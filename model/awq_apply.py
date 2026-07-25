"""AWQ front-end for SC models — native AWQ (INT-objective) scale search,
applied through the SAME per-linear ``smooth_scales`` slot as SmoothQuant.

Scope / rationale — this is a PTQ FRONT-END ablation, and SC stays a pure
downstream substrate. AWQ's per-input-channel equivalent-transform scale ``s``

    Y = A @ B.T = (A * s^-1) @ (B * s).T,    s_j > 0

is computed ONCE against an INT weight quantizer (never SC), then folded via
the ``smooth_scales`` buffer exactly like SmoothQuant — ``s`` uses the identical
convention (activations divided, weights multiplied), so it drops into the same
runtime path in ``model/sc_common.py`` with no kernel change. Mirrors
``model/smoothquant_apply.py``.

Difference from SmoothQuant: SmoothQuant sets ``s`` from statistics only
(``act_max^alpha / weight_max^(1-alpha)``, quantizer-agnostic). AWQ *searches*
``s = act_scale^ratio`` over a grid, picking the ratio that minimises the
INT-quantized-output MSE (Lin et al., "AWQ", MLSys 2024,
``AWQ-BitMoD/awq/quantize/auto_scale.py``). So AWQ's ``s`` is tuned to a
quantizer — here an INT proxy (``AWQ_OBJ_BITS``, default 4), NOT SC.

Caveat baked into the default: AWQ's objective minimises weight-GRID error. SC
at ``sc_prec=8`` is an 8-bit grid whose dominant error is STREAM VARIANCE, so
AWQ can only help SC via the activation-de-outliering side effect of ``s``. Run
the search against a near-lossless INT8 objective and it degenerates to
``s≈1``; ``AWQ_OBJ_BITS=4`` keeps the search in a regime where it produces real
de-outliering scales that the 8-bit SC substrate can still benefit from.

Only projection layers (``SCLinear``) are smoothed; the SC attention path
(Q·Kᵀ, softmax·V) is activation·activation and has no standard AWQ analog, so it
is left untouched — same coverage as SmoothQuant here.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from .sc_common import SCLinear


# ---- INT proxy quantizer (the AWQ search objective) --------------------------
@torch.no_grad()
def _fake_quant_int(w: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """Per-(output-row, ``group``-input-chunk) symmetric absmax RTN fake-quant.

    Matches the house g128 granularity (INT baselines, SC ``chunk_d=128``). Falls
    back to per-output-row when the input dim does not divide ``group``.
    """
    out_f, in_f = w.shape
    q_max = 2 ** (bits - 1) - 1
    if group and in_f % group == 0 and in_f > group:
        wg = w.view(out_f, in_f // group, group)
        scale = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / q_max
        wq = (wg / scale).round().clamp(-q_max, q_max) * scale
        return wq.view(out_f, in_f)
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / q_max
    return (w / scale).round().clamp(-q_max, q_max) * scale


@torch.no_grad()
def _search_scale(
    x_cache: torch.Tensor,       # (T, in) representative inputs
    weight: torch.Tensor,        # (out, in)
    x_scale: torch.Tensor,       # (in,) per-channel mean-abs (AWQ get_act_scale)
    obj_bits: int,
    n_grid: int = 20,
) -> torch.Tensor:
    """AWQ per-input-channel scale: ``s = x_scale^ratio`` (normalised) minimising
    ``|| X W^T - (X/s) q(W·s)^T ||`` over an ``n_grid`` sweep of ratio∈[0,1).

    ratio=0 → s=1 (no smoothing) is in the grid, so the search degrades
    gracefully to "no front-end" when smoothing never helps."""
    dev = weight.device
    W = weight.float()
    X = x_cache.float().to(dev)
    ref = X @ W.t()
    xs = x_scale.float().to(dev).clamp(min=1e-4)
    best_s: Optional[torch.Tensor] = None
    best_loss = float("inf")
    for i in range(n_grid):
        ratio = i / n_grid
        s = xs.pow(ratio)
        # AWQ normalisation: centre the scale so it neither blows up weights nor
        # activations on average (geometric-mean normalisation).
        s = (s / (s.max() * s.min()).sqrt()).clamp(min=1e-4)
        wq = _fake_quant_int(W * s.view(1, -1), obj_bits)
        out = (X / s.view(1, -1)) @ wq.t()
        loss = (ref - out).pow(2).mean().item()
        if loss < best_loss:
            best_loss = loss
            best_s = s.clone()
        del wq, out
    assert best_s is not None and torch.isnan(best_s).sum() == 0
    return best_s


# ---- calibration: capture per-linear act mean-abs + a capped input sample ----
@torch.no_grad()
def calibrate_awq_scales(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    *,
    ctx: int = 512,
    n_windows: int = 16,
    obj_bits: int = 4,
    n_grid: int = 20,
    tok_cap: int = 128,
) -> Dict[str, torch.Tensor]:
    """Compute AWQ per-input-channel scales for every ``SCLinear``.

    Runs an FP16 forward (SC disabled) over ``n_windows`` wikitext windows,
    hooking each ``SCLinear`` to accumulate (a) a running per-channel mean-abs
    and (b) up to ``tok_cap`` rows of its input. Then per layer searches the AWQ
    ratio against an ``obj_bits`` INT proxy. Returns ``{name: s (in,)}``.
    """
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    cache: Dict[str, List[torch.Tensor]] = {}
    cached_rows: Dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, args):
            x = args[0]
            if x.dim() < 2 or x.numel() == 0:
                return
            flat = x.detach().reshape(-1, x.shape[-1])
            if flat.shape[0] == 0:
                return
            fa = flat.abs().float()
            s = fa.sum(dim=0)
            sums[name] = s if name not in sums else sums[name] + s
            counts[name] = counts.get(name, 0) + fa.shape[0]
            have = cached_rows.get(name, 0)
            if have < tok_cap:
                take = min(tok_cap - have, flat.shape[0])
                cache.setdefault(name, []).append(flat[:take].detach().to(torch.float16).cpu())
                cached_rows[name] = have + take
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, SCLinear):
            handles.append(mod.register_forward_pre_hook(make_hook(name)))

    enc = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    n_tok = ctx * n_windows
    enc = enc[:n_tok] if enc.numel() >= n_tok else enc
    prev_attn = getattr(model.config, "use_sc_attn", True)
    prev_lin = getattr(model.config, "use_sc_linear", True)
    model.config.use_sc_attn = False
    model.config.use_sc_linear = False
    try:
        for w in range(0, enc.numel() - 1, ctx):
            ids = enc[w:w + ctx]
            if ids.numel() == 0:
                break
            model(input_ids=ids.unsqueeze(0).to(model.device))
    finally:
        model.config.use_sc_attn = prev_attn
        model.config.use_sc_linear = prev_lin
        for h in handles:
            h.remove()

    scales: Dict[str, torch.Tensor] = {}
    by_name = dict(model.named_modules())
    for name in sorted(sums):
        mod = by_name.get(name)
        if mod is None or name not in cache:
            continue
        x_scale = sums[name] / max(counts[name], 1)
        x_cache = torch.cat(cache[name], dim=0)
        s = _search_scale(
            x_cache, mod.weight.data, x_scale, obj_bits, n_grid=n_grid)
        scales[name] = s.to(torch.float16).cpu()
        del cache[name]                       # free as we go
    return scales


# ---- apply + env-driven entry point ------------------------------------------
@torch.no_grad()
def apply_awq_to_model(model: nn.Module, awq_scales: Dict[str, torch.Tensor]) -> int:
    """Attach a ``smooth_scales`` buffer (the AWQ ``s``) to every covered
    ``SCLinear`` — same slot/convention SmoothQuant uses. Returns count wired."""
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, SCLinear) or name not in awq_scales:
            continue
        s = awq_scales[name].to(mod.weight.device)
        if hasattr(mod, "smooth_scales"):
            mod.smooth_scales.copy_(s)
        else:
            mod.register_buffer("smooth_scales", s, persistent=False)
        n += 1
    return n


@torch.no_grad()
def apply_awq_frontend(
    model: nn.Module,
    tokenizer,
    model_path: str,
    *,
    cache_dir: str,
    obj_bits: Optional[int] = None,
    recalibrate: bool = False,
) -> int:
    """Load cached AWQ scales for ``model_path`` (keyed by obj_bits) or compute
    and cache them, then wire them onto the SC model. Mirrors the SmoothQuant
    cache-or-calibrate flow so AWQ costs one calibration per model, reused
    across budgets."""
    obj_bits = int(os.environ.get("AWQ_OBJ_BITS", obj_bits if obj_bits is not None else 4))
    safe = model_path.replace("/", "_")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"awq_scales_{safe}_b{obj_bits}.pt")
    if os.path.isfile(path) and not recalibrate:
        print(f"[awq] scales cache hit: {path}")
        awq_scales = torch.load(path, map_location="cpu", weights_only=True)
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [d["text"] for d in ds if d.get("text", "").strip()][:2000]
        print(f"[awq] calibrating AWQ scales ({model_path}, obj_bits={obj_bits}) ...")
        awq_scales = calibrate_awq_scales(
            model, tokenizer, texts,
            ctx=int(os.environ.get("CALIB_CTX", "512")),
            n_windows=int(os.environ.get("CALIB_WINDOWS", "16")),
            obj_bits=obj_bits,
            n_grid=int(os.environ.get("AWQ_N_GRID", "20")),
            tok_cap=int(os.environ.get("AWQ_TOK_CAP", "128")))
        torch.save(awq_scales, path)
        print(f"[awq] wrote {path} ({len(awq_scales)} layers)")
    n = apply_awq_to_model(model, awq_scales)
    print(f"[awq] applied to {n} SCLinear modules (obj_bits={obj_bits})")
    return n
