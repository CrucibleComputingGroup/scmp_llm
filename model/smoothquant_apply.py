"""SmoothQuant scale loading + per-layer application for SC models.

Two helpers:

* :func:`calibrate_act_scales` — hooks every ``SCLinear`` and accumulates
  per-input-channel max-abs across a list of input id tensors. Returns a
  dict keyed by the module's qualified name.

* :func:`apply_smoothquant_to_model` — given a calibrated ``act_scales``
  dict, builds the per-layer SmoothQuant scale vector via
  ``scmp_kernels.quant.compute_smooth_scales`` and attaches it as a
  ``smooth_scales`` buffer on each matching ``SCLinear``. ``SCLinear.forward``
  picks it up and passes it through to ``sc_matmul``.

SmoothQuant is applied only to projection layers (``SCLinear``). The SC
attention path (Q·Kᵀ, softmax·V) is activation·activation and has no
standard SmoothQuant analog, so it is left untouched — this is consistent
with upstream SmoothQuant practice.
"""
from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn

from scmp_kernels.quant.smoothquant import compute_smooth_scales

from .sc_common import SCLinear


@torch.no_grad()
def calibrate_act_scales(
    model: nn.Module,
    input_ids_iter: Iterable[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Per-channel max-abs of every ``SCLinear``'s input over the iterable.

    The model is run with SC disabled (FP16) so the calibration statistics
    match the operands the SC kernels will see at inference time.
    """
    act_scales: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, args):
            x = args[0]
            if x.dim() < 2 or x.numel() == 0:
                # Qwen3 MoE: an expert may receive zero tokens this batch.
                return
            flat = x.detach().abs().reshape(-1, x.shape[-1]).float()
            if flat.shape[0] == 0:
                return
            cur = flat.amax(dim=0)
            prev = act_scales.get(name)
            if prev is None:
                act_scales[name] = cur.clone()
            else:
                act_scales[name] = torch.maximum(prev.to(cur.device), cur)
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, SCLinear):
            handles.append(mod.register_forward_pre_hook(make_hook(name)))

    prev_attn = getattr(model.config, "use_sc_attn", True)
    prev_lin = getattr(model.config, "use_sc_linear", True)
    model.config.use_sc_attn = False
    model.config.use_sc_linear = False
    try:
        for input_ids in input_ids_iter:
            input_ids = input_ids.to(model.device)
            model(input_ids=input_ids)
    finally:
        model.config.use_sc_attn = prev_attn
        model.config.use_sc_linear = prev_lin
        for h in handles:
            h.remove()

    return act_scales


def _alpha_overrides_from_env() -> Dict[str, float]:
    """Per-operator SmoothQuant alpha overrides from ``SQ_ALPHA_OVERRIDES``
    (e.g. ``"down_proj:0.75,up_proj:0.65"``). Keys match the module name's
    last component. Parsed HERE — inside the one apply function every caller
    (eval_quant, calibrate_mp_thresholds, apply_smoothquant_from_env) shares —
    so calibration and eval CANNOT disagree on the smoothing geometry (the
    known calib==eval footgun). Empty/unset -> {} (global alpha everywhere,
    byte-identical to the old behavior)."""
    import os
    raw = os.environ.get("SQ_ALPHA_OVERRIDES", "").strip()
    if not raw:
        return {}
    out: Dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        op, _, val = part.partition(":")
        out[op.strip()] = float(val)
    return out


@torch.no_grad()
def apply_smoothquant_to_model(
    model: nn.Module,
    act_scales: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> int:
    """Attach a ``smooth_scales`` buffer to every ``SCLinear`` covered by
    ``act_scales``.

    Returns the number of layers wired.
    """
    overrides = _alpha_overrides_from_env()
    if overrides:
        print(f"[smoothquant] per-op alpha overrides active: {overrides} "
              f"(default alpha={alpha})")
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, SCLinear):
            continue
        if name not in act_scales:
            continue
        a = act_scales[name].to(mod.weight.device)
        mod_alpha = overrides.get(name.rsplit(".", 1)[-1], alpha)
        s = compute_smooth_scales(a, mod.weight.data, alpha=mod_alpha)
        # register_buffer is the canonical way to attach a non-parameter
        # tensor; survives model.to(device) and state_dict round-trips.
        if hasattr(mod, "smooth_scales"):
            mod.smooth_scales.copy_(s)
        else:
            mod.register_buffer("smooth_scales", s, persistent=False)
        n += 1
    return n


def apply_smoothquant_from_env(model: nn.Module) -> int:
    """Read ``USE_SMOOTHQUANT`` + ``SMOOTHQUANT_SCALES`` (+ optional
    ``SMOOTHQUANT_ALPHA``) and wire the model accordingly. No-op when
    ``USE_SMOOTHQUANT`` is not set to ``1``.
    """
    import os
    if os.environ.get("USE_SMOOTHQUANT", "0") != "1":
        return 0
    path = os.environ.get("SMOOTHQUANT_SCALES")
    if not path:
        raise SystemExit("USE_SMOOTHQUANT=1 requires SMOOTHQUANT_SCALES=<path.pt>")
    alpha = float(os.environ.get("SMOOTHQUANT_ALPHA", "0.5"))
    act_scales = torch.load(path, map_location="cpu", weights_only=True)
    n = apply_smoothquant_to_model(model, act_scales, alpha=alpha)
    print(f"[smoothquant] applied to {n} SCLinear modules (alpha={alpha}, scales={path})")
    return n
