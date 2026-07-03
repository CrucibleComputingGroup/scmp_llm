"""Uniform integer PTQ baseline (SmoothQuant + RTN fake-quant) for the HPCA
baselines — completely independent of the SC path.

W_xA_x, symmetric or asymmetric, at N bits:
  * weight     — per-OUTPUT-channel round-to-nearest (RTN), quantized once.
  * activation — per-TOKEN dynamic RTN, computed each forward.
  * SmoothQuant smoothing (per-input-channel scale s) applied first:
        w' = w * s   (broadcast over out dim) ;  x' = x / s
    using scmp_kernels.quant.smoothquant.compute_smooth_scales.

"Fake-quant" = quantize→dequantize in fp16 and run the matmul in fp16, so the
number we report is the *quantization error's* effect on PPL / accuracy (the
standard way PTQ accuracy is measured; real INT GEMM is a separate speed study).

Public API:
  QuantConfig(w_bits, a_bits, sym)
  load_quant_model(model_path, qcfg, act_scales, alpha)  -> HF CausalLM (patched)
  calibrate_act_scales(model, tokenizer, ...)            -> {name: amax-per-in-ch}
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# MoE router gates must NOT be quantized (top-k expert selection collapses),
# and the LM head is kept in fp16 by convention. Same skip set as the SC path.
_SKIP_LINEAR = {"gate", "lm_head"}


@dataclass
class QuantConfig:
    w_bits: int
    a_bits: int
    sym: bool                       # True: symmetric (zp=0); False: asymmetric
    per_channel_w: bool = True      # per-output-channel weight scale
    per_token_a: bool = True        # per-token dynamic activation scale

    def tag(self) -> str:
        return f"W{self.w_bits}A{self.a_bits}_{'symm' if self.sym else 'asymm'}"


def _fake_quant(x: torch.Tensor, bits: int, sym: bool, dim: int) -> torch.Tensor:
    """RTN quantize→dequantize along all dims, with scale/zp reduced over `dim`
    (kept per-slice on the OTHER dims). dim=-1 with x=(rows, in) → per-row.

    Returns a tensor of the same shape/dtype, holding dequantized values.
    """
    if bits >= 16:
        return x
    xf = x.float()
    if sym:
        qmax = float(2 ** (bits - 1) - 1)          # e.g. 4-bit → 7
        qmin = -float(2 ** (bits - 1))             #        → -8
        amax = xf.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        q = torch.clamp(torch.round(xf / scale), qmin, qmax)
        deq = q * scale
    else:
        qmax = float(2 ** bits - 1)                # e.g. 4-bit → 15
        qmin = 0.0
        xmin = xf.amin(dim=dim, keepdim=True)
        xmax = xf.amax(dim=dim, keepdim=True)
        scale = (xmax - xmin).clamp_min(1e-8) / (qmax - qmin)
        zp = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(xf / scale) + zp, qmin, qmax)
        deq = (q - zp) * scale
    return deq.to(x.dtype)


class QuantLinear(nn.Module):
    """Drop-in for nn.Linear: SmoothQuant smoothing + weight RTN (pre-quantized)
    + per-token activation RTN at forward. fp16 matmul on the dequantized values.
    """

    def __init__(self, linear: nn.Linear, qcfg: QuantConfig,
                 smooth_scale: Optional[torch.Tensor]):
        super().__init__()
        self.a_bits = qcfg.a_bits
        self.sym = qcfg.sym
        self.per_token_a = qcfg.per_token_a
        w = linear.weight.data.clone()                     # (out, in)
        dev, dt = w.device, w.dtype
        if smooth_scale is not None:
            s = smooth_scale.to(device=dev, dtype=torch.float32)   # (in,)
            self.register_buffer("smooth_scale", s, persistent=False)
            w = (w.float() * s.view(1, -1)).to(dt)         # w' = w * s
        else:
            self.smooth_scale = None
        # Weight quantized ONCE (per-output-channel → reduce over in-dim=1).
        wq_dim = 1 if qcfg.per_channel_w else None
        if wq_dim is None:
            wq = _fake_quant(w.reshape(1, -1), qcfg.w_bits, qcfg.sym, dim=-1).reshape_as(w)
        else:
            wq = _fake_quant(w, qcfg.w_bits, qcfg.sym, dim=1)
        self.register_buffer("weight", wq, persistent=False)
        self.register_buffer(
            "bias", linear.bias.data.clone() if linear.bias is not None else None,
            persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth_scale is not None:
            x = (x.float() / self.smooth_scale.view(1, -1)).to(x.dtype)  # x' = x/s
        if self.a_bits < 16:
            # Per-token dynamic: reduce over the feature dim (-1); each token row
            # gets its own scale. (per_token_a=False → per-tensor.)
            a_dim = -1 if self.per_token_a else None
            if a_dim is None:
                orig = x.shape
                x = _fake_quant(x.reshape(1, -1), self.a_bits, self.sym, dim=-1).reshape(orig)
            else:
                x = _fake_quant(x, self.a_bits, self.sym, dim=-1)
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out


def _iter_target_linears(model: nn.Module):
    """Yield (parent, attr, full_name, module) for every nn.Linear to quantize.

    Skip is by EXACT leaf name so the MoE router ``gate`` and ``lm_head`` are
    excluded but ``gate_proj`` (MLP gate projection) is NOT — a substring match
    would wrongly drop gate_proj.
    """
    for name, mod in model.named_modules():
        for attr, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and attr not in _SKIP_LINEAR:
                full = f"{name}.{attr}" if name else attr
                yield mod, attr, full, child


def apply_ptq(model: nn.Module, qcfg: QuantConfig,
              act_scales: Optional[Dict[str, torch.Tensor]] = None,
              alpha: float = 0.5) -> int:
    """Replace every (non-skipped) nn.Linear with a QuantLinear. Returns count.

    ``act_scales[name]`` (per-input-channel activation amax) enables SmoothQuant
    for that layer; missing → no smoothing for that layer.
    """
    from scmp_kernels.quant.smoothquant import compute_smooth_scales
    n = 0
    # The list holds every target Linear; if we don't free each original fp16
    # weight as we replace it, a 60GB MoE peaks at ~2x (originals + quantized
    # copies) and OOMs a 98GB GPU. QuantLinear clones what it needs in __init__,
    # so we can drop the original weight/bias immediately after.
    for parent, attr, full, lin in list(_iter_target_linears(model)):
        s = None
        if act_scales is not None and full in act_scales:
            s = compute_smooth_scales(
                act_scales[full].to(lin.weight.device), lin.weight.data, alpha=alpha)
        setattr(parent, attr, QuantLinear(lin, qcfg, s))
        lin.weight = None          # free the fp16 original NOW
        if getattr(lin, "bias", None) is not None:
            lin.bias = None
        n += 1
        if n % 2000 == 0:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return n


@torch.no_grad()
def calibrate_act_scales(model: nn.Module, tokenizer, texts, ctx: int,
                         n_windows: int) -> Dict[str, torch.Tensor]:
    """Per-input-channel activation amax over a few calib windows, keyed by the
    same module names apply_ptq uses (hooks the ORIGINAL nn.Linear inputs)."""
    scales: Dict[str, torch.Tensor] = {}
    handles = []

    def mk(full):
        def hook(_m, inp):
            x = inp[0].detach().float().abs().reshape(-1, inp[0].shape[-1])
            if x.shape[0] == 0:        # MoE expert that got zero routed tokens
                return
            x = x.amax(dim=0)          # (in,)
            prev = scales.get(full)
            scales[full] = x if prev is None else torch.maximum(prev.to(x.device), x)
        return hook

    for parent, attr, full, lin in _iter_target_linears(model):
        handles.append(lin.register_forward_pre_hook(mk(full)))
    enc = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    dev = next(model.parameters()).device
    for i in range(n_windows):
        s = i * ctx
        if s + 2 > enc.shape[0]:
            break
        ids = enc[s:s + ctx].unsqueeze(0).to(dev)
        model(input_ids=ids)
    for h in handles:
        h.remove()
    return {k: v.cpu() for k, v in scales.items()}


def _resolve_device_map(device_map):
    """Force whole-model-on-GPU-0 when exactly one GPU is visible.

    ``device_map="auto"`` lets accelerate shard/offload with ``meta`` tensors and
    per-layer materialization hooks. On a big MoE (Qwen3-30B-A3B: ~18k expert
    Linears) some weights stay on ``meta`` when apply_ptq wraps them, so the
    per-token ``x / smooth_scale`` divide hits a meta tensor ("device meta is not
    cuda:0"). Our models all fit on one 98GB GPU, so pin them to device 0.
    """
    if device_map == "auto" and torch.cuda.is_available() and torch.cuda.device_count() == 1:
        return {"": 0}
    return device_map


def load_plain_model(model_path: str, dtype=torch.float16, device_map="auto"):
    """Plain HF CausalLM (no SC), eager attention. Auto llama/qwen via id."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=_resolve_device_map(device_map),
        attn_implementation="eager")


def load_quant_model(model_path: str, qcfg: QuantConfig,
                     act_scales: Optional[Dict[str, torch.Tensor]] = None,
                     alpha: float = 0.5, device_map="auto"):
    """Load plain HF model and apply SmoothQuant + fake-quant per qcfg.
    FP16 baseline = QuantConfig(16, 16, ...) → apply_ptq is a near-no-op."""
    model = load_plain_model(model_path, device_map=device_map)
    model.eval()
    n = apply_ptq(model, qcfg, act_scales=act_scales, alpha=alpha)
    print(f"[ptq] {qcfg.tag()}: quantized {n} Linear layers "
          f"(smoothquant={'on' if act_scales else 'off'} alpha={alpha})")
    return model
