"""Uniform integer PTQ baseline (SmoothQuant + RTN fake-quant) for the HPCA
baselines — completely independent of the SC path.

W_xA_x, symmetric or asymmetric, at N bits:
  * Linear weight     — per-output-channel, per-128-input-chunk RTN.
  * Linear activation — per-token, per-128-input-chunk dynamic RTN.
  * Attention QK/AV   — both operands fake-quantized per row, per 128 values,
                        before the fp16 matmul.
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
    chunk_size: int = 128           # SC-compatible input-dimension grouping
    quantize_attention: bool = True # quantize QK and AV operands too

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


def _fake_quant_lastdim_chunks(
    x: torch.Tensor,
    bits: int,
    sym: bool,
    chunk_size: int,
) -> torch.Tensor:
    """RTN fake-quantize independently over chunks of the last dimension.

    This matches the SC linear fast path's ``chunk_d=128`` scale scope: each
    token/output row gets a separate quantization scale for each 128-wide slice
    of the reduction dimension. The final short slice is quantized as-is.
    """
    if bits >= 16:
        return x
    if chunk_size <= 0 or x.shape[-1] <= chunk_size:
        return _fake_quant(x, bits, sym, dim=-1)
    chunks = [
        _fake_quant(x[..., start:start + chunk_size], bits, sym, dim=-1)
        for start in range(0, x.shape[-1], chunk_size)
    ]
    return torch.cat(chunks, dim=-1)


class QuantLinear(nn.Module):
    """Drop-in for nn.Linear: SmoothQuant smoothing + weight RTN (pre-quantized)
    + per-token activation RTN at forward. Both use 128-wide input chunks by
    default, matching SCLinear ``chunk_d``. fp16 matmul runs on dequantized values.
    """

    def __init__(self, linear: nn.Linear, qcfg: QuantConfig,
                 smooth_scale: Optional[torch.Tensor]):
        super().__init__()
        self.a_bits = qcfg.a_bits
        self.sym = qcfg.sym
        self.per_token_a = qcfg.per_token_a
        self.chunk_size = int(qcfg.chunk_size)
        w = linear.weight.data.clone()                     # (out, in)
        dev, dt = w.device, w.dtype
        if smooth_scale is not None:
            s = smooth_scale.to(device=dev, dtype=torch.float32)   # (in,)
            self.register_buffer("smooth_scale", s, persistent=False)
            w = (w.float() * s.view(1, -1)).to(dt)         # w' = w * s
        else:
            self.smooth_scale = None
        # Weight quantized ONCE. Default is per-output-channel AND per-128
        # input chunk, so each (output row, input chunk) has its own scale.
        wq_dim = 1 if qcfg.per_channel_w else None
        if wq_dim is None:
            wq = _fake_quant(w.reshape(1, -1), qcfg.w_bits, qcfg.sym, dim=-1).reshape_as(w)
        else:
            wq = _fake_quant_lastdim_chunks(
                w, qcfg.w_bits, qcfg.sym, self.chunk_size)
        self.register_buffer("weight", wq, persistent=False)
        self.register_buffer(
            "bias", linear.bias.data.clone() if linear.bias is not None else None,
            persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth_scale is not None:
            x = (x.float() / self.smooth_scale.view(1, -1)).to(x.dtype)  # x' = x/s
        if self.a_bits < 16:
            # Per-token dynamic: each token row gets one scale per 128-wide
            # feature chunk. (per_token_a=False → legacy per-tensor fallback.)
            a_dim = -1 if self.per_token_a else None
            if a_dim is None:
                orig = x.shape
                x = _fake_quant(x.reshape(1, -1), self.a_bits, self.sym, dim=-1).reshape(orig)
            else:
                x = _fake_quant_lastdim_chunks(
                    x, self.a_bits, self.sym, self.chunk_size)
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA expansion: (B, H_kv, N, D) -> (B, H_kv*n_rep, N, D)."""
    batch, n_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, n_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv * n_rep, slen, head_dim)


def int_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """HF eager-attention replacement for full INT fake-quant coverage.

    The projection linears are already QuantLinear. This patch covers the two
    non-Linear matmuls so W*A* configs no longer mean "linear-only INT".
    """
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    config = getattr(module, "config", None)
    enabled = bool(getattr(config, "use_int_attention", False))
    bits = int(getattr(config, "int_attention_bits", 16))
    sym = bool(getattr(config, "int_attention_sym", True))
    chunk = int(getattr(config, "int_chunk_size", 128))

    if enabled and bits < 16:
        query_q = _fake_quant_lastdim_chunks(query, bits, sym, chunk)
        key_q = _fake_quant_lastdim_chunks(key_states, bits, sym, chunk)
        attn_weights = torch.matmul(
            query_q, key_q.transpose(2, 3)) * scaling
    else:
        attn_weights = torch.matmul(
            query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training)

    if enabled and bits < 16:
        # Express AV as a @ b.T so both operands are quantized over the shared
        # reduction dimension (sequence length), matching the SC attention path.
        attn_q = _fake_quant_lastdim_chunks(attn_weights, bits, sym, chunk)
        value_t_q = _fake_quant_lastdim_chunks(
            value_states.transpose(-2, -1), bits, sym, chunk)
        attn_output = torch.matmul(attn_q, value_t_q.transpose(-2, -1))
    else:
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


_ATTN_PATCHED = False
_ATTN_PATCH_COUNT = 0


def patch_int_attention_once() -> int:
    """Patch HF eager attention for Llama/Qwen3/Qwen3-MoE when available."""
    global _ATTN_PATCHED, _ATTN_PATCH_COUNT
    if _ATTN_PATCHED:
        return _ATTN_PATCH_COUNT
    patched = 0
    try:
        from transformers.models.llama import modeling_llama
        modeling_llama.eager_attention_forward = int_eager_attention_forward
        patched += 1
    except Exception:
        pass
    try:
        from transformers.models.qwen3 import modeling_qwen3
        modeling_qwen3.eager_attention_forward = int_eager_attention_forward
        patched += 1
    except Exception:
        pass
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe
        modeling_qwen3_moe.eager_attention_forward = int_eager_attention_forward
        patched += 1
    except Exception:
        pass
    _ATTN_PATCHED = True
    _ATTN_PATCH_COUNT = patched
    return _ATTN_PATCH_COUNT


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
    """Plain HF CausalLM (no SC). Auto llama/qwen via id.

    Attention impl is env-selectable via BASELINE_ATTN. Quantized INT configs
    must use eager attention so :func:`int_eager_attention_forward` can cover
    QK/AV; callers enforce that before loading quantized models.
    """
    import os
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=_resolve_device_map(device_map),
        attn_implementation=os.environ.get("BASELINE_ATTN", "eager"))


def load_quant_model(model_path: str, qcfg: QuantConfig,
                     act_scales: Optional[Dict[str, torch.Tensor]] = None,
                     alpha: float = 0.5, device_map="auto"):
    """Load plain HF model and apply SmoothQuant + fake-quant per qcfg.
    FP16 baseline = QuantConfig(16, 16, ...) → apply_ptq is a near-no-op."""
    if qcfg.quantize_attention:
        if os.environ.get("BASELINE_ATTN", "eager") != "eager":
            raise SystemExit(
                "INT baselines require BASELINE_ATTN=eager so QK/AV are "
                "fake-quantized; refusing to build a linear-only INT model.")
        n_patched = patch_int_attention_once()
        if n_patched == 0:
            raise SystemExit(
                "INT baseline requested, but no supported HF eager "
                "attention modules were patched.")
        print(f"[ptq] INT attention patch active "
              f"(modules_patched={n_patched}, chunk={qcfg.chunk_size})")
    model = load_plain_model(model_path, device_map=device_map)
    model.eval()
    n = apply_ptq(model, qcfg, act_scales=act_scales, alpha=alpha)
    model.config.use_int_attention = bool(qcfg.quantize_attention)
    model.config.int_attention_bits = int(qcfg.a_bits)
    model.config.int_attention_sym = bool(qcfg.sym)
    model.config.int_chunk_size = int(qcfg.chunk_size)
    print(f"[ptq] {qcfg.tag()}: quantized {n} Linear layers "
          f"(smoothquant={'on' if act_scales else 'off'} alpha={alpha}, "
          f"chunk={qcfg.chunk_size}, "
          f"qk_av={'on' if qcfg.quantize_attention else 'off'})")
    return model
