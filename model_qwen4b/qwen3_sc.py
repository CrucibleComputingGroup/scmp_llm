# SC integration for Qwen3-4B. Mirrors model/llama_sc.py semantics but
# stays as a thin adapter: subclass nothing, patch only what's necessary.
#
# Two surfaces are SC-ified:
#   1. Every nn.Linear inside each Qwen3DecoderLayer (q/k/v/o + gate/up/down)
#      is replaced with SCLinear, reusing the loaded weight tensors.
#   2. transformers.models.qwen3.modeling_qwen3.eager_attention_forward is
#      module-globally swapped for sc_eager_attention_forward, which routes
#      Q·Kᵀ and softmax·V through scmp_kernels.sc_matmul when use_sc_attn.
#
# Qwen3Attention.forward looks up `eager_attention_forward` as a free
# module global at call time (modeling_qwen3.py L212), so patching the
# module attribute is sufficient and avoids forking the file.
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModelForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as _qwen3_mod
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

try:
    from scmp_kernels import sc_matmul as _sc_matmul
    _HAS_SC = True
except ImportError:
    _sc_matmul = None
    _HAS_SC = False


_SC_DEFAULTS = dict(
    use_sc_attn=True,
    use_sc_linear=True,
    sc_prec=8,
    sc_stoc_len=256,
    sc_mode="bipolar",
    sc_granularity="per_head",
    sc_linear_granularity="per_row",
    sc_linear_chunk_d=128,
)


def _sc_attention_matmul_ab_t(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    granularity: str,
    mode: str,
    sc_prec: int,
    stoc_len: int,
) -> torch.Tensor:
    """4D-aware ``a @ b.T`` via sc_matmul. Matches llama_sc._sc_attention_matmul_ab_t."""
    orig_dtype = a.dtype
    B, H, N, K = a.shape
    M = b.shape[-2]
    a3 = a.reshape(B * H, N, K).to(torch.float32).contiguous()
    b3 = b.reshape(B * H, M, K).to(torch.float32).contiguous()
    out3 = _sc_matmul(
        a3, b3,
        granularity=granularity, mode=mode,
        sc_prec=sc_prec, stoc_len=stoc_len,
    )
    return out3.reshape(B, H, N, M).to(orig_dtype)


class SCLinear(nn.Linear):
    """nn.Linear subclass routing matmul through scmp_kernels.sc_matmul.
    Same semantics as model/llama_sc.py:SCLinear.
    """

    def __init__(self, in_features, out_features, bias=True, *, sc_config=None, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self._sc_config = sc_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        config = self._sc_config
        use_sc = _HAS_SC and bool(getattr(config, "use_sc_linear", True))
        if not use_sc:
            return F.linear(x, self.weight, self.bias)

        sc_gran = getattr(config, "sc_linear_granularity", "per_row")
        sc_mode = getattr(config, "sc_mode", "bipolar")
        sc_prec = int(getattr(config, "sc_prec", 8))
        sc_stoc_len = int(getattr(config, "sc_stoc_len", 256))
        sc_chunk_d = int(getattr(config, "sc_linear_chunk_d", 128))

        orig_dtype = x.dtype
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]).to(torch.float32).contiguous()
        w_fp32 = self.weight.to(torch.float32).contiguous()
        out_flat = _sc_matmul(
            x_flat, w_fp32,
            granularity=sc_gran, mode=sc_mode,
            sc_prec=sc_prec, stoc_len=sc_stoc_len,
            chunk_d=sc_chunk_d,
        )
        out = out_flat.reshape(*orig_shape[:-1], self.out_features).to(orig_dtype)
        if self.bias is not None:
            out = out + self.bias
        return out


def sc_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Drop-in for transformers.models.qwen3.modeling_qwen3.eager_attention_forward.

    Routes both attention matmuls (Q·Kᵀ and softmax·V) through sc_matmul when
    use_sc_attn is True. Accepts and ignores ``sliding_window`` kwarg that
    Qwen3 attention passes through.
    """
    config = getattr(module, "config", None)
    use_sc = _HAS_SC and bool(getattr(config, "use_sc_attn", True))
    sc_gran = getattr(config, "sc_granularity", "per_head")
    sc_mode = getattr(config, "sc_mode", "bipolar")
    sc_prec = int(getattr(config, "sc_prec", 8))
    sc_stoc_len = int(getattr(config, "sc_stoc_len", 256))

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if use_sc:
        attn_weights = _sc_attention_matmul_ab_t(
            query, key_states,
            granularity=sc_gran, mode=sc_mode,
            sc_prec=sc_prec, stoc_len=sc_stoc_len,
        ) * scaling
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    if use_sc:
        attn_output = _sc_attention_matmul_ab_t(
            attn_weights, value_states.transpose(-2, -1),
            granularity=sc_gran, mode=sc_mode,
            sc_prec=sc_prec, stoc_len=sc_stoc_len,
        )
    else:
        attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _replace_linear_with_sclinear(module: nn.Module, sc_config) -> int:
    """Recursively replace nn.Linear (non-SCLinear) under ``module`` in place.
    Returns count of replacements. Weight/bias Parameters are rebound, not copied.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, SCLinear):
            new = SCLinear(
                child.in_features, child.out_features,
                bias=child.bias is not None, sc_config=sc_config,
                device=child.weight.device, dtype=child.weight.dtype,
            )
            new.weight = child.weight
            if child.bias is not None:
                new.bias = child.bias
            setattr(module, name, new)
            n += 1
        else:
            n += _replace_linear_with_sclinear(child, sc_config)
    return n


def patch_eager_attention():
    """Monkey-patch Qwen3 eager_attention_forward in its module namespace."""
    _qwen3_mod.eager_attention_forward = sc_eager_attention_forward


def make_qwen3_sc(
    pretrained_name_or_path: str,
    *,
    torch_dtype=torch.float16,
    device_map="auto",
    **from_pretrained_kwargs,
):
    """Load Qwen3 with SC integration. Mirrors how scmp_llm/test.py loads Llama.

    - Forces ``attn_implementation="eager"`` so our patched
      eager_attention_forward is on the hot path.
    - Sets SC defaults (sc_prec=8, stoc_len=256, ...) on model.config if not set.
    - Replaces every nn.Linear under each decoder layer with SCLinear,
      reusing loaded weights (no second checkpoint pass).
    - lm_head / embed_tokens stay as native nn.Linear / nn.Embedding —
      same scope as model/llama_sc.py.
    """
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
        **from_pretrained_kwargs,
    )
    config = model.config
    for k, v in _SC_DEFAULTS.items():
        if not hasattr(config, k):
            setattr(config, k, v)

    n_replaced = 0
    for layer in model.model.layers:
        n_replaced += _replace_linear_with_sclinear(layer, config)
    model._sc_linear_replacements = n_replaced

    patch_eager_attention()
    model.config._attn_implementation = "eager"
    return model
