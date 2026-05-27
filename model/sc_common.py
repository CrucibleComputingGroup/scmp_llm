"""Shared SC building blocks for Llama and Qwen.

Both ``model/llama_sc.py`` and ``model_qwen4b/qwen3_sc.py`` are thin
adapters that:

  1. Load an upstream HF model with ``attn_implementation="eager"``.
  2. Monkey-patch the relevant ``eager_attention_forward`` in the HF
     ``modeling_*`` module with :func:`sc_eager_attention_forward`.
  3. Walk the decoder layers and replace each ``nn.Linear`` with
     :class:`SCLinear` via :func:`replace_linears_with_sc`.
  4. Apply SC config defaults via :func:`apply_sc_config_defaults`.

Knobs read from ``model.config``:

  * ``use_sc_attn`` (bool, default True) — route Q·Kᵀ and softmax·V through SC.
  * ``use_sc_linear`` (bool, default True) — route nn.Linear forwards through SC.
  * ``sc_prec`` (int, default 8) — quantization precision.
  * ``sc_stoc_len`` (int, default 256) — stochastic stream length.
  * ``sc_mode`` (str, default "bipolar") — bipolar vs unipolar.
  * ``sc_granularity`` (str, default "per_head") — attention SC granularity.
  * ``sc_linear_granularity`` (str, default "per_row") — linear SC granularity.
  * ``sc_linear_chunk_d`` (int, default 128) — D-chunking for linear path.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn

try:
    from scmp_kernels import sc_matmul as _sc_matmul
    _HAS_SC = True
except ImportError:
    _sc_matmul = None
    _HAS_SC = False


SC_CONFIG_DEFAULTS = {
    "use_sc_attn": True,
    "use_sc_linear": True,
    "sc_prec": 8,
    "sc_stoc_len": 256,
    "sc_mode": "bipolar",
    "sc_granularity": "per_head",
    "sc_linear_granularity": "per_row",
    "sc_linear_chunk_d": 128,
}


def apply_sc_config_defaults(config) -> None:
    """Set missing SC knobs to their default values on ``config``."""
    for k, v in SC_CONFIG_DEFAULTS.items():
        if not hasattr(config, k):
            setattr(config, k, v)


# ---------------------------------------------------------------------------
# SCLinear — nn.Linear subclass that routes the matmul through sc_matmul
# ---------------------------------------------------------------------------

class SCLinear(nn.Linear):
    """nn.Linear subclass whose forward dispatches to ``sc_matmul``.

    Weight / bias semantics, ``state_dict`` keys, and shapes match the
    parent ``nn.Linear`` exactly so HF checkpoints load as-is. When SC
    is unavailable or disabled the forward falls back to ``F.linear``.
    """

    def __init__(self, in_features, out_features, bias=True, *, sc_config=None, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self._sc_config = sc_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        config = self._sc_config
        use_sc = _HAS_SC and bool(getattr(config, "use_sc_linear", True))
        if not use_sc:
            return nn.functional.linear(x, self.weight, self.bias)

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
            sc_prec=sc_prec, stoc_len=sc_stoc_len, chunk_d=sc_chunk_d,
        )
        out = out_flat.reshape(*orig_shape[:-1], self.out_features).to(orig_dtype)
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_linears_with_sc(
    model: nn.Module,
    *,
    config,
    layer_filter,
    skip_names: Iterable[str] = (),
) -> int:
    """Replace every ``nn.Linear`` inside a ``layer_filter``-marked subtree.

    Descends ``model`` looking for any module where ``layer_filter(module)``
    returns True — typically a decoder layer class. Below such a module,
    every ``nn.Linear`` whose attribute name is NOT in ``skip_names`` is
    replaced in-place with an :class:`SCLinear` of the same shape, reusing
    the original weight and bias tensors so no extra GPU memory is
    consumed during conversion.

    Returns the number of replacements performed.
    """
    skip = set(skip_names)
    n_replaced = 0

    def _swap_within(parent: nn.Module) -> None:
        nonlocal n_replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, SCLinear):
                if name in skip:
                    continue
                new_lin = SCLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, sc_config=config,
                )
                new_lin.weight = child.weight
                if child.bias is not None:
                    new_lin.bias = child.bias
                # Preserve accelerate's dispatch/offload hook if present. With
                # device_map="auto" + CPU offload, the original Linear carries an
                # AlignDevicesHook whose weights_map holds the real (CPU) weights
                # while the module's own param is a `meta` placeholder. Dropping
                # the hook here leaves the SCLinear with a meta weight that is
                # never materialized at forward -> "Tensor on device meta" crash.
                # Transfer the existing hook (already initialized, weights_map
                # populated) so the replacement keeps offload semantics.
                hf_hook = getattr(child, "_hf_hook", None)
                if hf_hook is not None:
                    from accelerate.hooks import (
                        add_hook_to_module,
                        remove_hook_from_module,
                    )
                    remove_hook_from_module(child)
                    add_hook_to_module(new_lin, hf_hook)
                else:
                    new_lin.to(child.weight.device, dtype=child.weight.dtype)
                setattr(parent, name, new_lin)
                n_replaced += 1
            else:
                _swap_within(child)

    def _walk(parent: nn.Module) -> None:
        for child in parent.children():
            if layer_filter(child):
                _swap_within(child)
            else:
                _walk(child)

    _walk(model)
    return n_replaced


# ---------------------------------------------------------------------------
# SC attention forward — drop-in for HF's eager_attention_forward
# ---------------------------------------------------------------------------

def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA expansion: (B, H_kv, N, D) -> (B, H_kv*n_rep, N, D)."""
    batch, n_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv * n_rep, slen, head_dim)


def _sc_attention_matmul_ab_t(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    granularity: str,
    mode: str,
    sc_prec: int,
    stoc_len: int,
) -> torch.Tensor:
    """4D-aware wrapper around sc_matmul, computes ``a @ b.T``.

    a: (B, H, N, K), b: (B, H, M, K) -> (B, H, N, M).
    """
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
    """Drop-in for HF's ``eager_attention_forward``.

    Routes Q·Kᵀ and softmax·V through SC when ``module.config.use_sc_attn``.
    Accepts and ignores ``sliding_window`` kwarg that Qwen passes through.
    """
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    config = getattr(module, "config", None)
    use_sc = _HAS_SC and bool(getattr(config, "use_sc_attn", True))
    sc_gran = getattr(config, "sc_granularity", "per_head")
    sc_mode = getattr(config, "sc_mode", "bipolar")
    sc_prec = int(getattr(config, "sc_prec", 8))
    sc_stoc_len = int(getattr(config, "sc_stoc_len", 256))

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

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training)

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
