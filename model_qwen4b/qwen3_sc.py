"""SC adapter for Qwen3 dense + Qwen3 MoE.

Same shape as ``model/llama_sc.py``: load HF model with eager attention,
patch the module-level ``eager_attention_forward`` for both
``modeling_qwen3`` and (if importable) ``modeling_qwen3_moe``, then
replace every ``nn.Linear`` inside each decoder layer with ``SCLinear``.

The MoE router Linear (``Qwen3MoeSparseMoeBlock.gate``: hidden →
num_experts) is excluded from SC replacement. SC-quantizing the router
corrupts top-k expert selection and the model collapses to gibberish at
any ``stoc_len``, masking the actual SC quality of the experts themselves.
Note: ``gate_proj`` (MLP gate projection, a different role) is NOT skipped.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as _qwen3_mod
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

# Qwen3 MoE modeling may not be present in every transformers build.
try:
    from transformers.models.qwen3_moe import modeling_qwen3_moe as _qwen3_moe_mod
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
    _HAS_QWEN3_MOE = True
except ImportError:
    _qwen3_moe_mod = None
    Qwen3MoeDecoderLayer = None
    _HAS_QWEN3_MOE = False

# Reuse the shared SC core from model/sc_common.py at the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from model.sc_common import (  # noqa: E402
    SCLinear,
    apply_sc_config_defaults,
    replace_linears_with_sc,
    sc_eager_attention_forward,
)

__all__ = ["make_qwen3_sc", "SCLinear"]


# MoE router: see module docstring.
_SKIP_LINEAR_NAMES = frozenset({"gate"})

_PATCHED = False


def _patch_module_globals_once() -> None:
    """Swap Qwen3 (+ Qwen3 MoE) eager attention with the SC version (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    _qwen3_mod.eager_attention_forward = sc_eager_attention_forward
    if _HAS_QWEN3_MOE:
        _qwen3_moe_mod.eager_attention_forward = sc_eager_attention_forward
    _PATCHED = True


def _is_qwen3_layer(m) -> bool:
    if isinstance(m, Qwen3DecoderLayer):
        return True
    if _HAS_QWEN3_MOE and isinstance(m, Qwen3MoeDecoderLayer):
        return True
    return False


def make_qwen3_sc(
    pretrained_name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.float16,
    device_map: Any = "auto",
    **from_pretrained_kwargs,
):
    """Load Qwen3 (dense or MoE) with full SC integration."""
    _patch_module_globals_once()

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
        **from_pretrained_kwargs,
    )
    apply_sc_config_defaults(model.config)

    n = replace_linears_with_sc(
        model, config=model.config,
        layer_filter=_is_qwen3_layer,
        skip_names=_SKIP_LINEAR_NAMES,
    )
    model._sc_linear_replacements = n
    model.config._attn_implementation = "eager"
    return model
