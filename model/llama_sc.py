"""SC adapter for Llama 3.x.

Loads upstream HF ``LlamaForCausalLM`` with ``attn_implementation="eager"``,
then:

  * module-globally swaps ``transformers.models.llama.modeling_llama.\
eager_attention_forward`` with :func:`sc_common.sc_eager_attention_forward`.
  * replaces every ``nn.Linear`` inside each ``LlamaDecoderLayer`` with
    :class:`sc_common.SCLinear`, reusing weight tensors.

SC knobs live on ``model.config`` — see :data:`sc_common.SC_CONFIG_DEFAULTS`.

Replaces the previous 1286-line forked modeling file (kept at
``model/llama_sc_legacy.py`` for reference).
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import LlamaForCausalLM
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .sc_common import (
    SCLinear,
    apply_sc_config_defaults,
    replace_linears_with_sc,
    sc_eager_attention_forward,
)

__all__ = ["make_llama_sc", "SCLinear"]


_PATCHED = False


def _patch_module_globals_once() -> None:
    """Swap HF's Llama eager attention with the SC version (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    modeling_llama.eager_attention_forward = sc_eager_attention_forward
    _PATCHED = True


def make_llama_sc(
    model_path: str,
    *,
    torch_dtype: torch.dtype = torch.float16,
    device_map: Any = "auto",
    **from_pretrained_kwargs,
) -> LlamaForCausalLM:
    """Load Llama with full SC integration.

    Returns a regular HF ``LlamaForCausalLM`` instance; the SC behavior is
    enabled by the module-level patch and by the ``SCLinear`` replacements.
    Toggle on/off via ``model.config.use_sc_attn`` /
    ``model.config.use_sc_linear`` after this returns.
    """
    _patch_module_globals_once()

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map=device_map,
        **from_pretrained_kwargs,
    )
    apply_sc_config_defaults(model.config)

    n = replace_linears_with_sc(
        model, config=model.config,
        layer_filter=lambda m: isinstance(m, LlamaDecoderLayer),
    )
    model._sc_linear_replacements = n
    return model
