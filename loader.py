"""Unified model loader for both Llama and Qwen SC integrations.

Auto-dispatch based on the HuggingFace model id:

    "meta-llama/Llama-3.1-8B-Instruct"          → model.llama_sc.LlamaForCausalLM
    "Qwen/Qwen3-4B-Instruct-2507"               → model_qwen4b.qwen3_sc.make_qwen3_sc
    "Qwen/Qwen3-30B-A3B-Instruct-2507"          → model_qwen4b.qwen3_sc.make_qwen3_sc

Both code paths force ``attn_implementation="eager"`` so the SC attention
forward actually runs (the upstream HF default is ``sdpa`` which silently
bypasses SC attention).

All entry scripts (``test.py``, ``check_mse.py``, ``check_perlayer_mse.py``)
share the same env-var API for SC knobs — apply via ``apply_sc_env_overrides``.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import torch


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_sc_model(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    device_map: Any = "auto",
):
    """Load an SC-enabled CausalLM, auto-dispatching by model family.

    Both branches now share the same SC core (``model/sc_common.py``):
    an upstream HF model is loaded with eager attention, then patched in
    place — see ``model/llama_sc.py`` and ``model_qwen4b/qwen3_sc.py``.
    """
    p = model_path.lower()
    if "llama" in p:
        # LEGACY=1 -> use the original forked-modeling implementation
        # (model/llama_sc_legacy.py). Otherwise use the new adapter.
        # Used to A/B compare the two implementations.
        if os.environ.get("LEGACY", "0") == "1":
            from model.llama_sc_legacy import LlamaForCausalLM
            return LlamaForCausalLM.from_pretrained(
                model_path,
                attn_implementation="eager",
                torch_dtype=dtype,
                device_map=device_map,
            )
        from model.llama_sc import make_llama_sc
        return make_llama_sc(model_path, torch_dtype=dtype, device_map=device_map)
    if "qwen" in p:
        qwen_dir = os.path.join(_REPO_ROOT, "model_qwen4b")
        if qwen_dir not in sys.path:
            sys.path.insert(0, qwen_dir)
        from qwen3_sc import make_qwen3_sc
        return make_qwen3_sc(model_path, torch_dtype=dtype, device_map=device_map)
    raise ValueError(
        f"Unknown model family for {model_path!r}. "
        f"Expected a path/id containing 'llama' or 'qwen'.")


def apply_sc_env_overrides(model) -> None:
    """Apply shared SC env-var overrides to ``model.config``.

    Recognised env vars (all optional):

      * ``DISABLE_SC=1``           — turn off both attn and linear SC paths.
      * ``USE_SC_ATTN=0|1``        — fine-grained attention SC toggle.
      * ``USE_SC_LINEAR=0|1``      — fine-grained linear SC toggle.
      * ``SC_PREC``                — SC precision (int).
      * ``SC_STOC_LEN``            — SC stochastic stream length (int).
      * ``SC_ATTN_GRANULARITY``    — ``per_head`` or ``per_row``.
    """
    if os.environ.get("DISABLE_SC", "0") == "1":
        model.config.use_sc_attn = False
        model.config.use_sc_linear = False
    if "SC_PREC" in os.environ:
        model.config.sc_prec = int(os.environ["SC_PREC"])
    if "SC_STOC_LEN" in os.environ:
        model.config.sc_stoc_len = int(os.environ["SC_STOC_LEN"])
    if "USE_SC_ATTN" in os.environ:
        model.config.use_sc_attn = os.environ["USE_SC_ATTN"] == "1"
    if "USE_SC_LINEAR" in os.environ:
        model.config.use_sc_linear = os.environ["USE_SC_LINEAR"] == "1"
    if "SC_ATTN_GRANULARITY" in os.environ:
        model.config.sc_granularity = os.environ["SC_ATTN_GRANULARITY"]
    if "SC_HALVE_BIPOLAR_STOC_LEN" in os.environ:
        model.config.sc_halve_bipolar_stoc_len = (
            os.environ["SC_HALVE_BIPOLAR_STOC_LEN"] == "1")
    apply_mp_config_from_env(model)


def apply_mp_config_from_env(model) -> None:
    """Load per-row mixed-precision config from ``MP_CONFIG_JSON`` and attach
    it to ``model.config.sc_mp_config``.

    JSON schema (MPConfig, fixed-fraction quantile):

        {
          "type": "MPConfig",
          "stoc_len_levels": [128, 64, 32],
          "level_fractions": [0.2, 0.5, 0.3]
        }

    Unset env or empty path is a no-op (model stays in single-stoc_len mode).
    """
    path = os.environ.get("MP_CONFIG_JSON", "").strip()
    if not path:
        model.config.sc_mp_config = None
        return
    import json
    with open(path) as f:
        spec = json.load(f)
    kind = spec.get("type", "MPConfig")
    if kind != "MPConfig":
        raise SystemExit(
            f"MP_CONFIG_JSON: only MPConfig is wired into scmp_llm today, "
            f"got type={kind!r}"
        )
    from scmp_kernels.mp import MPConfig
    mp = MPConfig(
        stoc_len_levels=[int(x) for x in spec["stoc_len_levels"]],
        level_fractions=[float(x) for x in spec.get("level_fractions") or []] or None,
    )
    model.config.sc_mp_config = mp
    print(f"[mp] loaded MPConfig from {path}: "
          f"levels={mp.stoc_len_levels} fractions={mp.level_fractions}")


def describe_mode(model) -> str:
    """Short label for the current SC config (used in print output)."""
    use_sc_attn = getattr(model.config, "use_sc_attn", True)
    use_sc_linear = getattr(model.config, "use_sc_linear", True)
    sc_prec = getattr(model.config, "sc_prec", 8)
    sc_stoc_len = getattr(model.config, "sc_stoc_len", 256)
    sc_gran = getattr(model.config, "sc_granularity", "per_head")
    sc_halve = getattr(model.config, "sc_halve_bipolar_stoc_len", False)
    if not (use_sc_attn or use_sc_linear):
        return "FP16 baseline"
    # When halving is on, the effective stream length is 2**(sc_prec-1).
    eff_sl = (2 ** (sc_prec - 1)) if sc_halve else sc_stoc_len
    return (f"attn={'sc' if use_sc_attn else 'fp16'} "
            f"linear={'sc' if use_sc_linear else 'fp16'} "
            f"sc_prec={sc_prec} stoc_len={eff_sl}{' (halved)' if sc_halve else ''} "
            f"gran={sc_gran}")
