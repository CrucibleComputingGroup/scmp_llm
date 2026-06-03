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

try:
    from scmp_kernels.mp import (
        MPConfig,
        AdaptiveMPConfig,
        classify_rows_by_metric,
        adaptive_classify_rows,
    )
    _HAS_MP = True
except ImportError:
    MPConfig = None
    AdaptiveMPConfig = None
    classify_rows_by_metric = None
    adaptive_classify_rows = None
    _HAS_MP = False


SC_CONFIG_DEFAULTS = {
    "use_sc_attn": True,
    "use_sc_linear": True,
    "sc_prec": 8,
    "sc_stoc_len": 256,
    "sc_mode": "bipolar",
    "sc_granularity": "per_head",
    "sc_linear_granularity": "per_row",
    "sc_linear_chunk_d": 128,
    # When True, route through the wu-hpca2022 sign-magnitude cycle-halving
    # path in sc_matmul: forces stoc_len = rng_levels = 2**(sc_prec-1).
    # Overrides sc_stoc_len at call time (no resolution loss in bipolar).
    "sc_halve_bipolar_stoc_len": False,
    # Per-row mixed-precision config. When None (default), every row uses
    # the global sc_stoc_len. When set to a MPConfig instance, SCLinear and
    # sc_eager_attention_forward dispatch by row: classify each token row
    # (or query row) by its row metric, then call sc_matmul once per
    # stoc_len level with that level's row subset.
    "sc_mp_config": None,
    # STE noisy-trajectory gradient mode (calibration only). When True,
    # SCLinear / sc_eager_attention_forward return
    #     out = fp_out + (sc_out - fp_out).detach()
    # so the forward VALUE equals the SC-noisy output (noisy activations
    # propagate downstream) while the backward GRADIENT equals the clean FP
    # Jacobian (the .detach()'d SC term contributes none). Used by
    # benchmark/ppl/calibrate_mp_thresholds.py --grad-on-sc to capture
    # ‖∂L/∂y_row‖₂ along the SC-noisy trajectory rather than the clean FP
    # forward. The noisy forward runs at the uniform ``sc_stoc_len`` (which is
    # in halved space when ``sc_halve_bipolar_stoc_len`` is on), bypassing
    # MP row dispatch on purpose. Off by default: normal inference and the
    # FP-grad ('grad') calibration are byte-identical when this is False.
    "sc_ste_grad": False,
    # Calibration-only per-(operator, block_idx) UNIFORM stoc_len override map,
    # used by the measured-ΔLoss cross-layer sensitivity probe. When set to a
    # dict {(op_name, block_idx): stoc_len}, SCLinear and the attention path run
    # uniformly at the mapped stoc_len for that (op, block) — no MP dispatch, no
    # STE — falling back to sc_stoc_len for unmapped modules. None (default)
    # leaves every path byte-identical to normal inference.
    "sc_group_stoclen": None,
}


def _mp_tracker() -> dict:
    """Process-wide accumulator for effective per-row stoc_len.

    Set ``model.config.sc_mp_track = True`` to record. Use
    ``mp_tracker_snapshot()`` to read and reset between sweep runs.
    """
    if not hasattr(_mp_tracker, "_state"):
        _mp_tracker._state = {"weighted_sl": 0.0, "rows": 0}
    return _mp_tracker._state


def mp_tracker_reset() -> None:
    _mp_tracker()["weighted_sl"] = 0.0
    _mp_tracker()["rows"] = 0


def mp_tracker_avg_stoc_len() -> float:
    s = _mp_tracker()
    return s["weighted_sl"] / max(s["rows"], 1)


def _record_assignment(assignment) -> None:
    """Accumulate weighted-sum of per-row stoc_len for reporting.

    Each level's stoc_len is the *effective* cycle count seen by the kernel
    (MP levels are already in the halved space when halve_bipolar_stoc_len
    is on — they're just sliced from [0, 2**(sc_prec-1)]).
    """
    state = _mp_tracker()
    for sl, idxs in assignment.level_row_indices.items():
        n = int(idxs.numel())
        if n == 0:
            continue
        state["weighted_sl"] += float(sl) * n
        state["rows"] += n


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
        sc_halve = bool(getattr(config, "sc_halve_bipolar_stoc_len", False))
        mp_config = getattr(config, "sc_mp_config", None)

        orig_dtype = x.dtype
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]).to(torch.float32).contiguous()
        w_fp32 = self.weight.to(torch.float32).contiguous()
        smooth = getattr(self, "smooth_scales", None)

        group_map = getattr(config, "sc_group_stoclen", None)
        if group_map is not None:
            # Calibration-only per-(operator, block) UNIFORM stoc_len override,
            # used by the measured-ΔLoss knock-down probe: force this module to a
            # specified uniform stoc_len (halved space), no MP dispatch, no STE.
            # Modules not in the map fall back to the global sc_stoc_len.
            key = (getattr(self, "_sc_op_name", None), getattr(self, "_sc_block_idx", None))
            sl = int(group_map.get(key, sc_stoc_len))
            if sl <= 0:
                out = torch.zeros(
                    (*orig_shape[:-1], self.out_features),
                    dtype=orig_dtype, device=x.device,
                )
            else:
                out_flat = _sc_matmul(
                    x_flat, w_fp32,
                    granularity=sc_gran, mode=sc_mode,
                    sc_prec=sc_prec, stoc_len=sl, chunk_d=sc_chunk_d,
                    halve_bipolar_stoc_len=sc_halve, smooth_scales=smooth,
                )
                out = out_flat.reshape(*orig_shape[:-1], self.out_features).to(orig_dtype)
            if self.bias is not None:
                out = out + self.bias
            return out

        if bool(getattr(config, "sc_ste_grad", False)):
            # Straight-through SC: forward VALUE = SC-noisy output (propagates
            # downstream), backward GRADIENT = FP Jacobian. Runs at an explicit
            # uniform stoc_len so the noise level matches the deployment regime;
            # ``sc_stoc_len`` lives in halved space when ``sc_halve`` is on
            # (passed explicitly here, NOT as None, so the kernel uses it as the
            # stream length while keeping the halved RNG grid — mirrors
            # calibrate_mp_thresholds._sc_linear_at_level). MP row dispatch is
            # intentionally bypassed (uniform noise).
            fp_out = nn.functional.linear(x, self.weight, self.bias)
            with torch.no_grad():
                sc_flat = _sc_matmul(
                    x_flat, w_fp32,
                    granularity=sc_gran, mode=sc_mode,
                    sc_prec=sc_prec, stoc_len=int(sc_stoc_len), chunk_d=sc_chunk_d,
                    halve_bipolar_stoc_len=sc_halve, smooth_scales=smooth,
                )
                sc_out = sc_flat.reshape(*orig_shape[:-1], self.out_features).to(orig_dtype)
                if self.bias is not None:
                    sc_out = sc_out + self.bias
            return fp_out + (sc_out - fp_out).detach()

        if mp_config is not None and _HAS_MP:
            # Per-row mixed-precision dispatch. Classify each token row by
            # its abs-max along D, then call sc_matmul once per stoc_len
            # level on that level's row subset and scatter back.
            metric = x_flat.abs().amax(dim=-1)
            if AdaptiveMPConfig is not None and isinstance(mp_config, AdaptiveMPConfig):
                assignment = adaptive_classify_rows(
                    metric,
                    mp_config,
                    operator=getattr(self, "_sc_op_name", None),
                    block_idx=getattr(self, "_sc_block_idx", None),
                    total_blocks=getattr(config, "_sc_total_blocks", None),
                )
            else:
                assignment = classify_rows_by_metric(
                    metric,
                    mp_config.stoc_len_levels,
                    mp_config.level_fractions,
                )
            _record_assignment(assignment)
            out_flat = torch.empty(
                (x_flat.shape[0], self.out_features),
                dtype=torch.float32,
                device=x_flat.device,
            )
            for sl, indices in assignment.level_row_indices.items():
                if indices.numel() == 0:
                    continue
                if sl <= 0:
                    out_flat[indices] = 0.0
                    continue
                x_sub = x_flat.index_select(0, indices).contiguous()
                out_sub = _sc_matmul(
                    x_sub, w_fp32,
                    granularity=sc_gran, mode=sc_mode,
                    sc_prec=sc_prec, stoc_len=int(sl), chunk_d=sc_chunk_d,
                    halve_bipolar_stoc_len=sc_halve,
                    smooth_scales=smooth,
                )
                out_flat[indices] = out_sub
        else:
            # When halving, pass stoc_len=None so the kernel sets both stoc_len
            # and rng_levels to 2**(sc_prec-1). Passing the config int would
            # only halve rng_levels and silently keep the full-length stream.
            call_stoc_len = None if sc_halve else sc_stoc_len
            out_flat = _sc_matmul(
                x_flat, w_fp32,
                granularity=sc_gran, mode=sc_mode,
                sc_prec=sc_prec, stoc_len=call_stoc_len, chunk_d=sc_chunk_d,
                halve_bipolar_stoc_len=sc_halve,
                smooth_scales=smooth,
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

    Each SCLinear is tagged with ``_sc_op_name`` (the leaf attribute name,
    e.g. ``"q_proj"`` / ``"gate_proj"``) and ``_sc_block_idx`` (zero-based
    decoder-layer index). ``config._sc_total_blocks`` is set to the number
    of decoder layers found. AdaptiveMPConfig dispatch needs both.

    Returns the number of replacements performed.
    """
    skip = set(skip_names)
    n_replaced = 0
    block_counter = [0]

    def _swap_within(parent: nn.Module, block_idx: int) -> None:
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
                new_lin.to(child.weight.device, dtype=child.weight.dtype)
                new_lin._sc_op_name = name
                new_lin._sc_block_idx = block_idx
                setattr(parent, name, new_lin)
                n_replaced += 1
            else:
                _swap_within(child, block_idx)

    def _walk(parent: nn.Module) -> None:
        for child in parent.children():
            if layer_filter(child):
                block_idx = block_counter[0]
                block_counter[0] += 1
                _swap_within(child, block_idx)
            else:
                _walk(child)

    _walk(model)
    setattr(config, "_sc_total_blocks", block_counter[0])
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
    halve_bipolar_stoc_len: bool = False,
    mp_config=None,
    operator: Optional[str] = None,
    block_idx: Optional[int] = None,
    total_blocks: Optional[int] = None,
) -> torch.Tensor:
    """4D-aware wrapper around sc_matmul, computes ``a @ b.T``.

    a: (B, H, N, K), b: (B, H, M, K) -> (B, H, N, M).

    When ``mp_config`` is set, dispatches per-(B,H) slice with per-row
    stoc_len assignment based on ``a.abs().amax(-1)``. Each (B,H) slice
    runs the 2D per_row path so different rows can use different stoc_len
    levels — at the cost of losing the 3D batched kernel parallelism.

    ``operator`` / ``block_idx`` / ``total_blocks`` are used by AdaptiveMPConfig
    to look up calibrated thresholds. They are ignored for legacy MPConfig.
    """
    orig_dtype = a.dtype
    B, H, N, K = a.shape
    M = b.shape[-2]
    a3 = a.reshape(B * H, N, K).to(torch.float32).contiguous()
    b3 = b.reshape(B * H, M, K).to(torch.float32).contiguous()

    if mp_config is not None and _HAS_MP:
        out3 = torch.empty(
            (B * H, N, M), dtype=torch.float32, device=a3.device,
        )
        use_adaptive = (
            AdaptiveMPConfig is not None and isinstance(mp_config, AdaptiveMPConfig)
        )
        for bh in range(B * H):
            a_bh = a3[bh]                              # (N, K)
            b_bh = b3[bh]                              # (M, K)
            metric = a_bh.abs().amax(dim=-1)           # (N,)
            if use_adaptive:
                assignment = adaptive_classify_rows(
                    metric, mp_config,
                    operator=operator,
                    block_idx=block_idx,
                    total_blocks=total_blocks,
                )
            else:
                assignment = classify_rows_by_metric(
                    metric,
                    mp_config.stoc_len_levels,
                    mp_config.level_fractions,
                )
            _record_assignment(assignment)
            for sl, indices in assignment.level_row_indices.items():
                if indices.numel() == 0:
                    continue
                if sl <= 0:
                    out3[bh].index_fill_(0, indices, 0.0)
                    continue
                a_sub = a_bh.index_select(0, indices).contiguous()
                out_sub = _sc_matmul(
                    a_sub, b_bh,
                    granularity="per_row", mode=mode,
                    sc_prec=sc_prec, stoc_len=int(sl),
                    halve_bipolar_stoc_len=halve_bipolar_stoc_len,
                )
                out3[bh, indices] = out_sub
        return out3.reshape(B, H, N, M).to(orig_dtype)

    call_stoc_len = None if halve_bipolar_stoc_len else stoc_len
    out3 = _sc_matmul(
        a3, b3,
        granularity=granularity, mode=mode,
        sc_prec=sc_prec, stoc_len=call_stoc_len,
        halve_bipolar_stoc_len=halve_bipolar_stoc_len,
    )
    return out3.reshape(B, H, N, M).to(orig_dtype)


def _sc_attn_ab_t_at_stoc_len(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    mode: str,
    sc_prec: int,
    stoc_len: int,
    halve_bipolar_stoc_len: bool,
) -> torch.Tensor:
    """``a @ b.T`` at one EXPLICIT stoc_len. 4D in/out, per_head bipolar.

    Unlike :func:`_sc_attention_matmul_ab_t` (which passes ``stoc_len=None``
    when halving, forcing the kernel to the full 2**(sc_prec-1) stream), this
    helper always passes ``stoc_len`` through verbatim. With halving on, that
    means the halved RNG grid (2**(sc_prec-1) levels) but a stream truncated to
    ``stoc_len`` — i.e. ``stoc_len`` is interpreted in halved space, matching
    the calibration levels and ``calibrate_mp_thresholds._sc_attn_matmul_at_level``.
    Used only by the STE noisy-trajectory forward (``sc_ste_grad``).
    """
    orig_dtype = a.dtype
    B, H, N, K = a.shape
    M = b.shape[-2]
    a3 = a.reshape(B * H, N, K).to(torch.float32).contiguous()
    b3 = b.reshape(B * H, M, K).to(torch.float32).contiguous()
    out3 = _sc_matmul(
        a3, b3,
        granularity="per_head", mode=mode,
        sc_prec=sc_prec, stoc_len=int(stoc_len),
        halve_bipolar_stoc_len=halve_bipolar_stoc_len,
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
    sc_halve = bool(getattr(config, "sc_halve_bipolar_stoc_len", False))
    mp_config = getattr(config, "sc_mp_config", None)
    ste_grad = bool(getattr(config, "sc_ste_grad", False))
    group_map = getattr(config, "sc_group_stoclen", None)
    # AdaptiveMPConfig dispatch needs to know which (op, block) we're in.
    # MPConfig doesn't use these.
    block_idx = getattr(module, "layer_idx", None)
    total_blocks = getattr(config, "_sc_total_blocks", None)

    if use_sc and group_map is not None:
        # Calibration-only per-(op, block) UNIFORM stoc_len override (measured
        # ΔLoss knock-down probe). Explicit stoc_len in halved space; no MP.
        sl_qk = int(group_map.get(("qk", block_idx), sc_stoc_len))
        attn_weights = _sc_attn_ab_t_at_stoc_len(
            query, key_states, mode=sc_mode, sc_prec=sc_prec,
            stoc_len=sl_qk, halve_bipolar_stoc_len=sc_halve,
        ) * scaling
    elif use_sc and ste_grad:
        # Straight-through Q·Kᵀ: forward = SC-noisy, backward = FP Jacobian.
        # Uniform explicit stoc_len, MP dispatch bypassed.
        fp_attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        with torch.no_grad():
            sc_attn = _sc_attn_ab_t_at_stoc_len(
                query, key_states, mode=sc_mode, sc_prec=sc_prec,
                stoc_len=sc_stoc_len, halve_bipolar_stoc_len=sc_halve,
            ) * scaling
        attn_weights = fp_attn + (sc_attn - fp_attn).detach()
    elif use_sc:
        attn_weights = _sc_attention_matmul_ab_t(
            query, key_states,
            granularity=sc_gran, mode=sc_mode,
            sc_prec=sc_prec, stoc_len=sc_stoc_len,
            halve_bipolar_stoc_len=sc_halve,
            mp_config=mp_config,
            operator="qk",
            block_idx=block_idx,
            total_blocks=total_blocks,
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

    if use_sc and group_map is not None:
        sl_av = int(group_map.get(("av", block_idx), sc_stoc_len))
        attn_output = _sc_attn_ab_t_at_stoc_len(
            attn_weights, value_states.transpose(-2, -1),
            mode=sc_mode, sc_prec=sc_prec,
            stoc_len=sl_av, halve_bipolar_stoc_len=sc_halve,
        )
    elif use_sc and ste_grad:
        # Straight-through softmax·V: forward = SC-noisy, backward = FP Jacobian.
        fp_av = torch.matmul(attn_weights, value_states)
        with torch.no_grad():
            sc_av = _sc_attn_ab_t_at_stoc_len(
                attn_weights, value_states.transpose(-2, -1),
                mode=sc_mode, sc_prec=sc_prec,
                stoc_len=sc_stoc_len, halve_bipolar_stoc_len=sc_halve,
            )
        attn_output = fp_av + (sc_av - fp_av).detach()
    elif use_sc:
        attn_output = _sc_attention_matmul_ab_t(
            attn_weights, value_states.transpose(-2, -1),
            granularity=sc_gran, mode=sc_mode,
            sc_prec=sc_prec, stoc_len=sc_stoc_len,
            halve_bipolar_stoc_len=sc_halve,
            mp_config=mp_config,
            operator="av",
            block_idx=block_idx,
            total_blocks=total_blocks,
        )
    else:
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
