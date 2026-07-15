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
  * ``sc_linear_granularity`` (str, default "per_row") — linear SC granularity.
  (Attention granularity is NOT a knob — always per_row. The ``per_head`` kernel
  was removed 2026-07-03: catastrophic for small models at M=64, e.g. 1.7B
  uniform-128 attn-only PPL 17125 vs per_row 66.)
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
    # Precision-trace context (energy/latency simulator log). Tagging is
    # gated on _sc_trace._ENABLED so the off-path cost is one attr read.
    from scmp_kernels import trace as _sc_trace
except ImportError:
    class _sc_trace:            # kernels too old — tagging becomes a no-op
        _ENABLED = False

try:
    from scmp_kernels.mp import (
        MPConfig,
        AdaptiveMPConfig,
        classify_rows_by_metric,
        adaptive_classify_rows,
        compute_row_metric,
    )
    _HAS_MP = True
except ImportError:
    MPConfig = None
    AdaptiveMPConfig = None
    classify_rows_by_metric = None
    adaptive_classify_rows = None
    compute_row_metric = None
    _HAS_MP = False


def _mp_dispatch_metric(source: torch.Tensor, mp_config, operator) -> torch.Tensor:
    """Per-row dispatch metric for MP classify (act_global_v2 aware).

    AdaptiveMPConfig tables may carry a ρ-selected (metric, sign) per operator
    (``dispatch_metrics``); default — and every legacy table / MPConfig — is
    ("amax", +1), byte-identical to the original ``|x|.amax(-1)``. Sign −1
    inverts the ranking; ``adaptive_classify_rows``'s min–max normalization
    turns the negated metric into exactly 1 − normalized(raw), matching the
    calibration-side transform."""
    name, sign = "amax", 1.0
    if (AdaptiveMPConfig is not None and isinstance(mp_config, AdaptiveMPConfig)
            and compute_row_metric is not None):
        name, sign = mp_config.get_dispatch_metric(operator)
    metric = (source.abs().amax(dim=-1) if name == "amax"
              else compute_row_metric(source, name))
    return metric if sign >= 0 else -metric

try:
    from benchmark.quant.ptq import _fake_quant_lastdim_chunks as _int_fake_quant_chunks
    _HAS_INT_FAKE_QUANT = True
except ImportError:
    _int_fake_quant_chunks = None
    _HAS_INT_FAKE_QUANT = False


SC_CONFIG_DEFAULTS = {
    "use_sc_attn": True,
    "use_sc_linear": True,
    "sc_prec": 8,
    "sc_stoc_len": 256,
    "sc_mode": "bipolar",
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
    # Optional hybrid backend schedule:
    #   sc_hybrid_schedule[(op_name, block_idx)] = "sc" | "fp" | "int<N>"
    # Loaded by loader.apply_hybrid_config_from_env from a scmp_vit-style JSON.
    # None preserves the existing all-SC / MP behavior.
    "sc_hybrid_schedule": None,
    "sc_hybrid_default": "sc",
    "sc_hybrid_int_bits": 7,
    "sc_hybrid_int_sym": True,
    "sc_hybrid_chunk_size": 128,
    "sc_hybrid_force_int_bits": False,
}


def _mp_tracker() -> dict:
    """Process-wide accumulator for effective per-row stoc_len.

    Set ``model.config.sc_mp_track = True`` to record. Use
    ``mp_tracker_snapshot()`` to read and reset between sweep runs.
    """
    if not hasattr(_mp_tracker, "_state"):
        _mp_tracker._state = {"weighted_sl": 0.0, "rows": 0,
                              "mac_sl": 0.0, "macs": 0.0}
    return _mp_tracker._state


# Per-op MACs/row (from the calibration table's `mac_per_row`, exported since
# 2026-07-06). When set, the tracker ALSO accumulates a MAC-weighted average —
# the iso-compute (FLOP-avg) realized budget, matching the units the budget
# constraint is solved in. Empty dict => FLOP tracking off (older tables).
_MP_MAC_PER_ROW: dict = {}


def mp_tracker_set_mac_per_row(mac_per_row: dict) -> None:
    global _MP_MAC_PER_ROW
    _MP_MAC_PER_ROW = dict(mac_per_row or {})


def mp_tracker_reset() -> None:
    s = _mp_tracker()
    s["weighted_sl"] = 0.0
    s["rows"] = 0
    s["mac_sl"] = 0.0
    s["macs"] = 0.0


def mp_tracker_avg_stoc_len() -> float:
    s = _mp_tracker()
    return s["weighted_sl"] / max(s["rows"], 1)


def mp_tracker_flop_avg_stoc_len() -> float:
    """MAC-weighted realized average stoc_len (0.0 when no mac_per_row set)."""
    s = _mp_tracker()
    return (s["mac_sl"] / s["macs"]) if s.get("macs") else 0.0


def _record_stoc_len(sl: int, n: int, op=None, mac_scale: float = 1.0) -> None:
    if n == 0 or mac_scale <= 0.0:
        return
    state = _mp_tracker()
    weighted_n = float(n) * float(mac_scale)
    state["weighted_sl"] += float(sl) * weighted_n
    state["rows"] += weighted_n
    mac = _MP_MAC_PER_ROW.get(op) if op is not None else None
    if mac:
        state["mac_sl"] += float(sl) * weighted_n * mac
        state["macs"] += weighted_n * mac


def _record_assignment(assignment, op=None, mac_scale: float = 1.0) -> None:
    """Accumulate weighted-sum of per-row stoc_len for reporting.

    Each level's stoc_len is the *effective* cycle count seen by the kernel
    (MP levels are already in the halved space when halve_bipolar_stoc_len
    is on — they're just sliced from [0, 2**(sc_prec-1)]).
    With ``op`` + a mac_per_row map set, also accumulates the MAC-weighted
    (iso-compute) average. Observe-only — never affects dispatch.
    ``mac_scale`` is the input-channel fraction for split-channel matmuls;
    default 1.0 preserves legacy full-width accounting.
    """
    for sl, idxs in assignment.level_row_indices.items():
        n = int(idxs.numel())
        _record_stoc_len(int(sl), n, op=op, mac_scale=mac_scale)


def apply_sc_config_defaults(config) -> None:
    """Set missing SC knobs to their default values on ``config``."""
    for k, v in SC_CONFIG_DEFAULTS.items():
        if not hasattr(config, k):
            setattr(config, k, v)


def _channel_index_tensor(indices, width: int, device) -> torch.Tensor:
    vals = sorted({int(i) for i in (indices or []) if 0 <= int(i) < width})
    if not vals:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(vals, dtype=torch.long, device=device)


def _complement_channel_indices(width: int, selected: torch.Tensor, device) -> torch.Tensor:
    if selected.numel() == 0:
        return torch.arange(width, dtype=torch.long, device=device)
    mask = torch.ones(width, dtype=torch.bool, device=device)
    mask[selected] = False
    return mask.nonzero(as_tuple=True)[0]


def _hybrid_backend(config, op: Optional[str], block_idx: Optional[int]) -> str:
    schedule = getattr(config, "sc_hybrid_schedule", None)
    if schedule is None or op is None or block_idx is None:
        return "sc"
    return schedule.get(
        (op, int(block_idx)),
        getattr(config, "sc_hybrid_default", "sc"),
    )


def _hybrid_int_bits(config, backend: str) -> int:
    if (not bool(getattr(config, "sc_hybrid_force_int_bits", False))
            and backend.startswith("int") and backend[3:].isdigit()):
        return int(backend[3:])
    return int(getattr(config, "sc_hybrid_int_bits", 7))


def _int_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    bits: int,
    sym: bool,
    chunk_size: int,
    smooth_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not _HAS_INT_FAKE_QUANT:
        raise RuntimeError(
            "hybrid INT backend requested, but benchmark.quant.ptq "
            "_fake_quant_lastdim_chunks could not be imported")
    orig_dtype = x.dtype
    w = weight
    if smooth_scales is not None:
        s = smooth_scales.to(device=x.device, dtype=torch.float32)
        view_shape = [1] * x.ndim
        view_shape[-1] = s.numel()
        x = (x.float() / s.view(*view_shape)).to(orig_dtype)
        w = (weight.float() * s.view(1, -1)).to(orig_dtype)
    if bits < 16:
        x = _int_fake_quant_chunks(x, bits, sym, chunk_size)
        w = _int_fake_quant_chunks(w, bits, sym, chunk_size)
    return nn.functional.linear(x, w, bias)


def _int_attention_matmul_ab_t(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    bits: int,
    sym: bool,
    chunk_size: int,
) -> torch.Tensor:
    """INT fake-quant equivalent of ``a @ b.T`` for QK and AV."""
    if not _HAS_INT_FAKE_QUANT:
        raise RuntimeError(
            "hybrid INT attention requested, but benchmark.quant.ptq "
            "_fake_quant_lastdim_chunks could not be imported")
    if bits < 16:
        a = _int_fake_quant_chunks(a, bits, sym, chunk_size)
        b = _int_fake_quant_chunks(b, bits, sym, chunk_size)
    return torch.matmul(a, b.transpose(2, 3))


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
        if x_flat.shape[0] == 0:
            return torch.empty(
                (*orig_shape[:-1], self.out_features),
                dtype=orig_dtype,
                device=x.device,
            )
        w_fp32 = self.weight.to(torch.float32).contiguous()
        smooth = getattr(self, "smooth_scales", None)

        if _sc_trace._ENABLED:
            # Tag every sc_matmul this forward issues with this module's
            # identity for the precision trace. _sc_unit_idx = MoE expert
            # index (None for dense modules).
            _sc_trace.set_context(
                getattr(self, "_sc_op_name", None),
                getattr(self, "_sc_block_idx", None),
                getattr(self, "_sc_unit_idx", None),
            )

        op_name = getattr(self, "_sc_op_name", None)
        block_idx = getattr(self, "_sc_block_idx", None)
        backend = _hybrid_backend(config, op_name, block_idx)
        if backend == "fp":
            return nn.functional.linear(x, self.weight, self.bias)
        if backend.startswith("int"):
            bits = _hybrid_int_bits(config, backend)
            return _int_linear(
                x, self.weight, self.bias,
                bits=bits,
                sym=bool(getattr(config, "sc_hybrid_int_sym", True)),
                chunk_size=int(getattr(config, "sc_hybrid_chunk_size", sc_chunk_d)),
                smooth_scales=smooth,
            )

        group_map = getattr(config, "sc_group_stoclen", None)
        if group_map is not None:
            # Calibration-only per-(operator, block) UNIFORM stoc_len override,
            # used by the measured-ΔLoss knock-down probe: force this module to a
            # specified uniform stoc_len (halved space), no MP dispatch, no STE.
            # Modules not in the map fall back to the global sc_stoc_len.
            key = (op_name, block_idx)
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
            unit_idx = getattr(self, "_sc_unit_idx", None)
            protected_idx = torch.empty(0, dtype=torch.long, device=x_flat.device)
            rest_idx = None
            if AdaptiveMPConfig is not None and isinstance(mp_config, AdaptiveMPConfig):
                protected = mp_config.get_protected_channels(
                    operator=op_name, block_idx=block_idx, unit_idx=unit_idx)
                protected_idx = _channel_index_tensor(
                    protected, x_flat.shape[1], x_flat.device)
                if protected_idx.numel() > 0:
                    rest_idx = _complement_channel_indices(
                        x_flat.shape[1], protected_idx, x_flat.device)
            if rest_idx is not None and rest_idx.numel() == 0:
                metric = torch.zeros(x_flat.shape[0], dtype=torch.float32,
                                     device=x_flat.device)
            else:
                metric_source = x_flat if rest_idx is None else x_flat.index_select(1, rest_idx)
                metric = _mp_dispatch_metric(metric_source, mp_config, op_name)
            if AdaptiveMPConfig is not None and isinstance(mp_config, AdaptiveMPConfig):
                assignment = adaptive_classify_rows(
                    metric,
                    mp_config,
                    operator=op_name,
                    block_idx=block_idx,
                    total_blocks=getattr(config, "_sc_total_blocks", None),
                )
            else:
                assignment = classify_rows_by_metric(
                    metric,
                    mp_config.stoc_len_levels,
                    mp_config.level_fractions,
                )
            width = max(x_flat.shape[1], 1)
            protected_scale = float(protected_idx.numel()) / float(width)
            residual_scale = 1.0 - protected_scale
            if protected_idx.numel() > 0:
                out_flat = torch.zeros(
                    (x_flat.shape[0], self.out_features),
                    dtype=torch.float32,
                    device=x_flat.device,
                )
                x_prot = x_flat.index_select(1, protected_idx).contiguous()
                w_prot = w_fp32.index_select(1, protected_idx).contiguous()
                smooth_prot = (smooth.index_select(0, protected_idx).contiguous()
                               if smooth is not None else None)
                protect_sl = int(getattr(mp_config, "protected_channel_stoc_len", None)
                                 or max(mp_config.stoc_len_levels))
                out_flat += _sc_matmul(
                    x_prot, w_prot,
                    granularity=sc_gran, mode=sc_mode,
                    sc_prec=sc_prec, stoc_len=protect_sl, chunk_d=sc_chunk_d,
                    halve_bipolar_stoc_len=sc_halve,
                    smooth_scales=smooth_prot,
                )
                _record_stoc_len(protect_sl, x_flat.shape[0], op=op_name,
                                 mac_scale=protected_scale)
                x_dispatch = x_flat.index_select(1, rest_idx).contiguous()
                w_dispatch = w_fp32.index_select(1, rest_idx).contiguous()
                smooth_dispatch = (smooth.index_select(0, rest_idx).contiguous()
                                   if smooth is not None else None)
            else:
                out_flat = torch.empty(
                    (x_flat.shape[0], self.out_features),
                    dtype=torch.float32,
                    device=x_flat.device,
                )
                x_dispatch = x_flat
                w_dispatch = w_fp32
                smooth_dispatch = smooth
            _record_assignment(assignment, op_name, mac_scale=residual_scale)
            for sl, indices in assignment.level_row_indices.items():
                if residual_scale <= 0.0:
                    continue
                if indices.numel() == 0:
                    continue
                if sl <= 0:
                    if protected_idx.numel() == 0:
                        out_flat[indices] = 0.0
                    continue
                x_sub = x_dispatch.index_select(0, indices).contiguous()
                out_sub = _sc_matmul(
                    x_sub, w_dispatch,
                    granularity=sc_gran, mode=sc_mode,
                    sc_prec=sc_prec, stoc_len=int(sl), chunk_d=sc_chunk_d,
                    halve_bipolar_stoc_len=sc_halve,
                    smooth_scales=smooth_dispatch,
                )
                if protected_idx.numel() > 0:
                    out_flat[indices] += out_sub
                else:
                    out_flat[indices] = out_sub
        else:
            # UNIFORM linear (no MP / STE / group_map). Under halve the level is
            # in halved space (cap 2**(sc_prec-1)); clamp to that cap so the config
            # default (2**sc_prec) resolves to the halved ceiling — BIT-IDENTICAL
            # to the old stoc_len=None path — while a sub-cap request (e.g. 64) is
            # now EXPRESSIBLE instead of silently pinned to the ceiling. (Passing
            # the config int > cap verbatim would keep the full-length stream on a
            # halved grid, which the None path avoided; the min-clamp does too.)
            call_stoc_len = (min(int(sc_stoc_len), 2 ** (sc_prec - 1))
                             if sc_halve else sc_stoc_len)
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

    def _swap_within(parent: nn.Module, block_idx: int,
                     unit_idx: Optional[int] = None) -> None:
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
                # Sub-unit index for the precision trace: a digit-named
                # container on the path is a ModuleList entry — for MoE that
                # is the expert index (…experts.<i>.down_proj). Dense layers
                # have no digit-named ancestors, so this stays None.
                new_lin._sc_unit_idx = unit_idx
                setattr(parent, name, new_lin)
                n_replaced += 1
            else:
                _swap_within(child, block_idx,
                             int(name) if name.isdigit() else unit_idx)

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

    When ``mp_config`` is set, assigns a per-row stoc_len from
    ``a.abs().amax(-1)`` and dispatches each (B,H) slice through the 2D
    per_row path — at the cost of losing the 3D batched kernel parallelism.
    AdaptiveMPConfig rows are CLASSIFIED in one global pass over B*H*N (the
    normalization pools across heads, matching calibration — M1); the legacy
    fixed-fraction MPConfig keeps its by-design per-slice quantile split.

    ``operator`` / ``block_idx`` / ``total_blocks`` are used by AdaptiveMPConfig
    to look up calibrated thresholds. They are ignored for legacy MPConfig.
    """
    orig_dtype = a.dtype
    B, H, N, K = a.shape
    M = b.shape[-2]

    if _sc_trace._ENABLED:
        # Batched (uniform) attention calls carry no sub-unit; the per-(B,H)
        # MP dispatch loops below re-tag with unit = head index.
        _sc_trace.set_context(operator, block_idx, None)
    a3 = a.reshape(B * H, N, K).to(torch.float32).contiguous()
    b3 = b.reshape(B * H, M, K).to(torch.float32).contiguous()

    if mp_config is not None and _HAS_MP:
        out3 = torch.empty(
            (B * H, N, M), dtype=torch.float32, device=a3.device,
        )
        use_adaptive = (
            AdaptiveMPConfig is not None and isinstance(mp_config, AdaptiveMPConfig)
        )
        if use_adaptive:
            # M1 fix: classify ALL B*H*N rows in ONE global pass, so the
            # min/max normalization inside adaptive_classify_rows pools
            # across heads — matching the calibration, which normalizes the
            # full flattened metric. The previous per-(B,H) classify
            # stretched every head to [0,1] independently, erasing
            # cross-head scale: quiet heads claimed the same share of
            # high-precision rows as loud ones, and both consistency
            # directions were measured — per-slice calibration degrades
            # PPL badly (4B int7 act_global 16.9→19.2, grad_group 46→83),
            # so GLOBAL scope on both sides is the resolution.
            metric_all = _mp_dispatch_metric(a3, mp_config, operator)  # (B*H, N)
            assignment = adaptive_classify_rows(
                metric_all.reshape(-1), mp_config,
                operator=operator,
                block_idx=block_idx,
                total_blocks=total_blocks,
            )
            _record_assignment(assignment, operator)
            row_levels = assignment.row_levels.reshape(B * H, N)
            levels = mp_config.stoc_len_levels
            for bh in range(B * H):
                if _sc_trace._ENABLED:
                    _sc_trace.set_context(operator, block_idx, bh % H)
                a_bh = a3[bh]                          # (N, K)
                b_bh = b3[bh]                          # (M, K)
                lv_bh = row_levels[bh]                 # (N,) index into levels
                for li, sl in enumerate(levels):
                    indices = (lv_bh == li).nonzero(as_tuple=True)[0]
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
        # Legacy fixed-fraction MPConfig: quantile split is per-(B,H) slice
        # BY DESIGN (each head gets the configured level fractions), so the
        # per-slice classify is the correct semantics here — unchanged.
        for bh in range(B * H):
            if _sc_trace._ENABLED:
                _sc_trace.set_context(operator, block_idx, bh % H)
            a_bh = a3[bh]                              # (N, K)
            b_bh = b3[bh]                              # (M, K)
            metric = a_bh.abs().amax(dim=-1)           # (N,)
            assignment = classify_rows_by_metric(
                metric,
                mp_config.stoc_len_levels,
                mp_config.level_fractions,
            )
            _record_assignment(assignment, operator)
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

    # UNIFORM (mp_config is None) is the degenerate single-level case: all rows
    # at one stoc_len. This is THE single attention SC path — the STE and
    # knock-down-probe call sites route here too (mp_config=None), so uniform and
    # MP can never drift apart. per_head was removed 2026-07-03 (catastrophic for
    # small models at M=64: 1.7B uniform-128 attn-only PPL 17125 vs per_row 66);
    # attention is per_row only, so `granularity` is ignored.
    #
    # Under halve the level lives in halved space (cap 2**(sc_prec-1)); clamp to
    # that cap so the config default (2**sc_prec) resolves to the halved ceiling
    # — BIT-IDENTICAL to the old stoc_len=None path — while a sub-cap request
    # (e.g. 64) is now EXPRESSIBLE instead of silently pinned to the ceiling.
    # (Old bug: None always forced the ceiling, so uniform-<cap needed the
    # sc_group_stoclen={} escape hatch, and two attention functions existed.)
    call_stoc_len = (min(int(stoc_len), 2 ** (sc_prec - 1))
                     if halve_bipolar_stoc_len else stoc_len)
    out3 = _sc_matmul(
        a3, b3,
        granularity="per_row", mode=mode,
        sc_prec=sc_prec, stoc_len=call_stoc_len,
        halve_bipolar_stoc_len=halve_bipolar_stoc_len,
    )
    return out3.reshape(B, H, N, M).to(orig_dtype)


# NOTE: the former _sc_attn_ab_t_at_stoc_len (explicit single-stoc_len attention)
# was MERGED into _sc_attention_matmul_ab_t above — its uniform (mp_config=None)
# branch IS that computation (per_row, stoc_len passed through the min-clamp). The
# STE / knock-down-probe call sites below call _sc_attention_matmul_ab_t(mp_config
# =None) directly, so there is exactly ONE attention SC implementation.


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
    qk_backend = _hybrid_backend(config, "qk", block_idx)
    av_backend = _hybrid_backend(config, "av", block_idx)
    int_sym = bool(getattr(config, "sc_hybrid_int_sym", True))
    int_chunk = int(getattr(config, "sc_hybrid_chunk_size", 128))

    if qk_backend == "fp":
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    elif qk_backend.startswith("int"):
        attn_weights = _int_attention_matmul_ab_t(
            query, key_states,
            bits=_hybrid_int_bits(config, qk_backend),
            sym=int_sym,
            chunk_size=int_chunk,
        ) * scaling
    elif use_sc and group_map is not None:
        # Calibration-only per-(op, block) UNIFORM stoc_len override (measured
        # ΔLoss knock-down probe). Explicit stoc_len in halved space; no MP.
        sl_qk = int(group_map.get(("qk", block_idx), sc_stoc_len))
        attn_weights = _sc_attention_matmul_ab_t(
            query, key_states, mode=sc_mode, sc_prec=sc_prec,
            stoc_len=sl_qk, halve_bipolar_stoc_len=sc_halve,
            operator="qk", block_idx=block_idx,
        ) * scaling
    elif use_sc and ste_grad:
        # Straight-through Q·Kᵀ: forward = SC-noisy, backward = FP Jacobian.
        # Uniform explicit stoc_len, MP dispatch bypassed.
        fp_attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        with torch.no_grad():
            sc_attn = _sc_attention_matmul_ab_t(
                query, key_states, mode=sc_mode, sc_prec=sc_prec,
                stoc_len=sc_stoc_len, halve_bipolar_stoc_len=sc_halve,
                operator="qk", block_idx=block_idx,
            ) * scaling
        attn_weights = fp_attn + (sc_attn - fp_attn).detach()
    elif use_sc:
        attn_weights = _sc_attention_matmul_ab_t(
            query, key_states,
            mode=sc_mode,
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

    if av_backend == "fp":
        attn_output = torch.matmul(attn_weights, value_states)
    elif av_backend.startswith("int"):
        attn_output = _int_attention_matmul_ab_t(
            attn_weights, value_states.transpose(-2, -1),
            bits=_hybrid_int_bits(config, av_backend),
            sym=int_sym,
            chunk_size=int_chunk,
        )
    elif use_sc and group_map is not None:
        sl_av = int(group_map.get(("av", block_idx), sc_stoc_len))
        attn_output = _sc_attention_matmul_ab_t(
            attn_weights, value_states.transpose(-2, -1),
            mode=sc_mode, sc_prec=sc_prec,
            stoc_len=sl_av, halve_bipolar_stoc_len=sc_halve,
            operator="av", block_idx=block_idx,
        )
    elif use_sc and ste_grad:
        # Straight-through softmax·V: forward = SC-noisy, backward = FP Jacobian.
        fp_av = torch.matmul(attn_weights, value_states)
        with torch.no_grad():
            sc_av = _sc_attention_matmul_ab_t(
                attn_weights, value_states.transpose(-2, -1),
                mode=sc_mode, sc_prec=sc_prec,
                stoc_len=sc_stoc_len, halve_bipolar_stoc_len=sc_halve,
                operator="av", block_idx=block_idx,
            )
        attn_output = fp_av + (sc_av - fp_av).detach()
    elif use_sc:
        attn_output = _sc_attention_matmul_ab_t(
            attn_weights, value_states.transpose(-2, -1),
            mode=sc_mode,
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
