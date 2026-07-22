"""Calibrate per-operator, per-layer SC mixed-precision thresholds for an LLM.

Ported from ``scmp_diffusion/scripts/calibrate_mp_thresholds.py``. The
mechanism is the same — measure per-row reconstruction error of each SC
operator at each ``stoc_len`` level against an FP teacher, then solve a
budget-aware allocation problem to pick non-uniform thresholds — but the
forward driver is the wikitext2 PPL pipeline (`benchmark/ppl/ppl.py`)
instead of diffusion sampling, and the bucket schema collapses to a single
timestep bucket because LLM inference has no diffusion time axis.

Output JSON is consumed by
``scmp_kernels.mp.AdaptiveMPConfig.load_threshold_table`` and dispatched
through ``model/sc_common.py::adaptive_classify_rows``.

Usage (after `_setup_env.done` is touched and an HF token is configured):

  python -u benchmark/ppl/calibrate_mp_thresholds.py \\
    --model_path Qwen/Qwen3-4B-Instruct-2507 \\
    --sc_prec 8 --halve 1 \\
    --mp_levels 128,96,64 \\
    --budget_ratio 0.71 --budget_ref_stoc_len 128 \\
    --num_calib_sequences 16 --ctx_len 1024 \\
    --output_json benchmark/ppl/mp_calib/<safe>__int8_avg91.json

Operators recorded: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj, qk, av. The router ``gate`` Linear in MoE is skipped (same
``_SKIP_LINEAR_NAMES`` rule as the inference path).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from loader import load_sc_model, apply_sc_env_overrides  # noqa: E402
from model.sc_common import SCLinear  # noqa: E402
from scmp_kernels import sc_matmul as _sc_matmul  # noqa: E402
from benchmark.ppl.mp_objectives import objective_errors as _objective_errors  # noqa: E402


# -- defaults --
DEFAULT_OPERATORS = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,qk,av"
)
LINEAR_OPS = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}
ATTN_OPS = {"qk", "av"}


# =====================================================================
# Metric / error utilities
# =====================================================================


def _normalize_metric(metric: torch.Tensor) -> torch.Tensor:
    metric = metric.float()
    if metric.numel() == 0:
        # Empty row batch (e.g. an MoE expert that received zero tokens this
        # forward). Nothing to normalize; return as-is so callers can skip it.
        return metric
    m_min = metric.min()
    m_max = metric.max()
    if (m_max - m_min).item() < 1e-8:
        return torch.ones_like(metric, dtype=torch.float32)
    return (metric - m_min) / (m_max - m_min)


# M1 resolution note: calibration and runtime must share ONE normalization
# scope for the attention metric. Both now pool GLOBALLY over B*H*N rows
# (runtime: _sc_attention_matmul_ab_t classifies the whole flattened metric
# once; calibration: the qk/av hooks above use _normalize_metric on the full
# reshape). The per-slice alternative (normalize each (B,H) head independently)
# was tried and measurably degrades PPL: min-max stretching every head to [0,1]
# erases cross-head scale, so the allocator cannot move budget toward
# genuinely large heads — the very cross-group flow the global-λ solve needs
# (4B int7 act_global 16.9→19.2, grad_group 46→83).


def _metric_candidates(x_rows: torch.Tensor) -> dict:
    """act_global_v2 candidate dispatch metrics for one call, per-call
    NORMALIZED like the primary amax metric. ``x_rows`` is the SAME row-source
    tensor the amax metric was computed from (protected channels already
    excluded), flattened to (rows, D). Formulas must mirror
    ``scmp_kernels.mp.compute_row_metric`` exactly (runtime parity):
    l2 = ‖row‖₂, crest = ‖row‖_inf / ‖row‖₂ (eps 1e-12)."""
    xf = x_rows.float().reshape(-1, x_rows.shape[-1])
    l2 = xf.norm(dim=-1)
    amax = xf.abs().amax(dim=-1)
    crest = amax / (l2 + 1e-12)
    return {"l2": _normalize_metric(l2), "crest": _normalize_metric(crest)}


def _relative_l2_rows(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.float().reshape(pred.shape[0], -1)
    target_f = target.float().reshape(target.shape[0], -1)
    denom = target_f.norm(dim=-1).clamp_min(1e-8)
    return (pred_f - target_f).norm(dim=-1) / denom


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, numpy-only (rank then Pearson)."""
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    d = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def _relative_l2_heads(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target: [B, H, N, M] -> per-head error vector of length H.
    pred_h = pred.float().permute(1, 0, 2, 3).reshape(pred.shape[1], -1)
    target_h = target.float().permute(1, 0, 2, 3).reshape(target.shape[1], -1)
    denom = target_h.norm(dim=-1).clamp_min(1e-8)
    return (pred_h - target_h).norm(dim=-1) / denom


def _bucket_index(value: int, total: int, num_buckets: int) -> int:
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


# =====================================================================
# Budget allocation (Lagrangian binary search). Verbatim from diffusion.
# =====================================================================


def _cost_assignments(
    errors: np.ndarray,
    costs: np.ndarray,
    budget_total: float,
) -> np.ndarray:
    """Pick a level per unit that minimizes sum(error) under sum(cost) ≤ budget.

    errors: [n_units, n_levels]. costs: [n_levels] in stoc_len units.
    Lagrangian relaxation: for each lambda, each unit picks
    argmin_i (error[i] + lambda * cost[i]). Binary-search lambda to hit budget.
    """
    n_units = errors.shape[0]
    min_cost = float(n_units * costs[-1])
    max_cost = float(n_units * costs[0])
    if budget_total <= min_cost:
        return np.full(n_units, len(costs) - 1, dtype=np.int64)
    if budget_total >= max_cost:
        return np.zeros(n_units, dtype=np.int64)

    def solve_for_lambda(lmbd: float):
        objective = errors + lmbd * costs[None, :]
        assignment = objective.argmin(axis=1)
        total_cost = float(costs[assignment].sum())
        return assignment, total_cost

    lo = 0.0
    hi = 1.0
    _, cost_hi = solve_for_lambda(hi)
    while cost_hi > budget_total and hi < 1e6:
        hi *= 2.0
        _, cost_hi = solve_for_lambda(hi)

    best = np.zeros(n_units, dtype=np.int64)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        cand, cost_mid = solve_for_lambda(mid)
        best = cand
        if cost_mid > budget_total:
            lo = mid
        else:
            hi = mid
    return best


def _global_lambda(
    weighted_errors: list,
    costs: np.ndarray,
    budget_ratio: float,
    ref: float,
    row_counts: list,
    prices: Optional[list] = None,
) -> float:
    """Find ONE shared Lagrange multiplier λ across many groups for a GLOBAL,
    ROW-WEIGHTED budget — the cross-layer allocation core.

    ``weighted_errors`` is a list of [n_g, n_levels] arrays, each already scaled
    by its per-group importance weight W_g. ``row_counts[g]`` is group g's TRUE
    per-forward row count R_g (un-subsampled). The runtime cost of group g is
    ``R_g · mean_i(cost(level_i))`` — i.e. each stored (subsampled) row stands in
    for R_g/n_g real rows — and the budget is ``budget_ratio·ref·Σ R_g``. Without
    the R_g weighting, attention (qk/av), which carries far more rows per forward
    than the linears, is under-counted and the global allocation overspends at
    runtime (realized avg_sl ≫ target). R_g cancels in the per-row argmin, so the
    assignment shape is unchanged; it only fixes the cross-group budget balance.

    Binary-search λ so the global row-weighted total ≤ budget. Returns the λ to
    feed each group's assignment (see ``_fit_group(lam=...)``).
    """
    total_rows = float(sum(row_counts))
    if total_rows <= 0:
        return 0.0
    # rep_g = R_g / n_g: each stored (subsampled) row stands in for rep_g true
    # rows, so its cost in the assignment must be priced ·rep_g. WITHOUT this the
    # per-row assignment prices every group's cycle equally while the budget is
    # R_g-weighted — an inconsistent objective that over-prices the cheap (few-
    # row) linear groups and craters them to the floor, landing WORSE than
    # uniform on the very error it claims to minimize. With rep_g the assignment
    # and budget share one consistent R_g-weighted objective, uniform is feasible
    # and the optimum is ≤ uniform.
    reps = [float(R) / max(int(e.shape[0]), 1)
            for e, R in zip(weighted_errors, row_counts)]
    # act_global_v2 (--argmin-pricing macs): callers may override the per-row
    # cycle PRICE in the argmin (mac_per_row[op] — the KKT-consistent price of
    # "min Σ_true W·σ s.t. Σ_true macs·cycles ≤ B"; rep_g's extra true/stored
    # factor cancels between a stored row's benefit and cost). The BUDGET side
    # below is unchanged either way (still R_g-weighted, iso-compute).
    if prices is None:
        prices = reps
    global_budget = budget_ratio * ref * total_rows
    min_cost = float(total_rows * costs[-1])
    max_cost = float(total_rows * costs[0])
    if global_budget >= max_cost:
        return 0.0                      # afford highest level everywhere
    if global_budget <= min_cost:
        return 1e12                     # forced to lowest level everywhere

    def total_cost(lmbd: float) -> float:
        t = 0.0
        for e, R, price in zip(weighted_errors, row_counts, prices):
            assign = (e + lmbd * price * costs[None, :]).argmin(axis=1)
            t += float(R) * float(costs[assign].mean())   # R_g · avg_cost_g
        return t

    lo, hi = 0.0, 1.0
    while total_cost(hi) > global_budget and hi < 1e12:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if total_cost(mid) > global_budget:
            lo = mid
        else:
            hi = mid
    return hi


def _thresholds_from_counts(metrics: np.ndarray, counts: np.ndarray) -> list:
    """Convert per-level counts into n_levels-1 thresholds on the sorted metric."""
    if metrics.size == 0:
        return []
    sorted_metrics = np.sort(metrics)[::-1]
    thresholds: list = []
    offset = 0
    for count in counts[:-1]:
        offset += int(count)
        if offset <= 0:
            thresholds.append(1.0)
        elif offset >= len(sorted_metrics):
            thresholds.append(0.0)
        else:
            thresholds.append(
                float(0.5 * (sorted_metrics[offset - 1] + sorted_metrics[offset]))
            )
    return thresholds


# =====================================================================
# Per-level SC kernel invocations (re-implements the SCLinear / attention
# matmul calls used at runtime, but with explicit stoc_len per call).
# =====================================================================


def _sc_linear_at_level(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    sc_prec: int,
    stoc_len: int,
    halve: bool,
    chunk_d: int,
    smooth_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mimic SCLinear.forward for a single explicit stoc_len level.

    ``smooth_scales`` (open-issue #5): when the deployment applies SmoothQuant
    (eval runs USE_SMOOTHQUANT=1 α=0.5), the SC error must be measured on the
    SAME smoothed activation the kernel will quantize (apply_smoothing → a/s),
    else the per-level σ table — and thus the level allocation — is calibrated
    on the wrong (unsmoothed) distribution. None = original (byte-identical).

    Returns float32 output of shape (..., out_features).
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1]).to(torch.float32).contiguous()
    w = weight.to(torch.float32).contiguous()
    out_flat = _sc_matmul(
        x_flat, w,
        granularity="per_row", mode="bipolar",
        sc_prec=sc_prec, stoc_len=int(stoc_len), chunk_d=chunk_d,
        halve_bipolar_stoc_len=halve,
        smooth_scales=smooth_scales,
    )
    out = out_flat.reshape(*orig_shape[:-1], weight.shape[0])
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out


def _sc_linear_protected_at_level(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    protected_indices: list[int],
    *,
    protected_stoc_len: int,
    residual_stoc_len: int,
    sc_prec: int,
    halve: bool,
    chunk_d: int,
    smooth_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linear SC with salient input channels fixed at protected_stoc_len."""
    if not protected_indices:
        return _sc_linear_at_level(
            x, weight, bias, sc_prec=sc_prec, stoc_len=residual_stoc_len,
            halve=halve, chunk_d=chunk_d, smooth_scales=smooth_scales)
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1]).to(torch.float32).contiguous()
    w = weight.to(torch.float32).contiguous()
    idx = torch.tensor(
        sorted({int(i) for i in protected_indices
                if 0 <= int(i) < x_flat.shape[1]}),
        dtype=torch.long, device=x_flat.device)
    if idx.numel() == 0:
        return _sc_linear_at_level(
            x, weight, bias, sc_prec=sc_prec, stoc_len=residual_stoc_len,
            halve=halve, chunk_d=chunk_d, smooth_scales=smooth_scales)
    mask = torch.ones(x_flat.shape[1], dtype=torch.bool, device=x_flat.device)
    mask[idx] = False
    rest = mask.nonzero(as_tuple=True)[0]
    out_flat = torch.zeros(
        (x_flat.shape[0], weight.shape[0]), dtype=torch.float32,
        device=x_flat.device)
    smooth_p = (smooth_scales.index_select(0, idx).contiguous()
                if smooth_scales is not None else None)
    out_flat += _sc_matmul(
        x_flat.index_select(1, idx).contiguous(),
        w.index_select(1, idx).contiguous(),
        granularity="per_row", mode="bipolar",
        sc_prec=sc_prec, stoc_len=int(protected_stoc_len), chunk_d=chunk_d,
        halve_bipolar_stoc_len=halve, smooth_scales=smooth_p)
    if rest.numel() and residual_stoc_len > 0:
        smooth_r = (smooth_scales.index_select(0, rest).contiguous()
                    if smooth_scales is not None else None)
        out_flat += _sc_matmul(
            x_flat.index_select(1, rest).contiguous(),
            w.index_select(1, rest).contiguous(),
            granularity="per_row", mode="bipolar",
            sc_prec=sc_prec, stoc_len=int(residual_stoc_len), chunk_d=chunk_d,
            halve_bipolar_stoc_len=halve, smooth_scales=smooth_r)
    out = out_flat.reshape(*orig_shape[:-1], weight.shape[0])
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out


def _sc_attn_matmul_at_level(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    sc_prec: int,
    stoc_len: int,
    halve: bool,
) -> torch.Tensor:
    """Compute a @ b.T at one stoc_len. a/b: (B, H, N, K) / (B, H, M, K)."""
    B, H, N, K = a.shape
    M = b.shape[-2]
    a3 = a.reshape(B * H, N, K).to(torch.float32).contiguous()
    b3 = b.reshape(B * H, M, K).to(torch.float32).contiguous()
    # Must match the DEPLOYED attention kernel: runtime routes qk/av through
    # per_row (model/sc_common.py `_sc_attention_matmul_ab_t`; per_head was
    # removed 2026-07-03 as catastrophic for small models). Calibrating on
    # per_head would fit qk/av thresholds to a kernel that is never run.
    out3 = _sc_matmul(
        a3, b3,
        granularity="per_row", mode="bipolar",
        sc_prec=sc_prec, stoc_len=int(stoc_len),
        halve_bipolar_stoc_len=halve,
    )
    return out3.reshape(B, H, N, M)


# =====================================================================
# Calibrator: collects (metric, errors_by_level) per (operator, l_bucket)
# and fits per-bucket thresholds at the end.
# =====================================================================


class ThresholdCalibrator:
    def __init__(
        self,
        levels: list,
        operators: Iterable[str],
        total_blocks: int,
        layer_buckets: int,
        budget_ratio: float,
        budget_ref_stoc_len: Optional[int],
        max_units_per_call: int,
        min_bucket_units: int,
        rng_seed: int,
        loss_weight_by_grad: bool = False,
        grad_g_pow: float = 2.0,
        grad_s_pow: float = 2.0,
        budget_scope: str = "per_bucket",
        group_weights: Optional[dict] = None,
        grad_as_group_weight: bool = False,
        grad_group_pow: float = 1.0,
        grad_group_clip: float = 99.0,
        refine_mode: str = "none",
    ):
        # 2nd-stage refinement on top of the (global) solve. "none" = original
        # behaviour, byte-identical. "sigma" = greedy residual-budget fill: the
        # discrete argmin at the feasible-side λ leaves the realized row-weighted
        # avg_sl UNDER target; spend that slack on the rows with the highest
        # marginal error reduction per added cycle (Δσ/Δcost), strictly lowering
        # Σσ while staying ≤ the iso-budget target. See _refine_residual_fill.
        self.refine_mode = str(refine_mode)
        self._refined_assignments: dict = {}
        self.refine_stats: dict = {}
        # Allocation objective. "sigma" = relative-L2 recon error (original);
        # "sigma2" = total squared error; "delta_sigma2" (v9) = squared error
        # INCREASE relative to the highest available stream length. V9 does not
        # spend budget chasing an irreducible high-level error floor and
        # strongly prices the 32->16 degradation cliff.
        self.objective = "sigma"
        # open-issue #5: measure per-level σ on the SmoothQuant-transformed
        # activation (a/s) that the deployed kernel actually quantizes, so the
        # calibration distribution matches eval. Set by main() from the
        # SmoothQuant env/flag; the per-module smooth vector is read off the
        # module buffer in the hook. Off → byte-identical original.
        self.calib_smoothquant = False
        self.levels = levels
        self.operators = set(operators)
        # ---- act_global_v2 knobs (defaults = byte-identical act_global) ----
        # argmin_pricing: what multiplies the cycle cost in the PER-ROW argmin.
        #   "rep"  — rep_g = R_g/n_g (original). Includes the subsample ratio
        #            true/stored, which does NOT belong in the KKT solution of
        #            "min Σ_true W·σ  s.t.  Σ_true macs·cycles ≤ B": that ratio
        #            multiplies a stored row's benefit AND cost equally, so it
        #            cancels in the per-row argmin. Leaving it in over-prices
        #            heavily-subsampled groups' cycles (attention stored rows
        #            stand for ~128× more true rows than dense-linear ones).
        #   "macs" — mac_per_row[op] only: the KKT-consistent price. Budget
        #            accounting is UNCHANGED (still R_g-weighted, iso-compute).
        self.argmin_pricing = "rep"
        # metric_select: "amax" (original dispatch metric, byte-identical) or
        # "auto" — capture candidate metrics (amax/l2/crest) per row and pick,
        # per operator, the SIGNED candidate with the best Spearman ρ against
        # the true σ-benefit. Thresholds are then built on the winner and the
        # choice exported for the runtime (payload "dispatch_metrics").
        self.metric_select = "amax"
        self.metric_select_margin = 0.05
        self.dispatch_metric_selection: dict = {}
        # Cross-layer budget: "per_bucket" (each (op,layer-bucket) pinned to the
        # same avg budget — original) or "global" (one shared λ over all groups,
        # budget flows across layers/operators). group_weights maps
        # (operator, l_bucket) -> W_g for the measured-ΔLoss reweighting; None
        # (or missing key) means W_g=1.
        self.budget_scope = budget_scope
        self.group_weights = group_weights or {}
        self.total_blocks = total_blocks
        self.timestep_buckets = 1
        self.layer_buckets = layer_buckets
        self.budget_ratio = budget_ratio
        self.budget_ref_stoc_len = int(budget_ref_stoc_len) if budget_ref_stoc_len else max(levels)
        self.max_units_per_call = max_units_per_call
        self.min_bucket_units = min_bucket_units
        self.loss_weight_by_grad = bool(loss_weight_by_grad)
        self.grad_g_pow = float(grad_g_pow)
        self.grad_s_pow = float(grad_s_pow)
        # grad_group: instead of multiplying per-row σ by g (which over-
        # concentrates budget on a few high-gradient rows and crosses the
        # collapse cliff), aggregate g to ONE per-(operator, layer-bucket)
        # cross-layer weight W_g and keep act-style σ within group. The per-row
        # gradient noise averages out; g only nudges the cross-layer split.
        self.grad_as_group_weight = bool(grad_as_group_weight)
        self.grad_group_pow = float(grad_group_pow)
        self.grad_group_clip = float(grad_group_clip)
        self._grad_group_sum: dict = defaultdict(float)
        self._grad_group_cnt: dict = defaultdict(int)
        # FisherMP: reparameterization-invariant Gauss-Newton cross-group weight
        # Ŵ_g = winsorized mean_row(‖y_row‖² · ‖g_row‖²). Unlike grad_group's
        # mean|g|, the ‖y‖² factor makes it invariant to y→c·y / g→g/c rescaling,
        # so the post-softmax attention undercount (tiny y, large g) that clamped
        # grad_group's qk buckets to the floor (639 PPL) cannot recur. Linear ops
        # by default; qk/av weighted too when fisher_attn is set (--fisher-attn),
        # each normalized in its OWN pool. Set by main().
        self.fisher_group = False
        self.fisher_attn = False
        self._fisher_sum: dict = defaultdict(float)
        self._fisher_cnt: dict = defaultdict(int)
        self.costs = np.asarray(levels, dtype=np.float64)
        self.records: dict = defaultdict(
            lambda: {"metrics": [], "errors": [], "alts": defaultdict(list)})
        # True (un-subsampled) per-forward row count accumulated per group. The
        # global solve weights each group's budget by this so the realized
        # row-weighted avg_sl hits the target — attention (qk/av) carries far
        # more rows per forward than the linears, and subsampling to
        # max_units_per_call erases that, which otherwise lets the global
        # allocation silently overspend at runtime.
        self.true_counts: dict = defaultdict(int)
        # Budget weighting: "rows" (row-serial LATENCY, original) or "macs"
        # (FLOP/ENERGY). Row-weighting prices the FLOP-heavy linears as cheap
        # (few rows) and the cheap-FLOP attention as expensive (90% of rows), so
        # "iso-budget" is NOT iso-compute and act_global starves qk despite its
        # high σ. MAC-weighting (R_g → R_g·macs_per_row[op]) makes iso-budget
        # iso-compute: linears expensive, attention cheap. mac_per_row is set by
        # main() (per-operator MACs/row from a trace).
        self.budget_weight = "rows"
        self.mac_per_row: dict = {}
        # Offline salient-channel protection. Keys are
        # (operator, block_idx, unit_idx); unit_idx is only used for MoE experts.
        self.protect_channel_frac = 0.0
        self.protect_channel_metric = "none"
        self.protect_channel_stoc_len = 128
        self.protect_compensate_budget = False
        self.protected_channel_indices: dict = {}
        self.protected_channel_dims: dict = {}
        self.protected_channel_scores: dict = {}
        self._rng = np.random.default_rng(rng_seed)

    def use_operator(self, op: str) -> bool:
        return op in self.operators

    def protected_indices_for_module(
        self, operator: str, block_idx: int, unit_idx: Optional[int] = None,
    ) -> list[int]:
        key = (operator, int(block_idx), unit_idx)
        vals = self.protected_channel_indices.get(key)
        if vals is not None:
            return vals
        return self.protected_channel_indices.get((operator, int(block_idx), None), [])

    def _protected_key_to_str(self, key) -> str:
        op, block_idx, unit_idx = key
        s = f"{op}:b{int(block_idx)}"
        if unit_idx is not None:
            s += f":u{int(unit_idx)}"
        return s

    def _protected_frac_for_key(self, key) -> float:
        operator, _t_bucket, l_bucket = key
        vals = []
        for pkey, idxs in self.protected_channel_indices.items():
            op, block_idx, _unit_idx = pkey
            if op != operator:
                continue
            if _bucket_index(int(block_idx), self.total_blocks, self.layer_buckets) != l_bucket:
                continue
            dim = int(self.protected_channel_dims.get(pkey, 0))
            if dim > 0:
                vals.append(float(len(idxs)) / float(dim))
        return float(np.mean(vals)) if vals else 0.0

    def _protected_adjusted_avg(self, key, residual_avg: float) -> float:
        frac = self._protected_frac_for_key(key)
        if frac <= 0.0:
            return float(residual_avg)
        return frac * float(self.protect_channel_stoc_len) + \
            (1.0 - frac) * float(residual_avg)

    def _global_protected_fraction(self) -> float:
        num = den = 0.0
        for key, rec in self.records.items():
            n = sum(int(m.size) for m in rec["metrics"])
            if n < self.min_bucket_units:
                continue
            Rw = float(self._weight_count(key, n))
            num += Rw * self._protected_frac_for_key(key)
            den += Rw
        return (num / den) if den else 0.0

    def add(
        self,
        operator: str,
        block_idx: int,
        metric_norm: torch.Tensor,
        errors_by_level: list,
        grad_rows: Optional[torch.Tensor] = None,
        energy_rows: Optional[torch.Tensor] = None,
        metric_alts: Optional[dict] = None,
    ):
        """Record one matmul call's per-row metric + per-level SC error.

        When ``loss_weight_by_grad`` is on, ``grad_rows`` (= ‖∂L/∂y_row‖₂, one
        scalar per output row, captured by a backward hook on the SAME output
        tensor) must be supplied and aligned row-for-row with ``metric_norm``
        and ``errors_by_level``. The error matrix stored for the Lagrangian is
        then  (g_row**2) * (sigma_row_level**2)  so that the budget solver
        minimizes the loss-sensitive objective  Σ g_row² · σ²_row(level).

        Grad-weighting and the subsample MUST be applied to the same rows, so
        we merge ``grad_rows`` BEFORE the ``max_units_per_call`` subsample.
        """
        if not self.use_operator(operator):
            return
        metrics = metric_norm.detach().float().reshape(-1).cpu().numpy()
        err = torch.stack(
            [e.detach().float().reshape(-1) for e in errors_by_level], dim=-1
        )
        errors = err.cpu().numpy()
        if metrics.size == 0:
            return
        if self.loss_weight_by_grad:
            if grad_rows is None:
                raise RuntimeError(
                    f"loss_weight_by_grad is on but no grad captured for "
                    f"operator={operator} block_idx={block_idx}; the backward "
                    f"hook on the output tensor did not fire."
                )
            g = grad_rows.detach().float().reshape(-1).cpu().numpy()
            if g.shape[0] != metrics.shape[0]:
                raise RuntimeError(
                    f"grad rows ({g.shape[0]}) misaligned with metric rows "
                    f"({metrics.shape[0]}) for operator={operator} "
                    f"block_idx={block_idx}."
                )
            if self.fisher_group:
                # FisherMP: Ŵ_g = winsorized mean(‖y_row‖²·‖g_row‖²). g_row is the
                # L2 grad so g**2 = ‖g‖². energy_rows = ‖y_row‖² (captured for
                # linear ops, and qk/av too under --fisher-attn; otherwise attention
                # leaves energy None → no fisher weight → W_g=1). errors falls
                # through unchanged (act σ within group).
                if energy_rows is not None:
                    e = energy_rows.detach().float().reshape(-1).cpu().numpy()
                    if e.shape[0] == g.shape[0]:
                        fw = e * (g ** 2)
                        if 0.0 < self.grad_group_clip < 100.0 and fw.size:
                            cap = float(np.percentile(fw, self.grad_group_clip))
                            fw = np.minimum(fw, cap)
                        lb_f = _bucket_index(block_idx, self.total_blocks, self.layer_buckets)
                        self._fisher_sum[(operator, lb_f)] += float(fw.sum())
                        self._fisher_cnt[(operator, lb_f)] += int(fw.size)
            elif self.grad_as_group_weight:
                # grad_group: do NOT reweight per-row errors (keep act σ). Instead
                # accumulate a robust per-(op, layer-bucket) loss-sensitivity that
                # becomes the cross-layer weight W_g. Winsorize per call so a few
                # outlier rows cannot dominate the group mean. errors falls
                # through unchanged → within-group allocation stays act-style.
                gp = g ** self.grad_group_pow
                if 0.0 < self.grad_group_clip < 100.0 and gp.size:
                    cap = float(np.percentile(gp, self.grad_group_clip))
                    gp = np.minimum(gp, cap)
                lb_g = _bucket_index(block_idx, self.total_blocks, self.layer_buckets)
                self._grad_group_sum[(operator, lb_g)] += float(gp.sum())
                self._grad_group_cnt[(operator, lb_g)] += int(gp.size)
            else:
                # Objective term per (row, level): g_row^gp · σ_row(level)^sp.
                # Defaults (gp=sp=2) give the Gauss-Newton Σ g²σ² objective.
                # Setting gp=sp=1 gives the softened Σ g·σ (which makes act the
                # literal g≡1 special case and avoids over-concentrating budget
                # on a few high-gradient rows). Downstream solver is unchanged.
                errors = (g[:, None] ** self.grad_g_pow) * (errors ** self.grad_s_pow)
        # act_global_v2 candidate metrics (per-call NORMALIZED, aligned with
        # metric_norm row-for-row). Subsampled with the SAME keep indices so
        # candidates stay aligned with errors for the ρ selection.
        alts = {}
        if metric_alts:
            for name, alt in metric_alts.items():
                a = alt.detach().float().reshape(-1).cpu().numpy()
                if a.shape[0] != metrics.shape[0]:
                    raise RuntimeError(
                        f"metric_alts[{name}] rows ({a.shape[0]}) misaligned "
                        f"with metric rows ({metrics.shape[0]}) for "
                        f"operator={operator} block_idx={block_idx}.")
                alts[name] = a
        true_size = int(metrics.size)   # before subsampling — the runtime weight
        if self.max_units_per_call > 0 and metrics.size > self.max_units_per_call:
            keep = self._rng.choice(metrics.size, self.max_units_per_call, replace=False)
            metrics = metrics[keep]
            errors = errors[keep]
            alts = {name: a[keep] for name, a in alts.items()}
        l_bucket = _bucket_index(block_idx, self.total_blocks, self.layer_buckets)
        key = (operator, 0, l_bucket)
        self.records[key]["metrics"].append(metrics)
        self.records[key]["errors"].append(errors)
        for name, a in alts.items():
            self.records[key]["alts"][name].append(a)
        self.true_counts[key] += true_size

    def _weight_for_key(self, key) -> float:
        """Per-group importance weight W_g for the global cross-layer solve."""
        operator, _, l_bucket = key
        return float(self.group_weights.get((operator, l_bucket), 1.0))

    def _op_weight(self, operator) -> float:
        """Mean W_g over an operator's buckets (for the operator_default fallback)."""
        ws = [w for (op, _lb), w in self.group_weights.items() if op == operator]
        return float(np.mean(ws)) if ws else 1.0

    def grad_group_raw_weights(self) -> dict:
        """Per-(operator, layer-bucket) mean gradient importance for grad_group.

        Returns {(operator, l_bucket): mean_rows(g_row**grad_group_pow)} over all
        calibration rows seen (winsorized per call in ``add``). Fed through
        ``_normalize_group_weights`` to become the cross-layer W_g — the same
        slot the measured-ΔLoss probe fills, but sourced from one cheap backward
        instead of |ops|·buckets probe forwards."""
        return {k: self._grad_group_sum[k] / max(self._grad_group_cnt[k], 1)
                for k in self._grad_group_sum}

    def fisher_group_raw_weights(self) -> dict:
        """Per-(operator, layer-bucket) mean Gauss-Newton importance for FisherMP:
        mean_row(‖y_row‖²·‖g_row‖²), winsorized per call in ``add``. Fed through
        the bounded-blend in main() to become the cross-layer W_g. Only linear
        groups are populated (attention energy is not captured → W_g=1 there)."""
        return {k: self._fisher_sum[k] / max(self._fisher_cnt[k], 1)
                for k in self._fisher_sum}

    def apply_measured_curve(self, curves: dict) -> int:
        """measured_curve: replace each group's per-row recon-error matrix with
        that group's per-LEVEL ΔLoss curve (broadcast to all its rows), so the
        global-λ solve minimizes  Σ ΔLoss + λ·budget  DIRECTLY — the true
        loss-optimal allocation — instead of W_g × recon-error (which leaks
        recon-error's blindness to qk). Because the curve is constant across a
        group's rows, the per-row argmin lands every row on one level → per-group
        (operator × layer-bucket) uniform. W_g stays 1. Returns #groups overridden.
        Must be called AFTER the σ-collection (records populated) and BEFORE
        export()."""
        self.group_weights = {}          # ΔLoss is IN the errors now → no W_g tilt
        self._curve_applied = True       # enables the residual fill in export()
        L = len(self.levels)
        n_over = 0
        for key, rec in self.records.items():
            op, _, lb = key
            curve = curves.get((op, lb))
            if curve is None or not rec["metrics"]:
                continue
            n = int(sum(int(m.size) for m in rec["metrics"]))
            c = np.clip(np.asarray(curve, dtype=np.float64).reshape(1, L), 0.0, None)
            rec["errors"] = [np.tile(c, (n, 1))]   # [n, L], every row = the curve
            n_over += 1
        return n_over

    def metric_fidelity_rho(self) -> dict:
        """Diagnostic (the near-lossless lever). Runtime assigns each row a level
        by RANKING it on the dispatch metric (|x|.amax). This measures how well
        that ranking matches the row's TRUE need for cycles: Spearman ρ(metric,
        objective-benefit) per operator, where benefit = objective(min level) −
        objective(max level). Under the default objective this is the raw
        reconstruction-error benefit. ρ≈1 ⇒ metric ranks correctly
        (granularity is the only lever left); ρ low/negative ⇒ the METRIC
        misranks rows and a better metric — not more levels/buckets — is the win.
        av is the prime suspect (peak attention weight vs σ)."""
        per_op = defaultdict(lambda: {"m": [], "b": []})
        for key, rec in self.records.items():
            if not rec["metrics"]:
                continue
            m = np.concatenate(rec["metrics"], axis=0)
            e = np.concatenate(rec["errors"], axis=0)     # [n, L] descending levels
            eo = _objective_errors(e, self.objective)
            benefit = eo[:, -1] - eo[:, 0]  # objective benefit: min L -> max L
            per_op[key[0]]["m"].append(m)
            per_op[key[0]]["b"].append(benefit)
        out = {}
        for op, d in per_op.items():
            m = np.concatenate(d["m"]); b = np.concatenate(d["b"])
            out[op] = {"rho": _spearman(m, b), "n": int(m.size),
                       "benefit_mean": float(b.mean())}
        return out

    def select_dispatch_metrics(self) -> dict:
        """act_global_v2 (--metric-select auto): per-operator SIGNED dispatch-
        metric selection. For each operator, pool all buckets and compute
        Spearman ρ(candidate, objective-benefit) for amax + every captured alt; pick
        the candidate with the largest |ρ| (negative ρ ⇒ deploy with sign −1,
        which inverts the ranking). Switch away from raw amax only when the
        winner clears the incumbent by ``metric_select_margin`` (noise guard).

        On a switch, the stored per-bucket metric arrays are REWRITTEN to the
        selected candidate (sign −1 ⇒ 1−m, exactly what the runtime's negated-
        metric min–max normalization produces), so thresholds, summaries, and
        the post-selection ρ diagnostic all reflect the deployed metric. The
        choice is exported via payload["dispatch_metrics"] for the runtime.
        Assignments/budget are untouched — the metric only maps counts to
        thresholds."""
        if self.metric_select != "auto":
            return {}
        per_op: dict = defaultdict(lambda: defaultdict(lambda: {"m": [], "b": []}))
        for key, rec in self.records.items():
            if not rec["metrics"]:
                continue
            e = np.concatenate(rec["errors"], axis=0)
            eo = _objective_errors(e, self.objective)
            b = eo[:, -1] - eo[:, 0]  # objective benefit: min L -> max L
            cands = {"amax": np.concatenate(rec["metrics"], axis=0)}
            for name, lst in rec["alts"].items():
                if lst:
                    cands[name] = np.concatenate(lst, axis=0)
            for name, m in cands.items():
                if m.shape[0] != b.shape[0]:
                    # Partial capture (e.g. a call whose channels were all
                    # protected recorded no alts). Drop this key's contribution
                    # for this candidate — never kill an hours-long calibration
                    # over a diagnostic candidate; amax always stays available.
                    print(f"[calib] WARN metric-select: candidate '{name}' "
                          f"misaligned for group {key} "
                          f"({m.shape[0]} vs {b.shape[0]} rows) — skipped.")
                    continue
                per_op[key[0]][name]["m"].append(m)
                per_op[key[0]][name]["b"].append(b)
        chosen: dict = {}
        for op, cands in per_op.items():
            rhos = {name: _spearman(np.concatenate(d["m"]),
                                    np.concatenate(d["b"]))
                    for name, d in cands.items()}
            base = float(rhos.get("amax", 0.0))
            best_name = max(rhos, key=lambda n: abs(rhos[n]))
            best_rho = float(rhos[best_name])
            sign = 1.0 if best_rho >= 0.0 else -1.0
            if best_name == "amax" and sign > 0.0:
                continue                            # incumbent stands
            if abs(best_rho) < max(base, 0.0) + self.metric_select_margin:
                continue                            # not clearly better
            # A switched candidate must cover EVERY stored row of the op:
            # thresholds are quantiles of the metric array aligned with the
            # error-based counts, so a partial-capture candidate would shift
            # threshold occupancy. If coverage is incomplete, keep amax.
            if best_name != "amax":
                covered = all(
                    sum(int(a.size) for a in rec["alts"][best_name])
                    == sum(int(m.size) for m in rec["metrics"])
                    for key, rec in self.records.items() if key[0] == op)
                if not covered:
                    print(f"[calib] WARN metric-select: '{best_name}' wins on "
                          f"{op} but has partial capture — keeping amax.")
                    continue
            for key, rec in self.records.items():
                if key[0] != op:
                    continue
                src = (rec["metrics"] if best_name == "amax"
                       else rec["alts"][best_name])
                rec["metrics"] = [(1.0 - m) if sign < 0.0 else m for m in src]
            chosen[op] = {"metric": best_name, "sign": sign,
                          "rho_signed": abs(best_rho), "rho_amax": base}
        self.dispatch_metric_selection = chosen
        return chosen

    def _fit_group(self, metrics: np.ndarray, errors: np.ndarray,
                   lam: Optional[float] = None, weight: float = 1.0,
                   rep: float = 1.0,
                   override_assignment: Optional[np.ndarray] = None) -> dict:
        if override_assignment is not None:
            # sigma_refine 2nd stage: use the greedily-refined per-row levels
            # instead of the raw λ argmin (counts/thresholds recomputed below).
            assignment = np.asarray(override_assignment, dtype=np.int64)
        elif lam is None:
            # Per-group budget: pin this group to budget_ratio·ref average.
            budget_total = self.budget_ratio * self.budget_ref_stoc_len * metrics.size
            assignment = _cost_assignments(
                _objective_errors(errors, self.objective),
                self.costs,
                budget_total,
            )
        else:
            # Global cross-layer: assign at the shared price λ, with the cost
            # scaled by rep_g = R_g/n_g so cheap (few-row) groups buy precision
            # cheaply (consistent with the R_g-weighted budget — see
            # _global_lambda). weight = W_g importance; within-group quantile is
            # act-style (rep/weight are per-group constants).
            err_obj = _objective_errors(errors, self.objective)
            objective = (weight * err_obj) + lam * rep * self.costs[None, :]
            assignment = objective.argmin(axis=1)
        counts = np.bincount(assignment, minlength=len(self.levels))
        thresholds = _thresholds_from_counts(metrics, counts)
        avg_cost = float(self.costs[assignment].mean()) if assignment.size else 0.0
        avg_error = float(errors[np.arange(errors.shape[0]), assignment].mean()) if assignment.size else 0.0
        level_mean_error = [float(errors[:, i].mean()) for i in range(errors.shape[1])]
        return {
            "num_units": int(metrics.size),
            "counts": counts.tolist(),
            "fractions": (counts / max(metrics.size, 1)).tolist(),
            "thresholds": thresholds,
            "avg_stoc_len": avg_cost,
            "avg_error": avg_error,
            "level_mean_error": level_mean_error,
            "metric_mean": float(metrics.mean()) if metrics.size else 0.0,
            "metric_std": float(metrics.std()) if metrics.size else 0.0,
        }

    def export(self):
        summary_rows: list = []
        payload = {
            "stoc_len_levels": self.levels,
            "budget_ratio": self.budget_ratio,
            "budget_ref_stoc_len": self.budget_ref_stoc_len,
            "timestep_buckets": self.timestep_buckets,
            "layer_buckets": self.layer_buckets,
            "budget_scope": self.budget_scope,
            "operator_defaults": {},
            "buckets": {},
        }
        # act_global_v2 provenance + the runtime dispatch-metric switch table
        # (AdaptiveMPConfig.load_threshold_table consumes "dispatch_metrics";
        # absent/empty ⇒ runtime uses the original amax everywhere).
        payload["argmin_pricing"] = self.argmin_pricing
        if self.dispatch_metric_selection:
            payload["dispatch_metrics"] = self.dispatch_metric_selection
        # Global cross-layer: find ONE shared λ over all per-bucket groups so the
        # budget can flow across layers/operators. Per-bucket scope leaves λ=None
        # (each group independently pinned to budget_ratio·ref — original).
        if self.protected_channel_indices:
            protected = {
                "stoc_len": int(self.protect_channel_stoc_len),
                "frac": float(self.protect_channel_frac),
                "metric": self.protect_channel_metric,
                "compensate_budget": bool(self.protect_compensate_budget),
                "indices": {
                    self._protected_key_to_str(k): [int(i) for i in v]
                    for k, v in sorted(self.protected_channel_indices.items(),
                                       key=lambda kv: self._protected_key_to_str(kv[0]))
                },
                "dims": {
                    self._protected_key_to_str(k): int(v)
                    for k, v in sorted(self.protected_channel_dims.items(),
                                       key=lambda kv: self._protected_key_to_str(kv[0]))
                },
                "score_summary": {
                    self._protected_key_to_str(k): v
                    for k, v in sorted(self.protected_channel_scores.items(),
                                       key=lambda kv: self._protected_key_to_str(kv[0]))
                },
            }
            p_global = self._global_protected_fraction()
            protected["global_mac_weighted_frac"] = float(p_global)
            protected["uncompensated_target_avg_stoc_len"] = (
                float(self.budget_ratio) * float(self.budget_ref_stoc_len)
                + p_global * (float(self.protect_channel_stoc_len)
                              - float(self.budget_ratio) * float(self.budget_ref_stoc_len))
            )
            if self.protect_compensate_budget and p_global > 0.0:
                target = float(self.budget_ratio) * float(self.budget_ref_stoc_len)
                rest_target = (
                    target - p_global * float(self.protect_channel_stoc_len)
                ) / max(1.0 - p_global, 1e-12)
                if rest_target <= 0.0:
                    raise SystemExit(
                        "[calib] protected-channel compensation produced "
                        f"non-positive residual target {rest_target:.4f}; "
                        "lower --protect-channel-frac or stoc_len.")
                protected["original_budget_ratio"] = float(self.budget_ratio)
                protected["residual_budget_ratio"] = (
                    float(rest_target) / float(self.budget_ref_stoc_len))
                protected["residual_target_avg_stoc_len"] = float(rest_target)
                self.budget_ratio = protected["residual_budget_ratio"]
                payload["budget_ratio"] = self.budget_ratio
            payload["protected_channels"] = protected
        global_lam = None
        if self.budget_scope == "global":
            werrs, row_counts, prices = [], [], []
            for key, rec in self.records.items():
                e = _objective_errors(
                    np.concatenate(rec["errors"], axis=0), self.objective)
                werrs.append(self._weight_for_key(key) * e)
                # Per-group budget weight R_g (un-subsampled): rows (latency) or
                # MACs (FLOP/energy) per self.budget_weight. Fall back to the
                # stored count if (unexpectedly) missing.
                row_counts.append(self._weight_count(key, e.shape[0]))
                prices.append(self._price_for_key(key))
            global_lam = _global_lambda(
                werrs, self.costs, self.budget_ratio,
                float(self.budget_ref_stoc_len), row_counts, prices=prices)
            payload["global_lambda"] = float(global_lam)
            if self.refine_mode == "sigma" or getattr(self, "_curve_applied", False):
                # measured_curve: the per-group-constant ΔLoss curve makes the
                # cost landscape coarsely discrete, so the feasible-side λ
                # bisection leaves budget slack (measured 2.7–7% under target)
                # that nothing refilled — the residual greedy fill (MAC-priced
                # via rep_g) closes it so measured_curve is honestly iso-budget.
                self._refined_assignments = self._refine_residual_fill(global_lam)
                payload["refine_mode"] = ("sigma" if self.refine_mode == "sigma"
                                          else "curve_residual_fill")
                payload["refine_stats"] = self.refine_stats
        return self._export_body(payload, summary_rows, global_lam)

    def _refine_residual_fill(self, global_lam) -> dict:
        """sigma_refine 2nd stage (opt-in). act_global's per-row levels come from
        a hard argmin at the feasible-side shared λ, so the realized row-weighted
        avg_sl lands UNDER the iso-budget target (e.g. 63.2 vs 64). This spends
        the residual budget optimally: greedily upgrade the rows with the largest
        marginal error reduction per added cycle (Δσ/Δcost) until the target is
        reached. For a per-row error curve with diminishing returns this marginal
        greedy fill is the optimal way to spend the residual — it strictly lowers
        the calibrator's own Σσ objective while keeping avg_sl ≤ target, so it is
        a guaranteed, iso-budget-fair improvement over act_global.

        Operates ONLY on the per-bucket groups that will be exported (size ≥
        min_bucket_units), i.e. exactly the groups counted in
        expected_avg_stoc_len and used by the runtime per-bucket dispatch.
        Returns {key: refined_assignment}."""
        import heapq
        costs = self.costs                      # e.g. [128., 64., 32.]
        keys = [k for k in self.records
                if sum(int(m.size) for m in self.records[k]["metrics"])
                >= self.min_bucket_units]
        data, total_R, realized = {}, 0.0, 0.0
        for key in keys:
            errors = np.concatenate(self.records[key]["errors"], axis=0)  # raw sigma [n,L]
            objective_errors = _objective_errors(errors, self.objective)
            w = self._weight_for_key(key)
            rep = self._rep_for_key(key)
            # act_global_v2/v3: the argmin + greedy RANKING use the KKT price
            # (mac_per_row under --argmin-pricing macs; == rep in legacy mode,
            # keeping the original behavior byte-identical). BUDGET accounting
            # (realized/target/affordability) always uses the true rep_g so the
            # fill stays exactly iso-compute.
            price = self._price_for_key(key)
            R = self._weight_count(key, errors.shape[0])
            obj = (w * objective_errors) + global_lam * price * costs[None, :]
            assign = obj.argmin(axis=1).astype(np.int64)   # == _fit_group's argmin
            data[key] = {
                "errors": errors,
                "objective_errors": objective_errors,
                "w": w,
                "rep": rep,
                "price": price,
                "assign": assign,
            }
            total_R += R
            realized += rep * float(costs[assign].sum())    # = R_g · mean_row cost
        target = self.budget_ratio * self.budget_ref_stoc_len * total_R
        residual = target - realized
        base_avg = (realized / total_R) if total_R else 0.0
        if residual <= 0 or not keys:
            self.refine_stats = {"note": "no residual (>=target); no-op",
                                 "base_avg_sl": base_avg, "residual": residual}
            return {k: data[k]["assign"] for k in keys}
        # Max-heap (min-heap on -priority) of single-step upgrades j -> j-1
        # (fewer index = more cycles). Each row keeps exactly one live entry.
        heap = []
        for key in keys:
            d = data[key]
            for i in range(d["assign"].shape[0]):
                j = int(d["assign"][i])
                if j > 0:
                    dsig = d["w"] * (
                        float(d["objective_errors"][i, j])
                        - float(d["objective_errors"][i, j - 1]))
                    drank = d["price"] * (float(costs[j - 1]) - float(costs[j]))
                    if drank > 0 and dsig > 0:
                        heapq.heappush(heap, (-(dsig / drank), key, i))
        spent = upgrades = 0
        spent_cost = 0.0
        while heap and residual > 1e-9:
            _negpri, key, i = heapq.heappop(heap)
            d = data[key]
            j = int(d["assign"][i])
            if j <= 0:
                continue
            dcost = d["rep"] * (float(costs[j - 1]) - float(costs[j]))
            # Next step (j-1 -> j-2) costs strictly MORE for pow2-spaced levels,
            # so an unaffordable row is done (safe to drop without re-push).
            if dcost > residual + 1e-9:
                continue
            d["assign"][i] = j - 1
            residual -= dcost
            spent_cost += dcost
            upgrades += 1
            if j - 1 > 0:
                dsig2 = d["w"] * (
                    float(d["objective_errors"][i, j - 1])
                    - float(d["objective_errors"][i, j - 2]))
                drank2 = d["price"] * (float(costs[j - 2]) - float(costs[j - 1]))
                if drank2 > 0 and dsig2 > 0:
                    heapq.heappush(heap, (-(dsig2 / drank2), key, i))
        refined = {k: data[k]["assign"] for k in keys}
        new_realized = 0.0
        for key in keys:
            new_realized += data[key]["rep"] * float(costs[data[key]["assign"]].sum())
        self.refine_stats = {
            "base_avg_sl": base_avg,
            "target_avg_sl": self.budget_ratio * self.budget_ref_stoc_len,
            "refined_avg_sl": (new_realized / total_R) if total_R else 0.0,
            "upgrades": int(upgrades),
            "residual_cycles_left": float(residual),
            "n_groups_refined": len(keys),
        }
        return refined

    def _weight_count(self, key, n_fallback=None) -> float:
        """Per-group budget weight R_g: row count (default) OR MAC count
        (budget_weight='macs' → R_g·macs_per_row[op]). Drives the global-λ budget
        + rep_g, so with MAC weighting iso-budget = iso-compute."""
        n = (n_fallback if n_fallback is not None
             else sum(int(m.size) for m in self.records[key]["metrics"]))
        R = float(self.true_counts.get(key, n))
        if self.budget_weight == "macs":
            R *= float(self.mac_per_row.get(key[0], 1.0))
        return R

    def _rep_for_key(self, key) -> float:
        """rep_g = R_g / n_g for one (op, l_bucket) group (cost scale for the
        global assignment). n_g = stored subsampled row count. R_g is row- or
        MAC-weighted per self.budget_weight."""
        n = sum(int(m.size) for m in self.records[key]["metrics"])
        return self._weight_count(key, n) / max(n, 1)

    def _rep_for_op(self, operator) -> float:
        """rep for the operator_default fallback (sum over its buckets). R_g is
        row- or MAC-weighted per self.budget_weight."""
        n = R = 0.0
        for key, rec in self.records.items():
            if key[0] != operator:
                continue
            ng = sum(int(m.size) for m in rec["metrics"])
            n += ng
            R += self._weight_count(key, ng)
        return (R / max(n, 1)) if n else 1.0

    def _price_for_key(self, key) -> float:
        """Per-row cycle PRICE for the argmin (act_global_v2). "macs" prices a
        stored row's cycle by mac_per_row[op] — the KKT-consistent price (the
        subsample ratio in rep_g cancels between benefit and cost). "rep" is
        the original rep_g pricing."""
        if self.argmin_pricing == "macs":
            return float(self.mac_per_row.get(key[0], 1.0))
        return self._rep_for_key(key)

    def _price_for_op(self, operator) -> float:
        if self.argmin_pricing == "macs":
            return float(self.mac_per_row.get(operator, 1.0))
        return self._rep_for_op(operator)

    def _export_body(self, payload, summary_rows, global_lam):
        # Operator-level fallback (concat all layer buckets per op).
        for operator in sorted(self.operators):
            op_metrics, op_errors = [], []
            for (op, _, _), rec in self.records.items():
                if op != operator:
                    continue
                op_metrics.extend(rec["metrics"])
                op_errors.extend(rec["errors"])
            if op_metrics:
                fitted = self._fit_group(
                    np.concatenate(op_metrics, axis=0),
                    np.concatenate(op_errors, axis=0),
                    lam=global_lam, weight=self._op_weight(operator),
                    rep=self._price_for_op(operator),
                )
                payload["operator_defaults"][operator] = fitted
                summary_rows.append({
                    "scope": "operator_default",
                    "operator": operator,
                    "t_bucket": -1,
                    "l_bucket": -1,
                    **_flatten_summary(fitted),
                })
        # Per-bucket entries.
        for key in sorted(self.records.keys()):
            metrics_list = self.records[key]["metrics"]
            errors_list = self.records[key]["errors"]
            metrics = np.concatenate(metrics_list, axis=0)
            errors = np.concatenate(errors_list, axis=0)
            if metrics.size < self.min_bucket_units:
                continue
            operator, t_bucket, l_bucket = key
            fitted = self._fit_group(metrics, errors, lam=global_lam,
                                     weight=self._weight_for_key(key),
                                     rep=self._price_for_key(key),
                                     override_assignment=self._refined_assignments.get(key))
            pfrac = self._protected_frac_for_key(key)
            if pfrac > 0.0:
                fitted["protected_channel_frac"] = pfrac
                fitted["protected_channel_stoc_len"] = int(self.protect_channel_stoc_len)
                fitted["avg_stoc_len_with_protection"] = self._protected_adjusted_avg(
                    key, fitted["avg_stoc_len"])
            bucket_key = f"{operator}:t{t_bucket}:l{l_bucket}"
            payload["buckets"][bucket_key] = fitted
            summary_rows.append({
                "scope": "bucket",
                "operator": operator,
                "t_bucket": t_bucket,
                "l_bucket": l_bucket,
                **_flatten_summary(fitted),
            })
        # Predicted runtime row-weighted average stoc_len (weights each bucket's
        # avg by its true per-forward row count). For per_bucket scope this is
        # ≈ budget_ratio·ref by construction; for global scope it's the honest
        # iso-budget check — should also land near budget_ratio·ref.
        num = den = 0.0
        for bkey, fitted in payload["buckets"].items():
            operator, _t, lpart = bkey.split(":")
            l = int(lpart[1:])
            R = float(self.true_counts.get((operator, 0, l), fitted["num_units"]))
            avg = float(fitted.get("avg_stoc_len_with_protection",
                                   fitted["avg_stoc_len"]))
            num += R * avg
            den += R
        payload["expected_avg_stoc_len"] = (num / den) if den else 0.0
        # Budget-weighted expectation — under --budget-weight macs this is the
        # TRUE iso-compute (FLOP-avg) check. Summaries must read THIS, not the
        # operator_defaults fallback (whose mean-W_g fit is outside the joint
        # budget and produced the phantom "measured overspends" numbers).
        if self.budget_weight == "macs":
            fnum = fden = 0.0
            for bkey, fitted in payload["buckets"].items():
                operator, _t, lpart = bkey.split(":")
                l = int(lpart[1:])
                Rw = float(self._weight_count((operator, 0, l),
                                              fitted["num_units"]))
                avg = float(fitted.get("avg_stoc_len_with_protection",
                                       fitted["avg_stoc_len"]))
                fnum += Rw * avg
                fden += Rw
            payload["expected_flop_avg_stoc_len"] = (fnum / fden) if fden else 0.0
            # Per-op MACs/row (from --mac-weights-trace): lets the RUNTIME
            # tracker accumulate a realized FLOP-avg so the transfer check can
            # move to compute units (row-avg then demotes to a diagnostic).
            payload["mac_per_row"] = {op: float(v)
                                      for op, v in self.mac_per_row.items()}
        return payload, summary_rows


def _flatten_summary(fitted: dict) -> dict:
    row = {
        "num_units": fitted["num_units"],
        "avg_stoc_len": fitted["avg_stoc_len"],
        "avg_error": fitted["avg_error"],
        "metric_mean": fitted["metric_mean"],
        "metric_std": fitted["metric_std"],
    }
    row["counts"] = ",".join(str(int(x)) for x in fitted["counts"])
    row["fractions"] = ",".join(f"{float(x):.6f}" for x in fitted["fractions"])
    row["thresholds"] = ",".join(f"{float(x):.6f}" for x in fitted["thresholds"])
    row["level_mean_error"] = ",".join(f"{float(x):.6f}" for x in fitted["level_mean_error"])
    return row


def _write_summary_csv(path: str, rows: list):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# Hook wiring
# =====================================================================


class PendingMerger:
    """Holds per-call (op, block_idx, metric, errors) captured in the forward
    pass plus the per-row output gradient captured in the backward pass, and
    feeds them to ``calibrator.add`` after the backward completes.

    When grad-weighting is OFF this is a thin pass-through: each forward hook
    flushes its record to the calibrator immediately (g_row ≡ 1, behaviour is
    byte-identical to the original code path). When grad-weighting is ON the
    forward hook stashes the record under a unique call id and registers a
    backward hook on the same output tensor that stores ‖∂L/∂y_row‖₂ under that
    id; ``flush`` then merges the two by id and calls ``calibrator.add`` with
    the aligned ``grad_rows``.
    """

    def __init__(self, calibrator: ThresholdCalibrator):
        self.calibrator = calibrator
        self.enabled = calibrator.loss_weight_by_grad
        self._pending: dict = {}
        self._next_id = 0
        # Per-window draw accumulator, keyed by the STABLE (operator, block_idx)
        # identity of a matmul call (each fires exactly once per forward, so the
        # key is unique within a window and consistent across noise draws). Used
        # to average g_row over multiple SC noise draws before feeding the
        # calibrator. metric/errors are taken from the FIRST draw; g_row summed.
        self._accum: dict = {}

    def _new_id(self) -> int:
        cid = self._next_id
        self._next_id += 1
        return cid

    @staticmethod
    def _row_grad(grad_y: torch.Tensor) -> torch.Tensor:
        """Default g_row reduction: collapse all leading dims into rows and
        take the L2 over the trailing feature dim. ‖∂L/∂y_row‖₂."""
        gy = grad_y.detach()
        return gy.reshape(-1, gy.shape[-1]).norm(dim=-1)

    def record(self, operator, block_idx, metric_norm, errors_by_level, output,
               grad_reduce=None, metric_alts=None):
        """Called from a forward hook. Returns nothing.

        OFF: forward immediately to the calibrator (g_row ≡ 1).
        ON : stash and arm a backward hook on ``output`` to capture g_row.

        ``grad_reduce`` maps the raw output gradient to one g_row scalar per
        calibrator row. Defaults to ``_row_grad`` (token-row L2), which matches
        the SCLinear per-row units; the attention path overrides it (per-head
        for qk, per-(B*H*N)-row for av) so g_row stays aligned with the metric.
        ``metric_alts`` (act_global_v2): dict of per-call-normalized candidate
        metric tensors aligned row-for-row with ``metric_norm``.
        """
        if not self.enabled:
            self.calibrator.add(operator, block_idx, metric_norm,
                                errors_by_level, metric_alts=metric_alts)
            return
        if grad_reduce is None:
            grad_reduce = self._row_grad
        cid = self._new_id()
        # FisherMP: capture per-row output energy ‖y_row‖² (FP-teacher output y is
        # deterministic, so no draw-averaging needed). Linear ops always; qk/av too
        # when --fisher-attn (else attention stays energy=None → W_g=1). For qk/av
        # `output` is the qk_node/av_node, so output.reshape(-1, output.shape[-1])
        # yields the same B*H*N rows as the attention grad reduce → ‖y‖²‖g‖² aligns.
        energy = None
        _energy_ops = LINEAR_OPS | (
            ATTN_OPS if getattr(self.calibrator, "fisher_attn", False) else set())
        if getattr(self.calibrator, "fisher_group", False) and operator in _energy_ops:
            with torch.no_grad():
                y = output.detach().reshape(-1, output.shape[-1]).float()
                energy = (y * y).sum(dim=-1)          # ‖y_row‖²
        self._pending[cid] = {
            "operator": operator,
            "block_idx": block_idx,
            "metric": metric_norm.detach(),
            "errors": [e.detach() for e in errors_by_level],
            "grad": None,
            "energy": energy,
            "alts": ({name: a.detach() for name, a in metric_alts.items()}
                     if metric_alts else None),
            "n_rows": int(metric_norm.reshape(-1).shape[0]),
        }
        if output.requires_grad:
            def _grad_hook(grad_y, cid=cid, reduce=grad_reduce):
                self._pending[cid]["grad"] = reduce(grad_y)
                return None
            output.register_hook(_grad_hook)

    def flush(self):
        """Drain one draw's pending records into the per-window accumulator.

        No-op when grad-weighting is OFF (records were already flushed in
        ``record``). When ON, each pending record's g_row is added into
        ``self._accum`` keyed by call position (cid; unique per call so MoE
        experts that share op+block_idx don't collide); metric/errors are kept
        from the first draw. ``finalize`` then averages g_row over the draws
        seen and feeds the calibrator. With a single draw this is exactly the
        old behaviour (g_row passed through unchanged, same add order / RNG
        consumption), so the FP-grad path stays byte-identical.
        """
        if not self.enabled:
            return
        for cid, rec in self._pending.items():
            grad = rec["grad"]
            if grad is None:
                raise RuntimeError(
                    f"no output gradient captured for operator={rec['operator']} "
                    f"block_idx={rec['block_idx']}; loss.backward() did not reach "
                    f"this matmul output (check enable_input_require_grads / "
                    f"the autograd graph)."
                )
            g = grad.reshape(-1).float()
            if g.shape[0] != rec["n_rows"]:
                raise RuntimeError(
                    f"grad rows ({g.shape[0]}) misaligned with metric rows "
                    f"({rec['n_rows']}) for operator={rec['operator']} "
                    f"block_idx={rec['block_idx']}."
                )
            # Key by CALL POSITION (cid), not (operator, block_idx). In a MoE
            # layer many experts share the same op-name + block_idx but receive
            # DIFFERENT token counts, so (op, block_idx) is not unique per call
            # and summing their g_rows crashes on the size mismatch. cid is the
            # forward-order call index (reset each flush), so it is unique within
            # a draw AND stable across draws (routing is deterministic — the gate
            # is FP/excluded from SC). For dense models this is byte-identical to
            # the old (op, block_idx) keying (one call per key per forward).
            key = cid
            acc = self._accum.get(key)
            if acc is None:
                self._accum[key] = {
                    "operator": rec["operator"],
                    "block_idx": rec["block_idx"],
                    "metric": rec["metric"],
                    "errors": rec["errors"],
                    "energy": rec.get("energy"),
                    "alts": rec.get("alts"),
                    "g_sum": g,
                    "draws": 1,
                }
            elif acc["g_sum"].shape == g.shape:
                acc["g_sum"] = acc["g_sum"] + g
                acc["draws"] += 1
            # else: SC noise changed MoE routing across draws (grad_sc only), so
            # call cid hit a different expert with a different row count. Keep the
            # first draw's g rather than crash. draws==1 paths never reach here.
        self._pending.clear()
        self._next_id = 0

    def finalize(self):
        """Average g_row over the accumulated draws and feed the calibrator.

        Called once per window after all SC noise draws have been flushed.
        No-op when grad-weighting is OFF. The accumulator is drained in
        insertion order (= first-draw forward-hook firing order), so the
        calibrator's add order — and therefore its subsample RNG consumption —
        is identical to the single-draw path.
        """
        if not self.enabled:
            return
        for acc in self._accum.values():
            g_avg = acc["g_sum"] / max(acc["draws"], 1)
            self.calibrator.add(
                acc["operator"], acc["block_idx"], acc["metric"],
                acc["errors"], grad_rows=g_avg, energy_rows=acc.get("energy"),
                metric_alts=acc.get("alts"),
            )
        self._accum.clear()


def _make_sclinear_hook(
    calibrator: ThresholdCalibrator,
    merger: PendingMerger,
    levels: list,
    sc_prec: int,
    halve: bool,
    chunk_d: int,
    grad_on_sc: bool = False,
):
    """Forward hook for SCLinear. Records per-row metric + per-level error.

    In the FP / FP-grad paths the model is run with use_sc_linear=False, so the
    captured ``output`` already equals the FP teacher (F.linear(x, W, b)) and
    is used directly. When ``loss_weight_by_grad`` is on, the FP ``output``
    tensor is left attached to the autograd graph so a backward hook can later
    read ‖∂L/∂y_row‖₂ off it — the gradient flows through the real FP forward.

    In the STE noisy-trajectory path (``grad_on_sc``), SCLinear runs with
    ``sc_ste_grad`` on so ``output``'s VALUE is the SC-noisy activation (its
    gradient is still the FP Jacobian). The FP teacher is therefore recomputed
    here as ``F.linear(x, W, b)`` for the per-level σ measurement, while the
    backward hook still reads g_row off the (noisy-forward) ``output`` — which
    is exactly the noisy-trajectory loss sensitivity we want.

    The metric + per-level SC error computation always runs under ``no_grad``
    (it is a side measurement).
    """
    def hook(module: SCLinear, inputs, output: torch.Tensor):
        op = getattr(module, "_sc_op_name", None)
        if op is None or op not in LINEAR_OPS:
            return
        if not calibrator.use_operator(op):
            return
        block_idx = getattr(module, "_sc_block_idx", None)
        if block_idx is None:
            return
        x = inputs[0]
        x_flat = x.reshape(-1, x.shape[-1])
        # MoE experts can receive ZERO tokens in a given forward (sparse top-k
        # routing) → an empty (0, D) input. Nothing to calibrate; skip so we
        # don't reduce over an empty dim or record empty rows.
        if x_flat.shape[0] == 0:
            return
        with torch.no_grad():
            if grad_on_sc:
                # output is SC-noisy under STE; recompute the FP teacher.
                teacher_full = nn.functional.linear(x, module.weight, module.bias)
                teacher = teacher_full.reshape(-1, teacher_full.shape[-1])
            else:
                teacher = output.reshape(-1, output.shape[-1])
            # #5: match the deployed smoothed-activation error distribution.
            calib_smooth = (getattr(module, "smooth_scales", None)
                            if calibrator.calib_smoothquant else None)
            protected = calibrator.protected_indices_for_module(
                op, block_idx, getattr(module, "_sc_unit_idx", None))
            # Rank rows on the SAME channels runtime uses. When salient
            # (protected) channels are split off to their own stream, runtime
            # EXCLUDES them from the per-row abs-max metric (model/sc_common.py:
            # `metric_source = x_flat.index_select(1, rest_idx)`). Fitting the
            # metric on all channels here while runtime ranks on the complement
            # shifts row ordering / threshold occupancy and drifts the budget.
            prot_idx = torch.tensor(
                sorted({int(i) for i in (protected or [])
                        if 0 <= int(i) < x_flat.shape[1]}),
                dtype=torch.long, device=x_flat.device)
            if prot_idx.numel() == 0:
                metric_src = x_flat
            else:
                mask = torch.ones(x_flat.shape[1], dtype=torch.bool,
                                  device=x_flat.device)
                mask[prot_idx] = False
                rest_idx = mask.nonzero(as_tuple=True)[0]
                metric_src = (x_flat.index_select(1, rest_idx)
                              if rest_idx.numel() > 0 else None)
            metric_rows = (
                metric_src.float().abs().amax(dim=-1)
                if metric_src is not None
                else torch.zeros(x_flat.shape[0], dtype=torch.float32,
                                 device=x_flat.device)
            )
            row_metric = _normalize_metric(metric_rows)
            # act_global_v2: candidate metrics from the SAME channel-complement
            # source, so the ρ selection ranks exactly what runtime would see.
            metric_alts = (_metric_candidates(metric_src)
                           if (calibrator.metric_select == "auto"
                               and metric_src is not None) else None)
            level_errors = []
            for sl in levels:
                if protected:
                    sc_out = _sc_linear_protected_at_level(
                        x_flat, module.weight, module.bias, protected,
                        protected_stoc_len=calibrator.protect_channel_stoc_len,
                        residual_stoc_len=sl, sc_prec=sc_prec, halve=halve,
                        chunk_d=chunk_d, smooth_scales=calib_smooth)
                elif sl == 0:
                    sc_out = torch.zeros_like(teacher)
                else:
                    sc_out = _sc_linear_at_level(
                        x_flat, module.weight, module.bias,
                        sc_prec=sc_prec, stoc_len=sl, halve=halve, chunk_d=chunk_d,
                        smooth_scales=calib_smooth,
                    )
                level_errors.append(_relative_l2_rows(sc_out, teacher))
        merger.record(op, block_idx, row_metric, level_errors, output,
                      metric_alts=metric_alts)
    return hook


def _patch_attention_for_calibration(
    calibrator: ThresholdCalibrator,
    merger: PendingMerger,
    levels: list,
    sc_prec: int,
    halve: bool,
    grad_on_sc: bool = False,
    grad_sc_stoclen: Optional[int] = None,
):
    """Monkey-patch HF Qwen3(/MoE) eager_attention_forward.

    The patched function computes the FP teacher result and returns it
    (the rest of the forward proceeds in FP), while also recording per-head
    Q·Kᵀ error and per-(B*H)-row softmax·V error at every stoc_len level.

    When ``loss_weight_by_grad`` is on, the teacher matmul/softmax chain is
    kept on the autograd graph (NOT wrapped in ``no_grad``) so that
    ``loss.backward()`` can read ‖∂L/∂y_row‖₂ off the two intermediates:
      qk → gradient of the Q·Kᵀ output ([B,H,N,M]),
           reduced per-head to match the per-head qk error/metric;
      av → gradient of the softmax·V output ([B,H,N,D_head]),
           reduced per-(B*H*N)-row to match the av error/metric.
    The per-level SC error measurements always stay under ``no_grad``.

    STE noisy-trajectory mode (``grad_on_sc``): the Q·Kᵀ and softmax·V outputs
    are replaced by their straight-through form
        node = fp_teacher + (sc_uniform - fp_teacher).detach()
    where ``sc_uniform`` is the SC result at the uniform ``grad_sc_stoclen``.
    The softmax then sees the NOISY Q·Kᵀ and the rest of the network sees the
    NOISY softmax·V, so the gradients read off these nodes reflect the SC-noisy
    trajectory rather than the clean FP forward. The per-level σ measurements
    still compare SC-at-each-level against the FP teacher (unchanged objective).
    The backward hooks attach to the STE nodes (the values that actually
    propagate downstream).
    """
    try:
        from transformers.models.qwen3 import modeling_qwen3
    except ImportError:
        modeling_qwen3 = None
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe
    except ImportError:
        modeling_qwen3_moe = None
    try:
        from transformers.models.llama import modeling_llama
    except ImportError:
        modeling_llama = None

    # calib_eager uses the standardized HF attention-interface signature and
    # module.{num_key_value_groups,layer_idx}, so it patches Llama's
    # eager_attention_forward identically to Qwen3 (both loaded eager by
    # loader.make_*_sc). WITHOUT patching Llama here, Llama attention runs the
    # eval-side SC patch with use_sc_attn=False (FP, no recording) → qk/av are
    # never calibrated → the MP table has no qk/av thresholds → eval crashes at
    # the first attention layer. Patch every importable family exposing the fn.
    _mods = (modeling_qwen3, modeling_qwen3_moe, modeling_llama)
    targets = [m for m in _mods
               if m is not None and hasattr(m, "eager_attention_forward")]
    if not targets:
        raise RuntimeError(
            "no eager_attention_forward to patch among qwen3/qwen3_moe/llama "
            "(transformers version mismatch?).")
    originals = [(m, m.eager_attention_forward) for m in targets]

    from model.sc_common import _repeat_kv as repeat_kv

    grad_on = merger.enabled

    def calib_eager(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        # n1: attention keys the (op, layer) tables by module.layer_idx, while the
        # Linear path keys by the replace_linears traversal counter _sc_block_idx.
        # These agree for Llama/Qwen3 (ModuleList iterates in layer_idx order). If
        # onboarding a model that exposes layers out of registration order or
        # without layer_idx, verify the two indexings still line up.
        block_idx = getattr(module, "layer_idx", None)

        # Teacher chain: on-graph when grad-weighting is on, else no_grad
        # (byte-identical to the original code path).
        teacher_ctx = torch.enable_grad() if grad_on else torch.no_grad()
        with teacher_ctx:
            # ---- Q·Kᵀ teacher
            teacher_attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling

            # STE: feed the SC-noisy Q·Kᵀ into the softmax (value), keep the FP
            # Jacobian (gradient). qk_node is what propagates downstream.
            if grad_on_sc:
                with torch.no_grad():
                    # _sc_attn_matmul_at_level returns float32; cast to the
                    # teacher dtype so the STE node keeps fp16 (else downstream
                    # Linears see a float32 activation vs fp16 weights).
                    sc_attn_ste = (_sc_attn_matmul_at_level(
                        query, key_states,
                        sc_prec=sc_prec, stoc_len=int(grad_sc_stoclen), halve=halve,
                    ) * scaling).to(teacher_attn.dtype)
                qk_node = teacher_attn + (sc_attn_ste - teacher_attn).detach()
            else:
                qk_node = teacher_attn

            # Softmax (applied to qk_node).
            attn_weights = qk_node
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=dropout, training=False)

            # ---- A·V teacher
            teacher_av = torch.matmul(attn_weights, value_states)

            # STE: feed the SC-noisy softmax·V downstream, keep the FP Jacobian.
            if grad_on_sc:
                with torch.no_grad():
                    sc_av_ste = _sc_attn_matmul_at_level(
                        attn_weights, value_states.transpose(-2, -1),
                        sc_prec=sc_prec, stoc_len=int(grad_sc_stoclen), halve=halve,
                    ).to(teacher_av.dtype)
                av_node = teacher_av + (sc_av_ste - teacher_av).detach()
            else:
                av_node = teacher_av

        # ---- qk: per-(B*H*N)-ROW metric + per-level SC error (side measurement).
        # MUST mirror the RUNTIME qk classify, which is per query ROW (the
        # `for bh: metric = a_bh.amax(-1)` loop in _sc_attention_matmul_ab_t),
        # NOT per-head. A per-head calibration (the old code) records only H
        # units per call, so the cross-layer budget under-counts qk by ~N
        # (≈1000×) and the realized avg_sl blows past target (qk drift). Per-row
        # here makes the metric, the threshold granularity, AND true_counts[qk]
        # all match deployment — same structure as the av branch below.
        if calibrator.use_operator("qk") and block_idx is not None:
            with torch.no_grad():
                Bq, Hq, Nq, Kq = query.shape
                # Per-row query metric: amax over head_dim K, one value per
                # (B,H,N) query row — exactly the runtime metric. Normalized
                # GLOBALLY over all B*H*N rows: the runtime classify pools the
                # whole flattened metric before thresholding (M1 resolution —
                # global scope on BOTH sides; per-slice normalization erases
                # cross-head scale and measurably degrades PPL).
                q_metric = _normalize_metric(
                    query.float().abs().amax(dim=-1).reshape(Bq * Hq * Nq)
                )
                # act_global_v2 candidates on the same B*H*N query rows.
                qk_alts = (_metric_candidates(query.reshape(Bq * Hq * Nq, Kq))
                           if calibrator.metric_select == "auto" else None)
                level_errors = []
                for sl in levels:
                    if sl == 0:
                        sc_attn = torch.zeros_like(teacher_attn, dtype=torch.float32)
                    else:
                        sc_attn = _sc_attn_matmul_at_level(
                            query, key_states,
                            sc_prec=sc_prec, stoc_len=sl, halve=halve,
                        ) * scaling
                    # Per-row relative L2 over the M (key) dim.
                    pred = sc_attn.float().reshape(Bq * Hq * Nq, -1)
                    targ = teacher_attn.float().reshape(Bq * Hq * Nq, -1)
                    denom = targ.norm(dim=-1).clamp_min(1e-8)
                    level_errors.append((pred - targ).norm(dim=-1) / denom)

            def _qk_row_grad(grad_y):
                # grad_y: [B, H, N, M] -> [B*H*N, M], norm over M, matching the
                # per-row qk error flattening exactly.
                g = grad_y.detach().float()
                return g.reshape(-1, g.shape[-1]).norm(dim=-1)

            merger.record("qk", block_idx, q_metric, level_errors,
                          qk_node, grad_reduce=_qk_row_grad,
                          metric_alts=qk_alts)

        # ---- av: per-(B*H*N)-row metric + per-level SC error (side measurement).
        if calibrator.use_operator("av") and block_idx is not None:
            with torch.no_grad():
                # Per-(B*H)-row metric: amax over the kv-dim of attn_weights.
                # Same surface as the runtime classify in _sc_attention_matmul_ab_t,
                # normalized GLOBALLY over B*H*N (matches the runtime global
                # pooling — M1 resolution, see qk branch above).
                B, H, N, K = attn_weights.shape
                av_metric_full = _normalize_metric(
                    attn_weights.float().abs().amax(dim=-1).reshape(B * H * N)
                )
                # act_global_v2 candidates on the same B*H*N attn rows.
                av_alts = (_metric_candidates(
                    attn_weights.reshape(B * H * N, K))
                    if calibrator.metric_select == "auto" else None)
                # Per-level error over the full (B*H, N, D_head) flattened.
                level_errors = []
                v_t = value_states.transpose(-2, -1)  # (B, H, D_head, K_seq)
                for sl in levels:
                    if sl == 0:
                        sc_av = torch.zeros_like(teacher_av, dtype=torch.float32)
                    else:
                        sc_av = _sc_attn_matmul_at_level(
                            attn_weights, v_t,
                            sc_prec=sc_prec, stoc_len=sl, halve=halve,
                        )
                    # Per-row relative L2: flatten (B,H,N,D_head) to (B*H*N, D_head).
                    pred = sc_av.float().reshape(B * H * N, -1)
                    targ = teacher_av.float().reshape(B * H * N, -1)
                    denom = targ.norm(dim=-1).clamp_min(1e-8)
                    level_errors.append((pred - targ).norm(dim=-1) / denom)

            def _av_row_grad(grad_y):
                # grad_y: [B, H, N, D_head] -> [B*H*N, D_head], norm over D_head,
                # matching the av error flattening exactly.
                g = grad_y.detach().float()
                return g.reshape(-1, g.shape[-1]).norm(dim=-1)

            merger.record("av", block_idx, av_metric_full, level_errors,
                          av_node, grad_reduce=_av_row_grad,
                          metric_alts=av_alts)

        attn_output = av_node.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    for m in targets:
        m.eager_attention_forward = calib_eager

    def restore():
        for m, orig in originals:
            m.eager_attention_forward = orig
    return restore


# =====================================================================
# Driver: wikitext2 forward passes
# =====================================================================


def _load_calibration_token_stream(tokenizer, args) -> torch.Tensor:
    """Concatenate a chunk of wikitext2 and tokenize. Returns 1-D token tensor."""
    ds = load_dataset(
        args.calib_dataset, args.calib_dataset_config, split=args.calib_split
    )
    text = "\n\n".join(d["text"] for d in ds if d.get("text", "").strip())
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    return enc


def _iter_calib_windows(enc: torch.Tensor, ctx: int, num_seqs: int):
    """Yield non-overlapping windows of length ``ctx`` from a 1-D token tensor."""
    total = enc.shape[0]
    for i in range(num_seqs):
        start = i * ctx
        end = start + ctx
        if end > total:
            break
        yield enc[start:end]


def _select_int_swap_windows(
    enc: torch.Tensor, ctx: int, num_windows: int, *,
    sampling: str = "prefix", seed: int = 0,
):
    """Select deterministic, non-overlapping windows for an INT-swap sweep.

    ``prefix`` preserves the v18 behavior.  ``stratified`` divides the entire
    token stream into equal block-index strata and samples one ctx-aligned
    block from each stratum.  It therefore covers the corpus instead of letting
    one contiguous WikiText prefix decide the mask, while remaining exactly
    reproducible from ``seed``.
    """
    ctx = int(ctx)
    num_windows = int(num_windows)
    if ctx <= 0 or num_windows <= 0:
        raise ValueError("ctx and num_windows must be positive")
    total_blocks = int(enc.shape[0]) // ctx
    take = min(num_windows, total_blocks)
    if take <= 0:
        return [], []
    if sampling == "prefix":
        block_ids = list(range(take))
    elif sampling == "stratified":
        rng = np.random.default_rng(int(seed))
        # With take <= total_blocks, every integer stratum is non-empty and the
        # disjoint strata guarantee that no sampled window overlaps another.
        edges = np.linspace(0, total_blocks, take + 1, dtype=np.int64)
        block_ids = [int(rng.integers(int(edges[i]), int(edges[i + 1])))
                     for i in range(take)]
    else:
        raise ValueError(f"unknown int_swap window sampling mode: {sampling}")
    starts = [b * ctx for b in block_ids]
    return [enc[s:s + ctx] for s in starts], starts


def _reseed_sc_for_draw(draw_idx: int, base_seed: int) -> None:
    """Make SC draw ``draw_idx`` use independent noise — when the kernel's Owen
    scramble is in its stochastic ('random') mode.

    The SC kernel is a deterministic function of its inputs under the default
    Sobol + bitrev scramble (and under mode 'off'), so re-running an identical
    forward gives byte-identical output: draws would be identical and this
    reseed is a no-op. Under ``SC_OWEN_MODE=random`` the per-dimension
    scramble is drawn from ``_OWEN_SCRAMBLE_SEED``; bumping that constant per
    draw (and clearing the cached enable tables) yields genuinely independent
    noise draws. We vary it defensively regardless of mode so that turning on
    the random scramble Just Works.
    """
    torch.manual_seed(int(base_seed) + 1 + draw_idx)
    try:
        from scmp_kernels.sc import kernels as _sck
        _sck._OWEN_SCRAMBLE_SEED = (int(base_seed) + 1 + draw_idx) * 2654435761 & 0x7FFFFFFF
        _sck.clear_rng_cache()
    except Exception as e:
        # Don't hide a real import/attr error: under SC_OWEN_MODE=random a
        # silent failure here yields identical draws with no warning (m3).
        print(f"[calib] WARN: SC reseed for draw {draw_idx} failed "
              f"({type(e).__name__}: {e}); draws may be identical.", file=sys.stderr)


# All SC operators that can carry a per-group precision override.
_ALL_SC_OPS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "qk", "av",
)


def _measure_group_sensitivity(
    model, enc, levels, total_blocks, layer_buckets, operators,
    baseline_sl, probe_sl, n_windows, ctx, device,
):
    """Measured-ΔLoss per-group cross-layer sensitivity (knock-down probe).

    Runs the model in real SC inference at a uniform ``baseline_sl`` to get a
    reference loss L0, then for each (operator, layer-bucket) group re-runs with
    ONLY that group knocked down to ``probe_sl`` (all else at baseline) and
    records ΔL = L_group − L0. ΔL is the true marginal loss sensitivity of that
    group's precision budget — the ground-truth cross-layer signal that replaces
    the gradient proxy. Returns {(operator, l_bucket): ΔL}.

    Uses the per-(op, block) override map (config.sc_group_stoclen), so this is
    a pure forward measurement (no autograd). ~|ops|·layer_buckets + 1 forwards.
    """
    ops = [o for o in _ALL_SC_OPS if o in operators]
    cfg = model.config
    prev = {k: getattr(cfg, k, None) for k in (
        "use_sc_linear", "use_sc_attn", "sc_ste_grad", "sc_mp_config",
        "sc_stoc_len", "sc_group_stoclen")}
    cfg.use_sc_linear = True
    cfg.use_sc_attn = True
    cfg.sc_ste_grad = False
    cfg.sc_mp_config = None
    cfg.sc_stoc_len = baseline_sl

    def bucket(b):
        return _bucket_index(b, total_blocks, layer_buckets)

    base_map = {(op, b): baseline_sl for op in ops for b in range(total_blocks)}
    windows = list(_iter_calib_windows(enc, ctx, n_windows))

    def mean_loss(gmap):
        cfg.sc_group_stoclen = gmap
        tot, n = 0.0, 0
        with torch.no_grad():
            for w in windows:
                ids = w.unsqueeze(0).to(device)
                tot += float(model(input_ids=ids, labels=ids).loss)
                n += 1
        return tot / max(n, 1)

    L0 = mean_loss(base_map)
    print(f"[measure] baseline L0={L0:.5f} (uniform sl={baseline_sl}, "
          f"probe sl={probe_sl}, {len(windows)} windows × {ctx} ctx)")
    weights = {}
    for op in ops:
        for lb in range(layer_buckets):
            gmap = dict(base_map)
            n_blk = 0
            for b in range(total_blocks):
                if bucket(b) == lb:
                    gmap[(op, b)] = probe_sl
                    n_blk += 1
            if n_blk == 0:
                continue
            Lg = mean_loss(gmap)
            weights[(op, lb)] = Lg - L0
            print(f"[measure] {op:9s} bucket{lb}: ΔL={Lg - L0:+.5f}  (blocks={n_blk})")

    # restore config
    for k, v in prev.items():
        setattr(cfg, k, v)
    return weights, L0


def _measure_group_curve(
    model, enc, levels, total_blocks, layer_buckets, operators,
    n_windows, ctx, device,
):
    """measured_curve: per-group per-LEVEL ΔLoss CURVE (leave-one-out from max).

    Reference L0 = every group at the HIGHEST level. For each (operator,
    layer-bucket) group and each candidate level ℓ, re-run with ONLY that group
    knocked to ℓ (all else at max) and record ΔL(g, ℓ) = L_gℓ − L0. ΔL(g, max)=0
    by construction, so each group gets a monotone loss-vs-precision curve.

    Unlike ``_measure_group_sensitivity`` (ONE scalar W_g per group used to
    reweight recon-error), this is the TRUE loss curve, so the global solve can
    allocate by  min Σ ΔLoss + λ·budget  directly — no recon-error factor to
    blind it to qk. ~|ops|·layer_buckets·(|levels|−1) + 1 SC forwards.
    Returns {(operator, l_bucket): [ΔL per level, in ``levels`` order]}, L0.
    """
    ops = [o for o in _ALL_SC_OPS if o in operators]
    cfg = model.config
    prev = {k: getattr(cfg, k, None) for k in (
        "use_sc_linear", "use_sc_attn", "sc_ste_grad", "sc_mp_config",
        "sc_stoc_len", "sc_group_stoclen")}
    cfg.use_sc_linear = True
    cfg.use_sc_attn = True
    cfg.sc_ste_grad = False
    cfg.sc_mp_config = None
    top = int(max(levels))
    cfg.sc_stoc_len = top

    def bucket(b):
        return _bucket_index(b, total_blocks, layer_buckets)

    base_map = {(op, b): top for op in ops for b in range(total_blocks)}
    windows = list(_iter_calib_windows(enc, ctx, n_windows))

    def mean_loss(gmap):
        cfg.sc_group_stoclen = gmap
        tot, n = 0.0, 0
        with torch.no_grad():
            for w in windows:
                ids = w.unsqueeze(0).to(device)
                tot += float(model(input_ids=ids, labels=ids).loss)
                n += 1
        return tot / max(n, 1)

    L0 = mean_loss(base_map)
    print(f"[curve] baseline L0={L0:.5f} (all groups at max sl={top}, "
          f"{len(windows)} windows × {ctx} ctx)")
    curves = {}
    for op in ops:
        for lb in range(layer_buckets):
            blks = [b for b in range(total_blocks) if bucket(b) == lb]
            if not blks:
                continue
            row = []
            for lv in levels:
                if int(lv) >= top:
                    row.append(0.0)
                    continue
                gmap = dict(base_map)
                for b in blks:
                    gmap[(op, b)] = int(lv)
                row.append(float(mean_loss(gmap) - L0))
            curves[(op, lb)] = row
            print(f"[curve] {op:9s} bucket{lb}: "
                  f"ΔL={['%+.4f' % x for x in row]} blocks={len(blks)}")
    for k, v in prev.items():
        setattr(cfg, k, v)
    return curves, L0


# ---------------------------------------------------------------------------
# int_swap — the hybrid INT mask ranking (v18)
# ---------------------------------------------------------------------------

_INT_SWAP_CFG_KEYS = (
    "use_sc_linear", "use_sc_attn", "sc_ste_grad", "sc_mp_config",
    "sc_stoc_len", "sc_group_stoclen",
    "sc_hybrid_schedule", "sc_hybrid_default", "sc_hybrid_int_bits",
    "sc_hybrid_int_sym", "sc_hybrid_chunk_size",
)


def _int_swap_window_losses(model, cfg, windows, device, schedule) -> list:
    """Per-window losses with ``schedule`` as the active hybrid override map.

    Returns the per-window list (not the mean) so a split-half stability check
    costs nothing extra.
    """
    cfg.sc_hybrid_schedule = dict(schedule)
    out = []
    with torch.no_grad():
        for w in windows:
            ids = w.unsqueeze(0).to(device)
            out.append(float(model(input_ids=ids, labels=ids).loss))
    return out


def _int_swap_score(paired_gains, mode: str = "mean", lcb_z: float = 1.0):
    """Return mean, sample std, standard error, and the ranking score.

    Each sample is paired (the all-SC loss and swapped loss use the same text),
    so corpus difficulty cancels before uncertainty is estimated.  ``lcb``
    ranks by ``mean - z*SE`` and therefore requires evidence that an apparent
    gain survives calibration-text variation.
    """
    x = np.asarray(paired_gains, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    se = std / math.sqrt(int(x.size))
    if mode == "mean":
        score = mean
    elif mode == "lcb":
        score = mean - float(lcb_z) * se
    else:
        raise ValueError(f"unknown int_swap score mode: {mode}")
    return mean, std, se, float(score)


def _atomic_write_json(path, payload) -> None:
    """Atomically replace a JSON checkpoint so preemption cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _measure_group_int_swap(
    model, enc, total_blocks, operators, *,
    int_bits, ref_mode, ref_level, mp_config_path,
    n_windows, ctx, device, int_sym=True, int_chunk=128,
    window_sampling="prefix", window_seed=0,
    score_mode="mean", lcb_z=1.0, stability_folds=2,
    checkpoint_path=None,
):
    """int_swap: per-(operator, LAYER) ΔLoss of swapping that entry SC → INT<b>.

    The hybrid mask decides a BACKEND (SC vs INT), so the quantity that decides
    it is the backend-swap gain, not the SC precision-degradation curve that
    ``measured_curve`` measures. For each (operator, block) this flips exactly
    that entry to ``int<bits>`` — via ``config.sc_hybrid_schedule``, the same
    handle the deployed mask uses — leaving every other entry on SC, and records

        gain(op, b) = L_ref − L_swap        (>0 ⇒ INT<bits> HELPS here)

    Ranking by gain descending and masking the top fraction is then the direct
    criterion, replacing ``max(level_mean_error[1:])`` (which ranks by SC
    fragility and only equals the swap gain under "INT is lossless").

    Two references, both at the DEPLOYED operating point rather than at the top
    of the SC ladder:
      * ``mp_config``  — every entry on SC at its calibrated per-row MP levels,
        loaded from a V9 wrapper JSON. Closest to deployment. Mask OFF, so the
        measurement is a swap from SC in every cell.
      * ``uniform``    — every entry on SC at a single ``ref_level`` (halved).

    Granularity is native per-LAYER (no bucket broadcast). Cost is
    ``|ops|·total_blocks + 1`` forward-groups — cheaper than measured_curve's
    ``|ops|·buckets·(|levels|−1) + 1`` — which is what buys the extra windows
    that ``max`` over noisy per-level estimates cannot afford.

    Returns ``(gains, info)`` where gains maps (op, block) → float.
    """
    ops = [o for o in _ALL_SC_OPS if o in operators]
    cfg = model.config
    prev = {k: getattr(cfg, k, None) for k in _INT_SWAP_CFG_KEYS}

    old_mp_config_env = os.environ.get("MP_CONFIG_JSON")
    try:
        cfg.use_sc_linear = True
        cfg.use_sc_attn = True
        cfg.sc_ste_grad = False
        cfg.sc_hybrid_default = "sc"
        cfg.sc_hybrid_int_bits = int(int_bits)
        cfg.sc_hybrid_int_sym = bool(int_sym)
        cfg.sc_hybrid_chunk_size = int(int_chunk)

        if ref_mode == "mp_config":
            if not mp_config_path:
                raise SystemExit("[calib] int_swap --int-swap-ref mp_config requires "
                                 "--int-swap-mp-config <wrapper json>")
            # Reuse the deployment loader so the reference is byte-identical to
            # what eval runs.  The hybrid schedule remains empty while probing.
            os.environ["MP_CONFIG_JSON"] = mp_config_path
            from loader import apply_mp_config_from_env
            apply_mp_config_from_env(model)
            cfg.sc_group_stoclen = None
            ref_desc = f"mp_config={os.path.basename(mp_config_path)}"
        elif ref_mode == "uniform":
            cfg.sc_mp_config = None
            cfg.sc_group_stoclen = {}
            cfg.sc_stoc_len = int(ref_level)
            ref_desc = f"uniform_sl={int(ref_level)}"
        else:
            raise SystemExit(f"[calib] unknown --int-swap-ref: {ref_mode}")

        windows, window_starts = _select_int_swap_windows(
            enc, ctx, n_windows, sampling=window_sampling, seed=window_seed)
        if len(windows) < 4:
            print(f"[int_swap] WARN: {len(windows)} windows — split-half stability "
                  "is meaningless below 4; raise --measure-windows.")

        signature = {
            "total_blocks": int(total_blocks),
            "operators": list(ops),
            "int_bits": int(int_bits),
            "int_sym": bool(int_sym),
            "int_chunk_size": int(int_chunk),
            "ref_mode": ref_mode,
            "ref_level": int(ref_level),
            "mp_config_path": (os.path.abspath(mp_config_path)
                               if mp_config_path else None),
            "token_count": int(enc.shape[0]),
            "ctx": int(ctx),
            "windows": len(windows),
            "window_sampling": window_sampling,
            "window_seed": int(window_seed),
            "window_starts": window_starts,
            "score_mode": score_mode,
            "lcb_z": float(lcb_z),
        }
        checkpoint = None
        per_window = {}
        if checkpoint_path and Path(checkpoint_path).is_file():
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            if checkpoint.get("signature") != signature:
                raise SystemExit(
                    f"[int_swap] checkpoint signature mismatch: {checkpoint_path}; "
                    "use a new checkpoint path for changed sweep settings")
            for key, losses in checkpoint.get("per_window_loss", {}).items():
                op, bpart = key.split(":")
                if len(losses) != len(windows):
                    raise SystemExit(f"[int_swap] incomplete checkpoint row: {key}")
                per_window[(op, int(bpart[1:]))] = [float(x) for x in losses]
            print(f"[int_swap] resume: loaded {len(per_window)}/"
                  f"{len(ops) * total_blocks} completed probes from {checkpoint_path}")

        ref_losses = _int_swap_window_losses(model, cfg, windows, device, {})
        if checkpoint and checkpoint.get("baseline_window_losses") is not None:
            saved_ref = np.asarray(checkpoint["baseline_window_losses"], dtype=np.float64)
            if saved_ref.shape != (len(ref_losses),) or not np.allclose(
                    saved_ref, ref_losses, rtol=1e-6, atol=1e-7):
                raise SystemExit(
                    "[int_swap] checkpoint baseline changed on resume; refusing "
                    "to combine measurements from different model/runtime states")
        L0 = float(np.mean(ref_losses))
        print(f"[int_swap] reference L0={L0:.5f} ({ref_desc}, all-SC, mask OFF, "
              f"{len(windows)} {window_sampling} windows × {ctx} ctx)")

        def save_checkpoint():
            if not checkpoint_path:
                return
            _atomic_write_json(checkpoint_path, {
                "format": "scmp_llm_int_swap_checkpoint_v1",
                "signature": signature,
                "baseline_window_losses": ref_losses,
                "per_window_loss": {
                    f"{op}:b{b}": v for (op, b), v in per_window.items()
                },
            })

        save_checkpoint()
        gains, gain_std, gain_se, selection_scores = {}, {}, {}, {}
        n_total = len(ops) * total_blocks
        done = 0
        for op in ops:
            for b in range(total_blocks):
                key = (op, b)
                if key not in per_window:
                    per_window[key] = _int_swap_window_losses(
                        model, cfg, windows, device,
                        {key: f"int{int(int_bits)}"})
                    save_checkpoint()
                paired = np.asarray(ref_losses) - np.asarray(per_window[key])
                mean, std, se, score = _int_swap_score(
                    paired, score_mode, lcb_z)
                gains[key] = mean
                gain_std[key] = std
                gain_se[key] = se
                selection_scores[key] = score
                done += 1
            row = [gains[(op, b)] for b in range(total_blocks)]
            score_row = [selection_scores[(op, b)] for b in range(total_blocks)]
            print(f"[int_swap] {op:9s} gain: mean={np.mean(row):+.5f} "
                  f"min={np.min(row):+.5f} max={np.max(row):+.5f}; "
                  f"{score_mode} score max={np.max(score_row):+.5f} "
                  f"[{done}/{n_total}]")

        # For corpus-spread windows, even/odd halves each span the corpus.  The
        # legacy prefix mode keeps its contiguous split for byte-compatible
        # diagnostics on old invocations.
        half = len(windows) // 2
        split = None
        if half >= 2:
            if window_sampling == "stratified":
                idx_a = list(range(0, 2 * half, 2))
                idx_b = list(range(1, 2 * half, 2))
            else:
                idx_a = list(range(half))
                idx_b = list(range(half, 2 * half))

            def subset_maps(indices):
                gm, sm = {}, {}
                for key, losses in per_window.items():
                    paired = np.asarray([ref_losses[i] - losses[i]
                                         for i in indices])
                    mean, _std, _se, score = _int_swap_score(
                        paired, score_mode, lcb_z)
                    gm[key], sm[key] = mean, score
                return gm, sm

            gains_a, scores_a = subset_maps(idx_a)
            gains_b, scores_b = subset_maps(idx_b)
            split = {
                "windows_per_half": half,
                "indices_a": idx_a,
                "indices_b": idx_b,
                "gain_a": gains_a,
                "gain_b": gains_b,
                "score_a": scores_a,
                "score_b": scores_b,
            }

            k_folds = max(2, min(int(stability_folds), len(windows)))
            fold_scores = []
            for fold in range(k_folds):
                indices = list(range(fold, len(windows), k_folds))
                if indices:
                    _gm, sm = subset_maps(indices)
                    fold_scores.append(sm)
            split["fold_scores"] = fold_scores
            split["stability_folds"] = len(fold_scores)

        info = {
            "reference": ref_desc,
            "ref_mode": ref_mode,
            "baseline_loss": L0,
            "baseline_window_losses": ref_losses,
            "int_bits": int(int_bits),
            "int_sym": bool(int_sym),
            "int_chunk_size": int(int_chunk),
            "windows": len(windows),
            "ctx": int(ctx),
            "window_sampling": window_sampling,
            "window_seed": int(window_seed),
            "window_starts": window_starts,
            "ranking": {"score_method": score_mode, "lcb_z": float(lcb_z)},
            "selection_score": {
                f"{op}:b{b}": v for (op, b), v in selection_scores.items()
            },
            "gain_std": {f"{op}:b{b}": v for (op, b), v in gain_std.items()},
            "gain_se": {f"{op}:b{b}": v for (op, b), v in gain_se.items()},
            "per_window_loss": {
                f"{op}:b{b}": v for (op, b), v in per_window.items()
            },
        }
        return gains, info, split
    finally:
        for k, v in prev.items():
            setattr(cfg, k, v)
        if old_mp_config_env is None:
            os.environ.pop("MP_CONFIG_JSON", None)
        else:
            os.environ["MP_CONFIG_JSON"] = old_mp_config_env


def int_swap_stability(gains_a: dict, gains_b: dict, frac: float = 0.20) -> dict:
    """Split-half agreement of the mask ranking: Spearman ρ over all entries and
    overlap of the top-``frac`` selected SETS (the set is what ships, so set
    overlap — not ρ — is the gate)."""
    keys = sorted(set(gains_a) & set(gains_b))
    if not keys:
        return {"n": 0, "spearman": float("nan"), "top_overlap": float("nan")}
    a = np.array([gains_a[k] for k in keys], dtype=np.float64)
    b = np.array([gains_b[k] for k in keys], dtype=np.float64)
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        rho = float("nan")
    else:
        rho = float(np.corrcoef(ra, rb)[0, 1])
    t = max(1, int(math.ceil(len(keys) * frac)))
    top_a = {keys[i] for i in np.argsort(-a)[:t]}
    top_b = {keys[i] for i in np.argsort(-b)[:t]}
    return {"n": len(keys), "frac": frac, "top_n": t,
            "spearman": rho,
            "top_overlap": len(top_a & top_b) / t}


def int_swap_cross_fold_stability(fold_scores, frac: float = 0.20) -> dict:
    """Pairwise ranking agreement across corpus-interleaved folds."""
    pairwise = []
    for i in range(len(fold_scores)):
        for j in range(i + 1, len(fold_scores)):
            s = int_swap_stability(fold_scores[i], fold_scores[j], frac)
            pairwise.append({"fold_a": i, "fold_b": j, **s})
    if not pairwise:
        return {"folds": len(fold_scores), "pairs": [],
                "mean_top_overlap": float("nan"),
                "min_top_overlap": float("nan"),
                "mean_spearman": float("nan"), "min_spearman": float("nan")}
    overlaps = np.asarray([x["top_overlap"] for x in pairwise], dtype=np.float64)
    rhos = np.asarray([x["spearman"] for x in pairwise], dtype=np.float64)
    return {
        "folds": len(fold_scores),
        "pairs": pairwise,
        "mean_top_overlap": float(np.nanmean(overlaps)),
        "min_top_overlap": float(np.nanmin(overlaps)),
        "mean_spearman": float(np.nanmean(rhos)),
        "min_spearman": float(np.nanmin(rhos)),
    }


def build_int_swap_payload(gains: dict, info: dict, split: Optional[dict],
                           *, model_path: str, total_blocks: int,
                           operators, frac_report: float = 0.20) -> dict:
    """The ranking JSON consumed by ``hpca``'s ``auto_hybrid_config``.

    Deliberately a DIFFERENT schema from the measured_curve sensitivity file:
    ``swap_gain`` is a signed gain ranked DESCENDING, whereas that file's
    ``level_mean_error`` is an SC error ranked by ``max(...[1:])``. Reusing the
    same key for the opposite quantity would silently mis-rank if a stale file
    were pointed at the new selector.
    """
    measurement = dict(info)
    selection_score = measurement.pop("selection_score", None)
    gain_std = measurement.pop("gain_std", None)
    gain_se = measurement.pop("gain_se", None)
    payload = {
        "format": "scmp_llm_int_swap_v1",
        "method": "int_swap",
        "model_path": model_path,
        "total_blocks": int(total_blocks),
        "operators": sorted(operators),
        "swap_gain": {f"{op}:b{b}": float(g) for (op, b), g in gains.items()},
        "measurement": measurement,
    }
    if selection_score is not None:
        payload["selection_score"] = {
            k: float(v) for k, v in selection_score.items()
        }
    if gain_std is not None:
        payload["gain_std"] = {k: float(v) for k, v in gain_std.items()}
    if gain_se is not None:
        payload["gain_se"] = {k: float(v) for k, v in gain_se.items()}
    if split is not None:
        rank_a = split.get("score_a", split["gain_a"])
        rank_b = split.get("score_b", split["gain_b"])
        stab = int_swap_stability(rank_a, rank_b, frac_report)
        payload["split_half"] = {
            "windows_per_half": split["windows_per_half"],
            "indices_a": split.get("indices_a"),
            "indices_b": split.get("indices_b"),
            "spearman": stab["spearman"],
            "top_overlap": stab["top_overlap"],
            "top_frac": stab["frac"],
            "gain_a": {f"{op}:b{b}": float(v) for (op, b), v in split["gain_a"].items()},
            "gain_b": {f"{op}:b{b}": float(v) for (op, b), v in split["gain_b"].items()},
        }
        if "score_a" in split:
            payload["split_half"]["score_a"] = {
                f"{op}:b{b}": float(v) for (op, b), v in split["score_a"].items()
            }
            payload["split_half"]["score_b"] = {
                f"{op}:b{b}": float(v) for (op, b), v in split["score_b"].items()
            }
        if split.get("fold_scores"):
            fold_stab = int_swap_cross_fold_stability(
                split["fold_scores"], frac_report)
            payload["cross_fold"] = {
                **fold_stab,
                "top_frac": frac_report,
                "fold_scores": [
                    {f"{op}:b{b}": float(v) for (op, b), v in scores.items()}
                    for scores in split["fold_scores"]
                ],
            }
    return payload


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (scipy-free, so the calibrator keeps its dependency
    surface)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # average ties
    i = 0
    xs = x[order]
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def _normalize_group_weights(raw: dict, floor_frac: float = 0.1) -> dict:
    """Floor ΔL at a small positive fraction of the mean (so no group is fully
    starved by measurement noise / negative ΔL), then normalize to mean 1 so the
    global budget interpretation (avg ≈ budget_ratio·ref) is preserved."""
    vals = np.array(list(raw.values()), dtype=np.float64)
    pos = np.clip(vals, 0.0, None)
    mean_pos = float(pos.mean()) if pos.size and pos.mean() > 0 else 1.0
    floor = floor_frac * mean_pos
    floored = {k: max(v, floor) for k, v in raw.items()}
    m = float(np.mean(list(floored.values()))) or 1.0
    return {k: v / m for k, v in floored.items()}


# =====================================================================
# Args + main
# =====================================================================


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True,
                   help="HF model id (Qwen3-* or llama-3.1-*).")
    p.add_argument("--mp_levels", required=True,
                   help="Descending comma-separated stoc_len levels in halved space, "
                        "e.g. '128,96,64'.")
    p.add_argument("--budget_ratio", type=float, default=0.71,
                   help="Target avg stoc_len as a fraction of --budget_ref_stoc_len.")
    p.add_argument("--budget_ref_stoc_len", type=int, default=None,
                   help="Reference stoc_len for the budget. Defaults to max(mp_levels).")
    p.add_argument("--sc_prec", type=int, default=8)
    p.add_argument("--halve", type=int, default=1,
                   help="halve_bipolar_stoc_len for SC calls. Match the inference path.")
    p.add_argument("--chunk_d", type=int, default=128,
                   help="Inner-dim chunk for SCLinear calls (matches sc_linear_chunk_d).")
    p.add_argument("--operators", default=DEFAULT_OPERATORS,
                   help="Comma-separated operators to calibrate.")
    p.add_argument("--num_calib_sequences", type=int, default=16,
                   help="Number of ctx-length wikitext2 windows to run.")
    p.add_argument("--ctx_len", type=int, default=1024)
    p.add_argument("--calib_dataset", default="wikitext")
    p.add_argument("--calib_dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--calib_split", default="train")
    p.add_argument("--layer_buckets", type=int, default=4)
    p.add_argument("--max_units_per_call", type=int, default=512)
    p.add_argument("--min_bucket_units", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss-weight-by-grad", dest="loss_weight_by_grad",
                   action="store_true", default=False,
                   help="Weight each row's SC error by its loss sensitivity "
                        "g_row²=‖∂L/∂y_row‖₂². Runs one extra backward pass per "
                        "calibration window. Default OFF (g_row≡1, byte-identical "
                        "to the reconstruction-error 'act' calibrator).")
    p.add_argument("--grad-g-pow", dest="grad_g_pow", type=float, default=2.0,
                   help="Exponent on g_row in the loss-weighted objective "
                        "(only used with --loss-weight-by-grad). 2.0 = "
                        "Gauss-Newton g²; 1.0 = softened g. Default 2.0.")
    p.add_argument("--grad-s-pow", dest="grad_s_pow", type=float, default=2.0,
                   help="Exponent on σ_row(level) in the loss-weighted "
                        "objective. 2.0 = variance σ²; 1.0 = magnitude σ "
                        "(matches the un-squared act objective). Default 2.0.")
    p.add_argument("--grad-on-sc", dest="grad_on_sc",
                   action="store_true", default=False,
                   help="STE noisy-trajectory gradient (g_SC). Implies "
                        "--loss-weight-by-grad. Runs the MAIN calibration "
                        "forward with SC ENABLED via a straight-through "
                        "estimator (out = fp + (sc - fp).detach()) at a uniform "
                        "stoc_len, so g_row=‖∂L/∂y_row‖₂ is captured along the "
                        "SC-noisy trajectory instead of the clean FP forward. "
                        "The per-level σ measurement and the budget objective "
                        "are unchanged. Natural exponents here are "
                        "--grad-g-pow 1 --grad-s-pow 1 (g² is already known to "
                        "be harmful).")
    p.add_argument("--grad-sc-stoclen", dest="grad_sc_stoclen", type=int, default=0,
                   help="Uniform stoc_len (in the SAME halved/raw space as "
                        "--mp_levels) for the STE noisy forward. 0 (default) "
                        "→ round(budget_ratio · budget_ref_stoc_len), i.e. the "
                        "per-config budget target average, so the noise level "
                        "matches the deployment regime (len192→≈91 halved, "
                        "int7→64 halved). Only used with --grad-on-sc.")
    p.add_argument("--grad-sc-draws", dest="grad_sc_draws", type=int, default=2,
                   help="Number of independent SC noise draws to average g_row "
                        "over per window (one forward+backward each). Only used "
                        "with --grad-on-sc. NOTE: the SC kernel is deterministic "
                        "under the default Sobol+counter scramble, so draws>1 "
                        "only differ when SC_OWEN_MODE=random (the driver "
                        "reseeds per draw); otherwise draws are identical and "
                        "averaging is a harmless no-op (set 1 to skip the cost). "
                        "Default 2.")
    p.add_argument("--budget-scope", dest="budget_scope",
                   choices=["per_bucket", "global"], default="per_bucket",
                   help="Budget allocation scope. 'per_bucket' (default): every "
                        "(operator, layer-bucket) is independently pinned to the "
                        "same avg stoc_len (original; no cross-layer transfer). "
                        "'global': ONE shared λ over all groups, so budget flows "
                        "across layers/operators (SkewQ-style). With "
                        "--loss-weight-by-grad off → act-global; on → grad-global.")
    p.add_argument("--calib-smoothquant", dest="calib_smoothquant",
                   action="store_true", default=False,
                   help="open-issue #5 fix: measure per-level σ on the "
                        "SmoothQuant-transformed activation (a/s) that the "
                        "deployed kernel quantizes, matching the eval "
                        "(USE_SMOOTHQUANT=1 α=0.5). Also auto-enabled when the "
                        "USE_SMOOTHQUANT=1 env is set. Off → byte-identical.")
    p.add_argument("--objective", dest="objective",
                   choices=["sigma", "sigma2", "delta_sigma2"], default="sigma",
                   help="Allocation objective. 'sigma' (default): minimize Σ "
                        "relative-L2 recon error. 'sigma2': minimize Σσ² "
                        "(propagated-loss surrogate) — the near-lossless "
                        "water-filling objective; keeps high-σ linears pinned "
                        "near 128 while water-filling the attention pool. "
                        "'delta_sigma2' (v9): minimize the squared increase "
                        "relative to each row's error at the maximum MP level, "
                        "max(0, σ(L)-σ(Lmax))²; this prices the low-precision "
                        "cliff without chasing irreducible error at Lmax.")
    p.add_argument("--refine", dest="refine_mode",
                   choices=["none", "sigma"], default="none",
                   help="2nd-stage refinement on top of the (global) solve. "
                        "'none' (default): original, byte-identical. 'sigma': "
                        "greedy residual-budget fill — act_global's discrete "
                        "argmin leaves realized avg_sl under target; spend the "
                        "slack on the highest Δσ/Δcost row upgrades (guaranteed "
                        "iso-budget-fair improvement). Requires --budget-scope "
                        "global.")
    p.add_argument("--fisher-eps", dest="fisher_eps", type=float, default=0.5,
                   help="FisherMP bounded-blend strength: W_g = 1 + eps·(clip(Ŵ_g,"
                        "1/κ,κ)−1). eps=0 → act_global byte-identical.")
    p.add_argument("--fisher-kappa", dest="fisher_kappa", type=float, default=2.0,
                   help="FisherMP per-group weight clip [1/κ, κ] so no bounded "
                        "weight can starve a group (the primary anti-starvation "
                        "guardrail once attention is weighted).")
    p.add_argument("--fisher-attn", dest="fisher_attn", action="store_true",
                   help="FisherMP: also weight qk and av by their GN ‖y‖²‖g‖², "
                        "removing the W_g=1 attention protection. qk and av are "
                        "each normalized in their OWN pool (never a shared "
                        "attention weight). Off (default) = linear-only, "
                        "byte-identical to shipped fisher.")
    p.add_argument("--cross-layer-weight", dest="cross_layer_weight",
                   choices=["uniform", "measured", "measured_marg", "grad_group",
                            "fisher", "measured_curve", "int_swap"],
                   default="uniform",
                   help="Per-group importance weight W_g for --budget-scope "
                        "global. 'uniform' (default): W_g=1 (act_global). "
                        "'measured': ΔLoss knock-down probe to the LOWEST level "
                        "(original; inconsistent because the floor knock-down "
                        "saturates past the collapse cliff at low precision). "
                        "'measured_marg': marginal probe — knock each group down "
                        "by a SMALL step from baseline (--measure-marg-frac) so "
                        "ΔL stays in the locally-linear regime at every precision "
                        "(the recommended measured variant). 'grad_group': source "
                        "W_g from the per-group MEAN of the loss gradient (one "
                        "backward), instead of the per-row g·σ multiply that "
                        "over-concentrates budget and crosses the cliff. All three "
                        "keep act-style σ quantile WITHIN each group and imply "
                        "--budget-scope global. 'int_swap' does NOT produce an MP "
                        "table at all: it measures the per-(operator, layer) SC→INT "
                        "backend-swap gain and writes the hybrid-mask RANKING JSON, "
                        "then exits (no σ collection, no export).")
    p.add_argument("--int-swap-ref", dest="int_swap_ref",
                   choices=["mp_config", "uniform"], default="uniform",
                   help="int_swap reference point. 'mp_config': every entry on SC "
                        "at its calibrated per-row MP levels from "
                        "--int-swap-mp-config (closest to deployment). 'uniform' "
                        "(default): every entry on SC at --int-swap-ref-level.")
    p.add_argument("--int-swap-ref-level", dest="int_swap_ref_level", type=int,
                   default=None,
                   help="int_swap uniform reference stoc_len (HALVED space). "
                        "Defaults to the deployed AVERAGE, round(budget_ratio · "
                        "budget_ref_stoc_len), clamped to the ladder — NOT the top "
                        "of the ladder, which is where measured_curve measures and "
                        "where the mask does not operate.")
    p.add_argument("--int-swap-mp-config", dest="int_swap_mp_config", default=None,
                   help="int_swap: V9 wrapper JSON used as the --int-swap-ref "
                        "mp_config reference. The mask is OFF while measuring, so "
                        "every entry is swapped FROM SC.")
    p.add_argument("--int-swap-bits", dest="int_swap_bits", type=int, default=7,
                   help="int_swap: INT bit width probed (must match the width the "
                        "deployed mask uses — hpca's hybrid_int_bits_for_cfg).")
    p.add_argument("--int-swap-window-sampling",
                   dest="int_swap_window_sampling",
                   choices=["prefix", "stratified"], default="prefix",
                   help="int_swap calibration-text selection. 'prefix' keeps "
                        "the v18 contiguous-prefix behavior. 'stratified' "
                        "(recommended) samples deterministic ctx-aligned "
                        "windows across the full token stream.")
    p.add_argument("--int-swap-score", dest="int_swap_score",
                   choices=["mean", "lcb"], default="mean",
                   help="int_swap ranking statistic. 'mean' is the v18 mean "
                        "paired gain. 'lcb' ranks by mean - z*standard_error, "
                        "penalizing text-sensitive candidates.")
    p.add_argument("--int-swap-lcb-z", dest="int_swap_lcb_z", type=float,
                   default=1.0,
                   help="One-sided uncertainty penalty z for --int-swap-score "
                        "lcb. Default 1.0.")
    p.add_argument("--int-swap-stability-folds",
                   dest="int_swap_stability_folds", type=int, default=2,
                   help="Number of interleaved calibration-text folds used for "
                        "pairwise ranking diagnostics. Default 2; use 4 with "
                        "at least 24 windows.")
    p.add_argument("--int-swap-checkpoint", dest="int_swap_checkpoint",
                   default=None,
                   help="Resumable partial JSON written after every completed "
                        "operator-layer probe. 'auto' uses "
                        "<output_json>.partial.json.")
    p.add_argument("--measure-probe-level", dest="measure_probe_level", type=int,
                   default=0, help="stoc_len a group is knocked down to in the "
                        "'measured' probe. 0 (default) → lowest level (max noise). "
                        "Ignored by 'measured_marg' (which derives the probe from "
                        "--measure-marg-frac) unless explicitly set > 0.")
    p.add_argument("--measure-marg-frac", dest="measure_marg_frac", type=float,
                   default=0.25, help="measured_marg: probe stoc_len = "
                        "round(baseline·(1-frac)). A small frac keeps ΔL a "
                        "finite-difference marginal sensitivity ∂L/∂budget near "
                        "the operating point, avoiding the floor/cliff that makes "
                        "plain 'measured' inconsistent at int7/len96. Default 0.25.")
    p.add_argument("--measure-floor-frac", dest="measure_floor_frac", type=float,
                   default=0.1, help="Floor for per-group ΔL/grad weights as a "
                        "fraction of the positive mean (anti-starvation). Default "
                        "0.1; lower it (e.g. 0.05) for more cross-layer "
                        "discrimination once the probe is low-variance.")
    p.add_argument("--grad-group-pow", dest="grad_group_pow", type=float, default=1.0,
                   help="grad_group: exponent on g_row before the per-group mean "
                        "that forms W_g. 1.0 (default) = mean |∂L/∂y_row| (robust; "
                        "g² re-introduces heavy-tail concentration even at group "
                        "level).")
    p.add_argument("--grad-group-clip", dest="grad_group_clip", type=float,
                   default=99.0, help="grad_group: winsorize g_row at this "
                        "percentile per call before the group mean (robustness "
                        "against a few outlier rows). 100 = off. Default 99.")
    p.add_argument("--measure-baseline-stoclen", dest="measure_baseline_stoclen",
                   type=int, default=0, help="Uniform baseline stoc_len for the "
                        "measured probe. 0 (default) → round(budget_ratio·ref).")
    p.add_argument("--measure-windows", dest="measure_windows", type=int, default=2,
                   help="Number of windows for each measured-probe forward. Default 2.")
    p.add_argument("--measure-ctx", dest="measure_ctx", type=int, default=512,
                   help="ctx_len for the measured-probe forwards. Default 512.")
    p.add_argument("--budget-weight", dest="budget_weight",
                   choices=["rows", "macs"], default="rows",
                   help="Budget cost weighting. 'rows' (default): row-serial "
                        "latency Σ R_g·L_g — FLOP-heavy linears are cheap (few "
                        "rows), so 'iso-budget' is NOT iso-compute and act_global "
                        "starves high-σ qk. 'macs': FLOP/energy Σ MAC_g·L_g — "
                        "iso-budget = iso-compute (linears expensive, attention "
                        "cheap). Needs --mac-weights-trace for per-op MACs/row.")
    p.add_argument("--mac-weights-trace", dest="mac_weights_trace", default=None,
                   help="scmp trace JSON (summary) whose per-op rows+macs give "
                        "macs_per_row[op] for --budget-weight macs.")
    p.add_argument("--argmin-pricing", dest="argmin_pricing",
                   choices=["rep", "macs"], default="rep",
                   help="act_global_v2: per-row cycle PRICE in the assignment "
                        "argmin. 'rep' (default, original): rep_g = R_g/n_g — "
                        "includes the subsample ratio true/stored, which "
                        "over-prices heavily-subsampled groups (attention "
                        "~128x vs linears ~4x). 'macs': mac_per_row[op] only "
                        "— the KKT-consistent price (the subsample ratio "
                        "cancels between a stored row's benefit and cost). "
                        "Budget accounting is unchanged (R_g-weighted). "
                        "Requires --budget-weight macs + --budget-scope "
                        "global; incompatible with --refine sigma.")
    p.add_argument("--metric-select", dest="metric_select",
                   choices=["amax", "auto"], default="amax",
                   help="act_global_v2: per-operator dispatch-metric "
                        "selection. 'amax' (default): the original "
                        "|x|.amax(-1) everywhere. 'auto': capture candidate "
                        "metrics (amax/l2/crest) per row, pick per operator "
                        "the SIGNED candidate with the best Spearman rho "
                        "against the true sigma-benefit, build thresholds on "
                        "it, and export the choice (payload "
                        "'dispatch_metrics') for the runtime.")
    p.add_argument("--metric-select-margin", dest="metric_select_margin",
                   type=float, default=0.05,
                   help="Minimum |rho| improvement over raw amax before "
                        "--metric-select auto switches an operator's metric.")
    p.add_argument("--protect-channel-frac", dest="protect_channel_frac",
                   type=float, default=0.0,
                   help="Offline salient input-channel protection for SCLinear: "
                        "top fraction of channels per linear module run at "
                        "--protect-channel-stoc-len, residual channels still use "
                        "row MP. 0.0 (default) disables and preserves old tables.")
    p.add_argument("--protect-channel-frac-overrides",
                   dest="protect_channel_frac_overrides", default="",
                   help="Per-operator overrides for --protect-channel-frac, as "
                        "'op:frac' pairs (e.g. 'down_proj:0.05,up_proj:0.03'). "
                        "Ops not listed use --protect-channel-frac. Lets the "
                        "outlier-heavy MLP trio isolate more scale-setting "
                        "channels (SpQR-style) while the rest stay lean. Budget "
                        "compensation reads the realized per-op fraction, so "
                        "iso-compute accounting is automatic.")
    p.add_argument("--protect-channel-metric", dest="protect_channel_metric",
                   choices=["act_weight", "act_grad_weight", "weight",
                            "act_collapse"],
                   default="act_weight",
                   help="Salience metric for protected channels. act_weight "
                        "uses cached SmoothQuant act_scales^2 times weight-column "
                        "energy; act_grad_weight uses the offline Gauss-Newton "
                        "proxy E[x_j^2]·Σ_o W[o,j]^2·E[g_o^2] from one "
                        "calibration backward pass; weight uses weight-column "
                        "energy only.")
    p.add_argument("--protect-channel-stoc-len", dest="protect_channel_stoc_len",
                   type=int, default=128,
                   help="stoc_len for protected channels. Default 128.")
    p.add_argument("--protect-channel-stat-ctx",
                   dest="protect_channel_stat_ctx", type=int, default=0,
                   help="act_grad_weight only: ctx_len for the offline "
                        "E[x^2], E[(dL/dy)^2] prepass. 0 (default) uses "
                        "--ctx_len. Lower this for very large models whose "
                        "full-ctx backward OOMs; threshold calibration and eval "
                        "still use --ctx_len.")
    p.add_argument("--protect-channel-stat-sequences",
                   dest="protect_channel_stat_sequences", type=int, default=0,
                   help="act_grad_weight only: number of windows for the offline "
                        "GN-stat prepass. 0 (default) uses "
                        "--num_calib_sequences.")
    p.add_argument("--protect-compensate-budget",
                   dest="protect_compensate_budget", action="store_true",
                   help="Strict iso-compute mode: lower the residual MP target so "
                        "protected@stoc_len plus residual MP lands at the original "
                        "target. Off reports the intentional overhead.")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_summary_csv", default=None,
                   help="Defaults to <output_json without .json>_summary.csv.")
    return p


def _expand_levels(value: str) -> list:
    levels = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(levels) < 2:
        raise ValueError(f"Need at least 2 stoc_len levels, got {levels}")
    for i in range(len(levels) - 1):
        if levels[i] <= levels[i + 1]:
            raise ValueError(f"Levels must be strictly descending, got {levels}")
    return levels


def _parse_csv_set(value: str) -> set:
    return {x.strip() for x in value.split(",") if x.strip()}


def _module_protected_key(mod: SCLinear):
    op = getattr(mod, "_sc_op_name", None)
    block_idx = getattr(mod, "_sc_block_idx", None)
    if op not in LINEAR_OPS or block_idx is None:
        return None
    return (op, int(block_idx), getattr(mod, "_sc_unit_idx", None))


class ProtectedChannelGNStats:
    """Offline channel salience stats for protected-channel selection.

    For a linear y = xW^T, the diagonal Gauss-Newton contribution of input
    channel j is approximated as:

        E[x_j^2] * Σ_o W[o,j]^2 * E[(∂L/∂y_o)^2]

    This is still an offline selector: runtime only receives the top-channel
    index lists and pays the same split-matmul overhead as the existing PC path.
    """

    def __init__(self):
        self.x2_sum: dict = {}
        self.x_count: dict = defaultdict(int)
        self.g2_sum: dict = {}
        self.g_count: dict = defaultdict(int)
        self.dims: dict = {}
        self.out_dims: dict = {}

    def add_forward(self, key, x: torch.Tensor, output: torch.Tensor) -> None:
        if not output.requires_grad:
            return
        x_flat = x.detach().reshape(-1, x.shape[-1])
        if x_flat.shape[0] == 0:
            return
        with torch.no_grad():
            x2 = x_flat.float().pow(2).sum(dim=0).cpu()
        prev = self.x2_sum.get(key)
        self.x2_sum[key] = x2 if prev is None else prev + x2
        self.x_count[key] += int(x_flat.shape[0])
        self.dims[key] = int(x_flat.shape[-1])

        def _grad_hook(grad_y, key=key):
            g_flat = grad_y.detach().reshape(-1, grad_y.shape[-1])
            if g_flat.shape[0] == 0:
                return None
            with torch.no_grad():
                g2 = g_flat.float().pow(2).sum(dim=0).cpu()
            prev_g = self.g2_sum.get(key)
            self.g2_sum[key] = g2 if prev_g is None else prev_g + g2
            self.g_count[key] += int(g_flat.shape[0])
            self.out_dims[key] = int(g_flat.shape[-1])
            return None

        output.register_hook(_grad_hook)

    def score_for(self, key, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if key not in self.x2_sum or key not in self.g2_sum:
            return None
        x_count = max(int(self.x_count.get(key, 0)), 1)
        g_count = max(int(self.g_count.get(key, 0)), 1)
        x2 = self.x2_sum[key].float() / float(x_count)
        g2 = self.g2_sum[key].float() / float(g_count)
        w2 = weight.detach().float().cpu().pow(2)
        if w2.shape[1] != x2.numel() or w2.shape[0] != g2.numel():
            return None
        return x2 * (w2 * g2[:, None]).sum(dim=0)

    def summary_for(self, key) -> dict:
        x_count = max(int(self.x_count.get(key, 0)), 1)
        g_count = max(int(self.g_count.get(key, 0)), 1)
        x2 = self.x2_sum.get(key)
        g2 = self.g2_sum.get(key)
        return {
            "x_rows": int(self.x_count.get(key, 0)),
            "grad_rows": int(self.g_count.get(key, 0)),
            "x2_mean": float((x2.float() / float(x_count)).mean().item())
                       if x2 is not None and x2.numel() else 0.0,
            "grad2_mean": float((g2.float() / float(g_count)).mean().item())
                          if g2 is not None and g2.numel() else 0.0,
        }


def _register_protected_channel_gn_hooks(model: nn.Module,
                                         stats: ProtectedChannelGNStats):
    hooks = []
    for mod in model.modules():
        if not isinstance(mod, SCLinear):
            continue
        key = _module_protected_key(mod)
        if key is None:
            continue

        def _hook(module, inputs, output, key=key):
            stats.add_forward(key, inputs[0], output)

        hooks.append(mod.register_forward_hook(_hook))
    return hooks


def _prepare_model_for_backward(model: nn.Module, args, reason: str) -> None:
    """Enable output-gradient hooks without allocating parameter gradients."""
    if not getattr(model, "_scmp_backward_prepared", False):
        for p in model.parameters():
            p.requires_grad_(False)
        model.enable_input_require_grads()
        setattr(model, "_scmp_backward_prepared", True)
    mp_low = args.model_path.lower()
    is_big = (
        "a3b" in mp_low or "moe" in mp_low
        or "-14b" in mp_low or "-30b" in mp_low or "-32b" in mp_low
        or os.environ.get("CALIB_GRAD_CKPT", "0") == "1"
    )
    if is_big and not getattr(model, "_scmp_grad_ckpt_enabled", False):
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            setattr(model, "_scmp_grad_ckpt_enabled", True)
            print(f"[calib] gradient_checkpointing_enable() for large model "
                  f"({reason}; use_cache=False).")
        except Exception as e:
            print(f"[calib] WARN: gradient_checkpointing_enable failed: {e}")


def _collect_protected_channel_gn_stats(
    model: nn.Module,
    enc: torch.Tensor,
    args,
    device,
) -> ProtectedChannelGNStats:
    stats = ProtectedChannelGNStats()
    hooks = _register_protected_channel_gn_hooks(model, stats)
    if not hooks:
        raise SystemExit("[calib] act_grad_weight matched 0 SCLinear modules.")
    stat_ctx = int(args.protect_channel_stat_ctx) or int(args.ctx_len)
    stat_sequences = int(args.protect_channel_stat_sequences) or int(
        args.num_calib_sequences)
    print("[calib] protected-channel act_grad_weight: collecting "
          "E[x^2] and E[(dL/dy)^2] from one FP backward pass per window "
          f"({stat_sequences} × {stat_ctx}).")
    try:
        for i, window in enumerate(_iter_calib_windows(
            enc, stat_ctx, stat_sequences,
        )):
            print(f"[calib] protected-channel GN window "
                  f"{i + 1}/{stat_sequences}")
            ids = window.unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
    finally:
        for h in hooks:
            h.remove()
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[calib] protected-channel act_grad_weight: collected stats for "
          f"{len(stats.g2_sum)} modules.")
    return stats


def _configure_protected_channels(
    calibrator: ThresholdCalibrator,
    model: nn.Module,
    act_scales: Optional[dict],
    *,
    frac: float,
    metric: str,
    stoc_len: int,
    compensate_budget: bool,
    gn_stats: Optional[ProtectedChannelGNStats] = None,
    frac_overrides: Optional[dict] = None,
) -> int:
    """Select offline salient input channels for protected-channel MP."""
    frac = float(frac)
    frac_overrides = {k: float(v) for k, v in (frac_overrides or {}).items()}
    max_frac = max([frac, *frac_overrides.values()]) if frac_overrides else frac
    if max_frac <= 0.0:
        return 0
    if max_frac >= 1.0:
        raise SystemExit("[calib] --protect-channel-frac (incl. overrides) must be < 1.0")
    if metric in ("act_weight", "act_collapse") and not act_scales:
        raise SystemExit(
            f"[calib] --protect-channel-metric {metric} requires act scales; "
            "use --calib-smoothquant or choose --protect-channel-metric weight.")
    if metric == "act_grad_weight" and gn_stats is None:
        raise SystemExit(
            "[calib] --protect-channel-metric act_grad_weight requires "
            "offline GN stats; this is an internal wiring error.")
    calibrator.protect_channel_frac = frac
    calibrator.protect_channel_metric = metric
    calibrator.protect_channel_stoc_len = int(stoc_len)
    calibrator.protect_compensate_budget = bool(compensate_budget)
    n_mod = n_ch = missing = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, SCLinear):
            continue
        key = _module_protected_key(mod)
        if key is None:
            continue
        op, block_idx, _unit_idx = key
        w = mod.weight.detach().float()
        D = int(w.shape[1])
        if D <= 0:
            continue
        w2 = w.pow(2).sum(dim=0).detach().cpu()
        if metric == "act_weight":
            a = act_scales.get(name) if act_scales else None
            if a is None:
                missing += 1
                continue
            score = a.detach().float().cpu().pow(2) * w2
            stat_summary = {}
        elif metric == "act_collapse":
            # Scale-COLLAPSE selection (SpQR/LLM.int8-style): protect the
            # channels that SET the shared per-row quantization scale — the
            # largest per-channel activation amax — regardless of their weight
            # energy. Splitting them off shrinks the quantization step for
            # every OTHER channel in the row (the down_proj post-SiLU / k_proj
            # residual that survives SmoothQuant). act_scales is the cached
            # per-channel amax; any positive power preserves the ranking, so
            # the raw scale is the score. act_weight's ·‖W_col‖² factor is
            # deliberately DROPPED — it underweights exactly the modest-weight
            # outlier channels that break the grid.
            a = act_scales.get(name) if act_scales else None
            if a is None:
                missing += 1
                continue
            score = a.detach().float().cpu()
            stat_summary = {}
        elif metric == "act_grad_weight":
            score = gn_stats.score_for(key, mod.weight) if gn_stats else None
            if score is None:
                missing += 1
                continue
            stat_summary = gn_stats.summary_for(key)
        elif metric == "weight":
            score = w2
            stat_summary = {}
        else:  # pragma: no cover - argparse choices guard this.
            raise SystemExit(f"[calib] unknown protected-channel metric {metric}")
        op_frac = frac_overrides.get(op, frac)
        if op_frac <= 0.0:
            continue          # this operator opts out of protection entirely
        k = max(1, int(math.ceil(op_frac * D)))
        idx = torch.topk(score, k=min(k, D), largest=True).indices
        idx_list = sorted(int(i) for i in idx.tolist())
        calibrator.protected_channel_indices[key] = idx_list
        calibrator.protected_channel_dims[key] = D
        calibrator.protected_channel_scores[key] = {
            "score_min": float(score[idx].min().item()) if idx.numel() else 0.0,
            "score_max": float(score[idx].max().item()) if idx.numel() else 0.0,
            "score_mean": float(score[idx].mean().item()) if idx.numel() else 0.0,
            **stat_summary,
        }
        n_mod += 1
        n_ch += len(idx_list)
    if n_mod == 0:
        raise SystemExit(
            "[calib] protected-channel selection matched 0 SCLinear modules.")
    if missing:
        print(f"[calib] protected-channel: skipped {missing} modules missing "
              "act_scales.")
    ov = (" overrides=" + ",".join(f"{k}:{v:g}" for k, v in frac_overrides.items())
          if frac_overrides else "")
    print(f"[calib] protected-channel: selected {n_ch} channels across {n_mod} "
          f"linear modules (frac={frac}, metric={metric}, stoc_len={stoc_len}, "
          f"compensate={bool(compensate_budget)}{ov}).")
    return n_mod


def main():
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Calibration requires CUDA + Triton (sc_matmul is GPU-only)."
        )

    # Precision tracing during calibration would record σ-probe / FP-teacher /
    # knock-down work — with stale or null context — as if it were inference,
    # and (in trace mode) grow the buffer across the whole calibration. The
    # sweep exports SC_MP_TRACE for the PPL stage; explicitly disable it here.
    try:
        from scmp_kernels import trace as _sc_trace_mod
        if _sc_trace_mod.is_enabled():
            _sc_trace_mod.disable()
            print("[calib] SC_MP_TRACE set — tracing disabled during "
                  "calibration (probe workloads are not simulator input); "
                  "enable it on eval runs (ppl.py) instead.")
    except ImportError:
        pass

    torch.manual_seed(args.seed)

    levels = _expand_levels(args.mp_levels)
    operators = _parse_csv_set(args.operators)
    halve = bool(args.halve)
    sc_prec = int(args.sc_prec)
    chunk_d = int(args.chunk_d)
    # Sanity: levels must live in halved space when halve=True.
    halved_cap = 2 ** (sc_prec - 1) if halve else 2 ** sc_prec
    if max(levels) > halved_cap:
        raise ValueError(
            f"max(mp_levels)={max(levels)} exceeds halved cap {halved_cap} "
            f"for sc_prec={sc_prec} halve={halve}."
        )

    grad_weight = bool(args.loss_weight_by_grad)
    grad_on_sc = bool(args.grad_on_sc)
    protect_needs_grad = (
        float(args.protect_channel_frac) > 0.0
        and args.protect_channel_metric == "act_grad_weight"
    )
    if grad_on_sc and not grad_weight:
        # g_SC reuses the backward-hook g_row machinery; it is meaningless
        # without it, so turn it on implicitly.
        grad_weight = True
        print("[calib] --grad-on-sc implies --loss-weight-by-grad; enabling it.")

    grad_sc_stoclen = int(args.grad_sc_stoclen)
    grad_sc_draws = max(1, int(args.grad_sc_draws))
    if grad_on_sc:
        if grad_sc_stoclen <= 0:
            ref = args.budget_ref_stoc_len or max(levels)
            grad_sc_stoclen = max(1, int(round(args.budget_ratio * ref)))
        if grad_sc_stoclen > halved_cap:
            raise ValueError(
                f"--grad-sc-stoclen={grad_sc_stoclen} exceeds the halved cap "
                f"{halved_cap} for sc_prec={sc_prec} halve={halve}."
            )
        if not (min(levels) <= grad_sc_stoclen <= max(levels)):
            print(f"[calib] NOTE: grad_sc_stoclen={grad_sc_stoclen} is outside "
                  f"the MP level range [{min(levels)}, {max(levels)}].")
        owen_mode = os.environ.get("SC_OWEN_MODE", "bitrev").lower()
        if grad_sc_draws > 1 and owen_mode != "random":
            print(f"[calib] NOTE: grad_sc_draws={grad_sc_draws} but the SC RNG "
                  f"is deterministic (SC_OWEN_MODE={owen_mode}); draws will be "
                  f"identical. Set SC_OWEN_MODE=random for independent draws, "
                  f"or --grad-sc-draws 1 to skip the wasted compute.")

    # Cross-layer budget scope + per-group weighting. All weighted variants
    # share the SAME structure (global budget, per-group W_g, act σ quantile
    # WITHIN group) and differ only in how W_g is sourced.
    budget_scope = args.budget_scope
    cross_layer_weight = args.cross_layer_weight
    grad_as_group_weight = False
    if args.refine_mode == "sigma" and budget_scope != "global":
        budget_scope = "global"          # residual-fill only defined for the global solve
        print("[calib] --refine sigma forces --budget-scope global (the residual "
              "budget fill operates on the cross-layer global allocation).")
    if cross_layer_weight in ("measured", "measured_marg", "measured_curve"):
        budget_scope = "global"          # measured weights only make sense globally
        if grad_weight:
            grad_weight = False          # measured uses act objective within groups
            grad_on_sc = False
            print(f"[calib] --cross-layer-weight {cross_layer_weight} forces the "
                  "act (reconstruction-error) within-group objective; disabling "
                  "grad weighting.")
    elif cross_layer_weight == "grad_group":
        budget_scope = "global"          # group weights only make sense globally
        grad_weight = True               # need one backward to capture g_row
        grad_on_sc = False               # clean-trajectory grad, used as a GROUP mean
        grad_as_group_weight = True
        print("[calib] --cross-layer-weight grad_group: collecting clean "
              "loss-gradient, aggregating to a per-group cross-layer weight W_g "
              "(act σ quantile WITHIN group; no per-row g·σ multiply).")
    elif cross_layer_weight == "fisher":
        budget_scope = "global"          # group weights only make sense globally
        grad_weight = True               # need one backward to capture g_row
        grad_on_sc = False               # clean-trajectory grad
        grad_as_group_weight = False     # FisherMP has its own accumulation
        print("[calib] --cross-layer-weight fisher (FisherMP): per-group W_g = "
              "1+ε·(clip(mean‖y‖²‖g‖²)−1), reparam-invariant Gauss-Newton "
              "(linear ops%s). act σ quantile WITHIN group." %
              (" + qk/av per-class" if getattr(args, "fisher_attn", False)
               else "; attention W_g=1"))
    ref_sl = args.budget_ref_stoc_len or max(levels)
    measure_baseline_sl = int(args.measure_baseline_stoclen) or max(
        1, int(round(args.budget_ratio * ref_sl)))
    if cross_layer_weight == "measured_marg" and int(args.measure_probe_level) <= 0:
        # Marginal probe: a small step DOWN from baseline, grid-independent, so it
        # never lands on the floor/cliff (which is what makes plain 'measured'
        # inconsistent at int7/len96). ΔL is then a finite-difference ∂L/∂budget.
        measure_probe_sl = max(1, int(round(
            measure_baseline_sl * (1.0 - args.measure_marg_frac))))
        if measure_probe_sl >= measure_baseline_sl:
            measure_probe_sl = max(1, measure_baseline_sl - 1)
    else:
        measure_probe_sl = int(args.measure_probe_level) or min(levels)

    print(f"[calib] model={args.model_path} sc_prec={sc_prec} halve={halve} "
          f"levels={levels} budget_ratio={args.budget_ratio:.4f} "
          f"loss_weight_by_grad={grad_weight} grad_on_sc={grad_on_sc} "
          f"budget_scope={budget_scope} cross_layer_weight={cross_layer_weight}"
          + (f" grad_sc_stoclen={grad_sc_stoclen} grad_sc_draws={grad_sc_draws}"
             if grad_on_sc else "")
          + (f" measure_baseline_sl={measure_baseline_sl} "
             f"measure_probe_sl={measure_probe_sl}"
             if cross_layer_weight == "measured" else ""))
    if protect_needs_grad:
        print("[calib] --protect-channel-metric act_grad_weight: will run a "
              "pre-calibration backward pass for offline channel salience; "
              "the main row-threshold objective is unchanged.")

    # Load model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = load_sc_model(args.model_path, dtype=torch.float16, device_map="auto")
    model.eval()
    # open-issue #5: if deployment applies SmoothQuant, attach the SAME smooth
    # vectors here so the per-level σ measurement quantizes the deployed (a/s)
    # activation. calib_smoothquant is set on the calibrator below; the hook
    # reads module.smooth_scales. Enabled by --calib-smoothquant (or the
    # USE_SMOOTHQUANT env the eval already uses). No-op otherwise.
    calib_smoothquant = bool(args.calib_smoothquant or
                             os.environ.get("USE_SMOOTHQUANT", "") == "1")
    sq_alpha = sq_path = None
    sq_act_scales = None
    if calib_smoothquant:
        # Self-sufficient: resolve the SAME act_scales file + alpha the eval
        # applies (ACT_SCALES_DIR + SQ_ALPHA). The old delegation to
        # apply_smoothquant_from_env silently attached to 0 modules unless the
        # separate USE_SMOOTHQUANT/SMOOTHQUANT_SCALES env pair was ALSO set —
        # i.e. --calib-smoothquant alone still calibrated unsmoothed.
        from model.smoothquant_apply import apply_smoothquant_to_model
        sq_path = os.environ.get("SMOOTHQUANT_SCALES")
        if not sq_path:
            _root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            act_dir = os.environ.get(
                "ACT_SCALES_DIR",
                os.path.join(_root, "benchmark", "quant", "act_scales"))
            sq_path = os.path.join(
                act_dir,
                "act_scales_" + args.model_path.replace("/", "_") + ".pt")
        if not os.path.isfile(sq_path):
            raise SystemExit(
                f"--calib-smoothquant: act_scales file not found: {sq_path} "
                "(set ACT_SCALES_DIR or SMOOTHQUANT_SCALES to the same scales "
                "the eval applies)")
        sq_alpha = float(os.environ.get(
            "SQ_ALPHA", os.environ.get("SMOOTHQUANT_ALPHA", "0.5")))
        sq_act_scales = torch.load(sq_path, map_location="cpu", weights_only=True)
        n_sq = apply_smoothquant_to_model(model, sq_act_scales, alpha=sq_alpha)
        if n_sq == 0:
            raise SystemExit(
                f"--calib-smoothquant: 0 SCLinear modules matched {sq_path} — "
                "act_scales keys do not match this model's module names")
        print(f"[calib] #5 fix: SmoothQuant attached to {n_sq} SCLinear modules "
              f"for the σ measurement (alpha={sq_alpha}, scales={sq_path}).")
    # SC must be wired (so SCLinear instances exist and _sc_op_name is tagged).
    # In the act / FP-grad paths the SC dispatch is DISABLED so forwards run in
    # FP teacher mode. In the STE g_SC path SCLinear runs with sc_ste_grad on:
    # forward value = SC-noisy, gradient = FP Jacobian (the attention noisy
    # path is handled separately by the calib_eager patch).
    model.config.use_sc_attn = False
    model.config.sc_prec = sc_prec
    model.config.sc_halve_bipolar_stoc_len = halve
    model.config.sc_linear_chunk_d = chunk_d
    if grad_on_sc:
        model.config.use_sc_linear = True
        model.config.sc_ste_grad = True
        model.config.sc_mp_config = None          # uniform noise, no MP dispatch
        model.config.sc_stoc_len = grad_sc_stoclen
    else:
        model.config.use_sc_linear = False
        model.config.sc_ste_grad = False

    if grad_weight or protect_needs_grad:
        # Autograd setup: gradients must flow to matmul outputs so backward
        # hooks fire, but we must NOT allocate parameter-gradient buffers.
        _prepare_model_for_backward(
            model, args,
            "protected-channel GN stats" if protect_needs_grad and not grad_weight
            else "calibration gradients",
        )

    total_blocks = getattr(model.config, "_sc_total_blocks", None)
    if total_blocks is None:
        # Fallback: count decoder layers from model.model.layers.
        try:
            total_blocks = len(model.model.layers)
        except Exception:
            total_blocks = 1
    print(f"[calib] total_blocks={total_blocks}")

    # Calibration token stream.
    enc = _load_calibration_token_stream(tokenizer, args)
    total_tokens = enc.shape[0]
    needed = args.num_calib_sequences * args.ctx_len
    if total_tokens < needed:
        print(f"[calib] WARN: dataset only has {total_tokens} tokens, want {needed}.")
    print(f"[calib] using up to {min(total_tokens, needed)} tokens "
          f"({args.num_calib_sequences} × {args.ctx_len}).")

    # int_swap produces the hybrid-mask RANKING, not an MP table: it needs the
    # model and the token stream and nothing else, so it runs here and returns
    # before the (expensive) σ collection and the export chain.
    if cross_layer_weight == "int_swap":
        device = next(model.parameters()).device
        if args.int_swap_ref_level:
            ref_level = int(args.int_swap_ref_level)
        else:
            # The deployed AVERAGE (budget_ratio · ref), not the ladder floor and
            # not its top: the mask trades against what a typical entry actually
            # gets, and both ladder ends are corners the allocator rarely uses.
            ref_level = int(round(args.budget_ratio *
                                  (args.budget_ref_stoc_len or max(levels))))
            ref_level = min(max(ref_level, min(levels)), max(levels))
        print(f"[calib] int_swap: per-(operator, layer) SC→INT{args.int_swap_bits} "
              f"swap gain, ref={args.int_swap_ref} (level={ref_level} halved), "
              f"{args.measure_windows} {args.int_swap_window_sampling} windows "
              f"× {args.measure_ctx} ctx, score={args.int_swap_score}, "
              f"{len(operators)} ops × {total_blocks} layers.")
        checkpoint_path = args.int_swap_checkpoint
        if checkpoint_path == "auto":
            checkpoint_path = args.output_json + ".partial.json"
        gains, swap_info, split = _measure_group_int_swap(
            model, enc, total_blocks, operators,
            int_bits=args.int_swap_bits,
            ref_mode=args.int_swap_ref,
            ref_level=ref_level,
            mp_config_path=args.int_swap_mp_config,
            n_windows=args.measure_windows,
            ctx=args.measure_ctx,
            device=device,
            int_chunk=chunk_d,
            window_sampling=args.int_swap_window_sampling,
            window_seed=args.seed,
            score_mode=args.int_swap_score,
            lcb_z=args.int_swap_lcb_z,
            stability_folds=args.int_swap_stability_folds,
            checkpoint_path=checkpoint_path,
        )
        payload = build_int_swap_payload(
            gains, swap_info, split,
            model_path=args.model_path,
            total_blocks=total_blocks,
            operators=operators,
        )
        _atomic_write_json(args.output_json, payload)
        if "split_half" in payload:
            sh = payload["split_half"]
            print(f"[int_swap] SPLIT-HALF top-{sh['top_frac']:.0%} set overlap="
                  f"{sh['top_overlap']:.0%} spearman={sh['spearman']:+.3f} "
                  f"({sh['windows_per_half']} windows/half) — this is the gate: "
                  "a low overlap means the ranking is noise, not signal.")
        if "cross_fold" in payload:
            cf = payload["cross_fold"]
            print(f"[int_swap] {cf['folds']}-FOLD pairwise top-"
                  f"{cf['top_frac']:.0%} overlap mean={cf['mean_top_overlap']:.0%} "
                  f"min={cf['min_top_overlap']:.0%}; spearman "
                  f"mean={cf['mean_spearman']:+.3f} "
                  f"min={cf['min_spearman']:+.3f}.")
        print(f"[calib] int_swap: wrote {args.output_json}")
        return

    protected_gn_stats = None
    if protect_needs_grad:
        device = next(model.parameters()).device
        protected_gn_stats = _collect_protected_channel_gn_stats(
            model, enc, args, device)

    # Calibrator + hooks + monkey-patched attention.
    calibrator = ThresholdCalibrator(
        levels=levels,
        operators=operators,
        total_blocks=total_blocks,
        layer_buckets=args.layer_buckets,
        budget_ratio=args.budget_ratio,
        budget_ref_stoc_len=args.budget_ref_stoc_len,
        max_units_per_call=args.max_units_per_call,
        min_bucket_units=args.min_bucket_units,
        rng_seed=args.seed,
        loss_weight_by_grad=grad_weight,
        grad_g_pow=args.grad_g_pow,
        grad_s_pow=args.grad_s_pow,
        budget_scope=budget_scope,
        grad_as_group_weight=grad_as_group_weight,
        grad_group_pow=args.grad_group_pow,
        grad_group_clip=args.grad_group_clip,
        refine_mode=args.refine_mode,
    )
    calibrator.calib_smoothquant = calib_smoothquant
    calibrator.fisher_group = (cross_layer_weight == "fisher")
    calibrator.fisher_attn = bool(getattr(args, "fisher_attn", False)) and \
        (cross_layer_weight == "fisher")
    calibrator.objective = args.objective
    # Budget weighting: rows (latency, default) or MACs (FLOP/energy → iso-compute).
    calibrator.budget_weight = args.budget_weight
    if args.budget_weight == "macs":
        if not args.mac_weights_trace:
            raise SystemExit("[calib] --budget-weight macs requires "
                             "--mac-weights-trace <scmp trace json>.")
        _tr = json.load(open(args.mac_weights_trace))
        _r, _m = defaultdict(float), defaultdict(float)
        for g in _tr["groups"]:
            _r[g["op"]] += g["rows"]; _m[g["op"]] += g["macs"]
        calibrator.mac_per_row = {op: _m[op] / _r[op] for op in _m if _r[op] > 0}
        print("[calib] budget-weight=MACS (iso-compute). macs/row: "
              + ", ".join(f"{op}={calibrator.mac_per_row[op]:.2e}"
                          for op in sorted(calibrator.mac_per_row)))
    # ---- act_global_v2 knobs ----
    if args.argmin_pricing == "macs":
        if args.budget_weight != "macs":
            raise SystemExit("[calib] --argmin-pricing macs requires "
                             "--budget-weight macs (mac_per_row source).")
        if budget_scope != "global":
            raise SystemExit("[calib] --argmin-pricing macs requires "
                             "--budget-scope global.")
        print("[calib] argmin-pricing=MACS (KKT-consistent per-row price; "
              "budget stays R_g-weighted; residual fill ranks by the same "
              "price, budget-accounts by rep_g).")
    calibrator.argmin_pricing = args.argmin_pricing
    calibrator.metric_select = args.metric_select
    calibrator.metric_select_margin = float(args.metric_select_margin)
    if args.metric_select == "auto":
        print("[calib] metric-select=AUTO (per-operator signed rho selection "
              "over amax/l2/crest).")
    pc_frac_overrides = {}
    for pair in str(args.protect_channel_frac_overrides or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        op, _, val = pair.partition(":")
        if not val:
            raise SystemExit(
                f"[calib] --protect-channel-frac-overrides expects 'op:frac' pairs, got {pair!r}")
        pc_frac_overrides[op.strip()] = float(val)
    if pc_frac_overrides:
        print(f"[calib] protected-channel per-op frac overrides: {pc_frac_overrides}")
    _configure_protected_channels(
        calibrator, model, sq_act_scales,
        frac=args.protect_channel_frac,
        metric=args.protect_channel_metric,
        stoc_len=args.protect_channel_stoc_len,
        compensate_budget=args.protect_compensate_budget,
        gn_stats=protected_gn_stats,
        frac_overrides=pc_frac_overrides,
    )
    merger = PendingMerger(calibrator)

    hooks = []
    sclinear_hook = _make_sclinear_hook(
        calibrator, merger, levels, sc_prec, halve, chunk_d,
        grad_on_sc=grad_on_sc,
    )
    for mod in model.modules():
        if isinstance(mod, SCLinear):
            hooks.append(mod.register_forward_hook(sclinear_hook))

    restore_attn = _patch_attention_for_calibration(
        calibrator, merger, levels, sc_prec, halve,
        grad_on_sc=grad_on_sc, grad_sc_stoclen=grad_sc_stoclen,
    )

    draws = grad_sc_draws if grad_on_sc else 1

    try:
        device = next(model.parameters()).device
        for i, window in enumerate(_iter_calib_windows(
            enc, args.ctx_len, args.num_calib_sequences,
        )):
            print(f"[calib] window {i + 1}/{args.num_calib_sequences}")
            ids = window.unsqueeze(0).to(device)
            if grad_weight:
                # One forward + ONE backward per draw yields g_row for ALL
                # layers at once (∂L/∂y has the same shape as y). The forward
                # hooks stash (metric, errors) + arm output backward hooks; the
                # backward populates g_row; flush() accumulates per draw and
                # finalize() averages g_row over draws and feeds the calibrator.
                # labels = input_ids (next-token CE = the LM loss). For the FP /
                # FP-grad path draws==1, so this is byte-identical to one
                # forward+backward+flush+add.
                for d in range(draws):
                    if draws > 1:
                        _reseed_sc_for_draw(d, args.seed)
                    model.zero_grad(set_to_none=True)
                    out = model(input_ids=ids, labels=ids)
                    loss = out.loss
                    loss.backward()
                    merger.flush()
                merger.finalize()
            else:
                with torch.no_grad():
                    _ = model(input_ids=ids)
    finally:
        for h in hooks:
            h.remove()
        restore_attn()

    # Per-group cross-layer weights W_g. Done AFTER the act σ-collection (hooks
    # removed). Two sources, both feeding the SAME _normalize_group_weights →
    # calibrator.group_weights slot consumed by the global solve:
    #   measured / measured_marg → ΔLoss knock-down probe (extra SC forwards)
    #   grad_group               → per-group mean of g_row (already accumulated)
    # Metric-fidelity diagnostic — computed BEFORE any W_g / curve override so
    # every method stores the same, meaningful ρ (measured_curve used to replace
    # the per-row errors with a per-group-constant curve first, which reduced
    # its stored ρ to argsort tie-breaking noise).
    rho = calibrator.metric_fidelity_rho()
    print(f"[calib] metric-fidelity ρ(metric, {args.objective}-benefit) per op "
          "(≈1 good; low/neg ⇒ metric misranks → headroom):")
    for op in sorted(rho, key=lambda o: rho[o]["rho"] if rho[o]["rho"] == rho[o]["rho"] else 9):
        d = rho[op]
        print(f"    {op:<11} ρ={d['rho']:+.3f}  n={d['n']:>7}  "
              f"objective-benefit_mean={d['benefit_mean']:.4f}")

    # act_global_v2: signed per-operator dispatch-metric selection (rewrites
    # the stored metric arrays for switched ops; thresholds + the exported
    # dispatch_metrics then reflect the deployed metric). No-op unless
    # --metric-select auto.
    switched = calibrator.select_dispatch_metrics()
    if switched:
        print("[calib] metric-select switches (deployed by the runtime):")
        for op, d in sorted(switched.items()):
            print(f"    {op:<11} -> {d['metric']}"
                  f"{' (inverted)' if d['sign'] < 0 else ''}"
                  f"  ρ={d['rho_signed']:+.3f} (amax was {d['rho_amax']:+.3f})")
    elif calibrator.metric_select == "auto":
        print("[calib] metric-select auto: no operator cleared the margin — "
              "amax everywhere (table carries no dispatch_metrics).")

    measured_info = None
    if cross_layer_weight in ("measured", "measured_marg"):
        is_marg = cross_layer_weight == "measured_marg"
        print(f"[calib] measuring per-group ΔLoss sensitivity "
              f"({cross_layer_weight}: baseline_sl={measure_baseline_sl}, "
              f"probe_sl={measure_probe_sl}"
              + (f", marg_frac={args.measure_marg_frac}" if is_marg else "")
              + ")...")
        raw_w, L0 = _measure_group_sensitivity(
            model, enc, levels, total_blocks, args.layer_buckets, operators,
            baseline_sl=measure_baseline_sl, probe_sl=measure_probe_sl,
            n_windows=args.measure_windows, ctx=args.measure_ctx, device=device,
        )
        norm_w = _normalize_group_weights(raw_w, floor_frac=args.measure_floor_frac)
        calibrator.group_weights = norm_w
        measured_info = {
            "source": cross_layer_weight,
            "baseline_sl": measure_baseline_sl,
            "probe_sl": measure_probe_sl,
            "marg_frac": (args.measure_marg_frac if is_marg else None),
            "floor_frac": args.measure_floor_frac,
            "baseline_loss": L0,
            "raw_delta_loss": {f"{op}:l{lb}": float(v) for (op, lb), v in raw_w.items()},
            "normalized_weights": {f"{op}:l{lb}": float(v) for (op, lb), v in norm_w.items()},
        }
    elif cross_layer_weight == "grad_group":
        raw_w = calibrator.grad_group_raw_weights()
        norm_w = _normalize_group_weights(raw_w, floor_frac=args.measure_floor_frac)
        calibrator.group_weights = norm_w
        print(f"[calib] grad_group: built {len(norm_w)} per-group weights "
              f"from mean |∂L/∂y|^{args.grad_group_pow} "
              f"(clip p{args.grad_group_clip}, floor {args.measure_floor_frac}).")
        measured_info = {
            "source": "grad_group",
            "grad_group_pow": args.grad_group_pow,
            "grad_group_clip": args.grad_group_clip,
            "floor_frac": args.measure_floor_frac,
            "raw_group_grad": {f"{op}:l{lb}": float(v) for (op, lb), v in raw_w.items()},
            "normalized_weights": {f"{op}:l{lb}": float(v) for (op, lb), v in norm_w.items()},
        }

    elif cross_layer_weight == "fisher":
        raw_w = calibrator.fisher_group_raw_weights()
        eps, kappa = float(args.fisher_eps), float(args.fisher_kappa)
        # Normalize WITHIN each operator class so the linear pool, qk, and av each
        # get their OWN mean-1 scale before the bounded blend. qk/av GN magnitudes
        # (softmax scores / attention outputs) sit at very different scales than
        # linear-projection outputs; a single shared pool + κ clip would saturate
        # every attention group to κ and every linear to 1/κ (a degenerate 2-vs-0.5
        # split). Per-class keeps W_g a WITHIN-operator cross-layer reweight and
        # leaves cross-operator budget balance to the σ + rep_g global-λ solve.
        # With --fisher-attn OFF, raw_w has no qk/av keys → those pools are empty →
        # byte-identical to the shipped linear-only fisher.
        def _pool(pred):
            sub = {k: v for k, v in raw_w.items() if pred(k[0])}
            return (_normalize_group_weights(sub, floor_frac=args.measure_floor_frac)
                    if sub else {})
        norm_w = {}
        norm_w.update(_pool(lambda op: op in LINEAR_OPS))
        norm_w.update(_pool(lambda op: op == "qk"))   # own pool
        norm_w.update(_pool(lambda op: op == "av"))   # own pool
        # Bounded blend around 1 so budget interpretation holds and no group is
        # starved: W_g = 1 + eps·(clip(Ŵ_g_norm, 1/κ, κ) − 1).
        blended = {k: 1.0 + eps * (min(max(v, 1.0 / kappa), kappa) - 1.0)
                   for k, v in norm_w.items()}
        calibrator.group_weights = blended
        n_attn = sum(1 for (op, _lb) in blended if op in ATTN_OPS)
        print(f"[calib] fisher: built {len(blended)} W_g ({n_attn} attention) "
              f"(eps={eps}, kappa={kappa}, fisher_attn={calibrator.fisher_attn}).")
        measured_info = {
            "source": "fisher", "fisher_eps": eps, "fisher_kappa": kappa,
            "fisher_attn": bool(calibrator.fisher_attn),
            "floor_frac": args.measure_floor_frac,
            "raw_group_fisher": {f"{op}:l{lb}": float(v) for (op, lb), v in raw_w.items()},
            "blended_weights": {f"{op}:l{lb}": float(v) for (op, lb), v in blended.items()},
        }

    elif cross_layer_weight == "measured_curve":
        print("[calib] measuring per-group per-level ΔLoss CURVE (measured_curve): "
              f"reference = all groups at max sl={max(levels)}, "
              f"{args.measure_windows} windows × {args.measure_ctx} ctx...")
        curves, L0 = _measure_group_curve(
            model, enc, levels, total_blocks, args.layer_buckets, operators,
            n_windows=args.measure_windows, ctx=args.measure_ctx, device=device,
        )
        n_over = calibrator.apply_measured_curve(curves)
        print(f"[calib] measured_curve: overrode {n_over} group error curves with "
              "true per-level ΔLoss → allocation = min Σ ΔLoss + λ·budget "
              "(per-group uniform, no recon-error factor).")
        measured_info = {
            "source": "measured_curve",
            "baseline_loss": L0,
            "reference": f"all_max_sl={max(levels)}",
            "delta_loss_curve": {f"{op}:l{lb}": [float(x) for x in c]
                                 for (op, lb), c in curves.items()},
        }

    # Export. (metric_fidelity_rho was computed pre-override, before the W_g
    # chain above.)
    payload, summary_rows = calibrator.export()
    payload["metric_fidelity_rho"] = {op: rho[op]["rho"] for op in rho}
    payload["model_path"] = args.model_path
    payload["sc_prec"] = sc_prec
    payload["halve_bipolar_stoc_len"] = halve
    payload["operators"] = sorted(operators)
    payload["num_calib_sequences"] = args.num_calib_sequences
    payload["ctx_len"] = args.ctx_len
    payload["loss_weight_by_grad"] = grad_weight
    if grad_weight:
        payload["grad_g_pow"] = args.grad_g_pow
        payload["grad_s_pow"] = args.grad_s_pow
    payload["grad_on_sc"] = grad_on_sc
    if grad_on_sc:
        payload["grad_sc_stoclen"] = grad_sc_stoclen
        payload["grad_sc_draws"] = grad_sc_draws
    payload["cross_layer_weight"] = cross_layer_weight
    if measured_info is not None:
        payload["measured"] = measured_info
    # method tag (the JSON consumer / AdaptiveMPConfig ignores this — schema is
    # identical across methods):
    #   act / grad / grad_sc            (per_bucket scope)
    #   act_global / grad_global        (global scope, uniform W_g, per-row obj)
    #   measured_xlayer                 (global, floor-knockdown ΔLoss W_g, act within)
    #   measured_marg_xlayer            (global, MARGINAL ΔLoss W_g, act within)  ← fix 1
    #   grad_group_xlayer               (global, per-group mean-g W_g, act within) ← fix 2
    if cross_layer_weight == "measured":
        method = "measured_xlayer"
    elif cross_layer_weight == "measured_marg":
        method = "measured_marg_xlayer"
    elif cross_layer_weight == "grad_group":
        method = "grad_group_xlayer"
    elif cross_layer_weight == "fisher":
        method = "fisher_xlayer"
    elif cross_layer_weight == "measured_curve":
        method = "measured_curve"
    elif grad_on_sc:
        method = "grad_sc_global" if budget_scope == "global" else "grad_sc"
    elif grad_weight:
        method = "grad_global" if budget_scope == "global" else "grad"
    else:
        method = "act_global" if budget_scope == "global" else "act"
    if args.refine_mode == "sigma":
        method = method + "_refine"
    if args.budget_weight == "macs":
        method = method + "_fw"          # FLOP/energy-weighted (iso-compute) budget
    if calib_smoothquant:
        method = method + "_sq"
        payload["calib_smoothquant"] = True
        payload["calib_smoothquant_alpha"] = sq_alpha
        payload["calib_smoothquant_scales"] = sq_path
    if args.objective == "sigma2":
        method = method + "_s2"
        payload["objective"] = "sigma2"
    elif args.objective == "delta_sigma2":
        method = method + "_ds2"
        payload["objective"] = "delta_sigma2"
    if calibrator.protected_channel_indices:
        metric_suffix = (
            "" if args.protect_channel_metric == "act_weight"
            else f"_{args.protect_channel_metric}"
        )
        method = (
            method
            + f"_pc{args.protect_channel_frac:g}x{args.protect_channel_stoc_len}"
            + metric_suffix
        )
        if args.protect_compensate_budget:
            method = method + "_comp"
    payload["method"] = method
    # Preserve the EXACT settings so every table is self-documenting: reusable
    # (MP_CONFIG_JSON=<wrapper>, no recalibration) AND reproducible (re-run
    # calib_command to regenerate byte-identically). Grep the table or the
    # launcher MANIFEST for these fields — nothing has to be re-derived.
    payload["calib_command"] = " ".join(sys.argv)
    payload["mp_levels"] = args.mp_levels
    payload["seed"] = int(args.seed)
    payload["budget_weight"] = args.budget_weight
    if args.budget_weight == "macs":
        payload["mac_per_row"] = calibrator.mac_per_row
    if (args.protect_channel_metric == "act_grad_weight"
            and payload.get("protected_channels")):
        payload["protected_channels"]["stat_ctx_len"] = (
            int(args.protect_channel_stat_ctx) or int(args.ctx_len))
        payload["protected_channels"]["stat_num_sequences"] = (
            int(args.protect_channel_stat_sequences)
            or int(args.num_calib_sequences))

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[calib] wrote {out_json}")

    summary_csv = args.output_summary_csv
    if summary_csv is None:
        summary_csv = str(out_json.with_suffix("")) + "_summary.csv"
    _write_summary_csv(summary_csv, summary_rows)
    print(f"[calib] wrote {summary_csv}")


if __name__ == "__main__":
    main()
