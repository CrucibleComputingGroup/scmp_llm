"""Phase 3 allocator: solve per-band stream lengths inside a row.

Two decisions, deliberately made by different means:

BAND ASSIGNMENT (which chunks go where) uses a closed-form importance proxy
  A_c = mean_r |x[r, chunk c]|_inf  *  ||W[:, chunk c]||_F
which is the water-filling weight of that chunk's partial product: SC error on
a chunk scales with the quantization scale it is normalized by (its row amax --
the value the kernel ALREADY computes as the group scale) times the weight
column norms it multiplies. It needs no SC calls, and a 2-way split is coarse
enough to be robust to proxy error.

BAND LENGTHS are MEASURED, never proxied. For each (operator, layer bucket) and
each band we measure the band's partial deviation
  E_b(L) = sum_rows || SC(x_b, W_b, L) - x_b @ W_b^T ||^2
on a grid of L, then choose each rung's (L_0, L_1) to minimize E_0 + E_1 subject
to the exact per-rung iso-cost identity. Measuring per band is FLOP-NEUTRAL
versus measuring the row: the K-partials are additive, so per level you do
n_bands calls of width w_b instead of one call of width R = sum_b w_b.

MODELLING CAVEAT, stated because it bounds what this can claim: chunks of one
call share a single cum_indicator / RNG prefix, so per-band SC errors are
positively CORRELATED. Adding E_b across bands therefore UNDER-estimates the
row error. It is a monotone ranking signal, which is all the solve needs, but
it means Phase 3's predicted gain is not a promise -- the full-protocol PPL is
the arbiter, exactly as Phase 2 treats its own sigma model.

Usage:
    python -m benchmark.ppl.mp_kbands_calib \\
        --parent <bundle_dir> --model_path <hf> --out alloc.json \\
        --n-bands 2 --num-windows 8 --max-rows 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)

from model.sc_common import SCLinear  # noqa: E402
from scmp_kernels import sc_matmul as _sc_matmul  # noqa: E402
from scmp_kernels.mp.config import residual_chunk_widths  # noqa: E402
from benchmark.ppl.mp_kbands import (  # noqa: E402
    BANDABLE_OPS, CHUNK_D, bucket_ladder, load_parent, residual_widths,
    band_widths as _band_widths, rung_candidates,
)

SC_PREC = 8
HALVE = True


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def _chunk_cols(chunk_idx: int, residual_width: int, device):
    start = chunk_idx * CHUNK_D
    return torch.arange(start, min(start + CHUNK_D, residual_width),
                        device=device)


def chunk_importance(x_res: torch.Tensor, w_res: torch.Tensor,
                     n_chunks: int) -> torch.Tensor:
    """A_c = mean_r |x_c|_inf * ||W_c||_F, one scalar per residual chunk.

    Both factors are free at runtime: the first IS the group quantization scale
    the kernel already reduces, the second is a static weight norm.
    """
    R = x_res.shape[1]
    out = torch.zeros(n_chunks, dtype=torch.float64)
    for c in range(n_chunks):
        cols = _chunk_cols(c, R, x_res.device)
        xc = x_res.index_select(1, cols)
        wc = w_res.index_select(1, cols)
        scale = xc.abs().amax(dim=-1).mean()          # mean over rows
        out[c] = float(scale) * float(wc.norm())
    return out


def band_error_curve(x_band: torch.Tensor, w_band: torch.Tensor,
                     lengths, smooth_band=None) -> tuple[dict, dict]:
    """E(L) on two DISJOINT row halves: (solve-on, held-out).

    The solve searches ~100 candidates per rung against a measured objective.
    With one sample that is a textbook setup for selecting sampling noise --
    the same failure mode as the two earlier bugs here (a two-sided residue
    tolerance, and snapping to the nearest grid point) where the argmin found
    and exploited whatever slack existed. Splitting the rows lets the caller
    SCORE the chosen allocation on data it did not pick with, so
    "predicted gain" can be separated from "gain that generalizes".

    Splitting by row parity keeps both halves i.i.d. from the same call and
    costs one SC pass, not two: each half is half the rows.
    """
    a_idx = torch.arange(0, x_band.shape[0], 2, device=x_band.device)
    b_idx = torch.arange(1, x_band.shape[0], 2, device=x_band.device)
    out_a, out_b = {}, {}
    for half, idx, out in ((0, a_idx, out_a), (1, b_idx, out_b)):
        if idx.numel() == 0:
            continue
        xh = x_band.index_select(0, idx).contiguous()
        fp = (xh.float() @ w_band.float().t())
        for L in lengths:
            sc = _sc_matmul(
                xh, w_band, granularity="per_row", mode="bipolar",
                sc_prec=SC_PREC, stoc_len=int(L), chunk_d=CHUNK_D,
                halve_bipolar_stoc_len=HALVE, smooth_scales=smooth_band,
            )
            out[int(L)] = float(((sc - fp) ** 2).sum())
    return out_a, out_b


def monotonize(curve: dict) -> dict:
    """Force E(L) non-increasing in L (running minimum from the left).

    SC reconstruction error genuinely falls as the stream lengthens, so any
    rise between adjacent measured points is sampling noise. Leaving it in
    breaks two things at once: water-filling's convexity assumption, and the
    greedy slack-spender, which requires a POSITIVE marginal gain to take a
    step and therefore stalls with budget unspent the moment it meets a noisy
    uptick. That is not hypothetical -- it left q_proj:t0:l0 rung 3 at 44.93
    against a parent rung of 48, i.e. 3.07 cycles thrown away, which the
    one-sided underspend guard then (correctly) rejected.

    Running-min is the cheapest projection onto the monotone cone that never
    INCREASES a measured value, so it cannot manufacture an optimistic curve:
    every returned point is <= something actually measured.
    """
    out, best = {}, None
    for L in sorted(curve):
        v = curve[L]
        best = v if best is None else min(best, v)
        out[L] = best
    return out


def score_on(curves: list[dict], parent: list[int],
             ladders: list[list[int]]) -> dict:
    """Predicted error ratio of a FIXED allocation against a curve set.

    Used to score the solve-half's choice on the held-out half. Reuses the
    same log-log interpolation the solve uses so the two numbers are directly
    comparable.
    """
    import bisect
    import math

    def err(b, L):
        c = curves[b]
        grid = sorted(c)
        if L in c:
            return c[L]
        i = bisect.bisect_left(grid, L)
        if i == 0:
            return c[grid[0]]
        if i >= len(grid):
            return c[grid[-1]]
        x0, x1 = grid[i - 1], grid[i]
        y0, y1 = c[x0], c[x1]
        if y0 <= 0.0 or y1 <= 0.0:
            t = (L - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
        t = (math.log(L) - math.log(x0)) / (math.log(x1) - math.log(x0))
        return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))

    n_bands = len(curves)
    out = {}
    for k, L_par in enumerate(parent):
        base = sum(err(b, L_par) for b in range(n_bands))
        got = sum(err(b, ladders[b][k]) for b in range(n_bands))
        out[str(L_par)] = round(got / base, 6) if base else 1.0
    return out


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

def solve_bucket_waterfill(parent: list[int], widths: list[int],
                           curves: list[dict], cap: int = 128,
                           floor: int = 1) -> tuple[list[list[int]], dict]:
    """EXACT per-rung allocation by Lagrangian water-filling. Any band count.

    Why this replaces the one-hot-band sweep: that move set perturbs ONE band
    and makes the others pay back proportionally, which is adequate for 2 bands
    and increasingly hopeless as bands multiply -- with B bands the reachable
    set is O(B * max_delta) points out of a B-dimensional space. Water-filling
    solves the actual problem:

        minimise  sum_b E_b(L_b)      s.t.  sum_b w_b * L_b <= sum_b w_b * L_par

    E_b(L) is measured, decreasing and (empirically) convex in L, so the
    Lagrangian relaxation is tight: for a multiplier lam, each band independently
    takes L_b(lam) = argmin_L [E_b(L) + lam * w_b * L], and the total cost is
    monotone in lam. Bisect lam to land on the budget. That is exact for
    separable convex problems and costs O(B * |grid| * iters) with NO dependence
    on the number of bands beyond a linear factor -- which is what makes a
    per-chunk (B = n_chunks) allocation tractable at all.

    Budget is one-sided (<=), so this never overspends, same guarantee as
    rung_candidates. Lengths are clamped to [floor, cap]; cap is the halved
    stream limit, above which streams WRAP.
    """
    n_bands = len(widths)
    grid = sorted(curves[0])
    grid = [g for g in grid if floor <= g <= cap]
    total_w = float(sum(widths))

    def err(b, L):
        return _interp_logf(curves[b], grid, L)

    def alloc_at(lam):
        """Each band's independent argmin at multiplier lam."""
        out = []
        for b in range(n_bands):
            best_L, best_J = grid[0], None
            for L in grid:
                J = err(b, L) + lam * widths[b] * L
                if best_J is None or J < best_J:
                    best_L, best_J = L, J
            out.append(best_L)
        return out

    def cost(lens):
        return sum(widths[b] * lens[b] for b in range(n_bands)) / total_w

    ladders = [[0] * len(parent) for _ in range(n_bands)]
    report = {}
    for k, L_par in enumerate(parent):
        budget = float(L_par)
        # bracket lam: lam=0 -> longest (most expensive); large lam -> shortest
        lo_lam, hi_lam = 0.0, 1.0
        while cost(alloc_at(hi_lam)) > budget and hi_lam < 1e12:
            hi_lam *= 10.0
        best = None
        for _ in range(60):                      # bisection on the multiplier
            mid = 0.5 * (lo_lam + hi_lam)
            lens = alloc_at(mid)
            c = cost(lens)
            if c <= budget:
                best = lens                      # feasible; try cheaper lam
                hi_lam = mid
            else:
                lo_lam = mid
        if best is None:
            best = [L_par] * n_bands             # parent is always feasible
        # Spend any leftover budget greedily on the best marginal return, so a
        # conservative bisection does not leave cycles unused.
        improved = True
        while improved:
            improved = False
            slack = budget - cost(best)
            # -inf, NOT 0.0. With monotonized curves a flat region has gain
            # EXACTLY 0, and a strict `gain > 0` test then refuses to spend
            # remaining budget at all -- which is how 4-band q_proj still left
            # 3.07 cycles unspent even after monotonizing. Taking a zero-gain
            # step can never hurt (the curve is non-increasing), and spending
            # the budget is what keeps the cell honestly iso-compute.
            cand, cand_gain = None, float("-inf")
            for b in range(n_bands):
                nxt = [g for g in grid if g > best[b]]
                if not nxt:
                    continue
                L2 = nxt[0]
                dcost = widths[b] * (L2 - best[b]) / total_w
                if dcost > slack:
                    continue
                gain = err(b, best[b]) - err(b, L2)
                if gain > cand_gain:
                    cand, cand_gain = (b, L2), gain
            if cand is not None:
                best[cand[0]] = cand[1]
                improved = True
        base = sum(err(b, L_par) for b in range(n_bands))
        got = sum(err(b, best[b]) for b in range(n_bands))
        for b in range(n_bands):
            ladders[b][k] = int(best[b])
        realized = cost(best)
        report[str(L_par)] = {
            "lengths": [int(x) for x in best],
            "realized_mac_mean": round(realized, 5),
            "residue": round(realized - L_par, 5),
            "predicted_err_ratio": round(got / base, 6) if base else 1.0,
        }
    return ladders, report


def _interp_logf(curve: dict, grid: list, L: float) -> float:
    """Log-log interpolation of a measured error curve (see solve_bucket)."""
    import bisect
    import math
    if L in curve:
        return curve[L]
    i = bisect.bisect_left(grid, L)
    if i == 0:
        return curve[grid[0]]
    if i >= len(grid):
        return curve[grid[-1]]
    x0, x1 = grid[i - 1], grid[i]
    y0, y1 = curve[x0], curve[x1]
    if y0 <= 0.0 or y1 <= 0.0:
        t = (L - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    t = (math.log(L) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))


def solve_bucket(parent: list[int], widths: list[int], curves: list[dict],
                 tol: float, max_delta: int) -> tuple[list[list[int]], dict]:
    """Per-rung (L_0..L_{B-1}) minimizing summed band error at iso-cost.

    Sweeps the hot band's offset; the remaining bands pay it back in proportion
    to width (rung_candidates), which never overspends -- so every candidate is
    already at or UNDER the parent MAC cost.
    Band lengths are clamped to the measured grid rather than extrapolated --
    an unmeasured length would be a guess dressed as an optimum.

    SUPERSEDED by solve_bucket_waterfill for n_bands > 2; kept because the
    published 4B t48 result (-0.42% PPL, 6.3 sd) was produced with it and must
    stay reproducible.
    """
    n_bands = len(widths)
    grid = sorted(curves[0])
    lo, hi = grid[0], grid[-1]

    import bisect
    import math

    def err(b, L):
        """E_b(L) by LOG-LOG interpolation between bracketing measured points.

        Snapping to the nearest measured point instead was a silent
        resolution bug: with a grid of spacing 4 and a regular ladder such as
        14B t32's [96,64,48,32,24,16], every +-1 candidate mapped onto the SAME
        grid point as the parent, tied on error, and the solve reported "no
        improvement" for all 168 rungs. 4B t48's irregular ladder
        [111,85,49,48,33] produced overlapping offsets, a denser effective
        grid, and 114/140 improvements -- so the statistic was measuring
        LADDER REGULARITY, not allocation quality, and was not comparable
        across cells.

        Log-log is the right space: SC error falls as roughly A/L^p, which is a
        straight line in (log L, log E), so interpolation is near-exact rather
        than merely smooth.
        """
        c = curves[b]
        if L in c:
            return c[L]
        i = bisect.bisect_left(grid, L)
        if i == 0:
            return c[grid[0]]
        if i >= len(grid):
            return c[grid[-1]]
        x0, x1 = grid[i - 1], grid[i]
        y0, y1 = c[x0], c[x1]
        if y0 <= 0.0 or y1 <= 0.0:
            t = (L - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
        t = (math.log(L) - math.log(x0)) / (math.log(x1) - math.log(x0))
        return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))

    ladders = [[0] * len(parent) for _ in range(n_bands)]
    report = {}
    for k, L_par in enumerate(parent):
        # the parent itself: residue exactly 0, always admissible
        best = [L_par] * n_bands
        best_cost = sum(err(b, L_par) for b in range(n_bands))
        best_res = 0.0
        for hot in range(n_bands):
            for delta in range(1, max_delta + 1):
                for cand, residue in rung_candidates(
                        L_par, widths, hot, delta, tol, cap=hi):
                    if any(v < lo or v > hi for v in cand):
                        continue
                    cost = sum(err(b, cand[b]) for b in range(n_bands))
                    # strictly better error, or an equal-error candidate that
                    # gives budget back -- never trade cost for a tie
                    if cost < best_cost or (cost == best_cost
                                            and residue < best_res):
                        best, best_cost, best_res = cand, cost, residue
        base = sum(err(b, L_par) for b in range(n_bands))
        for b in range(n_bands):
            ladders[b][k] = best[b]
        realized = sum(widths[b] * best[b] for b in range(n_bands)) / sum(widths)
        report[str(L_par)] = {
            "lengths": best,
            "realized_mac_mean": round(realized, 5),
            "residue": round(realized - L_par, 5),
            "predicted_err_ratio": round(best_cost / base, 6) if base else 1.0,
        }
    return ladders, report


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True, help="deployed bundle dir")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bands", type=int, default=2)
    ap.add_argument("--num-windows", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=512,
                    help="rows sampled per call (mirrors --max_units_per_call)")
    ap.add_argument("--iso-tol", type=float, default=0.25)
    ap.add_argument("--solver", choices=("waterfill","hotband"),
                    default="waterfill",
                    help="waterfill = exact Lagrangian, any band count; "
                         "hotband = legacy 2-band sweep (reproduces the "
                         "published 4B t48 result)")
    ap.add_argument("--hot-frac", type=float, default=0.5,
                    help="fraction of chunks in the HOT (band 0) group; "
                         "small = concentrate outliers = larger band asymmetry")
    ap.add_argument("--max-delta", type=int, default=24,
                    help="largest cycle offset the hot band may take")
    ap.add_argument("--ctx-len", type=int, default=2048)
    args = ap.parse_args()

    parent_dir = Path(args.parent)
    wrapper, table, _ = load_parent(parent_dir)
    r_widths = residual_widths(table)
    layer_buckets = int(table.get("layer_buckets", 1))
    n_bands = args.n_bands

    # Candidate grid: the union of every bucket ladder plus the offsets the
    # solve may reach, so no chosen length is ever extrapolated.
    ladder_vals = set(int(x) for x in table["stoc_len_levels"])
    for op in BANDABLE_OPS:
        for lb in range(layer_buckets):
            ladder_vals.update(bucket_ladder(table, op, lb))
    grid = sorted({max(1, min(v + d, 128))
                   for v in ladder_vals
                   for d in range(-args.max_delta, args.max_delta + 1, 4)})
    # stoc_len must never exceed the halved cap 2**(sc_prec-1) = 128: above it
    # the stream wraps and the result is meaningless, not merely worse.
    print(f"[p3] measurement grid ({len(grid)}): {grid}")

    os.environ.setdefault("SC_OWEN_MODE", "bitrev")
    os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")
    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = str(parent_dir / "wrapper.json")
    hybrid = parent_dir / "hybrid_config.json"
    if hybrid.is_file():
        os.environ["SC_HYBRID_CONFIG_JSON"] = str(hybrid)

    # build_sc_model is the DEPLOYED eval's own loader: it attaches the MP
    # config, the SmoothQuant/AWQ front-end and the hybrid INT schedule exactly
    # as the cell under refinement runs them. Re-implementing any of that here
    # would measure a model the runtime never executes.
    from benchmark.quant.eval_quant import build_sc_model  # noqa: E402
    from benchmark.ppl.calibrate_mp_thresholds import _iter_calib_windows
    from datasets import load_dataset

    print(f"[p3] loading {args.model_path} via build_sc_model (deployed path)")
    model, tok = build_sc_model(args.model_path, "mp",
                                mp_table=str(parent_dir / "wrapper.json"))
    model.eval()
    device = next(model.parameters()).device
    mp_config = getattr(model.config, "sc_mp_config", None)
    if mp_config is None:
        raise SystemExit(
            "[p3] the loaded model carries no sc_mp_config; the parent wrapper "
            "did not attach, so any measurement here would be off-deployment.")
    total_blocks = getattr(model.config, "_sc_total_blocks", None)
    if not total_blocks:
        raise SystemExit("[p3] model.config._sc_total_blocks is unset")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    # (op, l_bucket) -> accumulators
    imp = defaultdict(lambda: None)          # per-chunk importance, summed
    curves = defaultdict(lambda: defaultdict(float))   # (op,lb,band) -> {L: E}
    seen = defaultdict(int)

    def hook(mod, inputs, output):
        op = getattr(mod, "_sc_op_name", None)
        blk = getattr(mod, "_sc_block_idx", None)
        if op not in r_widths or blk is None:
            return
        # MoE: the protected set is keyed by (op, block, EXPERT), and each
        # expert sees a different token subset. Averaging importance across
        # experts would band them all identically off one expert's statistics.
        # Fail loudly rather than emit a plausible-looking wrong allocation.
        if getattr(mod, "_sc_unit_idx", None) is not None:
            raise SystemExit(
                "[p3] this module carries a unit (MoE expert) index; the "
                "Phase-3 allocator is dense-only so far. Extend the band map "
                "key to (op, block, unit) before running a MoE model.")
        x = inputs[0]
        x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        if x_flat.shape[0] > args.max_rows:
            sel = torch.randperm(x_flat.shape[0], device=x_flat.device)[:args.max_rows]
            x_flat = x_flat.index_select(0, sel)
        w = mod.weight.to(torch.float32)
        smooth = getattr(mod, "smooth_scales", None)

        # residual slice, exactly as the runtime forms it
        prot = (mp_config.get_protected_channels(operator=op, block_idx=blk)
                if mp_config is not None else None)
        if prot:
            keep = torch.ones(x_flat.shape[1], dtype=torch.bool, device=x_flat.device)
            keep[torch.tensor(sorted(set(prot)), device=x_flat.device)] = False
            rest = keep.nonzero(as_tuple=True)[0]
            x_res, w_res = x_flat.index_select(1, rest), w.index_select(1, rest)
            s_res = smooth.index_select(0, rest) if smooth is not None else None
        else:
            x_res, w_res, s_res = x_flat, w, smooth
        if x_res.shape[1] != r_widths[op]:
            raise SystemExit(
                f"[p3] {op}:b{blk} residual width {x_res.shape[1]} != "
                f"{r_widths[op]} from the parent table -- the protected set "
                f"the runtime uses is not the one the table records.")

        lb = min(int(blk) * layer_buckets // int(total_blocks), layer_buckets - 1)
        n_chunks = len(residual_chunk_widths(x_res.shape[1], CHUNK_D))

        a = chunk_importance(x_res, w_res, n_chunks)
        key = (op, lb)
        imp[key] = a if imp[key] is None else imp[key] + a
        seen[key] += 1
        model._p3_last[key] = (x_res, w_res, s_res, n_chunks)

    model._p3_last = {}
    handles = [m.register_forward_hook(hook)
               for m in model.modules() if isinstance(m, SCLinear)]

    try:
        with torch.no_grad():
            for i, window in enumerate(_iter_calib_windows(
                    enc, args.ctx_len, args.num_windows)):
                print(f"[p3] window {i + 1}/{args.num_windows}")
                model(window.unsqueeze(0).to(device))
    finally:
        for h in handles:
            h.remove()

    # ---- band assignment + measured curves, one pass per (op, bucket) ------
    print("[p3] assigning bands and measuring per-band error curves")
    chunk_bands: dict[str, list[int]] = {}
    ladders_out: dict[str, list[list[int]]] = {}
    reports: dict[str, dict] = {}

    hot_frac = float(args.hot_frac)
    for (op, lb), a in sorted(imp.items()):
        R = r_widths[op]
        widths_c = residual_chunk_widths(R, CHUNK_D)
        n_chunks = len(widths_c)
        # PER-OPERATOR band count: n_bands is a MAXIMUM. Every band needs >= 2
        # chunks (a narrower band falls off the chunked kernel path), so a
        # 19-chunk projection caps at 9 bands while down_proj's 71 chunks
        # support 35. Clamping per op is what lets down_proj run deep without
        # silently dropping every narrow op out of the K-band path.
        # Cap from the chunks actually AVAILABLE for assignment. The tail chunk
        # is pinned to the last band, so only `full` chunks get distributed;
        # deriving the cap from n_chunks instead gave per_band = 19 // 10 = 1,
        # i.e. nine singleton bands, and every narrow op was then dropped by the
        # >=2-chunks guard. That is how a "per-operator" run still covered
        # down_proj alone.
        _n_full = n_chunks - (1 if widths_c[-1] != CHUNK_D else 0)
        n_bands_op = max(2, min(n_bands, _n_full // 2))
        if _n_full < 4:
            print(f"[p3] SKIP {op}:l{lb}: only {_n_full} full chunks")
            continue
        # Pin the ragged tail chunk to the LAST band for every block so band
        # WIDTHS stay constant per operator (the loader requires it: ladders
        # are per bucket, membership is per block).
        has_tail = widths_c[-1] != CHUNK_D
        full = list(range(n_chunks - 1)) if has_tail else list(range(n_chunks))
        order = sorted(full, key=lambda c: float(a[c]), reverse=True)
        # Band SIZE is the lever on asymmetry. An equal split (hot_frac=0.5)
        # dilutes the outlier chunks with ordinary ones, so both bands end up
        # with similar error and water-filling correctly declines to move
        # anything (14B t32: 6/168 rungs improved, mean ratio 0.9999). A small
        # hot band concentrates the outliers, which is what creates a gap worth
        # redistributing across -- at the cost of the hot band being narrow, so
        # a given cycle moved there costs less budget but buys less coverage.
        bmap = [n_bands_op - 1] * n_chunks
        if n_bands_op == 2:
            n_hot = max(2, min(len(full) - 2,
                               int(round(hot_frac * len(full)))))
            for c in order[:n_hot]:
                bmap[c] = 0
        else:
            # Even split of the importance-ordered chunks. Slicing by
            # b*len//B (rather than a fixed per_band with the remainder dumped
            # on the last band) keeps band WIDTHS balanced: the old form gave
            # down_proj [512 x 15, 1464], a final band 3x wider than the rest
            # holding the 11 least-important chunks, which is a poor partition
            # AND wastes the resolution more bands were meant to buy.
            n_full = len(full)
            for b in range(n_bands_op):
                lo = b * n_full // n_bands_op
                hi = (b + 1) * n_full // n_bands_op
                for c in order[lo:hi]:
                    bmap[c] = b
        counts = [bmap.count(b) for b in range(n_bands_op)]
        if min(counts) < 2:
            print(f"[p3] SKIP {op}:l{lb}: band counts {counts}")
            continue
        for blk in range(int(total_blocks)):
            chunk_bands.setdefault(f"{op}:b{blk}", bmap)

        cached = model._p3_last.get((op, lb))
        if cached is None:
            continue
        x_res, w_res, s_res, _ = cached
        w_band = _band_widths(bmap, R, n_bands_op)
        curves_solve, curves_held = [], []
        for b in range(n_bands_op):
            cols = torch.cat([_chunk_cols(c, R, x_res.device)
                              for c in range(n_chunks) if bmap[c] == b])
            ca, cb = band_error_curve(
                x_res.index_select(1, cols).contiguous(),
                w_res.index_select(1, cols).contiguous(), grid,
                s_res.index_select(0, cols).contiguous()
                if s_res is not None else None)
            # monotonize BEFORE solving: noisy upticks stall the greedy
            # slack-spender and violate water-filling's convexity assumption
            curves_solve.append(monotonize(ca))
            curves_held.append(monotonize(cb))
        parent_ladder = bucket_ladder(table, op, lb)
        if args.solver == "waterfill":
            lad, rep = solve_bucket_waterfill(parent_ladder, w_band,
                                              curves_solve)
        else:
            lad, rep = solve_bucket(parent_ladder, w_band, curves_solve,
                                    args.iso_tol, args.max_delta)
        # Score the SAME choice on rows it was not selected with. A held-out
        # ratio near 1.0 while the solve ratio is well below it means the gain
        # was sampling noise the argmin found, not allocation.
        held = score_on(curves_held, parent_ladder, lad)
        # Band asymmetry AT THE PARENT RUNG, per unit width. This is the single
        # number that predicts whether any redistribution can pay: bands with
        # equal error-density have nothing to trade. Reported so a null result
        # is diagnosable (bad split) rather than merely disappointing.
        asym = {}
        for k, L_par in enumerate(parent_ladder):
            dens = [curves_solve[b].get(L_par, float("nan")) / max(w_band[b], 1)
                    for b in range(n_bands_op)]
            lo_d, hi_d = min(dens), max(dens)
            asym[str(L_par)] = round(hi_d / lo_d, 4) if lo_d > 0 else None
        for L, r in rep.items():
            r["heldout_err_ratio"] = held.get(L, 1.0)
        ladders_out[f"{op}:t0:l{lb}"] = lad
        reports[f"{op}:t0:l{lb}"] = {
            "parent": parent_ladder, "band_widths": w_band,
            "band_share": [round(x / R, 4) for x in w_band],
            "hot_frac": hot_frac,
            "band_err_density_asymmetry": asym,
            "rungs": rep,
        }
        gains = [v["predicted_err_ratio"] for v in rep.values()]
        hgains = [v["heldout_err_ratio"] for v in rep.values()]
        amean = [v for v in asym.values() if v]
        asym_s = f"{sum(amean) / len(amean):.2f}x" if amean else "n/a"
        print(f"[p3] {op}:l{lb} widths={w_band} hot_frac={hot_frac:.3f} "
              f"asym={asym_s} "
              f"solve_err mean={sum(gains) / len(gains):.4f} "
              f"min={min(gains):.4f} "
              f"| HELDOUT mean={sum(hgains) / len(hgains):.4f} "
              f"min={min(hgains):.4f}")

    alloc = {
        "n_bands": n_bands,
        "chunk_d": CHUNK_D,
        "chunk_bands": chunk_bands,
        "ladders": ladders_out,
        "report": reports,
        "note": (f"phase3 k-bands v4 (held-out scored): importance-split chunks, "
                 f"measured per-band "
                 f"error curves, per-rung iso-cost solve (tol={args.iso_tol}, "
                 f"max_delta={args.max_delta}); parent={parent_dir}"),
    }
    Path(args.out).write_text(json.dumps(alloc, indent=1))
    print(f"[p3] wrote {args.out}")
    print(f"[p3] buckets solved: {len(ladders_out)}  ops: "
          f"{sorted({k.split(':')[0] for k in ladders_out})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
