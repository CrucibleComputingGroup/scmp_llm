"""Per-(row, chunk) thresholds calibrated on the DEPLOYMENT population (v2).

Pure allocation: the deployed rule is unchanged (per-(row, 128-chunk) absmax,
min-max normalized over the call, bucketized by per-(op, layer-bucket)
thresholds onto a per-bucket ladder). Only the TABLE changes. Each
(op, layer-bucket) is held to the per-row parent's TRACED cost, attention and
the INT mask are untouched, so the comparison against the parent isolates how
lengths are assigned to groups.

WHAT v1 (mp_per_row_chunk_calib.py, the "v7" tables) GOT WRONG, and the fix:

  1. Normalization population. v1 cut thresholds from x[:256] of one window,
     but the runtime min-max normalizes over the FULL call (2048 rows). The
     per-call max moves, so deployed histograms missed the solved ones by
     -6% to -24% mean L per op (down_proj to -47%). FIX: the metric is always
     computed and normalized on the full call; only the error measurement is
     subsampled (uniform random rows, so the sample is unbiased).
  2. Coverage. v1 saw the first 4 calls per key (the first ~4 blocks of each
     layer bucket) and stopped after one window. FIX: every block, several
     windows; MoE expert calls are subsampled uniformly per block.
  3. INT-masked modules. v1's hook sampled them (the hook fires even when the
     forward routes to INT7). FIX: skip any module whose hybrid backend is not
     "sc".
  4. Arithmetic. v1 measured error curves WITHOUT the module's smooth_scales,
     but the deployed sc_matmul applies them (a/s, b*s). FIX: pass them.
  5. Tail chunk. v1 dropped it; the runtime dispatches it. FIX: include it,
     priced at its true width.
  6. Ladder. v1 reused the parent's per-ROW ladder (e.g. 4B t64
     [48,60,61,62,63,64,105,128]): near-duplicate rungs and a floor that holds
     43-84% of linear MACs at t40-t64, so almost nothing could move. FIX: a
     dense ladder extended below the parent floor, 128 kept as the top rung.
  7. Histogram. v1 imposed the UNCONSTRAINED oracle's histogram and then cut
     thresholds, so the deployable (monotone-in-metric) rule inherited a
     histogram chosen for a rule it cannot express. FIX: solve directly over
     monotone staircases -- an exact Lagrangian DP over metric-sorted bins.
  8. Budget. v1 read per-bucket budgets from its 256-row sample (+9-23% vs the
     parent at t32) or from the trace INCLUDING the protected slice. FIX:
     budgets from the parent trace's DISPATCHED groups only.

PRE-FLIGHT. Before any PPL run the script evaluates, on HELD-OUT windows and
with the exact deploy rule, the realized cost and measured squared error of
(parent per-row, v1 table if given, this table) per bucket and per op. That is
the number to pre-register against; it is not a PPL result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

CHUNK_D = 128
SC_PREC = 8
HALVE = True
DENSE_LADDER = [8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96,
                112, 128]


# --------------------------------------------------------------------------
# Deploy-rule mirrors
# --------------------------------------------------------------------------
def call_metric(x: torch.Tensor) -> torch.Tensor:
    """(N, n_chunks) per-(row, chunk) absmax, min-max normalized over the CALL.

    Must equal model.sc_common.per_row_chunk_rungs' `mn` exactly (same padding,
    same clamp), including the tail chunk.
    """
    N, D = x.shape
    nch = (D + CHUNK_D - 1) // CHUNK_D
    pad = nch * CHUNK_D - D
    xa = x.abs()
    if pad:
        xa = torch.nn.functional.pad(xa, (0, pad))
    m = xa.view(N, nch, CHUNK_D).amax(-1)
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo).clamp_min(1e-8)


def deploy_rungs(mn: torch.Tensor, thresholds) -> torch.Tensor:
    th = torch.as_tensor(list(thresholds), dtype=mn.dtype, device=mn.device)
    return torch.bucketize(mn, th)


def bucket_of(block_idx, total_blocks, n_buckets):
    """Mirror of scmp_kernels.mp.config._bucket_index for the layer axis."""
    from scmp_kernels.mp.config import _bucket_index
    return _bucket_index(int(block_idx), total_blocks, n_buckets)


# --------------------------------------------------------------------------
# Solver: best MONOTONE staircase (the only family the deploy rule expresses)
# --------------------------------------------------------------------------
def staircase_dp(E: np.ndarray, C: np.ndarray, Lv: np.ndarray, lam: float):
    """Exact min over non-decreasing rung sequences of sum_b E[b,r_b] + lam*C[b]*Lv[r_b].

    E: (B, R) weighted error per metric-sorted bin and rung; C: (B,) weighted
    MACs per bin; Lv: (R,) ascending lengths. O(B*R).
    """
    B, R = E.shape
    cost = E + lam * C[:, None] * Lv[None, :]
    f = cost[0].copy()
    arg = np.zeros((B, R), dtype=np.int32)
    idx = np.arange(R)
    for b in range(1, B):
        # prefix-min over r' <= r of f, with its argmin
        best = np.minimum.accumulate(f)
        # argmin of the prefix: position where the running min was last set
        is_new = np.concatenate(([True], f[1:] < best[:-1]))
        pos = np.maximum.accumulate(np.where(is_new, idx, 0))
        arg[b] = pos
        f = cost[b] + best
    r = np.empty(B, dtype=np.int32)
    r[-1] = int(np.argmin(f))
    for b in range(B - 1, 0, -1):
        r[b - 1] = arg[b, r[b]]
    return r


def solve_bucket(metric, err, mac, Lv, target, n_bins=2048):
    """Monotone staircase at MAC-weighted mean length <= target.

    metric (P,), err (P, R) weighted, mac (P,) weighted MACs. Returns
    (thresholds len R-1, rung per bin, achieved mean, bins' metric edges).
    """
    order = np.argsort(metric, kind="stable")
    m_s = metric[order]
    P = m_s.size
    B = int(min(n_bins, P))
    # equal-count bins over the sorted pairs
    edges = np.linspace(0, P, B + 1).astype(np.int64)
    E = np.add.reduceat(err[order], edges[:-1], axis=0)
    C = np.add.reduceat(mac[order], edges[:-1])
    bin_max = m_s[edges[1:] - 1]
    Ctot = float(C.sum())

    def mean_len(r):
        return float((C * Lv[r]).sum() / Ctot)

    lo, hi = 0.0, 1e-12
    r_hi = staircase_dp(E, C, Lv, hi)
    while mean_len(r_hi) > target and hi < 1e30:
        hi *= 10.0
        r_hi = staircase_dp(E, C, Lv, hi)
    best = r_hi
    for _ in range(60):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e3
        r = staircase_dp(E, C, Lv, mid)
        if mean_len(r) <= target:
            best, hi = r, mid
        else:
            lo = mid
        if hi / max(lo, 1e-300) < 1.0 + 1e-6:
            break
    R = Lv.size
    th = []
    for k in range(R - 1):
        low = np.nonzero(best <= k)[0]
        high = np.nonzero(best > k)[0]
        if high.size == 0:
            th.append(1.0)
        elif low.size == 0:
            th.append(0.0)
        else:
            th.append(float(bin_max[low[-1]]))
    for i in range(1, len(th)):
        th[i] = max(th[i], th[i - 1])
    return th, best, mean_len(best), float(E[np.arange(B), best].sum())


def oracle_error(err, mac, Lv, target):
    """Unconstrained per-pair water-fill (needs the FP reference; bound only)."""
    lo, hi = 0.0, 1e-12

    def solve(lam):
        j = np.argmin(err + lam * mac[:, None] * Lv[None, :], axis=1)
        return j, float((mac * Lv[j]).sum() / mac.sum())

    j, m = solve(hi)
    while m > target and hi < 1e30:
        hi *= 10.0
        j, m = solve(hi)
    best = j
    for _ in range(60):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e3
        j, m = solve(mid)
        if m <= target:
            best, hi = j, mid
        else:
            lo = mid
    return float(err[np.arange(err.shape[0]), best].sum())


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def _sc(x, w, L, smooth):
    from scmp_kernels.sc.matmul import sc_matmul
    return sc_matmul(x, w, granularity="per_row", mode="bipolar",
                     sc_prec=SC_PREC, stoc_len=int(L), chunk_d=CHUNK_D,
                     halve_bipolar_stoc_len=HALVE, smooth_scales=smooth)


def error_curves(x, w, smooth, levels):
    """(R_rows, n_chunks, n_levels) squared error of each chunk's partial
    product, with the DEPLOYED smoothing, tail chunk at its true width."""
    N, D = x.shape
    nch = (D + CHUNK_D - 1) // CHUNK_D
    e = torch.zeros(N, nch, len(levels), device=x.device)
    widths = []
    for c in range(nch):
        sl = slice(c * CHUNK_D, min((c + 1) * CHUNK_D, D))
        xc = x[:, sl].contiguous()
        wc = w[:, sl].contiguous()
        sc_c = smooth[sl].contiguous() if smooth is not None else None
        widths.append(xc.shape[1])
        fpc = xc @ wc.t()
        for li, L in enumerate(levels):
            e[:, c, li] = ((_sc(xc, wc, L, sc_c) - fpc) ** 2).sum(dim=1)
    return e, torch.tensor(widths, dtype=torch.float32)


def parent_budgets(trace_path, total_blocks, n_buckets):
    """Per-(op, layer-bucket) MAC-weighted mean L over DISPATCHED groups only.

    The protected slice is a separate trace group with a smaller d_in; keep
    only groups whose d_in is the max for their (op, block, unit).
    """
    tr = json.loads(Path(trace_path).read_text())
    grps = tr.get("groups", tr)
    dmax = defaultdict(int)
    for g in grps:
        k = (g.get("op"), g.get("block"), g.get("unit"))
        dmax[k] = max(dmax[k], int(g.get("d_in") or 0))
    agg = defaultdict(lambda: [0.0, 0.0])
    for g in grps:
        op, blk = g.get("op"), g.get("block")
        if op is None or blk is None or op in ("qk", "av"):
            continue
        if int(g.get("d_in") or 0) != dmax[(op, blk, g.get("unit"))]:
            continue
        b = bucket_of(int(blk), total_blocks, n_buckets)
        a = agg[(op, b)]
        a[0] += float(g["macs"])
        a[1] += float(g["macs"]) * float(g["stoc_len"])
    return {k: v[1] / v[0] for k, v in agg.items() if v[0] > 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True, help="mp_best bundle dir")
    ap.add_argument("--parent-trace", default="",
                    help="parent arm's full-eval trace (per-bucket budgets)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True, help="wrapper json to write")
    ap.add_argument("--ladder", default=",".join(map(str, DENSE_LADDER)),
                    help="halved, ascending; 'parent' = parent ladder + escape")
    ap.add_argument("--compare-table", default="",
                    help="v1 prc TABLE json to score on the same held-out data")
    ap.add_argument("--calib-windows", type=int, default=4)
    ap.add_argument("--holdout-windows", type=int, default=2)
    ap.add_argument("--rows-per-call", type=int, default=256)
    ap.add_argument("--expert-calls-per-block", type=int, default=8,
                    help="MoE: expert calls measured per (op, block, window)")
    ap.add_argument("--bins", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frontend", default="awq", choices=("awq", "smoothquant"))
    ap.add_argument("--diag", default="", help="pre-flight json path")
    ap.add_argument("--budget-source", default="calib", choices=("calib", "trace"),
                    help="calib (default): each bucket's target = the PARENT's own "
                         "MAC-weighted L on the SAME calibration pairs (no test-set "
                         "information). trace: the parent's full-eval trace.")
    ap.add_argument("--windows", default="stratified",
                    choices=("stratified", "prefix"),
                    help="stratified: windows spread over the whole train stream "
                         "(the first windows are one article and over-spend "
                         "down_proj by ~22%% vs the test trace)")
    ap.add_argument("--scales", default="1.0,0.9,0.8,0.7,0.6",
                    help="budget scales for the held-out iso-error sweep")
    ap.add_argument("--emit-scales", default="",
                    help="extra tables to write at these budget scales "
                         "(suffix _s<pct>), e.g. 0.8")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0

    parent = Path(args.parent)
    table = json.loads((parent / "table.json").read_text())
    wrapper = json.loads((parent / "wrapper.json").read_text())
    cap = 2 ** (SC_PREC - 1) if HALVE else 2 ** SC_PREC
    par_levels = sorted(int(v) for v in table.get("stoc_len_levels", []))
    esc = int(wrapper.get("escape_stoc_len", 0) or 0)
    if args.ladder == "parent":
        ladder = sorted(set(par_levels + ([esc] if esc else [])))
    else:
        ladder = sorted(int(v) for v in args.ladder.split(","))
    if max(ladder) > cap or min(ladder) < 1:
        raise SystemExit(f"ladder {ladder} outside [1, {cap}]")
    if cap not in ladder:
        raise SystemExit(f"ladder must keep the top length {cap}: {ladder}")
    cmp_tbl = (json.loads(Path(args.compare_table).read_text())
               if args.compare_table else None)
    cmp_levels = set()
    if cmp_tbl:
        for e in (cmp_tbl.get("per_row_chunk") or {}).get("buckets", {}).values():
            cmp_levels.update(int(v) for v in e["levels"])
    # Every length the PARENT can emit: stoc_len_levels is only the global
    # ladder -- per-bucket ladders (V19-style) carry more (e.g. llama8B t32
    # 17/65/72-75). Scoring the parent at a level we never measured would be
    # a silent KeyError or, worse, a nearest-level substitute.
    tbl_levels = set()

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("levels", "ladder", "ladders", "stoc_len_levels") \
                        and isinstance(v, list):
                    for x in v:
                        if isinstance(x, (int, float)):
                            tbl_levels.add(int(x))
                        elif isinstance(x, list):
                            tbl_levels.update(int(y) for y in x
                                              if isinstance(y, (int, float)))
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
    _walk(table)
    measured = sorted(set(ladder) | set(par_levels) | cmp_levels | tbl_levels
                      | ({esc} if 0 < esc <= cap else set()))
    measured = [L for L in measured if 1 <= L <= cap]
    li = {L: i for i, L in enumerate(measured)}
    print(f"[prc2] ladder {ladder}; measured levels {measured}")

    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = str(parent / "wrapper.json")
    if (parent / "hybrid_config.json").is_file():
        os.environ["SC_HYBRID_CONFIG_JSON"] = str(parent / "hybrid_config.json")
    os.environ.setdefault(
        "ACT_SCALES_DIR",
        "/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales")
    os.environ["FRONTEND"] = args.frontend

    from benchmark.quant.eval_quant import build_sc_model
    from benchmark.ppl.calibrate_mp_thresholds import _iter_calib_windows
    from model.sc_common import (SCLinear, _hybrid_backend, _mp_dispatch_metric,
                                 _channel_index_tensor,
                                 _complement_channel_indices)
    from scmp_kernels.mp import adaptive_classify_rows
    from datasets import load_dataset

    model, tok = build_sc_model(args.model_path, "mp",
                                mp_table=str(parent / "wrapper.json"))
    model.eval()
    dev = next(model.parameters()).device
    config = model.config
    total_blocks = getattr(config, "num_hidden_layers", None)
    mp_cfg = getattr(config, "sc_mp_config", None)
    if mp_cfg is None:
        raise SystemExit("[prc2] model carries no sc_mp_config")
    n_lb = int(getattr(mp_cfg, "layer_buckets", 0) or table.get("layer_buckets") or 4)
    trace_budgets = (parent_budgets(args.parent_trace, total_blocks, n_lb)
                     if args.parent_trace else {})

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    nwin = args.calib_windows + args.holdout_windows
    if args.windows == "stratified":
        from benchmark.ppl.calibrate_mp_thresholds import _select_int_swap_windows
        windows, starts = _select_int_swap_windows(
            enc, 2048, nwin, sampling="stratified", seed=args.seed)
        # hold out every k-th stratum so both sets span the corpus
        k = max(nwin // max(args.holdout_windows, 1), 1)
        hold_idx = set(range(k - 1, nwin, k)[:args.holdout_windows])
        print(f"[prc2] stratified window starts {starts}; held-out idx {sorted(hold_idx)}")
    else:
        windows = list(_iter_calib_windows(enc, 2048, nwin))
        hold_idx = set(range(args.calib_windows, nwin))
    if len(windows) < nwin:
        raise SystemExit("[prc2] not enough train tokens for the windows")

    # key -> phase -> list of dict(mn, err, mac, parL)
    store = {"calib": defaultdict(list), "hold": defaultdict(list)}
    state = {"phase": "calib", "win": 0}
    expert_seen = defaultdict(int)

    def hook(mod, inp, out):
        op = getattr(mod, "_sc_op_name", None)
        blk = getattr(mod, "_sc_block_idx", None)
        unit = getattr(mod, "_sc_unit_idx", None)
        if op is None or blk is None:
            return
        if _hybrid_backend(config, op, blk) != "sc":
            return                                   # INT-masked: not SC
        if unit is not None:
            k = (op, blk, state["win"])
            expert_seen[k] += 1
            # deterministic pseudo-random subset of expert calls per block
            h = zlib.crc32(f"{args.seed}:{op}:{blk}:{unit}:{state['win']}".encode())
            n_units = getattr(config, "num_experts", None) or 128
            if (h % n_units) >= args.expert_calls_per_block:
                return
        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
        if x.shape[0] == 0:
            return
        w = mod.weight.float()
        smooth = getattr(mod, "smooth_scales", None)
        smooth = smooth.float() if smooth is not None else None
        prot = mp_cfg.get_protected_channels(operator=op, block_idx=blk,
                                             unit_idx=unit)
        pidx = _channel_index_tensor(prot, x.shape[1], x.device)
        if pidx.numel() > 0:
            rest = _complement_channel_indices(x.shape[1], pidx, x.device)
            if rest.numel() == 0:
                return
            x = x.index_select(1, rest)
            w = w.index_select(1, rest)
            smooth = smooth.index_select(0, rest) if smooth is not None else None
        N = x.shape[0]
        mn = call_metric(x)                          # FULL call normalization
        # parent's per-row lengths on the SAME call, deployed resolver
        met = _mp_dispatch_metric(x, mp_cfg, op)
        asg = adaptive_classify_rows(met, mp_cfg, operator=op, block_idx=blk,
                                     total_blocks=total_blocks)
        parL = torch.zeros(N, dtype=torch.float32, device=x.device)
        for sl, idx in asg.level_row_indices.items():
            parL[idx] = float(sl)
        g = torch.Generator(device="cpu")
        g.manual_seed(zlib.crc32(f"{args.seed}:{op}:{blk}:{unit}:{state['win']}".encode()))
        R = min(args.rows_per_call, N)
        rows = torch.randperm(N, generator=g)[:R].to(x.device)
        xs = x.index_select(0, rows)
        e, widths = error_curves(xs, w, smooth, measured)
        rep = N / R
        if unit is not None:
            # MEASUREMENT CORRECTION (2026-09-23, before any 30B t32/t40/t64/t96 c17
            # result existed): expert calls are sampled with probability
            # k/n_units, so each sampled pair must stand for n_units/k calls.
            # Without this the held-out roll-up and the cost gate weight MoE experts
            # at ~11% of linear MACs instead of ~67% (30B t32 gate read 1.028; at
            # true MAC weights it is 1.007). Per-(op,bucket) solves are unchanged:
            # every pair in a key is scaled alike.
            rep *= (getattr(config, "num_experts", None) or 128) / max(
                args.expert_calls_per_block, 1)
        mac = (widths * w.shape[0]).to(x.device)[None, :].expand(R, -1) * rep
        b = bucket_of(blk, total_blocks, n_lb)
        store[state["phase"]][(op, b)].append(dict(
            mn=mn.index_select(0, rows).reshape(-1).cpu().numpy(),
            err=(e * rep).reshape(-1, len(measured)).cpu().numpy(),
            mac=mac.reshape(-1).cpu().numpy(),
            parL=parL.index_select(0, rows)[:, None].expand(-1, mn.shape[1])
                 .reshape(-1).cpu().numpy()))

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, SCLinear)]
    try:
        with torch.no_grad():
            for i, wnd in enumerate(windows):
                state["phase"] = "hold" if i in hold_idx else "calib"
                state["win"] = i
                model(wnd.unsqueeze(0).to(dev))
                print(f"[prc2] window {i + 1}/{len(windows)} ({state['phase']}): "
                      f"{len(store[state['phase']])} keys")
    finally:
        for h in hs:
            h.remove()

    def cat(parts, f):
        return np.concatenate([p[f] for p in parts])

    calib_budgets = {}
    for key, parts in store["calib"].items():
        cm, cp = cat(parts, "mac"), cat(parts, "parL")
        calib_budgets[key] = float((cm * cp).sum() / cm.sum())
    budgets = calib_budgets if args.budget_source == "calib" else trace_budgets
    for key in sorted(calib_budgets):
        tb = trace_budgets.get(key)
        print(f"[prc2] budget {key[0]:10} l{key[1]}: parent on calib "
              f"{calib_budgets[key]:6.2f}" + (f"  trace {tb:6.2f}" if tb else ""))
    scales = [float(v) for v in args.scales.split(",") if v]
    emit_scales = [float(v) for v in args.emit_scales.split(",") if v]
    extra_buckets = {sc: {} for sc in emit_scales}
    Lv = np.asarray(ladder, dtype=np.float64)
    lad_cols = [li[L] for L in ladder]
    buckets, diag = {}, {"ladder": ladder, "measured": measured, "keys": {}}
    for key in sorted(store["calib"]):
        op, b = key
        if key not in budgets:
            print(f"[prc2] WARNING no parent budget for {key}; skipped")
            continue
        parts = store["calib"][key]
        mn, err, mac = cat(parts, "mn"), cat(parts, "err"), cat(parts, "mac")
        tgt = budgets[key]
        tgt = min(max(tgt, float(Lv[0])), float(Lv[-1]))
        th, rb, ach, e_st = solve_bucket(mn, err[:, lad_cols], mac, Lv, tgt,
                                         n_bins=args.bins)
        e_or = oracle_error(err[:, lad_cols], mac, Lv, tgt)
        buckets[f"{op}:t0:l{b}"] = {"levels": [int(v) for v in ladder],
                                    "thresholds": th}
        d = {"target": budgets[key], "calib_mean_L": ach,
             "calib_err_staircase": e_st, "calib_err_oracle": e_or}
        hp = store["hold"].get(key)
        if hp:
            hmn, herr, hmac = cat(hp, "mn"), cat(hp, "err"), cat(hp, "mac")
            hpar = cat(hp, "parL")
            r = deploy_rungs(torch.from_numpy(hmn), th).numpy()
            Lnew = Lv[r]
            cols_new = np.asarray(lad_cols)[r]
            miss = sorted({int(v) for v in np.unique(hpar)} - set(li))
            if miss:
                raise SystemExit(f"[prc2] parent emitted unmeasured lengths "
                                 f"{miss} for {key}; add them to --ladder")
            cols_par = np.asarray([li[int(v)] for v in hpar])
            M = hmac.sum()
            d["hold_new_L"] = float((hmac * Lnew).sum() / M)
            d["hold_new_err"] = float(herr[np.arange(herr.shape[0]), cols_new].sum())
            d["hold_par_L"] = float((hmac * hpar).sum() / M)
            d["hold_par_err"] = float(herr[np.arange(herr.shape[0]), cols_par].sum())
            if cmp_tbl:
                ce = (cmp_tbl.get("per_row_chunk") or {}).get("buckets", {}) \
                    .get(f"{op}:t0:l{b}")
                if ce:
                    cr = deploy_rungs(torch.from_numpy(hmn), ce["thresholds"]).numpy()
                    cl = np.asarray(ce["levels"], dtype=np.float64)[cr]
                    cc = np.asarray([li[int(v)] for v in cl])
                    d["hold_v1_L"] = float((hmac * cl).sum() / M)
                    d["hold_v1_err"] = float(herr[np.arange(herr.shape[0]), cc].sum())
            d["hold_macs"] = float(M)
        sweep = {}
        for sc in sorted(set(scales) | set(emit_scales)):
            t_s = min(max(budgets[key] * sc, float(Lv[0])), float(Lv[-1]))
            th_s = solve_bucket(mn, err[:, lad_cols], mac, Lv, t_s,
                                n_bins=args.bins)[0]
            if sc in extra_buckets:
                extra_buckets[sc][f"{op}:t0:l{b}"] = {
                    "levels": [int(v) for v in ladder], "thresholds": th_s}
            if hp:
                r_s = deploy_rungs(torch.from_numpy(hmn), th_s).numpy()
                c_s = np.asarray(lad_cols)[r_s]
                sweep[f"{sc:.2f}"] = {
                    "L": float((hmac * Lv[r_s]).sum() / M),
                    "err": float(herr[np.arange(herr.shape[0]), c_s].sum())}
        if hp:
            d["sweep"] = sweep
        diag["keys"][f"{op}:l{b}"] = d
        print(f"  {op:10} l{b} tgt {budgets[key]:6.2f} calib {ach:6.2f} "
              f"hold new {d.get('hold_new_L', float('nan')):6.2f} "
              f"par {d.get('hold_par_L', float('nan')):6.2f} "
              f"err new/par {d.get('hold_new_err', 0) / max(d.get('hold_par_err', 1e-30), 1e-30):.3f} "
              f"v1/par {d.get('hold_v1_err', float('nan')) / max(d.get('hold_par_err', 1e-30), 1e-30):.3f}")

    # op-level + total roll-up of the held-out pre-flight
    roll = defaultdict(lambda: defaultdict(float))
    for kk, d in diag["keys"].items():
        op = kk.split(":")[0]
        if "hold_macs" not in d:
            continue
        for tag in ("new", "par", "v1"):
            if f"hold_{tag}_L" in d:
                roll[op][f"{tag}_LM"] += d[f"hold_{tag}_L"] * d["hold_macs"]
                roll[op][f"{tag}_err"] += d[f"hold_{tag}_err"]
        roll[op]["M"] += d["hold_macs"]
    summ = {}
    tot = defaultdict(float)
    for op, r in roll.items():
        s = {"macs": r["M"]}
        for tag in ("new", "par", "v1"):
            if f"{tag}_LM" in r:
                s[f"{tag}_L"] = r[f"{tag}_LM"] / r["M"]
                s[f"{tag}_err"] = r[f"{tag}_err"]
                tot[f"{tag}_LM"] += r[f"{tag}_LM"]
        tot["M"] += r["M"]
        s["err_new_over_par"] = s["new_err"] / max(s["par_err"], 1e-30)
        if "v1_err" in s:
            s["err_v1_over_par"] = s["v1_err"] / max(s["par_err"], 1e-30)
        summ[op] = s
        print(f"[prc2] {op:10} L new {s['new_L']:6.2f} par {s['par_L']:6.2f}"
              f"{'  v1 %6.2f' % s['v1_L'] if 'v1_L' in s else ''}  "
              f"err new/par {s['err_new_over_par']:.3f}"
              f"{'  v1/par %.3f' % s['err_v1_over_par'] if 'err_v1_over_par' in s else ''}")
    for tag in ("new", "par", "v1"):
        if f"{tag}_LM" in tot:
            summ[f"total_{tag}_L"] = tot[f"{tag}_LM"] / tot["M"]
    print(f"[prc2] linear-dispatched MAC-weighted L (held-out): "
          + "  ".join(f"{t} {summ[f'total_{t}_L']:.2f}"
                      for t in ("new", "par", "v1") if f"total_{t}_L" in summ))
    diag["ops"] = summ
    # iso-error sweep roll-up over all keys: total held-out err and L per scale
    sw_tot = defaultdict(lambda: [0.0, 0.0])
    par_err_tot = sum(d.get("hold_par_err", 0.0) for d in diag["keys"].values())
    for d in diag["keys"].values():
        for sc, v in (d.get("sweep") or {}).items():
            sw_tot[sc][0] += v["err"]
            sw_tot[sc][1] += v["L"] * d["hold_macs"]
    Mtot = sum(d.get("hold_macs", 0.0) for d in diag["keys"].values())
    diag["sweep_total"] = {sc: {"err_over_par": v[0] / max(par_err_tot, 1e-30),
                                "L_over_par": (v[1] / Mtot) / summ["total_par_L"]}
                           for sc, v in sorted(sw_tot.items())}
    for sc, v in diag["sweep_total"].items():
        print(f"[prc2] sweep scale {sc}: held-out L/par {v['L_over_par']:.3f}  "
              f"err/par {v['err_over_par']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for sc, bk in extra_buckets.items():
        o2 = out.with_name(f"{out.stem}_s{int(round(sc * 100))}.json")
        t2 = o2.with_name(o2.stem + "_table.json")
        pt = dict(table)
        pt["per_row_chunk"] = {"buckets": bk}
        pt.setdefault("sc_prec", SC_PREC)
        pt.setdefault("halve_bipolar_stoc_len", HALVE)
        pt["prc2_calib"] = dict(vars(args), budget_scale=sc)
        t2.write_text(json.dumps(pt, indent=1))
        pw = dict(wrapper)
        pw["threshold_table_path"] = str(t2.resolve())
        pw.setdefault("sc_prec", SC_PREC)
        pw.setdefault("halve_bipolar_stoc_len", HALVE)
        o2.write_text(json.dumps(pw, indent=1))
        print(f"[prc2] wrote scale {sc} table {t2}")
    tbl_out = out.with_name(out.stem + "_table.json")
    payload_t = dict(table)
    payload_t["per_row_chunk"] = {"buckets": buckets}
    payload_t.setdefault("sc_prec", SC_PREC)
    payload_t.setdefault("halve_bipolar_stoc_len", HALVE)
    payload_t["prc2_calib"] = {k: v for k, v in vars(args).items()}
    tbl_out.write_text(json.dumps(payload_t, indent=1))
    payload_w = dict(wrapper)
    payload_w["threshold_table_path"] = str(tbl_out.resolve())
    payload_w.setdefault("sc_prec", SC_PREC)
    payload_w.setdefault("halve_bipolar_stoc_len", HALVE)
    out.write_text(json.dumps(payload_w, indent=1))
    from scmp_kernels.mp.config import AdaptiveMPConfig as _AMP
    lv = [int(v) for v in (payload_t.get("stoc_len_levels")
                           or sorted(ladder, reverse=True))]
    chk = _AMP(sorted(lv, reverse=True))
    chk.load_threshold_table(str(tbl_out))
    n = len(getattr(chk, "per_row_chunk", {}) or {})
    if n != len(buckets):
        raise SystemExit(f"[prc2] round-trip lost buckets: {len(buckets)} -> {n}")
    print(f"[prc2] wrote {tbl_out} ({n} buckets, round-trip verified)")
    if args.diag:
        Path(args.diag).write_text(json.dumps(diag, indent=1))
        print(f"[prc2] pre-flight -> {args.diag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
