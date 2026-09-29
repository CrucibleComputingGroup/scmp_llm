"""Calibrate per-(row, chunk) stream-length thresholds.

WHAT THIS ALLOCATES. Quantization is already per-(row, 128-chunk) -- each chunk
carries its own scale -- but dispatch has only ever been per ROW, so every chunk
in a row shares one length. Measured on real activations at equal mean cost that
mismatch is most of the available allocation gain: per-row buys -0.3% to -7.2%
squared error while per-(row, chunk) buys -15% to -75%, on every linear operator
of every model.

WHY A THRESHOLD IS ENOUGH. The oracle water-fills on MEASURED error, which needs
the FP16 reference at runtime. The deployed rule sees only the per-(row, chunk)
absmax -- which is free, being the group's own quantization scale. Rank-matched
to the oracle's length histogram that proxy captures 94-99% of the oracle's gain
(4B q_proj 99%, 14B down_proj 97%, 4B down_proj 96%, 4B up_proj 94%). So the
calibrator's job is only to (a) find each group's length histogram under one
global MAC budget and (b) turn it into thresholds on the normalized metric.

THE BUDGET IS MAC-WEIGHTED, not pair-weighted. Pricing every (row, chunk) pair
equally would repeat the row-weighting bug that made "iso-budget" not
iso-compute: a pair's cost is chunk_width x d_out MACs, and chunk widths differ
at the tail.

NORMALIZATION IS PER CALL and must match `model.sc_common.per_row_chunk_rungs`
exactly -- min-max over all (row, chunk) pairs of the call -- or the thresholds
do not transfer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

CHUNK_D = 128
SC_PREC = 8
HALVE = True


def _sc(x, w, L):
    from scmp_kernels.sc.matmul import sc_matmul
    return sc_matmul(x, w, granularity="per_row", mode="bipolar",
                     sc_prec=SC_PREC, stoc_len=int(L), chunk_d=CHUNK_D,
                     halve_bipolar_stoc_len=HALVE)


def chunk_metric(x: torch.Tensor) -> torch.Tensor:
    """Per-(row, chunk) absmax, min-max normalized over the whole call.

    Mirrors model.sc_common.per_row_chunk_rungs. Any divergence here silently
    shifts every threshold.
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


def error_curves(x: torch.Tensor, w: torch.Tensor, levels):
    """(n_pairs, n_levels) measured squared error, plus each pair's MAC count.

    One SC call per (chunk, level) yields the error for EVERY row at once, so
    the cost is n_chunks x n_levels narrow matmuls, not per-pair work.
    """
    N, D = x.shape
    nch = D // CHUNK_D
    e = torch.zeros(N, nch, len(levels), device=x.device)
    for c in range(nch):
        sl = slice(c * CHUNK_D, (c + 1) * CHUNK_D)
        xc = x[:, sl].contiguous()
        wc = w[:, sl].contiguous()
        fpc = xc.float() @ wc.float().t()
        for li, L in enumerate(levels):
            e[:, c, li] = ((_sc(xc, wc, L) - fpc) ** 2).sum(dim=1)
    macs = float(CHUNK_D * w.shape[0])
    return e.reshape(N * nch, len(levels)), macs


def per_group_waterfill(groups, levels, targets):
    """Water-fill each group under ITS OWN budget -- the parent's.

    WHY THIS IS THE DEFAULT. A single global multiplier over raw squared error
    silently re-decides the CROSS-LAYER split at the same time as the
    granularity, and raw squared error is not comparable across operators (a
    layer whose output has larger magnitude has larger error regardless of how
    sensitive it is). On 4B t32 the global solve pinned all of o_proj bucket 0
    at the floor while giving q_proj bucket 3 a mean of 58 -- a cross-layer
    reallocation riding along with the change under test, and the same failure
    mode this project already hit with sigma-weighted cross-layer solves.

    Holding each group at the parent's realized mean length isolates the ONE
    variable that is actually being tested, and makes the parent a member of
    the search space (per-row is per-(row, chunk) with every chunk of a row
    sharing its rung), so the refinement cannot lose by construction.
    """
    cost = torch.tensor([float(v) for v in levels], dtype=torch.float32)
    out = {}
    for k, (err, mac, _rep) in groups.items():
        c = cost.to(err.device)
        tgt = float(targets[k])
        lo, hi = 0.0, 1e-6

        def alloc(lam):
            j = (err + lam * c.unsqueeze(0)).argmin(dim=1)
            return j, float(c[j].mean())

        while alloc(hi)[1] > tgt and hi < 1e12:
            hi *= 10.0
        best = alloc(hi)[0]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            j, m = alloc(mid)
            if m <= tgt:
                best, hi = j, mid
            else:
                lo = mid
        out[k] = best
    return out


def global_waterfill(groups, levels, target, tol=1e-4):
    """One shared multiplier across ALL groups -- the cross-layer solve.

    `groups` maps key -> (err (n,L), mac_per_pair, rep). `rep` rescales a
    sampled group to the pairs it stands for; without it a group with many
    sampled pairs dominates the budget and the shared lambda craters the rest
    (the rep_g bug that gated cross-layer allocation the first time).

    Returns key -> chosen level index per pair, and the achieved MAC-weighted
    mean length.
    """
    cost = torch.tensor([float(v) for v in levels], dtype=torch.float32)

    def solve(lam):
        out, num, den = {}, 0.0, 0.0
        for k, (err, mac, rep) in groups.items():
            c = cost.to(err.device)
            j = (err + lam * c.unsqueeze(0) * mac).argmin(dim=1)
            out[k] = j
            wgt = mac * rep
            num += float(c.to(err.device)[j].sum()) * wgt
            den += float(j.numel()) * wgt
        return out, (num / den if den else 0.0)

    lo, hi = 0.0, 1e-6
    while solve(hi)[1] > target and hi < 1e12:
        hi *= 10.0
    best = None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        a, m = solve(mid)
        if m <= target:
            best, hi = (a, m), mid
        else:
            lo = mid
        if abs(m - target) < tol:
            break
    if best is None:
        best = solve(hi)
    return best


def thresholds_from_alloc(metric, alloc, n_levels):
    """Thresholds on the normalized metric that reproduce `alloc`'s histogram.

    The runtime compares the metric against ascending thresholds, so the rule
    can only express a MONOTONE map (bigger metric -> longer stream). Sort the
    pairs by metric and cut at the counts the water-fill chose; the cut points
    are the thresholds. Monotonicity is why this loses only 1-6% versus the
    oracle rather than reproducing it exactly.
    """
    order = torch.argsort(metric)
    counts = torch.bincount(alloc, minlength=n_levels).tolist()
    sorted_m = metric[order]
    th, cum = [], 0
    for r in range(n_levels - 1):
        cum += counts[r]
        if cum <= 0:
            th.append(0.0)
        elif cum >= sorted_m.numel():
            th.append(1.0)
        else:
            th.append(float(sorted_m[cum]))
    # Enforce ascending; equal cuts mean an empty rung, which is legal.
    for i in range(1, len(th)):
        th[i] = max(th[i], th[i - 1])
    return th


def bucket_of(block_idx, total_blocks, n_buckets):
    if total_blocks is None or n_buckets <= 1:
        return 0
    return min(n_buckets - 1, int(block_idx / max(total_blocks - 1, 1) * n_buckets))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True, help="mp_best bundle dir")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, required=True,
                    help="MAC-weighted mean stream length, HALVED space")
    ap.add_argument("--levels", default="", help="comma list, halved; default "
                                                 "= the parent ladder")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=256)
    ap.add_argument("--layer-buckets", type=int, default=4)
    ap.add_argument("--parent-trace", default=None,
                    help="Parent arm's TRACE (full eval). Per-group budgets are "
                         "read from it instead of from the calibration sample. "
                         "The sample overestimates the parent by 9-23% (it sees "
                         "256 rows and ~4 blocks per bucket), so children "
                         "calibrated against it OVERSPEND at tight budgets: 14B "
                         "t32 ran +7.8% compute and its -1.63%% PPL win was "
                         "partly purchased. Same principle as reading cost from "
                         "the trace rather than from a promise -- applied to the "
                         "budget itself.")
    ap.add_argument("--budget-mode", default="parent",
                    choices=("parent", "global"),
                    help="parent (DEFAULT): hold each (op, bucket) at the "
                         "PARENT's realized mean length and redistribute only "
                         "WITHIN it, so granularity is the only variable and "
                         "the parent stays inside the search space. global: one "
                         "shared multiplier, which also re-decides the "
                         "cross-layer split using raw squared error that is not "
                         "comparable across operators.")
    ap.add_argument("--frontend", default="awq", choices=("awq", "smoothquant"),
                    help="MUST match the deployed baseline (AWQ + INT7 20%%). "
                         "Calibrating on a different front-end shifts every "
                         "activation the thresholds are cut from.")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0

    parent = Path(args.parent)
    table = json.loads((parent / "table.json").read_text())
    # The escape-gate knobs live in wrapper.json, NOT table.json. Reading them
    # off `table` returns None silently and the parity rung never appears.
    wrapper = json.loads((parent / "wrapper.json").read_text())
    levels = ([int(v) for v in args.levels.split(",")] if args.levels
              else sorted(int(v) for v in table.get("stoc_len_levels", [])))
    if len(levels) < 2:
        raise SystemExit(f"need >= 2 levels, got {levels}")
    cap = 2 ** (SC_PREC - 1) if HALVE else 2 ** SC_PREC
    if max(levels) > cap:
        raise SystemExit(f"level {max(levels)} exceeds the stream cap {cap}; "
                         "results would wrap")
    if not (min(levels) <= args.target <= max(levels)):
        raise SystemExit(
            f"target {args.target} lies outside the ladder [{min(levels)}, "
            f"{max(levels)}] -- the budget is unreachable, not merely tight")
    # ESCAPE-GATE PARITY. The parent promotes outlier rows OFF the ladder to
    # `escape_stoc_len` (deployed: k=2.0, len=128), and `adaptive_classify_rows`
    # applies that gate INTERNALLY -- so the per-group budgets read from it
    # already include gate spending. A child without an equivalent top rung
    # would quietly drop a deployed feature AND undershoot the budget it was
    # matched against, which reads as "cheaper and worse" rather than as a bug.
    #
    # No new runtime machinery is needed: the gate exists to reach a length
    # ABOVE the ladder, so append that length as a top rung and let the
    # calibrated threshold decide how many pairs earn it. That keeps every
    # decision inside the one dispatch mechanism.
    esc = int(wrapper.get("escape_stoc_len", 0) or 0)
    gate_k = wrapper.get("escape_gate_k")
    if gate_k and esc > 0 and esc <= cap and esc not in levels:
        levels = sorted(levels + [esc])
        print(f"[prc] parent runs an escape gate (k={gate_k}, len={esc}); "
              f"appended {esc} as a top rung for parity")
    elif gate_k and esc > cap:
        raise SystemExit(
            f"[prc] parent escape_stoc_len {esc} exceeds the stream cap {cap}")
    print(f"[prc] ladder {levels} (halved), target {args.target}, "
          f"frontend {args.frontend}")

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
    from model.sc_common import SCLinear
    from datasets import load_dataset

    model, tok = build_sc_model(args.model_path, "mp",
                                mp_table=str(parent / "wrapper.json"))
    model.eval()
    dev = next(model.parameters()).device
    total_blocks = getattr(model.config, "num_hidden_layers", None)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    # (op, l_bucket) -> list of (metric_flat, err, mac, seen_rows, total_rows,
    #                             parent_mean_len)
    acc = defaultdict(list)
    mp_cfg = getattr(model.config, "sc_mp_config", None)
    if mp_cfg is None:
        raise SystemExit(
            "[prc] the loaded model carries no sc_mp_config; the parent budget "
            "cannot be read and --budget-mode parent would silently fall back "
            "to the global solve, which also re-decides the cross-layer split.")

    def hook(mod, inp, out):
        op = getattr(mod, "_sc_op_name", None)
        blk = getattr(mod, "_sc_block_idx", None)
        if op is None or blk is None:
            return
        b = bucket_of(blk, total_blocks, args.layer_buckets)
        key = (op, b)
        if len(acc[key]) >= args.windows:
            return
        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
        total = x.shape[0]
        if total > args.max_rows:
            x = x[:args.max_rows]
        w_full = mod.weight.float()
        # MIRROR THE RUNTIME: SCLinear dispatches on the RESIDUAL, with
        # protected channels split off and run at protected_stoc_len OUTSIDE
        # the MP ladder. Calibrating on the full input computes a different
        # metric than the deployed path sees AND prices channels that are not
        # the allocator's to spend, which is how three cells came out ABOVE
        # their parent's realized cost while the per-group cap said otherwise.
        try:
            prot = mp_cfg.get_protected_channels(
                operator=op, block_idx=blk,
                unit_idx=getattr(mod, "_sc_unit_idx", None))
        except Exception:                                  # noqa: BLE001
            prot = None
        if prot is not None and len(prot) > 0:
            keepmask = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
            pi = torch.as_tensor(list(prot), dtype=torch.long, device=x.device)
            pi = pi[(pi >= 0) & (pi < x.shape[1])]
            keepmask[pi] = False
            rest = keepmask.nonzero(as_tuple=True)[0]
            if rest.numel() == 0:
                return
            x = x.index_select(1, rest)
            w_full = w_full.index_select(1, rest)
        keep = (x.shape[1] // CHUNK_D) * CHUNK_D
        if keep < CHUNK_D or x.shape[0] == 0:
            return
        x = x[:, :keep].contiguous()
        w = w_full[:, :keep].contiguous()
        m = chunk_metric(x).reshape(-1)
        e, mac = error_curves(x, w, levels)
        # The parent's realized mean length on THESE rows, from the deployed
        # resolver -- never a reimplementation. Two resolvers disagreeing about
        # a bucket's ladder is a four-instance bug family in this repo.
        par_mean = None
        try:
            from scmp_kernels.mp import adaptive_classify_rows
            from model.sc_common import _mp_dispatch_metric
            # x is already the residual here, matching `metric_source` in
            # SCLinear -- the deployed path computes the metric AFTER the
            # protected split, so calibrating on the full input shifts every
            # threshold and the parent budget read from it.
            met = _mp_dispatch_metric(x, mp_cfg, op)
            asg = adaptive_classify_rows(
                met, mp_cfg, operator=op, block_idx=blk,
                total_blocks=total_blocks)
            tot = sum(int(i.numel()) for i in asg.level_row_indices.values())
            if tot:
                par_mean = sum(int(sl) * int(i.numel())
                               for sl, i in asg.level_row_indices.items()) / tot
        except Exception as exc:                      # noqa: BLE001
            par_mean = None
            if not hasattr(hook, "_warned"):
                print(f"[prc] WARNING: parent budget unavailable ({exc}); "
                      f"falling back to the global solve")
                hook._warned = True
        acc[key].append((m.cpu(), e.cpu(), mac, x.shape[0], total, par_mean))
        # rep prices the rows this sample stands for. Without it a densely
        # sampled operator dominates the shared multiplier.

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, SCLinear)]
    try:
        with torch.no_grad():
            for i, wnd in enumerate(_iter_calib_windows(enc, 2048, args.windows)):
                model(wnd.unsqueeze(0).to(dev))
                filled = sum(1 for v in acc.values() if len(v) >= args.windows)
                print(f"[prc] window {i + 1}/{args.windows}: "
                      f"{len(acc)} (op, bucket) groups, {filled} full")
                # A single forward already supplies ~n_blocks/n_buckets modules
                # per key, so every key usually fills on window 1 and further
                # forwards do measurable work for nothing.
                if acc and filled == len(acc):
                    print("[prc] all keys full — stopping early")
                    break
    finally:
        for h in hs:
            h.remove()
    if not acc:
        raise SystemExit("[prc] captured no SCLinear inputs")

    groups, metrics = {}, {}
    for key, parts in acc.items():
        m = torch.cat([p[0] for p in parts]).to(dev)
        e = torch.cat([p[1] for p in parts]).to(dev)
        mac = parts[0][2]
        rep = sum(p[4] for p in parts) / max(sum(p[3] for p in parts), 1)
        groups[key] = (e, mac, rep)
        metrics[key] = m

    par_targets = {}
    if args.parent_trace:
        tr = json.loads(Path(args.parent_trace).read_text())
        grps = tr.get("groups", tr)
        agg = defaultdict(lambda: [0.0, 0.0])       # key -> [macs, macs*L]
        for g in grps:
            op = g.get("op")
            blk = g.get("block")
            if op is None or blk is None:
                continue
            b = bucket_of(int(blk), total_blocks, args.layer_buckets)
            a = agg[(op, b)]
            a[0] += float(g["macs"])
            a[1] += float(g["macs"]) * float(g["stoc_len"])
        for k, (mm, ml) in agg.items():
            if mm > 0:
                par_targets[k] = ml / mm
        print(f"[prc] parent budgets from TRACE {args.parent_trace}: "
              f"{len(par_targets)} groups")
        for key in sorted(set(acc) - set(par_targets)):
            print(f"[prc]   WARNING: no trace budget for {key}")
    else:
        for key, parts in acc.items():
            vals = [p[5] for p in parts if p[5] is not None]
            if vals:
                par_targets[key] = sum(vals) / len(vals)
    # SUBSET, not equality. A parent TRACE legitimately covers MORE groups than
    # the allocator has (it includes qk/av, which per-(row,chunk) never touches),
    # so an equality check fails with a NEGATIVE "missing" count and silently
    # falls back to the GLOBAL solve -- the confounded cross-layer mode that was
    # removed as the default. Require only that every group being allocated has
    # a budget.
    have_all = bool(groups) and all(k in par_targets for k in groups)
    if args.budget_mode == "parent" and have_all:
        alloc = per_group_waterfill(groups, levels, par_targets)
        cst = torch.tensor([float(v) for v in levels], dtype=torch.float32)
        num = den = 0.0
        for k, (e, mac, rep) in groups.items():
            w_ = mac * rep
            num += float(cst.to(e.device)[alloc[k]].sum()) * w_
            den += float(alloc[k].numel()) * w_
        achieved = num / den if den else 0.0
        # APPLES-TO-APPLES: aggregate the PARENT's own per-group means with the
        # SAME mac x rep weights. Comparing the child's linear-only mean against
        # mp_best's realized_flop_avg_sl (which also covers qk/av, and the
        # allocator runs attention short) makes a compliant child look like a
        # 16% overspend. This is the comparison that actually means something.
        pnum = pden = 0.0
        for k, (e, mac, rep) in groups.items():
            w_ = mac * rep
            pnum += par_targets[k] * float(alloc[k].numel()) * w_
            pden += float(alloc[k].numel()) * w_
        parent_lin = pnum / pden if pden else 0.0
        print(f"[prc] PER-GROUP water-fill at the parent's own budgets:")
        print(f"[prc]   child  linear-only MAC-weighted mean length "
              f"{achieved:.2f}")
        print(f"[prc]   parent linear-only MAC-weighted mean length "
              f"{parent_lin:.2f}  -> child/parent = "
              f"{achieved / parent_lin if parent_lin else float('nan'):.4f}")
        print(f"[prc]   (NOT comparable to mp_best realized_flop_avg_sl, which "
              f"also includes qk/av)")
        if achieved > parent_lin + 1e-6:
            raise SystemExit(
                f"[prc] child {achieved:.2f} exceeds parent {parent_lin:.2f} on "
                f"the SAME basis — refusing to emit a non-iso-cost table")
        over = [k for k, a in alloc.items()
                if float(torch.tensor([float(v) for v in levels])
                         .to(a.device)[a].mean()) > par_targets[k] + 1e-6]
        if over:
            raise SystemExit(
                f"[prc] {len(over)} group(s) exceed their parent budget "
                f"({over[:4]}...); the per-group cap is not binding, refusing "
                f"to emit a table that is not iso-cost")
    else:
        if args.budget_mode == "parent":
            miss = [k for k in groups if k not in par_targets]
            raise SystemExit(
                f"[prc] --budget-mode parent but {len(miss)} allocated groups "
                f"have no parent budget ({miss[:4]}). Refusing to fall back to "
                f"the global solve, which would re-decide the cross-layer split "
                f"and confound the experiment.")
        alloc, achieved = global_waterfill(groups, levels, args.target)
        print(f"[prc] GLOBAL water-fill: MAC-weighted mean length "
              f"{achieved:.2f} (target {args.target})")
    if args.budget_mode == "global" and achieved > args.target + 0.5:
        raise SystemExit(
            f"[prc] solver overspent ({achieved:.2f} > {args.target}); refusing "
            "to emit a table that would not be iso-compute")

    buckets = {}
    for key, a in alloc.items():
        op, b = key
        th = thresholds_from_alloc(metrics[key], a, len(levels))
        buckets[f"{op}:t0:l{b}"] = {"levels": levels, "thresholds": th}
        hist = torch.bincount(a, minlength=len(levels)).tolist()
        mean_l = sum(h * L for h, L in zip(hist, levels)) / max(sum(hist), 1)
        print(f"  {op:11} l{b}  hist={hist}  mean_L={mean_l:6.2f}  "
              f"th={[round(v, 4) for v in th]}")

    # WHERE THE SECTION MUST LIVE. `_load_per_row_chunk` is called from
    # `AdaptiveMPConfig.load_threshold_table(path)`, so its payload is
    # table.json -- NOT wrapper.json. Writing the section into the wrapper
    # parses to ZERO entries and every SCLinear silently falls back to per-ROW
    # dispatch: the eval completes, reports a plausible PPL at the parent's
    # cost, and shows no change. That reads as "per-(row,chunk) does not help",
    # the most expensive possible wrong conclusion. k_bands lives in the table
    # for the same reason.
    #
    # So emit TWO files: a table (parent table + per_row_chunk) and a wrapper
    # pointing at it by ABSOLUTE path (the stored path is relative, and the
    # emitted files do not sit in the parent bundle).
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl_out = out.with_name(out.stem + "_table.json")

    table_payload = dict(table)
    table_payload["per_row_chunk"] = {"buckets": buckets}
    table_payload.setdefault("sc_prec", SC_PREC)
    table_payload.setdefault("halve_bipolar_stoc_len", HALVE)
    tbl_out.write_text(json.dumps(table_payload, indent=1))

    payload = dict(wrapper)
    payload["threshold_table_path"] = str(tbl_out.resolve())
    payload.setdefault("sc_prec", SC_PREC)
    payload.setdefault("halve_bipolar_stoc_len", HALVE)
    out.write_text(json.dumps(payload, indent=1))

    # Prove the round-trip rather than trusting it: load it back through the
    # deployed parser and require the buckets to actually be there.
    try:
        from scmp_kernels.mp.config import AdaptiveMPConfig as _AMP
        # AdaptiveMPConfig takes stoc_len_levels DESCENDING (the global per-row
        # ladder). The per_row_chunk section's own `levels` are ASCENDING on
        # purpose -- the rung index IS the position, and the runtime maps
        # bucketize() output straight onto them. Two different lists, two
        # different conventions; do not "harmonise" them.
        # Construct with the TABLE's own stoc_len_levels -- the loader
        # cross-checks those two. The per_row_chunk ladder is SEPARATE and may
        # be longer (it appends the escape length as a top rung); each bucket's
        # ladder is validated on its own by _load_per_row_chunk and is never
        # compared against stoc_len_levels. Passing the per-bucket ladder here
        # trips a validation that says nothing about the emitted table.
        _tbl_levels = [int(v) for v in (table_payload.get("stoc_len_levels")
                                        or sorted(levels, reverse=True))]
        chk = _AMP(sorted(_tbl_levels, reverse=True))
        chk.load_threshold_table(str(tbl_out))
        n = len(getattr(chk, "per_row_chunk", {}) or {})
    except Exception as exc:                                # noqa: BLE001
        raise SystemExit(f"[prc] emitted table fails to load: {exc}")
    if n != len(buckets):
        raise SystemExit(
            f"[prc] round-trip lost buckets: wrote {len(buckets)}, parsed {n}")
    print(f"[prc] wrote {tbl_out} ({len(buckets)} buckets, round-trip verified)")
    print(f"[prc] wrote {out} (wrapper -> {tbl_out.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
