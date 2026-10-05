"""Read, check and summarize per-group precision traces (benchmark/ppl/group_trace.py).

    python benchmark/ppl/group_trace_check.py --exact DIR [--fp DIR] [--sc-trace FILE]
        [--archived-trace FILE] [--verify-tokens] [--out report.json]

1. Identity (exact capture). The groups, re-aggregated per (block, op, unit, stoc_len, d_in,
   d_out, smoothed, rng_levels) the way scmp_kernels/trace.py aggregates what the kernels
   ran -- a per-(row, chunk) call by rung and chunk width, the protected slice as one call,
   attention per head -- must equal the scmp-trace-summary written for the same windows:
   rows, MACs and row-cycles of every group, exactly.
2. FP vs exact. How many groups (and MACs) the FP trajectory puts on the same length, how
   far the others move (ladder rungs), per operator and per block (MoE expert rows joined
   on token, since the trajectories route differently), and what that does to the trace
   cost and to tile-max avg_sl (tile_cost.tile_cycle_macs, R rows per tile).
3. Context. Per-operator avg_sl of the captured windows next to the archived full-test
   trace of the same cell (a 4-window sample is not the full test).
4. --verify-tokens: the captured windows are windows 0.. of the canonical wikitext-2 test
   stream (eval_quant's), token for token.

load_group_trace / sc_calls / expert_tokens / summary_from_groups / tile_avg_sl are the
reader API.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))      # benchmark.ppl.tile_cost
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TILE_ROWS = (1, 8, 16, 32, 64)


def load_group_trace(path):
    """(meta, arrays, ops): the json, the lazily loaded npz, {(op, block, unit): desc}."""
    path = Path(path)
    meta = json.loads((path / "group_trace.json").read_text())
    if meta.get("schema") != "scmp-group-trace-v1":
        raise ValueError(f"{path}: not a scmp-group-trace-v1 trace")
    arrays = np.load(path / "group_trace.npz")
    ops = {(o["op"], o["block"], o["unit"]): o for o in meta["ops"]}
    return meta, arrays, ops


def sc_calls(meta):
    """(window, call) for every SC call, in execution order."""
    for w in sorted(meta["calls"], key=int):
        for c in meta["calls"][w]:
            if c["backend"] == "sc":
                yield int(w), c


def _chunk_classes(lengths, desc):
    """Split a [rows, chunks] map into (sub-map, chunk width) as trace.py does."""
    widths, chunk_d = desc["chunk_widths"], desc["chunk_d"]
    if widths[-1] == chunk_d:
        return [(lengths, chunk_d)]
    return [(lengths[:, :-1], chunk_d), (lengths[:, -1:], widths[-1])]


def summary_from_groups(meta, arrays, ops):
    """{(block, op, unit, stoc_len, d_in, d_out, smoothed, rng_levels): [rows, macs,
    row_cycles]} -- the scmp-trace-summary-v1 aggregation of the SC calls, exact integers."""
    rng = meta["header"]["sc"]["rng_levels"]
    acc = defaultdict(lambda: [0, 0, 0])

    def add(c, unit, sl, rows, d_in, smoothed):
        a = acc[(c["block"], c["op"], unit, sl, d_in, c["d_out"], smoothed, rng)]
        a[0] += rows
        a[1] += rows * d_in * c["d_out"]
        a[2] += rows * sl

    for _, c in sc_calls(meta):
        desc = ops[(c["op"], c["block"], c["unit"])]
        if c["op"] in LINEARS:
            if c["groups"] is not None:
                for sub, width in _chunk_classes(arrays[c["groups"]], desc):
                    vals, cnt = np.unique(sub, return_counts=True)
                    for v, k in zip(vals.tolist(), cnt.tolist()):
                        add(c, c["unit"], v, k, width, desc["smoothed"])
            if desc["protected_cols"]:
                add(c, c["unit"], desc["protected_stoc_len"], c["rows"],
                    desc["protected_cols"], desc["smoothed"])
        else:
            lengths, heads = arrays[c["groups"]], c["heads"]
            for h in range(heads):
                vals, cnt = np.unique(lengths[h::heads], return_counts=True)
                for v, k in zip(vals.tolist(), cnt.tolist()):
                    add(c, h, v, k, c["d_in"], desc["smoothed"])
    return dict(acc)


def summary_from_trace(path):
    t = json.loads(Path(path).read_text())
    if t.get("schema") != "scmp-trace-summary-v1":
        raise ValueError(f"{path}: not a scmp-trace-summary-v1 trace")
    acc = defaultdict(lambda: [0, 0, 0])
    for g in t["groups"]:
        if g["dims_vary"]:
            raise ValueError(f"{path}: dims_vary group, shapes are not exact")
        a = acc[(g["block"], g["op"], g["unit"], g["stoc_len"], g["d_in"], g["d_out"],
                 g["smoothed"], g["rng_levels"])]
        a[0] += g["rows"]
        a[1] += g["macs"]
        a[2] += g["row_cycles"]
    return t["header"], dict(acc)


def identity(groups, trace):
    keys = set(groups) | set(trace)
    bad = sorted((k for k in keys if groups.get(k) != trace.get(k)), key=str)
    return dict(ok=not bad, n_keys=len(keys), n_mismatch=len(bad),
                macs_groups=sum(v[1] for v in groups.values()),
                macs_trace=sum(v[1] for v in trace.values()),
                first_mismatches=[dict(key=list(k), groups=groups.get(k), trace=trace.get(k))
                                  for k in bad[:20]])


def per_op_avg_sl(summary):
    """MAC-weighted mean stoc_len per op, plus 'all' / 'linear' / 'attention' -- the trace
    cost definition (every SC group, protected slice included)."""
    num, den = defaultdict(int), defaultdict(int)
    for key, (_, macs, _) in summary.items():
        op, sl = key[1], key[3]
        for k in (op, "all", "linear" if op in LINEARS else "attention"):
            num[k] += macs * sl
            den[k] += macs
    return {k: num[k] / den[k] for k in den}


def expert_tokens(selected, expert):
    """Token index of every row of an expert's map. HF 4.51 Qwen3MoeSparseMoeBlock gathers
    rows = torch.where(one_hot(selected).permute(2, 1, 0)[expert]): top-k slot major, then
    token."""
    return np.nonzero(np.asarray(selected).T == expert)[1]


def _group_macs(c, desc, shape):
    if c["op"] in LINEARS:
        return np.broadcast_to(np.asarray(desc["chunk_widths"], np.float64) * c["d_out"], shape)
    return np.full(shape, float(c["d_in"] * c["d_out"]))


def compare(fp, exact):
    """Group-by-group agreement of the FP-trajectory capture with the exact one.

    Dense calls pair row by row. A MoE expert's rows are joined on token (its token order
    rebuilt from each capture's router_selected), because the two trajectories route
    differently; groups routed in only one capture are counted, not compared. The
    dispatched_* lengths cover the compared groups only (no protected slice); the trace-cost
    comparison is per_op_avg_sl of each capture."""
    (meta_f, arr_f, _), (meta_e, arr_e, ops_e) = fp, exact
    struct = lambda m: {w: [(c["block"], c["op"], c["backend"], c["rows"], c["d_in"],
                             c["d_out"]) for c in cs if c["unit"] is None]
                        for w, cs in m["calls"].items()}
    stats = defaultdict(lambda: defaultdict(float))
    by_block = defaultdict(lambda: [0, 0])
    unmatched = defaultdict(int)
    for w, c in sc_calls(meta_e):
        key = c["groups"]
        if key is None:
            continue
        e = arr_e[key]
        if key not in arr_f.files:
            unmatched["exact_only_groups"] += e.size
            continue
        f = arr_f[key]
        if c["unit"] is not None:
            sel = f"w{w}/b{c['block']}/router_selected"
            te = expert_tokens(arr_e[sel], c["unit"])
            tf = expert_tokens(arr_f[sel], c["unit"])
            if len(te) != e.shape[0] or len(tf) != f.shape[0]:
                raise ValueError(f"{key}: expert rows do not match the captured routing")
            _, ie, jf = np.intersect1d(te, tf, assume_unique=True, return_indices=True)
            unmatched["exact_only_groups"] += (len(te) - len(ie)) * e.shape[1]
            unmatched["fp_only_groups"] += (len(tf) - len(jf)) * f.shape[1]
            e, f = e[ie], f[jf]
        elif e.shape != f.shape:
            raise ValueError(f"{key}: dense call with different shapes in the two captures")
        desc = ops_e[(c["op"], c["block"], c["unit"])]
        macs = _group_macs(c, desc, e.shape)
        ladder = np.asarray(sorted(set(desc["ladder"])))
        dr = np.abs(np.searchsorted(ladder, e) - np.searchsorted(ladder, f))
        ef, ff = e.astype(np.float64), f.astype(np.float64)
        for k in (c["op"], "all", "linear" if c["op"] in LINEARS else "attention"):
            s = stats[k]
            s["groups"] += e.size
            s["same"] += int((dr == 0).sum())
            s["within_1_rung"] += int((dr <= 1).sum())
            s["macs"] += macs.sum()
            s["macs_same"] += macs[dr == 0].sum()
            s["cycles_exact"] += (macs * ef).sum()
            s["cycles_fp"] += (macs * ff).sum()
            s["fp_longer_macs"] += macs[f > e].sum()
            s["fp_shorter_macs"] += macs[f < e].sum()
        by_block[c["block"]][0] += e.size
        by_block[c["block"]][1] += int((dr == 0).sum())
    for w, c in sc_calls(meta_f):
        if c["groups"] is not None and c["groups"] not in arr_e.files:
            unmatched["fp_only_groups"] += arr_f[c["groups"]].size
    routing = [float((np.sort(arr_e[k], 1) == np.sort(arr_f[k], 1)).all(1).mean())
               for k in arr_e.files if k.endswith("/router_selected") and k in arr_f.files]
    out = {}
    for k, s in stats.items():
        out[k] = dict(groups=int(s["groups"]), same_frac=s["same"] / s["groups"],
                      within_1_rung_frac=s["within_1_rung"] / s["groups"],
                      mac_same_frac=s["macs_same"] / s["macs"],
                      dispatched_avg_sl_exact=s["cycles_exact"] / s["macs"],
                      dispatched_avg_sl_fp=s["cycles_fp"] / s["macs"],
                      dispatched_rel_err=s["cycles_fp"] / s["cycles_exact"] - 1,
                      mac_frac_fp_longer=s["fp_longer_macs"] / s["macs"],
                      mac_frac_fp_shorter=s["fp_shorter_macs"] / s["macs"])
    return dict(dense_call_structure_identical=struct(meta_f) == struct(meta_e),
                unmatched=dict(unmatched),
                routing_same_set_frac=(sum(routing) / len(routing)) if routing else None,
                per_op=out,
                per_block_same_frac={b: v[1] / v[0] for b, v in sorted(by_block.items())})


def tile_avg_sl(meta, arrays, ops, tile_rows=TILE_ROWS):
    """Flat (R=1) and tile-max avg_sl of the captured windows: an R-row tile runs at its
    longest row, per 128-chunk (tile_cost.py's hardware model and function)."""
    import torch
    from benchmark.ppl.tile_cost import tile_cycle_macs
    cyc, macs = defaultdict(float), defaultdict(float)
    fixed_macs = fixed_cyc = 0.0
    for _, c in sc_calls(meta):
        desc = ops[(c["op"], c["block"], c["unit"])]
        fam = "linear" if c["op"] in LINEARS else "attention"
        if fam == "linear":
            if desc["protected_cols"]:
                m = c["rows"] * desc["protected_cols"] * c["d_out"]
                fixed_macs += m
                fixed_cyc += m * desc["protected_stoc_len"]
            if c["groups"] is None:
                continue
            lengths = torch.as_tensor(arrays[c["groups"]].astype(np.int64))[None]
            row_macs = torch.as_tensor(desc["chunk_widths"], dtype=torch.float64) * c["d_out"]
        else:
            lengths = torch.as_tensor(arrays[c["groups"]].astype(np.int64))[..., None]
            row_macs = torch.tensor([float(c["d_in"] * c["d_out"])], dtype=torch.float64)
        for r, v in tile_cycle_macs(lengths, row_macs, tile_rows).items():
            cyc[(fam, r)] += float(v)
        macs[fam] += float(row_macs.sum()) * lengths.shape[0] * lengths.shape[1]
    lin_m, att_m = macs["linear"] + fixed_macs, macs["attention"]
    res = {}
    for r in tile_rows:
        lin_c, att_c = cyc[("linear", r)] + fixed_cyc, cyc[("attention", r)]
        res[str(r)] = dict(all=(lin_c + att_c) / (lin_m + att_m),
                           linear=lin_c / lin_m if lin_m else None,
                           attention=att_c / att_m if att_m else None)
    return res


def verify_tokens(meta, arrays):
    """Windows 0.. must be eval_quant's canonical wikitext-2 test stream."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    h = meta["header"]
    tok = AutoTokenizer.from_pretrained(h["model"])
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    enc = tok(text, return_tensors="np").input_ids[0]
    ctx = h["ctx"]
    ok = {}
    for w in meta["header"]["windows"]:
        key = f"w{w}/token_ids"
        ok[w] = key in arrays.files and np.array_equal(
            arrays[key].astype(np.int64), enc[w * ctx:(w + 1) * ctx])
    return dict(ok=all(ok.values()), per_window=ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exact", required=True, help="group trace captured on the SC path")
    ap.add_argument("--fp", help="group trace captured on the FP trajectory")
    ap.add_argument("--sc-trace", help="scmp-trace-summary of the exact capture's windows")
    ap.add_argument("--archived-trace", help="archived full-test trace of the same cell")
    ap.add_argument("--verify-tokens", action="store_true")
    ap.add_argument("--out", help="write the report here (json)")
    args = ap.parse_args()

    exact = load_group_trace(args.exact)
    report = dict(exact=args.exact, fp=args.fp, windows=exact[0]["header"]["windows"])
    groups = summary_from_groups(*exact)
    report["avg_sl_exact"] = per_op_avg_sl(groups)
    report["tile_avg_sl_exact"] = tile_avg_sl(*exact)
    ok = True
    if args.sc_trace:
        _, trace = summary_from_trace(args.sc_trace)
        report["identity"] = identity(groups, trace)
        ok &= report["identity"]["ok"]
        print(f"[check] identity vs {args.sc_trace}: ok={report['identity']['ok']} "
              f"keys={report['identity']['n_keys']} mismatches={report['identity']['n_mismatch']} "
              f"macs {report['identity']['macs_groups']} vs {report['identity']['macs_trace']}")
    if args.verify_tokens:
        report["tokens"] = verify_tokens(exact[0], exact[1])
        ok &= report["tokens"]["ok"]
        print(f"[check] tokens = canonical stream windows: {report['tokens']}")
    if args.fp:
        fp = load_group_trace(args.fp)
        report["avg_sl_fp"] = per_op_avg_sl(summary_from_groups(*fp))
        report["tile_avg_sl_fp"] = tile_avg_sl(*fp)
        report["fp_vs_exact"] = cmp_all = compare(fp, exact)
        ok &= cmp_all["dense_call_structure_identical"]
        cmp, le, lf = cmp_all["per_op"], report["avg_sl_exact"], report["avg_sl_fp"]
        print(f"[check] fp vs exact: dense call structure identical = "
              f"{cmp_all['dense_call_structure_identical']}, unmatched {cmp_all['unmatched']}, "
              f"routing same-set frac {cmp_all['routing_same_set_frac']}")
        print("  same / <=1rung / MACsame: share of groups on the exact length / within one "
              "ladder rung / MAC-weighted same; L = trace cost (protected slice included)")
        print(f"  {'op':<10} {'groups':>10} {'same':>7} {'<=1rung':>8} {'MACsame':>8} "
              f"{'L exact':>8} {'L fp':>8} {'dL':>7}")
        for op in list(LINEARS) + ["qk", "av", "linear", "attention", "all"]:
            if op in cmp:
                s = cmp[op]
                print(f"  {op:<10} {s['groups']:>10} {s['same_frac']:>7.4f} "
                      f"{s['within_1_rung_frac']:>8.4f} {s['mac_same_frac']:>8.4f} "
                      f"{le[op]:>8.3f} {lf[op]:>8.3f} {100 * (lf[op] / le[op] - 1):>+6.2f}%")
        for r in TILE_ROWS:
            e, f = report["tile_avg_sl_exact"][str(r)], report["tile_avg_sl_fp"][str(r)]
            print(f"  tile R={r:<3d} avg_sl exact {e['all']:.3f} fp {f['all']:.3f} "
                  f"({100 * (f['all'] / e['all'] - 1):+.2f}%)  linear {e['linear']:.3f}/"
                  f"{f['linear']:.3f}  attention {e['attention']:.3f}/{f['attention']:.3f}")
    if args.archived_trace:
        _, arch = summary_from_trace(args.archived_trace)
        report["avg_sl_archived_full_test"] = per_op_avg_sl(arch)
        a, e = report["avg_sl_archived_full_test"], report["avg_sl_exact"]
        print("[check] avg_sl (captured windows, exact) vs archived full test: " + ", ".join(
            f"{k} {e[k]:.2f}/{a[k]:.2f}" for k in ("all", "linear", "attention")))
    report["ok"] = bool(ok)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"[check] wrote {args.out}")
    print(f"[check] ok={ok}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
