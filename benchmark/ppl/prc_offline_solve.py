"""Offline re-solve of the global per-group allocation from a calib7/8 per-pair dump.

Why: the per-token MC-Fisher weight is extremely heavy-tailed. On the 4B t32 calib8 dump one
of 12 calibration windows carries 80-99% of every operator's Fisher mass, ~20 token rows
around a wikitext section break ("\\n\\n = = Heading = = \\n\\n", delimiter tokens that act
as secondary attention sinks) carry 83% of it, and the global lambda solved on that mass
predicts +17.7% PPL on held-out windows. The deployable rule cannot see which token a group
belongs to, so per-token sensitivity is noise to the allocation; what it can use is sensitivity
that is predictable from (operator, block, chunk, metric). This script re-prices the dumped
pairs with ROBUST Fisher estimates and re-solves the same monotone-staircase global lambda,
emitting deployable tables (same runtime rule) for a held-out NLL comparison:

  fis          per-pair Fisher (= calib8's gfis; reference)
  fisclipC     each (window, block, token row)'s Fisher mass capped at C x the median row mass
               of its (op, block) over the calibration windows (winsorized per-token weight)
  kraw         raw squared error x kappa(op, block), kappa = MEDIAN over calibration windows of
               that window's sum(Fisher err)/sum(raw err) at a mid rung (robust per-layer
               sensitivity, no per-token weight)
Keys: q = (op, layer-quartile) [as calib6/7/8's gfis], blk = (op, block) [per_row_chunk.layer_buckets].
Optional per-variant fields (colon-separated after the keying):
  s=<x>     solve the linears at x times the parent's per-group budget
  att=<L>   raise the ATTENTION ladder's top rung to L (halved) in the emitted table; with
            s = 1 - (added attention cost)/(linear per-group cost) the pair is iso-cost
            (both read from the round-1 trace; see kbands/PRC2_OVERNIGHT.md)

  python benchmark/ppl/prc_offline_solve.py --dump X.npz --parent <mp_best cfg dir> \\
      --out-stem <dir>/<m>_t<T>_c8o --n-blocks 36 --variants fis:q,fisclip30:q,kraw:q,kraw:blk
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark.ppl.mp_per_row_chunk_calib5 import (  # noqa: E402
    HALVE, SC_PREC, deploy_rungs, staircase_dp)


def bisect(pick, cost, target):
    lo, hi = 0.0, 1e-30
    j = pick(hi)
    while cost(j) > target and hi < 1e40:
        hi *= 10.0
        j = pick(hi)
    best = j
    for _ in range(80):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e3
        j = pick(mid)
        if cost(j) <= target:
            best, hi = j, mid
        else:
            lo = mid
        if lo > 0 and hi / lo < 1 + 1e-7:
            break
    return best


def th_from(best, bin_max, n_lv):
    th = []
    for k in range(n_lv - 1):
        low, high = np.nonzero(best <= k)[0], np.nonzero(best > k)[0]
        th.append(1.0 if high.size == 0 else (0.0 if low.size == 0 else float(bin_max[low[-1]])))
    for i in range(1, len(th)):
        th[i] = max(th[i], th[i - 1])
    return th


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out-stem", required=True)
    ap.add_argument("--n-blocks", type=int, required=True)
    ap.add_argument("--variants", default="fis:q,fisclip10:q,fisclip100:q,kraw:q,fisclip10:blk,kraw:blk")
    ap.add_argument("--bins", type=int, default=2048)
    ap.add_argument("--min-pairs-per-bin", type=int, default=8)
    ap.add_argument("--ref-len", type=int, default=24, help="mid rung (halved) for masses/kappa")
    ap.add_argument("--holdout-windows", type=int, default=2)
    ap.add_argument("--rows-per-call", type=int, default=64)
    args = ap.parse_args()

    d = np.load(args.dump)
    measured = [int(v) for v in d["measured"]]
    ladder = [int(v) for v in d["ladder"]]
    li = {L: i for i, L in enumerate(measured)}
    lad_cols = [li[L] for L in ladder]
    top, ref = li[max(ladder)], li[min(ladder, key=lambda L: abs(L - args.ref_len))]
    Lv = np.asarray(ladder, dtype=np.float64)
    parent = Path(args.parent)
    table = json.loads((parent / "table.json").read_text())
    wrapper = json.loads((parent / "wrapper.json").read_text())

    data = {"calib": {}, "hold": {}}
    for k in sorted({tuple(f.split("|")[:3]) for f in d.files if "|" in f}):
        ph, op, q = k
        pre = f"{ph}|{op}|{q}|"
        if pre + "fis" not in d.files:
            raise SystemExit(f"[off] {pre} has no fis array")
        blk = d[pre + "blk"].astype(np.int64)
        # window index (dense models): records are appended per (window, block) in block
        # order, each rows_per_call x n_chunks pairs; a key with one SC block has no blk restart
        if pre + "win" in d.files:                  # calib8 dumps carry the window index
            win = d[pre + "win"].astype(np.int64)
        else:
            nch = int(d[pre + "cidx"].max()) + 1
            ub = np.unique(blk)
            rec_i = np.arange(blk.size) // (args.rows_per_call * nch)
            win = rec_i // ub.size
            assert np.array_equal(blk[::args.rows_per_call * nch],
                                  ub[rec_i[::args.rows_per_call * nch] % ub.size]), \
                f"[off] {pre}: record layout is not (window, block)-ordered with fixed size"
        rec = {f: d[pre + f] for f in ("mn", "mac", "parL", "rpos", "cidx")}
        rec["blk"], rec["win"] = blk, win
        rec["fis"] = d[pre + "fis"].astype(np.float64)
        rec["fis"] -= rec["fis"][:, [top]]
        rec["raw"] = d[pre + "raw"].astype(np.float64)
        rec["raw"] -= rec["raw"][:, [top]]
        data[ph][(op, int(q))] = rec
    keys = sorted(data["calib"])
    print(f"[off] {len(keys)} keys; calib windows per key "
          f"{sorted({int(r['win'].max()) + 1 for r in data['calib'].values()})}")

    # robust statistics from the CALIBRATION pairs only
    rowmed, kappa = {}, {}
    for k in keys:
        r = data["calib"][k]
        unit = (r["win"] * 1000 + r["blk"]) * 100_000 + r["rpos"].astype(np.int64)
        u, inv = np.unique(unit, return_inverse=True)
        mass = np.bincount(inv, weights=r["fis"][:, ref])
        r["_unit_inv"], r["_unit_mass"] = inv, mass
        ub = (u // 100_000) % 1000
        for b in np.unique(ub):
            rowmed[(k[0], int(b))] = float(np.median(mass[ub == b]))
        for b in np.unique(r["blk"]):
            sb = r["blk"] == b
            ratios = []
            for w in np.unique(r["win"][sb]):
                s = sb & (r["win"] == w)
                den = r["raw"][s, ref].sum()
                if den > 0:
                    ratios.append(r["fis"][s, ref].sum() / den)
            kappa[(k[0], int(b))] = float(np.median(ratios)) if ratios else 0.0

    def currency(ph, k, name):
        r = data[ph][k]
        if name == "fis":
            return r["fis"]
        if name.startswith("fisclip"):
            C = float(name[len("fisclip"):])
            if ph == "calib":
                inv, mass = r["_unit_inv"], r["_unit_mass"]
            else:
                unit = (r["win"] * 1000 + r["blk"]) * 100_000 + r["rpos"].astype(np.int64)
                _, inv = np.unique(unit, return_inverse=True)
                mass = np.bincount(inv, weights=r["fis"][:, ref])
            lut = np.zeros(int(r["blk"].max()) + 1)
            for b_ in np.unique(r["blk"]):
                lut[b_] = rowmed[(k[0], int(b_))]
            cap = C * lut[r["blk"]]
            scale = np.minimum(1.0, cap / np.maximum(mass[inv], 1e-300))
            return r["fis"] * scale[:, None]
        if name == "kraw":
            lut = np.zeros(int(r["blk"].max()) + 1)
            for b_ in np.unique(r["blk"]):
                lut[b_] = kappa[(k[0], int(b_))]
            kap = lut[r["blk"]]
            return r["raw"] * kap[:, None]
        raise SystemExit(f"unknown currency {name}")

    budget = {k: float((data["calib"][k]["mac"] * data["calib"][k]["parL"]).sum()
                       / data["calib"][k]["mac"].sum()) for k in keys}
    target = sum(budget[k] * float(data["calib"][k]["mac"].sum()) for k in keys)
    hkeys = [k for k in keys if k in data["hold"]]
    n_tok = args.holdout_windows * 2047

    def score(ph_tables, cur_name):
        """held-out (Fisher err in cur_name, cost) of a {(op, sub): th} table (or None = parent)"""
        tf = tc = 0.0
        for k in hkeys:
            r = data["hold"][k]
            cu = currency("hold", k, cur_name)
            if ph_tables is None:
                Lr = r["parL"].astype(np.float64)
            else:
                Lr = np.empty(r["mn"].size)
                sub, by_blk = ph_tables
                subk = r["blk"] if by_blk else np.full(r["blk"].shape, k[1])
                for s in np.unique(subk):
                    sel = subk == s
                    Lr[sel] = Lv[deploy_rungs(torch.from_numpy(r["mn"][sel]), sub[(k[0], int(s))]).numpy()]
            cols = np.asarray([li[int(v)] for v in Lr])
            tf += float(cu[np.arange(Lr.size), cols].sum())
            tc += float((r["mac"] * Lr).sum())
        return tf, tc

    par = {c: score(None, c) for c in ("fis", "fisclip10", "kraw")}
    diag = {"variants": {}}
    for spec in [v for v in args.variants.split(",") if v]:
        cname, keying, *opt = spec.split(":")
        opt = dict(o.split("=", 1) for o in opt)
        lin_scale = float(opt.get("s", 1.0))
        att_top = int(opt["att"]) if "att" in opt else None
        by_blk = keying == "blk"
        sb = {}
        for k in keys:
            r = data["calib"][k]
            cu = currency("calib", k, cname)[:, lad_cols]
            subk = r["blk"] if by_blk else np.full(r["blk"].shape, k[1])
            for s in np.unique(subk):
                sel = subk == s
                m = r["mn"][sel]
                order = np.argsort(m, kind="stable")
                P = m.size
                B = int(min(args.bins, P)) if not by_blk else \
                    int(max(1, min(args.bins, P // max(args.min_pairs_per_bin, 1))))
                edges = np.linspace(0, P, B + 1).astype(np.int64)
                E = np.add.reduceat(cu[sel][order], edges[:-1], axis=0)
                C = np.add.reduceat(r["mac"][sel][order], edges[:-1])
                sb[(k[0], int(s))] = (E, C, m[order][edges[1:] - 1])
        rs = bisect(lambda lam: {q: staircase_dp(v[0], v[1], Lv, lam) for q, v in sb.items()},
                    lambda rr: sum(float((sb[q][1] * Lv[rr[q]]).sum()) for q in sb), target * lin_scale)
        th = {q: th_from(rs[q], sb[q][2], len(Lv)) for q in sb}
        moves = defaultdict(lambda: [0.0, 0.0])
        for q, (E, C, _) in sb.items():
            moves[q[0]][0] += float(C.sum())
            moves[q[0]][1] += float((C * Lv[rs[q]]).sum())
        row = {"n_keys": len(sb),
               "op_L": {o: m[1] / m[0] for o, m in sorted(moves.items())}}
        for c in ("fis", "fisclip10", "kraw"):
            f_, c_ = score((th, by_blk), c)
            dn = 0.5 * (f_ - par[c][0]) / n_tok
            row[f"heldout_{c}_over_parent"] = f_ / par[c][0]
            row[f"heldout_pred_dppl_{c}_pct"] = (math.exp(dn) - 1) * 100
            row["heldout_cost_over_parent"] = c_ / par[c][1]
        diag["variants"][spec] = row
        print(f"[off] {spec:16s} keys {len(sb):4d}  cost/par {row['heldout_cost_over_parent']:.4f}  "
              f"held-out pred dPPL: fis {row['heldout_pred_dppl_fis_pct']:+.2f}%  fisclip10 "
              f"{row['heldout_pred_dppl_fisclip10_pct']:+.2f}%  kraw {row['heldout_pred_dppl_kraw_pct']:+.2f}%  | "
              + " ".join(f"{o[:4]} {v:.1f}" for o, v in row["op_L"].items()), flush=True)

        # emit
        name = f"{cname}{'blk' if by_blk else ''}" + (f"_s{round(lin_scale * 1e4):d}" if "s" in opt else "") + \
            (f"_att{att_top}" if att_top else "")
        stem = Path(args.out_stem)
        o = stem.with_name(f"{stem.name}_{name}.json")
        t_ = o.with_name(o.stem + "_table.json")
        pt = json.loads(json.dumps(table))
        bk = {f"{op}:t0:l{s}": {"levels": ladder, "thresholds": v} for (op, s), v in th.items()}
        pt["per_row_chunk"] = {"buckets": bk}
        if by_blk:
            pt["per_row_chunk"]["layer_buckets"] = int(args.n_blocks)
        pt.setdefault("sc_prec", SC_PREC)
        pt.setdefault("halve_bipolar_stoc_len", HALVE)
        pt["prc_offline"] = dict(vars(args), variant=spec)
        pw = dict(wrapper)
        if att_top:
            old = max(int(v) for v in pt["stoc_len_levels"])
            cap = 2 ** (SC_PREC - 1) if HALVE else 2 ** SC_PREC
            assert old < att_top <= cap, (old, att_top)
            for obj in (pt, pw):
                obj["stoc_len_levels"] = [att_top if int(v) == old else int(v) for v in obj["stoc_len_levels"]]
            for bk_, b_ in list(pt.get("buckets", {}).items()) + list(pt.get("operator_defaults", {}).items()):
                if isinstance(b_, dict) and b_.get("levels"):
                    b_["levels"] = [att_top if int(v) == old else int(v) for v in b_["levels"]]
            pt["prc_offline"]["attention_top_rung"] = {"from": old, "to": att_top}
        t_.write_text(json.dumps(pt, indent=1))
        pw["threshold_table_path"] = str(t_.resolve())
        o.write_text(json.dumps(pw, indent=1))
        from scmp_kernels.mp.config import AdaptiveMPConfig
        chk = AdaptiveMPConfig(sorted([int(v) for v in pt["stoc_len_levels"]], reverse=True))
        chk.load_threshold_table(str(t_))
        assert len(chk.per_row_chunk) == len(bk)
        nlb = int(args.n_blocks) if by_blk else int(table.get("layer_buckets", 4))
        for (op, s), v in th.items():
            blk0 = s if by_blk else next(b for b in range(args.n_blocks)
                                         if min(nlb - 1, int(b / max(args.n_blocks - 1, 1) * nlb)) == s)
            got = chk.get_per_row_chunk(op, blk0, int(args.n_blocks))
            assert got is not None and [float(x) for x in got[1]] == [float(x) for x in v], (op, s)
        print(f"[off] wrote {o.name} ({len(bk)} buckets, lookup verified)")
        row["wrapper"] = str(o)
    Path(str(args.out_stem) + "_offline_diag.json").write_text(json.dumps(diag, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
