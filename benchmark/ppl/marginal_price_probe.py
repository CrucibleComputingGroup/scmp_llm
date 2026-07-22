#!/usr/bin/env python
"""Measured marginal-value probe on the DEPLOYED config (V17).

Decides the last open Stage-1 question: is σ (reconstruction error) the wrong
PRICE for the per-group allocation? V9's Lagrangian prices each of the 36
(op, layer-bucket) groups by σ. This probe measures, on the DEPLOYED config
(post hybrid-mask, post SmoothQuant), the MEASURED marginal value of a cycle at
each group — dNLL/d(cycles) — via a small paired-window perturbation, and ranks
the groups.

Validation target (4B ground truth from the manual cells): protecting LATE
layers beat protecting early, and cutting down_proj hurt. So a CORRECT measured
price must rank late-layer groups ABOVE early-layer groups (a cycle there buys
more NLL), and must NOT flag down_proj as over-priced. If it does, the measured
price is validated and Stage 1 should re-price with it (finer granularity next).
If it does NOT, price alone is not the fix.

Perturbation = PROMOTE each group by a small mass at the 24/16 floor boundary
(spend a few cycles there); marginal value = -mean_paired_dNLL / d_cost. Higher
= more underpriced by V9 (a cycle there is worth more).

Run via sbatch (GPU). Writes marginal_price_probe.json + a ranked summary.
"""
import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/allenjin/Projects/scmp_llm")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "kernels"))
from benchmark.ppl.mp_v16_refine import (  # noqa: E402
    _build_parent_model, _load_eval_stream, eval_windows, paired_deltas,
)
from benchmark.ppl.mp_ladder_refine import (  # noqa: E402
    _atomic_json, _protected_info, _write_wrapper, weighted_cost,
)
from benchmark.ppl.mp_joint_refine import (  # noqa: E402
    _hist_values, _shift_one_boundary, profile_level_weights,
)


def exp_cost(table, profile):
    psl, pw = _protected_info(table)
    lw = profile_level_weights(table, profile, protected_weight=pw)
    return weighted_cost(table["stoc_len_levels"], lw, pw, psl or 0)


def promote_group(table, profile, op, lb, mass_delta):
    """Promote ONE (op, layer-bucket) group off the floor by mass_delta."""
    out = copy.deepcopy(table)
    values = _hist_values(profile)
    changed = False
    for key, group in (profile.get("groups") or {}).items():
        if str(group["op"]) != op or group.get("l_bucket") != lb:
            continue
        payload = (out.get("buckets") or {}).get(key)
        if payload is None:
            continue
        hist = [float(x) for x in group["mac_weighted_hist"]]
        shifted, mv = _shift_one_boundary(
            [float(x) for x in payload["thresholds"]], hist, values,
            boundary=-1, mass_delta=+mass_delta)
        payload["thresholds"] = shifted
        changed = changed or mv.get("changed", False)
    return out, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--mass-delta", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    table = json.load(open(args.table))
    profile = json.load(open(args.profile))
    levels = [int(x) for x in table["stoc_len_levels"]]
    base_cost = exp_cost(table, profile)

    model, tok = _build_parent_model(args.model_path, Path(args.wrapper),
                                     args.alpha)
    enc = _load_eval_stream(tok, split="validation",
                            max_tokens=args.ctx * (args.n_windows + 2),
                            ctx=args.ctx)
    windows = list(range(args.n_windows))

    base = eval_windows(model, enc, windows, ctx=args.ctx, levels=levels,
                        table_path=Path(args.table), wrapper_path=Path(args.wrapper),
                        trace_path=outdir / "base_trace.json",
                        split_label="probe_baseline")
    base_nll = base["window_nll"]
    print(f"[probe] baseline ppl={base['ppl']:.5f} cost={base_cost:.4f} "
          f"windows={len(windows)}", flush=True)

    ops = sorted({str(g["op"]) for g in profile["groups"].values()})
    lbs = sorted({g["l_bucket"] for g in profile["groups"].values()})
    results = []
    for op in ops:
        for lb in lbs:
            cand, changed = promote_group(table, profile, op, lb, args.mass_delta)
            if not changed:
                results.append({"op": op, "l_bucket": lb, "changed": False,
                                "marginal_value": None})
                continue
            ctab = outdir / f"cand_{op}_l{lb}.json"
            cwrap = outdir / f"cand_{op}_l{lb}_wrapper.json"
            _atomic_json(ctab, cand)
            _write_wrapper(cwrap, levels, ctab)
            d_cost = exp_cost(cand, profile) - base_cost   # >0 (promoted = spend)
            ev = eval_windows(model, enc, windows, ctx=args.ctx, levels=levels,
                              table_path=ctab, wrapper_path=cwrap,
                              trace_path=outdir / f"cand_{op}_l{lb}_trace.json",
                              split_label=f"probe_{op}_l{lb}")
            deltas = paired_deltas(ev["window_nll"], base_nll, windows)
            mean_d = sum(deltas) / len(deltas)          # <0 = promoting helped
            marginal = (-mean_d / d_cost) if d_cost > 1e-9 else None
            results.append({"op": op, "l_bucket": lb, "changed": True,
                            "mean_dNLL": mean_d, "d_cost": d_cost,
                            "marginal_value": marginal})
            print(f"[probe] {op:11s} l{lb}: dNLL={mean_d:+.6f} dcost={d_cost:+.4f}"
                  f" marginal={marginal if marginal is None else round(marginal,6)}",
                  flush=True)
            for p in (ctab, cwrap):
                try: p.unlink()
                except OSError: pass

    _atomic_json(outdir / "marginal_price_probe.json",
                 {"model": args.model_path, "base_ppl": base["ppl"],
                  "base_cost": base_cost, "mass_delta": args.mass_delta,
                  "n_windows": args.n_windows, "results": results})

    # ---- ranked summary + validation check --------------------------------
    valid = [r for r in results if r.get("marginal_value") is not None]
    valid.sort(key=lambda r: -r["marginal_value"])
    print("\n[probe] RANKED marginal value (high = a cycle here buys more NLL):")
    for r in valid:
        print(f"  {r['op']:11s} l{r['l_bucket']}  {r['marginal_value']:+.6f}")
    # aggregate: early (l0,l1) vs late (l2,l3) mean marginal value
    early = [r["marginal_value"] for r in valid if r["l_bucket"] in (0, 1)]
    late = [r["marginal_value"] for r in valid if r["l_bucket"] in (2, 3)]
    by_op = defaultdict(list)
    for r in valid:
        by_op[r["op"]].append(r["marginal_value"])
    e = sum(early) / len(early) if early else 0
    la = sum(late) / len(late) if late else 0
    print(f"\n[probe] early(l0,l1) mean marginal={e:+.6f}  "
          f"late(l2,l3) mean marginal={la:+.6f}")
    print(f"[probe] VALIDATION: measured price ranks {'LATE>early (MATCHES ground truth)' if la>e else 'early>=late (does NOT match)'}")
    print("[probe] per-op mean marginal (low = V9 already spends enough there):")
    for op in sorted(by_op, key=lambda o: -sum(by_op[o]) / len(by_op[o])):
        print(f"  {op:11s} {sum(by_op[op])/len(by_op[op]):+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
