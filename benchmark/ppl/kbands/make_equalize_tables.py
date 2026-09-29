#!/usr/bin/env python3
"""Price-equalizing reallocation: move cycles from cheap operators to dear ones.

THE ARGUMENT. At the optimum of  min loss  s.t. sum cycles <= B, every operator's
marginal price dLoss/dCycles must be EQUAL (KKT). The measured prices at the
deployed allocation are not equal -- they span 5x (4B, llama8B), 8x (14B) and 24x
(30B). So the deployed allocation is provably off-optimum, and the direction of
the fix is measured, not guessed: LENGTHEN the operators that are expensive to
starve, SHORTEN the ones that are cheap, keeping total cost at or below the
control's.

This is the allocator's own objective being corrected with a directly measured
price, rather than sigma (a proxy that misprices operators by up to 17x) or a
knock-down probe (8-56% impossible-sign entries, which killed six previous
attempts at exactly this).

MECHANISM. Per (op, layer-bucket) the per_row_chunk section holds an ASCENDING
ladder and thresholds on the min-max normalized per-(row,chunk) amax. Shifting
the threshold vector one slot down lengthens that operator's streams by one rung;
one slot up shortens them. Both are the SAME perturbation the price wave used, so
the measured dPPL/dcost per op apply directly and the predicted effect is a sum of
measured terms -- stated before the run, refutable after it.

Selection is greedy and cost-bounded: lengthen in descending price while the
budget (freed by shortening the cheapest ops) allows, never exceeding the
control's realized cost.

  python make_equalize_tables.py <model> [--target 32] [--prices <json>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys

PRC = pathlib.Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/prc")
OUT = pathlib.Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/equal")
LOGS = pathlib.Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data/"
                    "allenjin/hpca/logs/_kbands")
OPS = ["down_proj", "up_proj", "gate_proj", "o_proj", "q_proj", "k_proj", "v_proj"]
RES = re.compile(r"value=([0-9.]+).*?realized_flop_avg_sl=([0-9.]+)")


def read(pat: str):
    for f in sorted(glob.glob(str(LOGS / f"{pat}_*.out")), key=os.path.getmtime,
                    reverse=True):
        if not re.fullmatch(rf"{re.escape(pat)}_\d+\.out", os.path.basename(f)):
            continue
        got = None
        for line in open(f, errors="ignore"):
            if "[RESULT]" in line:
                m = RES.search(line)
                if m:
                    got = (float(m.group(1)), float(m.group(2)))
        if got:
            return got
    return None


def prices(model: str, target: int):
    """(price, dppl_pct, dcost_pct) per op, from the marginal wave."""
    ctl = read(f"margctl_{model}_t{target}")
    if not ctl:
        raise SystemExit(f"no control for {model} t{target}")
    out = {}
    for op in OPS:
        v = read(f"marg_{model}_t{target}_{op}")
        if not v:
            continue
        dp = (v[0] - ctl[0]) / ctl[0] * 100.0
        dc = (v[1] - ctl[1]) / ctl[1] * 100.0
        if dc >= -1e-9:                      # freed nothing measurable
            continue
        out[op] = (dp / (-dc), dp, dc)
    return ctl, out


def shift(entry: dict, longer: bool) -> tuple[dict, bool]:
    th = list(entry.get("thresholds") or [])
    if len(th) < 2:
        return entry, False
    new = ([th[0]] + th[:-1]) if longer else (th[1:] + [th[-1]])
    o = dict(entry)
    o["thresholds"] = new
    return o, new != th


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--target", type=int, default=32)
    ap.add_argument("--max-rungs", dest="max_rungs", type=int, default=4,
                    help="max lengthening passes; each pass adds one rung to "
                         "every affordable operator, dearest first")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="allowed cost change vs control, %% (<=0 keeps it cheaper)")
    a = ap.parse_args()
    ctl, pr = prices(a.model, a.target)
    if len(pr) < 4:
        raise SystemExit(f"only {len(pr)} priced ops; need the full wave")

    ranked = sorted(pr.items(), key=lambda kv: -kv[1][0])
    # Shorten the cheapest ops to fund lengthening the dearest. Greedy on price.
    lengthen, shorten, budget = [], [], 0.0
    for op, (p, dp, dc) in reversed(ranked):          # cheapest first
        if p > 0.35:                                   # only genuinely cheap ones
            break
        shorten.append(op)
        budget += -dc                                  # % of total cost freed
    # Spend the freed budget DOWN. v1 stopped after one rung per operator and
    # left 10% slack on 14B/30B, so those cells came in 8-10% CHEAPER than the
    # control -- which lowers quality for trivial reasons and is NOT the
    # iso-cost test the price model predicts. Keep lengthening (repeat rungs on
    # the dearest operators, `reps` counts them) until the slack is spent.
    reps = {}
    for _ in range(a.max_rungs):
        progressed = False
        for op, (p, dp, dc) in ranked:                 # dearest first
            if op in shorten:
                continue
            cost = -dc                                 # lengthening costs ~the same
            if budget - cost < -a.margin:
                continue
            reps[op] = reps.get(op, 0) + 1
            if op not in lengthen:
                lengthen.append(op)
            budget -= cost
            progressed = True
        if not progressed:
            break
    if not lengthen or not shorten:
        raise SystemExit("no cost-feasible move found")

    pred = (sum(-pr[o][1] * reps.get(o, 1) for o in lengthen)
            + sum(pr[o][1] for o in shorten))
    print(f"{a.model} t{a.target}: control {ctl[0]:.4f} @ {ctl[1]:.2f}")
    print(f"  LENGTHEN (dear) {[(o, reps.get(o,1)) for o in lengthen]}")
    print(f"  SHORTEN  (cheap) {shorten}")
    print(f"  predicted dPPL {pred:+.2f}%  (sum of measured terms), "
          f"cost slack {budget:+.2f}%")

    wrap = json.loads((PRC / f"{a.model}_t{a.target}_v7_prc.json").read_text())
    table = json.loads(pathlib.Path(wrap["threshold_table_path"]).read_text())
    buckets = table["per_row_chunk"]["buckets"]
    n = 0
    for k in list(buckets):
        op = k.split(":")[0]
        if op in lengthen:
            ch = False
            for _ in range(reps.get(op, 1)):           # multi-rung where afforded
                buckets[k], c1 = shift(buckets[k], True)
                ch = ch or c1
        elif op in shorten:
            buckets[k], ch = shift(buckets[k], False)
        else:
            continue
        n += int(ch)

    OUT.mkdir(parents=True, exist_ok=True)
    tdst = OUT / f"{a.model}_t{a.target}_equal_table.json"
    wdst = OUT / f"{a.model}_t{a.target}_equal.json"
    tdst.write_text(json.dumps(table))
    w2 = dict(wrap)
    w2["threshold_table_path"] = str(tdst)
    wdst.write_text(json.dumps(w2, indent=1))

    rt = "ROUND-TRIP NOT RUN (no torch — use the annstention python)"
    try:
        sys.path.insert(0, "/home/allenjin/Projects/SCMP/scmp_llm/kernels")
        from scmp_kernels.mp.config import AdaptiveMPConfig
        cfg = AdaptiveMPConfig(stoc_len_levels=w2["stoc_len_levels"])
        cfg.load_threshold_table(str(tdst))
        got = len((json.loads(tdst.read_text()).get("per_row_chunk") or {})
                  .get("buckets") or {})
        assert got == len(buckets)
        rt = f"round-trip OK, {got} buckets"
    except ImportError:
        pass
    print(f"  {n} buckets shifted -> {wdst.name}  [{rt}]")
    (OUT / f"{a.model}_t{a.target}_equal_plan.json").write_text(json.dumps(
        {"model": a.model, "target": a.target, "control_ppl": ctl[0],
         "control_cost": ctl[1], "lengthen": lengthen, "shorten": shorten,
         "predicted_dppl_pct": pred, "cost_slack_pct": budget,
         "prices": {k: {"price": v[0], "dppl_pct": v[1], "dcost_pct": v[2]}
                    for k, v in pr.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
