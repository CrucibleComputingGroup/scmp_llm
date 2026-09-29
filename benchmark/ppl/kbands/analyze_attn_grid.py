"""Gate decision for the per-operator-family attention enable grid.

Reads probe_attn_grid.py outputs and answers ONE question per (model, operator):
does any grid arm beat the deployed 128 on BOTH independent captures?

An arm that wins on capture A and loses on capture B is noise, not a lever.
Decision rule (stated before looking at any number):
  PASS for an operator  <=>  some arm improves the mean downstream quantity on
                             capture A AND capture B, and wins on a majority of
                             the individual (capture, block) cells.
  Otherwise KILL -- no GPU eval cells.

  python -m benchmark.ppl.kbands.analyze_attn_grid <gate_*.json> [...]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

VAL = {"qk": "kl", "av": "rel_l2"}
DEPLOYED = "deployed(128)"


def load(path):
    d = json.loads(Path(path).read_text())
    return d, Path(path).stem


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    verdicts = {}
    for path in argv:
        d, name = load(path)
        rows = d["rows"]
        arms = d["arms"]
        caps = sorted({r["capture"] for r in rows})
        print(f"\n{'='*78}\n{name}   model={d['model']}\n{'='*78}")

        for op in ("qk", "av"):
            key = VAL[op]
            # cell[(capture, block, arm)] = value
            cell = {(r["capture"], r["block"], r["arm"]): r[key]
                    for r in rows if r["op"] == op and key in r}
            if not cell:
                continue
            blocks = sorted({b for (_, b, _) in cell})
            Ls = {r["mean_L"] for r in rows if r["op"] == op}
            print(f"\n-- {op}  (mean deployed stream length "
                  f"{sum(Ls)/len(Ls):.1f} cycles; metric = "
                  f"{'masked softmax KL' if op=='qk' else 'output rel-L2'}) --")
            hdr = f"{'arm':>13}"
            for c in caps:
                hdr += f" | {'cap '+c+' mean':>16} {'Δ%':>8} {'wins':>6}"
            print(hdr)

            best = None
            for a in arms:
                line = f"{a:>13}"
                deltas, ok = [], True
                for c in caps:
                    vals = [cell[(c, b, a)] for b in blocks if (c, b, a) in cell]
                    base = [cell[(c, b, DEPLOYED)] for b in blocks
                            if (c, b, DEPLOYED) in cell]
                    mv, mb = sum(vals) / len(vals), sum(base) / len(base)
                    dp = 100.0 * (mv - mb) / mb
                    wins = sum(1 for b in blocks
                               if cell[(c, b, a)] < cell[(c, b, DEPLOYED)])
                    line += f" | {mv:16.6e} {dp:+7.2f}% {wins:>3}/{len(blocks)}"
                    deltas.append(dp)
                    ok = ok and dp < 0
                print(line + ("   <== improves on BOTH captures" if ok and a != DEPLOYED else ""))
                if a != DEPLOYED and ok:
                    m = max(deltas)          # the WORST of the two captures
                    if best is None or m < best[1]:
                        best = (a, m)
            verdicts[(name, op)] = best
            if best:
                print(f"  -> best consistent arm: {best[0]} "
                      f"(worst-capture Δ {best[1]:+.2f}%)")
            else:
                print("  -> NO arm improves on both captures")

    print(f"\n{'='*78}\nGATE VERDICT\n{'='*78}")
    anypass = False
    for (name, op), best in sorted(verdicts.items()):
        if best:
            anypass = True
            print(f"  PASS  {name:<24} {op:<3} -> {best[0]} ({best[1]:+.2f}% worst capture)")
        else:
            print(f"  KILL  {name:<24} {op:<3} -> deployed 128 is best")
    print("\n" + ("GATE PASSES for at least one operator family: a GPU wave is "
                  "justified for the passing families only."
                  if anypass else
                  "GATE FAILS on every operator family. Do NOT run eval cells."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
