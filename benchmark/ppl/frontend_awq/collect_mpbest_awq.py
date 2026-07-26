#!/usr/bin/env python3
"""Collect the Tier-1 mp_best AWQ transfer probe into a joined table.

Parses the [RESULT] lines out of the wave's Slurm logs and joins each cell
against the DEPLOYED SmoothQuant number from its own mp_best bundle
(eval_summary.json), so the PPL delta and the realized-cost drift are read
side by side.

The cost column is not decoration. The bundle's budget was solved for
SmoothQuant activations; under AWQ nothing re-solves it, so a cell can pay more
cycles than it was priced at. A PPL win at a drifted cost is NOT an iso-compute
win, and this script flags it rather than letting the PPL column stand alone.

Usage: python collect_mpbest_awq.py [--logs DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MP_BEST = Path("/home/allenjin/Projects/hpca_results/llm/ppl/mp_best")
DEFAULT_LOGS = Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data"
                    "/allenjin/hpca/logs/_mpbest_awq")
MODELS = ["4B", "llama8B", "14B", "30B"]
TARGETS = [32, 96]
# Drift beyond this fraction of the deployed realized cost means the cell is no
# longer comparable at iso-compute and the PPL cannot be quoted as a clean win.
DRIFT_TOL = 0.02

# 14B t96 does NOT use its mp_best bundle: the wave runs the 20%-INT-mask
# variant from mp_avg96_hyb20_20260724 so the mask is a constant across all 8
# cells. mp_best kept 10% for this cell because 20% lost there (8.6848 @ 96.16
# vs 8.6789 @ 95.779). The baseline below is the 20% SmoothQuant number — the
# correct A/B partner, since front-end is then the only difference.
BASELINE_OVERRIDE = {
    ("14B", 96): {"ppl": 8.6848, "flop_avg_sl": 96.16,
                  "src": "mp_avg96_hyb20_20260724 (20% mask)"},
}

RESULT_RE = re.compile(
    r"\[RESULT\].*?config=(?P<config>\S+).*?value=(?P<ppl>[0-9.]+)"
    r".*?tokens=(?P<tokens>\d+).*?realized_flop_avg_sl=(?P<flop>[0-9.]+)")
CELL_RE = re.compile(r"\[mpbest_awq\] cell=(?P<model>\S+) target=(?P<target>\d+)")


def parse_logs(logs: Path) -> dict:
    out = {}
    if not logs.is_dir():
        raise SystemExit(f"[collect] no log dir: {logs}")
    for f in sorted(logs.glob("*.out")):
        text = f.read_text(errors="replace")
        cell = CELL_RE.search(text)
        res = RESULT_RE.search(text)
        if not cell:
            continue
        key = (cell["model"], int(cell["target"]))
        if not res:
            out[key] = {"status": "running-or-failed", "log": f.name}
            continue
        out[key] = {
            "status": "done",
            "ppl": float(res["ppl"]),
            "tokens": int(res["tokens"]),
            "flop_avg_sl": float(res["flop"]),
            "log": f.name,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    args = ap.parse_args()

    awq = parse_logs(args.logs)
    rows = []
    print(f"{'model':9s} {'tgt':>4s} {'SQ ppl':>9s} {'AWQ ppl':>9s} {'dPPL%':>7s} "
          f"{'SQ cyc':>8s} {'AWQ cyc':>8s} {'dcyc%':>7s}  verdict")
    for m in MODELS:
        for t in TARGETS:
            bundle = MP_BEST / "configs" / m / f"target{t}"
            meta = json.loads((bundle / "metadata.json").read_text())
            ovr = BASELINE_OVERRIDE.get((m, t))
            if ovr:
                sq_ppl, sq_cyc, cfg_src = ovr["ppl"], ovr["flop_avg_sl"], ovr["src"]
            else:
                ev = json.loads((bundle / "eval_summary.json").read_text())
                sq_ppl = float(ev["ppl"])
                sq_cyc = float(ev["realized_flop_avg_stoc_len"])
                cfg_src = "mp_best (deployed)"
            a = awq.get((m, t))
            if not a or a["status"] != "done":
                print(f"{m:9s} {t:4d} {sq_ppl:9.4f} {'—':>9s} {'—':>7s} "
                      f"{sq_cyc:8.3f} {'—':>8s} {'—':>7s}  "
                      f"{(a or {}).get('status', 'not-started')}")
                continue
            d_ppl = (a["ppl"] - sq_ppl) / sq_ppl * 100
            d_cyc = (a["flop_avg_sl"] - sq_cyc) / sq_cyc * 100
            if abs(d_cyc) > DRIFT_TOL * 100:
                verdict = f"COST DRIFTED {d_cyc:+.1f}% — not iso-compute"
            elif d_ppl < 0:
                verdict = "AWQ wins at iso-cost"
            else:
                verdict = "no win (uninformative: allocator is stale)"
            print(f"{m:9s} {t:4d} {sq_ppl:9.4f} {a['ppl']:9.4f} {d_ppl:+7.2f} "
                  f"{sq_cyc:8.3f} {a['flop_avg_sl']:8.3f} {d_cyc:+7.2f}  {verdict}")
            rows.append(dict(
                model=m, target=t, fp16_ppl=meta["fp16_ppl"],
                ppl_smoothquant=f"{sq_ppl:.4f}", ppl_awq=f"{a['ppl']:.4f}",
                delta_ppl_pct=f"{d_ppl:+.2f}",
                flop_avg_sl_smoothquant=f"{sq_cyc:.3f}",
                flop_avg_sl_awq=f"{a['flop_avg_sl']:.3f}",
                delta_cost_pct=f"{d_cyc:+.2f}",
                iso_compute="yes" if abs(d_cyc) <= DRIFT_TOL * 100 else "NO",
                x_fp16_awq=f"{a['ppl'] / meta['fp16_ppl']:.4f}",
                config_source=cfg_src,
                eval_tokens=a["tokens"], log=a["log"]))
    if rows:
        out = Path(__file__).resolve().parent / "mpbest_awq_transfer.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}  ({len(rows)}/8 cells)")
        drifted = [r for r in rows if r["iso_compute"] == "NO"]
        if drifted:
            print(f"⚠ {len(drifted)}/{len(rows)} cells drifted off their priced "
                  f"cost — their PPL is NOT an iso-compute comparison.")


if __name__ == "__main__":
    main()
