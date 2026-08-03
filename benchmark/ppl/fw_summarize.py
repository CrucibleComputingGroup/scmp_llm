#!/usr/bin/env python3
"""fw_summarize.py — regenerate the MP summary TSV from results + manifest.

Replaces the prior session's ad-hoc summary computation, whose flop_avg_sl was
sourced from each table's `operator_defaults` — a never-exercised runtime
fallback fit OUTSIDE the joint budget (mean-W_g mispricing). That produced the
phantom "measured overspends" numbers (llama8B 74.1@64). The deployed
allocation lives in `buckets`, and its MAC-weighted average is the true
iso-compute check.

flop_avg_sl here = the table's `expected_flop_avg_stoc_len` (exported by
calibrate_mp_thresholds.py since 2026-07-05) or, for older tables, recomputed
from `buckets` with weights R_(op,bucket) x mac_per_row(op) taken from the
same --mac-weights-trace the calibration used (parsed from calib_command).

Usage:
  python benchmark/ppl/fw_summarize.py \
    --results  /home/allenjin/Projects/hpca_results/llm/mp/fw_results.tsv \
    --manifest /home/allenjin/Projects/hpca_results/llm/mp/fw_manifest.tsv \
    --out      /home/allenjin/Projects/hpca_results/llm/mp/fw_summary.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

UNIFORM_TWIN = {"int8": "sc_int8", "len192": "sc_avg192",
                "int7": "sc_int7", "len96": "sc_avg96", "int6": "sc_int6"}
RESULTS_BASE = "/home/allenjin/Projects/hpca_results/llm"


def _bucket_index(value: int, total: int, num_buckets: int) -> int:
    # Verbatim mirror of calibrate_mp_thresholds._bucket_index — block->bucket
    # mapping must match the calibration or the recompute is meaningless.
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


def flop_avg_from_table(table: dict) -> float:
    """expected_flop_avg_stoc_len, recomputed from buckets+trace when absent."""
    if "expected_flop_avg_stoc_len" in table:
        return float(table["expected_flop_avg_stoc_len"])
    m = re.search(r"--mac-weights-trace\s+(\S+)", table.get("calib_command", ""))
    if not m:
        raise SystemExit("table has neither expected_flop_avg_stoc_len nor a "
                         "--mac-weights-trace in calib_command — cannot compute "
                         "an iso-compute average (row-weighted-era table?)")
    trace = json.load(open(m.group(1)))
    rows_ob, macs_op, rows_op = defaultdict(float), defaultdict(float), defaultdict(float)
    total_blocks = max(g["block"] for g in trace["groups"]) + 1
    nb = int(table.get("layer_buckets", 4))
    for g in trace["groups"]:
        b = _bucket_index(int(g["block"]), total_blocks, nb)
        rows_ob[(g["op"], b)] += g["rows"]
        macs_op[g["op"]] += g["macs"]
        rows_op[g["op"]] += g["rows"]
    mac_per_row = {op: macs_op[op] / rows_op[op] for op in macs_op if rows_op[op]}
    num = den = 0.0
    for bkey, fitted in table["buckets"].items():
        op, _t, lpart = bkey.split(":")
        l = int(lpart[1:])
        w = rows_ob.get((op, l), 0.0) * mac_per_row.get(op, 1.0)
        num += w * float(fitted["avg_stoc_len"])
        den += w
    return num / den if den else 0.0


def load_uniform(path: str) -> dict:
    twins = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            twins[(r["model"], r["config"])] = float(r["ppl"])  # last wins
    return twins


def load_fp16(int_dir: str) -> dict:
    out = {}
    for fn in os.listdir(int_dir):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(int_dir, fn)) as f:
            for r in csv.DictReader(f):
                if r["config"] == "fp16" and r["metric"] == "ppl":
                    out[r["model"]] = float(r["value"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--uniform", default=os.path.join(RESULTS_BASE, "uniform", "uniform_all.csv"))
    ap.add_argument("--int-dir", default=os.path.join(RESULTS_BASE, "int"))
    args = ap.parse_args()

    tables = {}   # (model,budget,method) -> table path
    with open(args.manifest) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            tables[(r["model"], r["budget"], r["method"])] = \
                r["wrapper"].replace("_wrapper.json", ".json")

    twins, fp16 = load_uniform(args.uniform), load_fp16(args.int_dir)

    rows = []
    with open(args.results) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["status"] != "OK":
                continue
            key = (r["model"], r["budget"], r["method"])
            tpath = tables.get(key)
            if not tpath or not os.path.isfile(tpath):
                print(f"[warn] no table for {key} — skipped", file=sys.stderr)
                continue
            flop = flop_avg_from_table(json.load(open(tpath)))
            ppl = float(r["ppl"])
            uni = twins.get((r["model"], UNIFORM_TWIN.get(r["budget"], "")))
            base = fp16.get(r["model"])
            rows.append({
                "model": r["model"], "budget": r["budget"], "method": r["method"],
                "flop_avg_sl": f"{flop:.1f}", "ppl": f"{ppl:.4f}",
                "x_fp16": f"{ppl / base:.3f}" if base else "-",
                "uniform_ppl": f"{uni:.4f}" if uni else "-",
                "vs_uniform_pct": f"{(ppl / uni - 1) * 100:+.1f}" if uni else "-",
            })

    rows.sort(key=lambda r: (r["model"], r["budget"], r["method"]))
    cols = ["model", "budget", "method", "flop_avg_sl", "ppl", "x_fp16",
            "uniform_ppl", "vs_uniform_pct"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"[fw_summarize] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
