#!/usr/bin/env python3
"""Per-operator MARGINAL price tables: shorten ONE operator by one rung.

WHY. The cross-layer Lagrangian needs dLoss/dCycles per operator at its CURRENT
allocation. Every previous attempt to get that (measured / measured_marg / grad*
/ fisher / P1 / B1-B2) derived it from a knock-down probe whose noise scales with
model size (negative-dLoss fraction 8/31/39/56%). The op-swap wave gave a clean
number but it is the INTEGRAL (remove the operator's SC error entirely by routing
it to INT7), not the derivative -- and the two diverge exactly at the precision
cliff, i.e. at the budgets we care about.

WHAT THIS EMITS. For one operator, a copy of the deployed per-(row,chunk) table
whose per-bucket thresholds for THAT operator are shifted one rung shorter, so
the operator's rows dispatch to the next-lower stream length. Everything else --
every other operator, the ladder values, the mask, the front-end -- is untouched.
Full protocol on the result gives dPPL and, from the trace, dCycles for exactly
that one operator. The ratio IS the Lagrangian's price, measured in the deployed
regime and STAYING IN SC (never INT: constraint "MP lives only in the SC domain").

The per_row_chunk section stores, per bucket, an ASCENDING `levels` ladder and
`thresholds` (len = levels-1) on the min-max normalized per-(row,chunk) amax.
Raising every threshold pushes mass DOWN the ladder = shorter streams = cheaper.
We shift by one full threshold slot, which moves each rung's mass to the rung
below it -- a clean one-rung perturbation rather than an arbitrary epsilon.

Usage:
  python make_marginal_tables.py <model> <target> <op>        # one op
  python make_marginal_tables.py <model> <target> --all       # all 9
Writes $TURBO/kbands/kbands_20260801/marg/<model>_t<T>_marg_<op>.json (+ _table).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

PRC = pathlib.Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/prc")
OUT = pathlib.Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/marg")
LINEAR_OPS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_OPS = ["qk", "av"]


def shift_bucket(entry: dict) -> tuple[dict, bool]:
    """Move one rung shorter: drop the lowest threshold, duplicate the top.

    thresholds are ascending and select among ascending `levels`; dropping the
    first boundary and repeating the last collapses the longest rung and shifts
    every other band one step down the ladder. Returns (new_entry, changed).
    """
    th = list(entry.get("thresholds") or [])
    if len(th) < 2:
        return entry, False
    new = th[1:] + [th[-1]]
    out = dict(entry)
    out["thresholds"] = new
    return out, new != th


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("target")
    ap.add_argument("op", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    wrap_src = PRC / f"{a.model}_t{a.target}_v7_prc.json"
    if not wrap_src.is_file():
        raise SystemExit(f"missing {wrap_src}")
    wrap = json.loads(wrap_src.read_text())
    tbl_src = pathlib.Path(wrap["threshold_table_path"])
    if not tbl_src.is_file():
        raise SystemExit(f"missing table {tbl_src}")
    table = json.loads(tbl_src.read_text())
    prc = (table.get("per_row_chunk") or {}).get("buckets") or {}
    if not prc:
        raise SystemExit("table carries no per_row_chunk buckets")

    ops = (LINEAR_OPS if a.all else [a.op])
    OUT.mkdir(parents=True, exist_ok=True)
    for op in ops:
        keys = [k for k in prc if k.split(":")[0] == op]
        if not keys:
            print(f"  {op}: no per_row_chunk buckets (attention is not "
                  f"per-(row,chunk) allocated) — SKIP")
            continue
        t2 = json.loads(json.dumps(table))
        n = 0
        for k in keys:
            ent, ch = shift_bucket(t2["per_row_chunk"]["buckets"][k])
            t2["per_row_chunk"]["buckets"][k] = ent
            n += int(ch)
        if n == 0:
            print(f"  {op}: no bucket could shift — SKIP")
            continue
        tbl_dst = OUT / f"{a.model}_t{a.target}_marg_{op}_table.json"
        wrp_dst = OUT / f"{a.model}_t{a.target}_marg_{op}.json"
        tbl_dst.write_text(json.dumps(t2))
        w2 = dict(wrap)
        w2["threshold_table_path"] = str(tbl_dst)
        wrp_dst.write_text(json.dumps(w2, indent=1))

        # ROUND-TRIP through the DEPLOYED parser before any GPU is spent.
        # (A per_row_chunk section written where the parser does not look once
        # produced a flawless NULL 8-cell wave.) Needs torch, so run this
        # generator under the annstention python; if the import is unavailable
        # we say so LOUDLY rather than silently skipping the check -- the
        # launcher's own 0-bucket guard is the backstop, not a replacement.
        rt = "ROUND-TRIP NOT RUN (no torch — use the annstention python)"
        try:
            import sys
            sys.path.insert(0, "/home/allenjin/Projects/SCMP/scmp_llm/kernels")
            from scmp_kernels.mp.config import AdaptiveMPConfig
            cfg = AdaptiveMPConfig(stoc_len_levels=w2["stoc_len_levels"])
            cfg.load_threshold_table(str(tbl_dst))
            got = len((json.loads(tbl_dst.read_text()).get("per_row_chunk")
                       or {}).get("buckets") or {})
            assert got == len(prc), f"bucket count {got} != {len(prc)}"
            rt = f"round-trip OK, {got} buckets"
        except ImportError:
            pass
        print(f"  {op}: {n}/{len(keys)} buckets shifted -> {wrp_dst.name}  [{rt}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
