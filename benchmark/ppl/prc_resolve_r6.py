"""CPU re-solve of calib9_r6 tables from a --solve-state npz (round 6, 2026-09-28).

calib9 (benchmark/ppl/mp_per_row_chunk_calib9_r6.py) saves the binned solve inputs it used
(linear Fisher bins, attention bins, budgets, fixed escape cost, targets). This tool re-runs
the SAME module-level solve functions (solve_global / solve_joint / th_from_bins /
att_th_from) at any budget scale and emits deployable tables with the SAME emit_table code,
so a new scale needs no second GPU capture. At a scale calib9 also emitted, the thresholds
are bit-identical to calib9's in-process table (checked by `compare_thresholds`, used by the
round-6 driver as a state round-trip identity check).

  python benchmark/ppl/prc_resolve_r6.py --state <cell>_r6_state.npz \
      --parent <mp_best cfg dir> --out-stem <dir>/<cell>_r6x.json --scales 1.0161 --tables gfisla

No GPU, no model. Pure allocation: only per_row_chunk thresholds (and, for gfisla*, the
parent attention buckets' thresholds) change; ladders, escape, mask, RNG are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.ppl.mp_per_row_chunk_calib9_r6 import (  # noqa: E402
    base_table_name, emit_table, load_solve_state, parse_scales, resolve_tables)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare_thresholds(name, lin_th, att_th, table_payload):
    """Mismatches between a resolved (lin_th, att_th) and an emitted table JSON payload.
    Exact float equality after JSON round trip (json floats round-trip exactly)."""
    bad = []
    prc = (table_payload.get("per_row_chunk") or {}).get("buckets") or {}
    want = {f"{k[0]}:t0:l{k[1]}": th for k, th in lin_th.items() if k != "__levels__"}
    if set(prc) != set(want):
        bad.append(f"{name}: per_row_chunk bucket keys differ "
                   f"({sorted(set(prc) ^ set(want))[:6]})")
    for kk, th in want.items():
        got = (prc.get(kk) or {}).get("thresholds")
        if got is None or [float(x) for x in got] != [float(x) for x in json.loads(json.dumps(th))]:
            bad.append(f"{name}: per_row_chunk {kk} thresholds differ")
    if att_th is not None:
        for (op, b), th in att_th.items():
            kk = f"{op}:t0:l{b}"
            got = ((table_payload.get("buckets") or {}).get(kk) or {}).get("thresholds")
            if got is None or [float(x) for x in got] != [float(x) for x in json.loads(json.dumps(th))]:
                bad.append(f"{name}: attention {kk} thresholds differ")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True)
    ap.add_argument("--parent", required=True, help="parent cfg dir (table.json/wrapper.json)")
    ap.add_argument("--out-stem", default="", help="emit tables at <stem>_<name>.json")
    ap.add_argument("--scales", required=True)
    ap.add_argument("--tables", default="gfisla", help="gfis,gfisla")
    ap.add_argument("--compare-stem", default="",
                    help="compare against calib9's emitted <stem>_<name>_table.json instead of "
                         "(or before) emitting; exits 3 on any mismatch")
    args = ap.parse_args()
    meta, keys, akeys, lbins, abins = load_solve_state(args.state)
    parent = Path(args.parent)
    if sha256(parent / "table.json") != meta["parent_table_sha256"]:
        raise SystemExit("[resolve] parent table.json differs from the one the state was built on")
    if sha256(parent / "wrapper.json") != meta["parent_wrapper_sha256"]:
        raise SystemExit("[resolve] parent wrapper.json differs from the one the state was built on")
    which = [t.strip() for t in args.tables.split(",") if t.strip()]
    if not set(which) <= {"gfis", "gfisla"}:
        raise SystemExit(f"[resolve] --tables must be within gfis,gfisla: {which}")
    scales = parse_scales(args.scales)
    res = resolve_tables(meta, keys, akeys, lbins, abins, scales, which)
    bad = []
    if args.compare_stem:
        stem = Path(args.compare_stem)
        for name, (lin, att, rec) in res.items():
            tp = stem.with_name(f"{stem.stem}_{name}_table.json")
            if not tp.is_file():
                bad.append(f"{name}: missing {tp}")
                continue
            bad += compare_thresholds(name, lin, att, json.loads(tp.read_text()))
        print(f"[resolve] compare vs {stem}: {'IDENTICAL' if not bad else f'{len(bad)} mismatches'}")
        for b in bad[:20]:
            print("   ", b)
    if args.out_stem:
        table = json.loads((parent / "table.json").read_text())
        wrapper = json.loads((parent / "wrapper.json").read_text())
        for name, (lin, att, rec) in res.items():
            extra = dict(meta.get("c9_record") or {}, table=name, solve=rec,
                         resolved_from_state=str(Path(args.state).resolve()),
                         state_sha256=sha256(args.state), tool="prc_resolve_r6.py")
            emit_table(args.out_stem, name, lin, att if base_table_name(name) == "gfisla" else None,
                       table, wrapper, meta["ladder"], dict(meta["calib_record"], table=name), extra)
            print(f"[resolve] {name}: target x{rec['target_scale']:.4f}  calib cost/target "
                  f"{rec['calib_cost_over_target']:.6f}")
    return 3 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
