"""Build the frozen round-7 STEP-0 manifest and the round-7 window registry (CPU only).

Writes
  benchmark/ppl/kbands/prc_windows_r7_20260928.json   window registry (exclusion set, prefix guard,
                                                       16 fresh t48 screen + 32 confirmation windows)
  benchmark/ppl/kbands/prc_step0_r7_20260928.json     step-0 manifest (inputs, hashes, identity refs,
                                                       pre-registered kappa rule)
No arm tables are created: step 0 evaluates the EXISTING prc2 c17 / c17_s80 tables (30B t32 uses the
c17e32 pair, the c17 variant the c7 diag scored and the test measured) plus INC and the parent.
Tasks: 0 30B_t40 and 1 30B_t32 ("kappa" cells: c17/s80 + INC16 profile + PARENT16), 2 30B_t48
("reference" cell: INC16 profile + PARENT16 on the fresh t48 screen windows, the fixed-point input),
3 4B_t64 (optional "reference" cell on the round-6 4B diag windows; submit only when the user enables
4B t64).
No GPU, no model load, no Slurm. The registry is written exclusively (an existing file must be
byte-identical); the manifest exclusively unless --overwrite-manifest (pre-submission only).

  /nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python \
      benchmark/ppl/kbands/build_prc_step0_r7_20260928.py [--overwrite-manifest] [--skip-sweep]
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.ppl import prc_eval_r7_20260928 as E  # noqa: E402
from benchmark.ppl import prc_r6_attn_diag_arms as R  # noqa: E402

RUN_ID = "prc_step0_r7_20260928"
KDIR = REPO / "benchmark/ppl/kbands"
MANIFEST = KDIR / "prc_step0_r7_20260928.json"
REGISTRY = KDIR / "prc_windows_r7_20260928.json"
OUTPUT_ROOT = E.R7_ROOT / "step0"
K = E.KB
NEW_SOURCES = [
    "benchmark/ppl/prc_eval_r7_20260928.py",
    "benchmark/ppl/prc_step0_r7_20260928.py",
    "benchmark/ppl/test_prc_eval_r7_20260928.py",
    "benchmark/ppl/kbands/build_prc_step0_r7_20260928.py",
    "benchmark/ppl/kbands/run_prc_step0_r7_20260928.sbatch",
]
FROZEN_IMPORTED = [
    "benchmark/ppl/prc_r6_attn_diag_arms.py",
    "benchmark/ppl/prc_r6_retarget_20260928.py",
    "benchmark/ppl/prc_local_refine.py",
    "benchmark/ppl/prc_local_proposals.py",
    "model/sc_common.py",
    "kernels/scmp_kernels/mp/config.py",
    "kernels/scmp_kernels/sc/matmul.py",
    "kernels/scmp_kernels/sc/kernels.py",
    "kernels/scmp_kernels/trace.py",
    "loader.py",
    "model/awq_apply.py",
    "benchmark/quant/eval_quant.py",
]
CELLS = [
    dict(id="30B_t40", model="30B", target=40, kind="kappa", c17="30B_t40_c17", s80="30B_t40_c17_s80",
         field="r6_diag_windows_30B"),
    dict(id="30B_t32", model="30B", target=32, kind="kappa", c17="30B_t32_c17e32", s80="30B_t32_c17e32_s80",
         field="r6_diag_windows_30B"),
    dict(id="30B_t48", model="30B", target=48, kind="reference", field="screen16_fresh_t48"),
    dict(id="4B_t64", model="4B", target=64, kind="reference", field="r6_diag_windows_4B"),
]
TASKS = [{"index": 0, "id": "30B_t40", "cells": ["30B_t40"]},
         {"index": 1, "id": "30B_t32", "cells": ["30B_t32"]},
         {"index": 2, "id": "30B_t48", "cells": ["30B_t48"]},
         {"index": 3, "id": "4B_t64", "cells": ["4B_t64"]}]
DEFAULT_ARRAY = "0-2%1"
OPTIONAL_TASKS = {3: "4B t64 reference: submit only with the user's explicit OK to enable 4B t64 (round-6 gate "
                     "margin 0.02 pp; the calib-side capture manifest must also be rebuilt with --enable-4b)"}
EVALS = {"kappa": {"pair": True, "parent16": True, "inc16_profiled": True},
         "reference": {"pair": False, "parent16": True, "inc16_profiled": True}}


def write_registry() -> tuple[dict, str]:
    reg = E.derive_round7_windows()
    data = (json.dumps(reg, indent=1) + "\n").encode()
    if REGISTRY.exists():
        if REGISTRY.read_bytes() != data:
            raise SystemExit(f"{REGISTRY} exists with different content; the window draw is frozen")
    else:
        with REGISTRY.open("xb") as f:
            f.write(data)
    return reg, E.sha256_file(REGISTRY)


def _trace_reference(nll_file, trace_file, starts, nb, source) -> dict:
    """A stored INC run on exactly these windows: NLLs + summary trace (identity reference)."""
    nll = E.read_json(nll_file)
    agg = R.TraceAgg(E.load_trace(trace_file), nb, 4)
    if agg.header.get("windows") != starts:
        raise SystemExit(f"{trace_file}: trace windows differ from the step-0 windows")
    if nll.get("window_starts", starts) != starts or len(nll["window_nll"]) != 16:
        raise SystemExit(f"{nll_file}: NLL windows differ from the step-0 windows")
    if (agg.total_macs, agg.total_cycle_macs) != (int(nll["total_macs"]), int(nll["total_cycle_macs"])):
        raise SystemExit(f"{nll_file}: NLL file and trace disagree")
    return {"nll_file": str(nll_file), "trace_file": str(trace_file), "window_nll": nll["window_nll"],
            "total_macs": agg.total_macs, "total_cycle_macs": agg.total_cycle_macs, "cost": agg.cost,
            "source": source}


def build_cell(spec, reg, rows, m6, sweep=True) -> dict:
    cid, model, T, kind = spec["id"], spec["model"], spec["target"], spec["kind"]
    nb = E.TOTAL_BLOCKS[model]
    row = rows[(model, T)]
    inc_wrapper = Path(row["wrapper"])
    inc_table = E.resolve_table_path(inc_wrapper)
    r6c = m6["cells"].get(cid)
    if r6c is not None and Path(r6c["incumbent"]["wrapper"]) != inc_wrapper:
        raise SystemExit(f"{cid}: best(all) incumbent differs from round 6's")
    parent_dir = E.MP_BEST / f"configs/{model}/target{T}"
    arms = {
        "INC": {"wrapper": str(inc_wrapper), "table": str(inc_table), "label": f"best(all) {row['winning_arm']}"},
        "PARENT": {"wrapper": str(parent_dir / "wrapper.json"), "table": str(parent_dir / "table.json"),
                   "label": "parent (mp_best, AWQ + h20 mask)"},
    }
    if kind == "kappa":
        arms["C17"] = {"wrapper": str(K / f"prc2/{spec['c17']}.json"), "table": str(K / f"prc2/{spec['c17']}_table.json"),
                       "label": spec["c17"], "capture_score_name": "c17"}
        arms["S80"] = {"wrapper": str(K / f"prc2/{spec['s80']}.json"), "table": str(K / f"prc2/{spec['s80']}_table.json"),
                       "label": spec["s80"], "capture_score_name": "c17_s80"}
    inc_w = E.read_json(inc_wrapper)
    for name, a in arms.items():
        if E.resolve_table_path(a["wrapper"]).resolve() != Path(a["table"]).resolve():
            raise SystemExit(f"{cid} {name}: wrapper/table mismatch")
        E.check_wrapper_matches_inc(E.read_json(a["wrapper"]), inc_w, name)
        if E.read_json(a["table"])["model_path"] != E.MODEL_PATHS[model]:
            raise SystemExit(f"{cid} {name}: model mismatch")
    inc_t = E.read_json(inc_table)
    lineage = {}
    if kind == "kappa":
        lineage = {n: E.validate_lineage(inc_t, E.read_json(arms[n]["table"]), total_blocks=nb) for n in ("C17", "S80")}
        lineage["C17_vs_S80"] = E.validate_lineage(E.read_json(arms["C17"]["table"]), E.read_json(arms["S80"]["table"]),
                                                   total_blocks=nb)
        for n, rep in lineage.items():
            if not rep["ok"]:
                raise SystemExit(f"{cid} {n}: lineage failed {rep['failures'][:5]}")
        pair = lineage["C17_vs_S80"]
        if pair["attention_threshold_keys_changed"] or pair["attention_ladders_changed"] or \
                not pair["prc_threshold_keys_changed"]:
            raise SystemExit(f"{cid}: c17 vs s80 is not a pure PRC-threshold (linear budget) shift")
    starts = [int(s) for s in reg[spec["field"]]]
    files = []
    r6_runtime = None
    if kind == "kappa":
        ident = r6c["identity"]
        if ident["type"] != "reference_run" or [int(s) for s in r6c["windows"]["starts"]] != starts:
            raise SystemExit(f"{cid}: expected a round-4 reference run on the round-6 diag windows")
        nll = E.read_json(ident["nll_file"])
        if nll["window_nll"] != ident["window_nll"]:
            raise SystemExit(f"{cid}: round-4 reference NLLs changed")
        reference = _trace_reference(ident["nll_file"], ident["trace_file"], starts, nb,
                                     "round-4 confirm_incumbent")
        r6_runtime = {"inc_nll_file": str(E.R6_DIAG_ROOT / cid / "INC_nll.json"),
                      "note": "read at run time if present; must equal the stored round-4 reference"}
    elif cid == "4B_t64":
        d = E.R6_DIAG_ROOT / cid
        summ = E.read_json(d / "diag_summary.json")
        if (d / "failure.json").exists() or summ.get("complete") is not True or summ.get("simulated") is not False \
                or summ["identity"].get("restored_incumbent_exact") is not True:
            raise SystemExit(f"{cid}: the round-6 diagnostic is not a complete GPU run")
        if [int(s) for s in r6c["windows"]["starts"]] != starts:
            raise SystemExit(f"{cid}: round-6 diag windows differ from the registry")
        reference = _trace_reference(d / "INC_nll.json", d / "INC_trace.json", starts, nb,
                                     "round-6 INC run (prc_r6_20260928/4B_t64)")
        files.append(str(d / "diag_summary.json"))
    else:
        reference = None      # 30B t48: fresh windows, no stored INC value; INC16 becomes the reference
    if reference is not None:
        files += [reference["nll_file"], reference["trace_file"]]
    held = K / f"prc2/{cid}_c7_heldout_nll.json"
    hd = E.read_json(held)
    if hd["windows"][0] != 30720:
        raise SystemExit(f"{cid}: held-out file window 1 is not 30720")
    probe_arms = (("PARENT", "parent"), ("C17", "c17")) if kind == "kappa" else (("INC", None), ("PARENT", "parent"))
    probes = []
    for arm, tname in probe_arms:
        if tname is None:
            tname = next((k for k, v in hd["tables"].items()
                          if Path(v["wrapper"]).resolve() == Path(arms[arm]["wrapper"]).resolve()), None)
            if tname is None:
                raise SystemExit(f"{cid}: INC is not in the held-out file")
        ent = hd["tables"][tname]
        if Path(ent["wrapper"]).resolve() != Path(arms[arm]["wrapper"]).resolve():
            raise SystemExit(f"{cid}: held-out {tname} wrapper {ent['wrapper']} != {arms[arm]['wrapper']}")
        probes.append({"name": f"probe_{arm}_30720", "arm": arm, "start": 30720, "expected": ent["window_nll"][0],
                       "source": f"{held}:tables.{tname}.window_nll[0]"})
    if E.overlaps([30720], starts):
        raise SystemExit("probe overlaps the step-0 windows")
    hybrid = parent_dir / "hybrid_config.json"
    if r6c is not None and Path(r6c["hybrid_config"]) != hybrid:
        raise SystemExit(f"{cid}: hybrid config differs from round 6")
    awq = E.MP_BEST / f"act_scales/awq_scales/awq_scales_{E.MODEL_PATHS[model].replace('/', '_')}_b4.pt"
    if r6c is not None and Path(r6c["awq_scales"]) != awq:
        raise SystemExit(f"{cid}: AWQ cache differs from round 6")
    sweeps = {}
    if sweep:
        for n in [x for x in ("INC", "C17", "S80") if x in arms]:
            sweeps[n] = E.resolver_sweep(arms[n]["wrapper"], arms["INC"]["wrapper"], nb)["checked"]
    files += [arms[n][k] for n in arms for k in ("wrapper", "table")] + \
        [str(hybrid), str(awq), str(held), row["trace"]]
    if reference is not None:
        sim_standin = reference["trace_file"]
    else:
        sim_standin = row["trace"]   # dry run only: INC full-test trace (reduced for 30B by the scratch helper)
    return {
        "id": cid, "kind": kind, "model": model, "target": T, "nominal_cycles": 2 * T,
        "model_path": E.MODEL_PATHS[model], "total_blocks": nb, "layer_buckets": 4, "hybrid_config": str(hybrid),
        "awq_cache": str(awq),
        "incumbent": {"arm": row["winning_arm"], "wrapper": str(inc_wrapper), "table": str(inc_table),
                      "full_test_ppl": row["best_ppl"], "full_test_cost": row["best_cost"],
                      "full_test_trace": row["trace"]},
        "arms": arms,
        "windows": {"starts": starts, "token_ids_sha256": E.QWEN_TOKEN_SHA, "split": "train", "ctx": E.CTX,
                    "source": str(REGISTRY), "field": spec["field"],
                    "note": {"r6_diag_windows_30B": "the round-6 30B diagnostic windows = round-4 30B confirmation "
                                                    "windows = the 30B t32/t40 screen windows",
                             "screen16_fresh_t48": "the 16 fresh registry windows = the 30B t48 screen windows",
                             "r6_diag_windows_4B": "the round-6 4B diagnostic windows = the 4B screen windows"
                             }[spec["field"]]},
        "identity": {"reference": reference, "r6": r6_runtime},
        "probes": probes,
        "evals": dict(EVALS[kind]),
        "lineage": lineage, "resolver_sweep": sweeps,
        "hashes": {f: E.sha256_file(f) for f in dict.fromkeys(files)},
        "simulation": {"inc_standin": sim_standin,
                       "note": "CPU dry-run only (simulate); other arms synthesized onto their own ladders"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite-manifest", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true", help="skip the CPU resolver sweep (faster rebuilds)")
    args = ap.parse_args()
    if MANIFEST.exists() and not args.overwrite_manifest:
        raise SystemExit(f"{MANIFEST} exists; pass --overwrite-manifest (pre-submission only)")
    reg, reg_sha = write_registry()
    best = E.read_json(E.BEST_ALL)
    rows = {(r["model"], int(r["target"])): r for r in best["rows"]}
    m6 = E.read_json(E.R6_DIAG_MANIFEST)
    cells = {}
    for spec in CELLS:
        cells[spec["id"]] = build_cell(spec, reg, rows, m6, sweep=not args.skip_sweep)
        print(f"[r7s0build] {spec['id']} ({spec['kind']}): arms={sorted(cells[spec['id']]['arms'])} "
              f"reference={'yes' if cells[spec['id']]['identity']['reference'] else 'no'} probes="
              f"{[p['expected'] for p in cells[spec['id']]['probes']]}", flush=True)
    sources = [str(REPO / p) for p in NEW_SOURCES + FROZEN_IMPORTED]
    missing = [p for p in sources if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"missing source files: {missing}")
    manifest = {
        "schema": "prc-step0-r7-v1", "run_id": RUN_ID,
        "purpose": "Round-7 step 0 (CRIT L3): measure the budget-shift linear marginal on 30B t32/t40 by "
                   "evaluating the existing c17 / c17_s80 tables (a pure -20% linear-budget shift) on the "
                   "round-6 diagnostic windows, identity-checked against INC; plus, for every round-7 screen "
                   "cell (30B t32/t40/t48; 4B t64 optional), the exact parent cost P and an INC 16-window "
                   "profile on that cell's screen windows: the screen cost gate's P and the input of the "
                   "pre-registered target-scale fixed point (prc_fixedpoint_r7_20260928.py).",
        "units": "stream lengths HALVED (nominal = 2x); code cap 128; cost = MAC-weighted mean halved length",
        "max_total_gpus": 4, "array_gpu_cap": 1, "gpus_per_task": 1,
        "default_array": DEFAULT_ARRAY, "optional_tasks": {str(k): v for k, v in OPTIONAL_TASKS.items()},
        "concurrency_note": "One GPU, tasks sequential (%1: 30B_t40, 30B_t32, 30B_t48 [, 4B_t64 only with the "
                            "user's OK: --array=0-3%1]). With 62208440 (1) and the capture array 0-1%2 (2) the "
                            "total is 4.",
        "protocol": dict(E.PROTOCOL, window_count=16, full_test=False,
                         profile="observe-only ProfileCollector: first window of C17/S80; all 16 windows of INC16"),
        "scope": "Evaluation only of frozen existing tables; no table is created or edited. Frozen: INT mask, "
                 "AWQ, RNG (bitrev/64, 128-level grid), QK operands, escape semantics, runtime code.",
        "output_root": str(OUTPUT_ROOT),
        "windows_registry": {"path": str(REGISTRY), "sha256": reg_sha},
        "log_dir": "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands",
        "tasks": TASKS, "cells": cells,
        "kappa_rule": E.KAPPA_RULE,
        "preregistered_rules": {
            "kappa": E.KAPPA_RULE["decision"],
            "kappa_authority": E.KAPPA_RULE["authority"],
            "identity_fail_closed": "INC window 1 must reproduce the stored reference (30B t32/t40: round-4 "
                                    "confirm_incumbent and round-6 INC_nll.json when present; 4B t64: the round-6 "
                                    "INC run) bit for bit before, between and after the arms; held-out probes on "
                                    "window 30720 (kappa cells: PARENT, C17; reference cells: INC, PARENT) must "
                                    "reproduce the prc2 c7 held-out NLLs; INC16 must reproduce all 16 stored NLLs "
                                    "and its trace record for record where a stored reference exists, and its "
                                    "window 1 must equal identity_before everywhere (30B t48: INC16 becomes the "
                                    "screen reference). Any mismatch or audit failure stops the cell "
                                    "(failure.json).",
            "profile_exactness": "profile vs trace totals and the table replay must agree EXACTLY (integer totals "
                                 "below 2**53), not within a relative tolerance.",
            "not_citable": "TRAIN-window measurements feeding the round-7 currency; never a best(all) candidate.",
        },
        "source_files": sources,
        "code_hashes": {p: E.sha256_file(p) for p in sources},
        "frozen_imported_note": "round-3/6 modules and runtime files are imported unmodified; nothing hashed by "
                                "the round-6 manifests is edited.",
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    payload = json.dumps(manifest, indent=1, allow_nan=False) + "\n"
    if args.overwrite_manifest:
        tmp = MANIFEST.with_suffix(".json.tmp")
        tmp.write_text(payload)
        tmp.replace(MANIFEST)
    else:
        with MANIFEST.open("x") as f:
            f.write(payload)
    print(f"[r7s0build] wrote {MANIFEST} ({len(cells)} cells, {len(sources)} hashed sources); registry {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
