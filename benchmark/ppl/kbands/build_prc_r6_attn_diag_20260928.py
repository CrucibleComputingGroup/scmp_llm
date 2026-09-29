"""Build the frozen round-6 attention-diagnostic arms and manifest (CPU only).

Writes, per cell, INC/A/ATT128[/LIN128] wrappers + tables under
  /nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r6_20260928/attn_diag_arms/<cell>/
(INC table = byte copy of the best(all) incumbent; the others = validated table
edits) and the manifest benchmark/ppl/kbands/prc_r6_attn_diag_20260928.json with
source/input hashes, fixed windows, identity references, per-bucket qualification
statistics, best(all) chords, and the pre-registered Round-7 gate.

Idempotent: existing arm files must be byte-identical to what would be written.
The manifest is written exclusively unless --overwrite-manifest is given (only
before submission). No GPU, no model load, no Slurm.

  /nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python \
      benchmark/ppl/kbands/build_prc_r6_attn_diag_20260928.py [--overwrite-manifest]
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.ppl import prc_r6_attn_diag_arms as R  # noqa: E402

SCMP = REPO.parent
RUN_ID = "prc_r6_attn_diag_20260928"
K = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands")
OUTPUT_ROOT = K / "prc_r6_20260928"
ARMS_ROOT = OUTPUT_ROOT / "attn_diag_arms"
FULLTEST = K / "kbands_20260801/ppl"
BEST_ALL = SCMP / "hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.json"
MP_BEST = SCMP / "hpca_results/llm/ppl/mp_best"
MANIFEST = REPO / "benchmark/ppl/kbands/prc_r6_attn_diag_20260928.json"
LAUNCHER = REPO / "benchmark/ppl/kbands/run_prc_r6_attn_diag_20260928.sbatch"
MODEL_PATHS = {"30B": "Qwen/Qwen3-30B-A3B-Instruct-2507", "4B": "Qwen/Qwen3-4B-Instruct-2507"}

QK_ALL = ["qk:t0:l0", "qk:t0:l1", "qk:t0:l2", "qk:t0:l3"]
CELLS = [
    dict(id="30B_t40", model="30B", target=40, total_blocks=48,
         incumbent_wrapper=K / "prc_local_20260926/30B_t40/candidate07_mlp_from_projections.json",
         A=QK_ALL + ["av:t0:l3"], arms=["A", "ATT128", "LIN128"],
         windows_file=K / "prc_adjacent_20260927/30B_t40/windows.json",
         identity=dict(type="reference_run", dir=K / "prc_adjacent_20260927/30B_t40")),
    dict(id="30B_t32", model="30B", target=32, total_blocks=48,
         incumbent_wrapper=K / "prc2/30B_t32_c7_gfisla.json",
         A=QK_ALL + ["av:t0:l3"], arms=["A", "ATT128"],
         windows_file=K / "prc_adjacent_20260927/30B_t32/windows.json",
         identity=dict(type="reference_run", dir=K / "prc_adjacent_20260927/30B_t32")),
    dict(id="4B_t40", model="4B", target=40, total_blocks=36,
         incumbent_wrapper=K / "prc2/4B_t40_c7_gfisla.json",
         A=QK_ALL + ["av:t0:l1", "av:t0:l2", "av:t0:l3"], arms=["A", "ATT128"],
         windows_file=K / "prc_adjacent_20260927/4B_t32/windows.json",
         identity=dict(type="probe", file=K / "prc2/4B_t40_c7_heldout_nll.json", table="gfisla",
                       start=30720)),
    dict(id="4B_t64", model="4B", target=64, total_blocks=36,
         incumbent_wrapper=K / "prc2/4B_t64_c7_gfisla.json",
         A=QK_ALL + ["av:t0:l1", "av:t0:l2", "av:t0:l3"], arms=["A", "ATT128"],
         windows_file=K / "prc_adjacent_20260927/4B_t32/windows.json",
         identity=dict(type="probe", file=K / "prc2/4B_t64_c7_heldout_nll.json", table="gfisla",
                       start=30720)),
]
TASKS = [{"index": 0, "id": "30B_t40", "cells": ["30B_t40"]},
         {"index": 1, "id": "30B_t32", "cells": ["30B_t32"]},
         {"index": 2, "id": "4B_dense", "cells": ["4B_t40", "4B_t64"]}]
FROZEN_IMPORTED = [
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
NEW_SOURCES = [
    "benchmark/ppl/prc_r6_attn_diag.py",
    "benchmark/ppl/prc_r6_attn_diag_arms.py",
    "benchmark/ppl/test_prc_r6_attn_diag.py",
    "benchmark/ppl/kbands/build_prc_r6_attn_diag_20260928.py",
    "benchmark/ppl/kbands/run_prc_r6_attn_diag_20260928.sbatch",
]


def write_exact(path: Path, data: bytes) -> str:
    """Create ``path`` with ``data``; an existing file must already hold these bytes."""
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"refusing to replace a different existing arm file: {path}")
        return "unchanged"
    with path.open("xb") as f:
        f.write(data)
    return "created"


def best_rows():
    data = json.loads(BEST_ALL.read_text())
    return data, {(r["model"], int(r["target"])): r for r in data["rows"]}


def qualification(agg: "R.TraceAgg", table: dict, listed: list) -> dict:
    """Per-runtime-bucket top-rung / escape MAC shares and first-order +cycles."""
    g = R.global_ladder(table)
    out, total_cyc = {}, agg.total_cycle_macs
    for (kind, op, q), hist in sorted(agg.bucket_hist.items()):
        if kind != "attention":
            continue
        tot = sum(hist.values())
        out[f"{op}:t0:l{q}"] = {
            "macs": tot, "share_of_sc_macs": tot / agg.total_macs,
            "top_share": hist.get(g[0], 0) / tot, "at_128_share": hist.get(R.CAP, 0) / tot,
            "first_order_extra_cycles_pct_top_to_128": 100.0 * hist.get(g[0], 0) * (R.CAP - g[0]) / total_cyc,
            "first_order_extra_cycles_pct_all_to_128": 100.0 * sum(m * (R.CAP - L) for L, m in hist.items()) / total_cyc,
            "listed_in_A": f"{op}:t0:l{q}" in listed}
    literal = sorted(k for k, v in out.items() if v["top_share"] >= 0.85)
    lin_cyc = sum(L * m for (kind, _op, _q), h in agg.bucket_hist.items() if kind == "linear" for L, m in h.items())
    lin_mac = sum(m for (kind, _op, _q), h in agg.bucket_hist.items() if kind == "linear" for m in h.values())
    return {
        "source": "incumbent full-test trace, runtime layer buckets _bucket_index(block, total_blocks, 4)",
        "top_rung": g[0], "buckets": out,
        "A_listed": listed,
        "A_first_order_extra_cycles_pct": sum(out[k]["first_order_extra_cycles_pct_top_to_128"] for k in listed),
        "literal_ge85pct_rule_buckets": literal,
        "literal_ge85pct_rule_first_order_extra_cycles_pct":
            sum(out[k]["first_order_extra_cycles_pct_top_to_128"] for k in literal),
        "ATT128_first_order_extra_cycles_pct":
            sum(v["first_order_extra_cycles_pct_all_to_128"] for v in out.values()),
        "LIN128_first_order_extra_cycles_pct": 100.0 * (R.CAP * lin_mac - lin_cyc) / total_cyc,
        "choice": ("A = enumerated per-bucket list from investigation RC3/section 6 Q1 (the scout's A_inv); "
                   "the literal >=85% rule list is recorded for reference and is not run."),
    }


def build_cell(spec: dict, rows: dict, chords: dict) -> dict:
    cid, model, target = spec["id"], spec["model"], spec["target"]
    row = rows[(model, target)]
    if Path(row["wrapper"]) != spec["incumbent_wrapper"]:
        raise SystemExit(f"{cid}: best(all) wrapper {row['wrapper']} != pinned {spec['incumbent_wrapper']}")
    wrapper = json.loads(spec["incumbent_wrapper"].read_text())
    table_path = Path(wrapper["threshold_table_path"])
    base = json.loads(table_path.read_text())
    R.check_incumbent_shape(base)
    if base["model_path"] != MODEL_PATHS[model]:
        raise SystemExit(f"{cid}: model mismatch")
    if wrapper.get("escape_gate_k") != 2.0 or int(wrapper.get("escape_stoc_len", 128)) != 128:
        raise SystemExit(f"{cid}: expected the mu+2tau escape gate at 128")
    if [int(v) for v in wrapper["stoc_len_levels"]] != R.global_ladder(base):
        raise SystemExit(f"{cid}: wrapper/table ladder mismatch")
    hybrid = MP_BEST / f"configs/{model}/target{target}/hybrid_config.json"
    awq = MP_BEST / f"act_scales/awq_scales/awq_scales_{MODEL_PATHS[model].replace('/', '_')}_b4.pt"
    trace = Path(row["trace"])
    agg = R.TraceAgg.load(trace, spec["total_blocks"], int(base["layer_buckets"]))
    if agg.header.get("total_blocks") != spec["total_blocks"] or agg.header.get("mp_config_json") != str(spec["incumbent_wrapper"]):
        raise SystemExit(f"{cid}: full-test trace header does not match the incumbent")
    if abs(agg.cost - row["best_cost"]) > 1e-9 or agg.header.get("ppl") != row["best_ppl"]:
        raise SystemExit(f"{cid}: full-test trace does not reproduce best(all) cost/PPL")
    basics = R.audit_trace_basics(agg, "incumbent full-test", json.loads(hybrid.read_text()))
    basics += R.audit_lengths(agg, "INC", {}, base, 128)
    if basics:
        raise SystemExit(f"{cid}: incumbent full-test trace audit failed: {basics}")

    # --- arms ---------------------------------------------------------------
    cell_dir = ARMS_ROOT / cid
    cell_dir.mkdir(parents=True, exist_ok=True)
    arms, statuses, sweeps = {}, {}, {}
    inc_table = cell_dir / "INC_table.json"
    if inc_table.exists():
        if inc_table.read_bytes() != table_path.read_bytes():
            raise SystemExit(f"{inc_table} differs from the incumbent table")
        statuses["INC_table"] = "unchanged"
    else:
        shutil.copyfile(table_path, inc_table)
        statuses["INC_table"] = "created"
    provenance = {"source_table": str(table_path), "source_table_sha256": R.sha256_file(table_path),
                  "source_wrapper": str(spec["incumbent_wrapper"]), "cell": cid}
    specs = {"A": {"buckets": list(spec["A"])}, "ATT128": {}, "LIN128": {"protected": True}}
    for arm in ["INC"] + spec["arms"]:
        tpath = cell_dir / f"{arm}_table.json"
        if arm != "INC":
            table = R.build_arm_table(base, arm, specs[arm], provenance)
            statuses[f"{arm}_table"] = write_exact(tpath, (json.dumps(table, indent=1) + "\n").encode())
            R.validate_arm_table(base, json.loads(tpath.read_text()), arm, specs[arm])
        w = dict(wrapper)
        w["threshold_table_path"] = str(tpath)
        wpath = cell_dir / f"{arm}.json"
        statuses[arm] = write_exact(wpath, (json.dumps(w, indent=2, sort_keys=True) + "\n").encode())
        arms[arm] = {"wrapper": str(wpath), "table": str(tpath),
                     "spec": None if arm == "INC" else specs[arm],
                     "attribution_only": arm in R.ATTRIBUTION_ONLY_ARMS,
                     "best_all_candidate": False}
        if arm != "INC":
            sweeps[arm] = R.resolution_sweep(w, tpath, arm, specs[arm], base, inc_table, spec["total_blocks"])

    # --- windows and identity -----------------------------------------------
    wins = json.loads(spec["windows_file"].read_text())
    starts = [int(s) for s in wins["starts"]["confirm"]]
    if wins["split"] != "train" or wins["ctx"] != 2048 or len(starts) != 16:
        raise SystemExit(f"{cid}: unexpected windows file")
    ident = spec["identity"]
    hashes_extra = [spec["windows_file"]]
    if ident["type"] == "reference_run":
        d = ident["dir"]
        nll = json.loads((d / "confirm_incumbent_nll.json").read_text())
        ref_trace = d / "confirm_incumbent_trace.json"
        ref_agg = R.TraceAgg.load(ref_trace, spec["total_blocks"], int(base["layer_buckets"]))
        if ref_agg.header.get("windows") != starts:
            raise SystemExit(f"{cid}: reference trace windows differ from the diagnostic windows")
        snap = json.loads((d / "incumbent_table.json").read_text())
        if R.canonical(snap) != R.canonical(base):
            raise SystemExit(f"{cid}: round-4 incumbent snapshot is not the same table content")
        if (ref_agg.total_macs, ref_agg.total_cycle_macs) != (int(nll["total_macs"]), int(nll["total_cycle_macs"])):
            raise SystemExit(f"{cid}: reference NLL file and trace disagree")
        identity = {"type": "reference_run", "nll_file": str(d / "confirm_incumbent_nll.json"),
                    "trace_file": str(ref_trace), "window_nll": nll["window_nll"],
                    "mean_nll": nll["mean_nll"], "total_macs": ref_agg.total_macs,
                    "total_cycle_macs": ref_agg.total_cycle_macs, "cost": ref_agg.cost,
                    "round4_incumbent_snapshot": str(d / "incumbent_table.json"),
                    "note": "round-4 confirm_incumbent on the same 16 windows; its table snapshot is "
                            "content-identical to the incumbent (byte-identical for 30B t40)"}
        hashes_extra += [d / "confirm_incumbent_nll.json", ref_trace, d / "incumbent_table.json"]
        sim_standin = ref_trace
        del ref_agg
    else:
        held = json.loads(ident["file"].read_text())
        entry = held["tables"][ident["table"]]
        if held["windows"][0] != ident["start"] or Path(entry["wrapper"]) != spec["incumbent_wrapper"]:
            raise SystemExit(f"{cid}: held-out probe file does not match the incumbent")
        if any(abs(s - x) < 2048 for s in starts for x in held["windows"]):
            raise SystemExit(f"{cid}: diagnostic windows overlap the held-out windows")
        identity = {"type": "probe", "file": str(ident["file"]), "table_name": ident["table"],
                    "start": ident["start"], "nll": entry["window_nll"][0],
                    "note": "prc2 c7 held-out TRAIN window 1 (heldout_nll.py, same build/swap/loss path); "
                            "no prior NLL exists for this cell on the diagnostic windows"}
        hashes_extra += [ident["file"]]
        sim_standin = trace

    chord = R.binding_chord(chords, model, target)
    hashes = {str(p): R.sha256_file(p) for p in
              [spec["incumbent_wrapper"], table_path, hybrid, awq, trace] + hashes_extra}
    for arm, entry in arms.items():
        hashes[entry["wrapper"]] = R.sha256_file(entry["wrapper"])
        hashes[entry["table"]] = R.sha256_file(entry["table"])
    return {
        "id": cid, "model": model, "target": target, "nominal_cycles": 2 * target,
        "model_path": MODEL_PATHS[model], "total_blocks": spec["total_blocks"],
        "layer_buckets": int(base["layer_buckets"]),
        "incumbent": {"arm": row["winning_arm"], "wrapper": str(spec["incumbent_wrapper"]),
                      "table": str(table_path), "full_test_ppl": row["best_ppl"],
                      "full_test_cost": row["best_cost"], "full_test_trace": str(trace)},
        "hybrid_config": str(hybrid), "awq_scales": str(awq),
        "escape_gate_k": wrapper["escape_gate_k"], "escape_stoc_len": int(wrapper["escape_stoc_len"]),
        "global_ladder": R.global_ladder(base), "protected_stoc_len": int(base["protected_channels"]["stoc_len"]),
        "arms": arms, "arm_order": list(spec["arms"]),
        "windows": {"source": str(spec["windows_file"]), "field": "starts.confirm", "starts": starts,
                    "token_ids_sha256": wins["token_ids_sha256"], "data_seed": wins["data_seed"],
                    "split": "train", "ctx": 2048,
                    "note": "used once before (round-4 confirmation, failed); never used for incumbent selection"},
        "identity": identity,
        "qualification": qualification(agg, base, spec["A"]),
        "resolution_sweep": sweeps,
        "chord": chord,
        "gate_gross_dnll_threshold": R.GATE_GROSS_DNLL[model],
        "simulation": {"incumbent_trace_standin": str(sim_standin),
                       "note": "CPU dry-run only (--simulate); never a measurement"},
        "hashes": hashes,
        "build_status": statuses,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite-manifest", action="store_true")
    args = ap.parse_args()
    if MANIFEST.exists() and not args.overwrite_manifest:
        raise SystemExit(f"{MANIFEST} exists; pass --overwrite-manifest (pre-submission only)")
    data, rows = best_rows()
    trace_costs = {}
    for (model, target), row in sorted(rows.items()):
        if model not in MODEL_PATHS:
            continue
        payload = json.loads(Path(row["trace"]).read_text())
        m = sum(int(g["macs"]) for g in payload["groups"])
        c = sum(int(g["macs"]) * int(g["stoc_len"]) for g in payload["groups"])
        trace_costs[(model, target)] = c / m
        if abs(c / m - row["best_cost"]) > 1e-9:
            raise SystemExit(f"{model} t{target}: trace cost {c / m} != best_cost {row['best_cost']}")
        del payload
    chords = R.chord_table([r for k, r in rows.items() if k[0] in MODEL_PATHS], trace_costs)
    cells = {}
    for spec in CELLS:
        cells[spec["id"]] = build_cell(spec, rows, chords)
        c = cells[spec["id"]]
        print(f"[r6build] {spec['id']}: arms {c['arm_order']} status {c['build_status']} "
              f"chord(up) {c['chord']['chord']:.5f} A first-order +{c['qualification']['A_first_order_extra_cycles_pct']:.2f}%",
              flush=True)
    sources = [str(REPO / p) for p in NEW_SOURCES + FROZEN_IMPORTED]
    missing = [p for p in sources if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"missing source files: {missing}")
    rules = dict(R.PREREGISTERED_RULES)
    rules["per_cell"] = {
        cid: {"model": c["model"], "binding_chord": c["chord"]["chord"],
              "binding_chord_interval": f"t{c['chord']['upward']['from_target']}->t{c['chord']['upward']['to_target']}",
              "downward_chord_nonbinding": (c["chord"]["downward_sensitivity"] or {}).get("chord_pct_ppl_per_1pct_cycles"),
              "gross_dnll_threshold": c["gate_gross_dnll_threshold"],
              "marginal_fraction": R.GATE_MARGINAL_FRACTION, "se_multiplier": R.GATE_SE_MULTIPLIER,
              "A_first_order_extra_cycles_pct": c["qualification"]["A_first_order_extra_cycles_pct"],
              "implied_budget_term_at_first_order_pct":
                  -R.GATE_MARGINAL_FRACTION * c["chord"]["chord"] * c["qualification"]["A_first_order_extra_cycles_pct"]}
        for cid, c in cells.items()}
    manifest = {
        "schema": "prc-r6-attn-diag-v1", "run_id": RUN_ID,
        "purpose": "Round-6 attention diagnostic (investigation section 6 Q1 + Q9): does raising the "
                   "top-bound attention buckets to the 128 cap pay against the budget curve, and how "
                   "much of the residual is attention vs linears/routing. Gates the Round-7 "
                   "attention-ladder joint solve. Pure allocation (table-only) arms.",
        "units": "stream lengths are HALVED code units (nominal = 2x); code cap 128; costs = "
                 "MAC-weighted mean halved length over all SC trace records",
        "max_total_gpus": 4, "array_gpu_cap": 2, "gpus_per_task": 1,
        "concurrency_note": "This array runs at most 2 GPUs (throttle %2); another round-6 array of at most "
                            "2 GPUs may run concurrently, so the total stays <= 4.",
        "protocol": {"frontend": "awq", "owen_mode": "bitrev", "scramble_masks": 64, "hw_max_masks": 64,
                     "ctx": 2048, "stride": 2048, "split": "train", "window_count": 16,
                     "sc_prec": 8, "sc_halve": True, "qk_rebalance": False, "rng_grid": "fixed128",
                     "awq_obj_bits": 4, "sq_alpha": 0.5, "full_test": False,
                     "profile": "observe-only ProfileCollector on window 1 of every evaluated arm",
                     "escape": "mu+2tau gate (escape_gate_k 2.0, escape_stoc_len 128) unchanged in every arm"},
        "scope": "Table edits only: per-bucket attention ladders within existing runtime support "
                 "(bucket stoc_len_levels), attention thresholds (ATT128 only), PRC ladders and the "
                 "protected-slice length (LIN128 only). Frozen: INT mask, AWQ scales, RNG (bitrev/64, "
                 "128-level grid), QK operands (no rebalance/smooth-scale), escape semantics, weights, "
                 "runtime code. No INT rung, no asymmetric SC, no lengths above 128.",
        "output_root": str(OUTPUT_ROOT), "arms_root": str(ARMS_ROOT),
        "log_dir": "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands",
        "tasks": TASKS, "cells": cells,
        "chords": chords,
        "chord_source": {"best_all": str(BEST_ALL), "best_all_sha256": R.sha256_file(BEST_ALL),
                         "trace_costs": {f"{m}:t{t}": v for (m, t), v in sorted(trace_costs.items())},
                         "definition": "chord = [100*(PPL_t - PPL_next)/PPL_t] / [100*(C_next/C_t - 1)], "
                                       "best(all) full-test PPL and exact full-test trace cost"},
        "preregistered_rules": rules,
        "baseline_results": str(BEST_ALL), "baseline_results_sha256": R.sha256_file(BEST_ALL),
        "source_files": sources,
        "code_hashes": {p: R.sha256_file(p) for p in sources},
        "frozen_imported_note": "prc_local_refine/prc_local_proposals and the runtime files are imported "
                                "unmodified; round-4 prc_adjacent_* and calib7 are not used or edited.",
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
    print(f"[r6build] wrote {MANIFEST} ({len(cells)} cells, {len(sources)} hashed sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
