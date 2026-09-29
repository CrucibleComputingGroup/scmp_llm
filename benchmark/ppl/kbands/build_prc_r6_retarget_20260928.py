"""Build + freeze the round-6 budget re-target manifest (prc_r6_retarget_20260928.json).

Login node, CPU only, no model. Re-run after ANY edit to a hashed source file (the launcher
refuses to start on a hash mismatch). Records, per cell: the incumbent's exact calib7 argument
record, frozen-input SHA-256s, fp16 and the 1.05x / 1.10x flip thresholds (from
hpca_results/llm/int/<model>.csv), the stored held-out child-cost ratios (the ~1.0134 figure
recomputed with its source), informational test-trace fixed points, and the pre-registered
rules (scale formula, gates, reporting).

  python benchmark/ppl/kbands/build_prc_r6_retarget_20260928.py [--no-test-trace-check]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCMP = REPO.parent
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TURBO = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands")
PRC2 = TURBO / "prc2"
PPL = TURBO / "kbands_20260801/ppl"
OUT_ROOT = TURBO / "prc_r6_retarget_20260928"
HP = SCMP / "hpca_results/llm"
MANIFEST = REPO / "benchmark/ppl/kbands/prc_r6_retarget_20260928.json"
KB_TBL_PREFIX = "r6ts20260928"

SOURCE_FILES = [
    "benchmark/ppl/mp_per_row_chunk_calib9_r6.py",
    "benchmark/ppl/prc_resolve_r6.py",
    "benchmark/ppl/prc_r6_retarget_20260928.py",
    "benchmark/ppl/test_prc_r6_retarget.py",
    "benchmark/ppl/kbands/run_prc_r6_retarget_20260928.sbatch",
    "benchmark/ppl/kbands/build_prc_r6_retarget_20260928.py",
    "benchmark/ppl/kbands/run_prc_ppl.sbatch",
    "benchmark/ppl/mp_per_row_chunk_calib7.py",
    "benchmark/ppl/mp_per_row_chunk_calib5.py",
    "benchmark/ppl/heldout_nll.py",
    "benchmark/ppl/calibrate_mp_thresholds.py",
    "benchmark/quant/eval_quant.py",
    "loader.py",
    "model/sc_common.py",
    "model/awq_apply.py",
    "model/smoothquant_apply.py",
    "kernels/scmp_kernels/mp/config.py",
    "kernels/scmp_kernels/mp/__init__.py",
    "kernels/scmp_kernels/sc/matmul.py",
    "kernels/scmp_kernels/sc/kernels.py",
    "kernels/scmp_kernels/sc/rng.py",
    "kernels/scmp_kernels/sc/sng.py",
    "kernels/scmp_kernels/sc/constants.py",
    "kernels/scmp_kernels/sc/config_helpers.py",
    "kernels/scmp_kernels/trace.py",
]

CELLS = [
    # id, model, target, arm (calib9 table), incumbent arm label, held-out table name
    ("14B_t32", "14B", 32, "gfisla", "c7gfisla", "gfisla"),
    ("30B_t64", "30B", 64, "gfis", "c7gfis", "gfis"),
]

# Expected effect (pre-registered; r6/scout_calib.md TL;DR-5 and investigation RC6).
PREDICTIONS = {
    "14B_t32": {"dppl_pct_range": [-0.26, -0.16],
                "basis": "+1.34% cycles x chord elasticity 0.19-0.21 (c17 9.1672@33.19 vs c17_s80 "
                         "9.4615@28.19), marginal ~0.6x chord (investigation RC6)",
                "needed_for_1p05x_pct": None},
    "30B_t64": {"dppl_pct_range": [-0.06, -0.017],
                "basis": "+0.63% cycles x the 4B t64 chord e=0.0445 (no 30B t64 elasticity data); "
                         "30B t64 is UNLIKELY to flip at exactly parent cost (scout TL;DR-5)",
                "needed_for_1p05x_pct": None},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fp16_of(model):
    with (HP / "int" / f"{model}.csv").open() as f:
        for row in csv.DictReader(f):
            if row["config"] == "fp16" and row["metric"] == "ppl":
                return float(row["value"])
    raise SystemExit(f"no fp16 row for {model}")


def trace_cost(path):
    g = json.loads(Path(path).read_text())["groups"]
    m = sum(float(x["macs"]) for x in g)
    return sum(float(x["macs"]) * float(x["stoc_len"]) for x in g) / m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-test-trace-check", action="store_true",
                    help="skip the informational test-trace fixed point (loads ~115 MB)")
    args = ap.parse_args()
    best = json.loads((HP / "ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.json").read_text())
    cells = []
    for index, (cid, model, target, arm, inc_arm, hname) in enumerate(CELLS):
        parent = HP / f"ppl/mp_best/configs/{model}/target{target}"
        inc_w = PRC2 / f"{cid}_c7_{arm}.json"
        inc_t = PRC2 / f"{cid}_c7_{arm}_table.json"
        heldout = PRC2 / f"{cid}_c7_heldout_nll.json"
        diag = PRC2 / f"{cid}_c7_diag.json"
        inc_trace = PPL / f"{cid}_prc_p2{inc_arm}_trace.json"
        par_trace = PPL / f"{cid}_parent_trace.json"
        table = json.loads(inc_t.read_text())
        rec = dict(table["prc6_calib"])
        assert rec.pop("table") == arm
        rec.pop("out")
        rec.pop("diag")
        score = dict(s.split("=", 1) for s in rec["score_tables"].split(",") if s)
        c17_table = Path(score["c17"])
        row = next(r for r in best["rows"] if r["model"] == model and r["target"] == target)
        assert row["winning_arm"] == inc_arm and Path(row["wrapper"]) == inc_w, row
        cand = next(c for c in row["candidates"] if c["arm"] == inc_arm)
        inc_ppl = float(cand["ppl"])
        inc_hdr = json.loads(inc_trace.read_text())["header"]
        assert inc_hdr["ppl"] == inc_ppl and inc_hdr["mp_config_json"] == str(inc_w)
        assert inc_hdr["ctx"] == inc_hdr["stride"] == 2048 and inc_hdr["ppl_max_tokens"] == 0
        assert inc_hdr["ppl_window_batch_size"] == 1
        par_hdr = json.loads(par_trace.read_text())["header"]
        fp16 = fp16_of(model)
        h = json.loads(heldout.read_text())["tables"]
        awq = HP / f"ppl/mp_best/act_scales/awq_scales/awq_scales_{table['model_path'].replace('/', '_')}_b4.pt"
        kb_tbl = f"{KB_TBL_PREFIX}{inc_arm}"
        frozen = [parent / "wrapper.json", parent / "table.json", parent / "hybrid_config.json",
                  inc_w, inc_t, heldout, diag, inc_trace, par_trace, c17_table, awq,
                  HP / f"int/{model}.csv", HP / "ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.json"]
        ratio_tracker = h[hname]["cost"] / h["parent"]["cost"]
        info = {
            "heldout_tracker_ratio_child_over_parent": ratio_tracker,
            "heldout_tracker_inverse_ratio": 1.0 / ratio_tracker,
            "source": f"{heldout} tables.{{{hname},parent}}.cost (mp_tracker_flop_avg_stoc_len, "
                      "NOT a trace; per-group children read up to ~0.1-0.35% cheap)",
        }
        if cid == "14B_t32":
            h_c6 = json.loads((PRC2 / "14B_t32_c6_heldout_nll.json").read_text())["tables"]
            info["investigation_1p0134_is"] = {
                "value": h_c6["parent"]["cost"] / h_c6["gfis"]["cost"],
                "source": str(PRC2 / "14B_t32_c6_heldout_nll.json") + " tables.{parent,gfis}.cost "
                          "(c6gfis, NOT the c7gfisla incumbent)"}
        if not args.no_test_trace_check:
            from benchmark.ppl.prc_r6_retarget_20260928 import analyze_trace, retarget_scale
            ap_ = analyze_trace(json.loads(par_trace.read_text()), parent / "wrapper.json", arm=arm)
            ai_ = analyze_trace(json.loads(inc_trace.read_text()), inc_w, arm=arm)
            raw, _ = retarget_scale(ap_["cost"], ai_["cost"], ai_["U"])
            info["test_trace_fixed_point_s"] = raw
            info["test_trace_audit_ok"] = [ap_["audit_ok"], ai_["audit_ok"]]
            info["test_trace_note"] = ("informational ONLY (uses test traces); the job derives s "
                                       "from held-out TRAIN traces measured in stage-a")
        info["expected_job_scale_range"] = [1.013, 1.018] if cid == "14B_t32" else [1.009, 1.015]
        cells.append({
            "index": index, "id": cid, "model": model, "target": target, "arm": arm,
            "incumbent_arm": inc_arm, "incumbent_heldout_name": hname,
            "parent": str(parent), "model_path": table["model_path"],
            "hybrid_config": str(parent / "hybrid_config.json"),
            "incumbent_wrapper": str(inc_w), "incumbent_table": str(inc_t),
            "incumbent_heldout": str(heldout), "incumbent_diag": str(diag),
            "incumbent_trace": str(inc_trace), "incumbent_ppl": inc_ppl,
            "incumbent_test_cost": trace_cost(inc_trace),
            "parent_test_trace": str(par_trace), "parent_test_cost": trace_cost(par_trace),
            "parent_test_ppl_rerun": par_hdr["ppl"], "parent_ppl": float(row["submitted_ppl"]),
            "fp16_ppl": fp16, "fp16_source": str(HP / f"int/{model}.csv"),
            "flip_thresholds": {"1.05x": 1.05 * fp16, "1.10x": 1.10 * fp16},
            "incumbent_x_fp16": inc_ppl / fp16,
            "needed_for_1p05x_pct": (1.05 * fp16 / inc_ppl - 1) * 100,
            "calib_args": rec, "c17_score_table": str(c17_table), "awq_cache": str(awq),
            "cell_dir": str(OUT_ROOT / cid), "kb_tbl": kb_tbl,
            "test_trace": str(PPL / f"{cid}_prc_{kb_tbl}_trace.json"),
            "scale_info": info,
            "prediction": dict(PREDICTIONS[cid], needed_for_1p05x_pct=(1.05 * fp16 / inc_ppl - 1) * 100),
            "hashes": {str(p): sha256(p) for p in frozen},
        })
    manifest = {
        "schema": "prc-r6-retarget-v1", "run_id": "prc_r6_retarget_20260928",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "max_total_gpus": 4, "array_gpu_cap": 2,
        "concurrency_note": "This array: 2 one-GPU tasks, throttle %2. The round-6 attention "
                            "diagnostic array (prc_r6_attn_diag_20260928) is also capped at 2, so "
                            "both round-6 arrays together use <= 4 GPUs.",
        "output_root": str(OUT_ROOT),
        "scope": "Budget re-target only (investigation RC6 / section 6 Q3): the incumbent's own "
                 "calib7 recipe re-solved at s x its calibration-sample target. Fixed: parent "
                 "wrapper + INT mask, AWQ cache, SC RNG (bitrev, 64 masks, 128-level grid), dense "
                 "ladder, per-(row,chunk) rule, attention ladders + mu+2tau escape, protected "
                 "slice, calibration windows/seed/currency. No QK rebalance or smooth-scale change.",
        "protocol": {
            "frontend": "awq", "awq_obj_bits": 4, "owen_mode": "bitrev", "scramble_masks": 64,
            "sc_prec": 8, "sc_halve": True, "rng_grid": "fixed128", "qk_rebalance": False,
            "ctx": 2048, "stride": 2048, "ppl_max_tokens": 0, "ppl_window_batch_size": 1,
            "heldout": {"windows": 16, "seed": 101, "calib_seed": 0, "calib_n": [8],
                        "split": "train", "procedure": "benchmark/ppl/heldout_nll.py (same windows "
                        "as each incumbent's stored <cell>_c7_heldout_nll.json)"},
            "scale_band": [1.0, 1.03], "scale_rounding": 1e-4,
            "cost_gate_tolerance": 0.003, "sanity_abort_dnll": 0.02,
            "sbatch": {"cpus": 4, "mem": "180G", "time": "16:00:00", "gpus_per_task": 1,
                       "array": "0-1%2"},
        },
        "preregistered_rules": {
            "scale": "s = (P - U) / (C1 - U) rounded to 1e-4, where on the 16 stored held-out "
                     "TRAIN windows P = parent exact-trace SC cost, C1 = incumbent exact-trace cost, "
                     "U = incumbent uncontrolled cycles per SC MAC (protected slice; + qk/av for "
                     "the linear-only gfis arm). One fixed-point pass: C(s) = U + s (C1 - U) = P. "
                     "Stop if s is outside (1.0, 1.03]. No test data enters s.",
            "identity": "Stop before any test unless (a) parent and incumbent held-out window NLLs "
                        "equal the stored heldout_nll.json values bit for bit, (b) calib9's s=1.0 "
                        "table equals the incumbent table in every non-provenance field, (c) the "
                        "s table differs from the incumbent only in allocation thresholds, (d) the "
                        "CPU re-solve from the saved solve state reproduces both tables exactly, "
                        "(e) every realized stream length in every trace is a rung the runtime "
                        "resolver assigns to that (op, bucket) or the protected length, with the "
                        "fixed 128-level RNG grid.",
            "cost_gate": "Full test only if |C(s)/P - 1| <= 0.3% on the same held-out TRAIN windows "
                         "(exact traces). Otherwise record stage_b.json and stop.",
            "sanity_abort": "Also stop if the s table's mean held-out NLL is more than 0.02 nats "
                            "above the incumbent's (bug guard, ~+2% PPL; not a selection rule).",
            "no_selection": "No held-out NLL-based selection: one candidate per cell, tested iff "
                            "the identity checks and the cost gate pass.",
            "best_all_entry": "The full-test PPL enters best(all) for its cell as a comparable "
                              "pure-allocation candidate WHATEVER ITS SIGN; best(all) = lowest "
                              "full-test PPL across comparable allocation rounds, each candidate "
                              "keeping its provenance.",
            "disclosure": "Round-6 BUDGET CORRECTION (disclosed): the incumbent's calib7 recipe "
                          "re-solved at s x its calibration target so the deployed table spends the "
                          "parent's cost on held-out train windows (the incumbent under-spent the "
                          "parent on test by 1.34% [14B t32] / 0.64% [30B t64]). Not a new "
                          "algorithm and not a T1 granularity effect.",
            "flip_thresholds": "PPL <= 1.05 x fp16 with fp16 from hpca_results/llm/int/<model>.csv "
                               "(14B 8.6383 -> 9.070215; 30B 7.2613 -> 7.624365).",
        },
        "sources": {"investigation": "scratchpad inv/INVESTIGATION.md RC6, section 6 Q3",
                    "scout": "scratchpad r6/scout_calib.md sections 1-3, 5"},
        "time_estimate_gpu_h": {"14B_t32": "~4 (stage-a 0.5 + capture 0.6 + stage-b 0.3 + test 2.1 + loads)",
                                "30B_t64": "~6.5 (stage-a 0.9 + capture 1.4-1.8 + stage-b 0.5 + test 2.8-4.2 + loads)"},
        "cells": cells,
        "source_files": [str(REPO / p) for p in SOURCE_FILES],
    }
    manifest["code_hashes"] = {p: sha256(p) for p in manifest["source_files"]}
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[build] wrote {MANIFEST}")
    for c in cells:
        print(f"[build] {c['id']}: incumbent {c['incumbent_ppl']:.6f} ({c['incumbent_x_fp16']:.5f}x fp16), "
              f"1.05x threshold {c['flip_thresholds']['1.05x']:.6f} (needs {c['needed_for_1p05x_pct']:+.4f}%); "
              f"held-out tracker 1/r {c['scale_info']['heldout_tracker_inverse_ratio']:.6f}; "
              f"test-trace fixed point {c['scale_info'].get('test_trace_fixed_point_s', float('nan')):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
