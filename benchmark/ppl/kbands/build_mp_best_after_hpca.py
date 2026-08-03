"""Build hpca_results/llm/ppl/mp_best_after_hpca/ — the post-HPCA archive.

Kept SEPARATE from `mp_best/` on purpose. `mp_best/` is the frozen HPCA
deployment archive that the paper and every prior comparison reference; nothing
here should mutate it. This directory records what the post-HPCA work
(qk operand rebalance + K-band per-group allocation) achieves on top of it.

SELECTION RULE, and it matters: for each (model, target) the winner is the
LOWEST full-protocol PPL among the parent and every new-method cell measured
for that pair. When the new method is WORSE — which happens on 14B for every
lever — the PARENT wins and is recorded as such. An archive that silently kept
the new number where it lost would misrepresent the method.

BASELINE: AWQ front-end + INT7 20% mask. This differs from mp_best/SUMMARY.md,
whose headline table is SmoothQuant. Every number here is AWQ-baselined, and the
parent column is the archived AWQ measurement, so the deltas are like-for-like.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

RESULTS = Path("/home/allenjin/Projects/hpca_results/llm")
MP_BEST = RESULTS / "ppl" / "mp_best"
OUT = RESULTS / "ppl" / "mp_best_after_hpca"
KB = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands")
TRACES = KB / "kbands_20260801" / "traces"
QK = KB / "kbands_20260801" / "qk"
BUNDLES = KB / "bundles"

FP16 = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}

# (model, target) -> list of (label, trace_stem, qk_scales|None, band_bundle|None)
CELLS = {
    ("4B", 32): [("qk+kband", "4B_awq_qk_kband_t32",
                  "4B_t32_qk_alpha1.0.json", "4B_t32_s2_perop16"),
                 ("band_only", "4B_awq_perop16_t32", None, "4B_t32_s2_perop16")],
    ("4B", 40): [("qk", "4B_awq_qk_t40", "4B_t40_qk_alpha1.0.json", None)],
    ("4B", 48): [("qk+kband", "4B_awq_qk_kband_t48",
                  "4B_t48_qk_alpha1.0.json", "4B_t48_s2_hf0p25"),
                 ("qk", "4B_awq_qk_a10_t48", "4B_t48_qk_alpha1.0.json", None)],
    ("llama8B", 48): [("qk", "llama8B_awq_qk_a10_t48",
                       "llama8B_t48_qk_alpha1.0.json", None)],
    ("llama8B", 96): [("qk", "llama8B_awq_qk_t96",
                       "llama8B_t96_qk_alpha1.0.json", None)],
    ("14B", 32): [("band_only", "14B_awq_band_t32", None, "14B_t32_s2_hf0p25")],
    ("14B", 48): [("qk", "14B_awq_qk_a10_t48", "14B_t48_qk_alpha1.0.json", None),
                  ("band_only", "14B_awq_band_t48", None, "14B_t48_s2_hf0p25")],
    ("30B", 32): [("qk", "30B_awq_qk_t32", "30B_t32_qk_alpha1.0.json", None)],
    ("30B", 48): [("qk", "30B_awq_qk_t48", "30B_t48_qk_alpha1.0.json", None)],
    ("30B", 64): [("qk", "30B_awq_qk_t64", "30B_t64_qk_alpha1.0.json", None)],
}


def trace_ppl(stem: str):
    p = TRACES / f"{stem}_trace.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    return (d.get("header") or d).get("ppl")


def awq_parent():
    out = {}
    src = RESULTS / "frontend_awq" / "mpbest_awq_vs_smoothquant.csv"
    for r in csv.DictReader(src.open()):
        out[(r["model"], int(r["target"]))] = (
            float(r["awq_ppl"]), float(r["awq_cost"] or 0))
    return out


def main() -> int:
    parents = awq_parent()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(exist_ok=True)
    (OUT / "qk_scales").mkdir(exist_ok=True)

    manifest, table = {}, []
    for (model, target) in sorted(CELLS, key=lambda k: (k[0], k[1])):
        par = parents.get((model, target))
        if par is None:
            print(f"  SKIP {model} t{target}: no AWQ parent recorded")
            continue
        par_ppl = par[0]
        best = ("parent", par_ppl, None, None)
        for label, stem, qk, band in CELLS[(model, target)]:
            v = trace_ppl(stem)
            if v is None:
                print(f"  (no trace) {model} t{target} {label}")
                continue
            if v < best[1]:
                best = (label, v, qk, band)

        label, ppl, qk, band = best
        key = f"{model}/target{target}"
        dst = OUT / "configs" / model / f"target{target}"
        dst.mkdir(parents=True, exist_ok=True)
        # always ship the parent bundle (the MP table/wrapper the cell ran)
        srcb = BUNDLES / band if band else MP_BEST / "configs" / model / f"target{target}"
        if srcb.is_dir():
            for f in srcb.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
        if qk and (QK / qk).is_file():
            shutil.copy2(QK / qk, OUT / "qk_scales" / qk)

        manifest[key] = {
            "model": model, "target": target,
            "winner": label,
            "ppl": ppl,
            "x_fp16": ppl / FP16[model],
            "awq_parent_ppl": par_ppl,
            "awq_parent_x_fp16": par_ppl / FP16[model],
            "delta_pct": (ppl - par_ppl) / par_ppl * 100.0,
            "passes_1p05": ppl <= FP16[model] * 1.05,
            "qk_scales": qk,
            "band_bundle": band,
            "bundle_source": str(srcb),
        }
        table.append((model, target, par_ppl, ppl, label))
        print(f"  {model:8} t{target:<3} winner={label:10} {ppl:.4f} "
              f"(parent {par_ppl:.4f})")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    lines = [
        "# mp_best_after_hpca — post-HPCA deployment winners",
        "",
        "Separate from `mp_best/` on purpose: that directory is the FROZEN HPCA",
        "archive the paper references and must not change. This one records what",
        "the post-HPCA work adds on top of it.",
        "",
        "**Baseline is AWQ + INT7 20% mask**, not SmoothQuant. `mp_best/SUMMARY.md`'s",
        "headline table is SmoothQuant, so numbers are NOT directly comparable to it;",
        "the parent column here is the archived AWQ measurement",
        "(`../../frontend_awq/mpbest_awq_vs_smoothquant.csv`).",
        "",
        "## Methods",
        "",
        "* **qk** — exact score-invariant diagonal rebalance on the Q/K contracted",
        "  dim, `scores = sum_d (Q_d/s_d)(K_d s_d)`, applied inside sc_matmul via",
        "  `smooth_scales` (so it sits AFTER RoPE and needs no pair constraint).",
        "  `s_d = mq_d^a / mk_d^(1-a)` from post-RoPE per-dim maxima, a=1.0.",
        "  qk/av is ~90% of dispatch rows and the ONLY operator class neither",
        "  SmoothQuant nor AWQ reaches — both stop at SCLinear.",
        "* **kband** — per-group (K-band) stream-length allocation: the residual",
        "  contraction axis is split into bands of whole 128-chunks, band b runs",
        "  row-rung k at its own length, under an exact per-rung iso-cost identity.",
        "",
        "Selection: LOWEST full-protocol PPL among parent and all measured cells.",
        "**Where the new method LOSES, the parent is recorded as the winner** — 14B",
        "regresses on every lever and is archived as `parent`.",
        "",
        "| model | target | AWQ parent | winner | PPL | x_fp16 | delta | 1.05x |",
        "|---|---:|---:|---|---:|---:|---:|:--|",
    ]
    for k in sorted(manifest, key=lambda s: (s.split('/')[0], int(s.split('target')[1]))):
        m = manifest[k]
        lines.append(
            f"| {m['model']} | {m['target']} | {m['awq_parent_ppl']:.4f} | "
            f"{m['winner']} | {m['ppl']:.4f} | {m['x_fp16']:.4f} | "
            f"{m['delta_pct']:+.2f}% | {'**PASS**' if m['passes_1p05'] else ''} |")
    lines += [
        "",
        "## Rungs lowered vs the AWQ baseline",
        "",
        "| model | passing rung before | after |",
        "|---|---|---|",
        "| 4B | t64 | **t48** |",
        "| 30B | t96 | **t64** |",
        "| 14B | t40 | t40 (unchanged; every lever regresses it) |",
        "| llama8B | t128 | t128 (both levers marginal) |",
        "",
        "## Caveats",
        "",
        "* Cells are NOT a uniform method: 4B t32/t48 include K-bands, 30B is",
        "  qk-only because the band allocator cannot run on MoE (it raises on the",
        "  expert index), llama8B t48 includes bands that contributed nothing.",
        "  The `winner` column states which was used per cell.",
        "* Blank (model, target) pairs were not run, mostly because the baseline",
        "  already passed 1.05x or the model showed no response to either lever.",
        "* Provenance for every number: `../../..//scmp_llm/benchmark/ppl/kbands/",
        "  LOOP_STATE.md`, which also records four retracted intermediate claims.",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/SUMMARY.md and manifest.json ({len(manifest)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
