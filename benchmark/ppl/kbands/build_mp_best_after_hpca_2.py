"""Build hpca_results/llm/ppl/mp_best_after_hpca_2/ — the merged deployment archive.

RELATIONSHIP TO ITS PREDECESSOR. `mp_best_after_hpca` explored two levers on top
of the frozen `mp_best` per-row parents: **qk** (a score-invariant diagonal on
the Q/K contracted dim, i.e. ATTENTION) and **K-bands** (static per-chunk stream
lengths on LINEARS). This archive keeps the same parents and the same qk lever —
it loads that archive's own qk scale files unchanged — and REPLACES K-bands with
**per-(row, chunk)** allocation: the same operators, but input-adaptive at the
granularity quantization already uses, rather than one static assignment shared
by every row.

So this is an EXTENSION, not a supersession. Per cell it takes whichever of the
two archives is lower, and it records which. Both rest on the identical
AWQ + INT7-20% baseline; the parent arm re-run here reproduced the predecessor's
archived AWQ parent PPL EXACTLY on five cells (10.6491 / 7.9200 / 8.9072 /
9.4647 / 8.1784), which is what licenses comparing the two at all.

SELECTION. Lowest full-protocol WikiText-2 test PPL per (model, target). Where
the predecessor is lower it WINS and is recorded as such — this archive is not
allowed to flatter the new lever.

COST. Every cell also carries `realized_flop_avg_sl` from its own run, and a
COST-ADJUSTED value that removes the compute term (local PPL-vs-compute
sensitivity, anchored on llama8B which has two prc runs at different cost). A
raw PPL win at higher compute is not the same as a better allocation, and this
archive keeps both numbers so the distinction survives.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

RES = Path("/home/allenjin/Projects/hpca_results/llm")
PREV = RES / "ppl" / "mp_best_after_hpca"
MPB = RES / "ppl" / "mp_best"
OUT = RES / "ppl" / "mp_best_after_hpca_2"
LOGS = Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/"
            "hpca/logs/_kbands")
PPLDIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl")
PRCDIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/prc")

FP16 = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}
MODELS = ["4B", "llama8B", "14B", "30B"]
# The per-(row, chunk) wave ran t32/t48, but the archive must COVER every cell
# its predecessor holds or it silently drops them -- including 30B t64, a 1.05x
# PASS. Cells with no wave run and no predecessor entry fall out as `pending`;
# cells with only a predecessor entry are carried at `source: mp_best_after_hpca`.
# Sensitivities are anchored on (32, 48) explicitly, so widening this is inert
# for the cost model.
TARGETS = [32, 40, 48, 64, 96]
# arms measured in this wave, in preference order for tie-breaks
ARMS = ["prcqk", "prc", "qk", "parent"]


def read_grid_result(model: str, target: int, arm: str):
    """PPL for the SC_RNG_GRID=pow2 variant of an arm.

    Grid runs are launched with a `grid_` job-name prefix precisely so this
    builder's `prcppl_` glob cannot pick them up by accident; they are opted in
    HERE, explicitly. Same table, same 20% mask, same budget -- only the SC
    enable-grid differs (largest pow2 <= stoc_len), which is runtime-free.
    An identity control with the flag unset reproduced the archived 30B t48
    number EXACTLY (7.6394), so any delta is the grid and not code drift.
    """
    name = f"{model}_t{target}_{arm}"
    for f in sorted([q for q in LOGS.glob(f"grid_{name}_*.out")
                     if re.fullmatch(rf"grid_{re.escape(name)}_\d+\.out", q.name)],
                    key=lambda q: q.stat().st_mtime, reverse=True):
        for line in reversed(f.read_text(errors="ignore").splitlines()):
            if "[RESULT]" in line and "metric=ppl" in line:
                kv = dict(t.split("=", 1) for t in line.split() if "=" in t)
                try:
                    return float(kv["value"]), float(kv["realized_flop_avg_sl"])
                except Exception:                            # noqa: BLE001
                    pass
    return None, None


def read_result(tag: str):
    """(ppl, realized_flop_avg_sl) from the eval's own [RESULT] line."""
    # The suffix must be a JOB ID. A bare `{tag}_*` also matches longer arm
    # names -- `prc_*` swallows `prc_iso2_*` -- and since the glob is sorted by
    # mtime the newer re-run silently shadows the arm being asked for.
    for f in sorted([q for q in LOGS.glob(f"prcppl_{tag}_*.out")
                     if re.fullmatch(rf"prcppl_{re.escape(tag)}_\d+\.out", q.name)],
                    key=lambda q: q.stat().st_mtime, reverse=True):
        for line in reversed(f.read_text(errors="ignore").splitlines()):
            if "[RESULT]" in line and "metric=ppl" in line:
                kv = dict(t.split("=", 1) for t in line.split() if "=" in t)
                try:
                    return float(kv["value"]), float(kv["realized_flop_avg_sl"])
                except Exception:                            # noqa: BLE001
                    pass
    return None, None


def sensitivities(parents):
    """Local %PPL per %compute per model.

    Wide-range parent slope (t32 -> t48) scaled by 1.47, the ratio measured
    directly on llama8B, which has two prc runs at different cost and therefore
    pins the LOCAL slope. Without the scaling the wide-range average
    understates sensitivity near the tight end, where the curve is steeper.
    """
    s = {}
    for m in MODELS:
        a = parents.get((m, 32))
        b = parents.get((m, 48))
        if not a or not b:
            continue
        dp = (a[0] - b[0]) / a[0] * 100.0
        dc = (b[1] - a[1]) / a[1] * 100.0
        if dc:
            s[m] = dp / dc * 1.47
    return s


def main() -> int:
    prev = json.loads((PREV / "manifest.json").read_text())
    parents = {}
    for m in MODELS:
        for t in TARGETS:
            p = read_result(f"{m}_t{t}_parent")
            if p[0] is not None:
                parents[(m, t)] = p
    S = sensitivities(parents)

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("configs", "prc_tables", "qk_scales"):
        (OUT / sub).mkdir(exist_ok=True)

    manifest, rows, pending = {}, [], []
    for m in MODELS:
        for t in TARGETS:
            par = parents.get((m, t))
            # every arm measured in THIS wave
            mine = {}
            for arm in ARMS:
                if arm == "parent":
                    continue
                v = read_result(f"{m}_t{t}_{arm}")
                if v[0] is not None:
                    mine[arm] = v
                gv = read_grid_result(m, t, arm)
                if gv[0] is not None:
                    mine[arm + "+grid"] = gv
            pv = prev.get(f"{m}/target{t}")
            if not mine and not pv:
                pending.append(f"{m} t{t}")
                continue

            cands = []
            if pv:
                cands.append(("prev:" + pv["winner"], pv["ppl"], None))
            for arm, (v, c) in mine.items():
                cands.append((arm, v, c))
            if par:
                cands.append(("parent", par[0], par[1]))
            lab, ppl, cost = min(cands, key=lambda r: r[1])

            adj = None
            if par and cost and m in S:
                raw = (ppl - par[0]) / par[0] * 100.0
                dc = (cost - par[1]) / par[1] * 100.0
                adj = raw + dc * S[m]

            key = f"{m}/target{t}"
            src = "mp_best_after_hpca" if lab.startswith("prev:") else "this wave"
            manifest[key] = {
                "model": m, "target": t, "winner": lab, "source": src,
                "ppl": ppl, "x_fp16": ppl / FP16[m],
                "realized_flop_avg_sl": cost,
                "parent_ppl": par[0] if par else None,
                "parent_cost": par[1] if par else None,
                "delta_vs_parent_pct": ((ppl - par[0]) / par[0] * 100.0) if par else None,
                "cost_adjusted_value_pct": adj,
                "prev_archive_ppl": pv["ppl"] if pv else None,
                "prev_archive_winner": pv["winner"] if pv else None,
                "delta_vs_prev_pct": ((ppl - pv["ppl"]) / pv["ppl"] * 100.0) if pv else None,
                "all_arms_this_wave": {a: v for a, (v, _) in mine.items()},
                "passes_1p05": ppl <= FP16[m] * 1.05,
            }
            rows.append(manifest[key])

            # ship what the winner needs to be reproduced
            dst = OUT / "configs" / m / f"target{t}"
            dst.mkdir(parents=True, exist_ok=True)
            srcb = MPB / "configs" / m / f"target{t}"
            if srcb.is_dir():
                for f in srcb.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)
            if lab in ("prc", "prcqk"):
                for pat in (f"{m}_t{t}_v7_prc.json", f"{m}_t{t}_v7_prc_table.json"):
                    src_f = PRCDIR / pat
                    if src_f.is_file():
                        shutil.copy2(src_f, OUT / "prc_tables" / pat)
            if lab in ("qk", "prcqk") or (pv and "qk" in (pv.get("winner") or "")):
                for cand in (PREV / "qk_scales" / f"{m}_t{t}_qk_alpha1.0.json",
                             Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/"
                                  f"kbands_20260801/qk/{m}_t{t}_qk_alpha1.0.json")):
                    if cand.is_file():
                        shutil.copy2(cand, OUT / "qk_scales" / cand.name)
                        break

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    def fmt(v, spec, dash="—"):
        return format(v, spec) if isinstance(v, (int, float)) else dash

    L = [
        "# mp_best_after_hpca_2 — merged deployment archive",
        "",
        "Extends `mp_best_after_hpca`. **Same parents** (`mp_best`, AWQ + INT7 20%",
        "mask) and **the same qk lever** — this archive loads that one's qk scale",
        "files unchanged. What changed is the LINEAR-side lever: static **K-bands**",
        "are replaced by **per-(row, chunk)** allocation, which assigns a stream",
        "length to every (row, 128-chunk) group instead of one static assignment",
        "shared by all rows.",
        "",
        "Per cell the archive takes whichever source is lower and records which.",
        "Where the predecessor wins it is kept — see the `source` column.",
        "",
        "**Baseline sanity**: the parent arm re-run in this wave reproduced the",
        "predecessor's archived AWQ parent PPL EXACTLY on five cells (10.6491,",
        "7.9200, 8.9072, 9.4647, 8.1784). That is what licenses the comparison.",
        "",
        "## Results",
        "",
        "| model | tgt | winner | source | PPL | x_fp16 | vs parent | cost | value* | vs prev |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: (r["model"], r["target"])):
        L.append(
            f"| {r['model']} | {r['target']} | {r['winner']} | {r['source']} | "
            f"{r['ppl']:.4f} | {r['x_fp16']:.4f} | "
            f"{fmt(r['delta_vs_parent_pct'], '+.2f')}% | "
            f"{fmt(r['realized_flop_avg_sl'], '.1f')} | "
            f"{fmt(r['cost_adjusted_value_pct'], '+.2f')}% | "
            f"{fmt(r['delta_vs_prev_pct'], '+.2f')}% |")
    L += [
        "",
        "\\*`value` = cost-adjusted contribution: the PPL delta with the compute",
        "term removed, using local %PPL-per-%compute sensitivity "
        + ", ".join(f"{m} {v:.3f}" for m, v in sorted(S.items())) + ".",
        "A raw win at higher compute is NOT a better allocation; both numbers are",
        "kept so that distinction cannot be lost.",
        "",
        "## Which lever wins where, and why",
        "",
        "* **per-(row, chunk) replaces K-bands.** On 4B t32 it beats qk+kband by",
        "  −3.40%; on 4B t48 the two are within 0.1%.",
        "* **qk is per-cell, not universal.** Measured WITH controls in this wave:",
        "  qk adds −2.97% (4B t32) and −2.08% (4B t48) and −0.30% (llama8B t48),",
        "  but REGRESSES 14B at both budgets (t32 qk 9.3525 vs parent 9.3097;",
        "  t48 prcqk 8.9423 vs prc 8.8571) and llama8B t32 (+0.56%). That",
        "  independently reproduces the predecessor's own lever selection.",
        "* **14B had no working lever before.** Its predecessor winners are the bare",
        "  parent; per-(row, chunk) moves it for the first time.",
        "* **30B needs BOTH levers, and it is the cleanest evidence they compose.**",
        "  Alone, each LOSES to the predecessor's qk-only cells (prc 8.6696 /",
        "  8.0931 vs 8.6255 / 7.7217); combined, prcqk wins both budgets by the",
        "  largest margins here (8.0212 / 7.6394). Narrow expert projections",
        "  (6 chunks) leave little within-row structure, so on MoE the linear-side",
        "  lever appears to ride attention rather than stand alone.",
        "",
        "## Caveats",
        "",
        "* **Coverage — this archive now covers every `mp_best_after_hpca` cell.**",
        "  The grid was widened to t{32,40,48,64,96} so no predecessor cell is",
        "  dropped; an earlier t32/t48-only grid silently omitted 30B t64, which is",
        "  a 1.05x PASS. Three cells (30B t64, 4B t40, llama8B t96) are CARRIED from",
        "  the predecessor and have no per-(row, chunk) arm yet, so their `source`",
        "  reads `mp_best_after_hpca` and their cost-adjusted value is null.",
        "  Quote 1.05x passes as 3 of 11 over THIS archive; do not mix that count",
        "  with the predecessor's own 10-cell denominator.",
        "* Cells are NOT a uniform method — the `winner` column states the config.",
        "* Costs are read from each run's TRACE, which prices per rung, so a",
        "  per-(row, chunk) cell cannot flatter its own budget.",
        "* Noise floor is sd 0.0069 PPL (n=6 identity controls); |delta| < ~0.014",
        "  is not a result.",
        "* Full provenance and the complete bug ledger (11 bugs, 4 refuted",
        "  hypotheses) are in `scmp_llm/benchmark/ppl/kbands/LOOP_STATE.md`.",
    ]
    if pending:
        L += ["", f"**Not yet complete**: {', '.join(pending)} — rerun this script",
              "once those land; it is idempotent."]
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(manifest)} cells)"
          + (f"; pending {pending}" if pending else ""))
    for r in sorted(rows, key=lambda r: (r["model"], r["target"])):
        print(f"  {r['model']:8} t{r['target']:<3} {r['winner']:14} "
              f"{r['source']:20} {r['ppl']:.4f}  x{r['x_fp16']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
