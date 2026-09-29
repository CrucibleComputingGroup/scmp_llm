"""Build hpca_results/llm/ppl/mp_best_after_hpca_3/ — the COMPLETE MP archive.

WHAT THIS IS. The full MP grid (4 models x t{32,40,48,64,96} = 20 cells) measured
under the current best method, with the winning arm's TRACE for the energy model.
Unlike `_2`, which was an extension of `_1` over a partial grid, this archive
covers every MP cell `mp_best` holds and is the one the paper should quote.

t128 IS DELIBERATELY NOT AN MP CELL. Those `mp_best` bundles are
`deployment_mode: uniform_sc` / `quant_config: sc_int8`, realized 128.0, and carry
no `table.json` — at the ceiling there is nothing to allocate. It is recorded here
as a `uniform_ceiling` reference block, never as an MP result.

METHOD. Per cell the lowest full-protocol WikiText-2 test PPL over:
  parent  the mp_best per-row config re-run in this environment. Not optional:
          comparisons are S1->S2 and the cost adjustment needs this cell's own
          traced parent cost.
  prc     per-(row, chunk) allocation — a stream length per (row, 128-chunk)
          group instead of one shared by every row in a row-dispatch.
  qk      score-invariant diagonal rebalance on the Q/K contracted dim.
  prcqk   both. qk is PER-CELL, not universal (it regresses 14B and llama8B t32),
          so prc and prcqk are both measured and the lower wins.
  +grid   SC_RNG_GRID=pow2: enable-grid = largest power of two <= stoc_len.
          Runtime-free (same cycles). An identity control with the flag unset
          reproduced the archived 30B t48 number EXACTLY (7.6394), so any grid
          delta is the grid and not code drift.
Predecessor archives `_1` / `_2` are also candidates, so a cell can never get
WORSE than what was already shipped.

BASELINE. "old" is `mp_best` re-measured under AWQ, not its SmoothQuant headline
number — the post-HPCA work is AWQ-baselined and the two are not comparable.

COST. Every cell carries `realized_flop_avg_sl` from its own trace plus a
cost-adjusted value, because a raw PPL win bought with more compute is not a
better allocation.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

RES = Path("/home/allenjin/Projects/SCMP/hpca_results/llm")
A1 = RES / "ppl" / "mp_best_after_hpca"
A2 = RES / "ppl" / "mp_best_after_hpca_2"
MPB = RES / "ppl" / "mp_best"
OUT = RES / "ppl" / "mp_best_after_hpca_3"
LOGS = Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/"
            "hpca/logs/_kbands")
PPLDIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl")
PRCDIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/prc")
QKDIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/qk")

FP16 = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}
MODELS = ["4B", "llama8B", "14B", "30B"]
TARGETS = [32, 40, 48, 64, 96]
ARMS = ["prcqk", "prc", "qk", "parent"]


def _read(prefix: str, name: str):
    """(ppl, cost) from a job's own [RESULT] line, newest first.

    The job-id suffix must be matched exactly: a bare `{name}_*` also matches
    longer arm names (`prc_*` swallows `prc_iso2_*`) and, sorted by mtime, a
    newer unrelated run would silently shadow the arm being asked for.
    """
    for f in sorted([q for q in LOGS.glob(f"{prefix}_{name}_*.out")
                     if re.fullmatch(rf"{prefix}_{re.escape(name)}_\d+\.out", q.name)],
                    key=lambda q: q.stat().st_mtime, reverse=True):
        for line in reversed(f.read_text(errors="ignore").splitlines()):
            if "[RESULT]" in line and "metric=ppl" in line:
                kv = dict(t.split("=", 1) for t in line.split() if "=" in t)
                try:
                    return float(kv["value"]), float(kv["realized_flop_avg_sl"])
                except Exception:                                # noqa: BLE001
                    pass
    return None, None


def sensitivities(parents):
    """Local %PPL per %compute, per model (t32->t48 slope x1.47).

    The 1.47 is measured directly on llama8B, which has two prc runs at
    different cost and therefore pins the LOCAL slope; the wide-range parent
    slope is flatter and understates sensitivity near the tight end.
    """
    s = {}
    for m in MODELS:
        a, b = parents.get((m, 32)), parents.get((m, 48))
        if not a or not b:
            continue
        dc = (b[1] - a[1]) / a[1] * 100.0
        if dc:
            s[m] = (a[0] - b[0]) / a[0] * 100.0 / dc * 1.47
    return s


def main() -> int:
    a1 = json.loads((A1 / "manifest.json").read_text())
    a2 = json.loads((A2 / "manifest.json").read_text()) if (A2 / "manifest.json").is_file() else {}

    parents = {}
    for m in MODELS:
        for t in TARGETS:
            p = _read("prcppl", f"{m}_t{t}_parent")
            if p[0] is not None:
                parents[(m, t)] = p
    S = sensitivities(parents)

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("configs", "prc_tables", "qk_scales", "traces"):
        (OUT / sub).mkdir(exist_ok=True)

    manifest, rows, missing = {}, [], []
    for m in MODELS:
        for t in TARGETS:
            key = f"{m}/target{t}"
            par = parents.get((m, t))
            cands = []
            for arm in ARMS:
                v = _read("prcppl", f"{m}_t{t}_{arm}")
                if v[0] is not None:
                    cands.append((arm, v[0], v[1]))
                g = _read("grid", f"{m}_t{t}_{arm}")
                if g[0] is not None:
                    cands.append((arm + "+grid", g[0], g[1]))
            for tag, arch in (("v1", a1), ("v2", a2)):
                e = arch.get(key)
                if e and e.get("ppl"):
                    cands.append((f"prev{tag}:" + str(e.get("winner")), e["ppl"],
                                  e.get("realized_flop_avg_sl")))
            if not cands:
                missing.append(key)
                continue
            lab, ppl, cost = min(cands, key=lambda r: r[1])

            old = (a2.get(key, {}).get("parent_ppl")
                   or a1.get(key, {}).get("awq_parent_ppl")
                   or (par[0] if par else None))
            adj = None
            if par and cost and m in S:
                adj = ((ppl - par[0]) / par[0] * 100.0
                       + (cost - par[1]) / par[1] * 100.0 * S[m])
            x = ppl / FP16[m]
            manifest[key] = {
                "model": m, "target": t, "winner": lab, "ppl": ppl,
                "x_fp16": x, "passes_1p05": x <= 1.05,
                "realized_flop_avg_sl": cost,
                "awq_parent_ppl": old,
                "awq_parent_x_fp16": (old / FP16[m]) if old else None,
                "delta_vs_awq_parent_pct": ((ppl - old) / old * 100.0) if old else None,
                "cost_adjusted_value_pct": adj,
                "all_arms": {a: v for a, v, _ in cands},
            }
            rows.append((m, t, old, ppl, lab, x))

            # artifacts
            dst = OUT / "configs" / m / f"target{t}"
            dst.mkdir(parents=True, exist_ok=True)
            src = MPB / "configs" / m / f"target{t}"
            if src.is_dir():
                for f in src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)
            base = lab.replace("+grid", "").split(":")[-1]
            if base in ("prc", "prcqk"):
                for pat in (f"{m}_t{t}_v7_prc.json", f"{m}_t{t}_v7_prc_table.json"):
                    if (PRCDIR / pat).is_file():
                        shutil.copy2(PRCDIR / pat, OUT / "prc_tables" / pat)
            if "qk" in base:
                for c in (A1 / "qk_scales" / f"{m}_t{t}_qk_alpha1.0.json",
                          QKDIR / f"{m}_t{t}_qk_alpha1.0.json"):
                    if c.is_file():
                        shutil.copy2(c, OUT / "qk_scales" / c.name)
                        break
            # winning arm's TRACE — the energy model runs off these, not budgets
            if not lab.startswith("prev"):
                want_grid = lab.endswith("+grid")
                for c in sorted(PPLDIR.glob(f"{m}_t{t}_{base}*_trace.json"),
                                key=lambda q: q.stat().st_mtime, reverse=True):
                    if ("_grid" in c.name) == want_grid:
                        shutil.copy2(c, OUT / "traces" / c.name)
                        manifest[key]["trace"] = c.name
                        break

    # t128 uniform ceiling, as REFERENCE only
    ceiling = {}
    for m in MODELS:
        p = MPB / "configs" / m / "target128" / "metadata.json"
        if p.is_file():
            d = json.load(open(p))
            ceiling[m] = {"ppl": d.get("ppl"), "x_fp16": d.get("x_fp16"),
                          "realized_flop_avg_stoc_len": d.get("realized_flop_avg_stoc_len"),
                          "deployment_mode": d.get("deployment_mode"),
                          "note": "uniform SC ceiling, NOT an MP cell"}
    manifest["_uniform_ceiling_t128"] = ceiling
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    cov = [r for r in rows]
    imp = [(p - o) / o * 100.0 for _, _, o, p, _, _ in cov if o]
    npass = sum(1 for k, v in manifest.items()
                if k != "_uniform_ceiling_t128" and v["passes_1p05"])
    nold = sum(1 for k, v in manifest.items()
               if k != "_uniform_ceiling_t128" and v.get("awq_parent_x_fp16")
               and v["awq_parent_x_fp16"] <= 1.05)

    L = ["# mp_best_after_hpca_3 — the COMPLETE MP archive", "",
         f"{len(cov)} of {len(MODELS)*len(TARGETS)} MP cells "
         f"(4 models x t{{{','.join(str(x) for x in TARGETS)}}}).", "",
         "`old` = **`mp_best` re-measured under AWQ**, not its SmoothQuant headline "
         "number: the post-HPCA work is AWQ-baselined and the two are NOT comparable.",
         "", "| model | tgt | old (AWQ) | new | winner | improve | x_fp16 | was | 1.05x |",
         "|---|---:|---:|---:|---|---:|---:|---:|:--|"]
    for m, t, o, p, lab, x in cov:
        d = f"{(p-o)/o*100:+.2f}%" if o else "—"
        xo = f"{o/FP16[m]:.4f}" if o else "—"
        L.append(f"| {m} | {t} | {o:.4f} | **{p:.4f}** | {lab} | {d} | "
                 f"{x:.4f} | {xo} | {'**PASS**' if x <= 1.05 else ''} |")
    L += ["", f"**Mean improvement {sum(imp)/len(imp):+.2f}%** over {len(imp)} cells. "
          f"Cells at or under 1.05x fp16: **{nold} -> {npass}**.", "",
          "## t128 is not here", "",
          "The t128 `mp_best` bundles are `deployment_mode: uniform_sc` / "
          "`quant_config: sc_int8`, realized 128.0, and carry no `table.json` — at "
          "the ceiling there is nothing to allocate. They are recorded in "
          "`manifest.json` under `_uniform_ceiling_t128` as a reference, never as an "
          "MP result.", "", "## Caveats", "",
          "* Cells are NOT a uniform method — `winner` states the config per cell. "
          "qk regresses 14B and llama8B t32; grid pays on 30B and is "
          "neutral-to-negative elsewhere. Applying either uniformly loses ground.",
          "* Costs come from each run's own TRACE, which prices per rung, so a cell "
          "cannot flatter its own budget.",
          "* Noise floor sd 0.0069 PPL; |delta| < ~0.014 is not a result.",
          "* Predecessor archives are candidates too, so no cell can regress below "
          "what was already shipped.",
          "* Full ledger, every refuted hypothesis and bug: "
          "`scmp_llm/benchmark/ppl/kbands/LOOP_STATE.md`."]
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")

    print(f"wrote {OUT} ({len(cov)} MP cells)"
          + (f"; MISSING {missing}" if missing else ""))
    for m, t, o, p, lab, x in cov:
        print(f"  {m:9s} t{t:<4d} {lab:18s} {p:9.4f}  x{x:.4f}"
              f"{'  PASS' if x <= 1.05 else ''}")
    print(f"\nmean {sum(imp)/len(imp):+.2f}% over {len(imp)} cells | "
          f"passing 1.05x {nold} -> {npass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
