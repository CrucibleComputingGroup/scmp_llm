"""Build hpca_results/llm/ppl/prc2/ -- the per-group (per-(row,128-chunk)) allocation
archive, calibrated with mp_per_row_chunk_calib2.py ("prc2", tag c17).

Idempotent; login node; re-run as cells land. RAISES on any [RESULT] that is not full
protocol (ctx 2048, PPL_MAX_TOKENS=0, full token count, AWQ).

Arms (all: same mp_best wrapper, same INT7 mask, same attention thresholds, AWQ):
  parent   submitted per-row allocation            prcppl_<m>_t<T>_parent_*   (2026-08-03)
  v7       old per-(row,chunk) tables (calib v1)    prcppl_<m>_t<T>_prc_*      (2026-08-03)
  c17      prc2 tables at the parent's cost         p2_<m>_t<T>_c17_*  (+ c17e32 = 30B retry)
  c17_s80  prc2 tables at 0.8x linear budget        p2ppl_<m>_t<T>_c17_s80_*
  r17      per-ROW control, calibrated identically  p3_<m>_t<T>_r17_*  (SC_PRC_ROWSHARED=1)

  python benchmark/ppl/kbands/build_prc2_archive.py
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
import shutil
import statistics as st
from pathlib import Path

ROOT = Path("/home/allenjin/Projects/SCMP")
OUT = ROOT / "hpca_results/llm/ppl/prc2"
HARV = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/harvested_results_20260920.txt")
LOGS = Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands")
TRACES = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl")
P2 = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2")
LLM = ROOT / "scmp_llm"
FP16 = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}
TOKENS = {"4B": 298862, "llama8B": 288627, "14B": 298862, "30B": 298862}
MODELS = ("4B", "llama8B", "14B", "30B")
TARGETS = (32, 40, 48, 64, 96)
NOISE_SD = 0.0069
RX = re.compile(r"model=(\S+) config=mp metric=ppl value=([\d.]+) tokens=(\d+) sec=([\d.]+)"
                r".*?realized_flop_avg_sl=([\d.]+)")


def _check_protocol(text: str, where: str) -> None:
    if "PPL_MAX_TOKENS=0" not in text or "ctx=2048" not in text or "frontend=awq" not in text:
        raise SystemExit(f"[prc2-archive] NOT full protocol / AWQ: {where}")


def collect():
    res = {}          # (model, T, arm) -> dict
    for line in HARV.read_text().splitlines():
        m = re.match(r"(prcppl_(4B|llama8B|14B|30B)_t(\d+)_(parent|prc)_(\d+)\.out)\|", line)
        r = RX.search(line)
        if not (m and r):
            continue
        arm = "v7" if m[4] == "prc" else "parent"
        res[(m[2], int(m[3]), arm)] = dict(
            ppl=float(r[2]), cost=float(r[5]), tokens=int(r[3]), sec=float(r[4]),
            hf=r[1], log=f"(harvested) {m[1]}", job=m[5])
    for f in sorted(LOGS.glob("p*_*.out")):
        m = re.match(r"(p2|p3|p2ppl|p2s|p6|p6ev|p7|p7ev)_(4B|llama8B|14B|30B)_t(\d+)_(\w+?)_(\d+)\.out", f.name)
        if not m:
            continue
        text = f.read_text(errors="replace")
        r = RX.search(text)
        if not r:
            continue
        _check_protocol(text, str(f))
        arm = m[4]
        if m[1] in ("p6", "p7"):         # chained calib6/7 job: the evaluated table is in the tag
            t6 = re.search(r"\[prc-ppl\] \S+_prc_p2(c[67]\w+)\s", text)
            arm = t6.group(1) if t6 else m[1]
        if arm == "c17e32":          # 30B t32 retry with denser expert sampling
            arm = "c17"
        res[(m[2], int(m[3]), arm)] = dict(
            ppl=float(r[2]), cost=float(r[5]), tokens=int(r[3]), sec=float(r[4]),
            hf=r[1], log=str(f), job=m[5])
    for (mdl, _t, _a), v in res.items():
        if v["tokens"] != TOKENS[mdl]:
            raise SystemExit(f"[prc2-archive] token count {v['tokens']} != full "
                             f"{TOKENS[mdl]} for {mdl}")
    return res


def trace_cost(path: Path):
    """MAC-weighted mean stream length over EVERY SC group in a full-eval trace
    (linears incl. protected slice + attention). The in-process tracker behind the
    [RESULT] line prices each (row,chunk) pair at 1/n_chunks of the row, which
    over-weights narrow tail chunks and makes per-group arms read up to ~0.4% cheap;
    for per-row arms the two agree to 0.01."""
    if not path.is_file():
        return None
    g = json.loads(path.read_text())["groups"]
    m = sum(float(x["macs"]) for x in g)
    return sum(float(x["macs"]) * float(x["stoc_len"]) for x in g) / m if m else None


TRACE_OF = {"parent": "{c}_parent_trace.json", "v7": "{c}_prc_v7_trace.json",
            "c17": "{c}_prc_p2c17_trace.json", "c17_s80": "{c}_prc_p2c17_s80_trace.json",
            "r17": "{c}_prc_p2r17_trace.json"}


def attach_trace_costs(res):
    for (mdl, t, arm), v in res.items():
        c = f"{mdl}_t{t}"
        cand = []
        if arm in TRACE_OF:
            cand.append(TRACES / TRACE_OF[arm].format(c=c))
        if arm == "v7":
            cand.append(TRACES / f"{c}_prc_trace.json")
        if arm == "c17":
            cand.insert(0, TRACES / f"{c}_prc_p2c17e32_trace.json")
        cand.append(TRACES / f"{c}_prc_p2{arm}_trace.json")
        tc = None
        for pth in cand:
            if pth.is_file():
                tc = trace_cost(pth)
                v["trace"] = str(pth)
                break
        v["tracker_cost"] = v["cost"]
        if tc is not None:
            v["cost"] = tc
            v["cost_source"] = "trace"
        else:
            v["cost_source"] = "tracker"


def iso_ppl_saving(points, ppl_target, cost_ref):
    """Cost at which the per-group curve reaches `ppl_target`, by chord interpolation
    between measured per-group points (conservative: PPL(cost) is convex, so the chord
    lies above the curve and over-states the cost). Returns (saving_frac, kind)."""
    pts = sorted(points)                      # (cost, ppl)
    if not pts:
        return None, "none"
    # enforce monotone decreasing ppl in cost (drop dominated points)
    mono = []
    for c, p in pts:
        if not mono or p < mono[-1][1]:
            mono.append((c, p))
    if ppl_target >= mono[0][1]:
        # even the cheapest per-group point is at least as good: saving >= that
        return 1 - mono[0][0] / cost_ref, "lower_bound"
    for (c0, p0), (c1, p1) in zip(mono, mono[1:]):
        if p1 <= ppl_target <= p0:
            w = (p0 - ppl_target) / (p0 - p1)
            return 1 - (c0 + w * (c1 - c0)) / cost_ref, "interp"
    return None, "beyond"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    res = collect()
    attach_trace_costs(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    (OUT / "code").mkdir(exist_ok=True)

    # ---- copy tables / diags / traces --------------------------------------
    manifest = {"cells": {}, "arms": {}, "noise_sd_ppl": NOISE_SD, "fp16": FP16}
    for mdl in MODELS:
        for t in TARGETS:
            cell = f"{mdl}_t{t}"
            d = OUT / "configs" / mdl / f"target{t}"
            ent = {}
            stems = [("c17", f"{cell}_c17"), ("c17e32", f"{cell}_c17e32"),
                     ("c17_s80", f"{cell}_c17_s80"), ("r17", f"{cell}_r17")]
            stems += [(Path(x).stem.replace(f"{cell}_", ""), Path(x).stem)
                      for x in glob.glob(str(P2 / f"{cell}_c17iso_s*.json"))
                      if not x.endswith("_table.json")]
            for tag, stem in stems:
                w = P2 / f"{stem}.json"
                tb = P2 / f"{stem}_table.json"
                dg = P2 / f"{stem}_diag.json"
                if not w.is_file():
                    continue
                d.mkdir(parents=True, exist_ok=True)
                for src in (w, tb, dg):
                    if src.is_file():
                        shutil.copy2(src, d / src.name)
                ent[tag] = {"wrapper": str(w), "table": str(tb),
                            "diag": str(dg) if dg.is_file() else None}
            for arm, tr in (("parent", f"{cell}_parent_trace.json"),
                            ("v7", f"{cell}_prc_v7_trace.json" if (TRACES / f"{cell}_prc_v7_trace.json").is_file() else f"{cell}_prc_trace.json"),
                            ("c17", f"{cell}_prc_p2c17_trace.json"),
                            ("c17e32", f"{cell}_prc_p2c17e32_trace.json"),
                            ("c17_s80", f"{cell}_prc_p2c17_s80_trace.json"),
                            ("r17", f"{cell}_prc_p2r17_trace.json")):
                src = TRACES / tr
                if src.is_file() and arm != "parent":
                    shutil.copy2(src, OUT / "traces" / tr)
                ent.setdefault("traces", {})[arm] = str(src) if src.is_file() else None
            for src in glob.glob(str(TRACES / f"{cell}_prc_p2c17iso_*_trace.json")):
                shutil.copy2(src, OUT / "traces" / Path(src).name)
                ent.setdefault("traces", {})[Path(src).name.split("_prc_p2")[1][:-11]] = src
            for (m_, t_, arm), v in res.items():
                if m_ == mdl and t_ == t:
                    ent.setdefault("results", {})[arm] = v
            manifest["cells"][cell] = ent
    for src in ("benchmark/ppl/mp_per_row_chunk_calib2.py",
                "benchmark/ppl/mp_per_row_chunk_calib3.py",
                "benchmark/ppl/kbands/run_prc2_calib.sbatch",
                "benchmark/ppl/kbands/run_prc3_calib.sbatch",
                "benchmark/ppl/kbands/run_prc3_analysis.sbatch",
                "benchmark/ppl/kbands/launch_prc2_ppl.sh",
                "benchmark/ppl/kbands/run_prc2_scaled.sbatch",
                "benchmark/ppl/kbands/run_prc_ppl.sbatch",
                "benchmark/ppl/kbands/build_prc2_archive.py",
                "benchmark/ppl/mp_per_row_chunk_calib5.py",
                "benchmark/ppl/mp_per_row_chunk_calib6.py",
                "benchmark/ppl/heldout_nll.py",
                "benchmark/ppl/kbands/run_prc5.sbatch",
                "benchmark/ppl/kbands/run_prc6.sbatch"):
        shutil.copy2(LLM / src, OUT / "code" / Path(src).name)
    # the runtime edits (per_row_chunk path + SC_PRC_ROWSHARED) are uncommitted: keep a diff
    os.system(f"cd {LLM} && git diff -- model/sc_common.py > {OUT}/code/sc_common.uncommitted.diff")
    os.system(f"cd {LLM}/kernels && git diff > {OUT}/code/kernels.uncommitted.diff")

    # ---- results.csv -------------------------------------------------------
    rows = []
    for (mdl, t, arm), v in sorted(res.items()):
        rows.append(dict(model=mdl, target=t, arm=arm, ppl=v["ppl"],
                         cost=v["cost"], cost_source=v.get("cost_source"),
                         tracker_realized_flop_avg_sl=v.get("tracker_cost"), tokens=v["tokens"],
                         eval_sec=v["sec"], hf_model=v["hf"], job=v["job"], log=v["log"],
                         trace=v.get("trace")))
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- SUMMARY.md --------------------------------------------------------
    L = []
    L.append("# prc2 — per-group allocation, calibrated on the deployment population\n")
    L.append("Built by `scmp_llm/benchmark/ppl/kbands/build_prc2_archive.py` (idempotent). "
             "See README.md for method, protocol and caveats.\n")
    L.append("## 1. Iso-cost: per-group (c17) vs submitted per-row (parent)\n")
    L.append("PPL = full wikitext-2 test, ctx 2048. cost = realized MAC-weighted SC stream "
             "length (halved) from each run's trace, incl. attention. Excess = share of the "
             "parent's gap to fp16 that the arm removes. v7 = the earlier per-group tables.\n")
    L.append("| model | T | parent PPL @cost | **c17 PPL @cost** | ΔPPL | Δcost | excess removed | v7 ΔPPL (Δcost) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    deltas, per_model, by_t = [], {m: [] for m in MODELS}, {}
    for mdl in MODELS:
        for t in TARGETS:
            p = res.get((mdl, t, "parent"))
            c = res.get((mdl, t, "c17"))
            v = res.get((mdl, t, "v7"))
            if not p:
                continue
            vtxt = (f"{(v['ppl']/p['ppl']-1)*100:+.2f}% ({(v['cost']/p['cost']-1)*100:+.1f}%)"
                    if v else "")
            if not c:
                L.append(f"| {mdl} | {t} | {p['ppl']:.4f} @{p['cost']:.2f} | *pending* | | | | {vtxt} |")
                continue
            dp = (c["ppl"] / p["ppl"] - 1) * 100
            dc = (c["cost"] / p["cost"] - 1) * 100
            ex = ((p["ppl"] - c["ppl"]) / (p["ppl"] - FP16[mdl]) * 100
                  if p["ppl"] - FP16[mdl] > 2 * NOISE_SD else float("nan"))
            deltas.append(dp)
            by_t.setdefault(t, []).append(dp)
            per_model[mdl].append(dp)
            L.append(f"| {mdl} | {t} | {p['ppl']:.4f} @{p['cost']:.2f} | **{c['ppl']:.4f}** @{c['cost']:.2f} | "
                     f"**{dp:+.2f}%** | {dc:+.1f}% | {ex:.0f}% | {vtxt} |")
    if deltas:
        L.append("\n**Per budget (finished cells):** " + "; ".join(
            f"t{t} mean {st.mean(v):+.2f}% (n={len(v)})" for t, v in sorted(by_t.items())) + ".")
        L.append("`excess removed` = n/a where the parent is within 2 noise-sd of fp16.")
        L.append(f"\n{len(deltas)}/20 cells done: mean ΔPPL {st.mean(deltas):+.2f}%, median "
                 f"{st.median(deltas):+.2f}%, improved {sum(x < 0 for x in deltas)}/{len(deltas)}. "
                 + "; ".join(f"{m} mean {st.mean(v):+.2f}% (n={len(v)})" for m, v in per_model.items() if v)
                 + f". Noise floor sd {NOISE_SD} PPL (a dispatcher-reshuffle spread, not run-to-run "
                 "noise: evals are bit-deterministic). Cost = trace (see README).\n")

    L.append("## 1b. FINAL algorithm: GLOBAL per-group allocation (Fisher currency) vs c17\n")
    L.append("Round 1 (`c6gfis`; `c7gfis` where only calib7 produced the cell) = one λ across every "
             "linear (row,chunk) group, priced by the diagonal Fisher of the operator output (MC "
             "labels); attention = parent. Round 2 (`c7gfisla`) = linears AND attention rows in the "
             "same λ. **final** = the better of the two on test per cell (user rule 2026-09-24); "
             "round 2 is test-evaluated only where it beat round 1 on held-out TRAIN windows (or in "
             "the first three probe cells). Same wrapper/mask/AWQ.\n")
    L.append("| cell | parent PPL @cost | c17 PPL @cost | round 1 @cost | round 2 @cost | "
             "**final** | vs parent | vs c17 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mdl in MODELS:
        for t in TARGETS:
            p, c = res.get((mdl, t, "parent")), res.get((mdl, t, "c17"))
            r1 = res.get((mdl, t, "c6gfis")) or res.get((mdl, t, "c7gfis"))
            r2 = res.get((mdl, t, "c7gfisla"))
            if not (p and (r1 or r2)):
                continue
            g = min([x for x in (r1, r2) if x], key=lambda x: x["ppl"])
            tag = "r2" if (r2 and g is r2) else "r1"
            f_ = lambda x: f"{x['ppl']:.4f} @{x['cost']:.2f}" if x else "—"
            L.append(f"| {mdl} t{t} | {f_(p)} | {f_(c)} | {f_(r1)} | {f_(r2)} | "
                     f"**{g['ppl']:.4f}** ({tag}) | **{(g['ppl']/p['ppl']-1)*100:+.2f}%** | "
                     + (f"{(g['ppl']/c['ppl']-1)*100:+.2f}%" if c else "") + " |")
    L.append("")
    L.append("## 2. Iso-PPL: SC cycles saved by per-group at the parent's quality\n")
    L.append("### 2a. Within-cell (MEASURED): the same calibration run at a reduced linear budget\n")
    L.append("`sNN` = the c17 calibration emitted at NN% of each bucket's budget (same windows, "
             "attention thresholds and mask). If sNN's PPL ≤ parent's, the saving is at least "
             "1 − cost(sNN)/cost(parent); otherwise it is bracketed between c17 and sNN (linear "
             "interpolation inside one cell).\n")
    L.append("| cell | parent PPL @cost | c17 PPL @cost | reduced-budget arm PPL @cost | SC cycles saved at parent PPL |")
    L.append("|---|---:|---:|---:|---:|")
    for mdl in MODELS:
        for t in TARGETS:
            p, c = res.get((mdl, t, "parent")), res.get((mdl, t, "c17"))
            if not (p and c):
                continue
            for (m_, t_, a), sv in sorted(res.items()):
                if m_ != mdl or t_ != t or not (a.startswith("c17_s") or a.startswith("c17iso_s")):
                    continue
                if sv["ppl"] <= p["ppl"]:
                    txt = f"≥ {(1 - sv['cost'] / p['cost']) * 100:.1f}% (measured)"
                elif c["ppl"] <= p["ppl"] <= sv["ppl"]:
                    w = (sv["ppl"] - p["ppl"]) / (sv["ppl"] - c["ppl"])
                    cst = sv["cost"] + w * (c["cost"] - sv["cost"])
                    txt = f"{(1 - cst / p['cost']) * 100:.1f}% (bracketed c17–{a})"
                else:
                    txt = "n/a"
                L.append(f"| {mdl} t{t} | {p['ppl']:.4f} @{p['cost']:.2f} | {c['ppl']:.4f} @{c['cost']:.2f} | "
                         f"{a}: {sv['ppl']:.4f} @{sv['cost']:.2f} | {txt} |")
    L.append("")
    L.append("### 2b. Cross-target curve (INDICATIVE ONLY)\n")
    L.append("Chords between c17 cells calibrated at DIFFERENT targets (their attention thresholds "
             "and, for some cells, INT masks differ), so this is not a pure-allocation number; "
             "conservative only where PPL(cost) is convex.\n")
    L.append("`≥` = even the cheapest measured per-group point of the model is at least as good.\n")
    L.append("| model | T | parent PPL @cost | per-group cost at same PPL | SC cycles saved |")
    L.append("|---|---:|---:|---:|---:|")
    for mdl in MODELS:
        pts = [(v["cost"], v["ppl"]) for (m, t, a), v in res.items()
               if m == mdl and a.startswith("c17")]
        for t in TARGETS:
            p = res.get((mdl, t, "parent"))
            if not p or not pts:
                continue
            s, kind = iso_ppl_saving(pts, p["ppl"], p["cost"])
            if s is None:
                txt = "n/a (" + kind + ")"
            else:
                txt = f"{'≥ ' if kind == 'lower_bound' else ''}{s*100:.1f}%"
            gc = f"{p['cost'] * (1 - s):.2f}" if s is not None else ""
            L.append(f"| {mdl} | {t} | {p['ppl']:.4f} @{p['cost']:.2f} | {gc} | {txt} |")
    L.append("")

    L.append("## 3. Granularity vs recalibration (per-ROW control, calibrated identically)\n")
    L.append("r17 = the same calibrator, windows, AWQ, dense ladder, per-bucket budget and "
             "solver, but every chunk of a row shares the row's rung (`SC_PRC_ROWSHARED=1`).\n")
    L.append("r17 is NOT cost-matched to c17 (see costs). `granularity (cost-adj.)` rescales r17 to "
             "c17's cost with the within-cell elasticity d ln PPL / d ln cost measured between c17 "
             "and its reduced-budget arm (an INFERENCE, one elasticity per cell).\n")
    L.append("| cell | parent | r17 per-row (same calib) | c17 per-(row,chunk) | recalibration only (raw) | **granularity only (raw)** | granularity (cost-adj., inferred) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for mdl in MODELS:
        for t in TARGETS:
            for ctl in ("r17", "r17m"):
                p, r, c = (res.get((mdl, t, a)) for a in ("parent", ctl, "c17"))
                if p and r and c:
                    import math
                    sarm = next((v for (m_, t_, a), v in res.items() if m_ == mdl and t_ == t
                                 and (a.startswith("c17_s") or a.startswith("c17iso_s"))), None)
                    adj = ""
                    if sarm and sarm["cost"] != c["cost"]:
                        el = (math.log(sarm["ppl"]) - math.log(c["ppl"])) / (math.log(sarm["cost"]) - math.log(c["cost"]))
                        r_adj = r["ppl"] * math.exp(el * (math.log(c["cost"]) - math.log(r["cost"])))
                        adj = f"{(c['ppl'] / r_adj - 1) * 100:+.2f}% (elasticity {el:.3f})"
                    L.append(f"| {mdl} t{t} | {p['ppl']:.4f} @{p['cost']:.2f} | {r['ppl']:.4f} @{r['cost']:.2f} | "
                             f"{c['ppl']:.4f} @{c['cost']:.2f} | {(r['ppl']/p['ppl']-1)*100:+.2f}% | "
                             f"**{(c['ppl']/r['ppl']-1)*100:+.2f}%** | {adj} |")
    L.append("")

    L.append("## 4. Held-out pre-flight (train windows, no test data)\n")
    L.append("Linear-projection squared error vs the parent at the calibrated cost, and the "
             "linear-budget scale at which prc2 matches the parent's error (from the sweep).\n")
    L.append("| cell | L new/par | err new/par | iso-error linear budget | gate |")
    L.append("|---|---:|---:|---:|:--|")
    for f in sorted(P2.glob("*_c17_diag.json")) + sorted(P2.glob("*_c17e32_diag.json")):
        dd = json.loads(f.read_text())
        ops = [k for k in dd["ops"] if not k.startswith("total_")]
        en = sum(dd["ops"][o]["new_err"] for o in ops)
        ep = sum(dd["ops"][o]["par_err"] for o in ops)
        lr = dd["ops"]["total_new_L"] / dd["ops"]["total_par_L"]
        sw = sorted((float(k), v["err_over_par"], v["L_over_par"])
                    for k, v in (dd.get("sweep_total") or {}).items())
        iso = None
        for (s0, e0, l0), (s1, e1, l1) in zip(sw, sw[1:]):
            if (e0 - 1) * (e1 - 1) <= 0 and e0 != e1:
                iso = l0 + (1 - e0) / (e1 - e0) * (l1 - l0)
        gate = "pass" if (0.97 <= lr <= 1.02 and en < ep) else "FAIL"
        L.append(f"| {f.name.replace('_diag.json','')} | {lr:.3f} | {en/ep:.3f} | "
                 f"{'%.2fx' % iso if iso else 'n/a'} | {gate} |")
    L.append("")
    stat = sorted(P2.glob("*_a3_diag.json")) + sorted(P2.glob("*_r17_diag.json"))
    if stat:
        L.append("## 5. Dispatch-statistic / granularity analysis (held-out err / parent)\n")
        L.append("| cell | today's stat | smoothed | absolute | per-block | per-row same-calib | oracle |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        seen = set()
        for f in stat:
            dd = json.loads(f.read_text())
            sa = dd.get("statistic_analysis")
            cell = f.name.split("_a3")[0].split("_r17")[0]
            if not sa or cell in seen:
                continue
            seen.add(cell)
            g = lambda k: f"{sa[k]['err_over_parent']:.3f}" if k in sa else ""
            L.append(f"| {cell} | {g('mn')} | {g('mn_sm')} | {g('raw')} | {g('mn_perblock')} | "
                     f"{g('row_same_calib')} | {g('oracle')} |")
        L.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # ---- checksums ---------------------------------------------------------
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in files))
    print(f"[prc2-archive] {len(res)} results, {len(deltas)} c17 cells -> {OUT}")


if __name__ == "__main__":
    main()
