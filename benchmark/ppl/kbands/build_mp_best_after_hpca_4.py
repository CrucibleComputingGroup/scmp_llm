"""Build hpca_results/llm/ppl/mp_best_after_hpca_4/ — the LOSS-PRICED archive.

WHAT IS NEW vs `_3`. `_3` selected over allocation/operand arms (parent / prc /
qk / prcqk / +grid) under the DEPLOYED hybrid INT mask. `_4` adds the two levers
measured 2026-08-07/08, both of which act on things `_3` held fixed:

  attnfirst  the INT mask RECOMPOSED by measured per-operator loss instead of
             the deployed SC-fragility proxy, MAC-MATCHED to the archive mask's
             INT MAC share (2 dp) so the INT side of the energy comparison is
             unchanged. Whole-(operator, block) granularity ONLY -- a matmul is
             entirely SC or entirely INT7, never mixed (hardware constraint,
             user 2026-08-08).
  reinv      the same mask run against a LOOSER table, because the allocator is
             mask-blind and therefore underspends 11-23% once the mask removes
             attention (the longest-stream rows) from the SC pool. Reinvesting
             that underspend is where most of the quality gain appears.
  m128       SC_SCRAMBLE_MASKS=128 (Owen mask population doubled). Cycle-neutral.
             **SIMULATION-ONLY**: HW_MAX_MASKS=64 is the silicon cap, so these
             are an ARCHITECTURE-PARAMETER result (a 7-bit mask selector), never
             a deployable cell at the current spec.
  qka05      qk rebalance at alpha=0.5 rather than the pinned 1.0.

TWO WINNERS PER CELL, deliberately:
  `winner`        best HARDWARE-REALIZABLE cell (M=64). This is the deployable
                  archive number and what the paper should quote.
  `winner_ideal`  best including M=128 simulation cells, flagged
                  `hw_realizable: false`. Never quote it as deployed.

TARGET KEYING. Cells are keyed by their table's NOMINAL target, as in `_3`, and
each carries its own traced `realized_flop_avg_sl`. A reinvestment cell legitimately
competes at its nominal target while running cheaper than it -- the archive has
always tolerated realized-cost drift and disclosed it in that column. Because the
strongest claims are of the form "beats a LOOSER archive cell at LESS compute",
SUMMARY.md also carries an ISO-COST FRONTIER table: per model, the best PPL at or
under each reference cost.

NOT INCLUDED as cells: the op-swap diagnostics (one operator forced entirely to
INT7). They move the INT MAC share far off the fixed dose, so they are not
deployable configurations; they are the MEASUREMENT that produced the mask, and
are recorded under `_op_swap_prices` in the manifest.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

RES = Path("/home/allenjin/Projects/SCMP/hpca_results/llm")
A3 = RES / "ppl" / "mp_best_after_hpca_3"
MPB = RES / "ppl" / "mp_best"
OUT = RES / "ppl" / "mp_best_after_hpca_4"
LOGS = Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/"
            "hpca/logs/_kbands")
PPLDIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl")
PRCDIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/prc")
QKDIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/qk")
MASKDIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/attnmask")

FP16 = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}
SENS = {"4B": 0.348, "llama8B": 0.279, "14B": 0.134, "30B": 0.401}
MODELS = ["4B", "llama8B", "14B", "30B"]
TARGETS = [32, 40, 48, 64, 96]

# (log prefix, label, uses M=128 -> not hardware realizable)
NEW_ARMS = [
    ("attnmask", "attnfirst", False),
    ("reinv", "attnfirst", False),          # same config family, looser table
    ("combo", "attnfirst+m128", True),
    ("reinvcombo", "attnfirst+m128", True),
    ("m128mp", "m128", True),
]
SWAP_OPS = ["qk", "av", "down_proj", "up_proj", "gate_proj",
            "o_proj", "q_proj", "k_proj", "v_proj"]


def _read(prefix: str, name: str):
    """(ppl, cost) from a job's own [RESULT] line, newest first, exact job-id match."""
    pat = re.compile(rf"{re.escape(prefix)}_{re.escape(name)}_\d+\.out")
    for f in sorted([q for q in LOGS.glob(f"{prefix}_{name}_*.out") if pat.fullmatch(q.name)],
                    key=lambda q: q.stat().st_mtime, reverse=True):
        for line in reversed(f.read_text(errors="ignore").splitlines()):
            if "[RESULT]" in line and "metric=ppl" in line:
                kv = dict(t.split("=", 1) for t in line.split() if "=" in t)
                try:
                    return float(kv["value"]), float(kv["realized_flop_avg_sl"])
                except Exception:                                # noqa: BLE001
                    pass
    return None, None


def op_swap_prices():
    """Per-operator measured loss, the instrument behind the mask.

    value = dPPL% - (-dcost% * sensitivity): the loss removed per unit of
    compute FREED. Ranking on raw dPPL builds the wrong mask (4B's two
    smallest raw effects are its two best targets).
    """
    out = {}
    for m in MODELS:
        for t in (32, 96):
            base = _read("prcppl", f"{m}_t{t}_prcqk")
            if base[0] is None:
                base = _read("prcppl", f"{m}_t{t}_prc")
            if base[0] is None:
                continue
            tbl = {}
            for op in SWAP_OPS:
                v = _read("opswap", f"{m}_t{t}_{op}")
                if v[0] is None:
                    continue
                dppl = (v[0] - base[0]) / base[0] * 100.0
                dcost = (v[1] - base[1]) / base[1] * 100.0
                tbl[op] = {"ppl": v[0], "cost": v[1], "d_ppl_pct": dppl,
                           "d_cost_pct": dcost,
                           "value_pct": dppl + dcost * SENS[m]}
            if tbl:
                out[f"{m}/t{t}"] = {"baseline_ppl": base[0], "baseline_cost": base[1],
                                    "ops": tbl}
    return out


def main() -> int:
    a3 = json.loads((A3 / "manifest.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("configs", "prc_tables", "qk_scales", "masks", "traces"):
        (OUT / sub).mkdir(exist_ok=True)

    manifest, rows, sim_rows = {}, [], []
    for m in MODELS:
        for t in TARGETS:
            key = f"{m}/target{t}"
            prev = a3.get(key) or {}
            cands = []                       # (label, ppl, cost, hw_ok)
            if prev.get("ppl"):
                cands.append((f"a3:{prev.get('winner')}", prev["ppl"],
                              prev.get("realized_flop_avg_sl"), True))
            for pref, lab, sim in NEW_ARMS:
                v = _read(pref, f"{m}_t{t}")
                if v[0] is not None:
                    cands.append((lab, v[0], v[1], not sim))
            for arm in ("prcqk", "prc"):     # qk alpha=0.5 variants
                v = _read("qka05", f"{m}_t{t}_{arm}")
                if v[0] is not None:
                    cands.append((f"qk-a0.5({arm})", v[0], v[1], True))
            if not cands:
                continue

            hw = [c for c in cands if c[3]]
            lab, ppl, cost, _ = min(hw, key=lambda r: r[1])
            ilab, ippl, icost, _ = min(cands, key=lambda r: r[1])
            old = prev.get("ppl")
            oldc = prev.get("realized_flop_avg_sl")
            x = ppl / FP16[m]
            adj = None
            if old and oldc and cost:
                adj = ((ppl - old) / old * 100.0
                       + (cost - oldc) / oldc * 100.0 * SENS[m])
            manifest[key] = {
                "model": m, "target": t,
                "winner": lab, "ppl": ppl, "x_fp16": x, "passes_1p05": x <= 1.05,
                "realized_flop_avg_sl": cost, "hw_realizable": True,
                "winner_ideal": ilab, "ppl_ideal": ippl,
                "x_fp16_ideal": ippl / FP16[m],
                "realized_flop_avg_sl_ideal": icost,
                "ideal_hw_realizable": ilab == lab,
                "prev_a3_winner": prev.get("winner"), "prev_a3_ppl": old,
                "prev_a3_cost": oldc,
                "delta_vs_a3_pct": ((ppl - old) / old * 100.0) if old else None,
                "cost_delta_vs_a3_pct": ((cost - oldc) / oldc * 100.0) if (oldc and cost) else None,
                "cost_adjusted_value_pct": adj,
                "all_arms": {a: {"ppl": p, "cost": c, "hw": h} for a, p, c, h in cands},
            }
            rows.append((m, t, old, oldc, ppl, cost, lab, x))
            if ilab != lab:
                sim_rows.append((m, t, ppl, ippl, ilab, icost))

            # ---- artifacts ----
            dst = OUT / "configs" / m / f"target{t}"
            dst.mkdir(parents=True, exist_ok=True)
            src = MPB / "configs" / m / f"target{t}"
            if src.is_dir():
                for f in src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)
            if "attnfirst" in lab:
                # _mm = the MAC-matched masks that PRODUCED these results.
                # The plain name was later reused for the entry-matched
                # standard (dose = 20% of matmuls, user 2026-08-08), so pinning
                # _mm keeps a rebuild faithful instead of pairing new masks with
                # old numbers.
                msk = MASKDIR / f"{m}_t{t}_attnfirst_mm.json"
                if msk.is_file():
                    shutil.copy2(msk, OUT / "masks" / msk.name)
                    shutil.copy2(msk, dst / "hybrid_config.json")   # the DEPLOYED mask
            for pat in (f"{m}_t{t}_v7_prc.json", f"{m}_t{t}_v7_prc_table.json"):
                if (PRCDIR / pat).is_file():
                    shutil.copy2(PRCDIR / pat, OUT / "prc_tables" / pat)
            for a in ("1.0", "0.5"):
                q = QKDIR / f"{m}_t{t}_qk_alpha{a}.json"
                if q.is_file():
                    shutil.copy2(q, OUT / "qk_scales" / q.name)
            for c in sorted(PPLDIR.glob(f"{m}_t{t}_*_trace.json"),
                            key=lambda q: q.stat().st_mtime, reverse=True):
                if any(k in c.name for k in ("attnfirst", "combo", "reinv48", "qka05")):
                    shutil.copy2(c, OUT / "traces" / c.name)
                    manifest[key].setdefault("traces", []).append(c.name)

    manifest["_op_swap_prices"] = op_swap_prices()
    manifest["_uniform_ceiling_t128"] = a3.get("_uniform_ceiling_t128", {})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # ---------- iso-cost frontier: best PPL at or under each reference cost ----------
    frontier = {}
    for m in MODELS:
        ref = [(t, a3.get(f"{m}/target{t}", {}).get("realized_flop_avg_sl"),
                a3.get(f"{m}/target{t}", {}).get("ppl")) for t in TARGETS]
        for t, rc, rp in ref:
            if not rc:
                continue
            best = None
            for k, v in manifest.items():
                if not k.startswith(f"{m}/target"):
                    continue
                for a, d in v["all_arms"].items():
                    if d["hw"] and d["cost"] and d["cost"] <= rc * 1.005:
                        if best is None or d["ppl"] < best[0]:
                            best = (d["ppl"], d["cost"], f"{k.split('target')[1]}:{a}")
            if best:
                frontier[f"{m}@t{t}"] = {
                    "ref_cost": rc, "a3_ppl": rp, "best_ppl": best[0],
                    "best_cost": best[1], "config": best[2],
                    "delta_pct": (best[0] - rp) / rp * 100.0 if rp else None}
    manifest["_iso_cost_frontier"] = frontier
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    imp = [(p - o) / o * 100 for _, _, o, _, p, _, _, _ in rows if o]
    npass = sum(1 for k, v in manifest.items() if "/target" in k and v["passes_1p05"])
    nprev = sum(1 for k, v in a3.items() if "/target" in k and v.get("passes_1p05"))

    L = ["# mp_best_after_hpca_4 — the LOSS-PRICED archive", "",
         f"{len(rows)} MP cells (4 models x t{{{','.join(map(str,TARGETS))}}}). "
         "Supersedes `_3`, which is a candidate here, so no cell can regress.", "",
         "**What changed:** `_3` optimized ALLOCATION under the deployed INT mask. "
         "`_4` also recomposes the MASK by MEASURED per-operator loss "
         "(MAC-matched to the same INT MAC share) and reinvests the compute that "
         "frees. Mask granularity is whole-(operator, block) — a matmul is entirely "
         "SC or entirely INT7, never mixed.", "",
         "| model | tgt | _3 PPL @cost | **_4 PPL @cost** | winner | ΔPPL | Δcost | x_fp16 | 1.05x |",
         "|---|---:|---:|---:|---|---:|---:|---:|:--|"]
    for m, t, o, oc, p, c, lab, x in rows:
        # `_3` records a null cost for cells it carried without a trace (4B t48),
        # so every column that touches it must tolerate None rather than crash
        # the build after the artifacts are already written.
        d = f"{(p-o)/o*100:+.2f}%" if o else "—"
        dc = f"{(c-oc)/oc*100:+.1f}%" if (o and oc and c) else "—"
        oldcell = f"{o:.4f} @{oc:.1f}" if (o and oc) else (f"{o:.4f} @—" if o else "—")
        newcell = f"**{p:.4f} @{c:.1f}**" if c else f"**{p:.4f}** @—"
        L.append(f"| {m} | {t} | {oldcell} | {newcell} | {lab} | "
                 f"{d} | {dc} | {x:.4f} | {'**PASS**' if x <= 1.05 else ''} |")
    L += ["", f"**Mean {sum(imp)/len(imp):+.2f}%** vs `_3` over {len(imp)} cells. "
              f"Cells at or under 1.05x fp16: **{nprev} -> {npass}**.", ""]

    L += ["## Iso-cost frontier — best PPL at or under each `_3` cell's own cost", "",
          "This is where the headline claims live: a cell built for a LOOSER budget "
          "that realizes LESS compute than a tighter `_3` cell, because the mask "
          "evicted attention (the longest-stream rows) from the SC pool.", "",
          "| model @ ref | ref cost | `_3` PPL | best `_4` PPL @cost | config | Δ |",
          "|---|---:|---:|---:|---|---:|"]
    for k, v in frontier.items():
        if v["delta_pct"] is None:
            continue
        L.append(f"| {k} | {v['ref_cost']:.1f} | {v['a3_ppl']:.4f} | "
                 f"**{v['best_ppl']:.4f}** @{v['best_cost']:.1f} | {v['config']} | "
                 f"{v['delta_pct']:+.2f}% |")

    if sim_rows:
        L += ["", "## Simulation-only cells (NOT deployable, recorded for the "
              "architecture question)", "",
              "`SC_SCRAMBLE_MASKS=128` doubles the Owen mask population at identical "
              "cycles. `HW_MAX_MASKS=64` is the silicon cap, so these need a 7-bit "
              "mask selector. Listed as `winner_ideal` in the manifest; the `winner` "
              "column above is always hardware-realizable.", "",
              "| model | tgt | deployable | ideal (M=128) | config |",
              "|---|---:|---:|---:|---|"]
        for m, t, p, ip, il, ic in sim_rows:
            L.append(f"| {m} | {t} | {p:.4f} | **{ip:.4f}** | {il} |")

    L += ["", "## The instrument: measured per-operator loss (`_op_swap_prices`)", "",
          "Route ONE operator entirely to INT7, read full-protocol PPL. The mask is "
          "then composed by **ΔL per unit of compute freed**, not raw ΔL — on 4B the "
          "two operators with the SMALLEST raw effect (av, qk) are the two BEST "
          "masking targets, while the two largest (up/down_proj) are worthless once "
          "their compute is priced. These swap cells are NOT deployable "
          "configurations (they move the INT MAC share far off the fixed dose); they "
          "are the measurement behind the masks.", "",
          "## Caveats", "",
          "* **Attention runs on INT7 in every `attnfirst` cell**, i.e. SC executes "
          "no attention there. The 20% INT dose is fixed by decree and only its "
          "COMPOSITION changed (MAC-matched to 2 dp), so the energy comparison is "
          "fair on the INT side — but any 'SC reaches Nx' claim must state it.",
          "* Costs come from each run's own TRACE, which prices per rung.",
          "* Noise floor sd 0.0069 PPL; |delta| < ~0.014 absolute is not a result.",
          "* The allocator is MASK-BLIND: it does not know the mask removed whole "
          "matmuls from its pool, which is why these cells underspend and why "
          "`reinv` (same mask, looser table) recovers it. A mask-aware allocator "
          "would subsume that by construction — un-built.",
          "* Full ledger incl. every refutation and my own failed prediction: "
          "`scmp_llm/benchmark/ppl/kbands/LOOP_STATE.md`; narrative summary: "
          "`MORNING_RESULTS_20260808.md`."]
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")

    print(f"wrote {OUT} ({len(rows)} cells)")
    for m, t, o, oc, p, c, lab, x in rows:
        flag = "  PASS" if x <= 1.05 else ""
        dv = f"{(p-o)/o*100:+.2f}%" if o else "  —  "
        cs = f"@{c:6.2f}" if c else "@  n/a"
        print(f"  {m:9s} t{t:<3d} {lab:16s} {p:9.4f} {cs}  x{x:.4f}  vs_3 {dv}{flag}")
    print(f"\nmean {sum(imp)/len(imp):+.2f}% vs _3 | passing 1.05x {nprev} -> {npass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
