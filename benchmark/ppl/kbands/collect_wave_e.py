#!/usr/bin/env python3
"""Collect the 2026-08-07/08 diagnostic + Wave E/E5/E6 results into one table.

Reads the [RESULT] line and realized cost from each job log, joins against the
mp_best_after_hpca_3 archive baselines, and reports raw + cost-adjusted deltas.

Cost-adjustment (project convention): compute_part = -dcost_pct * sensitivity;
value = dppl_pct - compute_part. Sensitivities were fitted over a ~10% cost
range, so flag any row whose |dcost| exceeds that as an extrapolation.

  python benchmark/ppl/kbands/collect_wave_e.py
"""
import json, re, pathlib, collections

LOGS = pathlib.Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands")
ARCH = pathlib.Path("/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best_after_hpca_3")
SENS = {"4B": 0.348, "llama8B": 0.279, "14B": 0.134, "30B": 0.401}
FP16 = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}
PREFIXES = ("opswap_", "attnmask_", "m128mp_", "combo_", "reinv_", "reinvcombo_",
            "qka05_", "sqa085_", "mcmask_", "m128_")
RES = re.compile(r"value=([0-9.]+).*?realized_flop_avg_sl=([0-9.]+)")


def baselines():
    m = json.loads((ARCH / "manifest.json").read_text())
    out = {}
    for k, c in m.items():
        if "/target" in k:
            model, t = k.split("/target")
            out[(model, int(t))] = (c["ppl"], c.get("realized_flop_avg_sl"), c["winner"])
    return out


def main():
    base = baselines()
    rows = []
    for f in sorted(LOGS.glob("*.out")):
        name = f.name
        if not name.startswith(PREFIXES):
            continue
        txt = f.read_text(errors="ignore")
        hit = None
        for line in txt.splitlines():
            if "[RESULT]" in line:
                hit = RES.search(line)
        if not hit:
            continue
        ppl, cost = float(hit.group(1)), float(hit.group(2))
        stem = name.rsplit("_", 1)[0]
        parts = stem.split("_")
        model = next((p for p in parts if p in SENS), None)
        tgt = next((int(p[1:]) for p in parts if re.fullmatch(r"t\d+", p)), None)
        if model is None:
            continue
        rows.append(dict(job=stem, model=model, target=tgt, ppl=ppl, cost=cost))

    print(f"{'cell':<34}{'PPL':>10}{'x_fp16':>9}{'cost':>8}"
          f"{'dPPL%':>8}{'dcost%':>8}{'value%':>8}  vs")
    for r in sorted(rows, key=lambda x: (x["model"], x["target"] or 0, x["job"])):
        m, t = r["model"], r["target"]
        b = base.get((m, t))
        x = r["ppl"] / FP16[m]
        if not b or not b[1]:
            print(f"{r['job']:<34}{r['ppl']:>10.4f}{x:>9.4f}{r['cost']:>8.2f}"
                  f"{'':>24}  (no baseline)")
            continue
        bp, bc, bw = b
        dppl = 100 * (r["ppl"] - bp) / bp
        dcost = 100 * (r["cost"] - bc) / bc
        value = dppl - (-dcost * SENS[m])
        flag = " *extrap" if abs(dcost) > 10 else ""
        print(f"{r['job']:<34}{r['ppl']:>10.4f}{x:>9.4f}{r['cost']:>8.2f}"
              f"{dppl:>8.2f}{dcost:>8.2f}{value:>8.2f}  {bw}@t{t}{flag}")
    print("\n* value = dPPL - compute_part; compute_part = -dcost x sensitivity")
    print("* rows marked *extrap exceed the ~10% cost range the sensitivities "
          "were fitted over -- read the raw columns as the result.")
    print("* noise floor sd 0.0069 PPL; |dPPL| under ~0.014 absolute is not a result.")


if __name__ == "__main__":
    main()
