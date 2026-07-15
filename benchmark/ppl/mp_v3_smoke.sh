#!/bin/bash
# act_global_v3 smoke: 4B x {mp_avg48f_burst128, mp_avg32_burst128}, FULL
# protocol (citable — user rule: always run citable results; rows and traces
# are directly comparable/mergeable with the sweep). Decides whether the v3
# levers earn the full 4-model sweep:
#   gate 1: calibration completes with the 5-level ladders; budget holds
#           (expected_flop_avg within 2 of 48 / 32);
#   gate 2: avg48f PPL <= 11.71 (v2 avg48 full — v3 must at least match v2;
#           the real target is 11.05 = 1.10x fp16);
#   gate 3: avg32 PPL < 16.94   (uniform-int7 full; collapse toward
#           uniform-int6 (304) kills the avg32 point).
# fp16 4B = 10.0445.
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm

MARKER="$HOME/mp_v3_smoke_passed"
rm -f "$MARKER"
TAG="${1:?usage: mp_v3_smoke.sh <tag>}"
SENS_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

bash hpca --models 4B --configs mp_avg48f_burst128,mp_avg32_burst128 \
  --metrics ppl --mp-method act_global_v3 --mp-protect-metric act_collapse \
  --sc-backend hybrid --hybrid-int-frac 0.10 \
  --hybrid-sensitivity-dir "$SENS_DIR" \
  --hybrid-sensitivity-config sc_int8 \
  --tag "$TAG"

RESULTS=/nfs/turbo/coe-nbleier/allenjin/hpca/results/results_${TAG}.tsv
TABLES=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_${TAG}
python - "$RESULTS" "$TABLES" <<'PY'
import csv, glob, json, os, sys
results, tables = sys.argv[1], sys.argv[2]
ppl, flop = {}, {}
for line in open(results):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 4 and f[0] == "4B":
        if f[2] == "ppl":
            ppl[f[1]] = float(f[3])
        if f[2] == "mp_expected_flop_avg_sl":
            flop[f[1]] = float(f[3])
print(f"[smoke] ppl={ppl} expected_flop={flop}")
for tab in sorted(glob.glob(os.path.join(tables, "*hyb0.10.json"))):
    if "_wrapper" in tab or "_summary" in tab:
        continue
    d = json.load(open(tab))
    cfg = os.path.basename(tab).split("__")[1]
    print(f"[smoke] {cfg}: refine={d.get('refine_stats',{})} "
          f"dispatch_metrics={list((d.get('dispatch_metrics') or {}).keys())}")
    summ = tab.replace(".json", "_summary.csv")
    if os.path.isfile(summ):
        ops = [r for r in csv.DictReader(open(summ))
               if r["scope"] == "operator_default"]
        print("    alloc: " + "  ".join(
            f"{r['operator']}:{float(r['avg_stoc_len']):.0f}" for r in ops))
fails = []
t48 = ppl.get("mp_avg48f_burst128"); t32 = ppl.get("mp_avg32_burst128")
f48 = flop.get("mp_avg48f_burst128"); f32 = flop.get("mp_avg32_burst128")
if t48 is None or t32 is None:
    fails.append("missing ppl rows")
if f48 is not None and abs(f48 - 48.0) > 2.0:
    fails.append(f"avg48f flop-avg {f48:.2f} off target")
if f32 is not None and abs(f32 - 32.0) > 2.0:
    fails.append(f"avg32 flop-avg {f32:.2f} off target")
if t48 is not None and t48 > 11.71:
    fails.append(f"avg48f ppl {t48:.3f} > 11.71 gate (v2 avg48 full)")
if t32 is not None and t32 > 16.94:
    fails.append(f"avg32 ppl {t32:.3f} > 16.94 gate (uniform-int7 full)")
if fails:
    raise SystemExit("SMOKE FAIL: " + "; ".join(fails))
print("SMOKE PASS")
PY
rc=$?
[[ $rc -eq 0 ]] && touch "$MARKER" && echo "[smoke] marker: $MARKER"
exit $rc
