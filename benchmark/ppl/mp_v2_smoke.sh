#!/bin/bash
# act_global_v2 GPU smoke (guardrail: validate calibrator changes on reduced
# settings BEFORE the full sweep). 4B / mp_avg64_burst128 / hybrid 10% INT +
# protected channels, PPL truncated to 65536 tokens (NON-CITABLE, smoke tag).
#
# Pass criteria (exit 0 + touch ~/mp_v2_smoke_passed):
#   1. calibration + eval complete (hpca cell exits clean, table exists);
#   2. expected_flop_avg_stoc_len within 2 of the 64 target (iso-compute holds
#      under the new argmin pricing);
#   3. truncated PPL < 14.0 (v1 full-protocol = 10.83; uniform sc_int7 = 16.94
#      — a broken allocation shows up as >>14);
#   4. log the per-op allocation (qk/av vs v1) + any metric-select switches.
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm

MARKER="$HOME/mp_v2_smoke_passed"
rm -f "$MARKER"
TAG="${1:?usage: mp_v2_smoke.sh <tag>}"
SENS_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

bash hpca --models 4B --configs mp_avg64_burst128 --metrics ppl \
  --mp-method act_global_v2 \
  --sc-backend hybrid --hybrid-int-frac 0.10 \
  --hybrid-sensitivity-dir "$SENS_DIR" \
  --hybrid-sensitivity-config sc_int8 \
  --max-tokens 65536 \
  --tag "$TAG"

TABLES=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_${TAG}
TABLE=$(ls "$TABLES"/Qwen_Qwen3-4B-Instruct-2507__mp_avg64_burst128__act_global_v2_*_hyb0.10.json 2>/dev/null | grep -v _wrapper | grep -v _summary | head -1)
RESULTS=/nfs/turbo/coe-nbleier/allenjin/hpca/results/results_${TAG}.tsv

python - "$TABLE" "$RESULTS" <<'PY'
import json, sys, csv, os
table_path, results_path = sys.argv[1], sys.argv[2]
if not table_path or not os.path.isfile(table_path):
    raise SystemExit(f"SMOKE FAIL: calibration table missing ({table_path!r})")
d = json.load(open(table_path))
flop_avg = float(d.get("expected_flop_avg_stoc_len", -1))
print(f"[smoke] expected_flop_avg_stoc_len = {flop_avg:.4f} (target 64)")
if abs(flop_avg - 64.0) > 2.0:
    raise SystemExit(f"SMOKE FAIL: flop-avg {flop_avg:.2f} off target 64 (>2)")
print(f"[smoke] argmin_pricing = {d.get('argmin_pricing')}")
print(f"[smoke] dispatch_metrics = {d.get('dispatch_metrics', {})}")
# per-op allocation from the summary csv (same basename + _summary.csv)
summ = table_path.replace(".json", "_summary.csv")
if os.path.isfile(summ):
    print("[smoke] operator_default avg_stoc_len (v1 4B avg64 reference: "
          "qk 111, av 111, k 126, v 128, o 77, q 59, down 63, up 46, gate 32):")
    for row in csv.DictReader(open(summ)):
        if row["scope"] == "operator_default":
            print(f"    {row['operator']:<11} {float(row['avg_stoc_len']):7.2f}"
                  f"  err={float(row['avg_error']):.4f}")
ppl = None
if os.path.isfile(results_path):
    for line in open(results_path):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 4 and f[2] == "ppl":
            ppl = float(f[3])
print(f"[smoke] truncated-65k PPL = {ppl}")
if ppl is None:
    raise SystemExit("SMOKE FAIL: no ppl row recorded")
if ppl > 14.0:
    raise SystemExit(f"SMOKE FAIL: ppl {ppl:.3f} > 14.0 gate")
print("SMOKE PASS")
PY
rc=$?
if [[ $rc -eq 0 ]]; then
  touch "$MARKER"
  echo "[smoke] marker touched: $MARKER"
fi
exit $rc
