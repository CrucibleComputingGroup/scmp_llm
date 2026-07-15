#!/bin/bash
# v7 = v6 (L2 targeted PC) + per-op SmoothQuant alpha on the MLP trio.
# Raising alpha on down/up/gate migrates their activation-outlier difficulty
# into the (offline-quantized) weights, attacking the BROADBAND residue that
# survives after the top-6% scale-setter channels are already protected —
# the part v6's PC saturated on. Runtime-free: scales resolve inside
# apply_smoothquant_to_model from SQ_ALPHA_OVERRIDES, so calibration and eval
# geometry match by construction. If the SC-encoded weights absorb the moved
# outliers poorly, this trades even or negative — that is what the screen
# measures (SmoothQuant scaling is known to cap out at low precision; QuaRot
# rotation remains the escalation).
# Usage: mp_v7_lane.sh <model> <tag> <configs-csv> <pc-frac-overrides> <sq-alpha-overrides>
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm

M="${1:?usage: mp_v7_lane.sh <model> <tag> <configs-csv> <pc-ov> <sq-ov>}"
TAG="${2:?usage: mp_v7_lane.sh <model> <tag> <configs-csv> <pc-ov> <sq-ov>}"
CFGS="${3:?usage: mp_v7_lane.sh <model> <tag> <configs-csv> <pc-ov> <sq-ov>}"
PCOV="${4:?usage: mp_v7_lane.sh <model> <tag> <configs-csv> <pc-ov> <sq-ov>}"
SQOV="${5:?usage: mp_v7_lane.sh <model> <tag> <configs-csv> <pc-ov> <sq-ov>}"
SENS_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

bash hpca --models "$M" --configs "$CFGS" \
  --metrics ppl --mp-method act_global_v3 --mp-protect-metric act_collapse \
  --mp-protect-frac-overrides "$PCOV" \
  --sq-alpha-overrides "$SQOV" \
  --sc-backend hybrid --hybrid-int-frac 0.10 \
  --hybrid-sensitivity-dir "$SENS_DIR" \
  --hybrid-sensitivity-config sc_int8 \
  --tag "$TAG"
