#!/bin/bash
# Targeted protected-channel lane (citable, full protocol). v3 recipe + per-op
# protected-channel frac OVERRIDES on the outlier-heavy MLP trio (SpQR/LLM.int8
# style: pin the scale-setting amax channels to 128, pull them out of the short
# residual stream). Tests whether isolating MORE down/up/gate outlier channels
# lowers their sub-cliff sigma at iso-compute (budget compensation reads the
# realized per-op frac). If sigma plateaus as frac rises -> the error is
# broadband, not outlier-concentrated -> rotation (QuaRot) is the needed fix.
# Usage: mp_pc_lane.sh <model> <tag> <configs-csv> <frac-overrides>
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm

M="${1:?usage: mp_pc_lane.sh <model> <tag> <configs-csv> <frac-overrides>}"
TAG="${2:?usage: mp_pc_lane.sh <model> <tag> <configs-csv> <frac-overrides>}"
CFGS="${3:?usage: mp_pc_lane.sh <model> <tag> <configs-csv> <frac-overrides>}"
OV="${4:?usage: mp_pc_lane.sh <model> <tag> <configs-csv> <frac-overrides>}"
SENS_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

bash hpca --models "$M" --configs "$CFGS" \
  --metrics ppl --mp-method act_global_v3 --mp-protect-metric act_collapse \
  --mp-protect-frac-overrides "$OV" \
  --sc-backend hybrid --hybrid-int-frac 0.10 \
  --hybrid-sensitivity-dir "$SENS_DIR" \
  --hybrid-sensitivity-config sc_int8 \
  --tag "$TAG"
