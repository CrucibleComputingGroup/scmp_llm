#!/bin/bash
# v5 cliff-aware-objective lane (citable, full protocol). Same v3 pipeline
# (hybrid 10% INT mask + act_collapse protected channels + MAC-weighted budget)
# but with a CLIFF-AWARE MP objective instead of plain relative-L2 sigma:
#   act_global_v3s = --objective sigma2      (per-ROW, superlinear penalty)
#   act_global_v3c = --cross-layer-weight measured_curve (per-group measured dLoss)
# Motivation: v4 proved a 16-rung under the sigma objective is toxic because
# relative-L2 sigma understates deep-sub-cliff loss (it dumped e.g. 30B o_proj
# 100% -> 16, err 0.328, to fund down_proj upgrades). This lane provides the
# cliff-aware objective the v4 learning said must come FIRST, then re-tests the
# sub-cliff (16) rung on the *x ladders.
# Usage: mp_v5_lane.sh <model> <method> <tag> <configs-csv>
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm

M="${1:?usage: mp_v5_lane.sh <model> <method> <tag> <configs-csv>}"
METHOD="${2:?usage: mp_v5_lane.sh <model> <method> <tag> <configs-csv>}"
TAG="${3:?usage: mp_v5_lane.sh <model> <method> <tag> <configs-csv>}"
CFGS="${4:?usage: mp_v5_lane.sh <model> <method> <tag> <configs-csv>}"
SENS_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

bash hpca --models "$M" --configs "$CFGS" \
  --metrics ppl --mp-method "$METHOD" --mp-protect-metric act_collapse \
  --sc-backend hybrid --hybrid-int-frac 0.10 \
  --hybrid-sensitivity-dir "$SENS_DIR" \
  --hybrid-sensitivity-config sc_int8 \
  --tag "$TAG"
