#!/bin/bash
# One act_global_v3 full-protocol model lane (citable): hybrid 10% INT +
# v3 MP (KKT pricing + rho-selected metric + MAC-consistent fill) +
# act_collapse protected channels, at the configs given.
# Usage: mp_v3_lane.sh <model> <tag> <configs-csv>
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm

M="${1:?usage: mp_v3_lane.sh <model> <tag> <configs-csv>}"
TAG="${2:?usage: mp_v3_lane.sh <model> <tag> <configs-csv>}"
CFGS="${3:?usage: mp_v3_lane.sh <model> <tag> <configs-csv>}"
SENS_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

bash hpca --models "$M" --configs "$CFGS" \
  --metrics ppl --mp-method act_global_v3 --mp-protect-metric act_collapse \
  --sc-backend hybrid --hybrid-int-frac 0.10 \
  --hybrid-sensitivity-dir "$SENS_DIR" \
  --hybrid-sensitivity-config sc_int8 \
  --tag "$TAG"
