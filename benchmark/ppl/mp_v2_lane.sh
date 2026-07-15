#!/bin/bash
# One act_global_v2 full-protocol model lane: hybrid 10% INT + protected
# channels + v2 MP at all three budgets, FULL wikitext-2 PPL (citable).
# Gated on the smoke marker (belt) in addition to the sbatch afterok
# dependency (suspenders). Usage: mp_v2_lane.sh <model> <tag>
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm

M="${1:?usage: mp_v2_lane.sh <model> <tag>}"
TAG="${2:?usage: mp_v2_lane.sh <model> <tag>}"
if [[ ! -f "$HOME/mp_v2_smoke_passed" ]]; then
  echo "!!! smoke marker missing (~/mp_v2_smoke_passed) — refusing to run"
  exit 9
fi
SENS_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

bash hpca --models "$M" \
  --configs mp_avg96_burst128,mp_avg64_burst128,mp_avg48_burst128 \
  --metrics ppl --mp-method act_global_v2 \
  --sc-backend hybrid --hybrid-int-frac 0.10 \
  --hybrid-sensitivity-dir "$SENS_DIR" \
  --hybrid-sensitivity-config sc_int8 \
  --tag "$TAG"
