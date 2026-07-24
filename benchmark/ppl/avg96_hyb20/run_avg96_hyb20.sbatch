#!/bin/bash
# Fill the 20%-INT-mask grid at avg96 (target 96), 4 models.
# Reproduces the deployed 10% avg96 cell (mp_final_sweep) EXACTLY, changing only
# the hybrid INT fraction 0.10 -> 0.20. INT7 is forced via the legacy width table
# so the ONLY variable vs the 10% cell is the mask dose (the deployed 10% avg96
# cells are all INT7; the current hpca default would mask avg96 at INT8).
# One GPU per model. Full protocol (PPL_MAX_TOKENS=0, ctx 2048, SmoothQuant a=0.5).
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_avg96_hyb20/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${A96_MODEL:?A96_MODEL is required (4B|llama8B|14B|30B)}"
SENS=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

# Citability guards (full test split, no truncation, deploy-matched context).
export PPL_MAX_TOKENS=0
export CTX=2048

echo "[avg96 hyb20] model=$A96_MODEL  frac=0.20  int_bits=7(legacy)  tag=mp_avg96_hyb20_20260724"
bash hpca --models "$A96_MODEL" --configs mp_avg96f_burst128 \
  --metrics ppl --mp-method act_global_v3 --mp-protect-metric act_collapse \
  --mp-protect-frac-overrides down_proj:0.06,up_proj:0.03,gate_proj:0.03 \
  --sc-backend hybrid --hybrid-int-frac 0.20 \
  --hybrid-int-bits-mode legacy \
  --hybrid-sensitivity-dir "$SENS" --hybrid-sensitivity-config sc_int8 \
  --tag mp_avg96_hyb20_20260724
