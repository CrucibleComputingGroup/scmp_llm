#!/bin/bash
# INT%-ablation: the 5% dose point, uniform SC at budgets 64 / 96 / 128.
# Per-budget INT width held fixed to match the rest of the ablation:
#   64  -> sc_int7   @ INT7
#   96  -> sc_avg192 @ INT8
#   128 -> sc_int8   @ INT8
# Two hpca calls (INT7 then INT8) because a single call pins one width.
# Uniform eval only, no MP calibration. One job per model, same tag.
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_int_ablation/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${U_MODEL:?U_MODEL is required (4B|llama8B|14B|30B)}"
SENS=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921
TAG=int_abl_5pct_20260724

export PPL_MAX_TOKENS=0
export CTX=2048

echo "[int-abl 5pct] model=$U_MODEL  frac=0.05  tag=$TAG"

# budget 64 -> INT7
bash hpca --models "$U_MODEL" --configs sc_int7 --metrics ppl \
  --sc-backend hybrid --hybrid-int-frac 0.05 --hybrid-int-bits 7 \
  --hybrid-sensitivity-dir "$SENS" --hybrid-sensitivity-config sc_int8 --tag "$TAG"

# budgets 96 + 128 -> INT8
bash hpca --models "$U_MODEL" --configs sc_avg192,sc_int8 --metrics ppl \
  --sc-backend hybrid --hybrid-int-frac 0.05 --hybrid-int-bits 8 \
  --hybrid-sensitivity-dir "$SENS" --hybrid-sensitivity-config sc_int8 --tag "$TAG"
