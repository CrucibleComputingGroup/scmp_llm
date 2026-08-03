#!/bin/bash
# Complete the AWQ mask-dose grid: h in {0,10,20} x {256,192,128} nominal x 4
# models = 36 cells. 24 already existed, 6 are filled by run_dose_awq_h20.sbatch
# (the h=20% gaps), and this script runs the remaining 6:
#
#   h=10% at 192 / 128        -- 4B, llama8B, 30B only; 14B already deploys 10%
#                                there, so its cells are already measured.
#
# h=5% is deliberately NOT run: the dose grid is three points, not four. The
# four-point curve exists only on SmoothQuant, in hpca_results/llm/int_ablation/.
# 14B needs nothing here, so it gets no job in this wave.
#
# Uniform SC throughout (allocator OFF) so the grid isolates the mask, matching
# the SmoothQuant dose ablation in hpca_results/llm/int_ablation/.
# Widths follow the deployed convention: INT8 at 256, INT7 at 192 and 128.
# Mask source is the one ranking every other cell uses, so the doses stay
# strictly nested (5% subset of 10% subset of 20%).
# Full protocol: full wikitext-2, ctx 2048, PPL_MAX_TOKENS=0. AWQ scales cached.
#
# Submitted with --dependency=afterany on the matching h20 job so the two waves
# never load the same model on one node at the same time.
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_dose_awq_fill/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${U_MODEL:?U_MODEL is required (4B|llama8B|14B|30B)}"
SENS=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921
TAG=dose_awq_20260730

export PPL_MAX_TOKENS=0
export CTX=2048
export FRONTEND=awq
export AWQ_OBJ_BITS="${AWQ_OBJ_BITS:-4}"
export AWQ_TOK_CAP="${AWQ_TOK_CAP:-128}"

echo "[dose-awq-fill] model=$U_MODEL FRONTEND=awq obj_bits=$AWQ_OBJ_BITS tok_cap=$AWQ_TOK_CAP tag=$TAG"

# --- h=10% at 192 + 128 (14B already has these from its deployed config) -----
if [ "$U_MODEL" = "14B" ]; then
  echo "[dose-awq-fill] 14B needs no cells in this wave; exiting"
  exit 0
fi
bash hpca --models "$U_MODEL" --configs sc_avg192,sc_int7 --metrics ppl \
  --sc-backend hybrid --hybrid-int-frac 0.10 --hybrid-int-bits 7 \
  --hybrid-sensitivity-dir "$SENS" --hybrid-sensitivity-config sc_int8 --tag "$TAG"

echo "[dose-awq-fill] done model=$U_MODEL"
