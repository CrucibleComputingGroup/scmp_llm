#!/bin/bash
# Uniform SC + 20% INT7 hybrid mask = the fair same-mask comparator for the 20%
# MP grid. Mirrors the uniform_hybrid launch (2026-07-09) but at frac 0.20 and
# INT7 forced (so the mask is byte-identical to the deployed INT7 20% MP cells,
# isolating the per-row allocator). Uniform eval only, no MP calibration.
# One job per model; four SC budgets that mirror MP t32/t48/t64/avg96.
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

: "${U_MODEL:?U_MODEL is required (4B|llama8B|14B|30B)}"
SENS=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

export PPL_MAX_TOKENS=0
export CTX=2048

echo "[uniform hyb20] model=$U_MODEL  frac=0.20  int_bits=7  configs=sc_int6,sc_avg96,sc_int7,sc_avg192  tag=uni_hyb20_20260724"
bash hpca --models "$U_MODEL" --configs sc_int6,sc_avg96,sc_int7,sc_avg192 \
  --metrics ppl \
  --sc-backend hybrid --hybrid-int-frac 0.20 --hybrid-int-bits 7 \
  --hybrid-sensitivity-dir "$SENS" --hybrid-sensitivity-config sc_int8 \
  --tag uni_hyb20_20260724
