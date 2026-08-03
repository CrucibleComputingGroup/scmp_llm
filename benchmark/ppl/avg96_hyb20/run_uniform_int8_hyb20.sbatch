#!/bin/bash
# Uniform SC ceiling (sc_int8, 128 cyc) + 20% INT8 hybrid mask. INT8 (not the
# INT7 used for the aggressive-budget uniform grid) because at 128 cycles the SC
# already runs near-INT8 precision, so the masked entries must match at INT8 --
# this is the 20% version of the deployed t128 comparator (sc_int8 INT8 10%).
# Uniform eval only, no MP calibration. One job per model.
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

echo "[uniform int8 hyb20] model=$U_MODEL  config=sc_int8  frac=0.20  int_bits=8  tag=uni_hyb20_20260724"
bash hpca --models "$U_MODEL" --configs sc_int8 \
  --metrics ppl \
  --sc-backend hybrid --hybrid-int-frac 0.20 --hybrid-int-bits 8 \
  --hybrid-sensitivity-dir "$SENS" --hybrid-sensitivity-config sc_int8 \
  --tag uni_hyb20_20260724
