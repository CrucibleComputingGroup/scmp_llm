#!/bin/bash
# INT%-ablation gap-fill: uniform SC at budget 96 (sc_avg192) + 20% mask at INT8.
# Needed so the budget-96 dose row is width-consistent (0% pure / 10% INT8 from
# uniform_hybrid / 20% INT8 here). Tonight's uni_hyb20 ran 96@20% at INT7 (for the
# MP comparison), which mixes widths against the INT8 10% cell. Uniform eval only.
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_int_ablation/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${U_MODEL:?U_MODEL is required (4B|llama8B|14B|30B)}"
SENS=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921

export PPL_MAX_TOKENS=0
export CTX=2048

echo "[int-abl 96i8] model=$U_MODEL  config=sc_avg192  frac=0.20  int_bits=8  tag=int_abl_96i8_20260724"
bash hpca --models "$U_MODEL" --configs sc_avg192 \
  --metrics ppl \
  --sc-backend hybrid --hybrid-int-frac 0.20 --hybrid-int-bits 8 \
  --hybrid-sensitivity-dir "$SENS" --hybrid-sensitivity-config sc_int8 \
  --tag int_abl_96i8_20260724
