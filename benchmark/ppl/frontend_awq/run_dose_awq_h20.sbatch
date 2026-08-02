#!/bin/bash
# Make the AWQ mask dose UNIFORM at h=20% across every budget in the paper's
# quality tables. Two gaps are filled:
#
#   (a) 20% @ nominal 256 (sc_int8, INT8) on ALL FOUR models. The 256 row today
#       is the mp_best target-128 cell, which is uniform_sc mode at INT8 *10%* --
#       the AWQ uniform-comparator wave (uac_*) only covered t32/48/64/96, so a
#       20% cell at 256 was never run under AWQ.
#   (b) 20% @ nominal 192 and 128 (INT7) on 14B ONLY. Every other model already
#       deploys 20% at those budgets; 14B deploys 10% because 10% won its
#       SmoothQuant dose sweep, which makes it the lone dose outlier in the table.
#
# Uniform SC only (allocator OFF) -- these are the "Uniform SC" comparator rows.
# The matching 14B MP@20% cells ALREADY EXIST under AWQ and are not re-run here:
#   t96 -> 8.6418 @ 96.14   (mba_14B_t96_54972413.out)
#   t64 -> 8.7758 @ 64.01   (awq14bt64_55367988.out, better of two runs)
#
# Mask source is the SAME ranking every other cell uses, so the 20% mask is a
# strict superset of the 10% one it replaces.
# Full protocol: full wikitext-2, ctx 2048, PPL_MAX_TOKENS=0. AWQ scales are
# already cached (b4) for all four models, so no calibration cost.
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_dose_awq_h20/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${U_MODEL:?U_MODEL is required (4B|llama8B|14B|30B)}"
SENS=/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921
TAG=dose_awq_h20_20260730

export PPL_MAX_TOKENS=0
export CTX=2048
export FRONTEND=awq
export AWQ_OBJ_BITS="${AWQ_OBJ_BITS:-4}"
export AWQ_TOK_CAP="${AWQ_TOK_CAP:-128}"

echo "[dose-awq-h20] model=$U_MODEL frac=0.20 FRONTEND=awq obj_bits=$AWQ_OBJ_BITS tok_cap=$AWQ_TOK_CAP tag=$TAG"

# (a) nominal 256 -> INT8 mask, all four models
bash hpca --models "$U_MODEL" --configs sc_int8 --metrics ppl \
  --sc-backend hybrid --hybrid-int-frac 0.20 --hybrid-int-bits 8 \
  --hybrid-sensitivity-dir "$SENS" --hybrid-sensitivity-config sc_int8 --tag "$TAG"

# (b) nominal 192 + 128 -> INT7 mask, 14B only (the dose outlier)
if [ "$U_MODEL" = "14B" ]; then
  bash hpca --models 14B --configs sc_avg192,sc_int7 --metrics ppl \
    --sc-backend hybrid --hybrid-int-frac 0.20 --hybrid-int-bits 7 \
    --hybrid-sensitivity-dir "$SENS" --hybrid-sensitivity-config sc_int8 --tag "$TAG"
fi

echo "[dose-awq-h20] done model=$U_MODEL"
