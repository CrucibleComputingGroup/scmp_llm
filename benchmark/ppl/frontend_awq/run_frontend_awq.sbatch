#!/bin/bash
# PTQ front-end ablation: AWQ vs SmoothQuant on UNIFORM SC (pure, no hybrid mask).
# One job per model; three budgets sc_int7 / sc_avg192 / sc_int8 (nominal
# 128/192/256) = the SAME axis as the INT-sensitivity sweep. FRONTEND=awq swaps
# the per-linear smooth_scales for native-AWQ (INT-objective) scales; SC stays a
# pure downstream substrate. The FIRST config per job calibrates + caches the AWQ
# scales (obj_bits from AWQ_OBJ_BITS, default 4); the other two reuse the cache.
# SmoothQuant column = the existing pure-SC cells in ../../uniform/ (NOT rerun).
# Full protocol (PPL_MAX_TOKENS=0, ctx 2048, SmoothQuant-equiv alpha unused for AWQ).
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_frontend_awq/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${FE_MODEL:?FE_MODEL is required (4B|llama8B|14B|30B)}"

# Citability guards (full test split, no truncation, deploy-matched context).
export PPL_MAX_TOKENS=0
export CTX=2048
# Front-end selection + AWQ objective bit-width (low enough to find real
# de-outliering scales; see model/awq_apply.py caveat).
export FRONTEND=awq
export AWQ_OBJ_BITS="${AWQ_OBJ_BITS:-4}"
# 30B MoE has ~18k expert-linears in the search loop; a smaller per-linear token
# cap keeps calibration memory/time bounded. Override per model if needed.
export AWQ_TOK_CAP="${AWQ_TOK_CAP:-128}"

echo "[frontend_awq] model=$FE_MODEL FRONTEND=awq obj_bits=$AWQ_OBJ_BITS tok_cap=$AWQ_TOK_CAP tag=frontend_awq_20260724"
bash hpca --models "$FE_MODEL" --configs sc_int7,sc_avg192,sc_int8 \
  --metrics ppl \
  --tag frontend_awq_20260724
