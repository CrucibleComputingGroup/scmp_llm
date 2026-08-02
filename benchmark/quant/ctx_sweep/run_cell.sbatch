#!/bin/bash
# One (model x ctx) cell of the INT/fp16 CONTEXT sweep.
#
# Runs fp16 + W8A8_{symm,asymm} + W6A6_{symm,asymm} at a single eval context
# length through the standard HPCA PPL protocol (full wikitext-2 test,
# stride=ctx, PPL_MAX_TOKENS=0).
#
# FRONTEND=awq, both variants, to match the paper's INT convention exactly:
# model_quality.tex takes INT8/6/4 as "the lower of symmetric and asymmetric
# AWQ runs" (frontend_awq/int_awq_vs_smoothquant.csv).  A SmoothQuant or
# asymm-only arm would be front-end-mismatched to that table.  AWQ scales are
# LOADED from cache (act_scales/awq_scales/*_b4.pt, all four models present);
# the INT path cannot calibrate them itself, and they are calibration-derived
# so the same scales are correct at every eval ctx.
#
# fp16 is front-end-invariant: parse_config("fp16") is None so eval_quant.py
# returns plain HF before any SmoothQuant/AWQ branch.  It still runs eager
# attention (BASELINE_ATTN default), same as the INT cells.
#
# TAG carries the ctx because results_<TAG>.tsv records only
# (model, config, metric, value) -- no ctx column -- so a shared tag across
# contexts would collide indistinguishably.
#
# Submit via launch.sh; requires CTXSWEEP_MODEL and CTXSWEEP_CTX in --export.
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_ctx_sweep/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${CTXSWEEP_MODEL:?CTXSWEEP_MODEL is required}"
: "${CTXSWEEP_CTX:?CTXSWEEP_CTX is required}"
: "${CTXSWEEP_DATE:=20260730}"

CONFIGS="fp16,W8A8_symm,W8A8_asymm,W6A6_symm,W6A6_asymm"
TAG="ctxsweep_awq_ctx${CTXSWEEP_CTX}_${CTXSWEEP_DATE}"

# AWQ front-end. AWQ_OBJ_BITS defaults to 4, which is what the cached scale
# files are keyed on (awq_scales_<model>_b4.pt) -- do not override it or
# load_awq_scales will look for a cache that was never built.
export FRONTEND=awq

echo "[ctx_sweep] model=$CTXSWEEP_MODEL ctx=$CTXSWEEP_CTX tag=$TAG"
echo "[ctx_sweep] configs=$CONFIGS frontend=$FRONTEND"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash ./hpca \
  --models "$CTXSWEEP_MODEL" \
  --configs "$CONFIGS" \
  --metrics ppl \
  --ctx "$CTXSWEEP_CTX" \
  --max-tokens 0 \
  --tag "$TAG"
rc=$?
echo "[ctx_sweep] hpca rc=$rc"
exit $rc
