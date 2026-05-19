#!/bin/bash
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=_sbatch_logs/%x-%j.out
#
# Driver: runs one (mode, task) pair through ruler at 1K context, 50 samples,
# 64 output tokens. Submit with:
#
#   sbatch --job-name=ruler-<mode>-<task> \
#          --export=ALL,MODE=<fp16|sc>,TASK=<task>,SL=<stoc_len>,GRAN=<per_head|per_row> \
#          _sbatch_one.sh
#
# Required env (passed via --export):
#   MODE  — fp16 | sc
#   TASK  — one of synthetic tasks (niah_single_1, vt, ...)
# Optional:
#   SL    — SC stoc_len (default 128)
#   GRAN  — SC attn granularity (default per_row)

set -eo pipefail
# -u removed: /etc/bashrc references BASHRCSOURCED which trips `set -u`.

: "${MODE:?MODE not set (fp16|sc)}"
: "${TASK:?TASK not set}"
SL="${SL:-128}"
GRAN="${GRAN:-per_row}"

source ~/.bashrc
conda activate annstention
export HF_HUB_CACHE=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/hf_cache_hub
export MAX_NEW_TOKENS_OVERRIDE=64

cd /scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm_llama/benchmark/ruler

echo "=== host: $(hostname) ==="
echo "MODE=$MODE  TASK=$TASK  SL=$SL  GRAN=$GRAN  MAX_NEW_TOKENS=64  NUM_SAMPLES=50  CTX=1024"

NUM_SAMPLES=50 SC_ATTN_GRANULARITY="$GRAN" \
  bash ruler_run.sh llama-3.1-8b synthetic 1024 "$TASK" "$MODE" fp16 cuda:0 8 "$SL"
