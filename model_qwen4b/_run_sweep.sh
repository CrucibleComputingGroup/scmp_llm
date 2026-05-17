#!/usr/bin/env bash
# Overnight SC quality sweep for Qwen3-4B-Instruct.
#
# Phase 1: forward-pass logit MSE vs FP16, swept over a wide range of
#          stoc_len at sc_prec=8 — shows where quality degrades.
# Phase 2: per-matmul MSE at three representative stoc_len points
#          (256 = ~lossless, 64 = mid-quality, 16 = breakdown) — shows
#          *where* in the model the error concentrates.
#
# Usage:  bash _run_sweep.sh <tag>
# Writes  _run_<tag>.log  and touches  _run_<tag>.done  on clean exit.
set -eo pipefail
TAG="${1:-sweep}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=$HERE/_run_${TAG}.log
DONE=$HERE/_run_${TAG}.done
rm -f "$DONE"

exec >"$LOG" 2>&1
echo "=== sweep $TAG start $(date) on $(hostname) ==="
source ~/.bashrc
conda activate annstention
cd "$HERE"
export HF_HOME=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/hf_cache
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
echo "HF_HOME=$HF_HOME"
echo "QWEN_MODEL_PATH=$QWEN_MODEL_PATH"
echo

echo "============================================================"
echo "Phase 1: logit MSE sweep over stoc_len at sc_prec=8"
echo "============================================================"
python check_mse.py
echo

for SL in 256 64 16; do
  echo "============================================================"
  echo "Phase 2: per-layer MSE at sc_prec=8 stoc_len=$SL"
  echo "============================================================"
  SC_STOC_LEN=$SL python check_perlayer_mse.py
  echo
done

echo "=== sweep $TAG end $(date) ==="
touch "$DONE"
