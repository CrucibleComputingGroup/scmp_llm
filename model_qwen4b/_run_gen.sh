#!/usr/bin/env bash
# Qualitative generation sweep wrapper. One forward pass per stoc_len, decodes
# NEW_TOKENS (default 64) and prints the actual text so the operator can see
# where the prose collapses into rubbish.
#
# Usage: bash _run_gen.sh <tag> [QWEN_MODEL_PATH=...]
set -eo pipefail
TAG="${1:-gen}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=$HERE/_run_${TAG}.log
DONE=$HERE/_run_${TAG}.done
rm -f "$DONE"

exec >"$LOG" 2>&1
echo "=== gen $TAG start $(date) on $(hostname) ==="
source ~/.bashrc
conda activate annstention
cd "$HERE"
export HF_HOME=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/hf_cache
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
echo "QWEN_MODEL_PATH=$QWEN_MODEL_PATH"
echo
python check_gen.py
echo "=== gen $TAG end $(date) ==="
touch "$DONE"
