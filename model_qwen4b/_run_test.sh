#!/usr/bin/env bash
# Usage: bash _run_test.sh <tag> <env-vars...>
# Example: bash _run_test.sh fp16  DISABLE_SC=1
#          bash _run_test.sh sc256
# Writes _run_<tag>.log and touches _run_<tag>.done on clean exit.
# Models cache under the shared scratch dir per group policy.
set -eo pipefail
TAG="$1"; shift
HERE=/home/allenjin/Projects/scmp_llm/model_qwen4b
LOG=$HERE/_run_${TAG}.log
DONE=$HERE/_run_${TAG}.done
rm -f "$DONE"

exec >"$LOG" 2>&1
echo "=== run $TAG start $(date) on $(hostname) ==="
source ~/.bashrc
conda activate annstention
cd "$HERE"
export HF_HOME=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/hf_cache
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
echo "HF_HOME=$HF_HOME"
echo "QWEN_MODEL_PATH=$QWEN_MODEL_PATH"
env "$@" python test.py
echo "=== run $TAG end $(date) ==="
touch "$DONE"
