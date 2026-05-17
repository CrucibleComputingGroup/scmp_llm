#!/usr/bin/env bash
# Multi-model overnight SC quality sweep.
#
# For each Qwen3 model below, invokes _run_sweep.sh with a per-model tag
# (and HF_HOME pointed at the shared scratch cache). On failure (OOM, load
# error, etc.) prints a short note and continues to the next model — so an
# issue with one model does not block the others. Each model's output lives
# in _run_<tag>.log alongside its .done marker.
#
# Models intentionally ordered cheap -> expensive so early failures surface
# while the rest still has time to run before morning:
#   sweep_4b   Qwen/Qwen3-4B-Instruct-2507     (~8 GB,  dense, sanity)
#   sweep_8b   Qwen/Qwen3-8B                   (~16 GB, dense)
#   sweep_14b  Qwen/Qwen3-14B                  (~28 GB, dense)
#   sweep_30b  Qwen/Qwen3-30B-A3B-Instruct-2507(~60 GB, MoE — exercises the
#              new modeling_qwen3_moe.eager_attention_forward patch and the
#              "skip MoE router from SC replacement" path)
set -uo pipefail
HERE=/home/allenjin/Projects/scmp_llm/model_qwen4b
TOPLOG=$HERE/_run_sweep_all.log
TOPDONE=$HERE/_run_sweep_all.done
rm -f "$TOPDONE"

exec >"$TOPLOG" 2>&1
echo "=== multi-model sweep start $(date) on $(hostname) ==="

MODELS=(
  "sweep_4b   Qwen/Qwen3-4B-Instruct-2507"
  "sweep_8b   Qwen/Qwen3-8B"
  "sweep_14b  Qwen/Qwen3-14B"
  "sweep_30b  Qwen/Qwen3-30B-A3B-Instruct-2507"
)

for row in "${MODELS[@]}"; do
  TAG=$(echo "$row" | awk '{print $1}')
  MODEL=$(echo "$row" | awk '{print $2}')
  echo
  echo "############################################################"
  echo "# $(date)  TAG=$TAG  MODEL=$MODEL"
  echo "############################################################"
  if QWEN_MODEL_PATH="$MODEL" bash "$HERE/_run_sweep.sh" "$TAG"; then
    echo "# $(date)  $TAG OK  -> $HERE/_run_${TAG}.log"
  else
    RC=$?
    echo "# $(date)  $TAG FAILED rc=$RC  -> see $HERE/_run_${TAG}.log"
    echo "# (continuing to next model)"
  fi
done

echo
echo "=== multi-model sweep end $(date) ==="
touch "$TOPDONE"
