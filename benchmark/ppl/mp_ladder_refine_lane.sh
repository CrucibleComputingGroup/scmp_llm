#!/bin/bash
# Run the frozen-threshold validation-NLL ladder refinement after a completed
# hpca v9 cell.
#
# Usage: mp_ladder_refine_lane.sh <4B|llama8B|14B|30B> <parent-tag> <tag>
set -euo pipefail
set +u
source ~/.bashrc
conda activate annstention
set -u

cd /home/allenjin/Projects/scmp_llm

SHORT="${1:?usage: mp_ladder_refine_lane.sh <model> <parent-tag> <tag>}"
PARENT_TAG="${2:?usage: mp_ladder_refine_lane.sh <model> <parent-tag> <tag>}"
TAG="${3:?usage: mp_ladder_refine_lane.sh <model> <parent-tag> <tag>}"

declare -A HF=(
  [4B]="Qwen/Qwen3-4B-Instruct-2507"
  [llama8B]="meta-llama/Llama-3.1-8B-Instruct"
  [14B]="Qwen/Qwen3-14B"
  [30B]="Qwen/Qwen3-30B-A3B-Instruct-2507"
)
hf="${HF[$SHORT]:?unknown model $SHORT}"
safe_hf="${hf//\//_}"

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
SCRATCH=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca
PARENT_DIR="$TURBO/mp_calib_${PARENT_TAG}"
OUTDIR="$TURBO/mp_ladder_refine_${TAG}/${SHORT}"
LOGDIR="$SCRATCH/logs/_mp_ladder_refine_${TAG}"
HYBRID="${PASS2_HYBRID_CONFIG:-$TURBO/hybrid_configs/_hpca_${PARENT_TAG}/${SHORT}_mp_avg32_v9_rank-sc_int8_measured_curve_top0p10.json}"
mkdir -p "$OUTDIR" "$LOGDIR" "$SCRATCH/tmp"

if [[ -n "${PASS2_PARENT_WRAPPER:-}" ]]; then
  PARENT="$PASS2_PARENT_WRAPPER"
  [[ -s "$PARENT" ]] || { echo "missing explicit parent wrapper: $PARENT" >&2; exit 21; }
else
  wait_started=$(date +%s)
  wait_limit=${PASS2_PARENT_WAIT_SECONDS:-7200}
  while true; do
    mapfile -t wrappers < <(
      find "$PARENT_DIR" -maxdepth 1 -type f \
        -name "${safe_hf}__mp_avg32_v9__act_global_v9*_wrapper.json" -print \
        2>/dev/null
    )
    if [[ ${#wrappers[@]} -eq 1 ]]; then
      break
    fi
    if [[ ${#wrappers[@]} -gt 1 ]]; then
      echo "expected one v9 parent wrapper for $SHORT; found ${#wrappers[@]}" >&2
      printf '  %s\n' "${wrappers[@]}"
      exit 21
    fi
    waited=$(( $(date +%s) - wait_started ))
    if (( waited >= wait_limit )); then
      echo "timed out after ${waited}s waiting for v9 parent in $PARENT_DIR" >&2
      exit 21
    fi
    echo "[pass2 lane] waiting for v9 parent wrapper ($SHORT; ${waited}s elapsed)"
    sleep 30
  done
  PARENT="${wrappers[0]}"
fi
[[ -s "$HYBRID" ]] || { echo "missing v9 hybrid config: $HYBRID" >&2; exit 22; }

export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export ACT_SCALES_DIR="$TURBO/act_scales"
export TMPDIR="$SCRATCH/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SC_HYBRID_CONFIG_JSON="$HYBRID"
export SC_HYBRID_INT_BITS=7
export SC_HYBRID_FORCE_INT_BITS=1

echo "[pass2 lane] model=$SHORT parent=$PARENT out=$OUTDIR"
baseline_args=()
if [[ -n "${PASS2_BASELINE_TRACE:-}" ]]; then
  baseline_args=(--baseline-trace "$PASS2_BASELINE_TRACE")
fi
target_args=()
if [[ -n "${PASS2_TARGET_COST:-}" ]]; then
  target_args=(--target-cost "$PASS2_TARGET_COST")
fi
python -u benchmark/ppl/mp_ladder_refine.py \
  --parent-wrapper "$PARENT" \
  --model-path "$hf" \
  --output-dir "$OUTDIR" \
  --split "${PASS2_SPLIT:-validation}" \
  --max-tokens "${PASS2_MAX_TOKENS:-32768}" \
  --ctx 2048 \
  --max-rounds "${PASS2_MAX_ROUNDS:-4}" \
  --floor-step "${PASS2_FLOOR_STEP:-4}" \
  --donor-step "${PASS2_DONOR_STEP:-1}" \
  --min-gap "${PASS2_MIN_GAP:-1}" \
  --budget-tol "${PASS2_BUDGET_TOL:-0.35}" \
  --min-nll-improvement "${PASS2_MIN_NLL_IMPROVEMENT:-0.0}" \
  "${baseline_args[@]}" \
  "${target_args[@]}"

echo "[pass2 lane done] model=$SHORT summary=$OUTDIR/summary.json"
