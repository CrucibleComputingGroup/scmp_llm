#!/bin/bash
# One-model systematic value/topology/threshold search (V12/V14).
# Usage: mp_systematic_refine_lane.sh <4B|llama8B|14B|30B> <tag>
set -euo pipefail
set +u
source ~/.bashrc
conda activate annstention
set -u

cd /home/allenjin/Projects/scmp_llm

SHORT="${1:?usage: mp_systematic_refine_lane.sh <model> <tag>}"
TAG="${2:?usage: mp_systematic_refine_lane.sh <model> <tag>}"

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
PARENT_TAG="${V12_PARENT_TAG:-mp_final_avg32_20260714_010304}"
PARENT_DIR="$TURBO/mp_calib_${PARENT_TAG}"
PARENT_CONFIG="${V12_PARENT_CONFIG:-mp_avg32_burst128}"
PARENT_GLOB="${V12_PARENT_GLOB:-${safe_hf}__${PARENT_CONFIG}__act_global_v3*_wrapper.json}"
OUTDIR="$TURBO/mp_systematic_refine_${TAG}/${SHORT}"
LOGDIR="$SCRATCH/logs/_mp_systematic_refine_${TAG}"
mkdir -p "$OUTDIR" "$LOGDIR" "$SCRATCH/tmp"

if [[ -n "${V12_PARENT_WRAPPER:-}" ]]; then
  PARENT="$V12_PARENT_WRAPPER"
else
  mapfile -t wrappers < <(
    find "$PARENT_DIR" -maxdepth 1 -type f \
      -name "$PARENT_GLOB" -print
  )
  if [[ ${#wrappers[@]} -ne 1 ]]; then
    echo "expected one completed mp_final parent for $SHORT; found ${#wrappers[@]}" >&2
    printf '  %s\n' "${wrappers[@]}" >&2
    exit 21
  fi
  PARENT="${wrappers[0]}"
fi
HYBRID="${V12_HYBRID_CONFIG:-$TURBO/hybrid_configs/_hpca_${PARENT_TAG}/${SHORT}_${PARENT_CONFIG}_rank-sc_int8_measured_curve_top0p10.json}"
[[ -s "$PARENT" ]] || { echo "missing mp_final parent: $PARENT" >&2; exit 22; }
[[ -s "$HYBRID" ]] || { echo "missing mp_final hybrid config: $HYBRID" >&2; exit 23; }

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

echo "[v12 lane] model=$SHORT parent=$PARENT hybrid=$HYBRID out=$OUTDIR"
REMOVAL_ARG=()
if [[ "${V12_ALLOW_REMOVAL:-1}" == "0" ]]; then
  REMOVAL_ARG+=(--no-allow-removal)
fi
INSERTION_ARG=()
if [[ "${V12_ALLOW_INSERTION:-1}" == "0" ]]; then
  INSERTION_ARG+=(--no-allow-insertion)
fi
FIXED_COUNT_ARG=()
if [[ -n "${V12_FIXED_CLASS_COUNT:-}" ]]; then
  FIXED_COUNT_ARG+=(--fixed-class-count "${V12_FIXED_CLASS_COUNT}")
fi
declare -A FP16_PPL=(
  [4B]="10.0445"
  [llama8B]="7.2130"
  [14B]="8.6383"
  [30B]="7.2613"
)
FP16_ARG=(--fp16-ppl "${V12_FP16_PPL:-${FP16_PPL[$SHORT]}}"
          --ppl-fp16-ratio "${V12_PPL_FP16_RATIO:-1.1}")
python -u benchmark/ppl/mp_systematic_refine.py \
  --parent-wrapper "$PARENT" \
  --model-path "$hf" \
  --output-dir "$OUTDIR" \
  --targets "${V12_TARGETS:-parent:32:36:40:48}" \
  --screen-tokens "${V12_SCREEN_TOKENS:-4096}" \
  --guard-tokens "${V12_GUARD_TOKENS:-4096}" \
  --confirm-tokens "${V12_CONFIRM_TOKENS:-32768}" \
  --test-tokens "${V12_TEST_TOKENS:-0}" \
  --ctx 2048 \
  --max-sweeps "${V12_MAX_SWEEPS:-8}" \
  --patience "${V12_PATIENCE:-2}" \
  --shortlist "${V12_SHORTLIST:-3}" \
  --shortlist-per-family "${V12_SHORTLIST_PER_FAMILY:-0}" \
  --level-step "${V12_LEVEL_STEP:-4}" \
  --threshold-mass-steps "${V12_THRESHOLD_MASS_STEPS:-0.01:0.02}" \
  --split-fractions "${V12_SPLIT_FRACTIONS:-0.33:0.67}" \
  --topology-values-per-gap "${V12_TOPOLOGY_VALUES_PER_GAP:-1}" \
  --min-classes "${V12_MIN_CLASSES:-3}" \
  --max-classes "${V12_MAX_CLASSES:-8}" \
  --budget-tol "${V12_BUDGET_TOL:-0.35}" \
  --min-mean-nll-improvement "${V12_MIN_MEAN_NLL_IMPROVEMENT:-0.002}" \
  --max-guard-nll-regression "${V12_MAX_GUARD_NLL_REGRESSION:-0.002}" \
  --budget-corrections "${V12_BUDGET_CORRECTIONS:-2}" \
  --profile-bins "${V12_PROFILE_BINS:-257}" \
  --min-level 4 \
  --max-level 128 \
  "${REMOVAL_ARG[@]}" \
  "${INSERTION_ARG[@]}" \
  "${FIXED_COUNT_ARG[@]}" \
  "${FP16_ARG[@]}"

echo "[v12 lane done] model=$SHORT summary=$OUTDIR/summary.json"
