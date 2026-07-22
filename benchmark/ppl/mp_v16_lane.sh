#!/bin/bash
# One-model V16 paired-window refinement over the V9 parent.
# Usage: mp_v16_lane.sh <4B|llama8B|14B|30B> <tag>
# Targets come from V16_TARGETS (default 32:36:40); our wave submits one
# target per job so independent cells each hold one GPU.
set -euo pipefail
set +u
source ~/.bashrc
conda activate annstention
set -u

cd /home/allenjin/Projects/scmp_llm

SHORT="${1:?usage: mp_v16_lane.sh <model> <tag>}"
TAG="${2:?usage: mp_v16_lane.sh <model> <tag>}"

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
PARENT_TAG="${V16_PARENT_TAG:-mp_v9_int6_20260716_001908}"
PARENT_DIR="$TURBO/mp_calib_${PARENT_TAG}"
PARENT_CONFIG="${V16_PARENT_CONFIG:-mp_avg32_v9}"
PARENT_GLOB="${V16_PARENT_GLOB:-${safe_hf}__${PARENT_CONFIG}__act_global_v9*_wrapper.json}"
OUTDIR="$TURBO/mp_v16_refine_${TAG}/${SHORT}"
LOGDIR="$SCRATCH/logs/_mp_v16_refine_${TAG}"
FP16_REF_DIR="$TURBO/fp16_val_refs"
mkdir -p "$OUTDIR" "$LOGDIR" "$SCRATCH/tmp" "$FP16_REF_DIR"

if [[ -n "${V16_PARENT_WRAPPER:-}" ]]; then
  PARENT="$V16_PARENT_WRAPPER"
else
  mapfile -t wrappers < <(
    find "$PARENT_DIR" -maxdepth 1 -type f -name "$PARENT_GLOB" -print
  )
  if [[ ${#wrappers[@]} -ne 1 ]]; then
    echo "expected one V9 parent wrapper for $SHORT; found ${#wrappers[@]}" >&2
    printf '  %s\n' "${wrappers[@]}" >&2
    exit 21
  fi
  PARENT="${wrappers[0]}"
fi
HYBRID="${V16_HYBRID_CONFIG:-$TURBO/hybrid_configs/_hpca_${PARENT_TAG}/${SHORT}_${PARENT_CONFIG}_rank-sc_int8_measured_curve_top0p10.json}"
[[ -s "$PARENT" ]] || { echo "missing V9 parent: $PARENT" >&2; exit 22; }
[[ -s "$HYBRID" ]] || { echo "missing V9 hybrid config: $HYBRID" >&2; exit 23; }

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

declare -A FP16_TEST_PPL=(
  [4B]="10.0445"
  [llama8B]="7.2130"
  [14B]="8.6383"
  [30B]="7.2613"
)

# FP16 references on the SAME validation windows the search uses. The flock
# keeps concurrent same-model jobs from duplicating the ~15-minute FP16 eval;
# the reference build must not inherit the SC/MP environment.
FP16_REF="$FP16_REF_DIR/${safe_hf}_validation_ctx2048.json"
if [[ "${V16_SKIP_FP16_REF:-0}" != "1" ]]; then
  (
    flock -w 7200 9 || exit 0
    env -u MP_CONFIG_JSON -u SC_HYBRID_CONFIG_JSON \
      python -u benchmark/ppl/fp16_val_reference.py \
        --model-path "$hf" --ctx 2048 \
        --out "$FP16_REF" --skip-if-exists
  ) 9>"$FP16_REF.lock" || echo "[v16 lane] fp16 reference step failed; continuing without tier-1" >&2
fi
FP16_REF_ARG=()
[[ -s "$FP16_REF" ]] && FP16_REF_ARG=(--fp16-ref "$FP16_REF")

REMOVAL_ARG=()
if [[ "${V16_ALLOW_REMOVAL:-1}" == "0" ]]; then
  REMOVAL_ARG+=(--no-allow-removal)
fi

echo "[v16 lane] model=$SHORT parent=$PARENT hybrid=$HYBRID out=$OUTDIR targets=${V16_TARGETS:-32:36:40}"
python -u benchmark/ppl/mp_v16_refine.py \
  --parent-wrapper "$PARENT" \
  --model-path "$hf" \
  --output-dir "$OUTDIR" \
  --targets "${V16_TARGETS:-32:36:40}" \
  --ctx 2048 \
  --max-sweeps "${V16_MAX_SWEEPS:-6}" \
  --patience "${V16_PATIENCE:-2}" \
  --advance-total "${V16_ADVANCE_TOTAL:-12}" \
  --threshold-mass-steps "${V16_THRESHOLD_MASS_STEPS:-0.03}" \
  --split-fractions "${V16_SPLIT_FRACTIONS:-0.33:0.67}" \
  --min-classes 3 \
  --max-classes 8 \
  --min-level 4 \
  --max-level 128 \
  --budget-tol-under "${V16_BUDGET_TOL_UNDER:-0.35}" \
  --budget-tol-over "${V16_BUDGET_TOL_OVER:-1.0}" \
  --accept-floor "${V16_ACCEPT_FLOOR:-0.002}" \
  --fp16-test-ppl "${V16_FP16_TEST_PPL:-${FP16_TEST_PPL[$SHORT]}}" \
  --ppl-fp16-ratio "${V16_PPL_FP16_RATIO:-1.1}" \
  --tier1-ratio-scale "${V16_TIER1_RATIO_SCALE:-0.90}" \
  --test-tokens "${V16_TEST_TOKENS:-0}" \
  --walltime-hours "${V16_WALLTIME_HOURS:-22.5}" \
  "${FP16_REF_ARG[@]}" \
  "${REMOVAL_ARG[@]}"

echo "[v16 lane done] model=$SHORT summary=$OUTDIR/summary.json"
