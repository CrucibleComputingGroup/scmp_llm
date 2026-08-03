#!/bin/bash
# One-model v11 joint ladder/threshold search over several compute targets.
# Usage: mp_joint_refine_lane.sh <4B|llama8B|14B|30B> <tag>
set -euo pipefail
set +u
source ~/.bashrc
conda activate annstention
set -u

cd /home/allenjin/Projects/scmp_llm

SHORT="${1:?usage: mp_joint_refine_lane.sh <model> <tag>}"
TAG="${2:?usage: mp_joint_refine_lane.sh <model> <tag>}"

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
PARENT_TAG=mp_final_avg32_20260714_010304
PARENT_DIR="$TURBO/mp_calib_${PARENT_TAG}"
OUTDIR="$TURBO/mp_joint_refine_${TAG}/${SHORT}"
LOGDIR="$SCRATCH/logs/_mp_joint_refine_${TAG}"
mkdir -p "$OUTDIR" "$LOGDIR" "$SCRATCH/tmp"

if [[ -n "${V11_PARENT_WRAPPER:-}" ]]; then
  PARENT="$V11_PARENT_WRAPPER"
else
  mapfile -t wrappers < <(
    find "$PARENT_DIR" -maxdepth 1 -type f \
      -name "${safe_hf}__mp_avg32_burst128__act_global_v3*_wrapper.json" -print
  )
  if [[ ${#wrappers[@]} -ne 1 ]]; then
    echo "expected one mp_final avg32 parent for $SHORT; found ${#wrappers[@]}" >&2
    printf '  %s\n' "${wrappers[@]}" >&2
    exit 21
  fi
  PARENT="${wrappers[0]}"
fi
HYBRID="${V11_HYBRID_CONFIG:-$TURBO/hybrid_configs/_hpca_${PARENT_TAG}/${SHORT}_mp_avg32_burst128_rank-sc_int8_measured_curve_top0p10.json}"
[[ -s "$PARENT" ]] || { echo "missing parent: $PARENT" >&2; exit 22; }
[[ -s "$HYBRID" ]] || { echo "missing hybrid config: $HYBRID" >&2; exit 23; }

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

echo "[v11 lane] model=$SHORT parent=$PARENT hybrid=$HYBRID out=$OUTDIR"
python -u benchmark/ppl/mp_joint_refine.py \
  --parent-wrapper "$PARENT" \
  --model-path "$hf" \
  --output-dir "$OUTDIR" \
  --targets "${V11_TARGETS:-parent,nominal,36,40,48}" \
  --search-tokens "${V11_SEARCH_TOKENS:-8192}" \
  --confirm-tokens "${V11_CONFIRM_TOKENS:-32768}" \
  --ctx 2048 \
  --ladder-rounds "${V11_LADDER_ROUNDS:-2}" \
  --threshold-rounds "${V11_THRESHOLD_ROUNDS:-1}" \
  --threshold-mass-step "${V11_THRESHOLD_MASS_STEP:-0.02}" \
  --floor-step "${V11_FLOOR_STEP:-4}" \
  --budget-tol "${V11_BUDGET_TOL:-0.35}" \
  --budget-corrections "${V11_BUDGET_CORRECTIONS:-2}" \
  --profile-bins "${V11_PROFILE_BINS:-257}" \
  --max-level 128

echo "[v11 lane done] model=$SHORT summary=$OUTDIR/summary.json"
