#!/bin/bash
# Evaluate a frozen MP + k=2.0 wrapper with a replacement hybrid INT mask.
# This deliberately performs no calibration and no Phase-2 search: the purpose
# is a clean mask-only comparison against the existing full-test result.
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_v20_staged/%x_%j.out

set -euo pipefail
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${EVAL_MODEL:?EVAL_MODEL is required}"
: "${EVAL_WRAPPER:?EVAL_WRAPPER is required}"
: "${EVAL_HYBRID:?EVAL_HYBRID is required}"
: "${EVAL_OUTDIR:?EVAL_OUTDIR is required}"

[[ -s "$EVAL_WRAPPER" ]] || { echo "missing wrapper: $EVAL_WRAPPER" >&2; exit 21; }
[[ -s "$EVAL_HYBRID" ]] || { echo "missing hybrid mask: $EVAL_HYBRID" >&2; exit 22; }

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
SCRATCH=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca
mkdir -p "$EVAL_OUTDIR" "$SCRATCH/tmp"

export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export ACT_SCALES_DIR="$TURBO/act_scales"
export TMPDIR="$SCRATCH/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SC_HYBRID_CONFIG_JSON="$EVAL_HYBRID"
export SC_HYBRID_INT_BITS=7
export SC_HYBRID_FORCE_INT_BITS=1

echo "[mask A/B] model=$EVAL_MODEL"
echo "[mask A/B] frozen_wrapper=$EVAL_WRAPPER"
echo "[mask A/B] hybrid=$EVAL_HYBRID"
python -u benchmark/ppl/mp_ladder_refine.py \
  --parent-wrapper "$EVAL_WRAPPER" \
  --model-path "$EVAL_MODEL" \
  --output-dir "$EVAL_OUTDIR" \
  --split test --max-rounds 0 --max-tokens 0 --ctx 2048 --alpha 0.5

python - "$EVAL_OUTDIR/summary.json" <<'PY'
import json
import sys

s = json.load(open(sys.argv[1]))
print(
    f"[mask A/B result] ppl={s['best_ppl']:.6f} "
    f"realized_flop_avg_sl={s['best_trace_flop_avg_stoc_len']:.6f}"
)
PY
