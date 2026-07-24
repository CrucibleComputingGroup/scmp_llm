#!/bin/bash
# Recover the two final, batch-1 test evaluations for a completed 14B V20
# search winner.  This performs no calibration or measured search.
#
# Required export: EVAL_TARGET=32|40|48|64
# Optional export: EVAL_TAG=<search tag without the mp_final_search_ prefix>
#
# The historical 14B V20 wave searched all four targets with hybrid INT7, so
# this evaluator deliberately preserves INT7 even though the current launcher
# derives INT6 for a newly searched target-32 lane.
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_v20_staged/%x_%j.out

set -euo pipefail
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${EVAL_TARGET:?EVAL_TARGET is required}"
case "$EVAL_TARGET" in
  32|40|48|64) ;;
  *) echo "unsupported 14B target: $EVAL_TARGET" >&2; exit 20 ;;
esac

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
SCRATCH=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca
TAG="${EVAL_TAG:-mp_v20_structured_14B_t${EVAL_TARGET}_20260721}"
BRANCH="$TURBO/mp_final_search_${TAG}/14B/target${EVAL_TARGET}p000"
WINNER="$BRANCH/best_wrapper.json"
HYBRID="$TURBO/hybrid_configs/_hpca_mp_v9_hybdose_all_20260718_183035/14B_mp_avg32_v9_rank-sc_int8_measured_curve_top0p20.json"

[[ -s "$WINNER" ]] || { echo "missing completed winner: $WINNER" >&2; exit 21; }
[[ -s "$HYBRID" ]] || { echo "missing hybrid mask: $HYBRID" >&2; exit 22; }

mkdir -p "$SCRATCH/tmp"
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
export PPL_WINDOW_BATCH_SIZE=1

evaluate() {
  local wrapper="$1"
  local outdir="$2"
  local label="$3"
  if [[ -s "$outdir/summary.json" ]]; then
    echo "[14B final] target=$EVAL_TARGET $label already complete: $outdir/summary.json"
    return
  fi
  python -u benchmark/ppl/mp_ladder_refine.py \
    --parent-wrapper "$wrapper" \
    --model-path Qwen/Qwen3-14B \
    --output-dir "$outdir" \
    --split test --max-rounds 0 --max-tokens 0 --ctx 2048 --alpha 0.5
}

echo "[14B final] target=$EVAL_TARGET winner=$WINNER hybrid=INT7 batch=1"
evaluate "$WINNER" "$BRANCH/eval_test" ungated

GATED="$BRANCH/best_wrapper_gate2.0.json"
python - "$WINNER" "$GATED" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    wrapper = json.load(f)
wrapper["escape_gate_k"] = 2.0
wrapper["escape_stoc_len"] = 128
with open(sys.argv[2], "w") as f:
    json.dump(wrapper, f, indent=1)
PY
evaluate "$GATED" "$BRANCH/eval_gate2.0" gate2.0

python - "$BRANCH/eval_test/summary.json" "$BRANCH/eval_gate2.0/summary.json" "$EVAL_TARGET" <<'PY'
import json
import sys

ungated = json.load(open(sys.argv[1]))
gated = json.load(open(sys.argv[2]))
print(
    f"[14B final done] target={sys.argv[3]} "
    f"ungated_ppl={ungated['best_ppl']:.6f} "
    f"ungated_realized={ungated['best_trace_flop_avg_stoc_len']:.6f} "
    f"gate2_ppl={gated['best_ppl']:.6f} "
    f"gate2_realized={gated['best_trace_flop_avg_stoc_len']:.6f}"
)
PY
