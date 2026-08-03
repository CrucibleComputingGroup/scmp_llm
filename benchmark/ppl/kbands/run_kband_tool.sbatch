#!/bin/bash
# Phase 3 GPU tooling: the kernel equivalence test, or the allocator.
#
# Usage:
#   KB_JOB=equiv sbatch run_kband_tool.sbatch
#   KB_JOB=alloc KB_MODEL=4B KB_TARGET=48 sbatch run_kband_tool.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${KB_JOB:?KB_JOB is required (equiv|alloc)}"

HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
OUT="$TURBO/kbands/kbands_20260801"
mkdir -p "$OUT/alloc"

if [[ "$KB_JOB" == "equiv" ]]; then
  echo "[kbands] kernel equivalence test on $(hostname)"
  srun python -u benchmark/ppl/kbands/test_kband_equivalence.py
  exit $?
fi

if [[ "$KB_JOB" == "qkcalib" ]]; then
  : "${KB_MODEL:?}"; : "${KB_TARGET:?}"
  QKB=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}
  HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$QKB/table.json")
  [[ -n "$HF" ]] || { echo "[kbands] cannot read model_path" >&2; exit 2; }
  export ACT_SCALES_DIR=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/act_scales
  mkdir -p "$OUT/qk"
  srun python -u -m benchmark.ppl.kbands.qk_calib \
    --parent /home/allenjin/Projects/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET} \
    --model_path "$HF" \
    --out "$OUT/qk/${KB_MODEL}_t${KB_TARGET}_qk_alpha${KB_ALPHA:-0.5}.json" \
    --alpha "${KB_ALPHA:-0.5}"
  exit $?
fi

if [[ "$KB_JOB" == "ceilingreal" ]]; then
  : "${KB_MODEL:?}"
  echo "[kbands] per-group ceiling on REAL activations: $KB_MODEL"
  export ACT_SCALES_DIR=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/act_scales
  srun python -u benchmark/ppl/kbands/t32_allocation_ceiling.py \
    --target "${KB_TARGET:-32}" --real "$KB_MODEL" --op "${KB_OP:-down_proj}"
  exit $?
fi

if [[ "$KB_JOB" == "ceiling" ]]; then
  echo "[kbands] TRUE per-group allocation ceiling on $(hostname)"
  srun python -u benchmark/ppl/kbands/t32_allocation_ceiling.py --target "${KB_TARGET:-32}"
  exit $?
fi

if [[ "$KB_JOB" == "budget" ]]; then
  echo "[kbands] t32 error-budget decomposition on $(hostname)"
  srun python -u benchmark/ppl/kbands/t32_error_budget.py --target "${KB_TARGET:-32}"
  exit $?
fi

if [[ "$KB_JOB" == "qk" ]]; then
  echo "[kbands] qk rebalance invariance + SC-error test on $(hostname)"
  srun python -u benchmark/ppl/kbands/test_qk_rebalance.py
  exit $?
fi

: "${KB_MODEL:?KB_MODEL required}"
: "${KB_TARGET:?KB_TARGET required}"
# Model id comes from the PARENT TABLE, never a hardcoded string. The deployed
# 30B is Qwen3-30B-A3B-Instruct-2507; hardcoding "Qwen3-30B-A3B" selects a
# DIFFERENT model and a different act_scales cache -- the config substitution
# this project forbids inside a comparison. It failed loudly here (missing
# cache) but would have been silent had that cache existed.
BUNDLE=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$BUNDLE/table.json")
[[ -n "$HF" ]] || { echo "[kbands] cannot read model_path from $BUNDLE/table.json" >&2; exit 2; }
echo "[kbands] model_path (from parent table) = $HF"
export ACT_SCALES_DIR=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/act_scales

echo "[kbands] allocator: model=$KB_MODEL target=$KB_TARGET bundle=$BUNDLE"
srun python -u -m benchmark.ppl.mp_kbands_calib \
  --parent "$BUNDLE" --model_path "$HF" \
  --out "$OUT/alloc/${KB_MODEL}_t${KB_TARGET}${KB_SUFFIX:-}_alloc.json" \
  --n-bands "${KB_NBANDS:-2}" \
  --num-windows "${KB_WINDOWS:-8}" \
  --max-rows "${KB_MAXROWS:-512}" \
  --hot-frac "${KB_HOTFRAC:-0.5}"
