#!/bin/bash
# Rebuttal T10 (reviewer A): evaluate the FROZEN deployed configs on shifted
# data. Nothing is calibrated or allocated here; every MP row loads the
# best(all) t48 wrapper unchanged. One manifest row per array task.
#
#   kind=mp       best(all) t48 cell; env block == kbands/run_prc_ppl.sbatch
#   kind=uniform  uniform SC (sc_avg96 = t48, sc_int7 = t64) with the SAME t48
#                 INT mask + AWQ; env block == frontend_awq/run_uniform_awq_comparator.sbatch
#   kind=fp16     plain HF, eager attention (== ctx_sweep fp16)
#
# Outputs never touch the archived namespaces: traces/logs go to Turbo
# hpca/t10_20260929/, and an existing trace is never overwritten.
#
#   sbatch --array=0-30%4 run_t10_ppl.sbatch
#SBATCH --job-name=t10ppl
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/coe-nbleier/allenjin/hpca/t10_20260929/logs/%x_%A_%a.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/SCMP/scmp_llm
# The env's editable scmp_kernels install still maps to the pre-move path (see SCMP/CLAUDE.md).
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}

MANIFEST=${T10_MANIFEST:-/home/allenjin/Projects/SCMP/scmp_llm/benchmark/ppl/t10/t10_ppl_manifest_20260929.tsv}
LINE=$(awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" '$1==i' "$MANIFEST")
[[ -n "$LINE" ]] || { echo "[t10] no manifest row $SLURM_ARRAY_TASK_ID" >&2; exit 2; }
IFS=$'\t' read -r IDX MODEL KIND UCFG DATASET CTXV PURPOSE <<< "$LINE"
[[ "$UCFG" == "-" ]] && UCFG=""

SCMP=/home/allenjin/Projects/SCMP
TARGET=48
BUNDLE=$SCMP/hpca_results/llm/ppl/mp_best/configs/$MODEL/target$TARGET
TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
OUT=$TURBO/t10_20260929
mkdir -p "$OUT/traces"
# Model id comes from the deployed bundle, never a hardcoded string.
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$BUNDLE/table.json")
[[ -n "$HF" ]] || { echo "[t10] cannot read model_path" >&2; exit 2; }

HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export MODEL_PATH="$HF" CTX="$CTXV" PPL_DATASET="$DATASET" SQ_ALPHA=0.5
# FULL PROTOCOL on every scored text: no token cap.
export PPL_MAX_TOKENS=0
export SC_TRACE_PROVENANCE=1

case "$KIND" in
  mp)
    TAG="t10_${MODEL}_t${TARGET}_mp_${DATASET}_ctx${CTXV}"
    CFG=$(python - "$MODEL" "$TARGET" <<'PY'
import json, sys
rows = json.load(open("/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.json"))["rows"]
hit = [r for r in rows if r["model"] == sys.argv[1] and int(r["target"]) == int(sys.argv[2])]
assert len(hit) == 1, hit
print(hit[0]["wrapper"])
PY
)
    [[ -f "$CFG" ]] || { echo "[t10] missing best(all) wrapper $CFG" >&2; exit 2; }
    # Refuse a wrapper without per_row_chunk buckets: it would silently run
    # per-ROW dispatch and be recorded as the deployed per-group cell.
    python - "$CFG" <<'PY' || exit 2
import json, sys, pathlib
w = json.load(open(sys.argv[1]))
t = w.get("threshold_table_path")
if not t or not pathlib.Path(t).is_file():
    sys.exit(f"[t10] threshold_table_path missing: {t!r}")
n = len(((json.load(open(t)).get("per_row_chunk") or {}).get("buckets")) or {})
if n == 0:
    sys.exit("[t10] table carries NO per_row_chunk buckets")
print(f"[t10] per_row_chunk buckets: {n}")
PY
    export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64
    export SC_RNG_GRID="" SC_RNG_GRID_ATTN="" SC_RNG_GRID_QK="" SC_RNG_GRID_AV=""
    export QUANT_CONFIG=mp MP_CONFIG_JSON="$CFG"
    export SC_HYBRID_CONFIG_JSON="$BUNDLE/hybrid_config.json"
    export ACT_SCALES_DIR=$SCMP/hpca_results/llm/ppl/mp_best/act_scales
    export FRONTEND=awq
    ;;
  uniform)
    [[ "$UCFG" == "sc_avg96" || "$UCFG" == "sc_int7" ]] || {
      echo "[t10] uniform cfg must be sc_avg96|sc_int7, got '$UCFG'" >&2; exit 2; }
    TAG="t10_${MODEL}_t${TARGET}mask_${UCFG}_${DATASET}_ctx${CTXV}"
    export AWQ_SCALES_DIR="$TURBO/act_scales/awq_scales"
    AWQ_CACHE="$AWQ_SCALES_DIR/awq_scales_${HF//\//_}_b4.pt"
    [[ -f "$AWQ_CACHE" ]] || { echo "[t10] MISSING AWQ cache $AWQ_CACHE" >&2; exit 2; }
    export FRONTEND=awq AWQ_OBJ_BITS=4 AWQ_TOK_CAP=128
    export QUANT_CONFIG="$UCFG"              # uniform SC, allocator OFF
    export SC_HYBRID_CONFIG_JSON="$BUNDLE/hybrid_config.json"
    export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64
    export ACT_SCALES_DIR=$SCMP/hpca_results/llm/ppl/mp_best/act_scales
    export SC_MP_TRACE_MODE=summary
    ;;
  fp16)
    TAG="t10_${MODEL}_fp16_${DATASET}_ctx${CTXV}"
    export QUANT_CONFIG=fp16                 # BASELINE_ATTN default = eager
    ;;
  *) echo "[t10] unknown kind '$KIND'" >&2; exit 2 ;;
esac

if [[ "$KIND" != "fp16" ]]; then
  [[ -f "$SC_HYBRID_CONFIG_JSON" ]] || { echo "[t10] missing mask $SC_HYBRID_CONFIG_JSON" >&2; exit 2; }
  export SC_MP_TRACE="$OUT/traces/${TAG}_trace.json"
  [[ -e "$SC_MP_TRACE" ]] && { echo "[t10] refusing to overwrite $SC_MP_TRACE" >&2; exit 3; }
fi

echo "[t10] row=$IDX tag=$TAG purpose=$PURPOSE"
echo "[t10] model=$HF kind=$KIND cfg=${UCFG:-$QUANT_CONFIG} dataset=$DATASET ctx=$CTX max_tokens=$PPL_MAX_TOKENS"
echo "[t10] mask=${SC_HYBRID_CONFIG_JSON:-none} frontend=${FRONTEND:-none} mp_config=${MP_CONFIG_JSON:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun python -u benchmark/quant/eval_quant.py
