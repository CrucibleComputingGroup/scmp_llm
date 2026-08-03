#!/bin/bash
# run_ruler_baseline.sh — FP16 + INT RULER baselines, one (model,config,task) per cell.
#
# Why this exists (not just `hpca --metrics ruler`): the hpca run_ruler result
# parse is broken three ways — its glob omits the `/pred/` level, it looks for
# `quant_fp16` when fp16's dir is just `fp16`, and `grep -oE '[0-9.]+' | tail -1`
# grabs the Nulls denominator (200), not the Score. This driver reads the Score
# row of `<TAG>/pred/summary-<task>.csv` directly, and cleanly splits the run
# across A100 (small models) and pro6000 (30B) by taking an explicit device.
#
# Usage:
#   bash run_ruler_baseline.sh --models qwen3-4b,llama-3.1-8b,qwen3-14b \
#        --configs fp16,W8A8_symm,...,W4A4_asymm --task niah_multivalue \
#        --ctx 4096 --samples 200 --device cuda:0 --tag mvniah
#
# Each cell -> one row in $RESULTS (Turbo):  model \t config \t task \t ctx \t score \t nulls
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- knobs -----------------------------------------------------------------
MODELS_CSV="qwen3-4b,llama-3.1-8b,qwen3-14b"
CONFIGS_CSV="fp16,W8A8_symm,W8A8_asymm,W7A7_symm,W7A7_asymm,W6A6_symm,W6A6_asymm,W5A5_symm,W5A5_asymm,W4A4_symm,W4A4_asymm"
TASK="niah_multivalue"
CTX=4096
SAMPLES=200
BATCH=8         # RULER_BATCH: left-padded batched greedy decode (call_api.py); 1 = per-sample
DEVICE="cuda:0"
DTYPE="fp16"   # call_api.py --dtype accepts fp16|bf16 (NOT float16 — hpca's hardcoded 'float16' is a bug)
TAG="$(date +%Y%m%d 2>/dev/null || echo run)"
CONDA_ENV="annstention"
SQ_ALPHA=0.5
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)  MODELS_CSV="$2"; shift 2 ;;
    --configs) CONFIGS_CSV="$2"; shift 2 ;;
    --task)    TASK="$2"; shift 2 ;;
    --ctx)     CTX="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --batch)   BATCH="$2"; shift 2 ;;
    --device)  DEVICE="$2"; shift 2 ;;
    --tag)     TAG="$2"; shift 2 ;;
    --env)     CONDA_ENV="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

# rname -> short label for the results TSV (keeps parity with the PPL table).
declare -A SHORT=(
  [qwen3-1.7b]=1.7B [qwen3-4b]=4B [qwen3-14b]=14B
  [qwen3-30b-a3b]=30B [llama-3.1-8b]=llama8B
)

# ---- storage tiers (mirror hpca) -------------------------------------------
TURBO_BASE="/nfs/turbo/coe-nbleier/allenjin/hpca"
SCRATCH_BASE="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca"
export ACT_SCALES_DIR="$TURBO_BASE/act_scales"        # reuse PPL's cached SmoothQuant scales
export HF_HOME="/nfs/turbo/coe-nbleier/allenjin/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"                     # override ruler_run.sh's zhkangqi default
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NUM_SAMPLES="$SAMPLES"
export SC_ATTN_GRANULARITY="per_row"   # call_api.py only accepts per_row (per_head removed 2026-07-03); ruler_run.sh default per_head fails argparse
export RULER_BATCH="$BATCH"            # batched greedy decode in call_api.py
export BASELINE_ATTN="sdpa"           # efficient attention for fp16/INT baseline (eager OOMs at ctx4096+batch)

OUTDIR="$SCRATCH_BASE/logs/_ruler_${TAG}"
RESULTS="$TURBO_BASE/results/ruler_${TAG}.tsv"

if [[ $DRY -eq 0 ]]; then
  set +u; source ~/.bashrc; conda activate "$CONDA_ENV"; set -u
  mkdir -p "$OUTDIR" "$TURBO_BASE/results" "$SCRATCH_BASE/tmp"
  export TMPDIR="$SCRATCH_BASE/tmp"
  ( set -o noclobber; printf "model\tconfig\ttask\tctx\tscore\tnulls\n" > "$RESULTS" ) 2>/dev/null || true
fi

echo "[ruler] models=$MODELS_CSV"
echo "[ruler] configs=$CONFIGS_CSV task=$TASK ctx=$CTX samples=$SAMPLES device=$DEVICE"
echo "[ruler] results -> $RESULTS"
echo "[ruler] logs    -> $OUTDIR"

record() { printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$6" >> "$RESULTS"; }

N_FAILED=0
IFS=, read -ra MODELS  <<<"$MODELS_CSV"
IFS=, read -ra CONFIGS <<<"$CONFIGS_CSV"

for RNAME in "${MODELS[@]}"; do
  SH="${SHORT[$RNAME]:-$RNAME}"
  for CFG in "${CONFIGS[@]}"; do
    if [[ "$CFG" == "fp16" ]]; then MODE="fp16"; TAGDIR="fp16";
    else MODE="quant"; TAGDIR="quant_${CFG}"; fi
    SUMMARY="$HERE/ruler_eval_result/${RNAME}/synthetic/${CTX}/${TAGDIR}/pred/summary-${TASK}.csv"
    LOG="$OUTDIR/${SH}_${CFG}_${TASK}.log"

    if [[ -f "$SUMMARY" ]]; then
      echo "[ruler] [skip] $SH/$CFG — summary exists"
    else
      echo "[ruler] >>> $SH / $CFG / $TASK (mode=$MODE) -> $LOG"
      if [[ $DRY -eq 1 ]]; then
        echo "  DRY: QUANT_CONFIG=$CFG NUM_SAMPLES=$SAMPLES bash ruler_run.sh $RNAME synthetic $CTX $TASK $MODE $DTYPE $DEVICE"
        continue
      fi
      ( cd "$HERE" && QUANT_CONFIG="$CFG" SQ_ALPHA="$SQ_ALPHA" \
          bash ruler_run.sh "$RNAME" synthetic "$CTX" "$TASK" "$MODE" "$DTYPE" "$DEVICE" \
      ) >"$LOG" 2>&1
    fi

    # Parse the Score row of the single-task summary CSV (cols: label,value).
    if [[ -f "$SUMMARY" ]]; then
      SCORE=$(awk -F, '$1=="Score"{print $2}' "$SUMMARY" | tail -1)
      NULLS=$(awk -F, '$1=="Nulls"{print $2}' "$SUMMARY" | tail -1)
    else
      SCORE=""; NULLS=""
    fi
    if [[ $DRY -eq 0 ]]; then
      if [[ -n "$SCORE" ]]; then
        record "$SH" "$CFG" "$TASK" "$CTX" "$SCORE" "${NULLS:-?}"
        echo "[ruler]     score=$SCORE nulls=${NULLS:-?}"
      else
        record "$SH" "$CFG" "$TASK" "$CTX" "FAILED" "no-summary"
        N_FAILED=$((N_FAILED+1))
        echo "[ruler] !!! FAILED $SH/$CFG — no summary at $SUMMARY (see $LOG)"
      fi
    fi
  done
done

if [[ $DRY -eq 0 ]]; then
  echo; echo "===== RULER $TASK summary ($TAG) ====="
  column -t -s $'\t' "$RESULTS"
fi
echo "[ruler] done. results: $RESULTS"
[[ $N_FAILED -gt 0 ]] && { echo "[ruler] WARNING: $N_FAILED cell(s) FAILED"; exit 1; }
exit 0
