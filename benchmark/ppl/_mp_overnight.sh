#!/bin/bash
# Per-node MP sweep driver.
#
# Each node runs the 3 MP configs (MP_a, MP_m, MP_c) serially for ONE model.
# Halved + SQ@0.5 only, per_row attn, sc_prec=8, Owen scramble in rescale.
#
# Usage: _mp_overnight.sh <model_path> <out_subdir> [run_tag]
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env_common.sh"

MODEL="$1"
OUTDIR="$2"
TAG="${3:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTDIR"

# Safe filename suffix derived from model id (kept for matching the
# act_scales filename convention used by calibrate_smoothquant.py).
SAFE_NAME=$(echo "$MODEL" | tr '/' '_')
SQ_SCALES="$HERE/act_scales_${SAFE_NAME}.pt"
if [[ ! -f "$SQ_SCALES" ]]; then
  echo "[FATAL] missing SmoothQuant scales: $SQ_SCALES"
  exit 2
fi

for MP_NAME in mp_a mp_m mp_c; do
  MP_JSON="$HERE/${MP_NAME}.json"
  LOG="$OUTDIR/${SAFE_NAME}_${MP_NAME}.log"
  echo "[$(date)] >>> $SAFE_NAME / $MP_NAME"  | tee -a "$LOG"
  echo "          host=$(hostname) mp_json=$MP_JSON sq=$SQ_SCALES" | tee -a "$LOG"
  env MODEL_PATH="$MODEL" \
      PPL_MAX_TOKENS=65536 CTX=1024 \
      SC_PREC=8 STOC_LENS=128 SC_HALVE_BIPOLAR_STOC_LEN=1 \
      SC_ATTN_GRANULARITY=per_row \
      USE_SMOOTHQUANT=1 SMOOTHQUANT_ALPHA=0.5 \
      SMOOTHQUANT_SCALES="$SQ_SCALES" \
      MP_CONFIG_JSON="$MP_JSON" \
      python -u "$HERE/ppl.py" 2>&1 \
    | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[$(date)] [FAIL] $SAFE_NAME / $MP_NAME (exit=$status) — continuing"  | tee -a "$LOG"
  else
    echo "[$(date)] [OK]   $SAFE_NAME / $MP_NAME"  | tee -a "$LOG"
  fi
done

echo "[$(date)] $SAFE_NAME node done."
touch "$OUTDIR/${SAFE_NAME}.done"
