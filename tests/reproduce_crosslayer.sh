#!/bin/bash
# reproduce_crosslayer.sh — one command to reproduce the cross-layer SC
# mixed-precision comparison (act / act_global / grad_global / measured across
# int8 + the MP precisions, on Qwen3-4B/8B/14B).
#
# It wraps tests/run_mp_sweep.sh (which handles conda activate, HF cache, the
# SmoothQuant + threshold calibration, and the wikitext2 PPL eval) and then
# prints the summary table via benchmark/ppl/summarize_mp_results.py.
#
# Defaults reproduce the headline result:
#   models   4B,8B,14B
#   prec     int8,int7,len96,len192   (int8 = uniform ceiling)
#   method   act,act_global,grad_global,measured
#   Owen     bitrev (set inside run_mp_sweep.sh), --recalibrate forces fresh tables.
#
# Runs EVERYTHING SERIALLY on the current GPU node — this is the portable,
# click-and-run path. It is long (~1 day across all 3 models). To go faster,
# launch one model per GPU node in parallel, e.g. on three allocations:
#   MODELS=4B  TAG=crosslayer_repro bash tests/reproduce_crosslayer.sh   # node 1
#   MODELS=8B  TAG=crosslayer_repro bash tests/reproduce_crosslayer.sh   # node 2
#   MODELS=14B TAG=crosslayer_repro bash tests/reproduce_crosslayer.sh   # node 3
# (same TAG → shared outdir; the summarizer picks up whatever has finished.)
#
# Env overrides: MODELS, PREC, METHOD, TAG, plus anything run_mp_sweep.sh reads
# (CONDA_ENV, HF cache, MAX_TOKENS, CTX, ...). Extra flags pass through to the
# sweep, e.g.:  bash tests/reproduce_crosslayer.sh --max-tokens 4096   (smoke).
set -euo pipefail

MODELS="${MODELS:-4B,8B,14B}"
PREC="${PREC:-int8,int7,len96,len192}"
METHOD="${METHOD:-act,act_global,grad_global,measured}"
TAG="${TAG:-crosslayer_repro}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUTDIR="$REPO/benchmark/ppl/_mp_overnight_${TAG}"

echo "[reproduce] models=$MODELS prec=$PREC method=$METHOD tag=$TAG"
echo "[reproduce] outdir=$OUTDIR"

bash "$HERE/run_mp_sweep.sh" \
  --recalibrate \
  --tag "$TAG" \
  --models "$MODELS" \
  --prec "$PREC" \
  --method "$METHOD" \
  "$@"

echo
echo "[reproduce] ===== SUMMARY ====="
# Use the sweep's conda env python (run_mp_sweep.sh already activated it for its
# own subprocesses; re-activate here so the summarizer runs even in a fresh shell).
PY="$(command -v python)"
"$PY" "$REPO/benchmark/ppl/summarize_mp_results.py" "$OUTDIR" || \
  echo "[reproduce] summary parse failed; logs are in $OUTDIR"
