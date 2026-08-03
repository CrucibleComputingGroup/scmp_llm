#!/bin/bash
# Regenerate precision TRACES for the completed act_global_v2 sweep
# (tag mp_v2_hyb_pc_20260710_201348). The original evals did not set
# SC_MP_TRACE, so this re-runs eval_quant per cell REUSING the existing
# calibration wrappers + hybrid configs (no recalibration) with tracing on.
# Tracing is bit-identical on PPL, so the printed PPL doubles as a
# replication check against results_mp_v2_hyb_pc_20260710_201348.tsv.
# Trace naming follows hpca_results/llm/uniform/traces convention:
#   <short>_<config>_trace.json  ->  hpca_results/llm/mp_v2/traces/
# Usage: mp_v2_trace_lane.sh <model-short>
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm

SHORT="${1:?usage: mp_v2_trace_lane.sh <4B|llama8B|14B|30B>}"
TAG=mp_v2_hyb_pc_20260710_201348
TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
TABLES=$TURBO/mp_calib_${TAG}
HYB=$TURBO/hybrid_configs/_hpca_${TAG}
OUT=/home/allenjin/Projects/hpca_results/llm/mp_v2/traces

declare -A HF=(
  [4B]=Qwen/Qwen3-4B-Instruct-2507
  [llama8B]=meta-llama/Llama-3.1-8B-Instruct
  [14B]=Qwen/Qwen3-14B
  [30B]=Qwen/Qwen3-30B-A3B-Instruct-2507
)
hf="${HF[$SHORT]}"
sm="${hf//\//_}"

for CFG in mp_avg96_burst128 mp_avg64_burst128 mp_avg48_burst128; do
  # hpca budget->INT-bits mapping: avg96 -> INT8, avg64/avg48 -> INT7
  case "$CFG" in
    mp_avg96_burst128) BITS=8 ;;
    *)                 BITS=7 ;;
  esac
  WRAPPER=$TABLES/${sm}__${CFG}__act_global_v2_pc0.01x128_act_weight_comp_hyb0.10_wrapper.json
  HYBJSON=$HYB/${SHORT}_${CFG}_rank-sc_int8_measured_curve_top0p10.json
  TRACE=$OUT/${SHORT}_${CFG}_trace.json
  if [[ -s "$TRACE" ]]; then
    echo "=== $SHORT $CFG: trace exists, skipping"
    continue
  fi
  [[ -s "$WRAPPER" ]] || { echo "!!! missing wrapper $WRAPPER"; exit 21; }
  [[ -s "$HYBJSON" ]] || { echo "!!! missing hybrid config $HYBJSON"; exit 22; }
  echo "=== $SHORT $CFG trace eval start $(date)"
  env MODEL_PATH="$hf" QUANT_CONFIG=mp \
      SQ_ALPHA=0.5 CTX=2048 PPL_MAX_TOKENS=0 \
      SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 \
      MP_CONFIG_JSON="$WRAPPER" \
      ACT_SCALES_DIR=$TURBO/act_scales \
      SC_HYBRID_CONFIG_JSON="$HYBJSON" \
      SC_HYBRID_INT_BITS=$BITS SC_HYBRID_FORCE_INT_BITS=1 \
      SC_MP_TRACE="$TRACE" SC_MP_TRACE_MODE=summary \
      python -u benchmark/quant/eval_quant.py
  rc=$?
  if [[ $rc -ne 0 || ! -s "$TRACE" ]]; then
    echo "!!! FAILED $SHORT $CFG rc=$rc"
    exit 23
  fi
  echo "=== $SHORT $CFG trace done $(date)"
done
echo "=== lane $SHORT complete"
