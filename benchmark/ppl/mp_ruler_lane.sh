#!/bin/bash
# RULER MV-NIAH for SC-MP (frozen final recipe), one model per job.
#
# The MP budget is an input to the OFFLINE solve, so for RULER's ctx-4096
# deployment geometry we recalibrate the threshold tables at --ctx_len 4096
# with attention MAC weights corrected for 4096 (qk/av macs/row scale with N;
# traces *_sc_int7_trace_ctx4096.json hold the doubled composition). Same
# wikitext calibration data, same recipe, same code — only the deployment
# context changes, and the Lagrangian re-solves so the MAC-weighted average
# holds at the label AT RULER GEOMETRY (no drift, no manual tuning).
# Eval: RULER quant mode -> benchmark.quant.eval_quant.build_model, which
# routes QUANT_CONFIG=mp_* + MP_CONFIG_JSON to the same MP model as PPL.
# Protocol: niah_multivalue, ctx 4096, NUM_SAMPLES=50 (user decision
# 2026-07-14; prepare.py seed 42 is prefix-stable, so the 50 samples are
# byte-identical to the first 50 of the 200-sample baseline runs — baselines
# are rescored on the same first-50 subset). RULER_BATCH=1: batching was
# measured useless for MP (per-(B*H, level) kernel-launch loop in
# sc_common._sc_attention_matmul_ab_t dominates; batch multiplies launches).
# Usage: mp_ruler_lane.sh <short:4B|llama8B|14B|30B> <tag>
#   env BUDGETS="avg96f avg64f avg48f avg32"  — subset for per-cell jobs
#   env RULER_SAMPLES=50
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm
# same model cache as hpca (default ~/.cache hits the full home quota)
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"

SHORT="${1:?usage: mp_ruler_lane.sh <short> <tag>}"
TAG="${2:?usage: mp_ruler_lane.sh <short> <tag>}"

declare -A HF=(   [4B]="Qwen/Qwen3-4B-Instruct-2507" [llama8B]="meta-llama/Llama-3.1-8B-Instruct"
                  [14B]="Qwen/Qwen3-14B" [30B]="Qwen/Qwen3-30B-A3B-Instruct-2507" )
declare -A RNAME=([4B]="qwen3-4b" [llama8B]="llama-3.1-8b" [14B]="qwen3-14b" [30B]="qwen3-30b-a3b" )
hf="${HF[$SHORT]}"; rname="${RNAME[$SHORT]}"
sm="$(echo "$hf" | tr '/' '_')"

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
TABLES="$TURBO/mp_calib_${TAG}"
RESULTS="$TURBO/results/results_${TAG}.tsv"
TRACE="/home/allenjin/Projects/hpca_results/llm/uniform/traces/${SHORT}_sc_int7_trace_ctx4096.json"
HYB="$TURBO/hybrid_configs/_hpca_mp_final_sweep_20260713_143745/${SHORT}_mp_avg96f_burst128_rank-sc_int8_measured_curve_top0p10.json"
OV="down_proj:0.06,up_proj:0.03,gate_proj:0.03"
mkdir -p "$TABLES"
[[ -f "$RESULTS" ]] || printf "model\tconfig\tmetric\tvalue\n" > "$RESULTS"
[[ -f "$TRACE" ]] || { echo "missing ctx4096 trace: $TRACE"; exit 21; }
[[ -f "$HYB" ]]   || { echo "missing hybrid config: $HYB"; exit 22; }

run_budget() {  # <name> <ratio> <levels-csv>
  local name="$1" ratio="$2" levels="$3"
  local table="$TABLES/${sm}__${name}_ctx4096__final_recipe.json"
  local wrapper="$TABLES/${sm}__${name}_ctx4096__final_recipe_wrapper.json"

  if [[ ! -s "$table" ]]; then
    env SQ_ALPHA=0.5 ACT_SCALES_DIR="$TURBO/act_scales" \
        SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 \
        SC_HYBRID_CONFIG_JSON="$HYB" SC_HYBRID_INT_BITS=7 SC_HYBRID_FORCE_INT_BITS=1 \
        python -u benchmark/ppl/calibrate_mp_thresholds.py \
          --model_path "$hf" \
          --mp_levels "$levels" \
          --budget_ratio "$ratio" \
          --budget_ref_stoc_len 128 \
          --sc_prec 8 --halve 1 --seed 0 \
          --ctx_len 4096 \
          --budget-scope global \
          --budget-weight macs \
          --mac-weights-trace "$TRACE" \
          --calib-smoothquant \
          --protect-channel-frac 0.01 --protect-channel-stoc-len 128 \
          --protect-channel-metric act_collapse \
          --protect-channel-frac-overrides "$OV" \
          --protect-compensate-budget \
          --argmin-pricing macs --metric-select auto --refine sigma \
          --output_json "$table"
    local rc=$?
    [[ $rc -ne 0 || ! -s "$table" ]] && { echo "CALIB_FAILED $name rc=$rc"; return 1; }
  fi
  python -c "import json;json.dump({'type':'AdaptiveMPConfig','stoc_len_levels':[int(x) for x in '$levels'.split(',')],'threshold_table_path':'$table'},open('$wrapper','w'))"

  local qcfg="mp_${name}"
  # idempotent: skip the eval if a summary already exists (delete the pred dir
  # to force a re-run); keeps lane relaunches from redoing completed cells.
  local sumdir="benchmark/ruler/ruler_eval_result/${rname}/synthetic/4096/quant_${qcfg}/pred"
  if ls "$sumdir"/summary*.csv >/dev/null 2>&1; then
    echo "[skip] $SHORT $qcfg — summary exists"
  else
  ( cd benchmark/ruler && \
    env QUANT_CONFIG="$qcfg" MP_CONFIG_JSON="$wrapper" \
        SQ_ALPHA=0.5 ACT_SCALES_DIR="$TURBO/act_scales" \
        SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 \
        SC_HYBRID_CONFIG_JSON="$HYB" SC_HYBRID_INT_BITS=7 SC_HYBRID_FORCE_INT_BITS=1 \
        NUM_SAMPLES="${RULER_SAMPLES:-50}" RULER_BATCH=1 \
        bash ruler_run.sh "$rname" synthetic 4096 niah_multivalue quant fp16 cuda:0 )
  fi

  local summary="benchmark/ruler/ruler_eval_result/${rname}/synthetic/4096/quant_${qcfg}/pred/summary-niah_multivalue.csv"
  [[ -f "$summary" ]] || summary="benchmark/ruler/ruler_eval_result/${rname}/synthetic/4096/quant_${qcfg}/pred/summary.csv"
  if [[ -f "$summary" ]]; then
    local score nulls
    score=$(awk -F, '$1=="Score"{print $2}' "$summary" | tail -1)
    nulls=$(awk -F, '$1=="Nulls"{print $2}' "$summary" | tail -1)
    printf "%s\t%s\t%s\t%s\n" "$SHORT" "$qcfg" "ruler_niah_mv" "${score:-PARSE_FAIL}" >> "$RESULTS"
    printf "%s\t%s\t%s\t%s\n" "$SHORT" "$qcfg" "ruler_nulls" "${nulls:-?}" >> "$RESULTS"
    echo "[RULER-MP RESULT] model=$SHORT config=$qcfg score=${score:-PARSE_FAIL} nulls=${nulls:-?}"
  else
    printf "%s\t%s\t%s\t%s\n" "$SHORT" "$qcfg" "ruler_niah_mv" "FAILED(no summary)" >> "$RESULTS"
    echo "[RULER-MP RESULT] model=$SHORT config=$qcfg score=FAILED"
  fi
}

for b in ${BUDGETS:-avg96f avg64f avg48f avg32}; do
  case "$b" in
    avg96f) run_budget avg96f 0.75  128,96,64,48,32 ;;
    avg64f) run_budget avg64f 0.5   128,96,64,48,32 ;;
    avg48f) run_budget avg48f 0.375 128,96,64,48,32 ;;
    avg32)  run_budget avg32  0.25  128,64,48,32,16 ;;
    *) echo "unknown budget: $b"; exit 3 ;;
  esac
done
echo "[lane done] $SHORT ${BUDGETS:-all}"
