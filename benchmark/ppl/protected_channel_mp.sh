#!/bin/bash
# 4B protected-channel MP experiment.
# Tests whether a tiny offline-selected input-channel subspace at 128 cycles can
# pull the aggressive avg-48 point toward the INT baseline.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$HERE"

GPU="${GPU:-0}"
MODEL="${MODEL:-4B}"
HF="${HF:-Qwen/Qwen3-4B-Instruct-2507}"
CTX="${CTX:-2048}"
PPL_MAX_TOKENS="${PPL_MAX_TOKENS:-0}"
SQ_ALPHA="${SQ_ALPHA:-0.5}"
CELL_TIMEOUT="${CELL_TIMEOUT:-8h}"
SC_OWEN_MODE="${SC_OWEN_MODE:-bitrev}"
SC_SCRAMBLE_MASKS="${SC_SCRAMBLE_MASKS:-64}"

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
SCRATCH=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca
TABLES="${TABLES:-$TURBO/mp_calib_protected}"
RESULTS="${RESULTS:-/home/allenjin/Projects/hpca_results/llm/mp/protected_channel_results.tsv}"
MANIFEST="${MANIFEST:-/home/allenjin/Projects/hpca_results/llm/mp/protected_channel_manifest.tsv}"
LOGDIR="${LOGDIR:-$SCRATCH/logs/_mp_protected_$(date +%Y%m%d_%H%M)}"
TRACE="${TRACE:-/home/allenjin/Projects/hpca_results/llm/uniform/traces/${MODEL}_sc_int7_trace.json}"

export ACT_SCALES_DIR="${ACT_SCALES_DIR:-$TURBO/act_scales}"
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$TABLES" "$LOGDIR" "$TMPDIR" "$(dirname "$RESULTS")"

[[ -f "$RESULTS" ]] || printf "model\tbudget\tmethod\tprotect_frac\tcompensate\texpected_flop_avg_sl\trealized_avg_sl\trealized_flop_avg_sl\tppl\tstatus\n" > "$RESULTS"
[[ -f "$MANIFEST" ]] || printf "model\tbudget\tmethod\tprotect_frac\tcompensate\tppl\texpected_flop_avg_sl\trealized_flop_avg_sl\twrapper\tcalib_command\n" > "$MANIFEST"

safe(){ echo "$1" | tr '/' '_'; }

append_result(){
  ( flock 9; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$RESULTS" ) 9>"$RESULTS.lock"
}

append_manifest(){  # method frac comp ppl exp flop wrapper table
  python3 - "$MODEL" "len96_burst" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$MANIFEST" <<'PYEOF'
import json, sys, fcntl
model, budget, method, frac, comp, ppl, exp, flop, wrapper, table, manifest = sys.argv[1:12]
d = json.load(open(table))
row = [model, budget, method, frac, comp, ppl, exp, flop, wrapper, d.get("calib_command", "")]
with open(manifest, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write("\t".join(row) + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)
PYEOF
}

run_cell(){  # method protect_frac compensate
  local method="$1" frac="$2" comp="$3"
  local sm; sm="$(safe "$HF")"
  local tag="${MODEL}__len96_burst__${method}"
  local table="$TABLES/${sm}__len96_burst__${method}.json"
  local wrapper="$TABLES/${sm}__len96_burst__${method}_wrapper.json"
  local log="$LOGDIR/${tag}.log"
  if grep -qP "^${MODEL}\tlen96_burst\t${method}\t.*\tOK$" "$RESULTS" 2>/dev/null; then
    echo "[skip] $tag already complete"
    return
  fi
  local protect_flags=""
  if [[ "$frac" != "0" && "$frac" != "0.0" ]]; then
    protect_flags="--protect-channel-frac $frac --protect-channel-stoc-len 128 --protect-channel-metric act_weight"
    [[ "$comp" == "1" ]] && protect_flags="$protect_flags --protect-compensate-budget"
  fi
  echo "[gpu$GPU] >>> $tag frac=$frac comp=$comp $(date +%H:%M)"
  {
    echo "=== CELL $tag gpu=$GPU $(date) ==="
    echo "levels=128,64,32 target=48 protect_flags=[$protect_flags]"
    if [[ ! -s "$table" ]]; then
      env CUDA_VISIBLE_DEVICES="$GPU" SC_OWEN_MODE="$SC_OWEN_MODE" SC_SCRAMBLE_MASKS="$SC_SCRAMBLE_MASKS" SQ_ALPHA="$SQ_ALPHA" \
        timeout -k 2m "$CELL_TIMEOUT" python -u benchmark/ppl/calibrate_mp_thresholds.py \
          --model_path "$HF" --mp_levels 128,64,32 --budget_ratio 0.375 \
          --budget_ref_stoc_len 128 --sc_prec 8 --halve 1 --ctx_len "$CTX" \
          --budget-scope global --budget-weight macs --mac-weights-trace "$TRACE" \
          --calib-smoothquant $protect_flags --output_json "$table" || echo "CALIB_RC=$?"
    else
      echo "[calib] reuse $table"
    fi
    if [[ ! -s "$table" ]]; then echo "=== $tag CALIB FAILED ==="; exit 21; fi
    python -c "import json;json.dump({'type':'AdaptiveMPConfig','stoc_len_levels':[128,64,32],'threshold_table_path':'$table'},open('$wrapper','w'))"
    env CUDA_VISIBLE_DEVICES="$GPU" MODEL_PATH="$HF" QUANT_CONFIG=mp SQ_ALPHA="$SQ_ALPHA" \
      PPL_MAX_TOKENS="$PPL_MAX_TOKENS" CTX="$CTX" SC_OWEN_MODE="$SC_OWEN_MODE" \
      SC_SCRAMBLE_MASKS="$SC_SCRAMBLE_MASKS" MP_CONFIG_JSON="$wrapper" ACT_SCALES_DIR="$ACT_SCALES_DIR" \
      timeout -k 2m "$CELL_TIMEOUT" python -u benchmark/quant/eval_quant.py
  } >> "$log" 2>&1
  local ppl avg flop exp
  ppl=$(grep "\[RESULT\]" "$log" | grep -oE "value=[0-9.eE+-]+|value=nan|value=inf" | sed 's/value=//' | tail -1)
  avg=$(grep "\[RESULT\]" "$log" | grep -oE "realized_avg_sl=[0-9.]+" | sed 's/realized_avg_sl=//' | tail -1)
  flop=$(grep "\[RESULT\]" "$log" | grep -oE "realized_flop_avg_sl=[0-9.]+" | sed 's/realized_flop_avg_sl=//' | tail -1)
  exp=$(python -c "import json;print(round(json.load(open('$table')).get('expected_flop_avg_stoc_len',0),4))" 2>/dev/null)
  if [[ -n "$ppl" ]]; then
    append_result "$MODEL" "len96_burst" "$method" "$frac" "$comp" "${exp:--}" "${avg:--}" "${flop:--}" "$ppl" "OK"
    append_manifest "$method" "$frac" "$comp" "$ppl" "${exp:--}" "${flop:--}" "$wrapper" "$table"
    echo "[gpu$GPU] [done] $tag ppl=$ppl exp_flop=${exp:--} realized_flop=${flop:--}"
  elif [[ ! -s "$table" ]]; then
    append_result "$MODEL" "len96_burst" "$method" "$frac" "$comp" "-" "-" "-" "-" "CALIB_FAILED"
    echo "[gpu$GPU] [FAIL] $tag calib"
  else
    append_result "$MODEL" "len96_burst" "$method" "$frac" "$comp" "${exp:--}" "${avg:--}" "${flop:--}" "-" "EVAL_FAILED"
    echo "[gpu$GPU] [FAIL] $tag eval (see $log)"
  fi
}

echo "[protected] $(date) model=$MODEL gpu=$GPU results=$RESULTS logdir=$LOGDIR"
run_cell burst_act_global 0 0
run_cell pc1_act_weight 0.01 0
run_cell pc2_act_weight 0.02 0
run_cell pc1_act_weight_comp 0.01 1
echo "[protected] DONE $(date)"
column -t -s$'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
