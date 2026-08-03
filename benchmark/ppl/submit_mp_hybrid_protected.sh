#!/bin/bash
# Submit MP + hybrid INT + protected-channel experiments.
#
# Fresh calibration per cell:
#   1. top-fraction operator-layer hybrid mask -> INT
#   2. act_global MP calibrated on remaining SC graph
#   3. protected input channels run at 128 cycles with budget compensation
#
# Usage:
#   bash benchmark/ppl/submit_mp_hybrid_protected.sh
#   bash benchmark/ppl/submit_mp_hybrid_protected.sh --models 4B,llama8B --dry-run
set -uo pipefail

MODELS_CSV="4B,llama8B,14B,30B"
BUDGETS_CSV="len192,int7,len96_burst"
TAG="mp_hyb_pc_$(date +%Y%m%d_%H%M%S)"
SENS_DIR="/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921"
HYBRID_FRAC="0.10"
PROTECT_FRAC="0.01"
ACCOUNT="nbleier_owned1"
RESERVATION="rtx6000_arph_nodes"
PARTITION="gpu-rtx6000"
CONDA_ENV="annstention"
MEM="180G"
CPUS="12"
TIME="24:00:00"
DRY=0

TURBO="/nfs/turbo/coe-nbleier/allenjin/hpca"
SCRATCH="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS_CSV="$2"; shift 2 ;;
    --budgets) BUDGETS_CSV="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --sens-dir) SENS_DIR="$2"; shift 2 ;;
    --hybrid-frac) HYBRID_FRAC="$2"; shift 2 ;;
    --protect-frac) PROTECT_FRAC="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --reservation) RESERVATION="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '1,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

declare -A MODEL_ID=(
  [4B]="Qwen/Qwen3-4B-Instruct-2507"
  [14B]="Qwen/Qwen3-14B"
  [30B]="Qwen/Qwen3-30B-A3B-Instruct-2507"
  [llama8B]="meta-llama/Llama-3.1-8B-Instruct"
)

model_layers() {
  case "$1" in
    4B) echo 36 ;;
    llama8B) echo 32 ;;
    14B) echo 40 ;;
    30B) echo 48 ;;
    *) return 1 ;;
  esac
}

budget_spec() {
  case "$1" in
    len192) echo "128,96,64 0.75 8" ;;
    int7) echo "128,64,32 0.5 7" ;;
    len96_burst) echo "128,64,32 0.375 7" ;;
    *) return 1 ;;
  esac
}

safe() { echo "$1" | tr '/:' '__'; }

make_hybrid_config() {  # <model> <budget> <int_bits>
  local model="$1" budget="$2" int_bits="$3"
  local layers sens outdir out
  layers="$(model_layers "$model")" || return 1
  sens="$SENS_DIR/${model}_sc_int8_measured_curve.json"
  outdir="$TURBO/hybrid_configs/_hpca_${TAG}"
  out="$outdir/${model}_${budget}_rank-sc_int8_layer_top${HYBRID_FRAC//./p}.json"
  if [[ ! -s "$sens" ]]; then
    echo "[submit-mp-hyb] missing sensitivity: $sens" >&2
    return 1
  fi
  if [[ $DRY -eq 1 ]]; then
    echo "$out"
    return
  fi
  mkdir -p "$outdir"
  python - "$sens" "$out" "$layers" "$HYBRID_FRAC" "$int_bits" <<'PY'
import json, math, sys
sens_path, out_path, layers_s, frac_s, bits_s = sys.argv[1:6]
num_layers, frac, bits = int(layers_s), float(frac_s), int(bits_s)
with open(sens_path) as f:
    sens = json.load(f)
layer_buckets = int(sens.get("layer_buckets", 1))
schedule, candidates = {}, []
for key, rec in sens.get("buckets", {}).items():
    op, _t, lpart = key.split(":")
    lb = int(lpart[1:])
    errs = [float(x) for x in rec.get("level_mean_error", [])]
    if len(errs) < 2:
        continue
    score = max(errs[1:])
    blocks = [
        b for b in range(num_layers)
        if min(layer_buckets - 1, (b * layer_buckets) // max(num_layers, 1)) == lb
    ]
    for b in blocks:
        candidates.append((score, op, b, key))
target = int(math.ceil(len(candidates) * frac))
chosen = sorted(candidates, key=lambda x: x[0], reverse=True)[:target]
for _score, op, block, _key in chosen:
    row = schedule.setdefault(op, ["sc"] * num_layers)
    row[block] = f"int{bits}"
payload = {
    "format": "scmp_llm_hybrid_v1",
    "default": "sc",
    "int_bits": bits,
    "int_sym": True,
    "chunk_size": 128,
    "source_sensitivity": sens_path,
    "selection": {
        "method": "top_fraction_by_operator_layer_worst_delta_loss",
        "fraction": frac,
        "selected_entries": len(chosen),
        "total_entries": len(candidates),
        "selected_fraction": (len(chosen) / len(candidates)) if candidates else 0.0,
    },
    "schedule": schedule,
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
print(out_path)
PY
}

IFS=, read -ra MODELS <<<"$MODELS_CSV"
IFS=, read -ra BUDGETS <<<"$BUDGETS_CSV"

TABLES="$TURBO/mp_calib_hybrid_protected_${TAG}"
RESULTS="/home/allenjin/Projects/hpca_results/llm/mp/hybrid_protected_${TAG}_results.tsv"
MANIFEST="/home/allenjin/Projects/hpca_results/llm/mp/hybrid_protected_${TAG}_manifest.tsv"
LOGDIR="$SCRATCH/logs/_mp_hybrid_protected_${TAG}"

if [[ $DRY -eq 0 ]]; then
  mkdir -p "$TABLES" "$LOGDIR" "$(dirname "$RESULTS")"
  [[ -f "$RESULTS" ]] || printf "model\tbudget\tmethod\thybrid_frac\tprotect_frac\texpected_flop_avg_sl\trealized_avg_sl\trealized_flop_avg_sl\tppl\tstatus\n" > "$RESULTS"
  [[ -f "$MANIFEST" ]] || printf "model\tbudget\tmethod\thybrid_frac\tprotect_frac\tppl\texpected_flop_avg_sl\trealized_flop_avg_sl\twrapper\ttable\thybrid_config\n" > "$MANIFEST"
fi

echo "[submit-mp-hyb] tag=$TAG models=$MODELS_CSV budgets=$BUDGETS_CSV"
echo "[submit-mp-hyb] tables=$TABLES"
echo "[submit-mp-hyb] results=$RESULTS"

for model in "${MODELS[@]}"; do
  hf="${MODEL_ID[$model]:-}"
  if [[ -z "$hf" ]]; then echo "[submit-mp-hyb] skip unknown model $model" >&2; continue; fi
  sm="$(safe "$hf")"
  trace="/home/allenjin/Projects/hpca_results/llm/uniform/traces/${model}_sc_int7_trace.json"
  for budget in "${BUDGETS[@]}"; do
    spec="$(budget_spec "$budget")" || { echo "[submit-mp-hyb] bad budget $budget" >&2; exit 1; }
    read -r levels ratio int_bits <<<"$spec"
    hybrid_cfg="$(make_hybrid_config "$model" "$budget" "$int_bits")" || exit 1
    method="act_global_pc${PROTECT_FRAC}x128_comp_int${HYBRID_FRAC}"
    table="$TABLES/${sm}__${budget}__${method}.json"
    wrapper="$TABLES/${sm}__${budget}__${method}_wrapper.json"
    log="$LOGDIR/${model}__${budget}__${method}.log"
    job="mph_${model}_${budget}"
    safe_job="${job//[^A-Za-z0-9_.-]/_}"
    payload="source ~/.bashrc; conda activate $CONDA_ENV; cd $HERE; mkdir -p $TABLES $LOGDIR $(dirname "$RESULTS"); [[ -f $RESULTS ]] || printf 'model\\tbudget\\tmethod\\thybrid_frac\\tprotect_frac\\texpected_flop_avg_sl\\trealized_avg_sl\\trealized_flop_avg_sl\\tppl\\tstatus\\n' > $RESULTS; [[ -f $MANIFEST ]] || printf 'model\\tbudget\\tmethod\\thybrid_frac\\tprotect_frac\\tppl\\texpected_flop_avg_sl\\trealized_flop_avg_sl\\twrapper\\ttable\\thybrid_config\\n' > $MANIFEST; export ACT_SCALES_DIR=$TURBO/act_scales HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache TRANSFORMERS_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache HF_HUB_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache/hub HF_DATASETS_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache/datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$SCRATCH/tmp; mkdir -p \\$TMPDIR; { echo '=== CELL $model $budget $method ==='; echo 'levels=$levels ratio=$ratio int_bits=$int_bits hybrid=$hybrid_cfg'; env SC_HYBRID_CONFIG_JSON=$hybrid_cfg SC_HYBRID_INT_BITS=$int_bits SC_HYBRID_FORCE_INT_BITS=1 SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 SQ_ALPHA=0.5 timeout -k 2m 24h python -u benchmark/ppl/calibrate_mp_thresholds.py --model_path $hf --mp_levels $levels --budget_ratio $ratio --budget_ref_stoc_len 128 --sc_prec 8 --halve 1 --ctx_len 2048 --budget-scope global --budget-weight macs --mac-weights-trace $trace --calib-smoothquant --protect-channel-frac $PROTECT_FRAC --protect-channel-stoc-len 128 --protect-channel-metric act_weight --protect-compensate-budget --output_json $table; rc=\\$?; if [[ \\$rc -ne 0 || ! -s $table ]]; then echo CALIB_FAILED rc=\\$rc; exit 21; fi; python -c \"import json;json.dump({'type':'AdaptiveMPConfig','stoc_len_levels':[int(x) for x in '$levels'.split(',')],'threshold_table_path':'$table'},open('$wrapper','w'))\"; env MODEL_PATH=$hf QUANT_CONFIG=mp SQ_ALPHA=0.5 PPL_MAX_TOKENS=0 CTX=2048 SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 MP_CONFIG_JSON=$wrapper SC_HYBRID_CONFIG_JSON=$hybrid_cfg SC_HYBRID_INT_BITS=$int_bits SC_HYBRID_FORCE_INT_BITS=1 ACT_SCALES_DIR=$TURBO/act_scales timeout -k 2m 24h python -u benchmark/quant/eval_quant.py; } > $log 2>&1; ppl=\\$(grep '\\[RESULT\\]' $log | grep -oE 'value=[0-9.eE+-]+|value=nan|value=inf' | sed 's/value=//' | tail -1); avg=\\$(grep '\\[RESULT\\]' $log | grep -oE 'realized_avg_sl=[0-9.]+' | sed 's/realized_avg_sl=//' | tail -1); flop=\\$(grep '\\[RESULT\\]' $log | grep -oE 'realized_flop_avg_sl=[0-9.]+' | sed 's/realized_flop_avg_sl=//' | tail -1); exp=\\$(python -c \"import json;print(round(json.load(open('$table')).get('expected_flop_avg_stoc_len',0),4))\" 2>/dev/null); if [[ -n \\$ppl ]]; then status=OK; else status=EVAL_FAILED; ppl=-; fi; ( flock 9; printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' $model $budget $method $HYBRID_FRAC $PROTECT_FRAC \\${exp:--} \\${avg:--} \\${flop:--} \\$ppl \\$status >> $RESULTS ) 9>$RESULTS.lock; ( flock 8; printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' $model $budget $method $HYBRID_FRAC $PROTECT_FRAC \\$ppl \\${exp:--} \\${flop:--} $wrapper $table $hybrid_cfg >> $MANIFEST ) 8>$MANIFEST.lock"
    if [[ $DRY -eq 1 ]]; then
      echo "DRY: sbatch --job-name=$safe_job ... --wrap=$payload"
    else
      sbatch --job-name="$safe_job" \
        --account="$ACCOUNT" --reservation="$RESERVATION" \
        --partition="$PARTITION" --gres=gpu:1 --cpus-per-task="$CPUS" \
        --mem="$MEM" --time="$TIME" \
        --output="$SCRATCH/logs/${safe_job}_%j.out" \
        --wrap="$payload"
    fi
  done
done

echo "TAG=$TAG"
