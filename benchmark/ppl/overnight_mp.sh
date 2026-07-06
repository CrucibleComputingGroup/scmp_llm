#!/bin/bash
# Overnight MP sweep: per (model,budget,method) cell = calibrate table -> wrapper
# -> eval PPL (HPCA protocol, same as mp_all.csv). 8-wide GPU work-queue.
# Idempotent (skips cells already in RESULTS), failure-isolated, incremental
# writes so partial results accumulate. Budgets named in NOMINAL space
# (int7=nom128, len96=nom96, len192=nom192); --mp_levels are HALVED (CLAUDE.md).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$HERE"

NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"; NGPU="${NGPU:-1}"; [[ "$NGPU" -lt 1 ]] && NGPU=1
# CALIB_SQ=1 (default): calibrate σ under the SAME SmoothQuant transform the
# eval deploys (α from SQ_ALPHA, scales from ACT_SCALES_DIR) — fixes the
# calib/deploy mismatch (open issue #5). Set CALIB_SQ=0 for the legacy
# unsmoothed calibration (only for reproducing pre-2026-07-05 fw_* tables).
CALIB_SQ="${CALIB_SQ:-1}"
PPL_MAX_TOKENS="${PPL_MAX_TOKENS:-0}"        # 0 = full wikitext-2 (comparable to baselines)
CTX="${CTX:-2048}"; SQ_ALPHA="${SQ_ALPHA:-0.5}"
SC_OWEN_MODE="${SC_OWEN_MODE:-bitrev}"; SC_SCRAMBLE_MASKS="${SC_SCRAMBLE_MASKS:-64}"
CELL_TIMEOUT="${CELL_TIMEOUT:-6h}"
TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
SCRATCH=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca
TABLES="${TABLES:-$TURBO/mp_calib_overnight}"
RESULTS="${RESULTS:-/home/allenjin/Projects/hpca_results/llm/mp/overnight_results.tsv}"
MANIFEST="${MANIFEST:-/home/allenjin/Projects/hpca_results/llm/mp/manifest.tsv}"
LOGDIR="${LOGDIR:-$SCRATCH/logs/_mp_overnight_$(date +%Y%m%d_%H%M)}"
export ACT_SCALES_DIR="$TURBO/act_scales"
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="$SCRATCH/tmp"; mkdir -p "$TABLES" "$LOGDIR" "$TMPDIR" "$(dirname "$RESULTS")"
[[ -f "$RESULTS" ]] || printf "model\tbudget\tmethod\trealized_avg_sl\tppl\tstatus\n" > "$RESULTS"
[[ -f "$MANIFEST" ]] || printf "model\tbudget\tmethod\tppl\trealized_avg_sl\texpected_avg\tlevels\tbudget_ratio\twrapper\tcalib_command\n" > "$MANIFEST"

declare -A MODEL_ID=(
  [1.7B]="Qwen/Qwen3-1.7B" [4B]="Qwen/Qwen3-4B-Instruct-2507"
  [14B]="Qwen/Qwen3-14B" [30B]="Qwen/Qwen3-30B-A3B-Instruct-2507"
  [llama8B]="meta-llama/Llama-3.1-8B-Instruct")

budget_levels(){ case "$1" in           # -> "<halved levels> <budget_ratio>"  (ref=128)
  int7)   echo "128,64,32 0.5"  ;;       # nominal 128, target halved 64
  len96)  echo "64,48,32 0.375" ;;       # nominal 96,  target halved 48
  len192) echo "128,96,64 0.75" ;;       # nominal 192, target halved 96
  *) echo "";; esac; }
method_flags(){ case "$1" in
  act_global)  echo "--budget-scope global" ;;
  measured)    echo "--budget-scope global --cross-layer-weight measured" ;;
  fisher)      echo "--budget-scope global --cross-layer-weight fisher" ;;
  fisher_attn) echo "--budget-scope global --cross-layer-weight fisher --fisher-attn" ;;
  measured_curve)     echo "--budget-scope global --cross-layer-weight measured_curve" ;;
  measured_curve_sq)  echo "--budget-scope global --cross-layer-weight measured_curve --calib-smoothquant" ;;
  *) echo "";; esac; }
safe(){ echo "$1" | tr '/' '_'; }

append(){ ( flock 9; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$RESULTS" ) 9>"$RESULTS.lock"; }

# Grep-and-reuse index: one row per calibrated+evaluated cell -> deployable
# wrapper + achieved PPL + the EXACT calib_command (read from the table). To
# reuse a cell later without recalibration: grep this file for the wrapper, then
# MP_CONFIG_JSON=<wrapper> ... eval_quant.py. To reproduce: run its calib_command.
manifest_append(){  # model budget method ppl avg table wrapper
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$MANIFEST" <<'PYEOF'
import json, sys, os, fcntl
model, budget, method, ppl, avg, table, wrapper, manifest = sys.argv[1:9]
d = {}
try: d = json.load(open(table))
except Exception: pass
row = [model, budget, method, ppl, avg,
       str(round(d.get("expected_avg_stoc_len", 0), 2)),
       ",".join(str(x) for x in d.get("stoc_len_levels", [])),
       str(d.get("budget_ratio", "")), wrapper, d.get("calib_command", "")]
with open(manifest, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX); f.write("\t".join(row) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
PYEOF
}

run_cell(){  # <gpu> <model> <budget> <method>
  local gpu="$1" model="$2" budget="$3" method="$4"
  local hf="${MODEL_ID[$model]:-}"; [[ -z "$hf" ]] && { echo "[skip] unknown model $model"; return; }
  local sm; sm="$(safe "$hf")"; local tag="${model}__${budget}__${method}"
  local log="$LOGDIR/${tag}.log"
  grep -qP "^${model}\t${budget}\t${method}\t.*\tOK$" "$RESULTS" 2>/dev/null && { echo "[skip] $tag done"; return; }
  # Cross-node cell lock (mkdir is atomic on NFS): lets multiple sbatch workers
  # share one RESULTS file without duplicating in-flight cells. Held only while
  # the cell runs; done-ness is governed by the RESULTS grep above.
  local lockdir="$TABLES/.lock_${tag}"
  if ! mkdir "$lockdir" 2>/dev/null; then echo "[skip] $tag in flight elsewhere ($lockdir)"; return; fi
  local lv; lv="$(budget_levels "$budget")"; local levels="${lv% *}" ratio="${lv#* }"
  local mflags; mflags="$(method_flags "$method")"
  local table="$TABLES/${sm}__${budget}__${method}.json"
  local wrapper="$TABLES/${sm}__${budget}__${method}_wrapper.json"
  local ckpt=""; case "$model" in 14B|30B|32B) ckpt="CALIB_GRAD_CKPT=1";; esac
  # FLOP/energy-weighted budget (iso-compute): per-op MACs/row from the model's
  # sc_int7 trace. Falls back to row-weighting if no trace exists.
  local trace="${TRACE_DIR:-/home/allenjin/Projects/hpca_results/llm/uniform/traces}/${model}_sc_int7_trace.json"
  local budgetflags=""
  if [[ "${BUDGET_WEIGHT:-rows}" == "macs" ]]; then
    if [[ -s "$trace" ]]; then budgetflags="--budget-weight macs --mac-weights-trace $trace"
    else echo "[gpu$gpu] WARN: no trace $trace — row-weighted fallback for $tag"; fi
  fi
  echo "[gpu$gpu] >>> $tag  levels=$levels ratio=$ratio $(date +%H:%M)"
  {
    echo "=== CELL $tag gpu=$gpu $(date) ==="; echo "levels=$levels ratio=$ratio flags=[$mflags]"
    local sqflag=""
    [[ "$CALIB_SQ" == "1" ]] && sqflag="--calib-smoothquant"
    if [[ ! -s "$table" ]]; then
      env CUDA_VISIBLE_DEVICES="$gpu" $ckpt SC_OWEN_MODE="$SC_OWEN_MODE" SC_SCRAMBLE_MASKS="$SC_SCRAMBLE_MASKS" \
        SQ_ALPHA="$SQ_ALPHA" \
        timeout "$CELL_TIMEOUT" python -u benchmark/ppl/calibrate_mp_thresholds.py \
          --model_path "$hf" --mp_levels "$levels" --budget_ratio "$ratio" \
          --budget_ref_stoc_len 128 --sc_prec 8 --halve 1 --ctx_len "$CTX" \
          $mflags $budgetflags $sqflag --output_json "$table" || echo "CALIB_RC=$?"
    else echo "[calib] reuse $table"; fi
    if [[ ! -s "$table" ]]; then echo "=== $tag CALIB FAILED ==="; exit 21; fi
    python -c "import json;json.dump({'type':'AdaptiveMPConfig','stoc_len_levels':[int(x) for x in '$levels'.split(',')],'threshold_table_path':'$table'},open('$wrapper','w'))"
    env CUDA_VISIBLE_DEVICES="$gpu" MODEL_PATH="$hf" QUANT_CONFIG=mp SQ_ALPHA="$SQ_ALPHA" \
        PPL_MAX_TOKENS="$PPL_MAX_TOKENS" CTX="$CTX" SC_OWEN_MODE="$SC_OWEN_MODE" \
        SC_SCRAMBLE_MASKS="$SC_SCRAMBLE_MASKS" MP_CONFIG_JSON="$wrapper" ACT_SCALES_DIR="$ACT_SCALES_DIR" \
        timeout "$CELL_TIMEOUT" python -u benchmark/quant/eval_quant.py
  } >> "$log" 2>&1
  local ppl avg
  ppl=$(grep "\[RESULT\]" "$log" | grep -oE "value=[0-9.eE+-]+|value=nan|value=inf" | sed 's/value=//' | tail -1)
  avg=$(grep "\[RESULT\]" "$log" | grep -oE "realized_avg_sl=[0-9.]+" | sed 's/realized_avg_sl=//' | tail -1)
  [[ -z "$avg" && -s "$table" ]] && avg=$(python -c "import json;print(round(json.load(open('$table')).get('expected_avg_stoc_len',0),2))" 2>/dev/null)
  if [[ -n "$ppl" ]]; then append "$model" "$budget" "$method" "${avg:--}" "$ppl" "OK"; manifest_append "$model" "$budget" "$method" "$ppl" "${avg:--}" "$table" "$wrapper"; echo "[gpu$gpu] [done] $tag ppl=$ppl avg=${avg:--}"
  elif [[ ! -s "$table" ]]; then append "$model" "$budget" "$method" "-" "-" "CALIB_FAILED"; echo "[gpu$gpu] [FAIL] $tag calib"
  else append "$model" "$budget" "$method" "${avg:--}" "-" "EVAL_FAILED"; echo "[gpu$gpu] [FAIL] $tag eval (see $log)"; fi
  rmdir "$lockdir" 2>/dev/null
}

# ---------- build priority-ordered queue ----------
QUEUE="$LOGDIR/queue.txt"; : > "$QUEUE"
add(){ echo "$1 $2 $3" >> "$QUEUE"; }
if [[ "${SMOKE:-0}" == "1" ]]; then
  IFS=, read -ra _SM <<<"${SMOKE_METHODS:-act_global}"
  for w in "${_SM[@]}"; do add 4B int7 "$w"; done
else
  # Tier 0 — THE key comparison at int7, iso-compute (FLOP-weighted budget):
  #   does act_global RECOVER now that qk is cheap? is measured/measured_curve
  #   still ahead? all vs uniform sc_int7 (weighting-invariant baseline).
  for m in 4B llama8B; do for w in act_global measured measured_curve; do add $m int7 $w; done; done
  # Tier 1 — same at len96 (aggressive)
  for m in 4B llama8B; do for w in act_global measured measured_curve; do add $m len96 $w; done; done
  # Tier 2 — fisher control + cross-model breadth
  for m in 4B llama8B; do for b in int7 len96; do add $m $b fisher; done; done
  for b in int7 len96; do for w in act_global measured measured_curve; do add 14B $b $w; done; done
  for b in int7 len96; do for w in act_global measured measured_curve; do add 1.7B $b $w; done; done
  # Tier 3 — 30B tail (forward-only methods, OOM-safe)
  add 30B int7 act_global; add 30B int7 measured_curve
fi
# MODELS=14B,30B restricts the queue to those models (any other cell dropped).
if [[ -n "${MODELS:-}" ]]; then
  awk -v allow=",${MODELS}," 'index(allow, ","$1",")' "$QUEUE" > "$QUEUE.f" && mv "$QUEUE.f" "$QUEUE"
fi

pop(){ exec 8>"$QUEUE.lock"; flock 8; local l; l=$(head -n1 "$QUEUE"); [[ -n "$l" ]] && sed -i '1d' "$QUEUE"; flock -u 8; echo "$l"; }
worker(){ local g="$1"; while true; do local c; c=$(pop); [[ -z "$c" ]] && break; run_cell "$g" $c; done; }

echo "[overnight] $(date) NGPU=$NGPU cells=$(wc -l <"$QUEUE") results=$RESULTS"
echo "[overnight] logdir=$LOGDIR tables=$TABLES"
for g in $(seq 0 $((NGPU-1))); do worker "$g" & done
wait
echo "[overnight] ALL DONE $(date)"
column -t -s$'\t' "$RESULTS"
