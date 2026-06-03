#!/bin/bash
# tests/run_mp_sweep.sh — calibrated AdaptiveMP PPL sweep.
#
# 2026-05-27: replaces the fixed-fraction mp_a/mp_m/mp_c sweep entirely with
# calibrated thresholds produced by benchmark/ppl/calibrate_mp_thresholds.py.
# Per-row threshold tables come from per-operator SC reconstruction error
# against an FP teacher (--method act). The --method grad variant runs the
# SAME calibrator with --loss-weight-by-grad, which weights each row's SC
# error by its loss sensitivity g_row²=‖∂L/∂y_row‖₂² (one extra backward pass
# per window). No separate grad-score precompute step is used.
#
# Default sweep configs (all run the kernel with SC_PREC=8 — same 256-level
# Sobol generator. Lower-precision configs early-terminate the stream and/or
# mix per-row stoc_lens via MP):
#   int8   — uniform, no MP. stoc_len = 128 halved (= 256 raw). Baseline.
#   len192 — MP, target avg ≈ 91 halved (≈ 192 raw on average, hence the
#            vit_sc name 'len192' / 'log192'). Levels [128,96,64].
#   int7   — MP, target avg = 64 halved (≈ 128 raw, == uniform int7).
#            Levels [128,64,32].
#   len96  — MP, target avg ≈ 48 halved (≈ 96 raw on average, hence 'len96').
#            Levels [64,48,32].
#
# Each (model, prec_tag) pair gets:
#   benchmark/ppl/mp_calib/<safe>__<prec_tag>.json        ← calibration table
#   benchmark/ppl/mp_calib/<safe>__<prec_tag>_wrapper.json ← MP_CONFIG_JSON
#
# Calibration is lazy: if the table already exists, it is reused.
# -----------------------------------------------------------------------------
#
# How to run on Great Lakes (U-Mich ARC) — unchanged from the previous version:
#
# 1. ssh to a login node:  ssh gl-login.arc-ts.umich.edu
#
# 2. Allocate a GPU node (mirrors pro6000.sh; ~10-day wall):
#      salloc --no-shell \
#        --account=nbleier_owned1 --partition=gpu-rtx6000 \
#        --gres=gpu:1 --cpus-per-gpu=32 --mem=512G \
#        --time=9-21:00:00
#    Note the JOBID + the NODELIST (e.g. gl1806).
#
# 3. Inside one of those allocations, run:
#      ssh <NODE>       # e.g. ssh gl1806
#      cd /home/allenjin/Projects/scmp_llm
#      tmux new -s mp_sweep \
#        'bash tests/run_mp_sweep.sh; exec bash'
#      # detach with Ctrl-b d ; reattach with `tmux attach -t mp_sweep`
#
# 4. The sweep writes to:
#      benchmark/ppl/_mp_overnight_<ts>/
#        ├── <model>_<prec_tag>.log         per-config raw log
#        ├── <model>.done                   marker
#        └── SUMMARY.txt                    aggregated table
# -----------------------------------------------------------------------------
# Flags
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [flags]
  --models      Comma-separated subset of {4B,8B,14B,30B,32B}. Default: all.
  --prec        Comma-separated subset of {int8,len192,int7,len96}.
                  int8  — uniform stoc_len=128 (halved), no MP. Baseline.
                  len192 — MP avg ≈ 91 halved. Levels [128,96,64].
                  int7  — MP avg = 64 halved (== uniform int7). Levels [128,64,32].
                  len96 — MP avg ≈ 48 halved. Levels [64,48,32].
                Or 'none' as an alias for the int8 uniform baseline.
                Default: int8,len192,int7,len96.
  --method      Comma-separated subset of
                {act,grad,grad_sc,act_global,grad_global,measured}.
                Default: act,grad.
                  act_global  — cross-layer budget: ONE global shared-λ solve
                         (calibrate_mp_thresholds.py --budget-scope global), act
                         objective. Lets budget flow across layers/operators
                         instead of pinning every layer to the same avg.
                  grad_global — same global solve with FP-grad weighting
                         (--loss-weight-by-grad --budget-scope global).
                  measured    — measured-ΔLoss cross-layer + act within:
                         knock-down probe weights each (op,layer) group by its
                         measured loss sensitivity, global solve, act quantile
                         within (--cross-layer-weight measured). Env knobs:
                         MEASURE_PROBE_LEVEL, MEASURE_BASELINE_STOCLEN,
                         MEASURE_WINDOWS (def 2), MEASURE_CTX (def 512).
                  act  — per-row reconstruction-error calibration
                         (calibrate_mp_thresholds.py).
                  grad — loss-gradient-weighted per-row calibration
                         (calibrate_mp_thresholds.py --loss-weight-by-grad):
                         same calibrator, but each row's SC error is weighted
                         by g_row²=‖∂L/∂y_row‖₂² via one extra backward pass.
                  grad_sc — STE noisy-trajectory gradient (g_SC):
                         calibrate_mp_thresholds.py --grad-on-sc. The main
                         calibration forward runs with SC ENABLED through a
                         straight-through estimator at a uniform stoc_len (the
                         budget target average), so g_row is captured along the
                         SC-noisy trajectory. Defaults to --grad-g-pow 1
                         --grad-s-pow 1 (g² is known to be harmful). Tunables:
                         GRAD_SC_STOCLEN (default = budget avg), GRAD_SC_DRAWS
                         (default 2).
  --max-tokens  PPL_MAX_TOKENS cap. Default: 65536. Use 4096 for smoke.
  --ctx         Window length. Default: 1024.
  --no-halve    Disable cycle-halving (default: halved on).
  --no-sq       Disable SmoothQuant (default: SQ@0.5 on).
  --sq-alpha    SmoothQuant alpha. Default: 0.5.
  --attn-gran   Attention granularity. Default: per_row.
  --tag         Subdir suffix override. Default: timestamp.
  --outdir      Explicit output dir. Default: benchmark/ppl/_mp_overnight_<tag>.
  --env         Conda env name. Default: annstention.
  --hf-cache    HF_HOME. Default: /nfs/turbo/coe-nbleier/allenjin/hf_cache.
  --recalibrate Force re-running calibrator + grad-scores even when cached.
  --calib-only  Generate calibration tables, skip the PPL sweep.
  --calib-seqs  act-calib: number of ctx-length wikitext2 windows. Default: 4.
  --calib-buckets
                Calibration: number of layer buckets. Default: 4.
  --grad-windows
                grad-method: number of forward+backward calibration windows
                (maps to --num_calib_sequences). Default: same as --calib-seqs.
  --grad-ctx    grad-method: ctx_len for the forward+backward pass. Default:
                same as --ctx. Lower this (e.g. 512) for 30B-MoE to keep the
                backward activation memory under the GPU cap.
  --grad-alpha  Deprecated no-op (the budget solver now consumes g_row²
                directly; there is no separate grad-score exponent). Kept for
                backward-compatible CLI parsing. Default: 1.0.
  --gpu         CUDA_VISIBLE_DEVICES export for this invocation. Default: leave
                inherited (lets you fan one sweep out across two GPUs by
                launching twice with --gpu 0 / --gpu 1 + different --models).
  --dry-run     Print the commands without launching.
  -h | --help   Show this help.
EOF
}

# Defaults.
MODELS_CSV="4B,8B,14B,30B,32B"
PREC_CSV="int8,len192,int7,len96"
METHOD_CSV="act,grad"
MAX_TOKENS=65536
CTX=1024
HALVE=1
USE_SQ=1
SQ_ALPHA=0.5
ATTN_GRAN=per_row
TAG="$(date +%Y%m%d_%H%M%S)"
OUTDIR=""
CONDA_ENV="annstention"
HF_HOME_OVERRIDE="/nfs/turbo/coe-nbleier/allenjin/hf_cache"
DRY=0
RECALIBRATE=0
CALIB_ONLY=0
CALIB_SEQS=4
CALIB_BUCKETS=4
GRAD_WINDOWS=8
GRAD_CTX=""
GRAD_ALPHA=1.0
GPU_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)        MODELS_CSV="$2"; shift 2 ;;
    --prec)          PREC_CSV="$2"; shift 2 ;;
    --method)        METHOD_CSV="$2"; shift 2 ;;
    --max-tokens)    MAX_TOKENS="$2"; shift 2 ;;
    --ctx)           CTX="$2"; shift 2 ;;
    --no-halve)      HALVE=0; shift ;;
    --no-sq)         USE_SQ=0; shift ;;
    --sq-alpha)      SQ_ALPHA="$2"; shift 2 ;;
    --attn-gran)     ATTN_GRAN="$2"; shift 2 ;;
    --tag)           TAG="$2"; shift 2 ;;
    --outdir)        OUTDIR="$2"; shift 2 ;;
    --env)           CONDA_ENV="$2"; shift 2 ;;
    --hf-cache)      HF_HOME_OVERRIDE="$2"; shift 2 ;;
    --recalibrate)   RECALIBRATE=1; shift ;;
    --calib-only)    CALIB_ONLY=1; shift ;;
    --calib-seqs)    CALIB_SEQS="$2"; shift 2 ;;
    --calib-buckets) CALIB_BUCKETS="$2"; shift 2 ;;
    --grad-windows)  GRAD_WINDOWS="$2"; shift 2 ;;
    --grad-ctx)      GRAD_CTX="$2"; shift 2 ;;
    --grad-alpha)    GRAD_ALPHA="$2"; shift 2 ;;
    --gpu)           GPU_OVERRIDE="$2"; shift 2 ;;
    --dry-run)       DRY=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "unknown flag: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$GPU_OVERRIDE" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_OVERRIDE"
fi

# Per-precision recipe.
#   prec_tag : sc_prec  levels(halved CSV)  budget_ratio  budget_ref  stoc_lens
declare -A PREC_SC
declare -A PREC_LEVELS
declare -A PREC_RATIO
declare -A PREC_REF
declare -A PREC_STOC
# len192: MP, target avg ≈ 91 halved (≈ 192 raw). Bracketed by 128 (=int8)
# above and 64 below. Vit_sc convention names this 'len192' or 'log192'.
PREC_SC[len192]=8
PREC_LEVELS[len192]="128,96,64"
PREC_RATIO[len192]=0.7109          # 91 / 128
PREC_REF[len192]=128
PREC_STOC[len192]=128              # max halved level → SC envelope

# int7: MP, target avg = 64 halved (= uniform int7). Levels span 128..32 so
# the per-row dispatch actually moves precision around the int7 line rather
# than degenerating to a uniform single-level config.
PREC_SC[int7]=8
PREC_LEVELS[int7]="128,64,32"
PREC_RATIO[int7]=0.5               # 64 / 128
PREC_REF[int7]=128
PREC_STOC[int7]=128

# len96: MP, target avg ≈ 48 halved (≈ 96 raw). Bracketed by 64 (=int7)
# above and 32 below. Vit_sc convention 'len96' / 'log96'.
PREC_SC[len96]=8
PREC_LEVELS[len96]="64,48,32"
PREC_RATIO[len96]=0.75             # 48 / 64
PREC_REF[len96]=64
PREC_STOC[len96]=64

# int8 (uniform baseline) is handled inline in the dispatch loop — any
# precision tag NOT in PREC_LEVELS above is treated as "uniform sc_prec=8,
# stoc_len=128, no MP". 'int8' and the legacy 'none' both route there.

# Resolve repo layout.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PPL_DIR="$REPO/benchmark/ppl"
CALIB_DIR="$PPL_DIR/mp_calib"
[[ -z "$OUTDIR" ]] && OUTDIR="$PPL_DIR/_mp_overnight_${TAG}"

# Map model shortname -> HF id.
declare -A MODEL_ID=(
  [4B]="Qwen/Qwen3-4B-Instruct-2507"
  [8B]="Qwen/Qwen3-8B"
  [14B]="Qwen/Qwen3-14B"
  [30B]="Qwen/Qwen3-30B-A3B-Instruct-2507"
  [32B]="Qwen/Qwen3-32B"
)

if [[ $DRY -eq 0 ]]; then
  # /etc/bashrc references an unbound variable in some Slurm node images,
  # which trips `set -u`. Disable around the source, restore after.
  set +u
  source ~/.bashrc
  conda activate "$CONDA_ENV"
  set -u
fi

export HF_HOME="$HF_HOME_OVERRIDE"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p /scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/tmp || true
export TMPDIR=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/tmp

unset SC_DISABLE_OWEN
export SC_SCRAMBLE_RESCALE=1
# Owen scramble family. 'bitrev' keeps counter's equipartition but breaks the
# "low bits run consecutively" adjacency so per-D correlations don't resonate
# with the mask period. Calibration + PPL inherit this same env, so thresholds
# stay consistent. Override by exporting SC_OWEN_MODE before invoking.
export SC_OWEN_MODE="${SC_OWEN_MODE:-bitrev}"

if [[ $DRY -eq 0 ]]; then
  mkdir -p "$OUTDIR" "$CALIB_DIR"
  echo "out=$OUTDIR" > "$PPL_DIR/_mp_overnight_latest.path"
fi
echo "[$(date)] outdir=$OUTDIR  tag=$TAG  models=$MODELS_CSV  prec=$PREC_CSV"

IFS=, read -ra MODELS  <<<"$MODELS_CSV"
IFS=, read -ra PRECS   <<<"$PREC_CSV"
IFS=, read -ra METHODS <<<"$METHOD_CSV"

# Write a tiny wrapper JSON that points at a calibration table.
# Usage: write_wrapper <wrapper_path> <levels_csv> <table_relpath>
write_wrapper() {
  local wpath="$1"
  local levels="$2"
  local table_rel="$3"
  local lvl_json
  lvl_json="$(echo "$levels" | sed 's/,/, /g')"
  cat > "$wpath" <<EOF
{
  "type": "AdaptiveMPConfig",
  "stoc_len_levels": [${lvl_json}],
  "threshold_table_path": "${table_rel}"
}
EOF
}

# Run the appropriate calibrator for (model, prec_tag, method) if needed.
# Stdout: the wrapper-JSON path to feed MP_CONFIG_JSON.
ensure_calibration() {
  local hf="$1"; local safe="$2"; local prec_tag="$3"; local method="$4"
  local sc_prec="${PREC_SC[$prec_tag]}"
  local levels="${PREC_LEVELS[$prec_tag]}"
  local ratio="${PREC_RATIO[$prec_tag]}"
  local ref="${PREC_REF[$prec_tag]}"
  local table_abs="$CALIB_DIR/${safe}__${prec_tag}_${method}.json"
  local wrapper_abs="$CALIB_DIR/${safe}__${prec_tag}_${method}_wrapper.json"
  local table_rel
  table_rel="$(basename "$table_abs")"

  # All status output to stderr — stdout is reserved for the wrapper path.
  if [[ -f "$table_abs" && $RECALIBRATE -eq 0 ]]; then
    echo "[$(date)] [calib-cache] $safe / $prec_tag / $method -> $table_abs" >&2
  else
    echo "[$(date)] [calib] $safe / $prec_tag / $method -> $table_abs" >&2
    local cmd
    if [[ "$method" == "act" ]]; then
      cmd="MODEL_PATH=$hf python -u $PPL_DIR/calibrate_mp_thresholds.py \
        --model_path $hf \
        --mp_levels $levels \
        --budget_ratio $ratio \
        --budget_ref_stoc_len $ref \
        --sc_prec $sc_prec --halve $HALVE \
        --num_calib_sequences $CALIB_SEQS \
        --ctx_len $CTX \
        --layer_buckets $CALIB_BUCKETS \
        --output_json $table_abs"
    elif [[ "$method" == "grad" ]]; then
      # Loss-gradient-weighted calibration: same calibrator as 'act' plus the
      # --loss-weight-by-grad flag, which runs one extra backward pass per
      # window and weights each row's SC error by g_row²=‖∂L/∂y_row‖₂².
      local grad_seqs="${GRAD_WINDOWS:-$CALIB_SEQS}"
      local grad_ctx="${GRAD_CTX:-$CTX}"
      # Weighting exponents: defaults (2,2) = Gauss-Newton g²σ²; set
      # GRAD_G_POW=1 GRAD_S_POW=1 for the softened g·σ objective.
      local grad_g_pow="${GRAD_G_POW:-2.0}"
      local grad_s_pow="${GRAD_S_POW:-2.0}"
      cmd="MODEL_PATH=$hf python -u $PPL_DIR/calibrate_mp_thresholds.py \
        --model_path $hf \
        --mp_levels $levels \
        --budget_ratio $ratio \
        --budget_ref_stoc_len $ref \
        --sc_prec $sc_prec --halve $HALVE \
        --num_calib_sequences $grad_seqs \
        --ctx_len $grad_ctx \
        --layer_buckets $CALIB_BUCKETS \
        --loss-weight-by-grad \
        --grad-g-pow $grad_g_pow --grad-s-pow $grad_s_pow \
        --output_json $table_abs"
    elif [[ "$method" == "grad_sc" ]]; then
      # STE noisy-trajectory gradient (g_SC): same calibrator + the
      # --grad-on-sc flag, which runs the MAIN forward with SC enabled through
      # a straight-through estimator at a uniform stoc_len, capturing g_row
      # along the SC-noisy trajectory. Natural exponents here are 1,1 (g² is
      # already known harmful); GRAD_G_POW / GRAD_S_POW override.
      local grad_seqs="${GRAD_WINDOWS:-$CALIB_SEQS}"
      local grad_ctx="${GRAD_CTX:-$CTX}"
      local grad_g_pow="${GRAD_G_POW:-1.0}"
      local grad_s_pow="${GRAD_S_POW:-1.0}"
      # Uniform noisy-forward stoc_len. Default (empty) lets the calibrator
      # pick round(budget_ratio·budget_ref) = the per-config budget average.
      local sc_stoclen_arg=""
      if [[ -n "${GRAD_SC_STOCLEN:-}" ]]; then
        sc_stoclen_arg="--grad-sc-stoclen ${GRAD_SC_STOCLEN}"
      fi
      local sc_draws="${GRAD_SC_DRAWS:-2}"
      cmd="MODEL_PATH=$hf python -u $PPL_DIR/calibrate_mp_thresholds.py \
        --model_path $hf \
        --mp_levels $levels \
        --budget_ratio $ratio \
        --budget_ref_stoc_len $ref \
        --sc_prec $sc_prec --halve $HALVE \
        --num_calib_sequences $grad_seqs \
        --ctx_len $grad_ctx \
        --layer_buckets $CALIB_BUCKETS \
        --grad-on-sc \
        --grad-g-pow $grad_g_pow --grad-s-pow $grad_s_pow \
        $sc_stoclen_arg --grad-sc-draws $sc_draws \
        --output_json $table_abs"
    elif [[ "$method" == "act_global" ]]; then
      # Cross-layer: act objective, ONE global shared-λ solve (budget flows
      # across layers/operators instead of per-layer-equal).
      cmd="MODEL_PATH=$hf python -u $PPL_DIR/calibrate_mp_thresholds.py \
        --model_path $hf \
        --mp_levels $levels \
        --budget_ratio $ratio \
        --budget_ref_stoc_len $ref \
        --sc_prec $sc_prec --halve $HALVE \
        --num_calib_sequences $CALIB_SEQS \
        --ctx_len $CTX \
        --layer_buckets $CALIB_BUCKETS \
        --budget-scope global \
        --output_json $table_abs"
    elif [[ "$method" == "grad_global" ]]; then
      # Cross-layer with FP-grad weighting: global solve + --loss-weight-by-grad.
      local grad_seqs="${GRAD_WINDOWS:-$CALIB_SEQS}"
      local grad_ctx="${GRAD_CTX:-$CTX}"
      local grad_g_pow="${GRAD_G_POW:-1.0}"
      local grad_s_pow="${GRAD_S_POW:-1.0}"
      cmd="MODEL_PATH=$hf python -u $PPL_DIR/calibrate_mp_thresholds.py \
        --model_path $hf \
        --mp_levels $levels \
        --budget_ratio $ratio \
        --budget_ref_stoc_len $ref \
        --sc_prec $sc_prec --halve $HALVE \
        --num_calib_sequences $grad_seqs \
        --ctx_len $grad_ctx \
        --layer_buckets $CALIB_BUCKETS \
        --loss-weight-by-grad \
        --grad-g-pow $grad_g_pow --grad-s-pow $grad_s_pow \
        --budget-scope global \
        --output_json $table_abs"
    elif [[ "$method" == "measured" ]]; then
      # Measured-ΔLoss cross-layer + act within: knock-down probe sets per-group
      # weights, global solve, act quantile within each group.
      local m_probe="${MEASURE_PROBE_LEVEL:-0}"
      local m_base="${MEASURE_BASELINE_STOCLEN:-0}"
      local m_win="${MEASURE_WINDOWS:-2}"
      local m_ctx="${MEASURE_CTX:-512}"
      cmd="MODEL_PATH=$hf python -u $PPL_DIR/calibrate_mp_thresholds.py \
        --model_path $hf \
        --mp_levels $levels \
        --budget_ratio $ratio \
        --budget_ref_stoc_len $ref \
        --sc_prec $sc_prec --halve $HALVE \
        --num_calib_sequences $CALIB_SEQS \
        --ctx_len $CTX \
        --layer_buckets $CALIB_BUCKETS \
        --cross-layer-weight measured \
        --measure-probe-level $m_probe \
        --measure-baseline-stoclen $m_base \
        --measure-windows $m_win --measure-ctx $m_ctx \
        --output_json $table_abs"
    else
      echo "unknown method: $method" >&2
      exit 1
    fi
    if [[ $DRY -eq 1 ]]; then
      echo "  DRY: $cmd" >&2
    else
      eval "$cmd" 2>&1 | tee "$OUTDIR/_calib_${safe}_${prec_tag}_${method}.log" >&2
    fi
  fi

  if [[ $DRY -eq 0 ]]; then
    write_wrapper "$wrapper_abs" "$levels" "$table_rel"
  fi
  echo "$wrapper_abs"
}

for SHORT in "${MODELS[@]}"; do
  HF="${MODEL_ID[$SHORT]:-}"
  [[ -z "$HF" ]] && { echo "[skip] unknown model '$SHORT'"; continue; }
  SAFE="$(echo "$HF" | tr '/' '_')"

  # SmoothQuant scales — gated on existence, same calibration script as before.
  SQ_SCALES="$PPL_DIR/act_scales_${SAFE}.pt"
  if [[ "$USE_SQ" == "1" && ! -f "$SQ_SCALES" ]]; then
    echo "[$(date)] [missing-sq] $SHORT — running calibrate_smoothquant.py first"
    if [[ $DRY -eq 0 ]]; then
      MODEL_PATH="$HF" python -u "$PPL_DIR/calibrate_smoothquant.py" \
        2>&1 | tee "$OUTDIR/_calib_smoothquant_${SAFE}.log"
    fi
  fi

  for PREC in "${PRECS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      # Uniform configs: any prec tag with no entry in PREC_LEVELS is a
      # no-MP baseline. 'int8' is the canonical name; 'none' is a legacy
      # alias. Method has no meaning here — only run once per model.
      if [[ -z "${PREC_LEVELS[$PREC]:-}" ]]; then
        if [[ "$METHOD" != "${METHODS[0]}" ]]; then continue; fi
        TAG_RUN="$PREC"
        LOG="$OUTDIR/${SAFE}_${TAG_RUN}.log"
        echo "[$(date)] >>> $SHORT / $PREC (uniform, no MP) -> $LOG"
        MP_JSON_ARG=""
        SC_PREC_USE=8
        STOC_LENS_USE=128
      else
        SC_PREC_USE="${PREC_SC[$PREC]}"
        STOC_LENS_USE="${PREC_STOC[$PREC]}"
        WRAPPER="$(ensure_calibration "$HF" "$SAFE" "$PREC" "$METHOD")"
        MP_JSON_ARG="MP_CONFIG_JSON=$WRAPPER"
        TAG_RUN="${PREC}_${METHOD}"
        LOG="$OUTDIR/${SAFE}_${TAG_RUN}.log"
        echo "[$(date)] >>> $SHORT / $PREC / $METHOD (sc_prec=$SC_PREC_USE) -> $LOG"
      fi

      if [[ "$CALIB_ONLY" == "1" ]]; then
        echo "  [calib-only] skipping PPL run."
        continue
      fi

      # Skip PPL re-run if this config's log already contains a completed
      # SC line (avoids burning 30-90 min/config when resuming after a
      # disk-full / printer-bug crash on later configs).
      if [[ -f "$LOG" ]] && grep -q "SC AdaptiveMPConfig.*×.* vs fp16" "$LOG" 2>/dev/null; then
        echo "  [skip] $LOG already has a completed SC PPL line"
        continue
      fi

      SQ_ARGS=""
      if [[ "$USE_SQ" == "1" ]]; then
        SQ_ARGS="USE_SMOOTHQUANT=1 SMOOTHQUANT_ALPHA=$SQ_ALPHA SMOOTHQUANT_SCALES=$SQ_SCALES"
      fi

      CMD="env MODEL_PATH=$HF \
        PPL_MAX_TOKENS=$MAX_TOKENS CTX=$CTX \
        SC_PREC=$SC_PREC_USE STOC_LENS=$STOC_LENS_USE \
        SC_HALVE_BIPOLAR_STOC_LEN=$HALVE \
        SC_ATTN_GRANULARITY=$ATTN_GRAN \
        $SQ_ARGS \
        $MP_JSON_ARG \
        python -u $PPL_DIR/ppl.py"

      if [[ $DRY -eq 1 ]]; then
        echo "  DRY: $CMD"
        continue
      fi
      eval "$CMD" 2>&1 | tee -a "$LOG"
      status=${PIPESTATUS[0]}
      if [[ $status -ne 0 ]]; then
        echo "[$(date)] [FAIL] $SHORT / $TAG_RUN (exit=$status) — continuing"  | tee -a "$LOG"
      else
        echo "[$(date)] [OK]   $SHORT / $TAG_RUN"  | tee -a "$LOG"
      fi
    done
  done

  if [[ $DRY -eq 0 ]]; then
    touch "$OUTDIR/${SAFE}.done"
  fi
done

if [[ $DRY -eq 0 && "$CALIB_ONLY" != "1" ]]; then
  echo "[$(date)] sweep complete — building SUMMARY.txt"
  bash "$PPL_DIR/_mp_overnight_summary.sh" "$OUTDIR" || true
fi
echo "[$(date)] tests/run_mp_sweep.sh done. outdir=$OUTDIR"
