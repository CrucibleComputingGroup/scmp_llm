#!/bin/bash
# tests/run_mp_sweep.sh — reproduce the overnight per-row MP PPL sweep.
#
# DEFAULT BEHAVIOR (no flags) reproduces the 2026-05-26 overnight result:
#   5 models × 3 MP configs, halved (sl_max=128) + SQ@0.5 + per_row attention,
#   sc_prec=8, SC_SCRAMBLE_RESCALE=1, Owen-in-rescale scramble.
#
# All flags below have defaults — override any subset to narrow the run.
#
# -----------------------------------------------------------------------------
# How to run on Great Lakes (U-Mich ARC)
# -----------------------------------------------------------------------------
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
#    For multi-node parallelism (5-way fan-out, one model per node), repeat
#    the salloc 4 more times. salloc only allows ~5 concurrent allocations
#    on this reservation due to AssocGrpMemLimit — drop --mem to 128G for
#    the additional ones (LLM PPL eval doesn't need 512G of CPU RAM).
#
# 3. Inside one of those allocations, run:
#      ssh <NODE>       # e.g. ssh gl1806
#      cd /home/allenjin/Projects/scmp_llm
#      tmux new -s mp_sweep \
#        'bash tests/run_mp_sweep.sh; exec bash'
#      # detach with Ctrl-b d ; reattach with `tmux attach -t mp_sweep`
#
#    If the node hosts multiple of your allocations, prefix python with
#    `srun --jobid=<jobid> --overlap` so it lands in the correct cgroup
#    (otherwise every job grabs whichever GPU ssh handed it).
#
# 4. The sweep writes to:
#      benchmark/ppl/_mp_overnight_<ts>/
#        ├── <model>_<mp>.log         per-config raw log
#        ├── <model>.done             marker
#        └── SUMMARY.txt              aggregated table
#
# Reference timings (RTX PRO 6000 Blackwell, 65k tokens, MP attention loop):
#   4B  ~30  min/config    8B  ~32 min/config    14B ~40 min/config
#   30B-A3B ~75 min/config   32B ~95 min/config
#
# -----------------------------------------------------------------------------
# Flags
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [flags]
  --models      Comma-separated subset of {4B,8B,14B,30B,32B}. Default: all.
  --mp          Comma-separated subset of {mp_a,mp_m,mp_c}.  Default: all.
                Or 'none' to skip MP and run a fixed-sl baseline.
  --max-tokens  PPL_MAX_TOKENS cap. Default: 65536. Use 4096 for smoke.
  --ctx         Window length. Default: 1024.
  --prec        SC precision. Default: 8.
  --no-halve    Disable cycle-halving (default: halved on).
  --no-sq       Disable SmoothQuant (default: SQ@0.5 on).
  --sq-alpha    SmoothQuant alpha. Default: 0.5.
  --attn-gran   Attention granularity. Default: per_row.
  --tag         Subdir suffix override. Default: timestamp.
  --outdir      Explicit output dir. Default: benchmark/ppl/_mp_overnight_<tag>.
  --env         Conda env name. Default: annstention.
  --hf-cache    HF_HOME. Default: /nfs/turbo/coe-nbleier/allenjin/hf_cache.
  --dry-run     Print the commands without launching.
  -h | --help   Show this help.
EOF
}

# Defaults — reproduce overnight 2026-05-26.
MODELS_CSV="4B,8B,14B,30B,32B"
MP_CSV="mp_a,mp_m,mp_c"
MAX_TOKENS=65536
CTX=1024
PREC=8
HALVE=1
USE_SQ=1
SQ_ALPHA=0.5
ATTN_GRAN=per_row
TAG="$(date +%Y%m%d_%H%M%S)"
OUTDIR=""
CONDA_ENV="annstention"
HF_HOME_OVERRIDE="/nfs/turbo/coe-nbleier/allenjin/hf_cache"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)     MODELS_CSV="$2"; shift 2 ;;
    --mp)         MP_CSV="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --ctx)        CTX="$2"; shift 2 ;;
    --prec)       PREC="$2"; shift 2 ;;
    --no-halve)   HALVE=0; shift ;;
    --no-sq)      USE_SQ=0; shift ;;
    --sq-alpha)   SQ_ALPHA="$2"; shift 2 ;;
    --attn-gran)  ATTN_GRAN="$2"; shift 2 ;;
    --tag)        TAG="$2"; shift 2 ;;
    --outdir)     OUTDIR="$2"; shift 2 ;;
    --env)        CONDA_ENV="$2"; shift 2 ;;
    --hf-cache)   HF_HOME_OVERRIDE="$2"; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown flag: $1"; usage; exit 1 ;;
  esac
done

# Resolve repo layout from this script's location.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PPL_DIR="$REPO/benchmark/ppl"
[[ -z "$OUTDIR" ]] && OUTDIR="$PPL_DIR/_mp_overnight_${TAG}"

# Map model shortname -> HF id.
declare -A MODEL_ID=(
  [4B]="Qwen/Qwen3-4B-Instruct-2507"
  [8B]="Qwen/Qwen3-8B"
  [14B]="Qwen/Qwen3-14B"
  [30B]="Qwen/Qwen3-30B-A3B-Instruct-2507"
  [32B]="Qwen/Qwen3-32B"
)

# Activate env + storage knobs (turbo for read-heavy cache, scratch for tmp).
if [[ $DRY -eq 0 ]]; then
  source ~/.bashrc
  conda activate "$CONDA_ENV"
fi

export HF_HOME="$HF_HOME_OVERRIDE"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p /scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/tmp \
  || true
export TMPDIR=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/tmp

# SC kernel knobs locked in for the MP sweep semantics.
unset SC_DISABLE_OWEN
export SC_SCRAMBLE_RESCALE=1

if [[ $DRY -eq 0 ]]; then
  mkdir -p "$OUTDIR"
  echo "out=$OUTDIR" > "$PPL_DIR/_mp_overnight_latest.path"
fi
echo "[$(date)] outdir=$OUTDIR  tag=$TAG  models=$MODELS_CSV  mp=$MP_CSV"

IFS=, read -ra MODELS <<<"$MODELS_CSV"
IFS=, read -ra MPS    <<<"$MP_CSV"

for SHORT in "${MODELS[@]}"; do
  HF="${MODEL_ID[$SHORT]:-}"
  [[ -z "$HF" ]] && { echo "[skip] unknown model '$SHORT'"; continue; }
  SAFE="$(echo "$HF" | tr '/' '_')"

  # SmoothQuant scales path — produced by calibrate_smoothquant.py once per
  # model. The sweep is gated on its presence (calibrate first if missing).
  SQ_SCALES="$PPL_DIR/act_scales_${SAFE}.pt"
  if [[ "$USE_SQ" == "1" && ! -f "$SQ_SCALES" ]]; then
    echo "[$(date)] [missing-sq] $SHORT — running calibrate_smoothquant.py first"
    if [[ $DRY -eq 0 ]]; then
      MODEL_PATH="$HF" python -u "$PPL_DIR/calibrate_smoothquant.py" \
        2>&1 | tee "$OUTDIR/_calib_${SAFE}.log"
    fi
  fi

  for MP in "${MPS[@]}"; do
    LOG="$OUTDIR/${SAFE}_${MP}.log"
    echo "[$(date)] >>> $SHORT / $MP -> $LOG"

    if [[ "$MP" == "none" ]]; then
      MP_JSON_ARG=""
    else
      MP_JSON="$PPL_DIR/${MP}.json"
      [[ ! -f "$MP_JSON" ]] && { echo "[skip] missing MP json $MP_JSON"; continue; }
      MP_JSON_ARG="MP_CONFIG_JSON=$MP_JSON"
    fi

    SQ_ARGS=""
    if [[ "$USE_SQ" == "1" ]]; then
      SQ_ARGS="USE_SMOOTHQUANT=1 SMOOTHQUANT_ALPHA=$SQ_ALPHA SMOOTHQUANT_SCALES=$SQ_SCALES"
    fi

    CMD="env MODEL_PATH=$HF \
      PPL_MAX_TOKENS=$MAX_TOKENS CTX=$CTX \
      SC_PREC=$PREC STOC_LENS=$((2 ** (PREC - 1))) \
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
      echo "[$(date)] [FAIL] $SHORT / $MP (exit=$status) — continuing"  | tee -a "$LOG"
    else
      echo "[$(date)] [OK]   $SHORT / $MP"  | tee -a "$LOG"
    fi
  done

  touch "$OUTDIR/${SAFE}.done"
done

if [[ $DRY -eq 0 ]]; then
  echo "[$(date)] sweep complete — building SUMMARY.txt"
  bash "$PPL_DIR/_mp_overnight_summary.sh" "$OUTDIR" || true
fi
echo "[$(date)] tests/run_mp_sweep.sh done. outdir=$OUTDIR"
