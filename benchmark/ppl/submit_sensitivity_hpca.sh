#!/bin/bash
# Submit one Slurm job per model for HPCA sensitivity analysis.
#
# Usage:
#   bash benchmark/ppl/submit_sensitivity_hpca.sh
#   bash benchmark/ppl/submit_sensitivity_hpca.sh --configs sc_avg192 --models 4B,llama8B
#   bash benchmark/ppl/submit_sensitivity_hpca.sh --tag sens_test --dry-run
set -uo pipefail

MODELS_CSV="4B,llama8B,14B,30B"
CONFIGS_CSV="sc_avg192"
TAG="sens_$(date +%Y%m%d_%H%M%S)"
ACCOUNT="nbleier_owned1"
RESERVATION="rtx6000_arph_nodes"
PARTITION="gpu-rtx6000"
CONDA_ENV="annstention"
MEM="180G"
CPUS="12"
TIME="24:00:00"
DRY=0

SCRATCH_BASE="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS_CSV="$2"; shift 2 ;;
    --configs) CONFIGS_CSV="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --reservation) RESERVATION="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help)
      sed -n '1,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

IFS=, read -ra MODELS <<<"$MODELS_CSV"
IFS=, read -ra CONFIGS <<<"$CONFIGS_CSV"

mkdir -p "$SCRATCH_BASE/logs"
echo "[submit-sens] tag=$TAG models=$MODELS_CSV configs=$CONFIGS_CSV"

for MODEL in "${MODELS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    JOB="sens_${MODEL}_${CFG}"
    LOG="$SCRATCH_BASE/logs/${JOB}_%j.out"
    PAYLOAD="source ~/.bashrc; conda activate $CONDA_ENV; cd $HERE; bash hpca --sensitivity --models $MODEL --configs $CFG --tag $TAG"
    if [[ $DRY -eq 1 ]]; then
      echo "DRY: sbatch --job-name=$JOB --account=$ACCOUNT --reservation=$RESERVATION --partition=$PARTITION --gres=gpu:1 --cpus-per-task=$CPUS --mem=$MEM --time=$TIME --output=$LOG --wrap=$PAYLOAD"
    else
      sbatch --job-name="$JOB" \
        --account="$ACCOUNT" \
        --reservation="$RESERVATION" \
        --partition="$PARTITION" \
        --gres=gpu:1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --output="$LOG" \
        --wrap="$PAYLOAD"
    fi
  done
done

echo "TAG=$TAG"
