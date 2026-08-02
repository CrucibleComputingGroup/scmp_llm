#!/bin/bash
# Submit the 60-cell INT/fp16 context sweep under the AWQ front-end:
# 4 models x {fp16, W8A8_symm, W8A8_asymm, W6A6_symm, W6A6_asymm}
# x ctx {2048,4096,8192}.  One job per (model, ctx) = 12 jobs, 5 configs each.
#
# Walltimes are sized off the cancelled SmoothQuant wave, where 4B took ~23 min
# for 2 configs at ctx 2048 -- i.e. ~10 min/cell, not the 1-5 min first
# estimated. Five configs and the 8192 attention cost push the big models well
# past the original 4h, so this over-provisions rather than lose a wave to a
# walltime kill.
#
# 30B @ 8192 gets 2 GPUs: fp16 weights (~61G) + eager [1,32,8192,8192] scores +
# the 8192x151936 logits upcast peaks near ~85G, too thin on a 96G card.
# device_map="auto" shards it.
#
# Usage:  bash launch.sh [--dry]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=20260730
DRY=0
[[ "${1:-}" == "--dry" ]] && DRY=1

mkdir -p /scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_ctx_sweep

for model in 4B llama8B 14B 30B; do
  for ctx in 2048 4096 8192; do
    gpus=1; time="12:00:00"
    if [[ "$model" == "30B" ]]; then time="24:00:00"; fi
    if [[ "$model" == "30B" && "$ctx" == "8192" ]]; then gpus=2; fi
    name="ctxswawq_${model}_c${ctx}"
    cmd=(sbatch --job-name="$name" --gres=gpu:$gpus --time="$time"
         --export=ALL,CTXSWEEP_MODEL="$model",CTXSWEEP_CTX="$ctx",CTXSWEEP_DATE="$DATE"
         "$HERE/run_cell.sbatch")
    if [[ $DRY -eq 1 ]]; then
      echo "DRY: ${cmd[*]}"
    else
      "${cmd[@]}"
    fi
  done
done
