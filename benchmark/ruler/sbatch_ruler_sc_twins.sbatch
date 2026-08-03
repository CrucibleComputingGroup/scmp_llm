#!/bin/bash
#SBATCH --job-name=ruler_sctwin
#SBATCH --account=nbleier0
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/ruler_sctwin_%j.out
# SC-uniform RULER MV-NIAH cells at the MP budget points (sc_int7 / sc_avg96) —
# the downstream comparators ("uniform twins") for the iso-FLOP MP cells.
# One model per job: RULER_MODEL=qwen3-4b or llama-3.1-8b (passed via --export).
set -o pipefail
echo "[sbatch] node=$(hostname) model=${RULER_MODEL:?set RULER_MODEL} start=$(date)"
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm/benchmark/ruler
bash run_ruler_baseline.sh \
  --models "$RULER_MODEL" \
  --configs sc_int7,sc_avg96 \
  --task niah_multivalue --ctx 4096 --samples 100 \
  --device cuda:0 --tag "sctwin_${RULER_MODEL//[.\/]/_}"
echo "[sbatch] end=$(date)"
