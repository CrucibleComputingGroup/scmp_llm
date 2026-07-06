#!/bin/bash
#SBATCH --job-name=mp_sq
#SBATCH --account=nbleier0
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/mp_sq_%j.out
# Corrected-pipeline MP sweep worker (SQ-calibrated, iso-FLOP budget).
# Parametrize per submission:
#   sbatch --export=ALL,MODELS=4B [--account=... --gres=gpu:N --time=...] this.sbatch
# Multiple workers share sq_results.tsv safely (cross-node cell locks).
# Gated on the smoke marker so a queued worker can't run an unvalidated pipeline.
set -o pipefail
echo "[sbatch] node=$(hostname) models=${MODELS:-ALL} start=$(date)"
for _ in $(seq 1 24); do   # wait up to 2h for the smoke gate, then give up
  [[ -f "$HOME/sq_smoke_passed" ]] && break
  sleep 300
done
[[ -f "$HOME/sq_smoke_passed" ]] || { echo "[sbatch] smoke gate absent — refusing to run"; exit 1; }
set +u                       # /etc/bashrc references unbound vars
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm
MODELS="${MODELS:-4B,llama8B,14B,30B}" \
CALIB_SQ=1 \
BUDGET_WEIGHT=macs \
  TABLES=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_sq \
  RESULTS=/home/allenjin/Projects/hpca_results/llm/mp/sq_results.tsv \
  MANIFEST=/home/allenjin/Projects/hpca_results/llm/mp/sq_manifest.tsv \
  bash benchmark/ppl/overnight_mp.sh
echo "[sbatch] end=$(date)"
