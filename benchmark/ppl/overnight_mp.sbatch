#!/bin/bash
#SBATCH --job-name=mp_overnight
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/mp_overnight_%j.out
set -o pipefail
echo "[sbatch] node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} start=$(date)"
set +u                       # /etc/bashrc references unbound vars (BASHRCSOURCED)
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm
# FLOP/energy-weighted (iso-compute) budget run — separate output files so it
# never collides with / is skipped against the row-weighted results.
BUDGET_WEIGHT=macs \
  TABLES=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_fw \
  RESULTS=/home/allenjin/Projects/hpca_results/llm/mp/fw_results.tsv \
  MANIFEST=/home/allenjin/Projects/hpca_results/llm/mp/fw_manifest.tsv \
  bash benchmark/ppl/overnight_mp.sh   # NGPU auto-detected from SLURM_GPUS_ON_NODE
echo "[sbatch] end=$(date)"
