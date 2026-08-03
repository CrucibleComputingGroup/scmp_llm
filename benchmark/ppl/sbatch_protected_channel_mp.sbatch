#!/bin/bash
#SBATCH --job-name=mp_pc4b
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/mp_pc4b_%j.out

set -o pipefail
echo "[sbatch-pc] node=$(hostname) start=$(date)"
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm
GPU=0 bash benchmark/ppl/protected_channel_mp.sh
echo "[sbatch-pc] end=$(date)"
