#!/bin/bash
# Rotation-gate single-GPU job. RGTAG is a placeholder — sed-replace at launch.
#
# Launch (login node, NOT from this agent):
#   TAG=rot_gate_$(date +%Y%m%d_%H%M%S)
#   mkdir -p /scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_mp_rot_gate_llama8B_${TAG}
#   sed "s/RGTAG/${TAG}/g" /home/allenjin/Projects/scmp_llm/benchmark/ppl/rot_gate/rot_gate.sbatch \
#     > /tmp/rot_gate_${TAG}.sbatch
#   sbatch /tmp/rot_gate_${TAG}.sbatch
#
#SBATCH --job-name=mp_rot_gate_llama8B
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_mp_rot_gate_llama8B_RGTAG/%x_%j.out

bash /home/allenjin/Projects/scmp_llm/benchmark/ppl/rot_gate/payload.sh RGTAG
