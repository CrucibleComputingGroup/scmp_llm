#!/bin/bash
#SBATCH --job-name=op_14B_t32
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_oplayer_20260720/%x_%j.out
# Per-(operator-group x layer-band) ladders: 9 ladders = {score, attnlin, mlp}
# x {early, middle, late}, all SEEDED IDENTICAL to the flat parent so the
# start state is today's config.  Each move retunes one rung of one group and
# restores the budget through the other groups, so candidates are iso-cost.
# Rationale: measured occupancy shows MLP never touches the top 2 rungs while
# qk never touches the bottom 3 -- a shared ladder gives each ~half its
# resolution -- and early vs late MLP layers want different floors.
set +u
source ~/.bashrc
conda activate annstention
set -u
export FINAL_TARGETS=32
export FINAL_MAX_SWEEPS=4
export FINAL_WALLTIME_HOURS=21.0
export FINAL_GATE_K=2.0
export FINAL_LADDER_GROUPS=op:layer
export FINAL_SEED_PSL=80
bash /home/allenjin/Projects/scmp_llm/benchmark/ppl/final_mp_lane.sh 14B oplayer_20260720
