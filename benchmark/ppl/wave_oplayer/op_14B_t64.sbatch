#!/bin/bash
#SBATCH --job-name=op_14B_t64
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --exclude=gl1810
# gl1810 has failed this cell 5x with "CUDA-capable device(s) is/are
# busy or unavailable" during checkpoint load, always as the 4th
# concurrent job on that node, while its three siblings run fine there.
# Startup jitter did NOT help, so this is a bad GPU rather than a race.
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
# Stagger CUDA init: four jobs starting on the same node in the same
# second reproducibly lose the 4th to "CUDA-capable device(s) is/are
# busy or unavailable" during checkpoint load (seen on gl1810 in two
# consecutive waves, same cell both times).
sleep $(( (RANDOM % 90) + 30 ))
export FINAL_TARGETS=64
export FINAL_MAX_SWEEPS=4
export FINAL_WALLTIME_HOURS=21.0
export FINAL_GATE_K=2.0
export FINAL_LADDER_GROUPS=op:layer
export FINAL_SEED_PSL=80
bash /home/allenjin/Projects/scmp_llm/benchmark/ppl/final_mp_lane.sh 14B oplayer_20260720
