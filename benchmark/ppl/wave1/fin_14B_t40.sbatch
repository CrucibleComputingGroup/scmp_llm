#!/bin/bash
#SBATCH --job-name=fin_14B_t40
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_final_20260720/%x_%j.out
# WAVE 1 (post-fix rerun): Phase1 sigma parent -> Phase2 measured search
# (4 sweeps, uniform across models) -> Phase2b full-protocol test on the
# winner -> Phase3 escape gate k=2.0.  Global ladder; 20% INT dose.
# Engine fixes in play: counts/pass1_counts, observed-sweep walltime guard,
# V18 confirm veto (revert on pooled A+B+confirm regression), hoisted
# wrap-up, per-target Phase-3 gate.
set +u
source ~/.bashrc
conda activate annstention
set -u
export FINAL_TARGETS=40
export FINAL_MAX_SWEEPS=4
export FINAL_WALLTIME_HOURS=21.0
export FINAL_GATE_K=2.0
bash /home/allenjin/Projects/scmp_llm/benchmark/ppl/final_mp_lane.sh 14B final_20260720
