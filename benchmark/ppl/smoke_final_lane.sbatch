#!/bin/bash
#SBATCH --job-name=smoke_4B_t32
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_smoke_20260720/%x_%j.out
# End-to-end smoke for the post-fix engine, BEFORE any 190 GPU-h wave.
# Exercises, on a real model, every path changed today:
#   * the counts fix   -- 4B t32 accepted a topology insert at sweep 2 in the
#                         last wave, and the sweep AFTER that is exactly where
#                         KeyError: 'counts' killed the job
#   * the walltime guard with observed-sweep-window tracking
#   * the V18 confirm veto (accept -> pooled A+B+confirm re-test -> revert)
#   * PHASE 2b, the hoisted full-protocol test on the winner
#   * the PHASE 3 glob fix (must gate ONLY target 32, not every target dir)
# 3 sweeps is the minimum that can reach the post-accept generate() call.
set +u
source ~/.bashrc
conda activate annstention
set -u
export FINAL_TARGETS=32
export FINAL_MAX_SWEEPS=3
export FINAL_WALLTIME_HOURS=6.0
export FINAL_GATE_K=2.0
bash /home/allenjin/Projects/scmp_llm/benchmark/ppl/final_mp_lane.sh 4B smoke_20260720
