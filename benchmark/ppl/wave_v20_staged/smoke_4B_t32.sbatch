#!/bin/bash
# V20 smoke: one fresh-data global structural search from the clean V18 winner.
# The old A/B/C blocks are excluded so this is a genuine re-test, not a
# replay of the data that selected the parent.  The k=2.0 escape gate runs
# only after the fixed-budget search winner is frozen and full-test evaluated.
#SBATCH --job-name=v20_4B_t32
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=32:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_v20_staged/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

export FINAL_PARENT_WRAPPER=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_final_search_final_20260719_2130/4B/target32p000/s01_precision_048_pc_len_84_wrapper.json
export FINAL_EXCLUDE_PARTITION_CHECKPOINT=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_final_search_final_20260719_2130/4B/target32p000/checkpoint.json
export FINAL_TARGETS=32
export FINAL_MAX_SWEEPS=3
export FINAL_PATIENCE=2
export FINAL_CONFIRM_TOP_K=3
export FINAL_ADVANCE=12
export FINAL_LADDER_GROUPS=global
export FINAL_MIN_LEVEL=16
export FINAL_SURROGATE_PROPOSALS=0
export FINAL_WALLTIME_HOURS=28.0
export FINAL_GUARD_SLACK=1.10
export FINAL_GATE_K=2.0
export FINAL_SKIP_GATE=0
export FINAL_PC_LENGTHS=112:104:96:95:88:84:80:76:72:68:64
export FINAL_STRUCTURED_MASS_STEPS=0.10:0.15:0.30

bash benchmark/ppl/final_mp_lane.sh 4B mp_v20_structured_smoke_20260721
