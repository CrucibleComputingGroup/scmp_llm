#!/bin/bash
# One V20 fixed-target global structural search.  The parent is supplied by
# --export so each target starts from its own V18-clean incumbent and excludes
# the old A/B/C blocks that selected that incumbent.
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

: "${V20_MODEL:?V20_MODEL is required}"
: "${V20_PARENT:?V20_PARENT is required}"
: "${V20_EXCLUDE:?V20_EXCLUDE is required}"
: "${V20_TARGET:?V20_TARGET is required}"

export FINAL_PARENT_WRAPPER="$V20_PARENT"
export FINAL_EXCLUDE_PARTITION_CHECKPOINT="$V20_EXCLUDE"
export FINAL_TARGETS="$V20_TARGET"
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

# 30B's 128-row MoE expert projections need independent-window batching to
# make measured search feasible. The lane refuses this optimization unless the
# matching real-model B=1/B=8 torch.equal manifest exists. Cap Phase 2 at 18 h
# so the 32 h allocation retains time for both unbatched full-test evaluations.
if [[ "$V20_MODEL" == "30B" ]]; then
  export FINAL_WINDOW_BATCH_SIZE=8
  export FINAL_WALLTIME_HOURS=18.0
  if [[ "$V20_TARGET" == "32" || "$V20_TARGET" == "32.0" ]]; then
    _batch_bits=6
  else
    _batch_bits=7
  fi
  export FINAL_WINDOW_BATCH_IDENTITY_JSON="/nfs/turbo/coe-nbleier/allenjin/hpca/validation/v20_30B_window_batch_identity_b8_int${_batch_bits}.json"
fi

TAG="mp_v20_structured_${V20_MODEL}_t${V20_TARGET}_20260721"
bash benchmark/ppl/final_mp_lane.sh "$V20_MODEL" "$TAG"
