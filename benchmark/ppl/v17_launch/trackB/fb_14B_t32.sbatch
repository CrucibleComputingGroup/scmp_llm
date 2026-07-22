#!/bin/bash
#SBATCH --job-name=fb_14B_t32
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_mp_v16_refine_mp_v17_fbsearch_20260719_033500/%x_%j.out

# V17 Track B — fallback measured paired-window search, FIXED acceptance
# protocol (accept on A->B replication; 16-window confirm STOP-ONLY) + new
# lift_compound macro family. Branch: 14B target 32. Seeded from the V9 avg32
# parent (NOT a transplant cell). --patience 4. Auto-resubmits from checkpoint
# on the internal 22.5h walltime guard (bounded to 3 continuations).

SELF="/nfs/turbo/coe-nbleier/allenjin/hpca/mp_v17_fbsearch_20260719_033500/sbatch/fb_14B_t32.sbatch"
RESUB="${FBSEARCH_RESUB:-0}"
MAXRESUB=3

export V16_TARGETS="32"
export V16_PATIENCE=4
export V16_MAX_SWEEPS=8

# Run the lane (it does its own source ~/.bashrc + conda activate + env).
bash /home/allenjin/Projects/scmp_llm/benchmark/ppl/mp_v16_lane.sh 14B mp_v17_fbsearch_20260719_033500
rc=$?

# Auto-resubmit only if the branch stopped on the walltime guard (not converged).
BRANCH=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_v16_refine_mp_v17_fbsearch_20260719_033500/14B/target32p000
SUMMARY="$BRANCH/summary.json"
if [[ $rc -eq 0 && -f "$SUMMARY" ]]; then
  STOP=$(python3 -c "import json,sys;print(json.load(open('$SUMMARY')).get('stop_reason',''))" 2>/dev/null || echo "")
  echo "[fbsearch] branch 14B t32 stop_reason=$STOP resub=$RESUB"
  if [[ "$STOP" == "walltime_guard" && $RESUB -lt $MAXRESUB ]]; then
    echo "[fbsearch] resubmitting from checkpoint ($((RESUB+1))/$MAXRESUB)"
    FBSEARCH_RESUB=$((RESUB+1)) sbatch "$SELF"
  fi
fi
