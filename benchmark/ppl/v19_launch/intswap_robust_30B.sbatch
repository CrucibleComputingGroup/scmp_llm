#!/bin/bash
#SBATCH --job-name=iswap2_30B
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_hpca_mp_v19_intswap_robust_30B_20260722/%x_%j.out

# Robust INT7 allocation sweep for Qwen3-30B-A3B.  The statistical protocol
# matches the 4B/llama8B robust rankings: 24 deterministic corpus-stratified
# windows, paired mean-1SE ranking, and four-fold stability.  Eight independent
# windows share each forward only after the real-GPU exactness gate succeeds.
# Every operator-layer probe is atomically checkpointed.  Ten minutes before a
# walltime limit the batch job requeues itself and resumes that checkpoint.

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
export PPL_WINDOW_BATCH_SIZE=8
export PPL_WINDOW_BATCH_IDENTITY_JSON="$TURBO/validation/v19_30B_intswap_window_batch_identity_b8_int7.json"

requeue_on_timeout() {
  echo "[int_swap] USR1 before walltime; checkpoint is current, requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_on_timeout USR1

bash /home/allenjin/Projects/scmp_llm/hpca \
  --sensitivity \
  --models 30B \
  --configs mp_avg32_v9 \
  --sens-method int_swap \
  --sens-windows 24 \
  --sens-ctx 512 \
  --hybrid-int-bits 7 \
  --int-swap-ref mp_config \
  --int-swap-mp-config "$TURBO/mp_calib_mp_v9_hybdose_all_20260718_183035/Qwen_Qwen3-30B-A3B-Instruct-2507__mp_avg32_v9__act_global_v9_pc0.01x112_act_collapse_comp_pcovdown_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.20_wrapper.json" \
  --int-swap-window-sampling stratified \
  --int-swap-score lcb \
  --int-swap-lcb-z 1.0 \
  --int-swap-stability-folds 4 \
  --int-swap-checkpoint auto \
  --mp-seed 20260721 \
  --tag mp_v19_intswap_robust_30B_20260722 &

child_pid=$!
wait "$child_pid"
