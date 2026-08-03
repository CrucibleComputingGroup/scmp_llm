#!/bin/bash
#SBATCH --job-name=iswap2_l8B
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=40:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_hpca_mp_v19_intswap_robust_20260721/%x_%j.out

# Robust INT7 allocation sweep for Llama-3.1-8B-Instruct.  See the 4B launcher
# for the statistical design.  The V9 MP wrapper is the all-SC reference and
# the hybrid mask is kept off while each single operator-layer swap is probed.

bash /home/allenjin/Projects/scmp_llm/hpca \
  --sensitivity \
  --models llama8B \
  --configs mp_avg32_v9 \
  --sens-method int_swap \
  --sens-windows 24 \
  --sens-ctx 512 \
  --hybrid-int-bits 7 \
  --int-swap-ref mp_config \
  --int-swap-mp-config /nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_mp_v9_hybdose_all_20260718_183035/meta-llama_Llama-3.1-8B-Instruct__mp_avg32_v9__act_global_v9_pc0.01x112_act_collapse_comp_pcovdown_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.20_wrapper.json \
  --int-swap-window-sampling stratified \
  --int-swap-score lcb \
  --int-swap-lcb-z 1.0 \
  --int-swap-stability-folds 4 \
  --int-swap-checkpoint auto \
  --mp-seed 20260721 \
  --tag mp_v19_intswap_robust_20260721
