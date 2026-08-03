#!/bin/bash
# Real 30B gate for the int_swap-only independent-window batching optimization.
# Success requires exact B=1/B=8 per-window losses for SC, linear INT, and
# attention INT schedules.  The dependent ranking job validates this manifest.
#SBATCH --job-name=iswap2_30B_bid
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_hpca_mp_v19_intswap_robust_30B_20260722/%x_%j.out

set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export ACT_SCALES_DIR="$TURBO/act_scales"
export TMPDIR=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WRAPPER="$TURBO/mp_calib_mp_v9_hybdose_all_20260718_183035/Qwen_Qwen3-30B-A3B-Instruct-2507__mp_avg32_v9__act_global_v9_pc0.01x112_act_collapse_comp_pcovdown_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.20_wrapper.json"
OUT="$TURBO/validation/v19_30B_intswap_window_batch_identity_b8_int7.json"

python -u benchmark/ppl/verify_30b_intswap_batching.py \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --parent-wrapper "$WRAPPER" \
  --hybrid-int-bits 7 \
  --windows 8 --ctx 512 --batch-size 8 \
  --out "$OUT"
