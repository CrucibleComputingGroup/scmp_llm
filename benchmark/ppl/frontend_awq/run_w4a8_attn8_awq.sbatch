#!/bin/bash
# AWQ-front-end W4A8 projection baseline with INT8 attention operands.
#
# In benchmark/quant/ptq.py, W4A8 means:
#   * every non-router, non-lm-head Linear (q/k/v/o and gate/up/down projections)
#       - weights: 4-bit RTN
#       - input activations: 8-bit RTN
#   * eager-attention QK^T and softmax*V
#       - both operands: 8-bit RTN
#
# Run both symmetric and asymmetric RTN and retain the lower PPL per model,
# matching the methodology used for the INT8/INT6/INT4 rows in Table VI.
#
# Usage:
#   IA_MODEL=4B sbatch run_w4a8_attn8_awq.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_int_awq_w4a8attn8/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${IA_MODEL:?IA_MODEL is required (4B|llama8B|14B|30B)}"

export FRONTEND=awq
export AWQ_OBJ_BITS=4
export AWQ_SCALES_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/act_scales/awq_scales
export BASELINE_ATTN=eager
export PPL_MAX_TOKENS=0
export PPL_WINDOW_BATCH_SIZE=1
export CTX=2048

case "$IA_MODEL" in
  4B)      HF=Qwen_Qwen3-4B-Instruct-2507 ;;
  llama8B) HF=meta-llama_Llama-3.1-8B-Instruct ;;
  14B)     HF=Qwen_Qwen3-14B ;;
  30B)     HF=Qwen_Qwen3-30B-A3B-Instruct-2507 ;;
  *) echo "[w4a8-attn8] unknown IA_MODEL=$IA_MODEL" >&2; exit 2 ;;
esac
[[ -f "$AWQ_SCALES_DIR/awq_scales_${HF}_b4.pt" ]] || {
  echo "[w4a8-attn8] MISSING AWQ cache $AWQ_SCALES_DIR/awq_scales_${HF}_b4.pt" >&2
  exit 2
}

echo "[w4a8-attn8] model=$IA_MODEL frontend=awq projections=W4A8 attention=INT8 tag=int_awq_w4a8attn8_20260730"
bash hpca --models "$IA_MODEL" \
  --configs W4A8_symm,W4A8_asymm \
  --metrics ppl \
  --tag int_awq_w4a8attn8_20260730
