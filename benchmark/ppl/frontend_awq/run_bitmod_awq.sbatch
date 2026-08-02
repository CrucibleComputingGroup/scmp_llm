#!/bin/bash
# BitMoD weight-only baselines on the AWQ front-end (= AWQ + BitMoD, which is
# BitMoD's own strongest published setting; our wonly_bitmod baselines used RTN).
# AWQ is an activation-aware WEIGHT method, so it applies to weight-only configs
# (QuantLinear folds w=w*s BEFORE the BitMoD datatype quant, x/s at forward), and
# build_model now runs the AWQ front-end even at a_bits=16 (SmoothQuant stays
# skipped weight-only). One job per model; the two mixed-datatype cells.
#
# NOTE (caption honestly): the AWQ scale cache is obj_bits=4 (INT4 objective),
# reused here as the front-end scale for the BitMoD datatype — a proxy, not a
# scale searched against BitMoD. Mirrors how we reused it for INT-AWQ.
#
# Usage: BA_MODEL=4B sbatch run_bitmod_awq.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_bitmod_awq/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${BA_MODEL:?BA_MODEL is required (4B|llama8B|14B|30B)}"

export FRONTEND=awq
export AWQ_OBJ_BITS=4
export AWQ_SCALES_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/act_scales/awq_scales
export PPL_MAX_TOKENS=0
export CTX=2048

case "$BA_MODEL" in
  4B)      HF=Qwen_Qwen3-4B-Instruct-2507 ;;
  llama8B) HF=meta-llama_Llama-3.1-8B-Instruct ;;
  14B)     HF=Qwen_Qwen3-14B ;;
  30B)     HF=Qwen_Qwen3-30B-A3B-Instruct-2507 ;;
  *) echo "[bitmod_awq] unknown BA_MODEL=$BA_MODEL" >&2; exit 2 ;;
esac
[[ -f "$AWQ_SCALES_DIR/awq_scales_${HF}_b4.pt" ]] || {
  echo "[bitmod_awq] MISSING AWQ cache $AWQ_SCALES_DIR/awq_scales_${HF}_b4.pt" >&2; exit 2; }

echo "[bitmod_awq] model=$BA_MODEL FRONTEND=awq obj_bits=4 tag=bitmod_awq_20260729"
bash hpca --models "$BA_MODEL" \
  --configs W4A16_bitmod,W3A16_bitmod \
  --metrics ppl \
  --tag bitmod_awq_20260729
