#!/bin/bash
# INT baselines on the AWQ front-end — so SC-AWQ can be tabled against INT at a
# MATCHED front-end (mirrors the SmoothQuant INT baselines in hpca_results/llm/int/,
# only the front-end differs). One job per model runs the six cells the SUMMARY
# INT8/INT6/INT4 reference is drawn from: {W8A8,W6A6,W4A4} x {symm,asymm}; take
# the lower of symm/asymm per (model,precision) for the table.
#
# FRONTEND=awq reuses the SAME cached AWQ scale vectors the SC-AWQ wave built
# (byte-identical front-end); apply_ptq folds them in place of SmoothQuant. The
# cache MUST already exist — the INT path cannot calibrate AWQ (no SCLinear).
#
# NOTE (caption honestly): AWQ is an INT-native method, so it may help INT >= SC,
# especially at W4A4. That is expected and is why SC's case is made on ENERGY.
#
# Usage: IA_MODEL=4B sbatch run_int_awq.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_int_awq/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${IA_MODEL:?IA_MODEL is required (4B|llama8B|14B|30B)}"

# The AWQ front-end selection. hpca sets the HF caches itself; AWQ_SCALES_DIR is
# NOT set by hpca, so point it at the Turbo cache the SC-AWQ wave wrote.
export FRONTEND=awq
export AWQ_OBJ_BITS=4
export AWQ_SCALES_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/act_scales/awq_scales
export PPL_MAX_TOKENS=0
export CTX=2048

# Fail fast if the AWQ cache is missing (build_model would SystemExit anyway,
# but check here before Slurm burns the model load).
case "$IA_MODEL" in
  4B)      HF=Qwen_Qwen3-4B-Instruct-2507 ;;
  llama8B) HF=meta-llama_Llama-3.1-8B-Instruct ;;
  14B)     HF=Qwen_Qwen3-14B ;;
  30B)     HF=Qwen_Qwen3-30B-A3B-Instruct-2507 ;;
  *) echo "[int_awq] unknown IA_MODEL=$IA_MODEL" >&2; exit 2 ;;
esac
[[ -f "$AWQ_SCALES_DIR/awq_scales_${HF}_b4.pt" ]] || {
  echo "[int_awq] MISSING AWQ cache $AWQ_SCALES_DIR/awq_scales_${HF}_b4.pt" >&2; exit 2; }

echo "[int_awq] model=$IA_MODEL FRONTEND=awq obj_bits=4 tag=int_awq_20260727"
bash hpca --models "$IA_MODEL" \
  --configs W8A8_symm,W8A8_asymm,W6A6_symm,W6A6_asymm,W4A4_symm,W4A4_asymm \
  --metrics ppl \
  --tag int_awq_20260727
