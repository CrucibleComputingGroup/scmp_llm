#!/bin/bash
#SBATCH --job-name=fig3_sigma_4b
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_fig3_sigma_20260731/%x_%j.out

# Fig. 3 regeneration data: per-group SC reconstruction error (sigma) on
# Qwen3-4B at halved L in {128, 64} = NOMINAL {256, 128}, per (op x layer)
# call site (--layer_buckets 36 = per-layer on the 36-layer model), with the
# raw per-group arrays dumped via raw_sigma_dump.py. Clean sigma operating
# point: SmoothQuant front-end (deployment-matched), no protected channels,
# no hybrid mask, no gate.

set +u; source ~/.bashrc; conda activate annstention; set -u
set -eo pipefail
cd /home/allenjin/Projects/scmp_llm

OUT=/nfs/turbo/coe-nbleier/allenjin/hpca/fig3_sigma_qwen4b_20260731
mkdir -p "$OUT"

export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SQ_ALPHA=0.5
export ACT_SCALES_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/act_scales
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAW_SIGMA_NPZ=$OUT/qwen4b_raw_sigma_L128_64.npz

echo "[fig3_sigma] START $(date)"
nvidia-smi -L || true

python -u benchmark/ppl/fig3_sigma/raw_sigma_dump.py \
  --model_path Qwen/Qwen3-4B-Instruct-2507 \
  --mp_levels 128,64 \
  --budget_ratio 0.75 --budget_ref_stoc_len 128 \
  --sc_prec 8 --halve 1 --chunk_d 128 \
  --ctx_len 2048 --num_calib_sequences 16 \
  --layer_buckets 36 --max_units_per_call 512 --seed 0 \
  --calib-smoothquant \
  --output_json "$OUT/qwen4b_sigma_L128_64.json"

echo "[fig3_sigma] DONE $(date)"
