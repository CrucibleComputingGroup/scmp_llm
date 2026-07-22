#!/bin/bash
# Real 30B gate for the search-only independent-window batching optimization.
# This is a diagnostic, not a citable PPL cell and not a search wave.
# Success requires torch.equal on all eight per-window NLL scalars at B=1/B=8.
#SBATCH --job-name=v20_30B_batchid
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_v20_staged/%x_%j.out
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

PARENT_DIR="$TURBO/mp_calib_mp_v9_int6_20260716_001908"
mapfile -t wrappers < <(find "$PARENT_DIR" -maxdepth 1 -type f \
  -name 'Qwen_Qwen3-30B-A3B-Instruct-2507__mp_avg32_v9__act_global_v9*_wrapper.json' -print)
[[ ${#wrappers[@]} -eq 1 ]] || {
  echo "expected one 30B V9 wrapper, got ${#wrappers[@]}" >&2
  exit 21
}
HYBRID="$TURBO/hybrid_configs/_hpca_mp_v9_hybdose_all_20260718_183035/30B_mp_avg32_v9_rank-sc_int8_measured_curve_top0p20.json"
[[ -s "$HYBRID" ]] || { echo "missing $HYBRID" >&2; exit 22; }

for bits in 6 7; do
  python -u benchmark/ppl/verify_30b_window_batching.py \
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --parent-wrapper "${wrappers[0]}" \
    --hybrid-config "$HYBRID" \
    --hybrid-int-bits "$bits" \
    --windows 8 --ctx 2048 --batch-size 8 \
    --out "$TURBO/validation/v20_30B_window_batch_identity_b8_int${bits}.json"
done
