#!/bin/bash
# Re-calibrate a prc2 cell (same seed/windows => the same iso-cost table as c17) and
# emit an extra table at budget scale P2_SCALE, then run the full-protocol eval of THAT
# scaled table. For direct iso-PPL validation ("same PPL at fewer cycles").
#   KB_MODEL=llama8B KB_TARGET=48 P2_SCALE=0.9 sbatch run_prc2_scaled.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/SCMP/scmp_llm
: "${KB_MODEL:?}"; : "${KB_TARGET:?}"; : "${P2_SCALE:?}"
PCT=$(python -c "print(int(round(float('$P2_SCALE')*100)))")
TAG=c17iso
PARENT=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}
KB=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$PARENT/table.json")
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 CTX=2048 SQ_ALPHA=0.5
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
srun python -u benchmark/ppl/mp_per_row_chunk_calib2.py \
  --parent "$PARENT" --parent-trace "$KB/ppl/${KB_MODEL}_t${KB_TARGET}_parent_trace.json" \
  --model_path "$HF" --out "$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}.json" \
  --diag "$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}_diag.json" \
  --calib-windows 6 --holdout-windows 2 --rows-per-call 64 --expert-calls-per-block "${P2_EXPERTS:-8}" \
  --budget-source calib --windows stratified --emit-scales "$P2_SCALE" --frontend awq
W="$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}_s${PCT}.json"
[[ -f "$W" ]] || { echo "[prc2s] missing $W" >&2; exit 2; }
export KB_ARM=prc KB_PRC_OVERRIDE="$W" KB_TBL="p2${TAG}_s${PCT}"
bash benchmark/ppl/kbands/run_prc_ppl.sbatch
