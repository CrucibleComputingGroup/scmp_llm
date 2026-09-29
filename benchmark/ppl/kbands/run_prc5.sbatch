#!/bin/bash
# Stage 1 of the GLOBAL per-group allocation (no per-bucket budgets):
#   calib5 (one lambda across all linear (row,chunk) groups, sigma-pricing currency)
#   -> paired held-out TRAIN NLL of parent / c17 / g5 (selection signal, not a PPL)
#   -> gate: g5 beats BOTH parent and c17 on held-out NLL and its held-out linear L is
#      within [-3%, +2%] of the parent's -> full-protocol TEST eval of g5.
#   KB_MODEL=4B KB_TARGET=32 sbatch run_prc5.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/SCMP/scmp_llm
: "${KB_MODEL:?}"; : "${KB_TARGET:?}"
TAG=${P5_TAG:-g5}
PARENT=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}
KB=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2
C17=$OUTD/${KB_MODEL}_t${KB_TARGET}_c17.json
[[ -f $OUTD/${KB_MODEL}_t${KB_TARGET}_c17e32.json ]] && C17=$OUTD/${KB_MODEL}_t${KB_TARGET}_c17e32.json
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$PARENT/table.json")
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 CTX=2048 SQ_ALPHA=0.5
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
G=$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}.json
srun python -u benchmark/ppl/mp_per_row_chunk_calib5.py \
  --parent "$PARENT" --parent-trace "$KB/ppl/${KB_MODEL}_t${KB_TARGET}_parent_trace.json" \
  --model_path "$HF" --out "$G" --diag "$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}_diag.json" \
  --calib-windows 6 --holdout-windows 2 --rows-per-call 64 --expert-calls-per-block "${P2_EXPERTS:-8}" \
  --budget-source calib --windows stratified --frontend awq \
  --currency "${P5_CURRENCY:-rel}" --global-scale "${P5_GSCALE:-1.0}"
HN=$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}_heldout_nll.json
srun python -u benchmark/ppl/heldout_nll.py --model_path "$HF" --hybrid "$PARENT/hybrid_config.json" \
  --tables "parent=$PARENT/wrapper.json,c17=$C17,$TAG=$G" --windows "${P5_HWIN:-16}" --out "$HN"
python - "$HN" "$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}_diag.json" "$TAG" <<'PY' || { echo "[prc5] GATE FAILED -- no test eval" >&2; exit 3; }
import json, sys
h = json.load(open(sys.argv[1])); d = json.load(open(sys.argv[2])); tag = sys.argv[3]
t = h["tables"]
g, c, p = t[tag]["mean_nll"], t["c17"]["mean_nll"], t["parent"]["mean_nll"]
ho = d["global"]["holdout"]; r = ho["L_global"] / ho["L_parent"]
print(f"[prc5] gate: heldout NLL {tag} {g:.5f} c17 {c:.5f} parent {p:.5f}; L global/parent {r:.4f}")
sys.exit(0 if (g < c and g < p and 0.97 <= r <= 1.02) else 1)
PY
echo "[prc5] gate passed -> full-protocol eval"
export KB_ARM=prc KB_PRC_OVERRIDE="$G" KB_TBL="p2${TAG}"
bash benchmark/ppl/kbands/run_prc_ppl.sbatch
