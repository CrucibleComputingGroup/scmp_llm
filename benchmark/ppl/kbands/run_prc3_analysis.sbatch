#!/bin/bash
# Calibrate one per-(row, chunk) v2 table (mp_per_row_chunk_calib2.py) and its
# held-out pre-flight. No PPL here -- the eval is run_prc_ppl.sbatch with
# KB_PRC_OVERRIDE pointing at the wrapper this writes.
#
#   KB_MODEL=4B KB_TARGET=32 P2_TAG=d17 sbatch run_prc2_calib.sbatch
#
# P2_EVAL=1 chains the full-protocol PPL eval (run_prc_ppl.sbatch, KB_ARM=prc,
# KB_PRC_OVERRIDE=<this table>) in the SAME allocation, but only if the held-out
# pre-flight passes a gate fixed before any result: realized linear-dispatched L
# within [-3%, +2%] of the parent's, and total squared error below the parent's.
# Queue waits dominate on this partition, so re-queueing for the eval is the
# expensive step, not the eval.
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/SCMP/scmp_llm

: "${KB_MODEL:?}"; : "${KB_TARGET:?}"; : "${P2_TAG:?}"
PARENT=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}
KB=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801
PTRACE=$KB/ppl/${KB_MODEL}_t${KB_TARGET}_parent_trace.json
V1TBL=$KB/prc/${KB_MODEL}_t${KB_TARGET}_v7_prc_table.json
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2
mkdir -p "$OUTD"
[[ -f "$PTRACE" ]] || { echo "[prc2] missing parent trace $PTRACE" >&2; exit 2; }
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$PARENT/table.json")

HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 CTX=2048 SQ_ALPHA=0.5
# The conda env's editable scmp_kernels install still maps to the pre-move path
# /home/allenjin/Projects/scmp_llm/kernels (gone since the 2026-09-21 SCMP/ move),
# so `import scmp_kernels` fails in eval_quant. PathFinder (sys.path) runs before
# the editable finder, so PYTHONPATH restores it without touching the env.
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}

echo "[prc2] $KB_MODEL t$KB_TARGET tag=$P2_TAG ladder=${P2_LADDER:-dense} hf=$HF"
srun python -u benchmark/ppl/mp_per_row_chunk_calib3.py \
  --parent "$PARENT" --parent-trace "$PTRACE" --model_path "$HF" \
  --out "$OUTD/${KB_MODEL}_t${KB_TARGET}_${P2_TAG}.json" \
  --diag "$OUTD/${KB_MODEL}_t${KB_TARGET}_${P2_TAG}_diag.json" \
  --compare-table "$V1TBL" \
  ${P2_LADDER:+--ladder "$P2_LADDER"} \
  --calib-windows "${P2_CW:-4}" --holdout-windows "${P2_HW:-2}" \
  --rows-per-call "${P2_ROWS:-256}" \
  --expert-calls-per-block "${P2_EXPERTS:-8}" \
  --budget-source "${P2_BUDGET:-calib}" --windows "${P2_WINDOWS:-stratified}" \
  ${P2_EMIT:+--emit-scales "$P2_EMIT"} \
  --frontend awq

exit 0
DIAG="$OUTD/${KB_MODEL}_t${KB_TARGET}_${P2_TAG}_diag.json"
python - "$DIAG" <<'PY' || { echo "[prc2] GATE FAILED -- no eval" >&2; exit 3; }
import json, sys
d = json.load(open(sys.argv[1]))["ops"]
ops = [k for k in d if not k.startswith("total_")]
newL, parL = d["total_new_L"], d["total_par_L"]
en = sum(d[o]["new_err"] for o in ops); ep = sum(d[o]["par_err"] for o in ops)
r = newL / parL
print(f"[prc2] gate: L new/par {newL:.2f}/{parL:.2f} = {r:.4f}; err new/par {en/ep:.4f}")
sys.exit(0 if (0.97 <= r <= 1.02 and en < ep) else 1)
PY
echo "[prc2] gate passed -> full-protocol eval"
export KB_ARM=prc KB_PRC_OVERRIDE="$OUTD/${KB_MODEL}_t${KB_TARGET}_${P2_TAG}.json" KB_TBL="p2${P2_TAG}"
bash benchmark/ppl/kbands/run_prc_ppl.sbatch
