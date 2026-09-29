#!/bin/bash
# calib8 (round 3: finer deployable keys for the linears under one global Fisher lambda; attention = parent)
# -> paired held-out TRAIN NLL of parent, c17, the round-1 headline (c6 gfis) and the round-3 candidates
# -> full-protocol TEST eval of the best candidate if its held-out NLL beats the headline.
#   KB_MODEL=4B KB_TARGET=32 sbatch --job-name=p8_4B_t32_c8 run_prc8.sbatch
#   env: P8_CW (calib windows, default 12), P8_DUMP=1 (per-pair npz), P6_GRAD (full|block),
#        P2_EXPERTS, P8_HWIN (held-out windows, default 24), P6_EVAL (force the evaluated table)
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
TAG=${P6_TAG:-c8}
CW=${P8_CW:-12}
PARENT=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2
C17=$OUTD/${KB_MODEL}_t${KB_TARGET}_c17.json
[[ -f $OUTD/${KB_MODEL}_t${KB_TARGET}_c17e32.json ]] && C17=$OUTD/${KB_MODEL}_t${KB_TARGET}_c17e32.json
[[ -f $C17 ]] || C17=""
HEAD=$OUTD/${KB_MODEL}_t${KB_TARGET}_c6_gfis.json          # round-1 headline table
[[ -f $HEAD ]] || { echo "[prc8] no round-1 headline table $HEAD" >&2; exit 3; }
SCORE="c6gfis=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['threshold_table_path'])" "$HEAD")"
if [[ -n $C17 ]]; then
  SCORE="$SCORE,c17=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['threshold_table_path'])" "$C17")"
fi
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$PARENT/table.json")
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 CTX=2048 SQ_ALPHA=0.5
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
STEM=$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}
DIAG=${STEM}_diag.json; HN=${STEM}_heldout_nll.json
rm -f "$DIAG" "$HN" ${STEM}_*.json
DUMP=()
[[ ${P8_DUMP:-0} == 1 ]] && DUMP=(--dump "${STEM}.npz")
srun python -u benchmark/ppl/mp_per_row_chunk_calib8.py --parent "$PARENT" --model_path "$HF" \
  --out "${STEM}.json" --diag "$DIAG" --calib-windows "$CW" --holdout-windows 2 --rows-per-call 64 \
  --expert-calls-per-block "${P2_EXPERTS:-8}" --currencies fis --grad-mode "${P6_GRAD:-full}" \
  --score-tables "$SCORE" "${DUMP[@]}"
TABS="parent=$PARENT/wrapper.json,c6gfis=$HEAD${C17:+,c17=$C17}"
for t in gfis gfisblk; do [[ -f ${STEM}_${t}.json ]] && TABS="$TABS,${TAG}${t}=${STEM}_${t}.json"; done
# exclude both the round-1 calibration windows (8, seed 0) and this run's (CW+2, seed 0)
srun python -u benchmark/ppl/heldout_nll.py --model_path "$HF" --hybrid "$PARENT/hybrid_config.json" \
  --tables "$TABS" --windows "${P8_HWIN:-24}" --calib-n "8,$((CW + 2))" --out "$HN"
BEST=$(python - "$HN" "$TAG" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))["tables"]; tag = sys.argv[2]
ref = h["c6gfis"]["window_nll"]
best, bm = None, 0.0
for n, t in h.items():
    if not n.startswith(tag):
        continue
    d = [a - b for a, b in zip(t["window_nll"], ref)]
    m = sum(d) / len(d); sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    z = m / (sd / len(d) ** 0.5) if sd > 0 else 0.0
    print(f"[prc8] {n}: vs round-1 headline dNLL {m:+.5f} (z {z:+.2f}); cost {t.get('cost', 0):.3f}",
          file=sys.stderr)
    if m < 0 and (best is None or m < bm):
        best, bm = n[len(tag):], m
print(best or "")
PY
)
EV=${P6_EVAL:-$BEST}
if [[ -z "$EV" ]]; then
  echo "[prc8] no round-3 candidate beats the round-1 headline on held-out -- no test eval"
  exit 0
fi
[[ -f "${STEM}_${EV}.json" ]] || { echo "[prc8] no ${EV} table -- cannot eval" >&2; exit 4; }
echo "[prc8] full-protocol eval of ${TAG} ${EV}"
export KB_ARM=prc KB_PRC_OVERRIDE="${STEM}_${EV}.json" KB_TBL="p2${TAG}${EV}"
bash benchmark/ppl/kbands/run_prc_ppl.sbatch
