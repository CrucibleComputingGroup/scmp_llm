#!/bin/bash
# calib6 (within/global x sigma/Fisher tables) -> paired held-out TRAIN NLL of
# parent, c17 and the four candidates -> gate -> full-protocol TEST eval of the best.
# Gate: best candidate's held-out NLL beats c17 by >= 2 paired SE AND beats the parent.
#   KB_MODEL=4B KB_TARGET=32 sbatch run_prc6.sbatch
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
TAG=${P6_TAG:-c6}
# KB_PARENT: an alternate parent config dir (wrapper/table/hybrid_config), e.g. the 14B h=20% cells
PARENT=${KB_PARENT:-/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}}
KB=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2
C17=${P6_C17:-$OUTD/${KB_MODEL}_t${KB_TARGET}_c17.json}
[[ -f $OUTD/${KB_MODEL}_t${KB_TARGET}_c17e32.json ]] && C17=$OUTD/${KB_MODEL}_t${KB_TARGET}_c17e32.json
SCORE=""
if [[ -f $C17 ]]; then
  C17T=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['threshold_table_path'])" "$C17")
  SCORE="c17=$C17T"
else
  echo "[prc6] no c17 table for this cell -- gating against the parent only"
  C17=""
fi
G5=$OUTD/${KB_MODEL}_t${KB_TARGET}_g5_table.json
[[ -f $G5 ]] && SCORE="${SCORE:+$SCORE,}g5=$G5"
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$PARENT/table.json")
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 CTX=2048 SQ_ALPHA=0.5
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
STEM=$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}
DIAG=${STEM}_diag.json; HN=${STEM}_heldout_nll.json
rm -f "$DIAG" "$HN" ${STEM}_[wg]{rel,fis,fis_emp}.json ${STEM}_[wg]{rel,fis,fis_emp}_table.json
srun python -u benchmark/ppl/mp_per_row_chunk_calib6.py --parent "$PARENT" --model_path "$HF" \
  --out "${STEM}.json" --diag "$DIAG" --calib-windows 6 --holdout-windows 2 --rows-per-call 64 \
  --expert-calls-per-block "${P2_EXPERTS:-8}" --currencies "${P6_CUR:-rel,fis}" --grad-mode "${P6_GRAD:-full}" --score-tables "$SCORE"
TABS="parent=$PARENT/wrapper.json${C17:+,c17=$C17}"
for t in wrel grel wfis gfis wfis_emp gfis_emp; do [[ -f ${STEM}_${t}.json ]] && TABS="$TABS,$t=${STEM}_${t}.json"; done
srun python -u benchmark/ppl/heldout_nll.py --model_path "$HF" --hybrid "$PARENT/hybrid_config.json" \
  --tables "$TABS" --windows "${P6_HWIN:-16}" --out "$HN"
BEST=$(python - "$HN" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))["tables"]
p = h["parent"]["window_nll"]
c = h["c17"]["window_nll"] if "c17" in h else p   # no c17: gate against the parent
best, bz = None, 0.0
for n, t in h.items():
    if n in ("parent", "c17"):
        continue
    d = [a - b for a, b in zip(t["window_nll"], c)]
    m = sum(d) / len(d); sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    z = m / (sd / len(d) ** 0.5) if sd > 0 else 0.0
    dp = sum(a - b for a, b in zip(t["window_nll"], p)) / len(p)
    print(f"[prc6] {n}: vs c17 dNLL {m:+.5f} (z {z:+.2f}); vs parent {dp:+.5f}; cost {t.get('cost', 0):.3f}", file=sys.stderr)
    if z < -2.0 and dp < 0 and (best is None or z < bz):
        best, bz = n, z
print(best or "")
PY
)
if [[ -z "$BEST" ]]; then echo "[prc6] held-out gate: no candidate beats c17 by 2 SE (reported, not blocking)" >&2
else echo "[prc6] held-out gate: $BEST beats c17 by >=2 SE"; fi
# The headline is ONE algorithm on every cell: always test-evaluate the global MC-Fisher table.
EV=${P6_EVAL:-gfis}
[[ -f "${STEM}_${EV}.json" ]] || { echo "[prc6] no ${EV} table -- cannot eval" >&2; exit 4; }
echo "[prc6] full-protocol eval of ${EV}"
export KB_ARM=prc KB_PRC_OVERRIDE="${STEM}_${EV}.json" KB_TBL="p2${TAG}${EV}"
bash benchmark/ppl/kbands/run_prc_ppl.sbatch
