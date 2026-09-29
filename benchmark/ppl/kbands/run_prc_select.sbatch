#!/bin/bash
# Held-out selection among pre-built per-group tables (e.g. prc_offline_solve.py outputs)
# -> full-protocol TEST eval of the best candidate if its held-out NLL beats the round-1
# headline (c6 gfis). Reference tables: parent, c6gfis, c17.
#   KB_MODEL=4B KB_TARGET=32 PS_TAG=c8o PS_TABS="fisclip10=/.../4B_t32_c8o_fisclip10.json,..." \
#     PS_CALIBN=8,14 sbatch --job-name=ps_4B_t32_c8o run_prc_select.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
PARENT=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2
HEAD=$OUTD/${KB_MODEL}_t${KB_TARGET}_c6_gfis.json
C17=$OUTD/${KB_MODEL}_t${KB_TARGET}_c17.json
[[ -f $OUTD/${KB_MODEL}_t${KB_TARGET}_c17e32.json ]] && C17=$OUTD/${KB_MODEL}_t${KB_TARGET}_c17e32.json
[[ -f $C17 ]] || C17=""
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$PARENT/table.json")
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 CTX=2048 SQ_ALPHA=0.5
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
TAG=${PS_TAG:?set PS_TAG}
HN=$OUTD/${KB_MODEL}_t${KB_TARGET}_${TAG}_select_heldout_nll.json
TABS="parent=$PARENT/wrapper.json,c6gfis=$HEAD${C17:+,c17=$C17}"
IFS=',' read -ra CAND <<< "${PS_TABS:?set PS_TABS}"
for c in "${CAND[@]}"; do TABS="$TABS,${TAG}_${c}"; done
srun python -u benchmark/ppl/heldout_nll.py --model_path "$HF" --hybrid "$PARENT/hybrid_config.json" \
  --tables "$TABS" --windows "${PS_HWIN:-24}" --calib-n "${PS_CALIBN:-8}" --out "$HN"
BEST=$(python - "$HN" "$TAG" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))["tables"]; tag = sys.argv[2] + "_"
ref = h["c6gfis"]["window_nll"]
best, bm = None, 0.0
for n, t in h.items():
    if not n.startswith(tag):
        continue
    d = [a - b for a, b in zip(t["window_nll"], ref)]
    m = sum(d) / len(d); sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    z = m / (sd / len(d) ** 0.5) if sd > 0 else 0.0
    print(f"[select] {n}: vs round-1 headline dNLL {m:+.5f} (z {z:+.2f}); cost {t.get('cost', 0):.3f}",
          file=sys.stderr)
    if m < 0 and z < -1.5 and (best is None or m < bm):   # a test eval needs held-out evidence
        best, bm = n, m
print(best or "")
PY
)
EV=${PS_EVAL:-$BEST}
if [[ -z "$EV" ]]; then
  echo "[select] no candidate beats the round-1 headline on held-out (z < -1.5) -- no test eval"
  exit 0
fi
WR=$(python - "$HN" "$EV" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["tables"][sys.argv[2]]["wrapper"])
PY
)
echo "[select] full-protocol eval of ${EV} (${WR})"
export KB_ARM=prc KB_PRC_OVERRIDE="$WR" KB_TBL="p2${EV//_/}"
bash benchmark/ppl/kbands/run_prc_ppl.sbatch
