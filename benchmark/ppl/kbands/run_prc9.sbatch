#!/bin/bash
# Round 3 end-to-end for one dense cell: calib8 (12 windows, per-pair dump) -> robust-Fisher
# offline re-solve (prc_offline_solve.py: fisclipC / kraw currencies x quartile / per-block keys)
# -> paired held-out TRAIN NLL vs parent, c17 and the round-1 headline (c6 gfis)
# -> full-protocol TEST eval of the best candidate if it beats the headline on held-out.
#   KB_MODEL=llama8B KB_TARGET=32 sbatch --job-name=p9_llama8B_t32 run_prc9.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
CW=${P8_CW:-12}
PARENT=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2
HEAD=$OUTD/${KB_MODEL}_t${KB_TARGET}_c6_gfis.json
[[ -f $HEAD ]] || { echo "[prc9] no round-1 headline table $HEAD" >&2; exit 3; }
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$PARENT/table.json")
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
NB=$(python - "$HF" <<'PY'
import sys
from transformers import AutoConfig
print(AutoConfig.from_pretrained(sys.argv[1]).num_hidden_layers)
PY
)
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 CTX=2048 SQ_ALPHA=0.5
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
# P9_EXPAND=1: second (re-linearized) pass -- calibrate ON the round-1 table's trajectory
EXP=(); SFX=${P9_SFX:-c8}
if [[ ${P9_EXPAND:-0} == 1 ]]; then EXP=(--expand-at "$HEAD"); SFX=${P9_SFX:-c8x}; fi
STEM=$OUTD/${KB_MODEL}_t${KB_TARGET}_${SFX}
if [[ ! -f ${STEM}.npz ]]; then
  rm -f ${STEM}_*.json
  srun python -u benchmark/ppl/mp_per_row_chunk_calib8.py --parent "$PARENT" --model_path "$HF" \
    --out "${STEM}.json" --diag "${STEM}_diag.json" --calib-windows "$CW" --holdout-windows 2 \
    --rows-per-call 64 --currencies fis --grad-mode "${P6_GRAD:-full}" --cc-classes "" \
    --dump "${STEM}.npz" "${EXP[@]}"
else
  echo "[prc9] reusing dump ${STEM}.npz"
fi
srun python -u benchmark/ppl/prc_offline_solve.py --dump "${STEM}.npz" --parent "$PARENT" \
  --out-stem "${STEM}o" --n-blocks "$NB" \
  --variants "${P9_VARIANTS:-fisclip10:q,fisclip100:q,kraw:q,fisclip10:blk,fisclip100:blk,kraw:blk}"
TABS=$(python - "${STEM}o_offline_diag.json" "$(basename "${STEM}o")_" <<'PY'
import json, sys
from pathlib import Path
v = json.load(open(sys.argv[1]))["variants"]
print(",".join(f"{Path(r['wrapper']).stem.split(sys.argv[2], 1)[1]}={r['wrapper']}" for r in v.values()))
PY
)
export KB_MODEL KB_TARGET PS_TAG=${SFX}o PS_TABS="$TABS" PS_CALIBN="8,$((CW + 2))"
bash benchmark/ppl/kbands/run_prc_select.sbatch
