#!/bin/bash
# T7 figure capture: per-group stream-length maps of the deployed allocation.
# Observe-only (benchmark/ppl/group_map_dump.py): the deployed full-test eval environment
# of the tile-cost / prc-ppl runs, stopped after DUMP_WINDOWS windows. No PPL, no trace.
# Cells come from kbands/tile_cost_20260928.json (6 = 14B_t40, 7 = 30B_t40):
#   sbatch --array=6,7%2 run_group_map_20260929.sbatch
# Outputs: <Turbo>/hpca/kbands/group_map_20260929/<cell>/group_map.npz (+ .npz.json)
#SBATCH --job-name=groupmap_20260929
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/groupmap_%A_%a.out
set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}

MANIFEST=/home/allenjin/Projects/SCMP/scmp_llm/benchmark/ppl/kbands/tile_cost_20260928.json
IDX=${SLURM_ARRAY_TASK_ID:?run as an array}
eval "$(python - "$MANIFEST" "$IDX" <<'PY'
import json, shlex, sys
m = json.load(open(sys.argv[1])); c = m["cells"][int(sys.argv[2])]
for k, v in (("CELL", c["id"]), ("CFG", c["wrapper"]), ("HYB", c["hybrid"]), ("HF", c["hf"])):
    print(f"{k}={shlex.quote(str(v))}")
PY
)"
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/group_map_20260929/$CELL
mkdir -p "$OUTD"
[[ -f "$CFG" && -f "$HYB" ]] || { echo "[gmap] missing wrapper/hybrid for $CELL" >&2; exit 2; }

HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
# Environment of run_prc_ppl.sbatch with its defaults (as in run_tile_cost_20260928).
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SC_RNG_GRID="" SC_RNG_GRID_ATTN="" SC_RNG_GRID_QK="" SC_RNG_GRID_AV=""
export QUANT_CONFIG=mp
export MP_CONFIG_JSON="$CFG"
export SC_HYBRID_CONFIG_JSON="$HYB"
export ACT_SCALES_DIR=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales
export FRONTEND=awq
export MODEL_PATH="$HF"
export CTX=2048
export SQ_ALPHA=0.5
export PPL_MAX_TOKENS=0
unset SC_MP_TRACE
export DUMP_OUT="$OUTD/group_map.npz"
export DUMP_WINDOWS=${DUMP_WINDOWS:-4}
export DUMP_FP_MATMUL=${DUMP_FP_MATMUL:-0}
echo "[gmap] $CELL model=$HF windows=$DUMP_WINDOWS fp=$DUMP_FP_MATMUL"
echo "[gmap] config=$CFG"
echo "[gmap] hybrid=$HYB"

srun python -u benchmark/ppl/group_map_dump.py
