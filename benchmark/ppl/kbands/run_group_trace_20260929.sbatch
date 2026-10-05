#!/bin/bash
# Per-group precision trace (benchmark/ppl/group_trace.py): the stream length of every
# computation group, for the energy/latency simulator. One best(all) cell per array task
# (cells of kbands/tile_cost_20260928.json; 8 = 4B_t48), first GTRACE_WINDOWS windows:
#   1. fp     FP-trajectory capture -- the way that needs no SC inference
#   2. exact  the same windows on the real SC path, plus their scmp-trace-summary
#   3. check  exact groups reproduce that summary exactly; fp vs exact per group; tokens
#             are the canonical stream; context vs the archived full-test trace
#   sbatch --array=8 run_group_trace_20260929.sbatch
# Outputs: <Turbo>/hpca/kbands/group_trace_20260929/<cell>/{fp,exact}/group_trace.{npz,json},
#          exact/sc_trace_summary.json, check.json
#SBATCH --job-name=gtrace_20260929
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/gtrace_%A_%a.out
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
for k, v in (("CELL", c["id"]), ("ARM", c["arm"]), ("CFG", c["wrapper"]), ("HYB", c["hybrid"]),
             ("HF", c["hf"]), ("ARCHIVED", c["archived_trace"])):
    print(f"{k}={shlex.quote(str(v))}")
PY
)"
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/group_trace_20260929/$CELL
mkdir -p "$OUTD"
[[ -f "$CFG" && -f "$HYB" ]] || { echo "[gtrace] missing wrapper/hybrid for $CELL" >&2; exit 2; }

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
export GTRACE_CELL="$CELL" GTRACE_ARM="$ARM"
export GTRACE_WINDOWS=${GTRACE_WINDOWS:-4}
echo "[gtrace] $CELL ($ARM) model=$HF windows=$GTRACE_WINDOWS"
echo "[gtrace] config=$CFG"
echo "[gtrace] hybrid=$HYB"

GTRACE_FP_MATMUL=1 GTRACE_OUT="$OUTD/fp" srun python -u benchmark/ppl/group_trace.py
GTRACE_FP_MATMUL=0 GTRACE_OUT="$OUTD/exact" GTRACE_CHECK_TRACE="$OUTD/exact/sc_trace_summary.json" \
  srun python -u benchmark/ppl/group_trace.py
python -u benchmark/ppl/group_trace_check.py --exact "$OUTD/exact" --fp "$OUTD/fp" \
  --sc-trace "$OUTD/exact/sc_trace_summary.json" --archived-trace "$ARCHIVED" \
  --verify-tokens --out "$OUTD/check.json"
