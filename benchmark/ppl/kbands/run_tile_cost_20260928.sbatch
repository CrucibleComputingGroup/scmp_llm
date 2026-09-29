#!/bin/bash
# Tile-max avg_sl on the 20 best(all) cells (manifest tile_cost_20260928.json).
# Observe-only re-run of the deployed full-test eval through benchmark/ppl/tile_cost.py:
# same environment as run_prc_ppl.sbatch, nothing in dispatch or numerics changes.
# Identity (fail-closed, written to status.json): the new trace's PPL must equal the
# archived PPL exactly, and the flat (R=1) avg_sl must equal the archived trace cost.
#
# TILE_MODE=fp (cheap): SC matmuls become exact FP matmuls, dispatch unchanged, so rows
# are classified on the FP trajectory. No trace/PPL identity; instead the flat avg_sl must
# land within 1% of the archived trace cost. Outputs under <out>/fp/<cell>/.
#
# Index order is t32 -> t96, 4B/llama8B/14B/30B within a budget. Throttle to stay
# within the 4-GPU total alongside other jobs, e.g.
#   sbatch --array=0-19%3 --export=ALL,TILE_MODE=fp --time=4:00:00 run_tile_cost_20260928.sbatch
# Job name deliberately does NOT start with prcppl_ (the archive builder globs that).
#SBATCH --job-name=tilecost_20260928
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/tilecost_%A_%a.out
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
for k, v in (("CELL", c["id"]), ("CFG", c["wrapper"]), ("HYB", c["hybrid"]), ("HF", c["hf"]),
             ("OUTROOT", m["out"])):
    print(f"{k}={shlex.quote(str(v))}")
PY
)"
MODE=${TILE_MODE:-exact}
[[ "$MODE" == exact || "$MODE" == fp ]] || { echo "[tile] TILE_MODE must be exact|fp" >&2; exit 2; }
OUTD="$OUTROOT/$CELL"
[[ "$MODE" == fp ]] && OUTD="$OUTROOT/fp/$CELL"
mkdir -p "$OUTD"
[[ -f "$CFG" && -f "$HYB" ]] || { echo "[tile] missing wrapper/hybrid for $CELL" >&2; exit 2; }

HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
# Environment of run_prc_ppl.sbatch with its defaults (the deployed cells set none of
# KB_MASKS / SC_RNG_GRID* / KB_FRONTEND).
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
export TILE_OUT="$OUTD/tile.json"
if [[ "$MODE" == fp ]]; then
  export TILE_FP_MATMUL=1
  unset SC_MP_TRACE
else
  export SC_MP_TRACE="$OUTD/trace.json"
fi
echo "[tile] $CELL mode=$MODE model=$HF"
echo "[tile] config=$CFG"
echo "[tile] hybrid=$HYB"

srun python -u benchmark/ppl/tile_cost.py

python - "$MANIFEST" "$IDX" "$OUTD" "$MODE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); c = m["cells"][int(sys.argv[2])]; outd = sys.argv[3]
mode = sys.argv[4]
t = json.load(open(f"{outd}/tile.json"))
flat = t["avg_sl"]["1"]
flat_err = flat / c["archived_trace_cost"] - 1
if mode == "fp":
    ppl, trace_cost = None, None
    checks = {"fp_flat_within_1pct_of_archived_cost": abs(flat_err) < 0.01}
else:
    tr = json.load(open(f"{outd}/trace.json"))
    g = tr["groups"]
    trace_cost = sum(x["macs"] * x["stoc_len"] for x in g) / sum(x["macs"] for x in g)
    ppl = tr["header"]["ppl"]
    checks = {
        "ppl_equals_archived": ppl == c["archived_ppl"],
        "trace_cost_equals_archived": abs(trace_cost / c["archived_trace_cost"] - 1) < 1e-9,
        "flat_equals_trace_cost": abs(flat / trace_cost - 1) < 1e-6,
    }
status = dict(cell=c["id"], arm=c["arm"], mode=mode, ok=all(checks.values()), checks=checks,
              ppl=ppl, archived_ppl=c["archived_ppl"],
              trace_cost=trace_cost, archived_trace_cost=c["archived_trace_cost"],
              flat_avg_sl=flat, flat_rel_err_vs_archived=flat_err,
              tile_avg_sl=t["avg_sl"], linear_avg_sl=t["linear_avg_sl"],
              attention_avg_sl=t["attention_avg_sl"])
json.dump(status, open(f"{outd}/status.json", "w"), indent=1)
print(json.dumps(status, indent=1))
sys.exit(0 if status["ok"] else 3)
PY
