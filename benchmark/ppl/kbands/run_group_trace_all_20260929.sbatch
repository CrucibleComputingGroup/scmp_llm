#!/bin/bash
# Per-group precision traces (benchmark/ppl/group_trace.py) for ALL 20 best(all) cells
# (kbands/tile_cost_20260928.json), first GTRACE_WINDOWS=4 windows, exact SC path, on ONE GPU:
# the cells run one after another inside this single allocation, so the queue is paid once.
# Per cell: exact capture, then the check (identity vs the capture's own scmp-trace-summary,
# tokens = canonical stream, context vs the archived full-test trace). Cells whose check.json
# is already ok are skipped; a failed cell is logged and the loop moves on.
# Order: 4B, llama8B, 14B, then 30B (slowest), t32 -> t96 within a model.
#   sbatch run_group_trace_all_20260929.sbatch
# Outputs: <Turbo>/hpca/kbands/group_trace_20260929/<cell>/{exact/,check.json}; summary line
# per cell in the log.
#SBATCH --job-name=gtrace_all_20260929
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/gtrace_all_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -uo pipefail          # no -e: one failed cell must not stop the others
cd /home/allenjin/Projects/SCMP/scmp_llm
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}

MANIFEST=/home/allenjin/Projects/SCMP/scmp_llm/benchmark/ppl/kbands/tile_cost_20260928.json
ROOT=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/group_trace_20260929
ORDER=${GTRACE_ORDER:-"0 4 8 12 16 1 5 9 13 17 2 6 10 14 18 3 7 11 15 19"}

HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
# Environment of run_prc_ppl.sbatch with its defaults (as in run_group_trace_20260929).
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SC_RNG_GRID="" SC_RNG_GRID_ATTN="" SC_RNG_GRID_QK="" SC_RNG_GRID_AV=""
export QUANT_CONFIG=mp
export ACT_SCALES_DIR=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales
export FRONTEND=awq
export CTX=2048
export SQ_ALPHA=0.5
export PPL_MAX_TOKENS=0
unset SC_MP_TRACE
export GTRACE_WINDOWS=${GTRACE_WINDOWS:-4}
export GTRACE_FP_MATMUL=0

for IDX in $ORDER; do
  eval "$(python - "$MANIFEST" "$IDX" <<'PY'
import json, shlex, sys
m = json.load(open(sys.argv[1])); c = m["cells"][int(sys.argv[2])]
for k, v in (("CELL", c["id"]), ("ARM", c["arm"]), ("CFG", c["wrapper"]), ("HYB", c["hybrid"]),
             ("HF", c["hf"]), ("ARCHIVED", c["archived_trace"])):
    print(f"{k}={shlex.quote(str(v))}")
PY
)"
  OUTD=$ROOT/$CELL
  if python -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') else 1)" \
      "$OUTD/check.json" 2>/dev/null; then
    echo "[gtrace-all] $CELL already done, skipped"
    continue
  fi
  mkdir -p "$OUTD"
  export MP_CONFIG_JSON="$CFG" SC_HYBRID_CONFIG_JSON="$HYB" MODEL_PATH="$HF"
  export GTRACE_CELL="$CELL" GTRACE_ARM="$ARM"
  echo "[gtrace-all] $CELL ($ARM) model=$HF config=$CFG hybrid=$HYB"
  t0=$SECONDS
  if GTRACE_OUT="$OUTD/exact" GTRACE_CHECK_TRACE="$OUTD/exact/sc_trace_summary.json" \
       srun python -u benchmark/ppl/group_trace.py \
     && python -u benchmark/ppl/group_trace_check.py --exact "$OUTD/exact" \
          --sc-trace "$OUTD/exact/sc_trace_summary.json" --archived-trace "$ARCHIVED" \
          --verify-tokens --out "$OUTD/check.json"; then
    echo "[gtrace-all] $CELL OK in $((SECONDS - t0)) s"
  else
    echo "[gtrace-all] $CELL FAILED after $((SECONDS - t0)) s"
  fi
done
echo "[gtrace-all] done"
