#!/bin/bash
# TIER-1 TRANSFER PROBE — frozen mp_best config + AWQ front-end.
#
# Reuses a deployed `ppl/mp_best` bundle VERBATIM (wrapper / table / hybrid
# schedule / protected channels / gate) and swaps ONLY the PTQ front-end from
# SmoothQuant to AWQ. Nothing is recalibrated and nothing is re-searched.
#
# ⚠ THE ALLOCATOR IS MISMATCHED ON PURPOSE. The bundle's thresholds are ABSOLUTE
# metric values fit on SmoothQuant-smoothed activations, and its budget was
# solved to hit `residual_target_avg_stoc_len` to 7 significant figures. Under
# AWQ the metric distribution moves, rows get reassigned, and NOTHING re-solves
# the budget at runtime — so `realized_flop_avg_sl` may DRIFT OFF TARGET and the
# cell is no longer guaranteed iso-compute. Always read the realized cost next
# to the PPL; a PPL win at a higher realized cost is not a win.
# Interpretation is deliberately asymmetric:
#   AWQ wins anyway  -> strong (it won despite an allocator tuned for its rival)
#   AWQ loses        -> UNINFORMATIVE (front-end vs allocator staleness confounded)
# The clean iso-compute version is Tier 2 (re-fit thresholds under AWQ, keep
# topology), which needs calibrate_mp_thresholds.py taught the FRONTEND switch.
#
# Usage: FE_MODEL=4B FE_TARGET=32 sbatch run_mpbest_awq.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_mpbest_awq/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${FE_MODEL:?FE_MODEL is required (4B|llama8B|14B|30B)}"
: "${FE_TARGET:?FE_TARGET is required (32|96)}"

TAG="mpbest_awq_20260726"
MP_BEST=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best
BUNDLE="$MP_BEST/configs/${FE_MODEL}/target${FE_TARGET}"
TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
OUT="$TURBO/results/${TAG}"
mkdir -p "$OUT/traces"

case "$FE_MODEL" in
  4B)      HF="Qwen/Qwen3-4B-Instruct-2507" ;;
  llama8B) HF="meta-llama/Llama-3.1-8B-Instruct" ;;
  14B)     HF="Qwen/Qwen3-14B" ;;
  30B)     HF="Qwen/Qwen3-30B-A3B-Instruct-2507" ;;
  *) echo "[mpbest_awq] unknown FE_MODEL=$FE_MODEL" >&2; exit 2 ;;
esac

# ---- fail fast, BEFORE burning a GPU hour -----------------------------------
for f in wrapper.json table.json hybrid_config.json runtime.json; do
  [[ -f "$BUNDLE/$f" ]] || { echo "[mpbest_awq] missing $BUNDLE/$f" >&2; exit 2; }
done
# The AWQ scale cache MUST already exist. If it does not, apply_awq_frontend
# would silently spend hours recalibrating AND would write the new cache into
# whatever AWQ_SCALES_DIR points at — a different scale than the uniform wave
# used, which would break comparability with hpca_results/llm/frontend_awq/.
export AWQ_SCALES_DIR="$TURBO/act_scales/awq_scales"
AWQ_CACHE="$AWQ_SCALES_DIR/awq_scales_${HF//\//_}_b4.pt"
[[ -f "$AWQ_CACHE" ]] || {
  echo "[mpbest_awq] MISSING AWQ cache $AWQ_CACHE -- refusing to recalibrate" >&2
  echo "[mpbest_awq] (a fresh calibration would not be byte-identical to the" >&2
  echo "[mpbest_awq]  frontend_awq uniform wave, breaking the comparison)" >&2
  exit 2
}

# ---- front-end: the ONLY thing that differs from the deployed cell ----------
export FRONTEND=awq
export AWQ_OBJ_BITS=4          # must match the uniform wave (cache is keyed _b4)
export AWQ_TOK_CAP=128         # only used on a cache miss, which we refuse above

# ---- deployment knobs, read from the bundle's runtime.json -------------------
# int_bits is NOT forced: all 8 bundles record int_bits=7 in hybrid_config.json,
# and the loader treats the JSON as the authoritative deployment description
# unless SC_HYBRID_FORCE_INT_BITS=1. Leaving FORCE off keeps the schedule exactly
# as deployed. The escape gate (k=2.0, escape_stoc_len=128 on the t32 cells)
# travels inside wrapper.json, so it needs no env.
export QUANT_CONFIG=mp
export MP_CONFIG_JSON="$BUNDLE/wrapper.json"
export SC_HYBRID_CONFIG_JSON="$BUNDLE/hybrid_config.json"
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export ACT_SCALES_DIR="$MP_BEST/act_scales"   # unused on the AWQ branch; kept for provenance
export MODEL_PATH="$HF"

# ---- citability guards (full test split, deploy-matched context) ------------
export PPL_MAX_TOKENS=0
export CTX=2048
# PPL_WINDOW_BATCH_SIZE deliberately UNSET (=1). The V20 30B search used B=8;
# every citable eval is B=1 and this is a citable eval.

export SC_MP_TRACE="$OUT/traces/${FE_MODEL}_t${FE_TARGET}_awq_trace.json"
export SC_MP_TRACE_MODE=summary

echo "[mpbest_awq] ================================================="
echo "[mpbest_awq] cell=${FE_MODEL} target=${FE_TARGET} tag=$TAG"
echo "[mpbest_awq] bundle=$BUNDLE"
echo "[mpbest_awq] frontend=awq (obj_bits=4) cache=$AWQ_CACHE"
echo "[mpbest_awq] SmoothQuant baseline to beat (deployed cell):"
python - <<PY
import json
e = json.load(open("$BUNDLE/eval_summary.json"))
m = json.load(open("$BUNDLE/metadata.json"))
print(f"[mpbest_awq]   SQ ppl={e['ppl']:.4f}  realized_flop_avg_sl={e['realized_flop_avg_stoc_len']:.3f}"
      f"  fp16={m['fp16_ppl']}  x_fp16={e['ppl']/m['fp16_ppl']:.4f}")
print(f"[mpbest_awq]   iso-compute check: AWQ realized_flop_avg_sl must stay ~{e['realized_flop_avg_stoc_len']:.3f}")
PY
echo "[mpbest_awq] ================================================="

python -u benchmark/quant/eval_quant.py
