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

# ---- HF caches MUST live on Turbo, never /home (quota) ----------------------
# This driver calls eval_quant.py directly, so unlike `bash hpca` it does NOT
# inherit hpca's cache exports and nothing in ~/.bashrc sets them. Omitting them
# cost a full wave: the Qwen cells silently re-downloaded weights into
# ~/.cache/huggingface (634M -> 2.4G in three minutes, against an 80 GiB /home
# quota with two 30B cells inbound at ~68G each) and both llama8B cells died on
# a gated-repo 401 because the HF token lives at $HF_HOME/token.
# Both cache LAYOUTS are needed and are not interchangeable here:
#   $HF_CACHE/hub/models--*        <- Qwen 4B/14B/30B (current layout, HF_HUB_CACHE)
#   $HF_CACHE/models--*            <- Llama-3.1-8B    (legacy layout, TRANSFORMERS_CACHE)
# These four exports mirror `hpca` lines 244-247 verbatim.
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

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

# ---- config selection -------------------------------------------------------
# Default: the deployed mp_best bundle. OVERRIDE for 14B t96 so all 8 cells run
# a 20% INT mask, making the mask a CONSTANT across the wave and leaving the
# front-end as the only variable. mp_best kept the 10% mask for this one cell
# because 20% lost there (8.6848 @ 96.16 vs 8.6789 @ 95.779) — consistent with
# int_ablation, where 14B saturates at 10% and mildly regresses at 20%. The
# baseline to beat is therefore the 20% SmoothQuant number, NOT the deployed
# 10% one: same config, front-end the only difference.
W20="$TURBO/mp_calib_mp_avg96_hyb20_20260724"
H20="$TURBO/hybrid_configs/_hpca_mp_avg96_hyb20_20260724"
V3SUF="mp_avg96f_burst128__act_global_v3_pc0.01x128_act_collapse_comp_pcovdown_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.20"
FE_WRAPPER="$BUNDLE/wrapper.json"
FE_HYBRID="$BUNDLE/hybrid_config.json"
FE_CONFIG_SRC="mp_best bundle (deployed)"
if [[ "$FE_MODEL" == "14B" && "$FE_TARGET" == "96" ]]; then
  FE_WRAPPER="$W20/Qwen_Qwen3-14B__${V3SUF}_wrapper.json"
  FE_HYBRID="$H20/14B_mp_avg96f_burst128_rank-sc_int8_measured_curve_top0p20.json"
  FE_SQ_PPL=8.6848
  FE_SQ_CYC=96.16
  FE_CONFIG_SRC="mp_avg96_hyb20_20260724 (20% mask; mp_best kept 10% here)"
fi

# ---- fail fast, BEFORE burning a GPU hour -----------------------------------
for f in "$FE_WRAPPER" "$FE_HYBRID"; do
  [[ -f "$f" ]] || { echo "[mpbest_awq] missing config: $f" >&2; exit 2; }
done
# The wrapper's threshold_table_path must resolve — mp_best wrappers use a
# relative "table.json" next to them, the 20% wrapper an absolute Turbo path.
python - <<PY || exit 2
import json, os, sys
w = json.load(open("$FE_WRAPPER"))
p = w["threshold_table_path"]
p = p if os.path.isabs(p) else os.path.join(os.path.dirname("$FE_WRAPPER"), p)
if not os.path.isfile(p):
    sys.exit(f"[mpbest_awq] wrapper table does not resolve: {p}")
print(f"[mpbest_awq] table={p}")
print(f"[mpbest_awq] levels={w['stoc_len_levels']} gate={w.get('escape_gate_k', 'none')}")
PY
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
# Refuse to run if the model weights are not already cached on Turbo — that is
# the signal that the HF env is wrong and the job is about to download ~18-68G
# into /home. Checks BOTH layouts (see the HF cache note above).
SAFE_HF="${HF//\//--}"
[[ -d "$HF_HUB_CACHE/models--$SAFE_HF" || -d "$TRANSFORMERS_CACHE/models--$SAFE_HF" ]] || {
  echo "[mpbest_awq] $HF is NOT in the Turbo cache:" >&2
  echo "[mpbest_awq]   $HF_HUB_CACHE/models--$SAFE_HF" >&2
  echo "[mpbest_awq]   $TRANSFORMERS_CACHE/models--$SAFE_HF" >&2
  echo "[mpbest_awq] refusing to run -- it would download into /home (quota)." >&2
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
export MP_CONFIG_JSON="$FE_WRAPPER"
export SC_HYBRID_CONFIG_JSON="$FE_HYBRID"
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
echo "[mpbest_awq] config_src=$FE_CONFIG_SRC"
echo "[mpbest_awq] wrapper=$FE_WRAPPER"
echo "[mpbest_awq] hybrid=$FE_HYBRID"
echo "[mpbest_awq] frontend=awq (obj_bits=4) cache=$AWQ_CACHE"
python - <<PY
import json
sq_ppl, sq_cyc = "${FE_SQ_PPL:-}", "${FE_SQ_CYC:-}"
if sq_ppl and sq_cyc:
    sq_ppl, sq_cyc = float(sq_ppl), float(sq_cyc)
else:
    e = json.load(open("$BUNDLE/eval_summary.json"))
    sq_ppl, sq_cyc = float(e["ppl"]), float(e["realized_flop_avg_stoc_len"])
m = json.load(open("$BUNDLE/metadata.json"))
print(f"[mpbest_awq] SmoothQuant baseline (SAME config, front-end the only diff):")
print(f"[mpbest_awq]   SQ ppl={sq_ppl:.4f}  realized_flop_avg_sl={sq_cyc:.3f}"
      f"  fp16={m['fp16_ppl']}  x_fp16={sq_ppl/m['fp16_ppl']:.4f}")
print(f"[mpbest_awq]   iso-compute check: AWQ realized_flop_avg_sl must stay ~{sq_cyc:.3f}")
PY
echo "[mpbest_awq] ================================================="

python -u benchmark/quant/eval_quant.py
