#!/bin/bash
# TIER-1 TRANSFER PROBE (uniform_sc ceiling cells, target 128) — frozen mp_best
# t128 bundle + AWQ front-end. Companion to run_mpbest_awq.sbatch, which only
# handles the adaptive_mp bundles (those carry a wrapper.json). The deployed
# t128 cell is uniform_sc: sc_int8 (128 cyc) + a FROZEN 10% INT8 hybrid mask,
# with wrapper=null / mp_config_json=null (see the bundle's metadata.json). So it
# runs through QUANT_CONFIG=sc_int8 + the bundle's hybrid_config.json, NOT the MP
# wrapper path. Everything else (AWQ cache reuse, iso-cost caveat, HF-cache
# guards) is identical to the MP launcher.
#
# Reuses the deployed bundle VERBATIM and swaps ONLY the PTQ front-end
# SmoothQuant -> AWQ. Nothing recalibrated. Read realized cost next to PPL.
#
# Usage: FE_MODEL=4B sbatch run_mpbest_awq_uniform.sbatch
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
FE_TARGET=128

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
  *) echo "[mpbest_awq_u] unknown FE_MODEL=$FE_MODEL" >&2; exit 2 ;;
esac

FE_HYBRID="$BUNDLE/hybrid_config.json"
[[ -f "$FE_HYBRID" ]] || { echo "[mpbest_awq_u] missing hybrid_config: $FE_HYBRID" >&2; exit 2; }

# AWQ scale cache MUST already exist (byte-identical to the uniform frontend_awq
# wave and the MP AWQ wave); refuse to recalibrate.
export AWQ_SCALES_DIR="$TURBO/act_scales/awq_scales"
AWQ_CACHE="$AWQ_SCALES_DIR/awq_scales_${HF//\//_}_b4.pt"
[[ -f "$AWQ_CACHE" ]] || { echo "[mpbest_awq_u] MISSING AWQ cache $AWQ_CACHE -- refusing to recalibrate" >&2; exit 2; }

# Refuse to run if weights are not already cached on Turbo (else download to /home).
SAFE_HF="${HF//\//--}"
[[ -d "$HF_HUB_CACHE/models--$SAFE_HF" || -d "$TRANSFORMERS_CACHE/models--$SAFE_HF" ]] || {
  echo "[mpbest_awq_u] $HF not in Turbo cache -- refusing to run (would download into /home)." >&2; exit 2; }

# ---- front-end: the ONLY thing that differs from the deployed cell ----------
export FRONTEND=awq
export AWQ_OBJ_BITS=4
export AWQ_TOK_CAP=128

# ---- deployment knobs: uniform sc_int8 + the bundle's frozen hybrid mask -----
export QUANT_CONFIG=sc_int8            # uniform 128-cycle ceiling (NOT the MP wrapper path)
export SC_HYBRID_CONFIG_JSON="$FE_HYBRID"
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export ACT_SCALES_DIR="$MP_BEST/act_scales"
export MODEL_PATH="$HF"
export PPL_MAX_TOKENS=0
export CTX=2048
export SC_MP_TRACE="$OUT/traces/${FE_MODEL}_t${FE_TARGET}_awq_trace.json"
export SC_MP_TRACE_MODE=summary

echo "[mpbest_awq_u] ================================================="
echo "[mpbest_awq_u] cell=${FE_MODEL} target=128 (uniform_sc) tag=$TAG"
echo "[mpbest_awq_u] hybrid=$FE_HYBRID  frontend=awq (obj_bits=4) cache=$AWQ_CACHE"
python - <<PY
import json
m = json.load(open("$BUNDLE/metadata.json"))
print(f"[mpbest_awq_u] SmoothQuant baseline (SAME config, front-end the only diff):")
print(f"[mpbest_awq_u]   SQ ppl={m['ppl']:.4f}  realized_flop_avg_sl={m['realized_flop_avg_stoc_len']:.3f}"
      f"  fp16={m['fp16_ppl']}  x_fp16={m['ppl']/m['fp16_ppl']:.4f}")
print(f"[mpbest_awq_u]   iso-compute check: AWQ realized_flop_avg_sl must stay ~{m['realized_flop_avg_stoc_len']:.3f}")
PY
echo "[mpbest_awq_u] ================================================="

python -u benchmark/quant/eval_quant.py
