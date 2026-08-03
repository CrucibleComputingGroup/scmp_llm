#!/bin/bash
# FAIR UNIFORM COMPARATOR on AWQ — uniform SC at the iso-compute budget + the
# DEPLOYED mp_best cell's OWN hybrid mask, front-end = AWQ. This is the
# uniform_hybrid/uni_hyb20 comparator re-run on AWQ so "MP beats uniform at
# iso-compute" stays front-end-matched once MP deploys on AWQ.
#
# Reuses each MP cell's frozen hybrid_config.json (so the mask is byte-identical
# to the deployed cell: 20% INT7 for the bulk, 10% INT7 for 14B t64/t96), on the
# uniform budget aligned to the MP target. No MP wrapper (allocator OFF) — this
# isolates the allocator, exactly like uniform_hybrid vs mp_final.
#
#   MP target -> uniform config (iso-compute twin, halved cycles):
#     32 -> sc_int6   48 -> sc_avg96   64 -> sc_int7   96 -> sc_avg192
#   (t40 has no native uniform twin; t128 IS uniform_sc = the deployment cell.)
#
# Usage: FE_MODEL=4B FE_TARGET=32 sbatch run_uniform_awq_comparator.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_uniform_awq/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${FE_MODEL:?FE_MODEL is required (4B|llama8B|14B|30B)}"
: "${FE_TARGET:?FE_TARGET is required (32|48|64|96)}"
case "$FE_TARGET" in
  32) SC_CFG=sc_int6 ;;
  48) SC_CFG=sc_avg96 ;;
  64) SC_CFG=sc_int7 ;;
  96) SC_CFG=sc_avg192 ;;
  *) echo "[uac] FE_TARGET=$FE_TARGET has no uniform twin (use 32|48|64|96)" >&2; exit 2 ;;
esac

HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE"; export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"; export HF_HUB_CACHE="$HF_HOME/hub"

TAG="uniform_awq_20260726"
MP_BEST=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best
BUNDLE="$MP_BEST/configs/${FE_MODEL}/target${FE_TARGET}"
TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
OUT="$TURBO/results/${TAG}"; mkdir -p "$OUT/traces"

case "$FE_MODEL" in
  4B)      HF="Qwen/Qwen3-4B-Instruct-2507" ;;
  llama8B) HF="meta-llama/Llama-3.1-8B-Instruct" ;;
  14B)     HF="Qwen/Qwen3-14B" ;;
  30B)     HF="Qwen/Qwen3-30B-A3B-Instruct-2507" ;;
  *) echo "[uac] unknown FE_MODEL=$FE_MODEL" >&2; exit 2 ;;
esac

FE_HYBRID="$BUNDLE/hybrid_config.json"     # the DEPLOYED cell's own mask
[[ -f "$FE_HYBRID" ]] || { echo "[uac] missing hybrid_config: $FE_HYBRID" >&2; exit 2; }

export AWQ_SCALES_DIR="$TURBO/act_scales/awq_scales"
AWQ_CACHE="$AWQ_SCALES_DIR/awq_scales_${HF//\//_}_b4.pt"
[[ -f "$AWQ_CACHE" ]] || { echo "[uac] MISSING AWQ cache $AWQ_CACHE -- refusing to recalibrate" >&2; exit 2; }
SAFE_HF="${HF//\//--}"
[[ -d "$HF_HUB_CACHE/models--$SAFE_HF" || -d "$TRANSFORMERS_CACHE/models--$SAFE_HF" ]] || {
  echo "[uac] $HF not in Turbo cache -- refusing to run (would download into /home)." >&2; exit 2; }

export FRONTEND=awq
export AWQ_OBJ_BITS=4
export AWQ_TOK_CAP=128
export QUANT_CONFIG="$SC_CFG"              # uniform SC (NO MP wrapper -> allocator OFF)
export SC_HYBRID_CONFIG_JSON="$FE_HYBRID"
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export ACT_SCALES_DIR="$MP_BEST/act_scales"
export MODEL_PATH="$HF"
export PPL_MAX_TOKENS=0
export CTX=2048
export SC_MP_TRACE="$OUT/traces/${FE_MODEL}_t${FE_TARGET}_uac_trace.json"
export SC_MP_TRACE_MODE=summary

echo "[uac] ============================================================"
echo "[uac] cell=${FE_MODEL} target=${FE_TARGET} uniform_cfg=$SC_CFG tag=$TAG"
echo "[uac] hybrid(mask)=$FE_HYBRID  frontend=awq(obj_bits=4)  allocator=OFF"
python - <<PY
import json
m = json.load(open("$BUNDLE/metadata.json"))
print(f"[uac] (deployed MP cell for reference: ppl={m['ppl']:.4f} @ {m['realized_flop_avg_stoc_len']:.2f}, "
      f"mask int_bits={m.get('hybrid_int_bits')} frac={m.get('hybrid_int_fraction'):.3f})")
PY
echo "[uac] ============================================================"

python -u benchmark/quant/eval_quant.py
