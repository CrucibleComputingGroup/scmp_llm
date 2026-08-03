#!/bin/bash
# PHASE 3 (K-bands) — evaluate one bundle full-protocol.
#
# Drives the SAME eval as every citable cell (benchmark/quant/eval_quant.py,
# full wikitext-2 test, ctx 2048, PPL_MAX_TOKENS=0). The ONLY thing that varies
# between the sanity rungs is which bundle KB_BUNDLE points at:
#
#   S0  the deployed mp_best parent, unchanged.
#       Must reproduce the archived PPL EXACTLY. This is what proves the
#       escape-gate ladder fix is a no-op on shipped configs (every deployed
#       table has zero per-bucket ladders, so the corrected lookup returns the
#       same global list) -- if S0 moves at all, the fix was not a no-op and
#       nothing downstream is interpretable.
#
#   S1  the identity child: same bundle + a k_bands section whose every band
#       carries the parent ladder. Exercises the whole band path -- K-split,
#       per-band gather, per-band tracker pricing -- while computing the same
#       math. Must match S0 to fp32 ACCUMULATION REGROUPING only.
#       NOT bit-identical by construction: the chunked kernel keeps one fp32
#       accumulator across all chunks of a call, and a band split restarts it,
#       so the same addends are summed in a different order.
#
#   S2+ a real allocation. Iso-compute is enforced at table LOAD (the per-rung
#       identity), so a cell that starts is a cell whose budget already checks.
#
# Usage:
#   KB_MODEL=4B KB_BUNDLE=<dir> KB_LABEL=s0_parent sbatch run_kband_cell.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${KB_MODEL:?KB_MODEL is required (4B|llama8B|14B|30B)}"
: "${KB_BUNDLE:?KB_BUNDLE is required (bundle dir with wrapper.json)}"
: "${KB_LABEL:?KB_LABEL is required (e.g. s0_parent, s1_identity)}"

# ---- HF caches MUST live on Turbo, never /home (quota) ----------------------
# eval_quant.py is called directly, so nothing inherits hpca's cache exports.
# Both layouts are needed and are not interchangeable:
#   $HF_CACHE/hub/models--*  <- Qwen 4B/14B/30B (HF_HUB_CACHE)
#   $HF_CACHE/models--*      <- Llama-3.1-8B    (TRANSFORMERS_CACHE)
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

TAG="kbands_20260801"
MP_BEST=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best
TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
OUT="$TURBO/kbands/$TAG"
mkdir -p "$OUT/traces" "$OUT/results"

# Model id comes from the PARENT TABLE, never a hardcoded string. The deployed
# 30B is Qwen3-30B-A3B-Instruct-2507; hardcoding "Qwen3-30B-A3B" selects a
# DIFFERENT model and a different act_scales cache -- the config substitution
# this project forbids inside a comparison. It failed loudly here (missing
# cache) but would have been silent had that cache existed.
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$KB_BUNDLE/table.json")
[[ -n "$HF" ]] || { echo "[kbands] cannot read model_path from $KB_BUNDLE/table.json" >&2; exit 2; }
echo "[kbands] model_path (from parent table) = $HF"

WRAPPER="$KB_BUNDLE/wrapper.json"
HYBRID="$KB_BUNDLE/hybrid_config.json"
for f in "$WRAPPER" "$HYBRID"; do
  [[ -f "$f" ]] || { echo "[kbands] missing config: $f" >&2; exit 2; }
done

# Resolve the table and PRINT the band geometry. A child whose k_bands section
# failed to load would run as a silent per-row cell and still look in-budget,
# so the geometry is echoed into the log before any GPU work starts.
python - <<PY || exit 2
import json, os, sys
sys.path.insert(0, "kernels")
from scmp_kernels.mp.config import residual_chunk_widths
w = json.load(open("$WRAPPER"))
p = w["threshold_table_path"]
p = p if os.path.isabs(p) else os.path.join(os.path.dirname("$WRAPPER"), p)
if not os.path.isfile(p):
    sys.exit(f"[kbands] wrapper table does not resolve: {p}")
t = json.load(open(p))
print(f"[kbands] table={p}")
print(f"[kbands] levels={w['stoc_len_levels']} gate={w.get('escape_gate_k','none')}")
kb = t.get("k_bands")
if not kb:
    print("[kbands] k_bands: ABSENT -> per-row parent (expected for S0)")
else:
    n = kb["n_bands"]
    print(f"[kbands] k_bands: n_bands={n} note={t.get('k_bands_note','')}")
    for op, R in sorted(kb["residual_width"].items()):
        widths = residual_chunk_widths(R, kb.get("chunk_d", 128))
        first = next(k for k in kb["chunk_bands"] if k.startswith(op + ":"))
        bw = [0] * n
        for c, b in enumerate(kb["chunk_bands"][first]):
            bw[b] += widths[c]
        lad = kb["ladders"].get(f"{op}:t0:l0")
        print(f"[kbands]   {op:11} R={R:6} band_widths={bw} l0_ladders={lad}")
PY

# Refuse to run if the weights are not already on Turbo -- that is the signal
# the HF env is wrong and the job is about to download into /home (quota).
SAFE_HF="${HF//\//--}"
[[ -d "$HF_HUB_CACHE/models--$SAFE_HF" || -d "$TRANSFORMERS_CACHE/models--$SAFE_HF" ]] || {
  echo "[kbands] $HF is NOT in the Turbo cache -- refusing to run." >&2
  exit 2
}

# ---- deployment knobs, exactly as the parent cell ran them ------------------
# int_bits is not forced: hybrid_config.json is authoritative (int_bits=7 on
# every deployed t32/t48 bundle). The escape gate travels inside wrapper.json.
export QUANT_CONFIG=mp
export MP_CONFIG_JSON="$WRAPPER"
export SC_HYBRID_CONFIG_JSON="$HYBRID"
# qk/av contracted-dim rebalance, when this cell tests one. Unset = baseline.
if [[ -n "${KB_QK_SMOOTH:-}" ]]; then
  [[ -f "$KB_QK_SMOOTH" ]] || { echo "[kbands] missing qk scales: $KB_QK_SMOOTH" >&2; exit 2; }
  export SC_ATTN_SMOOTH_JSON="$KB_QK_SMOOTH"
  echo "[kbands] qk rebalance ON: $KB_QK_SMOOTH"
fi
# PTQ front-end. The project's real baseline is AWQ + INT7 20% mask, not
# SmoothQuant -- AWQ is already measured full-protocol on these exact bundles
# (frontend_awq/mpbest_awq_vs_smoothquant.csv) and is better on 23/24 cells.
# AWQ reaches only SCLinear; the qk rebalance reaches only attention, so they
# are disjoint and should compose.
if [[ "${KB_FRONTEND:-smoothquant}" == "awq" ]]; then
  export FRONTEND=awq
  export AWQ_OBJ_BITS=4
  export AWQ_SCALES_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/act_scales/awq_scales
  SAFE="${HF//\//_}"
  AWQ_CACHE="$AWQ_SCALES_DIR/awq_scales_${SAFE}_b4.pt"
  [[ -f "$AWQ_CACHE" ]] || {
    echo "[kbands] MISSING AWQ cache $AWQ_CACHE -- refusing to recalibrate," >&2
    echo "[kbands] which would not be byte-identical to the frontend_awq wave." >&2
    exit 2; }
  echo "[kbands] front-end: AWQ (cache $AWQ_CACHE)"
else
  echo "[kbands] front-end: SmoothQuant (default)"
fi
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export ACT_SCALES_DIR="$MP_BEST/act_scales"
export MODEL_PATH="$HF"

# ---- citability guards (full test split, deploy-matched context) ------------
export PPL_MAX_TOKENS=0
export CTX=2048
# PPL_WINDOW_BATCH_SIZE deliberately UNSET (=1); every citable eval is B=1.

export SC_MP_TRACE="$OUT/traces/${KB_MODEL}_${KB_LABEL}_trace.json"
export SC_MP_TRACE_MODE=summary

echo "[kbands] ================================================="
echo "[kbands] model=$KB_MODEL label=$KB_LABEL tag=$TAG"
echo "[kbands] bundle=$KB_BUNDLE"
echo "[kbands] job=$SLURM_JOB_ID host=$(hostname)"
echo "[kbands] ================================================="

# eval_quant.py has no summary-file hook: it prints one `[RESULT] ...` line and
# stamps ppl / eval_tokens into the trace header. Capture the line verbatim so
# the sanity rungs can be diffed without re-reading multi-hour logs.
RESULTS="$OUT/results/${KB_MODEL}_${KB_LABEL}.result"
srun python -u benchmark/quant/eval_quant.py 2>&1 | tee "$OUT/results/${KB_MODEL}_${KB_LABEL}.log"
rc=${PIPESTATUS[0]}

grep -h '^\[RESULT\]' "$OUT/results/${KB_MODEL}_${KB_LABEL}.log" > "$RESULTS" || true
echo "[kbands] eval_quant.py exit=$rc"
echo "[kbands] ---- $KB_LABEL ----"
cat "$RESULTS" || echo "[kbands] NO [RESULT] LINE -- the eval did not finish"
exit $rc
