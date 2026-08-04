#!/bin/bash
# Full-protocol PPL for one per-(row, chunk) cell, or its identity control.
#
# The comparison this feeds is S1 -> S2, never S0 -> S2: on 4B t48 with K-bands,
# 83% of the naive parent-to-result delta was numerics rather than allocation.
# So each (model, target) needs BOTH:
#   KB_ARM=prc      the allocated child  (per_row_chunk table)
#   KB_ARM=parent   the parent re-run in THIS environment  (wrapper.json)
# Comparing the child against the ARCHIVED parent number instead would fold in
# every environment difference since that run.
#
#   KB_MODEL=4B KB_TARGET=32 KB_ARM=prc sbatch run_prc_ppl.sbatch
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

: "${KB_MODEL:?KB_MODEL required}"
: "${KB_TARGET:?KB_TARGET required}"
: "${KB_ARM:?KB_ARM required (prc|parent)}"

PARENT=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}
PRC=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/prc/${KB_MODEL}_t${KB_TARGET}_${KB_TBL:-v7}_prc.json

# Model id comes from the PARENT TABLE, never a hardcoded string.
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$PARENT/table.json")
[[ -n "$HF" ]] || { echo "[prc-ppl] cannot read model_path" >&2; exit 2; }

if [[ "$KB_ARM" == "prc" ]]; then
  CFG="$PRC"
  [[ -f "$CFG" ]] || { echo "[prc-ppl] missing per-(row,chunk) table $CFG" >&2; exit 2; }
  # Refuse to run a "prc" cell whose table has no per_row_chunk section — that
  # would silently be a parent re-run recorded under the child's name.
  python - "$CFG" <<'PY' || exit 2
import json, sys, pathlib
w = json.load(open(sys.argv[1]))
t = w.get("threshold_table_path")
if not t or not pathlib.Path(t).is_file():
    sys.exit(f"[prc-ppl] threshold_table_path missing: {t!r}")
n = len(((json.load(open(t)).get("per_row_chunk") or {}).get("buckets")) or {})
if n == 0:
    sys.exit("[prc-ppl] table carries NO per_row_chunk buckets; this cell would "
             "silently run per-ROW dispatch and be recorded as the child")
print(f"[prc-ppl] per_row_chunk buckets: {n}")
PY
elif [[ "$KB_ARM" == "parent" ]]; then
  CFG="$PARENT/wrapper.json"
elif [[ "$KB_ARM" == "qk" || "$KB_ARM" == "prcqk" ]]; then
  # Combined arms. qk is a score-invariant transform on ATTENTION; per-(row,
  # chunk) is an allocation on LINEARS. Disjoint operators, so they compose --
  # qk+bands previously measured -2.54% against -2.51% predicted from summing
  # them. `qk` is the control that says how much of any prcqk win is qk alone.
  # Archive scales first, then the ones calibrated in this loop (14B t32/t48 and
  # llama8B t32 exist only there -- the archive has no 14B qk because qk
  # REGRESSED 14B at t48; at t32 it is untested, and qk pays more at tighter
  # budgets, so it is a real question rather than a known outcome.
  QKF=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best_after_hpca/qk_scales/${KB_MODEL}_t${KB_TARGET}_qk_alpha1.0.json
  [[ -f "$QKF" ]] || QKF=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/qk/${KB_MODEL}_t${KB_TARGET}_qk_alpha1.0.json
  # A named-but-missing qk path once fell back to a no-qk run recorded under a
  # "qk" label. Refuse instead.
  [[ -f "$QKF" ]] || { echo "[prc-ppl] qk scales missing: $QKF" >&2; exit 2; }
  export SC_ATTN_SMOOTH_JSON="$QKF"
  if [[ "$KB_ARM" == "prcqk" ]]; then
    CFG="$PRC"
    [[ -f "$CFG" ]] || { echo "[prc-ppl] missing $CFG" >&2; exit 2; }
    python - "$CFG" <<'PY2' || exit 2
import json, sys, pathlib
w = json.load(open(sys.argv[1]))
t = w.get("threshold_table_path")
if not t or not pathlib.Path(t).is_file():
    sys.exit(f"[prc-ppl] threshold_table_path missing: {t!r}")
n = len(((json.load(open(t)).get("per_row_chunk") or {}).get("buckets")) or {})
if n == 0:
    sys.exit("[prc-ppl] table carries NO per_row_chunk buckets")
print(f"[prc-ppl] per_row_chunk buckets: {n}")
PY2
  else
    CFG="$PARENT/wrapper.json"
  fi
  echo "[prc-ppl] qk scales: $QKF"
else
  echo "[prc-ppl] KB_ARM must be prc|parent|qk|prcqk, got '$KB_ARM'" >&2; exit 2
fi

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
OUT="$TURBO/kbands/kbands_20260801/ppl"
mkdir -p "$OUT"
# The table variant is part of the TAG. Without it an iso-cost re-run
# OVERWRITES the original prc trace and destroys the evidence for the first
# result -- including the overspend that motivated the re-run.
TAG="${KB_MODEL}_t${KB_TARGET}_${KB_ARM}${KB_TBL:+_${KB_TBL}}"

# Environment mirrors the archived parent runtime.json EXACTLY
# (owen bitrev, masks 64, sc_prec 8, halve, INT7 sym hybrid, chunk 128),
# with FRONTEND=awq because the baseline is AWQ + INT7 20%.
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64
export QUANT_CONFIG=mp
export MP_CONFIG_JSON="$CFG"
export SC_HYBRID_CONFIG_JSON="$PARENT/hybrid_config.json"
export ACT_SCALES_DIR=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/act_scales
export FRONTEND="${KB_FRONTEND:-awq}"
export MODEL_PATH="$HF"
export CTX=2048
export SQ_ALPHA=0.5
# FULL PROTOCOL. Any truncation makes the row non-citable; this project takes
# no truncated evals, smokes included.
export PPL_MAX_TOKENS=0
export SC_MP_TRACE="$OUT/${TAG}_trace.json"

echo "[prc-ppl] $TAG  model=$HF"
echo "[prc-ppl] config=$CFG"
echo "[prc-ppl] frontend=$FRONTEND  ctx=$CTX  PPL_MAX_TOKENS=$PPL_MAX_TOKENS"
srun python -u benchmark/quant/eval_quant.py
