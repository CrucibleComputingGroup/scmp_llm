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
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/SCMP/scmp_llm
# The env's editable scmp_kernels install still maps to the pre-move path (see SCMP/CLAUDE.md).
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}

: "${KB_MODEL:?KB_MODEL required}"
: "${KB_TARGET:?KB_TARGET required}"
: "${KB_ARM:?KB_ARM required (prc|parent)}"

# KB_PARENT: an alternate parent config dir (wrapper/table/hybrid_config), e.g. the 14B h=20% cells
PARENT=${KB_PARENT:-/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}}
# KB_PRC_OVERRIDE points the prc/prcqk arms at an ALTERNATE wrapper (e.g. the
# marginal-price tables, which shift ONE operator one rung shorter). Without it
# an override was silently ignored and the cell ran the DEPLOYED table under the
# variant's label -- and the 0-bucket guard below cannot catch that, because the
# deployed table has buckets. Same silent-fallback class as the qk-alpha bug.
PRC=${KB_PRC_OVERRIDE:-/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/prc/${KB_MODEL}_t${KB_TARGET}_${KB_TBL:-v7}_prc.json}

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
  # KB_QKALPHA selects the migration strength (default 1.0 = deployed).
  # alpha=1.0 flattens Q entirely and dumps all imbalance into K -- right when
  # |K| spread >> |Q| spread (4B 81 vs 37, 30B 117 vs 47), WRONG when they are
  # equal (llama8B 4.1 vs 4.6). Measured llama8B t96: alpha=0.5 7.5647 vs
  # alpha=1.0 7.5992 (-0.45%). This path used to hardcode the 1.0 filename, so
  # an alpha override was silently ignored and ran 1.0 under the new label.
  QKA="${KB_QKALPHA:-1.0}"
  QKF=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best_after_hpca/qk_scales/${KB_MODEL}_t${KB_TARGET}_qk_alpha${QKA}.json
  [[ -f "$QKF" ]] || QKF=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/qk/${KB_MODEL}_t${KB_TARGET}_qk_alpha${QKA}.json
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
# KB_TAGSUFFIX keeps a variant's trace out of the deployed cell's namespace.
# The archive builder globs `prcppl_<model>_t<target>_<arm>_<jobid>.out` and takes
# the NEWEST by mtime, so an unsuffixed variant run would silently SHADOW the
# archived result and corrupt the next rebuild. Launch variants with a job name
# that does not start with `prcppl_` as well.
TAG="${KB_MODEL}_t${KB_TARGET}_${KB_ARM}${KB_TBL:+_${KB_TBL}}${KB_TAGSUFFIX:+_${KB_TAGSUFFIX}}"

# Environment mirrors the archived parent runtime.json EXACTLY
# (owen bitrev, masks 64, sc_prec 8, halve, INT7 sym hybrid, chunk 128),
# with FRONTEND=awq because the baseline is AWQ + INT7 20%.
HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export SC_OWEN_MODE=bitrev
# KB_MASKS overrides the Owen scramble-mask count (default 64 = the deployed,
# archived setting -- byte-identical when unset). M=128 needs SC_HW_MAX_MASKS
# raised too and is SIMULATION-ONLY at that width (the 64-pattern selector is
# the hardware cap), so any M>64 row must be labeled an ideal/ablation, never
# a deployed cell. Measured 2026-08-07 at the ceiling: 4B -2.63%, llama8B
# -0.26% on pure uniform sc_int8.
export SC_SCRAMBLE_MASKS="${KB_MASKS:-64}"
[[ "${KB_MASKS:-64}" -gt 64 ]] && export SC_HW_MAX_MASKS="${KB_MASKS}"
echo "[prc-ppl] SC_SCRAMBLE_MASKS=${SC_SCRAMBLE_MASKS} HW_MAX=${SC_HW_MAX_MASKS:-64}"
# SC_RNG_GRID=pow2 matches the enable-grid to the stream length (largest
# pow2 <= L, 128 kept at L>=96). Same cycles, different grid -- runtime-free.
export SC_RNG_GRID="${SC_RNG_GRID:-}"
echo "[prc-ppl] SC_RNG_GRID=${SC_RNG_GRID:-<unset, deployed 128>}"
# SC_RNG_GRID_{ATTN,QK,AV} set the enable grid for the ATTENTION PRODUCTS only,
# independently of the linears (SC_RNG_GRID above). Cycle-neutral: stoc_len is
# untouched, so cost and the iso-compute identity are unchanged. Unset =>
# byte-identical to the deployed path (proven by attn_grid_identity.py).
export SC_RNG_GRID_ATTN="${SC_RNG_GRID_ATTN:-}"
export SC_RNG_GRID_QK="${SC_RNG_GRID_QK:-}"
export SC_RNG_GRID_AV="${SC_RNG_GRID_AV:-}"
echo "[prc-ppl] attn grid: ATTN=${SC_RNG_GRID_ATTN:-<unset>} " \
     "QK=${SC_RNG_GRID_QK:-<unset>} AV=${SC_RNG_GRID_AV:-<unset>}"
export QUANT_CONFIG=mp
export MP_CONFIG_JSON="$CFG"
export SC_HYBRID_CONFIG_JSON="${KB_HYBRID:-$PARENT/hybrid_config.json}"
[[ -f "$SC_HYBRID_CONFIG_JSON" ]] || {
  echo "[prc-ppl] missing hybrid config $SC_HYBRID_CONFIG_JSON" >&2; exit 2; }
echo "[prc-ppl] hybrid mask: $SC_HYBRID_CONFIG_JSON"
export ACT_SCALES_DIR=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales
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
