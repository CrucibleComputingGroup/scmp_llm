#!/bin/bash
# FINAL MP algorithm — systematic Phase-2 search at the paper's budget grid.
#
#   Phase 1  sigma (act_global_v9) parent, reprojected by the engine to each budget
#   Phase 2  MEASURED search (mp_v16_refine): paired-window dNLL selection over
#            ladder shape (value/insert/remove), protected length (pc_length),
#            the measured-direction compound (lift_compound), and surrogate
#            proposals. This is the search, not a hardcoded recipe: every move is
#            proposed cheaply and ACCEPTED only on measured A->B replication.
#   Phase 3  additive escape gate k>=2.0 applied to each budget's winner LAST
#            (nothing decreased elsewhere; realized cost floats up and is reported).
#
# Budgets: 32 / 40 / 48 / 64 (halved cycles) — min PPL at each fixed budget.
# Mask: 20% INT dose (mask-blind calibration: the MP table is byte-identical
# across doses, so only the hybrid config swaps).
#
# Usage: final_mp_lane.sh <4B|llama8B|14B|30B> <tag>
set -euo pipefail
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

SHORT="${1:?usage: final_mp_lane.sh <model> <tag>}"
TAG="${2:?usage: final_mp_lane.sh <model> <tag>}"

declare -A HF=(
  [4B]="Qwen/Qwen3-4B-Instruct-2507"
  [llama8B]="meta-llama/Llama-3.1-8B-Instruct"
  [14B]="Qwen/Qwen3-14B"
  [30B]="Qwen/Qwen3-30B-A3B-Instruct-2507"
)
declare -A FP16=( [4B]="10.0445" [llama8B]="7.2130" [14B]="8.6383" [30B]="7.2613" )
hf="${HF[$SHORT]:?unknown model $SHORT}"
safe_hf="${hf//\//_}"

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
SCRATCH=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca
PARENT_TAG=mp_v9_int6_20260716_001908
PARENT_DIR="$TURBO/mp_calib_${PARENT_TAG}"
OUTDIR="$TURBO/mp_final_search_${TAG}/${SHORT}"
LOGDIR="$SCRATCH/logs/_mp_final_search_${TAG}"
FP16_REF_DIR="$TURBO/fp16_val_refs"
mkdir -p "$OUTDIR" "$LOGDIR" "$SCRATCH/tmp" "$FP16_REF_DIR"

# Phase-1 parent. A V18-clean target-specific winner can be supplied for a
# chained run; otherwise retain the V9 calibration fallback.
if [[ -n "${FINAL_PARENT_WRAPPER:-}" ]]; then
  PARENT="$FINAL_PARENT_WRAPPER"
else
  mapfile -t wrappers < <(find "$PARENT_DIR" -maxdepth 1 -type f \
    -name "${safe_hf}__mp_avg32_v9__act_global_v9*_wrapper.json" -print)
  [[ ${#wrappers[@]} -eq 1 ]] || { echo "expected 1 V9 parent, got ${#wrappers[@]}" >&2; exit 21; }
  PARENT="${wrappers[0]}"
fi

# 20% INT dose (Stage 0). Mask-blind => the same MP table is valid at any dose.
DOSE_DIR_4B="$TURBO/hybrid_configs/_hpca_mp_v9_hybdose_4B_20260718_143110"
DOSE_DIR_ALL="$TURBO/hybrid_configs/_hpca_mp_v9_hybdose_all_20260718_183035"
HYBRID="$DOSE_DIR_ALL/${SHORT}_mp_avg32_v9_rank-sc_int8_measured_curve_top0p20.json"
[[ -s "$HYBRID" ]] || HYBRID="$DOSE_DIR_4B/${SHORT}_mp_avg32_v9_rank-sc_int8_measured_curve_top0p20.json"
[[ -s "$PARENT" ]] || { echo "missing parent $PARENT" >&2; exit 22; }
[[ -s "$HYBRID" ]] || { echo "missing 20% hybrid config for $SHORT" >&2; exit 23; }

export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export ACT_SCALES_DIR="$TURBO/act_scales"
export TMPDIR="$SCRATCH/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SC_HYBRID_CONFIG_JSON="$HYBRID"
export SC_HYBRID_INT_BITS=7
export SC_HYBRID_FORCE_INT_BITS=1

# FP16 reference on the SAME validation windows (tier-1 stop trigger only).
FP16_REF="$FP16_REF_DIR/${safe_hf}_validation_ctx2048.json"
if [[ ! -s "$FP16_REF" ]]; then
  ( flock -w 7200 9 || exit 0
    env -u MP_CONFIG_JSON -u SC_HYBRID_CONFIG_JSON \
      python -u benchmark/ppl/fp16_val_reference.py --model-path "$hf" \
        --ctx 2048 --out "$FP16_REF" --skip-if-exists
  ) 9>"$FP16_REF.lock" || echo "[final] fp16 ref failed; continuing" >&2
fi
FP16_ARG=(); [[ -s "$FP16_REF" ]] && FP16_ARG=(--fp16-ref "$FP16_REF")

TARGETS="${FINAL_TARGETS:-32:40:48:64}"
PARTITION_ARG=()
if [[ -n "${FINAL_EXCLUDE_PARTITION_CHECKPOINT:-}" ]]; then
  PARTITION_ARG=(--exclude-partition-checkpoint \
    "$FINAL_EXCLUDE_PARTITION_CHECKPOINT")
fi
QUOTAS="${FINAL_FAMILY_QUOTAS:-macro=2,value_move=1,topology_insert=3,topology_remove=0,surrogate=0,pc_length=2,lift_compound=3,structured_transfer=3,group_exchange=3,threshold_move=0}"
echo "[final] model=$SHORT targets=$TARGETS dose=20% parent=$PARENT"
echo "[final] PHASE 2: measured search"
python -u benchmark/ppl/mp_v16_refine.py \
  --parent-wrapper "$PARENT" \
  --model-path "$hf" \
  --output-dir "$OUTDIR" \
  --targets "$TARGETS" \
  --ctx 2048 \
  --max-sweeps "${FINAL_MAX_SWEEPS:-8}" \
  --patience "${FINAL_PATIENCE:-4}" \
  --advance-total "${FINAL_ADVANCE:-12}" \
  --confirm-top-k "${FINAL_CONFIRM_TOP_K:-3}" \
  --family-quotas "$QUOTAS" \
  --split-fractions 0.33:0.67 \
  --structured-mass-steps "${FINAL_STRUCTURED_MASS_STEPS:-0.10:0.15:0.30}" \
  --surrogate-proposals "${FINAL_SURROGATE_PROPOSALS:-0}" \
  --min-classes 3 --max-classes 8 \
  --min-level "${FINAL_MIN_LEVEL:-16}" --max-level 128 \
  --budget-tol-under 0.35 --budget-tol-over 1.0 \
  --accept-floor 0.002 \
  --pc-lengths "${FINAL_PC_LENGTHS:-112:104:96:95:88:84:80:76:72:68:64}" \
  --fp16-test-ppl "${FP16[$SHORT]}" \
  --ppl-fp16-ratio 1.1 --tier1-ratio-scale 0.90 \
  --test-tokens -1 \
  --walltime-hours "${FINAL_WALLTIME_HOURS:-22.0}" \
  --guard-slack "${FINAL_GUARD_SLACK:-1.10}" \
  --ladder-groups "${FINAL_LADDER_GROUPS:-global}" \
  ${FINAL_SEED_PSL:+--seed-protected-sl "$FINAL_SEED_PSL"} \
  "${PARTITION_ARG[@]}" \
  "${FP16_ARG[@]}"

# ---- PHASE 2b: full-protocol test on each ungated winner -------------------
# The 146-window test eval used to run INSIDE the search job, and the walltime
# guard reserved it before every sweep: 4.9h of a 21h budget on 30B (9.9h once
# the 1.2x slack was applied), which is why 30B never entered sweep 1.  Run it
# here instead, on the frozen winner, so the whole guard budget buys search.
# Consequence: --test-tokens -1 also disables the tier-2 FP16 early stop, which
# never fired in any of the 17 branches of the previous two waves.
echo "[final] PHASE 2b: full-protocol test on each winner"
IFS=':' read -r -a _test_targets <<< "$TARGETS"
for _t in "${_test_targets[@]}"; do
  branch="$OUTDIR/target$(printf '%.3f' "$_t" | tr '.' 'p')"
  win="$branch/best_wrapper.json"
  [[ -s "$win" ]] || { echo "[final]   $(basename "$branch"): no winner"; continue; }
  python -u benchmark/ppl/mp_ladder_refine.py \
    --parent-wrapper "$win" --model-path "$hf" \
    --output-dir "$branch/eval_test" \
    --split test --max-rounds 0 --max-tokens 0 --ctx 2048 --alpha 0.5
  python - "$branch/eval_test/summary.json" "$SHORT" "$(basename "$branch")" <<'PY'
import json, sys
fp16 = {"4B":10.0445,"llama8B":7.2130,"14B":8.6383,"30B":7.2613}[sys.argv[2]]
s = json.load(open(sys.argv[1])); p = s["best_ppl"]
print(f"[final] {sys.argv[2]} {sys.argv[3]} winner: ppl={p:.4f} "
      f"realized={s['best_trace_flop_avg_stoc_len']:.3f} ratio={p/fp16:.4f}x "
      f"{'PASS<=1.1x' if p <= 1.1*fp16 else 'above 1.1x'}")
PY
done

# ---- PHASE 3: additive escape gate on each budget's winner (applied LAST) ----
GATE_K="${FINAL_GATE_K:-2.0}"
if [[ "${FINAL_SKIP_GATE:-0}" == "1" ]]; then
  # No-search baseline cells compare against the UNGATED search winner from
  # PHASE 2b, so the gate would only double their full-test cost.
  echo "[final] PHASE 3 skipped (FINAL_SKIP_GATE=1)"
  echo "[final done] $SHORT -> $OUTDIR"
  exit 0
fi
echo "[final] PHASE 3: additive escape gate k=$GATE_K on each winner"
# Only THIS job's targets.  $OUTDIR is shared per-model, so globbing target*
# made every concurrent single-target job re-run the gate for all targets
# (16 full-test evals where 4 are needed, three jobs writing the same dir).
IFS=':' read -r -a _gate_targets <<< "$TARGETS"
for _t in "${_gate_targets[@]}"; do
  branch="$OUTDIR/target$(printf '%.3f' "$_t" | tr '.' 'p')"
  [[ -d "$branch" ]] || { echo "[final]   $(basename "$branch"): no branch dir"; continue; }
  win="$branch/best_wrapper.json"
  [[ -s "$win" ]] || { echo "[final]   $(basename "$branch"): no winner"; continue; }
  gated="$branch/best_wrapper_gate${GATE_K}.json"
  python - "$win" "$gated" "$GATE_K" <<'PY'
import json, sys
w = json.load(open(sys.argv[1]))
w["escape_gate_k"] = float(sys.argv[3])
w["escape_stoc_len"] = 128          # additive: nothing else is decreased
json.dump(w, open(sys.argv[2], "w"), indent=1)
print(f"  gate wrapper -> {sys.argv[2]}")
PY
  python -u benchmark/ppl/mp_ladder_refine.py \
    --parent-wrapper "$gated" --model-path "$hf" \
    --output-dir "$branch/eval_gate${GATE_K}" \
    --split test --max-rounds 0 --max-tokens 0 --ctx 2048 --alpha 0.5
  python - "$branch/eval_gate${GATE_K}/summary.json" "$SHORT" "$(basename "$branch")" <<'PY'
import json, sys
fp16 = {"4B":10.0445,"llama8B":7.2130,"14B":8.6383,"30B":7.2613}[sys.argv[2]]
s = json.load(open(sys.argv[1])); p = s["best_ppl"]
print(f"[final] {sys.argv[2]} {sys.argv[3]} +gate: ppl={p:.4f} "
      f"realized={s['best_trace_flop_avg_stoc_len']:.3f} ratio={p/fp16:.4f}x "
      f"{'PASS<=1.1x' if p <= 1.1*fp16 else 'above 1.1x'}")
PY
done
echo "[final done] $SHORT -> $OUTDIR"
