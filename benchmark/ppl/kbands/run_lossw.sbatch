#!/bin/bash
# P1 / P2 pooled loss-weighted sigma objective -- see
# benchmark/ppl/kbands/PREREG_LOSS_WEIGHTED_OBJECTIVE.md
#
#   KB_STAGE=calib KB_MODEL=4B KB_ARM=p1 sbatch run_lossw.sbatch
#   KB_STAGE=eval  KB_MODEL=4B KB_ARM=p1 sbatch --dependency=afterok:<calib> run_lossw.sbatch
#
# ARMS (identical in every respect except the cross-layer weighting):
#   control  --cross-layer-weight uniform                              (what every
#            deployed mp_best cell runs: an UNWEIGHTED sigma)
#   p1       --cross-layer-weight measured --measure-pool attn_linear  (2 weights)
#   p2       --cross-layer-weight measured --measure-pool operator     (9 weights)
#   p3       --cross-layer-weight measured --measure-pool none         (36; expected
#            to LOSE -- 22% of raw dLoss entries are negative)
#
# FRONT-END. Calibration is SmoothQuant because calibrate_mp_thresholds.py has
# no AWQ path at all; evaluation is AWQ because that is the post-HPCA baseline
# and what the rest of this loop runs. That split is NOT introduced here -- every
# deployed cell is a SmoothQuant-calibrated table measured under AWQ. Both arms
# share it identically, so it cannot move the control-vs-P1 delta.
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
cd /home/allenjin/Projects/scmp_llm

: "${KB_STAGE:?KB_STAGE required (calib|eval)}"
: "${KB_MODEL:?KB_MODEL required}"
: "${KB_ARM:?KB_ARM required (control|p1|p2|p3|b1|b2|pow2)}"
KB_TARGET="${KB_TARGET:-32}"

RES=/home/allenjin/Projects/hpca_results/llm
PARENT="$RES/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}"
TRACE="$RES/uniform/traces/${KB_MODEL}_sc_int7_trace.json"
OUT=/nfs/turbo/coe-nbleier/allenjin/hpca/lossw/lossw_20260804
mkdir -p "$OUT"
TAG="${KB_MODEL}_t${KB_TARGET}_${KB_ARM}"
TABLE="$OUT/${TAG}.json"
WRAPPER="$OUT/${TAG}_wrapper.json"

# Model id comes from the PARENT TABLE, never a hardcoded string.
HF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$PARENT/table.json")
[[ -n "$HF" ]] || { echo "[lossw] cannot read model_path" >&2; exit 2; }

LADDER=""
case "$KB_ARM" in
  control) WEIGHT_FLAGS="" ;;
  p1)      WEIGHT_FLAGS="--cross-layer-weight measured --measure-pool attn_linear" ;;
  p2)      WEIGHT_FLAGS="--cross-layer-weight measured --measure-pool operator" ;;
  p3)      WEIGHT_FLAGS="--cross-layer-weight measured --measure-pool none" ;;
  b1)      WEIGHT_FLAGS="--grad-source block" ;;
  b2)      WEIGHT_FLAGS="--grad-source block --grad-row-norm call --grad-g-pow 1 --grad-s-pow 1" ;;
  pow2)    WEIGHT_FLAGS="" ; LADDER="128,64,32,16" ;;
  *) echo "[lossw] bad KB_ARM '$KB_ARM'" >&2; exit 2 ;;
esac

HF_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets" HF_HUB_CACHE="$HF_CACHE/hub"
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64
# b1 holds one block's autograd graph at a time; fragmentation is the
# difference between fitting and not on the 95 GiB card.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ACT_SCALES_DIR="$RES/ppl/mp_best/act_scales"

# ----------------------------------------------------------------- calibrate
if [[ "$KB_STAGE" == "calib" ]]; then
  echo "[lossw] calibrating $TAG  arm=$KB_ARM  model=$HF"
  echo "[lossw] weight flags: ${WEIGHT_FLAGS:-<none, uniform>}"
  srun python -u benchmark/ppl/calibrate_mp_thresholds.py \
    --model_path "$HF" \
    --mp_levels "${LADDER:-96,64,48,32,24,16}" \
    --budget_ratio 0.25 --budget_ref_stoc_len 128 \
    --sc_prec 8 --halve 1 --seed 0 --ctx_len 2048 \
    --budget-scope global --budget-weight macs --mac-weights-trace "$TRACE" \
    --calib-smoothquant \
    --protect-channel-frac 0.01 --protect-channel-stoc-len 112 \
    --protect-channel-metric act_collapse \
    --protect-channel-frac-overrides down_proj:0.06,up_proj:0.03,gate_proj:0.03 \
    --protect-compensate-budget --argmin-pricing macs --metric-select auto \
    --refine sigma --objective delta_sigma2 \
    $WEIGHT_FLAGS \
    --output_json "$TABLE" || exit $?

  # GATE. A weighting flag that silently no-ops produces a table byte-identical
  # to the control, the eval completes, and the arm is recorded as "the idea
  # does not work". Refuse to hand such a table to a multi-hour eval.
  python - "$TABLE" "$KB_ARM" <<'PY' || exit 2
import json, sys
t, arm = json.load(open(sys.argv[1])), sys.argv[2]
m = t.get("measured") or {}
w = m.get("normalized_weights") or {}
if arm in ("b1", "b2"):
    if t.get("grad_source") != "block":
        sys.exit(f"[gate] b1 expected grad_source=block, got {t.get('grad_source')!r} "
                 "-- the flag no-opped and this would be an unweighted control")
    if arm == "b2" and t.get("grad_row_norm") != "call":
        sys.exit(f"[gate] b2 expected grad_row_norm=call, got "
                 f"{t.get('grad_row_norm')!r} -- unnormalized g blows up "
                 "(4B +123%, llama8B +489%)")
    if arm == "b2" and (t.get("grad_g_pow"), t.get("grad_s_pow")) != (1.0, 1.0):
        sys.exit(f"[gate] b2 expected g_pow=s_pow=1, got "
                 f"{t.get('grad_g_pow')}/{t.get('grad_s_pow')} -- squaring a "
                 "CV-20 gain re-concentrates the budget")
    print(f"[gate] PASS {arm}: grad_source=block, row_norm={t.get('grad_row_norm')}, "
          f"pows={t.get('grad_g_pow')}/{t.get('grad_s_pow')}")
    raise SystemExit(0)
if arm == "pow2":
    lv = t.get("stoc_len_levels") or []
    bad = [v for v in lv if v & (v - 1)]
    if bad:
        sys.exit(f"[gate] pow2 arm emitted NON-power-of-two rungs {bad} -- the "
                 "ladder override did not take, this would be the control")
    print(f"[gate] PASS pow2: ladder {lv}, all powers of two "
          f"(the only rungs fixed-point can express)")
    raise SystemExit(0)
if arm == "control":
    if t.get("cross_layer_weight") != "uniform":
        sys.exit(f"[gate] control must be uniform, got {t.get('cross_layer_weight')!r}")
    print("[gate] PASS control: cross_layer_weight=uniform, no group weights")
    raise SystemExit(0)
if not w:
    sys.exit("[gate] arm carries NO normalized_weights -- the weighting no-opped")
exp = {"p1": 2, "p2": 9, "p3": 36}[arm]
n = len(set(round(v, 9) for v in w.values()))
if n != exp:
    sys.exit(f"[gate] expected {exp} distinct weights for {arm}, got {n}")
# Judge the POOLED RAW ratio, never the normalized one. _normalize_group_weights
# floors at 0.1*mean_pos, so a NEGATIVE pooled linear mean (pure noise) gets
# clipped up and reappears as a huge positive ratio. On 30B t32 the raw ratio was
# -22.68x and the normalized one 45.00x: the floor MANUFACTURED the weight, and a
# post-floor check passed exactly when it should have failed.
pooled = m.get("pooled_delta_loss") or {}
if not pooled:
    sys.exit("[gate] table has no pooled_delta_loss -- cannot judge the raw ratio")
pa = [v for k, v in pooled.items() if k.split(":")[0] in ("qk", "av")]
pl = [v for k, v in pooled.items() if k.split(":")[0] not in ("qk", "av")]
mu_a, mu_l = sum(pa)/len(pa), sum(pl)/len(pl)
if mu_l <= 0:
    sys.exit(f"[gate] pooled LINEAR mean dLoss {mu_l:+.5f} <= 0 -- the probe is "
             "noise at this scale; any ratio is a flooring artifact. REFUSE.")
ra = mu_a / mu_l
negf = float(m.get("raw_negative_frac") or 0.0)
if negf > 0.35:
    sys.exit(f"[gate] raw negative fraction {negf:.0%} > 35% -- probe too noisy to "
             "price attention; REFUSE rather than spend eval hours")
if not (1.0 < ra < 40.0):
    sys.exit(f"[gate] pooled raw attention/linear {ra:.2f}x outside (1, 40) -- "
             "does not reproduce the documented 5-20x mispricing")
print(f"[gate] PASS {arm}: {n} distinct weights, pooled RAW attention/linear = "
      f"{ra:.2f}x, raw negative frac = {negf:.0%}")
PY

  # Wrapper (the calibrator does not emit one).
  python - "$TABLE" "$WRAPPER" <<'PY' || exit 2
import json, sys
t = json.load(open(sys.argv[1]))
json.dump({"type": "AdaptiveMPConfig",
           "stoc_len_levels": t["stoc_len_levels"],
           "threshold_table_path": sys.argv[1]}, open(sys.argv[2], "w"), indent=1)
print(f"[lossw] wrapper -> {sys.argv[2]}  levels={t['stoc_len_levels']}")
PY
  echo "[lossw] calib stage OK for $TAG"
  exit 0
fi

# ---------------------------------------------------------------------- eval
if [[ "$KB_STAGE" == "eval" ]]; then
  [[ -f "$WRAPPER" ]] || { echo "[lossw] missing wrapper $WRAPPER" >&2; exit 2; }
  export QUANT_CONFIG=mp
  export MP_CONFIG_JSON="$WRAPPER"
  export SC_HYBRID_CONFIG_JSON="$PARENT/hybrid_config.json"
  export FRONTEND=awq          # post-HPCA baseline; calibration is SQ (see header)
  export MODEL_PATH="$HF"
  export CTX=2048
  export SQ_ALPHA=0.5
  # FULL PROTOCOL. No truncated evals, smokes included.
  export PPL_MAX_TOKENS=0
  export SC_MP_TRACE="$OUT/${TAG}_trace.json"
  echo "[lossw] eval $TAG  model=$HF  frontend=$FRONTEND  PPL_MAX_TOKENS=0"
  echo "[lossw] config=$MP_CONFIG_JSON"
  srun python -u benchmark/quant/eval_quant.py
  exit $?
fi

echo "[lossw] KB_STAGE must be calib|eval, got '$KB_STAGE'" >&2
exit 2
