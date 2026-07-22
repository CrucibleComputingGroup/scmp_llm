#!/bin/bash
# Rotation-gate + FULL-PROTOCOL EVAL payload for Qwen/Qwen3-4B-Instruct-2507,
# single rotation seed 0 in ONE single-GPU pass:
#   a) rotate (R1+R2 offline fold; Qwen3: Paley H_20 Kronecker Hadamard for
#      hidden 2560, QK-Norms left untouched, tied embeddings untied on save)
#   b) fp16 PPL fold sanity on a fixed 32k wikitext-2 test prefix (tol 0.005)
#   c) act_scales on the rotated model (same recipe as the originals)
#   d) V9 calibration, byte-exact calib_command from the trusted 4B table
#      except --model_path/--output_json; rotated scales via SMOOTHQUANT_SCALES
#      env (the calibrator has NO CLI flag for a scales file)
#   e) rot_gate_report vs the trusted 4B baseline table (passed EXPLICITLY —
#      the report script's default baseline is the llama8B table)
#   f) full-protocol MP eval (eval_quant.py, full wikitext-2, ctx 2048,
#      PPL_MAX_TOKENS=0) of the rotated model under the fresh table
#
# SINGLE SEED 0: sign-magnitude SC is invariant to the +-1 rotation sign
# seeds (proven on the llama8B rot-gate run tonight — cross-seed sigma deltas
# were sign-consistent and seed-independent), so one seed carries the verdict.
#
# Markers: FOLD_DONE / SANITY_OK / SCALES_DONE / CALIB_DONE / REPORT_DONE /
#          EVAL_DONE / FINAL_PPL=<value>
#
# CRITICAL shell gotcha (cost us a job tonight): `source ~/.bashrc` must run
# BEFORE any `set -u` (bashrc trips on unset vars), and we never enable -u —
# only `set -eo pipefail` AFTER sourcing.
source ~/.bashrc
conda activate annstention
set -eo pipefail

TAG=${1:?usage: payload_4B.sh <tag>}

REPO=/home/allenjin/Projects/scmp_llm
TURBO=/nfs/turbo/coe-nbleier/allenjin
BASE=${TURBO}/hpca/mp_antiquarot_4B_${TAG}
Q1=${BASE}/q1_concentration.pt
BUDGET_RATIO=${BUDGET_RATIO:-0.25}
MODEL=Qwen/Qwen3-4B-Instruct-2507
BASELINE_TABLE=${TURBO}/hpca/mp_calib_mp_v9_int6_20260716_001908/Qwen_Qwen3-4B-Instruct-2507__mp_avg32_v9__act_global_v9_pc0.01x112_act_collapse_comp_pcovdown_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.10.json
ORIG_SCALES=${TURBO}/hpca/act_scales/act_scales_Qwen_Qwen3-4B-Instruct-2507.pt

ROT=${BASE}/rotated_seed0
SCALES=${BASE}/act_scales_rot_seed0.pt
TABLE=${BASE}/rot_calib_seed0.json
EVAL=${BASE}/eval_seed0

# HF caches on Turbo (never /home)
export HF_HOME=${TURBO}/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}

# calibration/eval-parity env (matches the V9 4B cell environment)
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SC_HYBRID_CONFIG_JSON=${TURBO}/hpca/hybrid_configs/_hpca_mp_v9_int6_20260716_001908/4B_mp_avg32_v9_rank-sc_int8_measured_curve_top0p10.json
export SC_HYBRID_INT_BITS=7
export SC_HYBRID_FORCE_INT_BITS=1
export SQ_ALPHA=0.5
export PYTHONUNBUFFERED=1

mkdir -p "${BASE}"
cd "${REPO}"

echo "[payload4B] tag=${TAG} base=${BASE} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"

# a) rotate + save bf16 (idempotent: manifest is written last -> completion
#    marker).  Qwen3 tied embeddings are untied inside rotate_r1r2.py (the
#    final-norm gamma fold cannot ride a shared E/lm_head tensor) and the
#    save is self-verified untied; the fold-sanity gate below catches any
#    residual issue end-to-end.
if [ -f "${ROT}/rotation_manifest.json" ]; then
  echo "FOLD_CACHED (${ROT})"
else
  if [ ! -s "${Q1}" ]; then
    echo "[antiquarot] learning concentration-maximizing R1"
    mkdir -p "${BASE}"
    python -u benchmark/ppl/rot_gate/learn_concentration_rotation.py \
      --model "${MODEL}" --out "${Q1}" --samples 8 --ctx 2048 \
      --iters 200 --device cuda
  fi
  echo "LEARN_DONE"
  python -u benchmark/ppl/rot_gate/rotate_r1r2.py \
    --model "${MODEL}" --out "${ROT}" --seed 0 --device cuda \
    --r1-matrix "${Q1}"
fi
echo "FOLD_DONE"

# b) fp16 PPL fold sanity, fixed 32k-token wikitext-2 test prefix
#    (NON-CITABLE diagnostic; exits nonzero on fail -> job stops here)
python -u benchmark/ppl/rot_gate/fold_sanity_ppl.py \
  --orig "${MODEL}" --rot "${ROT}" \
  --tokens 32768 --ctx 2048 --tol 0.005 --marker SEED0
echo "SANITY_OK"

# c) act_scales on the rotated model — same in-repo generator + recipe that
#    produced the original act_scales (wikitext-2 train, 128 seq x 512 tok,
#    per-channel max|x| at every SCLinear input, SC disabled)
if [ -f "${SCALES}" ]; then
  echo "SCALES_CACHED (${SCALES})"
else
  MODEL_PATH="${ROT}" OUTPUT="${SCALES}" \
    CALIB_DATASET=wikitext CALIB_CONFIG=wikitext-2-raw-v1 \
    CALIB_SPLIT=train N_SAMPLES=128 SEQ_LEN=512 \
    python -u benchmark/ppl/calibrate_smoothquant.py
fi
# entry-count + key-set parity vs the ORIGINAL 4B scales (252 = 36 layers x 7)
python - "${SCALES}" "${ORIG_SCALES}" <<'EOF'
import sys, torch
rot = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
ref = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
assert len(ref) == 252, f"original 4B act_scales drifted: {len(ref)} != 252"
assert len(rot) == len(ref), \
    f"entry count mismatch: rotated {len(rot)} vs original {len(ref)}"
assert set(rot) == set(ref), "act_scales module-name sets differ"
k = "model.layers.0.self_attn.q_proj"
assert k in rot and rot[k].dtype == torch.float32 and rot[k].dim() == 1, \
    "key format drift"
print(f"[payload4B] act_scales OK: {len(rot)} entries (== original), "
      f"e.g. {k} -> {tuple(rot[k].shape)} {rot[k].dtype}")
EOF
echo "SCALES_DONE"

# d) V9 calibration — byte-exact calib_command recorded in the trusted 4B
#    table (${BASELINE_TABLE}), changing ONLY --model_path and --output_json.
#    SAME levels 96,64,48,32,24,16 (isolate rotation; extended-menu combo is
#    a later cell).  Rotated scales injected via SMOOTHQUANT_SCALES (checked
#    before ACT_SCALES_DIR by calibrate_mp_thresholds.py; no CLI flag exists).
if [ -f "${TABLE}" ]; then
  echo "CALIB_CACHED (${TABLE})"
else
  SMOOTHQUANT_SCALES="${SCALES}" \
  python -u benchmark/ppl/calibrate_mp_thresholds.py \
    --model_path "${ROT}" \
    --mp_levels 96,64,48,32,24,16 \
    --budget_ratio ${BUDGET_RATIO} \
    --budget_ref_stoc_len 128 \
    --sc_prec 8 \
    --halve 1 \
    --seed 0 \
    --ctx_len 2048 \
    --budget-scope global \
    --budget-weight macs \
    --mac-weights-trace /home/allenjin/Projects/hpca_results/llm/uniform/traces/4B_sc_int7_trace.json \
    --calib-smoothquant \
    --protect-channel-frac 0.01 \
    --protect-channel-stoc-len 112 \
    --protect-channel-metric act_collapse \
    --protect-channel-frac-overrides down_proj:0.06,up_proj:0.03,gate_proj:0.03 \
    --protect-compensate-budget \
    --argmin-pricing macs \
    --metric-select auto \
    --refine sigma \
    --objective delta_sigma2 \
    --output_json "${TABLE}"
fi
echo "CALIB_DONE"

# e) sigma/rho gate report vs the trusted 4B baseline (EXPLICIT --baseline:
#    the script default is the llama8B table)
python -u benchmark/ppl/rot_gate/rot_gate_report.py \
  --baseline "${BASELINE_TABLE}" \
  --rotated "${TABLE}" \
  --out-prefix "${BASE}/rot_gate_report"
echo "REPORT_DONE"

# f) FULL-PROTOCOL EVAL (citable: full wikitext-2, ctx 2048, PPL_MAX_TOKENS=0)
#    eval_quant resolves ACT_SCALES_DIR/act_scales_{MODEL_PATH.replace('/','_')}.pt
#    and REFUSES to run if missing, so stage the rotated scales as a COPY
#    under that exact name (MODEL_PATH here is the absolute rotated dir).
mkdir -p "${EVAL}/act_scales"
SAFE_ROT=${ROT//\//_}
cp -f "${SCALES}" "${EVAL}/act_scales/act_scales_${SAFE_ROT}.pt"
cat > "${EVAL}/wrapper.json" <<WEOF
{"type": "AdaptiveMPConfig", "stoc_len_levels": [96, 64, 48, 32, 24, 16], "threshold_table_path": "${TABLE}"}
WEOF
EVAL_LOG=${EVAL}/eval_quant.log
MODEL_PATH="${ROT}" QUANT_CONFIG=mp MP_CONFIG_JSON="${EVAL}/wrapper.json" \
  ACT_SCALES_DIR="${EVAL}/act_scales" CTX=2048 SQ_ALPHA=0.5 PPL_MAX_TOKENS=0 \
  SC_MP_TRACE="${EVAL}/trace.json" \
  python -u benchmark/quant/eval_quant.py 2>&1 | tee "${EVAL_LOG}"
echo "EVAL_DONE"

FINAL_PPL=$(sed -n 's/.*metric=ppl value=\([0-9.]*\).*/\1/p' "${EVAL_LOG}" | tail -1)
if [ -z "${FINAL_PPL}" ]; then
  echo "FINAL_PPL_MISSING (no '[RESULT] ... metric=ppl value=' line in ${EVAL_LOG})"
  exit 1
fi
echo "FINAL_PPL=${FINAL_PPL}"
