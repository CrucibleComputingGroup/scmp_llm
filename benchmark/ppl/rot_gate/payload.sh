#!/bin/bash
# Rotation-gate payload: for rotation seeds 0,1,2 —
#   rotate (R1+R2 offline fold) -> fp16 PPL fold sanity (non-citable) ->
#   act_scales on the rotated model (in-repo calibrate_smoothquant.py) ->
#   V9 calib_command byte-exact except --model_path/--output_json (+ the
#   scales file, which the calibrator takes via SMOOTHQUANT_SCALES env —
#   it has NO --calib-smoothquant-scales flag) -> rot_gate_report over all
#   completed seeds.
# Markers: SEED<k>_FOLD_DONE / SEED<k>_SANITY_OK(ppl=..) / SEED<k>_CALIB_DONE
#          / SEED<k>_FAIL / REPORT_DONE.  Per-seed failures do not kill later
#   seeds (subshell + || marker).
set -euo pipefail

TAG=${1:?usage: payload.sh <tag>}

set +u
source ~/.bashrc
conda activate annstention
set -u

REPO=/home/allenjin/Projects/scmp_llm
TURBO=/nfs/turbo/coe-nbleier/allenjin
BASE=${TURBO}/hpca/mp_rot_gate_llama8B_${TAG}
MODEL=meta-llama/Llama-3.1-8B-Instruct
BASELINE_TABLE=${TURBO}/hpca/mp_calib_mp_v9_int6_20260716_001908/meta-llama_Llama-3.1-8B-Instruct__mp_avg32_v9__act_global_v9_pc0.01x112_act_collapse_comp_pcovdown_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.10.json

# HF caches on Turbo (never /home)
export HF_HOME=${TURBO}/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}

# calibration-parity env (matches the V9 run environment)
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SC_HYBRID_CONFIG_JSON=${TURBO}/hpca/hybrid_configs/_hpca_mp_v9_int6_20260716_001908/llama8B_mp_avg32_v9_rank-sc_int8_measured_curve_top0p10.json
export SC_HYBRID_INT_BITS=7
export SC_HYBRID_FORCE_INT_BITS=1
export SQ_ALPHA=0.5
export PYTHONUNBUFFERED=1

mkdir -p "${BASE}"
cd "${REPO}"

echo "[payload] tag=${TAG} base=${BASE} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"

for seed in 0 1 2; do
  (
    set -euo pipefail
    ROT=${BASE}/rotated_seed${seed}
    SCALES=${BASE}/act_scales_rot_seed${seed}.pt
    TABLE=${BASE}/rot_calib_seed${seed}.json

    # 1) rotate + save (idempotent: manifest is written last -> completion marker)
    if [ -f "${ROT}/rotation_manifest.json" ]; then
      echo "SEED${seed}_FOLD_CACHED (${ROT})"
    else
      python -u benchmark/ppl/rot_gate/rotate_r1r2.py \
        --model "${MODEL}" --out "${ROT}" --seed "${seed}" --device cuda
    fi
    echo "SEED${seed}_FOLD_DONE"

    # 2) fp16 PPL fold sanity on fixed 32k-token wikitext-2 test prefix
    #    (NON-CITABLE diagnostic; prints SEED<k>_SANITY_OK or FOLD_SANITY_FAIL;
    #    nonzero exit on fail aborts this seed's subshell -> next seed runs)
    python -u benchmark/ppl/rot_gate/fold_sanity_ppl.py \
      --orig "${MODEL}" --rot "${ROT}" \
      --tokens 32768 --ctx 2048 --tol 0.005 --marker "SEED${seed}"

    # 3) act_scales on the rotated model — the SAME in-repo generator +
    #    recipe that produced the original act_scales (wikitext-2 train,
    #    128 sequences x 512 tokens, per-channel max|x| at every SCLinear
    #    input, SC disabled)
    if [ -f "${SCALES}" ]; then
      echo "SEED${seed}_SCALES_CACHED (${SCALES})"
    else
      MODEL_PATH="${ROT}" OUTPUT="${SCALES}" \
        CALIB_DATASET=wikitext CALIB_CONFIG=wikitext-2-raw-v1 \
        CALIB_SPLIT=train N_SAMPLES=128 SEQ_LEN=512 \
        python -u benchmark/ppl/calibrate_smoothquant.py
    fi
    python - "${SCALES}" <<'EOF'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
assert len(d) == 224, f"expected 224 SCLinear entries for llama8B, got {len(d)}"
k = "model.layers.0.self_attn.q_proj"
assert k in d and d[k].dtype == torch.float32 and d[k].dim() == 1, "key format drift"
print(f"[payload] act_scales OK: {len(d)} entries, e.g. {k} -> {tuple(d[k].shape)} {d[k].dtype}")
EOF

    # 4) V9 calibration, byte-exact calib_command except --model_path /
    #    --output_json; rotated act_scales injected via SMOOTHQUANT_SCALES
    #    (checked before ACT_SCALES_DIR by calibrate_mp_thresholds.py)
    if [ -f "${TABLE}" ]; then
      echo "SEED${seed}_CALIB_CACHED (${TABLE})"
    else
      SMOOTHQUANT_SCALES="${SCALES}" \
      python -u benchmark/ppl/calibrate_mp_thresholds.py \
        --model_path "${ROT}" \
        --mp_levels 96,64,48,32,24,16 \
        --budget_ratio 0.25 \
        --budget_ref_stoc_len 128 \
        --sc_prec 8 \
        --halve 1 \
        --seed 0 \
        --ctx_len 2048 \
        --budget-scope global \
        --budget-weight macs \
        --mac-weights-trace /home/allenjin/Projects/hpca_results/llm/uniform/traces/llama8B_sc_int7_trace.json \
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
    echo "SEED${seed}_CALIB_DONE"
  ) || echo "SEED${seed}_FAIL"
done

# 5) report over all completed seeds
shopt -s nullglob
TABLES=("${BASE}"/rot_calib_seed*.json)
if [ "${#TABLES[@]}" -gt 0 ]; then
  if python -u benchmark/ppl/rot_gate/rot_gate_report.py \
      --baseline "${BASELINE_TABLE}" \
      --rotated "${TABLES[@]}" \
      --out-prefix "${BASE}/rot_gate_report"; then
    echo "REPORT_DONE"
  else
    echo "REPORT_FAIL"
  fi
else
  echo "REPORT_SKIPPED_NO_TABLES"
fi
