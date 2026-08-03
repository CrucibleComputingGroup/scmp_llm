#!/bin/bash
# RULER batch-size smoke (NON-CITABLE, timing only): 4B mp_avg96f, 32 samples
# at RULER_BATCH in {8,16,32}. Uses the cached ctx-4096 table. Separate
# RULER_TAG_SUFFIX per arm so nothing collides with real cells.
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"

TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
WRAPPER=$TURBO/mp_calib_mp_ruler_ctx4096_20260714_021537/Qwen_Qwen3-4B-Instruct-2507__avg96f_ctx4096__final_recipe_wrapper.json
HYB=$TURBO/hybrid_configs/_hpca_mp_final_sweep_20260713_143745/4B_mp_avg96f_burst128_rank-sc_int8_measured_curve_top0p10.json

for BS in 8 16 32; do
  echo "=== SMOKE BATCH=$BS start $(date +%H:%M:%S) ==="
  t0=$(date +%s)
  ( cd benchmark/ruler && \
    env QUANT_CONFIG=mp_avg96f MP_CONFIG_JSON="$WRAPPER" \
        SQ_ALPHA=0.5 ACT_SCALES_DIR="$TURBO/act_scales" \
        SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 \
        SC_HYBRID_CONFIG_JSON="$HYB" SC_HYBRID_INT_BITS=7 SC_HYBRID_FORCE_INT_BITS=1 \
        NUM_SAMPLES=32 RULER_BATCH=$BS RULER_TAG_SUFFIX="_bs${BS}smoke" \
        bash ruler_run.sh qwen3-4b synthetic 4096 niah_multivalue quant fp16 cuda:0 )
  rc=$?
  t1=$(date +%s)
  sec=$((t1-t0))
  S=benchmark/ruler/ruler_eval_result/qwen3-4b/synthetic/4096/quant_mp_avg96f_bs${BS}smoke/pred
  done_n=$(wc -l < $S/*.jsonl 2>/dev/null || echo 0)
  score=$(awk -F, '$1=="Score"{print $2}' $S/summary*.csv 2>/dev/null | tail -1)
  echo "[SMOKE RESULT] batch=$BS rc=$rc samples=$done_n wall=${sec}s per_sample=$((sec/32))s score=${score:-n/a}"
done
echo "[smoke done]"
