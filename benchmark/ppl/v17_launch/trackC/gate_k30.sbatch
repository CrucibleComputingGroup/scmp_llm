#!/bin/bash
#SBATCH --job-name=gate_k30
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_mp_gate_ab_20260719_032500/%x_%j.out

# V17 Track C — absolute escape gate A/B (2026-07-19). 4B, k=3.0.
# Wrapper = V9 4B avg32 wrapper + escape_gate_k/escape_stoc_len (TABLE UNTOUCHED).
# Predicted realized cost delta +0.21 cyc; predicted fire rate 0.061% (rows).
# Judged on (PPL, realized-energy): escaped rows priced at 128 in the trace,
# so realized flop-avg accounts for the budget float. Trace ON for fire-rate
# post-hoc check. Gate defaults OFF for wrappers without escape_gate_k (proven
# byte-identical by benchmark/ppl/test_mp_escape_gate.py).

WRAP=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_gate_ab_20260719_032500/Qwen_Qwen3-4B_gate_k3_wrapper.json
OUT=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_gate_ab_20260719_032500/eval_k3

source ~/.bashrc
conda activate annstention
set -euo pipefail
cd /home/allenjin/Projects/scmp_llm

export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SC_HYBRID_INT_BITS=7
export SC_HYBRID_FORCE_INT_BITS=1
export SC_HYBRID_CONFIG_JSON=/nfs/turbo/coe-nbleier/allenjin/hpca/hybrid_configs/_hpca_mp_v9_int6_20260716_001908/4B_mp_avg32_v9_rank-sc_int8_measured_curve_top0p10.json
export ACT_SCALES_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/act_scales
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hpca/../hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[gate_ab] START k=3.0 $(date)"
nvidia-smi -L || true

python -u benchmark/ppl/mp_ladder_refine.py \
  --parent-wrapper "$WRAP" \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir "$OUT" \
  --split test --max-rounds 0 --max-tokens 0 --ctx 2048 --alpha 0.5

echo "[gate_ab] EVAL_DONE k=3.0 $(date)"
python - "$OUT/summary.json" "3.0" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"FINAL gate k={sys.argv[2]} ppl={s['best_ppl']:.6f} "
      f"realized={s['best_trace_flop_avg_stoc_len']:.4f} levels={s['best_levels']}")
PY
echo "[gate_ab] END k=3.0 $(date)"
