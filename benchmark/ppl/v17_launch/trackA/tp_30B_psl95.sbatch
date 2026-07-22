#!/bin/bash
#SBATCH --job-name=tp_30B_psl95
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_mp_handbuilt_mp_v17_transplant_20260719_022423/%x_%j.out

# V17 track-A transplant wave (2026-07-19). 30B PSL-ONLY variant (user decision: full lift infeasible, -7.21): psl95 + MLP b16 reinvest (MoE-measured weights); predicted 31.5388 vs parent 31.5382
# Eval-only; full HPCA protocol. Build gates all passed (see build_report.json).

CELL=psl95_reinvest
CELLDIR=/nfs/turbo/coe-nbleier/allenjin/hpca/mp_handbuilt_mp_v17_transplant_20260719_022423/30B/psl95_reinvest

source ~/.bashrc
conda activate annstention
set -euo pipefail
cd /home/allenjin/Projects/scmp_llm

export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export SC_HYBRID_INT_BITS=7
export SC_HYBRID_FORCE_INT_BITS=1
export SC_HYBRID_CONFIG_JSON=/nfs/turbo/coe-nbleier/allenjin/hpca/hybrid_configs/_hpca_mp_v9_int6_20260716_001908/30B_mp_avg32_v9_rank-sc_int8_measured_curve_top0p10.json
export ACT_SCALES_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/act_scales
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[tp_wave] START cell=$CELL $(date)"
nvidia-smi -L || true

python -u benchmark/ppl/mp_ladder_refine.py \
  --parent-wrapper "$CELLDIR/wrapper.json" \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --output-dir "$CELLDIR/eval" \
  --split test --max-rounds 0 --max-tokens 0 --ctx 2048 --alpha 0.5

echo "[tp_wave] EVAL_DONE cell=$CELL $(date)"
python - "$CELLDIR/eval/summary.json" "$CELL" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"FINAL cell={sys.argv[2]} model={s['model']} ppl={s['best_ppl']:.6f} "
      f"realized={s['best_trace_flop_avg_stoc_len']:.4f} levels={s['best_levels']}")
PY
echo "[tp_wave] END cell=$CELL $(date)"
