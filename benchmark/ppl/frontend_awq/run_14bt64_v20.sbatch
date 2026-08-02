#!/bin/bash
# Fill the AWQ@20% 14B-t64 dose-split cell, exact-match to the SmoothQuant 8.8520.
# That number is the V20 structured search's 14B t64 @20% UNGATED eval
# (mp_final_search_mp_v20_structured_14B_t64_20260721/14B/target64p000/eval_test,
# ppl 8.85204791, levels [89,72,66,63,61,58], table s02_precision_040_macro_ceilcomp_8,
# escape_gate None). The 20% mask is the one the V20 lane set via env.
#
# Two evals on the SAME reconstructed config:
#   1) FRONTEND=smoothquant  -> MUST reproduce 8.8520 (deterministic kernel).
#   2) FRONTEND=awq          -> the AWQ@20% 14B t64 number we want.
# If (1) != 8.8520, the reconstruction is wrong — do NOT trust (2).
#
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=gl1810
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_mpbest_awq/%x_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
cd /home/allenjin/Projects/scmp_llm

HF="Qwen/Qwen3-14B"
TURBO=/nfs/turbo/coe-nbleier/allenjin/hpca
MP_BEST=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best
V20=$TURBO/mp_final_search_mp_v20_structured_14B_t64_20260721/14B/target64p000
TABLE="$V20/s02_precision_040_macro_ceilcomp_8.json"
MASK="$TURBO/hybrid_configs/_hpca_mp_v9_hybdose_all_20260718_183035/14B_mp_avg32_v9_rank-sc_int8_measured_curve_top0p20.json"
OUT="$TURBO/results/awq_14bt64_v20_20pct"
mkdir -p "$OUT/traces"

for f in "$TABLE" "$MASK"; do [[ -f "$f" ]] || { echo "[14bt64] missing $f" >&2; exit 2; }; done

# Reconstructed UNGATED wrapper (no escape_gate_k -> matches the eval_test 8.8520).
WRAP="$OUT/14B_t64_v20_ungated_wrapper.json"
cat > "$WRAP" <<JSON
{"type":"AdaptiveMPConfig","stoc_len_levels":[89,72,66,63,61,58],"threshold_table_path":"$TABLE"}
JSON

# HF caches on Turbo (this driver calls eval_quant directly; mirror hpca).
HF_CACHE=$TURBO/../hf_cache
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"; export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export AWQ_SCALES_DIR="$TURBO/act_scales/awq_scales"
export AWQ_OBJ_BITS=4
export QUANT_CONFIG=mp
export MP_CONFIG_JSON="$WRAP"
export SC_HYBRID_CONFIG_JSON="$MASK"
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
export ACT_SCALES_DIR="$MP_BEST/act_scales"
export MODEL_PATH="$HF"
export PPL_MAX_TOKENS=0
export CTX=2048

echo "[14bt64] ===== VERIFY: SmoothQuant (expect ppl 8.8520) ====="
export SC_MP_TRACE="$OUT/traces/14B_t64_sq_trace.json"; export SC_MP_TRACE_MODE=summary
FRONTEND=smoothquant python -u benchmark/quant/eval_quant.py

echo "[14bt64] ===== AWQ@20% 14B t64 (the number we want) ====="
export SC_MP_TRACE="$OUT/traces/14B_t64_awq_trace.json"
FRONTEND=awq python -u benchmark/quant/eval_quant.py
