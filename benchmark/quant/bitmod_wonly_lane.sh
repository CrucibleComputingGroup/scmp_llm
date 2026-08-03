#!/bin/bash
# Weight-only baseline lane: INT / FP(BitMoD-plain) / mixed_bitmod, W4+W3, A16,
# g128, SQ-free (A>=16 auto-skips SmoothQuant in build_model), full protocol.
# Usage: sbatch <resources> bitmod_wonly_lane.sh <model> <configs_csv> <tag> [crosscheck]
#   crosscheck=1 additionally runs the BitMoD reference-harness cross-check
#   (benchmark/quant/bitmod_ref_crosscheck.py) and prints both PPLs.
# NOTE: bashrc must be sourced BEFORE set -u (/etc/bashrc reads unset vars).
source ~/.bashrc
conda activate annstention
set -uo pipefail
cd /home/allenjin/Projects/scmp_llm

MODEL="$1"; CONFIGS="$2"; TAG="$3"; XCHECK="${4:-0}"

bash hpca --models "$MODEL" --configs "$CONFIGS" --metrics ppl --tag "$TAG"
rc=$?

if [[ "$XCHECK" == "1" ]]; then
  export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
  export TRANSFORMERS_CACHE="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" \
         HF_DATASETS_CACHE="$HF_HOME/datasets"
  MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 WQ_BITS=4 WQ_DATATYPE=mixed_bitmod \
    WQ_GROUPSIZE=128 python -u benchmark/quant/bitmod_ref_crosscheck.py
  TSV="/nfs/turbo/coe-nbleier/allenjin/hpca/results/results_${TAG}.tsv"
  ours=$(awk -F'\t' '$1=="4B" && $2=="W4A16_bitmod" && $3=="ppl"{v=$4} END{print v}' "$TSV")
  echo "[CROSSCHECK-DELTA] harness_W4A16_bitmod=${ours:-MISSING} vs [CROSSCHECK] ppl above (expect ~0.01 agreement)"
fi
exit "$rc"
