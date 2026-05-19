#!/bin/bash
# Usage: longbench_run.sh <model> <mode> <dtype> <device> [sc_prec] [sc_stoc_len]
#
#   model     — key in config/model2path.json (e.g. llama-3.1-8b)
#   mode      — fp16 | sc
#   dtype     — fp16 | bf16
#   device    — cuda:0, cuda:1, ...
#   sc_prec   — SC precision (default 8; only used when mode=sc)
#   sc_stoc_len — SC stochastic stream length (default 256; only used when mode=sc)
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <model> <mode> <dtype> <device> [sc_prec] [sc_stoc_len]"
    exit 1
fi

MODEL=${1}
MODE=${2}
DTYPE=${3}
DEVICE=${4}
SC_PREC=${5:-8}
SC_STOC_LEN=${6:-256}

if [ "${MODE}" = "fp16" ]; then
    TAG="fp16"
else
    TAG="sc_prec${SC_PREC}_stoc${SC_STOC_LEN}"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/hf_cache_hub}"

RESULT_DIR="./results/pred/${MODEL}/${TAG}"

tasks=(2wikimqa gov_report hotpotqa lcc multi_news multifieldqa_en musique narrativeqa passage_retrieval_en qasper qmsum repobench-p triviaqa)

for task in "${tasks[@]}"; do
    echo "=== ${MODEL} ${task} mode=${MODE} dtype=${DTYPE} device=${DEVICE} sc_prec=${SC_PREC} sc_stoc_len=${SC_STOC_LEN} ==="
    bash pred.sh "${MODEL}" "${task}" "${MODE}" "${DTYPE}" "${DEVICE}" "${SC_PREC}" "${SC_STOC_LEN}"
done

echo "Start to evaluate..."
python -u eval.py \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --sc_prec "${SC_PREC}" \
    --sc_stoc_len "${SC_STOC_LEN}"

echo "Results:"
cat "${RESULT_DIR}/result.json"
