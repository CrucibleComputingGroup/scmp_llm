#!/bin/bash
# Usage: pred.sh <model> <task> <mode> <dtype> <device> [sc_prec] [sc_stoc_len] [num_examples]
#
#   model     — key in config/model2path.json (e.g. llama-3.1-8b)
#   task      — LongBench task name (e.g. qasper)
#   mode      — fp16 | sc
#   dtype     — fp16 | bf16
#   device    — cuda:0, cuda:1, ...
#   sc_prec   — SC precision (default 8; only used when mode=sc)
#   sc_stoc_len — SC stochastic stream length (default 256; only used when mode=sc)
#   num_examples — cap per-task examples (default -1 = all)
set -euo pipefail

if [ $# -lt 5 ]; then
    echo "Usage: $0 <model> <task> <mode> <dtype> <device> [sc_prec] [sc_stoc_len] [num_examples]"
    exit 1
fi

MODEL=${1}
TASK=${2}
MODE=${3}
DTYPE=${4}
DEVICE=${5}
SC_PREC=${6:-8}
SC_STOC_LEN=${7:-256}
NUM_EXAMPLES=${8:--1}

if [ "${MODE}" = "fp16" ]; then
    TAG="fp16"
else
    TAG="sc_prec${SC_PREC}_stoc${SC_STOC_LEN}"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

# Model weights live on scratch; pin the HF hub cache so we never fall back
# to a stale copy under $HOME.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/hf_cache_hub}"

RESULT_DIR="./results/pred/${MODEL}/${TAG}"
RESULT_DIR_E="./results/pred_e/${MODEL}/${TAG}"

echo "remove previous result file..."
rm -f "${RESULT_DIR}/${TASK}.jsonl"
rm -f "${RESULT_DIR_E}/${TASK}.jsonl"

echo "Start to predict..."
python -u pred.py \
    --model "${MODEL}" \
    --task "${TASK}" \
    --mode "${MODE}" \
    --sc_prec "${SC_PREC}" \
    --sc_stoc_len "${SC_STOC_LEN}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --num_examples "${NUM_EXAMPLES}"
