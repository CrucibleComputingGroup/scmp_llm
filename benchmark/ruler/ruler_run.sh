#!/bin/bash
# Usage: ruler_run.sh <model> <benchmark> <ctx_len> <task> <mode> <dtype> <device> [sc_prec] [sc_stoc_len]
#
#   model       — key in ruler_config_models.sh (e.g. llama-3.1-8b)
#   benchmark   — synthetic | ... (key in ruler_config_tasks.sh)
#   ctx_len     — max sequence length (e.g. 4096, 8192, 16384, ...)
#   task        — task name (e.g. niah_single_1, vt, qa_1, ...)
#   mode        — fp16 | sc
#   dtype       — fp16 | bf16
#   device      — cuda:0, cuda:1, ...
#   sc_prec     — SC precision (default 8; only used when mode=sc)
#   sc_stoc_len — SC stochastic stream length (default 256; only used when mode=sc)
#
# Note: full SC at ctx_len>=4096 currently OOMs on a 96 GB GPU because the
# per_head bipolar kernel sizes its cum_indicator table by inner-dim (= N for
# softmax·V). Run --mode sc with ctx_len <= 1024 until per_head SC gets a
# chunk_d path. --mode fp16 is unconstrained.
set -euo pipefail

if [ $# -lt 7 ]; then
    echo "Usage: $0 <model> <benchmark> <ctx_len> <task> <mode> <dtype> <device> [sc_prec] [sc_stoc_len]"
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/nfs/turbo/coe-nbleier/zhkangqi/hf_cache_hub}"

ROOT_DIR="./ruler_eval_result"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
SC_ATTN_GRANULARITY="${SC_ATTN_GRANULARITY:-per_head}"

MODEL_NAME=${1}
BENCHMARK=${2}
MAX_SEQ_LENGTH=${3}
TASK=${4}
MODE=${5}
DTYPE=${6}
DEVICE=${7}
SC_PREC=${8:-8}
SC_STOC_LEN=${9:-256}

case "${MODE}" in
    fp16)      TAG="fp16" ;;
    sc_linear) TAG="sc_linear_prec${SC_PREC}_stoc${SC_STOC_LEN}" ;;
    sc)        TAG="sc_prec${SC_PREC}_stoc${SC_STOC_LEN}_gran-${SC_ATTN_GRANULARITY}" ;;
    *) echo "unknown mode: ${MODE}"; exit 1 ;;
esac

# Model and Tokenizer (HF id + chat template + framework)
source ruler_config_models.sh
MODEL_CONFIG=$(MODEL_SELECT "${MODEL_NAME}")
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported (check ruler_config_models.sh)."
    exit 1
fi

# Benchmark and Tasks (for REMOVE_NEWLINE_TAB / STOP_WORDS exports)
source ruler_config_tasks.sh
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/${TAG}"
DATA_DIR="${RESULTS_DIR}/data"
PRED_DIR="${RESULTS_DIR}/pred"
mkdir -p "${DATA_DIR}" "${PRED_DIR}"

echo "=== prepare data ==="
python -u data/prepare.py \
    --save_dir "${DATA_DIR}" \
    --benchmark "${BENCHMARK}" \
    --task "${TASK}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --tokenizer_type "${TOKENIZER_TYPE}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --model_template_type "${MODEL_TEMPLATE_TYPE}" \
    --num_samples "${NUM_SAMPLES}" \
    ${REMOVE_NEWLINE_TAB}

echo "=== predict ==="
python -u pred/call_api.py \
    --model_name "${MODEL_PATH}" \
    --mode "${MODE}" \
    --sc_prec "${SC_PREC}" \
    --sc_stoc_len "${SC_STOC_LEN}" \
    --max_len "${MAX_SEQ_LENGTH}" \
    --batch_size 1 \
    --data_dir "${DATA_DIR}" \
    --save_dir "${PRED_DIR}" \
    --benchmark "${BENCHMARK}" \
    --task "${TASK}" \
    --dtype "${DTYPE}" \
    --server_type "${MODEL_FRAMEWORK}" \
    --device "${DEVICE}" \
    --synthetic_len "${MAX_SEQ_LENGTH}" \
    --sc_attn_granularity "${SC_ATTN_GRANULARITY}"

echo "=== evaluate ==="
python -u eval/evaluate.py \
    --data_dir "${PRED_DIR}" \
    --benchmark "${BENCHMARK}"
