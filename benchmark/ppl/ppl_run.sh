#!/bin/bash
# Usage: ppl_run.sh <model> <mode> <ctx> [sc_prec] [sc_stoc_len] [sc_attn_granularity]
#
#   model       — HF id (e.g. meta-llama/Llama-3.1-8B-Instruct or
#                          Qwen/Qwen3-4B-Instruct-2507)
#   mode        — fp16 | sc | sc_linear
#   ctx         — context window for the slide (e.g. 1024)
#   sc_prec     — default 8
#   sc_stoc_len — default 256
#   sc_attn_granularity — per_head | per_row (default per_head)
#
# Honors env vars: PPL_DATASET, PPL_DATASET_CONFIG, PPL_SPLIT,
#                  PPL_MAX_TOKENS, STRIDE, STOC_LENS.
set -eo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <model> <mode> <ctx> [sc_prec] [sc_stoc_len] [sc_attn_granularity]"
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/nfs/turbo/coe-nbleier/zhkangqi/hf_cache_hub}"

MODEL=${1}
MODE=${2}
CTX=${3}
SC_PREC=${4:-8}
SC_STOC_LEN=${5:-256}
SC_ATTN_GRANULARITY=${6:-per_head}

# Map mode -> SC config env vars consumed by ppl.py via loader.apply_sc_env_overrides.
case "${MODE}" in
    fp16)
        EXTRA="DISABLE_SC=1 STOC_LENS=" ;;
    sc_linear)
        EXTRA="USE_SC_ATTN=0 USE_SC_LINEAR=1 STOC_LENS=${SC_STOC_LEN}" ;;
    sc)
        EXTRA="STOC_LENS=${SC_STOC_LEN}" ;;
    *)
        echo "unknown mode: ${MODE}"; exit 1 ;;
esac

env MODEL_PATH="${MODEL}" \
    CTX="${CTX}" \
    SC_PREC="${SC_PREC}" \
    SC_ATTN_GRANULARITY="${SC_ATTN_GRANULARITY}" \
    ${EXTRA} \
    python -u ppl.py
