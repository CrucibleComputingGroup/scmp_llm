"""Shared argparse helpers for benchmark/* entry scripts.

This repo only exercises two attention paths:

* ``fp16`` — vanilla HF eager attention, no SC.
* ``sc``   — full SC matmul (attention + linear) via
  ``model.llama_sc.LlamaForCausalLM``.

The upstream RetroInfer / Quest plumbing (``generate_config``,
``parse_attn_args`` with budget/estimate/coverage knobs) has been removed.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def parse_sc_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--mode", type=str, default="fp16",
        choices=["fp16", "sc", "sc_linear", "quant"],
        help="fp16 baseline, full SC (attention + linear), sc_linear "
             "(attention=fp16, linear=SC), or quant (integer PTQ baseline: "
             "SmoothQuant + per-channel-weight / per-token-activation RTN "
             "fake-quant — set --quant_config, e.g. W8A8_symm).")
    parser.add_argument(
        "--quant_config", type=str, default="fp16",
        help="Quant baseline config tag for --mode quant: "
             "fp16 | W{8..4}A{8..4}_{symm,asymm}. See benchmark/quant/ptq.py.")
    parser.add_argument(
        "--sc_prec", type=int, default=8,
        help="SC precision (quantization grid). Used when --mode sc.")
    parser.add_argument(
        "--sc_stoc_len", type=int, default=256,
        help="SC stochastic stream length. Used when --mode sc.")
    parser.add_argument(
        "--sc_attn_granularity", type=str, default="per_head",
        choices=["per_head", "per_row"],
        help="SC quant granularity for the two attention matmuls "
             "(Q·Kᵀ and softmax·V). per_head = one scale per head; "
             "per_row = one scale per row within each head (finer).")
    return parser
