#!/usr/bin/env python3
"""Real-GPU exactness gate for batched 30B ``int_swap`` measurements.

The sensitivity sweep uses a private per-window loss driver rather than the
normal PPL evaluator.  Exercise that exact driver at B=1 and B>1 for the all-SC
reference, one linear swap, and one attention swap.  The resulting manifest is
required by the batched sweep and becomes part of its checkpoint signature.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--parent-wrapper", required=True, type=Path)
    p.add_argument("--hybrid-int-bits", required=True, type=int)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True, type=Path)
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.windows < 1 or args.batch_size < 2:
        raise SystemExit("--windows >=1 and --batch-size >=2 are required")
    wrapper = args.parent_wrapper.resolve()
    if not wrapper.is_file():
        raise SystemExit(f"missing parent wrapper: {wrapper}")

    os.environ["MP_CONFIG_JSON"] = str(wrapper)
    os.environ.pop("SC_HYBRID_CONFIG_JSON", None)
    os.environ.setdefault("SC_OWEN_MODE", "bitrev")
    os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

    from benchmark.ppl.calibrate_mp_thresholds import (
        _int_swap_window_losses,
    )
    from benchmark.ppl.mp_ladder_refine import _load_eval_stream
    from benchmark.quant.eval_quant import build_model

    model, tokenizer = build_model(args.model_path, "mp")
    cfg = model.config
    if getattr(cfg, "sc_mp_config", None) is None:
        raise SystemExit("parent wrapper did not load an MP configuration")
    cfg.use_sc_linear = True
    cfg.use_sc_attn = True
    cfg.sc_ste_grad = False
    cfg.sc_group_stoclen = None
    cfg.sc_hybrid_default = "sc"
    cfg.sc_hybrid_int_bits = int(args.hybrid_int_bits)
    cfg.sc_hybrid_int_sym = True
    cfg.sc_hybrid_chunk_size = 128

    enc = _load_eval_stream(
        tokenizer, split="validation", max_tokens=0, ctx=args.ctx)
    enc = enc[:args.windows * args.ctx]
    if int(enc.numel()) != args.windows * args.ctx:
        raise SystemExit("validation stream does not contain requested windows")
    windows = [enc[i:i + args.ctx]
               for i in range(0, int(enc.numel()), args.ctx)]

    schedules = {
        "all_sc": {},
        "linear_q_proj_b0": {("q_proj", 0): f"int{args.hybrid_int_bits}"},
        "attention_qk_b0": {("qk", 0): f"int{args.hybrid_int_bits}"},
    }
    results = {}
    total_seconds = {1: 0.0, args.batch_size: 0.0}
    all_exact = True
    for name, schedule in schedules.items():
        runs = {}
        for batch_size in (1, args.batch_size):
            os.environ["PPL_WINDOW_BATCH_SIZE"] = str(batch_size)
            torch.cuda.synchronize()
            start = time.time()
            losses = _int_swap_window_losses(
                model, cfg, windows, next(model.parameters()).device, schedule)
            torch.cuda.synchronize()
            seconds = time.time() - start
            total_seconds[batch_size] += seconds
            runs[str(batch_size)] = {
                "seconds": seconds,
                "window_loss": losses,
                "mean_loss": sum(losses) / len(losses),
            }
        serial = torch.tensor(runs["1"]["window_loss"], dtype=torch.float64)
        batched = torch.tensor(
            runs[str(args.batch_size)]["window_loss"], dtype=torch.float64)
        exact = bool(torch.equal(serial, batched))
        mean_exact = (runs["1"]["mean_loss"]
                      == runs[str(args.batch_size)]["mean_loss"])
        all_exact = all_exact and exact and mean_exact
        results[name] = {
            "torch_equal_window_loss": exact,
            "mean_loss_equal": mean_exact,
            "runs": runs,
        }

    payload = {
        "schema": "scmp-30b-intswap-window-batch-identity-v1",
        "model": args.model_path,
        "wrapper": str(wrapper),
        "hybrid_int_bits": args.hybrid_int_bits,
        "ctx": args.ctx,
        "windows": args.windows,
        "batch_size": args.batch_size,
        "all_schedules_torch_equal": all_exact,
        "speedup": total_seconds[1] / total_seconds[args.batch_size],
        "total_seconds": {str(k): v for k, v in total_seconds.items()},
        "schedules": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not all_exact:
        raise SystemExit("FAIL: batched and serial int_swap window losses differ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
