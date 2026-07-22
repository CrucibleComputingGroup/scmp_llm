#!/usr/bin/env python3
"""Real-GPU bit-identity/throughput gate for 30B MoE window batching.

This loads the production model and a frozen MP deployment, evaluates the same
full validation windows at B=1 and at the requested batch size, and refuses
success unless every stored per-window NLL is exactly equal. Run it before
setting FINAL_WINDOW_BATCH_SIZE > 1.
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
    p.add_argument("--hybrid-config", required=True, type=Path)
    p.add_argument("--hybrid-int-bits", required=True, type=int)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True, type=Path)
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.windows < 1 or args.batch_size < 2:
        raise SystemExit("--windows >=1 and --batch-size >=2 are required")
    os.environ["MP_CONFIG_JSON"] = str(args.parent_wrapper.resolve())
    os.environ["SC_HYBRID_CONFIG_JSON"] = str(args.hybrid_config.resolve())
    os.environ["SC_HYBRID_INT_BITS"] = str(args.hybrid_int_bits)
    os.environ["SC_HYBRID_FORCE_INT_BITS"] = "1"
    os.environ.setdefault("SC_OWEN_MODE", "bitrev")
    os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

    from benchmark.ppl.mp_ladder_refine import _load_eval_stream
    from benchmark.quant.eval_quant import build_model, compute_ppl

    model, tokenizer = build_model(args.model_path, "mp")
    loaded_bits = int(getattr(model.config, "sc_hybrid_int_bits", -1))
    forced = bool(getattr(model.config, "sc_hybrid_force_int_bits", False))
    if loaded_bits != args.hybrid_int_bits or not forced:
        raise SystemExit(
            "hybrid override did not reach the real model: "
            f"requested INT{args.hybrid_int_bits}, loaded INT{loaded_bits}, "
            f"force={forced}")
    enc = _load_eval_stream(
        tokenizer, split="validation", max_tokens=0, ctx=args.ctx)
    enc = enc[:args.windows * args.ctx]
    if enc.numel() != args.windows * args.ctx:
        raise SystemExit("validation stream does not contain requested windows")

    runs = {}
    for batch_size in (1, args.batch_size):
        losses = []
        torch.cuda.synchronize()
        start = time.time()
        ppl, tokens, seconds = compute_ppl(
            model, enc, args.ctx, args.ctx,
            window_losses=losses, window_batch_size=batch_size)
        torch.cuda.synchronize()
        runs[str(batch_size)] = {
            "ppl": ppl,
            "tokens": tokens,
            "seconds": seconds,
            "wall_seconds": time.time() - start,
            "window_nll": [loss for _start, _valid, loss in losses],
        }

    a = torch.tensor(runs["1"]["window_nll"], dtype=torch.float64)
    b = torch.tensor(
        runs[str(args.batch_size)]["window_nll"], dtype=torch.float64)
    identical = bool(torch.equal(a, b))
    payload = {
        "schema": "scmp-30b-window-batch-identity-v1",
        "model": args.model_path,
        "wrapper": str(args.parent_wrapper.resolve()),
        "hybrid_config": str(args.hybrid_config.resolve()),
        "hybrid_int_bits": args.hybrid_int_bits,
        "loaded_hybrid_int_bits": loaded_bits,
        "loaded_force_int_bits": forced,
        "ctx": args.ctx,
        "windows": args.windows,
        "batch_size": args.batch_size,
        "torch_equal_window_nll": identical,
        "speedup": runs["1"]["seconds"] / runs[str(args.batch_size)]["seconds"],
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not identical or runs["1"]["ppl"] != runs[str(args.batch_size)]["ppl"]:
        raise SystemExit("FAIL: batched and unbatched per-window NLL differ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
