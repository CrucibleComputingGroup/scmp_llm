"""Per-window FP16 NLL references on the WikiText-2 validation split.

V16 gates its cheap tier-1 stop check against FP16 measured on the SAME
validation windows the search uses, instead of the full-test FP16 number
(the V14 gate compared 8k validation windows against full-test FP16 and
spuriously stopped every 4B/14B/30B branch in sweep 1).  This script runs
one plain-HF FP16 evaluation over the whole validation split and records
one mean NLL per non-overlapping ctx window, so any window subset's FP16
reference can be composed later.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.ppl.mp_ladder_refine import _atomic_json, _load_eval_stream


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--split", default="validation",
                   choices=["validation", "test"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--skip-if-exists", action="store_true")
    args = p.parse_args(argv)

    out = args.out.resolve()
    if args.skip_if_exists and out.exists():
        print(f"[fp16 ref] exists, skipping: {out}", flush=True)
        return 0

    # build_model routes to the MP builder whenever MP_CONFIG_JSON is set,
    # regardless of the requested tag; the FP16 reference must be plain HF.
    os.environ.pop("MP_CONFIG_JSON", None)
    from benchmark.quant.eval_quant import build_model, compute_ppl

    model, tokenizer = build_model(args.model_path, "fp16")
    enc = _load_eval_stream(
        tokenizer, split=args.split, max_tokens=0, ctx=args.ctx)
    n_windows = enc.numel() // args.ctx
    window_losses: list = []
    ppl, tokens, seconds = compute_ppl(
        model, enc, args.ctx, args.ctx, window_losses=window_losses)
    if len(window_losses) != n_windows:
        raise SystemExit(
            f"expected {n_windows} scored windows, got {len(window_losses)}")
    payload = {
        "schema": "scmp-fp16-window-reference-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model_path,
        "split": args.split,
        "ctx": int(args.ctx),
        "n_windows": int(n_windows),
        "stream_tokens": int(enc.numel()),
        "loss_tokens": int(tokens),
        "full_ppl": float(ppl),
        "eval_seconds": float(seconds),
        "window_nll": {
            str(start // args.ctx): loss
            for start, _valid, loss in window_losses
        },
    }
    _atomic_json(out, payload)
    print(
        f"[fp16 ref] model={args.model_path} split={args.split} "
        f"windows={n_windows} ppl={ppl:.4f} out={out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
