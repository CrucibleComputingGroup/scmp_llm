#!/usr/bin/env python
"""Fold sanity check: fp16 PPL of rotated vs unrotated Llama on a fixed
32k-token wikitext-2 TEST prefix.  DIAGNOSTIC ONLY — NON-CITABLE (truncated
eval; the citable protocol is full wikitext-2 via eval_quant.py).

Prints "<marker>_SANITY_OK(ppl=..,orig=..,rel=..)" when |dPPL|/PPL < --tol,
else "FOLD_SANITY_FAIL ..." and exits 1 (payload.sh then skips this seed's
calibration).
"""
import argparse
import math
import sys

import torch


def ppl_of(path: str, enc: torch.Tensor, ctx: int, device: str) -> float:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model.eval()
    nll, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, enc.shape[0], ctx):
            ids = enc[start:start + ctx]
            if ids.shape[0] < 2:
                break
            ids = ids.unsqueeze(0).to(device)
            out = model(input_ids=ids, labels=ids)
            n = ids.shape[1] - 1
            nll += out.loss.item() * n
            count += n
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return math.exp(nll / count)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True, help="unrotated model (HF id/dir)")
    p.add_argument("--rot", required=True, help="rotated model dir")
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--marker", default="SEED?")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.orig)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds)
    enc = tok(text, return_tensors="pt").input_ids[0][:args.tokens]
    print(f"[sanity] fixed test prefix: {enc.shape[0]} tokens, ctx {args.ctx}, "
          f"non-overlapping windows, fp16, device {args.device}", flush=True)

    po = ppl_of(args.orig, enc, args.ctx, args.device)
    pr = ppl_of(args.rot, enc, args.ctx, args.device)
    rel = abs(pr - po) / po
    print(f"[sanity][NON-CITABLE truncated diagnostic] fp16 PPL on 32k test "
          f"prefix: orig={po:.4f} rotated={pr:.4f} rel_delta={rel:.5f} "
          f"(gate < {args.tol})", flush=True)
    if rel < args.tol:
        print(f"{args.marker}_SANITY_OK(ppl={pr:.4f},orig={po:.4f},"
              f"rel={rel:.5f})", flush=True)
    else:
        print(f"FOLD_SANITY_FAIL {args.marker} rel={rel:.5f} >= {args.tol} "
              f"(orig={po:.4f} rotated={pr:.4f})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
