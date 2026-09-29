"""Measure post-RoPE per-dim |Q|/|K| spread and emit qk rebalance vectors.

The synthetic test showed a 73-75% SC-error reduction from a contracted-dim
rebalance -- but on operands with a deliberately extreme per-dim imbalance
(648x / 364x). The gain is entirely a function of how imbalanced the REAL
post-RoPE Q and K are, so that spread is the number that decides whether this
lever is worth anything. Measure it before spending any eval cells.

Emits, per (layer):
    s_d = mq_d^alpha / mk_d^(1-alpha)      (SmoothQuant migration form)
with mq/mk the per-dim absmax of post-RoPE Q/K over sampled rows. sc_matmul
applies it as (Q/s, K*s) INSIDE the matmul -- after RoPE -- so all head_dim
dims are free (a q_norm fold would need s[d] == s[d+head_dim/2]).

Usage:
    python -m benchmark.ppl.kbands.qk_calib --parent <bundle> \\
        --model_path <hf> --out qk_scales.json --alpha 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--num-windows", type=int, default=4)
    ap.add_argument("--ctx-len", type=int, default=2048)
    args = ap.parse_args()

    parent = Path(args.parent)
    os.environ.setdefault("SC_OWEN_MODE", "bitrev")
    os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")
    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = str(parent / "wrapper.json")
    hyb = parent / "hybrid_config.json"
    if hyb.is_file():
        os.environ["SC_HYBRID_CONFIG_JSON"] = str(hyb)

    from benchmark.quant.eval_quant import build_sc_model
    from benchmark.ppl.calibrate_mp_thresholds import _iter_calib_windows
    from datasets import load_dataset
    import model.sc_common as scc

    print(f"[qk] loading {args.model_path}")
    model, tok = build_sc_model(args.model_path, "mp",
                                mp_table=str(parent / "wrapper.json"))
    model.eval()
    device = next(model.parameters()).device

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    # Capture Q and K exactly where the SC qk product sees them: post-q_norm,
    # post-RoPE, post-GQA-repeat. Wrapping the attention matmul is the only
    # point where all three are already applied.
    stats: dict = defaultdict(lambda: {"mq": None, "mk": None, "n": 0})
    orig = scc._sc_attention_matmul_ab_t

    def wrapper(a, b, *, operator=None, block_idx=None, **kw):
        if operator == "qk" and block_idx is not None:
            # a: (B,H,N,D) queries, b: (B,H,M,D) repeated keys
            mq = a.detach().abs().amax(dim=(0, 1, 2)).float()
            mk = b.detach().abs().amax(dim=(0, 1, 2)).float()
            st = stats[int(block_idx)]
            st["mq"] = mq if st["mq"] is None else torch.maximum(st["mq"], mq)
            st["mk"] = mk if st["mk"] is None else torch.maximum(st["mk"], mk)
            st["n"] += 1
        return orig(a, b, operator=operator, block_idx=block_idx, **kw)

    scc._sc_attention_matmul_ab_t = wrapper
    try:
        with torch.no_grad():
            for i, w in enumerate(_iter_calib_windows(enc, args.ctx_len,
                                                      args.num_windows)):
                print(f"[qk] window {i + 1}/{args.num_windows}")
                model(w.unsqueeze(0).to(device))
    finally:
        scc._sc_attention_matmul_ab_t = orig

    if not stats:
        raise SystemExit("[qk] captured nothing — is the qk path SC-enabled?")

    out, rows = {}, []
    for blk in sorted(stats):
        mq = stats[blk]["mq"].clamp_min(1e-8)
        mk = stats[blk]["mk"].clamp_min(1e-8)
        s = (mq ** args.alpha) / (mk ** (1.0 - args.alpha))
        s = (s / s.mean()).clamp_min(1e-6)
        out[f"qk:b{blk}"] = [float(x) for x in s.cpu()]
        rows.append((blk, float(mq.max() / mq.min()), float(mk.max() / mk.min()),
                     float(s.max() / s.min())))

    print(f"\n[qk] per-dim SPREAD (max/min over head_dim), post-RoPE:")
    print(f"  {'layer':>6} {'|Q| spread':>12} {'|K| spread':>12} {'s spread':>10}")
    for blk, qs, ks, ss in rows[:8]:
        print(f"  {blk:6} {qs:12.1f} {ks:12.1f} {ss:10.1f}")
    if len(rows) > 8:
        print(f"  ... {len(rows) - 8} more layers")
    aq = sum(r[1] for r in rows) / len(rows)
    ak = sum(r[2] for r in rows) / len(rows)
    print(f"\n  MEAN |Q| spread = {aq:.1f}x   MEAN |K| spread = {ak:.1f}x")
    print(f"  (the synthetic test that gave 73-75% used 648x / 364x, so a much"
          f" smaller real spread means a much smaller real gain)")

    Path(args.out).write_text(json.dumps(
        {"alpha": args.alpha, "model": args.model_path,
         "mean_q_spread": aq, "mean_k_spread": ak,
         "scales": out}, indent=1))
    print(f"[qk] wrote {args.out}  ({len(out)} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
