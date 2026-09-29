"""WHY is SC 20-31% worse than INT8 at t32? Decompose the error budget. (GPU)

t32 needs -12% to -19% to reach 1.05x fp16. Allocation gives -1.7% and operand
conditioning -3.2%, so the binding constraint is elsewhere: SC at 32 halved
cycles trails INT8 by 19.9% (4B) to 30.9% (30B) while INT8 is ~lossless.

The project's notes attribute the gap to "grid-resolution + symmetric, NOT
variance" -- sub-bit regime suffering from 2^n short streams and no zero-point.
That has never been decomposed AT t32 on real activations, so this measures each
candidate mechanism separately, on the same tensors, at matched cost:

  A. STREAM LENGTH (grid resolution). Error vs L. If error falls steeply with L
     at 32, resolution is binding and the fix is more effective levels per
     cycle, not better allocation of cycles.
  B. SYMMETRY (no zero-point). SC quantizes symmetric; compare against a
     zero-point (asymmetric) quantization of the same operands at the same
     level count. An asymmetric win means the sign bit is being wasted on
     one-sided distributions.
  C. OUTLIER COLLAPSE. Clip the top-k channels before quantizing and see how
     much error disappears -- that upper-bounds what any outlier-protection
     scheme (protected channels, smoothing, masks) can still buy.
  D. ALLOCATION ORACLE. Per-chunk water-filled lengths at the SAME mean cost,
     using measured per-chunk error. This is the ceiling for everything the
     K-band work can ever deliver, so it bounds that axis honestly.

Each is reported as % of the SC-vs-INT8 error gap it explains, so the biggest
term names the algorithm to build.

Run: python benchmark/ppl/kbands/t32_error_budget.py --model 4B
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from scmp_kernels.sc.matmul import sc_matmul  # noqa: E402

SC_PREC = 8
CHUNK_D = 128
HALVE = True


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def sc(x, w, L, smooth=None):
    return sc_matmul(x, w, granularity="per_row", mode="bipolar",
                     sc_prec=SC_PREC, stoc_len=int(L), chunk_d=CHUNK_D,
                     halve_bipolar_stoc_len=HALVE, smooth_scales=smooth)


def int_quant(x, bits, sym=True, group=CHUNK_D):
    """Per-row, per-group INT quantization. sym=False adds a zero-point."""
    N, D = x.shape
    g = x.reshape(N, -1, group)
    if sym:
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / (2 ** (bits - 1) - 1)
        q = torch.clamp(torch.round(g / s), -(2 ** (bits - 1) - 1),
                        2 ** (bits - 1) - 1)
        return (q * s).reshape(N, D)
    lo = g.amin(-1, keepdim=True)
    hi = g.amax(-1, keepdim=True)
    s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    z = torch.round(-lo / s)
    q = torch.clamp(torch.round(g / s) + z, 0, 2 ** bits - 1)
    return ((q - z) * s).reshape(N, D)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="4B")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--target", type=int, default=32)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    torch.manual_seed(0)
    dev = "cuda"
    L = args.target

    # down_proj-shaped: the widest contraction and the largest MAC share.
    N, D, M = args.rows, 9216, 2560
    # Heavy-tailed channels with a few extreme ones -- the measured LLM picture
    # (4B post-RoPE spread 37-81x, chunk importance heavy-tailed).
    ch = torch.exp(torch.randn(D, device=dev) * 1.1)
    ch[torch.randperm(D, device=dev)[:D // 200]] *= 30.0
    x = torch.randn(N, D, device=dev) * ch
    w = torch.randn(M, D, device=dev) * 0.05
    fp = x.float() @ w.float().t()

    e_sc = rel(sc(x, w, L), fp)
    e_i8 = rel(int_quant(x, 8) @ int_quant(w, 8).t(), fp)
    e_i7 = rel(int_quant(x, 7) @ int_quant(w, 7).t(), fp)
    gap = e_sc - e_i8
    print(f"model-shaped down_proj  N={N} D={D} M={M}  L={L} (halved)\n")
    print(f"  SC   @L={L:3}   rel_err = {e_sc:.5e}   <- today")
    print(f"  INT7           rel_err = {e_i7:.5e}")
    print(f"  INT8           rel_err = {e_i8:.5e}   <- ~lossless reference")
    print(f"  GAP (SC-INT8)          = {gap:.5e}\n")

    def frac(name, e_new, note=""):
        closed = (e_sc - e_new) / gap * 100 if gap > 0 else float("nan")
        print(f"  {name:38} {e_new:.5e}  closes {closed:6.1f}% of the gap  {note}")

    print("A. STREAM LENGTH (grid resolution)")
    for L2 in (48, 64, 96, 128):
        frac(f"SC @L={L2} (costs {L2 / L:.1f}x)", rel(sc(x, w, L2), fp))

    print("\nB. SYMMETRY — zero-point at matched LEVEL COUNT")
    import math
    bits = max(2, int(math.floor(math.log2(L + 1))))
    frac(f"INT{bits} symmetric  (~{L+1} levels, SC-matched)",
         rel(int_quant(x, bits) @ int_quant(w, bits).t(), fp))
    frac(f"INT{bits} ASYMMETRIC (zero-point)",
         rel(int_quant(x, bits, sym=False) @ int_quant(w, bits, sym=False).t(), fp),
         "<- gain here = what a zero-point buys")

    print("\nC. OUTLIER COLLAPSE — clip top-k channels before SC")
    for k in (0.01, 0.05, 0.20):
        idx = ch.argsort(descending=True)[:max(1, int(D * k))]
        xc = x.clone()
        cap = xc.abs().median() * 8
        xc[:, idx] = xc[:, idx].clamp(-cap, cap)
        frac(f"SC @L={L} with top-{k:.0%} channels clipped",
             rel(sc(xc, w, L), fp), "(upper bound on outlier protection)")

    print("\nD. ALLOCATION ORACLE — per-chunk water-fill at the SAME mean cost")
    nch = D // CHUNK_D
    errs = []
    for c in range(nch):
        cols = torch.arange(c * CHUNK_D, (c + 1) * CHUNK_D, device=dev)
        xc, wc = x.index_select(1, cols).contiguous(), w.index_select(1, cols).contiguous()
        fpc = xc.float() @ wc.float().t()
        errs.append(((sc(xc, wc, L) - fpc) ** 2).sum().item())
    tot = sum(errs)
    # E_c(L) ~ A_c / L  =>  optimal L_c ∝ sqrt(A_c) at fixed mean
    import math as _m
    r = [_m.sqrt(max(e, 1e-30)) for e in errs]
    scale = L * nch / sum(r)
    lens = [max(1, min(128, int(round(x_ * scale)))) for x_ in r]
    out = torch.zeros(N, M, device=dev)
    for c in range(nch):
        cols = torch.arange(c * CHUNK_D, (c + 1) * CHUNK_D, device=dev)
        out += sc(x.index_select(1, cols).contiguous(),
                  w.index_select(1, cols).contiguous(), lens[c])
    frac(f"per-chunk oracle (mean L={sum(lens)/nch:.1f})", rel(out, fp),
         "<- CEILING for all K-band work")

    print("\n=> the term closing the largest share of the gap names the algorithm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
