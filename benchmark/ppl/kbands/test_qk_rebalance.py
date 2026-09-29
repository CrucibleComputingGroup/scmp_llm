"""Q/K contracted-dim rebalance: exact in FP, and does it help SC? (GPU)

attention scores = sum_d Q[t,d] K[s,d]. Scaling Q[:,d] by 1/s_d and K[:,d] by
s_d leaves that sum EXACTLY unchanged, while moving the per-row absmax of both
operands -- and the per-row absmax is precisely what sets the SC quantization
scale. qk/av is ~90% of dispatch rows and is the one operator class no PTQ
front-end reaches (SmoothQuant and AWQ both stop at SCLinear), so this is the
transform attention has never had.

Three claims:
  A. FP invariance. (Q/s) @ (K*s)^T == Q @ K^T to float tolerance, for ANY
     positive s. If this fails the whole idea is unsound.
  B. It is applied INSIDE sc_matmul, hence AFTER RoPE -- so unlike folding into
     q_norm/k_norm it carries no rotate_half pair constraint and all head_dim
     dims are free. (A fold would require s[d] == s[d+head_dim/2].)
  C. SC error actually falls. With Q/K whose per-dim magnitudes are imbalanced
     (the real situation), the migration scale s_d = mq_d^a / mk_d^(1-a) should
     beat s = 1 at the same stream length. If it does not, the lever is dead
     and no search over alpha will save it.

Run on a GPU node: python benchmark/ppl/kbands/test_qk_rebalance.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "kernels"))

import torch  # noqa: E402
from scmp_kernels.sc.matmul import sc_matmul  # noqa: E402

SC_PREC = 8
HALVE = True


def sc_qk(q, k, stoc_len, s=None):
    """One SC qk product on the deployed attention path (per_row, no chunking)."""
    return sc_matmul(q, k, granularity="per_row", mode="bipolar",
                     sc_prec=SC_PREC, stoc_len=stoc_len,
                     halve_bipolar_stoc_len=HALVE, smooth_scales=s)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    torch.manual_seed(0)
    dev = "cuda"
    N, M, D = 256, 256, 128          # one head: 256 queries x 256 keys x head_dim

    # Imbalanced per-dim magnitudes: Q large on some dims, K large on others.
    # This is the structure a rebalance can exploit; with balanced operands
    # there is nothing to migrate.
    gq = torch.exp(torch.randn(D, device=dev) * 1.2)
    gk = torch.exp(torch.randn(D, device=dev) * 1.2)
    q = torch.randn(N, D, device=dev) * gq
    k = torch.randn(M, D, device=dev) * gk
    fp = q.float() @ k.float().t()

    print(f"N={N} M={M} head_dim={D}")
    print(f"per-dim |Q| spread {gq.max() / gq.min():.1f}x, "
          f"|K| spread {gk.max() / gk.min():.1f}x\n")

    ok = True

    print("A. FP invariance of the rebalance (expect exact to float tol):")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        mq = q.abs().amax(0).clamp_min(1e-8)
        mk = k.abs().amax(0).clamp_min(1e-8)
        s = (mq ** a) / (mk ** (1.0 - a))
        s = (s / s.mean()).clamp_min(1e-6)
        got = (q / s).float() @ (k * s).float().t()
        r = rel(got, fp)
        good = r < 1e-5
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] alpha={a:.2f}  rel={r:.3e}")

    print("\nB. no RoPE pair constraint (applied inside the matmul, post-RoPE):")
    # a deliberately pair-INCONSISTENT s would break a q_norm fold; here it is fine
    s_odd = torch.rand(D, device=dev) + 0.5
    got = (q / s_odd).float() @ (k * s_odd).float().t()
    r = rel(got, fp)
    good = r < 1e-5
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] free per-dim s (s[d] != s[d+64])  "
          f"rel={r:.3e}   <- a q_norm fold could NOT do this")

    print("\nC. does it reduce SC error at fixed stream length?")
    print(f"  {'stoc_len':>9} {'s=1 (today)':>13} {'best alpha':>11} "
          f"{'best rel_err':>13} {'improvement':>12}")
    for L in (128, 64, 48, 32):
        base = rel(sc_qk(q, k, L), fp)
        best, best_a = base, None
        for a in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
            mq = q.abs().amax(0).clamp_min(1e-8)
            mk = k.abs().amax(0).clamp_min(1e-8)
            s = (mq ** a) / (mk ** (1.0 - a))
            s = (s / s.mean()).clamp_min(1e-6)
            r = rel(sc_qk(q, k, L, s), fp)
            if r < best:
                best, best_a = r, a
        imp = (1 - best / base) * 100
        print(f"  {L:9} {base:13.5e} {str(best_a):>11} {best:13.5e} "
              f"{imp:11.1f}%")
        if best_a is None:
            print(f"      (no alpha beat s=1 at L={L})")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
