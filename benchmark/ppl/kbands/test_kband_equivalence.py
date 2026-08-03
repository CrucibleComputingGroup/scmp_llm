"""Is a per-band K-split numerically equivalent to the unsplit call? (GPU)

The load-bearing assumption behind Phase 3. Three claims:

  A. CHUNK-ALIGNED splits preserve the math. _sc_matmul_bipolar_mlp_chunked
     builds k_table / rng_b / cum_indicator ONCE over chunk_d dims and reuses
     them for every chunk (sc/kernels.py), and the Owen mask index is
     `arange(D) % m` with D == chunk_d -- so the scramble depends on the
     WITHIN-chunk offset, not the absolute column. Each chunk also carries its
     own per-row quantization scale. A chunk relocated by a multiple of chunk_d
     is therefore bit-for-bit the same computation.
     NOT bit-identical overall: the kernel keeps ONE fp32 accumulator across a
     call's chunks (`output += partial * ...`) and a band split restarts it, so
     the identical addends are summed in a different ORDER. Hence a tolerance.

  B. NON-ALIGNED gathers are NOT equivalent -- they repack columns into
     different within-chunk offsets, changing both the mask and the group
     scale. This is why bands must be whole chunks. (The deployed
     protected-channel path gathers arbitrary indices, so it already lives with
     this; Phase 3 deliberately does not.)

  C. Per-band lengths at the same average MOVE the result, or there is nothing
     for the allocator to find.

Run on a GPU node: python benchmark/ppl/kbands/test_kband_equivalence.py
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
CHUNK_D = 128
HALVE = True
# Relative tolerance for fp32 accumulation REGROUPING across a K-split. Sums of
# ~10^4 addends of mixed sign; 1e-5 relative is far above float32 epsilon
# accumulation yet far below any real numeric divergence (a changed mask or
# scale moves results by O(1e-2) relative, three orders up).
REGROUP_RTOL = 1e-5


def call(x, w, stoc_len):
    return sc_matmul(
        x, w, granularity="per_row", mode="bipolar", sc_prec=SC_PREC,
        stoc_len=stoc_len, chunk_d=CHUNK_D, halve_bipolar_stoc_len=HALVE)


def split_call(x, w, chunk_levels):
    """Per-band SC matmul: chunk c runs at chunk_levels[c]; partials summed."""
    n_chunks = (x.shape[1] + CHUNK_D - 1) // CHUNK_D
    assert len(chunk_levels) == n_chunks
    out = torch.zeros(x.shape[0], w.shape[0], dtype=torch.float32,
                      device=x.device)
    for level in sorted(set(chunk_levels), reverse=True):
        cols = torch.cat([
            torch.arange(c * CHUNK_D, min((c + 1) * CHUNK_D, x.shape[1]),
                         device=x.device)
            for c, lv in enumerate(chunk_levels) if lv == level])
        out += call(x.index_select(1, cols).contiguous(),
                    w.index_select(1, cols).contiguous(), level)
    return out


def report(name, got, ref, expect_equal):
    d = (got - ref).abs()
    scale = ref.abs().max().clamp_min(1e-12)
    rel = (d.max() / scale).item()
    exact = torch.equal(got, ref)
    ok = (rel <= REGROUP_RTOL) if expect_equal else (rel > REGROUP_RTOL)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} "
          f"bit_exact={str(exact):<5} rel={rel:.3e}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    torch.manual_seed(0)
    dev = "cuda"
    N, M = 64, 512
    ok = True

    for D, label in ((1024, "even chunks"), (9144, "ragged tail (down_proj)")):
        n_chunks = (D + CHUNK_D - 1) // CHUNK_D
        x = torch.randn(N, D, device=dev)
        # heterogeneous chunk magnitudes -- the reason per-group MP can pay
        x[:, 2 * CHUNK_D:3 * CHUNK_D] *= 30.0
        x[:, 5 * CHUNK_D:6 * CHUNK_D] *= 10.0
        w = torch.randn(M, D, device=dev) * 0.05
        print(f"\n=== D={D} ({label}), {n_chunks} chunks ===")

        print("A. chunk-aligned split at a uniform length (expect equivalent):")
        for L in (128, 64, 48, 32):
            ref = call(x, w, L)
            ok &= report(f"L={L}, one band", split_call(x, w, [L] * n_chunks),
                         ref, True)
        L = 64
        ref = call(x, w, L)
        half = n_chunks // 2
        got = (call(x[:, :half * CHUNK_D].contiguous(),
                    w[:, :half * CHUNK_D].contiguous(), L)
               + call(x[:, half * CHUNK_D:].contiguous(),
                      w[:, half * CHUNK_D:].contiguous(), L))
        ok &= report(f"L={L}, two aligned bands", got, ref, True)

        print("B. non-chunk-aligned gather (expect NOT equivalent):")
        perm = torch.randperm(D, device=dev)
        a, b = perm[:D // 2].contiguous(), perm[D // 2:].contiguous()
        got = (call(x.index_select(1, a).contiguous(),
                    w.index_select(1, a).contiguous(), L)
               + call(x.index_select(1, b).contiguous(),
                      w.index_select(1, b).contiguous(), L))
        ok &= report(f"L={L}, random channel split", got, ref, False)

    print("\nC. per-band lengths at ~the same average (expect a real change):")
    D = 1024
    n_chunks = D // CHUNK_D
    x = torch.randn(N, D, device=dev)
    x[:, 2 * CHUNK_D:3 * CHUNK_D] *= 30.0
    w = torch.randn(M, D, device=dev) * 0.05
    fp = x.float() @ w.float().t()
    uni = call(x, w, 64)
    lv = [56] * n_chunks
    lv[2] = 128                      # the outlier chunk runs long
    mean = sum(lv) / len(lv)
    mp = split_call(x, w, lv)
    err = lambda y: ((y - fp).norm() / fp.norm()).item()
    print(f"  uniform L=64                    rel_err={err(uni):.5e}")
    print(f"  per-band mean={mean:.1f}            rel_err={err(mp):.5e}")
    delta = (err(mp) - err(uni)) / err(uni) * 100
    print(f"  -> per-band moves reconstruction error {delta:+.2f}% "
          f"at {'~iso' if abs(mean - 64) < 2 else 'NON-iso'} budget")
    if torch.equal(mp, uni):
        print("  [FAIL] per-band produced an identical tensor")
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
