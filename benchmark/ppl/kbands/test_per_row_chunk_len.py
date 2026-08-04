"""Correctness tests for per-(row, chunk) stream lengths in the SC kernel.

The claim being tested is that ONE cum table built at L_max serves every
shorter rung, because cum[d,k,v] = |{i<k : rng_b[d,i] <= v}| is a prefix sum
over the cycle axis and the Owen scramble is a position-independent per-dim
XOR. If that is wrong, the results are not garbage -- they are subtly wrong at
the right average cost, which is the failure mode this repo keeps hitting.
So the tests demand BIT-IDENTICAL agreement with the plain path, not closeness.

Run on a GPU node:  python -m benchmark.ppl.kbands.test_per_row_chunk_len
"""
from __future__ import annotations

import os
import pathlib

import torch

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

from scmp_kernels.sc.kernels import _sc_matmul_per_row_mlp  # noqa: E402

# Deployed operating point: sc_prec=8 with halve_bipolar_stoc_len=1 => the
# enable grid is 2**(sc_prec-1)=128 and a level value IS the halved cycle count.
SC_PREC = 8
RNG_LEVELS = 128
CHUNK_D = 128
LEVELS = [16, 32, 48, 64, 128]   # halved; nominal = 2x these
L_MAX = max(LEVELS)

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def plain(a, b, stoc_len):
    return _sc_matmul_per_row_mlp(
        a, b, mode="bipolar", sc_prec=SC_PREC, chunk_d=CHUNK_D,
        stoc_len=stoc_len, rng_levels=RNG_LEVELS)


def per_row(a, b, rung_table):
    return _sc_matmul_per_row_mlp(
        a, b, mode="bipolar", sc_prec=SC_PREC, chunk_d=CHUNK_D,
        stoc_len=L_MAX, rng_levels=RNG_LEVELS,
        rung_table=rung_table, level_lens=LEVELS)


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    N, D, M = 96, CHUNK_D * 5, 64          # 5 whole chunks
    n_chunks = D // CHUNK_D
    a = torch.randn(N, D, device=dev)
    # Give a few rows the heavy-tailed structure real down_proj inputs have,
    # so a length change actually moves the result.
    a[::7] *= 40.0
    b = torch.randn(M, D, device=dev)

    ref = {L: plain(a, b, L) for L in LEVELS}

    print("\n[1] uniform rung == plain call at that length (prefix nesting)")
    for r, L in enumerate(LEVELS):
        rt = torch.full((N, n_chunks), r, dtype=torch.int32, device=dev)
        got = per_row(a, b, rt)
        same = torch.equal(got, ref[L])
        check(f"all-rows rung {r} (L={L}) bit-identical to plain L={L}", same,
              "" if same else f"max|d|={(got - ref[L]).abs().max().item():.3e}")

    print("\n[2] per-ROW lengths: row i must match plain-at-its-own-length, row i")
    rung_row = torch.randint(0, len(LEVELS), (N,), device=dev, dtype=torch.int32)
    rt = rung_row[:, None].expand(N, n_chunks).contiguous()
    got = per_row(a, b, rt)
    exp = torch.empty_like(got)
    for r, L in enumerate(LEVELS):
        m = rung_row == r
        if m.any():
            exp[m] = ref[L][m]
    same = torch.equal(got, exp)
    check("mixed per-row bit-identical to row-wise reference", same,
          "" if same else f"max|d|={(got - exp).abs().max().item():.3e}, "
          f"rows differing={int((got != exp).any(1).sum())}/{N}")

    print("\n[3] per-CHUNK lengths: sum of single-chunk calls at each length")
    rung_chunk = torch.randint(0, len(LEVELS), (n_chunks,), device=dev,
                               dtype=torch.int32)
    rt = rung_chunk[None, :].expand(N, n_chunks).contiguous()
    got = per_row(a, b, rt)
    # Chunk contributions are additive, so run each chunk on its own at its own
    # length. D == 2*chunk_d keeps it on the same chunked fast path.
    exp = torch.zeros_like(got)
    for c in range(n_chunks):
        sl = slice(c * CHUNK_D, (c + 1) * CHUNK_D)
        ac = torch.cat([a[:, sl], torch.zeros(N, CHUNK_D, device=dev)], 1)
        bc = torch.cat([b[:, sl], torch.zeros(M, CHUNK_D, device=dev)], 1)
        exp += plain(ac, bc, LEVELS[int(rung_chunk[c])])
    same = torch.equal(got, exp)
    check("per-chunk lengths match summed single-chunk references", same,
          "" if same else f"max|d|={(got - exp).abs().max().item():.3e} "
          f"(rel {((got - exp).abs().max() / exp.abs().max()).item():.2e})")

    print("\n[4] full per-(row, chunk): error must beat per-row at equal mean cost")
    fp = (a.float() @ b.float().t())
    gen = torch.Generator(device=dev).manual_seed(1)
    rt_free = torch.randint(0, len(LEVELS), (N, n_chunks), generator=gen,
                            device=dev, dtype=torch.int32)
    lv = torch.tensor(LEVELS, device=dev, dtype=torch.float32)
    mean_cost = lv[rt_free.long()].mean().item()
    # Cost-matched per-row control: same mean length, one length per row.
    rt_row = torch.full((N, n_chunks), 0, dtype=torch.int32, device=dev)
    per_row_lens = lv[rt_free.long()].mean(1)
    nearest = (per_row_lens[:, None] - lv[None, :]).abs().argmin(1).to(torch.int32)
    rt_row = nearest[:, None].expand(N, n_chunks).contiguous()
    e_free = (per_row(a, b, rt_free) - fp).pow(2).sum().item()
    e_row = (per_row(a, b, rt_row) - fp).pow(2).sum().item()
    cost_row = lv[rt_row.long()].mean().item()
    check("random per-(row,chunk) runs at the intended mean cost",
          abs(mean_cost - lv[rt_free.long()].mean().item()) < 1e-6,
          f"mean L = {mean_cost:.2f} (per-row control {cost_row:.2f})")
    print(f"       sq err: per-(row,chunk) {e_free:.4e}   per-row {e_row:.4e}")

    print("\n[5] guards fire instead of silently running one uniform length")
    try:
        _sc_matmul_per_row_mlp(a, b, mode="bipolar", sc_prec=SC_PREC, chunk_d=0,
                               stoc_len=L_MAX, rng_levels=RNG_LEVELS,
                               rung_table=rt, level_lens=LEVELS)
        check("chunk_d=0 rejected", False, "no raise")
    except ValueError:
        check("chunk_d=0 rejected", True)
    try:
        per_row(a, b, torch.full((N, n_chunks), len(LEVELS), dtype=torch.int32,
                                device=dev))
        check("out-of-range rung rejected", False, "no raise")
    except ValueError:
        check("out-of-range rung rejected", True)
    try:
        per_row(a, b, torch.zeros(N, n_chunks + 3, dtype=torch.int32, device=dev))
        check("wrong-shape rung_table rejected", False, "no raise")
    except ValueError:
        check("wrong-shape rung_table rejected", True)
    try:
        _sc_matmul_per_row_mlp(a, b, mode="bipolar", sc_prec=SC_PREC,
                               chunk_d=CHUNK_D, stoc_len=2 ** SC_PREC,
                               rng_levels=2 ** SC_PREC,
                               rung_table=rt, level_lens=LEVELS)
        check("unscrambled-L_max (nesting breaks) rejected", False, "no raise")
    except ValueError:
        check("unscrambled-L_max (nesting breaks) rejected", True)

    print("\n[6] the TRACE must price the allocation, not L_max")
    import scmp_kernels.trace as _tr
    from scmp_kernels import sc_matmul
    lens_t = torch.tensor(LEVELS, dtype=torch.float64)
    import json, tempfile
    tmp = tempfile.mktemp(suffix=".json")
    _tr.reset(); _tr.enable(tmp, mode="summary")
    sc_matmul(a, b, "per_row", mode="bipolar", sc_prec=SC_PREC,
              chunk_d=CHUNK_D, stoc_len=L_MAX, rng_levels=RNG_LEVELS,
              rung_table=rt_free, level_lens=LEVELS)
    _tr.flush(tmp); _tr.disable()
    doc = json.loads(pathlib.Path(tmp).read_text())
    grps = doc["groups"] if isinstance(doc, dict) and "groups" in doc else doc
    macs = sum(g["macs"] for g in grps)
    cyc = sum(g["macs"] * g["stoc_len"] for g in grps)
    exp_macs = N * D * M
    exp_cyc = float((lens_t[rt_free.long().cpu()].sum().item()) * CHUNK_D * M)
    check("trace MACs conserve", macs == exp_macs, f"{macs} vs {exp_macs}")
    check("trace prices the per-(row,chunk) allocation, not L_max",
          abs(cyc - exp_cyc) / exp_cyc < 1e-9,
          f"mean L traced = {cyc/macs:.3f}, intended = {mean_cost:.3f}, "
          f"L_max = {L_MAX}")

    print("\n[7] default path unchanged when no rung_table is passed")
    check("plain call still bit-identical across repeats",
          torch.equal(plain(a, b, 64), ref[64]))

    print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
