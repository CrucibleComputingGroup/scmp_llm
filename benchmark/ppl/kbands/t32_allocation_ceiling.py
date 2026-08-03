"""What is the TRUE ceiling of per-group allocation at t32? (GPU)

An earlier diagnostic put the allocation ceiling at 15.8% of the SC-vs-INT8 gap.
That number came from a restricted oracle and should not be trusted as the
ceiling of the idea:

  * it assigned ONE length per chunk, SHARED ACROSS ALL ROWS -- collapsing the
    512 x 72 decision space to 72;
  * it assumed E_c(L) = A_c / L, a MODEL fitted from a single measured length,
    rather than measured curves;
  * it therefore bounded one restricted policy class, not per-group MP.

This measures the real ladder, all at the SAME mean cost, using MEASURED
per-(row, chunk) error curves and an EXACT Lagrangian water-fill (separable
objective, so bisecting the multiplier is optimal, not heuristic):

  0. UNIFORM      one L everywhere                       (the floor)
  1. PER-ROW      one L_r per row                        (what MP does today)
  2. PER-CHUNK    one L_c per chunk                      (the earlier oracle)
  3. PER-(ROW,CHUNK)  L_{r,c} free                       (the full per-group space)

Measurement is cheap because one SC call per (chunk, length) yields the error
for EVERY row at once: 72 chunks x |grid| lengths of 128-wide matmuls.

Run: python benchmark/ppl/kbands/t32_allocation_ceiling.py --target 32
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


def sc(x, w, L):
    return sc_matmul(x, w, granularity="per_row", mode="bipolar",
                     sc_prec=SC_PREC, stoc_len=int(L), chunk_d=CHUNK_D,
                     halve_bipolar_stoc_len=HALVE)


def waterfill(err, cost, budget_mean, grid):
    """Exact separable water-fill: min sum_i err[i, L_i] s.t. mean cost = budget.

    err  : (n_items, n_levels) measured error
    cost : (n_levels,) cycles per level
    Bisect the Lagrange multiplier; for a separable objective this is optimal.
    Returns (chosen level index per item, achieved mean cost, total error).
    """
    lo, hi = 0.0, 1.0
    def alloc(lam):
        j = (err + lam * cost.unsqueeze(0)).argmin(dim=1)
        return j, cost[j].mean().item()
    while alloc(hi)[1] > budget_mean and hi < 1e14:
        hi *= 10.0
    best = None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        j, c = alloc(mid)
        if c <= budget_mean:
            best = j; hi = mid
        else:
            lo = mid
    if best is None:
        best = torch.full((err.shape[0],), len(grid) - 1, dtype=torch.long,
                          device=err.device)
    j = best
    return j, cost[j].mean().item(), err.gather(1, j.unsqueeze(1)).sum().item()


def capture_real(model_key: str, op: str, block: int, n_rows: int):
    """Capture one SCLinear's real (x, W) at the point the SC kernel sees them.

    Uses the DEPLOYED loader so smoothing, the hybrid mask and protected
    channels are all exactly as the cell runs them, and trims x/W to whole
    chunks so the chunk grid matches the runtime's.
    """
    import os
    parent = (f"/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/"
              f"configs/{model_key}/target32")
    os.environ.setdefault("SC_OWEN_MODE", "bitrev")
    os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")
    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = f"{parent}/wrapper.json"
    if os.path.isfile(f"{parent}/hybrid_config.json"):
        os.environ["SC_HYBRID_CONFIG_JSON"] = f"{parent}/hybrid_config.json"
    os.environ.setdefault(
        "ACT_SCALES_DIR",
        "/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/act_scales")
    import json as _json
    hf = _json.load(open(f"{parent}/table.json"))["model_path"]

    from benchmark.quant.eval_quant import build_sc_model
    from benchmark.ppl.calibrate_mp_thresholds import _iter_calib_windows
    from model.sc_common import SCLinear
    from datasets import load_dataset

    model, tok = build_sc_model(hf, "mp", mp_table=f"{parent}/wrapper.json")
    model.eval()
    dev = next(model.parameters()).device
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    grab = {}

    def hook(mod, inp, out):
        if getattr(mod, "_sc_op_name", None) == op and \
           getattr(mod, "_sc_block_idx", None) == block and "x" not in grab:
            t = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if t.shape[0] > n_rows:
                t = t[:n_rows]
            grab["x"] = t.contiguous()
            grab["w"] = mod.weight.float().contiguous()

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, SCLinear)]
    try:
        with torch.no_grad():
            for w_ in _iter_calib_windows(enc, 2048, 1):
                model(w_.unsqueeze(0).to(dev))
                if "x" in grab:
                    break
    finally:
        for h in hs:
            h.remove()
    if "x" not in grab:
        raise SystemExit(f"[ceiling] never saw {op} block {block}")
    x, w = grab["x"], grab["w"]
    keep = (x.shape[1] // CHUNK_D) * CHUNK_D          # whole chunks only
    return x[:, :keep].contiguous(), w[:, :keep].contiguous()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=32)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--nchunks", type=int, default=72)
    ap.add_argument("--real", metavar="MODEL",
                    help="capture REAL activations from this mp_best model "
                         "(4B|llama8B|14B|30B) instead of synthetic tensors. The "
                         "-44.8%% ceiling was measured on a synthetic heavy-tailed "
                         "draw; the whole per-(row,chunk) plan rests on it, so it "
                         "must be reproduced on activations the model actually "
                         "produces before any of it is trusted.")
    ap.add_argument("--op", default="down_proj")
    ap.add_argument("--block", type=int, default=12)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    torch.manual_seed(0)
    dev = "cuda"
    L0 = args.target
    N, nch, M = args.rows, args.nchunks, 2560
    D = nch * CHUNK_D
    grid = [8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 112, 128]

    if args.real:
        x, w = capture_real(args.real, args.op, args.block, N)
        N, D = x.shape
        M = w.shape[0]
        nch = D // CHUNK_D
        print(f"REAL activations: {args.real} {args.op} block {args.block} "
              f"-> x{tuple(x.shape)} w{tuple(w.shape)}, {nch} chunks")
    else:
        ch = torch.exp(torch.randn(D, device=dev) * 1.1)
        ch[torch.randperm(D, device=dev)[:D // 200]] *= 30.0
        x = torch.randn(N, D, device=dev) * ch
        w = torch.randn(M, D, device=dev) * 0.05
        print("SYNTHETIC tensors (pass --real <MODEL> to use captured activations)")

    print(f"measuring per-(row,chunk) error: {nch} chunks x {len(grid)} lengths")
    # e[r, c, l] = squared error of row r's partial product for chunk c at grid[l]
    e = torch.zeros(N, nch, len(grid), device=dev)
    for c in range(nch):
        cols = torch.arange(c * CHUNK_D, (c + 1) * CHUNK_D, device=dev)
        xc = x.index_select(1, cols).contiguous()
        wc = w.index_select(1, cols).contiguous()
        fpc = xc.float() @ wc.float().t()
        for li, L in enumerate(grid):
            e[:, c, li] = ((sc(xc, wc, L) - fpc) ** 2).sum(dim=1)
    cost = torch.tensor([float(g) for g in grid], device=dev)
    l0 = grid.index(L0)

    base = e[:, :, l0].sum().item()
    print(f"\nAll policies at mean cost = {L0} cycles. Error relative to UNIFORM:\n")
    print(f"  {'policy':34} {'decisions':>12} {'total err':>12} {'vs uniform':>11}")
    print(f"  {'0. UNIFORM (one L everywhere)':34} {1:12} {base:12.4e} "
          f"{0.0:10.1f}%")

    # 1. PER-ROW: one L per row -> sum the chunk errors within a row first
    er = e.sum(dim=1)                                  # (N, n_levels)
    _, c1, t1 = waterfill(er, cost, float(L0), grid)
    print(f"  {'1. PER-ROW (today MP)':34} {N:12} {t1:12.4e} "
          f"{(t1/base-1)*100:10.1f}%")

    # 2. PER-CHUNK: one L per chunk, shared across rows
    ec = e.sum(dim=0)                                  # (nch, n_levels)
    _, c2, t2 = waterfill(ec, cost, float(L0), grid)
    print(f"  {'2. PER-CHUNK (earlier oracle)':34} {nch:12} {t2:12.4e} "
          f"{(t2/base-1)*100:10.1f}%")

    # 3. PER-(ROW,CHUNK): the full per-group space
    erc = e.reshape(N * nch, len(grid))
    _, c3, t3 = waterfill(erc, cost, float(L0), grid)
    print(f"  {'3. PER-(ROW,CHUNK) full space':34} {N*nch:12} {t3:12.4e} "
          f"{(t3/base-1)*100:10.1f}%")

    print(f"\n  realized mean cost: per-row {c1:.2f}, per-chunk {c2:.2f}, "
          f"per-(row,chunk) {c3:.2f}  (budget {L0})")
    print(f"\n  INCREMENT of the full per-group space over what MP does today:")
    print(f"    per-row -> per-(row,chunk):  {(t3/t1-1)*100:+.1f}% error")
    print(f"    per-chunk -> per-(row,chunk):{(t3/t2-1)*100:+.1f}% error")
    print(f"\n  These are SQUARED-ERROR reductions on the SC partial products, an")
    print(f"  upper bound on the allocation axis -- transfer to PPL is separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
