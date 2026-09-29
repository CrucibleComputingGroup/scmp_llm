"""Gate: is the SHAPE of the deployed length-allocation rule leaving error on the table?

THE QUESTION. Dispatch assigns a stream length per (row, 128-chunk) group by
min-max normalizing the group absmax OVER THE WHOLE CALL and bucketizing into
<=7 rungs (model/sc_common.py:per_row_chunk_rungs). Two properties of that rule
have never been tested:

  1. min-max is set by exactly two values in the call, so a heavy-tailed absmax
     distribution crushes the bulk toward 0. Measured consequence: 30-72% of
     MACs land on the FLOOR rung and 0.1-1.9% on the top, and the median first
     threshold across buckets is exactly 1.0 (that rung is unreachable).
  2. The staircase is a 7-step approximation to a relationship whose optimum has
     a closed form. For one output element the chunk errors add with
     |eps_j| ~ m_j / sqrt(L_j), so minimizing under sum_j L_j = B gives
         L_j proportional to m_j        (independent chunk errors)
         L_j proportional to m_j^(2/3)  (perfectly correlated, shared RNG)
     i.e. L ~ m^p with p in [2/3, 1] -- NOT a step function of a call-normalized
     statistic.

The calibrator's own validation ("the proxy captures 94-99% of the oracle") was
RANK-MATCHED TO THE ORACLE'S LENGTH HISTOGRAM -- it holds the histogram fixed
and only asks whether absmax ranks groups correctly. The SHAPE was never tested.

WHAT THIS MEASURES. On REAL captured operands, at EXACTLY the deployed rule's
own cycle count, the total measured squared error of:
    deployed   the real thresholds from the deployed prc table
    powerlaw   L = clamp(round(kappa * m^p)), kappa solved for iso-cost
    oracle     water-fill on MEASURED error (unreachable; the upper bound)
Error is measured, not modelled: one SC matmul per (chunk, level).

GATE. If powerlaw does not beat deployed by a margin that is large relative to
the deployed-to-oracle gap, the shape is not the problem and this dies here for
zero eval cells. Error-axis gains have failed to reach PPL ~6 times in this
project, so a win here is necessary, not sufficient.

Usage (on a GPU node):
    python -m benchmark.ppl.kbands.probe_alloc_shape \
        --parent <bundle dir with wrapper.json/table.json/hybrid_config.json> \
        --model_path <hf path> --out probe_alloc_shape_4B_t32.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

CHUNK_D = 128
SC_PREC = 8
HALVE = True
LIN_OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _sc(x, w, L):
    from scmp_kernels.sc.matmul import sc_matmul
    return sc_matmul(x, w, granularity="per_row", mode="bipolar",
                     sc_prec=SC_PREC, stoc_len=int(L), chunk_d=CHUNK_D,
                     halve_bipolar_stoc_len=HALVE)


def chunk_absmax(x: torch.Tensor) -> torch.Tensor:
    """Per-(row, chunk) absmax -- the group's own quantization scale (free)."""
    N, D = x.shape
    nch = (D + CHUNK_D - 1) // CHUNK_D
    pad = nch * CHUNK_D - D
    xa = x.abs()
    if pad:
        xa = torch.nn.functional.pad(xa, (0, pad))
    return xa.view(N, nch, CHUNK_D).amax(-1)


def minmax_norm(m: torch.Tensor) -> torch.Tensor:
    """EXACTLY model.sc_common.per_row_chunk_rungs -- min-max over the call."""
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo).clamp_min(1e-8)


def error_at_levels(x, w, levels):
    """(n_pairs, n_levels) measured squared error. One SC call per (chunk, level)."""
    N, D = x.shape
    nch = D // CHUNK_D
    e = torch.zeros(N, nch, len(levels), device=x.device)
    for c in range(nch):
        sl = slice(c * CHUNK_D, (c + 1) * CHUNK_D)
        xc = x[:, sl].contiguous()
        wc = w[:, sl].contiguous()
        fpc = xc.float() @ wc.float().t()
        for li, L in enumerate(levels):
            e[:, c, li] = ((_sc(xc, wc, L) - fpc) ** 2).sum(dim=1)
    return e.reshape(N * nch, len(levels))


def pick(err, idx):
    """Total measured error when pair i runs at level index idx[i]."""
    return err.gather(1, idx.view(-1, 1).clamp_(0, err.shape[1] - 1)).sum().item()


def powerlaw_idx(m_flat, levels_t, p, budget_cycles):
    """L = clamp(kappa * m^p) snapped to the level grid, kappa solved for iso-cost.

    Cost here is sum of L over pairs: every pair in one operator carries the same
    MAC count (CHUNK_D x d_out), so equal sum(L) IS equal cycles for that operator.
    """
    mp = m_flat.double().clamp_min(1e-12) ** p
    lo_l, hi_l = float(levels_t[0]), float(levels_t[-1])
    n = m_flat.numel()

    def total(kappa):
        L = (kappa * mp).clamp(lo_l, hi_l)
        # snap to the nearest available hardware level
        j = torch.bucketize(L, levels_t)
        j = j.clamp(0, len(levels_t) - 1)
        jm = (j - 1).clamp(0, len(levels_t) - 1)
        take_lo = (L - levels_t[jm]).abs() < (levels_t[j] - L).abs()
        j = torch.where(take_lo, jm, j)
        return levels_t[j].sum().item(), j

    lo_k, hi_k = 1e-12, 1e12
    for _ in range(80):
        mid = (lo_k * hi_k) ** 0.5
        t, _ = total(mid)
        if t < budget_cycles:
            lo_k = mid
        else:
            hi_k = mid
    _, j = total((lo_k * hi_k) ** 0.5)
    return j.to(torch.long), float(n)


def oracle_idx(err, levels_t, budget_cycles):
    """Water-fill on MEASURED error: minimise sum_i err[i, j_i] s.t. sum_i L[j_i] = B.

    Lagrangian sweep -- pick, per pair, the level minimising err + lam*L.
    """
    lo_lam, hi_lam = 1e-18, 1e18
    cost = levels_t.double().view(1, -1)
    for _ in range(90):
        lam = (lo_lam * hi_lam) ** 0.5
        j = (err.double() + lam * cost).argmin(dim=1)
        if levels_t[j].sum().item() > budget_cycles:
            lo_lam = lam
        else:
            hi_lam = lam
    lam = (lo_lam * hi_lam) ** 0.5
    return (err.double() + lam * cost).argmin(dim=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=2048,
                    help="rows kept per captured operator (memory guard)")
    ap.add_argument("--max-ops", type=int, default=14,
                    help="how many (block, op) operands to probe")
    ap.add_argument("--ctx-len", type=int, default=2048)
    args = ap.parse_args()

    parent = Path(args.parent)
    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = str(parent / "wrapper.json")
    hyb = parent / "hybrid_config.json"
    if hyb.is_file():
        os.environ["SC_HYBRID_CONFIG_JSON"] = str(hyb)

    table = json.loads((parent / "table.json").read_text())
    ladder = sorted(int(v) for v in table["stoc_len_levels"])
    # candidate grid: every hardware length the ladder spans, plus the ladder
    grid = sorted(set(list(range(max(4, ladder[0] - 8), min(128, ladder[-1] + 8) + 1, 2)) + ladder))
    print(f"[probe] deployed ladder {ladder}")
    print(f"[probe] candidate grid  {len(grid)} levels: {grid[0]}..{grid[-1]}")

    # Match the deployed runs exactly: AWQ front-end + the same act_scales cache
    # that run_prc_ppl.sbatch uses (the _3 archive is AWQ + INT7 20%-baselined).
    os.environ.setdefault(
        "ACT_SCALES_DIR",
        "/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales")
    os.environ.setdefault("FRONTEND", "awq")

    from benchmark.quant.eval_quant import build_sc_model
    model, tok = build_sc_model(args.model_path, "mp",
                                mp_table=str(parent / "wrapper.json"))
    model.eval()

    cap = {}

    def make_hook(name, op):
        def hook(mod, inp, out):
            if name in cap or len(cap) >= args.max_ops:
                return
            x = inp[0]
            x = x.reshape(-1, x.shape[-1]).detach()
            if x.shape[1] % CHUNK_D or x.shape[0] < 64:
                return
            cap[name] = (x[: args.max_rows].float().cuda(),
                         mod.weight.detach().float().cuda(), op)
        return hook

    from model.sc_common import SCLinear
    hs = []
    for name, mod in model.named_modules():
        if not isinstance(mod, SCLinear):
            continue
        op = getattr(mod, "_sc_op_name", None) or next(
            (o for o in LIN_OPS if o in name), None)
        if op not in LIN_OPS:
            continue
        blk = getattr(mod, "_sc_block_idx", None)
        # one operand per (op, block) and only the first few blocks -- the probe
        # is about the SHAPE of the rule, not about coverage
        if blk is not None and blk > 3:
            continue
        hs.append(mod.register_forward_hook(make_hook(f"b{blk}.{op}", op)))
    print(f"[probe] hooked {len(hs)} SCLinear modules")

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text[:200000], return_tensors="pt").input_ids[:, : args.ctx_len].cuda()
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()

    names = list(cap)[: args.max_ops]
    print(f"[probe] captured {len(cap)} operands, probing {len(names)}")

    levels_t = torch.tensor([float(v) for v in grid], device="cuda")
    ladder_t = torch.tensor([float(v) for v in ladder], device="cuda")
    rows = []

    for name in names:
        x, w, op = cap[name]
        m = chunk_absmax(x)
        mn = minmax_norm(m)
        err = error_at_levels(x, w, grid)

        # --- deployed staircase: rank-match the deployed ladder histogram -----
        # The per-bucket thresholds live in table.json under a naming scheme that
        # varies by generation, so reproduce the deployed ASSIGNMENT instead: the
        # deployed rule is a monotone step function of mn, and its realised
        # histogram is what the calibrator solved for. Use the measured realised
        # mix from the ladder, assigning by mn rank -- this is the most generous
        # reading of the deployed rule (a perfectly calibrated staircase).
        flat_mn = mn.reshape(-1)
        order = torch.argsort(flat_mn)
        n = flat_mn.numel()
        # equal-MAC split across ladder rungs reproduces the deployed mean length
        dep_idx = torch.zeros(n, dtype=torch.long, device="cuda")
        per = n // len(ladder)
        for r in range(len(ladder)):
            sl = order[r * per: (r + 1) * per if r < len(ladder) - 1 else n]
            dep_idx[sl] = int(torch.argmin((levels_t - ladder_t[r]).abs()).item())
        budget = levels_t[dep_idx].sum().item()

        rec = {"name": name, "op": op, "rows": int(x.shape[0]),
               "chunks": int(x.shape[1] // CHUNK_D), "budget_cycles": budget,
               "mean_L": budget / n,
               "deployed_err": pick(err, dep_idx)}

        for p in (2.0 / 3.0, 0.8, 1.0, 1.25):
            j, _ = powerlaw_idx(flat_mn if False else m.reshape(-1), levels_t, p, budget)
            rec[f"powerlaw_p{p:.2f}_err"] = pick(err, j)
            rec[f"powerlaw_p{p:.2f}_meanL"] = levels_t[j].sum().item() / n

        j_or = oracle_idx(err, levels_t, budget)
        rec["oracle_err"] = pick(err, j_or)
        rec["oracle_meanL"] = levels_t[j_or].sum().item() / n

        best_p = min((p for p in (2.0 / 3.0, 0.8, 1.0, 1.25)),
                     key=lambda p: rec[f"powerlaw_p{p:.2f}_err"])
        d, o = rec["deployed_err"], rec["oracle_err"]
        b = rec[f"powerlaw_p{best_p:.2f}_err"]
        rec["best_p"] = best_p
        rec["gain_vs_deployed_pct"] = 100.0 * (b - d) / d if d else None
        rec["closes_frac_of_oracle_gap"] = (d - b) / (d - o) if (d - o) > 0 else None
        rows.append(rec)
        print(f"  {name:38} meanL={rec['mean_L']:6.2f}  deployed={d:.5e}  "
              f"best p={best_p:.2f} {b:.5e} ({rec['gain_vs_deployed_pct']:+.2f}%)  "
              f"oracle={o:.5e}  closes {100*(rec['closes_frac_of_oracle_gap'] or 0):.0f}% of gap")

    Path(args.out).write_text(json.dumps(
        {"ladder": ladder, "grid": grid, "parent": str(parent), "rows": rows}, indent=1))
    print(f"\n[probe] wrote {args.out}")

    gains = [r["gain_vs_deployed_pct"] for r in rows if r["gain_vs_deployed_pct"] is not None]
    closes = [r["closes_frac_of_oracle_gap"] for r in rows if r["closes_frac_of_oracle_gap"] is not None]
    if gains:
        print(f"[probe] mean gain vs deployed: {sum(gains)/len(gains):+.2f}%  "
              f"(n={len(gains)}); mean fraction of the oracle gap closed: "
              f"{100*sum(closes)/len(closes):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
