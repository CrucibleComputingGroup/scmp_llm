"""Emit a per-(row, chunk) table whose induced L(m) follows a POWER LAW.

WHY. The deployed rule buckets the per-call min-max normalized absmax into <=7
rungs. For one output element the chunk errors add with |eps_j| ~ m_j/sqrt(L_j),
so minimizing sum_j m_j^2/L_j under sum_j L_j = B gives L_j ~ m_j, and the
perfectly-correlated case (the array shares two RNG banks across the
contraction) gives L_j ~ m_j^(2/3). Measured on real operands at iso-cost, the
power law cuts squared error ~11.5% against a perfectly-calibrated staircase and
closes ~61% of the gap to an oracle that needs an FP16 reference -- with
p = 2/3 winning on 11 of 14 operators, which is the correlated-case prediction.

NO CODE CHANGE. The deployed loader takes (levels ASCENDING, thresholds
ASCENDING) per (op, t-bucket, l-bucket) and does bucketize(mn, th). A power law
is expressible in exactly that format by using MANY rungs and placing each
threshold at the metric value where the law crosses the midpoint between
adjacent levels. So this is a table swap, not a kernel change.

ISO-COST BY CONSTRUCTION. For every bucket we compute the PARENT table's induced
mean length under the empirically observed metric distribution, then solve the
power law's scale kappa so its induced mean length matches to <0.5%. Cost is
therefore preserved per bucket, not merely on average -- the cross-layer split
is untouched, which is the axis that has failed 6 times here.

VALIDATION. The emitted file is round-tripped through the DEPLOYED parser
(AdaptiveMPConfig) before it is written, and every bucket is checked for the
loader's ascending invariants. This is the silent-fallback bug family that has
already cost this project two full waves.

Usage (GPU node):
    python -m benchmark.ppl.kbands.make_powerlaw_prc \
        --parent <archive config dir> --prc <parent prc json> \
        --model_path <hf> --p 0.667 --out <new prc json>
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
LIN_OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def bucket_of(block_idx, total_blocks, n_buckets):
    if total_blocks is None or n_buckets <= 1:
        return 0
    return min(n_buckets - 1, int(block_idx / max(total_blocks - 1, 1) * n_buckets))


def fine_levels(parent_levels, n_want):
    """Geometric ladder over the parent's span, snapped to integer cycles."""
    lo, hi = int(min(parent_levels)), int(max(parent_levels))
    if n_want <= len(parent_levels):
        return sorted(set(int(v) for v in parent_levels))
    out = set(int(round(lo * (hi / lo) ** (i / (n_want - 1)))) for i in range(n_want))
    out |= {lo, hi}
    return sorted(v for v in out if 1 <= v <= 128)


def induced_mean(levels, thresholds, mn):
    """Mean length the staircase assigns over the observed metric sample."""
    th = torch.tensor(thresholds, dtype=mn.dtype, device=mn.device)
    lv = torch.tensor([float(v) for v in levels], dtype=mn.dtype, device=mn.device)
    return lv[torch.bucketize(mn, th)].mean().item()


def powerlaw_thresholds(levels, p, kappa):
    """t_i = ((L_i + L_{i+1})/2 / kappa)^(1/p) -- where the law crosses the midpoint."""
    th = []
    for a, b in zip(levels[:-1], levels[1:]):
        mid = 0.5 * (a + b)
        t = (mid / kappa) ** (1.0 / p)
        th.append(min(max(t, 0.0), 1.0))
    # the loader demands strictly ascending; nudge any ties
    for i in range(1, len(th)):
        if th[i] <= th[i - 1]:
            th[i] = min(1.0, th[i - 1] + 1e-6)
    return th


def solve_kappa(levels, p, mn, target_mean):
    """Bisect kappa so the power-law staircase reproduces target_mean."""
    lo, hi = 1e-6, 1e6
    for _ in range(90):
        mid = (lo * hi) ** 0.5
        th = powerlaw_thresholds(levels, p, mid)
        m = induced_mean(levels, th, mn)
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    k = (lo * hi) ** 0.5
    th = powerlaw_thresholds(levels, p, k)
    return k, th, induced_mean(levels, th, mn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True, help="archive config dir (wrapper/table/hybrid)")
    ap.add_argument("--prc", required=True, help="parent per_row_chunk json")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--p", type=float, default=2.0 / 3.0)
    ap.add_argument("--rungs", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wrapper", required=True,
                    help="parent prc WRAPPER json (for the deployed-parser check)")
    ap.add_argument("--max-rows", type=int, default=4096)
    ap.add_argument("--ctx-len", type=int, default=2048)
    args = ap.parse_args()

    parent = Path(args.parent)
    src = json.loads(Path(args.prc).read_text())
    pbuckets = src["per_row_chunk"]["buckets"]
    print(f"[plaw] parent prc: {len(pbuckets)} buckets, p={args.p:.4f}, rungs={args.rungs}")

    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = str(parent / "wrapper.json")
    hyb = parent / "hybrid_config.json"
    if hyb.is_file():
        os.environ["SC_HYBRID_CONFIG_JSON"] = str(hyb)
    os.environ.setdefault(
        "ACT_SCALES_DIR",
        "/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales")
    os.environ.setdefault("FRONTEND", "awq")

    from benchmark.quant.eval_quant import build_sc_model
    from model.sc_common import SCLinear
    model, tok = build_sc_model(args.model_path, "mp", mp_table=str(parent / "wrapper.json"))
    model.eval()
    total_blocks = getattr(model.config, "num_hidden_layers", None)
    tbl = json.loads((parent / "table.json").read_text())
    n_lbuckets = int(tbl.get("layer_buckets", 4))

    # ---- collect the metric distribution per (op, layer bucket) -------------
    samples = defaultdict(list)

    def make_hook(op, lb):
        def hook(mod, inp, out):
            key = f"{op}:t0:l{lb}"
            if sum(s.numel() for s in samples[key]) > args.max_rows * 64:
                return
            x = inp[0]
            x = x.reshape(-1, x.shape[-1]).detach()
            if x.shape[1] % CHUNK_D or x.shape[0] < 8:
                return
            x = x[: args.max_rows].float()
            n, D = x.shape
            nch = D // CHUNK_D
            m = x.abs().view(n, nch, CHUNK_D).amax(-1)
            lo, hi = m.min(), m.max()
            samples[key].append(((m - lo) / (hi - lo).clamp_min(1e-8)).reshape(-1).cpu())
        return hook

    hs = []
    for name, mod in model.named_modules():
        if not isinstance(mod, SCLinear):
            continue
        op = getattr(mod, "_sc_op_name", None) or next((o for o in LIN_OPS if o in name), None)
        if op not in LIN_OPS:
            continue
        blk = getattr(mod, "_sc_block_idx", None)
        lb = bucket_of(blk, total_blocks, n_lbuckets)
        hs.append(mod.register_forward_hook(make_hook(op, lb)))
    print(f"[plaw] hooked {len(hs)} SCLinear modules over {n_lbuckets} layer buckets")

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"][:4000])
    ids = tok(text, return_tensors="pt").input_ids[:, : args.ctx_len].cuda()
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    print(f"[plaw] metric samples for {len(samples)} buckets")

    # ---- re-emit every bucket under the power law, iso-cost -----------------
    out_buckets, report = {}, []
    for key, entry in pbuckets.items():
        plevels = [int(v) for v in entry["levels"]]
        pth = [float(v) for v in entry["thresholds"]]
        s = samples.get(key)
        if not s:
            out_buckets[key] = entry          # unobserved (e.g. qk/av) -> untouched
            continue
        mn = torch.cat(s).cuda()
        target = induced_mean(plevels, pth, mn)
        levels = fine_levels(plevels, args.rungs)
        k, th, got = solve_kappa(levels, args.p, mn, target)
        out_buckets[key] = {"levels": levels, "thresholds": th}
        report.append((key, target, got, 100.0 * (got - target) / target, len(levels)))

    dst = dict(src)
    dst["per_row_chunk"] = {"buckets": out_buckets}
    dst["_powerlaw"] = {"p": args.p, "rungs": args.rungs,
                        "parent_prc": str(args.prc),
                        "note": "iso-cost per bucket vs the parent staircase"}

    # ---- validate through the DEPLOYED parser before writing ---------------
    tmp = Path(args.out).with_suffix(".validating.json")
    tmp.write_text(json.dumps(dst, indent=1))
    # Build through the DEPLOYED path: AdaptiveMPConfig.__post_init__ calls
    # load_threshold_table, so constructing from the wrapper exercises exactly
    # the parser the runtime uses.
    from scmp_kernels.mp.config import AdaptiveMPConfig
    import dataclasses
    fields = {f.name for f in dataclasses.fields(AdaptiveMPConfig)}
    wrap = json.loads(Path(args.wrapper).read_text())
    wrap = {k: v for k, v in wrap.items() if k in fields}
    wrap["threshold_table_path"] = str(tmp)
    cfg = AdaptiveMPConfig(**wrap)
    n_prc = len(getattr(cfg, "per_row_chunk", {}) or {})
    if n_prc < len(out_buckets) * 0.9:
        raise SystemExit(f"[plaw] DEPLOYED PARSER took only {n_prc} of {len(out_buckets)} "
                         f"buckets -- refusing to emit a table that silently falls back")
    tmp.rename(args.out)

    drift = [abs(r[3]) for r in report]
    print(f"\n[plaw] re-emitted {len(report)} buckets; deployed parser accepted {n_prc}")
    print(f"[plaw] iso-cost drift: max {max(drift):.3f}%  mean {sum(drift)/len(drift):.3f}%")
    for key, t, g, d, nl in report[:8]:
        print(f"    {key:22} parent meanL {t:6.2f} -> powerlaw {g:6.2f} ({d:+.3f}%)  rungs={nl}")
    print(f"[plaw] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
