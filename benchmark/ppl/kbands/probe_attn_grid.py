"""GATE: does a per-operator-family enable grid help the ATTENTION products?

THE QUESTION. `sc_matmul` resolves `rng_levels` to `2**(sc_prec-1) = 128` for
every call, while the deployed attention rows run far shorter streams (deployed
4B t32 `table.json`, bucket `av:t0:l0`: `avg_stoc_len` 84.1). A stream of L
cycles resolves ~L magnitudes, so a 128-level grid under an ~84-cycle stream is
partly sampling noise. `SC_RNG_GRID=pow2` already exists but is ONE GLOBAL
policy over linears and attention together, and it switches OFF at L>=96 -- so
it never reaches the attention rungs sitting right at that boundary, nor the
escape-gate rung. What has never been measured is a grid chosen SEPARATELY for
`qk` and `av`.

Why attention specifically: ~78% of the network's SC error mass sits in qk+av,
the per-(row,chunk) allocator structurally cannot reach them (qk contracts over
head_dim=128 = one chunk, av is unchunked), and neither SmoothQuant nor AWQ
touches them (both stop at SCLinear). The grid is the one lever that does reach.

WHAT THIS MEASURES -- DOWNSTREAM quantities on REAL captured operands, not
reconstruction error. Error-axis gains have failed to reach PPL ~6 times in this
project, so this probe measures what the next operator actually consumes:

  qk : masked-softmax KL(P_fp16 || P_sc) over attention probabilities. The
       scores are never consumed -- softmax is, and softmax is shift-invariant
       and saturating, so score error and probability error differ.
  av : relative L2 of the attention OUTPUT after signed accumulation, computed
       from the FP16 reference probabilities so it isolates av from qk's error.

Per-row stream lengths are the DEPLOYED ones: the real AdaptiveMPConfig table,
classified through `adaptive_classify_rows` in one global pass over B*H*N rows
and through `classify_level_values` (escape rung included), exactly as
`model/sc_common._sc_attention_matmul_ab_t` does -- and the metric is computed
from the REAL operand via `_mp_dispatch_metric`, because the deployed tables use
`crest`/`l2` for attention on some models, not always `amax`. The grid is the
ONLY thing that varies, so every arm is exactly iso-cycle by construction.

Two INDEPENDENT captures (disjoint wikitext-2 windows): an arm chosen on capture
A must hold on capture B, or it is noise.

Usage (GPU node):
    python -m benchmark.ppl.kbands.probe_attn_grid \
        --parent <mp_best/configs/<model>/target<t>> --out out.json \
        [--qk-scales <..._qk_alpha1.0.json>]
"""
from __future__ import annotations

import argparse
import json
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
import torch.nn.functional as F  # noqa: E402

SC_PREC = 8
HALVE = True
# Every candidate is a power of two: the Owen/bit-reversal scramble is a pow2
# construction, and non-pow2 grids were measured dead (won only at pow2 L).
GRIDS = [16, 32, 64, 128]
ARMS = [None, 16, 32, 64, 128, "pow2", "pow2all"]


def pow2_floor(L: int) -> int:
    return 1 << (int(L).bit_length() - 1)


def resolve_grid(arm, L: int):
    """arm -> rng_levels for one rung. None == today's deployed path."""
    if arm is None:
        return None
    if arm == "pow2":
        return None if L >= 96 else pow2_floor(L)
    if arm == "pow2all":
        return pow2_floor(L)
    return int(arm)


def arm_name(arm):
    return "deployed(128)" if arm is None else str(arm)


def deployed_row_levels(a3, mp_config, operator, block_idx, total_blocks):
    """(row_levels, levels) exactly as _sc_attention_matmul_ab_t computes them.

    a3 is the REAL (B*H, N, K) left operand -- required, because the deployed
    dispatch metric is `crest`/`l2` for attention on some models and those are
    not recoverable from a per-row amax.
    """
    from model.sc_common import _mp_dispatch_metric
    from scmp_kernels.mp import adaptive_classify_rows

    metric_all = _mp_dispatch_metric(a3, mp_config, operator)      # (B*H, N)
    assignment = adaptive_classify_rows(
        metric_all.reshape(-1), mp_config, operator=operator,
        block_idx=block_idx, total_blocks=total_blocks)
    levels = mp_config.classify_level_values(
        operator=operator, block_idx=block_idx, total_blocks=total_blocks)
    return assignment.row_levels.reshape(a3.shape[0], a3.shape[1]), list(levels)


def sc_rows(a2, b2, row_lv, levels, arm, smooth_scales=None):
    """One (N,K) x (M,K) -> (N,M) slice at the deployed per-row lengths, one arm."""
    from scmp_kernels.sc.matmul import sc_matmul

    out = torch.zeros(a2.shape[0], b2.shape[0], dtype=torch.float32,
                      device=a2.device)
    for li, sl in enumerate(levels):
        idx = (row_lv == li).nonzero(as_tuple=True)[0]
        if idx.numel() == 0 or int(sl) <= 0:
            continue
        L = int(sl)
        out[idx] = sc_matmul(
            a2.index_select(0, idx).contiguous(), b2,
            granularity="per_row", mode="bipolar", sc_prec=SC_PREC,
            stoc_len=L, halve_bipolar_stoc_len=HALVE,
            rng_levels=resolve_grid(arm, L), smooth_scales=smooth_scales)
    return out


def mean_L(row_lv, levels):
    lv = torch.tensor([float(v) for v in levels], device=row_lv.device)
    return float(lv[row_lv.reshape(-1).clamp(0, len(levels) - 1)].mean())


def masked_kl(logp_ref, logp_sc):
    """sum_j P_ref log(P_ref/P_sc) per row, averaged. Masked cells contribute 0."""
    p = logp_ref.exp()
    term = torch.where(p > 0, p * (logp_ref - logp_sc), torch.zeros_like(p))
    return float(term.sum(dim=-1).mean())


def rel_l2(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True, help="bundle with wrapper.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qk-scales", default=None,
                    help="deployed qk rebalance JSON; when given the qk arms ALSO "
                         "run with it, so the gate says whether the grid survives "
                         "on top of the deployed winner")
    ap.add_argument("--blocks", default="")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx-len", type=int, default=2048)
    args = ap.parse_args()

    parent = Path(args.parent)
    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = str(parent / "wrapper.json")
    hyb = parent / "hybrid_config.json"
    if hyb.is_file():
        os.environ["SC_HYBRID_CONFIG_JSON"] = str(hyb)
    if args.qk_scales:
        os.environ["SC_ATTN_SMOOTH_JSON"] = args.qk_scales
    os.environ.setdefault(
        "ACT_SCALES_DIR",
        "/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales")
    os.environ.setdefault("FRONTEND", "awq")
    hf = json.loads((parent / "table.json").read_text())["model_path"]
    print(f"[attngrid] parent={parent}\n[attngrid] model={hf}")

    from benchmark.quant.eval_quant import build_sc_model
    model, tok = build_sc_model(hf, "mp", mp_table=str(parent / "wrapper.json"))
    model.eval()
    cfg = model.config
    mp_config = getattr(cfg, "sc_mp_config", None)
    assert mp_config is not None, "no sc_mp_config on the built model"
    total_blocks = getattr(cfg, "_sc_total_blocks", None)

    import model.sc_common as scc
    real_fwd = scc.sc_eager_attention_forward
    targets = [m for m in sys.modules.values()
               if m is not None
               and getattr(m, "eager_attention_forward", None) is real_fwd]
    assert targets, "attention was never patched -- capture would be silent"
    print(f"[attngrid] patched: {[m.__name__.split('.')[-1] for m in targets]}")
    if args.qk_scales:
        assert getattr(cfg, "sc_attn_smooth", None), \
            "--qk-scales given but cfg.sc_attn_smooth is empty (silent fallback)"

    cap: dict = {}
    want: set = set()

    def capture(module, query, key, value, attention_mask, scaling,
                dropout=0.0, **kw):
        bi = getattr(module, "layer_idx", None)
        if bi in want and bi not in cap:
            ks = scc._repeat_kv(key, module.num_key_value_groups)
            vs = scc._repeat_kv(value, module.num_key_value_groups)
            cap[bi] = {"q": query.detach().float(), "k": ks.detach().float(),
                       "v": vs.detach().float(), "scaling": float(scaling),
                       "mask": (attention_mask[:, :, :, : ks.shape[-2]]
                                .detach().float()
                                if attention_mask is not None else None)}
        return real_fwd(module, query, key, value, attention_mask, scaling,
                        dropout, **kw)

    # The hybrid INT mask puts SOME (operator, block) pairs on INT7, where the
    # deployed cell runs no SC at all and the enable grid is meaningless. Ask
    # the DEPLOYED resolver which pairs are SC (never re-derive it from the
    # schedule JSON -- two resolvers disagreeing is a recurring bug here).
    nb = len(model.model.layers)
    sc_qk = [b for b in range(nb) if scc._hybrid_backend(cfg, "qk", b) == "sc"]
    sc_av = [b for b in range(nb) if scc._hybrid_backend(cfg, "av", b) == "sc"]
    print(f"[attngrid] blocks on SC: qk {len(sc_qk)}/{nb}, av {len(sc_av)}/{nb}")
    if args.blocks:
        blocks = [int(b) for b in args.blocks.split(",") if b]
    else:
        both = [b for b in range(nb) if b in set(sc_qk) and b in set(sc_av)]
        assert both, "no block runs BOTH qk and av on SC"
        blocks = [both[int(round(f * (len(both) - 1)))]
                  for f in (0.15, 0.4, 0.6, 0.85)]
        blocks = sorted(set(blocks))
    print(f"[attngrid] blocks {blocks} of {nb}; heads/block {args.heads}")
    for b in blocks:
        print(f"[attngrid]   b{b}: qk={scc._hybrid_backend(cfg, 'qk', b)} "
              f"av={scc._hybrid_backend(cfg, 'av', b)}")

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids_all = tok("\n\n".join(ds["text"])[:900000],
                  return_tensors="pt").input_ids[0]
    C = args.ctx_len
    captures = {"A": ids_all[:C], "B": ids_all[4 * C: 5 * C]}

    rows = []
    for cname, seq in captures.items():
        cap.clear()
        want.clear()
        want.update(blocks)
        for m in targets:
            m.eager_attention_forward = capture
        try:
            with torch.no_grad():
                model(seq.unsqueeze(0).cuda())
        finally:
            for m in targets:
                m.eager_attention_forward = real_fwd
        print(f"[attngrid] capture {cname}: blocks {sorted(cap)}")

        for bi in sorted(cap):
            d = cap.pop(bi)
            q, k, v, scaling, mask = d["q"], d["k"], d["v"], d["scaling"], d["mask"]
            B, H, N, Kd = q.shape
            BH = B * H
            q3 = q.reshape(BH, N, Kd).contiguous()
            k3 = k.reshape(BH, N, Kd).contiguous()
            v3 = v.reshape(BH, N, Kd).contiguous()
            step = max(1, BH // args.heads)
            heads = list(range(0, BH, step))[: args.heads]

            def mask_for(bh):
                if mask is None:
                    return None
                b, h = bh // H, bh % H
                return mask[b, 0 if mask.shape[1] == 1 else h]

            # ---------------- qk ----------------------------------------------
            # The hybrid INT mask routes some (operator, block) pairs to INT7,
            # where the deployed cell runs no SC and the grid is meaningless.
            if scc._hybrid_backend(cfg, "qk", bi) != "sc":
                print(f"  [skip] b{bi} qk runs on "
                      f"{scc._hybrid_backend(cfg, 'qk', bi)}, not SC")
            else:
              rl, lv = deployed_row_levels(q3, mp_config, "qk", bi, total_blocks)
              Lqk = mean_L(rl, lv)
              ss = None
              if args.qk_scales:
                  # The deployed qk rebalance does NOT cover every block (the 4B
                  # t32 table carries 26 vectors for 36 blocks); _qk_smooth_scales
                  # returns None there and the deployed run leaves that block
                  # unrebalanced. Mirror that instead of asserting.
                  ss = scc._qk_smooth_scales(cfg, "qk", bi, Kd, q.device,
                                             torch.float32)
                  if ss is None:
                      print(f"  [note] block {bi} has no qk rebalance vector "
                            f"(deployed leaves it unrebalanced); plain arm only")
              logp_ref = {}
              for bh in heads:
                  s = (q3[bh] @ k3[bh].t()) * scaling
                  m = mask_for(bh)
                  logp_ref[bh] = F.log_softmax(s + m if m is not None else s, -1)
                  del s
              for arm in ARMS:
                  kls, kls_s = [], []
                  for bh in heads:
                      m = mask_for(bh)
                      sc = sc_rows(q3[bh], k3[bh], rl[bh], lv, arm) * scaling
                      kls.append(masked_kl(logp_ref[bh],
                                           F.log_softmax(sc + m if m is not None
                                                         else sc, -1)))
                      del sc
                      if ss is not None:
                          sc2 = sc_rows(q3[bh], k3[bh], rl[bh], lv, arm,
                                        smooth_scales=ss) * scaling
                          kls_s.append(masked_kl(logp_ref[bh],
                                                 F.log_softmax(sc2 + m if m is not None
                                                               else sc2, -1)))
                          del sc2
                  rec = {"capture": cname, "block": bi, "op": "qk",
                         "arm": arm_name(arm), "mean_L": Lqk,
                         "kl": sum(kls) / len(kls)}
                  if kls_s:
                      rec["kl_qkrebal"] = sum(kls_s) / len(kls_s)
                  rows.append(rec)
                  print(f"  {cname} b{bi:<3} qk {rec['arm']:>13} L={Lqk:6.2f} "
                        f"KL={rec['kl']:.6e}"
                        + (f"  KL|rebal={rec['kl_qkrebal']:.6e}" if kls_s else ""))
              del logp_ref
              torch.cuda.empty_cache()

            # ---------------- av ----------------------------------------------
            if scc._hybrid_backend(cfg, "av", bi) != "sc":
                print(f"  [skip] b{bi} av runs on "
                      f"{scc._hybrid_backend(cfg, 'av', bi)}, not SC")
                del q, k, v, q3, k3, v3, d
                torch.cuda.empty_cache()
                continue
            # The FULL probability tensor is materialised: the deployed dispatch
            # metric for av is `crest` on 30B, which a per-row amax cannot supply.
            pfp = torch.empty(BH, N, N, device=q.device)
            for bh in range(BH):
                s = (q3[bh] @ k3[bh].t()) * scaling
                m = mask_for(bh)
                pfp[bh] = F.softmax(s + m if m is not None else s, -1)
                del s
            rl, lv = deployed_row_levels(pfp, mp_config, "av", bi, total_blocks)
            Lav = mean_L(rl, lv)
            for arm in ARMS:
                errs = []
                for bh in heads:
                    ofp = pfp[bh] @ v3[bh]
                    osc = sc_rows(pfp[bh], v3[bh].t().contiguous(), rl[bh], lv, arm)
                    errs.append(rel_l2(osc, ofp))
                    del ofp, osc
                rec = {"capture": cname, "block": bi, "op": "av",
                       "arm": arm_name(arm), "mean_L": Lav,
                       "rel_l2": sum(errs) / len(errs)}
                rows.append(rec)
                print(f"  {cname} b{bi:<3} av {rec['arm']:>13} L={Lav:6.2f} "
                      f"relL2={rec['rel_l2']:.6e}")
            del pfp, q, k, v, q3, k3, v3, d
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"parent": str(parent), "model": hf, "arms": [arm_name(a) for a in ARMS],
         "qk_scales": args.qk_scales, "heads": args.heads, "blocks": blocks,
         "rows": rows}, indent=1))
    print(f"\n[attngrid] wrote {args.out}")

    # ---- verdict: best arm per (op, capture), and does A's pick hold on B? ----
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        key = (r["op"], r["capture"], r["arm"])
        agg[key].append(r["kl"] if r["op"] == "qk" else r["rel_l2"])
    for op in ("qk", "av"):
        if not any(k[0] == op for k in agg):
            print(f"\n== {op}: no SC blocks probed (INT mask) ==")
            continue
        print(f"\n== {op}: mean over blocks (lower is better) ==")
        base = {c: (sum(agg[(op, c, "deployed(128)")])
                    / len(agg[(op, c, "deployed(128)")]))
                for c in captures if agg.get((op, c, "deployed(128)"))}
        for a in ARMS:
            n = arm_name(a)
            line = f"  {n:>13}"
            for c in captures:
                vals = agg.get((op, c, n))
                if not vals or c not in base:
                    continue
                mv = sum(vals) / len(vals)
                line += f"   {c}: {mv:.6e} ({100*(mv-base[c])/base[c]:+7.2f}%)"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
