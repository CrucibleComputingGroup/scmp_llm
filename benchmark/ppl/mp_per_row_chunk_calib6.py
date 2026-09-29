"""calib6 — per-group allocation with a LOSS-AWARE (Fisher) currency, global and within-bucket.

Why: calib5's global solve (one lambda across every linear (row,chunk) group, no per-bucket
budgets) lost to the within-bucket c17 table on held-out NLL for 4B t32 when priced in
sigma (relative error): it lowered sigma-error but raised loss. A global allocation and a
global oracle are only as real as their currency. Here each group is priced by the
second-order loss change its SC error causes,

    e_c(L) = sum_j F_{row,j} * eps_{c,j}(L)^2 ,   F_{row,j} = g_{row,j}^2 ,  g = dLoss/dy_op

(diagonal Fisher on the operator output, per token row and output channel; in NLL units and
invariant to per-operator output scale). g is taken at the DEPLOYED parent SC trajectory:
during a gradient pass every SCLinear and every attention SC/INT matmul returns its SC value
in the forward but routes the gradient through the FP operation (straight-through), so the
point of expansion is exactly where eps is measured and where heldout_nll scores tables.
Two label choices: 'fis' = labels sampled from the model's own softmax (MC Fisher = GGN, the
allocation currency), 'fis_emp' = true next tokens (diagnostic).

Emits four tables (all deployable per-(op, layer-quartile) monotone staircases):
  <stem>_wrel / _grel : within-bucket / global, sigma currency (MoE rows / top_k)
  <stem>_wfis / _gfis : within-bucket / global, Fisher currency
and a diag with held-out scoring of every table (and --score-tables, e.g. c17, g5) in every
currency, the Fisher-PREDICTED dNLL per token of each table vs the parent (to validate the
currency against measured held-out NLL before trusting it), and within/global oracles per
currency over the ladder and over all measured lengths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")
REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark.ppl.mp_per_row_chunk_calib5 import (  # noqa: E402
    CHUNK_D, DENSE_LADDER, HALVE, SC_PREC, _sc, bucket_of, call_metric, deploy_rungs,
    solve_bucket, staircase_dp)

WEIGHT_TAGS = ("fis", "fis_emp")


def error_curves_w(x, w, smooth, levels, weights):
    """raw (R, n_chunks, n_levels) squared error plus the same reduced with each weight
    matrix in `weights` ({tag: (R, d_out)}): sum_j W[r, j] * eps_{r,c,j}(L)^2."""
    N, D = x.shape
    nch = (D + CHUNK_D - 1) // CHUNK_D
    raw = torch.zeros(N, nch, len(levels), device=x.device)
    wt = {k: torch.zeros(N, nch, len(levels), device=x.device) for k in weights}
    widths = []
    for c in range(nch):
        sl = slice(c * CHUNK_D, min((c + 1) * CHUNK_D, D))
        xc, wc = x[:, sl].contiguous(), w[:, sl].contiguous()
        sc_c = smooth[sl].contiguous() if smooth is not None else None
        widths.append(xc.shape[1])
        fpc = xc @ wc.t()
        for li, L in enumerate(levels):
            d2 = (_sc(xc, wc, L, sc_c) - fpc) ** 2
            raw[:, c, li] = d2.sum(dim=1)
            for k, W in weights.items():
                wt[k][:, c, li] = (d2 * W).sum(dim=1)
    return raw, wt, torch.tensor(widths, dtype=torch.float32)


class _STELinear(torch.autograd.Function):
    """Forward = the SC output EXACTLY (bit-identical, so MoE routing in the gradient pass
    matches the measurement pass); backward = the FP linear's input gradient g @ W."""

    @staticmethod
    def forward(ctx, x, y_sc, weight):
        ctx.save_for_backward(weight)
        ctx.xdtype = x.dtype
        return y_sc.clone()

    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        return g.matmul(w.to(g.dtype)).to(ctx.xdtype), None, None


class _STEMatmulABt(torch.autograd.Function):
    """Forward = SC/INT a @ b^T output exactly; backward = FP matmul gradients."""

    @staticmethod
    def forward(ctx, a, b, out):
        ctx.save_for_backward(a, b)
        return out.clone()

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        g = g.to(a.dtype)
        return torch.matmul(g, b), torch.matmul(g.transpose(-2, -1), a), None


class GradPass:
    """Straight-through gradient capture on the deployed SC trajectory (context manager)."""

    def __init__(self, sc_common, SCLinear, capture):
        self.scm, self.SCLinear, self.capture = sc_common, SCLinear, capture

    def __enter__(self):
        scm, SCL, cap = self.scm, self.SCLinear, self.capture
        self._lin, self._sca, self._inta = (SCL.forward, scm._sc_attention_matmul_ab_t,
                                            scm._int_attention_matmul_ab_t)
        orig_lin, orig_sca, orig_inta = self._lin, self._sca, self._inta

        def lin(mod, x):
            with torch.no_grad():
                y_sc = orig_lin(mod, x)
            y = _STELinear.apply(x, y_sc, mod.weight) if x.requires_grad else y_sc
            cap(mod, y)
            return y

        def ste(orig):
            def f(a, b, **kw):
                with torch.no_grad():
                    out = orig(a, b, **kw)
                if not (a.requires_grad or b.requires_grad):
                    return out
                return _STEMatmulABt.apply(a, b, out)
            return f

        SCL.forward = lin
        scm._sc_attention_matmul_ab_t = ste(orig_sca)
        scm._int_attention_matmul_ab_t = ste(orig_inta)
        return self

    def __exit__(self, *exc):
        self.SCLinear.forward = self._lin
        self.scm._sc_attention_matmul_ab_t = self._sca
        self.scm._int_attention_matmul_ab_t = self._inta
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True, help="output STEM path (…/<m>_t<T>_c6.json)")
    ap.add_argument("--diag", required=True)
    ap.add_argument("--ladder", default=",".join(map(str, DENSE_LADDER)))
    ap.add_argument("--calib-windows", type=int, default=6)
    ap.add_argument("--holdout-windows", type=int, default=2)
    ap.add_argument("--rows-per-call", type=int, default=64)
    ap.add_argument("--expert-calls-per-block", type=int, default=8)
    ap.add_argument("--bins", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frontend", default="awq")
    ap.add_argument("--currencies", default="rel,fis")
    ap.add_argument("--grad-mode", default="full", choices=("full", "block"),
                    help="full: one autograd graph for the whole model (4B/llama8B); block: "
                         "store each decoder layer's input in a no-grad forward, then back-"
                         "propagate one layer at a time (same math, one layer's graph in "
                         "memory; needed for 30B)")
    ap.add_argument("--score-tables", default="",
                    help="name=TABLE.json,... extra per_row_chunk tables to score (c17, g5)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0

    parent = Path(args.parent)
    table = json.loads((parent / "table.json").read_text())
    wrapper = json.loads((parent / "wrapper.json").read_text())
    cap = 2 ** (SC_PREC - 1) if HALVE else 2 ** SC_PREC
    ladder = sorted(int(v) for v in args.ladder.split(","))
    assert max(ladder) == cap and min(ladder) >= 1
    esc = int(wrapper.get("escape_stoc_len", 0) or 0)
    tbl_levels = set()

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("levels", "ladder", "ladders", "stoc_len_levels") and isinstance(v, list):
                    for x in v:
                        if isinstance(x, (int, float)):
                            tbl_levels.add(int(x))
                        elif isinstance(x, list):
                            tbl_levels.update(int(y) for y in x if isinstance(y, (int, float)))
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
    _walk(table)
    score_tables = {}
    for spec in [s for s in args.score_tables.split(",") if s]:
        nm, pth = spec.split("=", 1)
        t_ = json.loads(Path(pth).read_text())
        if "threshold_table_path" in t_:
            t_ = json.loads(Path(t_["threshold_table_path"]).read_text())
        score_tables[nm] = (t_.get("per_row_chunk") or {}).get("buckets", {})
        for e_ in score_tables[nm].values():
            tbl_levels.update(int(v) for v in e_["levels"])
    measured = sorted(L for L in (set(ladder) | tbl_levels | ({esc} if 0 < esc <= cap else set()))
                      if 1 <= L <= cap)
    li = {L: i for i, L in enumerate(measured)}
    lad_cols = [li[L] for L in ladder]
    top_col = li[max(ladder)]
    Lv = np.asarray(ladder, dtype=np.float64)
    currencies = [c for c in args.currencies.split(",") if c]
    need_grad = any(c.startswith("fis") for c in currencies)
    print(f"[c6] ladder {ladder}; measured {measured}; currencies {currencies}")

    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = str(parent / "wrapper.json")
    if (parent / "hybrid_config.json").is_file():
        os.environ["SC_HYBRID_CONFIG_JSON"] = str(parent / "hybrid_config.json")
    os.environ.setdefault("ACT_SCALES_DIR",
                          "/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales")
    os.environ["FRONTEND"] = args.frontend

    from benchmark.quant.eval_quant import build_sc_model
    from benchmark.ppl.calibrate_mp_thresholds import _select_int_swap_windows
    import model.sc_common as scm
    from model.sc_common import (SCLinear, _hybrid_backend, _mp_dispatch_metric,
                                 _channel_index_tensor, _complement_channel_indices)
    from scmp_kernels.mp import adaptive_classify_rows
    from datasets import load_dataset

    model, tok = build_sc_model(args.model_path, "mp", mp_table=str(parent / "wrapper.json"))
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    dev = next(model.parameters()).device
    config = model.config
    total_blocks = getattr(config, "num_hidden_layers", None)
    mp_cfg = config.sc_mp_config
    n_lb = int(getattr(mp_cfg, "layer_buckets", 0) or table.get("layer_buckets") or 4)
    n_units = getattr(config, "num_experts", None) or 0
    top_k = getattr(config, "num_experts_per_tok", None) or 1

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    nwin = args.calib_windows + args.holdout_windows
    windows, starts = _select_int_swap_windows(enc, 2048, nwin, sampling="stratified",
                                               seed=args.seed)
    k_ = max(nwin // max(args.holdout_windows, 1), 1)
    hold_idx = set(list(range(k_ - 1, nwin, k_))[:args.holdout_windows])
    print(f"[c6] windows {starts}; held-out idx {sorted(hold_idx)}")

    OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    sel_cache = {}

    def expert_selected(op, blk, unit, win):
        if unit is None:
            return True
        key = (op, blk, win)
        if key not in sel_cache:
            seed = int.from_bytes(hashlib.blake2b(f"{args.seed}:{op}:{blk}:{win}".encode(),
                                                  digest_size=8).digest(), "little")
            rng = np.random.default_rng(seed)
            sel_cache[key] = set(rng.choice(n_units, size=min(args.expert_calls_per_block,
                                                                n_units), replace=False).tolist())
        return int(unit) in sel_cache[key]

    def sel_rows(op, blk, unit, win, N, device):
        g = torch.Generator(device="cpu")
        g.manual_seed(zlib.crc32(f"{args.seed}:{op}:{blk}:{unit}:{win}".encode()))
        return torch.randperm(N, generator=g)[:min(args.rows_per_call, N)].to(device)

    state = {"phase": "calib", "win": 0, "measure": False, "gtag": None}
    G = {}

    def capture(mod, y):
        op, blk = getattr(mod, "_sc_op_name", None), getattr(mod, "_sc_block_idx", None)
        unit = getattr(mod, "_sc_unit_idx", None)
        if op not in OPS or blk is None or _hybrid_backend(config, op, blk) != "sc":
            return
        if not expert_selected(op, blk, unit, state["win"]) or not y.requires_grad:
            return
        d_out = y.shape[-1]
        N = y.numel() // d_out
        if N == 0:
            return
        rows = sel_rows(op, blk, unit, state["win"], N, y.device)
        key = (op, blk, unit)

        def hk(gr):
            tag = state["gtag"]
            g2 = gr.reshape(-1, d_out).index_select(0, rows).float().pow(2)
            ent = G.setdefault(key, {})
            ent[tag] = ent.get(tag, 0) + g2
        y.register_hook(hk)

    store = {"calib": defaultdict(list), "hold": defaultdict(list)}

    def hook(mod, inp, out):
        if not state["measure"]:
            return
        op, blk = getattr(mod, "_sc_op_name", None), getattr(mod, "_sc_block_idx", None)
        unit = getattr(mod, "_sc_unit_idx", None)
        if op not in OPS or blk is None or _hybrid_backend(config, op, blk) != "sc":
            return
        if not expert_selected(op, blk, unit, state["win"]):
            return
        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
        if x.shape[0] == 0:
            return
        w = mod.weight.float()
        x_full, w_full = x, w
        smooth = getattr(mod, "smooth_scales", None)
        smooth = smooth.float() if smooth is not None else None
        prot = mp_cfg.get_protected_channels(operator=op, block_idx=blk, unit_idx=unit)
        pidx = _channel_index_tensor(prot, x.shape[1], x.device)
        if pidx.numel() > 0:
            rest = _complement_channel_indices(x.shape[1], pidx, x.device)
            if rest.numel() == 0:
                return
            x, w = x.index_select(1, rest), w.index_select(1, rest)
            smooth = smooth.index_select(0, rest) if smooth is not None else None
        N = x.shape[0]
        mn = call_metric(x)
        met = _mp_dispatch_metric(x, mp_cfg, op)
        asg = adaptive_classify_rows(met, mp_cfg, operator=op, block_idx=blk,
                                     total_blocks=total_blocks)
        parL = torch.zeros(N, dtype=torch.float32, device=x.device)
        for sl, idx in asg.level_row_indices.items():
            parL[idx] = float(sl)
        rows = sel_rows(op, blk, unit, state["win"], N, x.device)
        R = rows.numel()
        xs = x.index_select(0, rows)
        gw = G.get((op, blk, unit), {})
        weights = {t: gw[t] for t in WEIGHT_TAGS if t in gw and gw[t].shape[0] == R}
        raw, wt, widths = error_curves_w(xs, w, smooth, measured, weights)
        yE = (x_full.index_select(0, rows) @ w_full.t()).pow(2).sum(dim=1).clamp_min(1e-12)
        rep = N / R
        if unit is not None:
            rep *= n_units / max(min(args.expert_calls_per_block, n_units), 1)
        mac = (widths * w.shape[0]).to(x.device)[None, :].expand(R, -1) * rep
        b = bucket_of(blk, total_blocks, n_lb)
        rec = dict(
            mn=mn.index_select(0, rows).reshape(-1).cpu().numpy(),
            raw=(raw * rep).reshape(-1, len(measured)).cpu().numpy(),
            mac=mac.reshape(-1).cpu().numpy(),
            yE=yE[:, None].expand(-1, mn.shape[1]).reshape(-1).cpu().numpy(),
            expert=np.full(R * mn.shape[1], unit is not None),
            parL=parL.index_select(0, rows)[:, None].expand(-1, mn.shape[1]).reshape(-1)
            .cpu().numpy())
        for t in WEIGHT_TAGS:
            if t in wt:
                rec[t] = (wt[t] * rep).reshape(-1, len(measured)).cpu().numpy()
        store[state["phase"]][(op, b)].append(rec)

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, SCLinear)]
    emb = model.get_input_embeddings()
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    layers = model.model.layers
    final_norm = model.model.norm

    def block_grad_pass(ids, win_i):
        """Exact block-sequential backward on the deployed SC trajectory."""
        saved = [None] * len(layers)
        hL = {}

        def pre(idx):
            def f(mod, args_, kwargs_):
                kw = dict(kwargs_)
                if args_:
                    saved[idx] = (args_[0].detach(), tuple(args_[1:]), kw, False)
                else:
                    saved[idx] = (kw.pop("hidden_states").detach(), (), kw, True)
            return f

        def npre(mod, args_):
            hL["h"] = args_[0].detach()
        hs_ = [layers[j].register_forward_pre_hook(pre(j), with_kwargs=True)
               for j in range(len(layers))]
        hs_.append(final_norm.register_forward_pre_hook(npre))
        try:
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
        finally:
            for h_ in hs_:
                h_.remove()
        tags = ("fis_emp", "fis")
        with torch.enable_grad():
            h = hL["h"].clone().requires_grad_(True)
            logits = model.lm_head(final_norm(h))[0, :-1].float()
            tgt = ids[0, 1:]
            torch.manual_seed(1000 * args.seed + win_i)
            with torch.no_grad():
                mc = torch.multinomial(torch.softmax(logits, -1), 1)[:, 0]
            Gn = {"fis_emp": torch.autograd.grad(ce(logits, tgt), h, retain_graph=True)[0],
                  "fis": torch.autograd.grad(ce(logits, mc), h)[0]}
            del logits
            with GradPass(scm, SCLinear, capture):
                for j in range(len(layers) - 1, -1, -1):
                    h0, rest, k_, as_kw = saved[j]
                    hin = h0.clone().requires_grad_(True)
                    k_ = dict(k_)
                    k_["use_cache"] = False
                    k_.pop("past_key_value", None)
                    k_.pop("past_key_values", None)
                    out = (layers[j](hidden_states=hin, **k_) if as_kw
                           else layers[j](hin, *rest, **k_))
                    out = out[0] if isinstance(out, (tuple, list)) else out
                    Gb = {}
                    for ti, tag in enumerate(tags):
                        state["gtag"] = tag
                        Gb[tag] = torch.autograd.grad(out, hin, grad_outputs=Gn[tag],
                                                      retain_graph=ti < len(tags) - 1)[0]
                    Gn = Gb
                    saved[j] = None
                    del out, hin
        del saved, Gn

    try:
        for i, wnd in enumerate(windows):
            state["phase"] = "hold" if i in hold_idx else "calib"
            state["win"] = i
            ids = wnd.unsqueeze(0).to(dev)
            G.clear()
            if need_grad and args.grad_mode == "block":
                block_grad_pass(ids, i)
                torch.cuda.empty_cache()
            elif need_grad:
                torch.manual_seed(1000 * args.seed + i)
                eh = emb.register_forward_hook(lambda m, a, o: o.detach().requires_grad_(True))
                try:
                    with GradPass(scm, SCLinear, capture), torch.enable_grad():
                        logits = model(input_ids=ids).logits[0, :-1].float()
                        tgt = ids[0, 1:]
                        with torch.no_grad():
                            mc = torch.multinomial(torch.softmax(logits, -1), 1)[:, 0]
                        state["gtag"] = "fis_emp"
                        ce(logits, tgt).backward(retain_graph=True)
                        state["gtag"] = "fis"
                        ce(logits, mc).backward()
                        del logits
                finally:
                    eh.remove()
                torch.cuda.empty_cache()
            state["measure"] = True
            with torch.no_grad():
                model(input_ids=ids)
            state["measure"] = False
            print(f"[c6] window {i + 1}/{nwin} ({state['phase']}): {len(store[state['phase']])} keys, "
                  f"grad keys {len(G)}")
    finally:
        for h in hs:
            h.remove()

    # ------------------------------------------------------------------ analysis
    def cat(parts, f):
        return np.concatenate([p[f] for p in parts])

    def cur_of(parts, name):
        if name == "raw":
            return cat(parts, "raw")
        if name == "rel":
            e = cat(parts, "raw")
            r = np.maximum(e - e[:, [top_col]], 0.0) / cat(parts, "yE")[:, None]
            ex = cat(parts, "expert")
            if ex.any():
                r[ex] /= float(top_k)          # MoE: block output = sum over top_k experts
            return r
        e = cat(parts, name)                   # fisher tags: absolute (constant per pair is irrelevant)
        return e - e[:, [top_col]]

    keys = [k for k in sorted(store["calib"]) if store["calib"][k]]
    budget = {k: float((cat(store["calib"][k], "mac") * cat(store["calib"][k], "parL")).sum()
                       / cat(store["calib"][k], "mac").sum()) for k in keys}

    def binned(parts, cur):
        m = cat(parts, "mn")
        order = np.argsort(m, kind="stable")
        B = int(min(args.bins, m.size))
        edges = np.linspace(0, m.size, B + 1).astype(np.int64)
        E = np.add.reduceat(cur[order][:, lad_cols], edges[:-1], axis=0)
        C = np.add.reduceat(cat(parts, "mac")[order], edges[:-1])
        return E, C, m[order][edges[1:] - 1]

    def th_from(best, bin_max):
        th = []
        for k in range(len(Lv) - 1):
            low, high = np.nonzero(best <= k)[0], np.nonzero(best > k)[0]
            th.append(1.0 if high.size == 0 else (0.0 if low.size == 0 else float(bin_max[low[-1]])))
        for i in range(1, len(th)):
            th[i] = max(th[i], th[i - 1])
        return th

    def bisect(pick, cost, target):
        lo, hi = 0.0, 1e-30
        j = pick(hi)
        while cost(j) > target and hi < 1e40:
            hi *= 10.0
            j = pick(hi)
        best = j
        for _ in range(80):
            mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e3
            j = pick(mid)
            if cost(j) <= target:
                best, hi = j, mid
            else:
                lo = mid
            if lo > 0 and hi / lo < 1 + 1e-7:
                break
        return best

    tables = {}                                # name -> {key: thresholds}
    diag = {"ladder": ladder, "measured": measured, "currencies": currencies, "tables": {}}
    for cname in list(currencies):
        if cname.startswith("fis") and not all(cname in p for k in keys for p in store["calib"][k]):
            miss = [k for k in keys if not all(cname in p for p in store["calib"][k])]
            print(f"[c6] WARNING currency {cname} missing gradients for {len(miss)} keys "
                  f"(e.g. {miss[:3]}) -- skipped")
            currencies.remove(cname)
            continue
        bins = {k: binned(store["calib"][k], cur_of(store["calib"][k], cname)) for k in keys}
        # within-bucket: each key at the parent's own calibration cost
        tw = {}
        for k in keys:
            parts = store["calib"][k]
            tw[k] = solve_bucket(cat(parts, "mn"), cur_of(parts, cname)[:, lad_cols],
                                 cat(parts, "mac"), Lv, min(max(budget[k], Lv[0]), Lv[-1]),
                                 n_bins=args.bins)[0]
        tables[f"w{cname}"] = tw
        # global: one lambda across all keys at the parent's total cost
        target = sum(budget[k] * float(bins[k][1].sum()) for k in keys)
        rs = bisect(lambda lam: {k: staircase_dp(bins[k][0], bins[k][1], Lv, lam) for k in keys},
                    lambda r: sum(float((bins[k][1] * Lv[r[k]]).sum()) for k in keys), target)
        tables[f"g{cname}"] = {k: th_from(rs[k], bins[k][2]) for k in keys}
        mv = defaultdict(lambda: [0.0, 0.0, 0.0])
        for k in keys:
            Cs = float(bins[k][1].sum())
            mv[k[0]][0] += Cs
            mv[k[0]][1] += float((bins[k][1] * Lv[rs[k]]).sum())
            mv[k[0]][2] += Cs * budget[k]
        print(f"[c6] global-{cname}: " + "  ".join(f"{o} {p / m:.1f}->{g / m:.1f}"
                                                   for o, (m, g, p) in sorted(mv.items())))
    for nm, bk in score_tables.items():
        tables[nm] = {k: bk[f"{k[0]}:t0:l{k[1]}"]["thresholds"] for k in keys
                      if f"{k[0]}:t0:l{k[1]}" in bk}
        tables[nm]["__levels__"] = {k: bk[f"{k[0]}:t0:l{k[1]}"]["levels"] for k in keys
                                    if f"{k[0]}:t0:l{k[1]}" in bk}

    # held-out scoring of every table in every currency + Fisher-predicted dNLL
    hkeys = [k for k in keys if store["hold"].get(k)]
    score_cur = ["raw", "rel"] + [t for t in WEIGHT_TAGS
                                  if all(t in p for k in hkeys for p in store["hold"][k])]
    n_tok = args.holdout_windows * 2047
    HS = {}
    for tname in ["parent"] + list(tables):
        acc = defaultdict(float)
        for k in hkeys:
            hp = store["hold"][k]
            mac, hm, parL = cat(hp, "mac"), cat(hp, "mn"), cat(hp, "parL")
            ar = np.arange(mac.size)
            if tname == "parent":
                Lrow = parL
            else:
                th = tables[tname].get(k)
                if th is None:
                    Lrow = parL
                else:
                    lv_ = tables[tname].get("__levels__", {}).get(k, ladder)
                    Lrow = np.asarray(lv_, dtype=np.float64)[deploy_rungs(torch.from_numpy(hm), th).numpy()]
            cols = np.asarray([li[int(v)] for v in Lrow])
            acc["LM"] += float((mac * Lrow).sum())
            acc["M"] += float(mac.sum())
            for c in score_cur:
                acc[c] += float(cur_of(hp, c)[ar, cols].sum())
        HS[tname] = acc
    par = HS["parent"]
    for tname, acc in HS.items():
        row = {"L_over_parent": (acc["LM"] / acc["M"]) / (par["LM"] / par["M"])}
        for c in score_cur:
            row[f"{c}_over_parent"] = acc[c] / max(par[c], 1e-30)
        for t in WEIGHT_TAGS:
            if t in score_cur:
                row[f"pred_dnll_{t}"] = 0.5 * (acc[t] - par[t]) / n_tok
                row[f"pred_dppl_{t}_pct"] = (math.exp(row[f"pred_dnll_{t}"]) - 1) * 100
        diag["tables"][tname] = row
        print(f"[c6] held-out {tname:10s} L/par {row['L_over_parent']:.3f}  " +
              "  ".join(f"{c} {row[f'{c}_over_parent']:.3f}" for c in score_cur) +
              "".join(f"  pred dPPL({t}) {row[f'pred_dppl_{t}_pct']:+.2f}%" for t in WEIGHT_TAGS
                      if f"pred_dppl_{t}_pct" in row))

    # oracles (need the FP reference): within / global per currency, ladder and all lengths
    orc = {}
    for c in score_cur:
        for lvl_name, cols_sel, LL in (("ladder", lad_cols, Lv),
                                       ("all", list(range(len(measured))), np.asarray(measured, float))):
            CUR = {k: cur_of(store["hold"][k], c)[:, cols_sel] for k in hkeys}
            MAC = {k: cat(store["hold"][k], "mac") for k in hkeys}
            T = {k: float((MAC[k] * cat(store["hold"][k], "parL")).sum()) for k in hkeys}
            wtot = 0.0
            for k in hkeys:
                j = bisect(lambda lam: np.argmin(CUR[k] + lam * MAC[k][:, None] * LL[None, :], 1),
                           lambda j: float((MAC[k] * LL[j]).sum()), T[k])
                wtot += float(CUR[k][np.arange(MAC[k].size), j].sum())
            CA = np.concatenate([CUR[k] for k in hkeys])
            MA = np.concatenate([MAC[k] for k in hkeys])
            j = bisect(lambda lam: np.argmin(CA + lam * MA[:, None] * LL[None, :], 1),
                       lambda j: float((MA * LL[j]).sum()), sum(T.values()))
            gtot = float(CA[np.arange(MA.size), j].sum())
            ent = {"within_over_parent": wtot / max(par[c], 1e-30),
                   "global_over_parent": gtot / max(par[c], 1e-30)}
            if c in WEIGHT_TAGS:
                ent["global_pred_dppl_pct"] = (math.exp(0.5 * (gtot - par[c]) / n_tok) - 1) * 100
                ent["within_pred_dppl_pct"] = (math.exp(0.5 * (wtot - par[c]) / n_tok) - 1) * 100
            orc[f"{c}/{lvl_name}"] = ent
            print(f"[c6] ORACLE {c:8s} {lvl_name:6s}: within {ent['within_over_parent']:.3f}  "
                  f"GLOBAL {ent['global_over_parent']:.3f}" +
                  (f"   pred dPPL within {ent['within_pred_dppl_pct']:+.2f}%  GLOBAL "
                   f"{ent['global_pred_dppl_pct']:+.2f}%" if c in WEIGHT_TAGS else ""))
    diag["oracles"] = orc

    # emit tables
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    from scmp_kernels.mp.config import AdaptiveMPConfig as _AMP
    for tname in [t for t in tables if t[0] in "wg" and t[1:] in currencies]:
        bk = {f"{k[0]}:t0:l{k[1]}": {"levels": [int(v) for v in ladder], "thresholds": th}
              for k, th in tables[tname].items() if k != "__levels__"}
        o = stem.with_name(f"{stem.stem}_{tname}.json")
        t_ = o.with_name(o.stem + "_table.json")
        pt = dict(table)
        pt["per_row_chunk"] = {"buckets": bk}
        pt.setdefault("sc_prec", SC_PREC)
        pt.setdefault("halve_bipolar_stoc_len", HALVE)
        pt["prc6_calib"] = dict(vars(args), table=tname)
        t_.write_text(json.dumps(pt, indent=1))
        pw = dict(wrapper)
        pw["threshold_table_path"] = str(t_.resolve())
        o.write_text(json.dumps(pw, indent=1))
        chk = _AMP(sorted([int(v) for v in pt["stoc_len_levels"]], reverse=True))
        chk.load_threshold_table(str(t_))
        assert len(chk.per_row_chunk) == len(bk), "round-trip lost buckets"
        print(f"[c6] wrote {o.name} ({len(bk)} buckets, round-trip verified)")
    Path(args.diag).write_text(json.dumps(diag, indent=1, default=float))
    print(f"[c6] diag -> {args.diag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
