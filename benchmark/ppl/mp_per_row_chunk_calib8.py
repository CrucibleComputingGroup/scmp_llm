"""calib8 — round 3: finer DEPLOYABLE keys for the linears under the same global Fisher lambda.

Round 2 (calib7, attention in the joint lambda) lost on held-out NLL on 4B and llama8B: the
diagonal Fisher under-prices attention error, so attention stays at the parent's allocation
here (--att 0, the default, skips the attention capture). What remains is the gap between the
deployable global-Fisher staircase (one threshold vector per (op, layer-quartile)) and the
loss-aware per-group oracle. calib8 measures which deployable refinements close it, all
under ONE global lambda at the parent's cost, all scored in Fisher-predicted dNLL on the
held-out calibration windows:
  q        (op, layer-quartile) keys             = calib6/7's gfis  [emitted: <stem>_gfis]
  blk      (op, block) keys                      [emitted: <stem>_gfisblk; needs
           per_row_chunk.layer_buckets = n_blocks, same runtime rule, finer table]
  q_ccK    (op, quartile, static chunk class)    [analysis only]
  blk_ccK  (op, block, static chunk class)       [analysis only]
Static chunk class: chunks of one (op, block) ranked by Fisher error per unit metric^2 at a
mid rung on the CALIBRATION windows (captures the weight-column / AWQ factor the absmax
statistic cannot see), cut into K equal-count classes. Dense models only (MoE experts
share (op, block) keys, so no per-expert class map).

calib7 docstring follows.
calib7 — calib6 + ATTENTION in the same global Fisher allocation (round 2).

Linears keep calib6's per-(row,chunk) groups. Attention rows (qk: one query row of the
score product; av: one probability row of the value product) are priced in the SAME
Fisher currency (g = dLoss/d(attention product output), captured on the deployed SC
trajectory through the exact straight-through matmul) and enter the SAME lambda, so the
solve moves cycles between the linears and attention wherever the loss says. Attention keeps
the runtime's per-row rule (call-normalized per-row metric, per-(op, layer-quartile) bucket
thresholds on the parent's bucket ladder, mu+2tau escape rows fixed at the escape length);
only its thresholds change. Emits:
  <stem>_gfis   : global Fisher, linears only (attention = parent)  [calib6's headline]
  <stem>_gfisla : global Fisher, linears + attention in one lambda   [round 2]

calib6 docstring follows.
calib6 — per-group allocation with a LOSS-AWARE (Fisher) currency, global and within-bucket.

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

    def __init__(self, sc_common, SCLinear, capture, capture_att=None):
        self.scm, self.SCLinear, self.capture = sc_common, SCLinear, capture
        self.capture_att = capture_att

    def __enter__(self):
        scm, SCL, cap, capa = self.scm, self.SCLinear, self.capture, self.capture_att
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
                z = _STEMatmulABt.apply(a, b, out)
                if capa is not None:
                    capa(kw, z)
                return z
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
    ap.add_argument("--currencies", default="fis")
    ap.add_argument("--att-rows-per-call", type=int, default=128)
    ap.add_argument("--dump", default="", help="npz path: per-pair arrays (analysis only)")
    ap.add_argument("--grad-mode", default="full", choices=("full", "block"),
                    help="full: one autograd graph for the whole model (4B/llama8B); block: "
                         "store each decoder layer's input in a no-grad forward, then back-"
                         "propagate one layer at a time (same math, one layer's graph in "
                         "memory; needed for 30B)")
    ap.add_argument("--expand-at", default="",
                    help="wrapper whose per_row_chunk table the model RUNS during calibration "
                         "(Fisher + activations taken on that trajectory = a second, "
                         "re-linearized allocation pass); budgets/parent lengths stay the parent's")
    ap.add_argument("--att", type=int, default=0,
                    help="1: also capture/measure attention and run calib7's round-2 joint solve")
    ap.add_argument("--min-pairs-per-bin", type=int, default=8,
                    help="round-3 sub-keys: bins = min(--bins, pairs // this)")
    ap.add_argument("--cc-classes", default="2,4", help="round-3 chunk-class counts to analyse")
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

    run_wrapper = args.expand_at or str(parent / "wrapper.json")
    if args.expand_at:
        os.environ["MP_CONFIG_JSON"] = args.expand_at
    model, tok = build_sc_model(args.model_path, "mp", mp_table=run_wrapper)
    if args.expand_at:
        nprc = len(getattr(model.config.sc_mp_config, "per_row_chunk", {}) or {})
        if nprc == 0:
            raise SystemExit(f"[c8] --expand-at {args.expand_at} loaded no per_row_chunk table")
        print(f"[c8] calibrating ON the trajectory of {args.expand_at} ({nprc} per-group buckets); "
              "budgets and parent lengths from the parent's per-row rule")
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
    ATT_OPS = ("qk", "av")

    def sel_att(op, blk, win, total, device):
        g = torch.Generator(device="cpu")
        g.manual_seed(zlib.crc32(f"att:{args.seed}:{op}:{blk}:{win}".encode()))
        return torch.randperm(total, generator=g)[:min(args.att_rows_per_call, total)].to(device)

    def capture_att(kw, z):
        op, blk = kw.get("operator"), kw.get("block_idx")
        if op not in ATT_OPS or blk is None or not z.requires_grad:
            return
        if _hybrid_backend(config, op, blk) != "sc":
            return
        B, H, N, M = z.shape
        idx = sel_att(op, blk, state["win"], B * H * N, z.device)
        key = ("att", op, blk)

        def hk(gr):
            g2 = gr.reshape(B * H * N, M).index_select(0, idx).float().pow(2)
            ent = G.setdefault(key, {})
            ent[state["gtag"]] = ent.get(state["gtag"], 0) + g2
        z.register_hook(hk)

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
            cidx=np.tile(np.arange(mn.shape[1], dtype=np.int16), R),
            rpos=np.repeat(rows.cpu().numpy().astype(np.int32), mn.shape[1]),
            blk=np.full(R * mn.shape[1], int(blk), dtype=np.int16),
            win=np.full(R * mn.shape[1], int(state["win"]), dtype=np.int16),
            rowmax=np.repeat(mn.index_select(0, rows).amax(dim=1).cpu().numpy(), mn.shape[1]),
            parL=parL.index_select(0, rows)[:, None].expand(-1, mn.shape[1]).reshape(-1)
            .cpu().numpy())
        for t in WEIGHT_TAGS:
            if t in wt:
                rec[t] = (wt[t] * rep).reshape(-1, len(measured)).cpu().numpy()
        store[state["phase"]][(op, b)].append(rec)

    att_store = {"calib": defaultdict(list), "hold": defaultdict(list)}
    att_ladder = {}

    def measure_att(a, b, kw):
        op, blk = kw.get("operator"), kw.get("block_idx")
        if op not in ATT_OPS or blk is None or _hybrid_backend(config, op, blk) != "sc":
            return
        B, H, N, K = a.shape
        M = b.shape[-2]
        a3 = a.reshape(B * H, N, K).float()
        b3 = b.reshape(B * H, M, K).float()
        met = _mp_dispatch_metric(a3, mp_cfg, op).reshape(-1)
        mn_all = (met - met.min()) / (met.max() - met.min()).clamp_min(1e-8)
        asg = adaptive_classify_rows(met, mp_cfg, operator=op, block_idx=blk,
                                     total_blocks=total_blocks)
        vals = mp_cfg.classify_level_values(operator=op, block_idx=blk,
                                            total_blocks=total_blocks)
        lad = sorted(int(v) for v in mp_cfg.get_levels(operator=op, block_idx=blk,
                                                       total_blocks=total_blocks))
        t_esc = mp_cfg.get_escape_threshold(0, 1, operator=op, block_idx=blk,
                                            total_blocks=total_blocks)
        esc_len = int(getattr(mp_cfg, "escape_stoc_len", 0) or 0)
        lv_meas = sorted(set(lad) | ({esc_len} if (t_esc is not None and esc_len) else set()))
        bkt = bucket_of(blk, total_blocks, n_lb)
        th_par = mp_cfg.get_thresholds(0, 1, operator=op, block_idx=blk,
                                       total_blocks=total_blocks)
        att_ladder[(op, bkt)] = (lad, t_esc, esc_len, lv_meas,
                                 [float(v) for v in th_par] if th_par is not None else None)
        idx = sel_att(op, blk, state["win"], B * H * N, a.device)
        R = idx.numel()
        parL = torch.as_tensor(vals, device=a.device, dtype=torch.float32)[
            asg.row_levels.index_select(0, idx)]
        bh, n = idx // N, idx % N
        raw = torch.zeros(R, len(lv_meas), device=a.device)
        gw = G.get(("att", op, blk), {})
        fis = {t: torch.zeros(R, len(lv_meas), device=a.device)
               for t in WEIGHT_TAGS if t in gw and gw[t].shape[0] == R}
        sm = kw.get("smooth_scales")
        for h in torch.unique(bh).tolist():
            sel = (bh == h).nonzero(as_tuple=True)[0]
            a_sub = a3[h].index_select(0, n.index_select(0, sel)).contiguous()
            ref = a_sub @ b3[h].t()
            for li, L in enumerate(lv_meas):
                o = scm._sc_matmul(a_sub, b3[h].contiguous(), granularity="per_row",
                                   mode=kw.get("mode", "bipolar"), sc_prec=kw.get("sc_prec", SC_PREC),
                                   stoc_len=int(L),
                                   halve_bipolar_stoc_len=kw.get("halve_bipolar_stoc_len", HALVE),
                                   smooth_scales=sm, rng_levels=scm._attn_grid_for(op, int(L)))
                d2 = (o.float() - ref) ** 2
                raw[sel, li] = d2.sum(dim=1)
                for t in fis:
                    fis[t][sel, li] = (d2 * gw[t].index_select(0, sel)).sum(dim=1)
        rep_ = (B * H * N) / R
        rec = dict(mn=mn_all.index_select(0, idx).cpu().numpy(),
                   raw=(raw * rep_).cpu().numpy(),
                   mac=np.full(R, float(K * M) * rep_),
                   parL=parL.cpu().numpy(), lv=np.asarray(lv_meas))
        for t in fis:
            rec[t] = (fis[t] * rep_).cpu().numpy()
        att_store[state["phase"]][(op, bkt)].append(rec)

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
            with GradPass(scm, SCLinear, capture, capture_att if args.att else None):
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
                    with GradPass(scm, SCLinear, capture, capture_att if args.att else None), torch.enable_grad():
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
            _orig_att = scm._sc_attention_matmul_ab_t

            def _att_meas(a, b, **kw):
                out = _orig_att(a, b, **kw)
                measure_att(a, b, kw)
                return out
            if args.att:
                scm._sc_attention_matmul_ab_t = _att_meas
            try:
                with torch.no_grad():
                    model(input_ids=ids)
            finally:
                scm._sc_attention_matmul_ab_t = _orig_att
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

    # ============ ROUND 3: finer deployable keys, one global Fisher lambda ============
    r3_tables = {}
    if "fis" in currencies:
        cur3 = "fis"
        blk_all = {k: cat(store["calib"][k], "blk") for k in keys}
        cidx_all = {k: cat(store["calib"][k], "cidx") for k in keys}
        # static chunk classes per (op, block): Fisher error per unit metric^2 at a mid rung
        ref_col = li[min(ladder, key=lambda L: abs(L - 24))]
        cls_of = {}
        dense = not any(bool(cat(store["calib"][k], "expert").any()) for k in keys)
        cc_list = [int(v) for v in args.cc_classes.split(",") if v] if dense else []
        if cc_list:
            for k in keys:
                cu = cur_of(store["calib"][k], cur3)[:, ref_col].astype(np.float64)
                m2 = cat(store["calib"][k], "mn").astype(np.float64) ** 2
                for b_ in np.unique(blk_all[k]):
                    sel = blk_all[k] == b_
                    c_ = cidx_all[k][sel].astype(np.int64)
                    nch = int(c_.max()) + 1
                    num = np.bincount(c_, weights=cu[sel], minlength=nch)
                    den = np.bincount(c_, weights=m2[sel], minlength=nch)
                    s_ = num / np.maximum(den, 1e-30)
                    rank = np.empty(nch, dtype=np.int64)
                    rank[np.argsort(s_, kind="stable")] = np.arange(nch)
                    cls_of[(k[0], int(b_))] = rank

        def subkey_fn(variant):
            by_blk = variant.startswith("blk")
            K = int(variant.split("cc")[1]) if "cc" in variant else 0

            def f(k, blk, cidx):
                base = blk.astype(np.int64) if by_blk else np.full(blk.shape, k[1], np.int64)
                if not K:
                    return base * 1000
                cl = np.empty(blk.shape, np.int64)
                for b_ in np.unique(blk):
                    sel = blk == b_
                    rk = cls_of[(k[0], int(b_))]
                    cl[sel] = rk[cidx[sel].astype(np.int64)] * K // rk.size
                return base * 1000 + cl
            return f

        variants = ["q", "blk"] + [f"{p_}_cc{K}" for K in cc_list for p_ in ("q", "blk")]
        target3 = sum(budget[k] * float(cat(store["calib"][k], "mac").sum()) for k in keys)
        n_tok3 = args.holdout_windows * 2047
        hkeys3 = [k for k in keys if store["hold"].get(k)]
        par_f = par_c = 0.0
        for k in hkeys3:
            hp = store["hold"][k]
            parL = cat(hp, "parL")
            cols = np.asarray([li[int(v)] for v in parL])
            par_f += float(cur_of(hp, cur3)[np.arange(parL.size), cols].sum())
            par_c += float((cat(hp, "mac") * parL).sum())
        diag["round3"] = {}
        for var in variants:
            fn = subkey_fn(var)
            sb = {}
            for k in keys:
                parts = store["calib"][k]
                sk = fn(k, blk_all[k], cidx_all[k])
                mn_, cu_, mac_ = cat(parts, "mn"), cur_of(parts, cur3)[:, lad_cols], cat(parts, "mac")
                for s_ in np.unique(sk):
                    sel = sk == s_
                    m = mn_[sel]
                    order = np.argsort(m, kind="stable")
                    P = m.size
                    if var == "q":
                        B = int(min(args.bins, P))       # identical to calib6/7's gfis bins
                    else:
                        B = int(max(1, min(args.bins, P // max(args.min_pairs_per_bin, 1))))
                    edges = np.linspace(0, P, B + 1).astype(np.int64)
                    E = np.add.reduceat(cu_[sel][order], edges[:-1], axis=0)
                    C = np.add.reduceat(mac_[sel][order], edges[:-1])
                    sb[(k, int(s_))] = (E, C, m[order][edges[1:] - 1])
            rs = bisect(lambda lam: {q_: staircase_dp(v[0], v[1], Lv, lam) for q_, v in sb.items()},
                        lambda r: sum(float((sb[q_][1] * Lv[r[q_]]).sum()) for q_ in sb), target3)
            th3 = {q_: th_from(rs[q_], sb[q_][2]) for q_ in sb}
            tot_f = tot_c = 0.0
            for k in hkeys3:
                hp = store["hold"][k]
                hm, mac, blk_h, cid_h = cat(hp, "mn"), cat(hp, "mac"), cat(hp, "blk"), cat(hp, "cidx")
                sk = fn(k, blk_h, cid_h)
                Lr = np.empty(hm.size)
                for s_ in np.unique(sk):
                    sel = sk == s_
                    th = th3.get((k, int(s_)))
                    if th is None:
                        raise SystemExit(f"[c8] held-out sub-key {k}/{s_} unseen in calibration")
                    Lr[sel] = Lv[deploy_rungs(torch.from_numpy(hm[sel]), th).numpy()]
                cols = np.asarray([li[int(v)] for v in Lr])
                tot_f += float(cur_of(hp, cur3)[np.arange(hm.size), cols].sum())
                tot_c += float((mac * Lr).sum())
            d = 0.5 * (tot_f - par_f) / n_tok3
            diag["round3"][var] = {"n_subkeys": len(sb), "cost_over_parent": tot_c / par_c,
                                   "fis_over_parent": tot_f / par_f, "pred_dnll": d,
                                   "pred_dppl_pct": (math.exp(d) - 1) * 100}
            print(f"[c8] round3 {var:9s}: {len(sb):5d} sub-keys  held-out cost/parent "
                  f"{tot_c / par_c:.4f}  fis/parent {tot_f / par_f:.3f}  pred dPPL(fis) "
                  f"{(math.exp(d) - 1) * 100:+.2f}%", flush=True)
            if var == "blk":
                r3_tables["gfisblk"] = {(k[0], int(s_) // 1000): th for (k, s_), th in th3.items()}

    # ============ ROUND 2: linears + attention in ONE lambda (Fisher currency) ============
    def deploy_att(mn, th_desc, lad_asc, t_esc, esc_len):
        """Mirror _classify_rows_by_thresholds (+ escape): descending ladder, level i if
        mn >= th[i] (first match), else the last (shortest) level; mn > t_esc -> escape."""
        D = list(reversed(lad_asc))
        lvl = np.full(mn.shape, len(D) - 1)
        for i in range(len(th_desc) - 1, -1, -1):
            lvl[mn >= th_desc[i]] = i
        L = np.asarray(D, dtype=np.float64)[lvl]
        if t_esc is not None and t_esc < 1.0 and esc_len:
            L[mn > t_esc] = esc_len
        return L

    joint = {}
    acur = "fis" if "fis" in currencies else None
    akeys = sorted(k for k in att_store["calib"]
                   if acur and all(acur in p for p in att_store["calib"][k]))
    if acur and set(att_store["calib"]) - set(akeys):
        print(f"[c7] WARNING attention keys without Fisher: {sorted(set(att_store['calib']) - set(akeys))}")
    if acur and akeys:
        # replay check: the parent's own attention thresholds through deploy_att must
        # reproduce the parent's per-row lengths (validates the emitted semantics)
        agree = []
        for k in akeys:
            lad, t_esc, esc_len, lv_meas, th_par = att_ladder[k]
            for p_ in att_store["hold"].get(k, []) + att_store["calib"][k][:2]:
                if th_par is not None:
                    agree.append(float(np.mean(deploy_att(p_["mn"], th_par, lad, t_esc, esc_len)
                                               == p_["parL"])))
        print(f"[c7] attention replay of parent thresholds: agreement "
              f"{np.mean(agree) if agree else float('nan'):.5f} over {len(agree)} calls")
        diag["att_replay_agreement"] = float(np.mean(agree)) if agree else None
        lbins = {k: binned(store["calib"][k], cur_of(store["calib"][k], acur)) for k in keys}
        abins, fixed, par_att = {}, 0.0, 0.0
        for k in akeys:
            lad, t_esc, esc_len, lv_meas, _ = att_ladder[k]
            parts = att_store["calib"][k]
            mn, cur, mac, parL = cat(parts, "mn"), cat(parts, acur), cat(parts, "mac"), cat(parts, "parL")
            col = {int(L): i for i, L in enumerate(parts[0]["lv"])}
            esc = (mn > t_esc) if (t_esc is not None and t_esc < 1.0 and esc_len) else np.zeros(mn.shape, bool)
            A = np.asarray(lad, dtype=np.float64)
            ok = ~esc
            m_ok = mn[ok]
            order = np.argsort(m_ok, kind="stable")
            Bn = int(min(args.bins, max(m_ok.size, 1)))
            edges = np.linspace(0, m_ok.size, Bn + 1).astype(np.int64)
            E = np.add.reduceat(cur[ok][order][:, [col[int(L)] for L in lad]], edges[:-1], axis=0)
            C = np.add.reduceat(mac[ok][order], edges[:-1])
            abins[k] = (E, C, m_ok[order][edges[:-1]], A)
            if esc.any():
                fixed += float((mac[esc] * esc_len).sum())
            par_att += float((mac * parL).sum())
        par_lin = sum(budget[k] * float(lbins[k][1].sum()) for k in keys)
        target = par_lin + par_att

        def solve_j(lam):
            return ({k: staircase_dp(lbins[k][0], lbins[k][1], Lv, lam) for k in keys},
                    {k: staircase_dp(abins[k][0], abins[k][1], abins[k][3], lam) for k in akeys})

        def cost_j(r):
            return (sum(float((lbins[k][1] * Lv[r[0][k]]).sum()) for k in keys) +
                    sum(float((abins[k][1] * abins[k][3][r[1][k]]).sum()) for k in akeys) + fixed)
        rj = bisect(solve_j, cost_j, target)
        tables["gfisla"] = {k: th_from(rj[0][k], lbins[k][2]) for k in keys}
        att_th = {}
        for k in akeys:
            E, C, bmin, A = abins[k]
            r, n_ = rj[1][k], len(A)
            th = []
            for i in range(n_ - 1):
                hit = np.nonzero(r >= n_ - 1 - i)[0]
                th.append(1.0 if hit.size == 0 else (0.0 if hit[0] == 0 else float(bmin[hit[0]])))
            for i in range(1, len(th)):
                th[i] = min(th[i], th[i - 1])
            att_th[k] = th
        joint["att_th"] = att_th
        lin_L = sum(float((lbins[k][1] * Lv[rj[0][k]]).sum()) for k in keys) / sum(float(lbins[k][1].sum()) for k in keys)
        lin_par = par_lin / sum(float(lbins[k][1].sum()) for k in keys)
        att_mac = sum(float(cat(att_store["calib"][k], "mac").sum()) for k in akeys)
        att_L = (sum(float((abins[k][1] * abins[k][3][rj[1][k]]).sum()) for k in akeys) + fixed) / att_mac
        print(f"[c7] JOINT lambda: linear L {lin_par:.2f} -> {lin_L:.2f}; attention L "
              f"{par_att / att_mac:.2f} -> {att_L:.2f} (total cost held at the parent's)")
        for k in akeys:
            A = abins[k][3]
            Ck = abins[k][1]
            print(f"[c7]   {k[0]} l{k[1]}: ladder {list(map(int, A))}  L -> "
                  f"{float((Ck * A[rj[1][k]]).sum() / max(Ck.sum(), 1e-9)):.1f}  th {[round(v, 4) for v in att_th[k]]}")
        # held-out: Fisher-predicted dNLL incl. attention, for parent / gfis / gfisla
        n_tok = args.holdout_windows * 2047
        HJ = {}
        for tname in ("parent", "gfis", "gfisla"):
            tot_f = tot_c = 0.0
            for k in keys:
                hp = store["hold"].get(k)
                if not hp:
                    continue
                mac, hm, parL = cat(hp, "mac"), cat(hp, "mn"), cat(hp, "parL")
                if tname == "parent" or tname not in tables:
                    Lr = parL
                else:
                    Lr = Lv[deploy_rungs(torch.from_numpy(hm), tables[tname][k]).numpy()]
                cols = np.asarray([li[int(v)] for v in Lr])
                tot_f += float(cur_of(hp, acur)[np.arange(mac.size), cols].sum())
                tot_c += float((mac * Lr).sum())
            for k in akeys:
                lad, t_esc, esc_len, lv_meas, th_par = att_ladder[k]
                for p_ in att_store["hold"].get(k, []):
                    col = {int(L): i for i, L in enumerate(p_["lv"])}
                    if tname == "gfisla":
                        Lr = deploy_att(p_["mn"], att_th[k], lad, t_esc, esc_len)
                    else:
                        Lr = p_["parL"]
                    cols = np.asarray([col[int(v)] for v in Lr])
                    tot_f += float(p_[acur][np.arange(Lr.size), cols].sum())
                    tot_c += float((p_["mac"] * Lr).sum())
            HJ[tname] = (tot_f, tot_c)
        for tname, (f_, c_) in HJ.items():
            d = 0.5 * (f_ - HJ["parent"][0]) / n_tok
            print(f"[c7] held-out (linear+attention) {tname:7s}: cost/parent {c_ / HJ['parent'][1]:.4f}  "
                  f"pred dPPL(fis) {(math.exp(d) - 1) * 100:+.2f}%")
            joint[tname] = {"cost_over_parent": c_ / HJ["parent"][1], "pred_dnll": d}
        diag["joint"] = {k: v for k, v in joint.items() if k != "att_th"}
        diag["joint"]["att_th"] = {f"{k[0]}:l{k[1]}": v for k, v in att_th.items()}

    # emit tables
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    from scmp_kernels.mp.config import AdaptiveMPConfig as _AMP
    for tname in [t for t in tables if (t[0] in "wg" and t[1:] in currencies) or t == "gfisla"]:
        bk = {f"{k[0]}:t0:l{k[1]}": {"levels": [int(v) for v in ladder], "thresholds": th}
              for k, th in tables[tname].items() if k != "__levels__"}
        o = stem.with_name(f"{stem.stem}_{tname}.json")
        t_ = o.with_name(o.stem + "_table.json")
        pt = json.loads(json.dumps(table))
        if tname == "gfisla":
            for (op_, b_), th in joint["att_th"].items():
                kk = f"{op_}:t0:l{b_}"
                if kk not in pt["buckets"]:
                    raise SystemExit(f"[c7] parent table has no bucket {kk}")
                if len(pt["buckets"][kk]["thresholds"]) != len(th):
                    raise SystemExit(f"[c7] threshold length mismatch for {kk}")
                pt["buckets"][kk]["thresholds"] = th
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
    for tname, th_ in r3_tables.items():
        bk = {f"{op_}:t0:l{b_}": {"levels": [int(v) for v in ladder], "thresholds": th}
              for (op_, b_), th in th_.items()}
        o = stem.with_name(f"{stem.stem}_{tname}.json")
        t_ = o.with_name(o.stem + "_table.json")
        pt = json.loads(json.dumps(table))
        pt["per_row_chunk"] = {"layer_buckets": int(total_blocks), "buckets": bk}
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
        for (op_, b_), th in th_.items():
            got = chk.get_per_row_chunk(op_, b_, int(total_blocks))
            assert got is not None and [float(v) for v in got[1]] == [float(v) for v in th], \
                f"per-block lookup mismatch at {op_} block {b_}"
        print(f"[c8] wrote {o.name} ({len(bk)} per-block buckets, lookup verified)")
    Path(args.diag).write_text(json.dumps(diag, indent=1, default=float))
    if args.dump:
        out_ = {"measured": np.asarray(measured), "ladder": np.asarray(ladder)}
        for ph in ("calib", "hold"):
            for k, parts in store[ph].items():
                tag = f"{ph}|{k[0]}|{k[1]}"
                for f_ in ("mn", "mac", "parL", "cidx", "rpos", "blk", "rowmax", "yE", "win", "expert"):
                    out_[f"{tag}|{f_}"] = cat(parts, f_)
                for f_ in ("raw", "fis", "fis_emp"):
                    if all(f_ in p_ for p_ in parts):
                        out_[f"{tag}|{f_}"] = cat(parts, f_).astype(np.float32)
        np.savez_compressed(args.dump, **out_)
        print(f"[c7] dumped per-pair arrays -> {args.dump}")
    print(f"[c6] diag -> {args.diag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
