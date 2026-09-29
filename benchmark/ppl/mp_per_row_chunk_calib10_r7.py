"""calib10_r7 — calib9_r6 + round-7 CAPTURE options (every one OFF by default). Created
2026-09-28 as a NEW file; calib7 and calib9_r6 stay frozen (both are hashed by running jobs).

Round 7 (T1 per-group allocation, kbands/ROUNDS_6_8_PLAN_20260928.md section 4) re-solves the
joint linear + attention lambda offline with (a) per-bucket attention ladders that reach the
code cap 128, (b) a family-calibrated currency, (c) 512 attention rows per call, (d) a
parent-cost target. This calibrator only MEASURES and DUMPS what those CPU solves need; the
solves, the kappa currency, per-bucket ladder emission and validation live in
benchmark/ppl/prc_r7_solve.py.

Identity with calib9_r6 (hence with calib7, the incumbents' recipe):
  * calib9's module-level solve / emit / solve-state / error-curve / straight-through code is
    IMPORTED from mp_per_row_chunk_calib9_r6 (not copied), so it is calib9's by identity.
  * main() is calib9's main() verbatim except lines tagged `# r7`. Every tagged line is inert
    at the defaults below; test_prc_r7_calib_20260928.py audits the diff line by line and
    unit-tests every helper at its default.
New flags (defaults reproduce calib9 exactly):
  --att-prefix-split P   (default 0)  measure each head's rows with sample rank < P as their
        own batch, then the rest. With --att-rows-per-call 512 and P = 128 the rank<128 rows
        are calib7's 128-row sample (same crc32-seeded randperm prefix) measured in calib7's
        exact per-head batches, so the incumbent's joint solve is reproducible bit for bit
        from the 512-row capture (prc_r7_solve.py identity, prefix x4 = rep 512/128).
  --r7-hold-dump PATH    (default "")  hold-phase linear per-(row,chunk) arrays (mn, mac, parL,
        blk, cidx, rpos, expert, fis, fis_emp at every measured length) + protected-slice
        records, for full-table held-out scoring on CPU (linears + attention + ladders).
  --prot-measure L1,..   (default "")  also measure the PROTECTED slice's Fisher error per
        sampled row at these lengths (the runtime's protected SC matmul: chunk_d 128, one
        length for the whole row). Dump only (a CRIT L7 diagnostic); no solve uses it.
  With any r7 flag set, --solve-state writes schema v2 = calib9's v1 + an `r7` meta block
  (attention part sizes, escape thresholds, windows, n_tok, dump paths).
calib9_r6 flags used by round 7, unchanged: --att-measure-extra (extra measured attention
lengths, <= 128), --att-dump (per-row attention records incl. sample rank), --solve-state.
Units: HALVED code lengths (nominal = 2x); the code cap is 128 and is asserted.

The calib9_r6 / calib7 / calib6 docstrings (the method this main() runs) are in
benchmark/ppl/mp_per_row_chunk_calib9_r6.py.
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
# calib9_r6's module-level code, imported (identity by object, not by copy).
from benchmark.ppl.mp_per_row_chunk_calib9_r6 import (  # noqa: E402,F401
    WEIGHT_TAGS, SCALE_NAME_RE, scale_suffix, base_table_name, emitted_table_names,
    parse_scales, scaled_target, th_from_bins, bisect_lambda, att_th_from, solve_global,
    joint_cost, solve_joint, emit_table, _kstr, _kparse, save_solve_state, load_solve_state,
    resolve_tables, error_curves_w, _STELinear, _STEMatmulABt, GradPass)

# ============================== calib10_r7: new helpers (pure) ==============================
R7_NEW_ARGS = ("att_prefix_split", "r7_hold_dump", "prot_measure")
R7_DEFAULTS = {"att_prefix_split": 0, "r7_hold_dump": "", "prot_measure": ""}
R7_STATE_SCHEMA = "prc-calib10-r7-solve-state-v2"
R7_CALIBRATOR = "mp_per_row_chunk_calib10_r7.py"


def r7_active(args) -> bool:
    return any(getattr(args, a) != R7_DEFAULTS[a] for a in R7_NEW_ARGS)


def r7_adjust_records(args, calib7_view, c9_record, scales, c9_new_args):
    """Keep calib9's provenance exactly at the r7 defaults.

    Returns (calib7_view, c9_record, r7_record). calib7_view never carries the r7 flags (so the
    tables' prc6_calib record stays calib7's); with every r7 flag at its default the inputs
    come back unchanged (same objects) and r7_record is None. With an r7 flag set, the r7
    record rides inside calib9's prc9_r6 provenance block (created if calib9's was None)."""
    view = {k: v for k, v in calib7_view.items() if k not in R7_NEW_ARGS}
    if not r7_active(args):
        if view != calib7_view:
            raise AssertionError("r7 flags leaked into calib7_view")
        return calib7_view, c9_record, None
    r7_record = {"calibrator": R7_CALIBRATOR, **{a: getattr(args, a) for a in R7_NEW_ARGS},
                 "att_rows_per_call": int(args.att_rows_per_call)}
    base = (dict(c9_record) if c9_record is not None else
            {"calibrator": "mp_per_row_chunk_calib9_r6.py", "target_scales": scales,
             **{a: getattr(args, a) for a in c9_new_args}})
    base["calibrator"] = R7_CALIBRATOR
    base["r7"] = r7_record
    return view, base, r7_record


def r7_parse_lengths(spec: str, cap: int) -> list:
    """'48,64,128' -> sorted unique ints in [1, cap]; '' -> []."""
    out = sorted({int(v) for v in str(spec).split(",") if v.strip()})
    if any(not (1 <= L <= cap) for L in out):
        raise SystemExit(f"[c10] lengths {out} outside [1, {cap}] (stoc_len <= 2**sc_prec, halved)")
    return out


def r7_check_prefix(prefix: int, att_rows: int) -> int:
    """0 = off (calib9). Otherwise 0 < P < att_rows and att_rows / P a power of two, so the
    prefix rescale rep(P) = rep(att_rows) * (att_rows / P) is exact in floating point."""
    p = int(prefix)
    if p == 0:
        return 0
    ratio = int(att_rows) // p if p > 0 else 0
    if not (0 < p < int(att_rows)) or ratio * p != int(att_rows) or ratio & (ratio - 1):
        raise SystemExit(f"[c10] --att-prefix-split {p} must divide --att-rows-per-call "
                         f"{att_rows} by a power of two")
    return p


def r7_head_selections(bh, prefix: int):
    """The per-head row selections measure_att batches, in processing order.

    prefix == 0: calib9's loop exactly, [(h, (bh == h).nonzero()[0]) for h in unique(bh)].
    prefix  > 0: each head's selection split into sample ranks < prefix, then >= prefix
    (bh is indexed by sample rank, so `sel < prefix` is the rank test); order within each
    part is preserved and empty parts are dropped. Each part is measured as its own batch."""
    out = []
    for h in torch.unique(bh).tolist():
        sel = (bh == h).nonzero(as_tuple=True)[0]
        if prefix:
            for part in (sel[sel < prefix], sel[sel >= prefix]):
                if part.numel():
                    out.append((h, part))
        else:
            out.append((h, sel))
    return out


def r7_protected_record(xs_full, w_full, smooth_full, pidx, levels, weights, rep, blk, rows,
                        is_expert):
    """Fisher/raw error of the PROTECTED slice for the sampled rows (dump only).

    Mirrors the runtime's protected SC matmul (model/sc_common.py SCLinear MP path:
    x[:, protected] @ w[:, protected]^T at ONE length per row, chunk_d = 128, the protected
    channels' smooth scales) through calib9's error_curves_w, then sums over the protected
    chunks (the calibrators' additive per-chunk currency). mac = n_protected * d_out * rep."""
    xp = xs_full.index_select(1, pidx).contiguous()
    wp = w_full.index_select(1, pidx).contiguous()
    sp = smooth_full.index_select(0, pidx).contiguous() if smooth_full is not None else None
    raw, wt, _ = error_curves_w(xp, wp, sp, levels, weights)
    R = xp.shape[0]
    rec = dict(mac=np.full(R, float(pidx.numel() * wp.shape[0]) * rep),
               raw=(raw.sum(dim=1) * rep).cpu().numpy(),
               blk=np.full(R, int(blk), dtype=np.int16),
               rpos=rows.cpu().numpy().astype(np.int32),
               expert=np.full(R, bool(is_expert)))
    for t, v in wt.items():
        rec[t] = (v.sum(dim=1) * rep).cpu().numpy()
    return rec


def r7_state_meta(*, args, att_store, hold_store, measured, ladder, starts, hold_idx,
                  prot_levels, total_blocks, n_lb, n_units, top_k, currencies, cap):
    """The r7 block added to the solve-state meta (schema v2). Everything a CPU re-solve or
    scorer needs that calib9's v1 meta lacks: attention part sizes per phase/key (per-call
    record boundaries inside the --att-dump arrays), linear hold part sizes, windows, n_tok."""
    def sizes(st):
        return {_kstr(k): [int(len(p["mn"])) for p in parts] for k, parts in sorted(st.items())}
    return {"schema": R7_STATE_SCHEMA, "r7": {
        "calibrator": R7_CALIBRATOR,
        "att_rows_per_call": int(args.att_rows_per_call),
        "att_prefix_split": int(args.att_prefix_split),
        "att_measure_extra": [int(v) for v in str(args.att_measure_extra).split(",") if v.strip()],
        "prot_levels": [int(v) for v in prot_levels],
        "att_part_sizes": {ph: sizes(att_store[ph]) for ph in ("calib", "hold")},
        "lin_hold_part_sizes": sizes(hold_store),
        "measured_lin": [int(v) for v in measured], "ladder": [int(v) for v in ladder],
        "windows": [int(s) for s in starts], "hold_idx": sorted(int(i) for i in hold_idx),
        "calib_windows": int(args.calib_windows), "holdout_windows": int(args.holdout_windows),
        "n_tok": int(args.holdout_windows) * 2047, "bins": int(args.bins),
        "total_blocks": int(total_blocks), "layer_buckets": int(n_lb),
        "num_experts": int(n_units), "top_k": int(top_k), "currencies": list(currencies),
        "cap": int(cap), "files": {"att_dump": args.att_dump, "hold_dump": args.r7_hold_dump,
                                   "diag": args.diag, "out": args.out}}}


def r7_write_hold_dump(path, hold_store, prot_store, measured, prot_levels):
    """Hold-phase linear per-pair arrays (+ protected records, both phases), dtypes as
    measured (no casts), so a CPU scorer reproduces calib9's held-out sums bit for bit."""
    out = {"measured": np.asarray(measured, dtype=np.int64),
           "prot_levels": np.asarray(prot_levels, dtype=np.int64)}
    for k, parts in sorted(hold_store.items()):
        tag = f"lin|hold|{k[0]}|{k[1]}"
        for f_ in ("mn", "mac", "parL", "blk", "cidx", "rpos", "expert"):
            out[f"{tag}|{f_}"] = np.concatenate([p[f_] for p in parts])
        for f_ in WEIGHT_TAGS:
            if all(f_ in p for p in parts):
                out[f"{tag}|{f_}"] = np.concatenate([p[f_] for p in parts])
    for ph in ("calib", "hold"):
        for k, parts in sorted(prot_store[ph].items()):
            tag = f"prot|{ph}|{k[0]}|{k[1]}"
            for f_ in ("mac", "raw", "blk", "rpos", "expert"):
                out[f"{tag}|{f_}"] = np.concatenate([p[f_] for p in parts])
            for f_ in WEIGHT_TAGS:
                if all(f_ in p for p in parts):
                    out[f"{tag}|{f_}"] = np.concatenate([p[f_] for p in parts])
    np.savez(path, **out)
    return sorted(out)


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
    ap.add_argument("--score-tables", default="",
                    help="name=TABLE.json,... extra per_row_chunk tables to score (c17, g5)")
    # ---- calib9_r6 additions; every default reproduces calib7 exactly ----
    ap.add_argument("--target-scales", "--target-scale", dest="target_scales", default="1.0",
                    help="comma list of budget scales for the global and joint solves; 1.0 = "
                         "calib7's target and names; s != 1.0 adds <name>_sNNNNN tables")
    ap.add_argument("--solve-state", default="",
                    help="npz path: binned solve inputs for CPU re-solves (prc_resolve_r6.py)")
    ap.add_argument("--att-measure-extra", default="",
                    help="round 7: comma list of extra attention lengths to MEASURE (<= cap)")
    ap.add_argument("--att-dump", default="",
                    help="round 7: npz path for per-row attention records (analysis only)")
    # ---- calib10_r7 additions; every default reproduces calib9 (hence calib7) exactly  # r7
    ap.add_argument("--att-prefix-split", type=int, default=0, help="r7: measure sample ranks < P as their own per-head batch (0 = off)")  # r7
    ap.add_argument("--r7-hold-dump", default="", help="r7: npz of hold-phase linear pairs + protected records")  # r7
    ap.add_argument("--prot-measure", default="", help="r7: comma list of protected-slice lengths to measure (dump only)")  # r7
    args = ap.parse_args()
    C9_NEW_ARGS = ("target_scales", "solve_state", "att_measure_extra", "att_dump")
    C9_DEFAULTS = {"target_scales": "1.0", "solve_state": "", "att_measure_extra": "",
                   "att_dump": ""}
    scales = parse_scales(args.target_scales)
    c9_active = any(getattr(args, a) != C9_DEFAULTS[a] for a in C9_NEW_ARGS)
    # prc6_calib keeps calib7's record (vars(args) without the new flags); the new flags go
    # to prc9_r6, which is written only when one of them is non-default.
    calib7_view = {k: v for k, v in vars(args).items() if k not in C9_NEW_ARGS}
    c9_record = ({"calibrator": "mp_per_row_chunk_calib9_r6.py", "target_scales": scales,
                  **{a: getattr(args, a) for a in C9_NEW_ARGS}} if c9_active else None)
    calib7_view, c9_record, r7_record = r7_adjust_records(args, calib7_view, c9_record, scales, C9_NEW_ARGS)  # r7
    c9_active = c9_active or r7_record is not None  # r7: the r7 record rides in prc9_r6
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0

    parent = Path(args.parent)
    table = json.loads((parent / "table.json").read_text())
    wrapper = json.loads((parent / "wrapper.json").read_text())
    cap = 2 ** (SC_PREC - 1) if HALVE else 2 ** SC_PREC
    ladder = sorted(int(v) for v in args.ladder.split(","))
    assert max(ladder) == cap and min(ladder) >= 1
    att_extra = sorted({int(v) for v in args.att_measure_extra.split(",") if v.strip()})
    if any(not (1 <= L <= cap) for L in att_extra):
        raise SystemExit(f"[c9] --att-measure-extra {att_extra} outside [1, {cap}] "
                         "(stoc_len <= 2**sc_prec, halved)")
    prot_levels = r7_parse_lengths(args.prot_measure, cap)  # r7 ([] by default)
    att_prefix = r7_check_prefix(args.att_prefix_split, args.att_rows_per_call)  # r7 (0 by default)
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
    prot_store = {"calib": defaultdict(list), "hold": defaultdict(list)}  # r7 (stays empty by default)

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
        smooth_full = smooth  # r7: all-channel smooth scales, for the protected slice
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
            rowmax=np.repeat(mn.index_select(0, rows).amax(dim=1).cpu().numpy(), mn.shape[1]),
            parL=parL.index_select(0, rows)[:, None].expand(-1, mn.shape[1]).reshape(-1)
            .cpu().numpy())
        for t in WEIGHT_TAGS:
            if t in wt:
                rec[t] = (wt[t] * rep).reshape(-1, len(measured)).cpu().numpy()
        store[state["phase"]][(op, b)].append(rec)
        if prot_levels and pidx.numel() > 0 and weights:  # r7: protected slice, dump only
            prot_store[state["phase"]][(op, b)].append(r7_protected_record(  # r7
                x_full.index_select(0, rows), w_full, smooth_full, pidx, prot_levels, weights,  # r7
                rep, blk, rows, unit is not None))  # r7

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
        lv_meas = sorted(set(lad) | ({esc_len} if (t_esc is not None and esc_len) else set())
                         | set(att_extra))       # calib9: att_extra is empty by default
        assert 1 <= min(lv_meas) and max(lv_meas) <= cap, (op, blk, lv_meas)  # r7: code cap 128
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
        for h, sel in r7_head_selections(bh, att_prefix):  # r7: calib9's per-head loop when att_prefix == 0
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
        if args.att_dump:                        # calib9 round-7 fields; unused by the solve
            rec.update(blk=np.full(R, int(blk), dtype=np.int16),
                       bh=bh.cpu().numpy().astype(np.int32), pos=n.cpu().numpy().astype(np.int32),
                       rank=np.arange(R, dtype=np.int32), win=np.full(R, state["win"], np.int16))
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
            with GradPass(scm, SCLinear, capture, capture_att):
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
                    with GradPass(scm, SCLinear, capture, capture_att), torch.enable_grad():
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

    def th_from(best, bin_max):                # calib9: same code, now module-level
        return th_from_bins(best, bin_max, len(Lv))

    bisect = bisect_lambda                     # calib9: same code, now module-level
    c9_solve = {}                              # calib9: per scaled table, target + calib cost
    gbins_fis = None                           # calib9: the 'fis' bins, for --solve-state

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
        if cname == "fis":
            gbins_fis = bins
        # global: one lambda across all keys at the parent's total cost (x each target scale)
        target0 = sum(budget[k] * float(bins[k][1].sum()) for k in keys)
        for s_ in scales:
            target = scaled_target(target0, s_)
            rs = solve_global(bins, keys, Lv, target)
            tname_ = f"g{cname}{scale_suffix(s_)}"
            tables[tname_] = {k: th_from(rs[k], bins[k][2]) for k in keys}
            mv = defaultdict(lambda: [0.0, 0.0, 0.0])
            for k in keys:
                Cs = float(bins[k][1].sum())
                mv[k[0]][0] += Cs
                mv[k[0]][1] += float((bins[k][1] * Lv[rs[k]]).sum())
                mv[k[0]][2] += Cs * budget[k]
            lead = f"[c6] global-{cname}: " if s_ == 1.0 else f"[c9] global-{cname} x{s_:.4f}: "
            print(lead + "  ".join(f"{o} {p / m:.1f}->{g / m:.1f}"
                                   for o, (m, g, p) in sorted(mv.items())))
            ccost = sum(float((bins[k][1] * Lv[rs[k]]).sum()) for k in keys)
            c9_solve[tname_] = {"target_scale": float(s_), "target0": float(target0),
                                "target": float(target), "calib_cost": ccost,
                                "calib_cost_over_target": ccost / target}
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
        target0 = par_lin + par_att
        if gbins_fis is not None:              # calib9: the joint's linear bins ARE gfis's bins
            for k in keys:
                assert all(np.array_equal(a_, b_) for a_, b_ in zip(lbins[k], gbins_fis[k])), k
        att_th_by = {}                         # calib9: table name -> attention thresholds
        for s_ in scales:
            target = scaled_target(target0, s_)
            jname = f"gfisla{scale_suffix(s_)}"
            rj = solve_joint(lbins, abins, keys, akeys, Lv, fixed, target)
            tables[jname] = {k: th_from(rj[0][k], lbins[k][2]) for k in keys}
            att_th = {k: att_th_from(rj[1][k], abins[k][2], len(abins[k][3])) for k in akeys}
            att_th_by[jname] = att_th
            ccost = joint_cost(rj, lbins, abins, keys, akeys, Lv, fixed)
            c9_solve[jname] = {"target_scale": float(s_), "target0": float(target0),
                               "target": float(target), "calib_cost": ccost,
                               "calib_cost_over_target": ccost / target}
            if s_ == 1.0:
                joint["att_th"] = att_th
            tg = "[c7]" if s_ == 1.0 else f"[c9 x{s_:.4f}]"
            lin_L = sum(float((lbins[k][1] * Lv[rj[0][k]]).sum()) for k in keys) / sum(float(lbins[k][1].sum()) for k in keys)
            lin_par = par_lin / sum(float(lbins[k][1].sum()) for k in keys)
            att_mac = sum(float(cat(att_store["calib"][k], "mac").sum()) for k in akeys)
            att_L = (sum(float((abins[k][1] * abins[k][3][rj[1][k]]).sum()) for k in akeys) + fixed) / att_mac
            print(f"{tg} JOINT lambda: linear L {lin_par:.2f} -> {lin_L:.2f}; attention L "
                  f"{par_att / att_mac:.2f} -> {att_L:.2f} (total cost held at the parent's)")
            for k in akeys:
                A = abins[k][3]
                Ck = abins[k][1]
                print(f"{tg}   {k[0]} l{k[1]}: ladder {list(map(int, A))}  L -> "
                      f"{float((Ck * A[rj[1][k]]).sum() / max(Ck.sum(), 1e-9)):.1f}  th {[round(v, 4) for v in att_th[k]]}")
        att_th = att_th_by.get("gfisla", {})
        # held-out: Fisher-predicted dNLL incl. attention, for parent / gfis / gfisla
        # (calib9: + every scaled gfis_s*/gfisla_s* table, appended after calib7's three)
        n_tok = args.holdout_windows * 2047
        HJ = {}
        jnames = ["parent", "gfis", "gfisla"] if "gfisla" in att_th_by else ["parent"]
        jnames += [t for t in tables if SCALE_NAME_RE.match(t)
                   and base_table_name(t) in ("gfis", "gfisla")]
        for tname in jnames:
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
                    if base_table_name(tname) == "gfisla":
                        Lr = deploy_att(p_["mn"], att_th_by[tname][k], lad, t_esc, esc_len)
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
        if c9_active:
            diag["joint_att_th_by_table"] = {
                n_: {f"{k[0]}:l{k[1]}": v for k, v in th_.items()} for n_, th_ in att_th_by.items()}
    else:
        att_th_by, lbins, abins, fixed, par_lin, par_att = {}, None, {}, 0.0, None, None
    if c9_active:
        diag["prc9_r6"] = {"record": c9_record, "solves": c9_solve}

    # emit tables (calib9: calib7's name filter on the UNSCALED base name; body = emit_table)
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for tname in emitted_table_names(tables, currencies):
        extra = None
        if c9_active:
            extra = dict(c9_record, table=tname, solve=c9_solve.get(tname))
        emit_table(stem, tname, tables[tname],
                   att_th_by[tname] if base_table_name(tname) == "gfisla" else None,
                   table, wrapper, ladder, dict(calib7_view, table=tname), extra)
    if args.solve_state:
        if gbins_fis is None:
            raise SystemExit("[c9] --solve-state needs the 'fis' currency")
        meta = {"schema": "prc-calib9-r6-solve-state-v1", "ladder": [int(v) for v in ladder],
                "measured": [int(v) for v in measured],
                "keys": [[k[0], int(k[1])] for k in keys],
                "akeys": [[k[0], int(k[1])] for k in akeys] if lbins is not None else [],
                "budget": {f"{k[0]}|{k[1]}": budget[k] for k in keys},
                "target0_gfis": float(sum(budget[k] * float(gbins_fis[k][1].sum()) for k in keys)),
                "target0_gfisla": (float(par_lin + par_att) if lbins is not None else None),
                "par_lin": par_lin, "par_att": par_att, "fixed": float(fixed),
                "att_ladder": {f"{k[0]}|{k[1]}": [list(map(int, v[0])), v[1], v[2],
                                                  list(map(int, v[3])), v[4]]
                               for k, v in att_ladder.items()},
                "calib_record": calib7_view, "c9_record": c9_record, "solves": c9_solve,
                "parent_table_sha256": hashlib.sha256((parent / "table.json").read_bytes()).hexdigest(),
                "parent_wrapper_sha256": hashlib.sha256((parent / "wrapper.json").read_bytes()).hexdigest()}
        if r7_record is not None:  # r7: schema v2 = v1 + the capture facts CPU solves need
            meta.update(r7_state_meta(  # r7
                args=args, att_store=att_store, hold_store=store["hold"], measured=measured,  # r7
                ladder=ladder, starts=starts, hold_idx=hold_idx, prot_levels=prot_levels,  # r7
                total_blocks=total_blocks, n_lb=n_lb, n_units=n_units, top_k=top_k,  # r7
                currencies=currencies, cap=cap))  # r7
        save_solve_state(args.solve_state, meta, gbins_fis,
                         {k: abins[k] for k in akeys} if lbins is not None else {})
        print(f"[c9] solve state -> {args.solve_state}")
    if args.att_dump:
        out_ = {"measured": np.asarray(measured), "ladder": np.asarray(ladder)}
        for ph in ("calib", "hold"):
            for k, parts in att_store[ph].items():
                tag = f"att|{ph}|{k[0]}|{k[1]}"
                for f_ in ("mn", "mac", "parL", "blk", "bh", "pos", "rank", "win"):
                    out_[f"{tag}|{f_}"] = cat(parts, f_)
                for f_ in ("raw",) + WEIGHT_TAGS:
                    if all(f_ in p_ for p_ in parts):
                        out_[f"{tag}|{f_}"] = cat(parts, f_).astype(np.float32)
                out_[f"{tag}|lv"] = np.asarray(parts[0]["lv"])
        np.savez_compressed(args.att_dump, **out_)
        print(f"[c9] dumped per-row attention records -> {args.att_dump}")
    if args.r7_hold_dump:  # r7
        r7_write_hold_dump(args.r7_hold_dump, store["hold"], prot_store, measured, prot_levels)  # r7
        print(f"[c10] dumped hold-phase linear pairs + protected records -> {args.r7_hold_dump}")  # r7
    Path(args.diag).write_text(json.dumps(diag, indent=1, default=float))
    if args.dump:
        out_ = {"measured": np.asarray(measured), "ladder": np.asarray(ladder)}
        for ph in ("calib", "hold"):
            for k, parts in store[ph].items():
                tag = f"{ph}|{k[0]}|{k[1]}"
                for f_ in ("mn", "mac", "parL", "cidx", "rpos", "blk", "rowmax", "yE"):
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
