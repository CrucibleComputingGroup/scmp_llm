#!/usr/bin/env python3
"""Symmetric (bipolar) vs asymmetric (unipolar+zero-point) SC on REAL operands.

The open representation axis (SEARCH_SPACE_MAP A3): the deployed path is
sign-magnitude bipolar (127 magnitude levels, symmetric absmax per row-group).
The kernel ALREADY carries an asymmetric path — mode="unipolar" quantizes to
q_max = 2^prec - 1 = 255 levels with a per-row-group zero-point and applies the
zp correction on the host (kernels.py:1987-2021) — but nothing in the LLM
deployment uses it. Asymmetry pays only on SKEWED operands, so this probe
measures both modes at MATCHED actual cycles on operands captured from the
deployed model (AWQ front-end, protected channels removed, exactly what the
runtime quantizes):

  linears : down_proj (post-SiLU-gated input, skewed — the case FOR asym),
            up_proj / o_proj (post-LN / post-attention — controls),
            per 128-chunk for BOTH modes so chunking is not a confound
            (a chunked-unipolar deployment would sum per-chunk partials with
            per-chunk zp corrections, which is exactly what this computes).
  attention: qk (Q·K^T, ~symmetric — control) and av (softmax·V — probs are
            one-sided, the strongest a-priori asym case and the highest-sigma
            operator on every model), sampled per (batch,head) slice as 2D.

Cycle accounting: bipolar runs with halve_bipolar_stoc_len=True, so
stoc_len=L is L actual cycles; unipolar has no halving and stoc_len=L is also
L actual cycles. Same L = same compute.

References at matched LEVEL count: analytic INT sym/asym per 128-group
(no SC sampling noise), bounding how much of any unipolar win is zero-point
vs stream statistics.

Run:  python benchmark/ppl/kbands/probe_asym_real.py --model-key llama8B \
        --target 96   [--rows 256] [--windows 1]
Needs GPU. Prints one table; also dumps JSON next to the log.
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

MP_BEST = Path("/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs")
MODEL_PATHS = {
    "4B": "Qwen/Qwen3-4B-Instruct-2507",
    "llama8B": "meta-llama/Llama-3.1-8B-Instruct",
    "14B": "Qwen/Qwen3-14B",
    "30B": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}
CHUNK_D = 128
SC_PREC = 8
LENS = (16, 24, 32, 48, 64, 96, 128)  # actual cycles, both modes
LINEAR_OPS = ("down_proj", "up_proj", "o_proj")


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def int_quant(x, bits, sym, group=CHUNK_D):
    import torch
    N, D = x.shape
    keep = (D // group) * group
    g = x[:, :keep].reshape(N, -1, group)
    if sym:
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / (2 ** (bits - 1) - 1)
        q = torch.clamp(torch.round(g / s), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
        out = q * s
    else:
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
        z = torch.round(-lo / s)
        q = torch.clamp(torch.round(g / s) + z, 0, 2 ** bits - 1)
        out = (q - z) * s
    res = x.clone()
    res[:, :keep] = out.reshape(N, keep)
    return res


def sc_err_chunked(x, w, L, mode, sc_matmul):
    """Sum per-128-chunk sc_matmul partials (same chunking both modes)."""
    import torch
    D = x.shape[1]
    nch = D // CHUNK_D
    fp = x.float() @ w.float().t()
    out = torch.zeros_like(fp)
    for c in range(nch):
        sl = slice(c * CHUNK_D, (c + 1) * CHUNK_D)
        xc, wc = x[:, sl].contiguous(), w[:, sl].contiguous()
        if mode == "bipolar":
            out += sc_matmul(xc, wc, granularity="per_row", mode="bipolar",
                             sc_prec=SC_PREC, stoc_len=int(L),
                             halve_bipolar_stoc_len=True)
        else:
            out += sc_matmul(xc, wc, granularity="per_row", mode="unipolar",
                             sc_prec=SC_PREC, stoc_len=int(L))
    return rel(out, fp)


def sc_err_flat(a, b, L, mode, sc_matmul):
    """Unchunked 2D call (attention slices; D = head_dim or seq)."""
    fp = a.float() @ b.float().t()
    if mode == "bipolar":
        o = sc_matmul(a.contiguous(), b.contiguous(), granularity="per_row",
                      mode="bipolar", sc_prec=SC_PREC, stoc_len=int(L),
                      halve_bipolar_stoc_len=True)
    else:
        o = sc_matmul(a.contiguous(), b.contiguous(), granularity="per_row",
                      mode="unipolar", sc_prec=SC_PREC, stoc_len=int(L))
    return rel(o, fp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-key", required=True, choices=list(MODEL_PATHS))
    ap.add_argument("--target", type=int, default=96)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4, help="attention (b,h) slices")
    ap.add_argument("--frontend", default="awq")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA"); return 0
    torch.manual_seed(0)

    parent = MP_BEST / args.model_key / f"target{args.target}"
    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = str(parent / "wrapper.json")
    if (parent / "hybrid_config.json").is_file():
        os.environ["SC_HYBRID_CONFIG_JSON"] = str(parent / "hybrid_config.json")
    os.environ.setdefault(
        "ACT_SCALES_DIR",
        "/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales")
    os.environ["FRONTEND"] = args.frontend

    from benchmark.quant.eval_quant import build_sc_model
    from benchmark.ppl.calibrate_mp_thresholds import _iter_calib_windows
    from model.sc_common import SCLinear
    import model.sc_common as sc_common
    from scmp_kernels.sc.matmul import sc_matmul
    from datasets import load_dataset

    model, tok = build_sc_model(MODEL_PATHS[args.model_key], "mp",
                                mp_table=str(parent / "wrapper.json"))
    model.eval()
    dev = next(model.parameters()).device
    total_blocks = getattr(model.config, "num_hidden_layers", None)
    mp_cfg = getattr(model.config, "sc_mp_config", None)

    # ---- capture: linears (residual input, protected channels removed) ----
    lin: dict[tuple, tuple] = {}

    def hook(mod, inp, out):
        op = getattr(mod, "_sc_op_name", None)
        blk = getattr(mod, "_sc_block_idx", None)
        if op not in LINEAR_OPS or blk is None:
            return
        b = 0 if blk < total_blocks // 2 else 1  # early / late halves
        key = (op, b)
        if key in lin:
            return
        x = inp[0].reshape(-1, inp[0].shape[-1]).float()[:args.rows]
        w = mod.weight.float()
        try:
            prot = mp_cfg.get_protected_channels(
                operator=op, block_idx=blk,
                unit_idx=getattr(mod, "_sc_unit_idx", None)) if mp_cfg else None
        except Exception:  # noqa: BLE001
            prot = None
        if prot:
            keepm = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
            pi = torch.as_tensor(list(prot), dtype=torch.long, device=x.device)
            keepm[pi[(pi >= 0) & (pi < x.shape[1])]] = False
            r = keepm.nonzero(as_tuple=True)[0]
            x, w = x.index_select(1, r), w.index_select(1, r)
        keep = (x.shape[1] // CHUNK_D) * CHUNK_D
        if keep >= CHUNK_D and x.shape[0] > 0:
            lin[key] = (x[:, :keep].contiguous(), w[:, :keep].contiguous(), blk)

    # ---- capture: attention operand pairs via the sc attention entry ----
    attn: dict[tuple, tuple] = {}
    orig_ab_t = sc_common._sc_attention_matmul_ab_t

    def spy_ab_t(a, b, **kw):
        tag = kw.get("operator") or ("qk" if a.shape[-1] <= 256 else "av")
        blk = kw.get("block_idx")
        bkt = 0 if (blk or 0) < (total_blocks // 2) else 1
        key = (tag, bkt)
        if key not in attn and tag in ("qk", "av"):
            a3 = a.detach().float()
            b3 = b.detach().float()
            if a3.dim() == 4:  # (B,H,N,D) -> (B*H,N,D)
                a3 = a3.reshape(-1, *a3.shape[-2:])
            if b3.dim() == 4:
                b3 = b3.reshape(-1, *b3.shape[-2:])
            if a3.dim() == 3:
                attn[key] = (a3.cpu(), b3.cpu(), blk)
        return orig_ab_t(a, b, **kw)

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, SCLinear)]
    sc_common._sc_attention_matmul_ab_t = spy_ab_t
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        with torch.no_grad():
            for wnd in _iter_calib_windows(enc, 2048, 4):
                model(wnd.unsqueeze(0).to(dev))
                if len(lin) >= 2 * len(LINEAR_OPS) and len(attn) >= 4:
                    break
    finally:
        for h in hs:
            h.remove()
        sc_common._sc_attention_matmul_ab_t = orig_ab_t

    print(f"captured linears: {sorted(lin)}  attention: {sorted(attn)}")

    # ---- measure ----
    results = defaultdict(dict)

    def run(name, fn_b, fn_u):
        for L in LENS:
            try:
                eb = fn_b(L)
            except Exception as e:  # noqa: BLE001
                eb = None; print(f"  {name} L={L} bipolar FAILED: {e}")
            try:
                eu = fn_u(L)
            except Exception as e:  # noqa: BLE001
                eu = None; print(f"  {name} L={L} unipolar FAILED: {e}")
            results[name][L] = (eb, eu)
            if eb and eu:
                print(f"  {name:22} L={L:3}  bipolar {eb:.4e}  unipolar {eu:.4e}"
                      f"  asym gain {100 * (eb - eu) / eb:+6.1f}%")

    for (op, bkt), (x, w, blk) in sorted(lin.items()):
        name = f"{op}:h{bkt}(b{blk})"
        run(name,
            lambda L, x=x, w=w: sc_err_chunked(x, w, L, "bipolar", sc_matmul),
            lambda L, x=x, w=w: sc_err_chunked(x, w, L, "unipolar", sc_matmul))
        # analytic INT reference at 5 bits (~t32-matched) and 7 bits
        fp = x.float() @ w.float().t()
        for bits in (5, 7):
            es = rel(int_quant(x, bits, True) @ int_quant(w, bits, True).t(), fp)
            ea = rel(int_quant(x, bits, False) @ int_quant(w, bits, False).t(), fp)
            print(f"  {name:22} INT{bits}  sym {es:.4e}  asym {ea:.4e}"
                  f"  zp gain {100 * (es - ea) / es:+6.1f}%")
            results[name][f"int{bits}"] = (es, ea)

    for (tag, bkt), (a3, b3, blk) in sorted(attn.items()):
        a3, b3 = a3.to(dev), b3.to(dev)
        nsl = min(args.heads, a3.shape[0])
        for s in range(nsl):
            a2, b2 = a3[s], b3[s]
            if a2.shape[0] > args.rows:
                a2 = a2[:args.rows]
            if b2.shape[0] > args.rows:
                b2 = b2[:args.rows]
            name = f"{tag}:h{bkt}(b{blk}) s{s}"
            run(name,
                lambda L, a2=a2, b2=b2: sc_err_flat(a2, b2, L, "bipolar", sc_matmul),
                lambda L, a2=a2, b2=b2: sc_err_flat(a2, b2, L, "unipolar", sc_matmul))

    out = Path(os.environ.get(
        "ASYM_PROBE_OUT",
        f"/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/diag/"
        f"asym_probe_{args.model_key}_t{args.target}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: {str(kk): vv for kk, vv in v.items()}
                               for k, v in results.items()}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
