#!/usr/bin/env python
"""Anti-QuaRot: learn a CONCENTRATION-MAXIMIZING residual rotation for SC.

QuaRot/SpinQuant choose R1 to maximize INCOHERENCE (spread outliers, shrink
max|x|) because INT error is range-shaped. Our measured verdict + bit-level
stream simulation (papers/ROT_FAILURE_ANALYSIS.md, papers/rot_sc_sim.py) says SC
error is ACTIVITY-shaped instead: a row dominated by a few large products is
computed almost exactly (large p encodes accurately, near-zero channels emit
almost no stream activity), while a Gaussianized row spreads the output over
thousands of medium-p streams whose joint-discrepancy noise accumulates in the
popcount. Rotation therefore HURT (+0.62% on llama8B; +17..+39% matvec sigma in
sim at our smoothing level).

So the SC-native transplant is that objective NEGATED IN SPIRIT: the orthogonal
Q that MAXIMIZES per-row energy concentration of the residual stream.

Two methods, and the distinction is empirically real (measured on 4B):

  kurtosis (default) -- maximize  E_x[ ||Qx||_4^4 / ||x||_2^4 ]  by Cayley SGD.
      The ratio is computed PER ROW then averaged, so this targets per-row
      peakiness, which is what the SC stream argument actually asks for.
      Measured on 4B: 2.07x concentration vs identity.
  pca -- SliceGPT-style closed-form covariance eigenbasis. Concentrates the
      AGGREGATE (across-channel) variance, not per-row peakiness.
      Measured on 4B: only 1.26x, despite 89% of variance in the top 10% of
      components -- i.e. shared low-rank structure does NOT translate into
      per-row concentration. Kept for the comparison.

Both keep Q exactly orthogonal, so the offline weight fold in rotate_r1r2.py
stays exact and runtime-free (no online Hadamard, unlike QuaRot R3/R4).

NUMERICS (bug fixed 2026-07-19): the in-loop polar step runs in float32 and
leaves ~8e-4 orthogonality drift on a 2560x2560 matrix, which the fold's
1e-6 gate correctly rejects. The FINAL polar decomposition is therefore done in
float64, and the reported concentration is recomputed after that projection.

Writes Q1 (hidden x hidden, float64) for rotate_r1r2.py --r1-matrix.
"""
import argparse
import json
import sys
from pathlib import Path

import torch


def collect_residual_activations(model_path, n_samples, ctx, device, dtype):
    """Collect residual-stream vectors (the tensors R1 actually rotates)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device)
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    feats, hooks = [], []

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.detach().reshape(-1, h.shape[-1]).float()
        idx = torch.randperm(h.shape[0], device=h.device)[:256]
        feats.append(h[idx].cpu())

    for lyr in model.model.layers:          # every layer's input residual
        hooks.append(lyr.register_forward_hook(hook))
    with torch.no_grad():
        for i in range(n_samples):
            s = i * ctx
            if s + ctx > enc.numel():
                break
            model(enc[s:s + ctx].unsqueeze(0).to(device))
    for h in hooks:
        h.remove()
    X = torch.cat(feats, 0)
    del model
    torch.cuda.empty_cache()
    return X


def concentration(Q, X):
    """E[ ||Qx||_4^4 / ||x||_2^4 ] — higher = more concentrated (SC-friendly).

    Per-row ratio, then averaged: this is the per-row peakiness SC rewards, as
    opposed to PCA's aggregate-variance concentration."""
    Y = X @ Q.T
    num = (Y ** 4).sum(dim=1)
    den = (X ** 2).sum(dim=1) ** 2 + 1e-12      # orthogonal => ||Qx||=||x||
    return (num / den).mean()


def pca_rotation(X):
    """SliceGPT-style PCA rotation: eigenbasis of the residual covariance.

    Same computational invariance QuaRot uses, opposite geometry (concentrate
    rather than spread). We keep the FULL orthogonal Q and take only the
    rotation — SliceGPT's dimension deletion is a different compute axis and
    would confound the iso-compute MP comparison."""
    Xd = X.double()
    cov = (Xd.T @ Xd) / Xd.shape[0]
    cov = 0.5 * (cov + cov.T)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    return evecs[:, order].T.contiguous(), evals[order]


def cayley_step(Q, G, lr):
    """Cayley-transform update: stays on the orthogonal manifold."""
    A = G @ Q.T - Q @ G.T
    A = 0.5 * (A - A.T)
    I = torch.eye(Q.shape[0], dtype=Q.dtype, device=Q.device)
    return torch.linalg.solve(I + (lr / 2.0) * A, (I - (lr / 2.0) * A) @ Q)


def polar_orthogonalize(Q):
    """Nearest orthogonal matrix (float64 polar decomposition) — the fix."""
    U, _, Vh = torch.linalg.svd(Q.double(), full_matrices=False)
    return U @ Vh


def _save(out, Q, meta):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(Q.cpu(), out)                    # float64
    json.dump(meta, open(str(out) + ".json", "w"), indent=1)
    print(f"[anti-quarot] wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--method", choices=["kurtosis", "pca"], default="kurtosis")
    ap.add_argument("--orth-tol", type=float, default=1e-6,
                    help="must satisfy the fold's own gate")
    args = ap.parse_args()
    dev = args.device

    print(f"[anti-quarot] method={args.method} model={args.model}", flush=True)
    X = collect_residual_activations(args.model, args.samples, args.ctx, dev,
                                     torch.bfloat16)
    print(f"[anti-quarot] activations {tuple(X.shape)}", flush=True)
    H = X.shape[1]
    X = X.to(dev, torch.float32)
    X = X[X.norm(dim=1) > 1e-6]

    base = concentration(torch.eye(H, device=dev), X).item()
    print(f"[anti-quarot] identity concentration = {base:.6e}", flush=True)

    if args.method == "pca":
        Qp, evals = pca_rotation(X)
        Qp = polar_orthogonalize(Qp)
        cp = concentration(Qp.float(), X).item()
        top = (evals[:H // 10].sum() / evals.sum()).item()
        orth = (Qp @ Qp.T - torch.eye(H, dtype=torch.float64,
                                      device=Qp.device)).abs().max().item()
        print(f"[anti-quarot] PCA concentration={cp:.6e} (x{cp/base:.4f}) "
              f"top10%%_eigen_energy={top:.4f} orth={orth:.3e}", flush=True)
        if orth > args.orth_tol:
            raise SystemExit(f"orthogonality {orth:.3e} > {args.orth_tol}")
        _save(args.out, Qp, {"model": args.model, "hidden": H, "method": "pca",
                             "identity_concentration": base,
                             "final_concentration": cp,
                             "gain_vs_identity": cp / base,
                             "top10pct_eigen_energy": top,
                             "orthogonality_err": orth})
        return 0

    # ---- kurtosis: Cayley ascent on per-row peakiness --------------------
    Q, best_Q, best_c, lr = (torch.eye(H, device=dev),) * 2 + (base, args.lr)
    for it in range(args.iters):
        Q = Q.detach().requires_grad_(True)
        c = concentration(Q, X)
        g, = torch.autograd.grad(c, Q)
        with torch.no_grad():
            Qn = cayley_step(Q.detach(), -g, lr)        # ASCEND concentration
            U, _, Vh = torch.linalg.svd(Qn, full_matrices=False)
            Qn = U @ Vh
            cn = concentration(Qn, X).item()
        if cn > best_c:
            best_Q, best_c, Q = Qn.clone(), cn, Qn
        else:
            lr *= 0.5
            Q = best_Q.clone()
            if lr < 1e-4:
                print(f"[anti-quarot] converged at iter {it}", flush=True)
                break
        if it % 20 == 0:
            print(f"[anti-quarot] iter {it:4d} conc={best_c:.6e} "
                  f"(x{best_c/base:.4f}) lr={lr:.4g}", flush=True)

    # FIX: project to the nearest orthogonal matrix in FLOAT64, then re-score.
    best_Q = polar_orthogonalize(best_Q)
    best_c = concentration(best_Q.float(), X).item()
    orth = (best_Q @ best_Q.T - torch.eye(H, dtype=torch.float64,
                                          device=best_Q.device)).abs().max().item()
    print(f"[anti-quarot] FINAL concentration={best_c:.6e} "
          f"(x{best_c/base:.4f} vs identity) orth={orth:.3e}", flush=True)
    if orth > args.orth_tol:
        raise SystemExit(f"orthogonality {orth:.3e} > {args.orth_tol}")
    _save(args.out, best_Q, {"model": args.model, "hidden": H,
                             "method": "kurtosis", "iters": args.iters,
                             "identity_concentration": base,
                             "final_concentration": best_c,
                             "gain_vs_identity": best_c / base,
                             "orthogonality_err": orth,
                             "objective": "max E[||Qx||_4^4 / ||x||_2^4]"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
