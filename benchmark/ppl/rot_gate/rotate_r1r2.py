#!/usr/bin/env python
"""Offline-only QuaRot-style rotation folding for HF Llama AND Qwen3
checkpoints (architecture auto-detected from config.model_type).

R1: residual-stream randomized Hadamard rotation Q1 = H_hidden . diag(s1) / sqrt(hidden),
    folded entirely into weights (no runtime op).
R2: per-KV-head v->o rotation Q2_g = H_head_dim . diag(s2_{layer,g}) / sqrt(head_dim)
    (one shared Hadamard, per-(layer, kv-head) sign vectors), folded into v_proj
    output rows and the matching o_proj input columns, GQA-aware.
NO online R3/R4 — every transformation here is a weight-space fold; the saved
checkpoint is a plain LlamaForCausalLM that computes (up to float error) the
same function as the original.

Algebra (HF convention: Linear weight stored [out, in]; F.linear computes
y = x @ W.T):

  Define the rotated residual stream  x' = x @ Q1  (Q1 orthogonal, Q1^-1 = Q1.T).

  * residual READERS (q/k/v_proj, gate/up_proj, lm_head) receive x' and must
    reproduce their original output:
        x' @ W'.T = x @ W.T  for all x
        =>  W' = W @ Q1
        check: (W @ Q1).T = Q1.T @ W.T, so x @ Q1 @ Q1.T @ W.T = x @ W.T.
  * residual WRITERS (o_proj, down_proj) must emit the rotated vector:
        y' = y @ Q1 = x_in @ W.T @ Q1 = x_in @ (Q1.T @ W).T
        =>  W' = Q1.T @ W
    (a writer bias would transform b' = b @ Q1; Llama has none — asserted).
  * embedding rows are writers into the residual:  E' = E @ Q1  (row e -> e @ Q1).
  * RMSNorm with UNIT weight commutes with Q1 (orthogonal rotation preserves
    the per-token mean square: mean((xQ1)^2) = ||x||^2/n = mean(x^2)), so all
    RMSNorm gammas must be folded into their reader Linears FIRST:
        input_layernorm          -> q_proj, k_proj, v_proj
        post_attention_layernorm -> gate_proj, up_proj
        model.norm               -> lm_head
    fold: W <- W * gamma[None, :]  (scale input columns), gamma <- 1.

  R2, per kv head g with head_dim d:
  * v_proj output rows for kv head g write the (rotated) V:
        V'_g = V_g @ Q2_g
        =>  W_v[g*d:(g+1)*d, :] <- Q2_g.T @ W_v[g*d:(g+1)*d, :]
  * attn_out_h = softmax(P_h) @ V'_{g(h)} = (softmax(P_h) @ V_{g(h)}) @ Q2_{g(h)},
    i.e. the o_proj input slice of QUERY head h arrives rotated by the Q2 of the
    KV head it consumes.  HF repeat_kv repeats each kv head n_rep = H/H_kv times
    CONSECUTIVELY, and attn_out is reshaped [B, S, H*d] with query head h
    occupying columns [h*d, (h+1)*d), so
        g(h) = h // n_rep
        W_o[:, h*d:(h+1)*d] <- W_o[:, h*d:(h+1)*d] @ Q2_{g(h)}
    Every query head that consumes kv head g gets the SAME Q2_g — this is the
    GQA-correctness condition.

  qk (Q.K^T) and RoPE are untouched: q_proj/k_proj OUTPUTS are not rotated
  (R1 acts only on their input side), so attention probabilities are
  bit-identical in exact arithmetic.  down_proj INPUT (the SiLU(gate)*up
  intermediate) is likewise mathematically unchanged — qk and down_proj are the
  no-rotation-reaches-them control ops for the rot-gate experiment.

Qwen3 specifics (model_type == "qwen3", e.g. Qwen3-4B-Instruct-2507):

  * Module layout matches Llama naming, so all folds above apply unchanged.
  * Per-head q_norm/k_norm RMSNorms sit on q_proj/k_proj OUTPUTS (before
    RoPE).  R1 changes only the projections' INPUT side, so their operands
    are bit-identical; R2 touches only the v->o path.  They are therefore
    function-preserved WITHOUT any fold and MUST NOT be modified.  A census
    (`_census_rmsnorms`) enumerates every RMSNorm in the model and refuses
    to fold if any norm outside the known layout appears, so the gamma pass
    can never sweep them in accidentally.
  * hidden_size 2560 is not a power of 2 (2^7 * 20): the Hadamard is built
    as kron(Sylvester H_128, Paley-I H_20) — exact +-1 entries, integer
    self-checked.  head_dim 128 stays Sylvester.
  * TIED EMBEDDINGS (tie_word_embeddings=true): R1 alone WOULD preserve the
    tie — the reader fold (lm_head W <- W @ Q1) and the writer fold
    (E <- E @ Q1) are the same right-multiplication, applied once to the
    shared tensor.  But the final-norm gamma fold is reader-side ONLY:
    lm_head needs W <- W . diag(gamma) on its input columns, while the
    embedding-writer role of the same tensor must NOT see gamma (row
    e <- e * gamma would inject a per-channel scale into the residual
    stream at the input).  No compensation elsewhere exists: a per-channel
    diag(gamma) does not commute with the per-token RMS denominator of any
    downstream norm, and diag(gamma) commutes with Q1 = H.diag(s) only for
    constant gamma.  Exact function preservation therefore forces an
    UNTIE: lm_head gets its own clone of E, gamma + R1 fold into the clone
    (reader), E keeps the writer fold, and the checkpoint is saved with
    tie_word_embeddings=false (verified post-save).  Cost: +vocab*hidden
    params (~778 MB bf16 for 4B); correctness is enforced by the tiny-model
    logit-equivalence test with a tied config.

CLI: loads a checkpoint fp32, folds in float64 (on --device), casts back to the
checkpoint's original dtype, save_pretrained + tokenizer + rotation_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import torch


REACHED_OPS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "o_proj", "av"]
NOT_REACHED_OPS = ["qk", "down_proj"]


def _sylvester(n: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """Unnormalized Sylvester Hadamard H_n (+-1 entries). n = power of 2."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Sylvester size must be a power of 2, got {n}")
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def _hadamard20(dtype=torch.float64, device=None) -> torch.Tensor:
    """Unnormalized Paley-I Hadamard of order 20 (q = 19 ≡ 3 mod 4):
    H = I + S with S = [[0, 1^T], [-1, Q]] the skew conference matrix and
    Q the Jacobsthal matrix Q_ij = chi(i-j) over GF(19) (chi = quadratic
    character).  S skew + S S^T = 19 I  =>  H H^T = 20 I.  Exact +-1
    entries; self-checked in INTEGER arithmetic before use."""
    q = 19
    residues = {(i * i) % q for i in range(1, q)}

    def chi(x: int) -> int:
        x %= q
        return 0 if x == 0 else (1 if x in residues else -1)

    S = torch.zeros(q + 1, q + 1, dtype=torch.int64)
    S[0, 1:] = 1
    S[1:, 0] = -1
    for i in range(q):
        for j in range(q):
            S[1 + i, 1 + j] = chi(i - j)
    H = torch.eye(q + 1, dtype=torch.int64) + S
    assert bool((H.abs() == 1).all()), "H20 has a non +-1 entry"
    assert torch.equal(H @ H.T, 20 * torch.eye(q + 1, dtype=torch.int64)), \
        "H20 failed the exact H @ H.T == 20 I check"
    return H.to(dtype=dtype, device=device)


def _hadamard_supported(n: int) -> bool:
    """True iff we can build an exact Hadamard of order n: 2^k, or 2^k * 20
    (k >= 0; covers Qwen3 hidden 2560 = 128 * 20 and all pow2 dims)."""
    if n <= 0:
        return False
    m = n
    while m % 2 == 0:
        m //= 2
    return m == 1 or (m == 5 and n % 20 == 0)


def hadamard_orthogonal(n: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """Orthogonal Hadamard H_n / sqrt(n).  n = 2^k -> Sylvester (bit-identical
    to the original llama-only implementation); n = 2^k * 20 ->
    kron(Sylvester H_{n/20}, Paley H_20)."""
    if n > 0 and (n & (n - 1)) == 0:
        H = _sylvester(n, dtype, device)
    elif _hadamard_supported(n):
        H = torch.kron(_sylvester(n // 20, dtype, device),
                       _hadamard20(dtype, device))
    else:
        m = n
        while m > 0 and m % 2 == 0:
            m //= 2
        raise ValueError(
            f"Hadamard size {n} unsupported (odd part {m}); supported: "
            f"2^k or 2^k * 20")
    return H / math.sqrt(n)


def rand_signs(n: int, gen: torch.Generator) -> torch.Tensor:
    """Deterministic ±1 vector (CPU Philox, machine-independent)."""
    return (torch.randint(0, 2, (n,), generator=gen, dtype=torch.int64) * 2 - 1)


def orthogonality_error(Q: torch.Tensor) -> float:
    n = Q.shape[0]
    eye = torch.eye(n, dtype=Q.dtype, device=Q.device)
    return (Q.T @ Q - eye).abs().max().item()


def _sha256_signs(s: torch.Tensor) -> str:
    return hashlib.sha256(s.to(torch.int8).cpu().numpy().tobytes()).hexdigest()


def _apply(weight: torch.Tensor, left=None, right=None, device=None,
           compute_dtype=torch.float64) -> None:
    """In-place  W <- left @ W @ right  (either side optional) in compute_dtype."""
    w = weight.data.to(device=device, dtype=compute_dtype)
    if left is not None:
        w = left @ w
    if right is not None:
        w = w @ right
    weight.data.copy_(w.to(dtype=weight.dtype, device=weight.device))


def _scale_cols(weight: torch.Tensor, gamma: torch.Tensor,
                compute_dtype=torch.float64) -> None:
    """In-place  W <- W * gamma[None, :]  (fold a preceding RMSNorm gamma)."""
    w = weight.data.to(dtype=compute_dtype)
    w = w * gamma.to(device=w.device, dtype=compute_dtype).unsqueeze(0)
    weight.data.copy_(w.to(dtype=weight.dtype, device=weight.device))


def _layers(model):
    return model.model.layers


def _assert_no_bias(linear, name: str) -> None:
    if getattr(linear, "bias", None) is not None:
        raise AssertionError(
            f"{name} has a bias — the fold below does not handle writer biases; "
            "refuse rather than silently corrupt.")


_FOLDED_NORM_SUFFIXES = ("input_layernorm", "post_attention_layernorm")
_UNTOUCHED_NORM_SUFFIXES = ("self_attn.q_norm", "self_attn.k_norm")


def _census_rmsnorms(model):
    """Enumerate EVERY RMSNorm in the model and refuse to fold unless each one
    is either (a) a norm we fold (input_layernorm / post_attention_layernorm /
    model.norm — each feeds exactly the reader Linears handled in
    fold_rmsnorms), or (b) a norm that is provably function-preserved
    untouched: Qwen3's per-head q_norm/k_norm act on q_proj/k_proj OUTPUTS,
    which R1 (input-side fold) leaves bit-identical, and R2 never touches
    q/k.  Anything else (unknown architecture variant) raises BEFORE any
    weight is modified.  Returns (folded_names, untouched_names)."""
    folded, untouched, unexpected = [], [], []
    for name, mod in model.named_modules():
        if "RMSNorm" not in type(mod).__name__:
            continue
        if name == "model.norm" or name.endswith(_FOLDED_NORM_SUFFIXES):
            folded.append(name)
        elif name.endswith(_UNTOUCHED_NORM_SUFFIXES):
            untouched.append(name)
        else:
            unexpected.append(name)
    if unexpected:
        raise AssertionError(
            f"unexpected RMSNorm modules {unexpected} — fold layout unknown; "
            f"refusing to rotate")
    n_layers = len(_layers(model))
    assert len(folded) == 2 * n_layers + 1, \
        f"expected {2 * n_layers + 1} foldable norms, found {len(folded)}"
    mt = model.config.model_type
    expect_untouched = 2 * n_layers if mt == "qwen3" else 0
    assert len(untouched) == expect_untouched, \
        (f"{mt}: expected {expect_untouched} untouched q/k norms, "
         f"found {len(untouched)}: {untouched}")
    return folded, untouched


def fold_rmsnorms(model, compute_dtype=torch.float64) -> None:
    """Fold every RMSNorm gamma into its reader Linear(s); set gamma to 1."""
    for layer in _layers(model):
        g_in = layer.input_layernorm.weight.data.clone()
        for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                    layer.self_attn.v_proj):
            _scale_cols(lin.weight, g_in, compute_dtype)
        layer.input_layernorm.weight.data.fill_(1.0)

        g_post = layer.post_attention_layernorm.weight.data.clone()
        for lin in (layer.mlp.gate_proj, layer.mlp.up_proj):
            _scale_cols(lin.weight, g_post, compute_dtype)
        layer.post_attention_layernorm.weight.data.fill_(1.0)

    g_final = model.model.norm.weight.data.clone()
    _scale_cols(model.lm_head.weight, g_final, compute_dtype)
    model.model.norm.weight.data.fill_(1.0)


def apply_r1(model, seed: int, device=None,
             compute_dtype=torch.float64, r1_matrix=None) -> dict:
    """Residual-stream rotation. Requires fold_rmsnorms() to have run first.

    ``r1_matrix``: optional path to a saved orthogonal Q1 from
    learn_concentration_rotation.py (ANTI-QuaRot: a rotation chosen to MAXIMIZE
    per-row energy concentration, because SC error is activity-shaped, instead
    of the incoherence a Hadamard provides). The fold below is identical either
    way -- only the source of Q1 changes -- so the rewrite stays exact and
    runtime-free."""
    hidden = model.config.hidden_size
    if r1_matrix is not None:
        Q1 = torch.load(r1_matrix, map_location="cpu").to(
            device=device, dtype=compute_dtype)
        if tuple(Q1.shape) != (hidden, hidden):
            raise ValueError(f"R1 matrix {tuple(Q1.shape)} != ({hidden},{hidden})")
        s1 = None
    else:
        gen = torch.Generator().manual_seed(seed * 1000003 + 1)
        s1 = rand_signs(hidden, gen)
        H = hadamard_orthogonal(hidden, compute_dtype, device)
        Q1 = H * s1.to(device=device, dtype=compute_dtype).unsqueeze(0)
    orth = orthogonality_error(Q1)
    Q1T = Q1.T.contiguous()

    # writers: embedding rows E <- E @ Q1
    _apply(model.model.embed_tokens.weight, right=Q1, device=device,
           compute_dtype=compute_dtype)
    for layer in _layers(model):
        att, mlp = layer.self_attn, layer.mlp
        # readers of the residual stream: W <- W @ Q1
        for lin in (att.q_proj, att.k_proj, att.v_proj,
                    mlp.gate_proj, mlp.up_proj):
            _apply(lin.weight, right=Q1, device=device,
                   compute_dtype=compute_dtype)
        # writers to the residual stream: W <- Q1.T @ W
        for lin in (att.o_proj, mlp.down_proj):
            _apply(lin.weight, left=Q1T, device=device,
                   compute_dtype=compute_dtype)
    # lm_head reads the (final-norm'd) rotated residual: W <- W @ Q1
    _apply(model.lm_head.weight, right=Q1, device=device,
           compute_dtype=compute_dtype)
    return {"r1_seed_stream": None if s1 is None else seed * 1000003 + 1,
            "r1_sign_sha256": None if s1 is None else _sha256_signs(s1),
            "r1_source": "learned_concentration" if s1 is None else "hadamard",
            "r1_orthogonality_error": orth,
            "r1_dim": hidden}


def apply_r2(model, seed: int, device=None,
             compute_dtype=torch.float64) -> dict:
    """Per-KV-head v->o rotation (GQA-aware). Order vs R1 is irrelevant:
    R1 touches v_proj's input side / o_proj's output side, R2 the opposite
    sides, so the two folds commute exactly."""
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    assert n_q % n_kv == 0, (n_q, n_kv)
    n_rep = n_q // n_kv

    H2 = hadamard_orthogonal(hd, compute_dtype, device)
    gen = torch.Generator().manual_seed(seed * 1000003 + 2)
    orth_max = 0.0
    hasher = hashlib.sha256()
    for layer in _layers(model):
        Wv = layer.self_attn.v_proj.weight
        Wo = layer.self_attn.o_proj.weight
        assert Wv.shape[0] == n_kv * hd, (Wv.shape, n_kv, hd)
        assert Wo.shape[1] == n_q * hd, (Wo.shape, n_q, hd)
        Q2s = []
        for g in range(n_kv):
            s = rand_signs(hd, gen)
            hasher.update(s.to(torch.int8).numpy().tobytes())
            Q2 = H2 * s.to(device=device, dtype=compute_dtype).unsqueeze(0)
            orth_max = max(orth_max, orthogonality_error(Q2))
            Q2s.append(Q2)
            # v output rows of kv head g are a writer: block <- Q2.T @ block
            blk = Wv.data[g * hd:(g + 1) * hd, :].to(device=device,
                                                     dtype=compute_dtype)
            Wv.data[g * hd:(g + 1) * hd, :].copy_(
                (Q2.T @ blk).to(dtype=Wv.dtype, device=Wv.device))
        for h in range(n_q):
            g = h // n_rep          # repeat_kv: consecutive blocks of n_rep
            # o input cols of query head h are a reader: block <- block @ Q2_{g(h)}
            blk = Wo.data[:, h * hd:(h + 1) * hd].to(device=device,
                                                     dtype=compute_dtype)
            Wo.data[:, h * hd:(h + 1) * hd].copy_(
                (blk @ Q2s[g]).to(dtype=Wo.dtype, device=Wo.device))
    return {"r2_seed_stream": seed * 1000003 + 2,
            "r2_signs_sha256": hasher.hexdigest(),
            "r2_orthogonality_error": orth_max,
            "r2_dim": hd,
            "r2_signs_layout": "per (layer, kv_head), drawn layer-major",
            "num_heads": n_q, "num_key_value_heads": n_kv, "n_rep": n_rep}


def rotate_model(model, seed: int, device=None, r1_matrix=None,
                 compute_dtype=torch.float64, r2_only=False) -> dict:
    """Fold norms + R1 + R2 in place. Returns the rotation manifest dict.

    The model may be any HF LlamaForCausalLM (tiny test models included);
    parameter dtypes are preserved (each fold computes in compute_dtype and
    writes back in the parameter's own dtype).
    """
    cfg = model.config
    assert cfg.model_type in ("llama", "qwen3"), \
        f"llama/qwen3 only, got {cfg.model_type}"
    src_tied = bool(getattr(cfg, "tie_word_embeddings", False)) or (
        model.lm_head.weight.data_ptr() ==
        model.model.embed_tokens.weight.data_ptr())
    if src_tied and not r2_only:
        # UNTIE (see module docstring for the derivation): the final-norm
        # gamma fold is reader-side only, so a shared E/lm_head tensor cannot
        # carry it function-preservingly (the embedding-writer role must not
        # see gamma, and diag(gamma) commutes with neither Q1 nor any
        # downstream per-token RMS).  Clone E into a standalone lm_head; the
        # clone then takes the ordinary reader folds (gamma, then W <- W@Q1)
        # while E takes only the writer fold (E <- E@Q1).
        model.lm_head.weight = torch.nn.Parameter(
            model.model.embed_tokens.weight.data.clone())
        model.config.tie_word_embeddings = False
    if not r2_only:
        assert model.lm_head.weight.data_ptr() != \
            model.model.embed_tokens.weight.data_ptr(), "embeddings still tied"
    hidden = cfg.hidden_size
    hd = getattr(cfg, "head_dim", None) or hidden // cfg.num_attention_heads
    assert _hadamard_supported(hidden), \
        f"hidden {hidden} has no supported Hadamard (need 2^k or 2^k*20)"
    assert _hadamard_supported(hd), \
        f"head_dim {hd} has no supported Hadamard (need 2^k or 2^k*20)"
    folded_norms, untouched_norms = _census_rmsnorms(model)
    _assert_no_bias(model.lm_head, "lm_head")
    for i, layer in enumerate(_layers(model)):
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _assert_no_bias(getattr(layer.self_attn, nm), f"layers.{i}.{nm}")
        for nm in ("gate_proj", "up_proj", "down_proj"):
            _assert_no_bias(getattr(layer.mlp, nm), f"layers.{i}.{nm}")

    # R2-ONLY: R2 rotates only v_proj OUTPUT rows and the matching o_proj
    # INPUT columns -- it never touches the residual stream, so it is
    # function-preserving WITHOUT the RMSNorm gamma fold and WITHOUT the
    # lm_head untie (both exist solely so R1 commutes with the norms).
    # Skipping them keeps the isolation clean: the gamma fold itself shifts
    # the activation/weight distributions that SC codes, which would be a
    # confound when attributing the effect to R2.
    if r2_only:
        m1 = {"r1_source": "SKIPPED (r2_only)", "r1_orthogonality_error": None}
    else:
        fold_rmsnorms(model, compute_dtype)
        m1 = apply_r1(model, seed, device, compute_dtype, r1_matrix=r1_matrix)
    m2 = apply_r2(model, seed, device, compute_dtype)
    manifest = {
        "kind": "quarot_offline_r2_only" if r2_only else "quarot_offline_r1r2",
        "r2_only": r2_only,
        "model_type": cfg.model_type,
        "seed": seed,
        "hidden_size": hidden,
        "head_dim": hd,
        "num_layers": cfg.num_hidden_layers,
        "source_tie_word_embeddings": src_tied,
        "untied_lm_head": src_tied,
        "untie_note": ("lm_head = clone(E) with gamma+R1 reader folds; E "
                       "writer fold only; saved with tie_word_embeddings="
                       "false (gamma fold on a shared tensor is not "
                       "function-preserving)") if src_tied else None,
        "folded_norms": 0 if r2_only else len(folded_norms),
        "untouched_norms": untouched_norms,
        "compute_dtype": str(compute_dtype),
        "rotations": "R1 residual (H_hidden.diag(s)/sqrt(n)) + R2 per-kv-head "
                     "v->o (shared H_head_dim, per-(layer,kv-head) signs); "
                     "RMSNorm gammas folded to 1; NO online R3/R4",
        "reached_ops": ["v_proj", "o_proj", "av"] if r2_only else REACHED_OPS,
        "not_reached_ops": (["q_proj", "k_proj", "gate_proj", "up_proj",
                             "down_proj", "qk"] if r2_only
                            else NOT_REACHED_OPS),
    }
    manifest.update(m1)
    manifest.update(m2)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model", required=True,
                   help="HF id or local checkpoint dir (Llama family)")
    p.add_argument("--out", required=True, help="output dir for rotated model")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--r2-only", action="store_true",
                   help="apply ONLY R2 (per-head v->o); skip the gamma fold, untie and R1 -- isolates o_proj/v_proj/av")
    p.add_argument("--r1-matrix", default=None,
                   help="learned orthogonal Q1 (.pt); omit for Hadamard")
    p.add_argument("--device", default="cpu",
                   help="device for the fold matmuls (cpu | cuda)")
    p.add_argument("--save-dtype", default=None,
                   help="override save dtype (default: checkpoint torch_dtype)")
    args = p.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    src_cfg = AutoConfig.from_pretrained(args.model)
    orig_dtype = src_cfg.torch_dtype
    if isinstance(orig_dtype, str):
        orig_dtype = getattr(torch, orig_dtype)
    if args.save_dtype:
        save_dtype = getattr(torch, args.save_dtype)
    else:
        save_dtype = orig_dtype or torch.bfloat16
    print(f"[rotate] loading {args.model} fp32 (save dtype {save_dtype}) ...",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print(f"[rotate] loaded in {time.time()-t0:.1f}s", flush=True)

    dev = torch.device(args.device)
    t1 = time.time()
    with torch.no_grad():
        manifest = rotate_model(model, args.seed, device=dev,
                                r1_matrix=args.r1_matrix,
                                r2_only=args.r2_only,
                                compute_dtype=torch.float64)
    def _fmt(v):                      # r1 is None in --r2-only mode
        return "skipped" if v is None else f"{v:.3e}"
    print(f"[rotate] folded in {time.time()-t1:.1f}s  "
          f"r1_orth={_fmt(manifest['r1_orthogonality_error'])} "
          f"r2_orth={_fmt(manifest['r2_orthogonality_error'])}", flush=True)

    model.to(save_dtype)
    model.config.torch_dtype = save_dtype
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)

    if manifest.get("untied_lm_head"):
        # Tied-source case: the saved checkpoint MUST be untied (config flag
        # false + a real lm_head.weight tensor in the safetensors), otherwise
        # a reload would re-tie lm_head to E and silently drop the gamma+R1
        # reader fold.  Fail the run rather than save a corrupt checkpoint.
        with open(os.path.join(args.out, "config.json")) as f:
            saved_cfg = json.load(f)
        assert saved_cfg.get("tie_word_embeddings") is False, \
            f"saved config still ties embeddings: {saved_cfg.get('tie_word_embeddings')}"
        idx_path = os.path.join(args.out, "model.safetensors.index.json")
        if os.path.isfile(idx_path):
            with open(idx_path) as f:
                saved_keys = set(json.load(f)["weight_map"])
        else:
            from safetensors import safe_open
            with safe_open(os.path.join(args.out, "model.safetensors"),
                           framework="pt") as f:
                saved_keys = set(f.keys())
        assert "lm_head.weight" in saved_keys, \
            "untied save is missing lm_head.weight — reload would re-tie"
        print("[rotate] untie verified on save: tie_word_embeddings=false, "
              "lm_head.weight present in checkpoint", flush=True)

    manifest.update({
        "source_model": args.model,
        "save_dtype": str(save_dtype),
        "device": args.device,
        "wall_seconds": round(time.time() - t0, 1),
    })
    # manifest is written LAST -> its presence marks a complete rotated dir.
    with open(os.path.join(args.out, "rotation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[rotate] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
