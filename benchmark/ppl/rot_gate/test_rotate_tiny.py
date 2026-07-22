#!/usr/bin/env python
"""CPU unit test for rotate_r1r2.py — THE FOLD IS ONLY CORRECT IF THIS PASSES.

Three tiny random fp32 models, each rotated with rotate_model and gated on:

  * logits(rotated) == logits(original) to < 1e-4 relative on random batches
  * R1/R2 orthogonality error < 1e-6
  * every FOLDED RMSNorm weight is exactly all-ones after the fold
    (input_layernorm / post_attention_layernorm / model.norm); Qwen3's
    per-head q_norm/k_norm must be exactly UNCHANGED (they act on q/k_proj
    outputs, which the fold never touches — folding them would be a bug)
  * 2 rotation seeds (seed robustness)

Cases:
  llama    — hidden 64, 2 layers, 4 heads / 2 kv heads (GQA n_rep=2),
             head_dim 16, untied, vocab 128 (the original verified case)
  qwen3    — same dims, QK-Norm active, tie_word_embeddings=TRUE.  Extra
             gates: rotate_model must untie (config flag flips false, E and
             lm_head become distinct storage), and a save_pretrained ->
             from_pretrained round trip must stay untied with matching logits.
  qwen3p20 — hidden 80 = 2^4*5: exercises the Paley H_20 Kronecker Hadamard
             AND the decoupled hidden != heads*head_dim geometry — both are
             what the real Qwen3-4B (hidden 2560 = 2^7*20, 32*128 = 4096
             != 2560) hits, which the pow2-coupled cases never reach.

Plus a standalone orthogonality check of hadamard_orthogonal(2560), the exact
residual dimension of Qwen3-4B-Instruct-2507.

Run on the login node (CPU, < 5 min):
  source ~/.bashrc && conda activate annstention
  python benchmark/ppl/rot_gate/test_rotate_tiny.py
"""
import copy
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotate_r1r2 import hadamard_orthogonal, orthogonality_error, \
    rotate_model  # noqa: E402

from transformers import AutoModelForCausalLM  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaRMSNorm  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm  # noqa: E402


def _randomize_norms(model, rms_cls, init_seed: int) -> None:
    # Randomize every RMSNorm gamma away from the 1.0 init: with gamma == 1
    # the gamma-fold would be a no-op and the test would not exercise it (for
    # Qwen3 this includes q_norm/k_norm, so the function-preservation claim
    # is tested with NON-unit per-head norms too).
    gen = torch.Generator().manual_seed(init_seed + 1)
    for mod in model.modules():
        if isinstance(mod, rms_cls):
            mod.weight.data.uniform_(0.5, 1.5, generator=gen)


def build_tiny_llama(init_seed: int = 1234):
    torch.manual_seed(init_seed)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,            # 2^6 — exact Hadamard
        intermediate_size=176,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,     # < num_heads -> exercises GQA repeat_kv
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    model = LlamaForCausalLM(cfg).float().eval()
    _randomize_norms(model, LlamaRMSNorm, init_seed)
    return model


def build_tiny_qwen3(init_seed: int = 1234, hidden: int = 64,
                     intermediate: int = 176):
    torch.manual_seed(init_seed)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,     # GQA n_rep=2
        head_dim=16,               # explicit — Qwen3 decouples from hidden
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,  # the Qwen3-4B-Instruct-2507 case
        attention_bias=False,
        use_sliding_window=False,  # parity with the real 4B config
        sliding_window=None,
    )
    model = Qwen3ForCausalLM(cfg).float().eval()
    assert model.lm_head.weight.data_ptr() == \
        model.model.embed_tokens.weight.data_ptr(), "tiny qwen3 not tied?"
    _randomize_norms(model, Qwen3RMSNorm, init_seed)
    return model


CASES = {
    "llama": (build_tiny_llama, LlamaRMSNorm),
    "qwen3": (build_tiny_qwen3, Qwen3RMSNorm),
    "qwen3p20": (lambda: build_tiny_qwen3(hidden=80, intermediate=96),
                 Qwen3RMSNorm),
}


def _check_norms(orig, rot, rms_cls):
    """Folded norms all-ones; q_norm/k_norm bit-identical to the original."""
    orig_norms = {n: m.weight.data for n, m in orig.named_modules()
                  if isinstance(m, rms_cls)}
    n_folded = n_untouched = 0
    for name, mod in rot.named_modules():
        if not isinstance(mod, rms_cls):
            continue
        if name.endswith(("q_norm", "k_norm")):
            n_untouched += 1
            assert torch.equal(mod.weight.data, orig_norms[name]), \
                f"q/k norm {name} was MODIFIED by the fold (must be untouched)"
        else:
            n_folded += 1
            assert torch.equal(mod.weight.data,
                               torch.ones_like(mod.weight.data)), \
                f"RMSNorm {name} not all-ones after fold"
    L = orig.config.num_hidden_layers
    assert n_folded == 2 * L + 1, n_folded
    expect_untouched = 2 * L if orig.config.model_type == "qwen3" else 0
    assert n_untouched == expect_untouched, (n_untouched, expect_untouched)


def _check_tied_roundtrip(orig, rot, seed):
    """Tied source: rotate_model must have untied; save->reload stays untied
    and reproduces the rotated (== original) logits."""
    assert rot.config.tie_word_embeddings is False, \
        "rotate_model left config.tie_word_embeddings truthy on a tied model"
    assert rot.lm_head.weight.data_ptr() != \
        rot.model.embed_tokens.weight.data_ptr(), "still tied after rotate"
    with tempfile.TemporaryDirectory() as td:
        rot.config.torch_dtype = torch.float32
        rot.save_pretrained(td, safe_serialization=True)
        re_m = AutoModelForCausalLM.from_pretrained(
            td, torch_dtype=torch.float32).eval()
        assert not re_m.config.tie_word_embeddings, "reload re-tied (config)"
        assert re_m.lm_head.weight.data_ptr() != \
            re_m.model.embed_tokens.weight.data_ptr(), "reload re-tied (ptr)"
        gen = torch.Generator().manual_seed(4242 + seed)
        ids = torch.randint(0, orig.config.vocab_size, (2, 24), generator=gen)
        with torch.no_grad():
            lo = orig(input_ids=ids).logits
            lr = re_m(input_ids=ids).logits
        rel = ((lr - lo).norm() / lo.norm()).item()
        assert rel < 1e-4, f"save/reload logits drifted: rel {rel:.3e}"
    return rel


def run_one(case: str, seed: int):
    build, rms_cls = CASES[case]
    orig = build()
    cfg = orig.config
    hd = getattr(cfg, "head_dim", None) or \
        cfg.hidden_size // cfg.num_attention_heads
    assert hd == 16, hd
    assert cfg.num_key_value_heads < cfg.num_attention_heads
    src_tied = bool(cfg.tie_word_embeddings)

    rot = copy.deepcopy(orig)
    with torch.no_grad():
        manifest = rotate_model(rot, seed=seed, device="cpu",
                                compute_dtype=torch.float64)

    r1o = manifest["r1_orthogonality_error"]
    r2o = manifest["r2_orthogonality_error"]
    assert r1o < 1e-6, f"R1 orthogonality {r1o:.3e} >= 1e-6"
    assert r2o < 1e-6, f"R2 orthogonality {r2o:.3e} >= 1e-6"
    assert manifest["n_rep"] == 2, manifest["n_rep"]
    assert manifest["head_dim"] == 16, manifest["head_dim"]
    assert manifest["model_type"] == cfg.model_type
    assert manifest["untied_lm_head"] == src_tied

    _check_norms(orig, rot, rms_cls)

    gen = torch.Generator().manual_seed(999 + seed)
    rels = []
    with torch.no_grad():
        for _ in range(3):
            ids = torch.randint(0, orig.config.vocab_size, (4, 32),
                                generator=gen)
            lo = orig(input_ids=ids).logits
            lr = rot(input_ids=ids).logits
            rel = ((lr - lo).norm() / lo.norm()).item()
            rels.append(rel)
            assert rel < 1e-4, \
                f"{case} seed {seed}: relative logit error {rel:.3e}"

    reload_rel = _check_tied_roundtrip(orig, rot, seed) if src_tied else None
    return r1o, r2o, rels, reload_rel


def main() -> None:
    t0 = time.time()
    # Pre-verify the exact Qwen3-4B residual dimension (2560 = 2^7 * 20) on
    # CPU so the Paley-Kronecker path cannot first fail inside a GPU job.
    q4b = hadamard_orthogonal(2560, torch.float64)
    e4b = orthogonality_error(q4b)
    assert e4b < 1e-6, f"H_2560 orthogonality {e4b:.3e} >= 1e-6"
    print(f"hadamard_orthogonal(2560) [kron(H_128, PaleyH_20)]: "
          f"orth err {e4b:.3e} (gate < 1e-6)")

    for case in CASES:
        for seed in (0, 1):
            r1o, r2o, rels, reload_rel = run_one(case, seed)
            extra = ("  tied-save/reload rel %.3e" % reload_rel
                     if reload_rel is not None else "")
            print(f"{case:8s} seed {seed}: R1 orth {r1o:.3e}  "
                  f"R2 orth {r2o:.3e}  "
                  f"rel logit err {['%.3e' % r for r in rels]}{extra}  "
                  f"(gates: orth < 1e-6, rel < 1e-4)")
    print(f"PASS test_rotate_tiny ({time.time()-t0:.1f}s): gamma-fold + R1 + "
          f"R2(GQA n_rep=2) function-preserving on fp32 tiny Llama AND tiny "
          f"Qwen3 (QK-Norm untouched, tied-embeddings untied+verified, "
          f"Paley H_20 hidden 80), 2 seeds each")


if __name__ == "__main__":
    main()
