"""Validation for benchmark/quant/bitmod_dtypes.py + the bitmod/fp tag wiring.

Primary check: BIT-IDENTICAL agreement with the reference implementation in
BitMoD-HPCA-25/bitmod_quant/quant_utils/quant_weight.py (quant_datatype /
search_datatype) across datatypes, group sizes, and weight distributions.
The reference clone is only present on our machines — those tests skip cleanly
elsewhere. Runs on CPU; no GPU / HF download needed.

    python benchmark/quant/test_bitmod_dtypes.py      # or pytest -q <this file>
"""
from __future__ import annotations

import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from benchmark.quant.bitmod_dtypes import quant_weight_dtype  # noqa: E402

_BITMOD_REF = os.environ.get(
    "BITMOD_REF", "/home/allenjin/Projects/BitMoD-HPCA-25/bitmod_quant")


def _load_ref():
    if not os.path.isdir(_BITMOD_REF):
        return None
    if _BITMOD_REF not in sys.path:
        sys.path.insert(0, _BITMOD_REF)
    from quant_utils import quant_weight as ref  # type: ignore
    return ref


def _weights(k=16, c=512, seed=0, scale=0.05, outliers=True):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(k, c, generator=g) * scale
    if outliers:  # a few big channels, like real LLM weights post-fold
        w[:, ::97] *= 25.0
    return w.to(torch.float16)


def test_reference_parity_plain():
    ref = _load_ref()
    if ref is None:
        print("SKIP reference parity (no BitMoD clone)")
        return
    for bits, dt in [(3, "fp3"), (3, "int3"), (4, "fp4"), (4, "int4"),
                     (4, "flint4"), (4, "fp4_er_pos"), (4, "fp4_ea_neg")]:
        for gs in (128, 64, None):
            for seed in range(3):
                w = _weights(seed=seed)
                ours = quant_weight_dtype(w, bits, dt, group_size=gs)
                theirs = ref.quant_datatype(
                    w.clone(), wq_bits=bits, datatype=dt, group_size=gs)
                assert torch.equal(ours, theirs), \
                    f"plain mismatch: {dt} g{gs} seed{seed}"
    print("PASS reference parity: plain datatypes (fp3/fp4/int/flint/ER/EA)")


def test_reference_parity_mixed():
    ref = _load_ref()
    if ref is None:
        print("SKIP reference parity (no BitMoD clone)")
        return
    for bits in (3, 4):
        for dt in ("mixed_bitmod", "mixed_er", "mixed_ea", "mixed_ant"):
            for gs in (128, None):
                for seed in range(3):
                    w = _weights(seed=seed + 10)
                    ours = quant_weight_dtype(w, bits, dt, group_size=gs)
                    theirs = ref.search_datatype(
                        w.clone(), wq_bits=bits, datatype=dt, group_size=gs)
                    assert torch.equal(ours, theirs), \
                        f"mixed mismatch: {bits}b {dt} g{gs} seed{seed}"
    print("PASS reference parity: mixed datatypes (bitmod/er/ea/ant, 3b+4b)")


def test_tail_group():
    # C not divisible by group: remainder quantized as its own group,
    # consistent with quantizing the two slices independently.
    w = _weights(k=8, c=520)  # 4*128 + 8
    out = quant_weight_dtype(w, 4, "mixed_bitmod", group_size=128)
    main = quant_weight_dtype(w[:, :512], 4, "mixed_bitmod", group_size=128)
    tail = quant_weight_dtype(w[:, 512:], 4, "mixed_bitmod", group_size=None)
    assert torch.equal(out[:, :512], main) and torch.equal(out[:, 512:], tail)
    print("PASS tail group (C % group_size != 0)")


def test_parse_config():
    from benchmark.quant.eval_quant import parse_config
    c = parse_config("W4A16_bitmod")
    assert (c.w_bits, c.a_bits, c.w_dtype) == (4, 16, "mixed_bitmod"), c
    c = parse_config("W3A16_fp")
    assert (c.w_bits, c.a_bits, c.w_dtype) == (3, 16, "fp3"), c
    c = parse_config("W4A16_symm")
    assert (c.w_bits, c.a_bits, c.w_dtype, c.sym) == (4, 16, "int", True), c
    c = parse_config("W4A4_asymm")   # existing tags: byte-identical behavior
    assert (c.w_bits, c.a_bits, c.w_dtype, c.sym) == (4, 4, "int", False), c
    assert c.tag() == "W4A4_asymm"
    assert parse_config("W4A16_bitmod").tag() == "W4A16_bitmod"
    for bad in ("W4A16_foo", "W4A16_"):
        try:
            parse_config(bad)
            raise AssertionError(f"{bad} should have raised")
        except SystemExit:
            pass
    print("PASS parse_config tags")


def test_quantlinear_weight_only():
    import torch.nn as nn
    from benchmark.quant.ptq import QuantConfig, QuantLinear
    torch.manual_seed(0)
    lin = nn.Linear(256, 32).to(torch.float16)
    qcfg = QuantConfig(w_bits=4, a_bits=16, sym=True, w_dtype="mixed_bitmod")
    ql = QuantLinear(lin, qcfg, smooth_scale=None)
    # weights actually changed, and match the pure function
    expect = quant_weight_dtype(lin.weight.data, 4, "mixed_bitmod", 128)
    assert torch.equal(ql.weight, expect)
    assert not torch.equal(ql.weight, lin.weight.data)
    # A16 forward = plain fp16 matmul on quantized weights (activations untouched)
    x = torch.randn(5, 256).to(torch.float16)
    out = ql(x)
    manual = torch.nn.functional.linear(x, expect, lin.bias.data)
    assert torch.equal(out, manual)
    # int path untouched by the new field (default w_dtype="int")
    qcfg_int = QuantConfig(w_bits=4, a_bits=16, sym=True)
    assert qcfg_int.w_dtype == "int" and qcfg_int.tag() == "W4A16_symm"
    print("PASS QuantLinear weight-only bitmod path")


def test_grid_membership():
    # every dequantized value must be exactly (grid value) * (its group scale)
    from benchmark.quant.bitmod_dtypes import FP4_E2M1
    w = _weights(k=4, c=256)
    out = quant_weight_dtype(w, 4, "fp4", group_size=128)
    grid = torch.tensor(FP4_E2M1, dtype=torch.float16)
    for r in range(4):
        for gi in range(2):
            grp_w = w[r, gi * 128:(gi + 1) * 128]
            grp_q = out[r, gi * 128:(gi + 1) * 128]
            scale = (grp_w.abs().amax() / 12.0).clamp(1e-5, 1e4)
            ratio = grp_q.float() / scale.float()
            d = (ratio.unsqueeze(-1) - grid.float().unsqueeze(0)).abs().min(-1).values
            assert d.max() < 1e-2, f"off-grid value, row {r} group {gi}"
    print("PASS grid membership")


if __name__ == "__main__":
    test_reference_parity_plain()
    test_reference_parity_mixed()
    test_tail_group()
    test_parse_config()
    test_quantlinear_weight_only()
    test_grid_membership()
    print("ALL PASS")
