"""BitMoD (HPCA-25) weight datatypes — faithful port of the fake-quant core in
BitMoD-HPCA-25/bitmod_quant/quant_utils/quant_weight.py (quant_datatype +
search_datatype), for use as a weight-only baseline inside the HPCA harness.

Semantics (identical to the reference, validated by test_bitmod_dtypes.py):
  * grouping along the INPUT (last) dim: each (out_row, group) gets one scale
  * symmetric absmax scale: scale = amax(|w|) / max(|grid|), clamped [1e-5, 1e4]
  * NO zero-point ever — asymmetry comes from asymmetric grid VALUES (ER/EA)
  * snap to nearest grid value via midpoint thresholds (ties x<=mid -> lower)
  * mixed_* datatypes: per-group argmin-MSE choice among candidate grids
  * all math in fp16, matching the reference bit-for-bit

This module is deliberately model-agnostic (pure 2-D tensor in/out, no HF or
scmp_llm imports) so scmp_diffusion / CNN work can reuse it by flattening conv
weights to (C_out, C_in*kh*kw) around the call.

Deviation from the reference: the reference hard-fails when group_size does not
divide the input dim; here a trailing remainder is quantized as its own
(smaller) group, mirroring ptq._fake_quant_lastdim_chunks. All scmp_llm decoder
linears divide 128 exactly, so this path never fires in the LLM harness.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

# ---- value grids (verbatim from quant_weight.py) -----------------------------
INT3 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
FP3 = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_ER_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
FP3_ER_NEG = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_EA_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
FP3_EA_NEG = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

INT4 = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
FLINT4 = [-16.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]
FP4_E2M1 = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_ER_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
FP4_ER_NEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_EA_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
FP4_EA_NEG = [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

FP5_E2M2 = [-28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0,
            -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0]
FP6_E2M3 = [
    -60.0, -56.0, -52.0, -48.0, -44.0, -40.0, -36.0, -32.0, -30.0, -28.0, -26.0,
    -24.0, -22.0, -20.0, -18.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0,
    -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0,
    40.0, 44.0, 48.0, 52.0, 56.0, 60.0,
]

_DATATYPES_BY_BITS: Dict[int, Dict[str, List[float]]] = {
    3: {
        "int3": INT3, "fp3": FP3,
        "fp3_er_pos": FP3_ER_POS, "fp3_er_neg": FP3_ER_NEG,
        "fp3_ea_pos": FP3_EA_POS, "fp3_ea_neg": FP3_EA_NEG,
    },
    4: {
        "int4": INT4, "fp4": FP4_E2M1, "flint4": FLINT4,
        "fp4_er_pos": FP4_ER_POS, "fp4_er_neg": FP4_ER_NEG,
        "fp4_ea_pos": FP4_EA_POS, "fp4_ea_neg": FP4_EA_NEG,
    },
    5: {"fp5": FP5_E2M2},
    6: {"fp6": FP6_E2M3},
}

_MIXED_BY_BITS: Dict[int, Dict[str, List[str]]] = {
    3: {
        "mixed_bitmod": ["fp3_er_pos", "fp3_er_neg", "fp3_ea_pos", "fp3_ea_neg"],
        "mixed_er": ["fp3_er_pos", "fp3_er_neg"],
        "mixed_ea": ["fp3_ea_pos", "fp3_ea_neg"],
        "mixed_ant": ["int3", "fp3"],
    },
    4: {
        "mixed_bitmod": ["fp4_er_pos", "fp4_er_neg", "fp4_ea_pos", "fp4_ea_neg"],
        "mixed_er": ["fp4_er_pos", "fp4_er_neg"],
        "mixed_ea": ["fp4_ea_pos", "fp4_ea_neg"],
        "mixed_ant": ["int4", "flint4"],
    },
}


@torch.no_grad()
def _quant_grid(w: torch.Tensor, grid: List[float]) -> torch.Tensor:
    """Quantize->dequantize the last dim of ``w`` (fp16) onto ``grid`` with one
    symmetric absmax scale per leading index. Mirrors quant_datatype's inner
    loop exactly (same op order, fp16 throughout, ties x<=mid go DOWN)."""
    mid = [(grid[i] + grid[i + 1]) / 2 for i in range(len(grid) - 1)]
    rmax = torch.amax(w.abs(), dim=-1, keepdim=True)
    qmax = max(abs(v) for v in grid)
    scale = (rmax / qmax).clamp(min=1e-5, max=1e4)
    x = w / scale
    q = torch.zeros_like(x)
    for i, v in enumerate(grid):
        if i == 0:
            q += torch.where(x <= mid[i], v, 0)
        elif i == len(grid) - 1:
            q += torch.where(x > mid[i - 1], v, 0)
        else:
            q += torch.where((mid[i - 1] < x) & (x <= mid[i]), v, 0)
    return q * scale


@torch.no_grad()
def _search_grids(w: torch.Tensor, grids: List[List[float]]) -> torch.Tensor:
    """Per-group argmin-MSE datatype choice (search_datatype semantics):
    ``w`` is (..., group); every candidate quantizes every group, the lowest
    fp16 MSE wins. Calibration-free — weights only, no data."""
    q = torch.zeros_like(w)
    err = torch.full(w.shape[:-1], 1e3, dtype=w.dtype, device=w.device)
    for grid in grids:
        cand = _quant_grid(w, grid)
        cand_err = (cand - w).pow(2).mean(-1)
        upd = torch.lt(cand_err, err)
        err[upd] = cand_err[upd]
        q[upd] = cand[upd]
    return q


def resolve_datatype(wq_bits: int, datatype: str) -> List[List[float]]:
    """Return the candidate grid list for ``datatype`` at ``wq_bits`` (a single
    grid for plain datatypes, several for mixed_*). Raises on unknown combos."""
    if datatype.startswith("mixed"):
        table = _MIXED_BY_BITS.get(wq_bits, {})
        if datatype not in table:
            raise ValueError(
                f"unsupported mixed datatype {datatype!r} at {wq_bits}-bit "
                f"(supported: {sorted(table) or 'none'})")
        names = table[datatype]
        return [_DATATYPES_BY_BITS[wq_bits][n] for n in names]
    table = _DATATYPES_BY_BITS.get(wq_bits, {})
    if datatype not in table:
        raise ValueError(
            f"unsupported datatype {datatype!r} at {wq_bits}-bit "
            f"(supported: {sorted(table) or 'none'})")
    return [table[datatype]]


@torch.no_grad()
def quant_weight_dtype(w: torch.Tensor, wq_bits: int, datatype: str,
                       group_size: Optional[int] = 128) -> torch.Tensor:
    """Fake-quantize a 2-D weight (out, in) with a BitMoD datatype.

    group_size None/<=0 = one group per output row (per-channel). Returns the
    dequantized tensor in the input's dtype/shape. Pure function — the caller
    (QuantLinear) owns any SmoothQuant folding, skipping, or in-place update.
    """
    assert w.dim() == 2, f"expected 2-D weight, got {tuple(w.shape)}"
    grids = resolve_datatype(wq_bits, datatype)
    orig_dtype = w.dtype
    K, C = w.shape
    # Plain datatypes go straight to the grid snap (reference quant_datatype has
    # no MSE gate); only mixed_* runs the argmin-MSE search (search_datatype).
    quant = _quant_grid if len(grids) == 1 else _search_grids
    arg = grids[0] if len(grids) == 1 else grids
    if group_size is None or group_size <= 0 or group_size >= C:
        return quant(w.to(torch.float16), arg).to(orig_dtype)  # one group/row
    n_main = (C // group_size) * group_size
    main = w[:, :n_main].reshape(K, -1, group_size).to(torch.float16)
    out = quant(main, arg).reshape(K, n_main)
    if n_main < C:                                          # remainder = own group
        tail = quant(w[:, n_main:].to(torch.float16), arg)
        out = torch.cat([out, tail], dim=-1)
    return out.to(orig_dtype)
