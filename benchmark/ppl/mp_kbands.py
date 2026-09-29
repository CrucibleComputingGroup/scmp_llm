"""K-band (Phase 3) table construction: per-group stream lengths inside a row.

Phase 1 allocates stream length across rows; Phase 2 fine-tunes the ladder
end-to-end. Both leave every quantization chunk of a row at the SAME length,
even though the kernel already quantizes each 128-wide chunk with its own
scale. Phase 3 spends that unused axis: the residual (non-protected)
contraction dim is partitioned into bands of whole chunks, and band ``b`` runs
row-rung ``k`` at its own length ``L[b][k]``.

Row dispatch is untouched -- same metric, same thresholds, same escape gate,
same rung index per row -- so the per-row parent sits INSIDE this space at
``L[b][k] = L_parent[k]`` for every band, and Phase 3 can only redistribute,
never overspend. Iso-compute is a per-rung identity rather than a tolerance:

    sum_b (w_b / R) * L[b][k]  ==  L_parent[k]     for every rung k

with ``w_b`` the band's column count and ``R`` the residual width. Because MACs
are linear in the contraction dim, a column fraction IS the MAC fraction, so
this identity holds for any input, any rung occupancy, and any escape-gate
firing pattern.

Scope: LINEARS ONLY. ``qk`` contracts over head_dim=128 -- exactly one chunk,
so the row IS the group and there is nothing to partition. ``av`` contracts
over sequence length but runs unchunked today, so banding it would change
baseline numerics rather than extend the allocation. Attention therefore keeps
the parent's allocation byte-for-byte, which is also what makes the budget
identity above exact for the model as a whole.

Usage:
    # identity child (all bands = parent ladder) -- the S1 sanity control
    python -m benchmark.ppl.mp_kbands identity \\
        --parent <bundle_dir> --out <dir> --n-bands 2

    # apply a solved allocation
    python -m benchmark.ppl.mp_kbands apply \\
        --parent <bundle_dir> --alloc alloc.json --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "kernels") not in sys.path:
    sys.path.insert(0, str(REPO / "kernels"))

from scmp_kernels.mp.config import residual_chunk_widths  # noqa: E402

CHUNK_D = 128
# Operators Phase 3 can reach. qk/av are excluded structurally (see module
# docstring); keep this explicit so a future op cannot be banded by accident.
BANDABLE_OPS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# parent bundle
# ---------------------------------------------------------------------------

def load_parent(bundle: Path):
    """(wrapper, table, table_filename) for a deployed mp_best bundle."""
    wrapper = json.loads((bundle / "wrapper.json").read_text())
    table_name = wrapper.get("threshold_table_path", "table.json")
    table = json.loads((bundle / Path(table_name).name).read_text())
    return wrapper, table, Path(table_name).name


def parent_dims(table: dict) -> dict:
    """operator -> full contraction width D, from the protected-channel dims.

    The calibrator records ``protected_channels.dims`` per (op, block); the
    width is constant per operator, so collapse it and fail loudly if it is
    not (a per-block D would mean the residual chunk count varies within a
    layer bucket and no single ladder set could hold the iso-cost identity).
    """
    dims = (table.get("protected_channels") or {}).get("dims") or {}
    out: dict[str, int] = {}
    for key, d in dims.items():
        op = key.split(":")[0]
        prev = out.setdefault(op, int(d))
        if prev != int(d):
            raise ValueError(
                f"operator '{op}' has contraction width {d} in '{key}' but "
                f"{prev} elsewhere; band widths must be constant per operator.")
    return out


def residual_widths(table: dict) -> dict:
    """operator -> residual width R = D - |protected|, verified constant."""
    pc = table.get("protected_channels") or {}
    indices = pc.get("indices") or {}
    dims = parent_dims(table)
    if not indices:
        # No protected channels: the residual IS the full width.
        return dict(dims)
    out: dict[str, int] = {}
    for key, idx in indices.items():
        op = key.split(":")[0]
        r = int(dims[op]) - len(idx)
        prev = out.setdefault(op, r)
        if prev != r:
            raise ValueError(
                f"operator '{op}' has residual width {r} in '{key}' but "
                f"{prev} elsewhere; |protected| must be constant per operator "
                f"for one ladder set to satisfy the iso-cost identity.")
    return out


def parent_blocks(table: dict) -> dict:
    """operator -> sorted block indices the parent actually calibrated."""
    pc = (table.get("protected_channels") or {}).get("indices") or {}
    out: dict[str, list[int]] = {}
    for key in pc:
        op, b = key.split(":")[0], key.split(":")[1]
        out.setdefault(op, []).append(int(b[1:]))
    for op in out:
        out[op] = sorted(out[op])
    return out


def bucket_ladder(table: dict, op: str, l_bucket: int) -> list[int]:
    """The row ladder a bucket classifies against (bucket > op default > global)."""
    b = (table.get("buckets") or {}).get(f"{op}:t0:l{l_bucket}")
    if isinstance(b, dict) and "stoc_len_levels" in b:
        return [int(x) for x in b["stoc_len_levels"]]
    od = (table.get("operator_defaults") or {}).get(op)
    if isinstance(od, dict) and "stoc_len_levels" in od:
        return [int(x) for x in od["stoc_len_levels"]]
    return [int(x) for x in table["stoc_len_levels"]]


# ---------------------------------------------------------------------------
# band geometry
# ---------------------------------------------------------------------------

def equal_count_bands(n_chunks: int, n_bands: int) -> list[int]:
    """Contiguous, near-equal-count band map. The identity control's default.

    Contiguity is deliberate here and ONLY here: for the S1 identity check the
    map must not encode any importance signal, or a null result would be
    ambiguous between "bands do nothing" and "this particular split does
    nothing".
    """
    if n_chunks < 2 * n_bands:
        raise ValueError(
            f"{n_chunks} chunks cannot make {n_bands} bands of >= 2 chunks")
    return [min(c * n_bands // n_chunks, n_bands - 1) for c in range(n_chunks)]


def band_widths(band_of_chunk: list[int], residual_width: int,
                n_bands: int) -> list[int]:
    widths = residual_chunk_widths(residual_width, CHUNK_D)
    out = [0] * n_bands
    for c, b in enumerate(band_of_chunk):
        out[b] += widths[c]
    return out


def rung_candidates(parent_len: int, widths: list[int], hot: int, delta: int,
                    tol: float, cap: int = 128):
    """Feasible band lengths for one rung: ``hot`` gets +delta, others pay back.

    NEVER OVERSPENDS. Integer lengths make exact equality unreachable, so an
    earlier version accepted anything within +-tol -- and the search then
    systematically preferred the candidates that happened to round UP, because
    a longer stream always lowers error. That biased 83 of 140 rungs to
    overspend (mean +0.062% of budget): the optimizer was buying its win with
    compute rather than with allocation. Feasibility is now one-sided,

        -tol <= realized - parent_len <= 0,

    so any Phase-3 gain is achieved at NO MORE than the parent's MAC cost and
    "iso-compute" needs no asterisk. The parent (delta=0, residue exactly 0) is
    always feasible, so the search never comes back empty.

    Yields every integer payback rounding that satisfies this, letting the
    caller score them by measured error.
    """
    n = len(widths)
    total = float(sum(widths))
    rest = [b for b in range(n) if b != hot]
    if not rest:
        return
    rest_w = float(sum(widths[b] for b in rest))
    if rest_w <= 0:
        return
    exact_give = widths[hot] * delta / rest_w
    seen = set()
    # floor and ceil of the payback, plus one extra cycle of payback: with a
    # one-sided constraint the rounding that lands just over is inadmissible,
    # so the admissible neighbour must be reachable.
    import math
    for give in {math.floor(exact_give), math.ceil(exact_give),
                 math.ceil(exact_give) + 1}:
        lens = [0] * n
        lens[hot] = parent_len + delta
        for b in rest:
            lens[b] = parent_len - int(give)
        if any(v <= 0 or v > cap for v in lens):
            continue
        key = tuple(lens)
        if key in seen:
            continue
        seen.add(key)
        realized = sum(widths[b] * lens[b] for b in range(n)) / total
        residue = realized - parent_len
        if residue > 0.0 or residue < -tol:
            continue
        yield lens, residue


# ---------------------------------------------------------------------------
# table emission
# ---------------------------------------------------------------------------

def build_k_bands(table: dict, n_bands: int, *,
                  band_map_fn=None, ladder_fn=None, tol: float = 0.25) -> dict:
    """The ``k_bands`` section for a child table.

    band_map_fn(op, block, n_chunks) -> list[int]   (default: equal-count)
    ladder_fn(op, l_bucket, parent_ladder, widths) -> list[list[int]]
        (default: identity -- every band gets the parent ladder)
    """
    r_widths = residual_widths(table)
    blocks = parent_blocks(table)
    layer_buckets = int(table.get("layer_buckets", 1))

    chunk_bands: dict[str, list[int]] = {}
    ladders: dict[str, list[list[int]]] = {}
    used_ops: list[str] = []

    for op in BANDABLE_OPS:
        if op not in r_widths:
            continue
        R = r_widths[op]
        n_chunks = len(residual_chunk_widths(R, CHUNK_D))
        # PER-OPERATOR band count: n_bands is a MAXIMUM, clamped by how many
        # chunks this operator has (every band needs >= 2). Previously an op
        # that could not reach the global n_bands was DROPPED entirely, which
        # is how `--n-bands 16` silently covered down_proj alone.
        n_bands_op = max(2, min(n_bands, n_chunks // 2))
        if n_chunks < 4:
            print(f"[kbands] SKIP {op}: residual {R} = {n_chunks} chunks, "
                  f"too few to band at all")
            continue
        if n_bands_op != n_bands:
            print(f"[kbands] {op}: {n_chunks} chunks -> {n_bands_op} bands "
                  f"(capped from {n_bands})")
        used_ops.append(op)
        for blk in blocks.get(op, []):
            bmap = (band_map_fn(op, blk, n_chunks) if band_map_fn
                    else equal_count_bands(n_chunks, n_bands_op))
            if len(bmap) != n_chunks:
                raise ValueError(
                    f"band map for {op}:b{blk} has {len(bmap)} entries, "
                    f"expected {n_chunks}")
            chunk_bands[f"{op}:b{blk}"] = list(bmap)
        widths = band_widths(chunk_bands[f"{op}:b{blocks[op][0]}"], R,
                             max(chunk_bands[f"{op}:b{blocks[op][0]}"]) + 1)
        for lb in range(layer_buckets):
            parent = bucket_ladder(table, op, lb)
            if ladder_fn is None:
                per_band = [list(parent) for _ in range(len(widths))]
            else:
                per_band = ladder_fn(op, lb, parent, widths)
            if per_band is None:
                per_band = [list(parent) for _ in range(len(widths))]
            # verify the identity here too: the loader will reject it anyway,
            # but failing at build time names the offending bucket
            total = float(sum(widths))
            for k, pl in enumerate(parent):
                realized = sum(widths[b] * per_band[b][k]
                               for b in range(len(widths))) / total
                # one-sided, mirroring the loader: overspend is a hard error,
                # underspend only means the cell is cheaper than its parent
                if realized - pl > tol:
                    raise ValueError(
                        f"{op}:t0:l{lb} rung {k}: band mean {realized:.4f} "
                        f"OVERSPENDS vs parent {pl} (tol {tol})")
                if pl - realized > 2.0:
                    raise ValueError(
                        f"{op}:t0:l{lb} rung {k}: band mean {realized:.4f} "
                        f"underspends parent {pl} by {pl - realized:.3f} "
                        f"cycles - solver is leaving budget unspent")
            ladders[f"{op}:t0:l{lb}"] = per_band

    if not used_ops:
        raise ValueError("no operator could be banded; nothing to do")
    return {
        "n_bands": int(n_bands),
        "chunk_d": CHUNK_D,
        "residual_width": {op: int(r_widths[op]) for op in used_ops},
        "chunk_bands": chunk_bands,
        "ladders": ladders,
    }


def write_child(parent_dir: Path, out_dir: Path, k_bands: dict,
                note: str) -> Path:
    """Copy a parent bundle and inject the k_bands section into its table."""
    wrapper, table, table_name = load_parent(parent_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in parent_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
    child = copy.deepcopy(table)
    child["k_bands"] = k_bands
    child["k_bands_note"] = note
    (out_dir / table_name).write_text(json.dumps(child))
    return out_dir / "wrapper.json"


# ---------------------------------------------------------------------------

def _summarize(k_bands: dict, table: dict) -> None:
    n = k_bands["n_bands"]
    print(f"[kbands] n_bands={n} ops={sorted(k_bands['residual_width'])}")
    for op, R in sorted(k_bands["residual_width"].items()):
        first = next(k for k in k_bands["chunk_bands"] if k.startswith(op + ":"))
        w = band_widths(k_bands["chunk_bands"][first], R, n)
        print(f"  {op:11} R={R:6} chunks={len(residual_chunk_widths(R, CHUNK_D)):3} "
              f"band_widths={w} shares={[round(x / R, 4) for x in w]}")
    lb0 = {k: v for k, v in k_bands["ladders"].items() if k.endswith(":l0")}
    for key, per_band in sorted(lb0.items())[:3]:
        op = key.split(":")[0]
        print(f"  {key:22} parent={bucket_ladder(table, op, 0)}")
        for b, rungs in enumerate(per_band):
            print(f"      band{b} -> {rungs}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_id = sub.add_parser("identity",
                          help="all bands = parent ladder (S1 control)")
    p_id.add_argument("--parent", required=True)
    p_id.add_argument("--out", required=True)
    p_id.add_argument("--n-bands", type=int, default=2)

    p_ap = sub.add_parser("apply", help="apply a solved per-rung allocation")
    p_ap.add_argument("--parent", required=True)
    p_ap.add_argument("--alloc", required=True)
    p_ap.add_argument("--out", required=True)

    args = ap.parse_args()
    parent = Path(args.parent)
    wrapper, table, _ = load_parent(parent)

    if args.cmd == "identity":
        kb = build_k_bands(table, args.n_bands)
        note = (f"S1 identity control: {args.n_bands} equal-count bands, every "
                f"band = parent ladder. Must reproduce the parent to fp32 "
                f"accumulation regrouping.")
    else:
        alloc = json.loads(Path(args.alloc).read_text())
        n_bands = int(alloc["n_bands"])

        def band_map_fn(op, blk, n_chunks):
            return alloc["chunk_bands"][f"{op}:b{blk}"]

        def ladder_fn(op, lb, parent_ladder, widths):
            return alloc["ladders"].get(f"{op}:t0:l{lb}")

        kb = build_k_bands(table, n_bands, band_map_fn=band_map_fn,
                           ladder_fn=ladder_fn)
        note = alloc.get("note", "phase3 allocation")

    _summarize(kb, table)
    out = write_child(parent, Path(args.out), kb, note)
    print(f"[kbands] wrote child bundle -> {out.parent}")
    print(f"[kbands] MP_CONFIG_JSON={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
