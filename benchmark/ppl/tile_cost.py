"""Tile-max avg_sl: observe-only cost accounting on the DEPLOYED full-test eval.

Hardware model (user, 2026-09-28): the systolic array fetches one A tile of R rows x 8
contraction elements at a time, and the whole tile runs for the LONGEST stream length
among its rows. A stream length belongs to A's (row, 128-chunk) group on the linears and
to A's (head, query) row on attention. An 8-wide K slice never straddles a 128-chunk, so
a tile's length is the max over its R consecutive rows of that chunk's length:

    tile avg_sl(R) = sum_tiles maxL(tile) * MACs(tile) / sum MACs

Rows keep their execution order: tokens for the linears (the routed tokens of one call
for a MoE expert), query positions within one head for attention. MACs count real rows
only; a partial last tile adds no padding MACs. Protected columns run at one fixed length
for every row, so their tile cost equals their flat cost. INT-masked operators and FP16
ops are outside avg_sl, exactly as in the trace cost.

R=1 is today's flat avg_sl and must reproduce the trace cost. Nothing here touches
dispatch or numerics, so the PPL must equal the archived value bit-exactly.

TILE_FP_MATMUL=1 is the cheap mode: every SC matmul becomes an exact FP matmul (AWQ
scales cancel exactly in FP). Dispatch still runs the deployed code on every call, but on
the FP trajectory, so decisions can differ where SC noise would move a row across a
threshold. Its flat avg_sl is checked against the archived trace cost; the PPL is not
meaningful and no trace is written.

Usage -- the environment of kbands/run_prc_ppl.sbatch, plus TILE_OUT:
    TILE_OUT=/path/out.json [TILE_FP_MATMUL=1] python -u benchmark/ppl/tile_cost.py
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path

TILE_ROWS = (1, 8, 16, 32, 64)
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTENTION = ("qk", "av")


def tile_cycle_macs(lengths, row_macs, tile_rows=TILE_ROWS):
    """Cycle-MACs when every R-row tile runs at its longest row.

    lengths: (G, N, C) stream lengths of G independent row sequences of N rows, one
    column per contraction chunk. row_macs: (C,) MACs one row spends in each chunk.
    Returns {R: sum over tiles and chunks of max length * MACs of the tile's real rows}.
    """
    import torch
    g, n, c = lengths.shape
    lf = lengths.to(torch.float64)
    w = row_macs.to(torch.float64)
    out = {}
    for r in tile_rows:
        t = (n + r - 1) // r
        pad = t * r - n
        # Zero padding never wins the max: stream lengths are positive and the
        # last tile always holds at least one real row.
        lp = torch.nn.functional.pad(lf, (0, 0, 0, pad)) if pad else lf
        tile_max = lp.view(g, t, r, c).amax(2)
        real = torch.full((t,), float(r), dtype=torch.float64, device=lf.device)
        real[-1] = r - pad
        out[r] = (tile_max * real[None, :, None] * w[None, None, :]).sum()
    return out


class TileCollector:
    """Hooks the four sc_common entry points that fix a stream length.

    Mirrors prc_local_proposals.ProfileCollector's validated hook points, but keeps
    every row in execution order instead of histogramming the metric.
    """

    def __init__(self, tile_rows=TILE_ROWS):
        self.tile_rows = tuple(tile_rows)
        if self.tile_rows[0] != 1:
            raise ValueError("tile_rows must start at 1 (the flat avg_sl)")
        self.cycles = {}          # (op, R) -> device float64 cycle-MACs
        self.macs = {}            # op -> device float64 MACs
        self.min_len = None       # device scalar, checked at summary time
        self.fixed_macs = 0       # protected columns: one length for every row
        self.fixed_cycles = 0
        self.calls = {}
        self._linear = None
        self._attention = None

    def _add(self, op, lengths, row_macs):
        import torch
        cyc = tile_cycle_macs(lengths, row_macs, self.tile_rows)
        for r, v in cyc.items():
            key = (op, r)
            self.cycles[key] = self.cycles[key] + v if key in self.cycles else v
        g, n, _ = lengths.shape
        m = row_macs.to(torch.float64).sum() * (g * n)
        self.macs[op] = self.macs[op] + m if op in self.macs else m
        lo = lengths.min().to(torch.float64)
        self.min_len = lo if self.min_len is None else torch.minimum(self.min_len, lo)
        self.calls[op] = self.calls.get(op, 0) + 1

    @contextmanager
    def installed(self, scm):
        import torch
        orig = (scm.SCLinear.forward, scm.per_row_chunk_rungs,
                scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows)
        old_linear, old_prc, old_attention, old_classify = orig
        col = self

        def linear(mod, x):
            cfg = mod._sc_config
            op, block = getattr(mod, "_sc_op_name", None), getattr(mod, "_sc_block_idx", None)
            if (not getattr(cfg, "use_sc_linear", True) or x.numel() == 0
                    or scm._hybrid_backend(cfg, op, block) != "sc"):
                return old_linear(mod, x)
            mp = getattr(cfg, "sc_mp_config", None)
            if (op not in LINEARS or mp is None
                    or getattr(cfg, "sc_group_stoclen", None) is not None
                    or getattr(cfg, "sc_ste_grad", False)):
                raise ValueError(f"tile cost needs deployed adaptive PRC linears ({op}/{block})")
            unit = getattr(mod, "_sc_unit_idx", None)
            protected = mp.get_protected_channels(operator=op, block_idx=block, unit_idx=unit)
            count = int(scm._channel_index_tensor(protected, x.shape[-1], x.device).numel())
            rows = x.numel() // x.shape[-1]
            previous = col._linear
            col._linear = [mod, op, False]
            try:
                result = old_linear(mod, x)
                if count < x.shape[-1] and not col._linear[-1]:
                    raise ValueError(f"SC linear without a PRC dispatch: {op}/{block}")
                if count:
                    fixed_len = int(getattr(mp, "protected_channel_stoc_len", None)
                                    or max(mp.stoc_len_levels))
                    mac = rows * count * mod.out_features
                    col.fixed_macs += mac
                    col.fixed_cycles += mac * fixed_len
                return result
            finally:
                col._linear = previous

        def prc(x, chunk_d, levels, thresholds=None, target=0.0):
            result = old_prc(x, chunk_d, levels, thresholds=thresholds, target=target)
            if col._linear is None:
                raise ValueError("PRC dispatch outside an SCLinear call")
            mod, op, seen = col._linear
            if seen:
                raise ValueError("multiple PRC dispatches in one SCLinear call")
            col._linear[-1] = True
            n, d = x.shape
            if n:
                nch = result.shape[1]
                widths = torch.full((nch,), float(chunk_d), dtype=torch.float64, device=x.device)
                widths[-1] = d - (nch - 1) * chunk_d
                lengths = torch.as_tensor(list(levels), device=x.device)[result.long()]
                col._add(op, lengths[None], widths * mod.out_features)
            return result

        def attention(a, b, **kw):
            previous = col._attention
            col._attention = [tuple(a.shape), tuple(b.shape), kw, False]
            try:
                result = old_attention(a, b, **kw)
                if not col._attention[-1]:
                    raise ValueError("SC attention without adaptive row classification")
                return result
            finally:
                col._attention = previous

        def classify(metric, config, *args, **kw):
            result = old_classify(metric, config, *args, **kw)
            if col._attention is None:
                return result          # the linears' unused per-row assignment
            a_shape, b_shape, akw, seen = col._attention
            op = akw.get("operator")
            if op not in ATTENTION or kw.get("operator") != op or seen:
                raise ValueError("unrecognized attention dispatch context")
            col._attention[-1] = True
            # The same index -> length map the runtime executes (escape slot included).
            values = config.classify_level_values(
                operator=op, block_idx=akw.get("block_idx"),
                total_blocks=akw.get("total_blocks"))
            lengths = torch.as_tensor(list(values), device=metric.device)[result.row_levels.long()]
            bsz, heads, n, k = a_shape
            m = b_shape[-2]
            col._add(op, lengths.view(bsz * heads, n, 1),
                     torch.tensor([float(k * m)], dtype=torch.float64, device=metric.device))
            return result

        (scm.SCLinear.forward, scm.per_row_chunk_rungs,
         scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows) = (
            linear, prc, attention, classify)
        try:
            yield self
        finally:
            (scm.SCLinear.forward, scm.per_row_chunk_rungs,
             scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows) = orig

    def summary(self):
        if self.min_len is not None and float(self.min_len.item()) <= 0:
            raise ValueError("a dispatched stream length <= 0; the cost model assumes executed rows")
        ops = sorted(self.macs)
        macs = {op: float(self.macs[op].item()) for op in ops}
        cyc = {(op, r): float(self.cycles[(op, r)].item()) for op in ops for r in self.tile_rows}
        total = sum(macs.values()) + self.fixed_macs

        def avg(sel, r, fixed):
            num = sum(cyc[(op, r)] for op in sel) + (self.fixed_cycles if fixed else 0)
            den = sum(macs[op] for op in sel) + (self.fixed_macs if fixed else 0)
            return num / den if den else None

        lin = [op for op in ops if op in LINEARS]
        att = [op for op in ops if op in ATTENTION]
        return {
            "tile_rows": list(self.tile_rows),
            "total_macs": total,
            "protected_macs": self.fixed_macs,
            "attention_mac_share": sum(macs[op] for op in att) / total if total else None,
            "avg_sl": {str(r): avg(ops, r, True) for r in self.tile_rows},
            "linear_avg_sl": {str(r): avg(lin, r, True) for r in self.tile_rows},
            "attention_avg_sl": {str(r): avg(att, r, False) for r in self.tile_rows},
            "per_op_avg_sl": {op: {str(r): cyc[(op, r)] / macs[op] for r in self.tile_rows}
                              for op in ops},
            "calls": dict(self.calls),
        }


def main() -> int:
    out = os.environ.get("TILE_OUT")
    if not out:
        sys.exit("[tile] TILE_OUT is required")
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "kernels")]
    import model.sc_common as scm
    fp = os.environ.get("TILE_FP_MATMUL", "0") == "1"
    if fp:
        if os.environ.get("SC_MP_TRACE"):
            sys.exit("[tile] unset SC_MP_TRACE with TILE_FP_MATMUL=1 (no SC call is traced)")

        def fp_matmul(a, b, **_):
            # (a/s)(b*s)^T == a b^T exactly, so AWQ smooth_scales need no handling.
            return a.float() @ b.float().t()

        scm._sc_matmul = fp_matmul
    col = TileCollector()
    script = repo / "benchmark" / "quant" / "eval_quant.py"
    with col.installed(scm):
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name="__main__")
    s = col.summary()
    s.update(fp_matmul=fp, mp_config_json=os.environ.get("MP_CONFIG_JSON"),
             hybrid_config_json=os.environ.get("SC_HYBRID_CONFIG_JSON"),
             trace=os.environ.get("SC_MP_TRACE"))
    Path(out).write_text(json.dumps(s, indent=1))
    for r in col.tile_rows:
        print(f"[TILE] R={r:<3d} avg_sl={s['avg_sl'][str(r)]:.4f} "
              f"linear={s['linear_avg_sl'][str(r)]:.4f} "
              f"attention={s['attention_avg_sl'][str(r)] or 0:.4f}")
    print(f"[TILE] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
