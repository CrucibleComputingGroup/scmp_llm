"""CPU checks for tile-max avg_sl (no model/GPU required).

    python -m pytest benchmark/ppl/test_tile_cost.py -q      (from scmp_llm/)
"""
from __future__ import annotations

import unittest

import numpy as np


def brute_tiles(lengths, row_macs, r):
    """Reference: loop over tiles of r consecutive rows, per chunk."""
    n, c = len(lengths), len(row_macs)
    total = 0.0
    for t0 in range(0, n, r):
        rows = range(t0, min(t0 + r, n))
        for ch in range(c):
            total += max(lengths[i][ch] for i in rows) * len(rows) * row_macs[ch]
    return total


class TileMathTests(unittest.TestCase):
    def test_matches_brute_force(self):
        import torch
        from benchmark.ppl.tile_cost import tile_cycle_macs
        rng = np.random.default_rng(0)
        ladder = np.array([8, 10, 12, 16, 24, 32, 64, 128])
        for n, c in ((1, 1), (5, 3), (32, 2), (33, 4), (70, 3), (64, 1)):
            lengths = ladder[rng.integers(0, len(ladder), size=(3, n, c))]
            macs = rng.integers(1, 1000, size=c).astype(np.float64)
            got = tile_cycle_macs(torch.as_tensor(lengths), torch.as_tensor(macs), (1, 2, 8, 32))
            for r, v in got.items():
                want = sum(brute_tiles(lengths[g].tolist(), macs.tolist(), r) for g in range(3))
                self.assertAlmostEqual(float(v), want, places=6, msg=f"n={n} c={c} r={r}")

    def test_uniform_rows_cost_nothing_extra(self):
        import torch
        from benchmark.ppl.tile_cost import tile_cycle_macs
        lengths = torch.full((2, 45, 3), 24)
        got = tile_cycle_macs(lengths, torch.tensor([128.0, 128.0, 5.0]), (1, 8, 32))
        self.assertTrue(all(float(v) == float(got[1]) for v in got.values()))


class RealDispatchTests(unittest.TestCase):
    """Real config, SCLinear, per-(row,chunk) rungs, attention classifier and escape;
    only the SC kernel is replaced by a CPU FP matmul that records what it executes."""

    def test_real_dispatch_cpu_integration(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        import model.sc_common as scm
        from scmp_kernels.mp.config import AdaptiveMPConfig
        from benchmark.ppl.tile_cost import TileCollector

        mp = AdaptiveMPConfig([32, 16, 8])
        mp.per_row_chunk = {}
        mp.bucket_thresholds = {("q_proj", 0, 0): [.75, .25], ("qk", 0, 0): [.75, .25]}
        mp._load_per_row_chunk({"buckets": {"q_proj:t0:l0": {
            "levels": [8, 16, 32], "thresholds": [.25, .75]}}})
        mp.protected_channel_indices = {("q_proj", 0, None): [0, 1, 2, 3]}
        mp.protected_channel_stoc_len = 64
        mp.escape_gate_k = 2.0
        mp.escape_stoc_len = 64
        mp.bucket_escape_thresholds = {("qk", 0, 0): .8}
        cfg = SimpleNamespace(sc_mp_config=mp, use_sc_linear=True, _sc_total_blocks=2,
                              sc_linear_chunk_d=128, sc_halve_bipolar_stoc_len=True)
        mod = scm.SCLinear(261, 5, bias=False, sc_config=cfg)
        mod._sc_op_name, mod._sc_block_idx, mod._sc_unit_idx = "q_proj", 0, 1
        g = torch.Generator().manual_seed(0)
        x = torch.randn(6, 261, generator=g) * torch.linspace(0.2, 3.0, 261)
        # distinct attention rows so the stub can recover each row's position
        a = torch.randn(1, 2, 5, 17, generator=g) * torch.tensor([.2, 1., 3., .5, 2.])[:, None]
        b = torch.randn(1, 2, 7, 17, generator=g)
        kw = dict(mode="bipolar", sc_prec=8, stoc_len=32, halve_bipolar_stoc_len=True,
                  mp_config=mp, operator="qk", block_idx=0, total_blocks=2)

        executed = dict(macs=0.0, cycles=0.0)
        linear_tables = []          # (lengths (n, c), row_macs (c,)) in kernel order
        uniform = []                # (left, right, stoc_len)

        def fp_kernel(left, right, **options):
            n, d, m = left.shape[0], left.shape[1], right.shape[0]
            if options.get("rung_table") is not None:
                rungs = options["rung_table"].long()
                widths = torch.full((rungs.shape[1],), float(options["chunk_d"]), dtype=torch.float64)
                widths[-1] = d - options["chunk_d"] * (len(widths) - 1)
                lengths = torch.tensor(options["level_lens"], dtype=torch.float64)[rungs]
                executed["cycles"] += float((lengths * widths[None, :] * m).sum())
                linear_tables.append((lengths.tolist(), (widths * m).tolist()))
            else:
                executed["cycles"] += n * d * m * options["stoc_len"]
                uniform.append((left.clone(), right.clone(), options["stoc_len"]))
            executed["macs"] += n * d * m
            return left @ right.T

        def run():
            y = mod(x)
            y2 = mod(x[:2])      # another expert-style call with its own population
            mod(x[:0])           # empty expert call
            att = scm._sc_attention_matmul_ab_t(a, b, **kw)
            return y, y2, att

        old = (scm.SCLinear.forward, scm.per_row_chunk_rungs,
               scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows)
        rows = (1, 2, 4)
        with patch.object(scm, "_HAS_SC", True), patch.object(scm, "_sc_matmul", fp_kernel):
            with torch.no_grad():
                plain = run()
            executed.update(macs=0.0, cycles=0.0)
            linear_tables.clear()
            uniform.clear()
            state = torch.random.get_rng_state().clone()
            col = TileCollector(rows)
            with col.installed(scm), torch.no_grad():
                hooked = run()
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertTrue(all(torch.equal(p, h) for p, h in zip(plain, hooked)))
        self.assertEqual(old, (scm.SCLinear.forward, scm.per_row_chunk_rungs,
                               scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows))

        # R=1 reproduces exactly what the kernels executed (coverage + length map).
        total_macs = sum(float(v) for v in col.macs.values()) + col.fixed_macs
        flat = sum(float(col.cycles[(op, 1)]) for op in col.macs) + col.fixed_cycles
        self.assertAlmostEqual(total_macs, executed["macs"], places=3)
        self.assertAlmostEqual(flat, executed["cycles"], places=3)

        # Linear tiles: brute force over the rung tables the kernel actually received.
        for r in rows:
            want = sum(brute_tiles(lens, macs, r) for lens, macs in linear_tables)
            self.assertAlmostEqual(float(col.cycles[("q_proj", r)]), want, places=3)

        # Attention tiles: rebuild each row's executed length from the per-level kernel
        # calls (row identity by exact match), independently of the classifier hook.
        bh_rows = a.reshape(2, 5, 17)
        att_len = [[None] * 5 for _ in range(2)]
        for left, right, sl in uniform:
            if left.shape[1] != 17:
                continue                     # protected linear slice
            h = next(i for i in range(2) if torch.equal(right, b.reshape(2, 7, 17)[i]))
            for row in left:
                j = next(j for j in range(5) if torch.equal(row, bh_rows[h, j]))
                att_len[h][j] = sl
        self.assertTrue(all(v is not None for hr in att_len for v in hr))
        self.assertGreater(len({v for hr in att_len for v in hr}), 1, "fixture must mix lengths")
        for r in rows:
            want = sum(brute_tiles([[v] for v in att_len[h]], [17.0 * 7.0], r) for h in range(2))
            self.assertAlmostEqual(float(col.cycles[("qk", r)]), want, places=3)

        s = col.summary()
        self.assertEqual(s["calls"], {"q_proj": 2, "qk": 1})
        self.assertGreaterEqual(s["avg_sl"]["4"], s["avg_sl"]["1"])


if __name__ == "__main__":
    unittest.main()
