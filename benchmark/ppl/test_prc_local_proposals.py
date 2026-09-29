"""CPU checks for exact replay and bounded transfers (no model/GPU required)."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from benchmark.ppl.prc_local_proposals import (
    ProfileCollector, _group_cost, _lengths, baseline_cost, exact_profile_cost, propose,
)


def fixture():
    table = {"stoc_len_levels": [128, 64, 16], "layer_buckets": 4,
             "buckets": {}, "per_row_chunk": {"buckets": {}},
             "protected_channel_stoc_len": 128}
    groups = {}
    # Includes exact-threshold ties and different group widths/MACs.
    m = np.linspace(0, 1, 4097).astype(np.float32)
    for op, kind, scale in (("qk", "attention", 128 * 2048),
                            ("av", "attention", 2048 * 128),
                            ("q_proj", "prc", 128 * 2048),
                            ("down_proj", "prc", 17 * 2048)):
        levels = [16, 64, 128] if kind == "prc" else [128, 64, 16]
        th = [.25, .75] if kind == "prc" else [.75, .25]
        key = op + ":t0:l0"
        payload = {("levels" if kind == "prc" else "stoc_len_levels"): levels, "thresholds": th,
                   "metric_mean": .12, "metric_std": .09}
        (table["per_row_chunk"] if kind == "prc" else table)["buckets"][key] = payload
        groups[kind + "|" + key] = dict(kind=kind, op=op, key=key, levels=levels,
                                         thresholds=th, metric=m.tolist(),
                                         mac=np.full(m.size, scale).tolist(),
                                         edges=m.tolist(), calls=1)
    fixed_macs = 3_000_000  # protected slices + attention escape rows
    profile = dict(groups=groups, fixed_macs=fixed_macs,
                   fixed_cycle_macs=fixed_macs * 128)
    profile["total_macs"] = fixed_macs + sum(sum(g["mac"]) for g in groups.values())
    profile["total_cycle_macs"] = profile["fixed_cycle_macs"] + sum(
        _group_cost(g, g["thresholds"]) for g in groups.values())
    return table, profile


class LocalProposalTests(unittest.TestCase):
    def test_runtime_tie_semantics(self):
        g = dict(kind="prc", key="q_proj:t0:l0", levels=[8, 16, 32],
                 metric=[0, .25, .2501, .75, .7501, 1])
        self.assertEqual(_lengths(g, [.25, .75]).tolist(), [8, 8, 16, 16, 32, 32])
        g.update(kind="attention", levels=[32, 16, 8])
        self.assertEqual(_lengths(g, [.75, .25]).tolist(), [8, 16, 16, 32, 32, 32])

    def test_fixed_and_tail_costs(self):
        table, profile = fixture()
        self.assertEqual(baseline_cost(table, profile), exact_profile_cost(profile))
        g = profile["groups"]["prc|down_proj:t0:l0"]
        self.assertEqual(sum(g["mac"]), 4097 * 17 * 2048)

    def test_bounded_bidirectional_costmatched_proposals(self):
        table, profile = fixture()
        original = copy.deepcopy(table)
        proposals = propose(table, profile)
        self.assertEqual(len(proposals), 8)
        self.assertEqual(table, original)
        for result in proposals:
            d = result["diagnostics"]
            self.assertGreater(d["transferred_fraction"], .015)
            self.assertLessEqual(d["transferred_fraction"], .02000001)
            self.assertLessEqual(abs(d["predicted_cost_ratio"] - 1), .001)
            candidate = result["table"]
            for group in profile["groups"].values():
                section = "per_row_chunk" if group["kind"] == "prc" else None
                before = (table[section] if section else table)["buckets"][group["key"]]
                after = (candidate[section] if section else candidate)["buckets"][group["key"]]
                self.assertEqual({k: v for k, v in before.items() if k != "thresholds"},
                                 {k: v for k, v in after.items() if k != "thresholds"})
                self.assertTrue(all(0 <= t <= 1 for t in after["thresholds"]))

    def test_reject_wrong_incumbent_even_at_equal_cost(self):
        table, profile = fixture()
        table["per_row_chunk"]["buckets"]["q_proj:t0:l0"]["thresholds"] = [.3, .7]
        with self.assertRaises(ValueError):
            propose(table, profile)

    def test_reject_cost_replay_mismatch(self):
        table, profile = fixture()
        profile["total_cycle_macs"] *= 1.01
        with self.assertRaisesRegex(ValueError, "does not replay"):
            propose(table, profile)

    def test_pinned_attention_has_no_increase(self):
        table, profile = fixture()
        g = profile["groups"]["attention|qk:t0:l0"]
        g["thresholds"] = [0, 0]
        table["buckets"][g["key"]]["thresholds"] = [0, 0]
        profile["total_cycle_macs"] = profile["fixed_cycle_macs"] + sum(
            _group_cost(g, g["thresholds"]) for g in profile["groups"].values())
        names = [p["name"] for p in propose(table, profile)]
        self.assertNotIn("qk_from_linears", names)
        self.assertNotIn("qk_from_av", names)
        self.assertIn("linears_from_qk", names)

    def test_real_dispatch_cpu_integration(self):
        """Real config, SCLinear, classifiers, and attention; only matmul is CPU FP."""
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            self.skipTest("CPU torch unavailable in this interpreter")
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        import model.sc_common as scm
        from scmp_kernels.mp.config import AdaptiveMPConfig

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
        x = torch.arange(4 * 261, dtype=torch.float32).reshape(4, 261) / 37
        a = torch.arange(4 * 17, dtype=torch.float32).reshape(1, 1, 4, 17) / 13
        b = torch.arange(7 * 17, dtype=torch.float32).reshape(1, 1, 7, 17) / 19
        kw = dict(mode="bipolar", sc_prec=8, stoc_len=32,
                  halve_bipolar_stoc_len=True, mp_config=mp,
                  operator="qk", block_idx=0, total_blocks=2)
        observed = []

        def fp_kernel(left, right, **options):
            n, d, m = left.shape[0], left.shape[1], right.shape[0]
            if "rung_table" in options:
                rungs = options["rung_table"].long()
                widths = torch.full((rungs.shape[1],), options["chunk_d"], dtype=torch.float64)
                widths[-1] = d - options["chunk_d"] * (len(widths) - 1)
                lengths = torch.tensor(options["level_lens"])[rungs]
                cycles = float((lengths * widths[None, :] * m).sum())
            else:
                cycles = n * d * m * options["stoc_len"]
            observed.append((n * d * m, cycles))
            return left @ right.T

        def run():
            y = mod(x)
            mod(x[:2])  # another actual expert call, different routed population
            mod(x[:0])  # sparse empty expert
            att = scm._sc_attention_matmul_ab_t(a, b, **kw)
            scm._sc_attention_matmul_ab_t(torch.ones_like(a), b, **kw)
            return y, att

        old = (scm.SCLinear.forward, scm.per_row_chunk_rungs,
               scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows)
        with patch.object(scm, "_HAS_SC", True), patch.object(scm, "_sc_matmul", fp_kernel):
            plain = run()
            observed.clear()
            state = torch.random.get_rng_state().clone()
            collector = ProfileCollector(None, bins=256)
            with collector, torch.no_grad():
                profiled = run()
            profile = collector.snapshot()
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertTrue(all(torch.equal(a_, b_) for a_, b_ in zip(plain, profiled)))
            self.assertEqual(profile["total_macs"], sum(x_[0] for x_ in observed))
            self.assertEqual(profile["total_cycle_macs"], sum(x_[1] for x_ in observed))
            self.assertEqual(profile["groups"]["prc|q_proj:t0:l0"]["calls"], 2)
            self.assertGreater(profile["fixed_macs"], 6 * 4 * 5)
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with ProfileCollector(None):
                    raise RuntimeError("intentional")
        self.assertEqual(old, (scm.SCLinear.forward, scm.per_row_chunk_rungs,
                              scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows))


if __name__ == "__main__":
    unittest.main()
