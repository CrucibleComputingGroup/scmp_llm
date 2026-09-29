"""Tests for per-(operator-group x layer-band) ladders.

The load-bearing test here is
GlobalModeIsARegressionControlTest.test_matches_single_ladder_on_a_real_trace:
in `global` mode the grouped aggregator must reproduce the existing,
battle-tested single-ladder aggregate_trace_weights EXACTLY on a real trace
from the last wave.  If that holds, `global` is a true control and any
difference a grouped run shows is attributable to the grouping rather than to
a rewrite of the cost accounting.
"""

import json
import math
import os
import unittest

from benchmark.ppl.mp_group_ladders import (
    GROUP_MODES,
    aggregate_trace_weights_grouped,
    apply_group_ladders_to_table,
    assert_no_pin_collision,
    group_cost_contributions,
    group_key,
    group_keys_for_mode,
    layer_band,
    project_group_ladders_to_budget,
)
from benchmark.ppl.mp_ladder_refine import aggregate_trace_weights

REAL_TRACE = (
    "/nfs/turbo/coe-nbleier/allenjin/hpca/mp_final_search_final_20260719_2130"
    "/14B/target32p000/final_test_trace.json")
REAL_LEVELS = [97, 64, 49, 32, 24, 20]
REAL_PSL = 72
REAL_BLOCKS = 40


def _trace(groups):
    return {"groups": groups}


def _g(op, block, stoc_len, macs):
    return {"op": op, "block": block, "stoc_len": stoc_len, "macs": macs}


class LayerBandTest(unittest.TestCase):

    def test_bands_partition_every_layer(self):
        seen = [layer_band(b, 40) for b in range(40)]
        self.assertEqual(seen[0], "early")
        self.assertEqual(seen[-1], "late")
        self.assertEqual(set(seen), {"early", "middle", "late"})

    def test_bands_coarsen_buckets_rather_than_splitting_layers_evenly(self):
        # With 4 layer buckets the split is l0 | l1,l2 | l3, so on 48 layers
        # the bands are 12 | 24 | 12 -- NOT equal thirds.  Equal thirds would
        # put a band boundary inside bucket l1, and a bucket is the finest
        # unit that can carry its own ladder: its blocks would then be
        # attributed to two ladders while only one could be written, and
        # realized cost would fail to reconcile.
        counts = {}
        for b in range(48):
            band = layer_band(b, 48, 4)
            counts[band] = counts.get(band, 0) + 1
        self.assertEqual(counts, {"early": 12, "middle": 24, "late": 12})

    def test_every_block_in_a_bucket_shares_one_band(self):
        for layer_buckets in (2, 4, 8):
            for total in (12, 40, 48):
                seen = {}
                for b in range(total):
                    bucket = b * layer_buckets // total
                    band = layer_band(b, total, layer_buckets)
                    self.assertEqual(seen.setdefault(bucket, band), band,
                                     f"bucket {bucket} straddles two bands "
                                     f"({total} layers, {layer_buckets} buckets)")

    def test_single_layer_model_is_early(self):
        self.assertEqual(layer_band(0, 1), "early")


class GroupKeyTest(unittest.TestCase):

    def test_modes(self):
        self.assertEqual(group_key("qk", 0, 40, "global"), "global")
        self.assertEqual(group_key("qk", 0, 40, "op"), "score")
        self.assertEqual(group_key("qk", 0, 40, "layer"), "early")
        self.assertEqual(group_key("qk", 0, 40, "op:layer"), "score:early")
        self.assertEqual(group_key("down_proj", 39, 40, "op:layer"),
                         "mlp:late")
        self.assertEqual(group_key("o_proj", 20, 40, "op:layer"),
                         "attnlin:middle")

    def test_unknown_operator_raises_rather_than_silently_sharing(self):
        with self.assertRaisesRegex(ValueError, "no ladder group"):
            group_key("mystery_proj", 0, 40, "op")

    def test_unknown_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown ladder-group mode"):
            group_key("qk", 0, 40, "per_row")

    def test_key_sets_are_complete_and_unique(self):
        for mode in GROUP_MODES:
            keys = group_keys_for_mode(mode)
            self.assertEqual(len(keys), len(set(keys)), mode)
            produced = {group_key(op, b, 40, mode)
                        for op in ("qk", "av", "q_proj", "o_proj",
                                   "gate_proj", "down_proj")
                        for b in range(40)}
            self.assertTrue(produced.issubset(set(keys)), mode)


class GroupedAggregationTest(unittest.TestCase):

    LADDERS = {"score": [96, 64], "attnlin": [96, 64], "mlp": [32, 16]}

    def test_cost_is_ladder_independent(self):
        trace = _trace([
            _g("qk", 0, 96, 100.0),
            _g("o_proj", 0, 64, 100.0),
            _g("down_proj", 0, 16, 200.0),
        ])
        got = aggregate_trace_weights_grouped(
            trace, self.LADDERS, mode="op", total_blocks=40)
        expected = (96 * 100 + 64 * 100 + 16 * 200) / 400.0
        self.assertAlmostEqual(got["cost"], expected, places=9)

    def test_shares_sum_to_one_and_split_by_group(self):
        trace = _trace([
            _g("qk", 0, 96, 100.0),
            _g("down_proj", 0, 32, 100.0),
        ])
        got = aggregate_trace_weights_grouped(
            trace, self.LADDERS, mode="op", total_blocks=40)
        self.assertAlmostEqual(got["group_shares"]["score"], 0.5)
        self.assertAlmostEqual(got["group_shares"]["mlp"], 0.5)
        self.assertAlmostEqual(got["group_shares"]["attnlin"], 0.0)
        self.assertAlmostEqual(
            sum(got["group_shares"].values()) + got["protected_weight"], 1.0)

    def test_protected_pins_are_split_out_not_priced_as_a_rung(self):
        trace = _trace([
            _g("qk", 0, 96, 300.0),
            _g("down_proj", 0, 112, 100.0),   # pin length, outside every ladder
        ])
        got = aggregate_trace_weights_grouped(
            trace, self.LADDERS, mode="op", total_blocks=40,
            protected_stoc_len=112)
        self.assertAlmostEqual(got["protected_weight"], 0.25)
        self.assertAlmostEqual(got["cost"], (96 * 300 + 112 * 100) / 400.0)

    def test_length_outside_every_ladder_raises(self):
        trace = _trace([_g("qk", 0, 77, 100.0)])
        with self.assertRaisesRegex(ValueError, "outside the adaptive"):
            aggregate_trace_weights_grouped(
                trace, self.LADDERS, mode="op", total_blocks=40)

    def test_same_length_in_two_groups_is_attributed_to_each(self):
        # 64 is a rung of BOTH score and mlp here -- the single-ladder
        # aggregator would pop it once and mis-attribute the second group
        ladders = {"score": [96, 64], "attnlin": [96, 64], "mlp": [64, 16]}
        trace = _trace([
            _g("qk", 0, 64, 100.0),
            _g("down_proj", 0, 64, 300.0),
        ])
        got = aggregate_trace_weights_grouped(
            trace, ladders, mode="op", total_blocks=40)
        self.assertAlmostEqual(got["group_level_weights"]["score"][1], 0.25)
        self.assertAlmostEqual(got["group_level_weights"]["mlp"][0], 0.75)
        self.assertAlmostEqual(got["cost"], 64.0)


@unittest.skipUnless(os.path.exists(REAL_TRACE),
                     "real 14B trace not reachable")
class GlobalModeIsARegressionControlTest(unittest.TestCase):

    def setUp(self):
        with open(REAL_TRACE) as f:
            self.trace = json.load(f)

    def test_matches_single_ladder_on_a_real_trace(self):
        old = aggregate_trace_weights(
            self.trace, REAL_LEVELS, protected_stoc_len=REAL_PSL)
        new = aggregate_trace_weights_grouped(
            self.trace, {"global": REAL_LEVELS}, mode="global",
            total_blocks=REAL_BLOCKS, protected_stoc_len=REAL_PSL)
        self.assertAlmostEqual(new["cost"], old["cost"], places=9)
        self.assertAlmostEqual(new["protected_weight"],
                               old["protected_weight"], places=12)
        for got, want in zip(new["group_level_weights"]["global"],
                             old["level_weights"]):
            self.assertAlmostEqual(got, want, places=12)

    def test_op_layer_mode_reads_the_same_trace_and_agrees_on_cost(self):
        # grouping changes ATTRIBUTION, never the realized cost
        ladders = {k: list(REAL_LEVELS) for k in group_keys_for_mode("op:layer")}
        new = aggregate_trace_weights_grouped(
            self.trace, ladders, mode="op:layer", total_blocks=REAL_BLOCKS,
            protected_stoc_len=REAL_PSL)
        old = aggregate_trace_weights(
            self.trace, REAL_LEVELS, protected_stoc_len=REAL_PSL)
        self.assertAlmostEqual(new["cost"], old["cost"], places=9)
        self.assertGreater(len([k for k, v in new["group_shares"].items()
                                if v > 0]), 1)


class ProjectionTest(unittest.TestCase):

    def test_single_group_reproduces_the_target(self):
        got = project_group_ladders_to_budget(
            {"global": [96, 64, 48, 32]},
            {"global": [0.25, 0.25, 0.25, 0.25]},
            target_cost=40.0)
        self.assertAlmostEqual(got["cost"], 40.0, delta=0.5)

    def test_each_group_keeps_its_share_of_the_cost_budget(self):
        ladders = {"score": [96, 64], "mlp": [32, 16]}
        weights = {"score": [0.2, 0.1], "mlp": [0.4, 0.3]}
        before = group_cost_contributions(ladders, weights)
        frac_before = {k: v / sum(before.values()) for k, v in before.items()}
        got = project_group_ladders_to_budget(
            ladders, weights, target_cost=sum(before.values()) * 0.8)
        after = group_cost_contributions(got["group_ladders"], weights)
        frac_after = {k: v / sum(after.values()) for k, v in after.items()}
        for key in ladders:
            self.assertAlmostEqual(frac_after[key], frac_before[key],
                                   delta=0.02)

    def test_group_with_no_mac_mass_is_left_alone(self):
        got = project_group_ladders_to_budget(
            {"score": [96, 64], "mlp": [32, 16]},
            {"score": [0.5, 0.5], "mlp": [0.0, 0.0]},
            target_cost=60.0)
        self.assertEqual(got["group_ladders"]["mlp"], [32, 16])

    def test_budget_fully_consumed_by_pins_raises(self):
        with self.assertRaisesRegex(ValueError, "consumed by protected"):
            project_group_ladders_to_budget(
                {"g": [96, 64]}, {"g": [0.5, 0.5]},
                target_cost=10.0, protected_weight=0.5,
                protected_stoc_len=112)


class PinCollisionTest(unittest.TestCase):

    def test_collision_raises_with_the_offending_groups_named(self):
        with self.assertRaisesRegex(ValueError, "mlp"):
            assert_no_pin_collision(
                {"score": [96, 64], "mlp": [72, 32]}, 72)

    def test_no_collision_passes(self):
        assert_no_pin_collision({"score": [96, 64], "mlp": [32, 16]}, 72)

    def test_no_pin_is_a_no_op(self):
        assert_no_pin_collision({"score": [96, 64]}, None)


class ApplyToTableTest(unittest.TestCase):

    def _table(self):
        buckets = {}
        for op in ("qk", "down_proj"):
            for i in range(4):
                buckets[f"{op}:t0:l{i}"] = {"thresholds": [0.5]}
        return {"stoc_len_levels": [96, 64], "buckets": buckets}

    def test_global_mode_leaves_the_table_untouched(self):
        table = self._table()
        out = apply_group_ladders_to_table(
            table, {"global": [96, 64]}, mode="global", total_blocks=40,
            layer_buckets=4)
        for payload in out["buckets"].values():
            self.assertNotIn("stoc_len_levels", payload)

    def test_op_mode_writes_each_bucket_its_group_ladder(self):
        out = apply_group_ladders_to_table(
            self._table(), {"score": [112, 96], "attnlin": [96, 64],
                            "mlp": [32, 16]},
            mode="op", total_blocks=40, layer_buckets=4)
        for i in range(4):
            self.assertEqual(
                out["buckets"][f"qk:t0:l{i}"]["stoc_len_levels"], [112, 96])
            self.assertEqual(
                out["buckets"][f"down_proj:t0:l{i}"]["stoc_len_levels"],
                [32, 16])

    def test_op_layer_mode_differentiates_layer_buckets(self):
        ladders = {k: [96, 64] for k in group_keys_for_mode("op:layer")}
        ladders["mlp:early"] = [48, 24]
        ladders["mlp:late"] = [40, 20]
        out = apply_group_ladders_to_table(
            self._table(), ladders, mode="op:layer", total_blocks=40,
            layer_buckets=4)
        early = out["buckets"]["down_proj:t0:l0"]["stoc_len_levels"]
        late = out["buckets"]["down_proj:t0:l3"]["stoc_len_levels"]
        self.assertEqual(early, [48, 24])
        self.assertEqual(late, [40, 20])
        self.assertNotEqual(early, late)


if __name__ == "__main__":
    unittest.main()
