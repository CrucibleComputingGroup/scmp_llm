import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmark.ppl.mp_ladder_refine import (
    _build_parent_model,
    _validate_split_rounds,
    aggregate_trace_weights,
    make_candidate_table,
    nominal_target_cost,
    propose_floor_exchange,
    weighted_cost,
)


class SplitGuardTest(unittest.TestCase):
    def test_test_split_is_evaluation_only(self):
        _validate_split_rounds("test", 0)
        with self.assertRaises(SystemExit):
            _validate_split_rounds("test", 1)
        with self.assertRaises(SystemExit):
            _validate_split_rounds("validation", 0)


class TraceWeightsTest(unittest.TestCase):
    def test_v9_protected_length_is_separated_exactly(self):
        trace = {"groups": [
            {"stoc_len": 96, "macs": 20},
            {"stoc_len": 64, "macs": 30},
            {"stoc_len": 16, "macs": 40},
            {"stoc_len": 112, "macs": 10},
        ]}
        got = aggregate_trace_weights(
            trace, [96, 64, 16], protected_stoc_len=112)
        self.assertEqual(got["level_weights"], [0.2, 0.3, 0.4])
        self.assertEqual(got["protected_weight"], 0.1)
        self.assertAlmostEqual(got["cost"], 56.0)

    def test_rejects_unexplained_sc_length(self):
        trace = {"groups": [
            {"stoc_len": 96, "macs": 90},
            {"stoc_len": 80, "macs": 10},
        ]}
        with self.assertRaises(ValueError):
            aggregate_trace_weights(
                trace, [96, 64, 16], protected_stoc_len=None)


class FloorExchangeTest(unittest.TestCase):
    def test_floor_uplift_is_funded_at_same_budget(self):
        levels = [96, 64, 16]
        weights = [0.2, 0.3, 0.4]
        target = weighted_cost(levels, weights, 0.1, 112)
        got = propose_floor_exchange(
            levels, weights,
            target_cost=target,
            protected_weight=0.1,
            protected_stoc_len=112,
            floor_step=4,
            donor_step=1,
            min_gap=1,
        )
        self.assertEqual(got["levels"], [88, 64, 20])
        self.assertAlmostEqual(got["predicted_cost"], target)
        self.assertEqual(got["levels"][-1], levels[-1] + 4)

    def test_receiver_propagates_to_preserve_strict_order(self):
        levels = [96, 64, 24, 20]
        got = propose_floor_exchange(
            levels, [0.2, 0.2, 0.2, 0.4],
            target_cost=weighted_cost(levels, [0.2, 0.2, 0.2, 0.4]),
            floor_step=4,
            donor_step=1,
            min_gap=1,
        )
        self.assertEqual(got["receiver_levels"][-2:], [25, 24])
        self.assertTrue(all(
            got["levels"][i] > got["levels"][i + 1]
            for i in range(len(got["levels"]) - 1)
        ))


class CandidateTableTest(unittest.TestCase):
    def test_nominal_target_uses_pre_compensation_budget(self):
        table = {
            "budget_ratio": 0.228,
            "budget_ref_stoc_len": 128,
            "protected_channels": {"original_budget_ratio": 0.25},
        }
        self.assertEqual(nominal_target_cost(table), 32.0)

    def test_only_level_diagnostics_change(self):
        parent = {
            "stoc_len_levels": [96, 64, 16],
            "method": "act_global_refine_fw_sq_ds2",
            "operator_defaults": {
                "q_proj": {
                    "counts": [2, 3, 5],
                    "thresholds": [0.8, 0.2],
                    "avg_stoc_len": 40.0,
                    "avg_error": 0.4,
                    "level_mean_error": [0.1, 0.2, 0.8],
                },
            },
            "buckets": {},
            "expected_avg_stoc_len": 40.0,
            "expected_flop_avg_stoc_len": 32.0,
        }
        proposal = {"predicted_cost": 32.1}
        got = make_candidate_table(
            parent, [88, 64, 20],
            parent_table_path=Path("/tmp/parent.json"),
            round_index=1,
            target_cost=32.0,
            proposal=proposal,
            history=[],
            command="test",
        )
        group = got["operator_defaults"]["q_proj"]
        self.assertEqual(group["thresholds"], [0.8, 0.2])
        self.assertEqual(group["counts"], [2, 3, 5])
        self.assertAlmostEqual(group["avg_stoc_len"], 46.8)
        self.assertNotIn("avg_error", group)
        self.assertEqual(group["pass1_avg_error"], 0.4)
        self.assertEqual(got["stoc_len_levels"], [88, 64, 20])
        self.assertTrue(
            got["pass2_ladder_refine"]["thresholds_frozen"])
        self.assertNotIn("expected_avg_stoc_len", got)

    def test_parent_model_uses_eval_quants_env_entry_point(self):
        build = mock.Mock(return_value=("model", "tokenizer"))
        fake_module = SimpleNamespace(build_model=build)
        with mock.patch.dict(
                sys.modules, {"benchmark.quant.eval_quant": fake_module}):
            with mock.patch.dict(os.environ, {}, clear=False):
                got = _build_parent_model(
                    "example/model", Path("/tmp/parent_wrapper.json"), 0.5)
                self.assertEqual(
                    os.environ["MP_CONFIG_JSON"],
                    "/tmp/parent_wrapper.json",
                )
        self.assertEqual(got, ("model", "tokenizer"))
        build.assert_called_once_with("example/model", "mp", alpha=0.5)


if __name__ == "__main__":
    unittest.main()
