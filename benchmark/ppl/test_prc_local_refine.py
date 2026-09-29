"""CPU protocol checks for the allocation-only local refinement driver."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from benchmark.ppl.prc_local_refine import (
    choose_disjoint_starts,
    paired_stats,
    trace_cost,
    validate_allocation_only,
    within_cost,
)


def table_fixture():
    return {
        "stoc_len_levels": [128, 64, 32],
        "layer_buckets": 4,
        "sc_prec": 8,
        "halve_bipolar_stoc_len": True,
        "model_path": "unchanged/model",
        "dispatch_metrics": {"qk": "amax"},
        "protected_channels": {"q_proj": [0, 1]},
        "mac_per_row": {"q_proj": 8192, "qk": 2048, "av": 2048},
        "operator_defaults": {
            "q_proj": {"thresholds": [0.8, 0.2]},
            "qk": {"thresholds": [0.8, 0.2]},
            "av": {"thresholds": [0.8, 0.2]},
        },
        "buckets": {
            "q_proj:t0:l0": {"thresholds": [0.8, 0.2], "metric_mean": 0.3},
            "qk:t0:l0": {"thresholds": [0.8, 0.2], "metric_mean": 0.3},
            "av:t0:l0": {"thresholds": [0.8, 0.2], "metric_mean": 0.3},
        },
        "per_row_chunk": {
            "buckets": {
                "q_proj:t0:l0": {"levels": [8, 24, 128], "thresholds": [0.2, 0.8]},
            },
        },
    }


class WindowProtocolTests(unittest.TestCase):
    def test_aligned_disjoint_reproducible_and_full_length(self):
        ctx = 2048
        total = 40 * ctx + 121
        excluded = [0, 5 * ctx, 10 * ctx, 39 * ctx]
        args = (total, ctx, 6, 16, excluded, 1729)
        search, confirm = choose_disjoint_starts(*args)
        self.assertEqual((search, confirm), choose_disjoint_starts(*args))
        self.assertEqual(len(search), 6)
        self.assertEqual(len(confirm), 16)
        both = list(search) + list(confirm)
        self.assertEqual(len(set(both)), 22)
        self.assertFalse(set(both) & set(excluded))
        for start in both:
            self.assertEqual(start % ctx, 0)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start + ctx, total)
        starts = sorted(both + excluded)
        self.assertTrue(all(b - a >= ctx for a, b in zip(starts, starts[1:])))

    def test_insufficient_available_windows_fails(self):
        with self.assertRaises(ValueError):
            choose_disjoint_starts(5 * 2048, 2048, 2, 3, [0], 0)

    def test_historical_unaligned_intervals_are_excluded(self):
        ctx = 2048
        excluded = [ctx // 2]
        search, confirm = choose_disjoint_starts(8 * ctx, ctx, 3, 3, excluded, 7)
        self.assertEqual(set(search + confirm), set(range(2 * ctx, 8 * ctx, ctx)))


class AllocationInvariantTests(unittest.TestCase):
    def setUp(self):
        self.base = table_fixture()

    def test_noop_candidate_rejected_and_input_untouched(self):
        candidate = copy.deepcopy(self.base)
        with self.assertRaises(ValueError):
            validate_allocation_only(self.base, candidate)
        self.assertEqual(self.base, candidate)

    def test_per_group_and_attention_threshold_updates_allowed(self):
        candidate = copy.deepcopy(self.base)
        candidate["per_row_chunk"]["buckets"]["q_proj:t0:l0"]["thresholds"] = [0.3, 0.9]
        candidate["buckets"]["qk:t0:l0"]["thresholds"] = [0.7, 0.1]
        candidate["buckets"]["av:t0:l0"]["thresholds"] = [0.9, 0.3]
        validate_allocation_only(self.base, candidate)

    def test_execution_metadata_changes_rejected(self):
        replacements = {
            "sc_prec": 7,
            "halve_bipolar_stoc_len": False,
            "model_path": "different/model",
            "layer_buckets": 8,
            "stoc_len_levels": [128, 96, 32],
            "protected_channels": {"q_proj": [1, 2]},
            "dispatch_metrics": {"qk": "norm"},
            "mac_per_row": {"q_proj": 4096, "qk": 2048, "av": 2048},
        }
        for key, value in replacements.items():
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.base)
                candidate[key] = value
                with self.assertRaises(ValueError):
                    validate_allocation_only(self.base, candidate)

    def test_legacy_linear_thresholds_rejected(self):
        candidate = copy.deepcopy(self.base)
        candidate["buckets"]["q_proj:t0:l0"]["thresholds"] = [0.7, 0.1]
        with self.assertRaises(ValueError):
            validate_allocation_only(self.base, candidate)

    def test_group_ladder_and_bucket_coverage_changes_rejected(self):
        candidate = copy.deepcopy(self.base)
        candidate["per_row_chunk"]["buckets"]["q_proj:t0:l0"]["levels"] = [8, 32, 128]
        with self.assertRaises(ValueError):
            validate_allocation_only(self.base, candidate)
        candidate = copy.deepcopy(self.base)
        candidate["per_row_chunk"]["buckets"].pop("q_proj:t0:l0")
        with self.assertRaises(ValueError):
            validate_allocation_only(self.base, candidate)

    def test_invalid_threshold_arrays_rejected(self):
        for values in ([0.8, 0.2], [0.2], [-0.1, 0.8], [0.2, 1.1], [0.2, math.nan]):
            with self.subTest(values=values):
                candidate = copy.deepcopy(self.base)
                candidate["per_row_chunk"]["buckets"]["q_proj:t0:l0"]["thresholds"] = values
                with self.assertRaises(ValueError):
                    validate_allocation_only(self.base, candidate)


class PairedSelectionTests(unittest.TestCase):
    def test_noop_is_finite_tie(self):
        stats = paired_stats([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(stats["mean_dnll"], 0.0)
        self.assertEqual(stats["se"], 0.0)
        self.assertEqual(stats["z"], 0.0)
        self.assertEqual(stats["dppl_pct"], 0.0)

    def test_noisy_tiny_improvement_is_not_strong_evidence(self):
        stats = paired_stats([1.1, 0.89, 1.1, 0.9], [1.0] * 4)
        self.assertLess(stats["mean_dnll"], 0)
        self.assertGreater(stats["z"], -1.5)
        self.assertGreater(stats["se"], 0)

    def test_sign_and_effect_conversion(self):
        stats = paired_stats([1.98, 1.97, 1.99, 1.98], [2.0] * 4)
        self.assertAlmostEqual(stats["mean_dnll"], -0.02)
        self.assertLess(stats["z"], -1.5)
        self.assertAlmostEqual(stats["dppl_pct"], 100 * math.expm1(-0.02))

    def test_bad_paired_inputs_fail(self):
        for candidate, reference in (([], []), ([1], [1, 2]), ([math.nan, 1], [1, 1]),
                                     ([1, 2], [1, math.inf])):
            with self.subTest(candidate=candidate, reference=reference):
                with self.assertRaises(ValueError):
                    paired_stats(candidate, reference)

    def test_cost_cap_is_relative_and_fail_closed(self):
        self.assertTrue(within_cost(33.0, 33.0, 0.01))
        self.assertTrue(within_cost(33.3, 33.0, 0.01))
        self.assertFalse(within_cost(33.4, 33.0, 0.01))
        for candidate, baseline in ((0, 33), (-1, 33), (33, 0), (33, -1),
                                     (math.nan, 33), (33, math.nan),
                                     (math.inf, 33), (33, math.inf)):
            with self.subTest(candidate=candidate, baseline=baseline):
                self.assertFalse(within_cost(candidate, baseline, 0.01))


class TraceCostTests(unittest.TestCase):
    def test_cost_uses_true_macs_not_rows_or_group_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps({"schema": "scmp-trace-summary-v1", "groups": [
                {"macs": 128 * 100, "stoc_len": 16, "rows": 100},
                {"macs": 32 * 100, "stoc_len": 128, "rows": 100},
            ]}))
            cost = trace_cost(path)
            self.assertEqual(cost["total_macs"], 16000)
            self.assertEqual(cost["total_cycle_macs"], 614400)
            self.assertEqual(cost["cost"], 38.4)

    def test_empty_or_nonfinite_trace_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            for groups in ([], [{"macs": math.nan, "stoc_len": 32}],
                           [{"macs": 10, "stoc_len": math.inf}]):
                path.write_text(json.dumps({"schema": "scmp-trace-summary-v1", "groups": groups}))
                with self.assertRaises(ValueError):
                    trace_cost(path)


if __name__ == "__main__":
    unittest.main()
