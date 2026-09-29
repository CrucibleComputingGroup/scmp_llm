"""CPU protocol tests for the adjacent-rung allocation pilot.

These guard the round-3 failure modes: reusing confirmation inputs, selecting a
worse new table just because it is new, and silently changing fixed numerics.
No model, GPU, experiment archive, or scheduler is accessed by these tests.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from benchmark.ppl.prc_adjacent_refine import (
    confirmation_qualifies,
    load_excluded_windows,
    protocol_settings,
    search_eligible,
)
from benchmark.ppl.prc_local_refine import choose_disjoint_starts


def manifest_fixture():
    return {"protocol": {
        "frontend": "awq",
        "owen_mode": "bitrev",
        "scramble_masks": 64,
        "ctx": 2048,
        "stride": 2048,
        "ppl_max_tokens": 0,
        "ppl_window_batch_size": 1,
        "sc_prec": 8,
        "sc_halve": True,
        "qk_rebalance": False,
        "rng_grid": "fixed128",
        "awq_obj_bits": 4,
        "search_windows": 6,
        "confirmation_windows": 16,
        "max_candidates": 12,
        "transfer_fractions": [0.0025, 0.005],
        "operator_cost_fraction_cap": 0.02,
        "bucket_cost_fraction_cap": 0.05,
        "changed_group_fraction_cap": 0.05,
        "changed_mac_fraction_cap": 0.05,
        "cost_tolerance_fraction": 0.01,
        "confirmation_z_threshold": -1.5,
        "require_confirmation_improvement": True,
        "data_sampling_seed": 260927,
        "search_iterations": 1,
    }}


class ManifestProtocolTests(unittest.TestCase):
    def test_current_protocol_is_valid_and_input_unchanged(self):
        manifest = manifest_fixture()
        original = copy.deepcopy(manifest)
        settings = protocol_settings(manifest)
        for key, value in manifest["protocol"].items():
            self.assertEqual(settings[key], value)
        self.assertEqual(manifest, original)

    def test_settings_follow_parsed_manifest_values_not_old_cli_defaults(self):
        manifest = manifest_fixture()
        overrides = {
            "search_windows": 8,
            "confirmation_windows": 24,
            "max_candidates": 10,
            "transfer_fractions": [0.001, 0.003],
            "operator_cost_fraction_cap": 0.01,
            "bucket_cost_fraction_cap": 0.025,
            "changed_group_fraction_cap": 0.02,
            "changed_mac_fraction_cap": 0.03,
            "cost_tolerance_fraction": 0.005,
            "confirmation_z_threshold": -2.0,
            "data_sampling_seed": 1729,
        }
        manifest["protocol"].update(overrides)
        parsed = json.loads(json.dumps(manifest))
        settings = protocol_settings(parsed)
        for key, value in overrides.items():
            with self.subTest(key=key):
                self.assertEqual(settings[key], value)

    def test_fixed_numerical_variants_are_rejected(self):
        variants = {
            "frontend": "smoothquant",
            "owen_mode": "random",
            "scramble_masks": 32,
            "ctx": 1024,
            "stride": 1024,
            "ppl_max_tokens": 8192,
            "ppl_window_batch_size": 2,
            "sc_prec": 7,
            "sc_halve": False,
            "qk_rebalance": True,
            "rng_grid": "dynamic",
            "awq_obj_bits": 8,
            "require_confirmation_improvement": False,
            "search_iterations": 2,
        }
        for key, value in variants.items():
            with self.subTest(key=key):
                manifest = manifest_fixture()
                manifest["protocol"][key] = value
                with self.assertRaises(ValueError):
                    protocol_settings(manifest)

    def test_missing_protocol_or_required_settings_do_not_use_defaults(self):
        with self.assertRaises((KeyError, ValueError)):
            protocol_settings({})
        for key in ("search_windows", "transfer_fractions", "scramble_masks",
                    "require_confirmation_improvement"):
            with self.subTest(key=key):
                manifest = manifest_fixture()
                del manifest["protocol"][key]
                with self.assertRaises((KeyError, ValueError)):
                    protocol_settings(manifest)

    def test_search_window_and_candidate_limits(self):
        cases = {
            "search_windows": [0, 1, 17, 3.5, True],
            "confirmation_windows": [0, 1, 33, 3.5, True],
            "max_candidates": [0, -1, 17, 2.5, True],
            "data_sampling_seed": [-1, 1.5, True],
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    manifest = manifest_fixture()
                    manifest["protocol"][key] = value
                    with self.assertRaises(ValueError):
                        protocol_settings(manifest)

    def test_locality_and_cost_caps_cannot_be_disabled_or_expanded(self):
        maxima = {
            "operator_cost_fraction_cap": 0.02,
            "bucket_cost_fraction_cap": 0.05,
            "changed_group_fraction_cap": 0.05,
            "changed_mac_fraction_cap": 0.05,
            "cost_tolerance_fraction": 0.01,
        }
        for key, maximum in maxima.items():
            for value in (0, -0.001, maximum * 1.01, math.nan, math.inf):
                with self.subTest(key=key, value=value):
                    manifest = manifest_fixture()
                    manifest["protocol"][key] = value
                    with self.assertRaises(ValueError):
                        protocol_settings(manifest)

    def test_transfer_schedule_rejects_coarse_or_nonfinite_moves(self):
        for values in ([], [0], [-0.001], [0.02], [math.nan], [math.inf],
                       [0.0025, 0.0051]):
            with self.subTest(values=values):
                manifest = manifest_fixture()
                manifest["protocol"]["transfer_fractions"] = values
                with self.assertRaises(ValueError):
                    protocol_settings(manifest)

    def test_confirmation_threshold_cannot_be_weakened(self):
        for value in (-1.49, 0, 1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                manifest = manifest_fixture()
                manifest["protocol"]["confirmation_z_threshold"] = value
                with self.assertRaises(ValueError):
                    protocol_settings(manifest)


class HistoricalWindowExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, name, data):
        path = self.directory / name
        path.write_text(json.dumps(data))
        return str(path)

    def test_round3_search_confirmation_and_historical_windows_are_excluded(self):
        ctx = 2048
        prior = self.write("round3_windows.json", {
            "starts": {"search": [0, 5 * ctx], "confirm": [9 * ctx]},
        })
        historical = self.write("old_heldout.json", {"windows": [ctx // 2, 12 * ctx]})
        cell = {
            "excluded_windows_files": [prior],
            "historical_calibration_globs": [str(self.directory / "old_*.json")],
        }
        excluded, provenance = load_excluded_windows(cell)
        self.assertEqual(excluded, {0, ctx // 2, 5 * ctx, 9 * ctx, 12 * ctx})
        self.assertTrue(all(isinstance(item, dict) for item in provenance))
        provenance_text = json.dumps(provenance)
        self.assertIn(prior, provenance_text)
        self.assertIn(historical, provenance_text)
        search, confirm = choose_disjoint_starts(40 * ctx, ctx, 6, 16, excluded, 1729)
        self.assertFalse(set(search) & set(confirm))
        for start in search + confirm:
            for old in excluded:
                self.assertTrue(start + ctx <= old or old + ctx <= start)
        # The unaligned old interval overlaps BOTH aligned blocks 0 and 2048.
        self.assertNotIn(ctx, search + confirm)

    def test_missing_explicit_file_fails_even_if_optional_glob_has_no_matches(self):
        cell = {
            "excluded_windows_files": [str(self.directory / "missing.json")],
            "historical_calibration_globs": [str(self.directory / "nothing_*.json")],
        }
        with self.assertRaises((FileNotFoundError, ValueError)):
            load_excluded_windows(cell)

    def test_unmatched_historical_glob_is_allowed_with_valid_explicit_file(self):
        path = self.write("round3.json", {"starts": {"search": [0], "confirm": [2048]}})
        excluded, _ = load_excluded_windows({
            "excluded_windows_files": [path],
            "historical_calibration_globs": [str(self.directory / "absent_*.json")],
        })
        self.assertEqual(excluded, {0, 2048})

    def test_duplicate_start_values_are_deduplicated(self):
        path = self.write("round3.json", {"starts": {"search": [0, 2048], "confirm": [4096]}})
        self.write("historical.json", {"windows": [0, 4096, 6144]})
        excluded, _ = load_excluded_windows({
            "excluded_windows_files": [path, path],
            "historical_calibration_globs": [str(self.directory / "historical.json")],
        })
        self.assertEqual(excluded, {0, 2048, 4096, 6144})

    def test_malformed_explicit_start_schema_fails_closed(self):
        payloads = [
            {}, {"starts": []}, {"starts": {"search": [0]}},
            {"starts": {"search": "0", "confirm": [2048]}},
            {"starts": {"search": [0], "confirm": None}},
        ]
        for invalid in (-1, 1.5, "2048", True, None):
            payloads.append({"starts": {"search": [invalid], "confirm": [4096]}})
        for payload in payloads:
            with self.subTest(payload=payload):
                path = self.write("bad.json", payload)
                with self.assertRaises(ValueError):
                    load_excluded_windows({"excluded_windows_files": [path]})

    def test_invalid_historical_start_values_fail_closed(self):
        prior = self.write("round3.json", {"starts": {"search": [0], "confirm": [2048]}})
        for windows in ("4096", [True], [-1], [1.5], [None]):
            with self.subTest(windows=windows):
                old = self.write("old.json", {"windows": windows})
                with self.assertRaises(ValueError):
                    load_excluded_windows({
                        "excluded_windows_files": [prior],
                        "historical_calibration_globs": [old],
                    })


class SearchAndConfirmationTests(unittest.TestCase):
    def result(self, mean=-0.001, cost=True, locality=True):
        return {"cost_feasible": cost, "locality_feasible": locality,
                "paired_vs_incumbent": {"mean_dnll": mean}}

    def test_negative_search_loss_requires_both_guards(self):
        self.assertTrue(search_eligible(self.result()))
        for cost, locality in ((False, True), (True, False), (False, False)):
            self.assertFalse(search_eligible(self.result(cost=cost, locality=locality)))

    def test_no_worse_on_search_candidate_is_forced_to_confirmation_or_test(self):
        # Round 3 sent its least-bad new dense table to full TEST. This pilot
        # must instead stop when every new search result loses or ties.
        candidates = [self.result(0.0054324), self.result(0.0222885), self.result(0)]
        self.assertEqual([r for r in candidates if search_eligible(r)], [])

    def test_invalid_or_missing_search_effect_fails_closed(self):
        for mean in (math.nan, math.inf, -math.inf, None):
            with self.subTest(mean=mean):
                self.assertFalse(search_eligible(self.result(mean)))
        for result in ({}, {"cost_feasible": True, "locality_feasible": True},
                       {"paired_vs_incumbent": {"mean_dnll": -0.1}}):
            with self.subTest(result=result):
                self.assertFalse(search_eligible(result))

    def test_confirmation_requires_negative_mean_and_strict_z_threshold(self):
        self.assertTrue(confirmation_qualifies({"mean_dnll": -0.005, "z": -1.5001}, True, True))
        for mean, z in ((-0.005, -1.5), (-0.005, -1.49), (0, -2), (0.005, -2)):
            with self.subTest(mean=mean, z=z):
                self.assertFalse(confirmation_qualifies({"mean_dnll": mean, "z": z}, True, True))

    def test_confirmation_cost_and_locality_each_block_a_loss_win(self):
        stats = {"mean_dnll": -0.01, "z": -4.0}
        for cost, locality in ((False, True), (True, False), (False, False)):
            self.assertFalse(confirmation_qualifies(stats, cost, locality))

    def test_confirmation_rejects_nonfinite_or_undefined_effects(self):
        for key in ("mean_dnll", "z"):
            for invalid in (math.nan, math.inf, -math.inf, None):
                with self.subTest(key=key, invalid=invalid):
                    stats = {"mean_dnll": -0.01, "z": -2}
                    stats[key] = invalid
                    self.assertFalse(confirmation_qualifies(stats, True, True))
        self.assertFalse(confirmation_qualifies({}, True, True))

    def test_explicit_confirmation_threshold_is_honored(self):
        stats = {"mean_dnll": -0.01, "z": -1.8}
        self.assertTrue(confirmation_qualifies(stats, True, True))
        self.assertFalse(confirmation_qualifies(stats, True, True, z_threshold=-2.0))


if __name__ == "__main__":
    unittest.main()
