import copy
import unittest

from benchmark.ppl.mp_joint_refine import OP_GROUPS
from benchmark.ppl.mp_v16_refine import (
    _expected_profile_cost,
    build_window_partition,
    propose_structured_transfers,
    select_advance,
)


class WindowExclusionTest(unittest.TestCase):
    def test_old_search_blocks_are_excluded_from_new_partition(self):
        old = build_window_partition(117)
        self.assertEqual(
            old["screen_block_ids"], [1, 5, 9, 12, 16, 19, 23, 27])
        self.assertEqual(old["confirm_block_ids"], [3, 10, 18, 25])
        excluded = set(old["screen_block_ids"] + old["confirm_block_ids"])

        new = build_window_partition(117, excluded_block_ids=excluded)
        self.assertEqual(
            new["screen_block_ids"], [2, 6, 8, 13, 15, 20, 22, 26])
        self.assertEqual(new["confirm_block_ids"], [4, 11, 17, 24])
        self.assertEqual(new["excluded_block_ids"], sorted(excluded))

        assigned = set(new["screen_block_ids"] + new["confirm_block_ids"])
        holdout_blocks = {window // new["block"]
                          for window in new["holdout_windows"]}
        self.assertFalse(excluded & assigned)
        self.assertFalse(excluded & holdout_blocks)
        self.assertEqual(holdout_blocks, {0, 7, 14, 21, 28, 29})

    def test_exclusion_manifest_must_leave_enough_whole_blocks(self):
        # 117 windows contain 29 full four-window blocks.  Excluding 18
        # leaves only 11, one short of the 8 screen + 4 confirm requirement.
        with self.assertRaisesRegex(ValueError, r"11 eligible blocks.*need >= 12"):
            build_window_partition(117, excluded_block_ids=set(range(18)))


class GlobalBestSelectionTest(unittest.TestCase):
    def test_family_quota_cannot_displace_unlisted_global_best(self):
        ranked = [
            {"name": "group_best", "family": "group_value", "score": -0.012},
            {"name": "pc_second", "family": "pc_length", "score": -0.009},
            {"name": "macro_third", "family": "macro", "score": -0.007},
        ]
        got = select_advance(
            ranked, quotas={"pc_length": 1, "macro": 1}, total=2)
        self.assertIs(got[0], ranked[0])
        self.assertEqual([item["name"] for item in got],
                         ["group_best", "pc_second"])


def _structured_fixture():
    """A 4-layer-bucket table shaped like the deployed metric profiles."""
    levels = [96, 64, 48, 32, 24, 16]
    thresholds = [0.90, 0.75, 0.60, 0.45, 0.30]
    mlp_ops = tuple(OP_GROUPS["mlp"])
    ops = mlp_ops + ("q_proj",)
    op_scales = {
        "gate_proj": 1.00,
        "up_proj": 1.10,
        "down_proj": 0.90,
        "q_proj": 0.35,
    }
    # Smooth centre-heavy normalized-metric distribution at production's
    # 101-bin scale.  Operator and layer factors mimic unequal MAC shares.
    base_hist = [
        1.0 + 4.0 * (1.0 - abs(2.0 * i / 100.0 - 1.0))
        for i in range(101)
    ]
    buckets = {}
    groups = {}
    defaults = {}
    for op in ops:
        defaults[op] = {
            "thresholds": list(thresholds),
            "counts": [10] * len(levels),
        }
        for layer_bucket in range(4):
            key = f"{op}:t0:l{layer_bucket}"
            buckets[key] = {
                "thresholds": list(thresholds),
                "counts": [10] * len(levels),
                "fractions": [1.0 / len(levels)] * len(levels),
            }
            scale = op_scales[op] * (1.0 + 0.03 * layer_bucket)
            groups[key] = {
                "op": op,
                "l_bucket": layer_bucket,
                "mac_weighted_hist": [value * scale for value in base_hist],
            }
    table = {
        "method": "act_global_v9",
        "stoc_len_levels": levels,
        "operator_defaults": defaults,
        "buckets": buckets,
        "protected_channels": {
            "indices": {"down_proj@l0": [1, 2]},
            "stoc_len": 80,
            "global_mac_weighted_frac": 0.03,
        },
    }
    return table, {"bins": len(base_hist), "groups": groups}


class StructuredTransferTest(unittest.TestCase):
    def test_late_mlp_transfer_is_local_ordered_and_iso_cost(self):
        table, profile = _structured_fixture()
        original = copy.deepcopy(table)
        hard_tol = 0.08
        proposals = propose_structured_transfers(
            table, profile, mass_steps=[0.15], hard_tol=hard_tol)
        by_name = {item[0]: item for item in proposals}
        self.assertIn("xfer_mlp_late_from_early_m15", by_name)

        _, candidate, detail, moves, predicted_cost = by_name[
            "xfer_mlp_late_from_early_m15"]
        self.assertEqual(table, original, "proposal generation mutated parent")
        self.assertEqual(candidate["stoc_len_levels"],
                         original["stoc_len_levels"])
        self.assertEqual(candidate["protected_channels"],
                         original["protected_channels"])
        self.assertEqual(detail["scheme"], "mlp_late_from_early")

        mlp_ops = set(OP_GROUPS["mlp"])
        intended = {
            key for key, group in profile["groups"].items()
            if group["op"] in mlp_ops and group["l_bucket"] in (0, 1, 2, 3)
        }
        changed = {
            key for key in original["buckets"]
            if candidate["buckets"][key]["thresholds"]
            != original["buckets"][key]["thresholds"]
        }
        self.assertEqual(changed, intended)
        self.assertTrue(all(move["group"] in intended for move in moves))
        self.assertTrue(all(move["changed"] for move in moves))

        for payload in candidate["buckets"].values():
            ts = payload["thresholds"]
            self.assertTrue(
                all(left >= right for left, right in zip(ts, ts[1:])), ts)

        base_cost = _expected_profile_cost(original, profile)
        replayed_cost = _expected_profile_cost(candidate, profile)
        self.assertAlmostEqual(predicted_cost, replayed_cost, places=12)
        self.assertLessEqual(abs(predicted_cost - base_cost), hard_tol)
        self.assertLessEqual(
            abs(detail["predicted_budget_error"]), hard_tol)


if __name__ == "__main__":
    unittest.main()
