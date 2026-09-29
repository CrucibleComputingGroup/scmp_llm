import unittest
from pathlib import Path

from benchmark.ppl.mp_joint_refine import (
    _parse_targets,
    make_joint_table,
    profile_level_weights,
    project_levels_to_budget,
    shift_threshold_group,
)
from benchmark.ppl.mp_ladder_refine import weighted_cost


def _profile(hist):
    return {
        "bins": len(hist),
        "groups": {
            "gate_proj:t0:l0": {
                "op": "gate_proj",
                "mac_weighted_hist": hist,
            },
        },
    }


class BudgetProjectionTest(unittest.TestCase):
    def test_slurm_safe_target_separator(self):
        got = _parse_targets(
            "parent:nominal:40", parent_cost=30.7, nominal_cost=32.0)
        self.assertEqual(got, [30.7, 32.0, 40.0])

    def test_under_budget_spends_on_floor_first(self):
        levels = [12, 8, 4]
        weights = [0.2, 0.3, 0.5]
        target = weighted_cost([12, 8, 8], weights)
        got = project_levels_to_budget(
            levels, weights, target_cost=target, max_level=16)
        self.assertEqual(got["moves"][0]["index"], 2)
        self.assertGreater(got["levels"][-1], levels[-1])
        self.assertAlmostEqual(got["predicted_cost"], target)

    def test_over_budget_removes_from_high_rung_first(self):
        got = project_levels_to_budget(
            [12, 8, 4], [0.5, 0.3, 0.2], target_cost=7.5,
            max_level=16)
        self.assertEqual(got["moves"][0]["index"], 0)
        self.assertLess(got["levels"][0], 12)
        self.assertAlmostEqual(got["predicted_cost"], 7.5)
        self.assertTrue(all(
            got["levels"][i] > got["levels"][i + 1]
            for i in range(len(got["levels"]) - 1)))


class ThresholdMoveTest(unittest.TestCase):
    def setUp(self):
        self.table = {
            "stoc_len_levels": [12, 8, 6, 4, 2],
            "operator_defaults": {
                "gate_proj": {
                    "thresholds": [1.0, 1.0, 1.0, 1.0],
                    "counts": [0, 0, 0, 0, 100],
                },
            },
            "buckets": {
                "gate_proj:t0:l0": {
                    "thresholds": [1.0, 1.0, 1.0, 1.0],
                    "counts": [0, 0, 0, 0, 100],
                },
            },
        }
        # Uniform mass on normalized metric grid {0, .25, .5, .75, 1}.
        self.profile = _profile([20.0] * 5)

    def test_profile_replays_threshold_assignment(self):
        got = profile_level_weights(
            self.table, self.profile, protected_weight=0.1)
        self.assertAlmostEqual(got[0], 0.18)
        self.assertAlmostEqual(got[-1], 0.72)
        self.assertAlmostEqual(sum(got), 0.9)

    def test_bottom_boundary_promotes_profiled_mass(self):
        got, moves = shift_threshold_group(
            self.table,
            self.profile,
            ops=["gate_proj"],
            boundary=-1,
            mass_delta=0.20,
        )
        thresholds = got["buckets"]["gate_proj:t0:l0"]["thresholds"]
        self.assertEqual(thresholds[:3], [1.0, 1.0, 1.0])
        self.assertEqual(thresholds[-1], 0.75)
        self.assertTrue(any(move["changed"] for move in moves))
        replay = profile_level_weights(
            got, self.profile, protected_weight=0.0)
        self.assertAlmostEqual(replay[0], 0.2)
        self.assertAlmostEqual(replay[3], 0.2)
        self.assertAlmostEqual(replay[4], 0.6)


class JointTableTest(unittest.TestCase):
    def test_changes_levels_but_preserves_external_decisions(self):
        source = {
            "stoc_len_levels": [12, 8, 4],
            "method": "act_global_v3",
            "dispatch_metrics": {"gate_proj": {"metric": "l2", "sign": -1}},
            "protected_channels": {"stoc_len": 16, "indices": {"gate_proj:b0": [1]}},
            "operator_defaults": {
                "gate_proj": {
                    "thresholds": [0.8, 0.3],
                    "counts": [2, 3, 5],
                },
            },
            "buckets": {},
        }
        got = make_joint_table(
            source, [11, 8, 5],
            root_parent_table=Path("/tmp/parent.json"),
            target_cost=7.0,
            action={"type": "test", "predicted_cost": 7.0},
            command="unit-test",
        )
        self.assertEqual(got["stoc_len_levels"], [11, 8, 5])
        self.assertEqual(got["operator_defaults"]["gate_proj"]["thresholds"],
                         [0.8, 0.3])
        self.assertEqual(got["dispatch_metrics"], source["dispatch_metrics"])
        self.assertEqual(got["protected_channels"], source["protected_channels"])
        self.assertTrue(got["pass2_joint_refine"]["protected_channels_frozen"])


if __name__ == "__main__":
    unittest.main()
