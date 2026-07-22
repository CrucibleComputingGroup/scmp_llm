import unittest
from pathlib import Path

from benchmark.ppl.mp_systematic_refine import (
    _insertion_values,
    generate_candidates,
    insert_precision_rung,
    make_systematic_table,
    project_levels_constrained,
    remove_precision_rung,
)


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


def _table(levels, thresholds):
    return {
        "stoc_len_levels": levels,
        "method": "act_global_v9",
        "operator_defaults": {
            "gate_proj": {"thresholds": thresholds, "counts": [1] * len(levels)},
        },
        "buckets": {
            "gate_proj:t0:l0": {
                "thresholds": thresholds,
                "counts": [1] * len(levels),
            },
        },
        "protected_channels": {},
    }


class ConstrainedProjectionTest(unittest.TestCase):
    def test_preserves_locked_inserted_level(self):
        got = project_levels_constrained(
            [128, 64, 16], [0.1, 0.2, 0.7],
            target_cost=32.0, locked={1: 64}, min_level=4, max_level=128)
        self.assertEqual(got["levels"][1], 64)
        self.assertTrue(got["levels"][0] > 64 > got["levels"][2])
        self.assertLess(abs(got["predicted_cost"] - 32.0), 0.71)


class TopologyTest(unittest.TestCase):
    def setUp(self):
        self.table = _table([128, 16], [0.75])
        self.profile = _profile([10.0] * 9)

    def test_inserts_new_precision_type_and_threshold(self):
        got, _ = insert_precision_rung(
            self.table, self.profile,
            insert_idx=1, new_level=64, promote_fraction=0.5)
        self.assertEqual(got["stoc_len_levels"], [128, 64, 16])
        self.assertEqual(len(got["buckets"]["gate_proj:t0:l0"]["thresholds"]), 2)

    def test_removes_inserted_precision_type(self):
        inserted, _ = insert_precision_rung(
            self.table, self.profile,
            insert_idx=1, new_level=64, promote_fraction=0.5)
        got, _ = remove_precision_rung(inserted, remove_idx=1, merge="down")
        self.assertEqual(got["stoc_len_levels"], [128, 16])
        self.assertEqual(len(got["operator_defaults"]["gate_proj"]["thresholds"]), 1)

    def test_candidate_grid_includes_user_example(self):
        self.assertIn(64, _insertion_values(128, 16, limit=3))

    def test_top_and_floor_endpoints_are_real_candidates(self):
        self.assertIn(128, _insertion_values(
            128, 96, limit=4, include_high=True))
        self.assertIn(4, _insertion_values(
            16, 4, limit=4, include_low=True))

    def test_inserts_at_both_endpoints(self):
        top_source = _table([96, 16], [0.75])
        top, _ = insert_precision_rung(
            top_source, self.profile,
            insert_idx=0, new_level=128, promote_fraction=0.5)
        floor, _ = insert_precision_rung(
            self.table, self.profile,
            insert_idx=2, new_level=4, promote_fraction=0.5)
        self.assertEqual(top["stoc_len_levels"], [128, 96, 16])
        self.assertEqual(floor["stoc_len_levels"], [128, 16, 4])
        self.assertEqual(len(top["buckets"]["gate_proj:t0:l0"]["thresholds"]), 2)
        self.assertEqual(len(floor["buckets"]["gate_proj:t0:l0"]["thresholds"]), 2)

    def test_topology_metadata_does_not_claim_old_counts(self):
        inserted, _ = insert_precision_rung(
            self.table, self.profile,
            insert_idx=1, new_level=64, promote_fraction=0.5)
        got = make_systematic_table(
            inserted, [128, 64, 16],
            root_parent_table=Path("/tmp/v9.json"),
            target_cost=32.0,
            action={"type": "test", "predicted_cost": 32.0},
            command="unit-test")
        payload = got["operator_defaults"]["gate_proj"]
        self.assertNotIn("counts", payload)
        self.assertEqual(payload["pass1_counts"], [1, 1])
        self.assertEqual(got["mp_levels"], "128,64,16")


class NeighborhoodTest(unittest.TestCase):
    def test_enumerates_value_topology_and_threshold_families(self):
        table = _table([96, 64, 32, 16], [0.8, 0.5, 0.2])
        profile = _profile([10.0] * 9)
        occupancy = {
            "level_weights": [0.1, 0.2, 0.3, 0.4],
            "protected_weight": 0.0,
        }
        got = generate_candidates(
            table, profile, occupancy,
            target=40.0,
            root_parent_table=Path("/tmp/v9.json"),
            command="unit-test",
            level_step=4,
            threshold_mass_steps=[0.20],
            split_fractions=[0.5],
            topology_values_per_gap=1,
            min_classes=3,
            max_classes=6,
            min_level=4,
            max_level=128,
            families={"value", "topology", "threshold"},
        )
        families = {item["family"] for item in got}
        self.assertIn("value_move", families)
        self.assertIn("topology_insert", families)
        self.assertIn("topology_remove", families)
        self.assertIn("threshold_move", families)

    def test_topology_can_be_run_in_insertion_only_mode(self):
        table = _table([96, 64, 32, 16], [0.8, 0.5, 0.2])
        profile = _profile([10.0] * 9)
        occupancy = {"level_weights": [0.1, 0.2, 0.3, 0.4],
                     "protected_weight": 0.0}
        got = generate_candidates(
            table, profile, occupancy,
            target=40.0, root_parent_table=Path("/tmp/v12.json"),
            command="unit-test", level_step=4,
            threshold_mass_steps=[0.20], split_fractions=[0.5],
            topology_values_per_gap=1, min_classes=3, max_classes=6,
            min_level=4, max_level=128, families={"topology"},
            allow_removal=False)
        self.assertTrue(got)
        self.assertTrue(all(item["family"] == "topology_insert" for item in got))

    def test_fixed_class_count_filters_topology(self):
        table = _table([96, 64, 32, 16], [0.8, 0.5, 0.2])
        profile = _profile([10.0] * 9)
        occupancy = {"level_weights": [0.1, 0.2, 0.3, 0.4],
                     "protected_weight": 0.0}
        got = generate_candidates(
            table, profile, occupancy,
            target=40.0, root_parent_table=Path("/tmp/v12.json"),
            command="unit-test", level_step=4,
            threshold_mass_steps=[0.20], split_fractions=[0.5],
            topology_values_per_gap=1, min_classes=3, max_classes=6,
            fixed_class_count=4,
            min_level=4, max_level=128,
            families={"value", "topology", "threshold"})
        self.assertTrue(got)
        self.assertTrue(all(len(item["levels"]) == 4 for item in got))


if __name__ == "__main__":
    unittest.main()
