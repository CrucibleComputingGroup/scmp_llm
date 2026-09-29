"""CPU checks for bounded adjacent-rung allocation and exact profile replay."""
from __future__ import annotations

import copy
import importlib.util
import unittest

import numpy as np

from benchmark.ppl.prc_adjacent_proposals import (
    AdjacentProfileCollector, _indices, _universal_displacement,
    audit_transition, baseline_cost, propose,
)
from benchmark.ppl.prc_local_proposals import _entry, _group_cost, _lengths
from benchmark.ppl.test_prc_local_proposals import fixture as original_fixture


def fixture():
    table, profile = original_fixture()
    for group in profile["groups"].values():
        group["count"] = [1.] * len(group["mac"])
    return table, profile


def refresh_cost(profile):
    profile["total_macs"] = profile["fixed_macs"] + sum(
        sum(g["mac"]) for g in profile["groups"].values())
    profile["total_cycle_macs"] = profile["fixed_cycle_macs"] + sum(
        _group_cost(g, g["thresholds"]) for g in profile["groups"].values())


class AdjacentProposalTests(unittest.TestCase):
    def test_complete_proposals_cost_caps_and_only_two_boundary_edits(self):
        table, profile = fixture()
        original = copy.deepcopy(table)
        proposals = propose(table, profile)
        self.assertEqual(len(proposals), 12)
        self.assertEqual(table, original)
        for proposal in proposals:
            diagnostic = audit_transition(table, proposal["table"], profile)
            self.assertTrue(diagnostic["feasible"], diagnostic)
            self.assertEqual(diagnostic["edited_buckets"], 2)
            self.assertEqual(diagnostic["max_rung_displacement"], 1)
            self.assertLessEqual(abs(diagnostic["predicted_cost_ratio"] - 1), 1.00001e-4)
            self.assertLessEqual(diagnostic["transferred_fraction"], .005000001)
            normalized = copy.deepcopy(proposal["table"])
            for group in profile["groups"].values():
                _entry(normalized, group)["thresholds"] = copy.deepcopy(_entry(table, group)["thresholds"])
            self.assertEqual(normalized, table)
            for move in diagnostic["moves"]:
                self.assertLessEqual(move["changed_group_fraction"], .040000001)
                self.assertLessEqual(move["changed_mac_fraction"], .040000001)
                self.assertLessEqual(move["bucket_cost_fraction"], .040000001)
            self.assertTrue(all(x["gross_cost_fraction"] <= .016000001
                                for x in diagnostic["operators"].values()))

    def test_coincident_qk_boundaries_demote_only_96_to_74(self):
        table, profile = fixture()
        group = profile["groups"]["attention|qk:t0:l0"]
        group.update(levels=[96, 74, 48, 32, 24, 19], thresholds=[0.] * 5)
        _entry(table, group).update(stoc_len_levels=group["levels"], thresholds=group["thresholds"][:])
        refresh_cost(profile)
        self.assertEqual(_universal_displacement(group, [0.] * 5, [.321533] * 5), 5)
        proposals = propose(table, profile)
        touched = 0
        for proposal in proposals:
            new = _entry(proposal["table"], group)["thresholds"]
            if new == group["thresholds"]:
                continue
            touched += 1
            self.assertGreater(new[0], 0)
            self.assertEqual(new[1:], [0.] * 4)
            self.assertEqual(set(_lengths(group, new)), {96., 74.})
            self.assertEqual(_universal_displacement(group, group["thresholds"], new), 1)
        self.assertGreater(touched, 0)

    def test_float32_ties_and_endpoints_match_runtime(self):
        group = dict(kind="prc", key="q_proj:t0:l0", levels=[8, 16, 32],
                     metric=[0, .25, np.nextafter(np.float32(.25), np.float32(1)), .75, 1])
        self.assertEqual(_lengths(group, [.25, .75]).tolist(), [8, 8, 16, 16, 32])
        self.assertEqual(_universal_displacement(group, [0, 0], [0, .25]), 1)
        self.assertEqual(_universal_displacement(group, [1, 1], [.75, 1]), 1)
        self.assertEqual(_universal_displacement(group, [0, 0], [.25, .25]), 2)
        group.update(kind="attention", levels=[32, 16, 8])
        self.assertEqual(_lengths(group, [.75, .25]).tolist(), [8, 16, 16, 32, 32])
        self.assertEqual(_universal_displacement(group, [0, 0], [.25, 0]), 1)
        self.assertEqual(_universal_displacement(group, [1, 1], [1, .75]), 1)
        self.assertEqual(_universal_displacement(group, [0, 0], [.25, .25]), 2)

    def test_group_counts_are_not_inferred_from_mac_weights(self):
        table, profile = fixture()
        proposal = propose(table, profile, max_candidates=1)[0]
        move = proposal["diagnostics"]["moves"][0]
        group = profile["groups"][move["tag"]]
        new = _entry(proposal["table"], group)["thresholds"]
        changed = _indices(group, group["thresholds"]) != _indices(group, new)
        index = int(np.flatnonzero(changed)[0])
        group["count"][index] = 1_000_000.
        audit = audit_transition(table, proposal["table"], profile)
        self.assertFalse(audit["feasible"])
        self.assertTrue(any("group fraction exceeds" in r for r in audit["reasons"]))
        detail = next(d for d in audit["moves"] if d["tag"] == move["tag"])
        self.assertGreater(detail["changed_group_fraction"], .99)
        self.assertLessEqual(detail["changed_mac_fraction"], .05)

    def test_tail_width_cost_and_missing_count_rejection(self):
        table, profile = fixture()
        tail = profile["groups"]["prc|down_proj:t0:l0"]
        self.assertEqual(sum(tail["mac"]), 4097 * 17 * 2048)
        self.assertEqual(sum(tail["count"]), 4097)
        self.assertEqual(baseline_cost(table, profile), profile["total_cycle_macs"] / profile["total_macs"])
        del tail["count"]
        with self.assertRaisesRegex(ValueError, "count/MAC histogram"):
            propose(table, profile)

    def test_fields_and_multi_boundary_changes_rejected(self):
        table, profile = fixture()
        candidate = propose(table, profile, max_candidates=1)[0]["table"]
        bad = copy.deepcopy(candidate)
        bad["protected_channel_stoc_len"] += 1
        self.assertFalse(audit_transition(table, bad, profile)["feasible"])
        bad = copy.deepcopy(candidate)
        bucket = bad["per_row_chunk"]["buckets"]["q_proj:t0:l0"]
        bucket["thresholds"] = [.30, .80]
        self.assertTrue(any("one boundary" in reason for reason in
                            audit_transition(table, bad, profile)["reasons"]))

    def test_operator_gross_cap_cannot_cancel(self):
        table, profile = fixture()
        source = profile["groups"]["prc|q_proj:t0:l0"]
        extra = copy.deepcopy(source)
        extra["key"] = "q_proj:t0:l1"
        profile["groups"]["prc|q_proj:t0:l1"] = extra
        table["per_row_chunk"]["buckets"][extra["key"]] = copy.deepcopy(_entry(table, source))
        refresh_cost(profile)
        proposals = propose(table, profile, max_candidates=32)
        same_op = next(p for p in proposals if p["diagnostics"]["category"] == "within_q_proj")
        detail = same_op["diagnostics"]["operators"]["q_proj"]
        self.assertAlmostEqual(detail["net_cost_fraction"], 0, places=4)
        self.assertGreater(detail["gross_cost_fraction"], 0)
        audit = audit_transition(table, same_op["table"], profile,
                                 max_operator_cost_fraction=detail["gross_cost_fraction"] / 2)
        self.assertFalse(audit["feasible"])
        self.assertTrue(any("operator gross" in reason for reason in audit["reasons"]))

    def test_threshold_edges_required_for_exact_cross_replay(self):
        table, profile = fixture()
        candidate = propose(table, profile, max_candidates=1)[0]["table"]
        audit = audit_transition(table, candidate, profile)
        move = audit["moves"][0]
        group = profile["groups"][move["tag"]]
        group["edges"] = [x for x in group["edges"]
                          if np.float32(x) != np.float32(move["new_threshold"])]
        result = audit_transition(table, candidate, profile)
        self.assertFalse(result["feasible"])
        self.assertTrue(any("threshold edges" in reason for reason in result["reasons"]))

    def test_baseline_mismatch_rejected_and_deterministic(self):
        table, profile = fixture()
        first = propose(table, profile, max_candidates=2)
        self.assertEqual(first, propose(table, profile, max_candidates=2))
        profile["total_cycle_macs"] *= 1.01
        with self.assertRaisesRegex(ValueError, "does not replay"):
            propose(table, profile)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch not installed")
    def test_collector_counts_and_reference_edge_union(self):
        import torch
        from types import SimpleNamespace
        for kind, levels, old, new in (("prc", [8, 16, 32], [.25, .75], [.30, .75]),
                                        ("attention", [32, 16, 8], [.75, .25], [.75, .30])):
            op = "q_proj" if kind == "prc" else "qk"
            entry = dict(thresholds=old, **{("levels" if kind == "prc" else "stoc_len_levels"): levels})
            base = {"buckets": {}, "per_row_chunk": {"buckets": {}}}
            (base["per_row_chunk"] if kind == "prc" else base)["buckets"][op + ":t0:l0"] = entry
            collector = AdjacentProfileCollector(None, bins=32, reference_table=base)
            cfg = SimpleNamespace(layer_buckets=1, prc_layer_buckets=1)
            g = collector._group(cfg, op, 0, 1, kind, levels, new, torch.device("cpu"))
            metric = torch.tensor([0., .25, .26, .30, .5, .75, 1.])
            mac = torch.tensor([128., 17., 128., 17., 128., 17., 128.])
            indices = _indices(dict(kind=kind, levels=levels), new, metric.numpy())
            actual = torch.tensor(levels)[torch.tensor(indices)]
            collector._add(g, metric, mac, actual)
            profile = collector.snapshot()
            group = next(iter(profile["groups"].values()))
            self.assertEqual(sum(group["count"]), 7)
            self.assertEqual(sum(group["mac"]), float(mac.sum()))
            self.assertIn(float(np.float32(.25)), group["edges"])
            self.assertIn(float(np.float32(.30)), group["edges"])
            old_length = np.asarray(levels)[_indices(dict(kind=kind, levels=levels), old, metric.numpy())]
            self.assertEqual(baseline_cost(base, profile), float(np.dot(mac.numpy(), old_length)) / float(mac.sum()))
            self.assertEqual(profile["total_cycle_macs"], float((mac * actual).sum()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch not installed")
    def test_observer_identity_with_actual_dispatch(self):
        # Reuse the real SCLinear/attention integration fixture; only the collector changes.
        from unittest.mock import patch
        from benchmark.ppl import test_prc_local_proposals as previous
        with patch.object(previous, "ProfileCollector", AdjacentProfileCollector):
            previous.LocalProposalTests("test_real_dispatch_cpu_integration").test_real_dispatch_cpu_integration()


if __name__ == "__main__":
    unittest.main()
