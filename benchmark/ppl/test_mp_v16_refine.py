import math
import unittest
from pathlib import Path

from benchmark.ppl.mp_v16_refine import (
    LIFT_PSL,
    WalltimeClock,
    _solve_linear,
    build_window_partition,
    confirm_stop_check,
    confirm_veto_gate,
    fit_surrogate,
    generate_v16_candidates,
    ladder_features,
    make_v16_table,
    paired_deltas,
    pooled_sigma_w,
    propose_lift_compound,
    replication_accept_gate,
    select_advance,
    surrogate_candidates,
    surrogate_predict,
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


def _generate(table, profile, occupancy, *, phase, **kwargs):
    defaults = dict(
        target=34.0,
        root_parent_table=Path("/tmp/v9.json"),
        command="unit-test",
        phase=phase,
        threshold_mass_steps=[0.03],
        split_fractions=[0.33, 0.67],
        min_classes=3,
        max_classes=8,
        min_level=4,
        max_level=128,
        allow_removal=True,
    )
    defaults.update(kwargs)
    return generate_v16_candidates(table, profile, occupancy, **defaults)


class WindowPartitionTest(unittest.TestCase):
    def test_sets_are_disjoint_and_cover_everything(self):
        p = build_window_partition(117)
        screen = {w for s in p["subsets"] for w in s["A"] + s["B"]}
        confirm = set(p["confirm_windows"])
        holdout = set(p["holdout_windows"])
        self.assertEqual(len(screen), 32)
        self.assertEqual(len(confirm), 16)
        self.assertFalse(screen & confirm)
        self.assertFalse(screen & holdout)
        self.assertFalse(confirm & holdout)
        self.assertEqual(screen | confirm | holdout, set(range(117)))

    def test_subsets_pair_early_and_late_blocks(self):
        p = build_window_partition(117)
        for subset in p["subsets"]:
            self.assertEqual(len(subset["A"]), 4)
            self.assertEqual(len(subset["B"]), 4)
            self.assertLess(max(subset["A"]), min(subset["B"]))

    def test_deterministic(self):
        self.assertEqual(build_window_partition(113),
                         build_window_partition(113))

    def test_rejects_tiny_streams(self):
        with self.assertRaises(ValueError):
            build_window_partition(40)


class PairedStatsTest(unittest.TestCase):
    def test_paired_deltas_align_windows(self):
        cand = {3: 2.5, 7: 2.0}
        inc = {3: 2.4, 7: 2.2}
        self.assertEqual(paired_deltas(cand, inc, [7, 3]),
                         [-0.2 - 1e-18, 0.1] if False else
                         paired_deltas(cand, inc, [7, 3]))
        deltas = paired_deltas(cand, inc, [7, 3])
        self.assertAlmostEqual(deltas[0], -0.2)
        self.assertAlmostEqual(deltas[1], 0.1)

    def test_pooled_sigma_uses_fallback_when_thin(self):
        self.assertEqual(
            pooled_sigma_w([[0.1, 0.2]], fallback=0.5), 0.5)

    def test_pooled_sigma_matches_known_value(self):
        vecs = [[0.0, 0.02]] * 6  # per-vector sd around its mean: 0.01414
        got = pooled_sigma_w(vecs, min_candidates=5)
        self.assertAlmostEqual(got, math.sqrt(2 * (0.01 ** 2)), places=9)

    def test_replication_gate_accepts_on_sign_replication(self):
        # A-winner sign replicates on B and pooled A+B clears the pooled-
        # sigma threshold: max(0.002, 0.015/sqrt(8)) = 0.0053.
        good = replication_accept_gate(
            [-0.01] * 4, [-0.005] * 4,
            sigma_w=0.015, accept_floor=0.002, se_mult=1.0)
        self.assertTrue(good["sign_replicates"])
        self.assertTrue(good["accept"])
        self.assertAlmostEqual(good["threshold"],
                               -0.015 / math.sqrt(8.0))
        self.assertAlmostEqual(good["mean_ab"], -0.0075)

    def test_replication_gate_rejects_on_b_sign_flip(self):
        # A screen-lucky winner whose half-B mean is positive must reject
        # regardless of how strong half A looked.
        flipped = replication_accept_gate(
            [-0.05] * 4, [0.001] * 4,
            sigma_w=0.015, accept_floor=0.002, se_mult=1.0)
        self.assertFalse(flipped["sign_replicates"])
        self.assertFalse(flipped["accept"])

    def test_replication_gate_rejects_below_pooled_significance(self):
        # Sign replicates but the pooled mean is inside the noise floor.
        weak = replication_accept_gate(
            [-0.004] * 4, [-0.001] * 4,
            sigma_w=0.015, accept_floor=0.002, se_mult=1.0)
        self.assertTrue(weak["sign_replicates"])
        self.assertFalse(weak["accept"])

    def test_confirm_stop_check_cannot_accept(self):
        # STOP-ONLY: the returned dict must carry no accept/passes key, so
        # no caller can accept, select, or flip a decision from it.
        for deltas in ([-0.02] * 16, [0.03] * 16):
            check = confirm_stop_check(
                deltas, sigma_w=0.015, se_mult=1.0,
                regress_window_nats=0.02, regress_window_max=4)
            self.assertNotIn("accept", check)
            self.assertNotIn("passes", check)
            self.assertEqual(
                set(check), {"n", "mean", "threshold", "regressions",
                             "stop"})

    def test_confirm_stop_check_stop_semantics(self):
        # Strong improvement: never stops.
        improving = confirm_stop_check(
            [-0.02] * 16, sigma_w=0.015, se_mult=1.0,
            regress_window_nats=0.02, regress_window_max=4)
        self.assertFalse(improving["stop"])
        # Significant mean regression with many regressing windows: stops.
        regressing = confirm_stop_check(
            [0.03] * 16, sigma_w=0.015, se_mult=1.0,
            regress_window_nats=0.02, regress_window_max=4)
        self.assertTrue(regressing["stop"])
        # Significant mean but few spiky windows: does not stop.
        spiky = confirm_stop_check(
            [0.0] * 13 + [0.09] * 3, sigma_w=0.015, se_mult=1.0,
            regress_window_nats=0.02, regress_window_max=4)
        self.assertFalse(spiky["stop"])


class SelectAdvanceTest(unittest.TestCase):
    def test_quotas_then_global_fill(self):
        ranked = (
            [{"name": f"t{i}", "family": "threshold_move"} for i in range(9)]
            + [{"name": "m0", "family": "macro"},
               {"name": "v0", "family": "value_move"}]
        )
        got = select_advance(
            ranked, quotas={"threshold_move": 2, "macro": 2, "value_move": 2},
            total=5)
        names = [g["name"] for g in got]
        self.assertIn("m0", names)
        self.assertIn("v0", names)
        self.assertEqual(len(names), 5)
        self.assertEqual(names.count("t0"), 1)


class CandidateGenerationTest(unittest.TestCase):
    def setUp(self):
        self.levels = [96, 64, 48, 32, 24, 16]
        self.table = _table(self.levels, [0.9, 0.7, 0.5, 0.3, 0.1])
        self.profile = _profile([10.0] * 9)
        self.occupancy = {
            "level_weights": [0.08, 0.03, 0.02, 0.14, 0.20, 0.53],
            "protected_weight": 0.0,
        }

    def test_precision_families_present(self):
        got = _generate(
            self.table, self.profile, self.occupancy, phase="precision")
        families = {item["family"] for item in got}
        self.assertIn("value_move", families)
        self.assertIn("topology_insert", families)
        self.assertIn("topology_remove", families)
        self.assertIn("macro", families)
        self.assertNotIn("threshold_move", families)

    def test_bottom_rungs_move_fine_grained(self):
        got = _generate(
            self.table, self.profile, self.occupancy, phase="precision")
        requested = {
            (item["action"]["index"], item["action"]["requested_level"])
            for item in got if item["family"] == "value_move"}
        self.assertIn((5, 14), requested)   # floor moves by +-2
        self.assertIn((5, 18), requested)
        self.assertIn((0, 92), requested)   # top moves by +-4

    def test_macro_moves_exist_and_respect_bounds(self):
        got = _generate(
            self.table, self.profile, self.occupancy, phase="precision")
        macros = {item["action"]["macro"]
                  for item in got if item["family"] == "macro"}
        self.assertIn("floor_exchange", macros)
        self.assertIn("ceiling_compress", macros)
        self.assertIn("ceiling_raise", macros)
        for item in got:
            self.assertTrue(
                all(a > b for a, b in zip(item["levels"], item["levels"][1:])),
                item["name"])
            self.assertLessEqual(item["levels"][0], 128)

    def test_joint_pair_macro_from_previous_sweep(self):
        got = _generate(
            self.table, self.profile, self.occupancy, phase="precision",
            prev_pair_moves=[(1, 60), (3, 36)])
        names = {item["name"] for item in got}
        self.assertIn("macro_pair_i1_60_i3_36", names)
        pair = next(item for item in got
                    if item["name"] == "macro_pair_i1_60_i3_36")
        self.assertEqual(pair["levels"][1], 60)
        self.assertEqual(pair["levels"][3], 36)

    def test_threshold_phase_only_yields_threshold_moves(self):
        # The coarse 9-bin unit-test histogram cannot realize a 3% mass
        # move; use a coarse step here (production profiles have 257 bins).
        got = _generate(
            self.table, self.profile, self.occupancy, phase="threshold",
            threshold_mass_steps=[0.20])
        self.assertTrue(got)
        self.assertTrue(
            all(item["family"] == "threshold_move" for item in got))

    def test_removal_can_be_disabled(self):
        got = _generate(
            self.table, self.profile, self.occupancy, phase="precision",
            allow_removal=False)
        self.assertTrue(got)
        self.assertFalse(
            any(item["family"] == "topology_remove" for item in got))

    def test_protected_collision_is_skipped(self):
        table = _table(self.levels, [0.9, 0.7, 0.5, 0.3, 0.1])
        table["protected_channels"] = {"indices": [1], "stoc_len": 112,
                                       "global_mac_weighted_frac": 0.01}
        got = _generate(
            table, self.profile,
            {"level_weights": self.occupancy["level_weights"],
             "protected_weight": 0.028},
            phase="precision")
        for item in got:
            self.assertNotIn(112, item["levels"], item["name"])


class PcLengthFamilyTest(unittest.TestCase):
    def setUp(self):
        self.levels = [96, 64, 48, 32, 24, 16]
        self.table = _table(self.levels, [0.9, 0.7, 0.5, 0.3, 0.1])
        self.table["protected_channels"] = {
            "indices": {"down_proj@l0": [1, 2]},
            "stoc_len": 112,
            "global_mac_weighted_frac": 0.028,
        }
        self.profile = _profile([10.0] * 9)
        self.occupancy = {
            "level_weights": [0.08, 0.03, 0.02, 0.14, 0.20, 0.50],
            "protected_weight": 0.028,
        }

    def test_pc_candidates_change_table_and_carry_psl(self):
        got = _generate(
            self.table, self.profile, self.occupancy, phase="precision",
            pc_lengths=[96, 88, 80])
        pc = {item["name"]: item for item in got
              if item["family"] == "pc_length"}
        # 96 collides with an adaptive rung and must be skipped.
        self.assertNotIn("pc_len_96", pc)
        self.assertIn("pc_len_88", pc)
        self.assertIn("pc_len_80", pc)
        cand = pc["pc_len_88"]
        self.assertEqual(
            cand["table"]["protected_channels"]["stoc_len"], 88)
        self.assertEqual(cand["protected_sl"], 88)
        # Freed pin budget must be reinvested: cost projected back to target.
        self.assertLess(
            abs(cand["action"]["predicted_cost"] - 34.0), 1.0)

    def test_pc_family_absent_without_protected_channels(self):
        table = _table(self.levels, [0.9, 0.7, 0.5, 0.3, 0.1])
        got = _generate(
            table, self.profile,
            {"level_weights": self.occupancy["level_weights"],
             "protected_weight": 0.0},
            phase="precision", pc_lengths=[88, 80])
        self.assertFalse(
            any(item["family"] == "pc_length" for item in got))


class LiftCompoundTest(unittest.TestCase):
    """V17 macro family: measured-direction compound + partial variants."""

    LEVELS = [96, 64, 48, 32, 24, 16]
    THR = [0.9, 0.7, 0.5, 0.3, 0.1]
    ATTN_KEYS = tuple(f"{op}:t0:l{i}" for op in ("qk", "av")
                      for i in range(4))

    def _profile(self):
        groups = {}
        attn_hist = [0.0] * 50 + [0.1] * 51
        for key in self.ATTN_KEYS:
            groups[key] = {"op": key.split(":")[0],
                           "mac_weighted_hist": list(attn_hist)}
        groups["gate_proj:t0:l0"] = {
            "op": "gate_proj", "mac_weighted_hist": [10.0] * 101}
        return {"bins": 101, "groups": groups}

    def _table(self):
        buckets = {}
        for key in self.ATTN_KEYS:
            buckets[key] = {"thresholds": list(self.THR),
                            "counts": [2] * 6,
                            "fractions": [1.0 / 6.0] * 6}
        buckets["gate_proj:t0:l0"] = {"thresholds": list(self.THR),
                                      "counts": [2] * 6}
        return {
            "stoc_len_levels": list(self.LEVELS),
            "method": "act_global_v9",
            "operator_defaults": {
                "gate_proj": {"thresholds": list(self.THR),
                              "counts": [2] * 6}},
            "buckets": buckets,
            "protected_channels": {
                "indices": {"down_proj@l0": [1, 2]},
                "stoc_len": 112,
                "global_mac_weighted_frac": 0.03,
            },
        }

    def test_variants_are_iso_cost_psl95_ladder_frozen(self):
        from benchmark.ppl.mp_v16_refine import _expected_profile_cost
        table, profile = self._table(), self._profile()
        base = _expected_profile_cost(table, profile)
        got = propose_lift_compound(
            table, profile, protected_sl=112, protected_weight=0.03)
        names = {g[0] for g in got}
        self.assertEqual(
            names, {"lift_full", "lift_attn_only", "lift_psl_reinvest"})
        sentinel = [0.0] * 5
        for name, filled, psl, detail, moves, cost in got:
            # iso-cost in profile currency, ladder untouched, floor >= 16
            self.assertLessEqual(abs(cost - base), 0.05, name)
            self.assertEqual([int(x) for x in filled["stoc_len_levels"]],
                             self.LEVELS, name)
            self.assertGreaterEqual(
                min(int(x) for x in filled["stoc_len_levels"]), 16, name)
            if name in ("lift_full", "lift_psl_reinvest"):
                self.assertEqual(
                    filled["protected_channels"]["stoc_len"], LIFT_PSL)
                self.assertEqual(psl, LIFT_PSL)
                self.assertEqual(LIFT_PSL, 95)  # NEVER 96 (rung collision)
            else:
                self.assertEqual(
                    filled["protected_channels"]["stoc_len"], 112)
                self.assertIsNone(psl)
            if name in ("lift_full", "lift_attn_only"):
                for key in self.ATTN_KEYS:
                    payload = filled["buckets"][key]
                    self.assertEqual(
                        [float(x) for x in payload["thresholds"]],
                        sentinel, f"{name}:{key}")
                    total = sum(int(x) for x in payload["counts"])
                    self.assertEqual(payload["counts"],
                                     [total] + [0] * 5, f"{name}:{key}")

    def test_psl_variants_suppressed_on_rung_collision(self):
        table, profile = self._table(), self._profile()
        # A ladder already holding 95 would collide with the psl target:
        # only the attention-only partial may be proposed.
        table["stoc_len_levels"] = [95, 64, 48, 32, 24, 16]
        got = propose_lift_compound(
            table, profile, protected_sl=112, protected_weight=0.03)
        self.assertEqual({g[0] for g in got}, {"lift_attn_only"})

    def test_family_wired_into_candidate_generation(self):
        table, profile = self._table(), self._profile()
        occupancy = {
            "level_weights": [0.07, 0.03, 0.02, 0.15, 0.20, 0.50],
            "protected_weight": 0.03,
        }
        got = _generate(table, profile, occupancy, phase="precision")
        lifts = [item for item in got if item["family"] == "lift_compound"]
        self.assertTrue(lifts)
        for item in lifts:
            self.assertEqual(item["levels"], self.LEVELS)
            self.assertGreaterEqual(min(item["levels"]), 16)
            if item["name"] in ("lift_full", "lift_psl_reinvest"):
                self.assertEqual(item["protected_sl"], LIFT_PSL)


class SurrogateTest(unittest.TestCase):
    def _records(self, *, flip_b=False, n=48):
        # y correlates with 1/floor: low floors hurt (positive delta).
        records = []
        for i in range(n):
            floor = 12 + (i % 6) * 4  # 12..32
            levels = [96, 64, 48, max(33, floor + 1), floor]
            y = 0.4 / floor - 0.02 + ((i % 3) - 1) * 1e-4
            sweep = 1 + i // 24
            records.append({
                "sweep": sweep, "phase": "precision", "stage": "A",
                "candidate": f"c{i}", "family": "value_move",
                "levels": levels, "cost": 34.0,
                "mean_delta_a": y,
            })
            if i % 4 == 0:
                records.append({
                    "sweep": sweep, "phase": "precision", "stage": "B",
                    "candidate": f"c{i}", "family": "value_move",
                    "mean_delta_b": -y if flip_b else y,
                    "mean_delta_ab": -y if flip_b else y,
                    "passes": True, "sigma_w": 0.02,
                })
        return records

    def test_solve_linear(self):
        got = _solve_linear([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0])
        self.assertAlmostEqual(got[0], 1.0)
        self.assertAlmostEqual(got[1], 2.0)

    def test_fit_learns_and_transfers(self):
        surrogate = fit_surrogate(
            self._records(), target=34.0, min_history=40, min_transfer=0.2)
        self.assertTrue(surrogate["ok"])
        self.assertGreater(surrogate["r"], 0.5)
        low = surrogate_predict(surrogate, ladder_features(
            [96, 64, 48, 33, 12], 34.0, 34.0, None))
        high = surrogate_predict(surrogate, ladder_features(
            [96, 64, 48, 33, 32], 34.0, 34.0, None))
        self.assertGreater(low, high)  # low floor predicted worse

    def test_transfer_gate_rejects_anticorrelated_b(self):
        surrogate = fit_surrogate(
            self._records(flip_b=True), target=34.0,
            min_history=40, min_transfer=0.2)
        self.assertFalse(surrogate["ok"])

    def test_needs_history(self):
        surrogate = fit_surrogate(
            self._records(n=20), target=34.0,
            min_history=40, min_transfer=0.2)
        self.assertFalse(surrogate["ok"])

    def test_proposals_are_valid_unseen_ladders(self):
        table = _table([96, 64, 48, 32, 24, 16], [0.9, 0.7, 0.5, 0.3, 0.1])
        occupancy = {
            "level_weights": [0.08, 0.03, 0.02, 0.14, 0.20, 0.53],
            "protected_weight": 0.0,
        }
        surrogate = fit_surrogate(
            self._records(), target=34.0, min_history=40, min_transfer=0.2)
        seen = {(96, 64, 48, 32, 24, 16)}
        got = surrogate_candidates(
            table, occupancy, surrogate,
            target=34.0,
            root_parent_table=Path("/tmp/v9.json"),
            command="unit-test",
            seen_levels=seen,
            k=4,
            min_level=4,
            max_level=128,
        )
        self.assertTrue(0 < len(got) <= 4)
        tuples = {tuple(item["levels"]) for item in got}
        self.assertEqual(len(tuples), len(got))
        for item in got:
            self.assertEqual(item["family"], "surrogate")
            self.assertNotIn(tuple(item["levels"]), seen)
            self.assertTrue(all(
                a > b for a, b in zip(item["levels"], item["levels"][1:])))


class TableStampTest(unittest.TestCase):
    def test_method_stamp_is_idempotent(self):
        table = _table([96, 64, 48, 32, 24, 16], [0.9, 0.7, 0.5, 0.3, 0.1])
        once = make_v16_table(
            table, [96, 64, 48, 32, 24, 18],
            root_parent_table=Path("/tmp/v9.json"),
            target_cost=32.0,
            action={"type": "test", "predicted_cost": 32.0},
            command="unit-test")
        twice = make_v16_table(
            once, [96, 64, 48, 32, 24, 20],
            root_parent_table=Path("/tmp/v9.json"),
            target_cost=32.0,
            action={"type": "test", "predicted_cost": 32.0},
            command="unit-test")
        self.assertEqual(once["method"], "act_global_v9_ppl_v16")
        self.assertEqual(twice["method"], "act_global_v9_ppl_v16")
        self.assertIn("pass2_v16_refine", twice)
        self.assertNotIn("pass2_systematic_refine", twice)


class LiftCompoundTopologyTableTest(unittest.TestCase):
    """Regression: tables produced by topology_insert/topology_remove archive
    occupancy as pass1_counts/pass1_fractions, not counts/fractions.

    _attention_lift_table read payload["counts"] unconditionally, so the
    sweep AFTER any accepted insert/remove died with KeyError: 'counts' and
    took the whole branch with it.  Six of the sixteen cells of the
    mp_final_search_final_20260719_2130 wave were lost this way (4B t32/t40/
    t64, 14B t40/t48, llama8B t40); every surviving cell had accepted only
    pc_length / value_move / macro / lift moves, which keep "counts".

    Composed from LiftCompoundTest's fixtures rather than subclassing it, so
    the parent's counts-based assertions are not inherited.
    """

    LEVELS = LiftCompoundTest.LEVELS
    THR = LiftCompoundTest.THR
    ATTN_KEYS = LiftCompoundTest.ATTN_KEYS
    _profile = LiftCompoundTest._profile

    def _table(self):
        table = LiftCompoundTest._table(self)
        for payload in table["buckets"].values():
            if "counts" in payload:
                payload["pass1_counts"] = payload.pop("counts")
            if "fractions" in payload:
                payload["pass1_fractions"] = payload.pop("fractions")
        return table

    def test_topology_table_does_not_raise_and_keeps_pass1_naming(self):
        table, profile = self._table(), self._profile()
        got = propose_lift_compound(
            table, profile, protected_sl=112, protected_weight=0.03)
        self.assertEqual(
            {g[0] for g in got},
            {"lift_full", "lift_attn_only", "lift_psl_reinvest"})
        for name, filled, _psl, _detail, _moves, _cost in got:
            if name not in ("lift_full", "lift_attn_only"):
                continue          # psl-only variant never touches attention
            for key in self.ATTN_KEYS:
                payload = filled["buckets"][key]
                # occupancy restated under the table's OWN convention
                self.assertIn("pass1_counts", payload, name)
                self.assertNotIn("counts", payload, name)
                self.assertEqual(payload["pass1_counts"][0], sum([2] * 6))
                self.assertEqual(payload["pass1_counts"][1:], [0] * 5)
                self.assertEqual(payload["pass1_fractions"][0], 1.0)

    def test_missing_occupancy_disables_the_lift_without_raising(self):
        table, profile = self._table(), self._profile()
        for payload in table["buckets"].values():
            payload.pop("pass1_counts", None)
        # attention lift is impossible without occupancy, but the psl-only
        # variant does not depend on it and must still be offered
        self.assertEqual(
            {g[0] for g in propose_lift_compound(
                table, profile, protected_sl=112, protected_weight=0.03)},
            {"lift_psl_reinvest"})


class ConfirmVetoTest(unittest.TestCase):
    """V18: the confirm set may now REVERT an A->B acceptance."""

    KW = {"sigma_w": 0.015, "accept_floor": 0.002, "se_mult": 1.0}

    def test_keeps_a_move_that_holds_up_on_all_24_windows(self):
        got = confirm_veto_gate([-0.03] * 4, [-0.03] * 4, [-0.03] * 16,
                                **self.KW)
        self.assertTrue(got["keep"])
        self.assertEqual(got["n"], 24)

    def test_reverts_a_move_that_regresses_on_the_confirm_set(self):
        # accepted on A+B, then the 16 confirm windows come back worse --
        # the measured 14B t48 failure (confirm 9.9297 -> 9.9546, +0.25%).
        gate = replication_accept_gate([-0.02] * 4, [-0.02] * 4, **self.KW)
        self.assertTrue(gate["accept"], "fixture must pass the A->B gate")
        got = confirm_veto_gate([-0.02] * 4, [-0.02] * 4, [+0.02] * 16,
                                **self.KW)
        self.assertFalse(got["keep"])
        self.assertGreater(got["mean_abc"], got["threshold"])

    def test_uses_every_window_the_old_rule_used_plus_the_confirm_set(self):
        a, b, c = [-0.02] * 4, [-0.02] * 4, [-0.02] * 16
        self.assertEqual(
            confirm_veto_gate(a, b, c, **self.KW)["n"],
            replication_accept_gate(a, b, **self.KW)["n"] + len(c))

    def test_confirm_stop_check_still_cannot_accept(self):
        # the stop-only invariant must survive the veto being added
        self.assertNotIn(
            "accept",
            confirm_stop_check([0.0] * 16, sigma_w=0.015, se_mult=1.0,
                               regress_window_nats=0.01,
                               regress_window_max=4))


class WalltimeGuardTest(unittest.TestCase):
    """The guard sized sweeps from a hardcoded 50-candidate guess (288
    windows) and multiplied by 1.2; real sweeps measured 214-229 windows, so
    it demanded 21.57h of a 21.0h budget on 30B and fired before sweep 1."""

    def test_observed_sweep_supersedes_the_a_priori_estimate(self):
        clock = WalltimeClock(21.0, slack=1.10)
        self.assertEqual(clock.sweep_windows(288.0), 288.0)
        clock.begin_sweep()
        clock.note(100.0, 200)
        clock.end_sweep()
        self.assertEqual(clock.sweep_windows(288.0), 200.0)

    def test_guard_uses_the_worst_observed_sweep(self):
        clock = WalltimeClock(21.0)
        for windows in (200, 260, 230):
            clock.begin_sweep()
            clock.note(10.0, windows)
            clock.end_sweep()
        self.assertEqual(clock.sweep_windows(288.0), 260.0)

    def test_slack_is_configurable_and_applied(self):
        clock = WalltimeClock(21.0, slack=1.10)
        clock.note(121.0, 1)
        loose = WalltimeClock(21.0, slack=1.20)
        loose.note(121.0, 1)
        self.assertLess(clock.estimate(100), loose.estimate(100))


if __name__ == "__main__":
    unittest.main()
