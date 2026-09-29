"""CPU tests for the round-6 attention diagnostic (arm builder, validator, trace
audit, statistics, pre-registered gate, and a simulated end-to-end driver run).

Uses archived incumbent tables and full-test traces read-only; no model, GPU or
scheduler. Run from the repo root with the experiment env:
  PYTHONPATH=kernels python -m unittest benchmark.ppl.test_prc_r6_attn_diag -v
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from benchmark.ppl import prc_r6_attn_diag_arms as R
from benchmark.ppl import prc_r6_attn_diag as D
from benchmark.ppl.prc_local_refine import paired_stats

K = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands")
MP_BEST = Path("/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best")
BEST_ALL = Path("/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.json")
MANIFEST = Path(__file__).resolve().parent / "kbands/prc_r6_attn_diag_20260928.json"
A_4B = ["qk:t0:l0", "qk:t0:l1", "qk:t0:l2", "qk:t0:l3", "av:t0:l1", "av:t0:l2", "av:t0:l3"]
A_30B = ["qk:t0:l0", "qk:t0:l1", "qk:t0:l2", "qk:t0:l3", "av:t0:l3"]
CELLS = {
    "4B_t40": (K / "prc2/4B_t40_c7_gfisla.json", K / "kbands_20260801/ppl/4B_t40_prc_p2c7gfisla_trace.json", 36, A_4B),
    "4B_t64": (K / "prc2/4B_t64_c7_gfisla.json", K / "kbands_20260801/ppl/4B_t64_prc_p2c7gfisla_trace.json", 36, A_4B),
}
_CACHE = {}


def load(cell):
    if cell not in _CACHE:
        wpath, tpath, nb, lst = CELLS[cell]
        wrapper = json.loads(wpath.read_text())
        table = json.loads(Path(wrapper["threshold_table_path"]).read_text())
        payload = json.loads(tpath.read_text())
        hybrid = json.loads((MP_BEST / f"configs/4B/target{cell.split('_t')[1]}/hybrid_config.json").read_text())
        _CACHE[cell] = dict(wrapper=wrapper, table=table, payload=payload, nb=nb, A=lst, hybrid=hybrid,
                            table_path=wrapper["threshold_table_path"])
    return _CACHE[cell]


def arm_spec(arm, lst):
    return {"A": {"buckets": list(lst)}, "ATT128": {}, "LIN128": {"protected": True}}[arm]


class BucketIndexTest(unittest.TestCase):
    def test_mirror_matches_runtime(self):
        from scmp_kernels.mp.config import _bucket_index
        for total in (1, 2, 36, 48):
            for n in (1, 4):
                for b in range(total):
                    self.assertEqual(R.bucket_index(b, total, n), _bucket_index(b, total, n))

    def test_runtime_bucket_boundaries(self):
        self.assertEqual([R.bucket_index(b, 48, 4) for b in (0, 11, 12, 23, 24, 35, 36, 47)],
                         [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual([R.bucket_index(b, 36, 4) for b in (0, 8, 9, 17, 18, 26, 27, 35)],
                         [0, 0, 1, 1, 2, 2, 3, 3])


class ArmBuilderTest(unittest.TestCase):
    def test_build_validate_and_resolve_4b(self):
        for cell in CELLS:
            c = load(cell)
            with tempfile.TemporaryDirectory() as tmp:
                for arm in R.EDIT_ARMS:
                    spec = arm_spec(arm, c["A"])
                    table = R.build_arm_table(c["table"], arm, spec, {"source_table": c["table_path"]})
                    path = Path(tmp) / f"{arm}_table.json"
                    path.write_text(json.dumps(table, indent=1))
                    rep = R.validate_arm_table(c["table"], json.loads(path.read_text()), arm, spec)
                    self.assertEqual(rep["arm"], arm)
                    sweep = R.resolution_sweep(c["wrapper"], path, arm, spec, c["table"], c["table_path"], c["nb"])
                    self.assertEqual(sweep["attention_op_blocks_checked"], 2 * c["nb"])
                    self.assertEqual(sweep["protected_stoc_len"],
                                     128 if arm == "LIN128" else c["table"]["protected_channels"]["stoc_len"])
                    if arm != "LIN128":
                        self.assertGreater(sweep["synthetic_rows_changed"], 0)

    def test_arm_edits_are_exactly_the_definition(self):
        c = load("4B_t64")
        g = R.global_ladder(c["table"])
        a = R.build_arm_table(c["table"], "A", arm_spec("A", A_4B), {})
        for key in R.attention_bucket_keys(c["table"]):
            if key in A_4B:
                self.assertEqual(a["buckets"][key]["stoc_len_levels"], [128] + g[1:])
            else:
                self.assertNotIn("stoc_len_levels", a["buckets"][key])
            self.assertEqual(a["buckets"][key]["thresholds"], c["table"]["buckets"][key]["thresholds"])
        att = R.build_arm_table(c["table"], "ATT128", {}, {})
        for key in R.attention_bucket_keys(c["table"]):
            self.assertEqual(att["buckets"][key]["thresholds"], [0.0] * (len(g) - 1))
        lin = R.build_arm_table(c["table"], "LIN128", {"protected": True}, {})
        self.assertTrue(all(e["levels"] == [128] * 17 for e in lin["per_row_chunk"]["buckets"].values()))
        self.assertEqual(lin["protected_channels"]["stoc_len"], 128)
        self.assertEqual(lin["stoc_len_levels"], c["table"]["stoc_len_levels"])

    def test_30b_arm_a_resolves(self):
        wpath = K / "prc_local_20260926/30B_t40/candidate07_mlp_from_projections.json"
        wrapper = json.loads(wpath.read_text())
        base = json.loads(Path(wrapper["threshold_table_path"]).read_text())
        table = R.build_arm_table(base, "A", {"buckets": A_30B}, {})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "A_table.json"
            path.write_text(json.dumps(table))
            sweep = R.resolution_sweep(wrapper, path, "A", {"buckets": A_30B}, base,
                                       wrapper["threshold_table_path"], 48)
        self.assertEqual(sweep["attention_op_blocks_checked"], 96)


class ValidatorRejectionTest(unittest.TestCase):
    def setUp(self):
        c = load("4B_t40")
        self.base = c["table"]
        self.g = R.global_ladder(self.base)
        self.a = R.build_arm_table(self.base, "A", {"buckets": A_4B}, {})
        self.att = R.build_arm_table(self.base, "ATT128", {}, {})
        self.lin = R.build_arm_table(self.base, "LIN128", {"protected": True}, {})

    def reject(self, cand, arm, spec, msg=None):
        with self.assertRaises(ValueError) as ctx:
            R.validate_arm_table(self.base, cand, arm, spec)
        if msg:
            self.assertIn(msg, str(ctx.exception))

    def test_unlisted_bucket_edited(self):
        cand = copy.deepcopy(self.a)
        cand["buckets"]["av:t0:l0"]["stoc_len_levels"] = [128] + self.g[1:]
        self.reject(cand, "A", {"buckets": A_4B}, "unlisted bucket")

    def test_listed_bucket_not_edited(self):
        cand = copy.deepcopy(self.a)
        del cand["buckets"]["qk:t0:l2"]["stoc_len_levels"]
        self.reject(cand, "A", {"buckets": A_4B})

    def test_top_above_cap(self):
        cand = copy.deepcopy(self.a)
        cand["buckets"]["qk:t0:l0"]["stoc_len_levels"] = [129] + self.g[1:]
        self.reject(cand, "A", {"buckets": A_4B})

    def test_non_descending_ladder(self):
        cand = copy.deepcopy(self.a)
        cand["buckets"]["qk:t0:l0"]["stoc_len_levels"] = [128, 64, 64] + self.g[3:]
        self.reject(cand, "A", {"buckets": A_4B})

    def test_global_ladder_edit(self):
        cand = copy.deepcopy(self.a)
        cand["stoc_len_levels"] = [128] + self.g[1:]
        self.reject(cand, "A", {"buckets": A_4B}, "top-level")

    def test_threshold_edit_in_a(self):
        cand = copy.deepcopy(self.a)
        self.assertGreater(cand["buckets"]["qk:t0:l1"]["thresholds"][-1], 0.0)
        cand["buckets"]["qk:t0:l1"]["thresholds"][-1] = 0.0  # still monotone and in range
        self.reject(cand, "A", {"buckets": A_4B}, "beyond the arm's edits")
        cand = copy.deepcopy(self.a)
        cand["buckets"]["qk:t0:l1"]["thresholds"][0] *= 0.5  # breaks monotonicity
        self.reject(cand, "A", {"buckets": A_4B}, "invalid thresholds")

    def test_escape_constant_edit(self):
        cand = copy.deepcopy(self.a)
        cand["buckets"]["qk:t0:l1"]["metric_std"] = 0.0
        self.reject(cand, "A", {"buckets": A_4B}, "beyond the arm's edits")

    def test_psl_change_in_a_and_att128(self):
        for arm, table, spec in (("A", self.a, {"buckets": A_4B}), ("ATT128", self.att, {})):
            cand = copy.deepcopy(table)
            cand["protected_channels"]["stoc_len"] = 128
            self.reject(cand, arm, spec, "beyond the arm's edits")

    def test_prc_edit_in_a(self):
        cand = copy.deepcopy(self.a)
        entry = cand["per_row_chunk"]["buckets"]["down_proj:t0:l0"]
        entry["levels"] = [128] * len(entry["levels"])
        self.reject(cand, "A", {"buckets": A_4B}, "beyond the arm's edits")

    def test_att128_nonzero_threshold(self):
        cand = copy.deepcopy(self.att)
        cand["buckets"]["av:t0:l2"]["thresholds"][-1] = 0.01
        self.reject(cand, "ATT128", {}, "all zero")

    def test_lin128_without_psl_or_partial(self):
        cand = copy.deepcopy(self.lin)
        cand["protected_channels"]["stoc_len"] = self.base["protected_channels"]["stoc_len"]
        self.reject(cand, "LIN128", {"protected": True}, "protected")
        cand = copy.deepcopy(self.lin)
        cand["per_row_chunk"]["buckets"]["q_proj:t0:l3"]["levels"][0] = 112
        self.reject(cand, "LIN128", {"protected": True}, "all-128")

    def test_lin128_attention_edit(self):
        cand = copy.deepcopy(self.lin)
        cand["buckets"]["qk:t0:l0"]["thresholds"][0] = 0.0
        self.reject(cand, "LIN128", {"protected": True})

    def test_metadata_and_spec(self):
        cand = copy.deepcopy(self.a)
        cand[R.META_KEY]["arm"] = "ATT128"
        self.reject(cand, "A", {"buckets": A_4B}, "metadata")
        cand = copy.deepcopy(self.a)
        del cand[R.META_KEY]
        self.reject(cand, "A", {"buckets": A_4B}, "metadata")
        with self.assertRaises(ValueError):
            R.build_arm_table(self.base, "A", {"buckets": ["q_proj:t0:l0"]}, {})
        with self.assertRaises(ValueError):
            R.build_arm_table(self.base, "A", {"buckets": ["qk:t0:l0", "qk:t0:l0"]}, {})
        with self.assertRaises(ValueError):
            R.build_arm_table(self.base, "ATT128", {"x": 1}, {})
        with self.assertRaises(ValueError):
            R.build_arm_table(self.base, "LIN128", {}, {})
        with self.assertRaises(ValueError):
            R.build_arm_table(self.base, "B", {}, {})

    def test_base_must_be_an_incumbent(self):
        with self.assertRaises(ValueError):
            R.check_incumbent_shape(self.a)
        bad = copy.deepcopy(self.base)
        bad["buckets"]["qk:t0:l0"]["stoc_len_levels"] = list(self.g)
        with self.assertRaises(ValueError):
            R.check_incumbent_shape(bad)


class TraceAuditTest(unittest.TestCase):
    def audit(self, cell, arm, payload, spec=None, hybrid="default", **kw):
        c = load(cell)
        spec = arm_spec(arm, c["A"]) if spec is None else spec
        inc = R.TraceAgg(c["payload"], c["nb"], 4)
        return R.audit_arm_trace(inc, R.TraceAgg(payload, c["nb"], 4), arm, spec, c["table"],
                                 escape_len=128, hybrid=c["hybrid"] if hybrid == "default" else hybrid, **kw)

    def synth(self, cell, arm, spec=None):
        c = load(cell)
        spec = arm_spec(arm, c["A"]) if spec is None else spec
        return R.synthesize_arm_trace(c["payload"], arm, spec, c["table"], c["nb"])

    def test_intended_arms_pass(self):
        for cell in CELLS:
            for arm in R.EDIT_ARMS:
                rep = self.audit(cell, arm, self.synth(cell, arm))
                self.assertTrue(rep["ok"], (cell, arm, rep["failures"]))
                self.assertAlmostEqual(rep["extra_cycles_pct"], rep["direct_cycles_pct"] + rep["cascade_cycles_pct"])
                self.assertAlmostEqual(rep["direct_cycles_pct"], rep["predicted_first_order_direct_pct"])
                self.assertGreater(rep["first_edit_record_keys_checked"], 0)
        rep = self.audit("4B_t40", "A", self.synth("4B_t40", "A"))
        self.assertAlmostEqual(rep["extra_cycles_pct"], 9.0, delta=0.05)  # scout first-order +9.00%
        self.assertEqual(rep["first_edit_position"], [3, 1])  # first SC qk block on 4B

    def test_incumbent_trace_audits_clean(self):
        for cell in CELLS:
            c = load(cell)
            inc = R.TraceAgg(c["payload"], c["nb"], 4)
            self.assertEqual(R.audit_trace_basics(inc, "inc", c["hybrid"]), [])
            self.assertEqual(R.audit_lengths(inc, "INC", {}, c["table"], 128), [])

    def test_leftover_old_top(self):
        g0 = R.global_ladder(load("4B_t40")["table"])[0]
        syn = self.synth("4B_t40", "A")
        for grp in syn["groups"]:
            if grp["op"] == "qk" and grp["block"] == 20 and grp["stoc_len"] == 128:
                grp["stoc_len"] = g0
                break
        rep = self.audit("4B_t40", "A", syn)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("outside the allowed set" in f for f in rep["failures"]))

    def test_upstream_change_detected(self):
        syn = self.synth("4B_t40", "A")
        for grp in syn["groups"]:
            if grp["block"] == 0 and grp["op"] == "o_proj":
                grp["stoc_len"] = {8: 10}.get(grp["stoc_len"], 8)
                break
        rep = self.audit("4B_t40", "A", syn)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("upstream record changed" in f for f in rep["failures"]))

    def test_first_edit_transform_detected(self):
        syn = self.synth("4B_t40", "A")
        for grp in syn["groups"]:
            if grp["op"] == "qk" and grp["block"] == 3 and grp["stoc_len"] == 128:
                grp["stoc_len"] = 64  # an allowed rung, but not the intended map
                break
        rep = self.audit("4B_t40", "A", syn)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("intended length map" in f for f in rep["failures"]))

    def test_mac_and_numerics_changes_detected(self):
        syn = self.synth("4B_t40", "ATT128")
        syn["groups"][-1]["macs"] += 1
        self.assertFalse(self.audit("4B_t40", "ATT128", syn)["ok"])
        syn = self.synth("4B_t40", "ATT128")
        syn["groups"][-1]["rng_levels"] = 64
        rep = self.audit("4B_t40", "ATT128", syn)
        self.assertTrue(any("numerics" in f for f in rep["failures"]))

    def test_att128_and_lin128_leftovers(self):
        syn = self.synth("4B_t64", "ATT128")
        grp = next(g for g in syn["groups"] if g["op"] == "av" and g["block"] == 30)
        grp["stoc_len"] = 64
        self.assertFalse(self.audit("4B_t64", "ATT128", syn)["ok"])
        syn = self.synth("4B_t64", "LIN128")
        grp = next(g for g in syn["groups"] if g["op"] == "down_proj" and g["block"] == 30)
        grp["stoc_len"] = 96
        self.assertFalse(self.audit("4B_t64", "LIN128", syn)["ok"])

    def test_ladder_on_wrong_bucket_detected(self):
        # 4B t64 av:l0 holds ~42% of its MACs on the top rung; an A whose ladder
        # also landed on av:l0 would realize only allowed lengths (128 = escape).
        wrong = self.synth("4B_t64", "A", {"buckets": A_4B + ["av:t0:l0"]})
        rep = self.audit("4B_t64", "A", wrong)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("lost every MAC on the old top" in f for f in rep["failures"]))

    def test_mask_and_header_checks(self):
        c = load("4B_t40")
        syn = self.synth("4B_t40", "A")
        syn["groups"] = [g for g in syn["groups"] if not (g["op"] == "up_proj" and g["block"] == 20)]
        rep = self.audit("4B_t40", "A", syn)
        self.assertTrue(any("(op, block) set" in f for f in rep["failures"]))
        other = copy.deepcopy(c["hybrid"])
        other["schedule"]["qk"][3] = "int7"
        rep = self.audit("4B_t40", "A", self.synth("4B_t40", "A"), hybrid=other)
        self.assertTrue(any("hybrid INT mask" in f for f in rep["failures"]))
        rep = self.audit("4B_t40", "A", self.synth("4B_t40", "A"), expected_windows=[0, 2048])
        self.assertTrue(any("windows" in f for f in rep["failures"]))

    def test_reference_comparison(self):
        c = load("4B_t40")
        a = R.TraceAgg(c["payload"], 36, 4)
        self.assertEqual(R.compare_trace_to_reference(a, R.TraceAgg(c["payload"], 36, 4)), [])
        bumped = copy.deepcopy(c["payload"])
        bumped["groups"][5]["rows"] += 1
        self.assertTrue(R.compare_trace_to_reference(a, R.TraceAgg(bumped, 36, 4)))


class StatsAndGateTest(unittest.TestCase):
    def test_paired_matches_round4_helper(self):
        cand = [2.1, 1.9, 2.05, 2.2, 1.7]
        ref = [2.12, 1.93, 2.04, 2.25, 1.72]
        self.assertEqual(R.paired(cand, ref), paired_stats(cand, ref))

    def test_arm_metrics(self):
        m = R.arm_metrics([1.0, 1.1, 0.9, 1.0], [1.02, 1.11, 0.93, 1.01], 45.0, 40.0)
        diffs = [-0.02, -0.01, -0.03, -0.01]
        mean = sum(diffs) / 4
        self.assertAlmostEqual(m["dnll"], mean)
        self.assertAlmostEqual(m["extra_cycles_pct"], 12.5)
        self.assertAlmostEqual(m["dppl_pct"], 100 * math.expm1(mean))
        self.assertAlmostEqual(m["se_pct"], 100 * math.exp(mean) * m["se_dnll"])
        self.assertAlmostEqual(m["dppl_pct_per_1pct_extra_cycles"], m["dppl_pct"] / 12.5)

    def gate(self, model, dnll, se_pct, extra, chord, ok=True):
        metrics = {"dnll": dnll, "dppl_pct": 100 * math.expm1(dnll), "se_pct": se_pct,
                   "extra_cycles_pct": extra}
        return R.round7_gate(model, metrics, chord, ok)

    def test_gate_clauses(self):
        # 30B t40-like: chord 0.147, extra 8.6% -> budget term -0.759%
        g = self.gate("30B", -0.02, 0.3, 8.6, 0.147)
        self.assertAlmostEqual(g["threshold_dppl_pct"], -(0.6 * 0.147 * 8.6) - 1.5 * 0.3)
        self.assertTrue(g["proceed_round7_attention_ladder_solve"])
        self.assertFalse(self.gate("30B", -0.011, 0.3, 8.6, 0.147)["cond_beats_budget_marginal"])
        self.assertFalse(self.gate("30B", -0.011, 0.3, 8.6, 0.147)["proceed_round7_attention_ladder_solve"])
        # passes the budget clause but not the gross 30B clause
        g = self.gate("30B", -0.009, 0.01, 1.0, 0.1)
        self.assertTrue(g["cond_beats_budget_marginal"])
        self.assertFalse(g["cond_gross_dnll"])
        self.assertTrue(self.gate("30B", -0.01, 0.0, 0.0, 0.1)["cond_gross_dnll"])  # <= is inclusive
        # 4B threshold is -0.004
        self.assertTrue(self.gate("4B", -0.005, 0.05, 4.2, 0.0247)["proceed_round7_attention_ladder_solve"])
        self.assertFalse(self.gate("4B", -0.003, 0.0, 4.2, 0.0247)["cond_gross_dnll"])
        # verification failure blocks everything
        self.assertFalse(self.gate("30B", -0.05, 0.1, 8.6, 0.147, ok=False)["proceed_round7_attention_ladder_solve"])
        with self.assertRaises(ValueError):
            self.gate("14B", -0.05, 0.1, 8.6, 0.147)

    def test_chords_from_best_all(self):
        rows = [r for r in json.loads(BEST_ALL.read_text())["rows"] if r["model"] in ("4B", "30B")]
        costs = {(r["model"], int(r["target"])): r["best_cost"] for r in rows}
        chords = R.chord_table(rows, costs)
        c = chords["30B:t40->t48"]
        expect = (100 * (8.085610625763966 - 7.8570662271187865) / 8.085610625763966) / \
                 (100 * (50.60087211675625 / 42.46211528193216 - 1))
        self.assertAlmostEqual(c["chord_pct_ppl_per_1pct_cycles"], expect)
        self.assertAlmostEqual(expect, 0.14747, places=4)
        b = R.binding_chord(chords, "30B", 32)
        self.assertEqual((b["upward"]["from_target"], b["upward"]["to_target"]), (32, 40))
        self.assertIsNone(b["downward_sensitivity"])
        b = R.binding_chord(chords, "4B", 64)
        self.assertEqual(b["upward"]["to_target"], 96)
        self.assertEqual(b["downward_sensitivity"]["from_target"], 48)


@unittest.skipUnless(MANIFEST.is_file(), "manifest not built yet")
class ManifestAndSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = D.load_manifest(MANIFEST)

    def test_manifest_contract(self):
        m = self.manifest
        self.assertEqual([t["id"] for t in m["tasks"]], ["30B_t40", "30B_t32", "4B_dense"])
        self.assertEqual(m["tasks"][2]["cells"], ["4B_t40", "4B_t64"])
        self.assertEqual(m["cells"]["30B_t40"]["arm_order"], ["A", "ATT128", "LIN128"])
        self.assertEqual(m["cells"]["30B_t32"]["arm_order"], ["A", "ATT128"])
        for cid, cell in m["cells"].items():
            self.assertEqual(len(cell["windows"]["starts"]), 16)
            self.assertFalse(any(a["best_all_candidate"] for a in cell["arms"].values()))
            self.assertEqual(cell["gate_gross_dnll_threshold"], -0.01 if cell["model"] == "30B" else -0.004)
            self.assertGreater(cell["chord"]["chord"], 0)
        self.assertEqual(m["cells"]["30B_t40"]["arms"]["A"]["spec"]["buckets"], A_30B)
        self.assertEqual(m["cells"]["4B_t64"]["arms"]["A"]["spec"]["buckets"], A_4B)
        self.assertIn("round7_gate", m["preregistered_rules"])

    def test_preflight_every_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(len(self.manifest["tasks"])):
                report = D.preflight_task(self.manifest, index, tmp)
                self.assertEqual([r["cell"] for r in report], self.manifest["tasks"][index]["cells"])

    def test_simulated_4b_task_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cid in self.manifest["tasks"][2]["cells"]:
                cell = self.manifest["cells"][cid]
                out = Path(tmp) / cid
                out.mkdir()
                summary = D.run_cell(self.manifest, cell, out, D.SimBackend(self.manifest, cell, out))
                self.assertTrue(summary["complete"] and summary["simulated"])
                self.assertTrue(summary["identity"]["probe"]["exact"])
                self.assertTrue(all(r["audit"]["ok"] for r in summary["arms"].values()))
                gate = summary["round7_gate"]["binding"]
                self.assertIn("proceed_round7_attention_ladder_solve", gate)
                self.assertAlmostEqual(summary["arms"]["A"]["metrics"]["extra_cycles_pct"],
                                       cell["qualification"]["A_first_order_extra_cycles_pct"], places=6)
                for name in ("INC", "A", "ATT128", "identity_before", "identity_after", "probe_incumbent"):
                    self.assertTrue((out / f"{name}_trace.json").is_file())
                    self.assertTrue((out / f"{name}_nll.json").is_file())

    def test_simulated_identity_mismatch_fails_closed(self):
        cell = copy.deepcopy(self.manifest["cells"]["4B_t40"])
        cell["identity"]["nll"] = cell["identity"]["nll"] + 1e-7
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "4B_t40"
            out.mkdir()
            backend = D.SimBackend(self.manifest, cell, out)
            backend.build()
            backend.inc_nll[cell["identity"]["start"]] = self.manifest["cells"]["4B_t40"]["identity"]["nll"]
            backend.build = lambda: (cell["windows"]["token_ids_sha256"], None)
            with self.assertRaises(D.IdentityError):
                D.run_cell(self.manifest, cell, out, backend)

    def test_profile_check_branch(self):
        cell = self.manifest["cells"]["4B_t40"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ok"
            out.mkdir()
            summary = D.run_cell(self.manifest, cell, out,
                                 D.SimBackend(self.manifest, cell, out, synthetic_profile=True))
            checks = summary["arms"]["A"]["profile_checks"]
            self.assertIsInstance(checks, dict)
            self.assertEqual(checks["profile_total_macs"], checks["first_window_trace_total_macs"])
            self.assertTrue((out / "A_profile.json").is_file() and (out / "INC_profile.json").is_file())
            out = Path(tmp) / "skew"
            out.mkdir()
            with self.assertRaises(D.AuditError):
                D.run_cell(self.manifest, cell, out,
                           D.SimBackend(self.manifest, cell, out, synthetic_profile=True, profile_skew=1e-4))

    def test_simulation_refuses_turbo_root(self):
        cell = self.manifest["cells"]["4B_t40"]
        with self.assertRaises(ValueError):
            D.SimBackend(self.manifest, cell, Path(self.manifest["output_root"]) / "4B_t40")


if __name__ == "__main__":
    unittest.main()
