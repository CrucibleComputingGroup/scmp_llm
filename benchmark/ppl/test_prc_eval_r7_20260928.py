"""CPU tests for the round-7 EVAL library and the STEP-0 driver/builder (screen-side tests live in
test_prc_screen_r7_20260928.py so that screen fixes never touch a file the step-0 manifest hashes).

Archived tables/traces are read-only inputs; every write goes to a temporary directory. No model,
GPU or scheduler. Run from the repo root with the experiment env:
  PYTHONPATH=kernels python -m unittest benchmark.ppl.test_prc_eval_r7_20260928 -v
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark.ppl import prc_eval_r7_20260928 as E
from benchmark.ppl import prc_r6_attn_diag_arms as R
from benchmark.ppl import prc_step0_r7_20260928 as S0
from benchmark.ppl import prc_screen_r7_20260928 as SC

K = E.KB
REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
INC4 = K / "prc2/4B_t40_c7_gfisla.json"
INC4_TRACE = K / "kbands_20260801/ppl/4B_t40_prc_p2c7gfisla_trace.json"
HYB4 = E.MP_BEST / "configs/4B/target40/hybrid_config.json"
INC30 = K / "prc_local_20260926/30B_t40/candidate07_mlp_from_projections.json"
SCOUT_SCREEN16 = [126976, 301056, 522240, 602112, 843776, 851968, 1071104, 1257472, 1384448, 1482752, 1728512,
                  1923072, 1996800, 2107392, 2295808, 2447360]
SCOUT_CONFIRM32 = [65536, 163840, 245760, 376832, 428032, 516096, 538624, 643072, 716800, 823296, 890880, 966656,
                   1011712, 1105920, 1222656, 1290240, 1392640, 1433600, 1525760, 1613824, 1624064, 1757184, 1845248,
                   1900544, 1933312, 2060288, 2101248, 2193408, 2244608, 2336768, 2381824, 2484224]
_C = {}


def cached(key, fn):
    if key not in _C:
        _C[key] = fn()
    return _C[key]


def inc4():
    return cached("inc4", lambda: (E.read_json(INC4), E.read_json(E.resolve_table_path(INC4))))


def uk_table(base, buckets=("qk:t0:l0", "qk:t0:l1", "qk:t0:l2", "qk:t0:l3", "av:t0:l3")):
    t = copy.deepcopy(base)
    top = [128] + E.global_ladder(base)[1:]
    for k in buckets:
        t["buckets"][k]["stoc_len_levels"] = list(top)
    t["r7_test"] = {"note": "unit-test UK stand-in"}
    return t


def write_arm(tmp, name, wrapper, table):
    tp = Path(tmp) / f"{name}_table.json"
    tp.write_text(json.dumps(table))
    w = dict(wrapper)
    w["threshold_table_path"] = str(tp)
    wp = Path(tmp) / f"{name}.json"
    wp.write_text(json.dumps(w))
    return wp


class WindowTest(unittest.TestCase):
    def test_registry_reproduces_scout_draw(self):
        reg = cached("reg", E.derive_round7_windows)
        self.assertEqual(len(reg["excluded"]), 218)
        self.assertEqual(reg["screen16_fresh_t48"], SCOUT_SCREEN16)
        self.assertEqual(reg["confirm32"], SCOUT_CONFIRM32)
        self.assertTrue(set(reg["capture_windows_expected"]) <= set(reg["excluded"]))
        self.assertFalse(E.overlaps(reg["confirm32"], reg["excluded"] + list(range(0, 65536, 2048))))
        self.assertFalse(E.overlaps(reg["confirm32"], reg["screen16_fresh_t48"]))
        self.assertFalse(E.overlaps(reg["confirm32"], reg["r6_diag_windows_30B"]))
        self.assertEqual(reg["r6_diag_windows_30B"][0], 149504)

    def test_registry_file_matches_rederivation(self):
        p = REPO / "benchmark/ppl/kbands/prc_windows_r7_20260928.json"
        if p.exists():
            self.assertEqual(E.read_json(p), cached("reg", E.derive_round7_windows))

    def test_stratified_mirror(self):
        import torch
        from benchmark.ppl.calibrate_mp_thresholds import _select_int_swap_windows
        enc = torch.zeros(E.QWEN_NTOK, dtype=torch.long)
        for n, seed in ((8, 0), (24, 101), (6, 0)):
            _, starts = _select_int_swap_windows(enc, 2048, n, sampling="stratified", seed=seed)
            self.assertEqual([int(s) for s in starts], E.stratified_starts(n, seed))
        self.assertEqual(E.stratified_starts(8, 0), [266240, 514048, 788480, 1026048, 1353728, 1585152,
                                                     1908736, 2205696])

    def test_overlaps(self):
        self.assertEqual(E.overlaps([0, 4096], [2048]), [])
        self.assertEqual(E.overlaps([0, 4096], [1000]), [0])


class LineageTest(unittest.TestCase):
    def test_c17_pairs_are_allocation_only(self):
        for cell, c17, s80 in (("30B_t40", "30B_t40_c17", "30B_t40_c17_s80"),
                               ("30B_t32", "30B_t32_c17e32", "30B_t32_c17e32_s80")):
            row = {r["model"] + "_t" + str(r["target"]): r for r in E.read_json(E.BEST_ALL)["rows"]}[cell]
            inc = E.read_json(E.resolve_table_path(row["wrapper"]))
            a = E.read_json(K / f"prc2/{c17}_table.json")
            b = E.read_json(K / f"prc2/{s80}_table.json")
            for t in (a, b):
                rep = E.validate_lineage(inc, t, total_blocks=48)
                self.assertTrue(rep["ok"], rep["failures"])
                self.assertEqual(rep["attention_ladders_changed"], {})
            pair = E.validate_lineage(a, b, total_blocks=48)
            self.assertTrue(pair["ok"])
            self.assertEqual(pair["attention_threshold_keys_changed"], [])
            self.assertEqual(len(pair["prc_threshold_keys_changed"]), 28)

    def test_violations_are_caught(self):
        _w, base = inc4()
        nb = 36

        def bad(mut, **kw):
            t = copy.deepcopy(base)
            mut(t)
            return E.validate_lineage(base, t, total_blocks=nb, **kw)
        self.assertFalse(bad(lambda t: t["buckets"]["qk:t0:l1"].__setitem__("metric_mean", 0.1))["ok"])
        self.assertFalse(bad(lambda t: t["buckets"]["qk:t0:l1"].__setitem__(
            "stoc_len_levels", [136] + E.global_ladder(base)[1:]))["ok"])
        self.assertFalse(bad(lambda t: t["buckets"]["qk:t0:l1"].__setitem__(
            "stoc_len_levels", sorted(E.global_ladder(base))))["ok"])
        self.assertFalse(bad(lambda t: t["per_row_chunk"]["buckets"]["q_proj:t0:l0"].__setitem__(
            "levels", [8, 128]))["ok"])
        self.assertFalse(bad(lambda t: t["protected_channels"].__setitem__("stoc_len", 128))["ok"])
        self.assertTrue(bad(lambda t: t["protected_channels"].__setitem__("stoc_len", 128),
                            allow_psl_change=True)["ok"])
        self.assertFalse(bad(lambda t: t.__setitem__("global_lambda", 1.0))["ok"])
        self.assertFalse(bad(lambda t: t.__setitem__("foo_bar", 1))["ok"])
        self.assertFalse(bad(lambda t: t["buckets"]["q_proj:t0:l0"].__setitem__(
            "stoc_len_levels", E.global_ladder(base)))["ok"])
        self.assertFalse(bad(lambda t: t["buckets"]["q_proj:t0:l0"].__setitem__("thresholds",
                                                                                 [0.0] * (len(E.global_ladder(base)) - 1)))["ok"])
        # per-bucket ladder with wrong threshold count
        self.assertFalse(bad(lambda t: t["buckets"]["qk:t0:l1"].__setitem__("stoc_len_levels", [128, 64]))["ok"])
        # thresholds increasing
        self.assertFalse(bad(lambda t: t["buckets"]["av:t0:l1"].__setitem__(
            "thresholds", [0.1, 0.9] + t["buckets"]["av:t0:l1"]["thresholds"][2:]))["ok"])

    def test_valid_uk_and_provenance(self):
        _w, base = inc4()
        rep = E.validate_lineage(base, uk_table(base), total_blocks=36)
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertEqual(sorted(rep["attention_ladders_changed"]), ["av:t0:l3", "qk:t0:l0", "qk:t0:l1",
                                                                   "qk:t0:l2", "qk:t0:l3"])
        self.assertIn("r7_test", rep["provenance_keys"])
        self.assertNotEqual(E.allocation_signature(uk_table(base)), E.allocation_signature(base))
        self.assertEqual(E.allocation_signature(uk_table(base)), E.allocation_signature(uk_table(base)))

    def test_layer_rekey_requires_allowance_and_pinned_prc(self):
        _w, base = inc4()
        nb = 36
        t = copy.deepcopy(base)
        t["layer_buckets"] = nb
        new = {}
        for op in E.ALL_OPS:
            for b in range(nb):
                _k, p = E.json_bucket_payload(base, op, b, nb)
                new[f"{op}:t0:l{b}"] = copy.deepcopy(p)
        t["buckets"] = new
        self.assertFalse(E.validate_lineage(base, t, total_blocks=nb)["ok"])
        self.assertFalse(E.validate_lineage(base, t, total_blocks=nb, allow_layer_rekey=True)["ok"])  # PRC not pinned
        t["per_row_chunk"]["layer_buckets"] = 4
        rep = E.validate_lineage(base, t, total_blocks=nb, allow_layer_rekey=True)
        self.assertTrue(rep["ok"], rep["failures"])


class ResolverTest(unittest.TestCase):
    def test_numpy_dispatch_matches_runtime_with_128_rung(self):
        w, base = inc4()
        with tempfile.TemporaryDirectory() as tmp:
            wp = write_arm(tmp, "UK", w, uk_table(base))
            rep = E.resolver_sweep(wp, INC4, 36)
            self.assertEqual(rep["checked"]["attention"], 72)
            self.assertIn(128, rep["synthetic_rows_by_len"])
            rep_inc = E.resolver_sweep(INC4, INC4, 36)
            self.assertEqual(rep_inc["checked"]["prc"], rep["checked"]["prc"])

    def test_sweep_refuses_escape_constant_change(self):
        w, base = inc4()
        t = uk_table(base)
        t["buckets"]["qk:t0:l2"]["metric_std"] = 0.01
        with tempfile.TemporaryDirectory() as tmp:
            wp = write_arm(tmp, "BAD", w, t)
            with self.assertRaises(E.AuditError):
                E.resolver_sweep(wp, INC4, 36)

    def test_wrapper_escape_must_match(self):
        w, _ = inc4()
        w2 = dict(w, escape_gate_k=3.0)
        with self.assertRaises(E.AuditError):
            E.check_wrapper_matches_inc(w2, w, "x")
        E.check_wrapper_matches_inc(dict(w, sc_prec=8, halve_bipolar_stoc_len=True), w, "extra keys ignored")

    def test_escape_disclosure(self):
        w, base = inc4()
        d = E.escape_disclosure(uk_table(base), w, 36)
        self.assertEqual(sorted(d), ["av:t0:l3", "qk:t0:l0", "qk:t0:l1", "qk:t0:l2", "qk:t0:l3"])
        allowed = ("gate never fires (t_esc >= 1); 128 is a genuinely new rung",
                   "gate redundant: the calibrated 128 threshold lies at or below mu+2tau",
                   "gate still the only route to 128 above its threshold; escaped rows fold onto the 128 rung")
        for v in d.values():
            self.assertIn(v["status"], allowed)
        self.assertTrue(any(v["t_esc"] is not None and v["t_esc"] < 1 for k, v in d.items() if k.startswith("qk")))


class TraceAuditTest(unittest.TestCase):
    def setUp(self):
        self.w, self.base = inc4()
        self.payload = cached("inc4trace", lambda: E.load_trace(INC4_TRACE))
        self.hyb = E.read_json(HYB4)
        self.inc_agg = cached("inc4agg", lambda: R.TraceAgg(self.payload, 36, 4))

    def audit(self, payload, wrapper_path, inc_agg="default"):
        return E.audit_trace(payload, name="t", wrapper_path=wrapper_path,
                             inc_agg=self.inc_agg if inc_agg == "default" else inc_agg, inc_table=self.base,
                             inc_wrapper=self.w, hybrid=self.hyb, total_blocks=36, layer_buckets=4)[0]

    def test_inc_trace_passes(self):
        rep = self.audit(self.payload, INC4)
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertLessEqual(rep["max_len"], 128)

    def test_uk_synth_passes_own_table_fails_inc_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = uk_table(self.base)
            wp = write_arm(tmp, "UK", self.w, t)
            syn = E.synthesize_trace(self.payload, self.base, self.w, t, self.w, 36)
            rep = self.audit(syn, wp)
            self.assertTrue(rep["ok"], rep["failures"])
            b = rep["buckets_vs_inc"]["qk:l1"]
            self.assertGreater(b["share_at_128"], b["inc_share_at_128"] + 0.5)
            self.assertEqual(b["share_outside_inc_realizable"], 0.0)  # INC reaches 128 via the escape gate
            # [128]+g[1:] is a subset of INC's realizable set (INC reaches 128 via escape), so an allowed-set
            # audit cannot tell them apart (the profile replay check does). A rung INC cannot realize must fail:
            g = E.global_ladder(self.base)
            odd = g[1] + 1
            self.assertNotIn(odd, g)
            t2 = copy.deepcopy(self.base)
            t2["buckets"]["qk:t0:l1"]["stoc_len_levels"] = [128, odd] + g[2:]
            t2["r7_test"] = {}
            wp2 = write_arm(tmp, "ODD", self.w, t2)
            syn2 = E.synthesize_trace(self.payload, self.base, self.w, t2, self.w, 36)
            self.assertTrue(self.audit(syn2, wp2)["ok"])
            rep2 = self.audit(syn2, INC4)
            self.assertFalse(rep2["ok"])
            self.assertTrue(any("runtime-resolver" in f for f in rep2["failures"]))
            self.assertTrue(any("JSON-ladder" in f for f in rep2["failures"]))

    def test_length_above_cap_and_mac_mismatch(self):
        bad = copy.deepcopy(self.payload)
        bad["groups"][0]["stoc_len"] = 136
        rep = self.audit(bad, INC4)
        self.assertFalse(rep["ok"])
        bad2 = copy.deepcopy(self.payload)
        bad2["groups"][0]["macs"] += 1
        rep2 = self.audit(bad2, INC4)
        self.assertTrue(any("MACs" in f for f in rep2["failures"]))

    def test_parent_synth_passes(self):
        pw = E.MP_BEST / "configs/4B/target40/wrapper.json"
        syn = E.synthesize_trace(self.payload, self.base, self.w, E.read_json(E.resolve_table_path(pw)),
                                 E.read_json(pw), 36)
        rep = self.audit(syn, pw)
        self.assertTrue(rep["ok"], rep["failures"])


class StatsAndRulesTest(unittest.TestCase):
    def test_rules(self):
        self.assertTrue(E.pred_passes_mde(-0.012, 0.010))
        self.assertFalse(E.pred_passes_mde(-0.009, 0.010))
        self.assertFalse(E.pred_passes_mde(0.02, 0.010))
        self.assertFalse(E.pred_passes_mde(-0.02, 0.0))
        self.assertTrue(E.cost_gate(101.0, 100.0)["ok"])
        self.assertFalse(E.cost_gate(101.01, 100.0)["ok"])
        self.assertTrue(E.cost_gate(99.0, 100.0)["ok"])
        p = {"audit_ok": True, "cost_gate": {"ok": True}, "metrics": {"dnll": -1e-4}}
        self.assertTrue(E.screen_proceeds(p)["proceed_to_confirm"])
        self.assertFalse(E.screen_proceeds(dict(p, metrics={"dnll": 0.0}))["proceed_to_confirm"])
        self.assertFalse(E.screen_proceeds(dict(p, cost_gate={"ok": False}))["proceed_to_confirm"])
        self.assertTrue(E.confirm_passes({"z": -2.01}, True)["full_test"])
        self.assertFalse(E.confirm_passes({"z": -2.0}, True)["full_test"])
        self.assertFalse(E.confirm_passes({"z": -3.0}, False)["full_test"])

    def _measured(self, m, sd=0.004, n=16):
        wins = list(range(0, n * 2048, 2048))
        d = [m + sd * math.sin(1.7 * i + 0.3) for i in range(n)]
        return {"window_dnll": d, "windows": wins, "ok": True}

    def test_kappa_decisions(self):
        """ONE-SIDED rule (aligned with prc_r7_solve KAPPA_RULE v2): keep 0.42 only if kappa_lin_bs exceeds
        kappa_att by more than the band; inside, below, or undefined -> revert to kappa 1."""
        cells = E.KAPPA_RULE["cells"]
        keep = E.kappa_decision({c: self._measured(0.04) for c in cells}, {c: 0.02 for c in cells})
        self.assertAlmostEqual(keep["kappa_lin_bs"], 2.0, places=1)
        self.assertEqual((keep["decision"], keep["ratio"]), ("keep_0.42", 0.42))
        rev = E.kappa_decision({c: self._measured(0.0166) for c in cells}, {c: 0.02 for c in cells})
        self.assertEqual((rev["decision"], rev["ratio"]), ("revert_to_kappa_1", 1.0))
        low = E.kappa_decision({c: self._measured(0.004, sd=0.0005) for c in cells}, {c: 0.02 for c in cells})
        self.assertEqual((low["decision"], low["ratio"]), ("revert_to_kappa_1", 1.0))   # direction contradicted
        self.assertIn("direction_contradicted", low["flags"])
        self.assertTrue(low["requires_user_review"])
        gap = E.kappa_decision({c: self._measured(0.026, sd=0.0005) for c in cells}, {c: 0.02 for c in cells})
        self.assertEqual(gap["decision"], "keep_0.42")
        self.assertIn("gap_case", gap["flags"])
        und = E.kappa_decision({c: self._measured(0.03) for c in cells}, {c: -0.01 for c in cells})
        self.assertTrue(und["requires_user_review"])
        self.assertEqual((und["decision"], und["ratio"]), ("revert_to_kappa_1", 1.0))
        miss = E.kappa_decision({cells[0]: self._measured(0.03)}, {cells[0]: 0.02})
        self.assertTrue(miss["requires_user_review"])
        self.assertEqual(miss["decision"], "revert_to_kappa_1")
        diffw = {cells[0]: self._measured(0.04), cells[1]: dict(self._measured(0.04), windows=list(range(16)))}
        dw = E.kappa_decision(diffw, {c: 0.02 for c in cells})
        self.assertEqual(dw["decision"], "revert_to_kappa_1")
        self.assertTrue(any(f.startswith("undefined") for f in dw["flags"]))
        # pooled SE uses the per-window sum across cells
        ms = {c: self._measured(0.03) for c in cells}
        dec = E.kappa_decision(ms, {c: 0.02 for c in cells})
        sums = [sum(ms[c]["window_dnll"][i] for c in cells) for i in range(16)]
        mean = sum(sums) / 16
        sd = math.sqrt(sum((x - mean) ** 2 for x in sums) / 15)
        self.assertAlmostEqual(dec["se_kappa_lin_bs"], sd / 4 / 0.04, places=12)

    def test_kappa_sign_flag(self):
        """A -20% linear cut predicted (or measured) to IMPROVE NLL is a broken score/measurement: the pooled
        value must not hide it."""
        c32, c40 = E.KAPPA_RULE["cells"]
        dec = E.kappa_decision({c32: self._measured(0.05), c40: self._measured(0.02)}, {c32: 0.04, c40: -0.005})
        self.assertTrue(dec["requires_user_review"])
        self.assertTrue(any(f.startswith(f"cell_sign_inconsistent:{c40}") for f in dec["flags"]))
        dec2 = E.kappa_decision({c32: self._measured(0.05), c40: self._measured(-0.01, sd=0.001)}, {c32: 0.04, c40: 0.01})
        self.assertTrue(any(f.startswith(f"cell_sign_inconsistent:{c40}") for f in dec2["flags"]))
        ok = E.kappa_decision({c: self._measured(0.04) for c in (c32, c40)}, {c: 0.02 for c in (c32, c40)})
        self.assertFalse(ok["requires_user_review"])

    def _step0(self):
        p = REPO / "benchmark/ppl/kbands/prc_step0_r7_20260928.json"
        if not p.exists():
            self.skipTest("step-0 manifest not built yet")
        return S0.load_manifest(p)

    def _fake_capture(self, tmp, m, *, complete=True, identity_ok=True, record_in="diag", swap=False):
        """step0_measured.json (+ summary) per rule cell and a capture diag with its identity.json."""
        diags = {}
        for i, cid in enumerate(E.KAPPA_RULE["cells"]):
            d = Path(tmp) / cid
            d.mkdir()
            meas = self._measured(0.03 + 0.005 * i)
            (d / "step0_measured.json").write_text(json.dumps({
                "windows": meas["windows"], "ok": True, "simulated": True, "identity": {"identity_mid_exact": True},
                "pair": {"window_dnll": meas["window_dnll"], "dcycles_pct": -11.0}}))
            if complete:
                (d / "step0_summary.json").write_text(json.dumps({"complete": True, "identity": {
                    "restored_incumbent_exact": True}}))
            else:
                (d / "failure.json").write_text(json.dumps({"type": "IdentityError", "message": "INC16 mismatch"}))
            cell = m["cells"][cid]
            a, b = cell["arms"]["C17"]["table"], cell["arms"]["S80"]["table"]
            if swap:
                a, b = b, a
            spec = f"c17={a},c17_s80={b}"
            cap = Path(tmp) / f"cap_{cid}"
            cap.mkdir()
            diag = {"tables": {"c17": {"pred_dnll_fis": -0.02, "L_over_parent": 1.0},
                               "c17_s80": {"pred_dnll_fis": -0.005, "L_over_parent": 0.8}}}
            if record_in == "diag":
                diag["prc9_r6"] = {"record": {"score_tables": spec}}
            elif record_in == "gfis":
                (cap / f"{cid}_r7_gfis_table.json").write_text(json.dumps({"prc6_calib": {"score_tables": spec}}))
            diags[cid] = cap / f"{cid}_r7_diag.json"
            diags[cid].write_text(json.dumps(diag))
            (cap / "identity.json").write_text(json.dumps({"ok": identity_ok, "identity_ok": identity_ok,
                                                           "cell": cid,
                                                           "identity_level": "exact" if identity_ok else "fail"}))
        return diags

    def test_kappa_stage_verifies_score_tables(self):
        m = self._step0()
        with tempfile.TemporaryDirectory() as tmp:
            diags = self._fake_capture(tmp, m)
            out = Path(tmp) / "kappa.json"
            with self.assertRaises(ValueError):
                S0.kappa_stage(m, diags, {}, out, out_root=tmp)  # simulated measured files refused
            rec = S0.kappa_stage(m, diags, {}, out, allow_simulated=True, out_root=tmp)
            self.assertEqual(rec["decision"], "keep_0.42")
            self.assertFalse(rec["requires_user_review"], rec["flags"])
            self.assertAlmostEqual(rec["per_cell"]["30B_t32"]["p"], 0.015)
            self.assertTrue(rec["simulated_inputs"])
            self.assertEqual(rec["prediction_sources"]["30B_t32"]["capture_identity"]["identity_level"], "exact")
        with tempfile.TemporaryDirectory() as tmp:     # recorded tables swapped -> refused
            diags = self._fake_capture(tmp, m, swap=True)
            with self.assertRaises(ValueError):
                S0.kappa_stage(m, diags, {}, None, allow_simulated=True, out_root=tmp)

    def test_kappa_stage_prefers_recorded_score_tables(self):
        m = self._step0()
        with tempfile.TemporaryDirectory() as tmp:     # the record lives in the capture's emitted gfis table
            diags = self._fake_capture(tmp, m, record_in="gfis")
            rec = S0.kappa_stage(m, diags, {}, None, allow_simulated=True, out_root=tmp)
            self.assertTrue(rec["prediction_sources"]["30B_t40"]["score_tables_recorded_source"].endswith(
                "_gfis_table.json"))
            cell = m["cells"]["30B_t40"]
            wrong = {"30B_t40": f"c17={cell['arms']['S80']['table']},c17_s80={cell['arms']['C17']['table']}"}
            with self.assertRaises(ValueError):           # intended (manifest) spec disagrees with the record
                S0.kappa_stage(m, diags, wrong, None, allow_simulated=True, out_root=tmp)
        with tempfile.TemporaryDirectory() as tmp:     # no record anywhere -> cannot verify -> refused
            diags = self._fake_capture(tmp, m, record_in=None)
            with self.assertRaises(ValueError):
                S0.kappa_stage(m, diags, {}, None, allow_simulated=True, out_root=tmp)

    def test_kappa_stage_flags_failed_cells_and_bad_captures(self):
        m = self._step0()
        with tempfile.TemporaryDirectory() as tmp:     # failure.json after step0_measured.json -> review
            diags = self._fake_capture(tmp, m, complete=False)
            rec = S0.kappa_stage(m, diags, {}, None, allow_simulated=True, out_root=tmp)
            self.assertTrue(rec["requires_user_review"])
            self.assertTrue(any(f.startswith("step0_cell_failed_after_measurement:") for f in rec["flags"]))
        with tempfile.TemporaryDirectory() as tmp:     # capture identity failed -> no prediction, review
            diags = self._fake_capture(tmp, m, identity_ok=False)
            rec = S0.kappa_stage(m, diags, {}, None, allow_simulated=True, out_root=tmp)
            self.assertTrue(rec["requires_user_review"])
            self.assertIsNone(rec["kappa_lin_bs"])
            self.assertEqual(rec["decision"], "revert_to_kappa_1")          # undefined -> revert (one-sided)
            self.assertTrue(any(f.startswith("capture_identity_not_ok:") for f in rec["flags"]))

    def test_profile_checks_exact(self):
        from benchmark.ppl.prc_local_proposals import exact_profile_cost  # noqa: F401
        snap = {"total_macs": 1000.0, "total_cycle_macs": 64000.0, "fixed_macs": 1000.0,
                "fixed_cycle_macs": 64000.0, "groups": {}}
        E.profile_checks(snap, (1000, 64000), {})
        with self.assertRaises(E.AuditError):
            E.profile_checks(dict(snap, total_cycle_macs=64000.5), (1000, 64000), {})
        with self.assertRaises(E.AuditError):               # 1 part in 1e7: the old 1e-6 tolerance hid it
            E.profile_checks(dict(snap, total_macs=10_000_001.0, fixed_macs=10_000_001.0,
                                  total_cycle_macs=640_000_000.0, fixed_cycle_macs=640_000_000.0),
                             (10_000_000, 640_000_000), {})


class Step0ReferenceCellTest(unittest.TestCase):
    """The optional 4B t64 'reference' cell end to end in the CPU simulation (small 4B traces)."""

    def test_reference_cell_simulation(self):
        man = REPO / "benchmark/ppl/kbands/prc_step0_r7_20260928.json"
        if not man.exists():
            self.skipTest("step-0 manifest not built yet")
        m = S0.load_manifest(man)
        self.assertEqual([t["id"] for t in m["tasks"]], ["30B_t40", "30B_t32", "30B_t48", "4B_t64"])
        self.assertEqual(m["cells"]["30B_t48"]["kind"], "reference")
        self.assertIsNone(m["cells"]["30B_t48"]["identity"]["reference"])
        self.assertEqual(m["cells"]["30B_t48"]["windows"]["starts"],
                         E.read_json(REPO / "benchmark/ppl/kbands/prc_windows_r7_20260928.json")["screen16_fresh_t48"])
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=str(REPO / "kernels"), CUDA_VISIBLE_DEVICES="")
            r = subprocess.run([PY, str(REPO / "benchmark/ppl/prc_step0_r7_20260928.py"), "simulate",
                                "--manifest", str(man), "--task", "3", "--out-root", tmp],
                               cwd=REPO, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            d = Path(tmp) / "4B_t64"
            s = E.read_json(d / "step0_summary.json")
            self.assertTrue(s["complete"] and s["simulated"] and s["kind"] == "reference")
            self.assertFalse((d / "step0_measured.json").exists())
            pr = E.read_json(d / "parent_reference.json")
            self.assertTrue(pr["audit_ok"])
            self.assertTrue((d / "INC16_profile.json").exists())
            self.assertTrue(s["extras"]["inc16_profiled"]["trace_records_exact_vs_reference"])
            self.assertIn("U_prot", s["extras"]["inc16_profiled"])


class AdapterTest(unittest.TestCase):
    def test_capture_manifest_adapter(self):
        cm = REPO / "benchmark/ppl/kbands/prc_r7_capture_20260928.json"
        sm = REPO / "benchmark/ppl/kbands/prc_step0_r7_20260928.json"
        if not cm.exists() or not sm.exists():
            self.skipTest("capture or step-0 manifest not built")
        diags, specs, idents = S0.from_capture_manifest(cm)
        m = S0.load_manifest(sm)
        for cid in E.KAPPA_RULE["cells"]:
            self.assertTrue(diags[cid].endswith(f"{cid}_r7_diag.json"))
            self.assertTrue(idents[cid].endswith(f"{cid}/capture/identity.json"))
            smap = S0._spec_map(specs[cid])
            self.assertEqual(smap["c17"], E.sha256_file(m["cells"][cid]["arms"]["C17"]["table"]))
            self.assertEqual(smap["c17_s80"], E.sha256_file(m["cells"][cid]["arms"]["S80"]["table"]))

    def test_pred_formats(self):
        v, lo, f = S0._pred({"tables": {"c17": {"pred_dnll_fis": -0.01, "L_over_parent": 1.0}}}, "c17", "pred_dnll_fis")
        self.assertEqual((v, lo, f), (-0.01, 1.0, "tables.c17.pred_dnll_fis"))
        v, lo, f = S0._pred({"preds": {"c17": {"pred_dnll": -0.02, "lin_cost_over_ref": 0.9}}}, "c17", "pred_dnll_fis")
        self.assertEqual((v, lo, f), (-0.02, 0.9, "preds.c17.pred_dnll"))
        with self.assertRaises(ValueError):
            S0._pred({"tables": {}}, "c17", "pred_dnll_fis")


if __name__ == "__main__":
    unittest.main()
