"""CPU tests for the round-7 screen driver's REGISTERED realized-length audit (2026-09-28).

Registered (prc_r7_capture_20260928.json preregistered_rules.realized_length_audit): before any screen NLL is read,
the GPU trace's realized lengths must be rungs the runtime resolver assigns (or the protected length), none above
128, fixed numerics (sc_prec 8, halve, rng_levels 128), the SC (op, block) set of the INT mask, and every NEW
112/128 attention rung's realized MAC share within +-0.10 absolute of the solve's hold-window prediction
(att_mac_hist_hold); any failure stops the cell.

Inputs are REAL round-6 GPU traces (Turbo prc_r6_20260928: arm A of the attention diagnostic put a new 128 rung on
qk/av buckets; 4B t64 and 30B t40 MoE) and synthetic, adversarial mutations of them. The driver tests run the
screen driver's run_cell on a 4B t64 stand-in manifest with the CPU simulation evaluator (archived INC trace
re-mapped onto each arm's ladders). Read-only inputs; every write goes to a temporary directory. No model, GPU or
scheduler.
  cd scmp_llm && PYTHONPATH=kernels <annstention python> -m unittest benchmark.ppl.test_prc_screen_audit_r7_20260928 -v
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark.ppl import prc_eval_r7_20260928 as E
from benchmark.ppl import prc_r6_attn_diag_arms as R
from benchmark.ppl import prc_screen_r7_20260928 as SC

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
R6 = E.KB / "prc_r6_20260928"
ARMS = R6 / "attn_diag_arms"
CAP = 128


def lad(table, nb):
    return SC.bucket_attention_ladders(table, nb, 4)


def realized_hist(payload, nb) -> dict:
    agg = R.TraceAgg(payload, nb, 4)
    return {f"{op}:l{q}": {L: m for L, m in h.items() if m} for (kind, op, q), h in agg.bucket_hist.items()
            if kind == "attention"}


def shifted_hold_hist(real: dict, new: dict, arm_l: dict, shift: dict) -> dict:
    """A solve-record-shaped hold histogram whose predicted new-rung share = realized share + shift[key] (all other
    MACs of the bucket on its lowest rung). Buckets without new rungs keep their realized histogram."""
    out = {}
    for key, h in real.items():
        if key not in new:
            out[key] = {str(L): float(m) for L, m in h.items()}
            continue
        tot = float(sum(h.values()))
        shares = {L: h.get(L, 0) / tot for L in new[key]}
        tgt = {L: min(1.0, max(0.0, s + shift.get(key, 0.0))) for L, s in shares.items()}
        rest = 1.0 - sum(tgt.values())
        assert rest >= -1e-12, (key, tgt)
        hh = {str(L): s * tot for L, s in tgt.items()}
        low = str(arm_l[key][0])
        hh[low] = hh.get(low, 0.0) + max(rest, 0.0) * tot
        out[key] = hh
    return out


def relabel(payload, pick, new_len=None, **field) -> dict:
    """Copy of a trace with the picked groups (pick(g) True) changed: stoc_len -> new_len and/or other fields."""
    groups, n = [], 0
    for g in payload["groups"]:
        if pick(g):
            g = dict(g)
            if new_len is not None:
                g["stoc_len"] = int(new_len)
                g["row_cycles"] = int(g["rows"]) * int(new_len)
            g.update(field)
            n += 1
        groups.append(g)
    assert n > 0, "mutation picked no group"
    return {"schema": payload.get("schema"), "header": payload["header"], "groups": groups}


class RealArmA:
    """Round-6 arm A (real GPU trace) of one cell, with its table / INC / mask."""

    def __init__(self, cid, nb):
        self.cid, self.nb = cid, nb
        c6 = E.read_json(E.R6_DIAG_MANIFEST)["cells"][cid]
        self.hybrid = E.read_json(c6["hybrid_config"])
        self.a_w = ARMS / cid / "A.json"
        self.a_t = E.read_json(ARMS / cid / "A_table.json")
        self.i_t = E.read_json(ARMS / cid / "INC_table.json")
        self.i_w = E.read_json(ARMS / cid / "INC.json")
        self.inc_agg = R.TraceAgg(E.load_trace(R6 / cid / "INC_trace.json"), nb, 4)
        self.payload = E.load_trace(R6 / cid / "A_trace.json")
        self.arm_l, self.inc_l = lad(self.a_t, nb), lad(self.i_t, nb)
        self.new = SC._new_rungs(self.arm_l, self.inc_l)
        self.real = realized_hist(self.payload, nb)
        self.bucket_of = {b: E.bucket_index(b, nb, 4) for b in range(nb)}

    def prediction(self, shift=None, *, table=None, real=None, new=None, arm_l=None):
        table = table or self.a_t
        arm_l = arm_l or self.arm_l
        new = new if new is not None else self.new
        rec = {"table_path": "unused-in-unit-tests",
               "att_mac_hist_hold": shifted_hold_hist(real or self.real, new, arm_l, shift or {})}
        return SC.new_rung_prediction(rec, table, self.i_t, total_blocks=self.nb, layer_buckets=4,
                                      source={"table_name": "A", "note": "unit test"})

    def audit(self, payload, prediction, *, binding=True, wrapper=None, table=None, edit_runtime_hist=None):
        """What the driver's audited() does for a screened arm: standard audits + the registered new-rung check."""
        w = wrapper or self.a_w
        rep, agg = E.audit_trace(payload, name="A_screen", wrapper_path=w, inc_agg=self.inc_agg, inc_table=self.i_t,
                                 inc_wrapper=self.i_w, hybrid=self.hybrid, total_blocks=self.nb, layer_buckets=4,
                                 expected_windows=payload["header"]["windows"],
                                 expected_wrapper=payload["header"]["mp_config_json"])
        hist = rep.pop("histogram_macs")
        if edit_runtime_hist:
            edit_runtime_hist(hist)
        nr = SC.new_rung_audit(prediction, agg, hist, table or self.a_t, self.i_t, total_blocks=self.nb,
                               layer_buckets=4, binding=binding, stage="screen")
        return bool(rep["ok"] and (nr["ok"] or not binding)), rep, nr

    def in_bucket(self, op, q):
        return lambda g: g["op"] == op and g["block"] is not None and self.bucket_of[int(g["block"])] == q


class Audit4BRealTrace(unittest.TestCase):
    """4B t64 round-6 arm A: qk l0-l3 and av l1-l3 gained a 128 rung (realized 0.93-1.0 of their MACs)."""

    @classmethod
    def setUpClass(cls):
        cls.X = RealArmA("4B_t64", 36)

    def test_inputs_are_the_real_round6_arm(self):
        X = self.X
        self.assertEqual(sorted(X.new), ["av:l1", "av:l2", "av:l3", "qk:l0", "qk:l1", "qk:l2", "qk:l3"])
        self.assertTrue(all(v == [CAP] for v in X.new.values()))
        self.assertFalse(X.payload["header"].get("simulated"))

    def test_pass_within_tolerance(self):
        X = self.X
        shift = {k: (0.05 if i % 2 else -0.05) for i, k in enumerate(sorted(X.new))}
        ok, rep, nr = X.audit(X.payload, X.prediction(shift))
        self.assertTrue(ok, (rep["failures"], nr["failures"]))
        self.assertEqual(sorted(nr["buckets"]), sorted(X.new))
        for k, b in nr["buckets"].items():
            r = b["rungs"]["128"]
            self.assertTrue(r["ok"])
            self.assertLessEqual(r["abs_diff"], 0.0500001)
        ok2, _rep2, nr2 = X.audit(X.payload, X.prediction({"qk:l1": -0.0999}))      # the edge stays inside
        self.assertTrue(ok2, nr2["failures"])

    def test_share_beyond_tolerance_fails_only_the_new_check(self):
        X = self.X
        ok, rep, nr = X.audit(X.payload, X.prediction({"qk:l1": -0.11}))
        self.assertFalse(ok)
        self.assertTrue(rep["ok"], rep["failures"])                 # every standard audit still passes
        self.assertEqual(len(nr["failures"]), 1, nr["failures"])
        self.assertIn("qk:l1: new rung 128", nr["failures"][0])
        self.assertFalse(nr["buckets"]["qk:l1"]["rungs"]["128"]["ok"])

    def test_resolver_bug_valid_rungs_wrong_distribution(self):
        """The resolver-bug family that ladder membership cannot see: every realized length is a legal rung of
        the bucket, but the MACs sit on the wrong rungs (e.g. a bucket resolved to the old ladder top-cut)."""
        X = self.X
        mut = relabel(X.payload, lambda g: X.in_bucket("av", 2)(g) and int(g["stoc_len"]) == CAP and
                      int(g["block"]) % 2 == 0, new_len=64)
        share = realized_hist(mut, 36)["av:l2"]
        self.assertLess(share.get(CAP, 0) / sum(share.values()), 0.85)
        ok, rep, nr = X.audit(mut, X.prediction())
        self.assertTrue(rep["ok"], rep["failures"])                 # all lengths are legal rungs, MACs unchanged
        self.assertFalse(ok)
        self.assertTrue(any("av:l2: new rung 128" in f for f in nr["failures"]), nr["failures"])

    def test_silent_fallback_to_the_inc_ladder(self):
        """The runtime ignored the bucket ladders (INC's top rung 105 realized instead of 128): both the
        standard ladder audits and the new-rung check fail."""
        X = self.X
        mut = relabel(X.payload, lambda g: g["op"] == "qk" and int(g["stoc_len"]) == CAP, new_len=105)
        ok, rep, nr = X.audit(mut, X.prediction())
        self.assertFalse(ok)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("realized-length audit failed" in f for f in rep["failures"]), rep["failures"])
        self.assertEqual(sorted({f.split(": new rung")[0] for f in nr["failures"] if ": new rung" in f}),
                         ["qk:l0", "qk:l1", "qk:l2", "qk:l3"])

    def test_length_above_cap_fails(self):
        X = self.X
        blk = min(int(g["block"]) for g in X.payload["groups"] if g["op"] == "av" and int(g["stoc_len"]) == CAP)
        mut = relabel(X.payload, lambda g: g["op"] == "av" and int(g["block"]) == blk and int(g["stoc_len"]) == CAP,
                      new_len=256)
        ok, rep, _nr = X.audit(mut, X.prediction())
        self.assertFalse(ok)
        self.assertTrue(any("> 128" in f for f in rep["failures"]), rep["failures"])

    def test_fixed_numerics_fail(self):
        X = self.X
        first = {"done": False}

        def one(g):
            if first["done"] or g["op"] != "q_proj":
                return False
            first["done"] = True
            return True
        for field, val in (("rng_levels", 64), ("sc_prec", 7), ("halve", False)):
            with self.subTest(field=field):
                first["done"] = False
                mut = relabel(X.payload, one, **{field: val})
                ok, rep, _nr = X.audit(mut, X.prediction())
                self.assertFalse(ok)
                self.assertTrue(any("numerics" in f for f in rep["failures"]), rep["failures"])

    def test_sc_set_differs_from_the_mask(self):
        X = self.X
        blk = sorted({int(g["block"]) for g in X.payload["groups"] if g["op"] == "qk"})[2]
        mut = dict(X.payload, groups=[g for g in X.payload["groups"] if not (g["op"] == "qk" and g["block"] == blk)])
        self.assertLess(len(mut["groups"]), len(X.payload["groups"]))
        ok, rep, _nr = X.audit(mut, X.prediction())
        self.assertFalse(ok)
        self.assertTrue(any("hybrid INT mask" in f for f in rep["failures"]), rep["failures"])
        self.assertTrue(any("(op, block) set differs from INC" in f for f in rep["failures"]), rep["failures"])

    def test_runtime_bucket_mapping_must_agree(self):
        X = self.X

        def edit(h):
            k = next(L for L in h["qk:l2"])
            h["qk:l2"][k] = float(h["qk:l2"][k]) * 0.5
        ok, _rep, nr = X.audit(X.payload, X.prediction(), edit_runtime_hist=edit)
        self.assertFalse(ok)
        self.assertTrue(any("runtime-resolver histogram" in f for f in nr["failures"]), nr["failures"])

    def test_prediction_is_fail_closed(self):
        X = self.X
        good = X.prediction()
        cases = {"none": None,
                 "standin_binding": SC.standin_prediction("unit test"),
                 "stale_buckets": dict(good, buckets={k: v for k, v in good["buckets"].items() if k != "av:l3"}),
                 "wrong_tolerance": dict(good, tolerance=0.2),
                 "wrong_schema": dict(good, schema="other")}
        for name, pred in cases.items():
            with self.subTest(case=name):
                ok, rep, nr = X.audit(X.payload, pred)
                self.assertTrue(rep["ok"])
                self.assertFalse(ok)
                self.assertTrue(nr["failures"])
        ok, _rep, nr = X.audit(X.payload, SC.standin_prediction("dry run"), binding=False)   # descriptive only
        self.assertTrue(ok)
        self.assertTrue(nr["ok"] and nr["notes"] and not nr["binding"])

    def test_prediction_derivation_refuses_bad_histograms(self):
        X = self.X
        base = shifted_hold_hist(X.real, X.new, X.arm_l, {})
        bad = {"missing_bucket": {k: v for k, v in base.items() if k != "qk:l0"},
               "length_outside_ladder": dict(base, **{"qk:l0": dict(base["qk:l0"], **{"105": 1.0e9})}),
               "empty_bucket": dict(base, **{"qk:l0": {"128": 0.0}}),
               "negative": dict(base, **{"qk:l0": dict(base["qk:l0"], **{"128": -1.0})})}
        for name, h in bad.items():
            with self.subTest(case=name):
                with self.assertRaises(E.AuditError):
                    SC.new_rung_prediction({"att_mac_hist_hold": h}, X.a_t, X.i_t, total_blocks=36, layer_buckets=4,
                                           source={})
        with self.assertRaises(E.AuditError):              # no histogram at all
            SC.new_rung_prediction({}, X.a_t, X.i_t, total_blocks=36, layer_buckets=4, source={})
        with self.assertRaises(E.AuditError):              # the solve record's ladders disagree with the table
            SC.new_rung_prediction({"att_mac_hist_hold": base, "ladders": {"qk:l0": [48, 64, 105]}}, X.a_t, X.i_t,
                                   total_blocks=36, layer_buckets=4, source={})
        t = copy.deepcopy(X.a_t)                            # an unregistered new rung (100) is refused
        t["buckets"]["qk:t0:l0"]["stoc_len_levels"] = [128, 100] + list(t["buckets"]["qk:t0:l0"]["stoc_len_levels"][1:-1])
        t["buckets"]["qk:t0:l0"]["thresholds"] = list(t["buckets"]["qk:t0:l0"]["thresholds"])[:len(
            t["buckets"]["qk:t0:l0"]["stoc_len_levels"]) - 1]
        with self.assertRaises(E.AuditError):
            SC.new_rung_prediction({"att_mac_hist_hold": base}, t, X.i_t, total_blocks=36, layer_buckets=4, source={})

    def test_rederive_prediction_is_sha_locked(self):
        X = self.X
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "solve_summary.json"
            tp = str(ARMS / "4B_t64" / "A_table.json")
            rec = {"table_path": tp, "att_mac_hist_hold": shifted_hold_hist(X.real, X.new, X.arm_l, {})}
            sp.write_text(json.dumps({"arms": {"r7U": rec}}))
            src = {"solve_summary": str(sp), "solve_summary_sha256": E.sha256_file(sp), "table_name": "r7U"}
            pred = json.loads(json.dumps(SC.new_rung_prediction(rec, X.a_t, X.i_t, total_blocks=36, layer_buckets=4,
                                                                source=src)))
            cell = {"hashes": {str(sp): E.sha256_file(sp)}, "total_blocks": 36, "layer_buckets": 4}
            arm = {"table": tp, "realized_length_prediction": pred}
            self.assertEqual(SC.rederive_prediction(cell, "U", arm, X.a_t, X.i_t), pred)
            tampered = copy.deepcopy(arm)
            tampered["realized_length_prediction"]["buckets"]["qk:l1"]["pred_share"]["128"] = 0.5
            with self.assertRaises(E.AuditError):
                SC.rederive_prediction(cell, "U", tampered, X.a_t, X.i_t)
            with self.assertRaises(E.AuditError):              # another table than the summary's
                SC.rederive_prediction(cell, "U", dict(arm, table=str(ARMS / "4B_t64" / "INC_table.json")), X.a_t, X.i_t)
            with self.assertRaises(E.AuditError):              # the summary is not a hashed input of the cell
                SC.rederive_prediction(dict(cell, hashes={}), "U", arm, X.a_t, X.i_t)
            sp.write_text(json.dumps({"arms": {"r7U": dict(rec, extra=1)}}))
            with self.assertRaises(E.AuditError):              # summary changed after freezing
                SC.rederive_prediction(cell, "U", arm, X.a_t, X.i_t)

    def test_nll_log_mask(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(SC._WithheldNLL(buf)):
            print("[r7scr] 4B_t64 U_screen window 3/16 start=149504 NLL=2.2339019775390625", flush=True)
        self.assertNotIn("2.2339", buf.getvalue())
        self.assertIn("NLL=" + SC.WITHHELD, buf.getvalue())


class Audit30BRealTrace(unittest.TestCase):
    """30B t40 (MoE, 48 blocks) round-6 arm A: qk l0-l3 + av l3 gained 128; plus a synthetic 112 rung (the 'up'
    policy adds 112 on 30B because 112 >= 1.1 x the parent's top 96)."""

    @classmethod
    def setUpClass(cls):
        cls.X = RealArmA("30B_t40", 48)
        cls.tmp = tempfile.TemporaryDirectory()
        X, d = cls.X, Path(cls.tmp.name)
        t = copy.deepcopy(X.a_t)                           # A + a 112 rung on qk:l1 (same thresholds count + 1)
        b = t["buckets"]["qk:t0:l1"]
        levels = sorted(set(int(v) for v in b["stoc_len_levels"]) | {112}, reverse=True)
        th = [float(v) for v in b["thresholds"]]
        b["stoc_len_levels"], b["thresholds"] = levels, [th[0], th[0]] + th[1:]
        tp, wp = d / "A112_table.json", d / "A112.json"
        tp.write_text(json.dumps(t))
        w = E.read_json(X.a_w)
        w["threshold_table_path"] = str(tp)
        wp.write_text(json.dumps(w))
        cls.t112, cls.w112 = t, wp
        q1 = X.in_bucket("qk", 1)
        blocks = sorted({int(g["block"]) for g in X.payload["groups"] if q1(g)})
        moved = set(blocks[: max(1, len(blocks) // 3)])
        mut = relabel(X.payload, lambda g: q1(g) and int(g["stoc_len"]) == CAP and int(g["block"]) in moved,
                      new_len=112)
        mut["header"] = dict(mut["header"], mp_config_json=str(wp))
        cls.p112 = mut

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_30b_trace_and_a_112_rung(self):
        X = self.X
        self.assertEqual(sorted(X.new), ["av:l3", "qk:l0", "qk:l1", "qk:l2", "qk:l3"])
        new = SC._new_rungs(lad(self.t112, 48), X.inc_l)
        self.assertEqual(new["qk:l1"], [112, CAP])
        real = realized_hist(self.p112, 48)
        tot = sum(real["qk:l1"].values())
        s112 = real["qk:l1"][112] / tot
        self.assertTrue(0.05 < s112 < 0.6, s112)
        pred = X.prediction({"qk:l1": 0.04}, table=self.t112, real=real, new=new, arm_l=lad(self.t112, 48))
        ok, rep, nr = X.audit(self.p112, pred, wrapper=self.w112, table=self.t112)
        self.assertTrue(ok, (rep["failures"], nr["failures"]))
        self.assertEqual(sorted(nr["buckets"]["qk:l1"]["rungs"]), ["112", "128"])
        # the same trace against a prediction whose 112 share is 0.12 away: only the new-rung check fails
        h = shifted_hold_hist(real, new, lad(self.t112, 48), {})
        q = {int(L): m for L, m in h["qk:l1"].items()}
        mv = 0.12 * sum(q.values())
        q[112] -= min(mv, q[112])
        q[CAP] += mv
        h["qk:l1"] = {str(L): m for L, m in q.items()}
        pred_bad = SC.new_rung_prediction({"att_mac_hist_hold": h}, self.t112, X.i_t, total_blocks=48,
                                          layer_buckets=4, source={})
        ok2, rep2, nr2 = X.audit(self.p112, pred_bad, wrapper=self.w112, table=self.t112)
        self.assertTrue(rep2["ok"])
        self.assertFalse(ok2)
        self.assertTrue(any("qk:l1: new rung 112" in f for f in nr2["failures"]), nr2["failures"])

    def test_30b_resolver_bug_on_moe_trace(self):
        X = self.X
        mut = relabel(X.payload, lambda g: X.in_bucket("qk", 3)(g) and int(g["stoc_len"]) == CAP and
                      int(g["block"]) % 3 == 0, new_len=65)
        ok, rep, nr = X.audit(mut, X.prediction())
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertFalse(ok)
        self.assertTrue(any("qk:l3: new rung 128" in f for f in nr["failures"]), nr["failures"])


# ---------------------------------------------------------------------------------------------
# the driver: every screened trace audited before any candidate NLL is released; any failure stops the cell
# ---------------------------------------------------------------------------------------------
class DriverOrderingTest(unittest.TestCase):
    """run_cell on the 4B t64 cell (U primary with a new 128 rung, K secondary), CPU simulation evaluator."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        env = dict(os.environ, PYTHONPATH=str(REPO / "kernels"), CUDA_VISIBLE_DEVICES="")
        man = d / "manifest_dry.json"
        r = subprocess.run([PY, str(REPO / "benchmark/ppl/kbands/build_prc_screen_r7_20260928.py"), "--standins",
                            str(d / "standins"), "--assume-gate", "30B_t32=pass", "--assume-gate", "30B_t40=pass",
                            "--assume-gate", "4B_t40=fail", "--assume-gate", "4B_t64=pass", "--assume-kappa-ratio",
                            "0.42", "--enable-4b", "--skip-sweep", "--out", str(d / "standins" / "m.json")],
                           cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        m = json.loads((d / "standins" / "m.json").read_text())
        cell = m["cells"]["4B_t64"]
        assert cell["enabled"] and cell["primary"] == "U" and cell["secondaries"] == ["K"], cell.get("disabled_reason")
        assert cell["arms"]["U"]["realized_length_prediction"]["standin"] is True
        man.write_text(json.dumps(m))
        cls.dry_manifest = man
        # a REAL-audit variant: predictions from a (fake) solve summary, sha-locked in the cell's hashes
        inc_t = E.read_json(cell["arms"]["INC"]["table"])
        inc_w = E.read_json(cell["arms"]["INC"]["wrapper"])
        inc_standin = E.load_trace(cell["simulation"]["inc_standin"])
        cls.syn_hist = {}
        for n in ("U", "K"):
            a = cell["arms"][n]
            syn = E.synthesize_trace(inc_standin, inc_t, inc_w, E.read_json(a["table"]), E.read_json(a["wrapper"]), 36)
            cls.syn_hist[n] = {k: {str(L): float(m) for L, m in h.items()} for k, h in realized_hist(syn, 36).items()}
        cls.cell0, cls.m0, cls.inc_t = cell, m, inc_t

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def real_manifest(self, d: Path, *, shift=None):
        m = copy.deepcopy(self.m0)
        m["dry_run"] = False
        cell = m["cells"]["4B_t64"]
        sp = d / "solve_summary.json"
        arms = {}
        for n in ("U", "K"):
            a = cell["arms"][n]
            h = copy.deepcopy(self.syn_hist[n])
            for key, dv in (shift or {}).get(n, {}).items():
                q = {int(L): v for L, v in h[key].items()}
                tot = sum(q.values())
                low = min(q)
                q[CAP] -= dv * tot
                q[low] = q.get(low, 0.0) + dv * tot
                h[key] = {str(L): v for L, v in q.items()}
            arms[f"r7{n}"] = {"table_path": a["table"], "att_mac_hist_hold": h}
        sp.write_text(json.dumps({"arms": arms}))
        cell["hashes"][str(sp)] = E.sha256_file(sp)
        for n in ("U", "K"):
            a = cell["arms"][n]
            src = {"solve_summary": str(sp), "solve_summary_sha256": E.sha256_file(sp), "table_name": f"r7{n}"}
            a["realized_length_prediction"] = json.loads(json.dumps(SC.new_rung_prediction(
                arms[f"r7{n}"], E.read_json(a["table"]), self.inc_t, total_blocks=36, layer_buckets=4, source=src)))
        return m, cell

    def run_cell(self, m, cell, out, ev_cls=None):
        out.mkdir(parents=True)
        ev = SC.sim_evaluator(cell, out)
        if ev_cls is not None:
            ev.__class__ = ev_cls
        try:
            return SC.run_cell(m, cell, out, ev)
        finally:
            ev.close()

    def test_real_prediction_passes_and_binds(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            m, cell = self.real_manifest(d)
            self.assertIn("qk:l0", cell["arms"]["U"]["realized_length_prediction"]["buckets"])
            self.assertEqual(cell["arms"]["K"]["realized_length_prediction"]["buckets"], {})   # K: no new rung
            sel = self.run_cell(m, cell, d / "out")
            scr = E.read_json(d / "out" / "screen_result.json")
            self.assertTrue(scr["realized_length_audit"]["binding"])
            nr = scr["arms"]["U"]["new_rung_audit"]
            self.assertTrue(nr["ok"] and nr["binding"])
            self.assertTrue(all(r["abs_diff"] == 0.0 for b in nr["buckets"].values() for r in b["rungs"].values()))
            self.assertTrue(sel["evaluate_full_test"], sel)
            self.assertTrue((d / "out" / "U_screen_nll.json").is_file())
            self.assertTrue(E.read_json(d / "out" / "U_confirm_audit.json")["new_rung_audit"]["binding"] is False)

    def test_primary_share_failure_stops_the_cell_before_any_nll(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            m, cell = self.real_manifest(d, shift={"U": {"qk:l1": 0.2}})
            with self.assertRaises(E.AuditError) as cm:
                self.run_cell(m, cell, d / "out")
            self.assertIn("cell stopped", str(cm.exception))
            out = d / "out"
            fr = E.read_json(out / "U_screen_audit_failure.json")
            self.assertIsNone(fr["decision"])
            self.assertTrue(any("qk:l1: new rung 128" in f for f in fr["failures"]), fr["failures"])
            self.assertFalse((out / "U_screen_nll.json").exists())      # the candidate NLL was never written
            self.assertFalse((out / "screen_result.json").exists())     # no decision
            self.assertFalse((out / "selected.json").exists())
            prog = E.read_json(out / "screen_progress.json")
            self.assertEqual(prog["evaluations"]["U_screen"]["nll"], SC.WITHHELD)
            self.assertNotIn("mean_nll", prog["evaluations"]["U_screen"])
            self.assertFalse((out / "K_screen_trace.json").exists())    # stopped at the primary

    def test_secondary_failure_stops_the_cell_and_withholds_the_primary_nll(self):
        class CorruptK(E.SimEvaluator):
            def evaluate(self, name, wrapper_path, starts, **kw):
                r = super().evaluate(name, wrapper_path, starts, **kw)
                if name == "K_screen":
                    p = E.load_trace(r["trace"])
                    p["groups"][0] = dict(p["groups"][0], rng_levels=64)
                    Path(r["trace"]).write_text(json.dumps(p))
                return r
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            m, cell = self.real_manifest(d)
            with self.assertRaises(E.AuditError):
                self.run_cell(m, cell, d / "out", ev_cls=CorruptK)
            out = d / "out"
            self.assertTrue((out / "K_screen_audit_failure.json").is_file())
            self.assertTrue(E.read_json(out / "U_screen_audit.json")["ok"])   # the primary's trace passed ...
            self.assertFalse((out / "U_screen_nll.json").exists())            # ... but its NLL was never released
            self.assertFalse((out / "screen_result.json").exists())

    def test_candidate_nll_lines_are_masked_in_the_log(self):
        class Chatty(E.SimEvaluator):
            def evaluate(self, name, wrapper_path, starts, **kw):
                r = super().evaluate(name, wrapper_path, starts, **kw)
                for i, (s, v) in enumerate(zip(starts, r["window_nll"])):
                    print(f"[r7scr] 4B_t64 {name} window {i + 1}/{len(starts)} start={s} NLL={v!r}", flush=True)
                return r
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            m, cell = self.real_manifest(d)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.run_cell(m, cell, d / "out", ev_cls=Chatty)
            lines = buf.getvalue().splitlines()
            cand = [ln for ln in lines if " U_screen " in ln or " K_screen " in ln or " U_confirm " in ln]
            inc = [ln for ln in lines if " INC_screen " in ln]
            self.assertTrue(cand and inc)
            self.assertTrue(all("NLL=" + SC.WITHHELD in ln for ln in cand), cand[:3])
            self.assertTrue(all(SC.WITHHELD not in ln for ln in inc))

    def test_preflight_refuses_a_tampered_or_standin_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            m, cell = self.real_manifest(d)
            cell["arms"]["U"]["realized_length_prediction"]["buckets"]["qk:l0"]["pred_share"]["128"] = 0.3
            with self.assertRaises(ValueError):
                SC.verify_cell(m, cell)
            (d / "b").mkdir()
            m2, cell2 = self.real_manifest(d / "b")
            cell2["arms"]["K"]["realized_length_prediction"] = SC.standin_prediction("x")
            with self.assertRaises(ValueError):                                # stand-in outside a dry run
                SC.verify_cell(m2, cell2)
            SC.verify_cell(json.loads(self.dry_manifest.read_text()), copy.deepcopy(self.cell0))   # dry run: ok

    def test_dry_run_manifest_is_descriptive_and_never_runs_on_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            m = json.loads(self.dry_manifest.read_text())
            sel = self.run_cell(m, copy.deepcopy(self.cell0), d / "out")
            scr = E.read_json(d / "out" / "screen_result.json")
            self.assertFalse(scr["realized_length_audit"]["binding"])
            self.assertTrue(scr["arms"]["U"]["new_rung_audit"]["notes"])
            self.assertIn("evaluate_full_test", sel)
            env = dict(os.environ, PYTHONPATH=str(REPO / "kernels"), CUDA_VISIBLE_DEVICES="")
            r = subprocess.run([PY, str(REPO / "benchmark/ppl/prc_screen_r7_20260928.py"), "run", "--manifest",
                                str(self.dry_manifest), "--task", "4"], cwd=REPO, env=env, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("DRY-RUN", r.stderr)


if __name__ == "__main__":
    unittest.main()
