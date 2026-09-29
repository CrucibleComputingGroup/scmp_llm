"""CPU tests for the round-7 SCREEN side: pre-registration, target-scale fixed point (replay on a
step-0 INC16 profile), the screen builder (real and dry-run paths) and the screen driver's finalize.

Archived tables/traces/profiles are read-only inputs; every write goes to a temporary directory. No
model, GPU or scheduler. These tests live outside the step-0 manifest's hashed sources on purpose.
Run from the repo root with the experiment env:
  PYTHONPATH=kernels python -m unittest benchmark.ppl.test_prc_screen_r7_20260928 -v
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from benchmark.ppl import prc_eval_r7_20260928 as E
from benchmark.ppl import prc_fixedpoint_r7_20260928 as F
from benchmark.ppl import prc_r7_prereg_20260928 as PR
from benchmark.ppl import prc_screen_r7_20260928 as SC
from benchmark.ppl.kbands import build_prc_screen_r7_20260928 as B

K = E.KB
REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
R6 = K / "prc_r6_20260928"
INC4_64 = K / "prc2/4B_t64_c7_gfisla.json"
STEP0_MANIFEST = REPO / "benchmark/ppl/kbands/prc_step0_r7_20260928.json"
REG = REPO / "benchmark/ppl/kbands/prc_windows_r7_20260928.json"


def uk_table(base, buckets=("qk:t0:l0", "qk:t0:l1", "qk:t0:l2", "qk:t0:l3", "av:t0:l3"), tag="r7_test"):
    t = copy.deepcopy(base)
    top = [128] + E.global_ladder(base)[1:]
    for k in buckets:
        t["buckets"][k]["stoc_len_levels"] = list(top)
    t[tag] = {"note": "unit-test UK stand-in"}
    return t


def k_table(base, f=0.97):
    t = copy.deepcopy(base)
    for key, p in t["buckets"].items():
        if key.split(":")[0] in E.ATTN_OPS:
            p["thresholds"] = [min(1.0, max(0.0, float(x) * f)) for x in p["thresholds"]]
    t["r7_test"] = {"note": "unit-test K stand-in"}
    return t


def write_arm(tmp, name, wrapper, table):
    tp = Path(tmp) / f"{name}_table.json"
    tp.write_text(json.dumps(table))
    w = dict(wrapper)
    w["threshold_table_path"] = str(tp)
    wp = Path(tmp) / f"{name}.json"
    wp.write_text(json.dumps(w))
    return wp, tp


def _later(path):
    t = time.time() + 2
    os.utime(path, (t, t))


def hold_hist(table, total_blocks):
    """A valid prc_r7_solve-shaped att_mac_hist_hold for ``table``: every attention bucket spreads MACs over its
    resolved ladder (JSON keys, as solve_summary.json stores them)."""
    return {key: {str(L): 1.0e9 * (i + 1) for i, L in enumerate(lad)}
            for key, lad in SC.bucket_attention_ladders(table, total_blocks, 4).items()}


# ---------------------------------------------------------------------------------------------
class PreregTest(unittest.TestCase):
    def test_constants_reproduce_and_match_capture_manifest(self):
        d = PR.derive_mde()
        self.assertEqual({c: v["mde_nats"] for c, v in d.items()}, PR.MDE_NATS)
        for c, v in d.items():   # rounded UP: never below the raw MDE80
            self.assertGreaterEqual(v["mde_nats"], v["mde_raw"])
            self.assertTrue(all("ATT128" not in p["source"] and "LIN128" not in p["source"] for p in v["pairs"]))
        cm = E.read_json(REPO / "benchmark/ppl/kbands/prc_r7_capture_20260928.json")
        for c in cm["cells"]:   # ONE registration: the calib side's per-cell prereg must be this module's
            pre = c.get("prereg")
            if pre is None:
                self.assertEqual(PR.S0[c["id"]], c["solve_defaults_informational"]["initial_target_scale"])
                continue
            self.assertEqual((pre["mde_nats"], pre["s0"], pre["ladder_policy"]),
                             (PR.MDE_NATS[c["id"]], PR.S0[c["id"]], PR.LADDER_POLICY), c["id"])
            if c["model"] == "30B":
                self.assertEqual((pre["primary"], list(pre["arms"])), ("UK", list(PR.ARMS)))
        keep = PR.kappa_pins_for("30B", "keep_0.42")
        self.assertLessEqual(abs(keep["kappa_att"] / keep["kappa_lin"] - 0.42), PR.KAPPA_RATIO_TOL)
        rev = PR.kappa_pins_for("30B", "revert_to_kappa_1")
        self.assertEqual(rev["kappa_att"] / rev["kappa_lin"], 1.0)
        self.assertEqual(PR.kappa_pins_for("4B", None)["branch"], "dense")
        with self.assertRaises(ValueError):
            PR.kappa_pins_for("30B", None)
        s0m = E.read_json(STEP0_MANIFEST)
        self.assertEqual(s0m["kappa_rule"]["kappa_att"], 0.83)
        self.assertEqual(s0m["kappa_rule"]["kappa_att_se"], 0.07)
        self.assertIn("ONE kappa rule", s0m["kappa_rule"]["authority"])
        self.assertIn("ONE-SIDED", s0m["kappa_rule"]["decision"])
        self.assertEqual(s0m["kappa_rule"], E.KAPPA_RULE)

    def test_record_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = PR.write_record(Path(tmp) / "prereg.json", outputs_root=Path(tmp) / "none")
            rec, sha, t = PR.load_verified(p)
            self.assertEqual(rec["mde"]["nats"], PR.MDE_NATS)
            self.assertLess(abs(t - time.time()), 60)
            with self.assertRaises(FileExistsError):   # written once
                PR.write_record(p, outputs_root=Path(tmp) / "none")
            bad = json.loads(p.read_text())
            bad["mde"]["nats"]["30B_t40"] = 0.004
            q = Path(tmp) / "tampered.json"
            q.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                PR.load_verified(q)
            (Path(tmp) / "outs").mkdir()
            (Path(tmp) / "outs" / "x.json").write_text("{}")
            late = PR.write_record(Path(tmp) / "late.json", outputs_root=Path(tmp) / "outs")
            with self.assertRaises(ValueError):          # frozen after outputs existed
                PR.load_verified(late)


# ---------------------------------------------------------------------------------------------
class ReplayTest(unittest.TestCase):
    def test_inc_replays_its_own_profile_exactly(self):
        prof = E.read_json(R6 / "4B_t64/INC_profile.json")
        inc = E.read_json(R6 / "attn_diag_arms/4B_t64/INC_table.json")
        rp = F.replay_cost(inc, prof)
        self.assertEqual(rp["cost"], prof["total_cycle_macs"] / prof["total_macs"])
        self.assertEqual(rp["ladders_changed"], [])

    def test_replay_tracks_measured_round6_arm_cost(self):
        """First-order replay of round-6 arm A (qk/av ladders [128]+g[1:]) on INC's window-1 profile vs A's
        own measured window-1 cost: 30B within 0.2%, 4B within 1% (moves of +4.5..+8.4%)."""
        for cell, tol in (("4B_t64", 0.01), ("30B_t40", 0.002)):
            prof = E.read_json(R6 / f"{cell}/INC_profile.json")
            t = E.read_json(R6 / f"attn_diag_arms/{cell}/A_table.json")
            rp = F.replay_cost(t, prof)
            self.assertTrue(rp["ladders_changed"])
            pc = E.read_json(R6 / f"{cell}/A_nll.json")["profile_checks"]
            actual = pc["first_window_trace_total_cycle_macs"] / pc["first_window_trace_total_macs"]
            self.assertLess(abs(rp["cost"] / actual - 1), tol, (cell, rp["cost"], actual))
            del prof

    def test_prc_ladder_change_refused(self):
        prof = E.read_json(R6 / "4B_t64/INC_profile.json")
        inc = E.read_json(R6 / "attn_diag_arms/4B_t64/INC_table.json")
        t = copy.deepcopy(inc)
        k = next(iter(t["per_row_chunk"]["buckets"]))
        t["per_row_chunk"]["buckets"][k]["levels"] = [128] * len(t["per_row_chunk"]["buckets"][k]["levels"])
        with self.assertRaises(F.FixedPointError):
            F.replay_cost(t, prof)

    def test_fixed_point_scale(self):
        raw, s1 = F.fixed_point_scale(1.0043, 42.65, 42.40, 1.2)
        self.assertAlmostEqual(raw, 1.0043 * (42.65 - 1.2) / (42.40 - 1.2), places=12)
        self.assertEqual(s1, round(raw, 4))
        self.assertLess(abs(s1 * 1e4 - round(s1 * 1e4)), 1e-6)
        with self.assertRaises(F.FixedPointError):
            F.fixed_point_scale(1.0, 42.0, 30.0, 1.0)      # s1 ~ 1.35: outside the band
        with self.assertRaises(F.FixedPointError):
            F.fixed_point_scale(1.0, 1.0, 2.0, 3.0)        # degenerate


# ---------------------------------------------------------------------------------------------
class Fixture:
    """Synthetic step-0 reference + capture + solves for 4B t64 (dense), built from real round-6 files."""

    def __init__(self, tmp, *, identity_level="exact", identity_ok=True):
        self.tmp = Path(tmp)
        self.cid = "4B_t64"
        self.prereg = PR.write_record(self.tmp / "prereg.json", outputs_root=self.tmp / "no_outputs")
        rec, sha, t = PR.load_verified(self.prereg)
        self.prereg_dict = {"path": str(self.prereg), "sha256": sha, "frozen_utc": rec["frozen_utc"], "frozen_t": t,
                            "mde_nats": rec["mde"]["nats"]}
        s0m = E.read_json(STEP0_MANIFEST)
        c = s0m["cells"][self.cid]
        self.starts = c["windows"]["starts"]
        self.inc_w = E.read_json(c["arms"]["INC"]["wrapper"])
        self.inc_table_path = c["arms"]["INC"]["table"]
        self.inc_t = E.read_json(self.inc_table_path)
        # step-0 root
        self.root = self.tmp / "step0"
        d = self.root / self.cid
        d.mkdir(parents=True)
        shutil.copy(R6 / "4B_t64/INC_profile.json", d / "INC16_profile.json")
        prof = E.read_json(d / "INC16_profile.json")
        self.exact = prof["total_cycle_macs"] / prof["total_macs"]
        dummy_tr = self.tmp / "dummy_trace.json"
        dummy_tr.write_text('{"note": "unit-test stand-in trace"}')
        (d / "step0_summary.json").write_text(json.dumps({"complete": True, "simulated": False,
                                                         "identity": {"restored_incumbent_exact": True}}))
        (d / "INC16_nll.json").write_text(json.dumps({
            "window_starts": self.starts, "wrapper": c["arms"]["INC"]["wrapper"], "cost": self.exact,
            "total_macs": prof["total_macs"], "total_cycle_macs": prof["total_cycle_macs"], "trace": str(dummy_tr),
            "window_nll": E.read_json(R6 / "4B_t64/INC_nll.json")["window_nll"]}))
        self.U = 0.8
        (d / "INC16_audit.json").write_text(json.dumps({"ok": True, "runtime_audit": {
            "U_prot": self.U, "mac_share": {"prot": 0.01}}}))
        self.P = self.exact * 1.004
        (d / "parent_reference.json").write_text(json.dumps({
            "windows": self.starts, "audit_ok": True, "simulated": False, "P": self.P, "trace": str(dummy_tr),
            "trace_sha256": E.sha256_file(dummy_tr)}))
        # capture + identity
        cap = self.tmp / "capture"
        cap.mkdir()
        files = {}
        for k in ("state", "att_dump", "hold_dump"):
            f = cap / f"{k}.bin"
            f.write_bytes(k.encode())
            files[k] = {"path": str(f), "sha256": E.sha256_file(f)}
        self.capture_files = files
        self.identity = cap / "identity.json"
        self.identity.write_text(json.dumps({"ok": identity_ok, "identity_ok": identity_ok,
                                             "identity_level": identity_level, "cell": self.cid, "files": files}))
        self.cm = self.tmp / "capture_manifest.json"
        self.cm.write_text(json.dumps({"cells": [{"id": self.cid, "paths": {
            "identity": str(self.identity), "stem": str(cap / f"{self.cid}_r7.json"), "cell_dir": str(self.tmp / "cell")},
            "calib_args": {"parent": str(E.MP_BEST / "configs/4B/target64")}}]}))
        # arm tables: small allocation moves (a real s0 solve is already near the parent's cost)
        self.tabs = {"UK": uk_table(self.inc_t, buckets=("qk:t0:l1",)), "K": k_table(self.inc_t, f=0.995),
                     "U": uk_table(self.inc_t, buckets=("qk:t0:l1",), tag="r7_u")}

    FROZEN = object()

    def summary(self, name, scales, *, mde=FROZEN, kl=1.0, ka=1.0, preds=(-0.002, -0.008), tables=None, arms=None):
        """A prc_r7_solve.py-shaped solve summary; scales = {arm: target_scale}."""
        out = self.tmp / name
        out.mkdir()
        recs = {}
        for arm in (arms or ("UK", "K", "U")):
            s = scales[arm]
            wp, tp = write_arm(out, f"r7{arm}_s{int(round(s * 1e4)):05d}", self.inc_w, (tables or self.tabs)[arm])
            lin, att = preds
            r = 1.0 if arm == "U" else float(ka) / float(kl)
            recs[f"r7{arm}_k{int(round(r * 1000)):04d}_s{int(round(s * 1e4)):05d}"] = {
                "arm": arm, "ladder_policy": "inherit" if arm == "K" else "up", "kappa_ratio_used": r,
                "target_scale": s, "wrapper": str(wp), "table_path": str(tp), "wrapper_sha256": E.sha256_file(wp),
                "table_sha256": E.sha256_file(tp), "calib_cost_over_target": 1.0,
                "heldout_vs_inc": {"pred_dnll_lin": lin, "pred_dnll_att": att, "cost_over_ref": 1.001},
                "pred_true_vs_inc": float(kl) * float(lin) + float(ka) * float(att),
                "att_mac_hist_hold": hold_hist((tables or self.tabs)[arm], 36)}
        sm = {"tool": "prc_r7_solve.py", "cell": self.cid, "inc_table": self.inc_table_path,
              "inc_table_sha256": E.sha256_file(self.inc_table_path), "capture": self.capture_files,
              "kappa_lin": kl, "kappa_att": ka, "kappa_ratio": ka / kl, "ladder_policy": "up",
              "mde_nats": PR.MDE_NATS[self.cid] if mde is Fixture.FROZEN else mde, "arms": recs}
        p = out / "solve_summary.json"
        p.write_text(json.dumps(sm))
        _later(p)
        return p

    def compute(self, s0_summary, **kw):
        return F.compute(self.cid, s0_summary, out=self.tmp / "fp.json", step0_root=self.root,
                         step0_manifest=STEP0_MANIFEST, capture_manifest=self.cm, prereg_path=self.prereg, **kw)

    def build(self, spec_cell, **kw):
        reg = E.read_json(REG)
        rows = {(r["model"], int(r["target"])): r for r in E.read_json(E.BEST_ALL)["rows"]}
        m6 = E.read_json(E.R6_DIAG_MANIFEST)
        cdef = next(c for c in B.CELLS if c["id"] == self.cid)
        return B.build_cell(cdef, spec_cell, reg, rows, m6, {"ratio": 1.0, "decision": None}, {"passed": True},
                            dry_run=False, do_sweep=False, prereg=self.prereg_dict, step0_root=self.root,
                            capture_manifest=self.cm, step0_manifest=STEP0_MANIFEST, **kw)


class FixedPointPipelineTest(unittest.TestCase):
    """B1: s0 solve -> replay on the step-0 INC16 profile -> per-arm s1 -> the builder accepts only s1 tables."""

    def test_pipeline_dense_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            s0 = PR.S0[fx.cid]
            sp0 = fx.summary("solve_s0", {a: s0 for a in PR.ARMS})
            fp = fx.compute(sp0)
            self.assertEqual(fp["P"], fx.P)
            self.assertEqual(fp["U"], fx.U)
            self.assertEqual(fp["kappa_pins"]["branch"], "dense")
            for arm in PR.ARMS:
                a = fp["arms"][arm]
                self.assertEqual(a["status"], "ok")
                want = F.fixed_point_scale(s0, fx.P, a["C_replay_s0"], fx.U)[1]
                self.assertEqual(a["s1"], want)
            self.assertIn("--target-scales", fp["resolve_command"])
            self.assertIn(f"--mde-nats {PR.MDE_NATS[fx.cid]}", fp["resolve_command"])
            with self.assertRaises(FileExistsError):      # the record is written once
                fx.compute(sp0)
            s1 = {a: fp["arms"][a]["s1"] for a in PR.ARMS}
            sp1 = fx.summary("solve_s1", s1)
            spec = B.spec_from_solve_summaries([(fx.cid, sp1)], None, {fx.cid: fp["_path"]})
            cs = spec["cells"][fx.cid]
            self.assertEqual({a["kind"]: a["target_scale"] for a in cs["arms"]}, s1)
            cell = fx.build(cs)
            self.assertTrue(cell["enabled"], cell.get("disabled_reason"))
            self.assertEqual(cell["primary"], "U")                                     # B2: dense primary kept
            self.assertIn("dense cell", cell["refused"]["UK"])
            self.assertEqual(cell["secondaries"], ["K"])
            self.assertEqual(cell["arms"]["U"]["target_scale"], s1["U"])
            self.assertEqual(cell["arms"]["U"]["mde_nats"], PR.MDE_NATS[fx.cid])
            self.assertEqual(cell["fixed_point"]["s1"]["U"], s1["U"])
            self.assertEqual(cell["parent_cost"]["mode"], "reference")
            self.assertEqual(cell["parent_cost"]["P"], fx.P)
            self.assertEqual(cell["identity"]["screen_reference"]["source"], "round-6 INC run")
            self.assertEqual(cell["capture_identity"]["identity_level"], "exact")
            self.assertIsNotNone(cell["arms"]["U"]["replay_at_s1"])
            self.assertIn(str(fx.identity), cell["hashes"])
            self.assertIn(str(sp1), cell["hashes"])
            # an s0 summary passed alongside the s1 one cannot sneak in a second table for the same arm/scale
            with self.assertRaises(SystemExit):
                B.spec_from_solve_summaries([(fx.cid, sp1), (fx.cid, sp1)], None, {fx.cid: fp["_path"]})

    def test_target_scale_must_be_s1(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            s0 = PR.S0[fx.cid]
            fp = fx.compute(fx.summary("solve_s0", {a: s0 for a in PR.ARMS}))
            wrong = {a: round(fp["arms"][a]["s1"] + 0.0005, 4) for a in PR.ARMS}
            sp = fx.summary("solve_wrong", wrong)
            with self.assertRaises(SystemExit):          # no table at the pre-registered s1
                B.spec_from_solve_summaries([(fx.cid, sp)], None, {fx.cid: fp["_path"]})
            s1 = {a: fp["arms"][a]["s1"] for a in PR.ARMS}
            spec = B.spec_from_solve_summaries([(fx.cid, fx.summary("solve_s1", s1))], None, {fx.cid: fp["_path"]})
            cs = spec["cells"][fx.cid]
            for a in cs["arms"]:
                a["target_scale"] = round(a["target_scale"] + 0.0005, 4)   # hand-edited spec
            cell = fx.build(cs)
            self.assertFalse(cell["enabled"])
            self.assertIn("pre-registered fixed-point s1", cell["refused"]["U"])
            with self.assertRaises(SystemExit):
                B.spec_from_solve_summaries([(fx.cid, fx.summary("solve_s1b", s1))], None, {})  # no fixed point

    def test_mde_and_kappa_pins_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            s0 = PR.S0[fx.cid]
            with self.assertRaises(F.FixedPointError):                    # B4a: MDE other than the frozen one
                fx.compute(fx.summary("mde", {a: s0 for a in PR.ARMS}, mde=0.004))
            with self.assertRaises(F.FixedPointError):                    # B4b: dense cell with 30B pins
                fx.compute(fx.summary("kap", {a: s0 for a in PR.ARMS}, kl=1.96, ka=0.83))
            with self.assertRaises(F.FixedPointError):                    # None MDE (solver default) refused
                fx.compute(fx.summary("mdenone", {a: s0 for a in PR.ARMS}, mde=None))

    def test_absolute_kappa_scale_checked_on_30b(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            keep = PR.kappa_pins_for("30B", "keep_0.42")
            sp = fx.summary("k30", {a: 1.0 for a in PR.ARMS}, kl=1.0, ka=0.42)
            sm = json.loads(sp.read_text())
            sm["cell"], sm["mde_nats"] = "30B_t40", PR.MDE_NATS["30B_t40"]
            bad = F.check_solve_summary(sm, sp, "30B_t40", pins=keep, inc_table=fx.inc_table_path,
                                        identity={"files": fx.capture_files}, prereg_frozen_t=0.0)
            self.assertTrue(any("absolute kappa" in b for b in bad), bad)
            self.assertTrue(any("kappa_ratio_used" in b for b in bad), bad)
            sp2 = fx.summary("k30ok", {a: 1.0 for a in PR.ARMS}, kl=1.96, ka=0.83)
            sm2 = json.loads(sp2.read_text())
            sm2["cell"], sm2["mde_nats"] = "30B_t40", PR.MDE_NATS["30B_t40"]
            self.assertEqual(F.check_solve_summary(sm2, sp2, "30B_t40", pins=keep, inc_table=fx.inc_table_path,
                                                   identity={"files": fx.capture_files}, prereg_frozen_t=0.0), [])
            sm2["arms"][next(iter(sm2["arms"]))]["pred_true_vs_inc"] = -0.05   # pred_true not the pinned formula
            bad2 = F.check_solve_summary(sm2, sp2, "30B_t40", pins=keep, inc_table=fx.inc_table_path,
                                         identity={"files": fx.capture_files}, prereg_frozen_t=0.0)
            self.assertTrue(any("pred_true_vs_inc" in b for b in bad2), bad2)

    def test_summary_must_postdate_prereg_and_match_inc(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            sp = fx.summary("old", {a: PR.S0[fx.cid] for a in PR.ARMS})
            os.utime(sp, (1.0, 1.0))
            with self.assertRaises(F.FixedPointError):
                fx.compute(sp)
            sm = json.loads(sp.read_text())
            bad = F.check_solve_summary(sm, sp, fx.cid, pins=PR.kappa_pins_for("4B", None),
                                        inc_table=str(K / "prc2/4B_t64_c7_gfis_table.json"),
                                        identity={"files": fx.capture_files}, prereg_frozen_t=0.0)
            self.assertTrue(any("inc_table_sha256" in b for b in bad), bad)


class CaptureIdentityTest(unittest.TestCase):
    """B3: no screen from a capture whose identity failed; 'near' is recorded and flagged."""

    def test_failed_identity_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, identity_ok=False, identity_level="fail")
            with self.assertRaises(F.FixedPointError):
                fx.compute(fx.summary("s0", {a: PR.S0[fx.cid] for a in PR.ARMS}))
            with self.assertRaises(F.FixedPointError):
                F.capture_identity(fx.cid, capture_manifest=fx.cm)
            (Path(tmp) / "capture/identity.json").unlink()
            with self.assertRaises(F.FixedPointError):
                F.capture_identity(fx.cid, capture_manifest=fx.cm)

    def test_near_identity_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, identity_level="near")
            fp = fx.compute(fx.summary("s0", {a: PR.S0[fx.cid] for a in PR.ARMS}))
            self.assertTrue(fp["capture_identity"]["near_flag"])
            s1 = {a: fp["arms"][a]["s1"] for a in PR.ARMS}
            spec = B.spec_from_solve_summaries([(fx.cid, fx.summary("s1", s1))], None, {fx.cid: fp["_path"]})
            cell = fx.build(spec["cells"][fx.cid])
            self.assertTrue(cell["enabled"])
            self.assertEqual(cell["capture_identity"]["identity_level"], "near")
            self.assertTrue(any("near" in f for f in cell["flags"]))

    def test_solve_from_another_capture_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            sp = fx.summary("s0", {a: PR.S0[fx.cid] for a in PR.ARMS})
            sm = json.loads(sp.read_text())
            sm["capture"]["state"]["sha256"] = "0" * 64
            bad = F.check_solve_summary(sm, sp, fx.cid, pins=PR.kappa_pins_for("4B", None),
                                        inc_table=fx.inc_table_path, identity={"files": fx.capture_files},
                                        prereg_frozen_t=0.0)
            self.assertTrue(any("identity-checked capture" in b for b in bad), bad)


class CrossSideConsistencyTest(unittest.TestCase):
    """The eval-side kappa rule and pins must give the calib-side solver's branch for every statistic, and the
    calib side must read the same shared registration (one pre-registration, not two)."""

    def test_branch_function_equals_calib_side(self):
        from benchmark.ppl import prc_r7_solve as S
        self.assertEqual(json.loads(json.dumps(S.KAPPA_PINS)), json.loads(json.dumps(PR.KAPPA_PINS)))
        self.assertEqual(S.KAPPA_DECISION_RATIO, PR.KAPPA_DECISION_RATIO)
        self.assertEqual((S.KAPPA_RULE["kappa_att"], S.KAPPA_RULE["kappa_att_se"]),
                         (E.KAPPA_RULE["kappa_att"], E.KAPPA_RULE["kappa_att_se"]))
        self.assertEqual(S.KAPPA_RULE["step0_cells"], E.KAPPA_RULE["cells"])
        cells = E.KAPPA_RULE["cells"]
        wins = list(range(0, 16 * 2048, 2048))
        n = 0
        for m_ in (-0.01, 0.0, 0.005, 0.012, 0.0166, 0.018, 0.02, 0.025, 0.03, 0.04, 0.06):
            for sd in (0.0005, 0.004, 0.02):
                for p_ in (-0.01, 0.005, 0.02):
                    meas = {c: {"window_dnll": [m_ + sd * math.sin(1.7 * i + 0.3 + j) for i in range(16)],
                                "windows": wins, "ok": True} for j, c in enumerate(cells)}
                    mine = E.kappa_decision(meas, {c: p_ for c in cells})
                    rec = {"schema": "prc-step0-r7-v1-kappa", "kappa_lin_bs": mine["kappa_lin_bs"],
                           "se_kappa_lin_bs": mine["se_kappa_lin_bs"], "flags": mine["flags"],
                           "decision": mine["decision"], "simulated_inputs": False}
                    theirs = S.kappa_branch_from_record(rec)
                    self.assertEqual(theirs["decision"], mine["decision"], (m_, sd, p_, mine["kappa_lin_bs"]))
                    self.assertTrue(theirs["record_agrees"])
                    pooled = S.kappa_decision_pooled({c: dict(meas[c], simulated=False) for c in cells},
                                                     {c: p_ for c in cells})
                    self.assertEqual(pooled["decision"], mine["decision"], (m_, sd, p_))
                    if mine["kappa_lin_bs"] is not None:
                        self.assertAlmostEqual(pooled["kappa_lin_bs"], mine["kappa_lin_bs"], places=12)
                    n += 1
        self.assertEqual(n, 99)

    def test_identity_record_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            sp = fx.summary("s0", {a: PR.S0[fx.cid] for a in PR.ARMS})
            ident = F.capture_identity(fx.cid, capture_manifest=fx.cm)
            sm = json.loads(sp.read_text())
            sm["identity"] = {"path": ident["path"], "sha256": ident["sha256"]}
            self.assertEqual(F.check_solve_summary(sm, sp, fx.cid, pins=PR.kappa_pins_for("4B", None),
                                                   inc_table=fx.inc_table_path, identity=ident,
                                                   prereg_frozen_t=0.0, want_primary="U"), [])
            sm2 = dict(sm, identity={"path": ident["path"], "sha256": "f" * 64}, primary_kind="UK",
                       kappa_branch="keep_0.42")
            bad = F.check_solve_summary(sm2, sp, fx.cid, pins=PR.kappa_pins_for("4B", None),
                                        inc_table=fx.inc_table_path, identity=ident, prereg_frozen_t=0.0,
                                        want_primary="U")
            self.assertTrue(any("identity record" in b for b in bad), bad)
            self.assertTrue(any("registered primary" in b for b in bad), bad)
            self.assertTrue(any("kappa branch" in b for b in bad), bad)
            # a re-tagged identity record in the same capture dir (pipeline check failed, identity ok) is accepted
            tagged = Path(tmp) / "capture/identity_fix1.json"
            d = json.loads(fx.identity.read_text())
            tagged.write_text(json.dumps(dict(d, ok=False, identity_ok=True, pipeline_check={"ok": False})))
            it = F.capture_identity(fx.cid, capture_manifest=fx.cm, path=tagged)
            self.assertIs(it["pipeline_check_ok"], False)
            other = Path(tmp) / "elsewhere.json"
            other.write_text(tagged.read_text())
            with self.assertRaises(F.FixedPointError):     # outside the capture directory
                F.capture_identity(fx.cid, capture_manifest=fx.cm, path=other)
            nk = Path(tmp) / "capture/identity_near.json"
            nk.write_text(json.dumps(dict(d, identity_level="near", identity_ok=False, ok=False)))
            with self.assertRaises(F.FixedPointError):     # v2 capture manifest: near is a stop
                F.capture_identity(fx.cid, capture_manifest=fx.cm, path=nk)


class KappaDecisionLoaderTest(unittest.TestCase):
    def _kd(self, tmp, **over):
        m = E.read_json(STEP0_MANIFEST)
        meas = Path(tmp) / "meas.json"
        meas.write_text(json.dumps({"simulated": False}))
        kd = {"schema": "prc-step0-r7-v1-kappa", "rule": m["kappa_rule"], "decision": "keep_0.42", "ratio": 0.42,
              "requires_user_review": False, "flags": [], "simulated_inputs": False, "created": time.time(),
              "measured_files": {"30B_t32": str(meas)}}
        kd.update(over)
        p = Path(tmp) / f"kd_{len(list(Path(tmp).iterdir()))}.json"
        p.write_text(json.dumps(kd))
        return p

    def test_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(F.load_kappa_decision(self._kd(tmp))["decision"], "keep_0.42")
            for bad in ({"schema": "calib"}, {"ratio": 1.0}, {"decision": "third_branch"}, {"simulated_inputs": True},
                        {"rule": dict(E.KAPPA_RULE, kappa_att=0.82)}, {"requires_user_review": True}):
                with self.assertRaises(F.FixedPointError, msg=str(bad)):
                    F.load_kappa_decision(self._kd(tmp, **bad))
            ok = F.load_kappa_decision(self._kd(tmp, requires_user_review=True, flags=["gap_case"]), accept_review=True)
            self.assertTrue(ok["review_accepted"])


class BuilderCliTest(unittest.TestCase):
    def _run(self, *argv):
        env = dict(os.environ, PYTHONPATH=str(REPO / "kernels"), CUDA_VISIBLE_DEVICES="")
        return subprocess.run([PY, str(REPO / "benchmark/ppl/kbands/build_prc_screen_r7_20260928.py"), *argv],
                              cwd=REPO, env=env, capture_output=True, text=True)

    def test_overrides_refused(self):
        r = self._run("--from-solve", "4B_t64=/x.json", "--mde", "4B_t64=0.004")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--mde overrides are refused", r.stderr)
        r = self._run("--from-solve", "30B_t40=/x.json@1.0043")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("PATH@SCALE is refused", r.stderr)


class DedupOrderTest(unittest.TestCase):
    """B2 on the dry path: the primary kind is processed first, so it is never dropped as a duplicate."""

    def _cell(self, cid, arms, gate=True):
        reg = E.read_json(REG)
        rows = {(r["model"], int(r["target"])): r for r in E.read_json(E.BEST_ALL)["rows"]}
        m6 = E.read_json(E.R6_DIAG_MANIFEST)
        cdef = next(c for c in B.CELLS if c["id"] == cid)
        return B.build_cell(cdef, {"arms": arms}, reg, rows, m6, {"ratio": 0.42}, {"passed": gate}, dry_run=True,
                            do_sweep=False)

    def test_4b_uk_equal_u_keeps_u(self):
        w, base = E.read_json(INC4_64), E.read_json(E.resolve_table_path(INC4_64))
        with tempfile.TemporaryDirectory() as tmp:
            arms = []
            for name, tab in (("UK", uk_table(base)), ("K", k_table(base)), ("U", uk_table(base, tag="r7_u"))):
                wp, _ = write_arm(tmp, name, w, tab)
                arms.append({"name": name, "kind": name, "wrapper": str(wp), "pred_true_nats": -0.012,
                             "mde_nats": 0.0073, "kappa_ratio": 1.0})
            cell = self._cell("4B_t64", arms)
            self.assertTrue(cell["enabled"], cell.get("disabled_reason"))
            self.assertEqual(cell["primary"], "U")
            self.assertIn("dense cell", cell["refused"]["UK"])
            self.assertEqual(cell["secondaries"], ["K"])
            # U listed last and identical to K would still be the primary (K dropped as its duplicate)
            arms2 = [dict(arms[1]), dict(arms[2], wrapper=str(write_arm(tmp, "U2", w, k_table(base))[0]))]
            cell2 = self._cell("4B_t64", arms2)
            self.assertEqual(cell2["primary"], "U")
            self.assertIn("allocation-duplicate of U", cell2["refused"]["K"])


# ---------------------------------------------------------------------------------------------
class RunLogTest(unittest.TestCase):
    GOOD = """[prc-ppl] per_row_chunk buckets: 28
[prc-ppl] SC_SCRAMBLE_MASKS=64 HW_MAX=64
[prc-ppl] SC_RNG_GRID=<unset, deployed 128>
[prc-ppl] attn grid: ATTN=<unset>  QK=<unset> AV=<unset>
[prc-ppl] hybrid mask: {hyb}
[prc-ppl] 4B_t64_prc_r7U20260928  model=Qwen/Qwen3-4B-Instruct-2507
[prc-ppl] config={w}
[prc-ppl] frontend=awq  ctx=2048  PPL_MAX_TOKENS=0
"""

    def test_parse(self):
        hyb, w = "/a/hybrid_config.json", "/b/r7U.json"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "run.log"
            p.write_text(self.GOOD.format(hyb=hyb, w=w))
            self.assertEqual(SC.parse_run_log(p, hybrid=hyb, wrapper=w)["failed"], [])
            for old, new in (("frontend=awq", "frontend=smoothquant"), ("MASKS=64 HW", "MASKS=128 HW"),
                             ("<unset, deployed 128>", "pow2"), ("QK=<unset>", "QK=pow2"), (hyb, "/other/h.json")):
                p.write_text(self.GOOD.format(hyb=hyb, w=w).replace(old, new))
                self.assertTrue(SC.parse_run_log(p, hybrid=hyb, wrapper=w)["failed"], old)
            p.write_text(self.GOOD.format(hyb=hyb, w=w) + "[prc-ppl] qk scales: /q.json\n")
            self.assertTrue(any("qk_scales" in f for f in SC.parse_run_log(p, hybrid=hyb, wrapper=w)["failed"]))


class EndToEnd4BTest(unittest.TestCase):
    """Builder (stand-ins, dry run) -> screen simulate for 4B t40 -> finalize on a synthetic test trace."""

    def test_screen_pipeline_4b(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=str(REPO / "kernels"), CUDA_VISIBLE_DEVICES="")
            man = Path(tmp) / "manifest.json"
            r0 = subprocess.run([PY, str(REPO / "benchmark/ppl/kbands/build_prc_screen_r7_20260928.py"),
                                 "--standins", tmp, "--assume-gate", "30B_t32=fail", "--assume-gate", "30B_t40=pass",
                                 "--assume-gate", "4B_t40=pass", "--assume-gate", "4B_t64=pass",
                                 "--assume-kappa-ratio", "0.42", "--enable-4b", "--skip-sweep", "--out", str(man)],
                                cwd=REPO, env=env, capture_output=True, text=True)
            self.assertEqual(r0.returncode, 0, r0.stderr[-3000:])
            m = SC.load_manifest(man)
            self.assertTrue(m["dry_run"])
            cells = m["cells"]
            self.assertEqual(cells["30B_t32"]["primary"], "K")
            self.assertEqual(cells["30B_t40"]["primary"], "UK")
            self.assertEqual(cells["30B_t48"]["primary"], "UK")   # follows 30B t40's gate
            self.assertEqual(cells["4B_t40"]["primary"], "U")
            self.assertEqual(cells["4B_t64"]["primary"], "U")     # B2: a dense UK stand-in never disables U
            self.assertIn("dense cell", cells["4B_t64"]["refused"]["UK"])
            self.assertEqual(cells["30B_t48"]["windows"]["screen"], E.read_json(REG)["screen16_fresh_t48"])
            self.assertIsNone(cells["30B_t48"]["identity"]["screen_reference"])
            self.assertIsNotNone(cells["30B_t40"]["identity"]["screen_reference"])
            out_root = Path(tmp) / "screen"
            r = subprocess.run([PY, str(REPO / "benchmark/ppl/prc_screen_r7_20260928.py"), "simulate",
                                "--manifest", str(man), "--task", "3", "--out-root", str(out_root)],
                               cwd=REPO, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            d = out_root / "4B_t40"
            sel = E.read_json(d / "selected.json")
            scr = E.read_json(d / "screen_result.json")
            self.assertTrue(scr["screen_gate"]["proceed_to_confirm"], scr["screen_gate"])
            self.assertTrue(sel["evaluate_full_test"], sel)
            self.assertTrue(all(a["audit_ok"] for a in scr["arms"].values()))
            # without the simulation-only P override, stand-ins fail the cost gate -> no confirm, no test
            m2 = json.loads(man.read_text())
            m2["cells"]["4B_t40"]["simulation"]["force_cost_gate_pass"] = False
            man2 = Path(tmp) / "manifest_nogate.json"
            man2.write_text(json.dumps(m2))
            r3 = subprocess.run([PY, str(REPO / "benchmark/ppl/prc_screen_r7_20260928.py"), "simulate",
                                 "--manifest", str(man2), "--task", "3", "--out-root", str(Path(tmp) / "screen2")],
                                cwd=REPO, env=env, capture_output=True, text=True)
            self.assertEqual(r3.returncode, 0, r3.stderr[-3000:])
            sel2 = E.read_json(Path(tmp) / "screen2/4B_t40/selected.json")
            self.assertFalse(sel2["evaluate_full_test"])
            self.assertIn("cost_gate_ok", sel2["reason"])
            self.assertFalse((Path(tmp) / "screen2/4B_t40/confirm_result.json").exists())
            # finalize on a synthetic full-test trace of the primary (MACs = INC full test)
            cell = copy.deepcopy(cells["4B_t40"])
            pa = cell["arms"][cell["primary"]]
            inc_ft = E.load_trace(cell["incumbent"]["full_test_trace"])
            syn = E.synthesize_trace(inc_ft, E.read_json(cell["arms"]["INC"]["table"]),
                                     E.read_json(cell["arms"]["INC"]["wrapper"]), E.read_json(pa["table"]),
                                     E.read_json(pa["wrapper"]), 36,
                                     header={"mp_config_json": pa["wrapper"], "ppl": 10.70})
            tt = Path(tmp) / "test_trace.json"
            tt.write_text(json.dumps(syn))
            pa["test_trace"] = str(tt)
            rec = SC.finalize(m, cell, d, tt, allow_simulated=True)
            self.assertTrue(rec["protocol_ok"], rec["failed_checks"])
            self.assertLess(rec["ppl_vs_incumbent_pct"], 0)
            # N4: the stand-in spends ~+5.9% vs the parent on test -> flagged, NOT entered into best(all)
            self.assertGreater(rec["test_cost_vs_parent_pct"], 1.5)
            self.assertFalse(rec["best_all"]["eligible"])
            self.assertEqual(rec["best_all"]["new_best_arm"], cell["incumbent"]["arm"])
            self.assertEqual(rec["best_all"]["new_best_ppl"], cell["incumbent"]["full_test_ppl"])
            self.assertTrue(any("NOT entered" in f for f in rec["flags"]))
            self.assertIn("budget correction", rec["budget_correction"])
            # at parent cost (P := this cost) the same result would be eligible and enter best(all)
            cell_p = copy.deepcopy(cell)
            cell_p["parent"]["test_cost"] = rec["cost"]
            for f in ("full_test_result.json", "test_histogram.json"):
                (d / f).unlink()
            rec_p = SC.finalize(m, cell_p, d, tt, allow_simulated=True)
            self.assertTrue(rec_p["best_all"]["eligible"])
            self.assertEqual(rec_p["best_all"]["new_best_arm"], "r7_U")
            # a finalize with a mismatched wrapper header must fail its protocol check (and never enter)
            syn["header"]["mp_config_json"] = cell["arms"]["INC"]["wrapper"]
            tt2 = Path(tmp) / "test_trace2.json"
            tt2.write_text(json.dumps(syn))
            pa["test_trace"] = str(tt2)
            cell_p["arms"][cell["primary"]]["test_trace"] = str(tt2)
            for f in ("full_test_result.json", "test_histogram.json"):
                (d / f).unlink()
            rec2 = SC.finalize(m, cell_p, d, tt2, allow_simulated=True)
            self.assertIn("mp_config_json", rec2["failed_checks"])
            self.assertFalse(rec2["best_all"]["eligible"])
            self.assertEqual(rec2["best_all"]["new_best_ppl"], cell["incumbent"]["full_test_ppl"])


if __name__ == "__main__":
    unittest.main()
