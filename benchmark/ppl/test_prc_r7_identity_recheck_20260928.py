"""CPU tests for prc_r7_identity_recheck_20260928 (round-7 capture identity, corrected check 7').

  cd /home/allenjin/Projects/SCMP/scmp_llm && PYTHONPATH=$PWD/kernels \
      /nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python -m unittest \
      benchmark.ppl.test_prc_r7_identity_recheck_20260928 -v

Fixture. The synthetic 512-row capture comes from the pinned test module's build_capture. It is
rewritten into the diag layout calib10_r7 really writes:
  * tables = parent, wfis, gfis, c17, c17_s80, inc, and no gfisla;
  * joint = parent, gfis, gfisla.
The wfis table and its diag row come from the verbatim calib9 held-out code (CALIB9_diag_lin).

What is proven here:
  * on that layout, the pinned check (7) fails with exactly tables.gfisla = None, as on 30B_t32;
  * check (7') passes;
  * a genuinely wrong table still fails (7'). This covers gfisla linear thresholds, gfisla
    attention thresholds, a score table, the supplementary wfis table, and a tampered hold row
    outside the 128-row prefix. It also covers every diag-layout guard;
  * the end-to-end re-check writes identity_<tag>.json and restores the pinned function. The
    readers accept the record: prc_r7_solve.read_identity,
    prc_fixedpoint_r7_20260928.capture_identity and
    prc_step0_r7_20260928.capture_identity_status;
  * a re-run that does not reproduce the original record is written as identity_ok = false;
  * the structural fact holds: calib10 writes diag['tables'] before gfisla exists.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.ppl import prc_r7_capture_20260928 as D
from benchmark.ppl import prc_r7_identity_recheck_20260928 as R
from benchmark.ppl import prc_r7_solve as S
from benchmark.ppl import test_prc_r7_calib_20260928 as T   # pinned fixture (module import only)

PARENT = T.PARENT_4B_T40
CALIB10 = Path(T.CALIB10)
REAL_CAPTURE = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r7_20260928/30B_t32/capture")


def _try_import(name):
    try:
        return __import__(f"benchmark.ppl.{name}", fromlist=["x"])
    except Exception:  # noqa: BLE001  (a concurrently edited module may not import)
        return None


def to_calib10_layout(p, seed=5):
    """Rewrite build_capture's diag into calib10's real layout (see module docstring). Emits a
    wfis table next to the capture's own tables; returns the wfis thresholds."""
    cap = T.load_capture(p)
    assert Path(cap.r7["files"]["diag"]) == Path(p["capd"])
    rng = np.random.default_rng(seed)
    th_w = {k: sorted(float(np.float32(x)) for x in rng.uniform(0, 1, len(cap.ladder) - 1))
            for k in cap.keys}
    T.c9.emit_table(p["stem"], "wfis", th_w, None, p["table"], p["wrapper"], cap.ladder,
                    {"table": "wfis"})
    store = {k: [{f: v for f, v in d.items()}] for k, d in cap.hold_lin.items()}
    dl = T.CALIB9_diag_lin(store, cap.keys, {"wfis": th_w}, {}, cap.ladder, cap.li, cap.top_col,
                           cap.n_tok)
    capd = json.loads(Path(p["capd"]).read_text())
    assert "gfisla" in capd["tables"]            # the pinned fixture fabricates it
    tables = {"parent": {"pred_dnll_fis": 0.0}, "wfis": {"pred_dnll_fis": dl["wfis"]}}
    tables.update({n: v for n, v in capd["tables"].items() if n != "gfisla"})
    capd["tables"] = tables
    Path(p["capd"]).write_text(json.dumps(capd, indent=1))
    return th_w


@contextlib.contextmanager
def diag_variant(p, mutate):
    """Temporarily rewrite the capture diag (its path is baked into the solve state)."""
    path = Path(p["capd"])
    orig = path.read_text()
    d = json.loads(orig)
    mutate(d)
    path.write_text(json.dumps(d, indent=1))
    try:
        yield
    finally:
        path.write_text(orig)


def copy_stem(p, dst: Path, tamper=None):
    """Copy the capture's own tables (gfis, gfisla, wfis) under a new stem; tamper(name, table)
    may edit one. Returns the new stem."""
    stem = Path(p["stem"])
    new = dst / stem.name
    for nm in ("gfis", "gfisla", "wfis"):
        src = stem.with_name(f"{stem.stem}_{nm}_table.json")
        if not src.is_file():
            continue
        t = json.loads(src.read_text())
        if tamper:
            tamper(nm, t)
        new.with_name(f"{new.stem}_{nm}_table.json").write_text(json.dumps(t))
    return new


def zero_prc(t):
    for e in t["per_row_chunk"]["buckets"].values():
        e["thresholds"] = [0.0] * len(e["thresholds"])


class TestCheck7Prime(unittest.TestCase):
    """check (7') on a synthetic capture in calib10's real diag layout."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.p = T.build_capture(cls.d, PARENT)
        cls.th_w = to_calib10_layout(cls.p)
        cls.cap = T.load_capture(cls.p)
        cls.st = cls.p["score_tables"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pinned(self, stem=None, st=None):
        return R.PINNED_OWN_TABLE_SCORES(self.cap, stem or self.p["stem"], st or self.st)

    def prime(self, stem=None, st=None):
        stem = stem or self.p["stem"]
        st = st or self.st
        return R.correct_check7(self.pinned(stem, st), self.cap, stem, st)

    def test_pinned_check_fails_exactly_on_tables_gfisla(self):
        """The 30B_t32 failure, reproduced: only tables.gfisla is False, its reference None."""
        own = self.pinned()
        self.assertFalse(own["ok"])
        self.assertEqual(own["missing_own_tables"], [])
        self.assertIsNone(own["capture_diag"]["tables.gfisla"])
        self.assertEqual([n for n, v in own["equal"].items() if not v], ["tables.gfisla"])
        self.assertEqual(sorted(own["equal"]), sorted(["joint.gfis", "joint.gfisla", "tables.gfis",
                                                       "tables.gfisla", "tables.c17",
                                                       "tables.c17_s80", "tables.inc"]))

    def test_corrected_check_passes(self):
        c = self.prime()
        self.assertTrue(c["ok"], c["problems"])
        self.assertEqual(list(c["exempt"]), ["tables.gfisla"])
        self.assertEqual(c["absent_references"], ["tables.gfisla"])
        self.assertNotIn("tables.gfisla", c["equal"])
        self.assertTrue(c["equal"]["joint.gfisla"])
        self.assertEqual(list(c["supplementary"]), ["tables.wfis"])     # the row pinned (7) skips
        self.assertTrue(c["supplementary"]["tables.wfis"]["equal"])
        self.assertFalse(c["pinned_check"]["ok"])                       # kept verbatim
        self.assertEqual(c["pinned_check"], self.pinned())

    def test_wrong_gfisla_linear_thresholds_fail(self):
        """A genuinely wrong gfisla table still fails although tables.gfisla is exempt: its
        linear keys are part of the joint total, compared bit for bit."""
        with tempfile.TemporaryDirectory() as e_:
            stem = copy_stem(self.p, Path(e_), lambda nm, t: zero_prc(t) if nm == "gfisla" else None)
            c = self.prime(stem)
            self.assertFalse(c["ok"])
            self.assertFalse(c["pinned_check"]["equal"]["joint.gfisla"])
            self.assertEqual(c["exempt"], {})
            self.assertTrue(any("joint.gfisla" in s for s in c["problems"]), c["problems"])

    def test_wrong_gfisla_attention_thresholds_fail(self):
        def tamper(nm, t):
            if nm == "gfisla":
                for kk, b in t["buckets"].items():
                    if kk.startswith(("qk:", "av:")):
                        b["thresholds"] = [0.0] * len(b["thresholds"])
        with tempfile.TemporaryDirectory() as e_:
            c = self.prime(copy_stem(self.p, Path(e_), tamper))
            self.assertFalse(c["ok"])
            self.assertFalse(c["pinned_check"]["equal"]["joint.gfisla"])
            self.assertTrue(c["pinned_check"]["equal"]["tables.gfis"])   # linears untouched

    def test_wrong_score_table_fails(self):
        with tempfile.TemporaryDirectory() as e_:
            t = json.loads(Path(self.st["c17"]).read_text())
            zero_prc(t)
            bad = Path(e_) / "c17_table.json"
            bad.write_text(json.dumps(t))
            st = dict(self.st, c17=bad)
            c = self.prime(st=st)
            self.assertFalse(c["ok"])
            self.assertIn("tables.c17", " ".join(c["problems"]))

    def test_wrong_wfis_fails(self):
        with tempfile.TemporaryDirectory() as e_:
            stem = copy_stem(self.p, Path(e_), lambda nm, t: zero_prc(t) if nm == "wfis" else None)
            c = self.prime(stem)
            self.assertFalse(c["ok"])
            self.assertFalse(c["supplementary"]["tables.wfis"]["equal"])
            self.assertTrue(c["pinned_check"]["equal"]["joint.gfisla"])

    def test_missing_own_table_fails(self):
        with tempfile.TemporaryDirectory() as e_:
            stem = copy_stem(self.p, Path(e_))
            stem.with_name(f"{stem.stem}_gfisla_table.json").unlink()
            c = self.prime(stem)
            self.assertFalse(c["ok"])
            self.assertEqual(c["missing_own_tables"], ["gfisla"])

    def test_diag_layout_guards(self):
        cases = {
            "null gfisla key": lambda d: d["tables"].__setitem__("gfisla", {"pred_dnll_fis": None}),
            "no joint gfisla": lambda d: d["joint"].pop("gfisla"),
            "no tables.c17": lambda d: d["tables"].pop("c17"),
            "wrong tables.gfisla": lambda d: d["tables"].__setitem__("gfisla", {"pred_dnll_fis": -1.0}),
            "row without a table": lambda d: d["tables"].__setitem__("foo", {"pred_dnll_fis": 0.1}),
            "parent row not 0": lambda d: d["tables"].__setitem__("parent", {"pred_dnll_fis": 1e-9}),
            "joint gfisla off by 1 ulp": lambda d: d["joint"]["gfisla"].__setitem__(
                "pred_dnll", float(np.nextafter(d["joint"]["gfisla"]["pred_dnll"], 1.0))),
            "wfis row off by 1 ulp": lambda d: d["tables"]["wfis"].__setitem__(
                "pred_dnll_fis", float(np.nextafter(d["tables"]["wfis"]["pred_dnll_fis"], 1.0))),
        }
        for name, mut in cases.items():
            with self.subTest(name), diag_variant(self.p, mut):
                c = self.prime()
                self.assertFalse(c["ok"], name)
                self.assertTrue(c["problems"], name)

    def test_correct_tables_gfisla_reference_is_compared_not_exempted(self):
        """If a calibrator ever writes tables.gfisla, (7') compares it (no exemption)."""
        own = self.pinned()
        val = own["capture"]["tables.gfisla"]
        with diag_variant(self.p, lambda d: d["tables"].__setitem__("gfisla", {"pred_dnll_fis": val})):
            c = self.prime()
            self.assertTrue(c["ok"], c["problems"])
            self.assertEqual(c["exempt"], {})
            self.assertTrue(c["equal"]["tables.gfisla"])
            self.assertTrue(c["pinned_check"]["ok"])      # the pinned check passes on that layout

    def test_tampered_hold_row_outside_prefix_fails(self):
        """A hold attention row outside the 128 prefix is invisible to the c7 comparisons; the
        joint.gfisla re-score still catches it under (7')."""
        with tempfile.TemporaryDirectory() as e_:
            z = dict(np.load(self.p["att"]))
            key = next(f for f in z if f.startswith("att|hold|av|") and f.endswith("|fis"))
            rank = z[key[:-len("fis")] + "rank"]
            row = int(np.nonzero(rank >= 128)[0][3])
            z[key] = z[key].copy()
            z[key][row, :] *= np.float32(1.5)
            bad = Path(e_) / "att.npz"
            np.savez_compressed(bad, **z)
            cap = S.Capture(self.p["state"], bad, self.p["hold"])
            own = R.PINNED_OWN_TABLE_SCORES(cap, self.p["stem"], self.st)
            c = R.correct_check7(own, cap, self.p["stem"], self.st)
            self.assertFalse(c["ok"])
            self.assertFalse(own["equal"]["joint.gfisla"])

    def test_swap_is_restored_even_on_error(self):
        with self.assertRaises(ZeroDivisionError):
            with R.corrected_check7_installed():
                self.assertIsNot(S.own_table_scores, R.PINNED_OWN_TABLE_SCORES)
                raise ZeroDivisionError
        self.assertIs(S.own_table_scores, R.PINNED_OWN_TABLE_SCORES)


class TestRecheckEndToEnd(unittest.TestCase):
    """Driver-level: pinned identity (fails on the real layout) -> re-check -> readers."""

    @classmethod
    def setUpClass(cls):
        os.environ.pop("R7C_MANIFEST", None)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_cell(self, name, *, seed=0, tamper_hold=False):
        d = self.root / name
        d.mkdir()
        p = T.build_capture(d, PARENT, seed=seed)
        to_calib10_layout(p)
        att = p["att"]
        if tamper_hold:
            z = dict(np.load(att))
            key = next(f for f in z if f.startswith("att|hold|qk|") and f.endswith("|fis"))
            rank = z[key[:-len("fis")] + "rank"]
            row = int(np.nonzero(rank >= 128)[0][5])
            z[key] = z[key].copy()
            z[key][row, :] *= np.float32(2.0)
            att = d / "att_tampered.npz"
            np.savez_compressed(att, **z)
        capdir = d / "capture"
        capdir.mkdir()
        cell = {"id": "4B_t40", "index": 0, "enabled": True,
                "paths": {"state": str(p["state"]), "att_dump": str(att), "hold_dump": str(p["hold"]),
                          "stem": str(p["stem"]), "diag": str(p["capd"]), "cell_dir": str(d),
                          "calib_log": str(capdir / "calib10.log"),
                          "identity": str(capdir / "identity.json"),
                          "pipeline_dir": str(capdir / "pipeline_check")},
                "refs": {"c7_gfis_table": str(p["ref_gfis"]), "c7_gfisla_table": str(p["ref_gfisla"]),
                         "c7_diag": str(p["ref_diag"]), "inc_table": str(p["inc"]),
                         "expect_joint_line": p["joint_line"], "expect_global_line": "x",
                         "inc_relation": "identical"},
                "r7": {"att_prefix_split": 128, "att_rows_per_call": 512,
                       "score_tables": {k: str(v) for k, v in p["score_tables"].items()}},
                "calib_args": {"parent": str(PARENT)}}
        man = {"schema": D.SCHEMA, "preregistered_rules": {"identity_allow_near": False},
               "cells": [cell]}
        mf = d / "manifest.json"
        mf.write_text(json.dumps(man, indent=1))
        man["_path"] = str(mf)
        return man, cell, p

    def recheck(self, man, cell, **kw):
        return R.recheck(man, cell, run_preflight=False, require_pin=False, **kw)

    def test_full_recheck_and_acceptance(self):
        man, cell, p = self.make_cell("good")
        self.assertEqual(D.run_identity(man, cell), 3)                  # the pinned driver: fail
        orig = json.loads(Path(cell["paths"]["identity"]).read_text())
        failed = [k for g in ("cpu_exact", "exact") for k, v in orig["checks"][g].items() if not v]
        self.assertEqual(failed, [R.ORIG_KEY])
        self.assertEqual(orig["identity_level"], "fail")
        before = Path(cell["paths"]["identity"]).read_bytes()
        rc, rep, out = self.recheck(man, cell)
        self.assertEqual(rc, 0)
        self.assertEqual(out.name, "identity_check7fix.json")
        self.assertEqual(Path(cell["paths"]["identity"]).read_bytes(), before)   # never touched
        self.assertTrue((out.parent / "pipeline_check_check7fix").is_dir())
        self.assertIs(S.own_table_scores, R.PINNED_OWN_TABLE_SCORES)
        disk = json.loads(out.read_text())
        self.assertTrue(disk["identity_ok"] and disk["ok"])
        self.assertEqual(disk["identity_level"], "exact")
        self.assertEqual(disk["tag"], "check7fix")
        self.assertEqual(disk["cell"], "4B_t40")
        self.assertNotIn(R.ORIG_KEY, disk["checks"]["cpu_exact"])
        self.assertTrue(disk["checks"]["cpu_exact"][R.NEW_KEY])
        self.assertTrue(all(disk["checks"]["cpu_exact"].values()))
        self.assertTrue(all(disk["checks"]["exact"].values()))
        rc7 = disk["recheck"]
        self.assertFalse(rc7["original_failure"]["value"])
        self.assertEqual(rc7["original_failure"]["absent_references"], ["tables.gfisla"])
        self.assertTrue(rc7["corrected_check"]["value"])
        self.assertEqual(list(rc7["corrected_check"]["exempt"]), ["tables.gfisla"])
        self.assertTrue(rc7["reproduction_of_original"]["ok"], rc7["reproduction_of_original"])
        self.assertEqual(rc7["original_identity"]["failed_checks"], [f"cpu_exact.{R.ORIG_KEY}"])
        self.assertEqual(disk["notes"]["own_table_scores"]["pinned_check"],
                         orig["notes"]["own_table_scores"])
        pi = rc7["reproduction_of_original"]["pipeline_check_informational"]
        self.assertTrue(pi["record_equal_modulo_dir"])
        self.assertEqual(disk["files"], orig["files"])
        # readers
        cap = S.Capture(cell["paths"]["state"], cell["paths"]["att_dump"], cell["paths"]["hold_dump"])
        got = S.read_identity(out, cap, "4B_t40")
        self.assertEqual(got["identity_level"], "exact")
        with self.assertRaises(SystemExit):
            S.read_identity(cell["paths"]["identity"], cap, "4B_t40")
        P0 = _try_import("prc_step0_r7_20260928")
        if P0 is not None:
            self.assertTrue(P0.capture_identity_status(out)["ok"])
            self.assertFalse(P0.capture_identity_status(cell["paths"]["identity"])["ok"])
        F = _try_import("prc_fixedpoint_r7_20260928")
        if F is None:
            self.skipTest("prc_fixedpoint_r7_20260928 does not import (concurrently edited?)")
        r = F.capture_identity("4B_t40", capture_manifest=Path(man["_path"]), path=out)
        self.assertTrue(r["ok"])
        self.assertEqual(r["identity_level"], "exact")
        with self.assertRaises(F.FixedPointError):
            F.capture_identity("4B_t40", capture_manifest=Path(man["_path"]),
                               path=cell["paths"]["identity"])
        v = R.verify(cell, out, capture_manifest=man["_path"])
        self.assertTrue(all("refused" not in x for x in v.values()), v)
        # no overwrite, and the untagged path is refused
        with self.assertRaises(SystemExit):
            self.recheck(man, cell)
        with self.assertRaises(SystemExit):
            self.recheck(man, cell, tag="")

    def test_recheck_still_fails_a_wrong_capture(self):
        """A hold row tampered outside the 128 prefix: the pinned record fails, and the re-check
        fails too (identity_ok false, not accepted by any reader)."""
        man, cell, p = self.make_cell("bad", seed=2, tamper_hold=True)
        self.assertEqual(D.run_identity(man, cell), 3)
        rc, rep, out = self.recheck(man, cell)
        self.assertEqual(rc, 3)
        disk = json.loads(out.read_text())
        self.assertFalse(disk["identity_ok"])
        self.assertEqual(disk["identity_level"], "fail")
        self.assertFalse(disk["checks"]["cpu_exact"][R.NEW_KEY])
        self.assertTrue(disk["recheck"]["reproduction_of_original"]["ok"])
        self.assertTrue(any("joint.gfisla" in s for s in disk["recheck"]["corrected_check"]["problems"]))
        cap = S.Capture(cell["paths"]["state"], cell["paths"]["att_dump"], cell["paths"]["hold_dump"])
        with self.assertRaises(SystemExit):
            S.read_identity(out, cap, "4B_t40")
        F = _try_import("prc_fixedpoint_r7_20260928")
        if F is not None:
            with self.assertRaises(F.FixedPointError):
                F.capture_identity("4B_t40", capture_manifest=Path(man["_path"]), path=out)

    def test_recheck_refuses_when_the_original_is_not_reproduced(self):
        man, cell, p = self.make_cell("repro", seed=4)
        self.assertEqual(D.run_identity(man, cell), 3)
        orig = json.loads(Path(cell["paths"]["identity"]).read_text())
        for i, mut in enumerate((
                lambda o: o["notes"]["pred_abs_diff"].__setitem__("joint_gfis", 1e-12),
                lambda o: o["checks"]["exact"].__setitem__("gfis_table_equals_c7_gfis", False),
                lambda o: o["files"]["state"].__setitem__("sha256", "0" * 64),
                lambda o: o["notes"]["own_table_scores"]["capture"].__setitem__("tables.c17", 0.0))):
            o = copy.deepcopy(orig)
            mut(o)
            alt = Path(cell["paths"]["cell_dir"]) / f"orig_{i}.json"
            alt.write_text(json.dumps(o))
            rc, rep, out = self.recheck(man, cell, tag=f"repro{i}", original_identity=alt)
            with self.subTest(i):
                self.assertEqual(rc, 3)
                disk = json.loads(out.read_text())
                self.assertFalse(disk["identity_ok"])
                self.assertEqual(disk["identity_level"], "fail")
                self.assertFalse(disk["recheck"]["reproduction_of_original"]["ok"])
                self.assertTrue(disk["checks"]["cpu_exact"][R.NEW_KEY])   # (7') itself passed

    def test_original_ok_is_left_alone(self):
        man, cell, p = self.make_cell("already", seed=6)
        Path(cell["paths"]["identity"]).write_text(json.dumps({"identity_ok": True,
                                                               "identity_level": "exact"}))
        rc, rep, out = self.recheck(man, cell)
        self.assertEqual((rc, out), (0, None))
        self.assertFalse(D.identity_paths(cell, R.DEFAULT_TAG)[0].exists())


class TestStructuralFacts(unittest.TestCase):
    """The root cause, read from the pinned calibrator and the real captures (read-only)."""

    def test_calib10_writes_diag_tables_before_gfisla_exists(self):
        src = CALIB10.read_text().splitlines()
        writers = [i for i, s in enumerate(src, 1) if re.search(r'diag\["tables"\]\[[^\]]+\]\s*=', s)]
        loop = [i for i, s in enumerate(src, 1) if s.strip() == 'for tname in ["parent"] + list(tables):']
        add = [i for i, s in enumerate(src, 1) if s.strip().startswith("tables[jname] = ")]
        jn = [i for i, s in enumerate(src, 1) if s.strip().startswith('jname = f"gfisla')]
        self.assertEqual(len(writers), 1)
        self.assertEqual(len(loop), 1)
        self.assertEqual(len(add), 1)
        self.assertEqual(len(jn), 1)
        self.assertLess(loop[0], writers[0])
        self.assertLess(writers[0], add[0])        # diag.tables is complete before gfisla exists

    def test_pinned_fixture_fabricates_tables_gfisla(self):
        src = Path(T.__file__).read_text()
        self.assertIn('"gfisla": lin10', src)

    @unittest.skipUnless((REAL_CAPTURE / "30B_t32_r7_diag.json").is_file(), "real 30B_t32 capture absent")
    def test_real_30b_t32_diag_layout(self):
        d = json.loads((REAL_CAPTURE / "30B_t32_r7_diag.json").read_text())
        self.assertEqual(list(d["tables"]), ["parent", "wfis", "gfis", "c17", "c17_s80", "inc"])
        self.assertIn("gfisla", d["joint"])
        c7 = json.loads(Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2/30B_t32_c7_diag.json")
                        .read_text())
        self.assertNotIn("gfisla", c7["tables"])
        orig = json.loads((REAL_CAPTURE / "identity.json").read_text())
        own = orig["notes"]["own_table_scores"]
        self.assertEqual([n for n, v in own["equal"].items() if not v], ["tables.gfisla"])
        self.assertIsNone(own["capture_diag"]["tables.gfisla"])

    @unittest.skipUnless((REAL_CAPTURE / f"identity_{R.DEFAULT_TAG}.json").is_file(),
                         "30B_t32 re-check record not written yet")
    def test_real_30b_t32_recheck_record(self):
        rec = json.loads((REAL_CAPTURE / f"identity_{R.DEFAULT_TAG}.json").read_text())
        self.assertTrue(rec["identity_ok"] and rec["ok"])
        self.assertEqual(rec["identity_level"], "exact")
        self.assertEqual(rec["cell"], "30B_t32")
        rc7 = rec["recheck"]
        self.assertFalse(rc7["original_failure"]["value"])
        self.assertEqual(list(rc7["corrected_check"]["exempt"]), ["tables.gfisla"])
        self.assertEqual(list(rc7["corrected_check"]["supplementary"]), ["tables.wfis"])
        self.assertTrue(rc7["reproduction_of_original"]["ok"])
        orig = json.loads((REAL_CAPTURE / "identity.json").read_text())
        self.assertEqual(rec["files"], orig["files"])
        self.assertEqual(rec["notes"]["own_table_scores"]["pinned_check"], orig["notes"]["own_table_scores"])


if __name__ == "__main__":
    unittest.main()
