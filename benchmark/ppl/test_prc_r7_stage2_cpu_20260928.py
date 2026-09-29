"""End-to-end CPU test of the round-7 STAGE-2 chain (benchmark/ppl/kbands/run_r7_stage2_cpu_20260928.py).

Every registered step runs for real on a scratch round-7 root: the step-0 kappa stage CLI, prc_r7_solve.py solve at
s0, prc_fixedpoint_r7_20260928.py compute, the fixed point's printed re-solve command (verbatim), the screen builder
(with its resolver sweeps) and the screen launcher's dry run (driver deep preflight, incl. the realized-length
prediction re-derivation). Nothing is simulated inside the tools: the FIXTURE fabricates the GPU outputs they read.

TEST FIXTURE (scratch only, never under the Turbo round-7 root or the repo):
  * captures: the calib side's synthetic calib10 capture (test_prc_r7_calib_20260928.build_capture) on each cell's
    REAL parent (48 blocks) and REAL incumbent; its attention error curves are replaced by a monotone, metric-
    correlated err = w(mn)/L^2 so the solve spends its target (gain 30 -> eligible arms; the raw capture -> refused
    by sign); identity.json identity_ok / exact; capture diags carry the kappa predictions p_c;
  * step 0: fabricated, labelled GPU-style records on the registered windows: 30B t32/t40 use the REAL round-6 INC
    first-window profiles; 30B t48's profile is the t40 profile re-dispatched through the t48 incumbent (so the INC
    replays its own profile exactly); P is set from an in-process pre-solve so s1 lands in (or, on purpose, outside)
    the registered band; step0_measured m_c gives a keep (or a flagged gap-case) kappa decision;
  * Slurm: fake squeue / sacct on PATH (read-only answers), and a fake sbatch that records any call (must never run).
  cd scmp_llm && PYTHONPATH=kernels <annstention python> -m unittest benchmark.ppl.test_prc_r7_stage2_cpu_20260928 -v
(about 15 minutes; the login-node cgroup cap of 4 GB is respected: one solver process at a time.)
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from benchmark.ppl import prc_eval_r7_20260928 as E
from benchmark.ppl import prc_fixedpoint_r7_20260928 as F
from benchmark.ppl import prc_r7_prereg_20260928 as PR
from benchmark.ppl import prc_r7_solve as S
from benchmark.ppl.prc_local_proposals import _group_cost
from benchmark.ppl.test_prc_r7_calib_20260928 import build_capture

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
KDIR = REPO / "benchmark/ppl/kbands"
CHAIN = KDIR / "run_r7_stage2_cpu_20260928.py"
STEP0_MANIFEST = KDIR / "prc_step0_r7_20260928.json"
CAPTURE_MANIFEST = KDIR / "prc_r7_capture_20260928.json"
SCREEN_SBATCH = "benchmark/ppl/kbands/run_prc_screen_r7_20260928.sbatch"
R6 = E.KB / "prc_r6_20260928"
CELLS = ("30B_t32", "30B_t40", "30B_t48")
U_PROT = {"30B_t32": 0.8975882654809834, "30B_t40": 0.7651811644230064, "30B_t48": 1.2559891836113883}
KAPPA_P = {"30B_t32": 0.009, "30B_t40": 0.006}
KEEP_M = {"30B_t32": 0.0225, "30B_t40": 0.015}      # kappa_lin_bs 2.5 >> 0.83 + band: keep, no flag
GAP_M = {"30B_t32": 0.0108, "30B_t40": 0.0072}      # kappa_lin_bs 1.2: keep but gap_case -> requires_user_review
SACCT_OK = "\n".join(f"{j}_{i}|COMPLETED" for j in ("62263343", "62263344") for i in range(3)) + "\n"
_BASE = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wjson(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1))


def base_capture(cid, root: Path) -> Path:
    """The calib side's synthetic capture for one cell (built once per test run, copied per scenario)."""
    if cid not in _BASE:
        cell = next(c for c in E.read_json(CAPTURE_MANIFEST)["cells"] if c["id"] == cid)
        d = root / f"base_{cid}"
        d.mkdir(parents=True)
        build_capture(d, Path(cell["calib_args"]["parent"]), total_blocks=48)
        gc.collect()
        _BASE[cid] = d
    return _BASE[cid]


def structure_attention(src, dst, gain):
    """Replace the synthetic attention error curves by err = gain * rep * (0.05 + mn)^2 / L^2 (all fields)."""
    d = dict(np.load(src, allow_pickle=False))
    for tag in {k.rsplit("|", 1)[0] for k in d if k.startswith("att|")}:
        mn = d[f"{tag}|mn"].astype(np.float64)
        lv = d[f"{tag}|lv"].astype(np.float64)
        rep = d[f"{tag}|mac"].astype(np.float64) / (128.0 * 2048.0)
        f = (gain * rep * (0.05 + mn) ** 2)[:, None] / lv[None, :] ** 2
        d[f"{tag}|fis"] = f.astype(np.float32)
        d[f"{tag}|fis_emp"] = (2 * f).astype(np.float32)
        d[f"{tag}|raw"] = (3 * f).astype(np.float32)
    np.savez_compressed(dst, **d)


def derived_profile(src_profile, inc_table) -> dict:
    """A consistent INC16 profile for a cell without a real one: the round-6 t40 metric histograms re-dispatched
    through this cell's incumbent (levels/thresholds/cycles), so the INC replays it exactly (TEST FIXTURE)."""
    p = copy.deepcopy(E.read_json(src_profile))
    by = {"prc": 0.0, "attention": 0.0}
    for g in p["groups"].values():
        lv, th = F.table_group_spec(inc_table, g)
        g["levels"], g["thresholds"] = lv, th
        g["actual_cycle_macs"] = _group_cost(g, th)
        by[g["kind"]] += _group_cost(dict(g, levels=lv), th)
    p["total_cycle_macs"] = float(p["fixed_cycle_macs"]) + by["prc"] + by["attention"]
    p["actual_mean_length"] = p["total_cycle_macs"] / float(p["total_macs"])
    p["test_fixture"] = "derived from the round-6 30B t40 INC profile, re-dispatched through this incumbent"
    return p


class Stage2Fixture:
    """A complete scratch round-7 root the chain can run on (see the module docstring)."""

    def __init__(self, root: Path, base_root: Path, *, gain=30.0, p_mult=1.002, kappa_m=None, presolve=True,
                 identity_level="exact"):
        self.root = Path(root)
        self.r7 = self.root / "r7"
        self.cm = self.root / "capture_manifest.json"
        self.screen = self.root / "screen_manifest.json"
        self.cand = self.root / "candidates.json"
        self.bin = self.root / "bin"
        self.calls = self.root / "sbatch_calls.txt"
        self.squeue = self.root / "squeue.txt"
        self.sacct = self.root / "sacct.txt"
        self.squeue.write_text("")
        self.sacct.write_text(SACCT_OK)
        cm = E.read_json(CAPTURE_MANIFEST)
        s0m = E.read_json(STEP0_MANIFEST)
        kappa_m = kappa_m or KEEP_M
        self.P = {}
        for c in cm["cells"]:
            cid = c["id"]
            cd, capd = self.r7 / cid, self.r7 / cid / "capture"
            paths = dict(c["paths"])
            for k, v in list(paths.items()):
                paths[k] = str(self.r7 / Path(v).relative_to(E.R7_ROOT))
            if cid in CELLS:
                capd.mkdir(parents=True)
                b = base_capture(cid, base_root)
                paths.update(stem=str(capd / f"{cid}_r7.json"), state=str(capd / f"{cid}_r7_state.npz"),
                             att_dump=str(capd / f"{cid}_r7_att.npz"), hold_dump=str(capd / f"{cid}_r7_hold.npz"),
                             diag=str(capd / f"{cid}_r7_diag.json"), identity=str(capd / "identity.json"))
                shutil.copy(b / "cell_r7_state.npz", paths["state"])
                shutil.copy(b / "cell_r7_hold.npz", paths["hold_dump"])
                if gain is None:
                    shutil.copy(b / "cell_r7_att.npz", paths["att_dump"])
                else:
                    structure_attention(b / "cell_r7_att.npz", paths["att_dump"], gain)
                wjson(paths["stem"], {"test_fixture": "stand-in stem"})
                st = c["r7"]["score_tables"]
                diag = {"test_fixture": True, "score_tables": ",".join(f"{k}={v}" for k, v in st.items()),
                        "tables": {"c17": {"pred_dnll_fis": -0.001, "L_over_parent": 1.0},
                                   "c17_s80": {"pred_dnll_fis": -0.001 + KAPPA_P.get(cid, 0.004),
                                               "L_over_parent": 0.8}}}
                wjson(paths["diag"], diag)
                files = {k: {"path": paths[k], "sha256": sha(paths[k])} for k in ("state", "att_dump", "hold_dump")}
                wjson(paths["identity"], {"test_fixture": True, "cell": cid, "identity_ok": identity_level == "exact",
                                          "ok": identity_level == "exact", "identity_level": identity_level,
                                          "files": files, "pipeline_check": {"ok": True}})
                self._step0(cid, c, s0m["cells"][cid], paths, kappa_m, p_mult, presolve)
            c["paths"] = paths
        cm["output_root"] = str(self.r7)
        cm["test_fixture"] = "paths rewritten to a scratch round-7 root; registration fields unchanged"
        wjson(self.cm, cm)
        self.bin.mkdir()
        for name, body in (("squeue", 'cat "$FAKE_SQUEUE"'), ("sacct", 'cat "$FAKE_SACCT"'),
                           ("sbatch", 'echo "$*" >> "$FAKE_SBATCH_CALLS"; exit 97')):
            p = self.bin / name
            p.write_text(f"#!/bin/bash\n{body}\n")
            p.chmod(0o755)

    def _step0(self, cid, capc, s0c, paths, kappa_m, p_mult, presolve):
        d = self.r7 / "step0" / cid
        d.mkdir(parents=True)
        starts = [int(s) for s in s0c["windows"]["starts"]]
        inc_t = E.read_json(s0c["arms"]["INC"]["table"])
        if cid == "30B_t48":
            prof = derived_profile(R6 / "30B_t40" / "INC_profile.json", inc_t)
            wjson(d / "INC16_profile.json", prof)
        else:
            shutil.copy(R6 / cid / "INC_profile.json", d / "INC16_profile.json")
            prof = E.read_json(d / "INC16_profile.json")
        exact = prof["total_cycle_macs"] / prof["total_macs"]
        rp = F.replay_cost(inc_t, prof)
        assert not rp["ladders_changed"] and abs(rp["cost"] / exact - 1) <= 1e-12, (cid, rp["cost"], exact)
        U = U_PROT[cid]
        C = exact
        if presolve:        # the arm's s0 table replayed on this profile: P puts s1 at s0 * p_mult (approximately)
            cap = S.Capture(paths["state"], paths["att_dump"], paths["hold_dump"])
            kap = S.kappa_for_cell(capc["prereg"], S.kappa_decision_core(2.5, 0.05))
            lads = S.arm_ladders(cap, "UK", "up")
            parent = E.read_json(Path(capc["calib_args"]["parent"]) / "table.json")
            t = S._table_from_solve(parent, S.solve_arm(cap, lads, kap["currency_ratio"], PR.S0[cid]), lads, cap)
            C = F.replay_cost(t, prof, prot_macs=0.0, psl_inc=F.psl_of(inc_t), psl_arm=F.psl_of(t))["cost"]
            del cap
            gc.collect()
        P = U + (C - U) * p_mult
        self.P[cid] = P
        tr = d / "PARENT16_trace.json"
        wjson(tr, {"test_fixture": "stand-in trace"})
        itr = d / "INC16_trace.json"
        wjson(itr, {"test_fixture": "stand-in trace"})
        nll = [2.0 + 0.01 * ((7 * i) % 5) for i in range(16)]
        if cid != "30B_t48":
            nll = E.read_json(R6 / cid / "INC_nll.json")["window_nll"]
        wjson(d / "INC16_nll.json", {"window_starts": starts, "wrapper": s0c["arms"]["INC"]["wrapper"], "cost": exact,
                                     "total_macs": prof["total_macs"], "total_cycle_macs": prof["total_cycle_macs"],
                                     "trace": str(itr), "window_nll": nll, "test_fixture": True})
        wjson(d / "INC16_audit.json", {"ok": True, "runtime_audit": {"U_prot": U, "mac_share": {"prot": 0.0112}},
                                       "test_fixture": True})
        wjson(d / "parent_reference.json", {"windows": starts, "audit_ok": True, "simulated": False, "P": P,
                                            "trace": str(tr), "trace_sha256": sha(tr), "test_fixture": True})
        wjson(d / "step0_summary.json", {"complete": True, "simulated": False, "test_fixture": True,
                                         "identity": {"restored_incumbent_exact": True}})
        if cid in kappa_m:
            z = np.random.default_rng(7 if cid == "30B_t32" else 8).normal(0.0, 0.002, 16)
            wd = (kappa_m[cid] + z - z.mean()).tolist()
            wjson(d / "step0_measured.json", {
                "cell": cid, "simulated": False, "ok": True, "test_fixture": True, "windows": starts,
                "identity": {"identity_mid_exact": True},
                "c17": {"table": s0c["arms"]["C17"]["table"]}, "s80": {"table": s0c["arms"]["S80"]["table"]},
                "pair": {"window_dnll": wd, "dcycles_pct": -8.0}})

    def run(self, *args, squeue="", sacct=SACCT_OK, timeout=5400):
        self.squeue.write_text(squeue)
        self.sacct.write_text(sacct)
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", FAKE_SQUEUE=str(self.squeue),
                   FAKE_SACCT=str(self.sacct), FAKE_SBATCH_CALLS=str(self.calls),
                   PYTHONPATH=str(REPO / "kernels"), CUDA_VISIBLE_DEVICES="")
        argv = [PY, str(CHAIN), "--r7-root", str(self.r7), "--capture-manifest", str(self.cm),
                "--screen-manifest", str(self.screen), "--candidates-out", str(self.cand), "--python", PY, *args]
        r = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
        if os.environ.get("R7S2_TEST_LOG"):          # optional: keep every chain run's output for inspection
            with open(os.environ["R7S2_TEST_LOG"], "a") as f:
                f.write(f"\n##### chain {' '.join(args)} -> exit {r.returncode}\n{r.stdout}{r.stderr}")
        return r.returncode, r.stdout + r.stderr

    def outputs(self):
        files = sorted(p for p in self.r7.rglob("*") if p.is_file() and "stage2_cpu" not in p.parts)
        files += [p for p in (self.screen, self.cand) if p.exists()]
        return {str(p): sha(p) for p in files}


def setUpModule():
    global BASE_TMP
    BASE_TMP = tempfile.TemporaryDirectory()
    for cid in CELLS:
        base_capture(cid, Path(BASE_TMP.name))


def tearDownModule():
    BASE_TMP.cleanup()
    _BASE.clear()


class Stage2ChainEndToEnd(unittest.TestCase):
    maxDiff = None

    def fixture(self, tmp, **kw):
        return Stage2Fixture(Path(tmp), Path(BASE_TMP.name), **kw)

    def test_a_full_chain_three_cells_then_idempotent_reruns(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self.fixture(tmp)
            rc, out = fx.run(squeue="99999_0|RUNNING|gres/gpu:1\n")
            self.assertEqual(rc, 0, out[-6000:])
            cmd = f"  cd {REPO} && sbatch --parsable --array=0-2%3 {SCREEN_SBATCH}"
            self.assertIn(cmd, out.splitlines(), out[-4000:])
            self.assertFalse(fx.calls.exists(), "the chain must never call sbatch")
            for step in ("RUN kappa", "RUN s0solve_30B_t32", "RUN fixedpoint_30B_t40", "RUN s1solve_30B_t48",
                         "RUN screen_build", "launcher dry run: {'30B_t32': 'DRY RUN PASS', '30B_t40': 'DRY RUN PASS', "
                         "'30B_t48': 'DRY RUN PASS'}"):
                self.assertIn(step, out)
            self.assertIn("KAPPA: decision keep_0.42", out)
            self.assertEqual(out.count("SCREEN: ENABLED  primary UK"), 3, out[-4000:])
            self.assertIn("predicted cost vs parent (replay C(s1)/P)", out)
            self.assertIn("1 GPU(s) of ours queued/running now -> throttle %3", out)
            m = E.read_json(fx.screen)
            self.assertTrue(m["test_build"])
            self.assertFalse(m["dry_run"])
            self.assertEqual(m["enabled_tasks"], [0, 1, 2])
            kd = E.read_json(fx.r7 / "step0" / "kappa_decision.json")
            self.assertEqual(kd["decision"], "keep_0.42")
            for cid in CELLS:
                c = m["cells"][cid]
                self.assertTrue(c["enabled"], c.get("disabled_reason"))
                self.assertEqual(c["primary"], "UK")
                fp = E.read_json(fx.r7 / "fixed_point" / f"{cid}_fixed_point.json")
                for a in c["arms"].values():
                    if a["role"] in ("primary", "secondary"):
                        self.assertEqual(a["target_scale"], fp["arms"][a["kind"]]["s1"])
                        self.assertLessEqual(a["pred_true_nats"], -PR.MDE_NATS[cid])
                        rlp = a["realized_length_prediction"]
                        self.assertFalse(rlp["standin"])
                        self.assertIn(rlp["source"]["solve_summary"], c["hashes"])
                        if a["kind"] == "K":
                            self.assertEqual(rlp["buckets"], {})
                        else:
                            self.assertTrue(rlp["buckets"])
                            self.assertTrue(all(set(b["new_rungs"]) <= {112, 128} for b in rlp["buckets"].values()))
            self.assertTrue(list((fx.r7 / "stage2_cpu").glob("stage2_*.json")))
            snap = fx.outputs()
            # re-run: everything reused byte for byte; the throttle follows squeue
            rc2, out2 = fx.run("--skip-launcher-dry-run", squeue="".join(f"9999{i}_0|RUNNING|gres/gpu:1\n" for i in range(3)))
            self.assertEqual(rc2, 0, out2[-4000:])
            self.assertEqual(fx.outputs(), snap)
            self.assertNotIn("RUN ", out2)
            self.assertIn("kappa decision exists, reused", out2)
            self.assertIn("screen manifest exists and matches these inputs, reused", out2)
            self.assertIn(f"  cd {REPO} && sbatch --parsable --array=0-2%1 {SCREEN_SBATCH}", out2.splitlines())
            rc3, out3 = fx.run("--skip-launcher-dry-run", squeue="".join(f"9999{i}_0|RUNNING|gres/gpu:1\n" for i in range(4)))
            self.assertEqual(rc3, 0, out3[-3000:])
            self.assertIn(f"  cd {REPO} && sbatch --parsable --array=0-2%3 --dependency=afterany:99990:99991:99992:99993 "
                          f"{SCREEN_SBATCH}", out3.splitlines())
            self.assertFalse(fx.calls.exists())
            # an existing screen manifest built from other inputs is never overwritten
            fp = fx.r7 / "fixed_point" / "30B_t40_fixed_point.json"
            spec = E.read_json(fx.cand)
            spec["cells"]["30B_t40"]["fixed_point_sha256"] = "0" * 64
            fx.cand.chmod(0o644)
            fx.cand.write_text(json.dumps(spec))
            rc4, out4 = fx.run("--skip-launcher-dry-run")
            self.assertEqual(rc4, 3, out4[-3000:])
            self.assertIn("refusing to overwrite", out4)
            self.assertTrue(fp.is_file())

    def test_b_every_cell_refused_by_the_mde_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self.fixture(tmp, gain=None)          # the raw synthetic capture: arms predicted worse than INC
            rc, out = fx.run("--cells", "30B_t40", "--skip-launcher-dry-run")
            self.assertEqual(rc, 0, out[-6000:])
            self.assertIn("EVERY CELL REFUSED (screen manifest", out)
            self.assertNotIn("sbatch --parsable", out)
            self.assertIn("REFUSED (pred_true > -MDE)", out)
            self.assertIn("SCREEN: DISABLED - required primary kind UK not eligible", out)
            m = E.read_json(fx.screen)
            self.assertEqual(m["enabled_tasks"], [])
            self.assertFalse(fx.calls.exists())

    def test_c_flagged_kappa_stops_then_fixed_point_refuses_every_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self.fixture(tmp, kappa_m=GAP_M, p_mult=1.2)
            rc, out = fx.run("--cells", "30B_t48")
            self.assertEqual(rc, 3, out[-4000:])
            self.assertIn("flagged for USER REVIEW", out)
            self.assertIn("gap_case", out)
            self.assertFalse((fx.r7 / "30B_t48" / "solve_s0").exists())
            rc2, out2 = fx.run("--cells", "30B_t48", "--accept-kappa-review")
            self.assertEqual(rc2, 0, out2[-6000:])
            self.assertIn("kappa decision exists, reused", out2)
            self.assertIn("30B_t48: the fixed point REFUSED every arm", out2)
            self.assertIn("EVERY CELL REFUSED (the fixed point refused every arm", out2)
            self.assertFalse(fx.screen.exists())
            fp = E.read_json(fx.r7 / "fixed_point" / "30B_t48_fixed_point.json")
            self.assertTrue(all(a["status"].startswith("refused") for a in fp["arms"].values()
                                if a["status"] != "not solved at s0"), fp["arms"])
            self.assertIsNone(fp["resolve_command"])
            self.assertFalse(fx.calls.exists())

    def test_d_stop_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self.fixture(tmp, presolve=False)
            rc, out = fx.run(squeue="62263344_1|PENDING|gres/gpu:1\n")
            self.assertEqual(rc, 2, out)
            self.assertIn("not finished yet", out)
            self.assertFalse((fx.r7 / "stage2_cpu").exists())
            rc, out = fx.run(sacct=SACCT_OK.replace("62263343_2|COMPLETED", "62263343_2|FAILED"))
            self.assertEqual(rc, 3, out)
            self.assertIn("did not complete cleanly", out)
            ip = fx.r7 / "30B_t40" / "capture" / "identity.json"
            ident = E.read_json(ip)
            wjson(ip, dict(ident, identity_level="near"))
            rc, out = fx.run()
            self.assertEqual(rc, 3, out)
            self.assertIn("level='near'", out)
            wjson(ip, ident)
            rc, out = fx.run(sacct=SACCT_OK.replace("62263344_1|COMPLETED", "62263344_1|FAILED|3:0"))
            self.assertEqual(rc, 3, out)                     # capture exit 3 = identity fail / near: stop, ask
            self.assertIn("did not complete cleanly", out)
            (fx.r7 / "30B_t32" / "solve_s0").mkdir(parents=True)
            rc, out = fx.run()
            self.assertEqual(rc, 3, out)
            self.assertIn("exists without solve_summary.json", out)
            self.assertTrue((fx.r7 / "step0" / "kappa_decision.json").is_file())
            # capture exit 4 = identity ok, pipeline check failed: registered as a valid capture -> flagged, continues
            wjson(ip, dict(ident, pipeline_check={"ok": False, "error": "unit test"}))
            rc, out = fx.run(sacct=SACCT_OK.replace("62263344_1|COMPLETED", "62263344_1|FAILED|4:0"))
            self.assertEqual(rc, 3, out)                     # ... up to the partial solve directory
            self.assertIn("FLAG: capture task 1 (30B_t40) 62263344_1 exited 4", out)
            self.assertIn("FLAG: 30B_t40: capture pipeline check failed", out)
            self.assertIn("kappa decision exists, reused", out)
            self.assertFalse(fx.calls.exists())
            # test-mode safety: never write into the repo / partial overrides / test flags in a real run
            env = dict(os.environ, PYTHONPATH=str(REPO / "kernels"))
            for argv in ([ "--r7-root", str(fx.r7), "--capture-manifest", str(fx.cm), "--screen-manifest",
                           str(KDIR / "prc_screen_r7_20260928.json"), "--candidates-out", str(fx.cand)],
                         ["--r7-root", str(fx.r7)], ["--no-slurm"], ["--cells", "4B_t64"]):
                r = subprocess.run([PY, str(CHAIN), *argv], cwd=REPO, env=env, capture_output=True, text=True)
                self.assertEqual(r.returncode, 4, (argv, r.stdout, r.stderr))
            # argparse usage errors (unknown option, missing value) must exit 4, never argparse's 2 (= 'not ready
            # yet, re-run later'), and must run no chain step; --help still exits 0.
            # Abbreviations are refused (allow_abbrev=False): '--accept' / '--acc' / '--accept-k' must not resolve to
            # the user-review override --accept-kappa-review, nor '--py' to --python.
            # Every probe that a parser change could turn into a VALID command line also carries '--cells 4B_t64',
            # which _main rejects (exit 4) before the hash locks or any step: if the probe ever parses, the test
            # fails on the missing 'invalid command line' message instead of running the real chain against Slurm
            # and the Turbo round-7 root.
            nocell = ["--cells", "4B_t64"]
            for argv in (["--accept-identity", *nocell], ["--accept-i", *nocell], ["--cells"], ["--bogus-option", "x"],
                         ["--accept", *nocell], ["--acc", *nocell], ["--accept-k", *nocell],
                         ["--py", PY, *nocell]):
                r = subprocess.run([PY, str(CHAIN), *argv], cwd=REPO, env=env, capture_output=True, text=True)
                self.assertEqual(r.returncode, 4, (argv, r.stdout, r.stderr))
                self.assertIn("invalid command line", r.stdout, argv)
                self.assertNotIn("round-7 stage-2 CPU chain; cells", r.stdout, argv)
            # --accept-identity-recheck (user-approved 2026-09-28) now parses; the '--cells 4B_t64' guard still stops
            # it (exit 4) before the hash locks or any step
            r = subprocess.run([PY, str(CHAIN), "--accept-identity-recheck", *nocell], cwd=REPO, env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 4, (r.stdout, r.stderr))
            self.assertIn("--cells must be a subset", r.stdout)
            self.assertNotIn("round-7 stage-2 CPU chain; cells", r.stdout)
            r = subprocess.run([PY, str(CHAIN), "--help"], cwd=REPO, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, (r.stdout, r.stderr))
            self.assertFalse((KDIR / "prc_screen_r7_20260928.json").exists())


if __name__ == "__main__":
    unittest.main()
