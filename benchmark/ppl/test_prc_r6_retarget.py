"""CPU tests for the round-6 budget re-target (calib9_r6, prc_resolve_r6, the r6 driver).

  cd scmp_llm && PYTHONPATH=$PWD/kernels python -m unittest benchmark.ppl.test_prc_r6_retarget -v
  R6_REAL_DUMP=1 ...  additionally re-solves the real 4B t32 c8w6 calibration dump (~1-2 min)
                      and checks calib9's functions reproduce the deployed c7/c8w6 gfis tables.

The CALIB7_* functions below are VERBATIM copies of mp_per_row_chunk_calib7.py main()'s
inline code (th_from, bisect, the global solve, solve_j/cost_j, the attention threshold loop,
the emission block, the emission filter); only the enclosing function signatures were added
to pass the closure variables in. calib9 must match them bit for bit at scale 1.0.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.ppl import mp_per_row_chunk_calib9_r6 as c9  # noqa: E402
from benchmark.ppl.mp_per_row_chunk_calib5 import (  # noqa: E402
    DENSE_LADDER, HALVE, SC_PREC, staircase_dp)
from benchmark.ppl import prc_r6_retarget_20260928 as drv  # noqa: E402
from benchmark.ppl.prc_resolve_r6 import compare_thresholds  # noqa: E402

CALIB7 = REPO / "benchmark/ppl/mp_per_row_chunk_calib7.py"
CALIB7_SHA = "9d568562963bab7b7cd162708b560971e5ff8ba3ef87dc47c8ad6cdd548cee96"
PARENT_14B_T32 = REPO.parent / "hpca_results/llm/ppl/mp_best/configs/14B/target32"
PRC2 = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2")


# ------------------------------------------------------------------ calib7 verbatim reference
def CALIB7_th_from(best, bin_max, Lv):
    th = []
    for k in range(len(Lv) - 1):
        low, high = np.nonzero(best <= k)[0], np.nonzero(best > k)[0]
        th.append(1.0 if high.size == 0 else (0.0 if low.size == 0 else float(bin_max[low[-1]])))
    for i in range(1, len(th)):
        th[i] = max(th[i], th[i - 1])
    return th


def CALIB7_bisect(pick, cost, target):
    lo, hi = 0.0, 1e-30
    j = pick(hi)
    while cost(j) > target and hi < 1e40:
        hi *= 10.0
        j = pick(hi)
    best = j
    for _ in range(80):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e3
        j = pick(mid)
        if cost(j) <= target:
            best, hi = j, mid
        else:
            lo = mid
        if lo > 0 and hi / lo < 1 + 1e-7:
            break
    return best


def CALIB7_global(bins, keys, budget, Lv):
    bisect = CALIB7_bisect
    target = sum(budget[k] * float(bins[k][1].sum()) for k in keys)
    rs = bisect(lambda lam: {k: staircase_dp(bins[k][0], bins[k][1], Lv, lam) for k in keys},
                lambda r: sum(float((bins[k][1] * Lv[r[k]]).sum()) for k in keys), target)
    return rs, {k: CALIB7_th_from(rs[k], bins[k][2], Lv) for k in keys}


def CALIB7_joint(lbins, abins, keys, akeys, budget, Lv, fixed, par_att):
    bisect = CALIB7_bisect
    par_lin = sum(budget[k] * float(lbins[k][1].sum()) for k in keys)
    target = par_lin + par_att

    def solve_j(lam):
        return ({k: staircase_dp(lbins[k][0], lbins[k][1], Lv, lam) for k in keys},
                {k: staircase_dp(abins[k][0], abins[k][1], abins[k][3], lam) for k in akeys})

    def cost_j(r):
        return (sum(float((lbins[k][1] * Lv[r[0][k]]).sum()) for k in keys) +
                sum(float((abins[k][1] * abins[k][3][r[1][k]]).sum()) for k in akeys) + fixed)
    rj = bisect(solve_j, cost_j, target)
    lin = {k: CALIB7_th_from(rj[0][k], lbins[k][2], Lv) for k in keys}
    att_th = {}
    for k in akeys:
        E, C, bmin, A = abins[k]
        r, n_ = rj[1][k], len(A)
        th = []
        for i in range(n_ - 1):
            hit = np.nonzero(r >= n_ - 1 - i)[0]
            th.append(1.0 if hit.size == 0 else (0.0 if hit[0] == 0 else float(bmin[hit[0]])))
        for i in range(1, len(th)):
            th[i] = min(th[i], th[i - 1])
        att_th[k] = th
    return rj, lin, att_th, target


def CALIB7_emit(stem, tname, tables, joint, table, wrapper, ladder, args_vars):
    from scmp_kernels.mp.config import AdaptiveMPConfig as _AMP
    bk = {f"{k[0]}:t0:l{k[1]}": {"levels": [int(v) for v in ladder], "thresholds": th}
          for k, th in tables[tname].items() if k != "__levels__"}
    o = stem.with_name(f"{stem.stem}_{tname}.json")
    t_ = o.with_name(o.stem + "_table.json")
    pt = json.loads(json.dumps(table))
    if tname == "gfisla":
        for (op_, b_), th in joint["att_th"].items():
            kk = f"{op_}:t0:l{b_}"
            if kk not in pt["buckets"]:
                raise SystemExit(f"[c7] parent table has no bucket {kk}")
            if len(pt["buckets"][kk]["thresholds"]) != len(th):
                raise SystemExit(f"[c7] threshold length mismatch for {kk}")
            pt["buckets"][kk]["thresholds"] = th
    pt["per_row_chunk"] = {"buckets": bk}
    pt.setdefault("sc_prec", SC_PREC)
    pt.setdefault("halve_bipolar_stoc_len", HALVE)
    pt["prc6_calib"] = dict(args_vars, table=tname)
    t_.write_text(json.dumps(pt, indent=1))
    pw = dict(wrapper)
    pw["threshold_table_path"] = str(t_.resolve())
    o.write_text(json.dumps(pw, indent=1))
    chk = _AMP(sorted([int(v) for v in pt["stoc_len_levels"]], reverse=True))
    chk.load_threshold_table(str(t_))
    assert len(chk.per_row_chunk) == len(bk), "round-trip lost buckets"
    return o, t_


def CALIB7_filter(tables, currencies):
    return [t for t in tables if (t[0] in "wg" and t[1:] in currencies) or t == "gfisla"]


# ------------------------------------------------------------------ synthetic fixtures
OPS = ("down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj")


def synth_bins(seed=0, n_bins=96, lad=DENSE_LADDER, dtype=np.float32):
    """Per-key (E, C, bin_max) shaped like calib7's: E decreasing in L, bins metric-sorted."""
    rng = np.random.default_rng(seed)
    Lv = np.asarray(lad, dtype=np.float64)
    bins, budget = {}, {}
    for op in OPS:
        for b in range(4):
            scale = rng.lognormal(0.0, 1.0, size=n_bins).astype(dtype)
            curve = (1.0 / Lv[None, :] ** 2 - 1.0 / Lv[-1] ** 2).astype(dtype)
            E = (scale[:, None] * curve * np.linspace(0.5, 2.0, n_bins, dtype=dtype)[:, None]).astype(dtype)
            C = rng.uniform(1e6, 5e6, size=n_bins).astype(dtype)
            bm = np.sort(rng.uniform(0, 1, size=n_bins)).astype(dtype)
            bins[(op, b)] = (E, C, bm)
            budget[(op, b)] = float(rng.uniform(20, 40))
    return bins, budget, Lv


def synth_abins(seed=1, n_bins=64, ladder=(16, 24, 32, 48, 64, 96)):
    rng = np.random.default_rng(seed)
    A = np.asarray(ladder, dtype=np.float64)
    ab = {}
    par_att = 0.0
    for op in ("av", "qk"):
        for b in range(4):
            scale = rng.lognormal(0.0, 1.0, size=n_bins).astype(np.float32)
            curve = (1.0 / A[None, :] ** 2 - 1.0 / 128.0 ** 2).astype(np.float32)
            E = (scale[:, None] * curve).astype(np.float32)
            C = np.full(n_bins, float(rng.uniform(1e6, 3e6)))          # float64 like calib7
            bmin = np.sort(rng.uniform(0, 1, size=n_bins)).astype(np.float32)
            ab[(op, b)] = (E, C, bmin, A)
            par_att += float((C * 80.0).sum())
    return ab, par_att


class TestVerbatimSolve(unittest.TestCase):
    def test_calib7_is_the_frozen_file(self):
        self.assertEqual(hashlib.sha256(CALIB7.read_bytes()).hexdigest(), CALIB7_SHA)

    def test_global_scale1_identical_to_calib7(self):
        for seed in (0, 1, 2):
            bins, budget, Lv = synth_bins(seed)
            keys = sorted(bins)
            rs7, th7 = CALIB7_global(bins, keys, budget, Lv)
            t0 = sum(budget[k] * float(bins[k][1].sum()) for k in keys)
            tgt = c9.scaled_target(t0, 1.0)
            self.assertIs(tgt, t0)                      # no multiply at 1.0
            rs9 = c9.solve_global(bins, keys, Lv, tgt)
            for k in keys:
                self.assertTrue(np.array_equal(rs7[k], rs9[k]), k)
                self.assertEqual(th7[k], c9.th_from_bins(rs9[k], bins[k][2], len(Lv)))

    def test_joint_scale1_identical_to_calib7(self):
        bins, budget, Lv = synth_bins(3)
        ab, par_att = synth_abins(4)
        keys, akeys = sorted(bins), sorted(ab)
        fixed = 1.2345e9
        rj7, lin7, att7, tgt7 = CALIB7_joint(bins, ab, keys, akeys, budget, Lv, fixed, par_att)
        par_lin = sum(budget[k] * float(bins[k][1].sum()) for k in keys)
        t0 = par_lin + par_att
        self.assertEqual(t0, tgt7)
        rj9 = c9.solve_joint(bins, ab, keys, akeys, Lv, fixed, c9.scaled_target(t0, 1.0))
        for k in keys:
            self.assertTrue(np.array_equal(rj7[0][k], rj9[0][k]))
            self.assertEqual(lin7[k], c9.th_from_bins(rj9[0][k], bins[k][2], len(Lv)))
        for k in akeys:
            self.assertTrue(np.array_equal(rj7[1][k], rj9[1][k]))
            self.assertEqual(att7[k], c9.att_th_from(rj9[1][k], ab[k][2], len(ab[k][3])))

    def test_scaled_cost_bounded_and_monotone(self):
        bins, budget, Lv = synth_bins(5)
        ab, par_att = synth_abins(6)
        keys, akeys = sorted(bins), sorted(ab)
        t0 = sum(budget[k] * float(bins[k][1].sum()) for k in keys)
        prev = -1.0
        for s in (1.0, 1.005, 1.01, 1.016, 1.02, 1.03):
            tgt = c9.scaled_target(t0, s)
            rs = c9.solve_global(bins, keys, Lv, tgt)
            cost = sum(float((bins[k][1] * Lv[rs[k]]).sum()) for k in keys)
            self.assertLessEqual(cost, tgt)
            self.assertGreater(cost / tgt, 0.995)
            self.assertGreaterEqual(cost, prev)
            prev = cost
        jt0 = sum(budget[k] * float(bins[k][1].sum()) for k in keys) + par_att
        prev = -1.0
        for s in (1.0, 1.01, 1.02):
            tgt = c9.scaled_target(jt0, s)
            rj = c9.solve_joint(bins, ab, keys, akeys, Lv, 0.0, tgt)
            cost = c9.joint_cost(rj, bins, ab, keys, akeys, Lv, 0.0)
            self.assertLessEqual(cost, tgt)
            self.assertGreater(cost / tgt, 0.995)
            self.assertGreater(cost, prev)
            prev = cost

    def test_scale_names_and_filter(self):
        self.assertEqual(c9.scale_suffix(1.0), "")
        self.assertEqual(c9.scale_suffix(1.016), "_s10160")
        self.assertEqual(c9.scale_suffix(1.0123), "_s10123")
        self.assertEqual(c9.base_table_name("gfisla_s10160"), "gfisla")
        self.assertEqual(c9.base_table_name("gfis_emp"), "gfis_emp")
        self.assertEqual(c9.parse_scales("1.0,1.016"), [1.0, 1.016])
        for bad in ("", "1.0,1.00001", "2.0", "nan"):
            with self.assertRaises(SystemExit):
                c9.parse_scales(bad)
        cur = ["rel", "fis"]
        base = ["wrel", "grel", "wfis", "gfis", "c17", "g5", "gfisla"]
        self.assertEqual(c9.emitted_table_names(base, cur), CALIB7_filter(base, cur))
        scaled = base + ["grel_s10160", "gfis_s10160", "gfisla_s10160"]
        self.assertEqual(c9.emitted_table_names(scaled, cur),
                         ["wrel", "grel", "wfis", "gfis", "gfisla",
                          "grel_s10160", "gfis_s10160", "gfisla_s10160"])
        self.assertEqual(c9.emitted_table_names(base, ["fis"]), CALIB7_filter(base, ["fis"]))


@unittest.skipUnless((PARENT_14B_T32 / "table.json").is_file(), "parent config not available")
class TestEmission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.table = json.loads((PARENT_14B_T32 / "table.json").read_text())
        self.wrapper = json.loads((PARENT_14B_T32 / "wrapper.json").read_text())
        rng = np.random.default_rng(7)
        self.keys = [(op, b) for op in OPS for b in range(4)]
        self.lin = {k: sorted(float(np.float32(x)) for x in rng.uniform(0, 1, 16)) for k in self.keys}
        self.att = {}
        for op in ("av", "qk"):
            for b in range(4):
                n = len(self.table["buckets"][f"{op}:t0:l{b}"]["thresholds"])
                self.att[(op, b)] = sorted((float(np.float32(x)) for x in rng.uniform(0, 1, n)),
                                           reverse=True)
        self.vars7 = {"parent": str(PARENT_14B_T32), "model_path": "Qwen/Qwen3-14B",
                      "out": "x", "diag": "y", "ladder": ",".join(map(str, DENSE_LADDER)),
                      "calib_windows": 6, "holdout_windows": 2, "rows_per_call": 64,
                      "expert_calls_per_block": 8, "bins": 2048, "seed": 0, "frontend": "awq",
                      "currencies": "rel,fis", "att_rows_per_call": 128, "dump": "",
                      "grad_mode": "full", "score_tables": ""}

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_emission_bytes_equal_calib7(self):
        for tname, att in (("gfis", None), ("gfisla", self.att)):
            d7, d9 = self.dir / "c7", self.dir / "c9"
            d7.mkdir(exist_ok=True)
            d9.mkdir(exist_ok=True)
            o7, t7 = CALIB7_emit(d7 / "cell_c7.json", tname, {tname: self.lin},
                                 {"att_th": self.att}, self.table, self.wrapper, DENSE_LADDER,
                                 self.vars7)
            o9, t9, _ = c9.emit_table(d9 / "cell_c7.json", tname, self.lin, att, self.table,
                                      self.wrapper, DENSE_LADDER, dict(self.vars7, table=tname))
            self.assertEqual(t7.read_bytes(), t9.read_bytes(), tname)
            w7, w9 = json.loads(o7.read_text()), json.loads(o9.read_text())
            self.assertEqual({k: v for k, v in w7.items() if k != "threshold_table_path"},
                             {k: v for k, v in w9.items() if k != "threshold_table_path"})

    def test_scaled_emission_loads_and_is_allocation_only(self):
        o, t, n = c9.emit_table(self.dir / "cell_r6.json", "gfisla_s10160", self.lin, self.att,
                                self.table, self.wrapper, DENSE_LADDER,
                                dict(self.vars7, table="gfisla_s10160"),
                                {"target_scale": 1.016, "table": "gfisla_s10160"})
        self.assertTrue(o.name.endswith("_gfisla_s10160.json"))
        payload = json.loads(t.read_text())
        self.assertEqual(payload["prc9_r6"]["target_scale"], 1.016)
        cfg = drv.mp_from_wrapper(o)
        self.assertEqual(len(cfg.per_row_chunk), len(self.keys))
        for (op, b), th in self.lin.items():
            blk0 = next(x for x in range(40) if c9.bucket_of(x, 40, 4) == b)
            got = cfg.get_per_row_chunk(op, blk0, 40)
            self.assertEqual([float(x) for x in got[1]], th)
            self.assertEqual(list(got[0]), sorted(DENSE_LADDER))
        base = json.loads(json.dumps(self.table))
        base["per_row_chunk"] = {"buckets": {f"{k[0]}:t0:l{k[1]}": {
            "levels": DENSE_LADDER, "thresholds": sorted(self.lin[k])} for k in self.keys}}
        self.assertEqual(drv.allocation_only_diff(base, payload, allow_attention=True), [])
        self.assertTrue(drv.allocation_only_diff(base, payload, allow_attention=False))
        bad = copy.deepcopy(payload)
        bad["stoc_len_levels"] = [128] + bad["stoc_len_levels"][1:]
        self.assertTrue(drv.allocation_only_diff(base, bad, allow_attention=True))
        bad = copy.deepcopy(payload)
        bad["protected_channels"]["stoc_len"] = 96
        self.assertTrue(drv.allocation_only_diff(base, bad, allow_attention=True))


class TestSolveState(unittest.TestCase):
    def test_roundtrip_and_resolve(self):
        bins, budget, Lv = synth_bins(8, lad=DENSE_LADDER)
        ab, par_att = synth_abins(9)
        keys, akeys = sorted(bins), sorted(ab)
        fixed = 3.0e8
        t0g = float(sum(budget[k] * float(bins[k][1].sum()) for k in keys))
        par_lin = sum(budget[k] * float(bins[k][1].sum()) for k in keys)
        meta = {"ladder": list(DENSE_LADDER), "keys": [[k[0], k[1]] for k in keys],
                "akeys": [[k[0], k[1]] for k in akeys], "target0_gfis": t0g,
                "target0_gfisla": float(par_lin + par_att), "fixed": fixed,
                "calib_record": {"x": 1}, "c9_record": None}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state.npz"
            c9.save_solve_state(p, meta, bins, ab)
            meta2, keys2, akeys2, lb2, ab2 = c9.load_solve_state(p)
        self.assertEqual(keys2, keys)
        self.assertEqual(akeys2, akeys)
        for k in keys:
            for a_, b_ in zip(bins[k], lb2[k]):
                self.assertEqual(a_.dtype, b_.dtype)
                self.assertTrue(np.array_equal(a_, b_))
        for k in akeys:
            for a_, b_ in zip(ab[k], ab2[k]):
                self.assertEqual(a_.dtype, b_.dtype)
                self.assertTrue(np.array_equal(a_, b_))
        res = c9.resolve_tables(meta2, keys2, akeys2, lb2, ab2, [1.0, 1.016], ["gfis", "gfisla"])
        self.assertEqual(sorted(res), ["gfis", "gfis_s10160", "gfisla", "gfisla_s10160"])
        # scale 1.0 from the state == calib7 verbatim on the original arrays
        _, th7 = CALIB7_global(bins, keys, budget, Lv)
        self.assertEqual({k: v for k, v in res["gfis"][0].items()}, th7)
        _, lin7, att7, _ = CALIB7_joint(bins, ab, keys, akeys, budget, Lv, fixed, par_att)
        self.assertEqual(res["gfisla"][0], lin7)
        self.assertEqual(res["gfisla"][1], att7)
        self.assertGreater(res["gfisla_s10160"][2]["calib_cost"], res["gfisla"][2]["calib_cost"])
        self.assertLessEqual(res["gfisla_s10160"][2]["calib_cost_over_target"], 1.0)
        # compare_thresholds sees a JSON-emitted copy as identical, and catches an edit
        payload = {"per_row_chunk": {"buckets": {f"{k[0]}:t0:l{k[1]}": {"thresholds": th}
                                                 for k, th in lin7.items()}},
                   "buckets": {f"{k[0]}:t0:l{k[1]}": {"thresholds": th} for k, th in att7.items()}}
        payload = json.loads(json.dumps(payload))
        self.assertEqual(compare_thresholds("gfisla", lin7, att7, payload), [])
        payload["buckets"]["qk:t0:l0"]["thresholds"][0] += 1e-7
        self.assertEqual(len(compare_thresholds("gfisla", lin7, att7, payload)), 1)


class TestDriverMath(unittest.TestCase):
    def test_retarget_scale(self):
        # 14B t32-like numbers: P 33.5638, C1 33.076, U = protected 2.583
        raw, s = drv.retarget_scale(33.5638, 33.076, 2.583)
        self.assertAlmostEqual(raw, (33.5638 - 2.583) / (33.076 - 2.583))
        self.assertEqual(s, round(raw, 4))
        self.assertAlmostEqual(2.583 + raw * (33.076 - 2.583), 33.5638, places=9)
        with self.assertRaises(ValueError):
            drv.retarget_scale(2.0, 3.0, 2.5)

    def test_paired_stats(self):
        st = drv.paired_stats([1.0, 2.0, 3.5], [1.0, 2.5, 3.0])
        self.assertAlmostEqual(st["mean_dnll"], 0.0)
        with self.assertRaises(ValueError):
            drv.paired_stats([1.0], [1.0])

    def test_calib_argv_matches_incumbent_record(self):
        rec = {"parent": "/p", "model_path": "Qwen/Qwen3-14B", "ladder": "8,16,128",
               "calib_windows": 6, "holdout_windows": 2, "rows_per_call": 64,
               "expert_calls_per_block": 8, "bins": 2048, "seed": 0, "frontend": "awq",
               "currencies": "rel,fis", "att_rows_per_call": 128, "dump": "",
               "grad_mode": "full", "score_tables": "c17=/t.json"}
        cell = {"id": "14B_t32", "calib_args": rec}
        argv = drv.calib_argv({}, cell, "/o", 1.016)
        import argparse
        # parse with calib9's own parser options (mirror): every recorded key round-trips
        ap = argparse.ArgumentParser()
        for opt in ("--parent", "--model_path", "--out", "--diag", "--ladder", "--calib-windows",
                    "--holdout-windows", "--rows-per-call", "--expert-calls-per-block", "--bins",
                    "--seed", "--frontend", "--currencies", "--att-rows-per-call", "--dump",
                    "--grad-mode", "--score-tables", "--target-scales", "--solve-state",
                    "--att-measure-extra", "--att-dump"):
            ap.add_argument(opt, default="")
        ns = vars(ap.parse_args(argv))
        for k, v in rec.items():
            self.assertEqual(ns[k], str(v) if k != "dump" else "", k)
        self.assertEqual(ns["target_scales"], "1.0,1.0160")
        self.assertTrue(ns["solve_state"].endswith("14B_t32_r6_state.npz"))
        with self.assertRaises(ValueError):
            drv.calib_argv({}, cell, "/o", 1.0)

    def test_calib9_parser_accepts_argv(self):
        """The argv the driver builds parses with calib9's REAL parser (CUDA skip exits 0)."""
        import subprocess
        rec = {"parent": "/p", "model_path": "m", "ladder": ",".join(map(str, DENSE_LADDER)),
               "calib_windows": 6, "holdout_windows": 2, "rows_per_call": 64,
               "expert_calls_per_block": 16, "bins": 2048, "seed": 0, "frontend": "awq",
               "currencies": "fis", "att_rows_per_call": 128, "dump": "",
               "grad_mode": "block", "score_tables": "c17=/t.json"}
        argv = drv.calib_argv({}, {"id": "30B_t64", "calib_args": rec}, "/o", 1.012)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        p = subprocess.run([sys.executable, str(REPO / "benchmark/ppl/mp_per_row_chunk_calib9_r6.py")]
                           + argv, capture_output=True, text=True, env=env, cwd=str(REPO))
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        self.assertIn("SKIP: needs CUDA", p.stdout)

    def test_analyze_trace_synthetic(self):
        if not (PARENT_14B_T32 / "table.json").is_file():
            self.skipTest("parent config not available")
        table = json.loads((PARENT_14B_T32 / "table.json").read_text())
        widths, psl = drv.protected_widths(table)
        (op, blk, unit), w = next(iter(sorted(widths.items(), key=lambda kv: (kv[0][0], kv[0][1]))))
        lad = sorted(int(v) for v in table["stoc_len_levels"])
        groups = [
            {"op": op, "block": blk, "unit": unit, "stoc_len": psl, "d_in": w, "macs": 100,
             "rng_levels": 128, "sc_prec": 8, "halve": True},
            {"op": op, "block": blk, "unit": unit, "stoc_len": lad[0], "d_in": 128, "macs": 900,
             "rng_levels": 128, "sc_prec": 8, "halve": True},
            {"op": "qk", "block": blk, "unit": 0, "stoc_len": lad[-1], "d_in": 128, "macs": 1000,
             "rng_levels": 128, "sc_prec": 8, "halve": True},
        ]
        tr = {"header": {"total_blocks": 40}, "groups": groups}
        a = drv.analyze_trace(tr, PARENT_14B_T32 / "wrapper.json", arm="gfisla")
        self.assertTrue(a["audit_ok"], a["violations"])
        self.assertAlmostEqual(a["U"], 100 * psl / 2000)
        a2 = drv.analyze_trace(tr, PARENT_14B_T32 / "wrapper.json", arm="gfis")
        self.assertAlmostEqual(a2["U"], (100 * psl + 1000 * lad[-1]) / 2000)
        bad = copy.deepcopy(tr)
        bad["groups"][2]["stoc_len"] = 127          # not a rung of this qk bucket
        self.assertFalse(drv.analyze_trace(bad, PARENT_14B_T32 / "wrapper.json")["audit_ok"])
        bad = copy.deepcopy(tr)
        bad["groups"][1]["rng_levels"] = 64          # RNG grid change
        self.assertFalse(drv.analyze_trace(bad, PARENT_14B_T32 / "wrapper.json")["audit_ok"])


INC_TRACE_14B = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl/"
                     "14B_t32_prc_p2c7gfisla_trace.json")


@unittest.skipUnless((PARENT_14B_T32 / "table.json").is_file() and INC_TRACE_14B.is_file(),
                     "14B t32 parent config / incumbent trace not available")
class TestDriverStagesCPU(unittest.TestCase):
    """End-to-end CPU run of calib-identity and finalize on synthetic solve states emitted
    against the REAL 14B t32 parent table, and the REAL incumbent full-test trace."""

    def build(self, d, e):
        table = json.loads((PARENT_14B_T32 / "table.json").read_text())
        wrapper = json.loads((PARENT_14B_T32 / "wrapper.json").read_text())
        bins, budget, Lv = synth_bins(11, n_bins=64)
        ab, par_att = synth_abins(12, ladder=tuple(sorted(int(v) for v in table["stoc_len_levels"])))
        keys, akeys = sorted(bins), sorted(ab)
        par_lin = sum(budget[k] * float(bins[k][1].sum()) for k in keys)
        rec7 = {"parent": str(PARENT_14B_T32), "model_path": "Qwen/Qwen3-14B", "out": "OUT",
                "diag": "DIAG", "ladder": ",".join(map(str, DENSE_LADDER)), "seed": 0}
        c9rec = {"calibrator": "mp_per_row_chunk_calib9_r6.py", "target_scales": [1.0, 1.016]}
        meta = {"ladder": list(DENSE_LADDER), "keys": [[k[0], k[1]] for k in keys],
                "akeys": [[k[0], k[1]] for k in akeys],
                "target0_gfis": float(par_lin), "target0_gfisla": float(par_lin + par_att),
                "fixed": 0.0, "calib_record": dict(rec7, out="NEW_OUT", diag="NEW_DIAG"),
                "c9_record": c9rec,
                "parent_table_sha256": hashlib.sha256((PARENT_14B_T32 / "table.json").read_bytes()).hexdigest(),
                "parent_wrapper_sha256": hashlib.sha256((PARENT_14B_T32 / "wrapper.json").read_bytes()).hexdigest()}
        c9.save_solve_state(d / "14B_t32_r6_state.npz", meta, bins, ab)
        res = c9.resolve_tables(meta, keys, akeys, bins, ab, [1.0, 1.016], ["gfisla"])
        for name, (lin, att, rec) in res.items():
            c9.emit_table(d / "14B_t32_r6.json", name, lin, att, table, wrapper, DENSE_LADDER,
                          dict(meta["calib_record"], table=name), dict(c9rec, table=name, solve=rec))
        lin, att, _ = res["gfisla"]           # the "incumbent": calib7-style emission at 1.0
        o, t, _ = c9.emit_table(e / "14B_t32_c7.json", "gfisla", lin, att, table, wrapper,
                                DENSE_LADDER, dict(rec7, table="gfisla"))
        (d / "stage_a.json").write_text(json.dumps({"ok": True, "target_scale": 1.016,
                                                    "s_raw": 1.01603}))
        cell = {"id": "14B_t32", "model": "14B", "arm": "gfisla", "incumbent_arm": "c7gfisla",
                "model_path": "Qwen/Qwen3-14B", "incumbent_wrapper": str(o),
                "incumbent_table": str(t), "incumbent_diag": str(e / "missing_diag.json"),
                "incumbent_trace": str(INC_TRACE_14B), "incumbent_ppl": 9.076411710046006,
                "incumbent_test_cost": 33.10983222233671, "parent_test_cost": 33.55915013836878,
                "parent_ppl": 9.3097, "fp16_ppl": 8.6383,
                "flip_thresholds": {"1.05x": 1.05 * 8.6383, "1.10x": 1.10 * 8.6383}}
        manifest = {"preregistered_rules": {"best_all_entry": "BEST", "disclosure": "DISC"}}
        return manifest, cell

    def test_calib_identity_and_finalize(self):
        with tempfile.TemporaryDirectory() as d_, tempfile.TemporaryDirectory() as e_:
            d, e = Path(d_), Path(e_)
            manifest, cell = self.build(d, e)
            self.assertEqual(drv.calib_identity(manifest, cell, d), 0)
            ci = json.loads((d / "calib_identity.json").read_text())
            self.assertTrue(ci["ok"], ci["checks"])
            wrapper = ci["scaled_wrapper"]
            self.assertTrue(wrapper.endswith("14B_t32_r6_gfisla_s10160.json"))
            (d / "selected.json").write_text(json.dumps({"evaluate": True, "wrapper": wrapper}))
            (d / "stage_b.json").write_text(json.dumps({"cost_ratio_vs_parent": 1.0003}))
            tr = json.loads(INC_TRACE_14B.read_text())
            tr["header"]["mp_config_json"] = wrapper
            tp = d / "fake_test_trace.json"
            tp.write_text(json.dumps(tr))
            self.assertEqual(drv.finalize(manifest, cell, d, str(tp)), 0)
            r = json.loads((d / "full_test_result.json").read_text())
            self.assertTrue(r["protocol_ok"], r["failed_checks"])
            self.assertEqual(r["arm"], "r6ts_c7gfisla_s10160")
            self.assertFalse(r["passes_1p05x_fp16"])          # 9.0764 > 9.070215
            self.assertAlmostEqual(r["test_cost_vs_parent"], 33.10983222233671 / 33.55915013836878 - 1)
            self.assertEqual(r["best_all"]["entry_rule"], "BEST")
            # a protocol deviation in the test header is caught
            (d / "full_test_result.json").unlink()
            tr["header"]["ctx"] = 1024
            tp.write_text(json.dumps(tr))
            self.assertEqual(drv.finalize(manifest, cell, d, str(tp)), 7)

    def test_calib_identity_catches_non_allocation_edit(self):
        with tempfile.TemporaryDirectory() as d_, tempfile.TemporaryDirectory() as e_:
            d, e = Path(d_), Path(e_)
            manifest, cell = self.build(d, e)
            tp = d / "14B_t32_r6_gfisla_s10160_table.json"
            t = json.loads(tp.read_text())
            t["protected_channels"]["stoc_len"] = 96          # not an allocation threshold
            tp.write_text(json.dumps(t, indent=1))
            self.assertEqual(drv.calib_identity(manifest, cell, d), 6)
            ci = json.loads((d / "calib_identity.json").read_text())
            self.assertFalse(ci["checks"]["scaled_allocation_only"])
            self.assertTrue(ci["checks"]["state_roundtrip_exact"])

    def test_calib_identity_catches_state_mismatch(self):
        with tempfile.TemporaryDirectory() as d_, tempfile.TemporaryDirectory() as e_:
            d, e = Path(d_), Path(e_)
            manifest, cell = self.build(d, e)
            tp = d / "14B_t32_r6_gfisla_table.json"
            t = json.loads(tp.read_text())
            k0 = next(iter(t["per_row_chunk"]["buckets"]))
            t["per_row_chunk"]["buckets"][k0]["thresholds"][3] += 1e-6
            tp.write_text(json.dumps(t, indent=1))
            self.assertEqual(drv.calib_identity(manifest, cell, d), 6)
            ci = json.loads((d / "calib_identity.json").read_text())
            self.assertFalse(ci["checks"]["s1_equals_incumbent"])
            self.assertFalse(ci["checks"]["state_roundtrip_exact"])


@unittest.skipUnless(os.environ.get("R6_REAL_DUMP") == "1"
                     and (PRC2 / "4B_t32_c8w6.npz").is_file(), "set R6_REAL_DUMP=1 (slow)")
class TestRealDump(unittest.TestCase):
    """calib9's module-level solve on the REAL 4B t32 calibration arrays reproduces the deployed
    tables: c8w6_gfis (made in-process from these arrays) and c7_gfis (identical 28/28 keys)."""

    def test_real_gfis_reproduced(self):
        d = np.load(PRC2 / "4B_t32_c8w6.npz")
        measured = [int(v) for v in d["measured"]]
        ladder = [int(v) for v in d["ladder"]]
        li = {L: i for i, L in enumerate(measured)}
        lad_cols = [li[L] for L in ladder]
        top_col = li[max(ladder)]
        Lv = np.asarray(ladder, dtype=np.float64)
        keys = sorted({(f.split("|")[1], int(f.split("|")[2])) for f in d.files
                       if f.startswith("calib|")})
        bins, budget = {}, {}
        for k in keys:
            pre = f"calib|{k[0]}|{k[1]}|"
            mn, mac, parL = d[pre + "mn"], d[pre + "mac"], d[pre + "parL"]
            e = d[pre + "fis"]
            cur = e - e[:, [top_col]]                        # calib7 cur_of('fis'), verbatim
            budget[k] = float((mac * parL).sum() / mac.sum())
            order = np.argsort(mn, kind="stable")            # calib7 binned(), verbatim
            B = int(min(2048, mn.size))
            edges = np.linspace(0, mn.size, B + 1).astype(np.int64)
            E = np.add.reduceat(cur[order][:, lad_cols], edges[:-1], axis=0)
            C = np.add.reduceat(mac[order], edges[:-1])
            bins[k] = (E, C, mn[order][edges[1:] - 1])
            del e, cur
        t0 = sum(budget[k] * float(bins[k][1].sum()) for k in keys)
        rs = c9.solve_global(bins, keys, Lv, c9.scaled_target(t0, 1.0))
        th = {f"{k[0]}:t0:l{k[1]}": c9.th_from_bins(rs[k], bins[k][2], len(Lv)) for k in keys}
        th = json.loads(json.dumps(th))
        for ref in ("4B_t32_c8w6_gfis_table.json", "4B_t32_c7_gfis_table.json"):
            b = json.loads((PRC2 / ref).read_text())["per_row_chunk"]["buckets"]
            same = [kk for kk in th if b[kk]["thresholds"] == th[kk]]
            self.assertEqual(len(same), len(th), f"{ref}: {len(same)}/{len(th)} keys identical")
        rs2 = c9.solve_global(bins, keys, Lv, c9.scaled_target(t0, 1.016))
        c1 = sum(float((bins[k][1] * Lv[rs[k]]).sum()) for k in keys)
        c2 = sum(float((bins[k][1] * Lv[rs2[k]]).sum()) for k in keys)
        self.assertLessEqual(c1, t0)
        self.assertLessEqual(c2, t0 * 1.016)
        self.assertGreater(c2 / c1, 1.01)


if __name__ == "__main__":
    unittest.main()
