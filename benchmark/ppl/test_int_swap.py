"""int_swap (v18) — hybrid INT mask ranking by measured SC→INT swap gain.

Covers the three things that can silently produce a wrong mask:
  * the swap-gain measurement itself (sign, per-layer granularity, cfg restore),
  * the split-half stability statistic that gates the wave,
  * hpca's selector branch — including that it refuses to mix an int_swap
    ranking measured at one INT width with a mask applied at another.
"""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.ppl.calibrate_mp_thresholds import (
    _INT_SWAP_CFG_KEYS,
    _int_swap_score,
    _measure_group_int_swap,
    _rankdata,
    _select_int_swap_windows,
    build_int_swap_payload,
    int_swap_cross_fold_stability,
    int_swap_stability,
)

HPCA = Path(__file__).resolve().parents[2] / "hpca"


class _Cfg:
    """Stand-in for model.config: only the keys the measurement touches."""

    def __init__(self):
        self.use_sc_linear = False
        self.use_sc_attn = False
        self.sc_ste_grad = True
        self.sc_mp_config = "ORIGINAL"
        self.sc_stoc_len = 999
        self.sc_group_stoclen = {"sentinel": 1}
        self.sc_hybrid_schedule = {"sentinel": "int9"}
        self.sc_hybrid_default = "fp"
        self.sc_hybrid_int_bits = 3
        self.sc_hybrid_int_sym = False
        self.sc_hybrid_chunk_size = 7


class _FakeModel:
    """Loss = base − gain(op, block) for whichever single entry is masked, plus a
    deterministic per-window offset so window means are non-degenerate."""

    def __init__(self, cfg, gains, window_offset):
        self.config = cfg
        self._gains = gains
        self._off = window_offset
        self.calls = []
        self.seen_state = []

    def parameters(self):
        import torch
        yield torch.zeros(1)

    def __call__(self, input_ids=None, labels=None):
        import torch
        sched = self.config.sc_hybrid_schedule or {}
        key = next(iter(sched), None)
        gain = self._gains.get(key, 0.0) if key is not None else 0.0
        wid = int(input_ids.reshape(-1)[0].item())
        self.calls.append((key, wid))
        self.seen_state.append({
            "sc_mp_config": self.config.sc_mp_config,
            "sc_group_stoclen": self.config.sc_group_stoclen,
            "sc_stoc_len": self.config.sc_stoc_len,
            "use_sc_linear": self.config.use_sc_linear,
            "use_sc_attn": self.config.use_sc_attn,
            "sc_ste_grad": self.config.sc_ste_grad,
            "sc_hybrid_int_bits": self.config.sc_hybrid_int_bits,
        })

        class _Out:
            pass
        out = _Out()
        out.loss = torch.tensor(2.0 + self._off.get(wid, 0.0) - gain)
        return out


def _enc(n_windows, ctx):
    import torch
    # window i is a constant run of value i, so the fake model can identify it
    return torch.cat([torch.full((ctx,), i, dtype=torch.long)
                      for i in range(n_windows)])


class MeasureIntSwapTest(unittest.TestCase):
    def setUp(self):
        self.cfg = _Cfg()
        self.gains = {("qk", 0): 0.5, ("qk", 1): 0.1,
                      ("down_proj", 0): -0.2, ("down_proj", 1): 0.3}
        self.model = _FakeModel(self.cfg, self.gains, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})

    def _run(self, **kw):
        import torch
        return _measure_group_int_swap(
            self.model, _enc(4, 8), total_blocks=2,
            operators={"qk", "down_proj"},
            int_bits=7, ref_mode="uniform", ref_level=32,
            mp_config_path=None, n_windows=4, ctx=8,
            device=torch.device("cpu"), **kw)

    def test_gain_sign_and_granularity(self):
        gains, info, split = self._run()
        # one entry per (op, LAYER) — no bucket broadcast
        self.assertEqual(set(gains), set(self.gains))
        for k, want in self.gains.items():
            self.assertAlmostEqual(gains[k], want, places=5)
        # positive gain == INT helps == loss went DOWN
        self.assertGreater(gains[("qk", 0)], 0.0)
        self.assertLess(gains[("down_proj", 0)], 0.0)
        self.assertAlmostEqual(info["baseline_loss"], 2.0, places=5)
        self.assertEqual(info["ref_mode"], "uniform")
        self.assertEqual(info["int_bits"], 7)

    def test_one_entry_masked_at_a_time(self):
        self._run()
        for key, _wid in self.model.calls:
            self.assertIn(key, set(self.gains) | {None})

    def test_config_restored(self):
        before = {k: getattr(self.cfg, k) for k in _INT_SWAP_CFG_KEYS}
        self._run()
        for k, v in before.items():
            self.assertEqual(getattr(self.cfg, k), v,
                             f"{k} not restored after int_swap measurement")

    def test_completed_checkpoint_is_resumed_without_reprobing(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d) / "sweep.partial.json"
            self._run(checkpoint_path=checkpoint)
            self.assertTrue(checkpoint.is_file())
            self.model.calls.clear()
            gains, _info, _split = self._run(checkpoint_path=checkpoint)
            self.assertEqual(set(gains), set(self.gains))
            # The all-SC reference is intentionally rechecked; all four costly
            # candidate probes came from the verified checkpoint.
            self.assertEqual(len(self.model.calls), 4)

    def test_uniform_reference_state_during_measurement(self):
        self._run()
        self.assertTrue(self.model.seen_state)
        for st in self.model.seen_state:
            # SC live on both paths, STE off
            self.assertTrue(st["use_sc_linear"])
            self.assertTrue(st["use_sc_attn"])
            self.assertFalse(st["sc_ste_grad"])
            # uniform reference: MP dispatch off, empty group map so every module
            # falls back to sc_stoc_len — the deployed operating point, not the
            # ladder top the old measured_curve ranking used.
            self.assertIsNone(st["sc_mp_config"])
            self.assertEqual(st["sc_group_stoclen"], {})
            self.assertEqual(st["sc_stoc_len"], 32)
            self.assertEqual(st["sc_hybrid_int_bits"], 7)

    def test_mp_config_ref_requires_path(self):
        with self.assertRaises(SystemExit):
            import torch
            _measure_group_int_swap(
                self.model, _enc(4, 8), total_blocks=2,
                operators={"qk"}, int_bits=7, ref_mode="mp_config",
                ref_level=32, mp_config_path=None, n_windows=4, ctx=8,
                device=torch.device("cpu"))

    def test_unknown_ref_mode_raises(self):
        with self.assertRaises(SystemExit):
            import torch
            _measure_group_int_swap(
                self.model, _enc(4, 8), total_blocks=2,
                operators={"qk"}, int_bits=7, ref_mode="bogus",
                ref_level=32, mp_config_path=None, n_windows=4, ctx=8,
                device=torch.device("cpu"))


class SplitHalfTest(unittest.TestCase):
    def test_identical_halves_are_perfectly_stable(self):
        g = {("qk", i): float(i) for i in range(20)}
        s = int_swap_stability(g, g, frac=0.20)
        self.assertAlmostEqual(s["spearman"], 1.0, places=6)
        self.assertAlmostEqual(s["top_overlap"], 1.0, places=6)
        self.assertEqual(s["top_n"], 4)

    def test_reversed_halves_share_nothing(self):
        a = {("qk", i): float(i) for i in range(20)}
        b = {("qk", i): -float(i) for i in range(20)}
        s = int_swap_stability(a, b, frac=0.20)
        self.assertAlmostEqual(s["spearman"], -1.0, places=6)
        self.assertAlmostEqual(s["top_overlap"], 0.0, places=6)

    def test_split_half_uses_disjoint_windows(self):
        import torch
        cfg = _Cfg()
        # window offsets make half A and half B disagree about the ranking
        model = _FakeModel(cfg, {("qk", 0): 0.0, ("qk", 1): 0.0},
                           {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
        _g, _i, split = _measure_group_int_swap(
            model, _enc(4, 8), total_blocks=2, operators={"qk"},
            int_bits=7, ref_mode="uniform", ref_level=32,
            mp_config_path=None, n_windows=4, ctx=8,
            device=torch.device("cpu"))
        self.assertEqual(split["windows_per_half"], 2)
        self.assertEqual(set(split["gain_a"]), set(split["gain_b"]))

    def test_too_few_windows_yields_no_split(self):
        import torch
        cfg = _Cfg()
        model = _FakeModel(cfg, {("qk", 0): 0.1}, {})
        _g, _i, split = _measure_group_int_swap(
            model, _enc(2, 8), total_blocks=1, operators={"qk"},
            int_bits=7, ref_mode="uniform", ref_level=32,
            mp_config_path=None, n_windows=2, ctx=8,
            device=torch.device("cpu"))
        self.assertIsNone(split)

    def test_rankdata_averages_ties(self):
        r = _rankdata(np.array([1.0, 2.0, 2.0, 3.0]))
        self.assertEqual(list(r), [1.0, 2.5, 2.5, 4.0])

    def test_cross_fold_reports_pairwise_agreement(self):
        g = {("qk", i): float(i) for i in range(20)}
        s = int_swap_cross_fold_stability([g, g, g, g], frac=0.20)
        self.assertEqual(s["folds"], 4)
        self.assertEqual(len(s["pairs"]), 6)
        self.assertAlmostEqual(s["mean_top_overlap"], 1.0)


class RobustSamplingAndScoreTest(unittest.TestCase):
    def test_stratified_windows_are_reproducible_and_corpus_spread(self):
        enc = _enc(100, 8)
        windows_a, starts_a = _select_int_swap_windows(
            enc, 8, 10, sampling="stratified", seed=17)
        windows_b, starts_b = _select_int_swap_windows(
            enc, 8, 10, sampling="stratified", seed=17)
        self.assertEqual(starts_a, starts_b)
        self.assertEqual(len(set(starts_a)), 10)
        self.assertEqual([int(w[0]) for w in windows_a],
                         [int(w[0]) for w in windows_b])
        # One ctx-aligned block comes from each tenth of the full corpus.
        for i, start in enumerate(starts_a):
            block = start // 8
            self.assertGreaterEqual(block, i * 10)
            self.assertLess(block, (i + 1) * 10)

    def test_lcb_penalizes_text_sensitive_gain(self):
        stable = _int_swap_score([0.2, 0.2, 0.2, 0.2], "lcb", 1.0)
        noisy = _int_swap_score([0.5, -0.1, 0.5, -0.1], "lcb", 1.0)
        self.assertAlmostEqual(stable[0], noisy[0])
        self.assertGreater(stable[3], noisy[3])


class PayloadTest(unittest.TestCase):
    def test_schema_is_distinct_from_measured_curve(self):
        gains = {("qk", 0): 0.5, ("qk", 1): -0.1}
        p = build_int_swap_payload(
            gains, {"int_bits": 7}, None,
            model_path="m", total_blocks=2, operators={"qk"})
        self.assertEqual(p["format"], "scmp_llm_int_swap_v1")
        self.assertIn("swap_gain", p)
        # must NOT reuse the key that means "SC error, rank by max(...[1:])"
        self.assertNotIn("buckets", p)
        self.assertNotIn("level_mean_error", json.dumps(p))
        self.assertEqual(p["swap_gain"]["qk:b0"], 0.5)


def _write_int_swap_json(path, gains, int_bits=7, overlap=0.83,
                         selection_score=None, score_method="mean"):
    payload = {
        "format": "scmp_llm_int_swap_v1",
        "method": "int_swap",
        "total_blocks": 2,
        "swap_gain": gains,
        "measurement": {"int_bits": int_bits,
                        "ranking": {"score_method": score_method}},
        "split_half": {"top_overlap": overlap},
    }
    if selection_score is not None:
        payload["selection_score"] = selection_score
    path.write_text(json.dumps(payload))


def _write_curve_json(path, buckets, layer_buckets=2):
    path.write_text(json.dumps({
        "layer_buckets": layer_buckets,
        "buckets": buckets,
    }))


class HpcaSelectorTest(unittest.TestCase):
    """Drives the auto_hybrid_config python block exactly as hpca does."""

    def _select(self, sens_path, out_path, layers, frac, bits):
        src = HPCA.read_text()
        start = src.index("python - \"$sens_json\"")
        body = src[src.index("<<'PY'", start) + len("<<'PY'"):]
        body = body[:body.index("\nPY\n")]
        return subprocess.run(
            [sys.executable, "-c", body,
             str(sens_path), str(out_path), str(layers), str(frac), str(bits)],
            capture_output=True, text=True)

    def test_int_swap_ranks_by_gain_descending(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sens, out = d / "s.json", d / "o.json"
            _write_int_swap_json(sens, {
                "qk:b0": 0.9, "qk:b1": 0.1,
                "down_proj:b0": 0.5, "down_proj:b1": -0.4,
            })
            r = self._select(sens, out, 2, 0.5, 7)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = json.loads(out.read_text())
            self.assertEqual(got["selection"]["method"],
                             "top_fraction_by_int_swap_gain")
            self.assertEqual(got["selection"]["selected_entries"], 2)
            self.assertEqual(got["selection"]["split_half_top_overlap"], 0.83)
            # top-2 by gain = qk:b0 (0.9) and down_proj:b0 (0.5)
            self.assertEqual(got["schedule"]["qk"], ["int7", "sc"])
            self.assertEqual(got["schedule"]["down_proj"], ["int7", "sc"])

    def test_int_swap_refuses_width_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sens, out = d / "s.json", d / "o.json"
            _write_int_swap_json(sens, {"qk:b0": 0.9}, int_bits=8)
            r = self._select(sens, out, 2, 0.5, 7)   # mask at int7, ranked at int8
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("re-measure", r.stderr)

    def test_robust_int_swap_ranks_by_lcb_not_raw_mean(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sens, out = d / "s.json", d / "o.json"
            _write_int_swap_json(
                sens,
                {"qk:b0": 0.9, "qk:b1": 0.8,
                 "down_proj:b0": 0.1, "down_proj:b1": 0.0},
                selection_score={"qk:b0": 0.1, "qk:b1": 0.2,
                                 "down_proj:b0": 0.9,
                                 "down_proj:b1": 0.8},
                score_method="lcb")
            r = self._select(sens, out, 2, 0.5, 7)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = json.loads(out.read_text())
            self.assertEqual(got["selection"]["method"],
                             "top_fraction_by_int_swap_lcb")
            self.assertEqual(got["schedule"].get("qk", ["sc", "sc"]),
                             ["sc", "sc"])
            self.assertEqual(got["schedule"]["down_proj"], ["int7", "int7"])

    def test_measured_curve_path_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sens, out = d / "s.json", d / "o.json"
            _write_curve_json(sens, {
                "qk:t0:l0": {"level_mean_error": [0.0, 0.9]},
                "qk:t0:l1": {"level_mean_error": [0.0, 0.1]},
            })
            r = self._select(sens, out, 2, 0.5, 7)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = json.loads(out.read_text())
            self.assertEqual(got["selection"]["method"],
                             "top_fraction_by_bucket_worst_delta_loss")
            self.assertEqual(got["schedule"]["qk"], ["int7", "sc"])
            self.assertIsNone(got["selection"]["split_half_top_overlap"])

    def test_selected_count_matches_ceil_fraction(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sens, out = d / "s.json", d / "o.json"
            _write_int_swap_json(
                sens, {f"qk:b{i}": float(i) for i in range(9)})
            r = self._select(sens, out, 9, 0.20, 7)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = json.loads(out.read_text())
            self.assertEqual(got["selection"]["selected_entries"],
                             math.ceil(9 * 0.20))


if __name__ == "__main__":
    unittest.main()
