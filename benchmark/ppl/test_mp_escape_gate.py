"""Escape gate (R7) test suite — Track C of the V17 wave.

The gate: in the calibrated-table classify path, after min-max normalization,
rows whose normalized metric is STRICTLY above ``t_esc_b = mu_b + k * sigma_b``
escape to ``escape_stoc_len`` regardless of the band thresholds. mu_b / sigma_b
are per-bucket ``metric_mean`` / ``metric_std`` read from the calibrated table;
k is the global wrapper key ``escape_gate_k``.

Safety property proven here: ``escape_gate_k`` absent (None) is byte-identical
to the pre-gate classifier. The pre-gate reference is the kernels-submodule
``config.py`` pinned at commit 3ba991b, imported as a scratch module and
compared bit-for-bit against the working tree.

Env: conda annstention. No pytest; run
    python -m unittest benchmark.ppl.test_mp_escape_gate -v
"""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
KERNELS = REPO / "kernels"
if str(KERNELS) not in sys.path:
    sys.path.insert(0, str(KERNELS))

from scmp_kernels.mp.config import (  # noqa: E402
    AdaptiveMPConfig,
    _classify_rows_by_thresholds,
    adaptive_classify_rows,
)
from benchmark.ppl.mp_ladder_refine import (  # noqa: E402
    aggregate_trace_weights,
    wrapper_escape_gate,
)

LEVELS = [96, 64, 48, 32, 24, 16]
PREGATE_COMMIT = "3ba991b120fca922df30ab5ed94c6294ec09f164"


def _load_pregate_module():
    """Import the pinned pre-gate config.py as a scratch module.

    This is the byte-identity reference: the escape gate must not perturb the
    classifier when the wrapper carries no ``escape_gate_k``.
    """
    src = subprocess.check_output(
        ["git", "-C", str(KERNELS), "show",
         f"{PREGATE_COMMIT}:scmp_kernels/mp/config.py"],
        text=True,
    )
    if "escape_gate_k" in src:
        raise AssertionError(
            f"pinned commit {PREGATE_COMMIT} already contains the gate — "
            "not a clean pre-gate reference")
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix="_pregate_config.py", delete=False)
    tmp.write(src)
    tmp.close()
    spec = importlib.util.spec_from_file_location("_pregate_config", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves field types via sys.modules[cls.__module__]; the
    # module must be registered BEFORE exec so @dataclass processing works.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _table(metric_mean=0.30, metric_std=0.10, thresholds=None):
    """Minimal one-bucket + operator-default calibrated table.

    thresholds len == n_levels - 1, non-increasing in [0, 1]. Carries
    metric_mean / metric_std so the gate has stats to derive t_esc from.
    """
    if thresholds is None:
        thresholds = [0.80, 0.60, 0.40, 0.20, 0.10]
    payload = {
        "thresholds": list(thresholds),
        "metric_mean": metric_mean,
        "metric_std": metric_std,
    }
    return {
        "stoc_len_levels": list(LEVELS),
        "timestep_buckets": 1,
        "layer_buckets": 1,
        "operator_defaults": {"q_proj": dict(payload)},
        "buckets": {"q_proj:t0:l0": dict(payload)},
    }


def _write_table(tmpdir, table):
    import json
    p = Path(tmpdir) / "table.json"
    p.write_text(json.dumps(table))
    return p


def _config(tmpdir, table, *, escape_gate_k=None, escape_stoc_len=128):
    p = _write_table(tmpdir, table)
    return AdaptiveMPConfig(
        stoc_len_levels=list(LEVELS),
        threshold_table_path=str(p),
        escape_gate_k=escape_gate_k,
        escape_stoc_len=escape_stoc_len,
    )


class ByteIdentityWhenGateOff(unittest.TestCase):
    """Property 1: gate key absent => bit-exact pre-gate classification."""

    def test_row_levels_bit_identical_to_pregate(self):
        pregate = _load_pregate_module()
        with tempfile.TemporaryDirectory() as d:
            table = _table()
            p = _write_table(d, table)
            new_cfg = AdaptiveMPConfig(
                stoc_len_levels=list(LEVELS), threshold_table_path=str(p))
            old_cfg = pregate.AdaptiveMPConfig(
                stoc_len_levels=list(LEVELS), threshold_table_path=str(p))
            torch.manual_seed(0)
            for _ in range(25):
                metric = torch.randn(257).abs()
                a_new = adaptive_classify_rows(
                    metric, new_cfg, operator="q_proj",
                    block_idx=0, total_blocks=1)
                a_old = pregate.adaptive_classify_rows(
                    metric, old_cfg, operator="q_proj",
                    block_idx=0, total_blocks=1)
                self.assertTrue(
                    torch.equal(a_new.row_levels, a_old.row_levels))
                # index sets identical for every ladder rung
                self.assertEqual(
                    set(a_new.level_row_indices), set(a_old.level_row_indices))
                for sl in LEVELS:
                    self.assertTrue(torch.equal(
                        a_new.level_row_indices[sl].sort().values,
                        a_old.level_row_indices[sl].sort().values))

    def test_gate_off_equals_raw_threshold_classify(self):
        # Direct proof the gate addition is inert: the pre-gate body WAS exactly
        # _classify_rows_by_thresholds(metric_norm, levels, thresholds) with no
        # gate call. Gate-off full path must reproduce it bit-for-bit.
        with tempfile.TemporaryDirectory() as d:
            cfg = _config(d, _table())
            torch.manual_seed(1)
            for _ in range(20):
                metric = torch.randn(129).abs()
                a = adaptive_classify_rows(
                    metric, cfg, operator="q_proj", block_idx=0, total_blocks=1)
                m_min, m_max = metric.min(), metric.max()
                norm = (metric - m_min) / (m_max - m_min)
                thr = cfg.get_thresholds(
                    timestep=0, total_timesteps=1, operator="q_proj",
                    block_idx=0, total_blocks=1)
                ref = _classify_rows_by_thresholds(norm, list(LEVELS), thr)
                self.assertTrue(torch.equal(a.row_levels, ref.row_levels))

    def test_classify_level_values_unchanged_when_off(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _config(d, _table())
            self.assertEqual(cfg.classify_level_values(), list(LEVELS))


class FireBothSignConventions(unittest.TestCase):
    """Property 2: rows above t_esc escape, under both metric sign conventions.

    The gate compares in the SAME post-sign normalized space the band
    thresholds live in. A +1 (amax) metric and a -1 (inverted) metric that
    encode the same ranking must escape the SAME rows.
    """

    def _run(self, metric, k, sl=128):
        with tempfile.TemporaryDirectory() as d:
            # tight mu/sigma so a modest k still leaves t_esc < 1 and fires
            cfg = _config(d, _table(metric_mean=0.30, metric_std=0.10),
                          escape_gate_k=k, escape_stoc_len=sl)
            return adaptive_classify_rows(
                metric, cfg, operator="q_proj", block_idx=0, total_blocks=1)

    def test_top_row_escapes_when_above_threshold(self):
        # one dominant outlier => normalized 1.0 for it, small for the rest;
        # t_esc = 0.30 + 2.0*0.10 = 0.50, so only the outlier escapes.
        metric = torch.cat([torch.full((32,), 0.1), torch.tensor([10.0])])
        a = self._run(metric, k=2.0, sl=128)
        esc_idx = len(LEVELS)  # appended escape index (128 not a rung)
        self.assertEqual(int(a.row_levels[-1].item()), esc_idx)
        self.assertEqual(a.level_row_indices[128].tolist(), [32])
        # every other row stayed on the ladder
        self.assertTrue(bool((a.row_levels[:-1] < esc_idx).all().item()))

    def test_positive_scale_invariance(self):
        # The gate reads only metric_norm, and min-max normalization is
        # invariant to positive affine scaling. A +1 (amax) metric and the
        # same metric scaled 5x must escape identical rows — proving the gate
        # adds no magnitude/sign handling of its own.
        base = torch.linspace(0.1, 1.0, 40)
        a1 = self._run(base.clone(), k=1.5)
        a5 = self._run(base.clone() * 5.0, k=1.5)
        esc_idx = len(LEVELS)
        e1 = set((a1.row_levels == esc_idx).nonzero().flatten().tolist())
        e5 = set((a5.row_levels == esc_idx).nonzero().flatten().tolist())
        self.assertEqual(e1, e5)
        self.assertTrue(len(e1) >= 1)

    def test_inverted_convention_escapes_top_of_normalized_space(self):
        # Inverted (-1 sign) op: runtime negates the metric BEFORE classify, so
        # the "important" rows arrive as the most-negative raw values. Min-max
        # then maps those to the TOP of [0,1]. The gate escapes whatever is high
        # in normalized space, so for an inverted feed it escapes the raw-MIN
        # rows — the runtime negated precisely because small-raw = important.
        raw = torch.linspace(0.1, 1.0, 40)      # index 0 = smallest raw
        inv = -raw.clone()                       # post-sign metric at runtime
        a = self._run(inv, k=1.5)
        esc = set((a.row_levels == len(LEVELS)).nonzero().flatten().tolist())
        # gate escapes exactly the rows whose INVERTED-normalized metric > t_esc
        norm = (inv - inv.min()) / (inv.max() - inv.min())
        t_esc = 0.30 + 1.5 * 0.10               # mu + k*sigma = 0.45
        expected = set((norm > t_esc).nonzero().flatten().tolist())
        self.assertEqual(esc, expected)
        # they are the raw-SMALLEST rows (which normalize to the top), never
        # the raw-largest — proving no raw-magnitude/sign handling in the gate
        self.assertIn(0, esc)
        self.assertNotIn(39, esc)

    def test_exact_escaped_set_matches_norm_above_tesc(self):
        # With t_esc < 1 the escaped set is EXACTLY {rows: metric_norm > t_esc}.
        # min-max always sends the max row to 1.0, so at least one row escapes.
        metric = torch.linspace(0.0, 1.0, 100)   # already normalized
        k = 2.0
        mu, sigma = 0.30, 0.10
        t_esc = mu + k * sigma                    # 0.50
        a = self._run(metric, k=k, sl=128)        # table uses mu/sigma above
        esc = set((a.row_levels == len(LEVELS)).nonzero().flatten().tolist())
        norm = (metric - metric.min()) / (metric.max() - metric.min())
        expected = set((norm > t_esc).nonzero().flatten().tolist())
        self.assertEqual(esc, expected)
        self.assertTrue(len(esc) >= 1)


class TescGreaterEqualOneNeverFires(unittest.TestCase):
    """Property 3: t_esc >= 1.0 can never fire (strict compare, norm max 1)."""

    def test_high_k_yields_tesc_ge_one_no_escape(self):
        with tempfile.TemporaryDirectory() as d:
            # mu 0.30, sigma 0.10, k=7 => t_esc = 1.00 (>= 1.0) => never fires
            cfg = _config(d, _table(metric_mean=0.30, metric_std=0.10),
                          escape_gate_k=7.0, escape_stoc_len=128)
            metric = torch.cat([torch.full((16,), 0.1), torch.tensor([99.0])])
            a = adaptive_classify_rows(
                metric, cfg, operator="q_proj", block_idx=0, total_blocks=1)
            self.assertFalse(
                bool((a.row_levels == len(LEVELS)).any().item()))

    def test_tesc_exactly_one_excludes_the_max_row(self):
        with tempfile.TemporaryDirectory() as d:
            # k chosen so t_esc == 1.0 exactly; the max row normalizes to 1.0
            # and 1.0 > 1.0 is False, so it must NOT escape.
            cfg = _config(d, _table(metric_mean=0.0, metric_std=0.10),
                          escape_gate_k=10.0, escape_stoc_len=128)
            metric = torch.linspace(0.0, 1.0, 50)
            a = adaptive_classify_rows(
                metric, cfg, operator="q_proj", block_idx=0, total_blocks=1)
            self.assertFalse(
                bool((a.row_levels == len(LEVELS)).any().item()))


class WrapperRoundTrip(unittest.TestCase):
    """Property 4: wrapper_escape_gate parses new + legacy wrappers."""

    def test_legacy_wrapper_gate_off(self):
        self.assertEqual(wrapper_escape_gate({"type": "AdaptiveMPConfig"}), {})
        self.assertEqual(wrapper_escape_gate({"escape_gate_k": None}), {})

    def test_new_wrapper_parsed(self):
        g = wrapper_escape_gate({"escape_gate_k": 3.5, "escape_stoc_len": 128})
        self.assertEqual(g, {"escape_gate_k": 3.5, "escape_stoc_len": 128})

    def test_new_wrapper_defaults_stoc_len_128(self):
        g = wrapper_escape_gate({"escape_gate_k": 3.0})
        self.assertEqual(g["escape_stoc_len"], 128)


class CostAccounting(unittest.TestCase):
    """Property 5: escaped MAC share is priced at escape_stoc_len exactly."""

    def _trace(self, extra_128_frac):
        # ladder [96..16] + protected 112 + an escape 128 slice.
        base = [
            {"op": "qk", "stoc_len": 96, "macs": 100.0},
            {"op": "q_proj", "stoc_len": 64, "macs": 100.0},
            {"op": "up_proj", "stoc_len": 16, "macs": 100.0},
            {"op": "down_proj", "stoc_len": 112, "macs": 40.0},  # protected
        ]
        if extra_128_frac > 0:
            base.append({"op": "av", "stoc_len": 128, "macs": extra_128_frac})
        return {"groups": base}

    def test_escaped_rows_priced_at_128(self):
        trace = self._trace(extra_128_frac=60.0)
        agg = aggregate_trace_weights(
            trace, LEVELS, protected_stoc_len=112, escape_stoc_len=128)
        total = 100 + 100 + 100 + 40 + 60
        self.assertAlmostEqual(agg["escape_weight"], 60.0 / total, places=9)
        # cost = ladder + protected@112 + escape@128, all MAC-weighted
        expected = (96 * 100 + 64 * 100 + 16 * 100
                    + 112 * 40 + 128 * 60) / total
        self.assertAlmostEqual(agg["cost"], expected, places=6)

    def test_no_escape_length_is_backward_compatible(self):
        # Without an escape slice, escape_weight is 0 and cost matches the
        # gate-off call (escape_stoc_len argument omitted).
        trace = self._trace(extra_128_frac=0.0)
        with_arg = aggregate_trace_weights(
            trace, LEVELS, protected_stoc_len=112, escape_stoc_len=128)
        without = aggregate_trace_weights(
            trace, LEVELS, protected_stoc_len=112)
        self.assertEqual(with_arg["escape_weight"], 0.0)
        self.assertAlmostEqual(with_arg["cost"], without["cost"], places=9)

    def test_escape_colliding_with_rung_not_double_counted(self):
        # If escape length equals a ladder rung, its MACs price there already;
        # escape_weight must stay 0 and cost is unchanged.
        trace = {"groups": [
            {"op": "qk", "stoc_len": 96, "macs": 100.0},
            {"op": "q_proj", "stoc_len": 64, "macs": 100.0},
        ]}
        agg = aggregate_trace_weights(
            trace, LEVELS, protected_stoc_len=112, escape_stoc_len=64)
        self.assertEqual(agg["escape_weight"], 0.0)
        self.assertAlmostEqual(agg["cost"], (96 * 100 + 64 * 100) / 200,
                               places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
