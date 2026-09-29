"""Mocked end-to-end control-flow test for mp_v16_refine.main().

No GPU/torch: the model build, eval stream, and evaluator are replaced with
deterministic fakes; everything else (partition, funnel, gates, checkpoint,
resume, wrap-up, summary/tsv writing) runs for real.  The fake evaluator
gives any ladder containing a 28-cycle rung a -0.02-nat bonus, so exactly
the bottom-gap insertion should be found, replicated, confirmed, and then
the branch should stall to a patience stop.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark.ppl.mp_v16_refine as v16


CTX = 8


class FakeEnc:
    def __init__(self, n_tokens: int):
        self.n = int(n_tokens)

    def numel(self) -> int:
        return self.n


def fake_load_eval_stream(tokenizer, *, split, max_tokens, ctx):
    return FakeEnc((117 if split == "validation" else 20) * ctx)


def fake_windows_enc(enc_val, window_ids, ctx):
    return ("ENC", tuple(int(w) for w in window_ids))


def fake_eval_once(model, enc, *, ctx, levels, table_path, wrapper_path,
                   trace_path, split, metric_profile_path=None,
                   metric_profile_bins=257, window_losses=None):
    if isinstance(enc, tuple) and enc and enc[0] == "ENC":
        ids = list(enc[1])
    else:  # the full test stream is evaluated directly, not via windows
        ids = list(range(1000, 1000 + enc.numel() // ctx))
    levels = [int(x) for x in levels]
    bonus = -0.02 if 28 in levels else 0.0
    losses = [2.0 + 0.001 * (w % 997) + bonus for w in ids]
    if window_losses is not None:
        for loss in losses:
            window_losses.append((0, ctx - 1, loss))
    nll = sum(losses) / len(losses)
    top, floor = levels[0], levels[-1]
    frac = 0.0 if top == floor else (top - 32.0) / (top - floor)
    frac = min(max(frac, 0.0), 1.0)
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as f:
        json.dump({"groups": [
            {"macs": (1.0 - frac) * 1e9, "stoc_len": top},
            {"macs": frac * 1e9, "stoc_len": floor},
        ]}, f)
    if metric_profile_path is not None:
        with Path(metric_profile_path).open("w") as f:
            json.dump({"bins": 9, "groups": {"gate_proj:t0:l0": {
                "op": "gate_proj",
                "mac_weighted_hist": [10.0] * 9,
            }}}, f)
    return {
        "ppl": math.exp(nll),
        "nll": nll,
        "tokens": len(ids) * (ctx - 1),
        "seconds": 0.01,
        "row_avg_stoc_len": 0.0,
        "tracker_flop_avg_stoc_len": 0.0,
        "trace": str(trace_path),
        "table": str(table_path),
        "wrapper": str(wrapper_path),
        "levels": levels,
    }


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.table_path = root / "parent_table.json"
        with self.table_path.open("w") as f:
            json.dump({
                "stoc_len_levels": [96, 64, 48, 32, 24, 16],
                "method": "act_global_v9",
                "budget_ratio": 0.25,
                "budget_ref_stoc_len": 128,
                "operator_defaults": {"gate_proj": {
                    "thresholds": [0.9, 0.7, 0.5, 0.3, 0.1],
                    "counts": [1] * 6,
                }},
                "buckets": {"gate_proj:t0:l0": {
                    "thresholds": [0.9, 0.7, 0.5, 0.3, 0.1],
                    "counts": [1] * 6,
                }},
                "protected_channels": {},
            }, f)
        self.wrapper_path = root / "parent_wrapper.json"
        with self.wrapper_path.open("w") as f:
            json.dump({
                "type": "AdaptiveMPConfig",
                "stoc_len_levels": [96, 64, 48, 32, 24, 16],
                "threshold_table_path": str(self.table_path),
            }, f)
        self.outdir = root / "out"
        self.argv = [
            "--parent-wrapper", str(self.wrapper_path),
            "--model-path", "fake/model",
            "--output-dir", str(self.outdir),
            "--targets", "32",
            "--ctx", str(CTX),
            "--max-sweeps", "3",
            # Pin patience so this test deterministically exercises the
            # patience-STOP path (accept @ sweep 1, then 2 stalls) regardless
            # of the --patience default, which V17 raised from 2 to 3.
            "--patience", "2",
            "--test-tokens", "0",
        ]
        self.patches = [
            mock.patch.object(v16, "_build_parent_model",
                              lambda *a, **k: (object(), object())),
            mock.patch.object(v16, "_load_eval_stream",
                              fake_load_eval_stream),
            mock.patch.object(v16, "_windows_enc", fake_windows_enc),
            mock.patch.object(v16, "_eval_once", fake_eval_once),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _summary(self):
        branch = self.outdir / "target32p000"
        with (branch / "summary.json").open() as f:
            return branch, json.load(f)

    def test_full_branch_then_resume(self):
        self.assertEqual(v16.main(self.argv), 0)
        branch, summary = self._summary()
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(len(summary["accepted_moves"]), 1)
        self.assertIn("28", summary["accepted_moves"][0]["name"])
        self.assertIn(28, summary["best_levels"])
        self.assertEqual(summary["stop_reason"], "patience")
        self.assertLess(summary["confirm_ppl"],
                        math.exp(2.0 + 0.001 * 996))
        self.assertIsNotNone(summary["final_test_ppl"])
        self.assertAlmostEqual(summary["confirm_cost"], 32.0, places=6)
        self.assertTrue((branch / "checkpoint.json").exists())
        self.assertTrue((branch / "history.jsonl").exists())
        # v16.1: the threshold family is off by default.
        with (branch / "history.jsonl").open() as f:
            phases = {json.loads(line).get("phase") for line in f}
        self.assertNotIn("threshold", phases)
        self.assertTrue((self.outdir / "results.tsv").exists())
        self.assertTrue(Path(summary["best_wrapper"]).exists())

        # Completed-branch resume: second run must not redo the search.
        self.assertEqual(v16.main(self.argv), 0)
        _, again = self._summary()
        self.assertEqual(again["created_at"], summary["created_at"])

        # Mid-branch resume: drop the summary, keep the checkpoint; the
        # integrity probe must pass and the wrap-up must be rebuilt.
        (branch / "summary.json").unlink()
        self.assertEqual(v16.main(self.argv), 0)
        _, rebuilt = self._summary()
        self.assertEqual(rebuilt["status"], "ok")
        self.assertEqual(len(rebuilt["accepted_moves"]), 1)
        self.assertIn(28, rebuilt["best_levels"])
        self.assertEqual(rebuilt["stop_reason"], "patience")

    def test_corrupt_checkpoint_fails_loudly(self):
        self.assertEqual(v16.main(self.argv), 0)
        branch, _ = self._summary()
        (branch / "summary.json").unlink()
        with (branch / "checkpoint.json").open() as f:
            state = json.load(f)
        first = str(state["partition"]["confirm_windows"][0])
        state["incumbent"]["confirm_cache"][first] += 0.5
        with (branch / "checkpoint.json").open("w") as f:
            json.dump(state, f)
        with self.assertRaises(SystemExit):
            v16.main(self.argv)


if __name__ == "__main__":
    unittest.main()
