"""Mocked end-to-end control-flow test for --ladder-groups op:layer.

Same shape as test_mp_v16_refine_flow, but the fake evaluator is group-aware:
it accepts a {group: ladder} dict for ``levels`` and emits a trace carrying
``op`` and ``block`` so the grouped occupancy path is exercised for real.

This is the check a compile cannot give.  The last wave died on a path with
56 passing tests because no test ever fed it the object the real run produces
-- here that object is the ladder DICT, which flows through deploy(),
occupancy_of(), the proposal builder, the A->B gate, the confirm veto, the
checkpoint and the wrap-up.
"""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark.ppl.mp_v16_refine as v16

CTX = 8
LEVELS = [96, 64, 48, 32, 24, 16]
OPS = ("qk", "av", "q_proj", "o_proj", "gate_proj", "down_proj")
N_LAYERS = 12


class FakeEnc:
    def __init__(self, n):
        self.n = int(n)

    def numel(self):
        return self.n


class FakeConfig:
    num_hidden_layers = N_LAYERS


class FakeModel:
    config = FakeConfig()


def fake_load_eval_stream(tokenizer, *, split, max_tokens, ctx):
    return FakeEnc((117 if split == "validation" else 20) * ctx)


def fake_windows_enc(enc_val, window_ids, ctx):
    return ("ENC", tuple(int(w) for w in window_ids))


def _flat(levels):
    """Accept either a flat ladder or a {group: ladder} dict."""
    if isinstance(levels, dict):
        return sorted({int(x) for lad in levels.values() for x in lad},
                      reverse=True)
    return [int(x) for x in levels]


def fake_eval_once(model, enc, *, ctx, levels, table_path, wrapper_path,
                   trace_path, split, metric_profile_path=None,
                   metric_profile_bins=257, window_losses=None):
    if isinstance(enc, tuple) and enc and enc[0] == "ENC":
        ids = list(enc[1])
    else:
        ids = list(range(1000, 1000 + enc.numel() // ctx))
    # CONTRACT: the runtime only ever accepts a FLAT ladder.  _eval_once ->
    # _install_runtime_table builds an AdaptiveMPConfig and indexes
    # stoc_len_levels, so a dict here is a bug (it produced KeyError: 0 in
    # production).  Assert it rather than tolerating it, which is what let the
    # bug through the first time.
    assert isinstance(levels, list), (
        f"_eval_once must receive a flat ladder, got {type(levels).__name__}")
    with Path(table_path).open() as _tf:
        _tbl = json.load(_tf)
    # STRONGER CONTRACT: AdaptiveMPConfig.load_threshold_table REJECTS any
    # runtime ladder that differs from the deployed table's own top-level
    # stoc_len_levels.  Passing the union of the group ladders satisfied
    # "is a list" but failed this, so assert the real invariant.
    assert levels == [int(x) for x in _tbl["stoc_len_levels"]], (
        f"runtime levels {levels} != table stoc_len_levels "
        f"{_tbl['stoc_len_levels']}; AdaptiveMPConfig would reject this")
    flat = _flat(levels)
    per_group = {}
    for bkey, payload in (_tbl.get("buckets") or {}).items():
        if "stoc_len_levels" in payload:
            per_group[bkey] = [int(x) for x in payload["stoc_len_levels"]]
    per_group = per_group or {"global": flat}
    # reward giving the MLP groups a lower floor than the score groups --
    # exactly the move a single shared ladder cannot express
    mlp_floor = min((min(lad) for k, lad in per_group.items()
                     if k.split(":")[0] in ("gate_proj", "up_proj",
                                            "down_proj")), default=flat[-1])
    score_top = max((max(lad) for k, lad in per_group.items()
                     if k.split(":")[0] in ("qk", "av")), default=flat[0])
    bonus = -0.02 if (mlp_floor < 16 and score_top >= 96) else 0.0
    losses = [2.0 + 0.001 * (w % 997) + bonus for w in ids]
    if window_losses is not None:
        for loss in losses:
            window_losses.append((0, ctx - 1, loss))
    nll = sum(losses) / len(losses)

    groups = []
    for op in OPS:
        for block in range(N_LAYERS):
            lb = block * 4 // N_LAYERS
            lad = per_group.get(f"{op}:t0:l{lb}", flat)
            groups.append({"op": op, "block": block,
                           "stoc_len": int(lad[0]), "macs": 1e6})
            groups.append({"op": op, "block": block,
                           "stoc_len": int(lad[-1]), "macs": 1e6})
    # Protected pins, when the deployed table declares them: real traces carry
    # their MACs at the pin stream length, and pc_length moves are only
    # proposed when that mass is measurably non-zero.
    try:
        with Path(table_path).open() as f:
            pins = (json.load(f).get("protected_channels") or {})
    except (OSError, ValueError):
        pins = {}
    pin_sl = pins.get("stoc_len")
    if pin_sl:
        groups.append({"op": "down_proj", "block": 0,
                       "stoc_len": int(pin_sl), "macs": 2e6})
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as f:
        json.dump({"groups": groups}, f)
    if metric_profile_path is not None:
        with Path(metric_profile_path).open("w") as f:
            json.dump({"bins": 9, "groups": {"gate_proj:t0:l0": {
                "op": "gate_proj", "mac_weighted_hist": [10.0] * 9}}}, f)
    return {
        "ppl": math.exp(nll), "nll": nll,
        "tokens": len(ids) * (ctx - 1), "seconds": 0.01,
        "row_avg_stoc_len": 0.0, "tracker_flop_avg_stoc_len": 0.0,
        "trace": str(trace_path), "table": str(table_path),
        "wrapper": str(wrapper_path), "levels": flat,
    }


class GroupFlowTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        buckets = {}
        for op in OPS:
            for l in range(4):
                buckets[f"{op}:t0:l{l}"] = {
                    "thresholds": [0.9, 0.7, 0.5, 0.3, 0.1],
                    "counts": [1] * 6,
                    "fractions": [1 / 6] * 6,
                    "metric_mean": 0.5, "metric_std": 0.1,
                }
        self.table_path = root / "parent_table.json"
        with self.table_path.open("w") as f:
            json.dump({
                "stoc_len_levels": LEVELS,
                "method": "act_global_v9",
                "budget_ratio": 0.25,
                "budget_ref_stoc_len": 128,
                "layer_buckets": 4,
                "operator_defaults": {},
                "buckets": buckets,
                "protected_channels": {},
            }, f)
        self.wrapper_path = root / "parent_wrapper.json"
        with self.wrapper_path.open("w") as f:
            json.dump({
                "type": "AdaptiveMPConfig",
                "stoc_len_levels": LEVELS,
                "threshold_table_path": str(self.table_path),
            }, f)
        self.outdir = root / "out"
        self.argv = [
            "--parent-wrapper", str(self.wrapper_path),
            "--model-path", "fake/model",
            "--output-dir", str(self.outdir),
            "--targets", "32",
            "--ctx", str(CTX),
            "--max-sweeps", "2",
            "--patience", "2",
            "--test-tokens", "0",
            "--ladder-groups", "op:layer",
        ]
        self.patches = [
            mock.patch.object(v16, "_build_parent_model",
                              lambda *a, **k: (FakeModel(), object())),
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

    def test_group_mode_runs_end_to_end(self):
        self.assertEqual(v16.main(self.argv), 0)
        branch = self.outdir / "target32p000"
        with (branch / "summary.json").open() as f:
            summary = json.load(f)
        self.assertEqual(summary["status"], "ok")

    def test_seed_writes_nine_identical_group_ladders(self):
        v16.main(self.argv)
        branch = self.outdir / "target32p000"
        seeds = list(branch.glob("group00_seed.json"))
        self.assertTrue(seeds, "group seed table was never deployed")
        with seeds[0].open() as f:
            table = json.load(f)
        ladders = {json.dumps(p["stoc_len_levels"])
                   for p in table["buckets"].values()}
        # seeded identical: the seed must be the SAME config as the flat
        # incumbent, just expressed per group
        self.assertEqual(len(ladders), 1)

    def test_history_records_group_moves(self):
        v16.main(self.argv)
        branch = self.outdir / "target32p000"
        hist = branch / "history.jsonl"
        self.assertTrue(hist.exists(), "history.jsonl was never created")
        rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
        fams = {r.get("family") for r in rows}
        self.assertTrue(
            "group_exchange" in fams or "pc_length" in fams,
            f"no grouped candidate was ever proposed; families={fams}")


class GroupPcLengthTest(GroupFlowTest):
    """psl must stay adjustable in group mode, and must be seedable.

    The pins are a single global stream length. Prior waves accepted pc_len
    68/72/84 against V9's 112 on three of four models, so (a) dropping the
    family in group mode was a regression, and (b) starting at 112 burns
    sweeps rediscovering a known-bad default.
    """

    def setUp(self):
        super().setUp()
        # give the parent real pins so the pc_length family is reachable
        with self.table_path.open() as f:
            table = json.load(f)
        table["protected_channels"] = {
            "indices": {"down_proj@l0": [0, 1]},
            "stoc_len": 112,
            "global_mac_weighted_frac": 0.03,
        }
        with self.table_path.open("w") as f:
            json.dump(table, f)

    def test_pc_length_is_proposed_in_group_mode(self):
        v16.main(self.argv + ["--pc-lengths", "96:88:80:72"])
        branch = self.outdir / "target32p000"
        rows = [json.loads(l) for l
                in (branch / "history.jsonl").read_text().splitlines()
                if l.strip()]
        fams = {r.get("family") for r in rows}
        self.assertIn("pc_length", fams,
                      f"pc_length was not proposed in group mode; got {fams}")

    def test_seed_repins_before_the_search_starts(self):
        v16.main(self.argv + ["--seed-protected-sl", "80"])
        branch = self.outdir / "target32p000"
        seeds = sorted(branch.glob("seed_psl.json"))
        self.assertTrue(seeds, "no re-pinned seed table was deployed")
        with seeds[0].open() as f:
            table = json.load(f)
        self.assertEqual(table["protected_channels"]["stoc_len"], 80)

    def test_without_the_flag_psl_is_left_at_the_parent_value(self):
        v16.main(self.argv)
        branch = self.outdir / "target32p000"
        self.assertFalse(list(branch.glob("seed_psl.json")),
                         "psl was re-pinned without --seed-protected-sl")

    def test_resume_and_pc_move_validate_candidate_table_pin(self):
        """A resumed seed and a pc move must not reuse the parent pin.

        Production failure 54236683 had root pin 112, checkpoint/table pin
        80, and a legal adaptive rung 112.  Proposal construction correctly
        used 80, but deploy() closed over the root 112 and killed the job.
        The same captured-state bug also mis-validates a pc_length proposal:
        after changing the table pin to 88, rung 80 is adaptive again.
        """
        seed_argv = self.argv + [
            "--seed-protected-sl", "80", "--max-sweeps", "0"]
        self.assertEqual(v16.main(seed_argv), 0)
        branch = self.outdir / "target32p000"

        # Turn the completed zero-sweep fixture into the same mid-branch
        # checkpoint shape the lane's automatic resubmit consumes.
        (branch / "summary.json").unlink()
        checkpoint_path = branch / "checkpoint.json"
        with checkpoint_path.open() as f:
            checkpoint = json.load(f)
        checkpoint["stop_reason"] = None
        with checkpoint_path.open("w") as f:
            json.dump(checkpoint, f)

        def proposals(table, occupancy, group_ladders, **kwargs):
            legal_parent_pin = copy.deepcopy(group_ladders)
            legal_parent_pin["mlp:middle"] = [112, 64, 48, 32, 24, 16]

            changed_pin_levels = copy.deepcopy(group_ladders)
            changed_pin_levels["mlp:middle"] = [96, 80, 48, 32, 24, 16]
            changed_pin_table = copy.deepcopy(table)
            changed_pin_table["protected_channels"]["stoc_len"] = 88
            return [
                {
                    "name": "legal_parent_pin_112",
                    "family": "group_value",
                    "levels": legal_parent_pin,
                    "table": copy.deepcopy(table),
                    "action": {"type": "group_value_move"},
                    "protected_sl": None,
                },
                {
                    "name": "pc_len_88_legal_old_pin_80",
                    "family": "pc_length",
                    "levels": changed_pin_levels,
                    "table": changed_pin_table,
                    "action": {"type": "group_pc_length"},
                    "protected_sl": 88,
                },
            ]

        resume_argv = self.argv + [
            "--seed-protected-sl", "80", "--max-sweeps", "1"]
        with mock.patch.object(v16, "build_group_proposals", proposals):
            self.assertEqual(v16.main(resume_argv), 0)

        parent_pin_candidate = branch / (
            "s01_precision_000_legal_parent_pin_112.json")
        changed_pin_candidate = branch / (
            "s01_precision_001_pc_len_88_legal_old_pin_80.json")
        self.assertTrue(parent_pin_candidate.exists())
        self.assertTrue(changed_pin_candidate.exists())
        with parent_pin_candidate.open() as f:
            self.assertEqual(
                json.load(f)["protected_channels"]["stoc_len"], 80)
        with changed_pin_candidate.open() as f:
            self.assertEqual(
                json.load(f)["protected_channels"]["stoc_len"], 88)


if __name__ == "__main__":
    unittest.main()
