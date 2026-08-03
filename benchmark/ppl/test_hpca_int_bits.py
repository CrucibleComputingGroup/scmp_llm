"""Hybrid INT mask width is derived from the config budget, not hand-maintained.

The mask width must agree with the iso-precision comparator map in
``sc_cycles_for_cfg`` (an SC stream of NOMINAL length N resolves log2(N)
levels, and nominal = 2 x halved), because every iso-precision claim in the
paper assumes that correspondence. The old hand table silently disagreed on the
low-budget configs -- ``mp_avg32_v9`` masked at int7 while its budget is
compared against int6 -- which is how the 2026-07-18 dose wave ran every avg32
cell one bit richer than its own comparator.
"""

import subprocess
import unittest
from pathlib import Path

HPCA = Path(__file__).resolve().parents[2] / "hpca"

# The five comparator anchors: these MUST match sc_cycles_for_cfg exactly.
#   halved 128 / 96 / 64 / 48 / 32  ->  int 8 / 8 / 7 / 7 / 6
COMPARATOR_ANCHORS = {
    "sc_int8": 8,     # halved 128 -> nominal 256
    "sc_avg192": 8,   # halved  96 -> nominal 192
    "sc_int7": 7,     # halved  64 -> nominal 128
    "sc_avg96": 7,    # halved  48 -> nominal  96
    "sc_int6": 6,     # halved  32 -> nominal  64
}

DERIVED = {
    "mp_avg96_burst128": 8,
    "mp_avg96f_burst128": 8,
    "mp_avg64_burst128": 7,
    "mp_avg48_burst128": 7,
    "mp_avg40_v9": 7,    # halved 40 -> nominal 80  -> ceil(log2 80) = 7
    "mp_avg36_v9": 7,    # halved 36 -> nominal 72  -> ceil(log2 72) = 7
    "mp_avg32_v9": 6,    # halved 32 -> nominal 64  -> exactly 6
    "mp_avg32_burst128": 6,
    "mp_int7": 7,
}

# Configs the derivation deliberately moves off the legacy table. Each is a
# correction toward the config's own comparator.
CHANGED_VS_LEGACY = {
    "mp_avg32_v9": (6, 7),
    "mp_avg32_burst128": (6, 7),
    "mp_avg96f_burst128": (8, 7),
}


def _bits(cfg, *extra):
    r = subprocess.run(["bash", str(HPCA), *extra, "--print-int-bits", cfg],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"hpca --print-int-bits {cfg} failed: {r.stderr}")
    out = r.stdout.strip()
    assert out.isdigit(), f"--print-int-bits must print only the width, got {out!r}"
    return int(out)


def _bits_for_cycles(cycles):
    r = subprocess.run(
        ["bash", str(HPCA), "--print-int-bits-cycles", str(cycles)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(
            f"hpca --print-int-bits-cycles {cycles} failed: {r.stderr}")
    out = r.stdout.strip()
    assert out.isdigit(), out
    return int(out)


class IsoPrecisionWidthTest(unittest.TestCase):
    def test_direct_cycle_query_for_search_targets(self):
        self.assertEqual(
            {target: _bits_for_cycles(target)
             for target in (32, 40, 48, 64)},
            {32: 6, 40: 7, 48: 7, 64: 7})

    def test_matches_comparator_anchors(self):
        for cfg, want in COMPARATOR_ANCHORS.items():
            self.assertEqual(_bits(cfg), want, f"{cfg} width diverges from "
                             "sc_cycles_for_cfg's comparator map")

    def test_derived_widths(self):
        for cfg, want in DERIVED.items():
            self.assertEqual(_bits(cfg), want, cfg)

    def test_changed_configs_are_exactly_the_known_set(self):
        for cfg, (iso, legacy) in CHANGED_VS_LEGACY.items():
            self.assertEqual(_bits(cfg), iso, cfg)
            self.assertEqual(_bits(cfg, "--hybrid-int-bits-mode", "legacy"),
                             legacy, cfg)

    def test_unchanged_configs_stay_unchanged(self):
        for cfg in list(COMPARATOR_ANCHORS) + list(DERIVED):
            if cfg in CHANGED_VS_LEGACY:
                continue
            self.assertEqual(
                _bits(cfg), _bits(cfg, "--hybrid-int-bits-mode", "legacy"),
                f"{cfg} changed width but is not in the declared change set")

    def test_explicit_override_wins_over_derivation(self):
        # How a wave pins a width that is deliberately NOT iso-precision -- e.g.
        # the v18 int_swap wave staying at int7 on avg32 to remain comparable
        # with the 2026-07-18 dose anchors.
        self.assertEqual(_bits("mp_avg32_v9", "--hybrid-int-bits", "7"), 7)
        self.assertEqual(
            _bits("mp_avg32_v9", "--hybrid-int-bits", "7",
                  "--hybrid-int-bits-mode", "legacy"), 7)

    def test_bad_mode_is_rejected(self):
        r = subprocess.run(
            ["bash", str(HPCA), "--hybrid-int-bits-mode", "bogus",
             "--print-int-bits", "mp_avg32_v9"],
            capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown --hybrid-int-bits-mode", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
