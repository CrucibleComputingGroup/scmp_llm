"""Round-7 PRE-REGISTRATION (2026-09-28): the refusal inputs and the target-scale rule, frozen
BEFORE any round-7 solve, fixed point, kappa decision or step-0 result exists.

What is pinned here (the screen builder refuses anything else):
  * MDE per cell (nats), used for the refusal rule pred_true <= -MDE. Derived now from files that
    already exist (derive_mde); a --mde override no longer exists, and every solve summary must
    carry exactly this value (prc_r7_solve.py solve --mde-nats <MDE>).
  * the ABSOLUTE family kappa (kappa_lin, kappa_att) that sets pred_true's scale, per kappa-decision
    branch (keep 0.42 / revert to kappa = 1) and for dense cells.
  * the authoritative kappa decision (the eval-side kappa_decision.json only).
  * the one target-scale rule: s0 from the capture manifest, ONE replay-based fixed point per arm
    on the step-0 INC16 profile with the step-0 PARENT16 P, one re-solve at s1, no iteration and no
    re-screen (prc_fixedpoint_r7_20260928.py).
  * the pre-registered primary ladder policy ("up") and the best(all) test-cost bound.
The JSON record benchmark/ppl/kbands/prc_r7_prereg_20260928.json is written ONCE (exclusive) with
its freeze time; the builder checks that the record equals these constants, that it predates every
solve summary / fixed-point record / kappa decision it accepts, and records its sha.

  python benchmark/ppl/prc_r7_prereg_20260928.py derive     # recompute the MDE table (read-only)
  python benchmark/ppl/prc_r7_prereg_20260928.py write      # write the frozen JSON record (once)
  python benchmark/ppl/prc_r7_prereg_20260928.py verify
Units: nats (MDE); HALVED code lengths elsewhere (nominal = 2x).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KDIR = REPO / "benchmark/ppl/kbands"
RECORD = KDIR / "prc_r7_prereg_20260928.json"
KB = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands")
SCHEMA = "prc-r7-prereg-v1"

# ---------------------------------------------------------------------------------------------
# MDE (nats)
# ---------------------------------------------------------------------------------------------
Z_CONFIRM = 2.0                       # confirmation: one-sided paired z < -2
Z_POWER80 = 0.8416212335729143        # Phi^-1(0.80)
N_CONFIRM = 32
MDE_FACTOR = (Z_CONFIRM + Z_POWER80) / math.sqrt(N_CONFIRM)   # 0.502333
HELDOUT_PAIRS = (("gfisla", "gfis"), ("gfisla", "parent"), ("gfis", "parent"), ("c17", "parent"))
MDE_RULE = ("MDE80 = (2 + z_0.80)/sqrt(32) * sigma_w = 0.50233 * sigma_w (the confirmation test: 32 paired "
            "TRAIN windows, one-sided z < -2, power 0.80). sigma_w = the LARGEST paired per-window SD among the "
            "pre-existing 16-window allocation pairs of the cell: prc2/<cell>_c7_heldout_nll.json pairs "
            "gfisla-gfis, gfisla-parent, gfis-parent, c17-parent, and the round-6 arm A - INC on the diag windows "
            "where it exists (the attribution-only oracles ATT128/LIN128 are excluded). Rounded UP to 1e-4 nats. "
            "Conservative by construction (the largest SD of any comparable move); it is the upper end of the "
            "scout's MDE80 ranges (30B t32 0.010-0.013, t40 0.006-0.009, t48 0.007-0.009).")
MDE_NATS = {"30B_t32": 0.0129, "30B_t40": 0.0092, "30B_t48": 0.0091, "4B_t40": 0.0148, "4B_t64": 0.0073}


def _sd_pairs(cell: str) -> list:
    out = []
    held = KB / f"prc2/{cell}_c7_heldout_nll.json"
    h = json.loads(held.read_text())
    t = h["tables"]
    for a, b in HELDOUT_PAIRS:
        if a in t and b in t:
            d = [x - y for x, y in zip(t[a]["window_nll"], t[b]["window_nll"])]
            out.append({"source": f"{held}:{a}-{b}", "n": len(d), "sd_w": statistics.stdev(d)})
    r6 = KB / f"prc_r6_20260928/{cell}"
    if (r6 / "A_nll.json").is_file() and (r6 / "INC_nll.json").is_file():
        a = json.loads((r6 / "A_nll.json").read_text())["window_nll"]
        b = json.loads((r6 / "INC_nll.json").read_text())["window_nll"]
        d = [x - y for x, y in zip(a, b)]
        out.append({"source": f"{r6}/A_nll.json - INC_nll.json", "n": len(d), "sd_w": statistics.stdev(d)})
    return out


def derive_mde(cells=tuple(MDE_NATS)) -> dict:
    """Recompute the MDE table from the source files (read-only)."""
    res = {}
    for c in cells:
        pairs = _sd_pairs(c)
        sd = max(p["sd_w"] for p in pairs)
        res[c] = {"pairs": pairs, "sigma_w": sd, "mde_raw": MDE_FACTOR * sd,
                  "mde_nats": math.ceil(MDE_FACTOR * sd * 1e4 - 1e-9) / 1e4}
    return res


# ---------------------------------------------------------------------------------------------
# kappa: authority and absolute scale
# ---------------------------------------------------------------------------------------------
KAPPA_AUTHORITY = (
    "ONE kappa rule for round 7: kappa_att 0.83 +- 0.07, pooled over 30B t32 + t40, keep ratio 0.42 only if the "
    "step-0 budget-shift kappa_lin exceeds kappa_att by more than sqrt(SE_bs^2 + 0.07^2), otherwise (inside the "
    "band, below it, or undefined) revert to kappa = 1 (one-sided; prc_eval_r7_20260928.KAPPA_RULE == "
    "prc_r7_solve.KAPPA_RULE id prc-r7-kappa-rule-v2-20260928). The eval-side kappa_decision.json written by "
    "prc_step0_r7_20260928.py kappa (schema prc-step0-r7-v1-kappa) is the decision RECORD the screen builder "
    "requires; the calib-side solver re-derives the same branch from its statistic and refuses pins that "
    "disagree. The calib-side v1 constants (prc_r7_capture v1 text and scout_calib section 3: 0.82 +- 0.073, "
    "two-sided) are superseded and never read.")
# pred_true = kappa_lin * dL + kappa_att * dA (nats vs INC). Pinned per decision branch:
#  keep_0.42          CRIT / scout pooled held-out kappa_lin 1.96, kappa_att 0.83 (ratio 0.4235 ~ 0.42)
#  revert_to_kappa_1  r = 1 in the objective; the step-0 check did not confirm kappa_lin > kappa_att, so
#                     BOTH families are priced at kappa_att = 0.83 (the measured attention scale)
#  dense              no family correction on dense cells (plan section 1): kappa = 1 for both
KAPPA_PINS = {"keep_0.42": {"kappa_lin": 1.96, "kappa_att": 0.83},
              "revert_to_kappa_1": {"kappa_lin": 0.83, "kappa_att": 0.83},
              "dense": {"kappa_lin": 1.0, "kappa_att": 1.0}}
KAPPA_DECISION_RATIO = {"keep_0.42": 0.42, "revert_to_kappa_1": 1.0}
KAPPA_RATIO_TOL = 0.005    # 0.83/1.96 = 0.4235 vs the rule's 0.42
KAPPA_RULE_TEXT = (
    "Solve summaries must carry exactly (kappa_lin, kappa_att) = KAPPA_PINS[branch] (30B: the branch of the "
    "authoritative kappa_decision.json; dense 4B: 'dense'). UK and K arms use r = kappa_att/kappa_lin from those "
    "pins, U uses r = 1. pred_true = kappa_lin*dL + kappa_att*dA with the SAME pins for every arm of a cell.")


def kappa_pins_for(model: str, decision: str | None) -> dict:
    if model != "30B":
        return dict(KAPPA_PINS["dense"], branch="dense")
    if decision not in KAPPA_DECISION_RATIO:
        raise ValueError(f"unknown kappa decision {decision!r}")
    return dict(KAPPA_PINS[decision], branch=decision)


# ---------------------------------------------------------------------------------------------
# target scale: the one pre-registered rule
# ---------------------------------------------------------------------------------------------
S0 = {"30B_t32": 1.0037, "30B_t40": 1.0043, "30B_t48": 1.011, "4B_t40": 1.001, "4B_t64": 1.0015}
S0_SOURCE = ("scout_calib section 4c s_pop (30B: held-out, trace-equivalent INC re-target to the parent's cost); "
             "4B: 1 + (1 - r_test)/0.95 (approximate). Adopted verbatim as prc_r7_capture_20260928.json "
             "cells[].prereg.s0.")
FIXED_POINT = {
    "rounding": 1e-4, "band": [0.97, 1.05], "replay_flag_tolerance": 0.0025,
    "rule": ("s0 = S0[cell]. Each registered arm (30B: UK, K, U, with U omitted whenever the kappa ratio is 1 "
             "because it then equals UK; dense: U; never the CPU control J) is solved ONCE at s0 with the pinned "
             "kappa and MDE (prc_r7_solve.py solve). C(s0) = CPU replay of the arm's s0 table on the step-0 INC16 "
             "profile of the cell (INC's deployed trajectory on the cell's 16 screen windows; every profiled row "
             "re-dispatched through the arm's OWN ladders and thresholds; escaped attention rows and the protected "
             "slice stay fixed; an L7 psl arm moves the protected slice by psl_arm/psl_INC). P = the step-0 "
             "PARENT16 exact trace cost on the same windows (parent_reference.json). U = the protected-slice "
             "cycles per SC MAC of the step-0 INC16 trace (analyze_trace U_prot; escaped rows are inside the "
             "solve's target and scale with s). s1 = round(s0 (P - U)/(C(s0) - U), 1e-4), refused outside "
             "[0.97, 1.05]. Each arm is re-solved ONCE at its own s1 and only that table is screened (the builder "
             "refuses any other target_scale). No iteration, no second re-solve, no re-screen. The screen's "
             "measured |C/P - 1| <= 1% gate is unchanged; the replayed C(s1)/P is recorded (non-binding, flagged "
             "when |C(s1)/P - 1| > 0.25%). A cell without a complete, non-simulated step-0 reference (INC16 "
             "profile + PARENT16) is disabled."),
}
LADDER_POLICY = "up"
ARMS = ("UK", "K", "U")
BEST_ALL_TEST_COST_BOUND = 0.015
BEST_ALL_RULE = ("The full-test PPL of a confirmed primary enters best(all) whatever its sign, provided every "
                 "protocol check passed and its full-test cost is within +-1.5% of the parent's full-test cost "
                 "(the user's ~1% cost-drift rule, with 0.5 pp slack for test-vs-held-out shift); outside that "
                 "bound it is recorded but not entered, and flagged for the user. |C_test/P_test - 1| > 1% is "
                 "flagged either way.")


def record() -> dict:
    return {"schema": SCHEMA, "units": "MDE in nats (natural-log PPL units, %PPL ~ 100 x nats)",
            "mde": {"rule": MDE_RULE, "factor": MDE_FACTOR, "nats": dict(MDE_NATS)},
            "kappa": {"authority": KAPPA_AUTHORITY, "pins": KAPPA_PINS, "decision_ratio": KAPPA_DECISION_RATIO,
                      "ratio_tolerance": KAPPA_RATIO_TOL, "rule": KAPPA_RULE_TEXT},
            "target_scale": {"s0": dict(S0), "s0_source": S0_SOURCE, **FIXED_POINT},
            "ladder_policy_primary": LADDER_POLICY, "arms": list(ARMS),
            "best_all": {"test_cost_bound": BEST_ALL_TEST_COST_BOUND, "rule": BEST_ALL_RULE}}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


OUTPUTS_ROOT = KB / "prc_r7_20260928"


def write_record(path: Path = RECORD, *, outputs_root: Path = OUTPUTS_ROOT) -> Path:
    derived = derive_mde()
    for c, d in derived.items():
        if d["mde_nats"] != MDE_NATS[c]:
            raise SystemExit(f"{c}: derived MDE {d['mde_nats']} != frozen {MDE_NATS[c]}")
    outputs_root = Path(outputs_root)
    rec = dict(record(), mde_derivation=derived,
               frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               frozen_before=("any round-7 capture, solve, fixed point, kappa decision or step-0 result "
                              "(none existed at freeze time: see existing_outputs_at_freeze)"),
               outputs_root_scanned=str(outputs_root),
               existing_outputs_at_freeze=sorted(str(p) for p in outputs_root.glob("**/*"))
               if outputs_root.exists() else [])
    with Path(path).open("x") as f:
        f.write(json.dumps(rec, indent=1) + "\n")
    return Path(path)


def load_verified(path: Path = RECORD) -> tuple[dict, str, float]:
    """(record, sha256, frozen epoch seconds); raises unless the file equals these constants."""
    rec = json.loads(Path(path).read_text())
    body = {k: v for k, v in rec.items() if k not in ("mde_derivation", "frozen_utc", "frozen_before",
                                                     "existing_outputs_at_freeze", "outputs_root_scanned")}
    if json.loads(json.dumps(record())) != body:
        raise ValueError(f"{path} differs from the pre-registration constants in {Path(__file__).name}")
    if rec.get("existing_outputs_at_freeze"):
        raise ValueError(f"{path} was frozen after round-7 outputs existed: {rec['existing_outputs_at_freeze'][:3]}")
    t = datetime.datetime.fromisoformat(rec["frozen_utc"]).timestamp()
    return rec, sha256_file(path), t


def main(argv=None) -> int:
    cmd = (argv or sys.argv[1:] or ["verify"])[0]
    if cmd == "derive":
        print(json.dumps(derive_mde(), indent=1))
    elif cmd == "write":
        print(f"wrote {write_record()}")
    elif cmd == "verify":
        rec, sha, t = load_verified()
        print(json.dumps({"ok": True, "sha256": sha, "frozen_utc": rec["frozen_utc"], "mde": rec["mde"]["nats"]}))
    else:
        raise SystemExit(f"unknown command {cmd!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
