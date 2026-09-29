"""Round-7 offline CPU solver: joint linear + attention lambda with a family-calibrated
currency over per-bucket attention ladders that reach the code cap (2026-09-28).

Reads one calib10_r7 capture (benchmark/ppl/mp_per_row_chunk_calib10_r7.py):
  <stem>_state.npz  calib9 solve state (linear Fisher bins, parent-ladder attention bins,
                    budgets, par_lin/par_att/fixed) + the r7 meta block (schema v2)
  <stem>_att.npz    per-row attention records at every measured length, both phases, with
                    the crc32 sample rank (calib9 --att-dump)
  <stem>_hold.npz   hold-phase linear per-(row, chunk) pairs (+ protected-slice records)
and re-solves calib9's joint lambda (solve_joint / staircase_dp / att_th_from / th_from_bins,
imported unchanged) on CPU with three pure-allocation changes:
  (a) per-bucket attention ladders A (policies below; the runtime's existing
      buckets[<op>:t0:l<q>].stoc_len_levels support, max 128 asserted);
  (b) the family-kappa currency: minimize kappa_lin*dF_lin + kappa_att*dF_att + lambda*cost,
      i.e. the attention bins' error scaled by r = kappa_att/kappa_lin (lambda absorbs
      kappa_lin). r == 1 passes the bins through untouched, so kappa_att == kappa_lin
      reproduces the plain joint lambda exactly (calib9's gfisla);
  (d) the budget target0 * s (calib9's target0 = parent cost on the calibration sample;
      s from the pre-registered fixed point, see `fixed-point`).
Escape semantics are unchanged: rows with normalized metric > mu+2tau stay out of the bins,
are priced at 128 and run at 128 (the runtime folds them into the 128 rung's index when 128 is
a rung). Where a bucket's own calibrated 128 threshold lies at or below mu+2tau the gate adds
nothing there (recorded per bucket as `escape_redundant`); on av (mu+2tau >= 1) the gate
never fires and 128 is a genuinely new rung.

Arms (one capture, pure allocation; tables differ from the parent only in per_row_chunk
thresholds, attention thresholds and attention bucket ladders):
  K   inherited (parent) ladders, kappa-corrected      (primary where the r6 gate FAILED)
  UK  policy ladders (default "up"), kappa-corrected   (primary where the r6 gate PASSED)
  U   policy ladders, kappa = 1                        (secondary)
  J   inherited ladders, kappa = 1 = calib10's in-process gfisla at 512 rows (CPU control)
Ladder policies (per bucket, lad = the parent's ladder, all rungs must have been measured):
  inherit  lad
  up       lad + {112 if 112 >= 1.1*top} + {128}        (pre-registered primary for U/UK)
  dense    lad + every measured rung in [min(lad), 128]  (CPU sensitivity)
  full     lad + every measured rung                     (CPU sensitivity; sub-floor rungs)

Held-out prediction (the capture's hold windows, MC-Fisher 'fis', calib9's HJ arithmetic):
full-table scorer (per_row_chunk thresholds + attention thresholds + attention ladders), split
by family. pred_true(arm vs INC) = kappa_lin*dL + kappa_att*dA with the registered pins of the
cell's kappa branch (KAPPA_RULE['prediction']; CRIT L1); refuse unless pred_true <= -MDE (the
cell's pre-registered MDE in nats).

Pre-registration (2026-09-28, v2): KAPPA_RULE below is the registered step-0 rule; its decision
is ONE-SIDED (plan section 4) and `solve` re-derives the branch from any decision record with it
(supersedes the two-sided v1 text; the eval side's copy must be aligned). The capture manifest's
per-cell `prereg` block (primary, arms, ladder policy, MDE, s0; values shared with
prc_r7_prereg_20260928) is enforced by `solve`, which also refuses a capture whose identity.json
is not identity_ok and any target scale that is not s0 or a registered fixed-point s1.

Subcommands:
  identity    capture == incumbent recipe (linear gfis exact; 128-row prefix x4 joint solve ==
              c7_gfisla exact; dump == in-process bins; held-out preds == c7 diag; the capture's
              OWN tables re-scored from the dumps == its own diag, bit for bit); then a pipeline
              check (emit one extended-ladder arm, lineage + runtime resolver sweep)
  solve       pre-registered arms at s0 (or each arm at its registered fixed-point s1) ->
              tables + solve_summary.json (validated, scored, J controls, refusal)
  score       full-table held-out Fisher scores of arbitrary tables (e.g. c17 / c17_s80)
  fixed-point s1 = s0 * (P - U) / (C - U)
  kappa-decision  the pre-registered step-0 rule (pooled over 30B t32 + t40; one-sided)
No GPU, no model. Units: HALVED code lengths (nominal = 2x).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from benchmark.ppl.mp_per_row_chunk_calib5 import (  # noqa: E402
    HALVE, SC_PREC, deploy_rungs)
from benchmark.ppl.mp_per_row_chunk_calib9_r6 import (  # noqa: E402
    _kparse, att_th_from, joint_cost, load_solve_state, resolve_tables, scaled_target,
    solve_joint, th_from_bins)
from benchmark.ppl import prc_r6_attn_diag_arms as R6A  # noqa: E402  (validators by import)

CAP = 128
ESC_LEN = 128
ATT_OPS = ("qk", "av")
LINEAR_OPS = R6A.LINEAR_OPS
R7_KEY = "prc10_r7"
PROVENANCE_KEYS = ("prc2_calib", "prc6_calib", "prc9_r6", R7_KEY)
UP_RUNGS = (112, 128)
UP_MIN_RATIO = 1.10
LADDER_POLICIES = ("inherit", "up", "dense", "full")
ARM_DEFS = {"K": ("inherit", True), "UK": ("policy", True), "U": ("policy", False),
            "J": ("inherit", False)}
STATE_SCHEMA_V2 = "prc-calib10-r7-solve-state-v2"
TOOL = "prc_r7_solve.py"

# The registered step-0 kappa rule (v2, frozen before any capture exists; copied verbatim into
# kbands/prc_r7_capture_20260928.json preregistered_rules.kappa_rule, whose preflight asserts
# equality). Plan section 4: K/UK keep the family currency only "provided the step-0 check
# confirms kappa_lin > kappa_att" -> ONE-SIDED. The absolute pins per branch are the eval-side
# pre-registration's (prc_r7_prereg_20260928.KAPPA_PINS; the capture builder asserts equality),
# so both sides predict and price with ONE pair per branch.
KAPPA_PINS = {"keep_0.42": {"kappa_lin": 1.96, "kappa_att": 0.83},
              "revert_to_kappa_1": {"kappa_lin": 0.83, "kappa_att": 0.83},
              "dense": {"kappa_lin": 1.0, "kappa_att": 1.0}}
KAPPA_DECISION_RATIO = {"keep_0.42": 0.42, "revert_to_kappa_1": 1.0}
EVAL_KAPPA_SCHEMA = "prc-step0-r7-v1-kappa"
KAPPA_RULE = {
    "id": "prc-r7-kappa-rule-v2-20260928",
    "kappa_att": 0.83,
    "kappa_att_se": 0.07,
    "kappa_att_source": ("CRIT section 1 / scout_calib section 3 (r7_kappa_se.json _pooled): pooled "
                         "held-out 30B kappa_att 0.8305 +- 0.0728 (p = 2; p = 1.5: 0.806 +- 0.074), "
                         "rounded as in the plan and the step-0 manifest (0.83 +- 0.07)"),
    "kappa_lin_heldout": 1.96,
    "kappa_lin_heldout_source": ("CRIT section 1: pooled 30B held-out REALLOCATION kappa_lin, "
                                 "sum measured / sum predicted over t32..t96 = 0.1631/0.0834"),
    "step0_cells": ["30B_t32", "30B_t40"],
    "applies_to": ["30B_t32", "30B_t40", "30B_t48"],
    "t48": ("30B t48 has no step-0 measurement and takes the pooled decision (one 30B currency; "
            "per-cell held-out ratios are consistent with one value, chi2 3.6 on 2 df)"),
    "pred_key": "pred_dnll_fis",
    "score_names": {"c17": "c17", "s80": "c17_s80"},
    "statistic": ("kappa_lin_bs = sum_c m_c / sum_c p_c over step0_cells; m_c = mean over the 16 "
                  "paired round-6 diag TRAIN windows of NLL(c17_s80) - NLL(c17) (step-0 "
                  "step0_measured.json pair.window_dnll, ok, not simulated); p_c = "
                  "tables.c17_s80.pred_dnll_fis - tables.c17.pred_dnll_fis from cell c's round-7 "
                  "capture diag (capture identity ok; the step-0 c17/c17_s80 tables byte-identical to "
                  "the capture's score tables). Identical to the step-0 manifest's statistic."),
    "se": ("SE_bs = sd_w(sum_c d_cw) / sqrt(n_w) / sum_c p_c (the cells share one window list, so "
           "the per-window sum carries their covariance); prediction noise excluded (disclosed). "
           "Different window lists -> undefined."),
    "decision": ("keep_0.42 iff kappa_lin_bs - kappa_att > band, band = sqrt(SE_bs^2 + "
                 "kappa_att_se^2) (ONE-SIDED: the step-0 check must CONFIRM kappa_lin > kappa_att). "
                 "Otherwise revert_to_kappa_1: inside the band, below it (direction contradicted) or "
                 "undefined."),
    "undefined": ("a step-0 cell missing / not ok / simulated, a capture diag or identity missing or "
                  "not ok, different window lists, n_w < 2, or sum_c p_c <= 0 -> revert_to_kappa_1 "
                  "and requires_user_review = true"),
    "pins": KAPPA_PINS,
    "decision_ratio": KAPPA_DECISION_RATIO,
    "currency": ("UK and K solve with r = kappa_att/kappa_lin of the branch pins (keep: 0.83/1.96 = "
                 "0.4235, the nominal 0.42; revert: 1); U always r = 1. Under revert UK == U, so U "
                 "is not solved separately; K at r = 1 equals the J control at the same scale "
                 "(disclosed). Dense cells: pins 'dense' (kappa = 1)."),
    "prediction": ("pred_true(table vs INC) = kappa_lin * dL + kappa_att * dA with the branch pins "
                   "(keep: 1.96 / 0.83; revert: 0.83 / 0.83 = both families at kappa_att, since the "
                   "step-0 check did not confirm kappa_lin > kappa_att; dense: 1 / 1), the SAME pair "
                   "for every arm of a cell; dL / dA = MC-Fisher held-out dNLL vs INC on the capture's "
                   "hold windows, full table (prc_r7_solve score_tables). kappa_lin_bs never enters "
                   "pred_true (reported only; gap_case flag)."),
    "authority": ("The branch is a function of the pooled (kappa_lin_bs, SE_bs, undefined) statistic "
                  "only. Either side may compute the statistic (prc_step0_r7_20260928 kappa -> "
                  "kappa_decision.json, or prc_r7_solve kappa-decision); prc_r7_solve re-derives the "
                  "branch with THIS one-sided rule from the record's kappa_lin_bs / se_kappa_lin_bs "
                  "and refuses pins that disagree with it. It supersedes the two-sided v1 text of "
                  "this manifest and of the step-0 manifest / prc_eval_r7_20260928.KAPPA_RULE "
                  "(|kappa_lin_bs - kappa_att| <= band; undefined -> keep), which must be aligned."),
    "flags_nonbinding": ("gap_case: keep with kappa_att / kappa_lin_bs > 0.55 (0.42 may "
                         "over-correct); direction_contradicted: kappa_lin_bs < kappa_att - band. "
                         "Reported only."),
}


# ---------------------------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------------------------
def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    text = json.dumps(value, indent=1, sort_keys=False, default=_json_default) + "\n"
    with path.open("x" if exclusive else "w") as f:
        f.write(text)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def arm_table_name(arm: str, r: float, s: float) -> str:
    """r7<ARM>_k<round(1000 r):04d>_s<round(1e4 s):05d>, e.g. r7UK_k0420_s10043."""
    if arm not in ARM_DEFS:
        raise ValueError(f"unknown arm {arm!r}")
    return f"r7{arm}_k{int(round(float(r) * 1000)):04d}_s{int(round(float(s) * 1e4)):05d}"


def parse_scales(spec) -> list:
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        s = float(tok)
        if not (math.isfinite(s) and 0.9 < s < 1.1):
            raise SystemExit(f"[r7] target scale {tok!r} outside (0.9, 1.1)")
        if abs(s * 1e4 - round(s * 1e4)) > 1e-6:
            raise SystemExit(f"[r7] target scale {tok!r} is not on the 1e-4 grid")
        out.append(s)
    if not out or len(set(out)) != len(out):
        raise SystemExit(f"[r7] bad --target-scales {spec!r}")
    return out


# ---------------------------------------------------------------------------------------------
# ladders
# ---------------------------------------------------------------------------------------------
def up_rungs(lad) -> list:
    top = max(int(v) for v in lad)
    return sorted({L for L in UP_RUNGS if L > top and L >= UP_MIN_RATIO * top} | {CAP})


def ladder_for(policy: str, lad, lv_meas) -> list:
    """Ascending per-bucket ladder A for one attention key (see module docstring)."""
    lad = sorted(int(v) for v in lad)
    meas = {int(v) for v in lv_meas}
    if policy == "inherit":
        A = lad
    elif policy == "up":
        A = sorted(set(lad) | set(up_rungs(lad)))
    elif policy == "dense":
        A = sorted(set(lad) | {L for L in meas if L >= lad[0]} | set(up_rungs(lad)))
    elif policy == "full":
        A = sorted(set(lad) | meas | set(up_rungs(lad)))
    else:
        raise ValueError(f"unknown ladder policy {policy!r}")
    missing = sorted(set(A) - meas)
    if missing:
        raise ValueError(f"ladder {A} needs unmeasured rungs {missing} (measured {sorted(meas)})")
    if len(A) < 2 or A[0] < 1 or A[-1] > CAP or len(set(A)) != len(A):
        raise ValueError(f"invalid ladder {A} (cap {CAP})")
    return A


# ---------------------------------------------------------------------------------------------
# calib9 semantics, verbatim (attention binning, deploy rule)
# ---------------------------------------------------------------------------------------------
def deploy_att(mn, th_desc, lad_asc, t_esc, esc_len):
    """calib9 main().deploy_att, verbatim: descending ladder, level i if mn >= th[i] (first
    match), else the last (shortest) level; mn > t_esc -> escape length."""
    D = list(reversed(lad_asc))
    lvl = np.full(mn.shape, len(D) - 1)
    for i in range(len(th_desc) - 1, -1, -1):
        lvl[mn >= th_desc[i]] = i
    L = np.asarray(D, dtype=np.float64)[lvl]
    if t_esc is not None and t_esc < 1.0 and esc_len:
        L[mn > t_esc] = esc_len
    return L


def deploy_att_runtime(mn, th_desc, lad_asc, t_esc, esc_len):
    """The runtime's float32 comparisons (_classify_rows_by_thresholds + _apply_escape_gate):
    thresholds and t_esc are cast to the metric dtype before comparing."""
    mn = np.asarray(mn, dtype=np.float32)
    D = list(reversed(lad_asc))
    lvl = np.full(mn.shape, len(D) - 1)
    for i in range(len(th_desc) - 1, -1, -1):
        lvl[mn >= np.float32(th_desc[i])] = i
    L = np.asarray(D, dtype=np.float64)[lvl]
    if t_esc is not None and t_esc < 1.0 and esc_len:
        L[mn > np.float32(t_esc)] = esc_len
    return L


def att_bins(mn, cur, mac, lv, A, t_esc, esc_len, n_bins):
    """calib9 main()'s attention binning for ONE key, verbatim, with the ladder A a parameter
    (calib9 always used A = the parent's ladder). Returns ((E, C, bin_min, A), escape mask)."""
    col = {int(L): i for i, L in enumerate(lv)}
    esc = (mn > t_esc) if (t_esc is not None and t_esc < 1.0 and esc_len) else np.zeros(mn.shape, bool)
    A_arr = np.asarray([int(v) for v in A], dtype=np.float64)
    ok = ~esc
    m_ok = mn[ok]
    order = np.argsort(m_ok, kind="stable")
    Bn = int(min(n_bins, max(m_ok.size, 1)))
    edges = np.linspace(0, m_ok.size, Bn + 1).astype(np.int64)
    E = np.add.reduceat(cur[ok][order][:, [col[int(L)] for L in A]], edges[:-1], axis=0)
    C = np.add.reduceat(mac[ok][order], edges[:-1])
    return (E, C, m_ok[order][edges[:-1]], A_arr), esc


def kappa_bins(abins, r):
    """The family-kappa currency: attention error x r (r = kappa_att / kappa_lin). r == 1.0
    returns the SAME dict (no arithmetic), so the plain joint lambda is reproduced exactly."""
    r = float(r)
    if not (math.isfinite(r) and r > 0):
        raise ValueError(f"kappa ratio must be finite and > 0, got {r}")
    if r == 1.0:
        return abins
    return {k: (E * r, C, bmin, A) for k, (E, C, bmin, A) in abins.items()}


# ---------------------------------------------------------------------------------------------
# capture loading
# ---------------------------------------------------------------------------------------------
class Capture:
    """One calib10_r7 capture: solve state (v2), attention dump, hold dump."""

    ATT_FIELDS = ("mn", "mac", "parL", "blk", "bh", "pos", "rank", "win", "raw", "fis", "fis_emp")
    HOLD_FIELDS = ("mn", "mac", "parL", "fis", "blk")

    def __init__(self, state_path, att_path, hold_path=None):
        self.paths = {"state": str(state_path), "att_dump": str(att_path),
                      "hold_dump": str(hold_path) if hold_path else None}
        meta, keys, akeys, lbins, abins = load_solve_state(state_path)
        self.meta, self.keys, self.akeys, self.lbins, self.abins_state = meta, keys, akeys, lbins, abins
        self.r7 = meta.get("r7") or {}
        if meta.get("schema") != STATE_SCHEMA_V2 or not self.r7:
            raise ValueError(f"{state_path}: not a calib10_r7 v2 solve state ({meta.get('schema')})")
        self.Lv = np.asarray(meta["ladder"], dtype=np.float64)
        self.ladder = [int(v) for v in meta["ladder"]]
        self.budget = {_kparse(s): float(v) for s, v in meta["budget"].items()}
        self.par_lin = float(meta["par_lin"])
        self.att_ladder = {_kparse(s): v for s, v in meta["att_ladder"].items()}
        self.n_bins = int(self.r7["bins"])
        self.n_tok = int(self.r7["n_tok"])
        self.total_blocks = int(self.r7["total_blocks"])
        self.layer_buckets = int(self.r7["layer_buckets"])
        self.att_rows = int(self.r7["att_rows_per_call"])
        self.prefix = int(self.r7["att_prefix_split"])
        self.att = {"calib": {}, "hold": {}}
        self.att_sizes = {ph: {_kparse(s): [int(n) for n in v]
                               for s, v in self.r7["att_part_sizes"][ph].items()}
                          for ph in ("calib", "hold")}
        with np.load(att_path, allow_pickle=False) as d:
            files = set(d.files)
            for ph in ("calib", "hold"):
                for k in self.att_sizes[ph]:
                    tag = f"att|{ph}|{k[0]}|{k[1]}"
                    rec = {f_: d[f"{tag}|{f_}"] for f_ in self.ATT_FIELDS if f"{tag}|{f_}" in files}
                    rec["lv"] = d[f"{tag}|lv"]
                    n = sum(self.att_sizes[ph][k])
                    for f_, v in rec.items():
                        if f_ != "lv" and len(v) != n:
                            raise ValueError(f"att dump {tag}|{f_}: {len(v)} rows != part sizes {n}")
                    self.att[ph][k] = rec
        for k in self.akeys:
            lad, t_esc, esc_len, lv_meas, _ = self.att_ladder[k]
            lv = [int(v) for v in self.att["calib"][k]["lv"]]
            if lv != [int(v) for v in lv_meas] or max(lv) > CAP:
                raise ValueError(f"attention key {k}: dump lv {lv} != state lv_meas {lv_meas}")
        self.hold_lin, self.measured_lin, self.prot = {}, None, {}
        if hold_path:
            with np.load(hold_path, allow_pickle=False) as d:
                self.measured_lin = [int(v) for v in d["measured"]]
                self.prot_levels = [int(v) for v in d["prot_levels"]]
                for f in d.files:
                    parts = f.split("|")
                    if parts[0] == "lin" and parts[1] == "hold" and parts[4] in self.HOLD_FIELDS:
                        k = (parts[2], int(parts[3]))
                        self.hold_lin.setdefault(k, {})[parts[4]] = d[f]
                    elif parts[0] == "prot":
                        k = (parts[1], parts[2], int(parts[3]))
                        self.prot.setdefault(k, {})[parts[4]] = d[f]
            if self.measured_lin != [int(v) for v in self.r7["measured_lin"]]:
                raise ValueError("hold dump measured lengths != state r7.measured_lin")
            self.li = {L: i for i, L in enumerate(self.measured_lin)}
            self.top_col = self.li[max(self.ladder)]

    # ---- attention data views -------------------------------------------------------------
    def att_parts(self, phase, k, prefix=None):
        """Per-call records of one key (list of dicts), optionally restricted to sample rank <
        prefix with mac/raw/fis/fis_emp x (att_rows / prefix): the prefix-128 view is calib7's
        128-row capture exactly (same rows, same batches, rep 512 = 128 * 4, exact in fp)."""
        d = self.att[phase][k]
        sizes = self.att_sizes[phase][k]
        out, off = [], 0
        factor = None
        if prefix:
            ratio = self.att_rows // int(prefix)
            if ratio * int(prefix) != self.att_rows or ratio & (ratio - 1):
                raise ValueError("prefix must divide att_rows_per_call by a power of two")
            if self.prefix != int(prefix):
                raise ValueError(f"capture was measured with --att-prefix-split {self.prefix}, "
                                 f"not {prefix}: the prefix view would not be batch-exact")
            factor = float(ratio)
        for n in sizes:
            sl = slice(off, off + (int(prefix) if prefix else n))
            if prefix:
                if n < int(prefix) or not np.array_equal(d["rank"][sl], np.arange(int(prefix))):
                    raise ValueError(f"part of {k} ({phase}) lacks ranks 0..{prefix - 1}")
            rec = {f_: v[sl] for f_, v in d.items() if f_ != "lv"}
            if factor is not None:
                for f_ in ("mac", "raw", "fis", "fis_emp"):
                    if f_ in rec:
                        rec[f_] = rec[f_] * factor
            rec["lv"] = d["lv"]
            out.append(rec)
            off += n
        if off != len(d["mn"]):
            raise ValueError(f"part sizes of {k} ({phase}) do not cover the dump")
        return out

    def att_cat(self, phase, k, prefix=None):
        parts = self.att_parts(phase, k, prefix)
        cat = {f_: np.concatenate([p[f_] for p in parts]) for f_ in parts[0] if f_ != "lv"}
        cat["lv"] = parts[0]["lv"]
        return cat

    def abins_for(self, ladders=None, prefix=None):
        """calib9's attention bins/fixed/par_att over the calibration records, key by key in
        akeys order (same accumulation order as calib9), with per-key ladders (default: the
        parent's). Returns (abins, fixed, par_att)."""
        ladders = ladders or {}
        abins, fixed, par_att = {}, 0.0, 0.0
        for k in self.akeys:
            lad, t_esc, esc_len, _, _ = self.att_ladder[k]
            c = self.att_cat("calib", k, prefix)
            b, esc = att_bins(c["mn"], c["fis"], c["mac"], c["lv"], ladders.get(k, lad), t_esc,
                              esc_len, self.n_bins)
            abins[k] = b
            if esc.any():
                fixed += float((c["mac"][esc] * esc_len).sum())
            par_att += float((c["mac"] * c["parL"]).sum())
        return abins, fixed, par_att


# ---------------------------------------------------------------------------------------------
# tables: attention spec, scoring
# ---------------------------------------------------------------------------------------------
def table_att_spec(table, k):
    """(ascending ladder, descending thresholds) for attention key k of a table JSON, exactly
    as the runtime resolves it (bucket ladder, else the global ladder)."""
    b = table["buckets"][f"{k[0]}:t0:l{k[1]}"]
    raw = b.get("stoc_len_levels", table["stoc_len_levels"])
    A = sorted(int(v) for v in raw)
    th = [float(v) for v in b["thresholds"]]
    if len(th) != len(A) - 1:
        raise ValueError(f"{k}: {len(th)} thresholds for ladder {A}")
    return A, th


def score_tables(cap: Capture, tables: dict, *, prefix=None) -> dict:
    """Held-out Fisher score of whole tables on the capture's hold windows.

    tables: {name: table JSON dict, or None for the parent's own per-row lengths}. Linears are
    scored per (row, chunk) pair through each table's per_row_chunk levels/thresholds (calib9
    deploy_rungs), attention per row through the table's bucket ladder + thresholds + the
    unchanged escape gate (calib9 deploy_att). Each table's sums follow calib9's HJ order
    (linear keys, then attention per call record), so F_seq reproduces calib9's joint held-out
    totals bit for bit. Returns {name: {F_lin, F_att, F_seq, C_lin, C_att, att_mac_hist}}."""
    import torch
    acc = {n: {"F_seq": 0.0, "C_seq": 0.0, "F_lin": 0.0, "C_lin": 0.0, "F_att": 0.0,
               "C_att": 0.0, "att_mac_hist": {}} for n in tables}
    specs = {n: (None if t is None else {k: table_att_spec(t, k) for k in cap.akeys})
             for n, t in tables.items()}
    for k in cap.keys:
        hp = cap.hold_lin.get(k)
        if not hp:
            continue
        mac, hm, parL = hp["mac"], hp["mn"], hp["parL"]
        e = hp["fis"]
        cur = e - e[:, [cap.top_col]]              # calib9 cur_of(hold, 'fis'), verbatim
        ar = np.arange(mac.size)
        for name, tbl in tables.items():
            entry = None if tbl is None else (tbl.get("per_row_chunk") or {}).get(
                "buckets", {}).get(f"{k[0]}:t0:l{k[1]}")
            if entry is None:
                Lr = parL
            else:
                Lr = np.asarray(entry["levels"], dtype=np.float64)[
                    deploy_rungs(torch.from_numpy(hm), entry["thresholds"]).numpy()]
            cols = np.asarray([cap.li[int(v)] for v in Lr])
            fk = float(cur[ar, cols].sum())
            ck = float((mac * Lr).sum())
            a = acc[name]
            a["F_seq"] += fk
            a["C_seq"] += ck
            a["F_lin"] += fk
            a["C_lin"] += ck
        del cur
    for k in cap.akeys:
        lad, t_esc, esc_len, _, _ = cap.att_ladder[k]
        parts = cap.att_parts("hold", k, prefix)
        for name in tables:
            a = acc[name]
            hist = a["att_mac_hist"].setdefault(f"{k[0]}:l{k[1]}", {})
            spec = specs[name]
            for p_ in parts:
                col = {int(L): i for i, L in enumerate(p_["lv"])}
                if spec is None:
                    Lr = p_["parL"]
                else:
                    A, th = spec[k]
                    Lr = deploy_att(p_["mn"], th, A, t_esc, esc_len)
                cols = np.asarray([col[int(v)] for v in Lr])
                fa = float(p_["fis"][np.arange(Lr.size), cols].sum())
                ca = float((p_["mac"] * Lr).sum())
                a["F_seq"] += fa
                a["C_seq"] += ca
                a["F_att"] += fa
                a["C_att"] += ca
                u, inv = np.unique(np.asarray(Lr, dtype=np.float64), return_inverse=True)
                w = np.bincount(inv.reshape(-1), weights=p_["mac"])
                for L_, m_ in zip(u.tolist(), w.tolist()):
                    hist[int(L_)] = hist.get(int(L_), 0.0) + float(m_)
    return acc


def preds(scores: dict, ref: str, n_tok: int) -> dict:
    """Fisher-predicted dNLL per token of every scored table vs `ref`, split by family, and
    the held-out sample cost ratio."""
    r = scores[ref]
    out = {}
    for name, s in scores.items():
        out[name] = {
            "pred_dnll": 0.5 * (s["F_seq"] - r["F_seq"]) / n_tok,
            "pred_dnll_lin": 0.5 * (s["F_lin"] - r["F_lin"]) / n_tok,
            "pred_dnll_att": 0.5 * (s["F_att"] - r["F_att"]) / n_tok,
            "cost_over_ref": (s["C_lin"] + s["C_att"]) / (r["C_lin"] + r["C_att"]),
            "lin_cost_over_ref": s["C_lin"] / r["C_lin"] if r["C_lin"] else None,
            "att_cost_over_ref": s["C_att"] / r["C_att"] if r["C_att"] else None}
    return out


def pred_true(p_lin: float, p_att: float, kappa_lin: float, kappa_att: float) -> float:
    return float(kappa_lin) * float(p_lin) + float(kappa_att) * float(p_att)


# ---------------------------------------------------------------------------------------------
# solve + emit
# ---------------------------------------------------------------------------------------------
def arm_ladders(cap: Capture, arm: str, policy: str) -> dict:
    kind, _ = ARM_DEFS[arm]
    pol = "inherit" if kind == "inherit" else policy
    return {k: ladder_for(pol, cap.att_ladder[k][0], cap.att_ladder[k][3]) for k in cap.akeys}


def solve_arm(cap: Capture, ladders: dict, r: float, s: float, *, prefix=None) -> dict:
    """calib9's joint solve at target0 * s with ladders A and attention error x r."""
    abins, fixed, par_att = cap.abins_for(ladders, prefix)
    target0 = cap.par_lin + par_att
    tgt = scaled_target(target0, s)
    rj = solve_joint(cap.lbins, kappa_bins(abins, r), cap.keys, cap.akeys, cap.Lv, fixed, tgt)
    lin_th = {k: th_from_bins(rj[0][k], cap.lbins[k][2], len(cap.Lv)) for k in cap.keys}
    att_th = {k: att_th_from(rj[1][k], abins[k][2], len(abins[k][3])) for k in cap.akeys}
    cost = joint_cost(rj, cap.lbins, abins, cap.keys, cap.akeys, cap.Lv, fixed)
    lin_C = sum(float(cap.lbins[k][1].sum()) for k in cap.keys)
    lin_L = sum(float((cap.lbins[k][1] * cap.Lv[rj[0][k]]).sum()) for k in cap.keys) / lin_C
    att_bucket = {}
    for k in cap.akeys:
        E, C, bmin, A = abins[k]
        idx = rj[1][k]
        L = A[idx]
        lad = set(int(v) for v in cap.att_ladder[k][0])
        new = np.asarray([int(v) not in lad for v in L])
        att_bucket[f"{k[0]}:l{k[1]}"] = {
            "ladder": [int(v) for v in A], "L": float((C * L).sum() / max(C.sum(), 1e-30)),
            "mac_share_new_rungs": float(C[new].sum() / max(C.sum(), 1e-30)),
            "mac_share_128": float(C[L == CAP].sum() / max(C.sum(), 1e-30))}
    att_mac = sum(float(cap.att_cat("calib", k, prefix)["mac"].sum()) for k in cap.akeys)
    att_L = (sum(float((abins[k][1] * abins[k][3][rj[1][k]]).sum()) for k in cap.akeys)
             + fixed) / att_mac
    return {"rj": rj, "lin_th": lin_th, "att_th": att_th, "abins": abins, "fixed": fixed,
            "par_att": par_att, "target0": target0, "target": tgt, "calib_cost": cost,
            "calib_cost_over_target": cost / tgt, "lin_L_parent": cap.par_lin / lin_C,
            "lin_L": lin_L, "att_L_parent": par_att / att_mac, "att_L": att_L,
            "att_bucket": att_bucket}


def joint_line(sol: dict, tag: str = "[c7]") -> str:
    """calib9's printed JOINT lambda line for a solve (for identity against calib7 logs)."""
    return (f"{tag} JOINT lambda: linear L {sol['lin_L_parent']:.2f} -> {sol['lin_L']:.2f}; "
            f"attention L {sol['att_L_parent']:.2f} -> {sol['att_L']:.2f} (total cost held at "
            f"the parent's)")


def emit_r7_table(stem, tname, lin_th, att_th, att_A, table, wrapper, ladder, calib_record,
                  provenance=None):
    """calib9 emit_table's body, extended to per-bucket attention ladders.

    att_A[(op, bkt)] = ascending ladder. A key whose ladder equals the global one gets no
    bucket ladder (calib9's check: thresholds as many as the parent's); any other gets
    buckets[kk].stoc_len_levels = the descending ladder (strictly descending, >= 2 rungs,
    max <= 128) and |A|-1 thresholds. provenance: extra top-level keys (e.g. {prc10_r7: ..}).
    With no bucket ladders and provenance None/{} the bytes equal calib9 emit_table's with
    extra_record None ({"prc9_r6": x} equals extra_record x). Returns (wrapper, table)."""
    from scmp_kernels.mp.config import AdaptiveMPConfig as _AMP
    stem = Path(stem)
    bk = {f"{k[0]}:t0:l{k[1]}": {"levels": [int(v) for v in ladder], "thresholds": th}
          for k, th in lin_th.items() if k != "__levels__"}
    o = stem.with_name(f"{stem.stem}_{tname}.json")
    t_ = o.with_name(o.stem + "_table.json")
    pt = json.loads(json.dumps(table))
    glob = sorted(int(v) for v in pt["stoc_len_levels"])
    for (op_, b_), th in att_th.items():
        kk = f"{op_}:t0:l{b_}"
        if kk not in pt["buckets"]:
            raise SystemExit(f"[r7] parent table has no bucket {kk}")
        if "stoc_len_levels" in pt["buckets"][kk]:
            raise SystemExit(f"[r7] parent bucket {kk} already carries a ladder")
        A = [int(v) for v in att_A[(op_, b_)]]
        if A != sorted(A) or len(set(A)) != len(A) or len(A) < 2 or A[0] < 1 or A[-1] > CAP:
            raise SystemExit(f"[r7] invalid ladder for {kk}: {A}")
        if A == glob:
            if len(pt["buckets"][kk]["thresholds"]) != len(th):
                raise SystemExit(f"[r7] threshold length mismatch for {kk}")
        else:
            if len(th) != len(A) - 1:
                raise SystemExit(f"[r7] {len(th)} thresholds for the {len(A)}-rung ladder of {kk}")
            pt["buckets"][kk]["stoc_len_levels"] = sorted(A, reverse=True)
        pt["buckets"][kk]["thresholds"] = th
    pt["per_row_chunk"] = {"buckets": bk}
    pt.setdefault("sc_prec", SC_PREC)
    pt.setdefault("halve_bipolar_stoc_len", HALVE)
    pt["prc6_calib"] = calib_record
    for key, val in (provenance or {}).items():
        pt[key] = val
    t_.write_text(json.dumps(pt, indent=1))
    pw = dict(wrapper)
    pw["threshold_table_path"] = str(t_.resolve())
    o.write_text(json.dumps(pw, indent=1))
    chk = _AMP(sorted(glob, reverse=True))
    chk.load_threshold_table(str(t_))
    assert len(chk.per_row_chunk) == len(bk), "round-trip lost buckets"
    return o, t_


# ---------------------------------------------------------------------------------------------
# validation: lineage + runtime resolver sweep
# ---------------------------------------------------------------------------------------------
def _is_desc_ladder(lv) -> bool:
    return (isinstance(lv, list) and len(lv) >= 2 and all(isinstance(v, int) for v in lv)
            and all(b < a for a, b in zip(lv, lv[1:])) and lv[-1] >= 1 and lv[0] <= CAP)


def validate_lineage(parent: dict, cand: dict, *, allow_bucket_ladders=True, prc_keys=None,
                     prc_ladder=None) -> list:
    """Problems (empty = OK) with `cand` as a pure-allocation child of `parent`.

    Allowed to differ: per_row_chunk (new; thresholds only vs its levels), attention bucket
    thresholds and (if allowed) attention bucket stoc_len_levels, provenance keys. Everything
    else must be canonical-equal: global ladder, layer/timestep buckets, linear-op buckets,
    attention metric_mean/metric_std (the escape mu/tau), protected channels + psl,
    dispatch metrics, operator defaults."""
    bad = []
    canon = R6A.canonical
    try:
        R6A.table_cap(cand)                # round-6 validator: sc_prec 8 + halving => cap 128
        R6A.table_cap(parent)
    except ValueError as e:
        bad.append(str(e))
    for key in sorted(set(parent) | set(cand)):
        if key in PROVENANCE_KEYS or key in ("per_row_chunk", "buckets"):
            continue
        if canon(parent.get(key)) != canon(cand.get(key)):
            bad.append(f"top-level field {key!r} differs from the parent")
    glob = [int(v) for v in parent["stoc_len_levels"]]
    if not _is_desc_ladder(glob):
        bad.append(f"parent global ladder invalid {glob}")
    pb, cb = parent.get("buckets", {}), cand.get("buckets", {})
    if set(pb) != set(cb):
        bad.append("bucket key set differs from the parent")
    for kk in sorted(set(pb) & set(cb)):
        p, c = pb[kk], cb[kk]
        op = kk.split(":")[0]
        if op not in ATT_OPS:
            if canon(p) != canon(c):
                bad.append(f"{kk}: non-attention bucket changed")
            continue
        strip = ("thresholds", "stoc_len_levels")
        if canon({a: b for a, b in p.items() if a not in strip}) != \
                canon({a: b for a, b in c.items() if a not in strip}):
            bad.append(f"{kk}: non-allocation bucket fields (e.g. escape mu/tau) differ")
        if "stoc_len_levels" in p:
            bad.append(f"{kk}: parent bucket already has a ladder")
        if "stoc_len_levels" in c:
            lv = c["stoc_len_levels"]
            if not allow_bucket_ladders:
                bad.append(f"{kk}: bucket ladder not allowed here")
            if not _is_desc_ladder(lv):
                bad.append(f"{kk}: bucket ladder {lv} not strictly descending ints in [1, {CAP}]")
            lad = lv if isinstance(lv, list) else glob
        else:
            lad = glob
        th = c.get("thresholds")
        if not isinstance(th, list) or len(th) != len(lad) - 1:
            bad.append(f"{kk}: {len(th) if isinstance(th, list) else th} thresholds for {len(lad)} rungs")
        elif any(not (0.0 <= float(v) <= 1.0) for v in th) or \
                any(float(b) > float(a) for a, b in zip(th, th[1:])):
            bad.append(f"{kk}: thresholds not non-increasing in [0, 1]")
    prc = cand.get("per_row_chunk")
    if not isinstance(prc, dict) or set(prc) != {"buckets"} or not prc["buckets"]:
        bad.append("per_row_chunk must be exactly {'buckets': {...}} (no layer override)")
    else:
        if prc_keys is not None and set(prc["buckets"]) != set(prc_keys):
            bad.append("per_row_chunk bucket keys differ from the solve's")
        for kk, e in prc["buckets"].items():
            if kk.split(":")[0] not in LINEAR_OPS:
                bad.append(f"per_row_chunk key {kk} is not a linear op")
            lv = e.get("levels")
            if not (isinstance(lv, list) and lv == sorted(lv) and len(lv) >= 2 and
                    all(isinstance(v, int) and 1 <= v <= CAP for v in lv)):
                bad.append(f"per_row_chunk {kk}: levels {lv} invalid")
                continue
            if prc_ladder is not None and lv != [int(v) for v in prc_ladder]:
                bad.append(f"per_row_chunk {kk}: levels differ from the dense ladder")
            th = e.get("thresholds")
            if not isinstance(th, list) or len(th) != len(lv) - 1 or \
                    any(not (0.0 <= float(v) <= 1.0) for v in th) or \
                    any(float(b) < float(a) for a, b in zip(th, th[1:])):
                bad.append(f"per_row_chunk {kk}: thresholds invalid")
    return bad


def resolver_sweep(wrapper_path, table_path, parent_dir, total_blocks, *, synthetic_rows=4096,
                   seed=20260928) -> dict:
    """Load the table exactly as the runtime does and check, for every (op, block):
    get_levels == the JSON bucket ladder (or global), classify_level_values == that ladder
    (+ the escape entry only when 128 is not a rung), get_thresholds == JSON, the escape
    threshold == the parent's; a synthetic adaptive_classify_rows dispatch (with planted rows
    at every threshold and at mu+2tau) equals the float32 mirror of calib9's deploy rule;
    get_per_row_chunk == JSON for every linear (op, block); protected slice, global ladder and
    dispatch metrics == the parent's. Raises ValueError on the first violation."""
    import torch
    from scmp_kernels.mp.config import _bucket_index, adaptive_classify_rows
    parent_dir = Path(parent_dir)
    wrapper = json.loads(Path(wrapper_path).read_text())
    table = json.loads(Path(table_path).read_text())
    pw = json.loads((parent_dir / "wrapper.json").read_text())
    strip = lambda w: {a: b for a, b in w.items() if a != "threshold_table_path"}  # noqa: E731
    if strip(wrapper) != strip(pw):
        raise ValueError("wrapper differs from the parent wrapper beyond threshold_table_path")
    if Path(wrapper["threshold_table_path"]).resolve() != Path(table_path).resolve():
        raise ValueError("wrapper does not point at the table")
    cfg = R6A._runtime_cfg(wrapper, table_path)
    par = R6A._runtime_cfg(pw, parent_dir / "table.json")
    glob = [int(v) for v in table["stoc_len_levels"]]
    lb = int(table.get("layer_buckets", 1))
    gen = torch.Generator().manual_seed(int(seed))
    n_att, n_lin, planted, calib_vs_runtime_diff = 0, 0, 0, 0
    max_len = 0
    for op in ATT_OPS:
        for block in range(int(total_blocks)):
            q = R6A.bucket_index(block, total_blocks, lb)
            if q != _bucket_index(block, total_blocks, lb):
                raise ValueError("bucket_index mirror disagrees with the runtime")
            payload = table["buckets"][f"{op}:t0:l{q}"]
            want = [int(v) for v in payload.get("stoc_len_levels", glob)]
            if not _is_desc_ladder(want):
                raise ValueError(f"{op}:l{q}: ladder {want} invalid (cap {CAP})")
            got = list(cfg.get_levels(operator=op, block_idx=block, total_blocks=total_blocks))
            if got != want:
                raise ValueError(f"get_levels({op}, {block}) = {got}, JSON {want}")
            vals = list(cfg.classify_level_values(operator=op, block_idx=block,
                                                  total_blocks=total_blocks))
            esc_on = cfg.escape_gate_k is not None
            want_vals = want if (not esc_on or int(cfg.escape_stoc_len) in want) \
                else want + [int(cfg.escape_stoc_len)]
            if vals != want_vals:
                raise ValueError(f"classify_level_values({op}, {block}) = {vals}, expected {want_vals}")
            max_len = max(max_len, max(vals))
            th = [float(v) for v in cfg.get_thresholds(0, 1, operator=op, block_idx=block,
                                                       total_blocks=total_blocks)]
            if th != [float(v) for v in payload["thresholds"]]:
                raise ValueError(f"get_thresholds({op}, {block}) != JSON")
            t_esc = cfg.get_escape_threshold(0, 1, operator=op, block_idx=block,
                                             total_blocks=total_blocks)
            t_par = par.get_escape_threshold(0, 1, operator=op, block_idx=block,
                                             total_blocks=total_blocks)
            if t_esc != t_par:
                raise ValueError(f"escape threshold changed at ({op}, {block}): {t_esc} vs {t_par}")
            metric = torch.rand(synthetic_rows, generator=gen) ** 3
            metric[0], metric[1] = 0.0, 1.0            # normalization becomes the identity
            plant = []
            for v in th + ([t_esc] if (t_esc is not None and t_esc < 1.0) else []):
                f = np.float32(v)
                plant += [f, np.nextafter(f, np.float32(0)), np.nextafter(f, np.float32(1))]
            plant = [min(max(float(p_), 0.0), 1.0) for p_ in plant]
            if plant:
                metric[2:2 + len(plant)] = torch.tensor(plant, dtype=metric.dtype)
                planted += len(plant)
            asg = adaptive_classify_rows(metric, cfg, operator=op, block_idx=block,
                                         total_blocks=total_blocks)
            L_rt = np.asarray(vals, dtype=np.float64)[asg.row_levels.numpy()]
            mn = ((metric - metric.min()) / (metric.max() - metric.min())).numpy()
            esc_len = int(cfg.escape_stoc_len) if esc_on else 0
            L_mirror = deploy_att_runtime(mn, th, sorted(want), t_esc, esc_len)
            if not np.array_equal(L_rt, L_mirror):
                raise ValueError(f"synthetic dispatch ({op}, {block}) disagrees with the deploy rule")
            calib_vs_runtime_diff += int((deploy_att(mn, th, sorted(want), t_esc, esc_len)
                                          != L_mirror).sum())
            if float(L_rt.max()) > CAP:
                raise ValueError(f"({op}, {block}) dispatched above {CAP}")
            n_att += 1
    prc = table["per_row_chunk"]["buckets"]
    for op in LINEAR_OPS:
        for block in range(int(total_blocks)):
            key = f"{op}:t0:l{R6A.bucket_index(block, total_blocks, lb)}"
            got = cfg.get_per_row_chunk(op, block, total_blocks)
            e = prc.get(key)
            if e is None:
                if got is not None:
                    raise ValueError(f"unexpected per_row_chunk entry at {key}")
                continue
            want = ([int(v) for v in e["levels"]], [float(v) for v in e["thresholds"]])
            if got is None or ([int(v) for v in got[0]], [float(v) for v in got[1]]) != want:
                raise ValueError(f"get_per_row_chunk({op}, {block}) != JSON")
            if max(want[0]) > CAP:
                raise ValueError(f"per_row_chunk {key} above {CAP}")
            n_lin += 1
    if cfg.protected_channel_stoc_len != par.protected_channel_stoc_len or \
            cfg.protected_channel_indices != par.protected_channel_indices:
        raise ValueError("protected slice (psl or indices) differs from the parent")
    if [int(v) for v in cfg.stoc_len_levels] != [int(v) for v in par.stoc_len_levels] or \
            cfg.dispatch_metrics != par.dispatch_metrics:
        raise ValueError("global ladder or dispatch metrics differ from the parent")
    if int(cfg.protected_channel_stoc_len or 0) > CAP:
        raise ValueError("protected slice above the cap")
    return {"attention_op_blocks": n_att, "linear_op_blocks": n_lin, "planted_rows": planted,
            "max_attention_length": int(max_len),
            "calib_rule_vs_runtime_mirror_rows": calib_vs_runtime_diff,
            "calib_rule_vs_runtime_mirror_note": (
                "rows where calib9's deploy_att (Python-float thresholds) and the float32 runtime "
                "mirror disagree; 0 is expected under NumPy >= 2, which compares a float32 array "
                "with a Python float in float32 (weak scalar), exactly as the runtime does"),
            "escape_gate_k": cfg.escape_gate_k, "escape_stoc_len": int(cfg.escape_stoc_len)}


def escape_disclosure(table, cap: Capture) -> dict:
    """Per attention bucket: is 128 a rung, and is the gate redundant there (the calibrated
    128 threshold th[0] <= mu+2tau, so every escaped row is already on 128)?"""
    out = {}
    for k in cap.akeys:
        A, th = table_att_spec(table, k)
        t_esc = cap.att_ladder[k][1]
        fires = t_esc is not None and t_esc < 1.0
        has128 = CAP in A
        out[f"{k[0]}:l{k[1]}"] = {
            "ladder": A, "has_128_rung": has128, "t_esc": t_esc, "gate_can_fire": fires,
            "th_top": th[0] if th else None,
            "escape_redundant": bool(has128 and fires and th and A[-1] == CAP and th[0] <= t_esc)}
    return out


# ---------------------------------------------------------------------------------------------
# identity (incumbent recipe) + pipeline check
# ---------------------------------------------------------------------------------------------
def _thresholds_equal(a, b) -> bool:
    return [float(x) for x in json.loads(json.dumps(a))] == [float(x) for x in b]


def compare_table_thresholds(lin_th, att_th, ref: dict, att_ladders=None) -> list:
    """Exact threshold identity (after JSON round trip) against a table payload. att_ladders
    None: the reference must carry NO attention bucket ladder (calib7/calib9 tables); else
    {key: ascending ladder}: a bucket ladder must be present exactly where the ladder differs
    from the reference's global one, and equal it (descending)."""
    bad = []
    glob = sorted(int(v) for v in ref.get("stoc_len_levels", []))
    prc = (ref.get("per_row_chunk") or {}).get("buckets") or {}
    want = {f"{k[0]}:t0:l{k[1]}": th for k, th in lin_th.items()}
    if set(prc) != set(want):
        bad.append(f"per_row_chunk keys differ: {sorted(set(prc) ^ set(want))[:6]}")
    for kk, th in want.items():
        got = (prc.get(kk) or {}).get("thresholds")
        if got is None or not _thresholds_equal(th, got):
            bad.append(f"per_row_chunk {kk} thresholds differ")
    for k, th in (att_th or {}).items():
        kk = f"{k[0]}:t0:l{k[1]}"
        b = (ref.get("buckets") or {}).get(kk) or {}
        if att_ladders is None:
            if "stoc_len_levels" in b:
                bad.append(f"{kk}: reference carries a bucket ladder")
        else:
            A = sorted(int(v) for v in att_ladders[k])
            want = None if A == glob else sorted(A, reverse=True)
            if b.get("stoc_len_levels") != want:
                bad.append(f"{kk}: bucket ladder {b.get('stoc_len_levels')} != {want}")
        if b.get("thresholds") is None or not _thresholds_equal(th, b["thresholds"]):
            bad.append(f"attention {kk} thresholds differ")
    return bad


def strip_provenance(t: dict) -> dict:
    return {k: v for k, v in t.items() if k not in PROVENANCE_KEYS}


def threshold_agreement(lin_th, att_th, ref: dict) -> dict:
    """Fraction of identical threshold ENTRIES and the max |delta| vs a reference table (the
    'near' identity diagnostics; exact identity is compare_table_thresholds == [])."""
    n = same = 0
    mx = 0.0
    prc = (ref.get("per_row_chunk") or {}).get("buckets") or {}
    pairs = [(th, (prc.get(f"{k[0]}:t0:l{k[1]}") or {}).get("thresholds")) for k, th in lin_th.items()]
    pairs += [(th, ((ref.get("buckets") or {}).get(f"{k[0]}:t0:l{k[1]}") or {}).get("thresholds"))
              for k, th in (att_th or {}).items()]
    for a, b in pairs:
        a = [float(x) for x in json.loads(json.dumps(a))]
        if b is None or len(b) != len(a):
            n += len(a)
            mx = max(mx, 1.0)
            continue
        for x, y in zip(a, b):
            n += 1
            same += int(x == float(y))
            mx = max(mx, abs(x - float(y)))
    return {"entries": n, "identical_fraction": same / max(n, 1), "max_abs_diff": mx}


# Pre-registered 'near' identity (MoE backward last-bit effects): every CPU-deterministic
# check exact, printed calib lines equal, and the held-out Fisher predictions within
# NEAR_PRED_ABS nats (0.002% PPL; the effects round 7 tests are >= 0.5%).
NEAR_PRED_ABS = 2e-5


def own_table_scores(capture: Capture, stem, capture_score_tables=None) -> dict:
    """Check (7), CPU-deterministic: re-score the capture's OWN in-process tables (gfis, gfisla
    = the 512-row J table at scale 1.0) and its --score-tables from the dumps (all rows, no
    prefix) and compare with the capture's own diag bit for bit: joint.{gfis,gfisla}.pred_dnll
    and tables.<name>.pred_dnll_fis. Both sides come from the same in-process arrays, so any
    difference means the hold/attention dumps are not what the capture scored."""
    stem = Path(stem)
    capd = json.loads(Path(capture.r7["files"]["diag"]).read_text())
    tabs = {"parent": None}
    for nm in ("gfis", "gfisla"):
        p = stem.with_name(f"{stem.stem}_{nm}_table.json")
        if p.is_file():
            tabs[nm] = json.loads(p.read_text())
    for nm, p in (capture_score_tables or {}).items():
        tabs[nm] = json.loads(Path(p).read_text())
    sc = score_tables(capture, tabs)
    pr = preds(sc, "parent", capture.n_tok)
    got, want = {}, {}
    for nm in ("gfis", "gfisla"):
        if nm in tabs:
            got[f"joint.{nm}"] = pr[nm]["pred_dnll"]
            want[f"joint.{nm}"] = ((capd.get("joint") or {}).get(nm) or {}).get("pred_dnll")
    for nm in tabs:
        if nm == "parent":
            continue
        got[f"tables.{nm}"] = 0.5 * (sc[nm]["F_lin"] - sc["parent"]["F_lin"]) / capture.n_tok
        want[f"tables.{nm}"] = ((capd.get("tables") or {}).get(nm) or {}).get("pred_dnll_fis")
    missing = sorted(n for n in ("gfis", "gfisla") if n not in tabs)
    equal = {n: (want[n] is not None and got[n] == float(want[n])) for n in got}
    return {"ok": bool(got) and not missing and all(equal.values()), "equal": equal,
            "missing_own_tables": missing, "capture": got, "capture_diag": want}


def run_identity(*, capture: Capture, stem, parent_dir, ref_gfis, ref_gfisla, ref_diag,
                 inc_table, prefix=128, expect_joint_line=None, expect_global_line=None,
                 capture_log=None, inc_relation="identical", allow_near=False,
                 capture_score_tables=None) -> dict:
    """The incumbent-identity reproduction, run BEFORE any extended solve.

    exact  (1) the capture's in-process gfis table == c7_gfis in every non-provenance field and
           the state re-solve reproduces it; (2) attention bins rebuilt from the dump == the
           in-process bins (CPU determinism of the dump); (3) the 128-row prefix view (ranks <
           128, rep x4) re-solved jointly at scale 1.0 == c7_gfisla (28 linear + 8 attention
           keys); (4) held-out Fisher predictions on the prefix view == the c7 diag (gfis,
           gfisla joint; c17 and gfis linear-only); (5) parent attention replay 1.0; (6) INC
           is a lineage-clean child of the parent and equals c7_gfisla (or differs only in
           per_row_chunk thresholds: 30B t40 candidate07); (7) the capture's own tables
           re-scored from the dumps (all 512 rows) == the capture's own diag, bit for bit.
    near   (only if allow_near): the CPU-deterministic checks (2)(5)(6)(7) exact, the printed
           global-fis / JOINT lines equal the c7 log's, every prediction within NEAR_PRED_ABS.
           Round 7 registers allow_near = False: a 'near' capture is stop-and-report.
    identity_level = exact | near | fail; identity_ok = ok = level == exact (or near if
    allowed)."""
    parent_dir = Path(parent_dir)
    exact, cpu, near, notes = {}, {}, {}, {}
    parent_table = json.loads((parent_dir / "table.json").read_text())
    ref_g = json.loads(Path(ref_gfis).read_text())
    ref_j = json.loads(Path(ref_gfisla).read_text())
    diag = json.loads(Path(ref_diag).read_text())
    meta = capture.meta
    cpu["parent_sha_matches_state"] = (
        sha256(parent_dir / "table.json") == meta["parent_table_sha256"] and
        sha256(parent_dir / "wrapper.json") == meta["parent_wrapper_sha256"])
    # (1) linear identity
    stem = Path(stem)
    g = json.loads(stem.with_name(f"{stem.stem}_gfis_table.json").read_text())
    exact["gfis_table_equals_c7_gfis"] = (
        R6A.canonical(strip_provenance(g)) == R6A.canonical(strip_provenance(ref_g)))
    res = resolve_tables(meta, capture.keys, capture.akeys, capture.lbins, capture.abins_state,
                         [1.0], ["gfis"])
    cpu["state_resolve_equals_inprocess_gfis"] = not compare_table_thresholds(
        res["gfis"][0], None, g)
    exact["gfis_state_resolve_equals_c7_gfis"] = not compare_table_thresholds(
        res["gfis"][0], None, ref_g)
    notes["gfis_vs_c7"] = threshold_agreement(res["gfis"][0], None, ref_g)
    # (2) dump == in-process attention bins (all rows, parent ladder)
    ab, fx, pa = capture.abins_for(None, None)
    same = all(all(np.array_equal(x, y) and x.dtype == y.dtype
                   for x, y in zip(ab[k], capture.abins_state[k])) for k in capture.akeys)
    cpu["att_dump_bins_equal_state"] = bool(same and fx == float(meta["fixed"]) and
                                            pa == float(meta["par_att"]))
    # (3) 128-prefix joint re-solve == c7_gfisla
    sol = solve_arm(capture, {}, 1.0, 1.0, prefix=prefix)
    mism = compare_table_thresholds(sol["lin_th"], sol["att_th"], ref_j)
    exact["prefix_joint_equals_c7_gfisla"] = not mism
    notes["prefix_joint_mismatches"] = mism[:20]
    notes["prefix_joint_vs_c7"] = threshold_agreement(sol["lin_th"], sol["att_th"], ref_j)
    notes["prefix_joint_line"] = joint_line(sol)
    if expect_joint_line:
        near["prefix_joint_line_equals_c7_log"] = joint_line(sol) == expect_joint_line
    if expect_global_line:
        found = None
        if capture_log and Path(capture_log).is_file():
            for line in Path(capture_log).read_text(errors="replace").splitlines():
                if line.startswith("[c6] global-fis: "):
                    found = line.strip()
                    break
        notes["capture_global_line"] = found
        near["global_fis_line_equals_c7_log"] = found == expect_global_line.strip()
    # (4) held-out Fisher predictions (prefix view) == the c7 diag
    tabs = {"parent": None, "gfis": ref_g, "gfisla": ref_j}
    c17_ref = (diag.get("tables") or {}).get("c17")
    sc = score_tables(capture, tabs, prefix=prefix)
    pr = preds(sc, "parent", capture.n_tok)
    want = diag.get("joint", {})
    got = {"joint_gfis": pr["gfis"]["pred_dnll"], "joint_gfisla": pr["gfisla"]["pred_dnll"],
           "lin_gfis": 0.5 * (sc["gfis"]["F_lin"] - sc["parent"]["F_lin"]) / capture.n_tok}
    ref_vals = {"joint_gfis": (want.get("gfis") or {}).get("pred_dnll"),
                "joint_gfisla": (want.get("gfisla") or {}).get("pred_dnll"),
                "lin_gfis": (diag.get("tables", {}).get("gfis") or {}).get("pred_dnll_fis")}
    notes["pred_capture_prefix"] = got
    notes["pred_c7_diag"] = ref_vals
    exact["prefix_preds_equal_c7_diag"] = all(
        ref_vals[n] is not None and got[n] == float(ref_vals[n]) for n in got)
    diffs = {n: (abs(got[n] - float(ref_vals[n])) if ref_vals[n] is not None else None)
             for n in got}
    notes["pred_abs_diff"] = diffs
    near["preds_within_near_tolerance"] = all(d is not None and d <= NEAR_PRED_ABS
                                              for d in diffs.values())
    # (5) parent replay on the capture (calib9's own agreement number)
    capd = json.loads(Path(capture.r7["files"]["diag"]).read_text())
    cpu["attention_replay_agreement_1"] = capd.get("att_replay_agreement") == 1.0
    if c17_ref is not None and "c17" in (capd.get("tables") or {}):
        notes["c17_linear_pred"] = {"capture": capd["tables"]["c17"].get("pred_dnll_fis"),
                                    "c7_diag": c17_ref.get("pred_dnll_fis")}
        exact["c17_linear_pred_equals_c7_diag"] = (
            capd["tables"]["c17"].get("pred_dnll_fis") == c17_ref.get("pred_dnll_fis"))
        d17 = abs(float(capd["tables"]["c17"]["pred_dnll_fis"]) - float(c17_ref["pred_dnll_fis"]))
        notes["pred_abs_diff"]["lin_c17"] = d17
        near["c17_within_near_tolerance"] = d17 <= NEAR_PRED_ABS
    # (6) incumbent lineage
    inc = json.loads(Path(inc_table).read_text())
    cpu["inc_lineage_vs_parent"] = not validate_lineage(parent_table, inc, allow_bucket_ladders=False)
    if inc_relation == "identical":
        cpu["inc_equals_c7_gfisla"] = R6A.canonical(inc) == R6A.canonical(ref_j)
    elif inc_relation == "prc_thresholds_only":
        a = {k: v for k, v in inc.items() if k != "per_row_chunk"}
        b = {k: v for k, v in ref_j.items() if k != "per_row_chunk"}
        ib, rb = inc["per_row_chunk"]["buckets"], ref_j["per_row_chunk"]["buckets"]
        cpu["inc_differs_from_c7_gfisla_only_in_prc_thresholds"] = (
            R6A.canonical(a) == R6A.canonical(b) and set(ib) == set(rb) and
            all(ib[kk]["levels"] == rb[kk]["levels"] for kk in rb))
    else:
        raise ValueError(f"unknown inc_relation {inc_relation!r}")
    # (7) the capture's own tables re-scored from its dumps == its own diag (all rows)
    own = own_table_scores(capture, stem, capture_score_tables)
    cpu["own_tables_rescored_equal_capture_diag"] = own["ok"]
    notes["own_table_scores"] = own
    cpu_ok = all(bool(v) for v in cpu.values())
    exact_ok = cpu_ok and all(bool(v) for v in exact.values())
    near_ok = cpu_ok and bool(near) and all(bool(v) for v in near.values())
    level = "exact" if exact_ok else ("near" if near_ok else "fail")
    ok = level == "exact" or (level == "near" and allow_near)
    return {"ok": ok, "identity_ok": ok, "identity_level": level, "allow_near": bool(allow_near),
            "near_pred_abs_tolerance_nats": NEAR_PRED_ABS,
            "checks": {"cpu_exact": cpu, "exact": exact, "near": near}, "notes": notes,
            "files": {k: {"path": v, "sha256": sha256(v)} for k, v in capture.paths.items() if v}}


def pipeline_check(capture: Capture, parent_dir, out_dir, *, policy="up") -> dict:
    """Emit ONE extended-ladder table (arm U, kappa 1, scale 1.0; a pipeline check, never a
    candidate) and validate it: lineage, runtime resolver sweep, escape disclosure."""
    parent_dir = Path(parent_dir)
    table = json.loads((parent_dir / "table.json").read_text())
    wrapper = json.loads((parent_dir / "wrapper.json").read_text())
    lads = arm_ladders(capture, "U", policy)
    sol = solve_arm(capture, lads, 1.0, 1.0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    name = "pipelinecheck_" + arm_table_name("U", 1.0, 1.0)
    o, t_ = emit_r7_table(out_dir / "check.json", name, sol["lin_th"], sol["att_th"], lads, table,
                          wrapper, capture.ladder,
                          dict(capture.meta["calib_record"], table=name),
                          {R7_KEY: {"tool": TOOL, "purpose": "pipeline check only; NOT a candidate"}})
    cand = json.loads(t_.read_text())
    lineage = validate_lineage(table, cand, prc_keys={f"{k[0]}:t0:l{k[1]}" for k in capture.keys},
                               prc_ladder=capture.ladder)
    sweep = resolver_sweep(o, t_, parent_dir, capture.total_blocks)
    return {"ok": not lineage, "lineage_problems": lineage, "resolver_sweep": sweep,
            "table": str(t_), "wrapper": str(o), "ladders": {f"{k[0]}:l{k[1]}": v
                                                             for k, v in lads.items()},
            "escape": escape_disclosure(cand, capture),
            "calib_cost_over_target": sol["calib_cost_over_target"]}


def guarded_pipeline_check(capture: Capture, parent_dir, out_dir, *, policy="up") -> dict:
    """pipeline_check that never raises: an emitter/validator failure is recorded (ok False,
    error) so the identity record is still written and the capture is not discarded."""
    try:
        return pipeline_check(capture, parent_dir, out_dir, policy=policy)
    except Exception as e:  # noqa: BLE001  (recorded, never swallowed silently)
        import traceback
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-4000:]}


# ---------------------------------------------------------------------------------------------
# solve (arms)
# ---------------------------------------------------------------------------------------------
PREREG_KEYS = ("primary", "arms", "ladder_policy", "mde_nats", "s0", "family_kappa")


def check_prereg(prereg: dict) -> dict:
    """Validate one cell's manifest `prereg` block (the registered solve inputs)."""
    miss = [k for k in PREREG_KEYS if k not in prereg]
    if miss:
        raise ValueError(f"prereg lacks {miss}")
    arms = list(prereg["arms"])
    if not arms or any(a not in ARM_DEFS or a == "J" for a in arms) or len(set(arms)) != len(arms):
        raise ValueError(f"prereg arms {arms} invalid (J is a CPU control, never an arm)")
    if prereg["primary"] not in arms:
        raise ValueError(f"prereg primary {prereg['primary']!r} not among its arms {arms}")
    if prereg["ladder_policy"] != "up":
        raise ValueError("the registered ladder policy for candidate arms is 'up'")
    mde = float(prereg["mde_nats"])
    if not (math.isfinite(mde) and mde > 0):
        raise ValueError(f"prereg mde_nats {prereg['mde_nats']!r} invalid")
    parse_scales(str(prereg["s0"]))
    return prereg


def read_identity(identity_path, capture: Capture, cell=None) -> dict:
    """Fail closed: the capture's identity.json must say identity_ok = true for THESE files."""
    p = Path(identity_path)
    if not p.is_file():
        raise SystemExit(f"[r7] no identity record {p}: no arm may be solved from this capture")
    rep = json.loads(p.read_text())
    if rep.get("identity_ok") is not True:
        raise SystemExit(f"[r7] {p}: identity_ok is {rep.get('identity_ok')!r} (level "
                         f"{rep.get('identity_level')!r}); no arm may be solved from this capture")
    if cell is not None and rep.get("cell") not in (None, cell):
        raise SystemExit(f"[r7] {p} is for cell {rep.get('cell')!r}, not {cell!r}")
    for k, v in capture.paths.items():
        rec = (rep.get("files") or {}).get(k)
        if v and (rec is None or sha256(v) != rec.get("sha256")):
            raise SystemExit(f"[r7] {p}: capture file {k} differs from the one identity checked")
    return {"path": str(p), "sha256": sha256(p), "identity_level": rep.get("identity_level"),
            "pipeline_check_ok": (rep.get("pipeline_check") or {}).get("ok")}


def _pred_record(pr: dict, kap: dict) -> dict:
    pt = pred_true(pr["pred_dnll_lin"], pr["pred_dnll_att"], kap["kappa_lin"], kap["kappa_att"])
    return {"pred_dnll_lin": pr["pred_dnll_lin"], "pred_dnll_att": pr["pred_dnll_att"],
            "pred_dnll_fisher": pr["pred_dnll"], "pred_true": pt, "cost_over_inc": pr["cost_over_ref"]}


def allowed_scales(prereg: dict, cell: str, kap: dict, fixed_point_record: dict = None,
                   allow_simulated=False) -> dict:
    """{arm: set of target scales this arm may be solved at}: always s0; plus the arm's OWN s1
    from the eval-side fixed-point record (prc_fixedpoint_r7_20260928, the registered target-scale
    rule), which must be for this cell, this s0 and these kappa pins."""
    s0 = float(prereg["s0"])
    out = {a: {s0} for a in prereg["arms"]}
    fp = fixed_point_record
    if fp is None:
        return out
    if fp.get("cell") != cell or float(fp.get("s0")) != s0:
        raise SystemExit(f"[r7] fixed-point record is for {fp.get('cell')} at s0 {fp.get('s0')}, "
                         f"not {cell} at the registered s0 {s0}")
    if fp.get("simulated") and not allow_simulated:
        raise SystemExit("[r7] fixed-point record was built from simulated step-0 data")
    pins = fp.get("kappa_pins") or {}
    if (float(pins.get("kappa_lin", -1)), float(pins.get("kappa_att", -1))) != \
            (kap["kappa_lin"], kap["kappa_att"]):
        raise SystemExit(f"[r7] fixed-point record kappa pins {pins} != this cell's registered pins "
                         f"({kap['kappa_lin']}, {kap['kappa_att']})")
    for arm, rec in (fp.get("arms") or {}).items():
        if arm in out and rec.get("status") == "ok" and rec.get("s1") is not None:
            parse_scales(str(rec["s1"]))
            out[arm].add(float(rec["s1"]))
    return out


def run_solve(*, capture: Capture, parent_dir, inc_table, out_dir, cell, prereg, identity_path,
              kappa_decision=None, kappa_pins=None, arms=None, scales=None, fixed_point_record=None,
              emit=True, sensitivity_ratios=(), sensitivity_policies=(), r6_arm_a=None,
              allow_simulated=False) -> dict:
    """Solve the cell's pre-registered arms and write solve_summary.json.

    prereg          the manifest cell's `prereg` block (primary, arms, ladder policy 'up', MDE in
                    nats, s0, family_kappa); nothing here can be chosen after predictions exist.
    identity_path   the capture's identity.json; refused unless identity_ok is true.
    kappa_decision  a step-0 decision record (either side's) for family-kappa (30B) cells; the
                    branch is re-derived with the one-sided KAPPA_RULE; `kappa_pins` (optional,
                    e.g. from --kappa-lin/--kappa-att) must equal that branch's pins.
    scales          default [s0]; any other scale must be an arm's own s1 in the registered
                    fixed-point record (each arm is solved only at the scales allowed for it).
    Candidates are only the emitted arm tables; the J controls (inherited ladders, kappa 1, at
    each solve scale and at 1.0) and every sensitivity entry are scored in memory, never
    emitted, never candidates."""
    check_prereg(prereg)
    parent_dir = Path(parent_dir)
    table = json.loads((parent_dir / "table.json").read_text())
    wrapper = json.loads((parent_dir / "wrapper.json").read_text())
    if sha256(parent_dir / "table.json") != capture.meta["parent_table_sha256"]:
        raise SystemExit("[r7] parent table.json differs from the capture's")
    ident = read_identity(identity_path, capture, cell)
    inc = json.loads(Path(inc_table).read_text())
    R6A.check_incumbent_shape(inc)         # round-6 validator: cap 128, ladders, keys, PRC, psl
    inc_problems = validate_lineage(table, inc, allow_bucket_ladders=False)
    if inc_problems:
        raise SystemExit(f"[r7] INC is not a lineage-clean child of the parent: {inc_problems[:3]}")
    kap = kappa_for_cell(prereg, kappa_decision, kappa_pins)
    r_cur = kap["currency_ratio"]
    policy = prereg["ladder_policy"]
    mde = float(prereg["mde_nats"])
    s0 = float(prereg["s0"])
    arms = list(prereg["arms"]) if arms is None else list(arms)
    bad = [a for a in arms if a not in prereg["arms"]]
    if bad or not arms:
        raise SystemExit(f"[r7] arms {arms} not within the registered {prereg['arms']}")
    omitted = {}
    if r_cur == 1.0 and "U" in arms and "UK" in prereg["arms"]:
        arms.remove("U")
        omitted["U"] = "identical to UK at kappa ratio 1 (revert or dense): not solved separately"
    scales = [s0] if not scales else [float(x) for x in scales]
    allowed = allowed_scales(prereg, cell, kap, fixed_point_record, allow_simulated)
    plan = {a: [s for s in scales if s in allowed[a]] for a in arms}
    orphan = [s for s in scales if not any(s in plan[a] for a in arms)]
    if orphan:
        raise SystemExit(f"[r7] target scales {orphan} are neither s0 ({s0}) nor a registered "
                         f"fixed-point s1 of the requested arms")
    for a in arms:
        if not plan[a]:
            omitted[a] = f"no registered scale among {scales} for this arm"
    stage = "initial_s0" if scales == [s0] else "fixed_point_s1"
    if stage == "initial_s0" and prereg["primary"] not in arms:
        raise SystemExit(f"[r7] the initial solve must include the primary {prereg['primary']}")
    out_dir = Path(out_dir)
    if emit:
        out_dir.mkdir(parents=True, exist_ok=False)
    prc_keys = {f"{k[0]}:t0:l{k[1]}" for k in capture.keys}
    summary = {"tool": TOOL, "cell": cell, "stage": stage, "prereg": prereg,
               "kappa_rule_id": KAPPA_RULE["id"], "kappa": kap, "kappa_branch": kap["branch"],
               "kappa_lin": kap["kappa_lin"], "kappa_att": kap["kappa_att"],
               "kappa_currency_ratio": r_cur, "ladder_policy": policy, "mde_nats": mde,
               "primary_kind": prereg["primary"], "primary": None, "scales": scales,
               "fixed_point_record": ({"cell": fixed_point_record.get("cell"),
                                       "s0": fixed_point_record.get("s0"),
                                       "s1": {a: r.get("s1") for a, r in
                                              (fixed_point_record.get("arms") or {}).items()}}
                                      if fixed_point_record else None),
               "omitted_arms": omitted, "identity": ident,
               "capture": {k: {"path": v, "sha256": sha256(v)} for k, v in capture.paths.items() if v},
               "parent": str(parent_dir), "inc_table": str(inc_table),
               "inc_table_sha256": sha256(inc_table), "arms": {},
               "non_candidates": ("J controls, sensitivity_not_preregistered entries, the capture's "
                                  "in-process gfis/gfisla (J512) tables and the pipeline-check "
                                  "table are never candidates")}
    scored = {"parent": None, "INC": inc}
    for arm in arms:
        kind, use_kappa = ARM_DEFS[arm]
        lads = arm_ladders(capture, arm, policy)
        ra = r_cur if use_kappa else 1.0
        for s in plan[arm]:
            name = arm_table_name(arm, ra, s)
            sol = solve_arm(capture, lads, ra, s)
            rec = {"arm": arm, "candidate": True,
                   "role": "primary" if arm == prereg["primary"] else "secondary",
                   "ladders": {f"{k[0]}:l{k[1]}": v for k, v in lads.items()},
                   "ladder_policy": "inherit" if kind == "inherit" else policy,
                   "kappa_ratio_used": ra, "target_scale": float(s),
                   "target0": sol["target0"], "target": sol["target"],
                   "calib_cost": sol["calib_cost"],
                   "calib_cost_over_target": sol["calib_cost_over_target"],
                   "calib_lin_L": [sol["lin_L_parent"], sol["lin_L"]],
                   "calib_att_L": [sol["att_L_parent"], sol["att_L"]],
                   "calib_att_bucket": sol["att_bucket"], "joint_line": joint_line(sol, "[r7]")}
            if arm == "K" and ra == 1.0:
                rec["note"] = "K at kappa ratio 1 equals the J control at this scale (disclosed)"
            prov = {R7_KEY: {"tool": TOOL, "cell": cell, "arm": arm, "table": name,
                             "role": rec["role"], "stage": stage,
                             "ladder_policy": rec["ladder_policy"], "ladders": rec["ladders"],
                             "kappa_rule_id": KAPPA_RULE["id"], "kappa_branch": kap["branch"],
                             "kappa_lin": kap["kappa_lin"], "kappa_att": kap["kappa_att"],
                             "kappa_ratio_used": ra, "target_scale": float(s),
                             "target0": sol["target0"], "target": sol["target"],
                             "calib_cost_over_target": sol["calib_cost_over_target"],
                             "capture_state_sha256": summary["capture"]["state"]["sha256"],
                             "capture_att_dump_sha256": summary["capture"]["att_dump"]["sha256"],
                             "c9_record": capture.meta.get("c9_record")}}
            if emit:
                o, t_ = emit_r7_table(out_dir / f"{cell}.json", name, sol["lin_th"], sol["att_th"],
                                      lads, table, wrapper, capture.ladder,
                                      dict(capture.meta["calib_record"], table=name), prov)
                cand = json.loads(t_.read_text())
                lineage = validate_lineage(table, cand, prc_keys=prc_keys, prc_ladder=capture.ladder)
                if lineage:
                    raise SystemExit(f"[r7] {name}: lineage problems {lineage[:5]}")
                mism = compare_table_thresholds(sol["lin_th"], sol["att_th"], cand, lads)
                if mism:
                    raise SystemExit(f"[r7] {name}: emitted table differs from the solve: {mism[:3]}")
                rec["resolver_sweep"] = resolver_sweep(o, t_, parent_dir, capture.total_blocks)
                rec["wrapper"], rec["table_path"] = str(o), str(t_)
                rec["table_sha256"], rec["wrapper_sha256"] = sha256(t_), sha256(o)
                rec["escape"] = escape_disclosure(cand, capture)
                scored[name] = cand
            else:
                cand = _table_from_solve(table, sol, lads, capture)
                scored[name] = cand
            if arm == prereg["primary"]:
                summary["primary"] = name
            summary["arms"][name] = rec
    # J controls (CPU only, never emitted): inherited ladders, kappa 1, at each scale and 1.0
    jl = arm_ladders(capture, "J", policy)
    controls = {}
    for s in sorted(set(scales) | {1.0}):
        nm = "control_" + arm_table_name("J", 1.0, s)
        scored[nm] = _table_from_solve(table, solve_arm(capture, jl, 1.0, s), jl, capture)
        controls[nm] = {"target_scale": float(s)}
    if r6_arm_a:
        if sha256(r6_arm_a["table"]) != r6_arm_a["sha256"]:
            raise SystemExit("[r7] round-6 arm A table changed since the manifest was built")
        scored["r6_arm_A"] = json.loads(Path(r6_arm_a["table"]).read_text())
    sc = score_tables(capture, scored)
    pr_par = preds(sc, "parent", capture.n_tok)
    pr_inc = preds(sc, "INC", capture.n_tok)
    for nm in controls:
        controls[nm].update(heldout_vs_inc=pr_inc[nm], **_pred_record(pr_inc[nm], kap))
    j_at = {c["target_scale"]: nm for nm, c in controls.items()}
    for name, rec in summary["arms"].items():
        rec["heldout_vs_parent"] = pr_par[name]
        rec["heldout_vs_inc"] = pr_inc[name]
        p = _pred_record(pr_inc[name], kap)
        rec["pred_true_vs_inc"] = p["pred_true"]
        rec["pred_true_vs_inc_pct"] = (math.exp(p["pred_true"]) - 1) * 100
        rec["att_mac_hist_hold"] = sc[name]["att_mac_hist"]
        rec["refuse_below_mde"] = bool(abs(p["pred_true"]) < mde)
        rec["refuse_wrong_sign"] = bool(p["pred_true"] >= 0)
        rec["eligible"] = bool(p["pred_true"] <= -mde)
        js, j1 = controls[j_at[rec["target_scale"]]], controls[j_at[1.0]]
        rec["attribution_pred_true"] = {
            "allocation_vs_J_same_scale": p["pred_true"] - js["pred_true"],
            "budget_J_scale_vs_J_1": js["pred_true"] - j1["pred_true"],
            "resample_J_1_vs_INC": j1["pred_true"],
            "note": ("additive: arm - INC = (arm - J@s) [currency + ladders] + (J@s - J@1) [budget "
                     "correction] + (J@1 - INC) [512-row resample; t40 also candidate07's linear "
                     "thresholds]")}
    summary["controls"] = controls
    summary["inc_heldout_vs_parent"] = pr_par["INC"]
    summary["inc_att_mac_hist_hold"] = sc["INC"]["att_mac_hist"]
    if r6_arm_a:
        pa = pr_inc["r6_arm_A"]
        att = pa["pred_dnll_att"]
        k_a = (float(r6_arm_a["measured_dnll"]) / att) if att < 0 else None
        se_a = (abs(float(r6_arm_a["measured_se"]) / att)) if att < 0 else None
        summary["kappa_att_A"] = {
            "binding": False, "table": r6_arm_a["table"], "measured_dnll": r6_arm_a["measured_dnll"],
            "measured_se": r6_arm_a["measured_se"], "pred_vs_inc": pa, "kappa_att_A": k_a,
            "se": se_a,
            "qk_raise_underpriced": bool(k_a is not None and k_a - 2 * se_a > KAPPA_RULE["kappa_att"]),
            "use": ("descriptive only (registered): the round-6 qk->128 move measured / its Fisher "
                    "prediction on the capture's hold windows. It changes neither the currency, the "
                    "refusal nor the primary; a flag is reported to the user")}
    sens = {}
    s_sens = scales[0]
    for arm in [a for a in arms if ARM_DEFS[a][1]] if sensitivity_ratios else []:
        lads = arm_ladders(capture, arm, policy)
        for rr in sensitivity_ratios:
            nm = arm_table_name(arm, rr, s_sens)
            if nm in summary["arms"]:
                continue
            cand = _table_from_solve(table, solve_arm(capture, lads, float(rr), s_sens), lads, capture)
            p2 = preds(score_tables(capture, {"INC": inc, nm: cand}), "INC", capture.n_tok)[nm]
            sens[nm] = dict(_pred_record(p2, kap), kappa_ratio_used=float(rr))
    for pol in sensitivity_policies:
        if pol not in ("dense", "full"):
            raise SystemExit(f"[r7] sensitivity policy {pol!r} must be dense or full")
        for arm in [a for a in arms if ARM_DEFS[a][0] == "policy"]:
            lads = arm_ladders(capture, arm, pol)
            ra = r_cur if ARM_DEFS[arm][1] else 1.0
            nm = f"{arm_table_name(arm, ra, s_sens)}_{pol}"
            cand = _table_from_solve(table, solve_arm(capture, lads, ra, s_sens), lads, capture)
            p2 = preds(score_tables(capture, {"INC": inc, nm: cand}), "INC", capture.n_tok)[nm]
            sens[nm] = dict(_pred_record(p2, kap), kappa_ratio_used=ra, ladder_policy=pol)
    if sens:
        summary["sensitivity_not_preregistered"] = sens
    if emit:
        write_json(out_dir / "solve_summary.json", summary, exclusive=True)
    return summary


def _table_from_solve(table, sol, lads, capture):
    """In-memory table (no files) with the solve's thresholds and ladders, for scoring."""
    pt = copy.deepcopy(table)
    glob = sorted(int(v) for v in pt["stoc_len_levels"])
    for (op_, b_), th in sol["att_th"].items():
        kk = f"{op_}:t0:l{b_}"
        A = [int(v) for v in lads[(op_, b_)]]
        if A != glob:
            pt["buckets"][kk]["stoc_len_levels"] = sorted(A, reverse=True)
        pt["buckets"][kk]["thresholds"] = th
    pt["per_row_chunk"] = {"buckets": {f"{k[0]}:t0:l{k[1]}": {
        "levels": [int(v) for v in capture.ladder], "thresholds": th}
        for k, th in sol["lin_th"].items()}}
    return pt


# ---------------------------------------------------------------------------------------------
# pure decision math
# ---------------------------------------------------------------------------------------------
def fixed_point_scale(s0, P, C, U, digits=4, band=(0.97, 1.05)):
    """s1 = s0 * (P - U) / (C - U): the arm's controlled cycles (C - U at s0) scaled so the
    arm spends the parent's cost P on the same windows (U = uncontrolled cycles/MAC, the
    protected slice). Rounded to 1e-4; refuses results outside `band`. Disclosed approximation:
    attention escape cycles sit inside target0 * s but do not scale with s (about 1e-4
    relative), so they are treated as controlled here."""
    s0, P, C, U = float(s0), float(P), float(C), float(U)
    if not (C > U and P > U):
        raise ValueError("need C > U and P > U")
    raw = s0 * (P - U) / (C - U)
    s = round(raw, digits)
    if not (band[0] <= s <= band[1]):
        raise ValueError(f"fixed-point scale {s} outside {band}")
    return raw, s


def kappa_decision_core(k_bs, se_bs, rule=KAPPA_RULE, *, undefined_reasons=()) -> dict:
    """KAPPA_RULE['decision'] for a pooled kappa_lin_bs and its SE (None = undefined). ONE-SIDED:
    keep the family currency only if kappa_lin_bs exceeds kappa_att by more than the combined
    band; inside the band, below it, or undefined -> revert_to_kappa_1."""
    ka, ka_se = float(rule["kappa_att"]), float(rule["kappa_att_se"])
    reasons = list(undefined_reasons)
    if not reasons and (k_bs is None or se_bs is None or
                        not (math.isfinite(float(k_bs)) and math.isfinite(float(se_bs)))):
        reasons = ["kappa_lin_bs undefined"]
    out = {"rule_id": rule["id"], "kappa_att": ka, "kappa_att_se": ka_se,
           "kappa_lin_bs": None, "se_kappa_lin_bs": None, "band": None, "flags": [],
           "requires_user_review": False, "undefined_reasons": reasons,
           "implied_ratio_measured": None}
    if reasons:
        branch = "revert_to_kappa_1"
        out["requires_user_review"] = True
    else:
        band = math.sqrt(float(se_bs) ** 2 + ka_se ** 2)
        branch = "keep_0.42" if (float(k_bs) - ka) > band else "revert_to_kappa_1"
        out.update(kappa_lin_bs=float(k_bs), se_kappa_lin_bs=float(se_bs), band=band,
                   implied_ratio_measured=(ka / float(k_bs)) if float(k_bs) > 0 else None)
        if branch == "keep_0.42" and ka / float(k_bs) > 0.55:
            out["flags"].append("gap_case")
        if float(k_bs) < ka - band:
            out["flags"].append("direction_contradicted")
    pins = rule["pins"][branch]
    out.update(decision=branch, ratio=float(rule["decision_ratio"][branch]),
               kappa_pins=dict(pins), currency_ratio=float(pins["kappa_att"]) / float(pins["kappa_lin"]))
    return out


def kappa_decision_pooled(measured: dict, preds: dict, identity_ok: dict = None,
                          rule=KAPPA_RULE) -> dict:
    """KAPPA_RULE['statistic'/'se'/'decision'] from the step-0 measurements and the captures'
    Fisher predictions. measured[cell] = {"window_dnll": [s80 - c17 per window], "windows": [...],
    "ok": bool, "simulated": bool}; preds[cell] = p_c (nats); identity_ok[cell] = the capture's
    identity result (None = not checked here, e.g. unit tests)."""
    cells = list(rule["step0_cells"])
    reasons, per_cell = [], {}
    for c in cells:
        md = measured.get(c)
        if md is None:
            reasons.append(f"{c}: no step-0 measurement")
            continue
        if md.get("ok") is not True:
            reasons.append(f"{c}: step-0 measurement not ok")
        if md.get("simulated"):
            reasons.append(f"{c}: step-0 measurement is a SIMULATION")
        if c not in preds or preds[c] is None or not math.isfinite(float(preds[c])):
            reasons.append(f"{c}: no capture Fisher prediction p_c")
        if identity_ok is not None and identity_ok.get(c) is not True:
            reasons.append(f"{c}: capture identity not ok")
        d = [float(x) for x in md.get("window_dnll") or []]
        if len(d) >= 2 and c in preds and preds[c] is not None:
            m = sum(d) / len(d)
            se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d))
            p = float(preds[c])
            per_cell[c] = {"m": m, "se_m": se, "p": p, "n_windows": len(d),
                           "kappa_lin_bs_cell": (m / p) if p > 0 else None,
                           "se_kappa_cell": (se / p) if p > 0 else None}
    pooled, k_bs, se_bs = None, None, None
    if not reasons:
        wins = [tuple(measured[c].get("windows") or ()) for c in cells]
        n = len(measured[cells[0]]["window_dnll"])
        if len(set(wins)) != 1 or any(len(measured[c]["window_dnll"]) != n for c in cells) or \
                len(wins[0]) != n:
            reasons.append("step-0 cells do not share one window list")
        elif n < 2:
            reasons.append("fewer than 2 windows")
        else:
            sums = [sum(float(measured[c]["window_dnll"][i]) for c in cells) for i in range(n)]
            M = sum(sums) / n
            sd = math.sqrt(sum((x - M) ** 2 for x in sums) / (n - 1))
            P = sum(float(preds[c]) for c in cells)
            pooled = {"sum_m": M, "sum_p": P, "n_windows": n, "sd_window_sum": sd}
            if P <= 0:
                reasons.append(f"pooled prediction sum_c p_c = {P} <= 0")
            else:
                k_bs, se_bs = M / P, sd / math.sqrt(n) / P
    out = kappa_decision_core(k_bs, se_bs, rule, undefined_reasons=reasons)
    out.update(per_cell=per_cell, pooled=pooled, step0_cells=cells)
    return out


def kappa_branch_from_record(rec: dict, rule=KAPPA_RULE) -> dict:
    """Re-derive the branch with the registered ONE-SIDED rule from a decision record made by
    either side (this module's `kappa-decision`, or the eval-side prc_step0_r7 kappa_decision.json,
    schema prc-step0-r7-v1-kappa). The record supplies only the statistic (kappa_lin_bs,
    se_kappa_lin_bs) and whether it is defined; its own branch is recorded, never trusted."""
    if rec.get("rule_id") == rule["id"]:
        reasons = list(rec.get("undefined_reasons") or ())
        source = "prc_r7_solve"
    elif rec.get("schema") == EVAL_KAPPA_SCHEMA:
        reasons = [f for f in (rec.get("flags") or []) if str(f).startswith("undefined")]
        if rec.get("simulated_inputs"):
            reasons.append("eval-side decision built from simulated step-0 inputs")
        source = "prc_step0_r7 (eval side)"
    else:
        raise ValueError(f"not a round-7 kappa decision record (rule_id {rec.get('rule_id')!r}, "
                         f"schema {rec.get('schema')!r})")
    k_bs, se_bs = rec.get("kappa_lin_bs"), rec.get("se_kappa_lin_bs")
    out = kappa_decision_core(None if reasons else k_bs, None if reasons else se_bs, rule,
                              undefined_reasons=reasons)
    out["record_source"] = source
    out["record_decision"] = rec.get("decision")
    out["record_agrees"] = rec.get("decision") == out["decision"]
    if source == "prc_r7_solve" and not out["record_agrees"]:
        raise ValueError("kappa decision record is inconsistent with the registered rule")
    return out


def kappa_for_cell(prereg: dict, decision: dict = None, pins: dict = None, rule=KAPPA_RULE) -> dict:
    """Branch, pins and currency of one cell under its pre-registration. Family-kappa (30B) cells
    need a decision record (either side's; the branch is re-derived one-sidedly). Explicit `pins`
    (e.g. --kappa-lin/--kappa-att) must equal the branch's registered pins."""
    if not prereg.get("family_kappa"):
        branch, info = "dense", {"decision": "dense", "requires_user_review": False}
    else:
        if decision is None:
            raise ValueError("a family-kappa cell needs the step-0 kappa decision record")
        info = kappa_branch_from_record(decision, rule)
        branch = info["decision"]
    want = rule["pins"][branch]
    if pins is not None and (float(pins["kappa_lin"]), float(pins["kappa_att"])) != \
            (float(want["kappa_lin"]), float(want["kappa_att"])):
        raise ValueError(f"kappa pins ({pins['kappa_lin']}, {pins['kappa_att']}) are not the registered "
                         f"pins of branch {branch!r} ({want['kappa_lin']}, {want['kappa_att']})"
                         + ("" if info.get("record_agrees", True) else
                            f"; the record itself says {info.get('record_decision')!r} (two-sided "
                            f"v1 rule?) -- reconcile with the one-sided registered rule"))
    return {"branch": branch, "kappa_lin": float(want["kappa_lin"]),
            "kappa_att": float(want["kappa_att"]),
            "currency_ratio": float(want["kappa_att"]) / float(want["kappa_lin"]),
            "rule_id": rule["id"], "decision_info": info,
            "requires_user_review": bool(info.get("requires_user_review"))}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _capture_from_args(a):
    stem = Path(a.stem)
    state = a.state or str(stem.with_name(stem.stem + "_state.npz"))
    att = a.att_dump or str(stem.with_name(stem.stem + "_att.npz"))
    hold = a.hold_dump or str(stem.with_name(stem.stem + "_hold.npz"))
    return Capture(state, att, hold)


CAPTURE_MANIFEST_DEFAULT = REPO / "benchmark/ppl/kbands/prc_r7_capture_20260928.json"


def manifest_cell(manifest_path, cell_id) -> tuple:
    """(manifest, cell) from the round-7 capture manifest; the manifest's kappa rule must be
    this module's KAPPA_RULE (the registered one)."""
    m = json.loads(Path(manifest_path).read_text())
    rule = (m.get("preregistered_rules") or {}).get("kappa_rule")
    if rule != KAPPA_RULE:
        raise SystemExit("[r7] the manifest's registered kappa_rule differs from prc_r7_solve.KAPPA_RULE")
    cells = [c for c in m["cells"] if c["id"] == cell_id]
    if len(cells) != 1:
        raise SystemExit(f"[r7] manifest has no unique cell {cell_id!r}")
    return m, cells[0]


def kappa_decision_from_files(manifest_path, step0_measured: dict, *, allow_simulated=False) -> dict:
    """The authoritative step-0 decision: step0_measured.json (m_c) of 30B t32/t40 + the round-7
    captures' diags (p_c) and identity records, under KAPPA_RULE."""
    m = json.loads(Path(manifest_path).read_text())
    if (m.get("preregistered_rules") or {}).get("kappa_rule") != KAPPA_RULE:
        raise SystemExit("[r7] the manifest's registered kappa_rule differs from KAPPA_RULE")
    cells = {c["id"]: c for c in m["cells"]}
    measured, preds_, ident, sources = {}, {}, {}, {}
    names = KAPPA_RULE["score_names"]
    for cid in KAPPA_RULE["step0_cells"]:
        c = cells[cid]
        src = {}
        mp = step0_measured.get(cid)
        if mp and Path(mp).is_file():
            md = json.loads(Path(mp).read_text())
            want = {"c17": c["inputs"]["c17_table"]["sha256"],
                    "s80": c["inputs"]["c17_s80_table"]["sha256"]}
            same = all((md.get(k) or {}).get("table") and Path(md[k]["table"]).is_file() and
                       sha256(md[k]["table"]) == h for k, h in want.items())
            sim = bool(md.get("simulated"))
            measured[cid] = {"window_dnll": (md.get("pair") or {}).get("window_dnll"),
                             "windows": md.get("windows"),
                             "ok": md.get("ok") is True and same and (not sim or allow_simulated),
                             "simulated": sim and not allow_simulated}
            src.update(step0_measured=str(mp), step0_measured_sha256=sha256(mp),
                       step0_tables_equal_capture_score_tables=same, simulated=sim)
        dp = Path(c["paths"]["diag"])
        if dp.is_file():
            t = json.loads(dp.read_text()).get("tables") or {}
            try:
                preds_[cid] = (float(t[names["s80"]][KAPPA_RULE["pred_key"]]) -
                               float(t[names["c17"]][KAPPA_RULE["pred_key"]]))
            except (KeyError, TypeError, ValueError):
                pass
            src.update(capture_diag=str(dp), capture_diag_sha256=sha256(dp))
        ip = Path(c["paths"]["identity"])
        ident[cid] = json.loads(ip.read_text()).get("identity_ok") is True if ip.is_file() else False
        src["identity"] = str(ip)
        sources[cid] = src
    dec = kappa_decision_pooled(measured, preds_, ident)
    dec.update(sources=sources, rule=KAPPA_RULE, manifest=str(manifest_path),
               manifest_sha256=sha256(manifest_path), authoritative=not allow_simulated,
               applies_to=KAPPA_RULE["applies_to"])
    return dec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cap_args(p):
        p.add_argument("--stem", required=True, help="capture stem <dir>/<cell>_r7.json")
        p.add_argument("--state", default="")
        p.add_argument("--att-dump", default="")
        p.add_argument("--hold-dump", default="")
        p.add_argument("--parent", required=True, help="parent cfg dir (table.json/wrapper.json)")

    p = sub.add_parser("identity")
    cap_args(p)
    p.add_argument("--ref-gfis", required=True)
    p.add_argument("--ref-gfisla", required=True)
    p.add_argument("--ref-diag", required=True)
    p.add_argument("--inc-table", required=True)
    p.add_argument("--inc-relation", default="identical",
                   choices=("identical", "prc_thresholds_only"))
    p.add_argument("--prefix", type=int, default=128)
    p.add_argument("--expect-joint-line", default="")
    p.add_argument("--expect-global-line", default="")
    p.add_argument("--capture-log", default="")
    p.add_argument("--score-tables", default="", help="name=TABLE,... the capture's --score-tables")
    p.add_argument("--allow-near", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--pipeline-dir", default="", help="also run the pipeline check here")

    p = sub.add_parser("solve", help="the registered arms of one manifest cell")
    p.add_argument("--manifest", default=str(CAPTURE_MANIFEST_DEFAULT),
                   help="the round-7 capture manifest (registration source)")
    p.add_argument("--cell", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--kappa-decision", default="",
                   help="step-0 decision record (either side's); 30B default: <output_root>/step0/"
                        "kappa_decision.json")
    p.add_argument("--identity", default="", help="default: the manifest cell's identity.json")
    p.add_argument("--arms", default="", help="subset of the registered arms (default: all)")
    p.add_argument("--target-scales", default="", help="default s0; other values must be an arm's "
                   "registered fixed-point s1 (--fixed-point-record)")
    p.add_argument("--fixed-point-record", default="",
                   help="eval-side <cell>_fixed_point.json; default <output_root>/fixed_point/"
                        "<cell>_fixed_point.json when a non-s0 scale is requested")
    p.add_argument("--sensitivity-ratios", default="")
    p.add_argument("--sensitivity-policies", default="", help="dense,full (never emitted)")
    # explicit form (prc_fixedpoint_r7 resolve_command): every value is CHECKED against the
    # registration, never used to choose anything
    for f in ("--stem", "--parent", "--inc-table", "--ladder-policy"):
        p.add_argument(f, default="")
    for f in ("--kappa-lin", "--kappa-att", "--mde-nats"):
        p.add_argument(f, type=float, default=None)

    p = sub.add_parser("score")
    cap_args(p)
    p.add_argument("--tables", required=True, help="name=TABLE_OR_WRAPPER.json,...")
    p.add_argument("--ref", default="parent")
    p.add_argument("--out", default="")

    p = sub.add_parser("fixed-point")
    for f in ("--s0", "--P", "--C", "--U"):
        p.add_argument(f, type=float, required=True)

    p = sub.add_parser("kappa-decision", help="the registered step-0 rule (KAPPA_RULE)")
    p.add_argument("--manifest", default="", help="capture manifest (authoritative mode)")
    p.add_argument("--step0-measured", action="append", default=[],
                   help="CELL=<step0 output>/<cell>/step0_measured.json (30B_t32, 30B_t40)")
    p.add_argument("--out", default="", help="write the decision record here (new file)")
    p.add_argument("--allow-simulated", action="store_true", help="dry runs only (non-authoritative)")
    p.add_argument("--what-if-kappa-lin-bs", type=float, default=None,
                   help="print the rule's outcome for a hypothetical pooled kappa_lin_bs")
    p.add_argument("--what-if-se", type=float, default=None)

    a = ap.parse_args(argv)
    if a.cmd == "fixed-point":
        raw, s = fixed_point_scale(a.s0, a.P, a.C, a.U)
        print(json.dumps({"s_raw": raw, "s": s}))
        return 0
    if a.cmd == "kappa-decision":
        if a.what_if_kappa_lin_bs is not None:
            rec = kappa_decision_core(a.what_if_kappa_lin_bs, a.what_if_se)
            rec["authoritative"] = False
            print(json.dumps(rec, indent=1))
            return 0
        if not a.manifest:
            raise SystemExit("[r7] kappa-decision needs --manifest (or --what-if-*)")
        s0m = dict(x.split("=", 1) for x in a.step0_measured)
        rec = kappa_decision_from_files(a.manifest, s0m, allow_simulated=a.allow_simulated)
        if a.out:
            write_json(a.out, rec, exclusive=True)
        print(json.dumps({k: rec.get(k) for k in ("decision", "ratio", "kappa_lin_bs",
                                                  "se_kappa_lin_bs", "band", "flags",
                                                  "requires_user_review", "undefined_reasons",
                                                  "kappa_pins", "currency_ratio", "authoritative")},
                         indent=1))
        return 0
    if a.cmd == "solve":
        m, cell = manifest_cell(a.manifest, a.cell)
        if not cell.get("enabled"):
            raise SystemExit(f"[r7] {a.cell} is disabled in the manifest")
        p_, pre = cell["paths"], cell["prereg"]
        checks = {"--stem": (a.stem, p_["stem"]), "--parent": (a.parent, cell["calib_args"]["parent"]),
                  "--inc-table": (a.inc_table, cell["refs"]["inc_table"]),
                  "--ladder-policy": (a.ladder_policy, pre["ladder_policy"])}
        for flag, (given, want) in checks.items():
            if given and (Path(given).resolve() != Path(want).resolve() if flag != "--ladder-policy"
                          else given != want):
                raise SystemExit(f"[r7] {flag} {given} != the registered {want}")
        if a.mde_nats is not None and float(a.mde_nats) != float(pre["mde_nats"]):
            raise SystemExit(f"[r7] --mde-nats {a.mde_nats} != the registered {pre['mde_nats']}")
        pins = None
        if (a.kappa_lin is None) != (a.kappa_att is None):
            raise SystemExit("[r7] --kappa-lin and --kappa-att go together")
        if a.kappa_lin is not None:
            pins = {"kappa_lin": a.kappa_lin, "kappa_att": a.kappa_att}
        root = Path(m["output_root"])
        kd_path = a.kappa_decision or (str(root / "step0" / "kappa_decision.json")
                                       if pre.get("family_kappa") else "")
        dec = json.loads(Path(kd_path).read_text()) if kd_path else None
        scales = parse_scales(a.target_scales) if a.target_scales else None
        fp = None
        if scales and scales != [float(pre["s0"])]:
            fpp = a.fixed_point_record or str(root / "fixed_point" / f"{a.cell}_fixed_point.json")
            fp = json.loads(Path(fpp).read_text())
        cap = Capture(p_["state"], p_["att_dump"], p_["hold_dump"])
        arms = [x.strip() for x in a.arms.split(",") if x.strip()] or None
        s = run_solve(capture=cap, parent_dir=cell["calib_args"]["parent"],
                      inc_table=cell["refs"]["inc_table"], out_dir=a.out_dir, cell=a.cell,
                      prereg=pre, identity_path=a.identity or p_["identity"],
                      kappa_decision=dec, kappa_pins=pins, arms=arms, scales=scales,
                      fixed_point_record=fp,
                      sensitivity_ratios=[float(x) for x in a.sensitivity_ratios.split(",") if x.strip()],
                      sensitivity_policies=[x.strip() for x in a.sensitivity_policies.split(",")
                                            if x.strip()],
                      r6_arm_a=pre.get("r6_arm_A"))
        for n, rec in s["arms"].items():
            print(f"[r7] {n} ({rec['role']}): calib cost/target {rec['calib_cost_over_target']:.6f}  "
                  f"hold cost/INC {rec['heldout_vs_inc']['cost_over_ref']:.4f}  pred_true vs INC "
                  f"{rec['pred_true_vs_inc']:+.5f} nats  MDE {s['mde_nats']}  "
                  f"{'ELIGIBLE' if rec['eligible'] else 'REFUSED'}")
        return 0
    cap = _capture_from_args(a)
    if a.cmd == "identity":
        st = dict(x.split("=", 1) for x in a.score_tables.split(",") if x) or None
        rep = run_identity(capture=cap, stem=a.stem, parent_dir=a.parent, ref_gfis=a.ref_gfis,
                           ref_gfisla=a.ref_gfisla, ref_diag=a.ref_diag, inc_table=a.inc_table,
                           prefix=a.prefix, expect_joint_line=a.expect_joint_line or None,
                           expect_global_line=a.expect_global_line or None,
                           capture_log=a.capture_log or None, inc_relation=a.inc_relation,
                           allow_near=a.allow_near, capture_score_tables=st)
        if a.pipeline_dir:
            rep["pipeline_check"] = guarded_pipeline_check(cap, a.parent, a.pipeline_dir)
            rep["ok"] = bool(rep["identity_ok"] and rep["pipeline_check"]["ok"])
        write_json(a.out, rep, exclusive=True)
        print(json.dumps({"ok": rep["ok"], "identity_ok": rep["identity_ok"],
                          "identity_level": rep["identity_level"], "checks": rep["checks"]}, indent=1))
        if not rep["identity_ok"]:
            return 3
        return 0 if rep["ok"] else 4
    if a.cmd == "score":
        tabs = {"parent": None}
        for spec in [x for x in a.tables.split(",") if x]:
            nm, pth = spec.split("=", 1)
            t = json.loads(Path(pth).read_text())
            if "threshold_table_path" in t:
                t = json.loads(Path(t["threshold_table_path"]).read_text())
            tabs[nm] = t
        sc = score_tables(cap, tabs)
        out = {"ref": a.ref, "n_tok": cap.n_tok, "preds": preds(sc, a.ref, cap.n_tok),
               "capture": cap.paths}
        if a.out:
            write_json(a.out, out, exclusive=True)
        print(json.dumps(out["preds"], indent=1))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
