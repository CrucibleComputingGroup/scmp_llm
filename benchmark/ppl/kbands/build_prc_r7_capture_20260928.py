"""Build the round-7 CAPTURE manifest kbands/prc_r7_capture_20260928.json (2026-09-28).

  cd scmp_llm && PYTHONPATH=$PWD/kernels <annstention python> \
      benchmark/ppl/kbands/build_prc_r7_capture_20260928.py

Freezes: per-cell calib10_r7 arguments (the incumbent's recorded calib7 args + the round-7
measurement options), every input's sha256, the calib7 log lines the identity step must
reproduce, the round-7 sources + their imported dependencies, the hashes of every frozen
round-4/5/6 manifest (the preflight re-verifies each of THEIR hashed sources, so a round-7 job
cannot run on an edited frozen file), and the PRE-REGISTRATION (v2, 2026-09-28, before any
capture / step-0 measurement / solve exists): the one authoritative kappa rule
(prc_r7_solve.KAPPA_RULE, one-sided), and per cell the primary arm, arm set, ladder policy,
numeric MDE, s0, target-scale / refusal / one-full-test rules (the `prereg` block, enforced by
prc_r7_solve solve).
4B cells are written disabled (4B t40: round-6 gate failed; 4B t64: known-outcome, its plausible
dense upside is below its MDE). --enable-4b only writes a SEPARATE manifest (--out elsewhere).
Refuses to overwrite the manifest once it is pinned for submission or any capture dir exists.
No GPU, no Slurm. Re-run after ANY edit of a hashed source (the preflight refuses otherwise).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

SCMP = Path("/home/allenjin/Projects/SCMP")
REPO = SCMP / "scmp_llm"
PPL = REPO / "benchmark/ppl"
KB = PPL / "kbands"
TURBO = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands")
PRC2 = TURBO / "prc2"
OUT_ROOT = TURBO / "prc_r7_20260928"
LOG = Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands")
MPB = SCMP / "hpca_results/llm/ppl/mp_best"
MANIFEST = KB / "prc_r7_capture_20260928.json"
R6_DIAG_OUT = TURBO / "prc_r6_20260928"

# Attention rungs MEASURED beyond each parent ladder (halved; 128 is always measured as the
# escape length). The solve's primary ladder policy uses only lad + {112 if >= 1.1 top} + {128};
# the rest (interior and sub-floor rungs) are CPU sensitivity only.
ATT_SUPERSET = (16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112)
PROT_BASE = (48, 56, 64, 80, 96, 112, 128)
ATT_ROWS = 512
PREFIX = 128
WALL = "12:00:00"
R6_ARMS = R6_DIAG_OUT / "attn_diag_arms"

PREREG_JSON = KB / "prc_r7_prereg_20260928.json"   # the eval side's FROZEN registration record


def shared_prereg():
    """The eval side's frozen registration record (written once, exclusive; prc_r7_prereg_20260928
    load_verified checks it equals the module constants and predates every round-7 output): ONE
    source for MDE, s0, arms, ladder policy, kappa pins and the target-scale rule. Its pins /
    decision ratios must equal prc_r7_solve's, or the build stops."""
    for p_ in (str(REPO), str(REPO / "kernels")):
        if p_ not in sys.path:
            sys.path.insert(0, p_)
    from benchmark.ppl import prc_r7_prereg_20260928 as PR
    from benchmark.ppl import prc_r7_solve as S
    rec, sha, _t = PR.load_verified(PREREG_JSON)
    if rec["kappa"]["pins"] != json.loads(json.dumps(S.KAPPA_PINS)) or \
            rec["kappa"]["decision_ratio"] != S.KAPPA_DECISION_RATIO:
        raise SystemExit("[build] registration record kappa pins / decision ratios != prc_r7_solve's")
    if tuple(rec["arms"]) != ("UK", "K", "U") or rec["ladder_policy_primary"] != "up":
        raise SystemExit("[build] registration record arms / ladder policy changed")
    for c, d in rec["mde_derivation"].items():
        if d["mde_nats"] != rec["mde"]["nats"][c]:
            raise SystemExit(f"[build] {c}: record MDE derivation {d['mde_nats']} != {rec['mde']['nats'][c]}")
    return {"MDE_NATS": rec["mde"]["nats"], "MDE_RULE": rec["mde"]["rule"],
            "S0": rec["target_scale"]["s0"], "ARMS": tuple(rec["arms"]),
            "LADDER_POLICY": rec["ladder_policy_primary"], "FIXED_POINT_RULE": rec["target_scale"]["rule"],
            "mde_derivation": rec["mde_derivation"], "path": str(PREREG_JSON), "sha256": sha,
            "frozen_utc": rec["frozen_utc"]}


CELLS = [
    {"id": "30B_t32", "model": "30B", "target": 32, "enabled": True,
     "inc_table": PRC2 / "30B_t32_c7_gfisla_table.json", "inc_relation": "identical",
     "inc_name": "c7gfisla",
     "c17": PRC2 / "30B_t32_c17e32_table.json", "c17_s80": PRC2 / "30B_t32_c17e32_s80_table.json",
     "log": LOG / "p7_30B_t32_c7_61848816.out", "calib_job": 61848816,
     "initial_target_scale": 1.0037},
    {"id": "30B_t40", "model": "30B", "target": 40, "enabled": True,
     "inc_table": TURBO / "prc_local_20260926/30B_t40/candidate07_mlp_from_projections_table.json",
     "inc_relation": "prc_thresholds_only", "inc_name": "round3_local candidate07",
     "c17": PRC2 / "30B_t40_c17_table.json", "c17_s80": PRC2 / "30B_t40_c17_s80_table.json",
     "log": LOG / "p7_30B_t40_c7_61848817.out", "calib_job": 61848817,
     "initial_target_scale": 1.0043},
    {"id": "30B_t48", "model": "30B", "target": 48, "enabled": True,
     "inc_table": PRC2 / "30B_t48_c7_gfisla_table.json", "inc_relation": "identical",
     "inc_name": "c7gfisla",
     "c17": PRC2 / "30B_t48_c17_table.json", "c17_s80": PRC2 / "30B_t48_c17_s80_table.json",
     "log": LOG / "p7_30B_t48_c7_61848818.out", "calib_job": 61848818,
     "initial_target_scale": 1.0110},
    {"id": "4B_t40", "model": "4B", "target": 40, "enabled": False,
     "inc_table": PRC2 / "4B_t40_c7_gfisla_table.json", "inc_relation": "identical",
     "inc_name": "c7gfisla",
     "c17": PRC2 / "4B_t40_c17_table.json", "c17_s80": PRC2 / "4B_t40_c17_s80_table.json",
     "log": LOG / "p7_4B_t40_c7_61982983.out", "calib_job": 61982983,
     "initial_target_scale": 1.0010},
    {"id": "4B_t64", "model": "4B", "target": 64, "enabled": False,
     "inc_table": PRC2 / "4B_t64_c7_gfisla_table.json", "inc_relation": "identical",
     "inc_name": "c7gfisla",
     "c17": PRC2 / "4B_t64_c17_table.json", "c17_s80": PRC2 / "4B_t64_c17_s80_table.json",
     "log": LOG / "p7_4B_t64_c7_62015988.out", "calib_job": 62015988,
     "initial_target_scale": 1.0015},
]

SOURCES = [
    # round-7 sources
    PPL / "mp_per_row_chunk_calib10_r7.py",
    PPL / "prc_r7_solve.py",
    PPL / "prc_r7_capture_20260928.py",
    PPL / "test_prc_r7_calib_20260928.py",
    KB / "run_prc_r7_capture_20260928.sbatch",
    KB / "build_prc_r7_capture_20260928.py",
    # imported by the above (frozen elsewhere too; hashed here so a change is caught)
    PPL / "mp_per_row_chunk_calib9_r6.py",
    PPL / "mp_per_row_chunk_calib5.py",
    PPL / "prc_r6_attn_diag_arms.py",
    PPL / "test_prc_r6_retarget.py",
    PPL / "prc_r6_retarget_20260928.py",
    PPL / "prc_resolve_r6.py",
    PPL / "calibrate_mp_thresholds.py",
    REPO / "benchmark/quant/eval_quant.py",
    REPO / "loader.py",
    REPO / "model/sc_common.py",
    REPO / "model/awq_apply.py",
    REPO / "model/smoothquant_apply.py",
    REPO / "kernels/scmp_kernels/mp/config.py",
    REPO / "kernels/scmp_kernels/mp/__init__.py",
    REPO / "kernels/scmp_kernels/sc/matmul.py",
    REPO / "kernels/scmp_kernels/sc/kernels.py",
    REPO / "kernels/scmp_kernels/sc/rng.py",
    REPO / "kernels/scmp_kernels/sc/sng.py",
    REPO / "kernels/scmp_kernels/sc/constants.py",
    REPO / "kernels/scmp_kernels/sc/config_helpers.py",
    REPO / "kernels/scmp_kernels/trace.py",
    REPO / "model_qwen4b/qwen3_sc.py",        # loader.load_sc_model's Qwen path
]
FROZEN_MANIFESTS = [
    KB / "prc_r6_attn_diag_20260928.json",
    KB / "prc_r6_retarget_20260928.json",
    KB / "prc_adjacent_20260927.json",
    KB / "prc_adjacent_30b_20260927.json",
    KB / "prc_local_20260926.json",
]
CALIB7_ARG_KEYS = ("parent", "model_path", "ladder", "calib_windows", "holdout_windows",
                   "rows_per_call", "expert_calls_per_block", "bins", "seed", "frontend",
                   "currencies", "grad_mode")


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def first_line(log: Path, prefix: str) -> str:
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            return line.strip()
    raise SystemExit(f"[build] {log}: no line starting with {prefix!r}")


def gate_ok(cell_id: str):
    p = R6_DIAG_OUT / cell_id / "diag_summary.json"
    if not p.is_file():
        return None, p
    s = json.loads(p.read_text())
    return bool(((s.get("round7_gate") or {}).get("binding") or {}).get(
        "proceed_round7_attention_ladder_solve")), p


def r6_record(cell_id: str) -> dict:
    """The FINAL round-6 gate + arm-A numbers of one cell (diag_summary.json, complete, real)."""
    p = R6_DIAG_OUT / cell_id / "diag_summary.json"
    s = json.loads(p.read_text())
    if s.get("simulated") or s.get("complete") is not True:
        raise SystemExit(f"[build] {p}: round-6 summary is simulated or incomplete")
    g = s["round7_gate"]["binding"]
    a = s["arms"]["A"]["metrics"]
    return {"diag_summary": str(p), "diag_summary_sha256": sha256(p),
            "passed": bool(g["proceed_round7_attention_ladder_solve"]),
            "dppl_pct_A": g["dppl_pct_A"], "threshold_dppl_pct": g["threshold_dppl_pct"],
            "dnll_A": a["dnll"], "se_dnll_A": a["se_dnll"], "n_windows": a["n_windows"],
            "sigma_w_A": a["se_dnll"] * math.sqrt(a["n_windows"])}


def build_prereg(c: dict, R: dict) -> dict:
    """The registered solve inputs of one cell (enforced by prc_r7_solve.check_prereg/run_solve);
    MDE / s0 / arms / ladder policy are the shared registration's (prc_r7_prereg_20260928)."""
    cid = c["id"]
    if R["S0"][cid] != c["initial_target_scale"]:
        raise SystemExit(f"[build] {cid}: registration S0 {R['S0'][cid]} != {c['initial_target_scale']}")
    der = R["mde_derivation"][cid]
    mde = {"mde_nats": R["MDE_NATS"][cid],
           "mde_derivation": {"rule": R["MDE_RULE"], "sigma_w": der["sigma_w"],
                              "pairs": der["pairs"], "source": R["path"]}}
    if c["model"] == "30B":
        gate_cell = "30B_t40" if cid == "30B_t48" else cid
        g = r6_record(gate_cell)
        if not g["passed"]:
            raise SystemExit(f"[build] {gate_cell}: round-6 gate did not pass; the registered primary "
                             f"would be K -- re-plan before building")
        pre = {"primary": "UK", "arms": list(R["ARMS"]), "ladder_policy": R["LADDER_POLICY"], **mde,
               "s0": R["S0"][cid],
               "s0_source": ("scout_calib section 4c s_pop (held-out, trace-equivalent): the INC's "
                             "controlled-cycle shortfall vs the parent on held-out windows; = "
                             "prc_r7_prereg_20260928.S0"),
               "family_kappa": True,
               "round6_gate": dict(g, cell=gate_cell,
                                   follows=("30B_t40 (plan section 4 proposal, adopted: same model, "
                                            "qk and av l3 top-bound as at t40)"
                                            if cid == "30B_t48" else None)),
               "primary_rule": "30B cell whose round-6 gate passed -> UK (30B t48 follows 30B t40)"}
        if gate_cell == cid:
            at = R6_ARMS / cid / "A_table.json"
            pre["r6_arm_A"] = {"table": str(at), "sha256": sha256(at),
                               "measured_dnll": g["dnll_A"], "measured_se": g["se_dnll_A"],
                               "diag_summary": g["diag_summary"]}
        return pre
    g = r6_record(cid)
    return {"primary": "U", "arms": ["U"], "ladder_policy": R["LADDER_POLICY"], **mde,
            "s0": R["S0"][cid], "s0_source": "1 + (1 - r_test)/0.95 (approximate); = prc_r7_prereg S0",
            "family_kappa": False, "round6_gate": dict(g, cell=cid),
            "not_run_reason": (
                "round-6 gate FAILED (dPPL(A) %.2f%% vs %.2f%%)" % (g["dppl_pct_A"], g["threshold_dppl_pct"])
                if not g["passed"] else
                "known-outcome trial (feedback_no_known_outcome_trials): gate passed only marginally "
                "(dPPL(A) %.2f%% vs %.2f%%) and the plausible dense upside 0-0.3%% (<= 0.003 nats) is "
                "below this cell's MDE %.4f nats" % (g["dppl_pct_A"], g["threshold_dppl_pct"],
                                                      R["MDE_NATS"][cid]))}


def build_cell(i: int, c: dict, enable_4b: bool, R: dict) -> dict:
    ref_gfisla = PRC2 / f"{c['id']}_c7_gfisla_table.json"
    ref_gfis = PRC2 / f"{c['id']}_c7_gfis_table.json"
    diag = PRC2 / f"{c['id']}_c7_diag.json"
    tj = json.loads(ref_gfisla.read_text())
    tg = json.loads(ref_gfis.read_text())
    rec = tj["prc6_calib"]
    if {k: v for k, v in rec.items() if k != "table"} != \
            {k: v for k, v in tg["prc6_calib"].items() if k != "table"}:
        raise SystemExit(f"[build] {c['id']}: c7 gfis / gfisla calib records differ")
    parent = Path(rec["parent"])
    if parent != MPB / "configs" / c["model"] / f"target{c['target']}":
        raise SystemExit(f"[build] {c['id']}: unexpected parent {parent}")
    ptable = json.loads((parent / "table.json").read_text())
    pwrap = json.loads((parent / "wrapper.json").read_text())
    lad = sorted(int(v) for v in ptable["stoc_len_levels"])
    if lad != sorted(int(v) for v in pwrap["stoc_len_levels"]) or max(lad) > 128:
        raise SystemExit(f"[build] {c['id']}: parent ladder mismatch / above cap")
    if float(pwrap.get("escape_gate_k")) != 2.0 or int(pwrap.get("escape_stoc_len")) != 128:
        raise SystemExit(f"[build] {c['id']}: escape gate is not mu+2tau at 128")
    psl = int(ptable["protected_channels"]["stoc_len"])
    extra = sorted(set(ATT_SUPERSET) - set(lad) - {128})
    prot = sorted({psl} | set(PROT_BASE))
    if int(rec["att_rows_per_call"]) != PREFIX:
        raise SystemExit(f"[build] {c['id']}: incumbent att rows {rec['att_rows_per_call']} != {PREFIX}")
    calib_args = {k: rec[k] for k in CALIB7_ARG_KEYS}
    calib_args["att_rows_per_call_incumbent"] = int(rec["att_rows_per_call"])
    calib_args["score_tables_incumbent"] = rec["score_tables"]
    calib_args["dump_incumbent"] = rec["dump"]
    inc = json.loads(Path(c["inc_table"]).read_text())
    if c["inc_relation"] == "identical" and sha256(c["inc_table"]) != sha256(ref_gfisla):
        raise SystemExit(f"[build] {c['id']}: INC is supposed to be c7_gfisla byte for byte")
    if c["inc_relation"] == "prc_thresholds_only":
        a = {k: v for k, v in inc.items() if k != "per_row_chunk"}
        b = {k: v for k, v in tj.items() if k != "per_row_chunk"}
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            raise SystemExit(f"[build] {c['id']}: INC differs from c7_gfisla outside per_row_chunk")
    cell_dir = OUT_ROOT / c["id"]
    cap = cell_dir / "capture"
    stem = cap / f"{c['id']}_r7.json"
    awq = MPB / "act_scales/awq_scales" / (
        "awq_scales_" + rec["model_path"].replace("/", "_") + "_b4.pt")
    inputs = {
        "parent_table": parent / "table.json", "parent_wrapper": parent / "wrapper.json",
        "parent_hybrid": parent / "hybrid_config.json", "c7_gfis_table": ref_gfis,
        "c7_gfisla_table": ref_gfisla, "c7_diag": diag, "c7_log": c["log"],
        "inc_table": Path(c["inc_table"]), "c17_table": c["c17"], "c17_s80_table": c["c17_s80"]}
    inp = {k: {"path": str(v), "sha256": sha256(v)} for k, v in inputs.items()}
    inp["awq_cache"] = {"path": str(awq), "size": awq.stat().st_size}
    cell = {
        "index": i, "id": c["id"], "model": c["model"], "target": c["target"],
        "enabled": bool(c["enabled"]),
        "calib_args": calib_args,
        "r7": {"att_rows_per_call": ATT_ROWS, "att_prefix_split": PREFIX,
               "att_measure_extra": extra, "prot_measure": prot,
               "measured_attention_superset": sorted(set(lad) | set(extra) | {128}),
               "score_tables": {"c17": str(c["c17"]), "c17_s80": str(c["c17_s80"]),
                                "inc": str(c["inc_table"])}},
        "parent_ladder": lad, "psl": psl,
        "inputs": inp,
        "refs": {"c7_gfis_table": str(ref_gfis), "c7_gfisla_table": str(ref_gfisla),
                 "c7_diag": str(diag), "c7_log": str(c["log"]), "calib7_job": c["calib_job"],
                 "expect_global_line": first_line(c["log"], "[c6] global-fis: "),
                 "expect_joint_line": first_line(c["log"], "[c7] JOINT lambda: "),
                 "inc_table": str(c["inc_table"]), "inc_name": c["inc_name"],
                 "inc_relation": c["inc_relation"]},
        "paths": {"cell_dir": str(cell_dir), "capture_dir": str(cap), "stem": str(stem),
                  "diag": str(cap / f"{c['id']}_r7_diag.json"),
                  "state": str(cap / f"{c['id']}_r7_state.npz"),
                  "att_dump": str(cap / f"{c['id']}_r7_att.npz"),
                  "hold_dump": str(cap / f"{c['id']}_r7_hold.npz"),
                  "job_log": str(cap / "job.log"), "calib_log": str(cap / "calib10.log"),
                  "identity": str(cap / "identity.json"),
                  "pipeline_dir": str(cap / "pipeline_check")},
        "prereg": build_prereg(c, R),
        "solve_defaults_informational": {
            "initial_target_scale": c["initial_target_scale"],
            "note": "kept for references by name (prc_r7_prereg S0_SOURCE); the registered value is prereg.s0"},
    }
    cell["inputs"]["round6_gate_summary"] = {
        "path": cell["prereg"]["round6_gate"]["diag_summary"],
        "sha256": cell["prereg"]["round6_gate"]["diag_summary_sha256"]}
    if "r6_arm_A" in cell["prereg"]:
        cell["inputs"]["round6_arm_A_table"] = {"path": cell["prereg"]["r6_arm_A"]["table"],
                                               "sha256": cell["prereg"]["r6_arm_A"]["sha256"]}
    if c["model"] == "4B":
        cell["requires_r6_gate"] = {
            "diag_summary": str(R6_DIAG_OUT / c["id"] / "diag_summary.json"),
            "path": ["round7_gate", "binding", "proceed_round7_attention_ladder_solve"]}
        cell["enable_rule"] = ("NOT RUN in round 7: " + cell["prereg"]["not_run_reason"] + ". "
                               "Enabling needs an explicit user decision and a SEPARATE manifest "
                               "(--enable-4b --out <other path>); dense kappa = 1.")
        if enable_4b:
            ok, p = gate_ok(c["id"])
            if ok is None:
                raise SystemExit(f"[build] --enable-4b: {p} does not exist yet")
            cell["enabled"] = bool(ok)
    return cell


def preregistered_rules(R) -> dict:
    from benchmark.ppl.prc_r7_solve import KAPPA_RULE
    return {
        "version": "v2-20260928 (frozen before any round-7 capture, step-0 measurement or solve)",
        "capture": ("The capture is the incumbent's exact calib7 recipe (recorded prc6_calib "
                    "args, seed 0, same windows) run through calib10_r7, whose main() equals "
                    "calib9_r6's except '# r7'-tagged measurement/dump lines; attention sampling "
                    "512 rows/call with the rank<128 prefix measured as calib7's own batches."),
        "identity": ("Before any extended solve: (1) in-process gfis == c7_gfis (all "
                     "non-provenance fields) and the state re-solve reproduces it; (2) dump bins "
                     "== in-process bins; (3) the 128-row prefix view (rep x4) re-solved jointly "
                     "at scale 1.0 == c7_gfisla thresholds (28 linear + 8 attention keys); (4) "
                     "held-out Fisher predictions on the prefix view == the c7 diag (joint gfis, "
                     "gfisla; linear gfis, c17); (5) parent attention replay 1.0; (6) INC lineage "
                     "(identical to c7_gfisla, or t40 candidate07 = c7_gfisla + per_row_chunk "
                     "thresholds only); (7) the capture's OWN gfis/gfisla/c17/c17_s80/inc tables "
                     "re-scored from its dumps (all 512 rows) == its own diag (joint.*.pred_dnll, "
                     "tables.*.pred_dnll_fis) bit for bit. identity.json is always written."),
        "identity_allow_near": False,
        "identity_near_rule": ("Bit identity is required (user rule). identity_level 'near' (every "
                               "CPU-deterministic check exact incl. (7), calib7 log lines reproduced, "
                               "predictions within 2e-5 nats) is recorded but is STOP-AND-REPORT: "
                               "identity_ok = false, exit 3, no solve without a user decision."),
        "fail_closed": ("prc_r7_solve solve refuses unless the cell's identity.json has identity_ok "
                        "= true for the same capture files (sha256). Exit 4 (pipeline check failed, "
                        "identity ok) keeps the capture valid; identity is re-run with --tag after a "
                        "solver fix, never by deleting."),
        "pipeline_check": ("After identity: one extended-ladder table (arm U, kappa 1, scale 1.0) "
                           "is emitted and validated (lineage + runtime resolver sweep incl. escape "
                           "folding). It is never a candidate."),
        "ladder_policy": ("Measured per bucket: parent ladder + 128 + ATT_SUPERSET. Candidate UK/U "
                          "ladders = parent + {112 if 112 >= 1.1*top} + {128} ('up'), enforced by "
                          "prc_r7_solve.check_prereg; 'dense'/'full' exist only as in-memory CPU "
                          "sensitivity entries (never emitted, never candidates)."),
        "kappa_rule": KAPPA_RULE,
        "arms": ("Per cell `prereg.arms`; the primary is `prereg.primary`. 30B: UK primary (round-6 "
                 "gate passed on t32 and t40; t48 follows t40), K and U secondary. Under REVERT U is "
                 "not solved (== UK at kappa 1) and K runs at kappa 1 (== the J control at s0, "
                 "disclosed). J is a CPU control, never an arm."),
        "mde": ("Per cell `prereg.mde_nats` (numeric, frozen here; = the frozen registration record "
                "kbands/prc_r7_prereg_20260928.json mde.nats): " + R["MDE_RULE"]),
        "refusal": ("An arm is screened only if its pred_true (KAPPA_RULE prediction: branch pins, vs "
                    "INC, held-out Fisher on the capture) <= -MDE of its cell. A refused primary ends the "
                    "cell for round 7 (no confirmation, no full test); secondaries are screened for "
                    "information only."),
        "target_scale": ("= kbands/prc_r7_prereg_20260928.json target_scale.rule (shared registration): "
                         + R["FIXED_POINT_RULE"] + " prc_r7_solve solve accepts only s0 or an arm's "
                         "own s1 from that fixed-point record. The table screened is therefore the "
                         "table confirmed and full-tested (same sha256); the screen's |C/P - 1| <= 1% "
                         "gate applies to it."),
        "one_full_test_per_cell": ("Only the registered primary, only after the screen gate "
                                   "(identity/audits ok, cost gate) and confirmation z < -2 on 32 "
                                   "fresh disjoint windows; the full-protocol full test enters "
                                   "best(all) whatever its sign. Secondaries are never confirmed "
                                   "or full-tested."),
        "realized_length_audit": ("Before any screen NLL is read: GPU trace histograms per (op, "
                                  "bucket) resolved through get_levels; realized lengths within the "
                                  "table's bucket ladder (+128 escape), none above 128; every new "
                                  "rung (112/128) realized where the solve's hold-window "
                                  "att_mac_hist_hold predicts it, MAC share within +-0.10 absolute "
                                  "of that prediction; any failure stops the cell (resolver bug "
                                  "family)."),
        "non_candidates": ("Never screened, confirmed or full-tested: the capture's in-process "
                           "tables (<cell>_r7_{wfis,gfis,gfisla}, the 512-row J at scale 1.0), the "
                           "pipeline-check table, the J controls and every "
                           "sensitivity_not_preregistered entry of a solve."),
        "kappa_att_A": ("Descriptive only: kappa_att(A) = round-6 measured dNLL(A) / the capture's "
                        "Fisher prediction of the round-6 A table vs INC (prc_r7_solve solve records "
                        "it). It changes neither the currency, the refusal nor the primary; "
                        "qk_raise_underpriced (kappa_att(A) - 2 SE > 0.83) is reported to the user."),
        "disclosure": ("UK/U buckets carry up to 8-9 rungs vs the parent's 6-7 (more threshold "
                       "compares per row; existing bucket_stoc_len_levels support). Where 128 is a "
                       "bucket rung the mu+2tau escape is a no-op there. On av buckets with a 128 "
                       "rung and th[0] = 1.0 each call's max-metric row (mn = 1.0) runs at 128 "
                       "(pre-existing deploy semantics, 1 row per call; scorer == runtime). "
                       "Attention cost stays dense (mac = K*M). Round-7 gains are per-row attention "
                       "reallocation / calibration objective (+ the s0 budget correction), not T1 "
                       "granularity."),
        "4B": ("Not run: 4B t40 round-6 gate failed; 4B t64 is a known-outcome trial (plausible "
               "dense upside 0-0.3% below its MDE). Enabling needs a user decision and a separate "
               "manifest."),
        "not_citable": "Capture/solve/screen numbers are TRAIN or held-out; only full tests are citable.",
        "no_new_runtime": ("Tables change only per_row_chunk thresholds, attention thresholds "
                           "and attention bucket stoc_len_levels (existing runtime support, "
                           "max 128). RNG, QK transform, smooth scales, escape mu/tau, INT mask, "
                           "psl, global ladder unchanged (validate_lineage)."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enable-4b", action="store_true",
                    help="write a SEPARATE manifest with gate-passing 4B cells enabled (user decision)")
    ap.add_argument("--out", default=str(MANIFEST))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if a.enable_4b and out == MANIFEST.resolve():
        raise SystemExit("[build] --enable-4b must write a separate manifest (--out); the 30B "
                         "manifest is pinned/used by the captures")
    pin = out.with_name(out.stem + "_pin.json")
    if pin.exists():
        raise SystemExit(f"[build] {pin} exists: this manifest is pinned for submission; "
                         f"rebuilding would invalidate the queued jobs")
    R = shared_prereg()
    cells = [build_cell(i, c, a.enable_4b, R) for i, c in enumerate(CELLS)]
    if out == MANIFEST.resolve():
        live = [c["paths"]["capture_dir"] for c in cells if Path(c["paths"]["capture_dir"]).exists()]
        if live:
            raise SystemExit(f"[build] capture dirs exist {live}: never rebuild the manifest while "
                             f"or after captures run")
    for src in SOURCES:
        if not src.is_file():
            raise SystemExit(f"[build] missing source {src}")
    frozen = {}
    for mp in FROZEN_MANIFESTS:
        m = json.loads(mp.read_text())
        bad = [p for p, h in (m.get("code_hashes") or {}).items()
               if not Path(p).is_file() or sha256(p) != h]
        if bad:
            raise SystemExit(f"[build] frozen sources of {mp.name} changed: {bad[:3]}")
        frozen[str(mp)] = {"sha256": sha256(mp), "n_hashed_sources": len(m.get("code_hashes") or {})}
    frozen[R["path"]] = {"sha256": R["sha256"], "n_hashed_sources": 0,
                         "role": "shared registration record (MDE, s0, pins, fixed point)"}
    sbatch = "benchmark/ppl/kbands/run_prc_r7_capture_20260928.sbatch"
    cd = "cd /home/allenjin/Projects/SCMP/scmp_llm && "
    manifest = {
        "schema": "prc-r7-capture-v1", "run_id": "prc_r7_capture_20260928",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": ("Round 7 (T1 per-group allocation, ROUNDS_6_8_PLAN_20260928 section 4) "
                    "capture: one calib10_r7 run per cell measuring what the offline family-kappa "
                    "joint solve over per-bucket attention ladders up to 128 needs, plus the "
                    "incumbent-identity reproduction before any extension."),
        "units": "HALVED code lengths (nominal = 2x); code cap 128",
        "max_total_gpus": 4, "array_gpu_cap": 2, "gpus_per_task": 1,
        "concurrency_note": (
            "Round 6: 62208439 finished (14:18 EDT 2026-09-28); 62208440 (14B t32 re-target, 1 GPU, "
            "full test in-job) may still run. Step 0 (prc_step0_r7, %1) = 1 GPU. Submit (A) tasks 0-1 "
            "%2 now and (B) task 2 after 62208440: worst case 62208440 + step 0 + A = 4, then step 0 "
            "+ A + B = 4. No round-7 screen job while captures run. The sbatch header default is "
            "%1; always pass --array explicitly. 4B tasks 3-4 are disabled (not run)."),
        "output_root": str(OUT_ROOT),
        "protocol": {"frontend": "awq", "awq_obj_bits": 4, "owen_mode": "bitrev",
                     "scramble_masks": 64, "sc_prec": 8, "sc_halve": True,
                     "rng_grid": "fixed128", "qk_rebalance": False, "ctx": 2048,
                     "escape": "mu+2tau at 128, unchanged", "int_mask": "parent hybrid_config (h20)"},
        "preregistered_rules": preregistered_rules(R),
        "shared_registration": {"record": R["path"], "sha256": R["sha256"], "frozen_utc": R["frozen_utc"],
                                "note": ("MDE, s0, arms, ladder policy, kappa pins and the target-scale "
                                         "rule come from the eval side's frozen record (written once); "
                                         "the preflight re-checks its sha256 (frozen_manifests)")},
        "gpu_plan": {
            "per_cell_estimate_h": {"30B": "2.0-2.6 (calib7 1.4-1.8 h + 512 rows x ~18 columns x "
                                           "prefix split + dumps)", "4B": "0.4-0.6 (not run)"},
            "wall_time_request": WALL,
            "pin_first": (cd + "PYTHONPATH=$PWD/kernels /nfs/turbo/coe-nbleier/allenjin/conda-envs/"
                          "annstention/bin/python benchmark/ppl/prc_r7_capture_20260928.py pin "
                          "--manifest benchmark/ppl/kbands/prc_r7_capture_20260928.json"),
            "submit": [
                cd + "R7C_MANIFEST_SHA256=<SHA> sbatch --parsable --array=0-1%2 " + sbatch,
                cd + "R7C_MANIFEST_SHA256=<SHA> sbatch --parsable --array=2 "
                     "--dependency=afterany:62208440 " + sbatch],
            "conservative_alternative": (
                cd + "R7C_MANIFEST_SHA256=<SHA> sbatch --parsable --array=0-2%1 " + sbatch),
            "check_first": "squeue -u allenjin: anything else of ours holding a GPU counts toward the 4",
        },
        "cells": cells,
        "frozen_manifests": frozen,
        "source_files": [str(p) for p in SOURCES],
        "code_hashes": {str(p): sha256(p) for p in SOURCES},
    }
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[build] wrote {out}: {len(cells)} cells "
          f"({sum(c['enabled'] for c in cells)} enabled), {len(SOURCES)} hashed sources, "
          f"{len(frozen)} frozen manifests")
    for c in cells:
        pr = c["prereg"]
        print(f"  {c['index']} {c['id']:8s} enabled={c['enabled']!s:5s} primary={pr['primary']} "
              f"arms={pr['arms']} mde={pr['mde_nats']} s0={pr['s0']} extra={c['r7']['att_measure_extra']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
