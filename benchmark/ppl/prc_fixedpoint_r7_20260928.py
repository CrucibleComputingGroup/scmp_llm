"""Round-7 TARGET-SCALE FIXED POINT (2026-09-28): the pre-registered rule of
prc_r7_prereg_20260928.py (task item d), plus the validators the screen builder shares.

  1. The calib side solves every arm (UK, K, U) ONCE at s0 = PREREG.S0[cell] with the pinned
     kappa and MDE (prc_r7_solve.py solve ... --target-scales s0 --mde-nats MDE).
  2. ``compute`` (this file, CPU) replays each s0 table on the cell's step-0 INC16 profile (INC's
     deployed trajectory on the cell's 16 screen windows; every profiled row re-dispatched through
     the arm's OWN ladders/thresholds - baseline_cost refuses changed ladders, this replay does
     not; escaped attention rows and the protected slice stay fixed) -> C(s0); P = step-0
     PARENT16 exact cost; U = protected-slice cycles/MAC of the INC16 trace;
     s1 = round(s0 (P - U) / (C(s0) - U), 1e-4). Writes <cell>_fixed_point.json (exclusive) and
     prints the ONE re-solve command.
  3. The calib side re-solves each arm at its own s1; the screen builder accepts only those
     tables (build_prc_screen_r7_20260928.py --from-solve CELL=<s1 summary> --fixed-point CELL=<fp>).
No iteration, no second re-solve, no re-screen.

  python benchmark/ppl/prc_fixedpoint_r7_20260928.py compute --cell 30B_t40 --solve-summary S0SUMMARY \
      [--out FILE] [--accept-kappa-review]
  python benchmark/ppl/prc_fixedpoint_r7_20260928.py replay --cell 30B_t40 --table T.json   (informational)
Units: HALVED stream lengths/costs (nominal = 2x); max 128. No GPU, no model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.ppl import prc_eval_r7_20260928 as E  # noqa: E402  (step-0 frozen: import only)
from benchmark.ppl import prc_r7_prereg_20260928 as PR  # noqa: E402
from benchmark.ppl.prc_local_proposals import _group_cost  # noqa: E402  (round-3 frozen: import only)

SCHEMA = "prc-fixedpoint-r7-v1"
KDIR = REPO / "benchmark/ppl/kbands"
STEP0_ROOT = E.R7_ROOT / "step0"
FP_ROOT = E.R7_ROOT / "fixed_point"
CAPTURE_MANIFEST = KDIR / "prc_r7_capture_20260928.json"
STEP0_MANIFEST = KDIR / "prc_step0_r7_20260928.json"
KAPPA_DECISION = STEP0_ROOT / "kappa_decision.json"
SOLVER = "prc_r7_solve.py"


class FixedPointError(RuntimeError):
    pass


def model_of(cid: str) -> str:
    return cid.split("_")[0]


# ---------------------------------------------------------------------------------------------
# replay with the table's OWN ladders
# ---------------------------------------------------------------------------------------------
def table_group_spec(table: dict, group: dict) -> tuple[list, list]:
    """(levels, thresholds) the runtime resolves for a profiled group's bucket key: PRC ->
    per_row_chunk levels (ascending); attention -> bucket ladder, else operator default, else the
    global ladder (descending), the get_levels order."""
    key = group["key"]
    if group["kind"] == "prc":
        e = ((table.get("per_row_chunk") or {}).get("buckets") or {}).get(key)
        if e is None:
            raise FixedPointError(f"profiled PRC bucket {key} missing from the table")
        return [int(v) for v in e["levels"]], [float(v) for v in e["thresholds"]]
    payload = (table.get("buckets") or {}).get(key)
    if payload is None or payload.get("thresholds") is None:
        raise FixedPointError(f"profiled attention bucket {key} missing from the table")
    op = key.split(":")[0]
    if payload.get("stoc_len_levels") is not None:
        lv = payload["stoc_len_levels"]
    else:
        od = (table.get("operator_defaults") or {}).get(op)
        lv = od["stoc_len_levels"] if isinstance(od, dict) and od.get("stoc_len_levels") is not None \
            else E.global_ladder(table)
    lv = [int(v) for v in lv]
    if max(lv) > E.CAP:
        raise FixedPointError(f"{key}: ladder {lv} above the code cap {E.CAP}")
    return lv, [float(v) for v in payload["thresholds"]]


def replay_cost(table: dict, profile: dict, *, prot_macs: float = 0.0, psl_inc=None, psl_arm=None) -> dict:
    """Mean SC length of ``table`` replayed on a ProfileCollector snapshot (first order: INC's
    trajectory). Fixed work (protected slice, escaped attention rows, constant-metric calls) stays
    at its observed length, except that a changed protected-slice length moves prot_macs x dpsl."""
    fixed = float(profile["fixed_cycle_macs"])
    if psl_arm is not None and psl_inc is not None and psl_arm != psl_inc:
        fixed += float(prot_macs) * (int(psl_arm) - int(psl_inc))
    by_kind = {"prc": 0.0, "attention": 0.0}
    ladders_changed = []
    for tag, g in profile["groups"].items():
        lv, th = table_group_spec(table, g)
        if g["kind"] == "prc" and lv != [int(v) for v in g["levels"]]:
            raise FixedPointError(f"{tag}: PRC ladders may not change ({g['levels']} -> {lv})")
        if lv != [int(v) for v in g["levels"]]:
            ladders_changed.append(tag)
        by_kind[g["kind"]] += _group_cost(dict(g, levels=lv), th)
    cyc = fixed + by_kind["prc"] + by_kind["attention"]
    tm = float(profile["total_macs"])
    return {"cost": cyc / tm, "cycle_macs": cyc, "total_macs": tm, "fixed_cycle_macs": fixed,
            "prc_cycle_macs": by_kind["prc"], "attention_cycle_macs": by_kind["attention"],
            "ladders_changed": sorted(ladders_changed)}


def fixed_point_scale(s0, P, C, U) -> tuple[float, float]:
    """s1 = s0 (P - U)/(C - U), rounded to the pre-registered grid; refused outside the band."""
    s0, P, C, U = float(s0), float(P), float(C), float(U)
    if not (C > U and P > U):
        raise FixedPointError(f"degenerate costs P={P} C={C} U={U}")
    raw = s0 * (P - U) / (C - U)
    step = PR.FIXED_POINT["rounding"]
    s1 = round(round(raw / step) * step, 4)
    lo, hi = PR.FIXED_POINT["band"]
    if not (lo <= s1 <= hi):
        raise FixedPointError(f"fixed-point scale {s1} outside [{lo}, {hi}]")
    return raw, s1


# ---------------------------------------------------------------------------------------------
# inputs: step-0 reference, capture identity, kappa decision, solve summaries
# ---------------------------------------------------------------------------------------------
def step0_reference(cid: str, *, root: Path = STEP0_ROOT, step0_manifest: Path = STEP0_MANIFEST,
                    allow_simulated=False) -> dict:
    """The cell's complete step-0 screen-window reference: P, INC16 profile + trace, U."""
    m = E.read_json(step0_manifest)
    if cid not in m["cells"]:
        raise FixedPointError(f"{cid}: not a step-0 cell (no pre-registered fixed point possible)")
    c = m["cells"][cid]
    d = Path(root) / cid
    if (d / "failure.json").exists():
        raise FixedPointError(f"{cid}: step-0 cell failed ({d / 'failure.json'})")
    summ = E.read_json(d / "step0_summary.json")
    if summ.get("complete") is not True or summ.get("identity", {}).get("restored_incumbent_exact") is not True:
        raise FixedPointError(f"{cid}: step-0 cell is not complete")
    if bool(summ.get("simulated")) and not allow_simulated:
        raise FixedPointError(f"{cid}: step-0 cell is a SIMULATION")
    pref = E.read_json(d / "parent_reference.json")
    inc = E.read_json(d / "INC16_nll.json")
    starts = [int(s) for s in c["windows"]["starts"]]
    if pref["windows"] != starts or inc["window_starts"] != starts or not pref.get("audit_ok"):
        raise FixedPointError(f"{cid}: step-0 P / INC16 are not audited runs on the cell's 16 windows")
    if E.sha256_file(pref["trace"]) != pref["trace_sha256"]:
        raise FixedPointError(f"{cid}: PARENT16 trace changed")
    if Path(inc["wrapper"]).resolve() != Path(c["arms"]["INC"]["wrapper"]).resolve():
        raise FixedPointError(f"{cid}: INC16 was not run on the incumbent wrapper")
    audit = E.read_json(d / "INC16_audit.json")
    if not audit.get("ok"):
        raise FixedPointError(f"{cid}: INC16 audit failed")
    prof_path = d / "INC16_profile.json"
    files = {n: {"path": str(p), "sha256": E.sha256_file(p)} for n, p in (
        ("step0_summary", d / "step0_summary.json"), ("parent_reference", d / "parent_reference.json"),
        ("inc16_nll", d / "INC16_nll.json"), ("inc16_profile", prof_path), ("inc16_trace", Path(inc["trace"])),
        ("inc16_audit", d / "INC16_audit.json"), ("parent16_trace", Path(pref["trace"])))}
    ra = audit["runtime_audit"]
    return {"cell": cid, "dir": str(d), "windows": starts, "P": float(pref["P"]), "inc16_cost": float(inc["cost"]),
            "U_prot": float(ra["U_prot"]), "total_macs": float(inc["total_macs"]),
            "prot_macs": float(ra["mac_share"].get("prot", 0.0)) * float(inc["total_macs"]),
            "profile": str(prof_path), "simulated": bool(summ.get("simulated")), "files": files,
            "inc_wrapper": c["arms"]["INC"]["wrapper"], "inc_table": c["arms"]["INC"]["table"],
            "step0_manifest_sha256": E.sha256_file(step0_manifest)}


def capture_cell(cid: str, capture_manifest: Path = CAPTURE_MANIFEST) -> dict:
    cm = E.read_json(capture_manifest)
    cells = cm["cells"] if isinstance(cm["cells"], list) else list(cm["cells"].values())
    for c in cells:
        if c["id"] == cid:
            return c
    raise FixedPointError(f"{cid}: not in the capture manifest {capture_manifest}")


def capture_identity(cid: str, *, capture_manifest: Path = CAPTURE_MANIFEST, path=None) -> dict:
    """The capture's identity record must exist with identity_ok == true (the calib side writes
    identity_ok; a legacy record's ok is accepted when identity_ok is absent) and level exact or near
    ('near' is flagged; the v2 capture manifest makes near a stop). ``path`` may be a re-tagged
    identity record (calib side: identity re-run with --tag after a solver fix), but it must live in the
    manifest cell's capture directory. A failed pipeline check is recorded and flagged."""
    manifest_path = Path(capture_cell(cid, capture_manifest)["paths"]["identity"])
    p = Path(path) if path else manifest_path
    if p.resolve().parent != manifest_path.resolve().parent:
        raise FixedPointError(f"{cid}: identity record {p} is not in the capture directory {manifest_path.parent}")
    if not p.is_file():
        raise FixedPointError(f"{cid}: capture identity record missing ({p}); no arm may be solved from it")
    d = E.read_json(p)
    ok = d["identity_ok"] if "identity_ok" in d else d.get("ok")
    if ok is not True or d.get("identity_level") not in ("exact", "near"):
        raise FixedPointError(f"{cid}: capture identity failed (identity_ok {ok}, level {d.get('identity_level')})")
    if d.get("cell") not in (None, cid):
        raise FixedPointError(f"{cid}: identity record is for {d.get('cell')}")
    if not d.get("files"):
        raise FixedPointError(f"{cid}: identity record lists no capture files")
    pc = d.get("pipeline_check")
    pipeline_ok = None if pc is None else bool(pc.get("ok"))
    return {"path": str(p), "sha256": E.sha256_file(p), "ok": True, "identity_level": d["identity_level"],
            "near_flag": d["identity_level"] == "near", "pipeline_check_ok": pipeline_ok,
            "files": d.get("files")}


def load_kappa_decision(path=KAPPA_DECISION, *, step0_manifest: Path = STEP0_MANIFEST, accept_review=False,
                        allow_simulated=False) -> dict:
    """The authoritative eval-side kappa decision (prereg KAPPA_AUTHORITY), validated."""
    kd = E.read_json(path)
    if kd.get("schema") != "prc-step0-r7-v1-kappa":
        raise FixedPointError(f"{path}: not the eval-side step-0 kappa decision (schema {kd.get('schema')})")
    m = E.read_json(step0_manifest)
    if kd.get("rule") != m["kappa_rule"]:
        raise FixedPointError(f"{path}: kappa rule differs from the frozen step-0 manifest's")
    if kd.get("decision") not in PR.KAPPA_DECISION_RATIO or kd.get("ratio") != PR.KAPPA_DECISION_RATIO[kd["decision"]]:
        raise FixedPointError(f"{path}: decision {kd.get('decision')} / ratio {kd.get('ratio')} inconsistent")
    if (kd.get("simulated_inputs") or kd.get("simulated_inputs") is None) and not allow_simulated:
        raise FixedPointError(f"{path}: built from simulated step-0 measurements")
    for cid, mp in (kd.get("measured_files") or {}).items():
        if not allow_simulated and E.read_json(mp).get("simulated") is not False:
            raise FixedPointError(f"{path}: measured file {mp} is not a GPU measurement")
    if kd.get("requires_user_review") and not accept_review:
        raise FixedPointError(f"kappa decision flagged for user review ({kd.get('flags')}); needs the user's OK "
                              "(--accept-kappa-review)")
    return {"path": str(path), "sha256": E.sha256_file(path), "decision": kd["decision"], "ratio": kd["ratio"],
            "requires_user_review": bool(kd.get("requires_user_review")), "flags": kd.get("flags"),
            "review_accepted": bool(accept_review and kd.get("requires_user_review")),
            "created": float(kd["created"])}


def expected_ratio(arm: str, pins: dict) -> float:
    """kappa_ratio_used exactly as prc_r7_solve.run_solve computes it."""
    return 1.0 if arm == "U" else float(pins["kappa_att"]) / float(pins["kappa_lin"])


def check_solve_summary(sm: dict, path, cid: str, *, pins: dict, inc_table, identity: dict,
                        prereg_frozen_t: float, want_primary=None) -> list:
    """Every pinned input of a solve summary; returns the list of problems (empty = ok)."""
    bad = []
    if sm.get("tool") != SOLVER or sm.get("cell") != cid:
        bad.append(f"summary is from {sm.get('tool')} for {sm.get('cell')}, not {SOLVER} for {cid}")
    sid = sm.get("identity")
    if sid is not None and (Path(sid.get("path", "")).resolve() != Path(identity["path"]).resolve()
                            or sid.get("sha256") != identity.get("sha256")):
        bad.append(f"the solve read identity record {sid.get('path')} ({str(sid.get('sha256'))[:12]}), not the "
                   f"verified {identity.get('path')}")
    if want_primary is not None and sm.get("primary_kind") not in (None, want_primary):
        bad.append(f"the solve's registered primary {sm.get('primary_kind')} != the eval-side primary {want_primary}")
    if sm.get("kappa_branch") not in (None, pins["branch"]):
        bad.append(f"the solve's kappa branch {sm.get('kappa_branch')} != {pins['branch']}")
    if sm.get("inc_table_sha256") != E.sha256_file(inc_table):
        bad.append("inc_table_sha256 is not the best(all) incumbent table (pred_true must be vs INC)")
    cap = {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in (sm.get("capture") or {}).items()}
    idf = {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in (identity.get("files") or {}).items()}
    if not cap or cap != idf:
        bad.append("the solve's capture files are not the identity-checked capture's")
    for k, v in cap.items():
        if not Path(v["path"]).is_file() or E.sha256_file(v["path"]) != v["sha256"]:
            bad.append(f"capture file {k} changed since the solve")
    if (sm.get("kappa_lin"), sm.get("kappa_att")) != (pins["kappa_lin"], pins["kappa_att"]):
        bad.append(f"absolute kappa ({sm.get('kappa_lin')}, {sm.get('kappa_att')}) != pinned "
                   f"({pins['kappa_lin']}, {pins['kappa_att']}) for branch {pins['branch']}")
    if sm.get("mde_nats") != PR.MDE_NATS[cid]:
        bad.append(f"mde_nats {sm.get('mde_nats')} != pre-registered {PR.MDE_NATS[cid]}")
    if sm.get("ladder_policy") != PR.LADDER_POLICY:
        bad.append(f"ladder policy {sm.get('ladder_policy')} != pre-registered {PR.LADDER_POLICY!r}")
    if os.path.getmtime(path) <= prereg_frozen_t:
        bad.append("solve summary predates the pre-registration record")
    for name, rec in (sm.get("arms") or {}).items():
        arm = rec.get("arm")
        if arm == "J":
            continue
        if arm not in PR.ARMS:
            bad.append(f"{name}: unknown arm {arm!r}")
            continue
        if rec.get("kappa_ratio_used") != expected_ratio(arm, pins):
            bad.append(f"{name}: kappa_ratio_used {rec.get('kappa_ratio_used')} != {expected_ratio(arm, pins)}")
        want_pol = "inherit" if arm == "K" else PR.LADDER_POLICY
        if rec.get("ladder_policy") != want_pol:
            bad.append(f"{name}: ladder policy {rec.get('ladder_policy')} != {want_pol}")
        h = rec.get("heldout_vs_inc") or {}
        try:
            pt = float(pins["kappa_lin"]) * float(h["pred_dnll_lin"]) + float(pins["kappa_att"]) * float(h["pred_dnll_att"])
        except (KeyError, TypeError):
            bad.append(f"{name}: no family-split held-out prediction vs INC")
            continue
        if rec.get("pred_true_vs_inc") != pt:
            bad.append(f"{name}: pred_true_vs_inc {rec.get('pred_true_vs_inc')} != pinned-kappa recomputation {pt}")
        missing = [f for f in ("wrapper", "table_path") if not rec.get(f) or not Path(rec[f]).is_file()]
        if missing:
            bad.append(f"{name}: emitted {missing} missing")
        elif E.sha256_file(rec["wrapper"]) != rec.get("wrapper_sha256") or \
                E.sha256_file(rec["table_path"]) != rec.get("table_sha256"):
            bad.append(f"{name}: emitted wrapper/table changed since the solve")
    return bad


def arm_tables_at(sm: dict, scale: float) -> dict:
    """{arm: (table_name, rec)} for the non-J arms solved at ``scale`` (exactly one per arm)."""
    out = {}
    for name, rec in sm["arms"].items():
        if rec["arm"] == "J" or abs(float(rec["target_scale"]) - float(scale)) > 5e-9:
            continue
        if rec["arm"] in out:
            raise FixedPointError(f"several tables for arm {rec['arm']} at scale {scale}")
        out[rec["arm"]] = (name, rec)
    return out


def psl_of(table: dict):
    return (table.get("protected_channels") or {}).get("stoc_len") if (table.get("protected_channels") or {}).get("indices") else None


# ---------------------------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------------------------
def compute(cid: str, s0_summary, *, out=None, step0_root: Path = STEP0_ROOT, step0_manifest: Path = STEP0_MANIFEST,
            kappa_decision=KAPPA_DECISION, capture_manifest: Path = CAPTURE_MANIFEST, identity_path=None,
            accept_review=False, allow_simulated=False, prereg_path: Path = PR.RECORD, write=True) -> dict:
    prereg, prereg_sha, prereg_t = PR.load_verified(prereg_path)
    model = model_of(cid)
    kd = None
    if model == "30B":
        kd = load_kappa_decision(kappa_decision, step0_manifest=step0_manifest, accept_review=accept_review,
                                 allow_simulated=allow_simulated)
        if kd["created"] <= prereg_t:
            raise FixedPointError("the kappa decision predates the pre-registration record")
    pins = PR.kappa_pins_for(model, kd["decision"] if kd else None)
    ref = step0_reference(cid, root=step0_root, step0_manifest=step0_manifest, allow_simulated=allow_simulated)
    sm = E.read_json(s0_summary)
    ident = capture_identity(cid, capture_manifest=capture_manifest,
                             path=identity_path or (sm.get("identity") or {}).get("path"))
    bad = check_solve_summary(sm, s0_summary, cid, pins=pins, inc_table=ref["inc_table"], identity=ident,
                              prereg_frozen_t=prereg_t)
    if bad:
        raise FixedPointError(f"{cid}: s0 solve summary refused: {bad[:6]}")
    s0 = PR.S0[cid]
    arms0 = arm_tables_at(sm, s0)
    if not arms0:
        raise FixedPointError(f"{cid}: the s0 summary has no arm solved at the pre-registered s0 = {s0}")
    profile = E.read_json(ref["profile"])
    inc_t = E.read_json(ref["inc_table"])
    inc_rep = replay_cost(inc_t, profile)
    if inc_rep["ladders_changed"]:
        raise FixedPointError(f"{cid}: the INC table's JSON ladders disagree with the runtime-resolved profile "
                              f"ladders at {inc_rep['ladders_changed'][:3]} (resolver-bug guard)")
    exact = float(profile["total_cycle_macs"]) / float(profile["total_macs"])
    if inc_rep["cost"] != exact and abs(inc_rep["cost"] / exact - 1) > 1e-12:
        raise FixedPointError(f"{cid}: INC does not replay its own profile ({inc_rep['cost']} vs {exact})")
    if abs(exact / ref["inc16_cost"] - 1) > 1e-12:
        raise FixedPointError(f"{cid}: INC16 profile cost != INC16 trace cost")
    psl_inc = psl_of(inc_t)
    P, U = ref["P"], ref["U_prot"]
    arms = {}
    for arm in PR.ARMS:
        if arm not in arms0:
            arms[arm] = {"status": "not solved at s0"}
            continue
        name, rec = arms0[arm]
        t = E.read_json(rec["table_path"])
        lin = E.validate_lineage(inc_t, t, total_blocks=E.TOTAL_BLOCKS[model], allow_psl_change=False)
        if not lin["ok"]:
            raise FixedPointError(f"{cid} {name}: lineage vs INC failed {lin['failures'][:3]}")
        rp = replay_cost(t, profile, prot_macs=ref["prot_macs"], psl_inc=psl_inc, psl_arm=psl_of(t))
        C0 = rp["cost"]
        try:
            raw, s1 = fixed_point_scale(s0, P, C0, U)
            status = "ok"
        except FixedPointError as exc:
            raw, s1, status = None, None, f"refused: {exc}"
        arms[arm] = {"status": status, "s0_table_name": name, "s0_table": rec["table_path"],
                     "s0_table_sha256": rec["table_sha256"], "C_replay_s0": C0, "C_replay_s0_over_P": C0 / P,
                     "calib_cost_over_target_s0": rec.get("calib_cost_over_target"),
                     "heldout_cost_over_inc_s0": (rec.get("heldout_vs_inc") or {}).get("cost_over_ref"),
                     "s1_raw": raw, "s1": s1, "replay": {k: v for k, v in rp.items() if k != "total_macs"}}
    ok_s1 = sorted({a["s1"] for a in arms.values() if a.get("status") == "ok"})
    capc = capture_cell(cid, capture_manifest)
    fp_path = Path(out) if out else FP_ROOT / f"{cid}_fixed_point.json"
    resolve_cmd = None
    if ok_s1:
        resolve_cmd = (
            f"python benchmark/ppl/{SOLVER} solve --manifest {capture_manifest} --cell {cid} "
            f"--out-dir {Path(capc['paths']['cell_dir']) / 'solve_fixedpoint'} "
            f"--arms {','.join(a for a in PR.ARMS if arms[a].get('status') == 'ok')} "
            f"--target-scales {','.join(f'{s:.4f}' for s in ok_s1)} --fixed-point-record {fp_path} "
            f"--identity {ident['path']} "
            + (f"--kappa-decision {kd['path']} " if kd else "")
            + f"--stem {capc['paths']['stem']} --parent {capc['calib_args']['parent']} --inc-table {ref['inc_table']} "
            f"--ladder-policy {PR.LADDER_POLICY} --kappa-lin {pins['kappa_lin']} --kappa-att {pins['kappa_att']} "
            f"--mde-nats {PR.MDE_NATS[cid]}")
    rec = {"schema": SCHEMA, "cell": cid, "created": time.time(), "simulated": ref["simulated"],
           "rule": PR.FIXED_POINT["rule"], "prereg": {"path": str(prereg_path), "sha256": prereg_sha},
           "s0": s0, "P": P, "U": U, "inc16_cost": ref["inc16_cost"], "inc_replay_cost": inc_rep["cost"],
           "kappa_pins": pins, "kappa_decision": kd, "mde_nats": PR.MDE_NATS[cid],
           "capture_identity": {k: ident[k] for k in ("path", "sha256", "identity_level", "near_flag",
                                                       "pipeline_check_ok")},
           "s0_summary": {"path": str(s0_summary), "sha256": E.sha256_file(s0_summary)},
           "capture_files": sm["capture"], "step0": ref["files"], "step0_manifest_sha256": ref["step0_manifest_sha256"],
           "arms": arms, "resolve_command": resolve_cmd,
           "note": "Screen exactly each arm's table at its own s1 (build_prc_screen_r7_20260928.py --fixed-point). "
                   "No iteration and no re-screen."}
    if write:
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        E.write_json(fp_path, rec, exclusive=True)
        rec["_path"] = str(fp_path)
    return rec


def load_fixed_point(path, cid: str) -> dict:
    fp = E.read_json(path)
    if fp.get("schema") != SCHEMA or fp.get("cell") != cid:
        raise FixedPointError(f"{path}: not a {SCHEMA} record for {cid}")
    for k, v in fp["step0"].items():
        if E.sha256_file(v["path"]) != v["sha256"]:
            raise FixedPointError(f"{path}: step-0 input {k} changed since the fixed point")
    if E.sha256_file(fp["s0_summary"]["path"]) != fp["s0_summary"]["sha256"]:
        raise FixedPointError(f"{path}: s0 solve summary changed since the fixed point")
    return fp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("compute", "replay"))
    ap.add_argument("--cell", required=True)
    ap.add_argument("--solve-summary", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--accept-kappa-review", action="store_true")
    # CPU-test overrides of the registered locations (stage-2 chain end-to-end test); defaults = registered paths.
    ap.add_argument("--step0-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--capture-manifest", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--kappa-decision", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    over = {k: Path(v) for k, v in (("step0_root", a.step0_root), ("capture_manifest", a.capture_manifest),
                                    ("kappa_decision", a.kappa_decision)) if v}
    if over and (not a.out or str(Path(a.out).resolve()).startswith(str(E.R7_ROOT))):
        raise SystemExit("test overrides need an explicit --out outside the Turbo round-7 root")
    if a.cmd == "replay":
        ref = step0_reference(a.cell)
        t = E.read_json(a.table)
        if "threshold_table_path" in t:
            t = E.read_json(E.resolve_table_path(a.table))
        rp = replay_cost(t, E.read_json(ref["profile"]), prot_macs=ref["prot_macs"],
                         psl_inc=psl_of(E.read_json(ref["inc_table"])), psl_arm=psl_of(t))
        print(json.dumps({"cell": a.cell, "C_replay": rp["cost"], "P": ref["P"], "C_over_P": rp["cost"] / ref["P"],
                          "U": ref["U_prot"], "note": "informational replay; not a fixed-point record"}, indent=1))
        return 0
    if not a.solve_summary:
        raise SystemExit("--solve-summary (the s0 solve) is required")
    rec = compute(a.cell, a.solve_summary, out=a.out, accept_review=a.accept_kappa_review, **over)
    print(json.dumps({k: rec[k] for k in ("cell", "s0", "P", "U", "_path")} |
                     {"arms": {k: {x: v.get(x) for x in ("status", "C_replay_s0_over_P", "s1")}
                               for k, v in rec["arms"].items()}}, indent=1))
    if rec["resolve_command"]:
        print("[r7fp] re-solve ONCE (calib side):\n  " + rec["resolve_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
