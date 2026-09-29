"""Round-7 capture identity RE-CHECK with a corrected check (7) (CPU only, 2026-09-28).

Why this exists
---------------
Capture 62263344_0 (30B_t32) wrote identity.json with identity_level 'fail'. Every check passed
except checks.cpu_exact.own_tables_rescored_equal_capture_diag, and that check is wrong, not the
capture.

The pinned check (prc_r7_solve.own_table_scores) re-scores the capture's own tables from its
dumps. For EVERY one of them, including gfisla, it wants a linear-only reference
diag.tables.<name>.pred_dnll_fis. calib10_r7 main() (verbatim calib7/calib9 code) never writes
diag.tables.gfisla, for this reason:
  * diag["tables"] is filled only in the linear-only held-out loop
    `for tname in ["parent"] + list(tables)` (calib10 lines 755-785);
  * that loop runs BEFORE the round-2 joint solve adds tables["gfisla"] (line 883);
  * gfisla is scored only jointly, into diag["joint"] (lines 905-944).
So the pinned check reads capture_diag['tables.gfisla'] = None, and `equal` is False for any
real capture. The unit-test fixture did not catch this: test_prc_r7_calib_20260928.build_capture
fabricates diag.tables.gfisla, which calib10 never produces.

Corrected check (7') (CHECK7_RULE below)
-----------------------------------------
Start from the pinned own_table_scores output. It is called unchanged, with the capture driver's
exact arguments, and is kept verbatim in the record.
  (a) Compare bit for bit every entry whose capture-diag reference exists.
  (b) Exempt exactly one absent reference, tables.gfisla, and only under all three conditions:
      - the diag's `tables` dict lacks the 'gfisla' KEY (calib10's layout; a null value does not
        count);
      - diag.joint.gfisla exists;
      - joint.gfisla is bit-equal.
      The joint total has the same linear keys added into it first, so a wrong gfisla table,
      linear or attention, still fails there.
  (c) Any other absent reference, any unequal entry, or any missing own table fails.
  (d) Strengthening: every diag.tables row the pinned check never compares (calib10 writes
      'wfis') is re-scored from the dumps and compared bit for bit. It uses the pinned
      score_tables and the pinned check's own formula,
      0.5 * (F_lin - F_lin_parent) / n_tok. diag.tables.parent must be 0.0.

Everything else is imported, not re-implemented:
  * prc_r7_solve.run_identity (checks 1-6, the level rule, allow_near = False);
  * prc_r7_solve.guarded_pipeline_check;
  * prc_r7_capture_20260928.identity_paths (identity_<tag>.json + pipeline_check_<tag>/);
  * prc_r7_capture_20260928.preflight (code, frozen-manifest and input hashes).
Check (7') is installed by swapping the module attribute S.own_table_scores for the duration of
one run_identity call. It is always restored.

Fail-closed reproduction guard
------------------------------
The re-run must reproduce the ORIGINAL identity.json exactly:
  * the same capture files (sha256);
  * every other check value;
  * every note;
  * the pinned check-(7) output, bit for bit.
If it does not, the tagged record is written with identity_ok = false.

Record
------
<capture>/identity_<tag>.json (default tag 'check7fix'), written exclusively; never overwritten.
It carries:
  * the original failure;
  * the corrected check and the reason;
  * the reproduction comparison;
  * the pinned sources' hashes.
The original identity.json is never touched.

Acceptance (read, not edited)
-----------------------------
  * prc_r7_solve.read_identity: `solve --identity <record>`;
  * prc_fixedpoint_r7_20260928.capture_identity: same capture dir; the solve summary's identity
    path is carried into the fixed point, its resolve command and the screen build;
  * prc_step0_r7_20260928 `kappa --capture-identity CELL=<record>`.
Not accepted: prc_r7_solve kappa-decision and run_r7_stage2_cpu_20260928.py. Both read the
manifest's untagged identity.json.

Registered-rule note: under the round-7 pre-registration an identity 'fail' is stop-and-ask.
Using this record is therefore a user decision. The record says so (user_action).

  cd /home/allenjin/Projects/SCMP/scmp_llm && \
  PYTHONPATH=$PWD/kernels /nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python \
      benchmark/ppl/prc_r7_identity_recheck_20260928.py recheck --cell 30B_t32
  ... verify --cell 30B_t32       (acceptance by the solver / fixed-point / step-0 readers)
No GPU, no model, no sbatch. Units: HALVED code lengths.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.ppl import prc_r7_capture_20260928 as D  # noqa: E402  (pinned driver)
from benchmark.ppl import prc_r7_solve as S  # noqa: E402  (pinned solver / identity checks)

SCHEMA = "prc-r7-identity-recheck-v1"
DEFAULT_TAG = "check7fix"
MANIFEST_DEFAULT = REPO / "benchmark/ppl/kbands/prc_r7_capture_20260928.json"
ORIG_KEY = "own_tables_rescored_equal_capture_diag"
NEW_KEY = "own_tables_rescored_equal_capture_diag_corrected"
PINNED_OWN_TABLE_SCORES = S.own_table_scores          # captured at import, before any swap
CHECK7_RULE_ID = "prc-r7-check7-prime-v1-20260928"
CHECK7_RULE = (
    "Check (7'): the pinned prc_r7_solve.own_table_scores output (unchanged call, the capture "
    "driver's arguments). (a) every entry with a capture-diag reference is bit-equal; (b) the "
    "only absent reference allowed is tables.gfisla, and only if the diag's `tables` dict lacks "
    "the 'gfisla' key, diag.joint.gfisla exists and joint.gfisla is bit-equal (calib10 scores "
    "gfisla only jointly; its linear keys are summed first into that joint total); (c) any other "
    "absent reference, unequal entry or missing own table fails; (d) every diag.tables row the "
    "pinned check does not compare (calib10: wfis) is re-scored from the dumps with the pinned "
    "score_tables and must be bit-equal, and diag.tables.parent.pred_dnll_fis must be 0.0.")
EXEMPT_ABSENT = {
    "tables.gfisla": (
        "calib10_r7 main() (= calib7/calib9 verbatim) fills diag['tables'] only in the linear-only "
        "held-out loop `for tname in ['parent'] + list(tables)` (calib10 lines 755-785), which runs "
        "BEFORE the round-2 joint solve adds tables['gfisla'] (line 883); gfisla is scored only in "
        "diag['joint'] (lines 905-944). The reference cannot exist in any calib7/9/10 diag "
        "(prc2 c7 diags lack it too); the unit-test fixture fabricated it."),
}
REASON = (
    "Checker bug, not a capture defect. The pinned check (7) (prc_r7_solve.own_table_scores) compares "
    "the re-scored linear-only prediction of EVERY own table with diag.tables.<name>.pred_dnll_fis, "
    "including gfisla, a key calib10 never writes (see exempt.tables.gfisla). The comparison is "
    "`want is not None and got == want`, so tables.gfisla is False for every real capture and "
    "identity_level is 'fail' by construction. Every reference that exists matches bit for bit.")
USER_ACTION = (
    "Re-tagged identity record: the pinned check (7) failed (identity_level 'fail' = registered "
    "STOP and ask the user); this record replaces that one check by the corrected check (7') and "
    "re-runs every other check with the pinned code. Solving from it needs the user's explicit OK.")


def jround(x):
    """The JSON round trip the pinned writer applies (S.write_json / S._json_default)."""
    return json.loads(json.dumps(x, default=S._json_default))


# ---------------------------------------------------------------------------------------------
# check (7')
# ---------------------------------------------------------------------------------------------
def correct_check7(pinned: dict, capture, stem, capture_score_tables=None) -> dict:
    """Check (7') from the pinned own_table_scores output (see CHECK7_RULE). Pure: reads the
    capture's diag and (for (d)) its emitted tables and dumps; writes nothing."""
    stem = Path(stem)
    capd = json.loads(Path(capture.r7["files"]["diag"]).read_text())
    dt = capd.get("tables") or {}
    dj = capd.get("joint") or {}
    names = list(pinned.get("equal") or {})
    want = pinned.get("capture_diag") or {}
    problems, exempt = [], {}
    absent = sorted(n for n in names if want.get(n) is None)
    for n in absent:
        if n not in EXEMPT_ABSENT:
            problems.append(f"{n}: no capture-diag reference (not a structural exemption)")
    if "tables.gfisla" in absent:
        jg = dj.get("gfisla")
        if "gfisla" in dt:
            problems.append("diag.tables has a 'gfisla' key without pred_dnll_fis: not calib10's layout")
        elif not (isinstance(jg, dict) and jg.get("pred_dnll") is not None):
            problems.append("diag.joint.gfisla is missing: gfisla was never scored by the capture")
        elif pinned["equal"].get("joint.gfisla") is not True:
            problems.append("joint.gfisla is not bit-equal: the exemption needs the joint check")
        else:
            lin = pinned["capture"]["tables.gfisla"]
            exempt["tables.gfisla"] = {
                "reason": EXEMPT_ABSENT["tables.gfisla"],
                "covered_by": "joint.gfisla (bit-equal)",
                "rescored_linear_only": lin,
                "joint_pred_dnll": float(jg["pred_dnll"]),
                "implied_attention_part_informational": float(jg["pred_dnll"]) - lin}
    unequal = sorted(n for n in names if want.get(n) is not None and pinned["equal"][n] is not True)
    for n in unequal:
        problems.append(f"{n}: re-scored {pinned['capture'][n]!r} != capture diag {want[n]!r}")
    if pinned.get("missing_own_tables"):
        problems.append(f"own tables missing: {pinned['missing_own_tables']}")
    required = {"joint.gfis", "joint.gfisla", "tables.gfis", "tables.gfisla"}
    required |= {f"tables.{n}" for n in (capture_score_tables or {})}
    if required - set(names):
        problems.append(f"the pinned check did not compare {sorted(required - set(names))}")
    # (d) diag.tables rows the pinned check never compares
    if "parent" in dt and dt["parent"].get("pred_dnll_fis") != 0.0:
        problems.append(f"diag.tables.parent.pred_dnll_fis = {dt['parent'].get('pred_dnll_fis')!r} != 0.0")
    covered = {n.split(".", 1)[1] for n in names if n.startswith("tables.")}
    extra = [r for r in dt if r != "parent" and r not in covered]
    supp, tabs, tpath = {}, {"parent": None}, {}
    for r in extra:
        p = stem.with_name(f"{stem.stem}_{r}_table.json")
        if not p.is_file():
            problems.append(f"diag.tables.{r} has no emitted table {p.name} to re-score")
            continue
        tabs[r] = json.loads(p.read_text())
        tpath[r] = p
    if len(tabs) > 1:
        sc = S.score_tables(capture, tabs)
        for r in tpath:
            got = 0.5 * (sc[r]["F_lin"] - sc["parent"]["F_lin"]) / capture.n_tok   # = own_table_scores
            ref = (dt.get(r) or {}).get("pred_dnll_fis")
            eq = ref is not None and got == float(ref)
            supp[f"tables.{r}"] = {"capture": got, "capture_diag": ref, "equal": eq,
                                   "table": str(tpath[r]), "table_sha256": S.sha256(tpath[r])}
            if not eq:
                problems.append(f"tables.{r} (supplementary): re-scored {got!r} != capture diag {ref!r}")
    equal = {n: pinned["equal"][n] for n in names if n not in exempt}
    equal.update({n: v["equal"] for n, v in supp.items()})
    return {"ok": bool(names) and not problems, "rule_id": CHECK7_RULE_ID, "rule": CHECK7_RULE,
            "equal": equal, "exempt": exempt, "supplementary": supp, "problems": problems,
            "absent_references": absent, "missing_own_tables": pinned.get("missing_own_tables"),
            "capture": pinned.get("capture"), "capture_diag": want,
            "diag_layout": {"tables": list(dt), "joint": [k for k in dj if k != "att_th"]},
            "pinned_check": pinned}


@contextlib.contextmanager
def corrected_check7_installed():
    """Route prc_r7_solve.run_identity's check (7) through check (7') for one call; the pinned
    function is restored whatever happens. Yields the list of (7') results produced."""
    if S.own_table_scores is not PINNED_OWN_TABLE_SCORES:
        raise RuntimeError("prc_r7_solve.own_table_scores is not the pinned function")
    calls = []

    def own_table_scores_7prime(capture, stem, capture_score_tables=None):
        pinned = PINNED_OWN_TABLE_SCORES(capture, stem, capture_score_tables)
        out = correct_check7(pinned, capture, stem, capture_score_tables)
        calls.append(out)
        return out

    S.own_table_scores = own_table_scores_7prime
    try:
        yield calls
    finally:
        S.own_table_scores = PINNED_OWN_TABLE_SCORES


# ---------------------------------------------------------------------------------------------
# reproduction of the original record
# ---------------------------------------------------------------------------------------------
def _without(d: dict, *keys) -> dict:
    return {k: v for k, v in (d or {}).items() if k not in keys}


def reproduction(orig: dict, rep: dict, pinned7: dict, *, orig_pipe=None, new_pipe=None) -> dict:
    """Does the re-run reproduce the ORIGINAL identity record exactly (everything except check
    (7)'s verdict and the level derived from it)? rep: the re-run record BEFORE the key rename."""
    r = jround(rep)
    oc, nc = orig.get("checks") or {}, r["checks"]
    items = {
        "files": orig.get("files") == r["files"],
        "cpu_exact_other": _without(oc.get("cpu_exact"), ORIG_KEY) == _without(nc["cpu_exact"], ORIG_KEY),
        "exact": oc.get("exact") == nc["exact"],
        "near": oc.get("near") == nc["near"],
        "notes_other": _without(orig.get("notes"), "own_table_scores") ==
                       _without(r["notes"], "own_table_scores"),
        "pinned_check7_output": (orig.get("notes") or {}).get("own_table_scores") == jround(pinned7),
        "allow_near": orig.get("allow_near") is False and r["allow_near"] is False,
        "near_tolerance": orig.get("near_pred_abs_tolerance_nats") == r["near_pred_abs_tolerance_nats"],
        "cell": orig.get("cell") in (None, r.get("cell")),
        "manifest_sha256": orig.get("manifest_sha256") == r.get("manifest_sha256"),
    }
    out = {"ok": all(items.values()), "items": items,
           "original_check7_value": (oc.get("cpu_exact") or {}).get(ORIG_KEY),
           "original_failed_checks": sorted(f"{g}.{k}" for g in ("cpu_exact", "exact")
                                            for k, v in (oc.get(g) or {}).items() if not v)}
    opc, npc = orig.get("pipeline_check"), r.get("pipeline_check")
    if opc is not None and npc is not None and orig_pipe and new_pipe:
        norm = json.loads(json.dumps(opc).replace(str(orig_pipe), str(new_pipe)))
        tables = {}
        for key in ("table", "wrapper"):
            a, b = opc.get(key), npc.get(key)
            if a and b and Path(a).is_file() and Path(b).is_file():
                tables[key] = S.sha256(a) == S.sha256(b)
        out["pipeline_check_informational"] = {"record_equal_modulo_dir": norm == npc,
                                               "emitted_files_equal": tables}
    return out


# ---------------------------------------------------------------------------------------------
# the re-check
# ---------------------------------------------------------------------------------------------
def load_manifest_cell(manifest_path, cell_id):
    m = json.loads(Path(manifest_path).read_text())
    idx = [c["index"] for c in m["cells"] if c["id"] == cell_id]
    if len(idx) != 1:
        raise SystemExit(f"[r7rc] manifest has no unique cell {cell_id!r}")
    m, cell = D.load_manifest(manifest_path, idx[0])     # the pinned driver's loader
    m["_path"] = str(manifest_path)
    return m, cell


def pinned_source_check(manifest: dict) -> dict:
    """The modules imported here are the manifest's hashed files (preflight hashes them)."""
    code = manifest.get("code_hashes") or {}
    out = {}
    for mod in (S, D):
        f = str(Path(mod.__file__).resolve())
        out[f] = {"sha256": S.sha256(f), "manifest_sha256": code.get(f),
                  "equal": code.get(f) == S.sha256(f)}
    return out


def recheck(manifest: dict, cell: dict, *, tag=DEFAULT_TAG, manifest_path=None,
            original_identity=None, run_preflight=True, require_pin=True) -> tuple:
    """Re-run the full identity check with check (7') and write identity_<tag>.json.
    Returns (exit code, record, path): 0 ok, 3 identity not ok, 4 identity ok but pipeline check
    failed (the capture driver's codes); 0 with path None when the original is already ok."""
    manifest_path = Path(manifest_path or manifest["_path"])
    if not tag:
        raise SystemExit("[r7rc] a re-check needs a tag (the original identity.json is never rewritten)")
    out, pipe = D.identity_paths(cell, tag)                 # pinned: identity_<tag>.json, pipeline_check_<tag>
    if out.exists() or Path(pipe).exists():
        raise SystemExit(f"[r7rc] {out} or {pipe} exists; use a new --tag")
    orig_path = Path(original_identity or cell["paths"]["identity"])
    if not orig_path.is_file():
        raise SystemExit(f"[r7rc] no original identity record {orig_path}: the capture's identity "
                         "step has not run (capture still running?)")
    orig = json.loads(orig_path.read_text())
    if orig.get("identity_ok") is True:
        print(f"[r7rc] {orig_path}: identity_ok already true (level {orig.get('identity_level')}); "
              "nothing to re-check")
        return 0, orig, None
    pre = []
    if run_preflight:
        pre = D.preflight(manifest, cell, require_fresh=False)
        if pre:
            raise SystemExit(f"[r7rc] pinned preflight failed (the imported checks are not the "
                             f"registered code): {pre}")
    msha = S.sha256(manifest_path)
    if require_pin:
        pin = D.pin_path(manifest_path)
        if not pin.is_file() or json.loads(pin.read_text()).get("manifest_sha256") != msha:
            raise SystemExit(f"[r7rc] manifest {manifest_path} is not the pinned version ({pin})")
    src = pinned_source_check(manifest) if run_preflight else {}
    if src and not all(v["equal"] for v in src.values()):
        raise SystemExit(f"[r7rc] imported modules differ from the manifest's hashes: {src}")
    p, ref = cell["paths"], cell["refs"]
    cap = S.Capture(p["state"], p["att_dump"], p["hold_dump"])
    if cap.prefix != cell["r7"]["att_prefix_split"] or cap.att_rows != cell["r7"]["att_rows_per_call"]:
        raise SystemExit("[r7rc] capture was not measured with the manifest's attention sampling")
    with corrected_check7_installed() as calls:            # identical arguments to D.run_identity
        rep = S.run_identity(capture=cap, stem=p["stem"], parent_dir=cell["calib_args"]["parent"],
                             ref_gfis=ref["c7_gfis_table"], ref_gfisla=ref["c7_gfisla_table"],
                             ref_diag=ref["c7_diag"], inc_table=ref["inc_table"],
                             prefix=cell["r7"]["att_prefix_split"],
                             expect_joint_line=ref["expect_joint_line"],
                             expect_global_line=ref["expect_global_line"],
                             capture_log=p["calib_log"], inc_relation=ref["inc_relation"],
                             allow_near=bool(manifest["preregistered_rules"]["identity_allow_near"]),
                             capture_score_tables=cell["r7"]["score_tables"])
    if S.own_table_scores is not PINNED_OWN_TABLE_SCORES or len(calls) != 1:
        raise RuntimeError("check (7') was not applied exactly once / the pinned check was not restored")
    c7p = calls[0]
    rep["cell"] = cell["id"]
    rep["tag"] = tag
    rep["manifest_sha256"] = msha
    rep["pipeline_check"] = S.guarded_pipeline_check(cap, cell["calib_args"]["parent"], pipe,
                                                     policy="up")
    rep["ok"] = bool(rep["identity_ok"] and rep["pipeline_check"]["ok"])
    repro = reproduction(orig, rep, c7p["pinned_check"], orig_pipe=Path(p["pipeline_dir"]),
                         new_pipe=Path(pipe))
    cpu = rep["checks"]["cpu_exact"]
    cpu[NEW_KEY] = cpu.pop(ORIG_KEY)
    lvl_pinned = rep["identity_level"]
    if lvl_pinned == "exact" and not (all(cpu.values()) and all(rep["checks"]["exact"].values())):
        raise RuntimeError("pinned level rule disagrees with the recorded checks")
    if not repro["ok"]:
        rep["identity_ok"] = rep["ok"] = False
        rep["identity_level"] = "fail"
    rep["recheck"] = {
        "schema": SCHEMA, "tag": tag,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": str(Path(__file__).resolve()), "tool_sha256": S.sha256(Path(__file__).resolve()),
        "original_identity": {"path": str(orig_path), "sha256": S.sha256(orig_path),
                              "identity_ok": orig.get("identity_ok"),
                              "identity_level": orig.get("identity_level"),
                              "failed_checks": repro["original_failed_checks"]},
        "original_failure": {"check": f"checks.cpu_exact.{ORIG_KEY}",
                             "value": repro["original_check7_value"],
                             "pinned_function": "prc_r7_solve.own_table_scores",
                             "absent_references": c7p["absent_references"],
                             "unequal": sorted(n for n, v in (c7p["pinned_check"].get("equal") or {}).items()
                                               if not v)},
        "corrected_check": {"check": f"checks.cpu_exact.{NEW_KEY}", "value": c7p["ok"],
                            "rule_id": CHECK7_RULE_ID, "rule": CHECK7_RULE,
                            "exempt": c7p["exempt"], "supplementary": c7p["supplementary"],
                            "problems": c7p["problems"]},
        "reason": REASON,
        "level_rule": ("prc_r7_solve.run_identity's own level rule (imported), allow_near = "
                       f"{rep['allow_near']}; level from the pinned rule: {lvl_pinned}"),
        "reproduction_of_original": repro,
        "pinned_preflight_problems": pre if run_preflight else "not run (test)",
        "pinned_sources": src,
        "accepted_by": {
            "prc_r7_solve.read_identity": "solve --identity <this record>",
            "prc_fixedpoint_r7_20260928.capture_identity": "same capture dir; carried from the solve "
                                                           "summary into the fixed point and screen build",
            "prc_step0_r7_20260928.capture_identity_status": "kappa --capture-identity CELL=<this record>",
            "NOT": ["prc_r7_solve kappa-decision (reads the untagged identity.json)",
                    "kbands/run_r7_stage2_cpu_20260928.py preconditions (read the untagged identity.json)"]},
    }
    rep["user_action"] = USER_ACTION
    S.write_json(out, rep, exclusive=True)
    rc = 3 if not rep["identity_ok"] else (0 if rep["ok"] else 4)
    return rc, rep, out


def verify(cell: dict, record, *, capture_manifest=None) -> dict:
    """Run the downstream readers on a tagged record (read-only)."""
    rec = Path(record)
    cap = S.Capture(cell["paths"]["state"], cell["paths"]["att_dump"], cell["paths"]["hold_dump"])
    res = {}
    try:
        res["prc_r7_solve.read_identity"] = S.read_identity(rec, cap, cell["id"])
    except SystemExit as e:
        res["prc_r7_solve.read_identity"] = {"refused": str(e)}
    try:
        from benchmark.ppl import prc_fixedpoint_r7_20260928 as F
        kw = {"capture_manifest": Path(capture_manifest)} if capture_manifest else {}
        r = F.capture_identity(cell["id"], path=rec, **kw)
        res["prc_fixedpoint_r7_20260928.capture_identity"] = {k: r[k] for k in (
            "path", "sha256", "ok", "identity_level", "near_flag", "pipeline_check_ok")}
    except Exception as e:  # noqa: BLE001  (reported, a refusal is a result)
        res["prc_fixedpoint_r7_20260928.capture_identity"] = {"refused": f"{type(e).__name__}: {e}"}
    try:
        from benchmark.ppl import prc_step0_r7_20260928 as P0
        r = P0.capture_identity_status(rec)
        res["prc_step0_r7_20260928.capture_identity_status"] = {k: r.get(k) for k in (
            "path", "ok", "identity_level", "sha256", "pipeline_check_ok")}
    except Exception as e:  # noqa: BLE001
        res["prc_step0_r7_20260928.capture_identity_status"] = {"refused": f"{type(e).__name__}: {e}"}
    return res


def downstream_commands(manifest: dict, tag=DEFAULT_TAG) -> dict:
    """Exact CPU commands that consume the tagged records (after the user's OK)."""
    cells = {c["id"]: c for c in manifest["cells"] if c.get("enabled")}
    rec = {cid: str(D.identity_paths(c, tag)[0]) for cid, c in cells.items()}
    pre = ("cd /home/allenjin/Projects/SCMP/scmp_llm && PYTHONPATH=$PWD/kernels "
           "/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python ")
    kappa_cells = [c for c in S.KAPPA_RULE["step0_cells"] if c in rec]
    out = {"records": rec,
           "kappa": pre + "benchmark/ppl/prc_step0_r7_20260928.py kappa --manifest "
                    "benchmark/ppl/kbands/prc_step0_r7_20260928.json --capture-manifest "
                    "benchmark/ppl/kbands/prc_r7_capture_20260928.json " +
                    " ".join(f"--capture-identity {c}={rec[c]}" for c in kappa_cells),
           "solve_s0": {cid: pre + "benchmark/ppl/prc_r7_solve.py solve --manifest "
                        f"benchmark/ppl/kbands/prc_r7_capture_20260928.json --cell {cid} --out-dir "
                        f"{c['paths']['cell_dir']}/solve_s0 --identity {rec[cid]}"
                        for cid, c in cells.items()},
           "fixed_point_and_after": ("prc_fixedpoint_r7_20260928.py compute --cell C --solve-summary "
                                     "<cell_dir>/solve_s0/solve_summary.json takes the identity path "
                                     "recorded in the solve summary; its resolve command carries "
                                     "--identity <record>; build_prc_screen_r7_20260928.py reads it "
                                     "from the fixed point. No flag needed after the s0 solve.")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("recheck", "verify", "commands"))
    ap.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    ap.add_argument("--cell", default="")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    a = ap.parse_args(argv)
    if a.cmd == "commands":
        m = json.loads(Path(a.manifest).read_text())
        print(json.dumps(downstream_commands(m, a.tag), indent=1))
        return 0
    m, cell = load_manifest_cell(a.manifest, a.cell)
    if a.cmd == "recheck":
        rc, rep, out = recheck(m, cell, tag=a.tag, manifest_path=a.manifest)
        if out is None:
            return rc
        rc7 = rep["recheck"]
        print(json.dumps({"cell": cell["id"], "record": str(out), "exit": rc, "ok": rep["ok"],
                          "identity_ok": rep["identity_ok"], "identity_level": rep["identity_level"],
                          "checks": rep["checks"], "pipeline_check_ok": rep["pipeline_check"]["ok"],
                          "original_failure": rc7["original_failure"],
                          "corrected_check": {k: rc7["corrected_check"][k] for k in
                                              ("value", "exempt", "supplementary", "problems")},
                          "reproduction_of_original": rc7["reproduction_of_original"]},
                         indent=1, default=str))
        return rc
    if a.cmd == "verify":
        rec = D.identity_paths(cell, a.tag)[0]
        res = verify(cell, rec, capture_manifest=a.manifest)
        print(json.dumps(res, indent=1, default=str))
        return 0 if all("refused" not in v for v in res.values()) else 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
