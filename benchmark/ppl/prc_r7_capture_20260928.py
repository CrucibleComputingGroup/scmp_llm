"""Round-7 CAPTURE driver (2026-09-28): one calib10_r7 capture per cell + the incumbent-identity
reproduction BEFORE any extended solve.

Cells (manifest kbands/prc_r7_capture_20260928.json): 30B t32 / t40 / t48 enabled; 4B t40 / t64
present but disabled (they need the round-6 attention gate; rebuild the manifest with
--enable-4b after reading it). Each capture is the incumbent's exact calib7 recipe (parent,
seed 0, 6+2 stratified TRAIN windows, dense ladder, Fisher currency, rows/experts per call,
grad mode) run through calib10_r7 with the round-7 measurement options:
  --att-rows-per-call 512 --att-prefix-split 128   (prefix = calib7's 128-row sample, exact)
  --att-measure-extra <superset>   attention rungs 16..112 beyond the parent ladder (+128 via
                                   the escape length, already measured)
  --prot-measure <psl + 48..128>   protected-slice Fisher error (dump only)
  --score-tables c17=,c17_s80=,inc=   held-out Fisher scores (step-0 kappa_lin denominator)
  --solve-state / --att-dump / --r7-hold-dump   everything the CPU solver needs
Stages (the launcher chains them; every artifact goes to <cell>/capture/ on Turbo):
  preflight   CPU. Code hashes (round-7 sources + imported dependencies), every frozen round-4/5/6
              manifest's hashed sources unchanged, input hashes, fresh output dir, cell enabled.
  calib-argv  CPU. NUL-separated calib10 argv for the cell.
  identity    CPU, after the GPU capture. prc_r7_solve.run_identity against the incumbent
              recipe's own artifacts (c7_gfis / c7_gfisla tables, c7 diag, calib7 log lines) and
              the capture's own diag (own tables re-scored from the dumps, bit for bit), then the
              pipeline check (one extended-ladder table emitted and validated through the runtime
              resolver; a raise is recorded, never lost). identity.json is ALWAYS written.
              Exit 3 = identity not ok (level 'fail', or 'near' = stop and report to the user:
              bit identity is registered); exit 4 = identity ok but the pipeline check failed
              (the capture stays valid; fix the solver and re-run identity with --tag).
  pin         CPU, at submit time: write kbands/prc_r7_capture_20260928_pin.json (exclusive)
              with the manifest's sha256 and the exact submit commands; every job asserts
              R7C_MANIFEST_SHA256 == sha256(manifest) == the pin (verify-pin).
Units: HALVED code lengths (nominal = 2x); max 128. No sbatch/srun here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "prc-r7-capture-v1"
CALIB10 = "benchmark/ppl/mp_per_row_chunk_calib10_r7.py"


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path, index):
    m = json.loads(Path(path).read_text())
    if m.get("schema") != SCHEMA:
        raise SystemExit(f"[r7c] manifest schema {m.get('schema')!r} != {SCHEMA}")
    cells = m["cells"]
    if not (0 <= int(index) < len(cells)) or cells[int(index)]["index"] != int(index):
        raise SystemExit(f"[r7c] no cell with index {index}")
    return m, cells[int(index)]


def calib_argv(cell) -> list:
    """calib10 argv: the incumbent's recorded calib7 args + the round-7 options."""
    a = cell["calib_args"]
    p = cell["paths"]
    r7 = cell["r7"]
    score = ",".join(f"{k}={v}" for k, v in r7["score_tables"].items())
    argv = ["--parent", a["parent"], "--model_path", a["model_path"],
            "--out", p["stem"], "--diag", p["diag"],
            "--ladder", a["ladder"], "--calib-windows", str(a["calib_windows"]),
            "--holdout-windows", str(a["holdout_windows"]), "--rows-per-call", str(a["rows_per_call"]),
            "--expert-calls-per-block", str(a["expert_calls_per_block"]), "--bins", str(a["bins"]),
            "--seed", str(a["seed"]), "--frontend", a["frontend"], "--currencies", a["currencies"],
            "--att-rows-per-call", str(r7["att_rows_per_call"]), "--dump", "",
            "--grad-mode", a["grad_mode"], "--score-tables", score,
            "--target-scales", "1.0", "--solve-state", p["state"],
            "--att-measure-extra", ",".join(str(v) for v in r7["att_measure_extra"]),
            "--att-dump", p["att_dump"],
            "--att-prefix-split", str(r7["att_prefix_split"]),
            "--r7-hold-dump", p["hold_dump"],
            "--prot-measure", ",".join(str(v) for v in r7["prot_measure"])]
    return argv


def preflight(manifest, cell, *, require_fresh=True, allow_disabled=False) -> list:
    """Problems (empty = OK). Pure CPU, no writes."""
    bad = []
    if not cell.get("enabled") and not allow_disabled:
        bad.append(f"cell {cell['id']} is disabled in the manifest ({cell.get('enable_rule')})")
    for path, want in manifest["code_hashes"].items():
        if not Path(path).is_file():
            bad.append(f"missing source {path}")
        elif sha256(path) != want:
            bad.append(f"source changed since the manifest was built: {path}")
    for mpath, rec in manifest["frozen_manifests"].items():
        if not Path(mpath).is_file() or sha256(mpath) != rec["sha256"]:
            bad.append(f"frozen manifest changed or missing: {mpath}")
            continue
        fm = json.loads(Path(mpath).read_text())
        for path, want in (fm.get("code_hashes") or {}).items():
            if not Path(path).is_file() or sha256(path) != want:
                bad.append(f"frozen source {path} (hashed by {Path(mpath).name}) changed")
    for name, rec in cell["inputs"].items():
        path = rec["path"]
        if not Path(path).is_file():
            bad.append(f"input {name} missing: {path}")
        elif rec.get("sha256") and sha256(path) != rec["sha256"]:
            bad.append(f"input {name} changed: {path}")
        elif rec.get("size") is not None and Path(path).stat().st_size != rec["size"]:
            bad.append(f"input {name} size changed: {path}")
    cap = cell["r7"]
    try:                                   # the calibrator's and the solver's own parsers
        from benchmark.ppl.mp_per_row_chunk_calib10_r7 import r7_check_prefix, r7_parse_lengths
        from benchmark.ppl.prc_r7_solve import ladder_for
        r7_check_prefix(cap["att_prefix_split"], cap["att_rows_per_call"])
        for key in ("att_measure_extra", "prot_measure"):
            if r7_parse_lengths(",".join(map(str, cap[key])), 128) != sorted(cap[key]):
                bad.append(f"{key} not a sorted unique length list")
        meas = sorted(set(cell["parent_ladder"]) | set(cap["att_measure_extra"]) | {128})
        if meas != cap["measured_attention_superset"]:
            bad.append("measured_attention_superset inconsistent")
        for pol in ("inherit", "up", "dense", "full"):
            ladder_for(pol, cell["parent_ladder"], meas)
        if cell["psl"] not in cap["prot_measure"]:
            bad.append("the parent's protected length is not measured")
    except (SystemExit, ValueError) as e:
        bad.append(f"round-7 option check: {e}")
    try:                                   # the registered kappa rule and per-cell solve inputs
        from benchmark.ppl.prc_r7_solve import KAPPA_RULE, check_prereg
        if (manifest.get("preregistered_rules") or {}).get("kappa_rule") != KAPPA_RULE:
            bad.append("manifest kappa_rule != prc_r7_solve.KAPPA_RULE")
        if manifest["preregistered_rules"].get("identity_allow_near") is not False:
            bad.append("identity_allow_near must be false (bit identity is registered)")
        check_prereg(cell["prereg"])
        if cell["prereg"]["family_kappa"] != (cell["id"] in KAPPA_RULE["applies_to"]):
            bad.append("prereg family_kappa disagrees with KAPPA_RULE applies_to")
    except (KeyError, TypeError, ValueError, SystemExit) as e:
        bad.append(f"pre-registration check: {e}")
    lens = list(cap["att_measure_extra"]) + list(cap["prot_measure"])
    if any(not (1 <= int(v) <= 128) for v in lens):
        bad.append(f"measured lengths outside [1, 128]: {lens}")
    if cap["att_rows_per_call"] % cap["att_prefix_split"] or \
            cap["att_prefix_split"] != cell["calib_args"]["att_rows_per_call_incumbent"]:
        bad.append("prefix split must equal the incumbent's att rows and divide the r7 rows")
    if not str(cell["paths"]["capture_dir"]).startswith(manifest["output_root"] + "/"):
        bad.append("capture dir outside the round-7 output root")
    if require_fresh and Path(cell["paths"]["capture_dir"]).exists():
        bad.append(f"capture dir already exists: {cell['paths']['capture_dir']}")
    gate = cell.get("requires_r6_gate")
    if gate:
        g = Path(gate["diag_summary"])
        if not g.is_file():
            bad.append(f"round-6 gate summary missing: {g}")
        else:
            val = json.loads(g.read_text())
            for key in gate["path"]:
                val = val.get(key) if isinstance(val, dict) else None
            if val is not True:
                bad.append(f"round-6 gate did not pass for {cell['id']} ({gate['path']} = {val!r})")
    return bad


def identity_paths(cell, tag=""):
    """identity.json + pipeline dir of a cell; a --tag re-run writes NEW files next to them."""
    p = cell["paths"]
    if not tag:
        return Path(p["identity"]), Path(p["pipeline_dir"])
    if not tag.replace("_", "").isalnum():
        raise SystemExit(f"[r7c] bad --tag {tag!r}")
    ip = Path(p["identity"])
    return ip.with_name(f"{ip.stem}_{tag}.json"), Path(f"{p['pipeline_dir']}_{tag}")


def run_identity(manifest, cell, tag="") -> int:
    from benchmark.ppl import prc_r7_solve as S
    p, ref = cell["paths"], cell["refs"]
    out, pipe = identity_paths(cell, tag)
    if out.exists():
        raise SystemExit(f"[r7c] {out} exists; re-run with a new --tag")
    cap = S.Capture(p["state"], p["att_dump"], p["hold_dump"])
    if cap.prefix != cell["r7"]["att_prefix_split"] or cap.att_rows != cell["r7"]["att_rows_per_call"]:
        raise SystemExit("[r7c] capture was not measured with the manifest's attention sampling")
    rep = S.run_identity(capture=cap, stem=p["stem"], parent_dir=cell["calib_args"]["parent"],
                         ref_gfis=ref["c7_gfis_table"], ref_gfisla=ref["c7_gfisla_table"],
                         ref_diag=ref["c7_diag"], inc_table=ref["inc_table"],
                         prefix=cell["r7"]["att_prefix_split"],
                         expect_joint_line=ref["expect_joint_line"],
                         expect_global_line=ref["expect_global_line"],
                         capture_log=p["calib_log"], inc_relation=ref["inc_relation"],
                         allow_near=bool(manifest["preregistered_rules"]["identity_allow_near"]),
                         capture_score_tables=cell["r7"]["score_tables"])
    rep["cell"] = cell["id"]
    rep["tag"] = tag or None
    rep["manifest_sha256"] = sha256(os.environ.get("R7C_MANIFEST") or manifest["_path"])
    # the pipeline check runs whatever the identity level (it validates the emitter, not the
    # capture) and can never prevent identity.json from being written
    rep["pipeline_check"] = S.guarded_pipeline_check(cap, cell["calib_args"]["parent"], pipe,
                                                     policy="up")
    rep["ok"] = bool(rep["identity_ok"] and rep["pipeline_check"]["ok"])
    if rep["identity_level"] == "near":
        rep["user_action"] = ("identity_level 'near': STOP and report to the user (bit identity is "
                              "registered); no arm may be solved from this capture without a user "
                              "decision")
    S.write_json(out, rep, exclusive=True)
    print(json.dumps({"cell": cell["id"], "ok": rep["ok"], "identity_ok": rep["identity_ok"],
                      "identity_level": rep["identity_level"], "checks": rep["checks"],
                      "pipeline_check_ok": rep["pipeline_check"]["ok"],
                      "pipeline_check_error": rep["pipeline_check"].get("error"),
                      "pred_abs_diff": rep["notes"].get("pred_abs_diff")},
                     indent=1, default=str))
    if not rep["identity_ok"]:
        return 3
    return 0 if rep["ok"] else 4


def pin_path(manifest_path) -> Path:
    mp = Path(manifest_path)
    return mp.with_name(mp.stem + "_pin.json")


def write_pin(manifest, manifest_path) -> Path:
    """Submit-time record: the manifest sha256 the jobs must run on + the exact commands."""
    import datetime
    sha = sha256(manifest_path)
    pp = pin_path(manifest_path)
    rec = {"schema": "prc-r7-capture-pin-v1", "manifest": str(Path(manifest_path).resolve()),
           "manifest_sha256": sha,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "submit": [c.replace("<SHA>", sha) for c in manifest["gpu_plan"]["submit"]],
           "conservative_alternative": manifest["gpu_plan"]["conservative_alternative"].replace(
               "<SHA>", sha),
           "note": ("Jobs refuse to start unless R7C_MANIFEST_SHA256 (exported at submit) equals "
                    "sha256(manifest) and this pin. Record the job ids in a NEW "
                    "prc_r7_capture_20260928_submission.json after sbatch.")}
    with pp.open("x") as f:
        f.write(json.dumps(rec, indent=1) + "\n")
    return pp


def verify_pin(manifest_path) -> list:
    """Problems (empty = OK) with the submission pin for a running job."""
    bad = []
    sha = sha256(manifest_path)
    env = os.environ.get("R7C_MANIFEST_SHA256", "")
    if env != sha:
        bad.append(f"R7C_MANIFEST_SHA256={env or '<unset>'} != sha256(manifest) {sha}")
    pp = pin_path(manifest_path)
    if not pp.is_file():
        bad.append(f"missing submission pin {pp}")
    elif json.loads(pp.read_text()).get("manifest_sha256") != sha:
        bad.append(f"submission pin {pp} is for another manifest version")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("preflight", "calib-argv", "identity", "cells", "pin",
                                    "verify-pin"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="preflight: do not require a fresh dir")
    ap.add_argument("--allow-disabled", action="store_true")
    ap.add_argument("--tag", default="", help="identity: re-run into identity_<tag>.json")
    a = ap.parse_args(argv)
    m, cell = load_manifest(a.manifest, a.index)
    m["_path"] = a.manifest
    if a.cmd == "pin":
        print(f"[r7c] wrote {write_pin(m, a.manifest)} (manifest sha256 {sha256(a.manifest)})")
        return 0
    if a.cmd == "verify-pin":
        bad = verify_pin(a.manifest)
        for b in bad:
            print(f"[r7c] PIN FAIL: {b}", file=sys.stderr)
        if not bad:
            print(f"[r7c] manifest pin OK ({sha256(a.manifest)})")
        return 0 if not bad else 2
    if a.cmd == "cells":
        for c in m["cells"]:
            print(f"{c['index']}\t{c['id']}\t{'enabled' if c['enabled'] else 'disabled'}")
        return 0
    if a.cmd == "preflight":
        bad = preflight(m, cell, require_fresh=not a.dry, allow_disabled=a.allow_disabled)
        for b in bad:
            print(f"[r7c] PREFLIGHT FAIL {cell['id']}: {b}", file=sys.stderr)
        if not bad:
            print(f"[r7c] preflight OK {cell['id']} ({len(m['code_hashes'])} sources, "
                  f"{len(m['frozen_manifests'])} frozen manifests, {len(cell['inputs'])} inputs)")
        return 0 if not bad else 2
    if a.cmd == "calib-argv":
        sys.stdout.write("\0".join(calib_argv(cell)) + "\0")
        return 0
    if a.cmd == "identity":
        return run_identity(m, cell, a.tag)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
