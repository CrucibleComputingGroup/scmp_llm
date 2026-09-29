"""Round-7 STAGE-2 CPU chain (2026-09-28): everything between the GPU step 0 + captures and the screen submit.

Runs, in the registered order (PRC_R7_PREREG_20260928.md section 8 step 3 / the build result's 'LATER' chain):
  0. preconditions: frozen-manifest hash locks; step-0 (62263343) and capture (62263344) array tasks COMPLETED
     (sacct / squeue); every cell's step-0 outputs complete and non-simulated; every capture identity.json
     identity_ok = true at level 'exact' ('near' or not ok = STOP and ask the user).
  1. kappa stage     prc_step0_r7_20260928.py kappa --manifest <step0> --capture-manifest <capture>
  2. s0 solves       prc_r7_solve.py solve --manifest <capture> --cell C --out-dir $R7/C/solve_s0
  3. fixed point     prc_fixedpoint_r7_20260928.py compute --cell C --solve-summary $R7/C/solve_s0/solve_summary.json
  4. s1 re-solve     the fixed point's ONE printed resolve command, verbatim, once (out-dir $R7/C/solve_fixedpoint);
                     a cell whose fixed point refused every arm is omitted (registered)
  5. screen build    build_prc_screen_r7_20260928.py --from-solve C=<s1> --fixed-point C=<fp> --kappa-decision <kd>
  6. launcher dry run of every enabled screen task (R7SCR_DRY_RUN=1: frozen-input + driver deep preflight)
and prints a per-cell summary (kappa branch, per-arm pred_true vs MDE at s0 and s1, refused arms and why, primary,
predicted cost vs the parent) and, if any cell is enabled, the EXACT screen sbatch command with a throttle that keeps
this user's GPUs <= 4 (squeue). It NEVER calls sbatch.

Idempotent: every tool writes exclusively; an existing complete output is verified and reused, never rewritten; a
partial output (directory without its summary) stops the chain for a user decision (nothing is deleted).
Exit codes: 0 done (screen command printed, or every cell refused -> no command); 2 not ready yet (jobs still
queued/running or outputs missing while jobs run) - re-run later; 3 STOP and ask the user (identity fail/near,
kappa decision flagged for review, failed GPU task, partial outputs, hash lock broken, inconsistent existing
outputs); 4 a chain step failed (tool error / refusal of its inputs) or the command line itself is invalid
(unknown or malformed option: argparse's own exit 2 is remapped to 4 so it can never read as 'not ready yet').
Options must be spelled out in full: abbreviations are refused (allow_abbrev=False), so '--accept' or '--accept-k'
is an unknown option (exit 4) and can never silently turn on the user-review override --accept-kappa-review.

  bash benchmark/ppl/kbands/run_r7_stage2_cpu_20260928.sh [--cells 30B_t32,30B_t40,30B_t48] [--accept-kappa-review]
      [--accept-identity-recheck]   (user-approved 2026-09-28: accept captures that failed ONLY the buggy pinned
      check (7) through their validated identity_check7fix.json; see PRC_R7_CAPTURE_IDENTITY_NOTE_20260928.md)
(--accept-kappa-review only after the user reviewed a flagged kappa decision; --cells drops a cell = user decision.)
Units: HALVED stream lengths/costs (nominal = 2x). No GPU, no model load, no sbatch.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KDIR = REPO / "benchmark/ppl/kbands"
ENV_PY = "/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python"
R7_ROOT = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r7_20260928")
STEP0_MANIFEST = KDIR / "prc_step0_r7_20260928.json"
CAPTURE_MANIFEST = KDIR / "prc_r7_capture_20260928.json"
PREREG_RECORD = KDIR / "prc_r7_prereg_20260928.json"
WINDOWS_REGISTRY = KDIR / "prc_windows_r7_20260928.json"
STEP0_SUBMISSION = KDIR / "prc_step0_r7_20260928_submission.json"
CAPTURE_SUBMISSION = KDIR / "prc_r7_capture_20260928_submission.json"
SCREEN_MANIFEST = KDIR / "prc_screen_r7_20260928.json"
CANDIDATES = KDIR / "prc_candidates_r7_20260928.json"
SCREEN_SBATCH = "benchmark/ppl/kbands/run_prc_screen_r7_20260928.sbatch"
EXPECTED_SHA = {  # the frozen round-7 records (PRC_R7_PREREG_20260928.md section 9)
    STEP0_MANIFEST: "cdbfdbad5e3f168cd3ce8192c0af2b1c1e10e3c589f4edce9ee8373698495d68",
    CAPTURE_MANIFEST: "7a7d8121ffee45947ee24528fd30c5e9a7c29fd2b5c8f1913a3d092f009a9dac",
    PREREG_RECORD: "c7fcfaec091ce01a2d06808f36d12ca617b3306d5f8388ca05b2165cfab4b6f2",
    WINDOWS_REGISTRY: "b06d64d9272f29585b7d8e5322b9120bde14a3db078d4209e1abbd9cdbd8ae49",
}
FROZEN_MANIFESTS = [KDIR / n for n in (  # every manifest whose code_hashes / frozen_manifests must stay unchanged
    "prc_step0_r7_20260928.json", "prc_r7_capture_20260928.json", "prc_r6_attn_diag_20260928.json",
    "prc_r6_retarget_20260928.json", "prc_adjacent_20260927.json", "prc_adjacent_30b_20260927.json",
    "prc_local_20260926.json")]
CELLS = ("30B_t32", "30B_t40", "30B_t48")
KAPPA_CELLS = ("30B_t32", "30B_t40")          # the pooled step-0 kappa statistic (KAPPA_RULE cells)
SCREEN_TASK = {"30B_t32": 0, "30B_t40": 1, "30B_t48": 2, "4B_t40": 3, "4B_t64": 4}
GPU_CAP = 4
USER = os.environ.get("USER", "allenjin")
SLURM_DONE_BAD = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
                  "DEADLINE", "REVOKED")


class Stop(Exception):
    """Chain stop with an exit code (2 not ready, 3 ask the user, 4 step failed)."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def log(msg=""):
    print(f"[r7s2] {msg}" if msg else "", flush=True)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------
# hash locks
# ---------------------------------------------------------------------------------------------
def verify_hash_locks() -> dict:
    """Frozen round-7 records at their registered sha, and every source any round-4/6/7 manifest hashes unchanged."""
    bad, n = [], 0
    for p, want in EXPECTED_SHA.items():
        if sha256(p) != want:
            bad.append(f"{p.name}: sha {sha256(p)[:12]} != registered {want[:12]}")
    for mp in FROZEN_MANIFESTS:
        d = read_json(mp)
        for key in ("code_hashes", "frozen_manifests"):
            for path, h in (d.get(key) or {}).items():
                hh = h if isinstance(h, str) else (h.get("sha256") if isinstance(h, dict) else None)
                if hh is None:
                    continue
                n += 1
                if not Path(path).is_file():
                    bad.append(f"{mp.name}: hashed source missing: {path}")
                elif sha256(path) != hh:
                    bad.append(f"{mp.name}: hashed source CHANGED: {path}")
    if bad:
        raise Stop(3, "hash lock broken (a frozen source changed; do not proceed): " + "; ".join(bad[:8]))
    return {"frozen_records": len(EXPECTED_SHA), "hashed_sources_checked": n}


# ---------------------------------------------------------------------------------------------
# Slurm (read-only: sacct / squeue; never sbatch)
# ---------------------------------------------------------------------------------------------
def _run_ro(argv, timeout=90):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    return (r.stdout if r.returncode == 0 else None), r.stderr


def queued_jobs() -> list:
    """This user's queued/running jobs, one row per array element: [(jobid, state, gpus)]."""
    out, err = _run_ro(["squeue", "-u", USER, "-h", "-r", "-o", "%i|%T|%b"])
    if out is None:
        raise Stop(3, f"squeue failed ({err.strip()[:200]}); cannot check the GPU cap - run where Slurm answers")
    rows = []
    for ln in out.splitlines():
        parts = ln.strip().split("|")
        if len(parts) < 3:
            continue
        m = re.search(r"gpu(?::[A-Za-z0-9_]+)?:(\d+)", parts[2])
        rows.append((parts[0], parts[1], int(m.group(1)) if m else (1 if "gpu" in parts[2] else 0)))
    return rows


def job_states(job_ids) -> dict:
    """{array_task_id: (State, ExitCode)} from sacct for the given (array) job ids."""
    out, err = _run_ro(["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids), "-o", "JobID,State,ExitCode"])
    if out is None:
        raise Stop(3, f"sacct failed ({err.strip()[:200]})")
    states = {}
    for ln in out.splitlines():
        parts = ln.strip().split("|")
        if len(parts) >= 2 and parts[0]:
            states[parts[0]] = (parts[1].split()[0], parts[2].strip() if len(parts) > 2 else "")
    return states


def flag(flags: list, msg: str):
    flags.append(msg)
    log(f"FLAG: {msg}")


def check_jobs(no_slurm: bool, flags: list, accept_failed3=()) -> dict:
    """Step-0 array tasks must be COMPLETED; capture tasks COMPLETED, or FAILED with exit code 4 = the registered
    'identity ok, pipeline check failed: capture valid' (flagged; identity.json is checked next). Anything else
    (incl. capture exit 3 = identity fail / near) stops the chain for the user."""
    if no_slurm:
        return {"skipped": "--no-slurm (tests only)"}
    s0, cap = read_json(STEP0_SUBMISSION), read_json(CAPTURE_SUBMISSION)
    want = {}
    for sub, jid_key, tasks_key in ((s0, "step0_job", "step0_tasks"), (cap, "capture_job", "capture_tasks")):
        for i in range(len(sub[tasks_key])):
            want[f"{sub[jid_key]}_{i}"] = f"{jid_key.split('_')[0]} task {i} ({sub[tasks_key][i]})"
    ids = sorted({k.split("_")[0] for k in want})
    queued = {j for j, _st, _g in queued_jobs()}
    still = sorted(k for k in want if k in queued or any(q.startswith(k.split("_")[0] + "_[") for q in queued))
    if still:
        raise Stop(2, f"not finished yet: {', '.join(want[k] + ' ' + k for k in still)} still queued/running")
    st = job_states(ids)
    missing = [k for k in want if k not in st]
    if missing:
        raise Stop(3, f"sacct has no record of {missing}; check the jobs by hand")
    bad = {}
    for k in want:
        state, code = st[k]
        if state == "COMPLETED":
            continue
        if state == "FAILED" and code == "4:0" and k.startswith(f"{cap['capture_job']}_"):
            flag(flags, f"{want[k]} {k} exited 4: identity ok, pipeline check FAILED (registered: capture valid); "
                        "the solver and the builder re-validate every emitted table")
            continue
        if state == "FAILED" and code == "3:0" and k.startswith(f"{cap['capture_job']}_") and \
                cap["capture_tasks"][int(k.split("_")[1])] in accept_failed3:
            flag(flags, f"{want[k]} {k} exited 3 on the buggy pinned check (7) only; accepted through its validated "
                        "identity_check7fix.json (--accept-identity-recheck, user-approved 2026-09-28)")
            continue
        bad[k] = f"{state} {code}".strip()
    if bad:
        pend = {k: v for k, v in bad.items() if v.split()[0] in ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING",
                                                                "COMPLETING", "SUSPENDED", "RESIZING")}
        if pend and len(pend) == len(bad):
            raise Stop(2, f"not finished yet: {pend}")
        raise Stop(3, f"GPU tasks did not complete cleanly: {bad} - STOP and ask the user (see their logs; a capture "
                      "exit 3 = identity fail or 'near')")
    return {"completed_or_accepted": sorted(want)}


# ---------------------------------------------------------------------------------------------
# preconditions on the GPU outputs
# ---------------------------------------------------------------------------------------------
def capture_cells(cm_path) -> dict:
    cm = read_json(cm_path)
    cells = cm["cells"] if isinstance(cm["cells"], list) else list(cm["cells"].values())
    return {c["id"]: c for c in cells}


# ---------------------------------------------------------------------------------------------
# --accept-identity-recheck  (USER-APPROVED 2026-09-28; PRC_R7_CAPTURE_IDENTITY_NOTE_20260928.md)
# A capture whose identity failed ONLY the pinned check (7) (it demands diag.tables.gfisla, which calib10 never
# writes) is accepted through its tagged identity_check7fix.json, after every provenance check below. Without the
# flag nothing changes: a failed identity still stops the chain (exit 3).
# ---------------------------------------------------------------------------------------------
RECHECK_TAG = "check7fix"
RECHECK_TOOL = REPO / "benchmark/ppl/prc_r7_identity_recheck_20260928.py"
RECHECK_TOOL_SHA = "55e9340e28b46b68b9eeeb73530c9c1e651fbba0ded26445533d90abf998d1f5"
RECHECK_SCHEMA = "prc-r7-identity-recheck-v1"
RECHECK_ONLY_FAILED = ["cpu_exact.own_tables_rescored_equal_capture_diag"]
RECHECK_APPROVAL = {
    "user_answer": "Accept re-check (Recommended)", "date": "2026-09-28",
    "channel": "AskUserQuestion, answered by the user in the main Claude Code session",
    "question": "Round 7's three 30B captures failed only a buggy identity check (it requires a table entry the "
                "calibrator never writes); a CPU re-check shows every other identity check bit-exact. Accept the "
                "re-checked records and continue round 7?"}


def _false_checks(checks) -> list:
    return sorted(f"{g}.{k}" for g, d in (checks or {}).items() for k, v in (d or {}).items() if v is not True)


def recheck_path(cap_cell) -> Path:
    """== prc_r7_capture_20260928.identity_paths(cell, 'check7fix')[0]; never an arbitrary path."""
    ip = Path(cap_cell["paths"]["identity"])
    return ip.with_name(f"{ip.stem}_{RECHECK_TAG}.json")


def original_identity_ok(cap_cell) -> bool:
    ip = Path(cap_cell["paths"]["identity"])
    if not ip.is_file():
        return False
    d = read_json(ip)
    return d.get("identity_ok") is True and d.get("identity_level") == "exact"


def validate_recheck(ctx, cid) -> dict:
    c = ctx["caps"].get(cid)
    if c is None or not c.get("enabled"):
        raise Stop(3, f"{cid}: not an enabled cell of the capture manifest")
    ip, tp = Path(c["paths"]["identity"]), recheck_path(c)
    if not ip.is_file() or not tp.is_file():
        raise Stop(3, f"{cid}: --accept-identity-recheck needs both {ip} and {tp}")
    orig, rec = read_json(ip), read_json(tp)
    rc = rec.get("recheck") or {}
    bad = []
    if rec.get("cell") != cid or orig.get("cell") not in (None, cid):
        bad.append(f"cell mismatch (tagged {rec.get('cell')!r}, original {orig.get('cell')!r})")
    if rec.get("tag") != RECHECK_TAG or rc.get("tag") != RECHECK_TAG:
        bad.append(f"tag {rec.get('tag')!r}/{rc.get('tag')!r} != {RECHECK_TAG!r}")
    if rec.get("ok") is not True or rec.get("identity_ok") is not True or rec.get("identity_level") != "exact":
        bad.append(f"tagged record not ok/exact (ok {rec.get('ok')}, identity_ok {rec.get('identity_ok')}, level "
                   f"{rec.get('identity_level')})")
    if rec.get("allow_near") is not False:
        bad.append("tagged record allows 'near'")
    if _false_checks(rec.get("checks")):
        bad.append(f"tagged record has false checks {_false_checks(rec.get('checks'))}")
    if (rec.get("pipeline_check") or {}).get("ok") is not True:
        bad.append("tagged record's pipeline check is not ok")
    cm_sha = sha256(ctx["capture_manifest"])
    if rec.get("manifest_sha256") != cm_sha or (not ctx["test"] and cm_sha != EXPECTED_SHA[CAPTURE_MANIFEST]):
        bad.append(f"capture manifest sha {str(rec.get('manifest_sha256'))[:12]} != {cm_sha[:12]}")
    if rc.get("schema") != RECHECK_SCHEMA:
        bad.append(f"recheck schema {rc.get('schema')!r}")
    if Path(rc.get("tool", "")).resolve() != RECHECK_TOOL.resolve() or rc.get("tool_sha256") != RECHECK_TOOL_SHA \
            or sha256(RECHECK_TOOL) != RECHECK_TOOL_SHA:
        bad.append("re-check tool path/sha is not the reviewed tool 55e9340e")
    oi = rc.get("original_identity") or {}
    if Path(oi.get("path", "")).resolve() != ip.resolve() or oi.get("sha256") != sha256(ip):
        bad.append("the original identity.json on disk is not the one the re-check read")
    if _false_checks(orig.get("checks")) != RECHECK_ONLY_FAILED:
        bad.append(f"original failed {_false_checks(orig.get('checks'))}, not only {RECHECK_ONLY_FAILED}")
    ots = (rec.get("notes") or {}).get("own_table_scores") or {}
    if ots.get("ok") is not True or ots.get("problems") or sorted(ots.get("exempt") or {}) != ["tables.gfisla"] \
            or ots.get("absent_references") != ["tables.gfisla"] or ots.get("missing_own_tables"):
        bad.append("corrected check (7') is not exactly 'tables.gfisla exempt, no problems'")
    if (rc.get("corrected_check") or {}).get("value") is not True:
        bad.append("corrected check value is not true")
    rep = rc.get("reproduction_of_original") or {}
    if rep.get("ok") is not True or not rep.get("items") or any(v is not True for v in rep["items"].values()):
        bad.append("reproduction of the original identity run is not ok on every item")
    if not rec.get("files") or rec.get("files") != orig.get("files"):
        bad.append("capture files differ between the original and the tagged record")
    for k, v in (rec.get("files") or {}).items():
        if not Path(v.get("path", "")).is_file() or sha256(v["path"]) != v.get("sha256"):
            bad.append(f"capture file {k} on disk does not match its recorded sha256")
    if bad:
        raise Stop(3, f"{cid}: --accept-identity-recheck refused {tp}: " + "; ".join(bad))
    argv = ["@PY", RECHECK_TOOL, "verify", "--cell", cid]
    if ctx["test"]:
        argv += ["--manifest", ctx["capture_manifest"]]
    run_step(f"recheck_verify_{cid}", argv, ctx["logdir"], ctx["py"])     # exit 3 if any downstream reader refuses
    return {"path": str(tp), "sha256": sha256(tp), "original": {"path": str(ip), "sha256": sha256(ip)},
            "tool_sha256": RECHECK_TOOL_SHA, "verify": "all three downstream readers accept"}


def _check_bound(what, got: dict, want: dict):
    if not got or Path(got.get("path", "")).resolve() != Path(want["path"]).resolve() or \
            got.get("sha256") != want["sha256"]:
        raise Stop(3, f"{what} is not bound to the validated re-check record {want['path']} "
                      f"(got {(got or {}).get('path')} / {str((got or {}).get('sha256'))[:12]})")


def check_outputs(r7: Path, cm_path, cells, flags, recheck=None) -> dict:
    caps = capture_cells(cm_path)
    rep = {}
    for cid in sorted(set(cells) | set(KAPPA_CELLS)):
        d = r7 / "step0" / cid
        if (d / "failure.json").exists():
            f = read_json(d / "failure.json")
            raise Stop(3, f"{cid}: step-0 task failed ({f.get('type')}: {str(f.get('message'))[:200]}) - ask the user")
        need = ["step0_summary.json", "parent_reference.json", "INC16_nll.json", "INC16_profile.json", "INC16_audit.json"]
        if cid in KAPPA_CELLS:
            need.append("step0_measured.json")
        miss = [n for n in need if not (d / n).is_file()]
        if miss:
            raise Stop(3, f"{cid}: step-0 outputs missing {miss} under {d} although the jobs finished - ask the user")
        s = read_json(d / "step0_summary.json")
        if s.get("complete") is not True or s.get("simulated") is not False or \
                (s.get("identity") or {}).get("restored_incumbent_exact") is not True:
            raise Stop(3, f"{cid}: step0_summary.json is not a complete, restored, non-simulated GPU run")
        if cid in KAPPA_CELLS:
            md = read_json(d / "step0_measured.json")
            if md.get("ok") is not True or md.get("simulated") is not False:
                raise Stop(3, f"{cid}: step0_measured.json not ok / simulated")
        c = caps.get(cid)
        if c is None or not c.get("enabled"):
            raise Stop(3, f"{cid}: not an enabled cell of the capture manifest")
        ip = Path(c["paths"]["identity"])
        if recheck and cid in recheck:    # user-approved --accept-identity-recheck; validated in validate_recheck
            ip = Path(recheck[cid]["path"])
        if not ip.is_file():
            raise Stop(3, f"{cid}: capture identity record missing ({ip}) - the capture did not finish its identity step")
        ident = read_json(ip)
        lvl = ident.get("identity_level")
        if ident.get("identity_ok") is not True or lvl != "exact":
            raise Stop(3, f"{cid}: capture identity_ok={ident.get('identity_ok')!r} level={lvl!r} (registered: exit 3 "
                          "= identity fail or 'near' -> STOP and ask the user)")
        pc = ident.get("pipeline_check")
        if pc is not None and not pc.get("ok"):
            flag(flags, f"{cid}: capture pipeline check failed (identity ok, capture valid: registered exit 4); "
                        "the builder re-validates every table")
        rep[cid] = {"step0": str(d), "capture_identity": str(ip), "identity_level": lvl,
                    "pipeline_check_ok": None if pc is None else bool(pc.get("ok")),
                    "identity_source": "check7fix re-check (user-approved)" if recheck and cid in recheck
                    else "identity.json"}
    return rep


# ---------------------------------------------------------------------------------------------
# running tools
# ---------------------------------------------------------------------------------------------
def run_step(tag, argv, logdir: Path, py) -> str:
    argv = [py if a == "@PY" else str(a) for a in argv]
    logdir.mkdir(parents=True, exist_ok=True)
    lp = logdir / f"{time.strftime('%Y%m%dT%H%M%S')}_{tag}.log"
    env = dict(os.environ, PYTHONPATH=str(REPO / "kernels"), CUDA_VISIBLE_DEVICES="")
    log(f"RUN {tag}: {' '.join(shlex.quote(a) for a in argv)}")
    t0 = time.time()
    r = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True)
    lp.write_text(f"# {' '.join(shlex.quote(a) for a in argv)}\n# exit {r.returncode}\n--- stdout ---\n{r.stdout}"
                  f"\n--- stderr ---\n{r.stderr}")
    if r.returncode != 0:
        tail = (r.stderr.strip() or r.stdout.strip())[-1500:]
        hint = ""
        if r.returncode < 0 or r.returncode == 137:
            hint = ("\nKILLED by a signal - most likely a memory cap on a real-size capture (the login node's 4 GB user "
                    "slice, or this job's --mem). Move the step's partial output directory aside (nothing is deleted "
                    "automatically) and re-run this same command inside a CPU allocation with more memory (no GPU "
                    "needed).")
        raise Stop(4, f"step {tag} FAILED (exit {r.returncode}, log {lp}):\n{tail}{hint}")
    log(f"  ok ({time.time() - t0:.0f} s, log {lp})")
    return r.stdout


def _exists_complete(out_dir: Path, marker: str) -> bool:
    if (out_dir / marker).is_file():
        return True
    if out_dir.exists():
        raise Stop(3, f"{out_dir} exists without {marker} (a previous run stopped mid-step); nothing is deleted - "
                      "inspect it and move it aside, then re-run")
    return False


# ---------------------------------------------------------------------------------------------
# chain
# ---------------------------------------------------------------------------------------------
def kappa_stage(ctx) -> dict:
    kd = ctx["r7"] / "step0" / "kappa_decision.json"
    if kd.is_file():
        log(f"kappa decision exists, reused: {kd}")
    else:
        argv = ["@PY", "benchmark/ppl/prc_step0_r7_20260928.py", "kappa", "--manifest", STEP0_MANIFEST,
                "--capture-manifest", ctx["capture_manifest"]]
        for c in KAPPA_CELLS:
            if c in ctx.get("recheck", {}):
                argv += ["--capture-identity", f"{c}={ctx['recheck'][c]['path']}"]
        if ctx["test"]:
            argv += ["--out-root", ctx["r7"] / "step0", "--out", kd]
        run_step("kappa", argv, ctx["logdir"], ctx["py"])
    d = read_json(kd)
    if d.get("schema") != "prc-step0-r7-v1-kappa":
        raise Stop(3, f"{kd}: not the eval-side kappa decision record")
    if d.get("simulated_inputs"):
        raise Stop(3, f"{kd}: built from SIMULATED step-0 inputs")
    if d.get("requires_user_review") and not ctx["accept_kappa_review"]:
        raise Stop(3, f"kappa decision {d.get('decision')} is flagged for USER REVIEW: {d.get('flags')} "
                      f"(kappa_lin_bs {d.get('kappa_lin_bs')} +- {d.get('se_kappa_lin_bs')}, band {d.get('band')}). "
                      "Re-run with --accept-kappa-review only after the user's explicit OK.")
    for c in KAPPA_CELLS:
        if c in ctx.get("recheck", {}):
            _check_bound(f"kappa decision's capture identity for {c}",
                         ((d.get("prediction_sources") or {}).get(c) or {}).get("capture_identity"), ctx["recheck"][c])
    return {"path": str(kd), "sha256": sha256(kd), "decision": d["decision"], "ratio": d["ratio"],
            "kappa_lin_bs": d.get("kappa_lin_bs"), "se_kappa_lin_bs": d.get("se_kappa_lin_bs"), "band": d.get("band"),
            "flags": d.get("flags"), "requires_user_review": bool(d.get("requires_user_review")),
            "per_cell": {c: {k: v for k, v in (p or {}).items() if k in ("m", "p", "ok")}
                         for c, p in (d.get("per_cell") or {}).items()}}


def s0_solve(ctx, cid) -> Path:
    out = Path(ctx["caps"][cid]["paths"]["cell_dir"]) / "solve_s0"
    if _exists_complete(out, "solve_summary.json"):
        log(f"{cid}: s0 solve exists, reused: {out}")
    else:
        argv = ["@PY", "benchmark/ppl/prc_r7_solve.py", "solve", "--manifest", ctx["capture_manifest"], "--cell", cid,
                "--out-dir", out]
        if cid in ctx.get("recheck", {}):
            argv += ["--identity", ctx["recheck"][cid]["path"]]
        run_step(f"s0solve_{cid}", argv, ctx["logdir"], ctx["py"])
    sp = out / "solve_summary.json"
    sm = read_json(sp)
    if sm.get("cell") != cid or sm.get("stage") != "initial_s0":
        raise Stop(3, f"{sp}: not the initial s0 solve of {cid} (stage {sm.get('stage')})")
    if cid in ctx.get("recheck", {}):
        _check_bound(f"{cid} s0 solve", sm.get("identity"), ctx["recheck"][cid])
    return sp


def fixed_point(ctx, cid, s0_summary: Path) -> Path:
    fp = (ctx["r7"] / "fixed_point" / f"{cid}_fixed_point.json")
    if fp.is_file():
        rec = read_json(fp)
        if rec.get("cell") != cid or (rec.get("s0_summary") or {}).get("sha256") != sha256(s0_summary):
            raise Stop(3, f"{fp} exists but was computed from another s0 solve summary; refusing to reuse or overwrite")
        log(f"{cid}: fixed point exists, reused: {fp}")
        return fp
    argv = ["@PY", "benchmark/ppl/prc_fixedpoint_r7_20260928.py", "compute", "--cell", cid,
            "--solve-summary", s0_summary]
    if ctx["accept_kappa_review"]:
        argv.append("--accept-kappa-review")
    if ctx["test"]:
        argv += ["--out", fp, "--step0-root", ctx["r7"] / "step0", "--capture-manifest", ctx["capture_manifest"],
                 "--kappa-decision", ctx["r7"] / "step0" / "kappa_decision.json"]
    run_step(f"fixedpoint_{cid}", argv, ctx["logdir"], ctx["py"])
    if not fp.is_file():
        raise Stop(4, f"{cid}: the fixed point did not write {fp}")
    if cid in ctx.get("recheck", {}):
        _check_bound(f"{cid} fixed point", read_json(fp).get("capture_identity"), ctx["recheck"][cid])
    return fp


def s1_solve(ctx, cid, fp_path: Path):
    fp = read_json(fp_path)
    cmd = fp.get("resolve_command")
    if not cmd:
        return None
    toks = shlex.split(cmd)
    if toks[:3] != ["python", "benchmark/ppl/prc_r7_solve.py", "solve"] or "--fixed-point-record" not in toks or \
            Path(toks[toks.index("--fixed-point-record") + 1]).resolve() != fp_path.resolve() or "--out-dir" not in toks:
        raise Stop(3, f"{fp_path}: unexpected resolve command: {cmd}")
    if Path(toks[toks.index("--manifest") + 1]).resolve() != Path(ctx["capture_manifest"]).resolve():
        raise Stop(3, f"{fp_path}: the resolve command uses another capture manifest")
    if cid in ctx.get("recheck", {}) and ("--identity" not in toks or Path(toks[toks.index("--identity") + 1]).resolve()
                                          != Path(ctx["recheck"][cid]["path"]).resolve()):
        raise Stop(3, f"{fp_path}: the resolve command does not read the validated re-check record")
    out = Path(toks[toks.index("--out-dir") + 1])
    if _exists_complete(out, "solve_summary.json"):
        log(f"{cid}: s1 re-solve exists, reused: {out}")
    else:
        run_step(f"s1solve_{cid}", ["@PY"] + toks[1:], ctx["logdir"], ctx["py"])   # verbatim, once
    sp = out / "solve_summary.json"
    sm = read_json(sp)
    want = {a: r["s1"] for a, r in fp["arms"].items() if r.get("status") == "ok"}
    got = {}
    for r in sm["arms"].values():
        got.setdefault(r["arm"], []).append(float(r["target_scale"]))
    if sm.get("cell") != cid or set(got) != set(want) or \
            any(len(got[a]) != 1 or abs(got[a][0] - float(s)) > 5e-9 for a, s in want.items()):
        raise Stop(3, f"{sp}: not the re-solve at the fixed point's s1 {want} (found {got})")
    if cid in ctx.get("recheck", {}):
        _check_bound(f"{cid} s1 re-solve", sm.get("identity"), ctx["recheck"][cid])
    return sp


def build_screen(ctx, s1: dict, fps: dict, kd: dict):
    man = ctx["screen_manifest"]
    if man.exists():
        m = read_json(man)
        spec = read_json(m["candidate_spec"]["path"])
        have = {c: (v.get("solve_summaries"), v.get("fixed_point_sha256")) for c, v in spec["cells"].items()}
        want = {c: ([str(s1[c])], sha256(fps[c])) for c in s1}
        if have != want or (m.get("kappa") or {}).get("sha256") != kd["sha256"]:
            raise Stop(3, f"{man} exists but was built from other solves / fixed points / kappa decision; refusing to "
                          "overwrite (it may already be frozen for a submitted screen)")
        log(f"screen manifest exists and matches these inputs, reused: {man}")
        return m
    argv = ["@PY", "benchmark/ppl/kbands/build_prc_screen_r7_20260928.py"]
    for c in s1:
        argv += ["--from-solve", f"{c}={s1[c]}", "--fixed-point", f"{c}={fps[c]}"]
    argv += ["--kappa-decision", kd["path"]]
    if ctx["accept_kappa_review"]:
        argv.append("--accept-kappa-review")
    if ctx["test"]:
        argv += ["--out", man, "--candidates-out", ctx["candidates"], "--step0-root", ctx["r7"] / "step0",
                 "--capture-manifest", ctx["capture_manifest"]]
        if ctx["skip_sweep"]:
            argv.append("--skip-sweep")
    run_step("screen_build", argv, ctx["logdir"], ctx["py"])
    return read_json(man)


def launcher_dry_run(ctx, m) -> dict:
    out = {}
    for t in m["tasks"]:
        if not t["enabled"]:
            continue
        env = dict(os.environ, R7SCR_DRY_RUN="1", R7SCR_TASK=str(t["index"]), R7SCR_MANIFEST=str(ctx["screen_manifest"]))
        r = subprocess.run(["bash", SCREEN_SBATCH], cwd=REPO, env=env, capture_output=True, text=True)
        ok = r.returncode == 0 and "DRY RUN PASS" in r.stdout
        (ctx["logdir"] / f"{time.strftime('%Y%m%dT%H%M%S')}_launcher_dry_{t['id']}.log").write_text(r.stdout + r.stderr)
        if not ok:
            raise Stop(4, f"screen launcher dry run FAILED for task {t['index']} ({t['id']}):\n"
                          f"{(r.stderr or r.stdout)[-1500:]}")
        out[t["id"]] = "DRY RUN PASS"
    return out


# ---------------------------------------------------------------------------------------------
# summary + submit command
# ---------------------------------------------------------------------------------------------
def _pct(x):
    return f"{100.0 * (x - 1.0):+.3f}%" if isinstance(x, (int, float)) else "n/a"


def cell_summary(cid, s0p, fpp, s1p, mcell) -> dict:
    s0 = read_json(s0p) if s0p else None
    fp = read_json(fpp) if fpp else None
    s1 = read_json(s1p) if s1p else None
    arms = {}
    for stage, sm in (("s0", s0), ("s1", s1)):
        for tname, r in ((sm or {}).get("arms") or {}).items():
            a = arms.setdefault(r["arm"], {})
            a[stage] = {"table": tname, "target_scale": r["target_scale"], "pred_true": r["pred_true_vs_inc"],
                        "eligible": r["eligible"], "heldout_cost_vs_inc": r["heldout_vs_inc"]["cost_over_ref"]}
    for arm, r in ((fp or {}).get("arms") or {}).items():
        arms.setdefault(arm, {})["fixed_point"] = {"status": r.get("status"), "s1": r.get("s1"),
                                                   "C_replay_s0_over_P": r.get("C_replay_s0_over_P")}
    for arm in (s0 or {}).get("omitted_arms") or {}:
        arms.setdefault(arm, {})["omitted_at_s0"] = s0["omitted_arms"][arm]
    out = {"cell": cid, "kappa_branch": (s0 or {}).get("kappa_branch"),
           "kappa_pins": {"kappa_lin": (s0 or {}).get("kappa_lin"), "kappa_att": (s0 or {}).get("kappa_att"),
                          "currency_ratio": (s0 or {}).get("kappa_currency_ratio")},
           "mde_nats": (s0 or {}).get("mde_nats"), "s0": (fp or {}).get("s0"), "P": (fp or {}).get("P"),
           "U": (fp or {}).get("U"), "arms": arms, "screen": None}
    if mcell is not None:
        scr = {"enabled": mcell["enabled"], "disabled_reason": mcell.get("disabled_reason"),
               "primary": mcell.get("primary"), "secondaries": mcell.get("secondaries"),
               "refused": mcell.get("refused"), "flags": mcell.get("flags"), "arms": {}}
        for n, a in (mcell.get("arms") or {}).items():
            if n in ("INC", "PARENT"):
                continue
            rp = a.get("realized_length_prediction") or {}
            scr["arms"][n] = {"role": a.get("role"), "pred_true": a.get("pred_true_nats"), "mde": a.get("mde_nats"),
                              "target_scale": a.get("target_scale"),
                              "C_replay_s1_over_P": (a.get("replay_at_s1") or {}).get("C_replay_s1_over_P"),
                              "heldout_cost_vs_inc": a.get("predicted_cost_ratio"),
                              "new_rung_buckets": {k: {"new": b["new_rungs"], "pred_share": b["pred_share"]}
                                                   for k, b in (rp.get("buckets") or {}).items()}}
        out["screen"] = scr
    return out


def print_summary(kd, cells_out):
    log("=" * 100)
    log(f"KAPPA: decision {kd['decision']} (currency ratio {kd['ratio']}); kappa_lin_bs {kd['kappa_lin_bs']} +- "
        f"{kd['se_kappa_lin_bs']} vs band {kd['band']}; flags {kd['flags']}; review {kd['requires_user_review']}")
    for c in cells_out:
        log("-" * 100)
        log(f"{c['cell']}: kappa branch {c['kappa_branch']} pins {c['kappa_pins']}  MDE {c['mde_nats']} nats  "
            f"s0 {c['s0']}  P {c['P']}  U {c['U']}")
        for arm, a in c["arms"].items():
            parts = [f"  arm {arm:2s}"]
            if "omitted_at_s0" in a:
                parts.append(f"omitted ({a['omitted_at_s0']})")
            if "s0" in a:
                parts.append(f"s0 pred_true {a['s0']['pred_true']:+.5f} ({'<=' if a['s0']['eligible'] else '> '} -MDE)")
            if "fixed_point" in a:
                f = a["fixed_point"]
                parts.append(f"fixed point {f['status']} s1={f['s1']} C(s0)/P {_pct(f['C_replay_s0_over_P'])}")
            if "s1" in a:
                parts.append(f"s1 pred_true {a['s1']['pred_true']:+.5f} -> "
                             f"{'ELIGIBLE' if a['s1']['eligible'] else 'REFUSED (pred_true > -MDE)'}")
            log("  |  ".join(parts))
        s = c["screen"]
        if s is None:
            log(f"  SCREEN: not built for this cell (fixed point refused every arm, or no s1 solve)")
            continue
        if not s["enabled"]:
            log(f"  SCREEN: DISABLED - {s['disabled_reason']}; refused {s.get('refused')}")
            continue
        log(f"  SCREEN: ENABLED  primary {s['primary']}  secondaries {s['secondaries']}  refused {s.get('refused')}")
        for n, a in s["arms"].items():
            nr = a["new_rung_buckets"]
            share = ", ".join(f"{k}:{'/'.join(map(str, b['new']))}@" +
                              "/".join(f"{float(x):.3f}" for x in b["pred_share"].values()) for k, b in nr.items())
            log(f"    {n:2s} ({a['role']}): pred_true {a['pred_true']:+.5f} vs -MDE {-a['mde']:+.4f}; s1 "
                f"{a['target_scale']}; predicted cost vs parent (replay C(s1)/P) {_pct(a['C_replay_s1_over_P'])}; "
                f"held-out cost vs INC {_pct(a['heldout_cost_vs_inc'])}; new rungs {share or 'none'}")
        if s.get("flags"):
            log(f"    flags: {s['flags']}")


def screen_command(m, *, no_slurm, assume_gpus_used=0) -> dict:
    enabled = [t["index"] for t in m["tasks"] if t["enabled"]]
    if not enabled:
        return {"enabled": [], "command": None}
    if no_slurm:
        rows = [("assumed", "RUNNING", int(assume_gpus_used))] if assume_gpus_used else []
    else:
        rows = queued_jobs()
    active = [(j, s, g) for j, s, g in rows if g > 0 and s not in ("COMPLETED", "CANCELLED", "FAILED")]
    used = sum(g for _j, _s, g in active)
    head = GPU_CAP - used
    ids = ",".join(str(i) for i in enabled) if len(enabled) > 1 and enabled != list(range(enabled[0], enabled[-1] + 1)) \
        else (f"{enabled[0]}-{enabled[-1]}" if len(enabled) > 1 else str(enabled[0]))
    dep = ""
    if head >= 1:
        k = min(len(enabled), head)
        note = f"{used} GPU(s) of ours queued/running now -> throttle %{k} keeps the total <= {GPU_CAP}"
    else:
        k = min(len(enabled), GPU_CAP)
        jobs = sorted({j.split("_")[0] for j, _s, _g in active})
        dep = f" --dependency=afterany:{':'.join(jobs)}"
        note = (f"{used} GPU(s) of ours already queued/running (cap {GPU_CAP}): the screens wait for them "
                f"(afterany) and then run %{k}; do not queue other GPU jobs meanwhile")
    cmd = f"cd {REPO} && sbatch --parsable --array={ids}%{k}{dep} {SCREEN_SBATCH}"
    return {"enabled": enabled, "throttle": k, "gpus_in_use": used, "note": note, "command": cmd,
            "jobs_counted": [j for j, _s, _g in active]}


def _main(argv, outer: dict) -> int:
    # allow_abbrev=False: a prefix such as '--accept' must never resolve to a user-review override
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--cells", default=",".join(CELLS), help="subset of 30B_t32,30B_t40,30B_t48 (dropping one = user decision)")
    ap.add_argument("--accept-kappa-review", action="store_true",
                    help="continue past a kappa decision flagged requires_user_review (ONLY with the user's OK)")
    ap.add_argument("--accept-identity-recheck", action="store_true",
                    help="accept a capture whose identity failed ONLY the buggy pinned check (7), through its validated "
                         "identity_check7fix.json (user-approved 2026-09-28; PRC_R7_CAPTURE_IDENTITY_NOTE_20260928.md)")
    ap.add_argument("--python", default=ENV_PY)
    # CPU-test overrides (end-to-end chain test on scratch fixtures); a real run passes none of them.
    for f in ("--r7-root", "--capture-manifest", "--screen-manifest", "--candidates-out"):
        ap.add_argument(f, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--no-slurm", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--assume-gpus-used", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--skip-sweep", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--skip-launcher-dry-run", action="store_true", help=argparse.SUPPRESS)
    try:
        a = ap.parse_args(argv)
    except SystemExit as exc:  # argparse: 0 = --help; 2 = usage error, which would collide with 'not ready yet'
        if exc.code in (0, None):
            raise
        log("invalid command line (argparse usage error above); exit 4, not argparse's 2 = the chain's "
            "'not ready yet, re-run later' code")
        return 4
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    if not cells or any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        log(f"--cells must be a subset of {CELLS}")
        return 4
    overrides = [a.r7_root, a.capture_manifest, a.screen_manifest, a.candidates_out]
    test = any(x is not None for x in overrides)
    if test:
        if not all(x is not None for x in overrides):
            log("test mode needs --r7-root, --capture-manifest, --screen-manifest and --candidates-out together")
            return 4
        for x in overrides:
            rp = str(Path(x).resolve())
            if rp.startswith(str(R7_ROOT)) or rp.startswith(str(REPO)):
                log(f"test-mode path {x} must be outside the Turbo round-7 root and the repo")
                return 4
    elif a.no_slurm or a.skip_sweep or a.skip_launcher_dry_run or a.assume_gpus_used:
        log("--no-slurm / --skip-sweep / --skip-launcher-dry-run / --assume-gpus-used are test-mode options only")
        return 4
    r7 = Path(a.r7_root) if test else R7_ROOT
    outer["r7"] = r7
    ctx = outer
    ctx.update({"test": test, "r7": r7, "capture_manifest": Path(a.capture_manifest) if test else CAPTURE_MANIFEST,
           "screen_manifest": Path(a.screen_manifest) if test else SCREEN_MANIFEST,
           "candidates": Path(a.candidates_out) if test else CANDIDATES, "py": a.python,
           "accept_kappa_review": a.accept_kappa_review, "skip_sweep": a.skip_sweep,
           "logdir": r7 / "stage2_cpu" / "logs"})
    started = datetime.datetime.now(datetime.timezone.utc)
    record = {"schema": "prc-r7-stage2-cpu-v1", "started_utc": started.isoformat(), "cells": cells, "test_mode": test,
              "accept_kappa_review": a.accept_kappa_review, "accept_identity_recheck": a.accept_identity_recheck,
              "flags": []}
    ctx["recheck"] = {}
    lock_f = None
    try:
        log(f"round-7 stage-2 CPU chain; cells {cells}; R7 root {r7}{' (TEST MODE)' if test else ''}")
        record["hash_locks_before"] = verify_hash_locks()
        log(f"hash locks ok: {record['hash_locks_before']}")
        ctx["caps"] = capture_cells(ctx["capture_manifest"])
        if a.accept_identity_recheck:
            for cid in sorted(set(cells) | set(KAPPA_CELLS)):
                c = ctx["caps"].get(cid)
                if c is not None and not original_identity_ok(c):
                    ctx["recheck"][cid] = validate_recheck(ctx, cid)
                    log(f"{cid}: identity accepted through the validated re-check record "
                        f"{ctx['recheck'][cid]['path']} (sha {ctx['recheck'][cid]['sha256'][:12]})")
            record["identity_recheck"] = {"approval": RECHECK_APPROVAL, "cells": ctx["recheck"]}
            if ctx["recheck"]:
                flag(record["flags"], f"--accept-identity-recheck used for {sorted(ctx['recheck'])} (user-approved "
                                      "2026-09-28; deviation from 'identity fail = stop' disclosed in "
                                      "PRC_R7_CAPTURE_IDENTITY_NOTE_20260928.md)")
        record["jobs"] = check_jobs(a.no_slurm, record["flags"], accept_failed3=set(ctx["recheck"]))
        log(f"GPU jobs: {record['jobs']}")
        record["outputs"] = check_outputs(r7, ctx["capture_manifest"], cells, record["flags"], recheck=ctx["recheck"])
        (r7 / "stage2_cpu").mkdir(parents=True, exist_ok=True)
        lock_f = open(r7 / "stage2_cpu" / ".lock", "a")
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Stop(3, "another stage-2 chain holds the lock; wait for it")
        except OSError as exc:
            log(f"warning: could not lock ({exc}); every tool still writes exclusively")
        kd = kappa_stage(ctx)
        record["kappa"] = kd
        log(f"kappa: {kd['decision']} (ratio {kd['ratio']}; kappa_lin_bs {kd['kappa_lin_bs']} +- "
            f"{kd['se_kappa_lin_bs']}; band {kd['band']}; flags {kd['flags']})")
        s0p, fpp, s1p = {}, {}, {}
        for cid in cells:
            s0p[cid] = s0_solve(ctx, cid)
            fpp[cid] = fixed_point(ctx, cid, s0p[cid])
            s1p[cid] = s1_solve(ctx, cid, fpp[cid])
            if s1p[cid] is None:
                log(f"{cid}: the fixed point REFUSED every arm -> cell omitted from the screen (registered)")
        s1 = {c: p for c, p in s1p.items() if p is not None}
        m = build_screen(ctx, s1, fpp, kd) if s1 else None
        cells_out = [cell_summary(c, s0p[c], fpp[c], s1p[c], (m or {}).get("cells", {}).get(c) if m else None)
                     for c in cells]
        record["cells_summary"] = cells_out
        print_summary(kd, cells_out)
        if record["flags"]:
            log(f"chain flags: {record['flags']}")
        log("=" * 100)
        if m is None:
            record["result"] = "every cell refused at the fixed point: no screen manifest, no screen"
            log("EVERY CELL REFUSED (the fixed point refused every arm of every cell): no screen manifest, no screen "
                "command. Round 7 adds no candidate.")
            return 0
        record["screen_manifest"] = {"path": str(ctx["screen_manifest"]), "sha256": sha256(ctx["screen_manifest"]),
                                     "enabled_tasks": m.get("enabled_tasks"), "flags": m.get("flags")}
        if not m.get("enabled_tasks"):
            record["result"] = "every cell refused: the screen manifest enables no task"
            log(f"EVERY CELL REFUSED (screen manifest {ctx['screen_manifest']} enables no task): no screen command. "
                "Round 7 adds no candidate.")
            return 0
        if not a.skip_launcher_dry_run:
            record["launcher_dry_run"] = launcher_dry_run(ctx, m)
            log(f"launcher dry run: {record['launcher_dry_run']}")
        sc = screen_command(m, no_slurm=a.no_slurm, assume_gpus_used=a.assume_gpus_used)
        record["screen_command"] = sc
        log(f"screen manifest {ctx['screen_manifest']} (sha {record['screen_manifest']['sha256'][:12]}); enabled tasks "
            f"{sc['enabled']} ({', '.join(t['id'] for t in m['tasks'] if t['enabled'])})")
        log(sc["note"])
        log("SUBMIT (not executed by this script; one GPU per enabled cell):")
        print(f"  {sc['command']}", flush=True)
        log("then record the job id in a NEW benchmark/ppl/kbands/prc_screen_r7_20260928_submission.json")
        record["result"] = "screen command printed"
        return 0
    except Stop as st:
        record["result"] = f"STOPPED (exit {st.code}): {st}"
        log(f"STOP (exit {st.code}): {st}")
        return st.code
    finally:
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if lock_f is not None:
            lock_f.close()
        ctx["record"] = record


def main(argv=None) -> int:
    ctx = {}
    rc = _main(argv, ctx)
    record = ctx.get("record") or {}
    if record:
        try:
            record["hash_locks_after"] = verify_hash_locks()
            log(f"hash locks still ok after the chain: {record['hash_locks_after']}")
        except Stop as st:
            record["hash_locks_after"] = str(st)
            log(f"STOP (exit 3): {st}")
            rc = 3
        record["exit_code"] = rc
        d = ctx.get("r7")
        if d is not None and (Path(d) / "stage2_cpu").is_dir():
            p = Path(d) / "stage2_cpu" / f"stage2_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.json"
            with p.open("x") as f:
                json.dump(record, f, indent=1, default=str)
            log(f"chain record: {p}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
