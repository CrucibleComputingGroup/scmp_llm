"""Idempotent launcher: bring the FULL paper grid to the current best method.

The paper needs every `mp_best` cell (4 models x t{32,40,48,64,96,128}) measured
under the deployed method, with a trace for the energy model. This script
inventories what already exists and submits ONLY what is missing, so it is safe
to re-run as often as you like -- which matters because the account's CPU cap
(GrpTRES cpu=160, routinely ~153 in use by the lab) means a 150-GPU-hour wave
trickles over days rather than landing at once.

WHAT "CURRENT BEST METHOD" MEANS, and why the arms are what they are:
  parent  the cell's mp_best per-row config, re-run HERE. Not optional: every
          comparison is S1->S2, and the cost-adjustment needs this cell's own
          traced parent cost. Comparing against an archived number folds in
          every environment difference since that run.
  prc     per-(row, chunk) allocation. The core lever; wins broadly.
  prcqk   prc + the qk operand rebalance. qk is PER-CELL, not universal -- it
          helps 4B / llama8B t48 / 30B but REGRESSES 14B at both budgets and
          llama8B t32 -- so both prc and prcqk are measured and the lower wins.
  +grid   SC_RNG_GRID=pow2, matching the enable-grid to the stream length.
          Runtime-free. Strong on 30B, neutral-to-slightly-negative elsewhere,
          so again measured rather than assumed.

Selection is done by build_mp_best_after_hpca_2.py, which takes the LOWEST
full-protocol PPL per cell and records which arm produced it.

  python -m benchmark.ppl.kbands.recreate_all            # dry run, show plan
  python -m benchmark.ppl.kbands.recreate_all --submit   # submit missing work
  python -m benchmark.ppl.kbands.recreate_all --submit --max-inflight 8
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = Path("/home/allenjin/Projects/hpca_results/llm")
MPB = RES / "ppl" / "mp_best" / "configs"
QK_ARCH = RES / "ppl" / "mp_best_after_hpca" / "qk_scales"
TURBO = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801")
PRC, QKT = TURBO / "prc", TURBO / "qk"
LOGS = Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/"
            "hpca/logs/_kbands")
AWQ_TRACES = RES / "ppl" / "mp_best" / "awq_traces" / "mp"

MODELS = ["4B", "llama8B", "14B", "30B"]
TARGETS = [32, 40, 48, 64, 96, 128]
ARMS = ["parent", "prc", "qk", "prcqk"]


def read_ppl(prefix: str, name: str):
    """Best [RESULT] PPL for this arm, or None."""
    import re as _re
    for f in sorted(LOGS.glob(f"{prefix}_{name}_*.out"),
                    key=lambda q: q.stat().st_mtime, reverse=True):
        if not _re.fullmatch(rf"{prefix}_{_re.escape(name)}_\d+\.out", f.name):
            continue
        for line in reversed(f.read_text(errors="ignore").splitlines()):
            if "[RESULT]" in line and "metric=ppl" in line:
                kv = dict(x.split("=", 1) for x in line.split() if "=" in x)
                try:
                    return float(kv["value"])
                except Exception:                            # noqa: BLE001
                    pass
    return None


def have_log(prefix: str, name: str) -> bool:
    """A completed [RESULT] line for this arm. Anything less is not a result."""
    for f in LOGS.glob(f"{prefix}_{name}_*.out"):
        if not re.fullmatch(rf"{prefix}_{re.escape(name)}_\d+\.out", f.name):
            continue
        txt = f.read_text(errors="ignore")
        if "[RESULT]" in txt and "metric=ppl" in txt:
            return True
    return False


def queued() -> set:
    try:
        out = subprocess.run(["squeue", "-h", "-u", "allenjin", "-o", "%j"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:                                        # noqa: BLE001
        return set()
    return {l.strip() for l in out.splitlines() if l.strip()}


def sbatch(script: str, env: dict, name: str, dep: str | None = None) -> str | None:
    cmd = ["sbatch", "--parsable", f"--job-name={name}"]
    if dep:
        cmd.append(f"--dependency=afterok:{dep}")
    cmd.append(str(HERE / script))
    import os
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    if r.returncode != 0:
        print(f"    ! sbatch failed for {name}: {r.stderr.strip()[:160]}")
        return None
    return r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--max-inflight", type=int, default=10,
                    help="stop submitting once this many of our jobs are queued "
                         "or running; the account CPU cap makes over-submitting "
                         "pointless and it starves the rest of the lab")
    ap.add_argument("--grid-policy", choices=["winner", "all", "none"],
                    default="winner",
                    help="Which arms get the SC_RNG_GRID=pow2 variant. 'winner' "
                         "(default) runs it only on the cell's currently-best "
                         "measured arm, so grid FOLLOWS measurement instead of "
                         "doubling every cell up front -- grid pays on 30B and "
                         "is neutral-to-negative elsewhere, so running it 8x per "
                         "cell burns ~2x the wave for nothing.")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--targets", default=",".join(str(t) for t in TARGETS))
    a = ap.parse_args()
    models = a.models.split(",")
    targets = [int(t) for t in a.targets.split(",")]

    live = queued()
    inflight = len(live)
    plan, submitted = [], 0
    print(f"[recreate] {inflight} job(s) already queued/running; "
          f"cap {a.max_inflight}\n")

    for m in models:
        for t in targets:
            if not (MPB / m / f"target{t}").is_dir():
                continue
            tbl = PRC / f"{m}_t{t}_v7_prc.json"
            qk = (QK_ARCH / f"{m}_t{t}_qk_alpha1.0.json")
            if not qk.is_file():
                qk = QKT / f"{m}_t{t}_qk_alpha1.0.json"
            trace = AWQ_TRACES / f"{m}_t{t}_awq_trace.json"

            # 1. prc table (pitfall 1: budgets MUST come from the parent TRACE;
            #    a calibration sample overestimates the parent by 9-23% and the
            #    child then silently buys its win).
            calib_job = None
            cname = f"kb_prccalib_{m}_t{t}"
            if not tbl.is_file() and cname not in live:
                if not trace.is_file():
                    plan.append(f"  SKIP {m} t{t}: no AWQ parent trace {trace.name}")
                    continue
                plan.append(f"  CALIB prc  {m} t{t}")
                if a.submit and inflight < a.max_inflight:
                    calib_job = sbatch("run_kband_tool.sbatch",
                                       {"KB_JOB": "prccalib", "KB_MODEL": m,
                                        "KB_TARGET": t, "KB_FRONTEND": "awq",
                                        "KB_SUFFIX": "_v7",
                                        "KB_PARENT_TRACE": str(trace)}, cname)
                    if calib_job:
                        inflight += 1; submitted += 1

            # 2. qk scales
            qname = f"kb_qkcalib_{m}_t{t}"
            qk_job = None
            if not qk.is_file() and qname not in live:
                plan.append(f"  CALIB qk   {m} t{t}")
                if a.submit and inflight < a.max_inflight:
                    qk_job = sbatch("run_kband_tool.sbatch",
                                    {"KB_JOB": "qkcalib", "KB_MODEL": m,
                                     "KB_TARGET": t, "KB_ALPHA": "1.0"}, qname)
                    if qk_job:
                        inflight += 1; submitted += 1

            # 3. PPL arms (+ grid variant of each)
            # Which arm (if any) currently wins this cell -- grid follows it.
            measured = {arm: read_ppl("prcppl", f"{m}_t{t}_{arm}") for arm in ARMS}
            measured = {k: v for k, v in measured.items() if v is not None}
            best_arm = min(measured, key=measured.get) if measured else None

            for arm in ARMS:
                needs_tbl = arm in ("prc", "prcqk")
                needs_qk = arm in ("qk", "prcqk")
                grids = (False,)
                if a.grid_policy == "all":
                    grids = (False, True)
                elif a.grid_policy == "winner" and arm == best_arm:
                    grids = (False, True)
                for grid in grids:
                    pfx = "grid" if grid else "prcppl"
                    jname = (f"grid_{m}_t{t}_{arm}" if grid
                             else f"prcppl_{m}_t{t}_{arm}")
                    if have_log(pfx, f"{m}_t{t}_{arm}") or jname in live:
                        continue
                    if needs_tbl and not (tbl.is_file() or calib_job):
                        continue
                    if needs_qk and not (qk.is_file() or qk_job):
                        continue
                    plan.append(f"  PPL   {m} t{t} {arm}{'+grid' if grid else ''}")
                    if a.submit and inflight < a.max_inflight:
                        dep = calib_job if needs_tbl else (qk_job if needs_qk else None)
                        env = {"KB_MODEL": m, "KB_TARGET": t, "KB_ARM": arm}
                        if grid:
                            env["SC_RNG_GRID"] = "pow2"
                            env["KB_TAGSUFFIX"] = "grid"
                        j = sbatch("run_prc_ppl.sbatch", env, jname, dep)
                        if j:
                            inflight += 1; submitted += 1

    for line in plan:
        print(line)
    print(f"\n[recreate] {len(plan)} unit(s) of work outstanding; "
          f"submitted {submitted} this pass"
          f"{'' if a.submit else '  (dry run -- pass --submit)'}")
    if not a.submit:
        print("[recreate] re-run with --submit repeatedly as capacity frees; "
              "it only ever submits what is missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
