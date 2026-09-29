"""Round-7 SCREEN -> CONFIRM -> FULL-TEST driver (2026-09-28) for round-7 candidate tables.

One GPU job per cell (30B t32 / t40 / t48; optional 4B t40 / t64 when enabled in the manifest),
one model load for screen + confirm, then the launcher runs the full-protocol test of the
PRE-REGISTERED PRIMARY arm only, and ``finalize`` verifies it on CPU.

  screen   INC (bit identity vs a stored reference where one exists; otherwise a held-out probe)
           + PARENT (exact P on the same windows, or a hash-verified step-0 reference) + the primary
           + <= 2 secondaries on 16 paired TRAIN windows. Every arm trace is audited against the
           arm's OWN table (runtime resolver + JSON mirror; no length > 128; sc_prec 8, halve on,
           rng_levels 128; SC (op, block) set == the INT mask's), equal MACs vs INC, first-window
           profile replay exact, and - REGISTERED realized-length audit (capture manifest
           preregistered_rules.realized_length_audit) - every new 112/128 attention rung's realized MAC
           share within +-0.10 absolute of the solve's hold-window prediction (att_mac_hist_hold,
           re-derived here from the sha-locked solve summary). Every screened trace is audited BEFORE any
           candidate NLL is released (the evaluator's per-window NLL log lines are masked meanwhile); any
           screened arm's audit failure stops the cell with a failure record and no decision.
           Cost gate |C/P - 1| <= 1% on these held-out windows.
  confirm  only the primary, only if the screen gate passes: INC and primary on 32 fresh disjoint
           TRAIN windows; full test iff paired z < -2 (and audits pass).
  finalize full-protocol header checks + realized-length audit of the TEST trace + the run
           environment read back from run_prc_ppl.sbatch's own log (hybrid mask, FRONTEND=awq, masks
           64, deployed RNG grid, no qk scales); the result enters best(all) whatever its sign when
           every check passed and |C_test/P_test - 1| <= 1.5% (flagged beyond 1%), with provenance.
Secondaries are information only: never confirmed, never full-tested (no selection inflation).

  python benchmark/ppl/prc_screen_r7_20260928.py preflight --manifest M --task K [--deep]
  python benchmark/ppl/prc_screen_r7_20260928.py run       --manifest M --task K
  python benchmark/ppl/prc_screen_r7_20260928.py test-args --manifest M --task K
  python benchmark/ppl/prc_screen_r7_20260928.py finalize  --manifest M --task K --trace T --run-log L
  python benchmark/ppl/prc_screen_r7_20260928.py simulate  --manifest M --task K --out-root SCRATCH
Units: HALVED stream lengths/costs (nominal = 2x); max 128.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.ppl import prc_eval_r7_20260928 as E  # noqa: E402
from benchmark.ppl import prc_r6_attn_diag_arms as R  # noqa: E402

SCHEMA = "prc-screen-r7-v1"
TAG = "r7scr"
TEST_DIR = E.KB / "kbands_20260801/ppl"
EVAL_TOKENS = {"30B": 298862, "4B": 298862}
TEST_COST_FLAG = 0.01          # |C_test/P_test - 1| above this is flagged
BEST_ALL_TEST_COST_BOUND = 0.015   # prc_r7_prereg_20260928.BEST_ALL_TEST_COST_BOUND (asserted in finalize)

# ---- registered realized-length audit (prc_r7_capture_20260928.json preregistered_rules.realized_length_audit,
#      verbatim; the screen builder refuses to build if the capture manifest's text differs) --------------------
REALIZED_LENGTH_RULE = (
    "Before any screen NLL is read: GPU trace histograms per (op, bucket) resolved through get_levels; realized "
    "lengths within the table's bucket ladder (+128 escape), none above 128; every new rung (112/128) realized "
    "where the solve's hold-window att_mac_hist_hold predicts it, MAC share within +-0.10 absolute of that "
    "prediction; any failure stops the cell (resolver bug family).")
NEW_RUNG_TOL = 0.10            # absolute MAC-share tolerance per (attention bucket, new rung)
NEW_RUNGS = (112, 128)         # the only rungs the registered 'up' ladder policy can add (prc_r7_solve.up_rungs)
PREDICTION_SCHEMA = "prc-r7-new-rung-prediction-v1"
AUDIT_SCHEMA = "prc-r7-new-rung-audit-v1"
WITHHELD = "<withheld until every screened trace passes its audits>"


class _WithheldNLL(io.TextIOBase):
    """stdout filter while a CANDIDATE arm is evaluated: the evaluator (step-0-frozen, not editable) prints
    each window's NLL; the value is masked so no candidate NLL is visible before its trace passed the audits."""
    _PAT = re.compile(r"NLL=\S+")

    def __init__(self, stream):
        super().__init__()
        self._stream = stream

    def writable(self):
        return True

    def write(self, s):
        self._stream.write(self._PAT.sub("NLL=" + WITHHELD, s))
        return len(s)

    def flush(self):
        self._stream.flush()


def bucket_attention_ladders(table: dict, total_blocks: int, layer_buckets: int) -> dict:
    """{'op:l<q>': ascending ladder} exactly as the get_levels mirror resolves each attention block; every block of
    one bucket must resolve to the same ladder (a disagreement is the resolver-bug family and fails closed)."""
    lb = int(table.get("layer_buckets", 1) or 1)
    if lb != int(layer_buckets):
        raise E.AuditError(f"table layer_buckets {lb} != the cell's {layer_buckets}")
    out = {}
    for op in E.ATTN_OPS:
        for b in range(int(total_blocks)):
            key = f"{op}:l{E.bucket_index(b, total_blocks, layer_buckets)}"
            lad = sorted(int(v) for v in E.json_ladder(table, op, b, total_blocks))
            if out.setdefault(key, lad) != lad:
                raise E.AuditError(f"{key}: blocks of one bucket resolve to different ladders ({out[key]} vs {lad})")
    return out


def _new_rungs(arm_l: dict, inc_l: dict) -> dict:
    new = {k: sorted(set(v) - set(inc_l.get(k, []))) for k, v in arm_l.items()}
    return {k: v for k, v in new.items() if v}


def new_rung_prediction(rec: dict, arm_table: dict, inc_table: dict, *, total_blocks: int, layer_buckets: int,
                        source: dict) -> dict:
    """The registered prediction for one arm, from its prc_r7_solve.py solve-summary record: for every attention
    bucket whose ladder gains rungs vs INC (only 112/128 are registered), the predicted MAC share of each new rung
    = att_mac_hist_hold[bucket][L] / sum over the bucket (hold windows; escaped rows are inside the histogram at
    128, exactly as the runtime realizes them). Fails closed (AuditError) on a missing or inconsistent histogram."""
    hist = rec.get("att_mac_hist_hold")
    if not isinstance(hist, dict) or not hist:
        raise E.AuditError(f"{source.get('table_name')}: the solve record carries no att_mac_hist_hold")
    arm_l = bucket_attention_ladders(arm_table, total_blocks, layer_buckets)
    inc_l = bucket_attention_ladders(inc_table, total_blocks, layer_buckets)
    new = _new_rungs(arm_l, inc_l)
    unreg = {k: [L for L in v if L not in NEW_RUNGS] for k, v in new.items()}
    if any(unreg.values()):
        raise E.AuditError(f"new attention rungs outside the registered {NEW_RUNGS}: "
                           f"{ {k: v for k, v in unreg.items() if v} }")
    rec_lad = rec.get("ladders")
    if rec_lad is not None:
        bad = [k for k, v in rec_lad.items() if sorted(int(x) for x in v) != arm_l.get(k)]
        if bad or any(k not in rec_lad for k in new):
            raise E.AuditError(f"the solve record's ladders differ from the emitted table's resolved ladders at "
                               f"{bad or sorted(set(new) - set(rec_lad))}")
    buckets = {}
    for key, rungs in sorted(new.items()):
        h = hist.get(key)
        if not isinstance(h, dict):
            raise E.AuditError(f"{key}: no hold-window histogram for a bucket with new rungs {rungs}")
        h = {int(L): float(m) for L, m in h.items()}
        tot = sum(h.values())
        if not (tot > 0 and math.isfinite(tot)) or any(m < 0 or not math.isfinite(m) for m in h.values()):
            raise E.AuditError(f"{key}: hold-window histogram is empty or invalid")
        outside = sorted(set(h) - (set(arm_l[key]) | {E.CAP}))
        if outside:
            raise E.AuditError(f"{key}: predicted lengths {outside} are outside the arm ladder {arm_l[key]} (+128)")
        buckets[key] = {"arm_ladder": arm_l[key], "inc_ladder": inc_l[key], "new_rungs": rungs,
                        "pred_share": {str(L): h.get(L, 0.0) / tot for L in rungs},
                        "pred_share_all": {str(L): m / tot for L, m in sorted(h.items())}, "pred_macs": tot}
    return {"schema": PREDICTION_SCHEMA, "standin": False, "tolerance": NEW_RUNG_TOL, "rungs": list(NEW_RUNGS),
            "rule": REALIZED_LENGTH_RULE, "source": dict(source), "buckets": buckets}


def standin_prediction(reason: str) -> dict:
    return {"schema": PREDICTION_SCHEMA, "standin": True, "reason": reason, "tolerance": NEW_RUNG_TOL,
            "rungs": list(NEW_RUNGS), "rule": REALIZED_LENGTH_RULE}


def rederive_prediction(cell: dict, name: str, arm: dict, arm_table: dict, inc_table: dict) -> dict:
    """Re-derive an arm's registered prediction from its solve summary (sha-locked in the cell's hashes) and
    require it to equal the manifest copy exactly."""
    pred = arm.get("realized_length_prediction")
    if not isinstance(pred, dict) or pred.get("schema") != PREDICTION_SCHEMA or pred.get("standin"):
        raise E.AuditError(f"{name}: no registered (non-stand-in) realized-length prediction")
    src = pred.get("source") or {}
    sp = src.get("solve_summary")
    if not sp or sp not in cell["hashes"] or src.get("solve_summary_sha256") != cell["hashes"][sp] or \
            E.sha256_file(sp) != cell["hashes"][sp]:
        raise E.AuditError(f"{name}: prediction source {sp} is not a sha-locked input of this cell")
    rec = (E.read_json(sp).get("arms") or {}).get(src.get("table_name"))
    if rec is None:
        raise E.AuditError(f"{name}: {src.get('table_name')} is not an arm of {sp}")
    if Path(rec.get("table_path", "")).resolve() != Path(arm["table"]).resolve():
        raise E.AuditError(f"{name}: the prediction's solve record is for {rec.get('table_path')}, not {arm['table']}")
    fresh = new_rung_prediction(rec, arm_table, inc_table, total_blocks=cell["total_blocks"],
                                layer_buckets=cell["layer_buckets"], source=src)
    if json.loads(json.dumps(fresh)) != pred:
        raise E.AuditError(f"{name}: the manifest's realized-length prediction differs from its re-derivation")
    return pred


def new_rung_audit(prediction, agg, runtime_hist, arm_table: dict, inc_table: dict, *, total_blocks: int,
                   layer_buckets: int, binding: bool, stage: str) -> dict:
    """The registered new-rung check on one arm trace. ``agg`` = the trace's R.TraceAgg (bucket mapping mirror),
    ``runtime_hist`` = analyze_trace's histogram (runtime _bucket_index); both must agree on every new-rung
    bucket. For every (bucket, new rung): realized MAC share vs the prediction, |diff| <= 0.10. ``binding``
    False = descriptive (confirm / full test, or a dry-run stand-in manifest in simulation)."""
    rep = {"schema": AUDIT_SCHEMA, "stage": stage, "binding": bool(binding), "tolerance": NEW_RUNG_TOL,
           "rule": REALIZED_LENGTH_RULE, "buckets": {}, "failures": [], "notes": []}
    fail = rep["failures"]
    if not isinstance(prediction, dict) or prediction.get("schema") != PREDICTION_SCHEMA:
        fail.append("no registered realized-length prediction for this arm")
        prediction = None
    elif prediction.get("standin"):
        (fail if binding else rep["notes"]).append(
            f"stand-in prediction ({prediction.get('reason')}): the new-rung share check cannot be enforced")
        prediction = None
    elif prediction.get("tolerance") != NEW_RUNG_TOL or list(prediction.get("rungs") or []) != list(NEW_RUNGS):
        fail.append(f"prediction tolerance/rungs {prediction.get('tolerance')}/{prediction.get('rungs')} are not the "
                    f"registered {NEW_RUNG_TOL}/{list(NEW_RUNGS)}")
        prediction = None
    try:
        new = _new_rungs(bucket_attention_ladders(arm_table, total_blocks, layer_buckets),
                         bucket_attention_ladders(inc_table, total_blocks, layer_buckets))
    except E.AuditError as exc:
        fail.append(str(exc))
        new = {}
    unreg = {k: [L for L in v if L not in NEW_RUNGS] for k, v in new.items()}
    if any(unreg.values()):
        fail.append(f"new attention rungs outside the registered {NEW_RUNGS}: { {k: v for k, v in unreg.items() if v} }")
    pb = (prediction or {}).get("buckets") or {}
    if prediction is not None and {k: list(b.get("new_rungs") or []) for k, b in pb.items()} != new:
        fail.append(f"the prediction's new-rung buckets {sorted(pb)} differ from the arm-vs-INC tables' {sorted(new)}")
    for key, rungs in sorted(new.items()):
        op, q = key.split(":l")
        h = {int(L): int(m) for L, m in agg.bucket_hist.get(("attention", op, int(q)), {}).items() if m}
        if runtime_hist is not None:
            rt = {int(L): float(m) for L, m in (runtime_hist.get(key) or {}).items() if float(m)}
            if rt != {L: float(m) for L, m in h.items()}:
                fail.append(f"{key}: runtime-resolver histogram (get_levels bucket mapping) != trace aggregate")
        tot = sum(h.values())
        ent = {"new_rungs": rungs, "realized_macs": tot,
               "realized_share_all": {str(L): m / tot for L, m in sorted(h.items())} if tot else {}, "rungs": {}}
        if tot == 0:
            fail.append(f"{key}: no realized SC MACs in a bucket with new rungs {rungs}")
        pk = pb.get(key) or {}
        for L in rungs:
            real = h.get(L, 0) / tot if tot else None
            pred = pk.get("pred_share", {}).get(str(L))
            diff = abs(real - float(pred)) if (real is not None and pred is not None) else None
            ok = diff is not None and diff <= NEW_RUNG_TOL + 1e-12
            ent["rungs"][str(L)] = {"realized_share": real, "predicted_share": pred, "abs_diff": diff, "ok": ok}
            if prediction is not None and not ok:
                fail.append(f"{key}: new rung {L} realized MAC share {real} vs predicted {pred} "
                            f"(|diff| {diff} > {NEW_RUNG_TOL})")
        rep["buckets"][key] = ent
    rep["ok"] = not fail
    return rep


def parse_run_log(path, *, hybrid, wrapper) -> dict:
    """The full-test run environment as run_prc_ppl.sbatch itself printed it (the trace header cannot
    prove FRONTEND or the mask). Returns {"values": ..., "failed": [...]}."""
    import re
    text = Path(path).read_text(errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("[prc-ppl]")]

    def grab(pattern):
        vals = [m.group(1).strip() for ln in lines for m in [re.search(pattern, ln)] if m]
        return vals[-1] if vals else None
    vals = {"hybrid_mask": grab(r"\[prc-ppl\] hybrid mask: (.+)$"),
            "frontend": grab(r"\[prc-ppl\] frontend=(\S+)"),
            "ctx": grab(r"\bctx=(\d+)"), "ppl_max_tokens": grab(r"PPL_MAX_TOKENS=(\d+)"),
            "scramble_masks": grab(r"SC_SCRAMBLE_MASKS=(\d+)"), "hw_max_masks": grab(r"HW_MAX=(\S+)"),
            "rng_grid": grab(r"\[prc-ppl\] SC_RNG_GRID=(.+)$"),
            "attn_grid": grab(r"\[prc-ppl\] attn grid: (.+)$"),
            "config": grab(r"\[prc-ppl\] config=(.+)$"),
            "qk_scales": grab(r"\[prc-ppl\] qk scales: (.+)$")}
    want = {"hybrid_mask": str(hybrid), "frontend": "awq", "ctx": "2048", "ppl_max_tokens": "0",
            "scramble_masks": "64", "hw_max_masks": "64", "rng_grid": "<unset, deployed 128>",
            "config": str(wrapper)}
    failed = [f"run_log:{k}={vals[k]!r} (want {v!r})" for k, v in want.items() if vals[k] != v]
    ag = vals["attn_grid"] or ""
    if not all(f"{k}=<unset>" in ag for k in ("ATTN", "QK", "AV")):
        failed.append(f"run_log:attn_grid={ag!r} (want every attention grid unset)")
    if vals["qk_scales"] is not None:
        failed.append(f"run_log:qk_scales={vals['qk_scales']!r} (qk rebalance is forbidden)")
    return {"path": str(path), "sha256": E.sha256_file(path), "values": vals, "failed": failed}


def log(msg):
    print(f"[{TAG}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# manifest / verification
# ---------------------------------------------------------------------------
def load_manifest(path) -> dict:
    m = E.read_json(path)
    if m.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema {m.get('schema')!r} != {SCHEMA!r}")
    if m.get("max_total_gpus") != 4 or m.get("gpus_per_task") != 1:
        raise ValueError("manifest must record the four-GPU total and one GPU per task")
    E.check_protocol(m["protocol"])
    p = m["protocol"]
    if (p.get("screen_windows"), p.get("confirm_windows"), p.get("cost_gate_tolerance"), p.get("confirm_z")) != \
            (16, 32, E.COST_GATE_TOL, E.CONFIRM_Z):
        raise ValueError("screen/confirm protocol constants differ from the pre-registered values")
    return m


def verify_code_hashes(m) -> None:
    code = m.get("code_hashes") or {}
    if not code or set(code) != set(m.get("source_files", [])):
        raise ValueError("code hashes must cover exactly the manifest source_files")
    for p, h in code.items():
        if E.sha256_file(p) != h:
            raise ValueError(f"experiment source changed after freezing: {p}")


def task_cell(m, task_index):
    tasks = m["tasks"]
    if not 0 <= task_index < len(tasks) or tasks[task_index]["index"] != task_index:
        raise ValueError(f"no task {task_index}")
    t = tasks[task_index]
    return t, m["cells"][t["cell"]]


def test_trace_path(cell, kb_tbl) -> Path:
    return TEST_DIR / f"{cell['model']}_t{cell['target']}_prc_{kb_tbl}_trace.json"


def verify_cell(m: dict, cell: dict) -> dict:
    if not cell.get("enabled"):
        raise ValueError(f"cell {cell['id']} is disabled: {cell.get('disabled_reason')}")
    for p, h in cell["hashes"].items():
        if E.sha256_file(p) != h:
            raise ValueError(f"frozen input changed: {p}")
    reg_path, reg_sha = m["windows_registry"]["path"], m["windows_registry"]["sha256"]
    if E.sha256_file(reg_path) != reg_sha:
        raise ValueError("window registry changed")
    reg = E.read_json(reg_path)
    arms = cell["arms"]
    inc_w = E.read_json(arms["INC"]["wrapper"])
    inc_t = E.read_json(arms["INC"]["table"])
    loaded = {}
    for name, a in arms.items():
        w = E.read_json(a["wrapper"])
        if E.resolve_table_path(a["wrapper"]).resolve() != Path(a["table"]).resolve():
            raise ValueError(f"{name}: wrapper does not point at the frozen table")
        E.check_wrapper_matches_inc(w, inc_w, name)
        t = E.read_json(a["table"])
        if t["model_path"] != cell["model_path"]:
            raise ValueError(f"{name}: model mismatch")
        loaded[name] = (w, t)
    primary, secondaries = cell["primary"], list(cell["secondaries"])
    cands = [primary] + secondaries
    if len(cands) > 3 or len(set(cands)) != len(cands) or any(c in ("INC", "PARENT") for c in cands):
        raise ValueError("a cell screens one primary and at most two secondaries")
    if arms[primary]["role"] != "primary" or any(arms[s]["role"] != "secondary" for s in secondaries):
        raise ValueError("arm roles disagree with primary/secondaries")
    sigs = set()
    for name in cands:
        a = arms[name]
        rep = E.validate_lineage(inc_t, loaded[name][1], total_blocks=cell["total_blocks"],
                                 allow_psl_change=bool(a.get("allow_psl_change")),
                                 allow_layer_rekey=bool(a.get("allow_layer_rekey")))
        if not rep["ok"]:
            raise ValueError(f"{name}: lineage vs INC failed: {rep['failures'][:5]}")
        if "candidate is allocation-identical to INC" in rep["notes"]:
            raise ValueError(f"{name}: allocation-identical to INC")
        if not E.pred_passes_mde(a["pred_true_nats"], a["mde_nats"]):
            raise ValueError(f"{name}: pred_true {a['pred_true_nats']} does not clear -MDE {a['mde_nats']}")
        sig = E.allocation_signature(loaded[name][1])
        if sig in sigs:
            raise ValueError(f"{name}: duplicate of another screened arm")
        sigs.add(sig)
        # registered realized-length audit: the prediction must re-derive from the sha-locked solve summary
        pred = a.get("realized_length_prediction")
        if isinstance(pred, dict) and pred.get("standin"):
            if not m.get("dry_run"):
                raise ValueError(f"{name}: stand-in realized-length prediction in a non-dry-run manifest")
        else:
            try:
                rederive_prediction(cell, name, a, loaded[name][1], inc_t)
            except (E.AuditError, KeyError, TypeError, FileNotFoundError) as exc:
                raise ValueError(f"{name}: realized-length prediction refused: {exc}")
    win = cell["windows"]
    screen, confirm = [int(s) for s in win["screen"]], [int(s) for s in win["confirm"]]
    for lst, n in ((screen, 16), (confirm, 32)):
        if len(lst) != n or len(set(lst)) != n or any(s % E.CTX for s in lst):
            raise ValueError(f"expected {n} distinct ctx-aligned windows")
    if E.overlaps(screen, confirm) or win["token_ids_sha256"] != E.QWEN_TOKEN_SHA:
        raise ValueError("screen and confirmation windows must be disjoint TRAIN windows of the Qwen stream")
    if confirm != reg["confirm32"] or E.overlaps(confirm, reg["excluded"] + list(range(0, E.PREFIX_GUARD[1], E.CTX))):
        raise ValueError("confirmation windows must be the frozen registry draw")
    if E.overlaps(screen + confirm, win.get("capture_windows", [])):
        raise ValueError("screen/confirm windows overlap the round-7 capture windows")
    for pr in cell["identity"]["probes"]:
        if pr["arm"] not in arms or E.overlaps([pr["start"]], screen + confirm):
            raise ValueError(f"bad probe {pr}")
    ref = cell["identity"].get("screen_reference")
    if ref is None and not cell["identity"]["probes"]:
        raise ValueError("a cell without a screen reference needs a held-out identity probe")
    if ref is not None and len(ref["window_nll"]) != 16:
        raise ValueError("screen reference must hold 16 NLLs")
    pc = cell["parent_cost"]
    if pc["mode"] == "reference":
        pref = E.read_json(pc["file"])
        if pref["windows"] != screen or pref.get("simulated") or not pref.get("audit_ok"):
            raise ValueError("parent reference is not an audited GPU measurement on the screen windows")
        if E.sha256_file(pref["trace"]) != pref["trace_sha256"] or pref["trace"] != pc["trace"]:
            raise ValueError("parent reference trace changed")
    elif pc["mode"] != "evaluate":
        raise ValueError(f"unknown parent_cost mode {pc['mode']}")
    return {"loaded": loaded, "screen": screen, "confirm": confirm, "candidates": cands}


def preflight(m, task_index, out_root, deep=False, *, check_outputs=True) -> dict:
    verify_code_hashes(m)
    t, cell = task_cell(m, task_index)
    if not cell.get("enabled"):
        return {"cell": cell["id"], "enabled": False, "disabled_reason": cell.get("disabled_reason")}
    v = verify_cell(m, cell)
    out = Path(out_root) / cell["id"]
    if check_outputs:
        for name in ("driver_started.json", "screen_result.json", "selected.json", "failure.json"):
            if (out / name).exists():
                raise ValueError(f"refusing to reuse a used cell directory: {out / name}")
        tt = Path(cell["arms"][cell["primary"]]["test_trace"])
        if tt.exists():
            raise ValueError(f"refusing to overwrite the primary's test trace {tt}")
    if Path(cell["arms"][cell["primary"]]["test_trace"]) != test_trace_path(cell, cell["arms"][cell["primary"]]["kb_tbl"]):
        raise ValueError("primary test trace path does not follow run_prc_ppl.sbatch naming")
    rep = {"cell": cell["id"], "enabled": True, "primary": cell["primary"], "secondaries": cell["secondaries"],
           "parent_cost": cell["parent_cost"]["mode"], "screen_reference": cell["identity"]["screen_reference"] is not None,
           "probes": [p["name"] for p in cell["identity"]["probes"]], "out_dir": str(out)}
    if deep:
        rep["resolver_sweep"] = {}
        for name in ["INC"] + v["candidates"]:
            a = cell["arms"][name]
            rep["resolver_sweep"][name] = E.resolver_sweep(a["wrapper"], cell["arms"]["INC"]["wrapper"],
                                                           cell["total_blocks"],
                                                           allow_psl_change=bool(a.get("allow_psl_change")))["checked"]
    return rep


# ---------------------------------------------------------------------------
# per-cell protocol (GPU or simulation)
# ---------------------------------------------------------------------------
def run_cell(m: dict, cell: dict, out: Path, ev) -> dict:
    v = verify_cell(m, cell)
    screen, confirm = v["screen"], v["confirm"]
    arms = cell["arms"]
    nb, lb = cell["total_blocks"], cell["layer_buckets"]
    hybrid = E.read_json(cell["hybrid_config"])
    inc_w, inc_t = v["loaded"]["INC"]
    E.write_json(out / "driver_started.json", {"time": time.time(), "cell": cell["id"], "simulated": ev.simulated,
                                               "primary": cell["primary"], "secondaries": cell["secondaries"],
                                               "manifest_run_id": m["run_id"]}, exclusive=True)
    progress = {"cell": cell["id"], "simulated": ev.simulated, "evaluations": {}}
    # The new-rung share check binds on every screen trace of a real manifest; only a dry-run (stand-in) manifest
    # in CPU simulation runs it descriptively (verify_cell refuses stand-ins anywhere else).
    new_rung_binding = not (m.get("dry_run") and ev.simulated)
    if m.get("dry_run") and not ev.simulated:
        raise ValueError("a dry-run manifest never runs on a GPU")

    def evaluate(name, arm, wins, *, profile="none", expected=None, hold=False):
        """hold=True (candidate arms): the NLLs are returned separately, masked in the evaluator's log and
        written nowhere until release() - i.e. until every screened trace passed its audits."""
        header = {"cell": cell["id"], "arm": arm, "split": "train", "ctx": E.CTX,
                  "mp_config_json": str(arms[arm]["wrapper"]), "model": cell["model_path"],
                  "total_blocks": nb, "run_id": m["run_id"], "simulated": ev.simulated}
        mask = contextlib.redirect_stdout(_WithheldNLL(sys.stdout)) if hold else contextlib.nullcontext()
        with mask:
            r = ev.evaluate(name, arms[arm]["wrapper"], wins, profile=profile, expected=expected, header_extra=header)
        tm, tc = E.trace_totals(r["trace"])
        rec = {"name": name, "arm": arm, "wrapper": arms[arm]["wrapper"], "window_starts": [int(s) for s in wins],
               "seconds": r["seconds"], "trace": str(r["trace"]), "cost": tc / tm, "total_macs": tm,
               "total_cycle_macs": tc}
        if r["snapshot"] is not None:
            totals = E.trace_totals(r["first_trace"]) if profile == "first" else (tm, tc)
            rec["profile_checks"] = E.profile_checks(r["snapshot"], totals, E.read_json(arms[arm]["table"]))
            E.write_json(out / f"{name}_profile.json", r["snapshot"], exclusive=True)
        nll = [float(x) for x in r["window_nll"]]
        if hold:
            progress["evaluations"][name] = dict(rec, nll=WITHHELD)
            E.write_json(out / "screen_progress.json", progress)
            return rec, nll
        return release(rec, nll)

    def release(rec, nll):
        rec = dict(rec, window_nll=nll, mean_nll=sum(nll) / len(nll))
        E.write_json(out / f"{rec['name']}_nll.json", rec, exclusive=True)
        progress["evaluations"][rec["name"]] = {k: x for k, x in rec.items() if k != "window_nll"}
        E.write_json(out / "screen_progress.json", progress)
        return rec

    def audited(rec, arm, inc_agg, wins, *, stage="screen"):
        payload = E.load_trace(rec["trace"])
        rep, agg = E.audit_trace(payload, name=rec["name"], wrapper_path=arms[arm]["wrapper"], inc_agg=inc_agg,
                                 inc_table=inc_t, inc_wrapper=inc_w, hybrid=hybrid, total_blocks=nb, layer_buckets=lb,
                                 expected_windows=wins, expected_wrapper=arms[arm]["wrapper"])
        del payload
        hist_rt = rep.pop("histogram_macs")
        if arms[arm]["role"] in ("primary", "secondary"):
            binding = stage == "screen" and new_rung_binding
            nr = new_rung_audit(arms[arm].get("realized_length_prediction"), agg, hist_rt, v["loaded"][arm][1], inc_t,
                                total_blocks=nb, layer_buckets=lb, binding=binding, stage=stage)
            rep["new_rung_audit"] = nr
            if binding and not nr["ok"]:
                rep["ok"] = False
                rep["failures"] = list(rep["failures"]) + [f"{rec['name']}: new-rung audit: {f}" for f in nr["failures"]]
        E.write_json(out / f"{rec['name']}_histogram.json", hist_rt, exclusive=True)
        E.write_json(out / f"{rec['name']}_audit.json", rep, exclusive=True)
        return rep, agg

    def stop_arm(rec, arm, rep, stage):
        """Fail closed: a failure record (no NLL - it is discarded unread), then stop the cell."""
        E.write_json(out / f"{rec['name']}_audit_failure.json", {
            "schema": SCHEMA + "-arm-stopped", "cell": cell["id"], "arm": arm, "role": arms[arm]["role"],
            "stage": stage, "trace": rec["trace"], "failures": rep["failures"][:30],
            "new_rung_audit": rep.get("new_rung_audit"), "nll": "discarded unread (the audit precedes any NLL)",
            "rule": REALIZED_LENGTH_RULE, "decision": None, "simulated": ev.simulated, "time": time.time()},
            exclusive=True)
        raise E.AuditError(f"{cell['id']} {arm} ({arms[arm]['role']}) {stage} trace audit failed -> cell stopped, no "
                           f"decision (registered: any realized-length audit failure stops the cell): "
                           f"{rep['failures'][:3]}")

    token_sha, _ = ev.build()
    if token_sha != cell["windows"]["token_ids_sha256"]:
        raise E.IdentityError(f"TRAIN token stream sha {token_sha} != {cell['windows']['token_ids_sha256']}")
    identity = {"token_ids_sha256": token_sha}
    for pr in cell["identity"]["probes"]:
        rec = evaluate(pr["name"], pr["arm"], [pr["start"]], expected=[pr["expected"]])
        identity[pr["name"]] = {"expected": pr["expected"], "observed": rec["window_nll"][0], "exact": True,
                                "source": pr["source"]}
    ref = cell["identity"].get("screen_reference")
    before = evaluate("identity_before", "INC", screen[:1], expected=[ref["window_nll"][0]] if ref else None)
    inc = evaluate("INC_screen", "INC", screen, profile="first", expected=ref["window_nll"] if ref else None)
    if inc["window_nll"][0] != before["window_nll"][0] or \
            E.trace_totals(out / "INC_screen_first_window_trace.json") != (before["total_macs"], before["total_cycle_macs"]):
        raise E.IdentityError("profiled INC window 1 differs from the unprofiled run")
    inc_rep, inc_agg = audited(inc, "INC", None, screen)
    if not inc_rep["ok"]:
        raise E.AuditError(f"INC screen trace audit failed: {inc_rep['failures'][:5]}")
    if ref is not None:
        ref_agg = R.TraceAgg(E.load_trace(ref["trace_file"]), nb, lb)
        diffs = R.compare_trace_to_reference(inc_agg, ref_agg)
        del ref_agg
        if diffs:
            raise E.IdentityError(f"INC screen trace differs from the stored reference: {diffs}")
        identity["screen_reference"] = {"nll_file": ref["nll_file"], "trace_file": ref["trace_file"],
                                        "window_nll_exact": True, "trace_records_exact": True}
    progress["identity"] = identity

    pc = cell["parent_cost"]
    if pc["mode"] == "evaluate":
        prec = evaluate("PARENT_screen", "PARENT", screen)
        prep, _ = audited(prec, "PARENT", inc_agg, screen)
        if not prep["ok"]:
            raise E.AuditError(f"PARENT screen trace audit failed: {prep['failures'][:5]}")
        P = prec["cost"]
        p_src = {"mode": "evaluate", "trace": prec["trace"], "P": P}
    else:
        pref = E.read_json(pc["file"])
        tm, tc = E.trace_totals(pref["trace"])
        P = tc / tm
        if abs(P / pref["P"] - 1) > 1e-12:
            raise E.AuditError("parent reference P does not match its trace")
        p_src = {"mode": "reference", "file": pc["file"], "trace": pref["trace"], "P": P}
    results = {}
    primary = cell["primary"]
    # (a) every screened arm: GPU trace first, audited (standard + registered new-rung share); NLLs held unread.
    held = {}
    for name in [primary] + list(cell["secondaries"]):
        rec, nll = evaluate(f"{name}_screen", name, screen, profile="first", hold=True)
        rep, _agg = audited(rec, name, inc_agg, screen, stage="screen")
        del _agg
        if not rep["ok"]:
            stop_arm(rec, name, rep, "screen")
        held[name] = (rec, nll, rep)
    progress["screen_traces_audited_ok"] = sorted(held)
    # (b) only now are candidate NLLs released and read.
    for name, (rec, nll, rep) in held.items():
        a = arms[name]
        rec = release(rec, nll)
        met = E.arm_metrics(rec["window_nll"], inc["window_nll"], rec["cost"], inc["cost"])
        nr = rep.get("new_rung_audit") or {}
        results[name] = {"arm": name, "kind": a["kind"], "role": a["role"], "metrics": met, "cost": rec["cost"],
                         "cost_vs_inc_pct": 100.0 * (rec["cost"] / inc["cost"] - 1.0),
                         "audit_ok": bool(rep["ok"]), "audit_failures": rep["failures"][:10],
                         "new_rung_audit": {k: nr.get(k) for k in ("ok", "binding", "tolerance", "buckets", "notes")},
                         "profile_checks": rec.get("profile_checks"), "pred_true_nats": a["pred_true_nats"],
                         "mde_nats": a["mde_nats"], "kappa_ratio": a.get("kappa_ratio"),
                         "buckets_vs_inc": rep.get("buckets_vs_inc"), "window_nll": rec["window_nll"]}
    P_gate = P
    if ev.simulated and cell.get("simulation", {}).get("force_cost_gate_pass"):
        # SIMULATION ONLY: stand-in tables are not iso-cost under synthetic traces; exercising the
        # confirm/test path needs P := C(primary). Never reachable on a GPU run (ev.simulated False).
        P_gate = results[primary]["cost"]
        p_src["simulation_forced_P"] = P_gate
    for name, res in results.items():
        res["cost_gate"] = E.cost_gate(res["cost"], P_gate)
        E.write_json(out / f"{name}_screen_result.json", res, exclusive=True)
        met = res["metrics"]
        log(f"{cell['id']} screen {name} ({res['role']}): dNLL {met['dnll']:+.5f} (se {met['se_dnll']:.5f}, "
            f"z {met['z']}) C/P {res['cost_gate']['ratio']:.5f} audit {res['audit_ok']}")
    gate = E.screen_proceeds(results[primary])
    screen_result = {"schema": SCHEMA + "-screen", "cell": cell["id"], "simulated": ev.simulated,
                     "label": E.SIM_LABEL if ev.simulated else "GPU measurement, WikiText-2 TRAIN (not citable)",
                     "windows": screen, "P": p_src, "inc": {"cost": inc["cost"], "mean_nll": inc["mean_nll"],
                                                            "C_over_P": inc["cost"] / P},
                     "arms": {k: {x: y for x, y in r.items() if x != "window_nll"} for k, r in results.items()},
                     "primary": primary, "screen_gate": gate, "identity": identity,
                     "realized_length_audit": {"rule": REALIZED_LENGTH_RULE, "binding": new_rung_binding,
                                               "all_screened_traces_ok_before_any_nll": True}}
    E.write_json(out / "screen_result.json", screen_result, exclusive=True)

    confirm_res = None
    if gate["proceed_to_confirm"]:
        inc_c = evaluate("INC_confirm", "INC", confirm)
        rep_ic, agg_ic = audited(inc_c, "INC", None, confirm, stage="confirm")
        if not rep_ic["ok"]:
            raise E.AuditError(f"INC confirmation trace audit failed -> cell stopped: {rep_ic['failures'][:3]}")
        pr_rec, pr_nll = evaluate(f"{primary}_confirm", primary, confirm, hold=True)
        rep_pc, _ = audited(pr_rec, primary, agg_ic, confirm, stage="confirm")
        del agg_ic, _
        if not rep_pc["ok"]:
            stop_arm(pr_rec, primary, rep_pc, "confirm")
        pr_c = release(pr_rec, pr_nll)
        met = E.arm_metrics(pr_c["window_nll"], inc_c["window_nll"], pr_c["cost"], inc_c["cost"])
        dec = E.confirm_passes(met, rep_ic["ok"] and rep_pc["ok"])
        confirm_res = {"schema": SCHEMA + "-confirm", "cell": cell["id"], "simulated": ev.simulated, "arm": primary,
                       "windows": confirm, "metrics": met, "decision": dec,
                       "cost_vs_inc_pct_confirm": 100.0 * (pr_c["cost"] / inc_c["cost"] - 1.0),
                       "audits": {"INC": rep_ic["ok"], primary: rep_pc["ok"]},
                       "new_rung_audit_confirm_descriptive": rep_pc.get("new_rung_audit"),
                       "audit_failures": rep_ic["failures"][:5] + rep_pc["failures"][:5]}
        E.write_json(out / "confirm_result.json", confirm_res, exclusive=True)
        log(f"{cell['id']} confirm {primary}: dNLL {met['dnll']:+.5f} (se {met['se_dnll']:.5f}, z {met['z']}) "
            f"-> full test {dec['full_test']}")
    after = evaluate("identity_after", "INC", screen[:1])
    if after["window_nll"] != before["window_nll"] or (after["total_macs"], after["total_cycle_macs"]) != \
            (before["total_macs"], before["total_cycle_macs"]):
        raise E.IdentityError("restoring the incumbent changed its loss or SC work")
    identity["restored_incumbent_exact"] = True
    go = bool(confirm_res and confirm_res["decision"]["full_test"])
    pa = arms[primary]
    sel = {"schema": SCHEMA + "-selected", "cell": cell["id"], "simulated": ev.simulated,
           "evaluate_full_test": go, "arm": primary if go else None,
           "wrapper": pa["wrapper"] if go else None, "kb_tbl": pa["kb_tbl"] if go else None,
           "test_trace": pa["test_trace"] if go else None,
           "reason": ("confirmed at z < -2 on 32 fresh windows" if go else
                      ("screen gate not met: " + ", ".join(k for k, x in gate["conditions"].items() if not x))
                      if not gate["proceed_to_confirm"] else "confirmation z >= -2 or audit failure"),
           "screen_gate": gate, "confirm": confirm_res["decision"] if confirm_res else None,
           "preregistered_rules": m["preregistered_rules"]}
    E.write_json(out / "selected.json", sel, exclusive=True)
    log(f"{cell['id']} selected: full test {go} ({sel['reason']})")
    return sel


# ---------------------------------------------------------------------------
# finalize (CPU, after the launcher's full-protocol test)
# ---------------------------------------------------------------------------
def finalize(m, cell, out: Path, trace_path, *, allow_simulated=False, run_log=None) -> dict:
    sel = E.read_json(out / "selected.json")
    if not sel.get("evaluate_full_test"):
        raise ValueError("finalize called without a confirmed primary")
    if sel.get("simulated") and not allow_simulated:
        raise ValueError("selected.json is a simulation")
    pa = cell["arms"][cell["primary"]]
    if sel["arm"] != cell["primary"] or sel["wrapper"] != pa["wrapper"] or str(trace_path) != pa["test_trace"]:
        raise ValueError("selected arm / wrapper / trace do not match the pre-registered primary")
    for p in (pa["wrapper"], pa["table"]):
        if E.sha256_file(p) != cell["hashes"][p]:
            raise ValueError(f"primary input changed since freezing: {p}")
    bound = cell["full_test"].get("best_all_test_cost_bound", BEST_ALL_TEST_COST_BOUND)
    if bound != BEST_ALL_TEST_COST_BOUND:
        raise ValueError("best(all) test-cost bound differs from the pre-registered 1.5%")
    payload = E.load_trace(trace_path)
    h = payload["header"]
    failed = []
    for cond, msg in ((h.get("eval_tokens") == EVAL_TOKENS[cell["model"]], "eval_tokens"),
                      (h.get("ctx") == 2048 and h.get("stride") == 2048, "ctx/stride"),
                      (h.get("ppl_max_tokens") == 0, "ppl_max_tokens"),
                      (h.get("ppl_window_batch_size") == 1, "ppl_window_batch_size"),
                      (h.get("owen_mode") == "bitrev", "owen_mode"),
                      (str(h.get("scramble_masks")) == "64", "scramble_masks"),
                      (h.get("sc_prec") == 8 and h.get("sc_halve") is True, "sc_prec/halve"),
                      (h.get("model") == cell["model_path"], "model"),
                      (h.get("mp_config_json") == pa["wrapper"], "mp_config_json"),
                      (int(h.get("total_blocks", -1)) == cell["total_blocks"], "total_blocks"),
                      (isinstance(h.get("ppl"), (int, float)) and math.isfinite(h["ppl"]) and h["ppl"] > 0, "ppl")):
        if not cond:
            failed.append(msg)
    if run_log is not None:
        env = parse_run_log(run_log, hybrid=cell["hybrid_config"], wrapper=pa["wrapper"])
        failed += env["failed"]
    elif sel.get("simulated") and allow_simulated:
        env = {"note": "simulation: no run log"}
    else:
        env = None
        failed.append("run_log: missing (FRONTEND / hybrid mask / RNG grid unverifiable)")
    inc_t, inc_w = E.read_json(cell["arms"]["INC"]["table"]), E.read_json(cell["arms"]["INC"]["wrapper"])
    inc_agg = R.TraceAgg(E.load_trace(cell["incumbent"]["full_test_trace"]), cell["total_blocks"], cell["layer_buckets"])
    rep, agg = E.audit_trace(payload, name="full_test", wrapper_path=pa["wrapper"], inc_agg=inc_agg, inc_table=inc_t,
                             inc_wrapper=inc_w, hybrid=E.read_json(cell["hybrid_config"]),
                             total_blocks=cell["total_blocks"], layer_buckets=cell["layer_buckets"],
                             expected_windows=None, expected_wrapper=pa["wrapper"])
    del payload, inc_agg
    hist_rt = rep.pop("histogram_macs")
    nr_test = new_rung_audit(pa.get("realized_length_prediction"), agg, hist_rt, E.read_json(pa["table"]), inc_t,
                             total_blocks=cell["total_blocks"], layer_buckets=cell["layer_buckets"], binding=False,
                             stage="full_test")
    del agg
    E.write_json(out / "test_histogram.json", hist_rt, exclusive=True)
    if not rep["ok"]:
        failed.append("audit: " + "; ".join(rep["failures"][:5]))
    ppl = float(h["ppl"])
    ft = cell["full_test"]
    inc_ppl = cell["incumbent"]["full_test_ppl"]
    arm_label = f"r7_{cell['primary']}"
    cost_ratio = rep["cost"] / cell["parent"]["test_cost"]
    flags = []
    if abs(cost_ratio - 1.0) > TEST_COST_FLAG:
        flags.append(f"test cost {100 * (cost_ratio - 1):+.3f}% vs the parent (outside +-1%)")
    cost_ok = abs(cost_ratio - 1.0) <= bound
    eligible = bool(not failed and cost_ok)
    if not cost_ok:
        flags.append(f"test cost outside the pre-registered +-{100 * bound:.1f}% best(all) bound: recorded, "
                     "NOT entered into best(all)")
    inc_vs_parent = 100.0 * (cell["incumbent"]["full_test_cost"] / cell["parent"]["test_cost"] - 1.0)
    rec = {
        "schema": SCHEMA + "-full-test", "cell": cell["id"], "arm": arm_label, "kind": pa["kind"], "ppl": ppl,
        "cost": rep["cost"], "protocol_ok": not failed, "failed_checks": failed, "wrapper": pa["wrapper"],
        "table": pa["table"], "trace": str(trace_path), "trace_sha256": E.sha256_file(trace_path),
        "candidate_hashes": {p: cell["hashes"][p] for p in (pa["wrapper"], pa["table"])},
        "pred_true_nats": pa["pred_true_nats"], "mde_nats": pa["mde_nats"], "kappa_ratio": pa.get("kappa_ratio"),
        "kappa_pins": pa.get("kappa_pins"), "target_scale": pa.get("target_scale"),
        "test_cost_vs_parent_pct": 100.0 * (cost_ratio - 1.0),
        "test_cost_vs_incumbent_pct": 100.0 * (rep["cost"] / cell["incumbent"]["full_test_cost"] - 1.0),
        "incumbent_test_cost_vs_parent_pct": inc_vs_parent,
        "ppl_vs_incumbent_pct": 100.0 * (ppl / inc_ppl - 1.0),
        "ppl_vs_parent_pct": 100.0 * (ppl / cell["parent"]["ppl"] - 1.0),
        "x_fp16": ppl / ft["fp16_ppl"], "fp16_ppl": ft["fp16_ppl"], "flip_thresholds": ft["flip_thresholds"],
        "passes_1p05x_fp16": ppl <= ft["flip_thresholds"]["1.05x"],
        "passes_1p10x_fp16": ppl <= ft["flip_thresholds"]["1.10x"],
        "incumbent": cell["incumbent"], "flags": flags, "run_environment": env,
        "best_all": {"previous_best_ppl": inc_ppl, "eligible": eligible,
                     "test_cost_bound": bound,
                     "new_best_ppl": min(ppl, inc_ppl) if eligible else inc_ppl,
                     "new_best_arm": arm_label if (eligible and ppl < inc_ppl) else cell["incumbent"]["arm"],
                     "entry_rule": m["preregistered_rules"]["best_all_entry"]},
        "disclosure": m["preregistered_rules"]["disclosure"],
        "budget_correction": f"INC's full-test cost is {inc_vs_parent:+.3f}% vs the parent and this table was "
                             f"re-targeted to the parent's cost ({100.0 * (cost_ratio - 1.0):+.3f}% on test): part "
                             "of any gain vs INC is that budget correction",
        "escape_disclosure": pa.get("escape_disclosure"),
        "class_costs_test": rep["class_costs"], "buckets_vs_incumbent_test": rep.get("buckets_vs_inc"),
        "new_rung_audit_test_descriptive": nr_test,
        "trace_header": h, "simulated": bool(sel.get("simulated")),
    }
    E.write_json(out / "full_test_result.json", rec, exclusive=True)
    log(f"finalize {cell['id']}: PPL {ppl:.6f} ({rec['x_fp16']:.5f}x fp16) vs incumbent "
        f"{rec['ppl_vs_incumbent_pct']:+.3f}%; test cost vs parent {rec['test_cost_vs_parent_pct']:+.3f}%; "
        f"protocol_ok {rec['protocol_ok']}; best(all) eligible {eligible}; flags {flags}")
    return rec


def sim_evaluator(cell, out):
    arms, sim = cell["arms"], cell["simulation"]
    ref = cell["identity"].get("screen_reference")
    nll_ref = dict(zip([int(s) for s in cell["windows"]["screen"]], ref["window_nll"])) if ref else {}
    deltas = {arms[n]["wrapper"]: float(sim.get("deltas", {}).get(n, -0.004)) for n in
              [cell["primary"]] + list(cell["secondaries"])}
    deltas[arms["PARENT"]["wrapper"]] = 0.03
    return E.SimEvaluator(cell, out, inc_wrapper=arms["INC"]["wrapper"], inc_standin=sim["inc_standin"],
                          nll_ref=nll_ref, token_sha=cell["windows"]["token_ids_sha256"],
                          parent_wrapper=arms["PARENT"]["wrapper"], parent_standin=None, deltas=deltas,
                          nll_exact={(arms[p["arm"]]["wrapper"], p["start"]): p["expected"]
                                     for p in cell["identity"]["probes"]})


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("preflight", "run", "simulate", "test-args", "finalize"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--trace", default="")
    ap.add_argument("--run-log", default=None, help="finalize: run_prc_ppl.sbatch's own output (tee'd by the launcher)")
    ap.add_argument("--allow-simulated", action="store_true", help="finalize a simulation (tests only)")
    args = ap.parse_args()
    m = load_manifest(args.manifest)
    if args.stage == "run" and (m.get("dry_run") or m.get("test_build")):
        raise ValueError("refusing to run a DRY-RUN or TEST-BUILD manifest on a GPU")
    t, cell = task_cell(m, args.task)
    out_root = Path(args.out_root or m["output_root"]).resolve()
    if args.stage in ("run", "finalize", "test-args") and not args.allow_simulated and str(out_root) != m["output_root"]:
        raise ValueError("GPU runs must use the manifest output_root")
    if args.stage == "simulate" and str(out_root).startswith(str(E.R7_ROOT)):
        raise ValueError("simulation must not write under the Turbo round-7 root")
    out = out_root / cell["id"]
    if args.stage == "preflight":
        rep = preflight(m, args.task, out_root, deep=args.deep)
        print(json.dumps({"preflight": "PASS", "task": t["id"], "report": rep}, indent=1))
        return 0
    if args.stage == "test-args":
        verify_code_hashes(m)
        sel = E.read_json(out / "selected.json")
        if not sel.get("evaluate_full_test"):
            print("")
            return 0
        pa = cell["arms"][cell["primary"]]
        if sel["wrapper"] != pa["wrapper"] or sel["kb_tbl"] != pa["kb_tbl"] or sel["test_trace"] != pa["test_trace"] \
                or sel.get("simulated"):
            raise ValueError("selected.json does not match the pre-registered primary")
        fields = [cell["model"], str(cell["target"]), pa["wrapper"], pa["kb_tbl"], pa["test_trace"],
                  str(Path(cell["arms"]["PARENT"]["wrapper"]).parent)]
        assert all("\t" not in f and "\n" not in f for f in fields)
        print("\t".join(fields))
        return 0
    if args.stage == "finalize":
        verify_code_hashes(m)
        try:
            finalize(m, cell, out, args.trace, allow_simulated=args.allow_simulated, run_log=args.run_log)
        except BaseException as exc:
            E.write_json(out / "failure_finalize.json", {"type": type(exc).__name__, "message": str(exc),
                                                         "traceback": traceback.format_exc(), "time": time.time()})
            raise
        return 0
    rep = preflight(m, args.task, out_root)
    if not rep["enabled"]:
        log(f"task {t['id']} disabled: {rep['disabled_reason']}; nothing to do")
        return 0
    if not out.is_dir():
        if args.stage == "simulate":
            out.mkdir(parents=True)
        else:
            raise ValueError(f"launcher must create the cell directory first: {out}")
    ev = sim_evaluator(cell, out) if args.stage == "simulate" else \
        E.Evaluator(cell, out, build_wrapper=cell["arms"]["INC"]["wrapper"], tag=TAG)
    try:
        run_cell(m, cell, out, ev)
    except BaseException as exc:
        E.write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                            "traceback": traceback.format_exc(), "time": time.time()})
        raise
    finally:
        ev.close()
    verify_code_hashes(m)
    log(f"task {t['id']} screen/confirm COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
