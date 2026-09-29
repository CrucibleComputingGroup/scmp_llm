"""Round-7 STEP 0 driver (2026-09-28): the measured budget-shift linear marginal on 30B
(CRIT L3) that sets the round-7 family currency kappa, plus the per-cell screen-window
references (exact parent cost P + INC 16-window profile) that the pre-registered target-scale
fixed point (prc_fixedpoint_r7_20260928.py) replays round-7 tables on.

Two cell kinds (one GPU, one model load per cell, WikiText-2 TRAIN, tasks sequential):
  kappa     30B t40, 30B t32 (screen windows = the round-6 diag windows; stored INC reference =
            round-4 confirm_incumbent, bit identity)
  reference 30B t48 (screen windows = the 16 fresh registry windows; no stored INC reference:
            cross-job identity through the held-out probe, and INC16 BECOMES the screen
            reference), 4B t64 (round-6 4B diag windows; stored reference = the round-6 INC run;
            optional task, only when the user enables 4B t64)
Per cell:
  1. identity_before  INC on window 1; must equal the stored reference window 1 (30B t32/t40:
                      round-4 confirm_incumbent, and round-6 INC_nll.json when it exists; 4B t64:
                      the round-6 INC run) bit for bit. 30B t48 has no stored value here.
  2. probes           held-out window 30720 vs prc2 c7 held-out NLLs (kappa cells: PARENT and
                      C17; reference cells: INC and PARENT) - cross-job identity of the swap path.
  3. C17, S80         (kappa cells only) the existing c17 / c17_s80 tables (t32: c17e32 pair) on
                      the 16 windows, first window profiled; every trace audited against the
                      table's OWN ladders (runtime resolver + JSON mirror), equal MACs vs the
                      INC reference trace on the same windows, profile replay exact.
  4. identity_mid     (kappa cells only) INC window 1 again (same NLL, same SC work) ->
                      step0_measured.json: m = mean paired NLL(S80) - NLL(C17), its SE, the cycle
                      shift, the measured dNLL per 1% cycles. This file is final at this point.
  5. INC16 profiled   INC on all 16 windows under one observe-only ProfileCollector; NLLs and trace
                      must equal the stored reference exactly where one exists; window 1 must equal
                      identity_before -> INC16_profile.json (the fixed point's replay trajectory).
  6. PARENT (16)      exact parent cost P on the same windows (audited, equal MACs vs INC16) ->
                      parent_reference.json (the screen cost gate's P and the fixed point's P).
  7. identity_after   INC window 1 again -> step0_summary.json.
The Fisher prediction for the c17/s80 pair exists only in the round-7 capture diag
(--score-tables c17=...,c17_s80=...), so kappa_lin and the pre-registered decision come from
the CPU stage ``kappa`` once the captures exist.

  python benchmark/ppl/prc_step0_r7_20260928.py preflight --manifest M --task K [--deep]
  python benchmark/ppl/prc_step0_r7_20260928.py run       --manifest M --task K
  python benchmark/ppl/prc_step0_r7_20260928.py simulate  --manifest M --task K --out-root SCRATCH
  python benchmark/ppl/prc_step0_r7_20260928.py kappa     --manifest M --capture-manifest CM \
         [--capture-diag 30B_t32=P ...] [--capture-identity 30B_t32=P ...] [--score-tables-spec 30B_t32=SPEC ...]
         [--out FILE] [--allow-simulated]
Units: HALVED stream lengths/costs (nominal = 2x); max 128.
"""
from __future__ import annotations

import argparse
import json
import math
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

SCHEMA = "prc-step0-r7-v1"
TAG = "r7s0"


def log(msg):
    print(f"[{TAG}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# manifest / preflight
# ---------------------------------------------------------------------------
def load_manifest(path) -> dict:
    m = E.read_json(path)
    if m.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema {m.get('schema')!r} != {SCHEMA!r}")
    if m.get("max_total_gpus") != 4 or m.get("gpus_per_task") != 1:
        raise ValueError("manifest must record the four-GPU total and one GPU per task")
    E.check_protocol(m["protocol"])
    if m["protocol"].get("full_test") is not False:
        raise ValueError("step 0 never runs a full test")
    return m


def verify_code_hashes(m) -> None:
    code = m.get("code_hashes") or {}
    if not code or set(code) != set(m.get("source_files", [])):
        raise ValueError("code hashes must cover exactly the manifest source_files")
    for p, h in code.items():
        if E.sha256_file(p) != h:
            raise ValueError(f"experiment source changed after freezing: {p}")


CELL_KINDS = {"kappa": {"pair": True, "parent16": True, "inc16_profiled": True},
              "reference": {"pair": False, "parent16": True, "inc16_profiled": True}}
WINDOW_FIELDS = ("r6_diag_windows_30B", "screen16_fresh_t48", "r6_diag_windows_4B")


def verify_cell(m: dict, cell: dict) -> dict:
    for p, h in cell["hashes"].items():
        if E.sha256_file(p) != h:
            raise ValueError(f"frozen input changed: {p}")
    reg = E.read_json(m["windows_registry"]["path"])
    if E.sha256_file(m["windows_registry"]["path"]) != m["windows_registry"]["sha256"]:
        raise ValueError("window registry changed")
    kind = cell.get("kind")
    if kind not in CELL_KINDS or cell["evals"] != CELL_KINDS[kind]:
        raise ValueError(f"cell kind {kind!r} / evals {cell.get('evals')} are not a known step-0 cell type")
    arms = cell["arms"]
    want_arms = {"INC", "PARENT"} | ({"C17", "S80"} if kind == "kappa" else set())
    if set(arms) != want_arms:
        raise ValueError(f"{kind} cell must evaluate exactly {sorted(want_arms)}, got {sorted(arms)}")
    inc_w = E.read_json(arms["INC"]["wrapper"])
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
    inc_t = loaded["INC"][1]
    if kind == "kappa":
        for name in ("C17", "S80"):
            rep = E.validate_lineage(inc_t, loaded[name][1], total_blocks=cell["total_blocks"])
            if not rep["ok"]:
                raise ValueError(f"{name}: lineage vs INC failed: {rep['failures'][:5]}")
        pair = E.validate_lineage(loaded["C17"][1], loaded["S80"][1], total_blocks=cell["total_blocks"])
        if not pair["ok"] or pair["attention_threshold_keys_changed"] or pair["attention_ladders_changed"] \
                or not pair["prc_threshold_keys_changed"]:
            raise ValueError("C17 vs S80 must differ ONLY in per_row_chunk thresholds (a pure linear budget shift)")
    if (loaded["PARENT"][1].get("per_row_chunk") or {}).get("buckets"):
        raise ValueError("PARENT must be the per-row parent table")
    starts = [int(s) for s in cell["windows"]["starts"]]
    if len(starts) != 16 or len(set(starts)) != 16 or any(s % E.CTX for s in starts):
        raise ValueError("exactly 16 distinct ctx-aligned windows are required")
    field = cell["windows"].get("field")
    if field not in WINDOW_FIELDS or starts != reg[field] or cell["windows"]["token_ids_sha256"] != E.QWEN_TOKEN_SHA:
        raise ValueError(f"step-0 windows must be the registry's {field!r} windows of the Qwen TRAIN stream")
    if kind == "kappa" and field != "r6_diag_windows_30B":
        raise ValueError("kappa cells measure on the round-6 30B diagnostic windows")
    ref = cell["identity"].get("reference")
    if kind == "kappa" and ref is None:
        raise ValueError("kappa cells need the stored round-4 INC reference")
    if ref is not None and len(ref["window_nll"]) != 16:
        raise ValueError("stored INC reference must hold 16 window NLLs")
    if ref is None and not any(pr["arm"] == "INC" for pr in cell["probes"]):
        raise ValueError("a cell without a stored INC reference needs an INC held-out probe")
    for pr in cell["probes"]:
        if pr["arm"] not in arms or E.overlaps([pr["start"]], starts):
            raise ValueError(f"bad probe {pr}")
    return {"loaded": loaded, "starts": starts, "kind": kind}


def preflight(m, task_index, out_root, deep=False) -> list:
    verify_code_hashes(m)
    tasks = m["tasks"]
    if not 0 <= task_index < len(tasks) or tasks[task_index]["index"] != task_index:
        raise ValueError(f"no task {task_index}")
    report = []
    for cid in tasks[task_index]["cells"]:
        cell = m["cells"][cid]
        v = verify_cell(m, cell)
        out = Path(out_root) / cid
        for name in ("driver_started.json", "step0_measured.json", "step0_summary.json", "failure.json"):
            if (out / name).exists():
                raise ValueError(f"refusing to reuse a used cell directory: {out / name}")
        entry = {"cell": cid, "kind": v["kind"], "windows": len(v["starts"]), "arms": sorted(cell["arms"]),
                 "evals": cell["evals"], "out_dir": str(out)}
        if deep:
            entry["resolver_sweep"] = {n: E.resolver_sweep(cell["arms"][n]["wrapper"], cell["arms"]["INC"]["wrapper"],
                                                           cell["total_blocks"])["checked"]
                                       for n in ("INC", "C17", "S80") if n in cell["arms"]}
        report.append(entry)
    return report


# ---------------------------------------------------------------------------
# per-cell protocol
# ---------------------------------------------------------------------------
def r6_reference(cell):
    """Round-6 INC NLLs on the same windows, if round 6 has written them (read at run time)."""
    if not cell["identity"].get("r6"):
        return None
    p = Path(cell["identity"]["r6"]["inc_nll_file"])
    for attempt in range(3):
        if not p.exists():
            return None
        try:
            d = E.read_json(p)
            return {"file": str(p), "sha256": E.sha256_file(p), "window_nll": d["window_nll"],
                    "window_starts": d.get("window_starts")}
        except json.JSONDecodeError:
            time.sleep(20)
    return None


def run_cell(m: dict, cell: dict, out: Path, ev) -> dict:
    v = verify_cell(m, cell)
    starts = v["starts"]
    arms = cell["arms"]
    nb, lb = cell["total_blocks"], cell["layer_buckets"]
    hybrid = E.read_json(cell["hybrid_config"])
    inc_w, inc_t = v["loaded"]["INC"]
    E.write_json(out / "driver_started.json", {"time": time.time(), "cell": cell["id"], "simulated": ev.simulated,
                                               "windows": starts, "manifest_run_id": m["run_id"]}, exclusive=True)
    progress = {"cell": cell["id"], "simulated": ev.simulated, "evaluations": {}}

    def evaluate(name, arm, wins, *, profile="none", expected=None):
        header = {"cell": cell["id"], "arm": arm, "split": "train", "ctx": E.CTX,
                  "mp_config_json": str(arms[arm]["wrapper"]), "model": cell["model_path"],
                  "total_blocks": nb, "run_id": m["run_id"], "simulated": ev.simulated}
        r = ev.evaluate(name, arms[arm]["wrapper"], wins, profile=profile, expected=expected, header_extra=header)
        tm, tc = E.trace_totals(r["trace"])
        rec = {"name": name, "arm": arm, "wrapper": arms[arm]["wrapper"], "window_starts": [int(s) for s in wins],
               "window_nll": r["window_nll"], "mean_nll": sum(r["window_nll"]) / len(r["window_nll"]),
               "seconds": r["seconds"], "trace": str(r["trace"]), "cost": tc / tm, "total_macs": tm,
               "total_cycle_macs": tc}
        if r["snapshot"] is not None:
            totals = E.trace_totals(r["first_trace"]) if profile == "first" else (tm, tc)
            rec["profile_checks"] = E.profile_checks(r["snapshot"], totals, E.read_json(arms[arm]["table"]))
            E.write_json(out / f"{name}_profile.json", r["snapshot"], exclusive=True)
        E.write_json(out / f"{name}_nll.json", rec, exclusive=True)
        progress["evaluations"][name] = {k: v_ for k, v_ in rec.items() if k != "window_nll"}
        E.write_json(out / "step0_progress.json", progress)
        return rec

    token_sha, _ = ev.build()
    if token_sha != cell["windows"]["token_ids_sha256"]:
        raise E.IdentityError(f"TRAIN token stream sha {token_sha} != {cell['windows']['token_ids_sha256']}")
    identity = {"token_ids_sha256": token_sha}
    ref = cell["identity"].get("reference")
    r6 = r6_reference(cell)
    if r6 is not None:
        if ref is None or r6["window_nll"] != ref["window_nll"] or (r6["window_starts"] not in (None, starts)):
            raise E.IdentityError("round-6 INC NLLs differ from the stored reference on the same windows")
        identity["r6_reference"] = {k: r6[k] for k in ("file", "sha256")}
    elif cell["identity"].get("r6"):
        identity["r6_reference"] = "not yet written by round 6; the stored reference was used"
    before = evaluate("identity_before", "INC", starts[:1], expected=[ref["window_nll"][0]] if ref else None)
    identity["identity_before"] = {"expected": ref["window_nll"][0] if ref else None,
                                   "observed": before["window_nll"][0], "exact": ref is not None,
                                   "sources": ([ref["source"]] if ref else []) +
                                   (["round-6 INC_nll.json"] if r6 is not None else []),
                                   "note": None if ref else "no stored INC value on these windows: cross-job "
                                                            "identity rests on the INC held-out probe; INC16 "
                                                            "becomes the screen reference"}
    for pr in cell["probes"]:
        rec = evaluate(pr["name"], pr["arm"], [pr["start"]], expected=[pr["expected"]])
        identity[pr["name"]] = {"expected": pr["expected"], "observed": rec["window_nll"][0], "exact": True,
                                "source": pr["source"]}
    progress["identity"] = identity

    inc_ref = None
    if ref is not None:
        inc_ref_payload = E.load_trace(ref["trace_file"])
        inc_ref = R.TraceAgg(inc_ref_payload, nb, lb)
        del inc_ref_payload
        if (inc_ref.total_macs, inc_ref.total_cycle_macs) != (ref["total_macs"], ref["total_cycle_macs"]):
            raise E.IdentityError("stored INC reference trace totals changed")
    if cell["evals"]["pair"]:
        run_pair(cell, out, evaluate, inc_ref, before, identity, starts, v, hybrid, m, ev)
    del inc_ref

    extras = {}
    # INC16 first: it validates (or, without a stored reference, establishes) INC on these windows,
    # and PARENT16's MAC audit then compares against it.
    rec = evaluate("INC16", "INC", starts, profile="all", expected=ref["window_nll"] if ref else None)
    if rec["window_nll"][0] != before["window_nll"][0]:
        raise E.IdentityError("INC16 window 1 differs from identity_before")
    payload = E.load_trace(rec["trace"])
    inc16_rep, agg16 = E.audit_trace(payload, name="INC16", wrapper_path=arms["INC"]["wrapper"], inc_agg=None,
                                     inc_table=inc_t, inc_wrapper=inc_w, hybrid=hybrid, total_blocks=nb,
                                     layer_buckets=lb, expected_windows=starts,
                                     expected_wrapper=arms["INC"]["wrapper"])
    del payload
    E.write_json(out / "INC16_histogram.json", inc16_rep.pop("histogram_macs"), exclusive=True)
    E.write_json(out / "INC16_audit.json", inc16_rep, exclusive=True)
    if not inc16_rep["ok"]:
        raise E.AuditError(f"INC16 trace audit failed: {inc16_rep['failures'][:5]}")
    diffs = []
    if ref is not None:
        ref3 = R.TraceAgg(E.load_trace(ref["trace_file"]), nb, lb)
        diffs = R.compare_trace_to_reference(agg16, ref3)
        del ref3
        if diffs:
            raise E.IdentityError(f"INC16 trace differs from the stored reference: {diffs}")
    extras["inc16_profiled"] = {"window_nll_exact_vs_reference": ref is not None,
                                "trace_records_exact_vs_reference": (ref is not None and not diffs),
                                "reference_source": ref["source"] if ref else None,
                                "profile": str(out / "INC16_profile.json"), "trace": rec["trace"],
                                "nll": str(out / "INC16_nll.json"), "cost": rec["cost"],
                                "U_prot": inc16_rep["runtime_audit"]["U_prot"],
                                "profile_checks": rec.get("profile_checks")}
    rec = evaluate("PARENT", "PARENT", starts)
    payload = E.load_trace(rec["trace"])
    audit, _ = E.audit_trace(payload, name="PARENT", wrapper_path=arms["PARENT"]["wrapper"], inc_agg=agg16,
                             inc_table=inc_t, inc_wrapper=inc_w, hybrid=hybrid, total_blocks=nb, layer_buckets=lb,
                             expected_windows=starts, expected_wrapper=arms["PARENT"]["wrapper"])
    del payload, agg16
    E.write_json(out / "PARENT_histogram.json", audit.pop("histogram_macs"), exclusive=True)
    E.write_json(out / "PARENT_audit.json", audit, exclusive=True)
    inc16_cost = extras["inc16_profiled"]["cost"]
    pref = {"schema": SCHEMA + "-parent-reference", "cell": cell["id"], "simulated": ev.simulated,
            "windows": starts, "wrapper": arms["PARENT"]["wrapper"], "trace": rec["trace"],
            "trace_sha256": E.sha256_file(rec["trace"]), "P": rec["cost"], "window_nll": rec["window_nll"],
            "audit_ok": audit["ok"], "audit_failures": audit["failures"],
            "inc16_cost": inc16_cost, "r_inc_over_parent": inc16_cost / rec["cost"],
            "use": "exact parent cost P on these 16 windows: the round-7 screen cost gate |C/P-1| <= 1% and the "
                   "pre-registered target-scale fixed point (prc_fixedpoint_r7_20260928.py)"}
    E.write_json(out / "parent_reference.json", pref, exclusive=True)
    extras["parent16"] = {"P": rec["cost"], "audit_ok": audit["ok"]}
    if not audit["ok"]:
        raise E.AuditError(f"PARENT trace audit failed: {audit['failures'][:5]}")
    after = evaluate("identity_after", "INC", starts[:1])
    if after["window_nll"] != before["window_nll"] or (after["total_macs"], after["total_cycle_macs"]) != \
            (before["total_macs"], before["total_cycle_macs"]):
        raise E.IdentityError("restoring the incumbent changed its loss or SC work")
    identity["restored_incumbent_exact"] = True
    summary = {"schema": SCHEMA + "-summary", "run_id": m["run_id"], "cell": cell["id"], "kind": v["kind"],
               "simulated": ev.simulated,
               "measured": str(out / "step0_measured.json") if cell["evals"]["pair"] else None,
               "identity": identity, "extras": extras, "complete": True, "finished": time.time()}
    E.write_json(out / "step0_summary.json", summary, exclusive=True)
    log(f"{cell['id']} COMPLETE ({v['kind']}; extras {sorted(extras)})")
    return summary


def run_pair(cell, out, evaluate, inc_ref, before, identity, starts, v, hybrid, m, ev):
    """Kappa cells: C17 and S80 on the 16 windows, identity_mid, then step0_measured.json."""
    arms = cell["arms"]
    nb, lb = cell["total_blocks"], cell["layer_buckets"]
    inc_w, inc_t = v["loaded"]["INC"]
    ref = cell["identity"]["reference"]
    results, audits = {}, {}
    for arm in ("C17", "S80"):
        rec = evaluate(arm, arm, starts, profile="first")
        payload = E.load_trace(rec["trace"])
        audit, _agg = E.audit_trace(payload, name=arm, wrapper_path=arms[arm]["wrapper"], inc_agg=inc_ref,
                                    inc_table=inc_t, inc_wrapper=inc_w, hybrid=hybrid, total_blocks=nb,
                                    layer_buckets=lb, expected_windows=starts,
                                    expected_wrapper=arms[arm]["wrapper"])
        del payload
        hist = audit.pop("histogram_macs")
        E.write_json(out / f"{arm}_histogram.json", hist, exclusive=True)
        E.write_json(out / f"{arm}_audit.json", audit, exclusive=True)
        if not audit["ok"]:
            raise E.AuditError(f"{arm} trace audit failed: {audit['failures'][:5]}")
        results[arm], audits[arm] = rec, audit
    mid = evaluate("identity_mid", "INC", starts[:1])
    if mid["window_nll"] != before["window_nll"] or (mid["total_macs"], mid["total_cycle_macs"]) != \
            (before["total_macs"], before["total_cycle_macs"]):
        raise E.IdentityError("INC window 1 changed between identity_before and identity_mid")
    identity["identity_mid_exact"] = True

    c17, s80 = results["C17"], results["S80"]
    d = [a - b for a, b in zip(s80["window_nll"], c17["window_nll"])]
    st = E.paired(s80["window_nll"], c17["window_nll"])
    dcyc = 100.0 * (s80["cost"] / c17["cost"] - 1.0)
    lin = {a: audits[a]["class_costs"]["cycles_per_sc_mac"].get("linear") for a in ("C17", "S80")}
    att = {a: audits[a]["class_costs"]["cycles_per_sc_mac"].get("attention") for a in ("C17", "S80")}
    ref_nll = ref["window_nll"]
    measured = {
        "schema": SCHEMA + "-measured", "run_id": m["run_id"], "cell": cell["id"], "simulated": ev.simulated,
        "label": E.SIM_LABEL if ev.simulated else "GPU measurement, WikiText-2 TRAIN windows (not citable)",
        "units": "HALVED stream lengths; cost = MAC-weighted mean halved length over all SC trace records",
        "windows": starts, "identity": identity,
        "c17": {k: c17[k] for k in ("wrapper", "mean_nll", "cost", "trace", "window_nll")} |
               {"table": arms["C17"]["table"], "label": arms["C17"]["label"]},
        "s80": {k: s80[k] for k in ("wrapper", "mean_nll", "cost", "trace", "window_nll")} |
               {"table": arms["S80"]["table"], "label": arms["S80"]["label"]},
        "pair": {"window_dnll": d, "paired_s80_minus_c17": st,
                 "dcycles_pct": dcyc, "linear_cycles_per_sc_mac": lin, "attention_cycles_per_sc_mac": att,
                 "dlinear_cycles_pct_of_c17_total": 100.0 * (lin["S80"] - lin["C17"]) / c17["cost"],
                 "dnll_per_1pct_cycles": st["mean_dnll"] / dcyc,
                 "se_dnll_per_1pct_cycles": st["se"] / abs(dcyc),
                 "dppl_pct_per_1pct_cycles": st["dppl_pct"] / dcyc,
                 "sign_note": "dcycles_pct < 0 (s80 spends fewer cycles); dnll_per_1pct_cycles < 0 means "
                              "each +1% SC cycles lowers NLL by that many nats"},
        "vs_inc_reference": {"c17_minus_inc": E.paired(c17["window_nll"], ref_nll),
                             "s80_minus_inc": E.paired(s80["window_nll"], ref_nll),
                             "inc_reference_cost": inc_ref.cost},
        "audits": {a: {"ok": audits[a]["ok"], "file": str(out / f"{a}_audit.json")} for a in audits},
        "kappa_inputs_note": "m = pair.paired_s80_minus_c17.mean_dnll; the Fisher prediction p comes from the "
                             "round-7 capture diag (stage 'kappa')",
        "ok": True, "finished": time.time(),
    }
    E.write_json(out / "step0_measured.json", measured, exclusive=True)
    log(f"{cell['id']} measured: dNLL(s80-c17) {st['mean_dnll']:+.6f} (se {st['se']:.6f}) at "
        f"{dcyc:+.2f}% cycles -> {measured['pair']['dnll_per_1pct_cycles']:+.6f} nats per +1% cycles")
    return measured


# ---------------------------------------------------------------------------
# kappa stage (CPU; after the round-7 captures)
# ---------------------------------------------------------------------------
def _find_score_tables(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("score_tables", "score-tables") and isinstance(v, str) and "=" in v:
                return v
            found = _find_score_tables(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_score_tables(v)
            if found:
                return found
    return None


def _spec_map(spec: str, names=None) -> dict:
    """{name: sha256 of the TABLE} for a --score-tables spec (wrapper entries resolved to their table);
    only ``names`` when given (other entries, e.g. inc=..., are not opened)."""
    out = {}
    for part in [p for p in spec.split(",") if p]:
        name, path = part.split("=", 1)
        if names is not None and name not in names:
            continue
        t = E.read_json(path)
        tp = E.resolve_table_path(path) if "threshold_table_path" in t else Path(path)
        out[name] = E.sha256_file(tp)
    return out


def from_capture_manifest(path) -> tuple[dict, dict, dict]:
    """(diag paths, INTENDED score-table specs, identity.json paths) per cell from the calib side's
    round-7 capture manifest (cells[i].paths.diag / .identity, cells[i].r7.score_tables)."""
    cm = E.read_json(path)
    cells = cm["cells"] if isinstance(cm["cells"], list) else list(cm["cells"].values())
    diags, specs, idents = {}, {}, {}
    for c in cells:
        st = (c.get("r7") or {}).get("score_tables") or {}
        if c.get("paths", {}).get("diag") and st:
            diags[c["id"]] = c["paths"]["diag"]
            specs[c["id"]] = ",".join(f"{k}={v}" for k, v in st.items())
            if c["paths"].get("identity"):
                idents[c["id"]] = c["paths"]["identity"]
    return diags, specs, idents


def recorded_score_tables(diag_path, diag: dict):
    """The --score-tables the capture ACTUALLY ran with: the diag's own record when it carries one,
    else the prc6_calib record (calib7 args incl. score_tables) of the capture's emitted gfis table
    (<stem>_gfis_table.json next to <stem>_diag.json). Returns (spec, source) or (None, None)."""
    spec = _find_score_tables(diag)
    if spec:
        return spec, str(diag_path)
    p = Path(diag_path)
    if p.name.endswith("_diag.json"):
        g = p.with_name(p.name[:-len("_diag.json")] + "_gfis_table.json")
        if g.is_file():
            spec = (E.read_json(g).get("prc6_calib") or {}).get("score_tables")
            if isinstance(spec, str) and "=" in spec:
                return spec, str(g)
    return None, None


def capture_identity_status(path) -> dict:
    """identity.json of a round-7 capture (prc_r7_capture identity): identity_ok (a legacy record's ok when
    identity_ok is absent; the calib side keeps identity_ok true when only its pipeline check failed), level,
    sha."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return {"path": str(p) if p else None, "exists": False, "ok": False, "identity_level": None}
    d = E.read_json(p)
    ok = d["identity_ok"] if "identity_ok" in d else d.get("ok")
    return {"path": str(p), "exists": True, "ok": ok is True,
            "identity_level": d.get("identity_level"), "sha256": E.sha256_file(p),
            "pipeline_check_ok": (d.get("pipeline_check") or {}).get("ok"), "files": d.get("files")}


def _pred(diag: dict, name: str, key: str):
    """Fisher prediction of one scored table: calib9/10 diag rows (tables.<name>.<key>) or the
    round-7 solver's `score` output (preds.<name>.pred_dnll, full-table scorer)."""
    if "tables" in diag and name in diag["tables"]:
        return float(diag["tables"][name][key]), diag["tables"][name].get("L_over_parent"), f"tables.{name}.{key}"
    if "preds" in diag and name in diag["preds"]:
        p = diag["preds"][name]
        return float(p["pred_dnll"]), p.get("lin_cost_over_ref"), f"preds.{name}.pred_dnll"
    raise ValueError(f"no prediction for {name!r} in the capture diag / score output")


def step0_cell_integrity(root: Path, cid: str, md: dict) -> list:
    """Review flags for a rule cell whose step-0 job did not finish cleanly after writing
    step0_measured.json (the measurement itself is bracketed by identity_before / identity_mid)."""
    d = root / cid
    flags = []
    if (d / "failure.json").exists():
        f = E.read_json(d / "failure.json")
        flags.append(f"step0_cell_failed_after_measurement:{cid} ({f.get('type')}: {str(f.get('message'))[:160]})")
    summ = d / "step0_summary.json"
    if not summ.exists():
        flags.append(f"step0_cell_incomplete:{cid} (no step0_summary.json)")
    else:
        s = E.read_json(summ)
        if s.get("complete") is not True or s.get("identity", {}).get("restored_incumbent_exact") is not True:
            flags.append(f"step0_cell_incomplete:{cid} (step0_summary.json not complete / not restored)")
    return flags


def kappa_stage(m, capture_diags: dict, specs: dict, out_path, *, allow_simulated=False, out_root=None,
                capture_identities: dict | None = None) -> dict:
    """Pre-registered kappa decision. ``specs`` = the INTENDED score-table specs (capture manifest /
    CLI); the tables verified are the ones the capture RECORDS having scored, which must agree.
    ``capture_identities`` = identity.json per cell (default: next to the diag); a capture whose
    identity is missing or not ok contributes no prediction (the cell counts as missing)."""
    rule = m["kappa_rule"]
    root = Path(out_root or m["output_root"])
    capture_identities = dict(capture_identities or {})
    measured, preds, sources, review, integrity = {}, {}, {}, [], {}
    for cid in rule["cells"]:
        cell = m["cells"][cid]
        mp = root / cid / "step0_measured.json"
        if not mp.exists():
            continue
        md = E.read_json(mp)
        if md.get("simulated") and not allow_simulated:
            raise ValueError(f"{cid}: step0_measured.json is a SIMULATION")
        ok = md.get("ok") is True and (md.get("identity") or {}).get("identity_mid_exact") is True
        measured[cid] = {"window_dnll": md["pair"]["window_dnll"], "windows": md["windows"], "ok": ok,
                         "dcycles_pct": md["pair"]["dcycles_pct"]}
        integrity[cid] = step0_cell_integrity(root, cid, md)
        review += integrity[cid]
        if cid not in capture_diags:
            continue
        ident = capture_identity_status(capture_identities.get(cid) or Path(capture_diags[cid]).parent / "identity.json")
        if not ident["ok"]:
            review.append(f"capture_identity_not_ok:{cid} ({ident['path']}, exists {ident['exists']}, "
                          f"level {ident['identity_level']}) -> no prediction from this capture")
            sources[cid] = {"capture_diag": str(capture_diags[cid]), "capture_identity": ident, "p": None}
            continue
        diag = E.read_json(capture_diags[cid])
        names = rule["score_names"]
        spec, spec_src = recorded_score_tables(capture_diags[cid], diag)
        if not spec:
            raise ValueError(f"{cid}: the capture records no --score-tables (diag or emitted gfis table); "
                             "cannot verify which tables it scored")
        wanted_names = (names["c17"], names["s80"])
        smap = _spec_map(spec, wanted_names)
        if specs.get(cid):
            intended = _spec_map(specs[cid], wanted_names)
            for n in (names["c17"], names["s80"]):
                if intended.get(n) != smap.get(n):
                    raise ValueError(f"{cid}: recorded score table {n!r} differs from the intended one")
        want = {names["c17"]: E.sha256_file(cell["arms"]["C17"]["table"]),
                names["s80"]: E.sha256_file(cell["arms"]["S80"]["table"])}
        for n, h in want.items():
            if smap.get(n) != h:
                raise ValueError(f"{cid}: capture score table {n!r} is not the step-0 table (sha mismatch)")
        pk = rule["pred_key"]
        p_c17, lo_c17, f_c17 = _pred(diag, names["c17"], pk)
        p_s80, lo_s80, f_s80 = _pred(diag, names["s80"], pk)
        preds[cid] = p_s80 - p_c17
        lo = {names["c17"]: lo_c17, names["s80"]: lo_s80}
        sources[cid] = {"capture_diag": str(capture_diags[cid]), "capture_diag_sha256": E.sha256_file(capture_diags[cid]),
                        "capture_identity": ident, "score_tables_recorded": spec,
                        "score_tables_recorded_source": spec_src, "score_tables_intended": specs.get(cid),
                        "pred_c17": p_c17, "pred_s80": p_s80, "p": preds[cid],
                        "pred_fields": [f_c17, f_s80], "L_over_parent": lo,
                        "pred_dlinear_L_pct": (100.0 * (lo[names["s80"]] / lo[names["c17"]] - 1.0))
                        if all(lo.values()) else None}
    dec = E.kappa_decision(measured, preds, rule)
    if review:
        dec["flags"] = dec["flags"] + review
        dec["requires_user_review"] = True
    for cid, s in sources.items():
        pc = dec["per_cell"].get(cid, {})
        s["measured_dnll_per_1pct_cycles"] = (pc.get("m") / measured[cid]["dcycles_pct"]) if pc else None
        s["fisher_pred_dnll_per_1pct_linear_L"] = (s["p"] / s["pred_dlinear_L_pct"]) \
            if s.get("p") is not None and s.get("pred_dlinear_L_pct") else None
    pooled = dec.get("pooled") or {}
    calib_equiv = ({"meas_dnll": pooled["sum_m_per_cell_mean"], "meas_se": pooled["sd_window_sum"] /
                    math.sqrt(pooled["n_windows"]), "pred_dnll": pooled["sum_p"],
                    "kappa_att": rule["kappa_att"], "kappa_att_se": rule["kappa_att_se"],
                    "ratio": rule["primary_ratio"],
                    "note": "the same pooled inputs for prc_r7_solve.py kappa-decision (identical rule); "
                            "informational only - this file is the authoritative decision"}
                   if pooled.get("sum_p") else None)
    rec = {"schema": SCHEMA + "-kappa", "run_id": m["run_id"], "created": time.time(),
           "authority": rule.get("authority"),
           "calib_side_equivalent_inputs": calib_equiv,
           "decision": dec["decision"], "ratio": dec["ratio"], "kappa_lin_bs": dec["kappa_lin_bs"],
           "se_kappa_lin_bs": dec["se_kappa_lin_bs"], "band": dec["band"],
           "requires_user_review": dec["requires_user_review"], "flags": dec["flags"],
           "per_cell": dec["per_cell"], "pooled": dec.get("pooled"),
           "sensitivity_se_bs_only": dec.get("sensitivity_se_bs_only"),
           "implied_ratio_measured": dec["implied_ratio_measured"],
           "prediction_sources": sources, "rule": rule, "step0_integrity": integrity,
           "simulated_inputs": bool(allow_simulated),
           "measured_files": {cid: str(root / cid / "step0_measured.json") for cid in measured},
           "use": "round-7 solves: attention bins' error scaled by r = ratio for UK/K arms (U always r = 1); "
                  "the absolute (kappa_lin, kappa_att) per decision branch are pinned in "
                  "prc_r7_prereg_20260928.json"}
    if out_path:
        E.write_json(out_path, rec, exclusive=True)
    log(f"kappa: kappa_lin_bs={dec['kappa_lin_bs']} (se {dec['se_kappa_lin_bs']}, band {dec['band']}) -> "
        f"{dec['decision']} (ratio {dec['ratio']}); flags {dec['flags']}")
    return rec


def sim_evaluator(cell, out, *, s80_delta=0.012, c17_delta=-0.004):
    """CPU dry-run evaluator for one step-0 cell (synthetic NLLs, archived stand-in traces)."""
    arms, sim = cell["arms"], cell["simulation"]
    ref = cell["identity"].get("reference")
    deltas = {arms["PARENT"]["wrapper"]: 0.03}
    prc_shift = {}
    if "S80" in arms:
        deltas.update({arms["S80"]["wrapper"]: c17_delta + s80_delta, arms["C17"]["wrapper"]: c17_delta})
        prc_shift[arms["S80"]["wrapper"]] = 2
    return E.SimEvaluator(
        cell, out, inc_wrapper=arms["INC"]["wrapper"], inc_standin=sim["inc_standin"],
        nll_ref=dict(zip([int(s) for s in cell["windows"]["starts"]], ref["window_nll"])) if ref else {},
        token_sha=cell["windows"]["token_ids_sha256"], parent_wrapper=arms["PARENT"]["wrapper"],
        parent_standin=None,  # synthesized from the INC stand-in so MACs match the INC windows
        deltas=deltas, prc_shift=prc_shift,
        nll_exact={(arms[p["arm"]]["wrapper"], p["start"]): p["expected"] for p in cell["probes"]})


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("preflight", "run", "simulate", "kappa"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", type=int)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--capture-diag", action="append", default=[])
    ap.add_argument("--capture-manifest", default=None,
                    help="calib-side prc_r7_capture manifest: diag / identity.json paths + intended score tables")
    ap.add_argument("--capture-identity", action="append", default=[],
                    help="CELL=identity.json (default: the capture manifest's, else next to the diag)")
    ap.add_argument("--score-tables-spec", action="append", default=[],
                    help="CELL=SPEC: intended score tables (must agree with what the capture recorded)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-simulated", action="store_true")
    args = ap.parse_args()
    m = load_manifest(args.manifest)
    if args.stage == "kappa":
        verify_code_hashes(m)
        diags, specs, idents = (from_capture_manifest(args.capture_manifest) if args.capture_manifest
                                else ({}, {}, {}))
        diags.update(dict(x.split("=", 1) for x in args.capture_diag))
        specs.update(dict(x.split("=", 1) for x in args.score_tables_spec))
        idents.update(dict(x.split("=", 1) for x in args.capture_identity))
        diags = {c: d for c, d in diags.items() if c in m["kappa_rule"]["cells"]}
        if args.allow_simulated and not args.out:
            raise ValueError("a simulated kappa decision must be written to an explicit scratch --out")
        out = args.out or str(Path(m["output_root"]) / "kappa_decision.json")
        kappa_stage(m, diags, specs, out, allow_simulated=args.allow_simulated, out_root=args.out_root,
                    capture_identities=idents)
        return 0
    out_root = Path(args.out_root or m["output_root"]).resolve()
    if args.stage == "run" and str(out_root) != m["output_root"]:
        raise ValueError("GPU runs must write to the manifest output_root")
    if args.stage == "simulate" and str(out_root).startswith(str(E.R7_ROOT)):
        raise ValueError("simulation must not write under the Turbo round-7 root")
    report = preflight(m, args.task, out_root, deep=args.deep)
    if args.stage == "preflight":
        print(json.dumps({"preflight": "PASS", "task": m["tasks"][args.task]["id"], "cells": report}, indent=1))
        return 0
    for cid in m["tasks"][args.task]["cells"]:
        cell = m["cells"][cid]
        out = out_root / cid
        if not out.is_dir():
            if args.stage == "simulate":
                out.mkdir(parents=True)
            else:
                raise ValueError(f"launcher must create the cell directory first: {out}")
        if args.stage == "simulate":
            ev = sim_evaluator(cell, out)
        else:
            ev = E.Evaluator(cell, out, build_wrapper=cell["arms"]["INC"]["wrapper"], tag=TAG)
        try:
            run_cell(m, cell, out, ev)
        except BaseException as exc:
            E.write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                "traceback": traceback.format_exc(), "time": time.time()})
            raise
        finally:
            ev.close()
    verify_code_hashes(m)
    log(f"task {m['tasks'][args.task]['id']} COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
