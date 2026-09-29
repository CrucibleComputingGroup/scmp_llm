"""Build the frozen round-7 SCREEN manifest from the round-7 solves (CPU only).

Real builds take the round-7 solver output plus the pre-registered fixed point, never hand-picked
values:
  python benchmark/ppl/kbands/build_prc_screen_r7_20260928.py \
      --from-solve 30B_t40=<s1 solve_summary.json> [--from-solve 30B_t40=<another summary> ...] \
      --fixed-point 30B_t40=<prc_r7_20260928/fixed_point/30B_t40_fixed_point.json> ... \
      --kappa-decision <prc_r7_20260928/step0/kappa_decision.json> [--accept-kappa-review] [--overwrite-manifest]
Every arm is re-derived and re-verified from its solve summary (sha-locked) and the fixed point:
  * PRE-REGISTRATION (benchmark/ppl/prc_r7_prereg_20260928.py + .json, frozen before any solve):
    MDE per cell (no override exists; the summary's mde_nats must equal it), the absolute
    (kappa_lin, kappa_att) of the authoritative kappa decision's branch (dense: 1/1), the "up"
    primary ladder policy, and the one target-scale rule. The record must predate every summary,
    fixed point and kappa decision it is used with.
  * TARGET SCALE: each arm's table must be the one solved at ITS OWN pre-registered s1 from the
    fixed-point record (replay of the s0 table on the step-0 INC16 profile with the step-0 P);
    any other scale is refused; no re-screen exists.
  * CAPTURE IDENTITY: <capture>/identity.json must be ok (level exact or near; near is flagged)
    and the solve's capture files must be exactly the identity-checked ones.
  * INC: every summary's inc_table_sha256 must be the best(all) incumbent table (pred_true is vs
    INC; 30B t40's INC is candidate07, not c7_gfisla), which must also be step 0's INC.
Rules applied here (pre-registered; the manifest records them):
  * primary per cell from the ROUND-6 GATE: 30B cell gate passed -> kind "UK", failed/missing -> kind "K";
    30B t48 follows 30B t40's gate; 4B t40/t64 are enabled only if their own gate passed (primary kind
    "U", kappa 1; a "UK" arm on a dense cell IS the U arm and is refused). Cells absent from the spec
    are disabled. The primary kind is processed FIRST, so it can never be dropped as a duplicate.
  * refusal: an arm is refused unless pred_true <= -MDE; a refused primary disables the cell.
  * <= 2 secondaries (spec order), allocation-duplicates of an earlier arm dropped.
  * every arm: lineage vs INC (allocation fields only, ladders <= 128), runtime resolver sweep, escape
    disclosure, unique full-test trace name, replayed C(s1)/P (recorded; flagged beyond 0.25%).
  * windows: 30B t32/t40 screen on the round-6 diag windows (INC reference = round-4 confirm, bit identity),
    30B t48 on the 16 fresh registry windows (INC reference = the step-0 INC16 run + held-out probe),
    4B on the round-6 4B diag windows (INC reference = round-6 INC run); confirmation = the 32 registry
    windows for every cell. P = the step-0 parent_reference.json on the same windows.
Dry runs (--standins) synthesize STAND-IN candidate tables from INC in a scratch directory and write a
manifest marked dry_run (the launcher refuses to run it for real); they skip the solve/fixed-point/
identity/step-0 checks and say so in the manifest.

  python benchmark/ppl/kbands/build_prc_screen_r7_20260928.py --standins SCRATCH_DIR \
      [--assume-gate 30B_t32=pass ...] [--assume-kappa-ratio 0.42] [--enable-4b]
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.ppl import prc_eval_r7_20260928 as E  # noqa: E402
from benchmark.ppl import prc_fixedpoint_r7_20260928 as F  # noqa: E402
from benchmark.ppl import prc_r7_prereg_20260928 as PR  # noqa: E402
from benchmark.ppl import prc_screen_r7_20260928 as SC  # noqa: E402  (realized-length audit prediction)

RUN_ID = "prc_screen_r7_20260928"
KDIR = REPO / "benchmark/ppl/kbands"
MANIFEST = KDIR / "prc_screen_r7_20260928.json"
REGISTRY = KDIR / "prc_windows_r7_20260928.json"
STEP0_MANIFEST = KDIR / "prc_step0_r7_20260928.json"
CAPTURE_MANIFEST = KDIR / "prc_r7_capture_20260928.json"
OUTPUT_ROOT = E.R7_ROOT / "screen"
STEP0_ROOT = E.R7_ROOT / "step0"
K = E.KB
NEW_SOURCES = [
    "benchmark/ppl/prc_eval_r7_20260928.py",
    "benchmark/ppl/prc_screen_r7_20260928.py",
    "benchmark/ppl/prc_fixedpoint_r7_20260928.py",
    "benchmark/ppl/prc_r7_prereg_20260928.py",
    "benchmark/ppl/test_prc_eval_r7_20260928.py",
    "benchmark/ppl/test_prc_screen_r7_20260928.py",
    "benchmark/ppl/test_prc_screen_audit_r7_20260928.py",
    "benchmark/ppl/test_prc_r7_stage2_cpu_20260928.py",
    "benchmark/ppl/kbands/build_prc_screen_r7_20260928.py",
    "benchmark/ppl/kbands/run_prc_screen_r7_20260928.sbatch",
    "benchmark/ppl/kbands/run_r7_stage2_cpu_20260928.py",
    "benchmark/ppl/kbands/run_r7_stage2_cpu_20260928.sh",
    "benchmark/ppl/kbands/run_prc_ppl.sbatch",
]
FROZEN_IMPORTED = [
    "benchmark/ppl/prc_r6_attn_diag_arms.py",
    "benchmark/ppl/prc_r6_retarget_20260928.py",
    "benchmark/ppl/prc_local_refine.py",
    "benchmark/ppl/prc_local_proposals.py",
    "model/sc_common.py",
    "kernels/scmp_kernels/mp/config.py",
    "kernels/scmp_kernels/sc/matmul.py",
    "kernels/scmp_kernels/sc/kernels.py",
    "kernels/scmp_kernels/trace.py",
    "loader.py",
    "model/awq_apply.py",
    "benchmark/quant/eval_quant.py",
]
CELLS = [
    dict(id="30B_t32", model="30B", target=32, screen="r6diag30B", gate_cell="30B_t32"),
    dict(id="30B_t40", model="30B", target=40, screen="r6diag30B", gate_cell="30B_t40"),
    dict(id="30B_t48", model="30B", target=48, screen="fresh", gate_cell="30B_t40"),
    dict(id="4B_t40", model="4B", target=40, screen="r6diag4B", gate_cell="4B_t40"),
    dict(id="4B_t64", model="4B", target=64, screen="r6diag4B", gate_cell="4B_t64"),
]
MAIN_KINDS = PR.ARMS
KAPPA_RATIO_TOL = PR.KAPPA_RATIO_TOL
CANDIDATES = KDIR / "prc_candidates_r7_20260928.json"
KIND_RE = re.compile(r"^(UK|K|U)(_(L6|L7|L6L7))?$")
PREREGISTERED = {
    "primary": "One pre-registered primary per cell from the round-6 gate: 30B gate passed -> UK (unpinned "
               "ladders up to 128, kappa-corrected currency), failed or missing -> K (inherited ladders, "
               "kappa-corrected); 30B t48 follows 30B t40's gate; 4B cells only if their own gate passed "
               "(primary U, kappa 1; a UK arm on a dense cell is the same allocation as U and is refused). U "
               "(unpinned, kappa 1) and other listed arms are secondary: screened for information only, never "
               "confirmed or full-tested. The primary kind is processed first (never dropped as a duplicate).",
    "mde_refusal": "An arm is screened only if its pre-registered pred_true = kappa_lin*dL + kappa_att*dA "
                   "(nats vs INC; absolute kappa pinned per kappa-decision branch in prc_r7_prereg_20260928.json) "
                   "satisfies pred_true <= -MDE, with MDE the frozen per-cell value of that record (no override); "
                   "a refused primary disables the cell. " + PR.MDE_RULE,
    "kappa": PR.KAPPA_AUTHORITY + " " + PR.KAPPA_RULE_TEXT,
    "target_scale": PR.FIXED_POINT["rule"],
    "capture_identity": "Every screened table comes from a capture whose identity record (the capture manifest's "
                        "identity.json, or a re-tagged record in the same capture directory) says identity_ok = "
                        "true, whose files equal the solve summary's capture record, and which is the record the "
                        "solve itself read; 'near' (a stop under the v2 capture manifest) and a failed pipeline "
                        "check are flagged.",
    "screen_gate": "The primary proceeds to confirmation iff (a) every identity check and audit passed, (b) "
                   "|C/P - 1| <= 1% on the 16 screen windows (C, P exact trace costs of the primary and the "
                   "parent on the same held-out TRAIN windows; P from step 0), and (c) its paired mean dNLL vs "
                   "the re-evaluated INC on those windows is < 0.",
    "confirm_gate": "Full-protocol test iff the paired z = mean/SE of dNLL(primary - INC) over 32 fresh disjoint "
                    "TRAIN windows is < -2 and every audit passed.",
    "best_all_entry": PR.BEST_ALL_RULE + " best(all) = lowest eligible full-test PPL across comparable "
                      "allocation rounds, each candidate keeping its provenance.",
    "disclosure": "Round-7 = one algorithm update: joint linear+attention lambda with a family-calibrated "
                  "currency (kappa from held-out data, step-0 checked) over attention ladders that may reach "
                  "L_max = 128 (halved), re-solved at the parent's cost. Reported as per-row attention "
                  "reallocation plus calibration objective, NOT as a T1 granularity effect. Every round-7 table "
                  "also re-targets from INC's cost (0.45-1.0% below the parent on test) to the parent's cost: "
                  "that part of any gain is a budget correction and is disclosed as such. Escape: mu+2tau "
                  "semantics are unchanged; the effect of a 128 rung is reported PER BUCKET from each arm's "
                  "escape_disclosure (qk: escaped rows fold onto the 128 rung and the gate adds nothing where the "
                  "calibrated 128 threshold lies at or below mu+2tau; av: mu+2tau >= 1, the gate never fires and "
                  "128 is a genuinely new rung), never as a blanket 'no-op'. For 30B t32/t40 the kappa step-0 "
                  "measurement used the same 16 windows as the screen (screen optimistic by construction; the "
                  "confirmation windows are fresh). Attention MACs counted dense.",
    "fail_closed": "Any identity mismatch (INC vs stored reference or held-out probe; unprofiled vs profiled vs "
                   "restored INC) or any screened arm's audit failure stops the cell with failure.json (plus "
                   "<arm>_screen_audit_failure.json); no decision, no test follows.",
    "realized_length_audit": "REGISTERED (prc_r7_capture_20260928.json preregistered_rules.realized_length_audit, "
                             "verbatim): " + SC.REALIZED_LENGTH_RULE + " Implementation: every screened arm's GPU "
                             "trace is audited (runtime resolver + JSON mirror ladders or the protected length, max "
                             "128, sc_prec 8 / halve / rng_levels 128, SC (op, block) set == the INT mask's, MACs == "
                             "INC, and each new 112/128 rung's realized MAC share per attention bucket within +-0.10 "
                             "absolute of realized_length_prediction = the arm's solve-summary att_mac_hist_hold, "
                             "re-derived by the driver from the sha-locked summary) before any candidate NLL is "
                             "released; the evaluator's per-window NLL log lines are masked meanwhile. Descriptive "
                             "(non-binding) on the confirmation and full-test traces.",
}


def write_exact(path: Path, data: bytes):
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"refusing to replace a different existing file: {path}")
        return "unchanged"
    with path.open("xb") as f:
        f.write(data)
    return "created"


def r6_gate(gate_cell, assume):
    if gate_cell in assume:
        return {"source_cell": gate_cell, "passed": assume[gate_cell] == "pass", "assumed": True,
                "detail": "ASSUMED for a dry run; never valid for a real manifest"}
    d = E.R6_DIAG_ROOT / gate_cell
    summ = d / "diag_summary.json"
    if (d / "failure.json").exists() or not summ.exists():
        return {"source_cell": gate_cell, "passed": False, "assumed": False, "summary": None,
                "detail": "round-6 diagnostic missing or failed -> gate treated as failed"}
    s = E.read_json(summ)
    g = s["round7_gate"]["binding"]
    ok = bool(s.get("complete") is True and s.get("simulated") is False
              and s["identity"].get("restored_incumbent_exact") is True
              and g["proceed_round7_attention_ladder_solve"] is True)
    return {"source_cell": gate_cell, "passed": ok, "assumed": False, "summary": str(summ),
            "summary_sha256": E.sha256_file(summ), "binding": g}


def step0_parent_cost(cid, screen, root=STEP0_ROOT) -> dict:
    pref_path = Path(root) / cid / "parent_reference.json"
    pr = E.read_json(pref_path)
    if not pr.get("audit_ok") or pr.get("simulated") is not False or pr["windows"] != screen:
        raise F.FixedPointError(f"{cid}: step-0 P is not an audited GPU measurement on the screen windows")
    return {"mode": "reference", "file": str(pref_path), "trace": pr["trace"], "P": pr["P"],
            "trace_sha256": pr["trace_sha256"]}


def _disabled(base, reason, **extra):
    return {**base, **extra, "enabled": False, "disabled_reason": reason}


def build_cell(spec, cspec, reg, rows, m6, kappa, gate, *, dry_run, do_sweep=True, prereg=None,
               step0_root=STEP0_ROOT, capture_manifest=CAPTURE_MANIFEST, step0_manifest=STEP0_MANIFEST) -> dict:
    cid, model, T = spec["id"], spec["model"], spec["target"]
    nb = E.TOTAL_BLOCKS[model]
    row = rows[(model, T)]
    inc_wrapper = Path(row["wrapper"])
    inc_table = E.resolve_table_path(inc_wrapper)
    parent_dir = E.MP_BEST / f"configs/{model}/target{T}"
    hybrid = parent_dir / "hybrid_config.json"
    awq = E.MP_BEST / f"act_scales/awq_scales/awq_scales_{E.MODEL_PATHS[model].replace('/', '_')}_b4.pt"
    parent_trace = K / f"kbands_20260801/ppl/{model}_t{T}_parent_trace.json"
    base = {"id": cid, "model": model, "target": T, "nominal_cycles": 2 * T, "model_path": E.MODEL_PATHS[model],
            "total_blocks": nb, "layer_buckets": 4, "hybrid_config": str(hybrid), "awq_cache": str(awq),
            "r6_gate": gate, "mde_nats": PR.MDE_NATS.get(cid)}
    if cspec is None:
        return _disabled(base, "not in the candidate spec")
    if model == "4B" and not gate["passed"]:
        return _disabled(base, "4B cells run only if their round-6 gate passed")
    inc_w, inc_t = E.read_json(inc_wrapper), E.read_json(inc_table)
    arms = {"INC": {"wrapper": str(inc_wrapper), "table": str(inc_table), "role": "incumbent", "kind": "INC"},
            "PARENT": {"wrapper": str(parent_dir / "wrapper.json"), "table": str(parent_dir / "table.json"),
                       "role": "parent", "kind": "PARENT"}}
    for n in ("INC", "PARENT"):
        E.check_wrapper_matches_inc(E.read_json(arms[n]["wrapper"]), inc_w, n)
    want_primary = ("UK" if gate["passed"] else "K") if model == "30B" else "U"
    flags, extra_files = [], []
    # ---- real builds: step-0 reference, capture identity, fixed point, kappa pins
    ref0 = ident = fp = pins = None
    if not dry_run:
        if prereg is None:
            raise SystemExit("a real build needs the verified pre-registration record")
        s0m = E.read_json(step0_manifest)
        if cid not in s0m["cells"]:
            return _disabled(base, "no step-0 reference cell -> no pre-registered fixed point")
        if Path(s0m["cells"][cid]["arms"]["INC"]["wrapper"]).resolve() != inc_wrapper.resolve():
            return _disabled(base, "the best(all) incumbent changed since step 0")
        try:
            ref0 = F.step0_reference(cid, root=step0_root, step0_manifest=step0_manifest)
        except (F.FixedPointError, FileNotFoundError) as exc:
            return _disabled(base, f"no complete step-0 reference: {exc}")
        pins = PR.kappa_pins_for(model, kappa.get("decision") if model == "30B" else None)
        fpp = cspec.get("fixed_point")
        if not fpp:
            return _disabled(base, "no fixed-point record in the spec")
        fp = F.load_fixed_point(fpp, cid)
        try:
            ident = F.capture_identity(cid, capture_manifest=capture_manifest, path=fp["capture_identity"]["path"])
        except (F.FixedPointError, FileNotFoundError, KeyError) as exc:
            return _disabled(base, f"capture identity not ok: {exc}")
        if ident["sha256"] != fp["capture_identity"]["sha256"]:
            raise SystemExit(f"{cid}: capture identity record changed since the fixed point")
        if ident["near_flag"]:
            flags.append(f"{cid}: capture identity_level 'near' (disclosed)")
        if ident["pipeline_check_ok"] is False:
            flags.append(f"{cid}: capture pipeline check failed (identity ok); every screened table is re-validated "
                         "here (lineage + resolver sweep)")
        fp_bad = []
        if fp["prereg"]["sha256"] != prereg["sha256"]:
            fp_bad.append("fixed point used a different pre-registration record")
        if fp["s0"] != PR.S0[cid] or fp["mde_nats"] != PR.MDE_NATS[cid]:
            fp_bad.append("fixed point s0 / MDE differ from the pre-registration")
        if {k: fp["kappa_pins"][k] for k in ("kappa_lin", "kappa_att", "branch")} != pins:
            fp_bad.append(f"fixed point kappa pins {fp['kappa_pins']} != {pins}")
        if fp["P"] != ref0["P"] or fp["U"] != ref0["U_prot"]:
            fp_bad.append("fixed point P / U differ from the step-0 reference")
        capf = {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in fp["capture_files"].items()}
        if capf != {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in (ident["files"] or {}).items()}:
            fp_bad.append("fixed point s0 solve is not from the identity-checked capture")
        if float(fp["created"]) <= prereg["frozen_t"]:
            fp_bad.append("fixed point predates the pre-registration record")
        if model == "30B" and (fp.get("kappa_decision") or {}).get("sha256") != kappa.get("sha256"):
            fp_bad.append("fixed point used a different kappa decision")
        if fp_bad:
            raise SystemExit(f"{cid}: fixed-point record refused: {fp_bad}")
        extra_files += [fpp, ident["path"], prereg["path"]] + [v["path"] for v in ref0["files"].values()]
        if model == "30B":
            extra_files.append(kappa["path"])
    ratio = kappa["ratio"] if model == "30B" else 1.0
    refused, notes, order, sigs = {}, [], [], {}
    spec_arms = sorted(cspec["arms"], key=lambda a: (0 if a["kind"] == want_primary else 1))  # primary first
    for a in spec_arms:
        name, kind = a["name"], a["kind"]
        if not re.fullmatch(r"[A-Za-z0-9]+", name) or name in ("INC", "PARENT") or name in arms:
            raise SystemExit(f"{cid}: bad or duplicate arm name {name!r}")
        mk = KIND_RE.fullmatch(kind)
        if not mk:
            raise SystemExit(f"{cid}: unknown arm kind {kind!r}")
        main = mk.group(1)
        if model != "30B" and main == "UK":
            refused[name] = "dense cell: kappa = 1 makes UK the same allocation as U (U is the dense primary)"
            continue
        if not dry_run and kind not in MAIN_KINDS:
            refused[name] = "fold-in arm: no pre-registered fixed point / MDE path exists for it"
            continue
        want_r = 1.0 if (main == "U" or model != "30B") else ratio
        if abs(float(a["kappa_ratio"]) - want_r) > KAPPA_RATIO_TOL:
            raise SystemExit(f"{cid} {name}: kappa_ratio {a['kappa_ratio']} != required {want_r} "
                             f"(tolerance {KAPPA_RATIO_TOL})")
        pred_true, mde, target_scale, fp_arm = float(a["pred_true_nats"]), float(a["mde_nats"]), a.get("target_scale"), None
        if not dry_run:
            # re-derive everything from the sha-locked solve summary; the spec copy must agree
            det = a.get("pred_detail") or {}
            sp = det.get("solve_summary")
            if not sp or E.sha256_file(sp) != det.get("solve_summary_sha256"):
                raise SystemExit(f"{cid} {name}: solve summary missing or changed since the spec was written")
            sm = E.read_json(sp)
            bad = F.check_solve_summary(sm, sp, cid, pins=pins, inc_table=inc_table, identity=ident,
                                        prereg_frozen_t=prereg["frozen_t"], want_primary=want_primary)
            if bad:
                raise SystemExit(f"{cid} {name}: solve summary refused: {bad[:6]}")
            rec = sm["arms"][det["table_name"]]
            if rec["arm"] != kind or Path(rec["wrapper"]).resolve() != Path(a["wrapper"]).resolve():
                raise SystemExit(f"{cid} {name}: spec arm is not the summary's {det['table_name']}")
            if (float(rec["pred_true_vs_inc"]), float(sm["mde_nats"]), float(rec["kappa_ratio_used"])) != \
                    (pred_true, mde, float(a["kappa_ratio"])) or mde != PR.MDE_NATS[cid]:
                raise SystemExit(f"{cid} {name}: spec pred_true / MDE / kappa ratio differ from the solve summary")
            fp_arm = fp["arms"].get(kind) or {}
            if fp_arm.get("status") != "ok":
                refused[name] = f"no pre-registered s1 for {kind} ({fp_arm.get('status')})"
                continue
            if abs(float(rec["target_scale"]) - float(fp_arm["s1"])) > 5e-9 or target_scale != rec["target_scale"]:
                refused[name] = (f"target_scale {rec['target_scale']} != its pre-registered fixed-point s1 "
                                 f"{fp_arm['s1']}")
                continue
            extra_files.append(sp)
        w = Path(a["wrapper"])
        t = E.resolve_table_path(w)
        wd = E.read_json(w)
        E.check_wrapper_matches_inc(wd, inc_w, name)
        td = E.read_json(t)
        allow_psl = bool(a.get("allow_psl_change")) and "L7" in kind
        allow_rekey = bool(a.get("allow_layer_rekey")) and "L6" in kind
        lin = E.validate_lineage(inc_t, td, total_blocks=nb, allow_psl_change=allow_psl, allow_layer_rekey=allow_rekey)
        if not lin["ok"]:
            raise SystemExit(f"{cid} {name}: lineage vs INC failed: {lin['failures'][:5]}")
        if "candidate is allocation-identical to INC" in lin["notes"]:
            refused[name] = "allocation-identical to INC"
            continue
        if not E.pred_passes_mde(pred_true, mde):
            refused[name] = f"pred_true {pred_true} > -MDE ({mde})"
            continue
        sig = E.allocation_signature(td)
        if sig in sigs:
            refused[name] = f"allocation-duplicate of {sigs[sig]}"
            continue
        sigs[sig] = name
        kb_tbl = f"r7{name}20260928"
        tt = K / f"kbands_20260801/ppl/{model}_t{T}_prc_{kb_tbl}_trace.json"
        if tt.exists():
            raise SystemExit(f"{cid} {name}: test trace {tt} already exists")
        sweep = E.resolver_sweep(w, inc_wrapper, nb, allow_psl_change=allow_psl)["checked"] if do_sweep else None
        # registered realized-length audit: each new 112/128 rung's predicted MAC share from the solve's hold windows
        if dry_run:
            rlp = SC.standin_prediction("dry-run stand-in table: no solve summary exists")
        else:
            try:
                rlp = SC.new_rung_prediction(rec, td, inc_t, total_blocks=nb, layer_buckets=base["layer_buckets"],
                                             source={"solve_summary": str(sp), "solve_summary_sha256": E.sha256_file(sp),
                                                     "table_name": det["table_name"]})
            except E.AuditError as exc:
                raise SystemExit(f"{cid} {name}: the registered realized-length prediction cannot be derived: {exc}")
        replay = None
        if not dry_run:
            rp = F.replay_cost(td, E.read_json(ref0["profile"]), prot_macs=ref0["prot_macs"],
                               psl_inc=F.psl_of(inc_t), psl_arm=F.psl_of(td))
            ratio_s1 = rp["cost"] / ref0["P"]
            replay = {"C_replay_s1": rp["cost"], "C_replay_s1_over_P": ratio_s1, "P": ref0["P"],
                      "flag": abs(ratio_s1 - 1) > PR.FIXED_POINT["replay_flag_tolerance"]}
            if replay["flag"]:
                flags.append(f"{cid} {name}: replayed C(s1)/P = {ratio_s1:.5f} (outside +-0.25%; non-binding)")
        arms[name] = {"wrapper": str(w), "table": str(t), "kind": kind, "role": None,
                      "pred_true_nats": pred_true, "mde_nats": mde,
                      "kappa_ratio": float(a["kappa_ratio"]), "kappa_pins": pins, "pred_detail": a.get("pred_detail"),
                      "target_scale": target_scale, "fixed_point_arm": fp_arm, "replay_at_s1": replay,
                      "predicted_cost_ratio": a.get("predicted_cost_ratio"),
                      "allow_psl_change": allow_psl, "allow_layer_rekey": allow_rekey,
                      "lineage": lin, "resolver_sweep": sweep, "realized_length_prediction": rlp,
                      "escape_disclosure": E.escape_disclosure(td, wd, nb),
                      "kb_tbl": kb_tbl, "test_trace": str(tt)}
        order.append(name)
    extra = {"flags": flags, "capture_identity": ident and {k: ident[k] for k in ("path", "sha256", "identity_level",
                                                                                   "near_flag", "pipeline_check_ok")},
             "fixed_point": fp and {"path": cspec.get("fixed_point"), "sha256": E.sha256_file(cspec["fixed_point"]),
                                    "s0": fp["s0"], "P": fp["P"], "U": fp["U"],
                                    "s1": {k: v.get("s1") for k, v in fp["arms"].items()}},
             "kappa_pins": pins}
    prim = [n for n in order if arms[n]["kind"] == want_primary]
    if not prim:
        why = refused.get(next((a["name"] for a in spec_arms if a["kind"] == want_primary), ""), "absent")
        return _disabled(base, f"required primary kind {want_primary} not eligible ({why})", refused=refused, **extra)
    if len(prim) > 1:
        raise SystemExit(f"{cid}: more than one arm of the primary kind {want_primary}")
    primary = prim[0]
    secondaries = [n for n in order if n != primary][:2]
    for n in [n for n in order if n != primary][2:]:
        notes.append(f"{n} not screened (cap: primary + 2 secondaries)")
        arms.pop(n)
    arms[primary]["role"] = "primary"
    for n in secondaries:
        arms[n]["role"] = "secondary"
    # ---- windows and identity
    capture = [int(s) for s in cspec.get("capture_windows") or reg["capture_windows_expected"]]
    pc = {"mode": "evaluate"}
    if spec["screen"] == "r6diag30B":
        screen = reg["r6_diag_windows_30B"]
        r6c = m6["cells"][cid]
        ident6 = r6c["identity"]
        nll = E.read_json(ident6["nll_file"])
        if nll["window_nll"] != ident6["window_nll"] or r6c["windows"]["starts"] != screen:
            raise SystemExit(f"{cid}: round-4 reference mismatch")
        ref = {"nll_file": ident6["nll_file"], "trace_file": ident6["trace_file"], "window_nll": ident6["window_nll"],
               "total_macs": ident6["total_macs"], "total_cycle_macs": ident6["total_cycle_macs"],
               "source": "round-4 confirm_incumbent"}
        r6nll = E.R6_DIAG_ROOT / cid / "INC_nll.json"
        if r6nll.exists() and E.read_json(r6nll)["window_nll"] != ident6["window_nll"]:
            raise SystemExit(f"{cid}: round-6 INC NLLs differ from round 4 on the same windows")
        probes = []
    elif spec["screen"] == "r6diag4B":
        screen = reg["r6_diag_windows_4B"]
        r6nll, r6tr = E.R6_DIAG_ROOT / cid / "INC_nll.json", E.R6_DIAG_ROOT / cid / "INC_trace.json"
        if r6nll.exists() and r6tr.exists():
            d = E.read_json(r6nll)
            if d.get("window_starts") != screen:
                raise SystemExit(f"{cid}: round-6 INC_nll.json window_starts differ from the 4B screen windows")
            ref = {"nll_file": str(r6nll), "trace_file": str(r6tr), "window_nll": d["window_nll"],
                   "total_macs": d["total_macs"], "total_cycle_macs": d["total_cycle_macs"],
                   "source": "round-6 INC run"}
        else:
            ref = None
        p = m6["cells"][cid]["identity"]
        probes = [{"name": "probe_INC_30720", "arm": "INC", "start": p["start"], "expected": p["nll"],
                   "source": f"{p['file']}:tables.{p['table_name']}.window_nll[0]"}]
    else:
        screen = reg["screen16_fresh_t48"]
        ref = None
        held = K / f"prc2/{cid}_c7_heldout_nll.json"
        hd = E.read_json(held)
        ent = next(v for v in hd["tables"].values() if Path(v["wrapper"]).resolve() == inc_wrapper.resolve())
        tname = next(k for k, v in hd["tables"].items() if Path(v["wrapper"]).resolve() == inc_wrapper.resolve())
        if hd["windows"][0] != 30720:
            raise SystemExit(f"{cid}: held-out window 1 is not 30720")
        probes = [{"name": "probe_INC_30720", "arm": "INC", "start": 30720, "expected": ent["window_nll"][0],
                   "source": f"{held}:tables.{tname}.window_nll[0]"}]
        if not dry_run:
            inc16 = E.read_json(Path(ref0["dir"]) / "INC16_nll.json")
            if inc16["window_starts"] != screen:
                raise SystemExit(f"{cid}: step-0 INC16 windows differ from the t48 screen windows")
            ref = {"nll_file": str(Path(ref0["dir"]) / "INC16_nll.json"), "trace_file": inc16["trace"],
                   "window_nll": inc16["window_nll"], "total_macs": inc16["total_macs"],
                   "total_cycle_macs": inc16["total_cycle_macs"], "source": "step-0 INC16 run (round 7)"}
    if not dry_run:
        try:
            pc = step0_parent_cost(cid, screen, root=step0_root)
        except (F.FixedPointError, FileNotFoundError) as exc:
            return _disabled(base, f"no step-0 P on the screen windows: {exc}", **extra)
    confirm = reg["confirm32"]
    for lst, nm in ((screen, "screen"), (confirm, "confirm")):
        if E.overlaps(lst, capture):
            raise SystemExit(f"{cid}: {nm} windows overlap the round-7 capture windows; redraw needed (user decision)")
    if E.overlaps(screen, confirm):
        raise SystemExit(f"{cid}: screen and confirm windows overlap")
    # ---- full-test reference numbers
    ptr = E.load_trace(parent_trace)
    p_cost = sum(int(g["macs"]) * int(g["stoc_len"]) for g in ptr["groups"]) / sum(int(g["macs"]) for g in ptr["groups"])
    p_hdr_ppl = ptr["header"].get("ppl")
    del ptr
    fp16 = E.FP16_PPL[model]
    files = [arms[n][k] for n in arms for k in ("wrapper", "table")] + \
        [str(hybrid), str(awq), row["trace"], str(parent_trace)] + extra_files
    if ref is not None:
        files += [ref["nll_file"], ref["trace_file"]]
    for p in probes:
        files.append(p["source"].split(":tables.")[0])
    if pc["mode"] == "reference":
        files += [pc["file"], pc["trace"]]
    inc_cost_vs_parent = 100.0 * (row["best_cost"] / p_cost - 1.0)
    cell = {**base, **extra, "enabled": True, "disabled_reason": None,
            "incumbent": {"arm": row["winning_arm"], "wrapper": str(inc_wrapper), "table": str(inc_table),
                          "full_test_ppl": row["best_ppl"], "full_test_cost": row["best_cost"],
                          "full_test_trace": row["trace"], "full_test_cost_vs_parent_pct": inc_cost_vs_parent},
            "parent": {"wrapper": arms["PARENT"]["wrapper"], "table": arms["PARENT"]["table"],
                       "test_trace": str(parent_trace), "test_cost": p_cost, "ppl": row["submitted_ppl"],
                       "test_trace_header_ppl": p_hdr_ppl},
            "arms": arms, "primary": primary, "secondaries": secondaries, "refused": refused, "notes": notes,
            "windows": {"screen": screen, "confirm": confirm, "capture_windows": capture,
                        "token_ids_sha256": E.QWEN_TOKEN_SHA, "screen_source": spec["screen"],
                        "registry": str(REGISTRY)},
            "identity": {"screen_reference": ref, "probes": probes},
            "parent_cost": pc,
            "full_test": {"fp16_ppl": fp16, "flip_thresholds": {"1.05x": round(fp16 * 1.05, 6),
                                                                "1.10x": round(fp16 * 1.10, 6)},
                          "eval_tokens": 298862, "best_all_test_cost_bound": PR.BEST_ALL_TEST_COST_BOUND,
                          "budget_correction_note": f"INC's full-test cost is {inc_cost_vs_parent:+.3f}% vs the "
                                                    "parent; a round-7 table re-targeted to the parent's cost "
                                                    "carries that budget correction in any gain"},
            "hashes": {f: E.sha256_file(f) for f in dict.fromkeys(files)},
            "simulation": {"inc_standin": ref["trace_file"] if ref else row["trace"],
                           "deltas": {primary: -0.01, **{n: -0.002 for n in secondaries}},
                           "force_cost_gate_pass": bool(dry_run),
                           "note": "CPU dry-run only; other arms synthesized onto their own ladders; "
                                   "force_cost_gate_pass is honoured only by the simulation evaluator"}}
    return cell


# ---------------------------------------------------------------------------
# dry-run stand-ins (scratch only)
# ---------------------------------------------------------------------------
def make_standins(scratch: Path, rows, kappa_ratio, enable_4b) -> Path:
    """STAND-IN candidate tables derived from each INC (NOT round-7 solves): UK = qk l0-l3 + av l3 ladders
    [128]+g[1:] (thresholds kept), K = attention thresholds scaled by 0.97 (pinned ladders), U = UK with
    attention thresholds scaled by 0.99. Written only under ``scratch``."""
    scratch.mkdir(parents=True, exist_ok=True)
    if str(scratch.resolve()).startswith((str(E.R7_ROOT), str(REPO))):
        raise SystemExit("stand-ins must be written to a scratch directory")
    spec = {"schema": "prc-r7-candidates-v1", "kappa_decision": None, "cells": {}, "standin": True}
    for c in CELLS:
        if c["model"] == "4B" and not enable_4b:
            continue
        row = rows[(c["model"], c["target"])]
        w = E.read_json(row["wrapper"])
        t = E.read_json(E.resolve_table_path(row["wrapper"]))
        g = E.global_ladder(t)
        top = [128] + g[1:]
        arms = []
        for name, kind in (("UK", "UK"), ("K", "K"), ("U", "U")):
            tt = copy.deepcopy(t)
            for key, p in tt["buckets"].items():
                op = key.split(":")[0]
                if op not in E.ATTN_OPS:
                    continue
                if kind in ("UK", "U") and (op == "qk" or key == "av:t0:l3"):
                    p["stoc_len_levels"] = list(top)
                if kind in ("K", "U"):
                    f = 0.97 if kind == "K" else 0.99
                    p["thresholds"] = [min(1.0, max(0.0, float(x) * f)) for x in p["thresholds"]]
            tt["r7_standin"] = {"kind": kind, "note": "DRY-RUN STAND-IN, not a round-7 solve"}
            tp = scratch / f"{c['id']}_{name}_standin_table.json"
            write_exact(tp, (json.dumps(tt, indent=1) + "\n").encode())
            ww = dict(w)
            ww["threshold_table_path"] = str(tp)
            wp = scratch / f"{c['id']}_{name}_standin.json"
            write_exact(wp, (json.dumps(ww, indent=2, sort_keys=True) + "\n").encode())
            r = 1.0 if (kind == "U" or c["model"] != "30B") else kappa_ratio
            arms.append({"name": name, "kind": kind, "wrapper": str(wp), "pred_true_nats": -0.012,
                         "mde_nats": 0.010, "kappa_ratio": r, "pred_detail": {"standin": True}})
        spec["cells"][c["id"]] = {"arms": arms}
    sp = scratch / "candidates_standin_spec.json"
    sp.write_text(json.dumps(spec, indent=1) + "\n")
    return sp


def spec_from_solve_summaries(items, kappa_path, fixed_points) -> dict:
    """Candidate spec from the round-7 solver's solve_summary.json files and the pre-registered fixed
    point. ``items`` = [(cell, summary path)] (several summaries per cell allowed); ``fixed_points`` =
    {cell: fixed-point record}. For every arm with an ok fixed point, EXACTLY ONE table solved at that
    arm's own s1 must exist across the cell's summaries; other scales are ignored. The CPU control J
    is never screened. Values are copied for readability only: build_cell re-derives and re-verifies
    every one of them from the sha-locked summaries."""
    spec = {"schema": "prc-r7-candidates-v1", "kappa_decision": kappa_path, "cells": {},
            "source": "prc_r7_solve.py solve_summary.json + prc_fixedpoint_r7_20260928.py"}
    by_cell = {}
    for cell, path in items:
        by_cell.setdefault(cell, []).append(path)
    for cell, paths in by_cell.items():
        if cell not in fixed_points:
            raise SystemExit(f"{cell}: --fixed-point {cell}=<record> is required (pre-registered target scale)")
        fp = F.load_fixed_point(fixed_points[cell], cell)
        sms = [(p, E.read_json(p)) for p in paths]
        for p, sm in sms:
            if sm.get("cell") != cell:
                raise SystemExit(f"{p}: summary is for {sm.get('cell')}, not {cell}")
        arms, fp_refused = [], {}
        for arm in PR.ARMS:
            fa = fp["arms"].get(arm) or {}
            if fa.get("status") != "ok":
                fp_refused[arm] = fa.get("status", "absent from the fixed point")
                continue
            hits = [(p, sm, name, rec) for p, sm in sms for name, rec in sm["arms"].items()
                    if rec["arm"] == arm and abs(float(rec["target_scale"]) - float(fa["s1"])) <= 5e-9]
            if len(hits) != 1:
                raise SystemExit(f"{cell}: arm {arm} needs exactly one table at its pre-registered s1 "
                                 f"{fa['s1']} across the given summaries, found {len(hits)} (run the fixed "
                                 "point's resolve command; pass only its summary)")
            p, sm, name, rec = hits[0]
            if "wrapper" not in rec or E.sha256_file(rec["wrapper"]) != rec["wrapper_sha256"] or \
                    E.sha256_file(rec["table_path"]) != rec["table_sha256"]:
                raise SystemExit(f"{cell} {name}: emitted wrapper/table missing or changed since the solve")
            arms.append({"name": arm, "kind": arm, "wrapper": rec["wrapper"],
                         "pred_true_nats": float(rec["pred_true_vs_inc"]), "mde_nats": float(sm["mde_nats"]),
                         "kappa_ratio": float(rec["kappa_ratio_used"]), "target_scale": rec["target_scale"],
                         "predicted_cost_ratio": rec["heldout_vs_inc"]["cost_over_ref"],
                         "pred_detail": {"table_name": name, "solve_summary": str(p),
                                         "solve_summary_sha256": E.sha256_file(p),
                                         "kappa_lin": sm["kappa_lin"], "kappa_att": sm["kappa_att"],
                                         "heldout_vs_inc": rec["heldout_vs_inc"],
                                         "calib_cost_over_target": rec["calib_cost_over_target"],
                                         "fixed_point_s1": fa["s1"],
                                         "ladders": rec.get("ladders"), "escape": rec.get("escape")}})
        spec["cells"][cell] = {"arms": arms, "fixed_point": str(fixed_points[cell]),
                               "fixed_point_sha256": E.sha256_file(fixed_points[cell]),
                               "fixed_point_refused": fp_refused, "solve_summaries": [str(p) for p in paths]}
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=None, help="a candidate spec previously written by --from-solve (re-verified)")
    ap.add_argument("--from-solve", action="append", default=[],
                    help="CELL=<solve_summary.json> (repeatable, several per cell allowed)")
    ap.add_argument("--fixed-point", action="append", default=[],
                    help="CELL=<fixed-point record from prc_fixedpoint_r7_20260928.py compute>")
    ap.add_argument("--kappa-decision", default=None, help="the eval-side step-0 kappa_decision.json")
    ap.add_argument("--mde", action="append", default=[], help=argparse.SUPPRESS)
    ap.add_argument("--standins", default=None, help="scratch dir: build stand-in candidates + a DRY-RUN manifest")
    ap.add_argument("--assume-gate", action="append", default=[], help="dry run only: CELL=pass|fail")
    ap.add_argument("--assume-kappa-ratio", type=float, default=None, help="dry run only")
    ap.add_argument("--enable-4b", action="store_true", help="dry run only: include 4B stand-ins")
    ap.add_argument("--out", default=None, help="manifest path (dry runs: must be under the scratch dir)")
    ap.add_argument("--overwrite-manifest", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--accept-kappa-review", action="store_true",
                    help="build although the step-0 kappa decision is flagged requires_user_review (needs the "
                         "user's explicit OK; recorded in the manifest)")
    # CPU-test overrides of the registered input/output locations (the stage-2 chain's end-to-end test); every
    # default is the registered path and a real build never passes them.
    ap.add_argument("--step0-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--capture-manifest", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--candidates-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.mde:
        raise SystemExit("--mde overrides are refused: the MDE is frozen per cell in prc_r7_prereg_20260928.json")
    step0_root = Path(args.step0_root) if args.step0_root else STEP0_ROOT
    capture_manifest = Path(args.capture_manifest) if args.capture_manifest else CAPTURE_MANIFEST
    candidates_out = Path(args.candidates_out) if args.candidates_out else CANDIDATES
    overridden = any(x is not None for x in (args.step0_root, args.capture_manifest, args.candidates_out))
    if overridden and (args.out is None or str(Path(args.out).resolve()).startswith(str(KDIR.resolve()))
                       or str(candidates_out.resolve()).startswith(str(KDIR.resolve()))):
        raise SystemExit("test overrides need an explicit --out and --candidates-out outside benchmark/ppl/kbands")
    reg_rule = (E.read_json(CAPTURE_MANIFEST).get("preregistered_rules") or {}).get("realized_length_audit")
    if reg_rule != SC.REALIZED_LENGTH_RULE:
        raise SystemExit("the capture manifest's registered realized_length_audit differs from the rule the screen "
                         "driver enforces (prc_screen_r7_20260928.REALIZED_LENGTH_RULE)")
    items = []
    for x in args.from_solve:
        cell, path = x.split("=", 1)
        if "@" in path:
            raise SystemExit("CELL=PATH@SCALE is refused: the scale is the pre-registered fixed point's s1")
        items.append((cell, path))
    dry = args.standins is not None
    if not dry and (args.assume_gate or args.assume_kappa_ratio is not None or args.enable_4b):
        raise SystemExit("--assume-* / --enable-4b are dry-run options only")
    best = E.read_json(E.BEST_ALL)
    rows = {(r["model"], int(r["target"])): r for r in best["rows"]}
    prereg = None
    if (not dry) or PR.RECORD.exists():
        try:
            rec, sha, t = PR.load_verified()
            prereg = {"path": str(PR.RECORD), "sha256": sha, "frozen_utc": rec["frozen_utc"], "frozen_t": t,
                      "mde_nats": rec["mde"]["nats"]}
        except (FileNotFoundError, ValueError) as exc:
            if not dry:
                raise SystemExit(f"pre-registration record not verified: {exc}")
    if dry:
        scratch = Path(args.standins)
        spec_path = make_standins(scratch, rows, args.assume_kappa_ratio or 0.42, args.enable_4b)
        out = Path(args.out or scratch / "prc_screen_r7_DRYRUN.json")
        if not str(out.resolve()).startswith(str(scratch.resolve())):
            raise SystemExit("a dry-run manifest must be written inside the scratch dir")
    else:
        if items:
            fps = dict(x.split("=", 1) for x in args.fixed_point)
            spec = spec_from_solve_summaries(items, args.kappa_decision, fps)
            data = (json.dumps(spec, indent=1) + "\n").encode()
            write_exact(candidates_out, data)
            args.spec = str(candidates_out)
        if not args.spec:
            raise SystemExit("--from-solve (with --fixed-point) or --spec is required")
        spec_path = Path(args.spec)
        out = Path(args.out or MANIFEST)
        if out.exists() and not args.overwrite_manifest:
            raise SystemExit(f"{out} exists; pass --overwrite-manifest (pre-submission only)")
    spec = E.read_json(spec_path)
    if spec.get("schema") != "prc-r7-candidates-v1":
        raise SystemExit("candidate spec schema must be prc-r7-candidates-v1")
    if not dry and spec.get("standin"):
        raise SystemExit("a stand-in spec can only build a dry-run manifest")
    reg = E.read_json(REGISTRY)
    if reg != E.derive_round7_windows():
        raise SystemExit("window registry no longer re-derives exactly")
    m6 = E.read_json(E.R6_DIAG_MANIFEST)
    if dry and args.assume_kappa_ratio is not None:
        kappa = {"ratio": args.assume_kappa_ratio, "assumed": True, "path": None, "decision": None}
    else:
        kp = spec.get("kappa_decision") or args.kappa_decision
        if not kp or not Path(kp).exists():
            raise SystemExit("the step-0 kappa decision file is required")
        try:
            kd = F.load_kappa_decision(kp, accept_review=args.accept_kappa_review)
        except F.FixedPointError as exc:
            raise SystemExit(str(exc))
        if prereg is not None and kd["created"] <= prereg["frozen_t"]:
            raise SystemExit("the kappa decision predates the pre-registration record")
        kappa = dict(kd, assumed=False, pins=PR.kappa_pins_for("30B", kd["decision"]))
    assume = dict(x.split("=", 1) for x in args.assume_gate)
    cells, tasks, flags = {}, [], []
    for i, c in enumerate(CELLS):
        gate = r6_gate(c["gate_cell"], assume)
        cells[c["id"]] = build_cell(c, spec["cells"].get(c["id"]), reg, rows, m6, kappa, gate, dry_run=dry,
                                    do_sweep=not args.skip_sweep, prereg=prereg, step0_root=step0_root,
                                    capture_manifest=capture_manifest)
        cc = cells[c["id"]]
        flags += cc.get("flags") or []
        tasks.append({"index": i, "id": c["id"], "cell": c["id"], "enabled": cc["enabled"]})
        print(f"[r7sbuild] {c['id']}: enabled={cc['enabled']} "
              + (f"primary={cc['primary']} secondaries={cc['secondaries']} refused={cc['refused']} "
                 f"P={cc['parent_cost']['mode']}" if cc["enabled"] else f"({cc['disabled_reason']})"), flush=True)
    if kappa.get("review_accepted"):
        flags.append(f"kappa decision flagged {kappa.get('flags')}; built with --accept-kappa-review (user OK)")
    sources = [str(REPO / p) for p in NEW_SOURCES + FROZEN_IMPORTED]
    missing = [p for p in sources if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"missing source files: {missing}")
    enabled = [t["index"] for t in tasks if t["enabled"]]
    manifest = {
        "schema": "prc-screen-r7-v1", "run_id": RUN_ID + ("_DRYRUN" if dry else "") + ("_TESTBUILD" if overridden else ""),
        "dry_run": dry, "test_build": overridden,
        "test_overrides": ({"step0_root": str(step0_root), "capture_manifest": str(capture_manifest),
                            "candidates_out": str(candidates_out)} if overridden else None),
        "purpose": "Round-7 screen -> confirm -> full test of the pre-registered primary per cell (family-kappa "
                   "joint lambda over attention ladders up to 128, re-solved at parent cost).",
        "units": "stream lengths HALVED (nominal = 2x); code cap 128; cost = MAC-weighted mean halved length",
        "max_total_gpus": 4, "gpus_per_task": 1,
        "concurrency_note": "One GPU per task; submit with --array=<enabled>%K and --dependency so that K plus every "
                            "other running GPU job stays <= 4.",
        "enabled_tasks": enabled,
        "protocol": dict(E.PROTOCOL, screen_windows=16, confirm_windows=32, cost_gate_tolerance=E.COST_GATE_TOL,
                         confirm_z=E.CONFIRM_Z, full_test="primary only, full WikiText-2 test, ctx 2048, "
                                                          "PPL_MAX_TOKENS=0, batch 1, via run_prc_ppl.sbatch"),
        "output_root": str(OUTPUT_ROOT),
        "windows_registry": {"path": str(REGISTRY), "sha256": E.sha256_file(REGISTRY)},
        "prereg": ({k: prereg[k] for k in ("path", "sha256", "frozen_utc", "mde_nats")} if prereg else
                   {"note": "dry run without a pre-registration record"}),
        "kappa": kappa, "candidate_spec": {"path": str(spec_path), "sha256": E.sha256_file(spec_path)},
        "flags": flags,
        "tasks": tasks, "cells": cells, "preregistered_rules": PREREGISTERED,
        "source_files": sources, "code_hashes": {p: E.sha256_file(p) for p in sources},
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    payload = json.dumps(manifest, indent=1, allow_nan=False) + "\n"
    if out.exists() and (args.overwrite_manifest or dry):
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(payload)
        tmp.replace(out)
    else:
        with out.open("x") as f:
            f.write(payload)
    print(f"[r7sbuild] wrote {out} (dry_run={dry}); enabled tasks {enabled}; flags {flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
