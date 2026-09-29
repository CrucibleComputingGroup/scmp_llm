"""Round-6 attention diagnostic driver (investigation section 6 Q1 + Q9).

Per cell, evaluates the frozen arm wrappers (INC, A, ATT128[, LIN128]) on the
same 16 fixed WikiText-2 TRAIN windows, one model load per cell, and writes
per-window NLL, the full SC summary trace of every arm, and diag_summary.json.

Fail-closed identity checks (the eval is bit-deterministic):
  * 30B: INC must reproduce round-4 ``confirm_incumbent`` window NLLs exactly,
    window by window, and its trace must equal the reference trace record by
    record (total MACs and cycle-MACs included);
  * 4B: INC on held-out window 30720 must reproduce the prc2 c7 held-out NLL;
  * all cells: unprofiled vs profiled vs restored incumbent on window 1 must
    match in NLL and traced SC work.
Every edit arm's own trace is audited (prc_r6_attn_diag_arms.audit_arm_trace):
A changes only the listed buckets' top rung to 128, ATT128 puts all attention
MACs at 128, LIN128 all SC linear MACs at 128, and nothing else moves. The
observe-only profiler additionally replays each arm's first window through the
runtime ladder resolvers. Any failure stops the cell with failure.json.

The pre-registered Round-7 gate (manifest ``preregistered_rules``) is evaluated
for arm A only; ATT128/LIN128 are attribution-only and never best(all)
candidates. No full TEST is run here.

  python benchmark/ppl/prc_r6_attn_diag.py --manifest M --task K --out-root R
  python benchmark/ppl/prc_r6_attn_diag.py --manifest M --task K --preflight [--deep]
  python benchmark/ppl/prc_r6_attn_diag.py --manifest M --task K --simulate --out-root SCRATCH
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
for _path in (str(REPO), str(REPO / "kernels")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmark.ppl import prc_r6_attn_diag_arms as R  # noqa: E402

SCHEMA = "prc-r6-attn-diag-v1"
TURBO_ROOT = "/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r6_20260928"
# Profiler-vs-trace agreement: both are exact integer sums in principle; a missing
# operator would be >= 1e-4 of MACs, so 1e-6 only absorbs float bookkeeping.
PROFILE_TOL = 1e-6
SIMULATION_LABEL = "SIMULATION (CPU dry-run from archived traces; synthetic NLL deltas; NOT a measurement)"


class IdentityError(RuntimeError):
    """An incumbent re-evaluation did not reproduce a recorded result exactly."""


class AuditError(RuntimeError):
    """An arm's realized lengths are not exactly the intended change."""


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    payload = json.dumps(value, indent=1, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as f:
            f.write(payload)
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload)
        tmp.replace(path)


# ---------------------------------------------------------------------------
# manifest / preflight (CPU)
# ---------------------------------------------------------------------------
def load_manifest(path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema {manifest.get('schema')!r} != {SCHEMA!r}")
    if manifest.get("max_total_gpus") != 4:
        raise ValueError("manifest must record the user's four-GPU total limit")
    fixed = manifest.get("protocol", {})
    expect = dict(frontend="awq", owen_mode="bitrev", scramble_masks=64, ctx=2048, split="train",
                  sc_prec=8, sc_halve=True, qk_rebalance=False, rng_grid="fixed128",
                  awq_obj_bits=4, window_count=16, full_test=False)
    for key, value in expect.items():
        if fixed.get(key) != value or type(fixed.get(key)) is not type(value):
            raise ValueError(f"fixed protocol mismatch in manifest: {key}")
    return manifest


def verify_code_hashes(manifest) -> None:
    hashes = manifest.get("code_hashes") or {}
    if not hashes or set(hashes) != set(manifest.get("source_files", [])):
        raise ValueError("code hashes must cover exactly the manifest source_files")
    for filename, expected in hashes.items():
        if R.sha256_file(filename) != expected:
            raise ValueError(f"experiment source changed after freezing: {filename}")


def verify_cell_inputs(cell: dict) -> dict:
    """Hash every frozen input, re-validate every arm table against the incumbent
    (pure python), and return the loaded incumbent table + wrappers."""
    for filename, expected in cell["hashes"].items():
        if R.sha256_file(filename) != expected:
            raise ValueError(f"frozen input changed: {filename}")
    inc_arm = cell["arms"]["INC"]
    src_wrapper = json.loads(Path(cell["incumbent"]["wrapper"]).read_text())
    if src_wrapper["threshold_table_path"] != cell["incumbent"]["table"]:
        raise ValueError("incumbent wrapper/table mismatch")
    if R.sha256_file(inc_arm["table"]) != R.sha256_file(cell["incumbent"]["table"]):
        raise ValueError("INC table is not a byte-identical copy of the incumbent table")
    base = json.loads(Path(inc_arm["table"]).read_text())
    R.check_incumbent_shape(base)
    if base["model_path"] != cell["model_path"]:
        raise ValueError("model mismatch between table and cell")
    if R.global_ladder(base) != cell["global_ladder"]:
        raise ValueError("global ladder mismatch")
    wrappers = {}
    for arm in ["INC"] + cell["arm_order"]:
        entry = cell["arms"][arm]
        wrapper = json.loads(Path(entry["wrapper"]).read_text())
        expect = copy.deepcopy(src_wrapper)
        expect["threshold_table_path"] = entry["table"]
        if wrapper != expect:
            raise ValueError(f"{arm} wrapper must equal the incumbent wrapper except its table path")
        if wrapper.get("escape_gate_k") != cell["escape_gate_k"] \
                or int(wrapper.get("escape_stoc_len", 128)) != cell["escape_stoc_len"]:
            raise ValueError(f"{arm} wrapper escape settings changed")
        if arm != "INC":
            R.validate_arm_table(base, json.loads(Path(entry["table"]).read_text()), arm, entry["spec"])
        wrappers[arm] = wrapper
    if cell["arm_order"][0] != "A" or not set(cell["arm_order"]) <= set(R.EDIT_ARMS):
        raise ValueError("arm A must be evaluated first; only edit arms are allowed")
    starts = cell["windows"]["starts"]
    if len(starts) != 16 or len(set(starts)) != 16 or any(s % 2048 for s in starts):
        raise ValueError("exactly 16 distinct ctx-aligned diagnostic windows are required")
    return {"base": base, "wrappers": wrappers}


def preflight_task(manifest: dict, task_index: int, out_root, deep: bool = False) -> list:
    verify_code_hashes(manifest)
    tasks = manifest["tasks"]
    if not 0 <= task_index < len(tasks) or tasks[task_index]["index"] != task_index:
        raise ValueError(f"no task {task_index}")
    report = []
    for cell_id in tasks[task_index]["cells"]:
        cell = manifest["cells"][cell_id]
        loaded = verify_cell_inputs(cell)
        out = Path(out_root) / cell_id
        for name in ("driver_started.json", "diag_summary.json", "failure.json"):
            if (out / name).exists():
                raise ValueError(f"refusing to reuse a used cell directory: {out / name}")
        entry = {"cell": cell_id, "arms": cell["arm_order"], "windows": len(cell["windows"]["starts"]),
                 "identity": cell["identity"]["type"], "out_dir": str(out)}
        if deep:
            sweeps = {}
            for arm in cell["arm_order"]:
                sweeps[arm] = R.resolution_sweep(loaded["wrappers"][arm], cell["arms"][arm]["table"], arm,
                                                 cell["arms"][arm]["spec"], loaded["base"],
                                                 cell["arms"]["INC"]["table"], cell["total_blocks"])
            entry["resolution_sweep"] = sweeps
        report.append(entry)
    return report


# ---------------------------------------------------------------------------
# backends: the real GPU evaluator and a CPU simulation for dry-runs
# ---------------------------------------------------------------------------
class GPUBackend:
    simulated = False

    def __init__(self, manifest, cell, out):
        self.manifest, self.cell, self.out = manifest, cell, Path(out)

    def build(self):
        from benchmark.ppl.prc_local_refine import configure_environment
        import torch
        from datasets import load_dataset
        from benchmark.quant.eval_quant import build_sc_model
        env_cell = {"incumbent_wrapper": self.cell["arms"]["INC"]["wrapper"],
                    "hybrid_config": self.cell["hybrid_config"], "model_path": self.cell["model_path"]}
        configure_environment(env_cell)
        for key in ("SC_MP_TRACE", "SC_MP_TRACE_MODE"):
            if os.environ.get(key):
                raise ValueError(f"{key} must be unset; the driver owns tracing")
        self.torch = torch
        self.model, tok = build_sc_model(self.cell["model_path"], "mp",
                                         mp_table=self.cell["arms"]["INC"]["wrapper"])
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        if self.dev.type != "cuda":
            raise ValueError("the diagnostic must run on its assigned Slurm GPU")
        cfg = self.model.config
        total = getattr(cfg, "_sc_total_blocks", None) or getattr(cfg, "num_hidden_layers", None)
        if int(total) != int(self.cell["total_blocks"]):
            raise ValueError(f"model has {total} blocks, manifest says {self.cell['total_blocks']}")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        return hashlib.sha256(self.enc.numpy().tobytes()).hexdigest(), int(self.enc.numel())

    def _swap(self, wrapper_path):
        from loader import apply_mp_config_from_env
        from model.sc_common import SCLinear
        os.environ["MP_CONFIG_JSON"] = str(wrapper_path)
        self.model.config.sc_mp_config = None
        apply_mp_config_from_env(self.model)
        mp = self.model.config.sc_mp_config
        modules = [m for m in self.model.modules() if isinstance(m, SCLinear)]
        if mp is None or not modules or any(m._sc_config.sc_mp_config is not mp for m in modules):
            raise ValueError("table swap did not reach every SCLinear")
        if Path(mp.threshold_table_path).resolve() != \
                Path(json.loads(Path(wrapper_path).read_text())["threshold_table_path"]).resolve():
            raise ValueError("loaded MP config does not point at the arm table")

    def evaluate(self, name, arm, wrapper_path, starts, *, profile_first, expected, header_extra):
        """Evaluate one arm; returns (window_nll, seconds, snapshot, first_trace_path)."""
        torch = self.torch
        from benchmark.ppl.prc_local_proposals import ProfileCollector
        from model.sc_common import mp_tracker_reset
        from scmp_kernels import trace
        self._swap(wrapper_path)
        trace_path = self.out / f"{name}_trace.json"
        if trace_path.exists():
            raise ValueError(f"refusing to overwrite {trace_path}")
        trace.reset()
        trace.enable(str(trace_path), mode="summary")
        mp_tracker_reset()
        losses, snapshot, first_trace = [], None, None
        started = time.time()
        for i, start in enumerate(starts):
            ids = self.enc[start:start + 2048].unsqueeze(0).to(self.dev)
            if ids.numel() != 2048:
                raise ValueError(f"window {start} is shorter than ctx")
            if profile_first and i == 0:
                collector = ProfileCollector(self.model)
                with collector:
                    with torch.no_grad():
                        loss = float(self.model(input_ids=ids, labels=ids).loss)
                snapshot = collector.snapshot()  # raises on any ladder-resolver replay mismatch
                first_trace = self.out / f"{name}_first_window_trace.json"
                trace.flush(path=str(first_trace), reset_after=False,
                            header_extra={**header_extra, "stage": name + "_first_window",
                                          "windows": [int(start)]})
            else:
                with torch.no_grad():
                    loss = float(self.model(input_ids=ids, labels=ids).loss)
            if not math.isfinite(loss):
                raise ValueError(f"nonfinite NLL in {name} window {start}")
            losses.append(loss)
            print(f"[r6diag] {self.cell['id']} {name} window {i + 1}/{len(starts)} "
                  f"start={start} NLL={loss!r}", flush=True)
            if expected is not None and loss != expected[i]:
                raise IdentityError(f"{self.cell['id']} {name} window {start}: NLL {loss!r} "
                                    f"!= recorded {expected[i]!r}")
        trace.flush(header_extra={**header_extra, "stage": name, "windows": [int(s) for s in starts]},
                    reset_after=True)
        trace.disable()
        return losses, time.time() - started, snapshot, first_trace

    def close(self):
        for attr in ("model", "enc"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class SimBackend:
    """CPU dry-run: exercises the whole driver flow (identity checks, audits,
    statistics, gate, summary) with archived traces standing in for GPU traces
    and synthetic NLL deltas. Never writes under the Turbo output root."""
    simulated = True
    SYN_DELTA = {"A": -0.012, "ATT128": -0.02, "LIN128": -0.035}

    def __init__(self, manifest, cell, out, *, synthetic_profile=False, profile_skew=0.0):
        self.manifest, self.cell, self.out = manifest, cell, Path(out)
        self.synthetic_profile, self.profile_skew = synthetic_profile, float(profile_skew)
        if str(self.out.resolve()).startswith(TURBO_ROOT):
            raise ValueError("simulation must not write under the Turbo output root")

    def build(self):
        sim = self.cell["simulation"]
        self.base = json.loads(Path(self.cell["arms"]["INC"]["table"]).read_text())
        self.payload = json.loads(Path(sim["incumbent_trace_standin"]).read_text())
        ident = self.cell["identity"]
        starts = self.cell["windows"]["starts"]
        if ident["type"] == "reference_run":
            self.inc_nll = dict(zip(starts, ident["window_nll"]))
        else:
            self.inc_nll = {s: 2.0 + 0.1 * math.sin(s / 7919.0) for s in starts}
            self.inc_nll[ident["start"]] = ident["nll"]
        return self.cell["windows"]["token_ids_sha256"], None

    def evaluate(self, name, arm, wrapper_path, starts, *, profile_first, expected, header_extra):
        trace_path = self.out / f"{name}_trace.json"
        if trace_path.exists():
            raise ValueError(f"refusing to overwrite {trace_path}")
        spec = self.cell["arms"][arm]["spec"] if arm != "INC" else {}
        payload = R.synthesize_arm_trace(self.payload, arm, spec, self.base, self.cell["total_blocks"],
                                         header_extra={**header_extra, "stage": name,
                                                       "windows": [int(s) for s in starts]})
        trace_path.write_text(json.dumps(payload))
        first_trace = None
        if profile_first:
            first_trace = self.out / f"{name}_first_window_trace.json"
            first = copy.deepcopy(payload)
            first["header"].update(stage=name + "_first_window", windows=[int(starts[0])])
            first_trace.write_text(json.dumps(first))
        snapshot = None
        if profile_first and self.synthetic_profile:
            # Minimal profile-shaped snapshot (all work pooled as fixed) so the
            # driver's profiler-vs-trace and table-replay checks are exercised.
            tm = sum(int(g["macs"]) for g in payload["groups"])
            tc = sum(int(g["macs"]) * int(g["stoc_len"]) for g in payload["groups"])
            tm_p = tm * (1.0 + self.profile_skew)
            snapshot = {"version": 1, "bins": 0, "groups": {}, "fixed_macs": tm_p,
                        "fixed_cycle_macs": float(tc), "total_macs": tm_p,
                        "total_cycle_macs": float(tc), "actual_mean_length": tc / tm_p,
                        "simulated": True}
        losses = []
        for i, s in enumerate(starts):
            delta = self.SYN_DELTA.get(arm, 0.0) * (1.0 + 0.3 * math.cos(s / 104729.0))
            loss = self.inc_nll[s] + delta
            losses.append(loss)
            if expected is not None and loss != expected[i]:
                raise IdentityError(f"simulated {name} window {s} mismatch")
        return losses, 0.0, snapshot, first_trace

    def close(self):
        pass


# ---------------------------------------------------------------------------
# per-cell protocol
# ---------------------------------------------------------------------------
def _identity_totals(path, cell):
    agg = R.TraceAgg.load(path, cell["total_blocks"], cell["layer_buckets"])
    return agg.total_macs, agg.total_cycle_macs


def run_cell(manifest: dict, cell: dict, out: Path, backend) -> dict:
    loaded = verify_cell_inputs(cell)
    base, wrappers = loaded["base"], loaded["wrappers"]
    arms = cell["arms"]
    starts = [int(s) for s in cell["windows"]["starts"]]
    escape_len = cell["escape_stoc_len"] if cell["escape_gate_k"] is not None else None
    hybrid = json.loads(Path(cell["hybrid_config"]).read_text())
    write_json(out / "driver_started.json", {
        "time": time.time(), "cell": cell["id"], "simulated": backend.simulated,
        "arms": ["INC"] + cell["arm_order"], "windows": starts,
        "manifest_run_id": manifest["run_id"]}, exclusive=True)
    progress = {"cell": cell["id"], "simulated": backend.simulated, "evaluations": {}}

    def record(name, arm, window_list, nll, seconds, trace_path, snapshot, first_trace):
        agg_cost = R.TraceAgg.load(trace_path, cell["total_blocks"], cell["layer_buckets"])
        result = {"name": name, "arm": arm, "wrapper": arms[arm]["wrapper"],
                  "window_starts": [int(s) for s in window_list],
                  "window_nll": nll, "mean_nll": sum(nll) / len(nll), "seconds": seconds,
                  "trace": str(trace_path), "cost": agg_cost.cost,
                  "total_macs": agg_cost.total_macs, "total_cycle_macs": agg_cost.total_cycle_macs}
        if snapshot is not None:
            from benchmark.ppl.prc_local_proposals import baseline_cost, exact_profile_cost
            tm, tc = _identity_totals(first_trace, cell)
            table = json.loads(Path(arms[arm]["table"]).read_text())
            replay = baseline_cost(table, snapshot)
            exact = exact_profile_cost(snapshot)
            checks = {"first_window_trace_total_macs": tm, "first_window_trace_total_cycle_macs": tc,
                      "profile_total_macs": snapshot["total_macs"],
                      "profile_total_cycle_macs": snapshot["total_cycle_macs"],
                      "profile_exact_cost": exact, "profile_table_replay_cost": replay}
            if abs(snapshot["total_macs"] / tm - 1) > PROFILE_TOL or abs(snapshot["total_cycle_macs"] / tc - 1) > PROFILE_TOL:
                raise AuditError(f"{name}: profiler totals disagree with the first-window trace: {checks}")
            if abs(replay / exact - 1) > PROFILE_TOL:
                raise AuditError(f"{name}: table ladders do not replay the dispatched lengths: {checks}")
            write_json(out / f"{name}_profile.json", snapshot, exclusive=True)
            result["profile_checks"] = checks
        elif first_trace is not None:
            result["profile_checks"] = "skipped (simulation)"
        write_json(out / f"{name}_nll.json", result, exclusive=True)
        progress["evaluations"][name] = {k: v for k, v in result.items() if k != "window_nll"}
        write_json(out / "diag_progress.json", progress)
        return result, agg_cost

    def run(name, arm, window_list, *, profile_first=False, expected=None):
        header = {"cell": cell["id"], "arm": arm, "split": "train", "ctx": 2048,
                  "mp_config_json": str(arms[arm]["wrapper"]), "model": cell["model_path"],
                  "total_blocks": cell["total_blocks"], "run_id": manifest["run_id"],
                  "simulated": backend.simulated}
        nll, seconds, snapshot, first_trace = backend.evaluate(
            name, arm, arms[arm]["wrapper"], window_list, profile_first=profile_first,
            expected=expected, header_extra=header)
        return record(name, arm, window_list, nll, seconds, out / f"{name}_trace.json",
                      snapshot, first_trace), first_trace

    token_sha, _ntok = backend.build()
    if token_sha != cell["windows"]["token_ids_sha256"]:
        raise IdentityError(f"TRAIN token stream sha {token_sha} != recorded {cell['windows']['token_ids_sha256']}")
    identity = {"token_ids_sha256": token_sha}
    ident = cell["identity"]

    if ident["type"] == "probe":  # 4B cross-job identity, fail fast
        (probe, _), _ = run("probe_incumbent", "INC", [int(ident["start"])], expected=[ident["nll"]])
        identity["probe"] = {"start": ident["start"], "expected": ident["nll"],
                             "observed": probe["window_nll"][0], "exact": True, "source": ident["file"]}

    ref_first = [ident["window_nll"][0]] if ident["type"] == "reference_run" else None
    (before, before_agg), _ = run("identity_before", "INC", starts[:1], expected=ref_first)
    (inc, inc_agg), inc_first = run("INC", "INC", starts, profile_first=True,
                                    expected=ident["window_nll"] if ident["type"] == "reference_run" else None)
    if inc["window_nll"][0] != before["window_nll"][0]:
        raise IdentityError("profiled incumbent window 1 differs from the unprofiled run")
    if _identity_totals(inc_first, cell) != (before_agg.total_macs, before_agg.total_cycle_macs):
        raise IdentityError("profiled incumbent first-window SC work differs from the unprofiled run")
    identity["profiler_observe_only"] = True
    if ident["type"] == "reference_run":
        if (inc_agg.total_macs, inc_agg.total_cycle_macs) != (ident["total_macs"], ident["total_cycle_macs"]):
            raise IdentityError(f"incumbent trace totals {(inc_agg.total_macs, inc_agg.total_cycle_macs)} "
                                f"!= reference {(ident['total_macs'], ident['total_cycle_macs'])}")
        ref_agg = R.TraceAgg.load(ident["trace_file"], cell["total_blocks"], cell["layer_buckets"])
        diffs = R.compare_trace_to_reference(inc_agg, ref_agg)
        del ref_agg
        if diffs:
            raise IdentityError(f"incumbent trace differs from the round-4 reference: {diffs}")
        identity["reference_run"] = {"nll_file": ident["nll_file"], "trace_file": ident["trace_file"],
                                     "window_nll_exact": True, "trace_records_exact": True}
    inc_audit = R.audit_trace_basics(inc_agg, "incumbent", hybrid) + \
        R.audit_lengths(inc_agg, "INC", {}, base, escape_len)
    if inc_audit:
        raise AuditError(f"incumbent trace audit failed: {inc_audit}")
    identity["incumbent_trace_audit"] = "passed"
    progress["identity"] = identity
    write_json(out / "diag_progress.json", progress)

    arm_results = {}
    for arm in cell["arm_order"]:
        (res, arm_agg), _ = run(arm, arm, starts, profile_first=True)
        audit = R.audit_arm_trace(inc_agg, arm_agg, arm, arms[arm]["spec"], base,
                                  escape_len=escape_len, hybrid=hybrid,
                                  expected_windows=starts, expected_wrapper=arms[arm]["wrapper"])
        write_json(out / f"{arm}_audit.json", audit, exclusive=True)
        if not audit["ok"]:
            raise AuditError(f"{arm} realized lengths are not the intended change: {audit['failures'][:5]}")
        metrics = R.arm_metrics(res["window_nll"], inc["window_nll"], res["cost"], inc["cost"])
        if abs(metrics["extra_cycles_pct"] - audit["extra_cycles_pct"]) > 1e-9:
            raise AuditError("cost bookkeeping disagrees between the audit and the metrics")
        arm_results[arm] = {
            "metrics": metrics,
            "window_nll": res["window_nll"],
            "cost": res["cost"], "trace": res["trace"],
            "audit": {k: audit[k] for k in ("ok", "first_edit_position", "upstream_record_keys_checked",
                                            "first_edit_record_keys_checked", "extra_cycles_pct",
                                            "direct_cycles_pct", "cascade_cycles_pct",
                                            "predicted_first_order_direct_pct", "A_old_top_share")},
            "audit_file": str(out / f"{arm}_audit.json"),
            "profile_checks": res.get("profile_checks"),
            "attribution_only": arm in R.ATTRIBUTION_ONLY_ARMS,
            "best_all_candidate": False,
        }
        progress["arms"] = {a: {"metrics": v["metrics"], "audit_ok": v["audit"]["ok"]}
                            for a, v in arm_results.items()}
        write_json(out / "diag_progress.json", progress)
        m = metrics
        print(f"[r6diag] {cell['id']} {arm}: dNLL={m['dnll']:+.6f} (se {m['se_dnll']:.6f}, z {m['z']}) "
              f"dPPL={m['dppl_pct']:+.3f}% extra cycles={m['extra_cycles_pct']:+.3f}% "
              f"dPPL/1%cyc={m['dppl_pct_per_1pct_extra_cycles']}", flush=True)

    (after, after_agg), _ = run("identity_after", "INC", starts[:1])
    if after["window_nll"] != before["window_nll"] or \
            (after_agg.total_macs, after_agg.total_cycle_macs) != (before_agg.total_macs, before_agg.total_cycle_macs):
        raise IdentityError("restoring the incumbent changed its loss or SC work")
    identity["restored_incumbent_exact"] = True

    gate = R.round7_gate(cell["model"], arm_results["A"]["metrics"], cell["chord"]["chord"], True)
    sensitivity = None
    down = cell["chord"].get("downward_sensitivity")
    if down:
        sensitivity = R.round7_gate(cell["model"], arm_results["A"]["metrics"],
                                    down["chord_pct_ppl_per_1pct_cycles"], True)
    q9 = None
    if {"ATT128", "LIN128"} <= set(arm_results):
        att, lin = arm_results["ATT128"]["metrics"], arm_results["LIN128"]["metrics"]
        q9 = {"dnll_attention_all128": att["dnll"], "dnll_linears_all128": lin["dnll"],
              "se_attention": att["se_dnll"], "se_linears": lin["se_dnll"],
              "paired_att_minus_lin": R.paired(arm_results["ATT128"]["window_nll"],
                                               arm_results["LIN128"]["window_nll"]),
              "note": "descriptive attribution only; no decision rule attached"}
    summary = {
        "schema": SCHEMA + "-summary", "run_id": manifest["run_id"], "cell": cell["id"],
        "model": cell["model"], "target": cell["target"], "simulated": backend.simulated,
        "label": SIMULATION_LABEL if backend.simulated else "GPU measurement, TRAIN windows",
        "units": "stream lengths HALVED (nominal = 2x); costs = MAC-weighted mean halved length",
        "incumbent": cell["incumbent"], "windows": starts,
        "identity": identity,
        "incumbent_eval": {"mean_nll": inc["mean_nll"], "cost": inc["cost"],
                           "total_macs": inc["total_macs"], "window_nll": inc["window_nll"]},
        "arms": arm_results,
        "preregistered_rules": manifest["preregistered_rules"],
        "round7_gate": {"binding": gate, "binding_chord": cell["chord"]["upward"],
                        "downward_chord_sensitivity_nonbinding": sensitivity},
        "q9_attribution": q9,
        "reporting": "A is per-row attention reallocation (not a T1 granularity effect); no arm here is a "
                     "best(all) candidate; best(all) = best full-test PPL across comparable rounds.",
        "complete": True, "finished": time.time(),
    }
    write_json(out / "diag_summary.json", summary, exclusive=True)
    print(f"[r6diag] {cell['id']} COMPLETE: Round-7 proceed={gate['proceed_round7_attention_ladder_solve']} "
          f"(dPPL(A)={gate['dppl_pct_A']:+.3f}% vs threshold {gate['threshold_dppl_pct']:+.3f}%, "
          f"dNLL(A)={gate['dnll_A']:+.5f} vs {gate['gross_dnll_threshold']})", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--preflight", action="store_true", help="CPU checks only; no model load")
    ap.add_argument("--deep", action="store_true", help="with --preflight: runtime resolver sweep (torch, CPU)")
    ap.add_argument("--simulate", action="store_true",
                    help="CPU dry-run of the full flow from archived traces (scratch out-root only)")
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    out_root = Path(args.out_root or manifest["output_root"]).resolve()
    if not args.simulate and not args.preflight and str(out_root) != manifest["output_root"]:
        raise ValueError("GPU runs must write to the manifest output_root")
    report = preflight_task(manifest, args.task, out_root, deep=args.deep)
    if args.preflight:
        print(json.dumps({"preflight": "PASS", "task": manifest["tasks"][args.task]["id"],
                          "cells": report}, indent=1), flush=True)
        return 0
    cells = manifest["tasks"][args.task]["cells"]
    for cell_id in cells:
        cell = manifest["cells"][cell_id]
        out = out_root / cell_id
        if not out.is_dir():
            if args.simulate:
                out.mkdir(parents=True)
            else:
                raise ValueError(f"launcher must create the cell directory first: {out}")
        backend = (SimBackend if args.simulate else GPUBackend)(manifest, cell, out)
        try:
            run_cell(manifest, cell, out, backend)
        except BaseException as exc:
            write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                              "traceback": traceback.format_exc(), "time": time.time()})
            raise
        finally:
            backend.close()
    verify_code_hashes(manifest)
    print(f"[r6diag] task {manifest['tasks'][args.task]['id']} COMPLETE ({len(cells)} cell(s))", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
