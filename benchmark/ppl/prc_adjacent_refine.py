"""Round 4: paired TRAIN loss search with bounded adjacent-rung exchanges.

The numerical execution and incumbent are frozen. Only thresholds may change.
Search and confirmation both check locality on the candidate's own trajectory;
full TEST is requested only for a cost-feasible, independently confirmed gain.
Final reporting remains best(full TEST) across comparable allocation rounds.
"""
from __future__ import annotations

import argparse
import copy
import glob
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

from benchmark.ppl.prc_local_refine import (
    choose_disjoint_starts, configure_environment, paired_stats, preflight,
    sha256, trace_cost, validate_allocation_only, within_cost, write_json,
)


def protocol_settings(manifest):
    """Use manifest settings, rejecting numerical variants or unbounded sweeps."""
    p = copy.deepcopy(manifest["protocol"])
    fixed = dict(frontend="awq", owen_mode="bitrev", scramble_masks=64,
                 ctx=2048, stride=2048, ppl_max_tokens=0,
                 ppl_window_batch_size=1, sc_prec=8, sc_halve=True,
                 qk_rebalance=False, rng_grid="fixed128", awq_obj_bits=4,
                 require_confirmation_improvement=True, search_iterations=1)
    for key, value in fixed.items():
        if key not in p or type(p[key]) is not type(value) or p[key] != value:
            raise ValueError(f"fixed round-4 protocol mismatch: {key}")
    for key, low, high in (("search_windows", 2, 16),
                           ("confirmation_windows", 2, 32),
                           ("max_candidates", 1, 16)):
        if type(p.get(key)) is not int or not low <= p[key] <= high:
            raise ValueError(f"invalid bounded protocol setting: {key}")
    if type(p.get("data_sampling_seed")) is not int or p["data_sampling_seed"] < 0:
        raise ValueError("nonnegative integer corpus sampling seed required")
    for key, cap in (("operator_cost_fraction_cap", .02),
                     ("bucket_cost_fraction_cap", .05),
                     ("changed_group_fraction_cap", .05),
                     ("changed_mac_fraction_cap", .05),
                     ("cost_tolerance_fraction", .01)):
        value = p.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= cap):
            raise ValueError(f"invalid locality/cost limit: {key}")
    transfers = p.get("transfer_fractions")
    if (not isinstance(transfers, list) or not 1 <= len(transfers) <= 4
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or not 0 < x <= .005 for x in transfers)):
        raise ValueError("transfer_fractions must be positive and at most 0.5%")
    z = p.get("confirmation_z_threshold")
    if (isinstance(z, bool) or not isinstance(z, (int, float))
            or not math.isfinite(z) or z > -1.5):
        raise ValueError("confirmation threshold must be at least as strict as z < -1.5")
    return p


def load_excluded_windows(cell):
    """Read prior search/confirmation blocks and calibration diagnostic blocks."""
    paths = set()
    for name in cell.get("excluded_windows_files", []):
        path = Path(name).resolve()
        if not path.is_file():
            raise ValueError(f"required prior-window file missing: {path}")
        paths.add(path)
    for pattern in cell.get("historical_calibration_globs", []):
        paths.update(Path(name).resolve() for name in glob.glob(pattern))
    excluded, provenance = set(), []
    for path in sorted(paths):
        data = json.loads(path.read_text())
        if isinstance(data.get("starts"), dict):
            if not all(key in data["starts"] for key in ("search", "confirm")):
                raise ValueError(f"incomplete prior search/confirmation schema: {path}")
            lists = list(data["starts"].values())
        elif "windows" in data:
            lists = [data["windows"]]
        else:
            raise ValueError(f"unknown prior-window schema: {path}")
        if any(not isinstance(values, list) or any(type(s) is not int or s < 0
                                                   for s in values) for values in lists):
            raise ValueError(f"invalid prior-window starts: {path}")
        starts = sorted({s for values in lists for s in values})
        excluded.update(starts)
        provenance.append({"path": str(path), "sha256": sha256(path), "starts": starts})
    return excluded, provenance


def search_eligible(result):
    mean = result.get("paired_vs_incumbent", {}).get("mean_dnll")
    return (result.get("cost_feasible") is True
            and result.get("locality_feasible") is True
            and isinstance(mean, (int, float)) and math.isfinite(mean) and mean < 0)


def confirmation_qualifies(stats, cost_ok, locality_ok, z_threshold=-1.5):
    mean, z = stats.get("mean_dnll"), stats.get("z")
    return (cost_ok is True and locality_ok is True
            and isinstance(mean, (int, float)) and math.isfinite(mean) and mean < 0
            and isinstance(z, (int, float)) and math.isfinite(z) and z < z_threshold)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    protocol = protocol_settings(manifest)
    if not manifest.get("code_hashes"):
        raise ValueError("code hashes must be frozen before submission")
    cell, source_wrapper, source_table = preflight(manifest, args.index)
    excluded, historical = load_excluded_windows(cell)
    if not cell.get("excluded_windows_files"):
        raise ValueError("round-3 search/confirmation exclusions are required")
    if args.preflight:
        print(f"[adjacent] preflight PASS {cell['id']}; {len(excluded)} historical starts", flush=True)
        return 0
    configure_environment(cell)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "driver_started.json", {"time": time.time(), "args": vars(args),
               "protocol": protocol, "cell": cell}, exclusive=True)
    try:
        import torch
        from datasets import load_dataset
        from benchmark.quant.eval_quant import build_sc_model
        from benchmark.ppl.calibrate_mp_thresholds import _select_int_swap_windows
        from benchmark.ppl.prc_adjacent_proposals import (
            AdjacentProfileCollector, audit_transition, baseline_cost, propose,
        )
        from loader import apply_mp_config_from_env
        from model.sc_common import SCLinear, mp_tracker_reset, mp_tracker_flop_avg_stoc_len
        from scmp_kernels import trace

        limits = dict(max_operator_cost_fraction=protocol["operator_cost_fraction_cap"],
                      max_bucket_cost_fraction=protocol["bucket_cost_fraction_cap"],
                      max_changed_group_fraction=protocol["changed_group_fraction_cap"],
                      max_changed_mac_fraction=protocol["changed_mac_fraction_cap"])
        base_table, base_wrapper = out / "incumbent_table.json", out / "incumbent.json"
        write_json(base_table, source_table, exclusive=True)
        snapshot_wrapper = copy.deepcopy(source_wrapper)
        snapshot_wrapper["threshold_table_path"] = str(base_table)
        write_json(base_wrapper, snapshot_wrapper, exclusive=True)
        model, tok = build_sc_model(cell["model_path"], "mp", mp_table=str(base_wrapper))
        model.eval()
        dev = next(model.parameters()).device
        if dev.type != "cuda":
            raise ValueError("round 4 must run on its assigned Slurm GPU")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        for n in (6, 8, 12, 14):
            _, starts = _select_int_swap_windows(enc, 2048, n, sampling="stratified", seed=0)
            excluded.update(int(s) for s in starts)
        search_starts, confirm_starts = choose_disjoint_starts(
            len(enc), 2048, protocol["search_windows"], protocol["confirmation_windows"],
            excluded, protocol["data_sampling_seed"])
        write_json(out / "windows.json", {"split": "train", "ctx": 2048,
                   "starts": {"search": search_starts, "confirm": confirm_starts},
                   "excluded": sorted(excluded), "historical_files": historical,
                   "data_seed": protocol["data_sampling_seed"],
                   "token_ids_sha256": hashlib.sha256(enc.numpy().tobytes()).hexdigest()}, exclusive=True)

        def swap(wrapper):
            os.environ["MP_CONFIG_JSON"] = str(wrapper)
            model.config.sc_mp_config = None
            apply_mp_config_from_env(model)
            mp = model.config.sc_mp_config
            modules = [m for m in model.modules() if isinstance(m, SCLinear)]
            if mp is None or not modules or any(m._sc_config.sc_mp_config is not mp for m in modules):
                raise ValueError("table swap did not reach every SCLinear")

        def evaluate(name, wrapper, starts, *, profile=False):
            from contextlib import nullcontext
            swap(wrapper)
            trace.reset()
            trace_path = out / f"{name}_trace.json"
            trace.enable(str(trace_path), mode="summary")
            mp_tracker_reset()
            collector = AdjacentProfileCollector(model, reference_table=source_table) if profile else None
            losses, started = [], time.time()
            with collector if collector is not None else nullcontext():
                for i, start in enumerate(starts):
                    ids = enc[start:start + 2048].unsqueeze(0).to(dev)
                    with torch.no_grad():
                        loss = float(model(input_ids=ids, labels=ids).loss)
                    if not math.isfinite(loss):
                        raise ValueError("nonfinite model NLL")
                    losses.append(loss)
                    if profile and i == 0:
                        first_trace = out / f"{name}_first_window_trace.json"
                        trace.flush(path=str(first_trace), reset_after=False)
                    print(f"[adjacent] {name} window {i+1}/{len(starts)} NLL={loss:.7f}", flush=True)
            trace.flush(header_extra={"stage": name, "split": "train", "windows": starts,
                                     "mp_config_json": str(wrapper)}, reset_after=True)
            trace.disable()
            result = {"name": name, "wrapper": str(wrapper), "window_nll": losses,
                      "mean_nll": sum(losses) / len(losses), "seconds": time.time() - started,
                      "tracker_cost": float(mp_tracker_flop_avg_stoc_len()),
                      "trace": str(trace_path), **trace_cost(trace_path)}
            observed = collector.snapshot() if collector is not None else None
            if profile:
                result["first_window_cost"] = trace_cost(first_trace)
                table = json.loads(Path(json.loads(Path(wrapper).read_text())["threshold_table_path"]).read_text())
                replay = baseline_cost(table, observed)
                if not within_cost(replay, result["cost"], .001):
                    raise ValueError(f"candidate profile replay mismatch: {name}")
                write_json(out / f"{name}_profile.json", observed, exclusive=True)
            write_json(out / f"{name}_nll.json", result, exclusive=True)
            return result, observed

        noop, _ = evaluate("identity_before", base_wrapper, search_starts[:1])
        baseline, profile = evaluate("search_incumbent", base_wrapper, search_starts, profile=True)
        if (baseline["window_nll"][0] != noop["window_nll"][0]
                or any(baseline["first_window_cost"][key] != noop[key]
                       for key in ("total_macs", "total_cycle_macs"))):
            raise ValueError("observe-only profiler changed incumbent loss or cost")
        write_json(out / "profile.json", profile, exclusive=True)
        write_json(out / "identity.json", {"exact_match": True,
                   "unprofiled_nll": noop["window_nll"][0],
                   "profiled_nll": baseline["window_nll"][0],
                   "replayed_cost": baseline_cost(source_table, profile),
                   "trace_cost": baseline["cost"]}, exclusive=True)
        proposals = propose(source_table, profile, transfer_fractions=protocol["transfer_fractions"],
                            max_candidates=protocol["max_candidates"], **limits)
        write_json(out / "proposal_summary.json", {"count": len(proposals), "protocol": protocol,
                   "pool": propose.last_diagnostics,
                   "proposals": [{"name": p["name"], "diagnostics": p["diagnostics"]}
                                 for p in proposals]}, exclusive=True)
        print(f"[adjacent] proposed {len(proposals)} bounded exchanges", flush=True)
        results = []
        for i, proposal in enumerate(proposals):
            name = f"candidate{i:02d}_{proposal['name']}"
            if not all(c.isalnum() or c in "_-" for c in name):
                raise ValueError("unsafe proposal name")
            candidate_table = proposal["table"]
            validate_allocation_only(source_table, candidate_table)
            prior_audit = audit_transition(source_table, candidate_table, profile, **limits)
            if prior_audit.get("feasible") is not True:
                raise ValueError(f"generator emitted a nonlocal candidate: {prior_audit}")
            table_path, wrapper_path = out / f"{name}_table.json", out / f"{name}.json"
            wrapper = copy.deepcopy(source_wrapper)
            wrapper["threshold_table_path"] = str(table_path)
            write_json(table_path, candidate_table, exclusive=True)
            write_json(wrapper_path, wrapper, exclusive=True)
            result, actual_profile = evaluate("search_" + name, wrapper_path, search_starts, profile=True)
            actual_audit = audit_transition(source_table, candidate_table, actual_profile, **limits)
            result.update(proposal=proposal["diagnostics"], locality=actual_audit,
                          locality_feasible=actual_audit.get("feasible") is True,
                          paired_vs_incumbent=paired_stats(result["window_nll"], baseline["window_nll"]),
                          cost_feasible=within_cost(result["cost"], baseline["cost"],
                                                    protocol["cost_tolerance_fraction"]))
            results.append(result)
            write_json(out / "search_results.json", {"baseline": baseline, "candidates": results})
            print(f"[adjacent] {name}: {result['paired_vs_incumbent']} "
                  f"cost_ratio={result['cost']/baseline['cost']:.6f} "
                  f"locality={result['locality_feasible']} eligible={search_eligible(result)}", flush=True)

        restored, _ = evaluate("identity_after", base_wrapper, search_starts[:1])
        if (restored["window_nll"] != noop["window_nll"]
                or any(restored[key] != noop[key] for key in ("total_macs", "total_cycle_macs"))):
            raise ValueError("restoring incumbent changed loss or cost")
        eligible = [r for r in results if search_eligible(r)]
        selection = {"evaluate": False, "wrapper": None, "incumbent_wrapper": str(base_wrapper),
                     "reason": "no cost-feasible local candidate improved search loss",
                     "candidate_count": len(results),
                     "reporting_rule": "best full-test PPL across comparable allocation rounds"}
        if eligible:
            chosen = min(eligible, key=lambda r: r["mean_nll"])
            chosen_wrapper = Path(chosen["wrapper"])
            chosen_table = json.loads(Path(json.loads(chosen_wrapper.read_text())["threshold_table_path"]).read_text())
            ref, _ = evaluate("confirm_incumbent", base_wrapper, confirm_starts)
            new, confirm_profile = evaluate("confirm_candidate", chosen_wrapper, confirm_starts, profile=True)
            stats = paired_stats(new["window_nll"], ref["window_nll"])
            cost_ok = within_cost(new["cost"], ref["cost"], protocol["cost_tolerance_fraction"])
            locality = audit_transition(source_table, chosen_table, confirm_profile, **limits)
            locality_ok = locality.get("feasible") is True
            confirmed = confirmation_qualifies(stats, cost_ok, locality_ok,
                                               protocol["confirmation_z_threshold"])
            selection.update(wrapper=str(chosen_wrapper), evaluate=confirmed, search=chosen,
                             reason="confirmed local improvement; evaluate full test" if confirmed else
                                    "fresh confirmation failed loss, cost, or locality requirement; retain incumbent",
                             confirmation={"baseline": ref, "candidate": new,
                                           "paired_vs_incumbent": stats, "cost_feasible": cost_ok,
                                           "locality": locality, "locality_feasible": locality_ok,
                                           "loss_improvement_confirmed": confirmed})
        preflight(manifest, args.index)
        write_json(out / "selected.json", selection, exclusive=True)
        print(f"[adjacent] COMPLETE: {selection['reason']} evaluate={selection['evaluate']}", flush=True)
        return 0
    except BaseException as exc:
        write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                         "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
