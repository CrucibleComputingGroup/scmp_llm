"""Bounded allocation-only search around an exact archived PRC incumbent.

All search/confirmation losses use WikiText TRAIN. Only the launcher runs full
TEST, once for the best new cost-matched search candidate. Confirmation is
reported separately; it does not silently change best-across-rounds reporting.
No model weights, kernels, RNG, masks, ladders or dispatch statistics are edited.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as f:
            f.write(payload)
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload)
        tmp.replace(path)


def choose_disjoint_starts(total_tokens, ctx, n_search, n_confirm,
                           excluded_starts, seed):
    """Sample aligned disjoint TRAIN blocks without touching any SC RNG state."""
    if min(ctx, n_search, n_confirm) <= 0:
        raise ValueError("positive context and window counts required")
    excluded = [int(s) for s in excluded_starts]
    # Interval exclusion also handles historical files with unaligned starts.
    available = [s for s in range(0, total_tokens - ctx + 1, ctx)
                 if all(s + ctx <= old or old + ctx <= s for old in excluded)]
    if len(available) < n_search + n_confirm:
        raise ValueError("not enough unused TRAIN windows")
    rng = random.Random(seed)  # local corpus sampling; no torch/kernel reseeding

    def stratified(pool, n):
        return [pool[rng.randrange(i * len(pool) // n,
                                   (i + 1) * len(pool) // n)] for i in range(n)]

    search = stratified(available, n_search)
    search_set = set(search)
    confirm = stratified([s for s in available if s not in search_set], n_confirm)
    assert len(set(search + confirm)) == n_search + n_confirm
    return search, confirm


def validate_allocation_only(base, candidate):
    """Permit only deployed PRC and attention threshold changes, nothing else."""
    if base == candidate:
        raise ValueError("candidate is unchanged")
    normalized = copy.deepcopy(candidate)
    for section, allowed_ops in (("per_row_chunk", None), (None, {"qk", "av"})):
        before = (base.get(section, {}) if section else base).get("buckets", {})
        after = (candidate.get(section, {}) if section else candidate).get("buckets", {})
        norm = (normalized.get(section, {}) if section else normalized).get("buckets", {})
        if before.keys() != after.keys():
            raise ValueError("allocation bucket keys changed")
        for key, entry in before.items():
            if allowed_ops is not None and key.split(":")[0] not in allowed_ops:
                continue
            a, b = entry.get("thresholds"), after[key].get("thresholds")
            if a is None or b is None or len(a) != len(b):
                raise ValueError(f"threshold shape changed: {key}")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in b):
                raise ValueError(f"nonfinite or out-of-range thresholds: {key}")
            if list(b) != sorted(b, reverse=section is None):
                raise ValueError(f"nonmonotone thresholds: {key}")
            norm[key]["thresholds"] = copy.deepcopy(a)
    if normalized != base:
        raise ValueError("candidate changes fields other than allowed thresholds")


def paired_stats(candidate, reference):
    if len(candidate) != len(reference) or len(candidate) < 2:
        raise ValueError("paired losses require equal counts >= 2")
    if not all(math.isfinite(x) for x in list(candidate) + list(reference)):
        raise ValueError("nonfinite loss")
    diffs = [a - b for a, b in zip(candidate, reference)]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs)
                   / (len(diffs) - 1) / len(diffs))
    return {"mean_dnll": mean, "se": se,
            "z": mean / se if se else (0.0 if mean == 0 else None),
            "dppl_pct": math.expm1(mean) * 100}


def within_cost(candidate_cost, baseline_cost, tolerance):
    return (all(math.isfinite(x) and x > 0 for x in (candidate_cost, baseline_cost))
            and abs(candidate_cost / baseline_cost - 1) <= tolerance + 1e-12)


def trace_cost(path):
    payload = json.loads(Path(path).read_text())
    macs = sum(float(g["macs"]) for g in payload["groups"])
    cycles = sum(float(g["macs"]) * float(g["stoc_len"])
                 for g in payload["groups"])
    if not math.isfinite(macs) or not math.isfinite(cycles) or min(macs, cycles) <= 0:
        raise ValueError("empty/nonfinite exact SC trace")
    return {"cost": cycles / macs, "total_macs": macs, "total_cycle_macs": cycles}


def preflight(manifest, index):
    for filename, expected in manifest.get("code_hashes", {}).items():
        if sha256(filename) != expected:
            raise ValueError(f"submitted experiment code changed: {filename}")
    cell = manifest["cells"][index]
    for filename, expected in cell["hashes"].items():
        if sha256(filename) != expected:
            raise ValueError(f"frozen input changed: {filename}")
    wrapper = json.loads(Path(cell["incumbent_wrapper"]).read_text())
    table = json.loads(Path(cell["incumbent_table"]).read_text())
    actual_table = Path(wrapper["threshold_table_path"])
    if not actual_table.is_absolute():
        actual_table = Path(cell["incumbent_wrapper"]).parent / actual_table
    if actual_table.resolve() != Path(cell["incumbent_table"]).resolve():
        raise ValueError("incumbent wrapper/table mismatch")
    if not table.get("per_row_chunk", {}).get("buckets"):
        raise ValueError("incumbent has no deployed per-group thresholds")
    if table["model_path"] != cell["model_path"]:
        raise ValueError("model mismatch")
    return cell, wrapper, table


def configure_environment(cell):
    for key in ("SC_RNG_GRID", "SC_RNG_GRID_ATTN", "SC_RNG_GRID_QK", "SC_RNG_GRID_AV",
                "SC_ATTN_SMOOTH_JSON", "SC_HYBRID_FORCE_INT_BITS"):
        if os.environ.get(key):
            raise ValueError(f"forbidden inherited numerical variant: {key}")
    if os.environ.get("SC_PRC_ROWSHARED", "0") != "0":
        raise ValueError("row-shared ablation is forbidden")
    if os.environ.get("SC_DISABLE_OWEN", "0") != "0":
        raise ValueError("Owen disabling is forbidden")
    if os.environ.get("SC_SCRAMBLE_RESCALE", "1") != "1":
        raise ValueError("scramble rescale variant is forbidden")
    for key, value in (("SC_OWEN_MODE", "bitrev"), ("SC_SCRAMBLE_MASKS", "64")):
        if os.environ.get(key, value) != value:
            raise ValueError(f"wrong protocol: {key}")
        os.environ[key] = value
    os.environ.update(QUANT_CONFIG="mp", MP_CONFIG_JSON=cell["incumbent_wrapper"],
                      SC_HYBRID_CONFIG_JSON=cell["hybrid_config"], FRONTEND="awq",
                      CTX="2048", STRIDE="2048", PPL_WINDOW_BATCH_SIZE="1",
                      PPL_MAX_TOKENS="0", SQ_ALPHA="0.5",
                      ACT_SCALES_DIR=str(REPO.parent / "hpca_results/llm/ppl/mp_best/act_scales"))
    scales_dir = Path(os.environ["ACT_SCALES_DIR"]) / "awq_scales"
    os.environ["AWQ_SCALES_DIR"] = str(scales_dir)
    os.environ["AWQ_OBJ_BITS"] = "4"
    cache = scales_dir / f"awq_scales_{cell['model_path'].replace('/', '_')}_b4.pt"
    if not cache.is_file():
        raise ValueError(f"frozen AWQ cache missing; refusing frontend recalibration: {cache}")
    os.environ.pop("SC_MP_TRACE", None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--search-windows", type=int, default=6)
    ap.add_argument("--confirm-windows", type=int, default=16)
    ap.add_argument("--data-seed", type=int, default=260926)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--transfer-fraction", type=float, default=0.02)
    ap.add_argument("--cost-tolerance", type=float, default=0.01)
    args = ap.parse_args()
    if not (0 < args.transfer_fraction <= .03 and 0 < args.cost_tolerance <= .01
            and 0 < args.max_candidates <= 8 and args.search_windows >= 2
            and args.confirm_windows >= 2):
        raise ValueError("pilot limits exceeded")
    manifest = json.loads(Path(args.manifest).read_text())
    cell, source_wrapper, source_table = preflight(manifest, args.index)
    if args.preflight:
        print(f"[local] preflight PASS {cell['id']}", flush=True)
        return 0
    configure_environment(cell)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "driver_started.json", {"time": time.time(), "args": vars(args),
                                            "cell": cell}, exclusive=True)
    try:
        import torch
        from datasets import load_dataset
        from benchmark.quant.eval_quant import build_sc_model
        from benchmark.ppl.calibrate_mp_thresholds import _select_int_swap_windows
        from benchmark.ppl.prc_local_proposals import ProfileCollector, propose, baseline_cost
        from loader import apply_mp_config_from_env
        from model.sc_common import SCLinear, mp_tracker_reset, mp_tracker_flop_avg_stoc_len
        from scmp_kernels import trace

        # Snapshot the exact incumbent, including immutable numerical settings.
        base_table = out / "incumbent_table.json"
        base_wrapper = out / "incumbent.json"
        write_json(base_table, source_table, exclusive=True)
        snapshot_wrapper = copy.deepcopy(source_wrapper)
        snapshot_wrapper["threshold_table_path"] = str(base_table)
        write_json(base_wrapper, snapshot_wrapper, exclusive=True)
        model, tok = build_sc_model(cell["model_path"], "mp", mp_table=str(base_wrapper))
        model.eval()
        dev = next(model.parameters()).device
        if dev.type != "cuda":
            raise ValueError("pilot must run on its Slurm GPU")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        excluded = set()
        historical = []
        for n in (6, 8, 12, 14):
            _, starts = _select_int_swap_windows(enc, 2048, n, sampling="stratified", seed=0)
            excluded.update(int(s) for s in starts)
        for p in sorted(Path(cell["incumbent_wrapper"]).parent.glob(
                f"{cell['model']}_t*_heldout_nll.json")):
            values = json.loads(p.read_text()).get("windows", [])
            if not all(isinstance(s, int) for s in values):
                raise ValueError(f"unexpected historical window schema: {p}")
            excluded.update(values)
            historical.append({"path": str(p), "sha256": sha256(p)})
        search_starts, confirm_starts = choose_disjoint_starts(
            len(enc), 2048, args.search_windows, args.confirm_windows, excluded, args.data_seed)
        windows = {"search": search_starts, "confirm": confirm_starts}
        write_json(out / "windows.json", {"split": "train", "ctx": 2048, "starts": windows,
                   "excluded": sorted(excluded), "historical_files": historical,
                   "data_seed": args.data_seed,
                   "token_ids_sha256": hashlib.sha256(enc.numpy().tobytes()).hexdigest()}, exclusive=True)

        def swap(wrapper):
            os.environ["MP_CONFIG_JSON"] = str(wrapper)
            model.config.sc_mp_config = None
            apply_mp_config_from_env(model)
            mp = model.config.sc_mp_config
            modules = [m for m in model.modules() if isinstance(m, SCLinear)]
            if mp is None or not modules or any(m._sc_config.sc_mp_config is not mp for m in modules):
                raise ValueError("MP table swap did not reach every SCLinear")

        def one_loss(start):
            ids = enc[start:start + 2048].unsqueeze(0).to(dev)
            with torch.no_grad():
                value = float(model(input_ids=ids, labels=ids).loss)
            if not math.isfinite(value):
                raise ValueError("nonfinite model NLL")
            return value

        def evaluate(name, wrapper, starts, *, profile=False):
            swap(wrapper)
            trace.reset()
            trace_path = out / f"{name}_trace.json"
            trace.enable(str(trace_path), mode="summary")
            mp_tracker_reset()
            losses = []
            collector = ProfileCollector(model) if profile else None
            from contextlib import nullcontext
            started = time.time()
            with collector if collector is not None else nullcontext():
                for i, start in enumerate(starts):
                    losses.append(one_loss(start))
                    if profile and i == 0:
                        first_trace = out / f"{name}_first_window_trace.json"
                        trace.flush(path=str(first_trace), reset_after=False)
                    print(f"[local] {name} window {i+1}/{len(starts)} NLL={losses[-1]:.7f}", flush=True)
            trace.flush(header_extra={"stage": name, "split": "train", "windows": starts,
                                     "mp_config_json": str(wrapper)}, reset_after=True)
            trace.disable()
            result = {"name": name, "wrapper": str(wrapper), "window_nll": losses,
                      "mean_nll": sum(losses) / len(losses), "seconds": time.time() - started,
                      "tracker_cost": float(mp_tracker_flop_avg_stoc_len()),
                      "trace": str(trace_path), **trace_cost(trace_path)}
            if profile:
                result["first_window_cost"] = trace_cost(first_trace)
            write_json(out / f"{name}_nll.json", result, exclusive=True)
            return result, collector.snapshot() if collector is not None else None

        noop, _ = evaluate("identity_before", base_wrapper, search_starts[:1])
        noop_loss = noop["window_nll"][0]
        baseline, profile = evaluate("search_incumbent", base_wrapper, search_starts, profile=True)
        if baseline["window_nll"][0] != noop_loss:
            raise ValueError("observe-only profiler changed baseline NLL")
        if any(baseline["first_window_cost"][key] != noop[key]
               for key in ("total_macs", "total_cycle_macs")):
            raise ValueError("observe-only profiler changed baseline SC cost")
        replay = baseline_cost(source_table, profile)
        if not within_cost(replay, baseline["cost"], 0.001):
            raise ValueError(f"profile replay cost {replay} differs from exact trace {baseline['cost']}")
        write_json(out / "profile.json", profile, exclusive=True)
        write_json(out / "identity.json", {"unprofiled_nll": noop_loss,
                   "profiled_nll": baseline["window_nll"][0], "exact_match": True,
                   "replayed_cost": replay, "trace_cost": baseline["cost"]}, exclusive=True)

        proposals = propose(source_table, profile, transfer_fraction=args.transfer_fraction,
                            max_candidates=args.max_candidates)
        results = []
        for index, proposal in enumerate(proposals):
            name = f"candidate{index:02d}_{proposal['name']}"
            if not all(c.isalnum() or c in "_-" for c in name):
                raise ValueError("unsafe proposal name")
            validate_allocation_only(source_table, proposal["table"])
            table_path, wrapper_path = out / f"{name}_table.json", out / f"{name}.json"
            wrapper = copy.deepcopy(source_wrapper)
            wrapper["threshold_table_path"] = str(table_path)
            write_json(table_path, proposal["table"], exclusive=True)
            write_json(wrapper_path, wrapper, exclusive=True)
            result, _ = evaluate("search_" + name, wrapper_path, search_starts)
            result["proposal"] = proposal["diagnostics"]
            result["paired_vs_incumbent"] = paired_stats(result["window_nll"], baseline["window_nll"])
            result["cost_feasible"] = within_cost(result["cost"], baseline["cost"], args.cost_tolerance)
            results.append(result)
            write_json(out / "search_results.json", {"baseline": baseline, "candidates": results})
            print(f"[local] {name}: {result['paired_vs_incumbent']} cost_ratio="
                  f"{result['cost']/baseline['cost']:.6f} feasible={result['cost_feasible']}", flush=True)

        restored, _ = evaluate("identity_after", base_wrapper, search_starts[:1])
        if (restored["window_nll"] != noop["window_nll"]
                or any(restored[key] != noop[key] for key in ("total_macs", "total_cycle_macs"))):
            raise ValueError("incumbent replay after search changed NLL or SC cost")
        feasible = [r for r in results if r["cost_feasible"]]
        selection = {"evaluate": False, "wrapper": None, "incumbent_wrapper": str(base_wrapper),
                     "reason": "no cost-matched new candidate", "candidate_count": len(results),
                     "reporting_rule": "best full-test PPL across comparable allocation rounds"}
        if feasible:
            chosen = min(feasible, key=lambda r: r["mean_nll"])
            ref, _ = evaluate("confirm_incumbent", base_wrapper, confirm_starts)
            new, _ = evaluate("confirm_candidate", chosen["wrapper"], confirm_starts)
            stats = paired_stats(new["window_nll"], ref["window_nll"])
            cost_ok = within_cost(new["cost"], ref["cost"], args.cost_tolerance)
            confirmed = stats["mean_dnll"] < 0 and stats["z"] is not None and stats["z"] < -1.5
            selection.update(wrapper=chosen["wrapper"], evaluate=cost_ok,
                             reason="full-test best new search candidate; confirmation reported independently"
                             if cost_ok else "confirmation cost mismatch; no comparable full test",
                             search=chosen, confirmation={"baseline": ref, "candidate": new,
                             "paired_vs_incumbent": stats, "cost_feasible": cost_ok,
                             "loss_improvement_confirmed": confirmed and cost_ok})
        # Recheck sources to ensure a simultaneous process did not replace an incumbent.
        preflight(manifest, args.index)
        write_json(out / "selected.json", selection, exclusive=True)
        print(f"[local] COMPLETE: {selection['reason']} evaluate={selection['evaluate']}", flush=True)
        return 0
    except BaseException as exc:
        write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                         "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
