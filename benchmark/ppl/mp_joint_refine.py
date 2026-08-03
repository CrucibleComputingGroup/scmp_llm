"""Validation-loss joint refinement of MP precision levels and thresholds.

V11 starts from an existing AdaptiveMPConfig (not from v9/v10), keeps the
dispatch metric, protected-channel set, and hybrid INT schedule fixed, and
alternates two discrete coordinate steps:

* project precision levels to an exact MAC-weighted target and exchange high
  precision for a higher floor;
* move a small amount of profiled MAC mass across selected threshold
  boundaries, then re-project the levels to the same target.

Candidate selection uses a fixed WikiText validation prefix.  An optional
larger validation confirmation is reported after search; the test split is
never loaded here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from benchmark.ppl.mp_ladder_refine import (
    _atomic_json,
    _build_parent_model,
    _eval_once,
    _load_eval_stream,
    _load_trace,
    _protected_info,
    _write_wrapper,
    aggregate_trace_weights,
    nominal_target_cost,
    propose_floor_exchange,
    resolve_parent_table,
    weighted_cost,
)


OP_GROUPS = {
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "attn_linear": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "attn_matmul": ("qk", "av"),
}

THRESHOLD_SPECS = (
    {
        "name": "promote_mlp_floor",
        "moves": (("mlp", -1, +1.0),),
    },
    {
        "name": "promote_attn_linear_floor",
        "moves": (("attn_linear", -1, +1.0),),
    },
    {
        "name": "demote_attn_matmul_top",
        "moves": (("attn_matmul", 0, -1.0),),
    },
    {
        "name": "transfer_attn_to_mlp",
        "moves": (
            ("mlp", -1, +1.0),
            ("attn_matmul", 0, -1.0),
        ),
    },
)


def project_levels_to_budget(
    levels: Iterable[int],
    level_weights: Iterable[float],
    *,
    target_cost: float,
    protected_weight: float = 0.0,
    protected_stoc_len: int = 0,
    min_gap: int = 1,
    min_level: int = 1,
    max_level: int = 128,
) -> dict:
    """Project an ordered integer ladder to a target measured occupancy.

    Under-budget ladders spend on the lowest rung first (propagating upward
    only when ordering requires it).  Over-budget ladders remove cycles from
    the highest rung first.  Every one-cycle state is considered and the
    closest state is returned.
    """
    current = [int(x) for x in levels]
    weights = [float(x) for x in level_weights]
    if len(current) < 2 or len(current) != len(weights):
        raise ValueError("levels and weights must have equal length >= 2")
    if any(current[i] <= current[i + 1]
           for i in range(len(current) - 1)):
        raise ValueError(f"levels must be strictly descending: {current}")
    if not (0 < min_level <= max_level):
        raise ValueError("invalid level bounds")

    def cost(xs: list[int]) -> float:
        return weighted_cost(
            xs, weights, protected_weight, protected_stoc_len)

    start = current.copy()
    start_cost = cost(current)
    best = current.copy()
    best_cost = start_cost
    path: list[dict] = []

    for _ in range(20000):
        current_cost = cost(current)
        under = current_cost < target_cost
        trials = []
        if under:
            # Prefer floor uplift, then progressively higher receivers.
            for receiver in range(len(current) - 1, -1, -1):
                trial = current.copy()
                trial[receiver] += 1
                for i in range(receiver - 1, -1, -1):
                    trial[i] = max(trial[i], trial[i + 1] + min_gap)
                if trial[0] > max_level:
                    continue
                trials.append((receiver, trial, cost(trial)))
        else:
            # Prefer taking cycles from the high rungs.  Do not propagate a
            # decrement downward: another independently feasible donor is
            # cleaner and preserves the floor whenever possible.
            for donor in range(0, len(current)):
                lowered = current[donor] - 1
                lower_bound = (current[donor + 1] + min_gap
                               if donor + 1 < len(current) else min_level)
                if lowered < lower_bound:
                    continue
                trial = current.copy()
                trial[donor] = lowered
                trials.append((donor, trial, cost(trial)))
        if not trials:
            break
        # Direction preference is already encoded in the index ordering.
        idx, trial, trial_cost = min(
            trials, key=lambda x: (abs(x[2] - target_cost),
                                   -x[0] if under else x[0]))
        if trial == current:
            break
        current = trial
        path.append({
            "direction": "up" if under else "down",
            "index": int(idx),
            "levels": current.copy(),
            "predicted_cost": float(trial_cost),
        })
        if abs(trial_cost - target_cost) < abs(best_cost - target_cost):
            best = current.copy()
            best_cost = trial_cost
        # Once the next directed step crossed the target and did not improve
        # the closest state, further steps in that direction cannot help.
        if ((under and trial_cost >= target_cost)
                or (not under and trial_cost <= target_cost)):
            break

    return {
        "type": "integer_budget_projection",
        "old_levels": start,
        "levels": best,
        "target_cost": float(target_cost),
        "start_cost": float(start_cost),
        "predicted_cost": float(best_cost),
        "predicted_budget_error": float(best_cost - target_cost),
        "min_gap": int(min_gap),
        "min_level": int(min_level),
        "max_level": int(max_level),
        "moves": path,
    }


def _hist_values(profile: dict) -> list[float]:
    bins = int(profile["bins"])
    if bins < 2:
        raise ValueError("metric profile requires at least two bins")
    return [i / float(bins - 1) for i in range(bins)]


def _payload_thresholds(table: dict, group_key: str, op: str) -> list[float]:
    payload = (table.get("buckets") or {}).get(group_key)
    if payload is None:
        payload = (table.get("operator_defaults") or {}).get(op)
    if payload is None or "thresholds" not in payload:
        raise ValueError(f"no thresholds for profiled group {group_key}")
    return [float(x) for x in payload["thresholds"]]


def profile_level_weights(
    table: dict,
    profile: dict,
    *,
    protected_weight: float,
) -> list[float]:
    """Predict global adaptive MAC occupancy from normalized-metric hists."""
    n_levels = len(table["stoc_len_levels"])
    raw = [0.0] * n_levels
    values = _hist_values(profile)
    for key, group in (profile.get("groups") or {}).items():
        hist = [float(x) for x in group["mac_weighted_hist"]]
        if len(hist) != len(values):
            raise ValueError(f"profile histogram length mismatch for {key}")
        thresholds = _payload_thresholds(table, key, str(group["op"]))
        if len(thresholds) != n_levels - 1:
            raise ValueError(f"threshold count mismatch for {key}")
        for value, weight in zip(values, hist):
            level_idx = n_levels - 1
            for idx, threshold in enumerate(thresholds):
                upper_ok = idx == 0 or value < thresholds[idx - 1]
                if value >= threshold and upper_ok:
                    level_idx = idx
                    break
            raw[level_idx] += weight
    total = sum(raw)
    adaptive_weight = 1.0 - float(protected_weight)
    if total <= 0.0 or adaptive_weight <= 0.0:
        raise ValueError("metric profile contains no adaptive MAC weight")
    return [x / total * adaptive_weight for x in raw]


def _shift_one_boundary(
    thresholds: list[float],
    hist: list[float],
    values: list[float],
    *,
    boundary: int,
    mass_delta: float,
) -> tuple[list[float], dict]:
    n_boundaries = len(thresholds)
    idx = boundary if boundary >= 0 else n_boundaries + boundary
    if not 0 <= idx < n_boundaries:
        raise ValueError(f"invalid boundary {boundary} for {thresholds}")
    total = sum(hist)
    if total <= 0.0:
        return thresholds.copy(), {"changed": False, "reason": "empty_hist"}
    old = float(thresholds[idx])
    old_high = sum(w for v, w in zip(values, hist) if v >= old)
    target_high = min(max(old_high + mass_delta * total, 0.0), total)
    lower = float(thresholds[idx + 1]) if idx + 1 < n_boundaries else 0.0
    upper = float(thresholds[idx - 1]) if idx > 0 else 1.0
    candidates = sorted({lower, upper, old, *(
        v for v in values if lower <= v <= upper
    )})
    chosen = min(
        candidates,
        key=lambda t: (
            abs(sum(w for v, w in zip(values, hist) if v >= t)
                - target_high),
            abs(t - old),
        ),
    )
    out = thresholds.copy()
    out[idx] = float(chosen)
    if any(out[i] < out[i + 1] - 1e-12
           for i in range(len(out) - 1)):
        raise AssertionError(f"threshold shift broke ordering: {out}")
    new_high = sum(w for v, w in zip(values, hist) if v >= chosen)
    return out, {
        "changed": abs(chosen - old) > 1e-12,
        "boundary": int(idx),
        "old_threshold": old,
        "new_threshold": float(chosen),
        "requested_mass_delta": float(mass_delta),
        "realized_mass_delta": float((new_high - old_high) / total),
    }


def shift_threshold_group(
    table: dict,
    profile: dict,
    *,
    ops: Iterable[str],
    boundary: int,
    mass_delta: float,
) -> tuple[dict, list[dict]]:
    """Shift one boundary for every profiled bucket in an operator group."""
    out = copy.deepcopy(table)
    ops = set(ops)
    values = _hist_values(profile)
    moves = []
    op_hists: dict[str, list[float]] = {}
    for key, group in (profile.get("groups") or {}).items():
        op = str(group["op"])
        if op not in ops:
            continue
        hist = [float(x) for x in group["mac_weighted_hist"]]
        payload = (out.get("buckets") or {}).get(key)
        if payload is None:
            continue
        shifted, move = _shift_one_boundary(
            [float(x) for x in payload["thresholds"]],
            hist,
            values,
            boundary=boundary,
            mass_delta=mass_delta,
        )
        payload["thresholds"] = shifted
        move.update({"group": key, "op": op})
        moves.append(move)
        if op not in op_hists:
            op_hists[op] = [0.0] * len(hist)
        op_hists[op] = [a + b for a, b in zip(op_hists[op], hist)]
    # Keep the fallback operator thresholds consistent with the same move.
    for op, hist in op_hists.items():
        payload = (out.get("operator_defaults") or {}).get(op)
        if payload is None:
            continue
        shifted, move = _shift_one_boundary(
            [float(x) for x in payload["thresholds"]],
            hist,
            values,
            boundary=boundary,
            mass_delta=mass_delta,
        )
        payload["thresholds"] = shifted
        move.update({"group": f"operator_default:{op}", "op": op})
        moves.append(move)
    return out, moves


def make_joint_table(
    source_table: dict,
    levels: Iterable[int],
    *,
    root_parent_table: Path,
    target_cost: float,
    action: dict,
    command: str,
) -> dict:
    levels = [int(x) for x in levels]
    out = copy.deepcopy(source_table)
    old_levels = [int(x) for x in out["stoc_len_levels"]]
    if len(levels) != len(old_levels):
        raise ValueError("joint refinement must preserve class count")
    out["stoc_len_levels"] = levels
    for section in ("operator_defaults", "buckets"):
        for payload in (out.get(section) or {}).values():
            counts = [int(x) for x in payload.get("counts") or []]
            if len(counts) == len(levels) and sum(counts) > 0:
                payload["avg_stoc_len"] = (
                    sum(c * sl for c, sl in zip(counts, levels)) / sum(counts))
            if "level_mean_error" in payload:
                payload["pass1_level_mean_error"] = payload.pop("level_mean_error")
            if "avg_error" in payload:
                payload["pass1_avg_error"] = payload.pop("avg_error")
    out.pop("expected_avg_stoc_len", None)
    out["expected_flop_avg_stoc_len"] = float(
        action.get("predicted_cost", target_cost))
    method = str(source_table.get("method", "adaptive_mp"))
    if not method.endswith("_ppl_joint_refine_v11"):
        out["method"] = method + "_ppl_joint_refine_v11"
    out["pass2_joint_refine"] = {
        "schema": "scmp-ppl-joint-refine-v1",
        "root_parent_table": str(root_parent_table),
        "source_levels": old_levels,
        "target_flop_avg_stoc_len": float(target_cost),
        "dispatch_metrics_frozen": True,
        "protected_channels_frozen": True,
        "hybrid_mask_external_and_frozen": True,
        "action": action,
        "command": command,
    }
    return out


def _candidate_paths(branch: Path, stem: str) -> tuple[Path, Path, Path, Path]:
    return (
        branch / f"{stem}.json",
        branch / f"{stem}_wrapper.json",
        branch / f"{stem}_trace.json",
        branch / f"{stem}_profile.json",
    )


def _evaluate_candidate(
    model,
    enc,
    *,
    ctx: int,
    branch: Path,
    stem: str,
    table: dict,
    levels: list[int],
    split: str = "validation",
    profile_bins: int = 257,
) -> tuple[dict, Path, dict]:
    table_path, wrapper_path, trace_path, profile_path = _candidate_paths(
        branch, stem)
    _atomic_json(table_path, table)
    _write_wrapper(wrapper_path, levels, table_path)
    result = _eval_once(
        model,
        enc,
        ctx=ctx,
        levels=levels,
        table_path=table_path,
        wrapper_path=wrapper_path,
        trace_path=trace_path,
        split=split,
        metric_profile_path=profile_path,
        metric_profile_bins=profile_bins,
    )
    with profile_path.open() as f:
        profile = json.load(f)
    return result, table_path, profile


def _actual_occupancy(
    result: dict,
    levels: list[int],
    *,
    protected_sl: int | None,
    protected_hint: float,
) -> dict:
    occupancy = aggregate_trace_weights(
        _load_trace(Path(result["trace"])),
        levels,
        protected_stoc_len=protected_sl,
        protected_weight_hint=protected_hint,
    )
    result["trace_flop_avg_stoc_len"] = float(occupancy["cost"])
    return occupancy


def _short_action(action: dict) -> dict:
    out = copy.deepcopy(action)
    if len(out.get("moves") or []) > 8:
        moves = out["moves"]
        out["moves"] = moves[:4] + [{"omitted": len(moves) - 8}] + moves[-4:]
    return out


def _target_slug(target: float) -> str:
    return f"target{target:.3f}".replace(".", "p")


def _parse_targets(spec: str, *, parent_cost: float, nominal_cost: float) -> list[float]:
    targets = []
    # ':' is the Slurm-safe separator: sbatch --export reserves commas for
    # separating environment assignments.
    for token in spec.replace(":", ",").split(","):
        token = token.strip().lower()
        value = (parent_cost if token == "parent"
                 else nominal_cost if token == "nominal"
                 else float(token))
        if value <= 0.0:
            raise ValueError(f"invalid target {token!r}")
        if not any(abs(value - x) < 1e-6 for x in targets):
            targets.append(float(value))
    return targets


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-wrapper", required=True, type=Path)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--targets", default="parent,nominal,36,40,48")
    p.add_argument("--search-tokens", type=int, default=16384)
    p.add_argument("--confirm-tokens", type=int, default=32768)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--ladder-rounds", type=int, default=2)
    p.add_argument("--threshold-rounds", type=int, default=1)
    p.add_argument("--threshold-mass-step", type=float, default=0.02)
    p.add_argument("--floor-step", type=int, default=4)
    p.add_argument("--budget-tol", type=float, default=0.35)
    p.add_argument("--min-nll-improvement", type=float, default=0.0)
    p.add_argument("--budget-corrections", type=int, default=2)
    p.add_argument("--profile-bins", type=int, default=257)
    p.add_argument("--min-level", type=int, default=1)
    p.add_argument("--max-level", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.search_tokens <= 0 or args.confirm_tokens < args.search_tokens:
        raise SystemExit("require 0 < search-tokens <= confirm-tokens")
    if not 0.0 < args.threshold_mass_step < 0.25:
        raise SystemExit("threshold-mass-step must be in (0, 0.25)")
    root_parent_wrapper = args.parent_wrapper.resolve()
    wrapper, parent_table_path, parent_table = resolve_parent_table(
        root_parent_wrapper)
    root_parent_table_path = parent_table_path
    parent_levels = [int(x) for x in wrapper["stoc_len_levels"]]
    if args.max_level > 128 or max(parent_levels) > 128:
        raise SystemExit(
            "v11 is fixed to sc_prec=8: maximum stream length is 128")
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    runtime_parent_wrapper = root_parent_wrapper
    protected_sl, protected_hint = _protected_info(parent_table)
    command = " ".join(sys.argv)

    model, tokenizer = _build_parent_model(
        args.model_path, runtime_parent_wrapper, args.alpha)
    enc_full = _load_eval_stream(
        tokenizer, split="validation", max_tokens=args.confirm_tokens,
        ctx=args.ctx)
    search_n = min(args.search_tokens, int(enc_full.numel()))
    search_n = search_n // args.ctx * args.ctx
    enc_search = enc_full[:search_n]

    baseline_dir = outdir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_profile_path = baseline_dir / "parent_profile.json"
    baseline = _eval_once(
        model,
        enc_search,
        ctx=args.ctx,
        levels=parent_levels,
        table_path=parent_table_path,
        wrapper_path=runtime_parent_wrapper,
        trace_path=baseline_dir / "parent_trace.json",
        split="validation",
        metric_profile_path=baseline_profile_path,
        metric_profile_bins=args.profile_bins,
    )
    with baseline_profile_path.open() as f:
        baseline_profile = json.load(f)
    baseline_occupancy = _actual_occupancy(
        baseline, parent_levels,
        protected_sl=protected_sl,
        protected_hint=protected_hint,
    )
    parent_cost = float(baseline_occupancy["cost"])
    nominal_cost = nominal_target_cost(parent_table)
    targets = _parse_targets(
        args.targets, parent_cost=parent_cost, nominal_cost=nominal_cost)
    print(
        f"[v11 baseline] levels={parent_levels} ppl={baseline['ppl']:.6f} "
        f"realized={parent_cost:.4f} nominal={nominal_cost:.4f} "
        f"targets={targets}", flush=True)

    all_summaries = []
    for target_idx, target in enumerate(targets):
        branch = outdir / _target_slug(target)
        branch.mkdir(parents=True, exist_ok=True)
        history = [{
            "stage": "parent_reference",
            "accepted": abs(parent_cost - target) <= args.budget_tol,
            "target": target,
            "budget_error": parent_cost - target,
            **baseline,
        }]
        current = dict(baseline)
        current_levels = parent_levels.copy()
        current_table = copy.deepcopy(parent_table)
        current_table_path = parent_table_path
        current_profile = baseline_profile
        current_occupancy = baseline_occupancy

        # First make every branch deployably close to its requested budget.
        for correction in range(args.budget_corrections + 1):
            if abs(current_occupancy["cost"] - target) <= args.budget_tol:
                break
            projection = project_levels_to_budget(
                current_levels,
                current_occupancy["level_weights"],
                target_cost=target,
                protected_weight=current_occupancy["protected_weight"],
                protected_stoc_len=int(protected_sl or 0),
                min_level=args.min_level,
                max_level=args.max_level,
            )
            levels = projection["levels"]
            if levels == current_levels:
                history.append({
                    "stage": "budget_projection",
                    "accepted": False,
                    "reason": "projection_cannot_improve_budget_error",
                    "proposal": _short_action(projection),
                })
                break
            action = _short_action(projection)
            table = make_joint_table(
                current_table, levels,
                root_parent_table=root_parent_table_path,
                target_cost=target,
                action=action,
                command=command,
            )
            stem = f"budget{correction:02d}_levels" + "-".join(map(str, levels))
            result, table_path, profile = _evaluate_candidate(
                model, enc_search,
                ctx=args.ctx, branch=branch, stem=stem,
                table=table, levels=levels,
                profile_bins=args.profile_bins,
            )
            occupancy = _actual_occupancy(
                result, levels,
                protected_sl=protected_sl,
                protected_hint=protected_hint,
            )
            budget_error = occupancy["cost"] - target
            accepted = abs(budget_error) < abs(current_occupancy["cost"] - target)
            history.append({
                "stage": "budget_projection",
                "accepted": accepted,
                "budget_error": budget_error,
                "proposal": action,
                **result,
            })
            if not accepted:
                break
            current = result
            current_levels = levels
            current_table = table
            current_table_path = table_path
            current_profile = profile
            current_occupancy = occupancy

        if abs(current_occupancy["cost"] - target) > args.budget_tol:
            summary = {
                "target": target,
                "status": "budget_projection_failed",
                "history": history,
            }
            _atomic_json(branch / "summary.json", summary)
            all_summaries.append(summary)
            continue

        # V10-style level coordinate steps at the requested branch budget.
        for round_idx in range(1, args.ladder_rounds + 1):
            try:
                proposal = propose_floor_exchange(
                    current_levels,
                    current_occupancy["level_weights"],
                    target_cost=target,
                    protected_weight=current_occupancy["protected_weight"],
                    protected_stoc_len=int(protected_sl or 0),
                    floor_step=args.floor_step,
                )
            except ValueError as exc:
                history.append({
                    "stage": "ladder", "round": round_idx,
                    "accepted": False, "reason": str(exc),
                })
                break
            levels = proposal["levels"]
            action = _short_action(proposal)
            table = make_joint_table(
                current_table, levels,
                root_parent_table=root_parent_table_path,
                target_cost=target,
                action=action,
                command=command,
            )
            stem = f"ladder{round_idx:02d}_levels" + "-".join(map(str, levels))
            result, table_path, profile = _evaluate_candidate(
                model, enc_search,
                ctx=args.ctx, branch=branch, stem=stem,
                table=table, levels=levels,
                profile_bins=args.profile_bins,
            )
            occupancy = _actual_occupancy(
                result, levels,
                protected_sl=protected_sl,
                protected_hint=protected_hint,
            )
            budget_error = occupancy["cost"] - target
            improvement = current["nll"] - result["nll"]
            accepted = (
                abs(budget_error) <= args.budget_tol
                and improvement > args.min_nll_improvement
            )
            history.append({
                "stage": "ladder", "round": round_idx,
                "accepted": accepted,
                "budget_error": budget_error,
                "nll_improvement": improvement,
                "proposal": action,
                **result,
            })
            if not accepted:
                break
            current = result
            current_levels = levels
            current_table = table
            current_table_path = table_path
            current_profile = profile
            current_occupancy = occupancy

        # End-to-end threshold coordinate search.  All candidates in a round
        # start from the same current state; accept the best valid NLL only.
        for threshold_round in range(1, args.threshold_rounds + 1):
            candidates = []
            for spec in THRESHOLD_SPECS:
                threshold_table = copy.deepcopy(current_table)
                threshold_moves = []
                for group_name, boundary, direction in spec["moves"]:
                    threshold_table, moves = shift_threshold_group(
                        threshold_table,
                        current_profile,
                        ops=OP_GROUPS[group_name],
                        boundary=boundary,
                        mass_delta=direction * args.threshold_mass_step,
                    )
                    threshold_moves.extend(moves)
                if not any(m.get("changed") for m in threshold_moves):
                    history.append({
                        "stage": "threshold", "round": threshold_round,
                        "candidate": spec["name"], "accepted": False,
                        "reason": "no_threshold_changed",
                    })
                    continue
                predicted_weights = profile_level_weights(
                    threshold_table,
                    current_profile,
                    protected_weight=current_occupancy["protected_weight"],
                )
                projection = project_levels_to_budget(
                    current_levels,
                    predicted_weights,
                    target_cost=target,
                    protected_weight=current_occupancy["protected_weight"],
                    protected_stoc_len=int(protected_sl or 0),
                    min_level=args.min_level,
                    max_level=args.max_level,
                )
                levels = projection["levels"]
                action = {
                    "type": "threshold_mass_move_with_budget_projection",
                    "candidate": spec["name"],
                    "threshold_mass_step": args.threshold_mass_step,
                    "threshold_moves": threshold_moves,
                    "level_projection": _short_action(projection),
                    "predicted_cost": projection["predicted_cost"],
                }
                table = make_joint_table(
                    threshold_table, levels,
                    root_parent_table=root_parent_table_path,
                    target_cost=target,
                    action=action,
                    command=command,
                )
                stem = (f"threshold{threshold_round:02d}_{spec['name']}_levels"
                        + "-".join(map(str, levels)))
                result, table_path, profile = _evaluate_candidate(
                    model, enc_search,
                    ctx=args.ctx, branch=branch, stem=stem,
                    table=table, levels=levels,
                    profile_bins=args.profile_bins,
                )
                occupancy = _actual_occupancy(
                    result, levels,
                    protected_sl=protected_sl,
                    protected_hint=protected_hint,
                )
                budget_error = occupancy["cost"] - target
                improvement = current["nll"] - result["nll"]
                valid = (
                    abs(budget_error) <= args.budget_tol
                    and improvement > args.min_nll_improvement
                )
                record = {
                    "stage": "threshold", "round": threshold_round,
                    "candidate": spec["name"],
                    "accepted": False,
                    "eligible": valid,
                    "budget_error": budget_error,
                    "nll_improvement": improvement,
                    "action": action,
                    **result,
                }
                history.append(record)
                candidates.append({
                    "valid": valid,
                    "record": record,
                    "result": result,
                    "levels": levels,
                    "table": table,
                    "table_path": table_path,
                    "profile": profile,
                    "occupancy": occupancy,
                })
            valid = [x for x in candidates if x["valid"]]
            if not valid:
                break
            winner = min(valid, key=lambda x: x["result"]["nll"])
            winner["record"]["accepted"] = True
            winner["record"]["reason"] = "best_end_to_end_validation_nll"
            current = winner["result"]
            current_levels = winner["levels"]
            current_table = winner["table"]
            current_table_path = winner["table_path"]
            current_profile = winner["profile"]
            current_occupancy = winner["occupancy"]

        best_wrapper = branch / "best_wrapper.json"
        _write_wrapper(best_wrapper, current_levels, current_table_path)
        confirm_trace = branch / "confirm_validation_trace.json"
        confirm = _eval_once(
            model,
            enc_full,
            ctx=args.ctx,
            levels=current_levels,
            table_path=current_table_path,
            wrapper_path=best_wrapper,
            trace_path=confirm_trace,
            split="validation",
        )
        confirm_occupancy = _actual_occupancy(
            confirm, current_levels,
            protected_sl=protected_sl,
            protected_hint=protected_hint,
        )
        summary = {
            "schema": "scmp-ppl-joint-refine-v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "ok",
            "target": target,
            "model": args.model_path,
            "root_parent_wrapper": str(root_parent_wrapper),
            "root_parent_table": str(root_parent_table_path),
            "parent_levels": parent_levels,
            "parent_search_ppl": baseline["ppl"],
            "parent_search_cost": parent_cost,
            "target_flop_avg_stoc_len": target,
            "best_levels": current_levels,
            "best_search_ppl": current["ppl"],
            "best_search_nll": current["nll"],
            "best_search_cost": current_occupancy["cost"],
            "confirm_validation_ppl": confirm["ppl"],
            "confirm_validation_nll": confirm["nll"],
            "confirm_validation_cost": confirm_occupancy["cost"],
            "best_wrapper": str(best_wrapper),
            "best_table": str(current_table_path),
            "search_tokens": int(enc_search.numel()),
            "confirm_tokens": int(enc_full.numel()),
            "history": history,
            "command": command,
        }
        _atomic_json(branch / "summary.json", summary)
        all_summaries.append(summary)
        print(
            f"[V11 TARGET RESULT] target={target:.4f} "
            f"levels={current_levels} search_ppl={current['ppl']:.6f} "
            f"confirm_ppl={confirm['ppl']:.6f} "
            f"confirm_cost={confirm_occupancy['cost']:.4f}",
            flush=True,
        )

    root_summary = {
        "schema": "scmp-ppl-joint-refine-wave-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model_path,
        "targets": targets,
        "parent_search_ppl": baseline["ppl"],
        "parent_search_cost": parent_cost,
        "branches": all_summaries,
        "command": command,
    }
    _atomic_json(outdir / "summary.json", root_summary)
    with (outdir / "results.tsv").open("w") as f:
        f.write("target\tstatus\tlevels\tsearch_ppl\tsearch_cost\t"
                "confirm_ppl\tconfirm_cost\tbest_wrapper\n")
        for summary in all_summaries:
            f.write(
                f"{summary['target']:.6f}\t{summary['status']}\t"
                f"{','.join(map(str, summary.get('best_levels', [])))}\t"
                f"{summary.get('best_search_ppl', math.nan):.8f}\t"
                f"{summary.get('best_search_cost', math.nan):.6f}\t"
                f"{summary.get('confirm_validation_ppl', math.nan):.8f}\t"
                f"{summary.get('confirm_validation_cost', math.nan):.6f}\t"
                f"{summary.get('best_wrapper', '')}\n"
            )
    print(f"[V11 DONE] summary={outdir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
