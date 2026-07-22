"""Systematic validation-loss refinement of an existing mixed-precision table.

V12 is a second pass over a completed AdaptiveMPConfig.  Unlike V10/V11, the
adaptive class topology is a search variable: a coordinate sweep may change
rung values, insert a new precision class (and its threshold) in any internal
gap or at either endpoint, remove a class, or move an existing threshold
boundary.  Every proposal is projected back to the requested measured MAC
budget before end-to-end PPL evaluation.

Candidates are screened on one validation window and the best few are checked
on a disjoint guard window.  Only a candidate that improves their mean NLL is
accepted.  The selected branch is then confirmed on a longer validation
prefix and evaluated on the full test split for reporting.  The test split is
never used for search.
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


from benchmark.ppl.mp_joint_refine import (
    OP_GROUPS,
    _actual_occupancy,
    _evaluate_candidate,
    _hist_values,
    _parse_targets,
    _short_action,
    _target_slug,
    profile_level_weights,
    project_levels_to_budget,
    shift_threshold_group,
)
from benchmark.ppl.mp_ladder_refine import (
    _atomic_json,
    _build_parent_model,
    _eval_once,
    _load_eval_stream,
    _protected_info,
    _write_wrapper,
    nominal_target_cost,
    resolve_parent_table,
    weighted_cost,
)


def project_levels_constrained(
    levels: Iterable[int],
    level_weights: Iterable[float],
    *,
    target_cost: float,
    protected_weight: float = 0.0,
    protected_stoc_len: int = 0,
    locked: dict[int, int] | None = None,
    min_level: int = 1,
    max_level: int = 128,
) -> dict:
    """Budget-project a ladder without undoing selected coordinate moves."""
    start = [int(x) for x in levels]
    weights = [float(x) for x in level_weights]
    locked = {int(i): int(v) for i, v in (locked or {}).items()}
    if len(start) < 2 or len(start) != len(weights):
        raise ValueError("levels and weights must have equal length >= 2")
    if any(start[i] <= start[i + 1] for i in range(len(start) - 1)):
        raise ValueError(f"levels must be strictly descending: {start}")
    if start[0] > max_level or start[-1] < min_level:
        raise ValueError("ladder is outside configured bounds")
    if any(not 0 <= i < len(start) or start[i] != value
           for i, value in locked.items()):
        raise ValueError("locked coordinates must match the input ladder")

    def cost(xs: list[int]) -> float:
        return weighted_cost(
            xs, weights, protected_weight, protected_stoc_len)

    current = start.copy()
    best = current.copy()
    best_cost = cost(current)
    path = []
    for _ in range(20000):
        current_cost = cost(current)
        under = current_cost < target_cost
        trials = []
        order = (range(len(current) - 1, -1, -1)
                 if under else range(len(current)))
        for idx in order:
            if idx in locked:
                continue
            trial = current.copy()
            trial[idx] += 1 if under else -1
            if not min_level <= trial[idx] <= max_level:
                continue
            if idx > 0 and trial[idx - 1] <= trial[idx]:
                continue
            if idx + 1 < len(trial) and trial[idx] <= trial[idx + 1]:
                continue
            trials.append((idx, trial, cost(trial)))
        if not trials:
            break
        idx, trial, trial_cost = min(
            trials,
            key=lambda item: (
                abs(item[2] - target_cost),
                -item[0] if under else item[0],
            ),
        )
        current = trial
        path.append({
            "direction": "up" if under else "down",
            "index": idx,
            "levels": current.copy(),
            "predicted_cost": trial_cost,
        })
        if abs(trial_cost - target_cost) < abs(best_cost - target_cost):
            best = current.copy()
            best_cost = trial_cost
        if ((under and trial_cost >= target_cost)
                or (not under and trial_cost <= target_cost)):
            break
    return {
        "type": "constrained_integer_budget_projection",
        "old_levels": start,
        "levels": best,
        "target_cost": float(target_cost),
        "predicted_cost": float(best_cost),
        "predicted_budget_error": float(best_cost - target_cost),
        "locked": locked,
        "min_level": int(min_level),
        "max_level": int(max_level),
        "moves": path,
    }


def _profile_hists(profile: dict) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    by_bucket = {}
    by_op: dict[str, list[float]] = {}
    for key, group in (profile.get("groups") or {}).items():
        hist = [float(x) for x in group["mac_weighted_hist"]]
        by_bucket[key] = hist
        op = str(group["op"])
        if op not in by_op:
            by_op[op] = [0.0] * len(hist)
        by_op[op] = [a + b for a, b in zip(by_op[op], hist)]
    return by_bucket, by_op


def _split_boundary(
    thresholds: list[float],
    hist: list[float] | None,
    values: list[float],
    *,
    insert_idx: int,
    promote_fraction: float,
) -> tuple[list[float], dict]:
    """Split one class at ``insert_idx`` and add one threshold.

    ``promote_fraction`` is the fraction of the split class assigned to its
    upper sub-interval.  For an internal/top insertion that is the new class;
    for a new bottom class it is the retained old floor.  Searching symmetric
    fractions therefore covers both endpoint orientations.
    """
    old_class_count = len(thresholds) + 1
    if not 0 <= insert_idx <= old_class_count:
        raise ValueError("insert_idx is outside the ladder endpoints")
    if insert_idx == 0:
        upper = 1.0
        lower = float(thresholds[0]) if thresholds else 0.0
    elif insert_idx == old_class_count:
        upper = float(thresholds[-1]) if thresholds else 1.0
        lower = 0.0
    else:
        upper = float(thresholds[insert_idx - 1])
        lower = (float(thresholds[insert_idx])
                 if insert_idx < len(thresholds) else 0.0)
    if upper < lower:
        raise ValueError(f"unordered threshold interval [{lower}, {upper}]")
    if hist is None or sum(hist) <= 0.0 or upper - lower <= 1e-12:
        chosen = lower + (upper - lower) * (1.0 - promote_fraction)
        realized = math.nan
    else:
        upper_inclusive = upper >= 1.0 - 1e-12
        class_mass = sum(
            weight for value, weight in zip(values, hist)
            if lower <= value and (value <= upper if upper_inclusive else value < upper))
        target = promote_fraction * class_mass
        choices = sorted({lower, upper, (lower + upper) / 2.0, *(
            value for value in values if lower <= value <= upper
        )})
        chosen = min(
            choices,
            key=lambda boundary: (
                abs(sum(
                    weight for value, weight in zip(values, hist)
                    if boundary <= value and (value <= upper if upper_inclusive else value < upper)
                ) - target),
                abs(boundary - (lower + upper) / 2.0),
            ),
        )
        promoted = sum(
            weight for value, weight in zip(values, hist)
            if chosen <= value and (value <= upper if upper_inclusive else value < upper))
        realized = promoted / class_mass if class_mass > 0.0 else math.nan
    if insert_idx == 0:
        out = [float(chosen)] + thresholds
    elif insert_idx == old_class_count:
        out = thresholds + [float(chosen)]
    else:
        out = thresholds[:insert_idx] + [float(chosen)] + thresholds[insert_idx:]
    return out, {
        "insert_idx": int(insert_idx),
        "old_lower": lower,
        "old_upper": upper,
        "new_threshold": float(chosen),
        "requested_promote_fraction": float(promote_fraction),
        "realized_promote_fraction": float(realized),
    }


def insert_precision_rung(
    table: dict,
    profile: dict,
    *,
    insert_idx: int,
    new_level: int,
    promote_fraction: float,
) -> tuple[dict, list[dict]]:
    """Insert a stream-length class and split one old class by MAC mass."""
    old_levels = [int(x) for x in table["stoc_len_levels"]]
    if not 0 <= insert_idx <= len(old_levels):
        raise ValueError("a new rung must be inserted in or beside the ladder")
    if insert_idx == 0:
        valid = new_level > old_levels[0]
    elif insert_idx == len(old_levels):
        valid = new_level < old_levels[-1]
    else:
        valid = old_levels[insert_idx - 1] > new_level > old_levels[insert_idx]
    if not valid:
        raise ValueError("new level is not strictly ordered at its insertion site")
    out = copy.deepcopy(table)
    out["stoc_len_levels"] = (
        old_levels[:insert_idx] + [int(new_level)] + old_levels[insert_idx:])
    values = _hist_values(profile)
    by_bucket, by_op = _profile_hists(profile)
    moves = []
    for section in ("operator_defaults", "buckets"):
        for key, payload in (out.get(section) or {}).items():
            hist = by_op.get(key) if section == "operator_defaults" else by_bucket.get(key)
            thresholds, move = _split_boundary(
                [float(x) for x in payload["thresholds"]],
                hist,
                values,
                insert_idx=insert_idx,
                promote_fraction=promote_fraction,
            )
            payload["thresholds"] = thresholds
            move.update({"section": section, "group": key})
            moves.append(move)
    return out, moves


def remove_precision_rung(
    table: dict,
    *,
    remove_idx: int,
    merge: str,
) -> tuple[dict, list[dict]]:
    """Remove a class, merging its metric interval upward or downward."""
    old_levels = [int(x) for x in table["stoc_len_levels"]]
    if len(old_levels) <= 2:
        raise ValueError("cannot remove a rung from a two-class ladder")
    if not 0 <= remove_idx < len(old_levels):
        raise ValueError("invalid remove_idx")
    if merge not in {"up", "down"}:
        raise ValueError("merge must be 'up' or 'down'")
    if remove_idx == 0:
        threshold_idx = 0
        merge = "down"
    elif remove_idx == len(old_levels) - 1:
        threshold_idx = len(old_levels) - 2
        merge = "up"
    else:
        threshold_idx = remove_idx - 1 if merge == "up" else remove_idx
    out = copy.deepcopy(table)
    out["stoc_len_levels"] = old_levels[:remove_idx] + old_levels[remove_idx + 1:]
    moves = []
    for section in ("operator_defaults", "buckets"):
        for key, payload in (out.get(section) or {}).items():
            old = [float(x) for x in payload["thresholds"]]
            payload["thresholds"] = old[:threshold_idx] + old[threshold_idx + 1:]
            moves.append({
                "section": section,
                "group": key,
                "remove_idx": int(remove_idx),
                "removed_threshold_idx": int(threshold_idx),
                "merge": merge,
            })
    return out, moves


def _validate_table_topology(table: dict, levels: list[int]) -> None:
    if [int(x) for x in table["stoc_len_levels"]] != levels:
        raise ValueError("table/level mismatch")
    if len(levels) < 2 or any(levels[i] <= levels[i + 1]
                              for i in range(len(levels) - 1)):
        raise ValueError(f"invalid level ordering: {levels}")
    for section in ("operator_defaults", "buckets"):
        for key, payload in (table.get(section) or {}).items():
            thresholds = [float(x) for x in payload.get("thresholds") or []]
            if len(thresholds) != len(levels) - 1:
                raise ValueError(
                    f"{section}:{key} has {len(thresholds)} thresholds for "
                    f"{len(levels)} levels")
            if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
                raise ValueError(f"threshold outside [0,1] in {section}:{key}")
            if any(thresholds[i] < thresholds[i + 1] - 1e-9
                   for i in range(len(thresholds) - 1)):
                raise ValueError(f"unordered thresholds in {section}:{key}")


def make_systematic_table(
    source_table: dict,
    levels: Iterable[int],
    *,
    root_parent_table: Path,
    target_cost: float,
    action: dict,
    command: str,
) -> dict:
    """Finalize one V12 candidate and preserve all frozen pass-1 decisions."""
    levels = [int(x) for x in levels]
    out = copy.deepcopy(source_table)
    source_levels = [int(x) for x in source_table["stoc_len_levels"]]
    out["stoc_len_levels"] = levels
    out["mp_levels"] = ",".join(map(str, levels))
    _validate_table_topology(out, levels)
    topology_changed = len(levels) != len(source_levels)
    for section in ("operator_defaults", "buckets"):
        for payload in (out.get(section) or {}).values():
            counts = [int(x) for x in payload.get("counts") or []]
            if counts and (topology_changed or len(counts) != len(levels)):
                payload.setdefault("pass1_counts", payload.pop("counts"))
                payload.pop("avg_stoc_len", None)
            elif len(counts) == len(levels) and sum(counts) > 0:
                payload["avg_stoc_len"] = (
                    sum(c * sl for c, sl in zip(counts, levels)) / sum(counts))
            fractions = [float(x) for x in payload.get("fractions") or []]
            if fractions and (topology_changed or len(fractions) != len(levels)):
                payload.setdefault("pass1_fractions", payload.pop("fractions"))
            if "level_mean_error" in payload:
                payload.setdefault(
                    "pass1_level_mean_error", payload.pop("level_mean_error"))
            if "avg_error" in payload:
                payload.setdefault("pass1_avg_error", payload.pop("avg_error"))
    out.pop("expected_avg_stoc_len", None)
    out["expected_flop_avg_stoc_len"] = float(
        action.get("predicted_cost", target_cost))
    method = str(source_table.get("method", "adaptive_mp"))
    if "_ppl_systematic_refine_v12" not in method:
        out["method"] = method + "_ppl_systematic_refine_v12"
    out["pass2_systematic_refine"] = {
        "schema": "scmp-ppl-systematic-refine-v12",
        "root_parent_table": str(root_parent_table),
        "source_levels": source_levels,
        "target_flop_avg_stoc_len": float(target_cost),
        "class_topology_searchable": True,
        "dispatch_metrics_frozen": True,
        "protected_channels_frozen": True,
        "hybrid_mask_external_and_frozen": True,
        "sc_prec": 8,
        "max_stream_length": 128,
        "action": action,
        "command": command,
    }
    return out


def _insertion_values(
    high: int,
    low: int,
    limit: int,
    *,
    include_high: bool = False,
    include_low: bool = False,
) -> list[int]:
    """Representative precision types in a gap, including 16-aligned ones."""
    candidates = set()
    arithmetic = (high + low) / 2.0
    geometric = math.sqrt(high * low)
    for raw in (arithmetic, geometric):
        for quantum in (4, 8, 16):
            candidates.add(int(round(raw / quantum) * quantum))
    if include_high:
        candidates.add(int(high))
    if include_low:
        candidates.add(int(low))
    candidates.update(value for value in range(8, 129, 8)
                      if low < value < high)
    candidates = sorted(
        (value for value in candidates
         if (low < value < high)
         or (include_high and value == high)
         or (include_low and value == low)),
        key=lambda value: (
            0 if value % 16 == 0 else 1,
            abs(value - arithmetic),
            value,
        ),
    )
    return candidates[:limit]


def _threshold_specs(n_levels: int) -> list[tuple[str, int, float, str]]:
    boundaries = range(n_levels - 1)
    specs = []
    for group in OP_GROUPS:
        for boundary in boundaries:
            for direction in (-1.0, 1.0):
                name = f"{group}_b{boundary}_{'promote' if direction > 0 else 'demote'}"
                specs.append((group, boundary, direction, name))
    return specs


def _compact_moves(moves: list[dict], limit: int = 8) -> list[dict]:
    """Keep candidate provenance useful without making summaries enormous."""
    if len(moves) <= limit:
        return moves
    half = limit // 2
    return (
        moves[:half]
        + [{"omitted": len(moves) - 2 * half}]
        + moves[-half:]
    )


def generate_candidates(
    table: dict,
    profile: dict,
    occupancy: dict,
    *,
    target: float,
    root_parent_table: Path,
    command: str,
    level_step: int,
    threshold_mass_steps: list[float],
    split_fractions: list[float],
    topology_values_per_gap: int,
    min_classes: int,
    max_classes: int,
    fixed_class_count: int | None = None,
    min_level: int,
    max_level: int,
    families: set[str],
    allow_insertion: bool = True,
    allow_removal: bool = True,
) -> list[dict]:
    """Enumerate one coordinate neighborhood around the incumbent."""
    levels = [int(x) for x in table["stoc_len_levels"]]
    protected_weight = float(occupancy["protected_weight"])
    protected_sl = int((table.get("protected_channels") or {}).get("stoc_len", 0))
    proposals = []

    def finish(name: str, family: str, raw_table: dict, seed: list[int],
               projection: dict, detail: dict) -> None:
        new_levels = [int(x) for x in projection["levels"]]
        if fixed_class_count is not None and len(new_levels) != fixed_class_count:
            return
        # Keep the fixed protected stream distinguishable in traces.  Existing
        # parents may already have a colliding rung; do not create a new one.
        if protected_sl and protected_sl not in levels and protected_sl in new_levels:
            return
        if new_levels == levels and raw_table == table:
            return
        action = {
            "type": family,
            "name": name,
            **detail,
            "level_projection": _short_action(projection),
            "predicted_cost": float(projection["predicted_cost"]),
        }
        candidate_table = make_systematic_table(
            raw_table,
            new_levels,
            root_parent_table=root_parent_table,
            target_cost=target,
            action=action,
            command=command,
        )
        proposals.append({
            "name": name,
            "family": family,
            "levels": new_levels,
            "table": candidate_table,
            "action": action,
        })

    if "value" in families:
        for idx, old in enumerate(levels):
            upper = max_level if idx == 0 else levels[idx - 1] - 1
            lower = min_level if idx + 1 == len(levels) else levels[idx + 1] + 1
            for delta in (-level_step, level_step):
                moved = min(max(old + delta, lower), upper)
                if moved == old:
                    continue
                seed = levels.copy()
                seed[idx] = moved
                if protected_sl and protected_sl not in levels and protected_sl in seed:
                    continue
                projection = project_levels_constrained(
                    seed,
                    occupancy["level_weights"],
                    target_cost=target,
                    protected_weight=protected_weight,
                    protected_stoc_len=protected_sl,
                    locked={idx: moved},
                    min_level=min_level,
                    max_level=max_level,
                )
                finish(
                    f"value_i{idx}_{moved}", "value_move", table, seed,
                    projection, {"index": idx, "old_level": old,
                                 "requested_level": moved},
                )

    if "topology" in families and allow_insertion and len(levels) < max_classes:
        insertion_sites: list[tuple[int, list[int]]] = []
        if levels[0] < max_level:
            insertion_sites.append((0, _insertion_values(
                max_level, levels[0], topology_values_per_gap,
                include_high=True)))
        for insert_idx in range(1, len(levels)):
            insertion_sites.append((insert_idx, _insertion_values(
                levels[insert_idx - 1], levels[insert_idx],
                topology_values_per_gap)))
        if levels[-1] > min_level:
            insertion_sites.append((len(levels), _insertion_values(
                levels[-1], min_level, topology_values_per_gap,
                include_low=True)))
        for insert_idx, insertion_values in insertion_sites:
            for new_level in insertion_values:
                if protected_sl and protected_sl not in levels and new_level == protected_sl:
                    continue
                for fraction in split_fractions:
                    raw_table, moves = insert_precision_rung(
                        table, profile,
                        insert_idx=insert_idx,
                        new_level=new_level,
                        promote_fraction=fraction,
                    )
                    seed = [int(x) for x in raw_table["stoc_len_levels"]]
                    predicted_weights = profile_level_weights(
                        raw_table, profile, protected_weight=protected_weight)
                    projection = project_levels_constrained(
                        seed, predicted_weights,
                        target_cost=target,
                        protected_weight=protected_weight,
                        protected_stoc_len=protected_sl,
                        locked={insert_idx: new_level},
                        min_level=min_level,
                        max_level=max_level,
                    )
                    fraction_slug = int(round(fraction * 100))
                    finish(
                        f"insert_i{insert_idx}_l{new_level}_f{fraction_slug}",
                        "topology_insert", raw_table, seed, projection,
                        {"insert_idx": insert_idx, "new_level": new_level,
                         "promote_fraction": fraction,
                         "threshold_moves": _compact_moves(moves)},
                    )

    if "topology" in families and allow_removal and len(levels) > min_classes:
        for remove_idx in range(len(levels)):
            merges = ("down",) if remove_idx == 0 else (
                ("up",) if remove_idx == len(levels) - 1 else ("up", "down"))
            for merge in merges:
                raw_table, moves = remove_precision_rung(
                    table, remove_idx=remove_idx, merge=merge)
                seed = [int(x) for x in raw_table["stoc_len_levels"]]
                predicted_weights = profile_level_weights(
                    raw_table, profile, protected_weight=protected_weight)
                projection = project_levels_to_budget(
                    seed, predicted_weights,
                    target_cost=target,
                    protected_weight=protected_weight,
                    protected_stoc_len=protected_sl,
                    min_level=min_level,
                    max_level=max_level,
                )
                finish(
                    f"remove_i{remove_idx}_{merge}", "topology_remove",
                    raw_table, seed, projection,
                    {"remove_idx": remove_idx, "merge": merge,
                     "threshold_moves": _compact_moves(moves)},
                )

    if "threshold" in families:
        for mass_step in threshold_mass_steps:
            for group, boundary, direction, name in _threshold_specs(len(levels)):
                raw_table, moves = shift_threshold_group(
                    table, profile,
                    ops=OP_GROUPS[group],
                    boundary=boundary,
                    mass_delta=direction * mass_step,
                )
                if not any(move.get("changed") for move in moves):
                    continue
                predicted_weights = profile_level_weights(
                    raw_table, profile, protected_weight=protected_weight)
                projection = project_levels_to_budget(
                    levels, predicted_weights,
                    target_cost=target,
                    protected_weight=protected_weight,
                    protected_stoc_len=protected_sl,
                    min_level=min_level,
                    max_level=max_level,
                )
                mass_slug = int(round(mass_step * 1000))
                finish(
                    f"threshold_{name}_m{mass_slug}", "threshold_move",
                    raw_table, levels, projection,
                    {"group": group, "boundary": boundary,
                     "mass_delta": direction * mass_step,
                     "threshold_moves": _compact_moves(moves)},
                )

    # Different coordinates sometimes collapse to the same deployable table.
    unique = []
    seen = set()
    for proposal in proposals:
        signature = json.dumps({
            "levels": proposal["levels"],
            "operator_defaults": proposal["table"].get("operator_defaults"),
            "buckets": proposal["table"].get("buckets"),
        }, sort_keys=True, separators=(",", ":"))
        if signature not in seen:
            seen.add(signature)
            unique.append(proposal)
    return unique


def _parse_float_list(spec: str) -> list[float]:
    values = []
    for token in spec.replace(":", ",").split(","):
        value = float(token)
        if value not in values:
            values.append(value)
    return values


def _guard_eval(
    model,
    enc,
    *,
    ctx: int,
    branch: Path,
    stem: str,
    result: dict,
) -> dict:
    return _eval_once(
        model, enc,
        ctx=ctx,
        levels=[int(x) for x in result["levels"]],
        table_path=Path(result["table"]),
        wrapper_path=Path(result["wrapper"]),
        trace_path=branch / f"{stem}_guard_trace.json",
        split="validation_guard",
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-wrapper", required=True, type=Path)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--targets", default="24:28:32:36:40:48")
    p.add_argument("--screen-tokens", type=int, default=4096)
    p.add_argument("--guard-tokens", type=int, default=4096)
    p.add_argument("--confirm-tokens", type=int, default=32768)
    p.add_argument(
        "--test-tokens", type=int, default=0,
        help=("final test-evaluation token cap after validation confirmation; "
              "0 uses the complete test split, -1 disables final evaluation"),
    )
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--max-sweeps", type=int, default=2)
    p.add_argument("--patience", type=int, default=2,
                   help="stop after this many consecutive rejected sweeps")
    p.add_argument("--shortlist", type=int, default=3)
    p.add_argument(
        "--shortlist-per-family", type=int, default=0,
        help="guard-test this many candidates per family (0=global shortlist)",
    )
    p.add_argument("--level-step", type=int, default=4)
    p.add_argument("--threshold-mass-steps", default="0.01:0.02")
    p.add_argument("--split-fractions", default="0.33:0.67")
    p.add_argument("--topology-values-per-gap", type=int, default=2)
    p.add_argument("--min-classes", type=int, default=3)
    p.add_argument("--max-classes", type=int, default=8)
    p.add_argument(
        "--fixed-class-count", type=int, default=None,
        help="restrict proposals to this many adaptive precision levels",
    )
    p.add_argument(
        "--allow-insertion", action=argparse.BooleanOptionalAction, default=True,
        help="allow topology search to add a precision rung",
    )
    p.add_argument(
        "--allow-removal", action=argparse.BooleanOptionalAction, default=True,
        help="allow topology search to remove a precision rung",
    )
    p.add_argument("--budget-tol", type=float, default=0.35)
    p.add_argument("--min-mean-nll-improvement", type=float, default=0.0)
    p.add_argument("--max-guard-nll-regression", type=float, default=0.002)
    p.add_argument(
        "--fp16-ppl", type=float, default=None,
        help=("full-precision reference PPL. When supplied, stop searching "
              "once both candidate windows are below "
              "--ppl-fp16-ratio times this value"),
    )
    p.add_argument(
        "--ppl-fp16-ratio", type=float, default=1.1,
        help="relative PPL stopping ratio (default: 1.1)",
    )
    p.add_argument("--budget-corrections", type=int, default=2)
    p.add_argument("--profile-bins", type=int, default=257)
    p.add_argument("--min-level", type=int, default=4)
    p.add_argument("--max-level", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_level > 128:
        raise SystemExit("V12 is fixed to sc_prec=8: maximum stream length is 128")
    if min(args.screen_tokens, args.guard_tokens) < args.ctx:
        raise SystemExit("screen and guard windows must each contain at least one ctx")
    needed = args.screen_tokens + args.guard_tokens
    if args.confirm_tokens < needed:
        raise SystemExit("confirm-tokens must include the screen and guard windows")
    if args.test_tokens < -1:
        raise SystemExit("test-tokens must be -1 or nonnegative")
    if args.min_classes < 2 or args.max_classes < args.min_classes:
        raise SystemExit("invalid class-count bounds")
    if args.fixed_class_count is not None and not (
        args.min_classes <= args.fixed_class_count <= args.max_classes
    ):
        raise SystemExit("fixed-class-count must lie within class-count bounds")
    if args.patience < 1:
        raise SystemExit("patience must be >= 1")
    if args.shortlist < 1 or args.shortlist_per_family < 0:
        raise SystemExit("shortlist must be >=1 and shortlist-per-family >=0")
    if args.fp16_ppl is not None and args.fp16_ppl <= 0:
        raise SystemExit("fp16-ppl must be > 0")
    if args.ppl_fp16_ratio <= 0:
        raise SystemExit("ppl-fp16-ratio must be > 0")
    threshold_steps = _parse_float_list(args.threshold_mass_steps)
    split_fractions = _parse_float_list(args.split_fractions)
    if any(not 0.0 < value < 0.25 for value in threshold_steps):
        raise SystemExit("threshold mass steps must be in (0, 0.25)")
    if any(not 0.0 < value < 1.0 for value in split_fractions):
        raise SystemExit("split fractions must be in (0, 1)")

    root_wrapper = args.parent_wrapper.resolve()
    wrapper, root_table_path, root_table = resolve_parent_table(root_wrapper)
    parent_levels = [int(x) for x in wrapper["stoc_len_levels"]]
    if max(parent_levels) > 128:
        raise SystemExit("parent exceeds the sc_prec=8 stream-length cap")
    protected_sl, protected_hint = _protected_info(root_table)
    command = " ".join(sys.argv)
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _build_parent_model(
        args.model_path, root_wrapper, args.alpha)
    enc_full = _load_eval_stream(
        tokenizer, split="validation", max_tokens=args.confirm_tokens,
        ctx=args.ctx)
    screen_n = args.screen_tokens // args.ctx * args.ctx
    guard_n = args.guard_tokens // args.ctx * args.ctx
    enc_screen = enc_full[:screen_n]
    enc_guard = enc_full[screen_n:screen_n + guard_n]
    enc_test = None
    if args.test_tokens >= 0:
        enc_test = _load_eval_stream(
            tokenizer, split="test", max_tokens=args.test_tokens,
            ctx=args.ctx)
        print(
            f"[V12 TEST STREAM] tokens={enc_test.numel()} "
            f"mode={'full' if args.test_tokens == 0 else 'capped'}",
            flush=True,
        )

    baseline_dir = outdir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_result, _, baseline_profile = _evaluate_candidate(
        model, enc_screen,
        ctx=args.ctx,
        branch=baseline_dir,
        stem="parent_reference",
        table=root_table,
        levels=parent_levels,
        profile_bins=args.profile_bins,
    )
    baseline_occupancy = _actual_occupancy(
        baseline_result, parent_levels,
        protected_sl=protected_sl,
        protected_hint=protected_hint)
    baseline_guard = _guard_eval(
        model, enc_guard,
        ctx=args.ctx,
        branch=baseline_dir,
        stem="parent_reference",
        result=baseline_result)
    baseline_guard_occupancy = _actual_occupancy(
        baseline_guard, parent_levels,
        protected_sl=protected_sl,
        protected_hint=protected_hint)
    parent_cost = float(baseline_occupancy["cost"])
    nominal_cost = nominal_target_cost(root_table)
    usefulness_threshold = (
        None if args.fp16_ppl is None
        else float(args.fp16_ppl) * float(args.ppl_fp16_ratio)
    )
    targets = _parse_targets(
        args.targets, parent_cost=parent_cost, nominal_cost=nominal_cost)
    print(
        f"[V12 BASELINE] levels={parent_levels} "
        f"screen_ppl={baseline_result['ppl']:.6f} "
        f"guard_ppl={baseline_guard['ppl']:.6f} "
        f"cost={parent_cost:.4f} targets={targets}", flush=True)

    summaries = []
    for target in targets:
        branch = outdir / _target_slug(target)
        branch.mkdir(parents=True, exist_ok=True)
        summary_path = branch / "summary.json"
        if args.resume and summary_path.exists():
            with summary_path.open() as f:
                completed = json.load(f)
            if completed.get("status") == "ok":
                # Older waves predate automatic final-test evaluation.  Reuse
                # their frozen best wrapper/table and backfill the test result
                # without repeating the search.
                if enc_test is not None and "final_test_ppl" not in completed:
                    best_levels = [int(x) for x in completed["best_levels"]]
                    best_wrapper = Path(completed["best_wrapper"])
                    best_table = Path(completed["best_table"])
                    final_test = _eval_once(
                        model, enc_test,
                        ctx=args.ctx,
                        levels=best_levels,
                        table_path=best_table,
                        wrapper_path=best_wrapper,
                        trace_path=branch / "final_test_trace.json",
                        split="test",
                    )
                    final_test_occupancy = _actual_occupancy(
                        final_test, best_levels,
                        protected_sl=protected_sl,
                        protected_hint=protected_hint)
                    completed.update({
                        "final_test_split": "test",
                        "final_test_ppl": final_test["ppl"],
                        "final_test_nll": final_test["nll"],
                        "final_test_tokens": final_test["tokens"],
                        "final_test_cost": final_test_occupancy["cost"],
                        "final_test_trace": final_test["trace"],
                    })
                    _atomic_json(summary_path, completed)
                summaries.append(completed)
                print(f"[V12 RESUME] target={target} already complete", flush=True)
                continue

        history = [{
            "stage": "parent_reference",
            "target": target,
            "screen": baseline_result,
            "guard": baseline_guard,
            "cost": parent_cost,
        }]
        current_result = copy.deepcopy(baseline_result)
        current_guard = copy.deepcopy(baseline_guard)
        current_profile = baseline_profile
        current_occupancy = baseline_occupancy
        current_table = copy.deepcopy(root_table)
        current_table_path = root_table_path
        current_levels = parent_levels.copy()

        # Establish an in-budget incumbent before optimizing PPL.
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
                break
            if protected_sl and protected_sl not in parent_levels and protected_sl in levels:
                break
            action = {
                "type": "initial_budget_projection",
                "predicted_cost": projection["predicted_cost"],
                "projection": _short_action(projection),
            }
            table = make_systematic_table(
                current_table, levels,
                root_parent_table=root_table_path,
                target_cost=target,
                action=action,
                command=command,
            )
            result, table_path, profile = _evaluate_candidate(
                model, enc_screen,
                ctx=args.ctx,
                branch=branch,
                stem=f"budget{correction:02d}_" + "-".join(map(str, levels)),
                table=table,
                levels=levels,
                profile_bins=args.profile_bins,
            )
            occupancy = _actual_occupancy(
                result, levels,
                protected_sl=protected_sl,
                protected_hint=protected_hint)
            improved_budget = (
                abs(occupancy["cost"] - target)
                < abs(current_occupancy["cost"] - target))
            history.append({
                "stage": "initial_budget_projection",
                "accepted": improved_budget,
                "budget_error": occupancy["cost"] - target,
                "action": action,
                "screen": result,
            })
            if not improved_budget:
                break
            current_result = result
            current_profile = profile
            current_occupancy = occupancy
            current_table = table
            current_table_path = table_path
            current_levels = levels

        if abs(current_occupancy["cost"] - target) > args.budget_tol:
            summary = {
                "schema": "scmp-ppl-systematic-refine-v12",
                "status": "budget_projection_failed",
                "target": target,
                "history": history,
            }
            _atomic_json(summary_path, summary)
            summaries.append(summary)
            continue

        if current_levels != parent_levels:
            current_guard = _guard_eval(
                model, enc_guard,
                ctx=args.ctx,
                branch=branch,
                stem="budget_incumbent",
                result=current_result)
        current_guard_occupancy = _actual_occupancy(
            current_guard, current_levels,
            protected_sl=protected_sl,
            protected_hint=protected_hint)
        current_mean_nll = (current_result["nll"] + current_guard["nll"]) / 2.0

        stalled_sweeps = 0
        usefulness_reached = False
        stop_reason = None
        for sweep in range(1, args.max_sweeps + 1):
            accepted_in_sweep = False
            for phase, families in (
                ("precision", {"value", "topology"}),
                ("threshold", {"threshold"}),
            ):
                proposals = generate_candidates(
                    current_table, current_profile, current_occupancy,
                    target=target,
                    root_parent_table=root_table_path,
                    command=command,
                    level_step=args.level_step,
                    threshold_mass_steps=threshold_steps,
                    split_fractions=split_fractions,
                    topology_values_per_gap=args.topology_values_per_gap,
                    min_classes=args.min_classes,
                    max_classes=args.max_classes,
                    fixed_class_count=args.fixed_class_count,
                    min_level=args.min_level,
                    max_level=args.max_level,
                    families=families,
                    allow_insertion=args.allow_insertion,
                    allow_removal=args.allow_removal,
                )
                print(
                    f"[V12 SWEEP] target={target:.2f} sweep={sweep} "
                    f"phase={phase} candidates={len(proposals)}", flush=True)
                evaluated = []
                for candidate_idx, proposal in enumerate(proposals):
                    stem = (
                        f"s{sweep:02d}_{phase}_{candidate_idx:03d}_"
                        f"{proposal['name']}")
                    result, table_path, profile = _evaluate_candidate(
                        model, enc_screen,
                        ctx=args.ctx,
                        branch=branch,
                        stem=stem,
                        table=proposal["table"],
                        levels=proposal["levels"],
                        profile_bins=args.profile_bins,
                    )
                    occupancy = _actual_occupancy(
                        result, proposal["levels"],
                        protected_sl=protected_sl,
                        protected_hint=protected_hint)
                    eligible = abs(occupancy["cost"] - target) <= args.budget_tol
                    record = {
                        "stage": phase,
                        "sweep": sweep,
                        "candidate": proposal["name"],
                        "family": proposal["family"],
                        "accepted": False,
                        "eligible": eligible,
                        "budget_error": occupancy["cost"] - target,
                        "action": proposal["action"],
                        "screen": result,
                    }
                    history.append(record)
                    if eligible:
                        evaluated.append({
                            "proposal": proposal,
                            "record": record,
                            "result": result,
                            "table_path": table_path,
                            "profile": profile,
                            "occupancy": occupancy,
                            "stem": stem,
                        })
                if args.shortlist_per_family > 0:
                    shortlisted = []
                    for family in sorted({
                        item["proposal"]["family"] for item in evaluated
                    }):
                        family_items = [
                            item for item in evaluated
                            if item["proposal"]["family"] == family
                        ]
                        shortlisted.extend(sorted(
                            family_items,
                            key=lambda item: item["result"]["nll"],
                        )[:args.shortlist_per_family])
                    shortlist = sorted(
                        shortlisted, key=lambda item: item["result"]["nll"]
                    )
                else:
                    shortlist = sorted(
                        evaluated, key=lambda item: item["result"]["nll"]
                    )[:args.shortlist]
                guarded = []
                for candidate in shortlist:
                    guard = _guard_eval(
                        model, enc_guard,
                        ctx=args.ctx,
                        branch=branch,
                        stem=candidate["stem"],
                        result=candidate["result"])
                    guard_occupancy = _actual_occupancy(
                        guard, candidate["proposal"]["levels"],
                        protected_sl=protected_sl,
                        protected_hint=protected_hint)
                    mean_cost = (candidate["occupancy"]["cost"]
                                 + guard_occupancy["cost"]) / 2.0
                    mean_nll = (candidate["result"]["nll"] + guard["nll"]) / 2.0
                    improvement = current_mean_nll - mean_nll
                    eligible = (
                        improvement > args.min_mean_nll_improvement
                        and guard["nll"]
                        <= current_guard["nll"] + args.max_guard_nll_regression
                        and abs(mean_cost - target) <= args.budget_tol)
                    candidate.update({
                        "guard": guard,
                        "guard_occupancy": guard_occupancy,
                        "mean_cost": mean_cost,
                        "mean_nll": mean_nll,
                        "mean_nll_improvement": improvement,
                        "guard_eligible": eligible,
                    })
                    # This is deliberately checked on both independent
                    # windows.  A short screen-only win can be sampling noise;
                    # requiring the guard window makes the stopping rule a
                    # conservative early-termination signal.  The final
                    # confirmation below remains mandatory.
                    near_fp16 = (
                        usefulness_threshold is not None
                        and candidate["result"]["ppl"] < usefulness_threshold
                        and guard["ppl"] < usefulness_threshold
                    )
                    candidate["near_fp16"] = near_fp16
                    candidate["record"].update({
                        "guard": guard,
                        "guard_cost": guard_occupancy["cost"],
                        "mean_cost": mean_cost,
                        "mean_nll": mean_nll,
                        "mean_nll_improvement": improvement,
                        "guard_eligible": eligible,
                        "near_fp16": near_fp16,
                        "fp16_ppl": args.fp16_ppl,
                        "ppl_fp16_ratio": args.ppl_fp16_ratio,
                        "usefulness_threshold": usefulness_threshold,
                    })
                    if eligible:
                        guarded.append(candidate)
                if not guarded:
                    continue
                winner = min(guarded, key=lambda item: item["mean_nll"])
                winner["record"]["accepted"] = True
                winner["record"]["reason"] = "best_two_window_mean_validation_nll"
                current_result = winner["result"]
                current_guard = winner["guard"]
                current_guard_occupancy = winner["guard_occupancy"]
                current_mean_nll = winner["mean_nll"]
                current_profile = winner["profile"]
                current_occupancy = winner["occupancy"]
                current_table = winner["proposal"]["table"]
                current_table_path = winner["table_path"]
                current_levels = winner["proposal"]["levels"]
                accepted_in_sweep = True
                print(
                    f"[V12 ACCEPT] target={target:.2f} sweep={sweep} "
                    f"phase={phase} candidate={winner['proposal']['name']} "
                    f"levels={current_levels} mean_nll={current_mean_nll:.8f}",
                    flush=True)
                if winner.get("near_fp16"):
                    usefulness_reached = True
                    stop_reason = "ppl_within_fp16_ratio"
                    print(
                        f"[V12 USEFULNESS] target={target:.2f} "
                        f"candidate={winner['proposal']['name']} "
                        f"screen_ppl={winner['result']['ppl']:.6f} "
                        f"guard_ppl={winner['guard']['ppl']:.6f} "
                        f"threshold={usefulness_threshold:.6f}; "
                        "stopping search before another sweep",
                        flush=True)
                    break
            if usefulness_reached:
                break
            if accepted_in_sweep:
                stalled_sweeps = 0
            else:
                stalled_sweeps += 1
                if stalled_sweeps >= args.patience:
                    print(
                        f"[V12 CONVERGED] target={target:.2f} "
                        f"consecutive_rejected_sweeps={stalled_sweeps}",
                        flush=True)
                    break

        best_wrapper = branch / "best_wrapper.json"
        _write_wrapper(best_wrapper, current_levels, current_table_path)
        confirm = _eval_once(
            model, enc_full,
            ctx=args.ctx,
            levels=current_levels,
            table_path=current_table_path,
            wrapper_path=best_wrapper,
            trace_path=branch / "confirm_validation_trace.json",
            split="validation",
        )
        confirm_occupancy = _actual_occupancy(
            confirm, current_levels,
            protected_sl=protected_sl,
            protected_hint=protected_hint)
        confirm_near_fp16 = (
            usefulness_threshold is not None
            and confirm["ppl"] < usefulness_threshold)
        final_test = None
        final_test_occupancy = None
        if enc_test is not None:
            final_test = _eval_once(
                model, enc_test,
                ctx=args.ctx,
                levels=current_levels,
                table_path=current_table_path,
                wrapper_path=best_wrapper,
                trace_path=branch / "final_test_trace.json",
                split="test",
            )
            final_test_occupancy = _actual_occupancy(
                final_test, current_levels,
                protected_sl=protected_sl,
                protected_hint=protected_hint)
        summary = {
            "schema": "scmp-ppl-systematic-refine-v12",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": ("ok" if abs(confirm_occupancy["cost"] - target)
                       <= args.budget_tol else "confirm_budget_miss"),
            "target": target,
            "model": args.model_path,
            "root_parent_wrapper": str(root_wrapper),
            "root_parent_table": str(root_table_path),
            "parent_levels": parent_levels,
            "parent_screen_ppl": baseline_result["ppl"],
            "parent_guard_ppl": baseline_guard["ppl"],
            "parent_cost": parent_cost,
            "best_levels": current_levels,
            "best_class_count": len(current_levels),
            "best_screen_ppl": current_result["ppl"],
            "best_guard_ppl": current_guard["ppl"],
            "best_two_window_mean_nll": current_mean_nll,
            "best_search_cost": current_occupancy["cost"],
            "best_guard_cost": current_guard_occupancy["cost"],
            "fp16_ppl": args.fp16_ppl,
            "ppl_fp16_ratio": args.ppl_fp16_ratio,
            "usefulness_threshold": usefulness_threshold,
            "usefulness_reached_during_search": usefulness_reached,
            "confirm_within_fp16_ratio": confirm_near_fp16,
            "stop_reason": stop_reason,
            "confirm_validation_ppl": confirm["ppl"],
            "confirm_validation_nll": confirm["nll"],
            "confirm_validation_cost": confirm_occupancy["cost"],
            "final_test_split": ("test" if final_test is not None else None),
            "final_test_ppl": (None if final_test is None
                                else final_test["ppl"]),
            "final_test_nll": (None if final_test is None
                                else final_test["nll"]),
            "final_test_tokens": (None if final_test is None
                                   else final_test["tokens"]),
            "final_test_cost": (None if final_test_occupancy is None
                                 else final_test_occupancy["cost"]),
            "final_test_trace": (None if final_test is None
                                  else final_test["trace"]),
            "best_wrapper": str(best_wrapper),
            "best_table": str(current_table_path),
            "screen_tokens": int(enc_screen.numel()),
            "guard_tokens": int(enc_guard.numel()),
            "confirm_tokens": int(enc_full.numel()),
            "history": history,
            "command": command,
        }
        _atomic_json(summary_path, summary)
        summaries.append(summary)
        _atomic_json(outdir / "summary.json", {
            "schema": "scmp-ppl-systematic-refine-v12-wave",
            "model": args.model_path,
            "targets": targets,
            "branches": summaries,
            "command": command,
        })
        print(
            f"[V12 TARGET RESULT] target={target:.2f} levels={current_levels} "
            f"confirm_ppl={confirm['ppl']:.6f} "
            f"confirm_cost={confirm_occupancy['cost']:.4f} "
            f"final_test_ppl={(final_test['ppl'] if final_test else math.nan):.6f}",
            flush=True)

    _atomic_json(outdir / "summary.json", {
        "schema": "scmp-ppl-systematic-refine-v12-wave",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model_path,
        "targets": targets,
        "branches": summaries,
        "command": command,
    })
    with (outdir / "results.tsv").open("w") as f:
        f.write("target\tstatus\tlevels\tclass_count\tscreen_ppl\tguard_ppl\t"
                "confirm_ppl\tconfirm_cost\tfinal_test_ppl\tfinal_test_cost\t"
                "best_wrapper\n")
        for summary in summaries:
            f.write(
                f"{summary['target']:.6f}\t{summary['status']}\t"
                f"{','.join(map(str, summary.get('best_levels', [])))}\t"
                f"{summary.get('best_class_count', 0)}\t"
                f"{summary.get('best_screen_ppl', math.nan):.8f}\t"
                f"{summary.get('best_guard_ppl', math.nan):.8f}\t"
                f"{summary.get('confirm_validation_ppl', math.nan):.8f}\t"
                f"{summary.get('confirm_validation_cost', math.nan):.6f}\t"
                f"{summary.get('final_test_ppl', math.nan):.8f}\t"
                f"{summary.get('final_test_cost', math.nan):.6f}\t"
                f"{summary.get('best_wrapper', '')}\n")
    print(f"[V12 DONE] summary={outdir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
