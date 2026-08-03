"""V16 systematic refinement with paired-window statistics and macro-moves.

V16 replaces the V12/V14 screen/guard/confirm construction, whose failure
modes were established forensically on the V14 wave:

* the FP16 usefulness gate referenced full-test FP16 against much-easier
  validation windows and truncated 9/10 branches to a single greedy move;
* screening reused one fixed easy validation prefix (val[:8k]/test PPL
  ratio 0.75) and shortlists selected on it carried a measured
  +0.0069-nat winner's-curse bias;
* the 32k confirmation stream contained the search windows;
* acceptance thresholds (0.002 nats) sat at ~0.3 sigma of the measured
  single-window candidate noise (sigma_8k ~ 0.0076 nats);
* a walltime kill lost an entire branch (no checkpointing).

Design here:

* The whole validation split is partitioned at 4-window-block granularity
  into a 32-window screen pool (4 rotating subsets of 8, each split into a
  4-window select half A and a 4-window replicate half B), a 16-window
  confirm set, and a held-out remainder evaluated once per branch.
* Every comparison is a PAIRED per-window mean NLL delta against the
  incumbent on identical windows; sigma is pooled across the sweep's
  candidates, never estimated from one candidate's four windows.
* Stage funnel per phase (V17 acceptance fix, user-approved 2026-07-19):
  rank all candidates on half A (family quotas), replicate the shortlist on
  half B.  ACCEPTANCE happens at the A->B stage: the stage-A winner is
  accepted iff its sign replicates on the disjoint half B and the pooled
  A+B mean clears the pooled-sigma significance threshold.  The 16-window
  confirm set is STOP-ONLY: it refreshes the incumbent cache after an
  acceptance and may terminate the branch (confirm_stop_check has no
  "accept" key and runs only after an acceptance is final), but it can
  never select between candidates or flip an accept/reject.  This mirrors
  the two-tier fp16 design (cheap trigger, authoritative stop) and fixes
  the V16 bottleneck where the confirm gate rejected every 14B candidate
  while a hand-built compound won -1.0% on the full test split.
* The move set adds compound macro-moves (floor-exchange, ceiling
  compress/raise, joint application of the previous sweep's two best value
  moves) — the only move family with a documented >2% validation win.
* Stopping: a branch ends on patience, sweep cap, or the walltime guard.
  The optional FP16 stop is two-tier: a cheap check against FP16 measured
  on the SAME confirm windows (fp16_val_reference.py) merely triggers the
  authoritative check — a full test-protocol evaluation that stops the
  branch only if final test PPL is within --ppl-fp16-ratio of full-test
  FP16.  The test split never selects candidates.
* Budget: realized cost must sit in [target - budget-tol-under,
  target + budget-tol-over] (asymmetric by user decision); candidates are
  ranked only by paired delta inside the band and every report quotes the
  realized trace cost.
* A checkpoint is written after every sweep; resume verifies incumbent
  reproducibility on one window before continuing (the evaluator is
  deterministic under SC_OWEN_MODE=bitrev).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from statistics import NormalDist


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from benchmark.ppl.mp_joint_refine import (
    OP_GROUPS,
    _actual_occupancy,
    _hist_values,
    _parse_targets,
    _shift_one_boundary,
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
    _load_trace,
    _protected_info,
    _write_wrapper,
    nominal_target_cost,
    propose_floor_exchange,
    resolve_parent_table,
    weighted_cost,
)
from benchmark.ppl.mp_group_ladders import (
    GROUP_MODES,
    aggregate_trace_weights_grouped,
    apply_group_ladders_to_table,
    assert_no_pin_collision,
    group_keys_for_mode,
    project_group_ladders_to_budget,
)
from benchmark.ppl.mp_systematic_refine import (
    _insertion_values,
    insert_precision_rung,
    make_systematic_table,
    project_levels_constrained,
    remove_precision_rung,
)


V16_SCHEMA = "scmp-ppl-v16-refine-v1"
DEFAULT_SIGMA_W = 0.015  # per-2k-window paired-delta noise, measured on V14

# lift_compound (V17): the measured-direction compound that beat the search
# on 14B (oracle wave 2026-07-18, cell attn_lift96_pc95_reinvest).  95 is
# deliberate — 96 is the V9 top rung and would collide with the adaptive
# ladder (aggregate_trace_weights cannot separate protected work that lands
# on an adaptive level).  NEVER change this to 96.
LIFT_PSL = 95
ATTN_LIFT_BUCKETS = tuple(
    f"{op}:t0:l{i}" for op in ("qk", "av") for i in range(4))


# --------------------------------------------------------------------------
# window partition
# --------------------------------------------------------------------------

def _spread_indices(count: int, total: int, taken: set[int]) -> list[int]:
    """Evenly spread ``count`` block indices over ``total``, avoiding taken."""
    out = []
    for i in range(count):
        idx = int((i + 0.5) * total / count)
        while idx in taken or idx in out:
            idx = (idx + 1) % total
        out.append(idx)
    return sorted(out)


def build_window_partition(
    n_windows: int,
    *,
    block: int = 4,
    screen_blocks: int = 8,
    confirm_blocks: int = 4,
    n_subsets: int = 4,
    excluded_block_ids: set[int] | None = None,
) -> dict:
    """Assign whole blocks of adjacent windows to screen/confirm/holdout.

    Blocks (not single windows) are the assignment unit so adjacent windows
    of one article rarely straddle two sets.  Screen and confirm blocks are
    spread across the whole split, killing the easy-prefix bias of the old
    val[:32k] construction.  Screen subset j pairs one early and one late
    block; its first block is the select half A, the second the replicate
    half B.
    """
    full_blocks = n_windows // block
    excluded = {
        int(i) for i in (excluded_block_ids or set())
        if 0 <= int(i) < full_blocks
    }
    eligible = [i for i in range(full_blocks) if i not in excluded]
    if len(eligible) < screen_blocks + confirm_blocks:
        raise ValueError(
            f"{n_windows} windows / block {block} gives {len(eligible)} "
            f"eligible blocks after excluding {len(excluded)}; need >= "
            f"{screen_blocks + confirm_blocks}")
    if screen_blocks % n_subsets != 0 or screen_blocks // n_subsets != 2:
        raise ValueError("screen pool must hold exactly 2 blocks per subset")
    if excluded:
        def spread_allowed(count: int, values: list[int], taken: set[int]):
            available = [v for v in values if v not in taken]
            out = []
            for i in range(count):
                pos = min(int((i + 0.5) * len(available) / count),
                          len(available) - 1)
                value = available[pos]
                while value in out:
                    pos = (pos + 1) % len(available)
                    value = available[pos]
                out.append(value)
            return sorted(out)

        screen_ids = spread_allowed(screen_blocks, eligible, set())
        confirm_ids = spread_allowed(
            confirm_blocks, eligible, set(screen_ids))
    else:
        # Preserve the historical partition byte-for-byte when no exclusion
        # manifest is supplied.
        screen_ids = _spread_indices(screen_blocks, full_blocks, set())
        confirm_ids = _spread_indices(
            confirm_blocks, full_blocks, set(screen_ids))
    used = set(screen_ids) | set(confirm_ids)

    def windows(block_id: int) -> list[int]:
        return list(range(block_id * block, block_id * block + block))

    subsets = []
    for j in range(n_subsets):
        subsets.append({
            "A": windows(screen_ids[j]),
            "B": windows(screen_ids[j + n_subsets]),
        })
    confirm = [w for b in confirm_ids for w in windows(b)]
    holdout = [w for w in range(n_windows)
               if ((w // block) not in used
                   and (w // block) not in excluded)]
    result = {
        "schema": ("scmp-v16-window-partition-v2"
                   if excluded else "scmp-v16-window-partition-v1"),
        "n_windows": int(n_windows),
        "block": int(block),
        "screen_block_ids": screen_ids,
        "confirm_block_ids": confirm_ids,
        "subsets": subsets,
        "confirm_windows": confirm,
        "holdout_windows": holdout,
    }
    if excluded:
        result["excluded_block_ids"] = sorted(excluded)
    return result


# --------------------------------------------------------------------------
# paired statistics
# --------------------------------------------------------------------------

def paired_deltas(
    cand_nll: dict[int, float],
    incumbent_nll: dict[int, float],
    window_ids: list[int],
) -> list[float]:
    return [float(cand_nll[w]) - float(incumbent_nll[w]) for w in window_ids]


def pooled_sigma_w(
    delta_vectors: list[list[float]],
    *,
    fallback: float = DEFAULT_SIGMA_W,
    min_candidates: int = 5,
) -> float:
    """Across-window sd of paired deltas, pooled over the sweep's candidates."""
    vecs = [v for v in delta_vectors if len(v) >= 2]
    if len(vecs) < min_candidates:
        return float(fallback)
    num, dof = 0.0, 0
    for v in vecs:
        m = sum(v) / len(v)
        num += sum((x - m) ** 2 for x in v)
        dof += len(v) - 1
    if dof <= 0 or num <= 0.0:
        return float(fallback)
    return math.sqrt(num / dof)


def replication_accept_gate(
    deltas_a: list[float],
    deltas_b: list[float],
    *,
    sigma_w: float,
    accept_floor: float,
    se_mult: float,
) -> dict:
    """Sole acceptance authority (V17): A->B replication of the stage-A winner.

    The stage-A winner is accepted iff (1) its SIGN replicates on the
    disjoint half B (mean_b < 0) and (2) the pooled A+B mean clears the
    pooled-sigma significance threshold.  The confirm set never
    participates in this decision — see confirm_stop_check.
    """
    n = len(deltas_a) + len(deltas_b)
    mean_b = sum(deltas_b) / len(deltas_b)
    mean_ab = (sum(deltas_a) + sum(deltas_b)) / n
    threshold = -max(float(accept_floor),
                     float(se_mult) * float(sigma_w) / math.sqrt(n))
    return {
        "n": n,
        "mean_b": float(mean_b),
        "mean_ab": float(mean_ab),
        "threshold": float(threshold),
        "sign_replicates": bool(mean_b < 0.0),
        "accept": bool(mean_b < 0.0 and mean_ab < threshold),
    }


def confirm_stop_check(
    deltas: list[float],
    *,
    sigma_w: float,
    se_mult: float,
    regress_window_nats: float,
    regress_window_max: int,
) -> dict:
    """STOP-ONLY confirm-set check — provably unable to accept anything.

    Acceptance is decided at the A->B stage (replication_accept_gate)
    BEFORE the confirm set is ever evaluated; this check runs only AFTER
    an acceptance is final, and its sole authority is terminating the
    sweep loop when the freshly accepted incumbent significantly regresses
    on the confirm windows.  The returned dict deliberately carries no
    "accept" key, so no caller can accept or reject a candidate from it.
    """
    n = len(deltas)
    mean = sum(deltas) / n
    threshold = max(float(se_mult) * float(sigma_w) / math.sqrt(n), 0.0)
    regressions = sum(1 for d in deltas if d > float(regress_window_nats))
    return {
        "n": n,
        "mean": float(mean),
        "threshold": float(threshold),
        "regressions": int(regressions),
        "stop": bool(mean > threshold
                     and regressions > int(regress_window_max)),
    }


def confirm_veto_gate(
    deltas_a: list[float],
    deltas_b: list[float],
    deltas_c: list[float],
    *,
    sigma_w: float,
    accept_floor: float,
    se_mult: float,
) -> dict:
    """Pooled A+B+confirm re-test of an A->B acceptance (V18, user-approved).

    The A->B gate decides on 8 windows (reliability rho ~= 0.64) and the
    16-window confirm set used to be STOP-ONLY, so a move that lost on the
    confirm windows was adopted anyway.  Measured on 14B t48: an accepted
    topology insert regressed the confirm PPL 9.9297 -> 9.9546 (+0.25%) and
    stayed.  All three window sets hold per-window deltas against the SAME
    incumbent, so they pool directly; the move is kept only if it still
    clears significance over all 24 windows, else the caller reverts.

    This strictly dominates the old rule: it uses every window the old rule
    used, plus 16 the old rule discarded for the accept decision.
    """
    pooled = list(deltas_a) + list(deltas_b) + list(deltas_c)
    n = len(pooled)
    mean = sum(pooled) / n
    threshold = -max(float(accept_floor),
                     float(se_mult) * float(sigma_w) / math.sqrt(n))
    return {
        "n": n,
        "mean_abc": float(mean),
        "threshold": float(threshold),
        "keep": bool(mean < threshold),
    }


def select_advance(
    ranked: list[dict],
    *,
    quotas: dict[str, int],
    total: int,
) -> list[dict]:
    """Family-quota shortlist that can never drop the global A-best.

    Quotas diversify the expensive B stage; they do not get to replace the
    best measured candidate.  The old quota-first implementation could put a
    worse pc_length candidate at index zero when ``group_value`` was absent
    from the quota map, and the caller then treated that item as the sole
    acceptance candidate.
    """
    if total <= 0 or not ranked:
        return []
    chosen: list[dict] = []
    names = set()
    counts: dict[str, int] = {}
    best = ranked[0]
    chosen.append(best)
    names.add(best["name"])
    counts[best["family"]] = 1
    for item in ranked:
        if len(chosen) >= total:
            break
        if item["name"] in names:
            continue
        fam = item["family"]
        if counts.get(fam, 0) < int(quotas.get(fam, 0)):
            chosen.append(item)
            names.add(item["name"])
            counts[fam] = counts.get(fam, 0) + 1
    for item in ranked:
        if len(chosen) >= total:
            break
        if item["name"] not in names:
            chosen.append(item)
            names.add(item["name"])
    return chosen[:total]


# --------------------------------------------------------------------------
# candidate tables
# --------------------------------------------------------------------------

def make_v16_table(
    source_table: dict,
    levels: list[int],
    *,
    root_parent_table: Path,
    target_cost: float,
    action: dict,
    command: str,
) -> dict:
    out = make_systematic_table(
        source_table, levels,
        root_parent_table=root_parent_table,
        target_cost=target_cost,
        action=action,
        command=command,
    )
    meta = out.pop("pass2_systematic_refine")
    meta["schema"] = V16_SCHEMA
    out["pass2_v16_refine"] = meta
    method = str(out.get("method", "adaptive_mp"))
    if method.endswith("_ppl_systematic_refine_v12"):
        method = method[: -len("_ppl_systematic_refine_v12")]
    if not method.endswith("_ppl_v16"):
        method += "_ppl_v16"
    out["method"] = method
    return out


def _ladder_state(value):
    """Normalize an incumbent ladder: {group: ladder} dict, or a flat list.

    The incumbent's ladder is a plain list in global mode and a dict in group
    mode.  Every consumer goes through here so a dict can never be silently
    iterated as if it were a list of rungs (which yields group NAMES and blows
    up in int()).
    """
    if isinstance(value, dict):
        return {k: [int(x) for x in v] for k, v in value.items()}
    return [int(x) for x in value]


def _runtime_levels(levels, table_path=None):
    """Flat ladder for the RUNTIME.

    _install_runtime_table builds an AdaptiveMPConfig, which (a) indexes
    stoc_len_levels and (b) REJECTS any value that differs from the deployed
    table's own top-level stoc_len_levels.  So in group mode this must be the
    TABLE's list -- not the union of the group ladders, which fails that
    equality check.  The authoritative per-bucket ladders travel inside the
    table and are read by AdaptiveMPConfig.get_levels(); the top-level list is
    only the fallback for a bucket that declares none.
    """
    if not isinstance(levels, dict):
        return [int(x) for x in levels]
    if table_path is not None:
        try:
            with Path(table_path).open() as f:
                return [int(x) for x in json.load(f)["stoc_len_levels"]]
        except (OSError, ValueError, KeyError):
            pass
    return sorted({int(x) for lad in levels.values() for x in lad},
                  reverse=True)


def group_state_from_levels(levels: list[int], mode: str) -> dict[str, list[int]]:
    """Seed every ladder group from the (single) parent ladder.

    Starting every group identical means the initial per-group state IS the
    parent allocation, so the first budget projection is a no-op and any
    divergence between groups afterwards was measured and accepted.
    """
    return {key: [int(x) for x in levels]
            for key in group_keys_for_mode(mode)}


def _valid_group_rung_change(
    ladder: list[int],
    index: int,
    delta: int,
    *,
    protected_sl: int,
    min_gap: int,
    min_level: int,
    max_level: int,
) -> list[int] | None:
    candidate = [int(x) for x in ladder]
    candidate[index] += int(delta)
    value = candidate[index]
    if not min_level <= value <= max_level:
        return None
    if protected_sl and value == int(protected_sl):
        return None
    if any(candidate[i] - candidate[i + 1] < min_gap
           for i in range(len(candidate) - 1)):
        return None
    return candidate


def _group_compensators(
    group_ladders: dict[str, list[int]],
    group_weights: dict[str, list[float]],
    *,
    required_cost_delta: float,
    excluded_group: str | None,
    protected_sl: int,
    min_gap: int,
    min_level: int,
    max_level: int,
    min_weight: float,
    max_delta: int,
    max_error: float,
    keep: int = 1,
) -> list[dict]:
    """Best explicit one-coordinate compensators, from distinct groups."""
    ranked = []
    for key in sorted(group_ladders):
        if key == excluded_group:
            continue
        ladder = group_ladders[key]
        weights = group_weights.get(key) or []
        for index, weight in enumerate(weights):
            weight = float(weight)
            if weight < min_weight:
                continue
            for delta in range(-int(max_delta), int(max_delta) + 1):
                if delta == 0:
                    continue
                changed = _valid_group_rung_change(
                    ladder, index, delta,
                    protected_sl=protected_sl, min_gap=min_gap,
                    min_level=min_level, max_level=max_level)
                if changed is None:
                    continue
                residual = weight * delta - float(required_cost_delta)
                if abs(residual) > max_error:
                    continue
                ranked.append({
                    "group": key,
                    "index": index,
                    "delta": delta,
                    "from": int(ladder[index]),
                    "to": int(changed[index]),
                    "weight": weight,
                    "residual": float(residual),
                    "ladder": changed,
                })
    ranked.sort(key=lambda x: (
        abs(x["residual"]), -x["weight"], abs(x["delta"]),
        x["group"], x["index"], x["delta"]))
    out, used_groups = [], set()
    for item in ranked:
        if item["group"] in used_groups:
            continue
        out.append(item)
        used_groups.add(item["group"])
        if len(out) >= keep:
            break
    return out


def generate_group_candidates(
    group_ladders: dict[str, list[int]],
    group_weights: dict[str, list[float]],
    *,
    target_cost: float,
    protected_weight: float,
    protected_sl: int,
    steps: tuple[int, ...] = (2, 4),
    min_gap: int = 1,
    min_level: int = 16,
    max_level: int = 128,
    min_weight: float = 0.005,
    max_compensation: int = 16,
    max_budget_error: float = 0.15,
) -> list[tuple[str, dict[str, list[int]], dict]]:
    """Explicit two-coordinate, cross-group fixed-cost exchanges.

    The old implementation nudged one named rung and then re-projected every
    other ladder.  A nominal one-rung proposal consequently changed 14--22
    coordinates, often outside the named group, so neither attribution nor
    pin safety was meaningful.  Here the primary and one active compensator
    are the only coordinates allowed to move.  Dead rungs are skipped and the
    compensation is bounded.
    """
    out: list[tuple[str, dict[str, list[int]], dict]] = []
    base_cost = (
        float(protected_weight) * int(protected_sl)
        + sum(weighted_cost(group_ladders[k], group_weights[k])
              for k in group_ladders)
    )
    for key in sorted(group_ladders):
        ladder = group_ladders[key]
        weights = group_weights.get(key) or []
        # At most the two most occupied coordinates per group.  This removes
        # behavior-identical dead-rung screens and bounds the candidate bank.
        active = sorted(
            (i for i, w in enumerate(weights) if float(w) >= min_weight),
            key=lambda i: (-float(weights[i]), i))[:2]
        for idx in active:
            weight = float(weights[idx])
            for step in steps:
                for sign in (+1, -1):
                    delta = sign * int(step)
                    primary = _valid_group_rung_change(
                        ladder, idx, delta,
                        protected_sl=protected_sl, min_gap=min_gap,
                        min_level=min_level, max_level=max_level)
                    if primary is None:
                        continue
                    after_primary = base_cost + weight * delta
                    required = float(target_cost) - after_primary
                    compensators = _group_compensators(
                        group_ladders, group_weights,
                        required_cost_delta=required,
                        excluded_group=key,
                        protected_sl=protected_sl, min_gap=min_gap,
                        min_level=min_level, max_level=max_level,
                        min_weight=min_weight,
                        max_delta=max_compensation,
                        max_error=max_budget_error,
                        keep=1)
                    if not compensators and abs(required) > max_budget_error:
                        continue
                    comp = compensators[0] if compensators else None
                    merged = {
                        k: [int(x) for x in v]
                        for k, v in group_ladders.items()
                    }
                    merged[key] = primary
                    if comp is not None:
                        merged[comp["group"]] = comp["ladder"]
                    predicted = (after_primary
                                 + (0.0 if comp is None else
                                    comp["weight"] * comp["delta"]))
                    name = (
                        f"xgrp_{key.replace(':', '-')}_i{idx}_{delta:+d}_"
                        + (f"by_{comp['group'].replace(':', '-')}_"
                           f"i{comp['index']}_{comp['delta']:+d}"
                           if comp is not None else "_local"))
                    out.append((name, merged, {
                        "type": "group_exchange",
                        "primary": {
                            "group": key, "rung_index": idx,
                            "delta": delta, "from": ladder[idx],
                            "to": primary[idx], "weight": weight,
                        },
                        "compensator": (None if comp is None else {
                            k: comp[k] for k in (
                                "group", "index", "delta", "from", "to",
                                "weight", "residual")}),
                        "predicted_cost": float(predicted),
                        "predicted_budget_error": float(
                            predicted - target_cost),
                        "changed_coordinates": 2,
                    }))
    return out


def build_group_proposals(
    table: dict,
    occupancy: dict,
    group_ladders: dict[str, list[int]],
    *,
    target: float,
    table_levels: list[int],
    root_parent_table: Path,
    command: str,
    protected_sl: int | None,
    min_level: int,
    max_level: int,
    pc_lengths: list[int] | None = None,
) -> list[dict]:
    """Group-scoped candidates in the same record shape the sweep consumes.

    ``levels`` carries the {group: ladder} dict instead of a flat list; deploy()
    and occupancy_of() are polymorphic on that, so the rest of the sweep --
    stage A/B screening, the A->B replication gate, the confirm veto, the
    budget band -- is shared verbatim with the global path.
    """
    weights = occupancy["group_level_weights"]
    out: list[dict] = []
    for name, merged, detail in generate_group_candidates(
            group_ladders, weights,
            target_cost=float(target),
            protected_weight=float(occupancy.get("protected_weight") or 0.0),
            protected_sl=int(protected_sl or 0),
            min_level=min_level, max_level=max_level):
        if merged == group_ladders:
            continue
        if protected_sl and any(int(protected_sl) in lad
                                for lad in merged.values()):
            continue          # pin/rung collision would corrupt realized cost
        action = {
            "type": "group_exchange",
            "name": name,
            **detail,
            "group_ladders": {k: list(v) for k, v in merged.items()},
        }
        try:
            candidate = make_v16_table(
                table, list(table_levels),
                root_parent_table=root_parent_table,
                target_cost=float(target),
                action=action,
                command=command,
            )
        except ValueError:
            continue
        out.append({
            "name": name,
            "family": "group_exchange",
            "levels": merged,
            "table": candidate,
            "action": action,
            "protected_sl": None,
        })

    # -- protected-channel length -------------------------------------------
    # The pins are a SINGLE global stream length, and prior waves accepted
    # pc_len 68 / 72 / 84 on three of four models against a 112 default, so
    # 112 is measurably over-provisioned (it holds ~3% of MACs but ~10% of the
    # cycle budget at target 32).  Shortening the pins frees budget the
    # projector reinvests across every group, so this move has to stay
    # available in group mode -- omitting it was a regression.
    pw = float(occupancy.get("protected_weight") or 0.0)
    if protected_sl and pw > 0.0:
        base_cost = (
            pw * int(protected_sl)
            + sum(weighted_cost(group_ladders[k], weights[k])
                  for k in group_ladders)
        )
        for new_psl in (pc_lengths or []):
            new_psl = int(new_psl)
            if new_psl == int(protected_sl):
                continue
            if not (min_level < new_psl <= max_level):
                continue
            after_pin = base_cost + pw * (new_psl - int(protected_sl))
            required = float(target) - after_pin
            merged = {
                k: [int(x) for x in v] for k, v in group_ladders.items()
            }
            compensation = None
            if abs(required) > 0.05:
                choices = _group_compensators(
                    group_ladders, weights,
                    required_cost_delta=required,
                    excluded_group=None,
                    protected_sl=new_psl, min_gap=1,
                    min_level=min_level, max_level=max_level,
                    min_weight=0.005, max_delta=16,
                    max_error=0.05, keep=1)
                if not choices:
                    continue
                compensation = choices[0]
                merged[compensation["group"]] = compensation["ladder"]
            predicted = after_pin + (
                0.0 if compensation is None else
                compensation["weight"] * compensation["delta"])
            if abs(predicted - float(target)) > 0.05:
                continue
            # a pin length that lands on an adaptive rung cannot be separated
            # from that rung in the trace, which corrupts realized cost
            if any(new_psl in lad for lad in merged.values()):
                continue
            raw_table = copy.deepcopy(table)
            raw_table.setdefault("protected_channels", {})
            raw_table["protected_channels"]["stoc_len"] = new_psl
            action = {
                "type": "group_pc_length",
                "name": f"pc_len_{new_psl}",
                "old_protected_stoc_len": int(protected_sl),
                "new_protected_stoc_len": new_psl,
                "compensator": (None if compensation is None else {
                    k: compensation[k] for k in (
                        "group", "index", "delta", "from", "to", "weight",
                        "residual")
                }),
                "predicted_cost": float(predicted),
                "predicted_budget_error": float(predicted - float(target)),
                "group_ladders": {k: list(v) for k, v in merged.items()},
            }
            try:
                candidate = make_v16_table(
                    raw_table, list(table_levels),
                    root_parent_table=root_parent_table,
                    target_cost=float(target),
                    action=action,
                    command=command,
                )
            except ValueError:
                continue
            out.append({
                "name": f"pc_len_{new_psl}",
                "family": "pc_length",
                "levels": merged,
                "table": candidate,
                "action": action,
                "protected_sl": new_psl,
            })
    return out


def _expected_profile_cost(table: dict, profile: dict) -> float:
    """Expected MAC-weighted cost in profile currency (halved cycles).

    This is the same expected-cost machinery the working hand-built 14B
    compound used: profile_level_weights prices the table's thresholds
    against the profiled metric histograms, weighted_cost adds the
    protected pins.
    """
    psl, pw = _protected_info(table)
    lw = profile_level_weights(table, profile, protected_weight=pw)
    return weighted_cost(table["stoc_len_levels"], lw, pw, psl or 0)


def _attention_lift_table(table: dict) -> dict | None:
    """Re-emit all 8 attention buckets to the top rung via the sentinel.

    The dispatch space is the min-max-normalized metric in [0, 1] with the
    sign folded in BEFORE normalization by the runtime, so the sign-safe
    below-min sentinel is thresholds = [0.0] * (n-1): every row's
    normalized metric is >= 0.0 -> level 0 (top rung), and all lower bands
    are empty regardless of the bucket's dispatch_metrics sign.  Counts and
    fractions are updated to the designed occupancy so stored diagnostics
    stay truthful (proven mechanics: the 14B attn_lift96_pc95_reinvest
    build script).  Returns None if a bucket is missing or the lift is a
    no-op (already lifted).
    """
    buckets = table.get("buckets") or {}
    if not all(key in buckets for key in ATTN_LIFT_BUCKETS):
        return None
    # Occupancy lives under "counts"/"fractions" on parent, value_move,
    # pc_length and macro tables, but topology_insert/topology_remove archive
    # the (now stale) pass-1 occupancy as "pass1_counts"/"pass1_fractions".
    # Read whichever the table carries and write the SAME key back, so a
    # lifted topology table keeps the topology naming convention.
    keyed = []
    for key in ATTN_LIFT_BUCKETS:
        payload = buckets[key]
        ckey = "counts" if "counts" in payload else "pass1_counts"
        fkey = "fractions" if "fractions" in payload else "pass1_fractions"
        if payload.get(ckey) is None:
            return None  # occupancy unknown -> cannot restate it truthfully
        keyed.append((key, ckey, fkey))
    out = copy.deepcopy(table)
    n = len(out["stoc_len_levels"])
    sentinel = [0.0] * (n - 1)
    changed = False
    for key, ckey, fkey in keyed:
        payload = out["buckets"][key]
        if [float(x) for x in payload["thresholds"]] != sentinel:
            changed = True
        total_rows = int(sum(int(x) for x in payload[ckey]))
        payload["thresholds"] = list(sentinel)
        payload[ckey] = [total_rows] + [0] * (n - 1)
        payload[fkey] = [1.0] + [0.0] * (n - 1)
    return out if changed else None


def _iso_fill_mlp(
    table: dict,
    profile: dict,
    *,
    iso_target: float,
    tol: float = 0.02,
    hard_tol: float = 0.05,
    max_iter: int = 48,
) -> tuple[dict, list[dict], float]:
    """Restore iso-cost through the MLP floor boundaries (profile currency).

    Positive need (under iso_target) -> floor RAISES: promote MAC mass
    upward across the bottom boundary (-1, e.g. the 24/16 boundary), then
    the one above (-2) if the first saturates — exactly the reinvest rule
    of the working 14B compound.  Negative need -> the mirrored demotes pay
    for an attention lift.  The ladder is never touched, so this can never
    introduce a rung below the parent floor.  Raises ValueError when the
    achievable cost misses iso_target by more than hard_tol.
    """
    def cost_of(t: dict) -> float:
        return _expected_profile_cost(t, profile)

    current = table
    all_moves: list[dict] = []
    base_cost = cost_of(current)
    if abs(base_cost - iso_target) <= tol:
        return current, all_moves, base_cost
    sign = 1.0 if iso_target > base_cost else -1.0
    for boundary in (-1, -2):
        sat_tab, sat_moves = shift_threshold_group(
            current, profile, ops=OP_GROUPS["mlp"], boundary=boundary,
            mass_delta=sign * 1.0)
        sat_cost = cost_of(sat_tab)
        overshoots = (sat_cost >= iso_target if sign > 0
                      else sat_cost <= iso_target)
        if not overshoots:
            # Full spend at this boundary still short: keep it, go deeper.
            current = sat_tab
            all_moves.extend(sat_moves)
            continue
        lo, hi = 0.0, 1.0
        best: tuple = (None, None, float("inf"), None)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            cand, moves = shift_threshold_group(
                current, profile, ops=OP_GROUPS["mlp"], boundary=boundary,
                mass_delta=sign * mid)
            c = cost_of(cand)
            if abs(c - iso_target) < best[2]:
                best = (cand, moves, abs(c - iso_target), c)
            if (c < iso_target) == (sign > 0):
                lo = mid
            else:
                hi = mid
        if best[0] is not None:
            current = best[0]
            all_moves.extend(best[1])
            if best[2] <= tol:
                return current, all_moves, best[3]
        break
    final = cost_of(current)
    if abs(final - iso_target) > hard_tol:
        raise ValueError(
            f"lift_compound iso fill missed target: {final:.4f} vs "
            f"{iso_target:.4f} (hard tol {hard_tol})")
    return current, all_moves, final


def propose_lift_compound(
    table: dict,
    profile: dict,
    *,
    protected_sl: int,
    protected_weight: float,
) -> list[tuple[str, dict, int | None, dict, list[dict], float]]:
    """The measured-direction compound as ONE candidate plus partials.

    Variants (each iso-cost vs the incumbent's expected profile cost):
      lift_full         — attention buckets -> top rung, protected pins ->
                          95 cycles, MLP floor raises reinvest the residual;
      lift_attn_only    — attention -> top rung, paid by the mirrored MLP
                          boundary demotes;
      lift_psl_reinvest — protected pins -> 95 only, freed budget
                          reinvested into MLP floor raises.
    Ladder untouched in every variant (no sub-floor rung can appear); psl
    target is LIFT_PSL=95, never 96 (rung collision).
    """
    levels = [int(x) for x in table["stoc_len_levels"]]
    try:
        iso_target = _expected_profile_cost(table, profile)
    except (KeyError, ValueError):
        return []
    lifted = _attention_lift_table(table)
    can_psl = bool(
        protected_sl and protected_weight > 0.0
        and int(protected_sl) != LIFT_PSL and LIFT_PSL not in levels)
    variants: list[tuple[str, dict, int | None, dict]] = []
    if lifted is not None and can_psl:
        full = copy.deepcopy(lifted)
        full["protected_channels"]["stoc_len"] = LIFT_PSL
        variants.append(("lift_full", full, LIFT_PSL, {
            "attention_lift": list(ATTN_LIFT_BUCKETS),
            "old_protected_stoc_len": int(protected_sl),
            "new_protected_stoc_len": LIFT_PSL,
        }))
    if lifted is not None:
        variants.append(("lift_attn_only", copy.deepcopy(lifted), None, {
            "attention_lift": list(ATTN_LIFT_BUCKETS),
        }))
    if can_psl:
        psl_only = copy.deepcopy(table)
        psl_only["protected_channels"]["stoc_len"] = LIFT_PSL
        variants.append(("lift_psl_reinvest", psl_only, LIFT_PSL, {
            "old_protected_stoc_len": int(protected_sl),
            "new_protected_stoc_len": LIFT_PSL,
        }))
    out = []
    for name, raw, psl, detail in variants:
        try:
            filled, fill_moves, cost = _iso_fill_mlp(
                raw, profile, iso_target=iso_target)
        except (KeyError, ValueError):
            continue
        if [int(x) for x in filled["stoc_len_levels"]] != levels:
            continue  # defensive: the ladder must never change here
        detail = dict(detail)
        detail.update({
            "mlp_iso_fill_moves": len(fill_moves),
            "iso_target_profile_cost": float(iso_target),
            "iso_cost_currency": "profile_expected",
        })
        out.append((name, filled, psl, detail, fill_moves, float(cost)))
    return out


def _shift_bucket_filter(
    table: dict,
    profile: dict,
    *,
    keyfilter,
    boundary: int,
    mass_delta: float,
) -> tuple[dict, list[dict]]:
    """Shift one threshold boundary for an explicit (op, layer-bucket) set."""
    out = copy.deepcopy(table)
    values = _hist_values(profile)
    moves = []
    for key, group in (profile.get("groups") or {}).items():
        op = str(group["op"])
        layer_bucket = group.get("l_bucket")
        if not keyfilter(op, layer_bucket):
            continue
        payload = (out.get("buckets") or {}).get(key)
        if payload is None:
            continue
        shifted, move = _shift_one_boundary(
            [float(x) for x in payload["thresholds"]],
            [float(x) for x in group["mac_weighted_hist"]],
            values,
            boundary=boundary,
            mass_delta=mass_delta,
        )
        payload["thresholds"] = shifted
        move.update({"group": key, "op": op,
                     "l_bucket": layer_bucket})
        moves.append(move)
    return out, moves


def propose_structured_transfers(
    table: dict,
    profile: dict,
    *,
    mass_steps: list[float] | tuple[float, ...] = (0.10, 0.15, 0.30),
    boundary: int = -1,
    hard_tol: float = 0.08,
) -> list[tuple[str, dict, dict, list[dict], float]]:
    """Atomic, iso-cost transfers across layer and operator populations.

    A global ladder cannot express "protect late MLP and fund it from early
    MLP" because both populations share the same rung values.  Per-bucket
    threshold moves can express it, but the old search moved a whole operator
    in isolation and never paired the receiver with a donor.  This family
    makes the coordinated transfer one candidate: promote the receiver at the
    floor boundary, then binary-search the donor demotion back to the
    incumbent's expected cost.

    Both directions are enumerated.  The historically strong late<-early
    move is therefore available without hard-coding it as an answer, and a
    model whose sensitivity is reversed can select the mirror image.
    """
    mlp = set(OP_GROUPS["mlp"])
    schemes = [
        (
            "mlp_late_from_early",
            lambda op, lb: op in mlp and lb in (2, 3),
            lambda op, lb: op in mlp and lb in (0, 1),
        ),
        (
            "mlp_early_from_late",
            lambda op, lb: op in mlp and lb in (0, 1),
            lambda op, lb: op in mlp and lb in (2, 3),
        ),
    ]
    # Also expose sensitivity-aligned per-operator transfers.  Each MLP op is
    # tried as receiver and the other two jointly fund it; measured results
    # show the useful direction is model-dependent.
    for receiver in sorted(mlp):
        donors = mlp - {receiver}
        schemes.append((
            f"mlp_{receiver}_from_peers",
            lambda op, lb, receiver=receiver: op == receiver,
            lambda op, lb, donors=donors: op in donors,
        ))

    try:
        iso_target = _expected_profile_cost(table, profile)
    except (KeyError, ValueError):
        return []
    out = []
    for scheme, promote_filter, fund_filter in schemes:
        for mass in mass_steps:
            mass = float(mass)
            if not 0.0 < mass <= 1.0:
                continue
            promoted, promote_moves = _shift_bucket_filter(
                table, profile, keyfilter=promote_filter,
                boundary=boundary, mass_delta=mass)
            if not any(m.get("changed") for m in promote_moves):
                continue
            try:
                promoted_cost = _expected_profile_cost(promoted, profile)
            except (KeyError, ValueError):
                continue
            # A promotion that costs nothing cannot identify a receiver.
            if promoted_cost <= iso_target + 1e-9:
                continue
            lo, hi = 0.0, 1.0
            best = None
            for _ in range(48):
                demote = 0.5 * (lo + hi)
                candidate, fund_moves = _shift_bucket_filter(
                    promoted, profile, keyfilter=fund_filter,
                    boundary=boundary, mass_delta=-demote)
                try:
                    cost = _expected_profile_cost(candidate, profile)
                except (KeyError, ValueError):
                    break
                error = abs(cost - iso_target)
                if best is None or error < best[0]:
                    best = (error, candidate, fund_moves, cost, demote)
                if cost > iso_target:
                    lo = demote
                else:
                    hi = demote
            if best is None or best[0] > hard_tol:
                continue
            error, candidate, fund_moves, cost, demote = best
            if not any(m.get("changed") for m in fund_moves):
                continue
            detail = {
                "scheme": scheme,
                "boundary": int(boundary),
                "promote_mass": mass,
                "fund_mass": float(demote),
                "promote_buckets_changed": sum(
                    bool(m.get("changed")) for m in promote_moves),
                "fund_buckets_changed": sum(
                    bool(m.get("changed")) for m in fund_moves),
                "iso_target_profile_cost": float(iso_target),
                "predicted_budget_error": float(cost - iso_target),
            }
            name = f"xfer_{scheme}_m{int(round(mass * 100)):02d}"
            out.append((name, candidate, detail,
                        promote_moves + fund_moves, float(cost)))
    return out


def generate_v16_candidates(
    table: dict,
    profile: dict,
    occupancy: dict,
    *,
    target: float,
    root_parent_table: Path,
    command: str,
    phase: str,
    threshold_mass_steps: list[float],
    split_fractions: list[float],
    min_classes: int,
    max_classes: int,
    min_level: int,
    max_level: int,
    allow_removal: bool,
    structured_mass_steps: list[float] | None = None,
    prev_pair_moves: list[tuple[int, int]] | None = None,
    pc_lengths: list[int] | None = None,
) -> list[dict]:
    """One coordinate-plus-macro neighborhood around the incumbent."""
    levels = [int(x) for x in table["stoc_len_levels"]]
    n = len(levels)
    protected_weight = float(occupancy["protected_weight"])
    protected_sl = int((table.get("protected_channels") or {}).get("stoc_len", 0))
    proposals: list[dict] = []

    def collides_protected(new_levels: list[int]) -> bool:
        return bool(protected_sl and protected_sl not in levels
                    and protected_sl in new_levels)

    def finish(name, family, raw_table, projection, detail, psl=None):
        try:
            new_levels = [int(x) for x in projection["levels"]]
            own_psl = int(psl) if psl else protected_sl
            if own_psl and own_psl not in levels and own_psl in new_levels:
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
            candidate = make_v16_table(
                raw_table, new_levels,
                root_parent_table=root_parent_table,
                target_cost=target,
                action=action,
                command=command,
            )
        except ValueError:
            return
        proposals.append({
            "name": name,
            "family": family,
            "levels": new_levels,
            "table": candidate,
            "action": action,
            "protected_sl": (int(psl) if psl else None),
        })

    def constrained(seed, locked, weights=None, psl=None):
        return project_levels_constrained(
            seed,
            weights if weights is not None else occupancy["level_weights"],
            target_cost=target,
            protected_weight=protected_weight,
            protected_stoc_len=int(psl) if psl else protected_sl,
            locked=locked,
            min_level=min_level,
            max_level=max_level,
        )

    if phase == "precision":
        # -- single value moves: fine near the floor, coarse above it -------
        for idx, old in enumerate(levels):
            step = 2 if idx >= n - 2 else 4
            upper = max_level if idx == 0 else levels[idx - 1] - 1
            lower = min_level if idx + 1 == n else levels[idx + 1] + 1
            for delta in (-step, step):
                moved = min(max(old + delta, lower), upper)
                if moved == old:
                    continue
                seed = levels.copy()
                seed[idx] = moved
                if collides_protected(seed):
                    continue
                try:
                    projection = constrained(seed, {idx: moved})
                except ValueError:
                    continue
                finish(
                    f"value_i{idx}_{moved}", "value_move", table, projection,
                    {"index": idx, "old_level": old, "requested_level": moved},
                )

        # -- insertions (extra values in the sub-floor gap) -----------------
        if n < max_classes:
            sites: list[tuple[int, list[int]]] = []
            if levels[0] < max_level:
                sites.append((0, _insertion_values(
                    max_level, levels[0], 2, include_high=True)))
            for i in range(1, n):
                sites.append((i, _insertion_values(
                    levels[i - 1], levels[i], 2)))
            if levels[-1] > min_level:
                sites.append((n, _insertion_values(
                    levels[-1], min_level, 3, include_low=True)))
            for insert_idx, values in sites:
                for new_level in values:
                    if protected_sl and protected_sl not in levels \
                            and new_level == protected_sl:
                        continue
                    for fraction in split_fractions:
                        try:
                            raw_table, moves = insert_precision_rung(
                                table, profile,
                                insert_idx=insert_idx,
                                new_level=new_level,
                                promote_fraction=fraction,
                            )
                            seed = [int(x) for x in raw_table["stoc_len_levels"]]
                            weights = profile_level_weights(
                                raw_table, profile,
                                protected_weight=protected_weight)
                            projection = constrained(
                                seed, {insert_idx: new_level}, weights)
                        except ValueError:
                            continue
                        finish(
                            f"insert_i{insert_idx}_l{new_level}"
                            f"_f{int(round(fraction * 100))}",
                            "topology_insert", raw_table, projection,
                            {"insert_idx": insert_idx, "new_level": new_level,
                             "promote_fraction": fraction},
                        )

        # -- removals (accepted only through the confirm gate) --------------
        if allow_removal and n > min_classes:
            for remove_idx in range(n):
                merges = ("down",) if remove_idx == 0 else (
                    ("up",) if remove_idx == n - 1 else ("up", "down"))
                for merge in merges:
                    try:
                        raw_table, moves = remove_precision_rung(
                            table, remove_idx=remove_idx, merge=merge)
                        seed = [int(x) for x in raw_table["stoc_len_levels"]]
                        weights = profile_level_weights(
                            raw_table, profile,
                            protected_weight=protected_weight)
                        projection = project_levels_to_budget(
                            seed, weights,
                            target_cost=target,
                            protected_weight=protected_weight,
                            protected_stoc_len=protected_sl,
                            min_level=min_level,
                            max_level=max_level,
                        )
                    except ValueError:
                        continue
                    finish(
                        f"remove_i{remove_idx}_{merge}", "topology_remove",
                        raw_table, projection,
                        {"remove_idx": remove_idx, "merge": merge},
                    )

        # -- macro-moves -----------------------------------------------------
        for floor_step in (4, 8):
            try:
                proposal = propose_floor_exchange(
                    levels, occupancy["level_weights"],
                    target_cost=target,
                    protected_weight=protected_weight,
                    protected_stoc_len=protected_sl,
                    floor_step=floor_step,
                )
                if proposal["levels"][0] > max_level:
                    raise ValueError("floor exchange exceeded max level")
            except ValueError:
                continue
            finish(
                f"macro_floorx_{floor_step}", "macro", table, proposal,
                {"macro": "floor_exchange", "floor_step": floor_step},
            )
        for drop in (8, 16):
            top = levels[0] - drop
            if top <= levels[1]:
                continue
            seed = levels.copy()
            seed[0] = top
            try:
                projection = constrained(seed, {0: top})
            except ValueError:
                continue
            finish(
                f"macro_ceilcomp_{drop}", "macro", table, projection,
                {"macro": "ceiling_compress", "drop": drop},
            )
        for top in {min(levels[0] + 16, max_level), max_level}:
            if top <= levels[0]:
                continue
            seed = levels.copy()
            seed[0] = top
            try:
                projection = constrained(seed, {0: top})
            except ValueError:
                continue
            finish(
                f"macro_ceilraise_{top}", "macro", table, projection,
                {"macro": "ceiling_raise", "top": top},
            )
        pair = [(int(i), int(v)) for i, v in (prev_pair_moves or [])
                if 0 <= int(i) < n]
        if len({i for i, _ in pair}) >= 2:
            (i1, v1), (i2, v2) = pair[0], next(
                (m for m in pair[1:] if m[0] != pair[0][0]))
            seed = levels.copy()
            seed[i1], seed[i2] = v1, v2
            try:
                projection = constrained(seed, {i1: v1, i2: v2})
            except ValueError:
                projection = None
            if projection is not None:
                finish(
                    f"macro_pair_i{i1}_{v1}_i{i2}_{v2}", "macro", table,
                    projection,
                    {"macro": "joint_pair",
                     "moves": [[i1, v1], [i2, v2]]},
                )

        # -- protected-channel length (v16.1): the 112-cycle pins hold
        # ~3% of MACs but ~10% of the cycle budget at target 32 and were
        # never ablated; shortening them frees budget the projector
        # reinvests in the adaptive rungs. Table-driven — the runtime
        # reads protected_channels.stoc_len from the deployed table.
        if protected_sl and protected_weight > 0.0:
            for new_psl in (pc_lengths or []):
                new_psl = int(new_psl)
                if new_psl == protected_sl or new_psl in levels:
                    continue
                if not min_level < new_psl <= max_level:
                    continue
                raw_table = copy.deepcopy(table)
                raw_table["protected_channels"]["stoc_len"] = new_psl
                try:
                    projection = constrained(levels, {}, psl=new_psl)
                    if new_psl in projection["levels"]:
                        continue
                except ValueError:
                    continue
                finish(
                    f"pc_len_{new_psl}", "pc_length", raw_table, projection,
                    {"old_protected_stoc_len": protected_sl,
                     "new_protected_stoc_len": new_psl},
                    psl=new_psl,
                )

        # -- lift_compound (v17): measured-direction compound as ONE
        # candidate + partial variants.  Iso-cost is restored in profile
        # (expected-cost) currency through the MLP floor boundaries; the
        # ladder never changes, so no new rung (in particular none below
        # the parent floor of 16) can be introduced by this family.
        for name, raw_table, vpsl, detail, fill_moves, cost in \
                propose_lift_compound(
                    table, profile,
                    protected_sl=int(protected_sl or 0),
                    protected_weight=protected_weight):
            projection = {
                "type": "lift_compound_iso_fill",
                "levels": levels.copy(),
                "predicted_cost": float(cost),
                "moves": [
                    {k: v for k, v in m.items() if k != "hist"}
                    for m in fill_moves],
            }
            finish(name, "lift_compound", raw_table, projection, detail,
                   psl=vpsl)

        # -- structured transfers (v20): coordinated, iso-cost population
        # reallocations.  This restores the large-jump axis omitted by the
        # op:layer experiment.  In particular, late-MLP<-early-MLP improved
        # full-test PPL on all four measured models, while its reverse hurt;
        # both directions remain candidates and must earn A->B->C acceptance.
        for name, raw_table, detail, moves, cost in \
                propose_structured_transfers(
                    table, profile,
                    mass_steps=(structured_mass_steps
                                or [0.15, 0.30, 0.40])):
            projection = {
                "type": "structured_transfer_iso_cost",
                "levels": levels.copy(),
                "predicted_cost": float(cost),
                "moves": [
                    {k: v for k, v in m.items() if k != "hist"}
                    for m in moves],
            }
            finish(name, "structured_transfer", raw_table, projection,
                   detail)

    if phase == "threshold":
        for mass_step in threshold_mass_steps:
            for group, ops in OP_GROUPS.items():
                for boundary in range(n - 1):
                    for direction in (-1.0, 1.0):
                        try:
                            raw_table, moves = shift_threshold_group(
                                table, profile,
                                ops=ops,
                                boundary=boundary,
                                mass_delta=direction * mass_step,
                            )
                            if not any(m.get("changed") for m in moves):
                                continue
                            weights = profile_level_weights(
                                raw_table, profile,
                                protected_weight=protected_weight)
                            projection = project_levels_to_budget(
                                levels, weights,
                                target_cost=target,
                                protected_weight=protected_weight,
                                protected_stoc_len=protected_sl,
                                min_level=min_level,
                                max_level=max_level,
                            )
                        except ValueError:
                            continue
                        tag = "promote" if direction > 0 else "demote"
                        finish(
                            f"threshold_{group}_b{boundary}_{tag}"
                            f"_m{int(round(mass_step * 1000))}",
                            "threshold_move", raw_table, projection,
                            {"group": group, "boundary": boundary,
                             "mass_delta": direction * mass_step},
                        )

    unique, seen = [], set()
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


# --------------------------------------------------------------------------
# v16.1 surrogate-guided compound proposals
#
# Mining the completed v16 histories showed (a) single-coordinate moves near
# the incumbent barely replicate across windows while multi-coordinate moves
# replicate strongly (A->B reliability +0.87/+0.56/+0.37 vs ~0), and (b) the
# in-region candidate variance is 74-93% explainable.  The surrogate is a
# ridge fit on the branch's own history that nominates a few compound ladders
# per sweep; it only chooses what to TRY — every proposal still passes the
# A/B/confirm gates — and it is only used when its predictions transfer to
# the branch's held-out B-stage results.
# --------------------------------------------------------------------------

def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gauss-Jordan solve for the small dense ridge systems used here."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        a[col] = [v / div for v in a[col]]
        for row in range(n):
            if row != col and a[row][col] != 0.0:
                factor = a[row][col]
                a[row] = [v - factor * w for v, w in zip(a[row], a[col])]
    return [a[i][n] for i in range(n)]


def ladder_features(
    levels: list[int],
    cost: float,
    target: float,
    pc_len: int | None,
) -> list[float]:
    """Interpretable ladder-shape features; per-rung one-hots do not transfer."""
    levels = [int(x) for x in levels]
    floor = levels[-1]
    top = levels[0]
    n = len(levels)
    gaps = [levels[i] - levels[i + 1] for i in range(n - 1)] or [1]
    return [
        1.0 / max(floor, 1),
        float(floor),
        float(levels[-2]) if n >= 2 else float(floor),
        float(top),
        float(n),
        (top - floor) / max(n - 1, 1),
        float(min(gaps)),
        float(sum(1 for x in levels if x <= 32)),
        float(cost) - float(target),
        float(pc_len or 0),
    ]


def _center_by_era(rows: list[dict], key: str) -> list[float]:
    sums: dict[tuple, list[float]] = {}
    for r in rows:
        sums.setdefault((r["sweep"], r["phase"]), []).append(float(r[key]))
    means = {era: sum(v) / len(v) for era, v in sums.items()}
    return [float(r[key]) - means[(r["sweep"], r["phase"])] for r in rows]


def fit_surrogate(
    records: list[dict],
    *,
    target: float,
    min_history: int = 40,
    min_transfer: float = 0.2,
    lam: float = 1.0,
) -> dict:
    """Ridge fit on A-stage history, gated on transfer to B-stage results."""
    a_recs = [r for r in records
              if r.get("stage") == "A" and r.get("levels")]
    out = {"ok": False, "n_a": len(a_recs), "n_b": 0, "r": None}
    if len(a_recs) < min_history:
        return out
    feats = [ladder_features(r["levels"], r["cost"], target,
                             r.get("protected_sl")) for r in a_recs]
    y = _center_by_era(a_recs, "mean_delta_a")
    k = len(feats[0])
    mu = [sum(f[j] for f in feats) / len(feats) for j in range(k)]
    sd = []
    for j in range(k):
        var = sum((f[j] - mu[j]) ** 2 for f in feats) / max(len(feats) - 1, 1)
        sd.append(math.sqrt(var) if var > 1e-18 else 1.0)
    xs = [[(f[j] - mu[j]) / sd[j] for j in range(k)] for f in feats]
    xtx = [[sum(x[i] * x[j] for x in xs) + (lam if i == j else 0.0)
            for j in range(k)] for i in range(k)]
    xty = [sum(x[i] * yi for x, yi in zip(xs, y)) for i in range(k)]
    try:
        beta = _solve_linear(xtx, xty)
    except ValueError:
        return out
    out.update({"beta": beta, "mu": mu, "sd": sd})

    a_by_key = {(r["sweep"], r["phase"], r["candidate"]): f
                for r, f in zip(a_recs, feats)}
    b_recs = [r for r in records
              if r.get("stage") == "B"
              and (r["sweep"], r["phase"], r["candidate"]) in a_by_key]
    out["n_b"] = len(b_recs)
    if len(b_recs) < 8:
        return out
    y_b = _center_by_era(b_recs, "mean_delta_ab")
    preds = [surrogate_predict(out, a_by_key[
        (r["sweep"], r["phase"], r["candidate"])]) for r in b_recs]
    mp_, mb = sum(preds) / len(preds), sum(y_b) / len(y_b)
    cov = sum((p - mp_) * (b - mb) for p, b in zip(preds, y_b))
    vp = sum((p - mp_) ** 2 for p in preds)
    vb = sum((b - mb) ** 2 for b in y_b)
    if vp <= 1e-18 or vb <= 1e-18:
        return out
    r = cov / math.sqrt(vp * vb)
    out["r"] = r
    out["ok"] = bool(math.isfinite(r) and r >= min_transfer)
    return out


def surrogate_predict(surrogate: dict, features: list[float]) -> float:
    return sum(b * (f - m) / s for b, f, m, s in zip(
        surrogate["beta"], features, surrogate["mu"], surrogate["sd"]))


def surrogate_candidates(
    table: dict,
    occupancy: dict,
    surrogate: dict,
    *,
    target: float,
    root_parent_table: Path,
    command: str,
    seen_levels: set[tuple],
    k: int,
    min_level: int,
    max_level: int,
) -> list[dict]:
    """Top-k surrogate-scored compound (two-coordinate) ladder proposals."""
    levels = [int(x) for x in table["stoc_len_levels"]]
    n = len(levels)
    protected_weight = float(occupancy["protected_weight"])
    protected_sl = int(
        (table.get("protected_channels") or {}).get("stoc_len", 0))
    scored = []
    deltas = (-8, -4, -2, 2, 4, 8)
    for i in range(n):
        for j in range(i + 1, n):
            for di in deltas:
                for dj in deltas:
                    seed = levels.copy()
                    seed[i] += di
                    seed[j] += dj
                    if any(seed[p] <= seed[p + 1] for p in range(n - 1)):
                        continue
                    if seed[0] > max_level or seed[-1] < min_level:
                        continue
                    try:
                        projection = project_levels_constrained(
                            seed, occupancy["level_weights"],
                            target_cost=target,
                            protected_weight=protected_weight,
                            protected_stoc_len=protected_sl,
                            locked={i: seed[i], j: seed[j]},
                            min_level=min_level,
                            max_level=max_level,
                        )
                    except ValueError:
                        continue
                    new_levels = [int(x) for x in projection["levels"]]
                    key = tuple(new_levels)
                    if key in seen_levels:
                        continue
                    if protected_sl and protected_sl not in levels \
                            and protected_sl in new_levels:
                        continue
                    score = surrogate_predict(surrogate, ladder_features(
                        new_levels, projection["predicted_cost"], target,
                        None))
                    scored.append((score, i, di, j, dj, projection))
    scored.sort(key=lambda item: item[0])
    proposals = []
    used: set[tuple] = set()
    for score, i, di, j, dj, projection in scored:
        new_levels = tuple(int(x) for x in projection["levels"])
        if new_levels in used:
            continue
        used.add(new_levels)
        name = f"surr_i{i}{'+' if di > 0 else ''}{di}_i{j}{'+' if dj > 0 else ''}{dj}"
        action = {
            "type": "surrogate",
            "name": name,
            "moves": [[i, di], [j, dj]],
            "predicted_delta": float(score),
            "level_projection": _short_action(projection),
            "predicted_cost": float(projection["predicted_cost"]),
        }
        try:
            candidate = make_v16_table(
                table, list(new_levels),
                root_parent_table=root_parent_table,
                target_cost=target,
                action=action,
                command=command,
            )
        except ValueError:
            continue
        proposals.append({
            "name": name,
            "family": "surrogate",
            "levels": list(new_levels),
            "table": candidate,
            "action": action,
            "protected_sl": None,
        })
        if len(proposals) >= k:
            break
    return proposals


# --------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------

def _windows_enc(enc_val, window_ids: list[int], ctx: int):
    import torch

    return torch.cat(
        [enc_val[w * ctx:(w + 1) * ctx] for w in window_ids])


def eval_windows(
    model,
    enc_val,
    window_ids: list[int],
    *,
    ctx: int,
    levels: list[int],
    table_path: Path,
    wrapper_path: Path,
    trace_path: Path,
    split_label: str,
    profile_path: Path | None = None,
    profile_bins: int = 257,
) -> dict:
    window_losses: list = []
    # The RUNTIME always needs a flat ladder: _install_runtime_table builds an
    # AdaptiveMPConfig whose stoc_len_levels is a list, and indexes it.  In
    # group mode the authoritative per-bucket ladders travel in the TABLE
    # (bucket["stoc_len_levels"], read by AdaptiveMPConfig.get_levels), so the
    # global list is only the fallback for a bucket that declares none -- the
    # union of the group ladders is the correct, strictly-descending value.
    runtime_levels = _runtime_levels(levels, table_path)
    result = _eval_once(
        model, _windows_enc(enc_val, window_ids, ctx),
        ctx=ctx,
        levels=runtime_levels,
        table_path=table_path,
        wrapper_path=wrapper_path,
        trace_path=trace_path,
        split=split_label,
        metric_profile_path=profile_path,
        metric_profile_bins=profile_bins,
        window_losses=window_losses,
    )
    if len(window_losses) != len(window_ids):
        raise RuntimeError(
            f"{split_label}: expected {len(window_ids)} scored windows, "
            f"got {len(window_losses)}")
    result["window_nll"] = {
        int(w): float(loss)
        for w, (_start, _valid, loss) in zip(window_ids, window_losses)
    }
    return result


class WalltimeClock:
    """Track spend and a sec/window EMA to guard sweeps and the wrap-up."""

    def __init__(self, budget_hours: float, slack: float = 1.2):
        self.start = time.monotonic()
        self.budget = float(budget_hours) * 3600.0
        self.sec_per_window: float | None = None
        self.slack = float(slack)
        # Observed sweep sizes.  The a-priori formula assumes 50 candidates
        # (288 windows); real sweeps measured 214-229.  Once a sweep has
        # actually completed we guard on what it COST, not on the guess.
        self.windows_seen = 0.0
        self._sweep_mark = 0.0
        self.sweep_windows_seen: list[float] = []

    def note(self, seconds: float, n_windows: int) -> None:
        if n_windows <= 0 or seconds <= 0:
            return
        self.windows_seen += float(n_windows)
        rate = seconds / n_windows
        self.sec_per_window = (
            rate if self.sec_per_window is None
            else 0.7 * self.sec_per_window + 0.3 * rate)

    def begin_sweep(self) -> None:
        self._sweep_mark = self.windows_seen

    def end_sweep(self) -> None:
        used = self.windows_seen - self._sweep_mark
        if used > 0:
            self.sweep_windows_seen.append(used)

    def sweep_windows(self, fallback: float) -> float:
        """Worst observed sweep so far, else the a-priori estimate."""
        if self.sweep_windows_seen:
            return max(self.sweep_windows_seen)
        return float(fallback)

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)

    def estimate(self, n_windows: float, slack: float | None = None) -> float:
        rate = self.sec_per_window if self.sec_per_window else 60.0
        s = self.slack if slack is None else float(slack)
        return rate * float(n_windows) * s + 300.0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _parse_float_list(spec: str) -> list[float]:
    out = []
    for token in spec.replace(":", ",").split(","):
        value = float(token)
        if value not in out:
            out.append(value)
    return out


def _parse_quotas(spec: str) -> dict[str, int]:
    out = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        key, _, value = token.partition("=")
        out[key.strip()] = int(value)
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-wrapper", required=True, type=Path)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--targets", default="32:36:40")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--block-windows", type=int, default=4)
    p.add_argument("--max-sweeps", type=int, default=6)
    p.add_argument("--patience", type=int, default=3,
                   help="consecutive no-accept sweeps before the branch "
                        "stops (V17 default 3, was 2)")
    p.add_argument("--advance-total", type=int, default=12)
    p.add_argument(
        "--family-quotas",
        default=("macro=3,value_move=3,topology_insert=3,"
                 "topology_remove=1,surrogate=4,pc_length=2,"
                 "lift_compound=3,structured_transfer=3,"
                 "group_exchange=3,threshold_move=6"))
    p.add_argument(
        "--confirm-top-k", type=int, default=1,
        help=("number of the A-ranked shortlist eligible to reach the "
              "16-window confirmation veto; K>1 uses a one-sided "
              "Bonferroni-adjusted SE multiplier and still accepts at most "
              "one move"))
    p.add_argument(
        "--threshold-phase", action=argparse.BooleanOptionalAction,
        default=False,
        help=("v16.1: threshold moves measured at/below the noise floor in "
              "every v16 branch (0/165 accepts); default off"))
    p.add_argument(
        "--pc-lengths", default="112:104:96:95:88:84:80:76:72:68:64",
        help=("candidate protected-channel stream lengths (halved cycles); "
              "empty string disables the pc_length family"))
    p.add_argument("--surrogate-proposals", type=int, default=5,
                   help="compound ladders proposed per sweep by the surrogate")
    p.add_argument("--surrogate-min-history", type=int, default=40)
    p.add_argument("--surrogate-min-transfer", type=float, default=0.2,
                   help="min Pearson r of A-fit predictions on held-out "
                        "B-stage results before surrogate proposals are used")
    p.add_argument("--threshold-mass-steps", default="0.03")
    p.add_argument(
        "--structured-mass-steps", default="0.10:0.15:0.30",
        help=("receiver mass steps for atomic iso-cost layer/operator "
              "transfers; 0.15 reproduces the measured late-MLP transfer"))
    p.add_argument("--split-fractions", default="0.33:0.67")
    p.add_argument("--min-classes", type=int, default=3)
    p.add_argument("--max-classes", type=int, default=8)
    p.add_argument("--min-level", type=int, default=16)
    p.add_argument("--max-level", type=int, default=128)
    p.add_argument(
        "--allow-removal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--budget-tol-under", type=float, default=0.35)
    p.add_argument("--budget-tol-over", type=float, default=1.0)
    p.add_argument("--budget-corrections", type=int, default=3)
    p.add_argument("--accept-floor", type=float, default=0.002)
    p.add_argument("--confirm-se-mult", type=float, default=1.0)
    p.add_argument("--sigma-fallback", type=float, default=DEFAULT_SIGMA_W)
    p.add_argument("--regress-window-nats", type=float, default=0.02)
    p.add_argument("--regress-window-max", type=int, default=4)
    p.add_argument("--fp16-ref", type=Path, default=None,
                   help="fp16_val_reference.py JSON for tier-1 stop checks")
    p.add_argument("--fp16-test-ppl", type=float, default=None,
                   help="full-test FP16 PPL for the tier-2 stop check")
    p.add_argument("--ppl-fp16-ratio", type=float, default=1.1)
    p.add_argument("--tier1-ratio-scale", type=float, default=0.90,
                   help="measured val/test PPL ratio pre-scaling for tier 1")
    p.add_argument("--test-tokens", type=int, default=0,
                   help="0 = full test split; -1 disables the final test eval")
    p.add_argument(
        "--holdout", action=argparse.BooleanOptionalAction, default=True,
        help="evaluate the frozen winner on the unused validation windows")
    p.add_argument("--walltime-hours", type=float, default=22.5)
    p.add_argument(
        "--guard-slack", type=float, default=1.10,
        help=("safety factor the walltime guard applies to its sweep and "
              "wrap-up estimates (was a hardcoded 1.20; measured sweeps came "
              "in ~1.3x under the a-priori window count already, so 1.20 "
              "compounded to ~1.56x and stopped 14B branches with 7-11h "
              "unused and 30B branches before sweep 1)"))
    p.add_argument(
        "--guard-cand-est", type=float, default=50.0,
        help=("a-priori candidate count used to size sweep 1 only; after any "
              "sweep completes the guard uses the worst OBSERVED sweep"))
    p.add_argument(
        "--confirm-veto", action=argparse.BooleanOptionalAction, default=True,
        help=("V18: re-test an A->B acceptance on the pooled A+B+confirm "
              "windows and REVERT to the previous incumbent if it no longer "
              "clears significance.  The confirm set was stop-only before, so "
              "a move that lost on it was still adopted (14B t48: confirm "
              "9.9297 -> 9.9546, +0.25%%, kept).  --no-confirm-veto restores "
              "the old stop-only behaviour"))
    p.add_argument(
        "--seed-protected-sl", type=int, default=None,
        help=("re-pin the protected channels to this stream length before the "
              "search starts. The V9 default of 112 is measurably "
              "over-provisioned -- prior waves accepted pc_len 68/72/84 on "
              "three of four models, and 112 holds ~3%% of MACs but ~10%% of "
              "the cycle budget at target 32 -- so starting there wastes "
              "sweeps rediscovering it. Freed budget is reprojected into the "
              "ladders at seed time."))
    p.add_argument(
        "--ladder-groups", default="global", choices=list(GROUP_MODES),
        help=("ladder granularity. 'global' (default) = one ladder shared by "
              "all 36 (op x layer) buckets, the historical behaviour and a "
              "byte-identical control. 'op' = separate ladders for score ops "
              "(qk/av), attention linears and MLP. 'layer' = early/middle/"
              "late. 'op:layer' = both (9 ladders). Measured occupancy shows "
              "MLP never touches the top 2 rungs while qk never touches the "
              "bottom 3, so a shared ladder gives each about half its "
              "resolution."))
    p.add_argument(
        "--exclude-partition-checkpoint", type=Path, default=None,
        help=("checkpoint or partition JSON whose screen+confirm blocks are "
              "excluded from this run's screen, confirmation, and holdout; "
              "use when chaining a winner selected on an earlier run"))
    p.add_argument("--profile-bins", type=int, default=257)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction,
                   default=True)
    return p


def _append_history(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _cache_ppl(cache: dict[int, float]) -> float:
    return math.exp(_mean(cache.values()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_fingerprint(
    args,
    *,
    target: float,
    root_wrapper: Path,
    root_table_path: Path,
    partition: dict,
) -> dict:
    """Identity of every input that changes candidate or accept semantics."""
    config = {
        "target": float(target),
        "model_path": str(args.model_path),
        "ctx": int(args.ctx),
        "partition": partition,
        "ladder_groups": args.ladder_groups,
        "advance_total": int(args.advance_total),
        "family_quotas": str(args.family_quotas),
        "confirm_top_k": int(args.confirm_top_k),
        "threshold_phase": bool(args.threshold_phase),
        "pc_lengths": str(args.pc_lengths),
        "structured_mass_steps": str(args.structured_mass_steps),
        "surrogate_proposals": int(args.surrogate_proposals),
        "surrogate_min_history": int(args.surrogate_min_history),
        "surrogate_min_transfer": float(args.surrogate_min_transfer),
        "threshold_mass_steps": str(args.threshold_mass_steps),
        "split_fractions": str(args.split_fractions),
        "min_classes": int(args.min_classes),
        "max_classes": int(args.max_classes),
        "min_level": int(args.min_level),
        "max_level": int(args.max_level),
        "allow_removal": bool(args.allow_removal),
        "budget_tol_under": float(args.budget_tol_under),
        "budget_tol_over": float(args.budget_tol_over),
        "accept_floor": float(args.accept_floor),
        "confirm_se_mult": float(args.confirm_se_mult),
        "confirm_veto": bool(args.confirm_veto),
        "sigma_fallback": float(args.sigma_fallback),
        "seed_protected_sl": args.seed_protected_sl,
        "ppl_window_batch_size": int(
            os.environ.get("PPL_WINDOW_BATCH_SIZE", "1")),
        "ppl_window_batch_identity": (
            str(Path(os.environ["PPL_WINDOW_BATCH_IDENTITY_JSON"]).resolve())
            if os.environ.get("PPL_WINDOW_BATCH_IDENTITY_JSON") else None),
        "ppl_window_batch_identity_sha256": (
            _sha256_file(Path(os.environ["PPL_WINDOW_BATCH_IDENTITY_JSON"]))
            if os.environ.get("PPL_WINDOW_BATCH_IDENTITY_JSON") else None),
        "parent_wrapper": str(root_wrapper),
        "parent_wrapper_sha256": _sha256_file(root_wrapper),
        "parent_table": str(root_table_path),
        "parent_table_sha256": _sha256_file(root_table_path),
        "source_sha256": _sha256_file(Path(__file__)),
        "group_source_sha256": _sha256_file(
            PROJECT_ROOT / "benchmark/ppl/mp_group_ladders.py"),
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "scmp-v20-run-fingerprint-v1",
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "config": config,
    }


def _bonferroni_se_mult(base: float, tests: int) -> float:
    """One-sided Bonferroni correction; exactly identity for one test."""
    if tests <= 1:
        return float(base)
    normal = NormalDist()
    alpha = normal.cdf(-float(base))
    return float(normal.inv_cdf(1.0 - alpha / int(tests)))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_level > 128:
        raise SystemExit("sc_prec=8 caps stream length at 128")
    if args.patience < 1 or args.max_sweeps < 0:
        raise SystemExit("patience must be >=1 and max-sweeps >=0")
    if args.confirm_top_k < 1:
        raise SystemExit("confirm-top-k must be >=1")
    if args.budget_tol_under <= 0 or args.budget_tol_over <= 0:
        raise SystemExit("budget tolerances must be positive")
    threshold_steps = _parse_float_list(args.threshold_mass_steps)
    structured_steps = _parse_float_list(args.structured_mass_steps)
    split_fractions = _parse_float_list(args.split_fractions)
    quotas = _parse_quotas(args.family_quotas)
    pc_lengths = [int(float(tok)) for tok in
                  args.pc_lengths.replace(":", ",").split(",") if tok.strip()]

    root_wrapper = args.parent_wrapper.resolve()
    wrapper, root_table_path, root_table = resolve_parent_table(root_wrapper)
    parent_levels = [int(x) for x in wrapper["stoc_len_levels"]]
    protected_sl, protected_hint = _protected_info(root_table)
    # Snapshot: --seed-protected-sl re-pins per branch, and each target must
    # start from the parent value rather than inherit the previous branch's.
    protected_sl_parent = protected_sl
    command = " ".join(sys.argv)
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    clock = WalltimeClock(args.walltime_hours, slack=args.guard_slack)

    model, tokenizer = _build_parent_model(
        args.model_path, root_wrapper, args.alpha)
    # Layer count, for mapping a layer-quartile bucket onto an early/middle/
    # late band.  Taken from the model config so band boundaries match the
    # runtime's own _bucket_index arithmetic rather than a guess.
    total_blocks_hint = int(
        getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    # Bands are a COARSENING of the table's layer buckets, never an
    # independent split of the layer index -- a bucket is the finest unit that
    # can carry its own ladder.
    layer_buckets_hint = int(root_table.get("layer_buckets") or 4)
    if args.ladder_groups != "global" and total_blocks_hint <= 0:
        raise ValueError(
            "--ladder-groups needs the model's layer count to place layer "
            "bands; config.num_hidden_layers was not readable")
    enc_val = _load_eval_stream(
        tokenizer, split="validation", max_tokens=0, ctx=args.ctx)
    n_windows = enc_val.numel() // args.ctx
    excluded_blocks: set[int] = set()
    excluded_partition_sha256 = None
    if args.exclude_partition_checkpoint is not None:
        source = args.exclude_partition_checkpoint.resolve()
        with source.open("rb") as f:
            raw_partition_source = f.read()
        excluded_partition_sha256 = hashlib.sha256(
            raw_partition_source).hexdigest()
        payload = json.loads(raw_partition_source)
        old_partition = payload.get("partition", payload)
        if (int(old_partition.get("n_windows", -1)) != n_windows
                or int(old_partition.get("block", -1))
                != args.block_windows):
            raise SystemExit(
                f"excluded partition {source} does not match this stream "
                f"(n_windows={n_windows}, block={args.block_windows})")
        excluded_blocks = {
            int(x) for x in (
                list(old_partition.get("screen_block_ids") or [])
                + list(old_partition.get("confirm_block_ids") or []))
        }
        if not excluded_blocks:
            raise SystemExit(
                f"excluded partition {source} has no screen/confirm blocks")
    partition = build_window_partition(
        n_windows, block=args.block_windows,
        excluded_block_ids=excluded_blocks)
    if excluded_partition_sha256 is not None:
        partition["excluded_partition_source"] = str(
            args.exclude_partition_checkpoint.resolve())
        partition["excluded_partition_sha256"] = excluded_partition_sha256
    print(
        f"[V16 WINDOWS] n={n_windows} screen_blocks="
        f"{partition['screen_block_ids']} confirm_blocks="
        f"{partition['confirm_block_ids']} "
        f"holdout={len(partition['holdout_windows'])}", flush=True)
    enc_test = None
    if args.test_tokens >= 0:
        enc_test = _load_eval_stream(
            tokenizer, split="test", max_tokens=args.test_tokens, ctx=args.ctx)
        print(f"[V16 TEST STREAM] tokens={enc_test.numel()}", flush=True)

    fp16_ref = None
    if args.fp16_ref is not None and args.fp16_ref.exists():
        with args.fp16_ref.open() as f:
            candidate_ref = json.load(f)
        if (int(candidate_ref.get("n_windows", -1)) == n_windows
                and int(candidate_ref.get("ctx", -1)) == args.ctx):
            fp16_ref = candidate_ref
        else:
            print(
                f"[V16 WARN] fp16 reference {args.fp16_ref} does not match "
                f"this stream (windows/ctx); tier-1 stop disabled", flush=True)
    fp16_confirm_ppl = None
    if fp16_ref is not None:
        fp16_confirm_ppl = math.exp(_mean(
            float(fp16_ref["window_nll"][str(w)])
            for w in partition["confirm_windows"]))
        print(
            f"[V16 FP16 REF] full_val={fp16_ref['full_ppl']:.4f} "
            f"confirm_set={fp16_confirm_ppl:.4f}", flush=True)

    def occupancy_of(result: dict, levels, psl: int | None = None) -> dict:
        """Realized occupancy. Polymorphic on the ladder state.

        ``levels`` is a plain list in global mode and a {group: ladder} dict
        in group mode, so every existing call site works unchanged and the
        global path stays exactly as it was.
        """
        if isinstance(levels, dict):
            occ = aggregate_trace_weights_grouped(
                _load_trace(Path(result["trace"])), levels,
                mode=args.ladder_groups,
                total_blocks=total_blocks_hint,
                layer_buckets=layer_buckets_hint,
                protected_stoc_len=(psl if psl else protected_sl),
                protected_weight_hint=protected_hint)
            result["trace_flop_avg_stoc_len"] = float(occ["cost"])
            # expose a flat view so cost-only consumers need no changes
            occ["level_weights"] = [
                w for key in sorted(occ["group_level_weights"])
                for w in occ["group_level_weights"][key]]
            return occ
        return _actual_occupancy(
            result, levels,
            protected_sl=(psl if psl else protected_sl),
            protected_hint=protected_hint)

    def in_band(cost: float, target: float) -> bool:
        return (target - args.budget_tol_under
                <= cost <= target + args.budget_tol_over)

    parent_probe: dict | None = None
    summaries = []
    targets: list[float] | None = None

    def deploy(branch: Path, stem: str, table: dict, levels):
        """Write table + wrapper. Polymorphic on the ladder state.

        In group mode the per-bucket ladders are stamped into the table (the
        runtime reads bucket["stoc_len_levels"] and falls back to the
        table-level list when absent), and the wrapper keeps the parent
        ladder purely to satisfy AdaptiveMPConfig's global-list validation --
        no bucket resolves to it once every bucket declares its own.
        """
        table_path = branch / f"{stem}.json"
        wrapper_path = branch / f"{stem}_wrapper.json"
        if isinstance(levels, dict):
            # The table being deployed is authoritative for its protected
            # length.  The outer ``protected_sl`` is only branch state and
            # may legitimately differ here: a resumed branch can carry a
            # re-pinned incumbent, and a pc_length proposal changes the pin
            # before it is accepted.  Consulting the captured value caused a
            # legal rung equal to the *parent* pin to kill resumed searches.
            deployed_protected_sl, _ = _protected_info(table)
            assert_no_pin_collision(levels, deployed_protected_sl)
            table = apply_group_ladders_to_table(
                table, levels,
                mode=args.ladder_groups,
                total_blocks=total_blocks_hint,
                layer_buckets=layer_buckets_hint)
            wrapper_levels = parent_levels
        else:
            wrapper_levels = levels
        _atomic_json(table_path, table)
        _write_wrapper(wrapper_path, wrapper_levels, table_path)
        return table_path, wrapper_path

    # Parent occupancy probe on subset 0 establishes the target list.
    probe_branch = outdir / "baseline"
    probe_branch.mkdir(parents=True, exist_ok=True)
    # PER-JOB probe filenames.  $OUTDIR is shared by every target of a model,
    # so concurrent single-target jobs all wrote the SAME probe trace and one
    # job would json.load it while another was mid-write -- observed as
    # JSONDecodeError at ~1 MiB, exactly a partially flushed file.  The trace
    # is ~1-28 MB and is not written atomically, so uniqueness is the fix.
    _probe_tag = re.sub(r"[^0-9A-Za-z]+", "_", str(args.targets)).strip("_")
    probe_trace = probe_branch / f"parent_probe_{_probe_tag}_trace.json"
    probe_profile = probe_branch / f"parent_probe_{_probe_tag}_profile.json"
    subset0 = partition["subsets"][0]
    parent_probe = eval_windows(
        model, enc_val, subset0["A"] + subset0["B"],
        ctx=args.ctx,
        levels=parent_levels,
        table_path=root_table_path,
        wrapper_path=root_wrapper,
        trace_path=probe_trace,
        split_label="validation_screen",
        profile_path=probe_profile,
        profile_bins=args.profile_bins,
    )
    clock.note(parent_probe["seconds"], 8)
    parent_occupancy = occupancy_of(parent_probe, parent_levels)
    parent_cost = float(parent_occupancy["cost"])
    targets = _parse_targets(
        args.targets, parent_cost=parent_cost,
        nominal_cost=nominal_target_cost(root_table))
    print(
        f"[V16 BASELINE] levels={parent_levels} cost={parent_cost:.4f} "
        f"subset0_ppl={parent_probe['ppl']:.6f} targets={targets}", flush=True)

    for target in targets:
        # Branch-local state must never leak from a previous target.  A fresh
        # branch starts from the parent pin; a resumed branch replaces this
        # below with the checkpoint/table value.
        protected_sl = protected_sl_parent
        branch = outdir / _target_slug(target)
        branch.mkdir(parents=True, exist_ok=True)
        summary_path = branch / "summary.json"
        history_path = branch / "history.jsonl"
        checkpoint_path = branch / "checkpoint.json"
        manifest_path = branch / "run_manifest.json"
        run_fingerprint = _run_fingerprint(
            args, target=target, root_wrapper=root_wrapper,
            root_table_path=root_table_path, partition=partition)
        if manifest_path.exists():
            with manifest_path.open() as f:
                existing_manifest = json.load(f)
            if existing_manifest.get("sha256") != run_fingerprint["sha256"]:
                raise SystemExit(
                    f"run fingerprint mismatch in {branch}; refusing to "
                    "reuse artifacts produced by different code, parent, "
                    "partition, or search settings")
        else:
            material = [summary_path, history_path, checkpoint_path]
            if any(p.exists() for p in material):
                raise SystemExit(
                    f"legacy artifacts without run fingerprint in {branch}; "
                    "use a new output directory")
            _atomic_json(manifest_path, run_fingerprint)
        if args.resume and summary_path.exists():
            with summary_path.open() as f:
                completed = json.load(f)
            # A walltime_guard summary is NOT a converged branch: the
            # auto-resubmit helper relaunches the same tag and the branch
            # must continue from its checkpoint instead of being skipped.
            walltime_stopped = (
                completed.get("stop_reason") == "walltime_guard")
            if (completed.get("status") == "budget_projection_failed"
                    or (completed.get("status") == "ok"
                        and not walltime_stopped)):
                summaries.append(completed)
                print(f"[V16 RESUME] target={target} already complete",
                      flush=True)
                continue
            if walltime_stopped:
                print(
                    f"[V16 RESUME] target={target} summary stopped on "
                    "walltime_guard; continuing from checkpoint", flush=True)

        # ------------------------------------------------------------------
        # incumbent state (fresh or from checkpoint)
        # ------------------------------------------------------------------
        state = None
        if args.resume and checkpoint_path.exists():
            with checkpoint_path.open() as f:
                state = json.load(f)
            if state.get("run_fingerprint") != run_fingerprint["sha256"]:
                raise SystemExit(
                    f"checkpoint run fingerprint mismatch in {branch}; "
                    "refusing incompatible resume")
            if state.get("partition") != partition:
                raise SystemExit(
                    f"checkpoint window partition mismatch in {branch}; "
                    "refusing to resume with different windows")
            incumbent_state = state["incumbent"]
            with Path(incumbent_state["table"]).open() as f:
                resumed_table = json.load(f)
            table_protected_sl, _ = _protected_info(resumed_table)
            checkpoint_protected_sl = incumbent_state.get("protected_sl")
            if (checkpoint_protected_sl is not None
                    and table_protected_sl is not None
                    and int(checkpoint_protected_sl)
                    != int(table_protected_sl)):
                raise SystemExit(
                    f"checkpoint protected pin mismatch in {branch}: state "
                    f"has {checkpoint_protected_sl}, table has "
                    f"{table_protected_sl}")
            protected_sl = (
                int(checkpoint_protected_sl)
                if checkpoint_protected_sl is not None
                else table_protected_sl)
            probe_window = partition["confirm_windows"][0]
            probe = eval_windows(
                model, enc_val, [probe_window],
                ctx=args.ctx,
                levels=_ladder_state(state["incumbent"]["levels"]),
                table_path=Path(state["incumbent"]["table"]),
                wrapper_path=Path(state["incumbent"]["wrapper"]),
                trace_path=branch / "resume_probe_trace.json",
                split_label="resume_probe",
            )
            recorded = float(
                state["incumbent"]["confirm_cache"][str(probe_window)])
            observed = float(probe["window_nll"][probe_window])
            if abs(observed - recorded) > 1e-6:
                raise SystemExit(
                    f"resume integrity check failed in {branch}: window "
                    f"{probe_window} nll {observed!r} != checkpointed "
                    f"{recorded!r}")
            if state.get("stop_reason") == "walltime_guard":
                state["stop_reason"] = None  # a resumed branch keeps going
            print(
                f"[V16 RESUME] target={target} from sweep "
                f"{state['sweep_completed']} (integrity ok)", flush=True)

        if state is None:
            # -------- establish an in-band incumbent --------------------
            current_levels = parent_levels.copy()
            current_table = copy.deepcopy(root_table)
            protected_sl = protected_sl_parent
            if args.seed_protected_sl and protected_sl:
                # Re-pin BEFORE the budget projection below, so the existing
                # (tested) projector spends the freed pin budget on the
                # ladder and the incumbent starts in band.  V9's 112 is
                # measurably over-provisioned: prior waves accepted pc_len
                # 68/72/84 on three of four models, and the pins hold ~3% of
                # MACs but ~10% of the cycle budget at target 32, so starting
                # at 112 spends sweeps rediscovering that.
                protected_sl = int(args.seed_protected_sl)
                current_table.setdefault("protected_channels", {})
                current_table["protected_channels"]["stoc_len"] = protected_sl
                current_table_path, current_wrapper_path = deploy(
                    branch, "seed_psl", current_table, current_levels)
                seed_probe = eval_windows(
                    model, enc_val, subset0["A"] + subset0["B"],
                    ctx=args.ctx,
                    levels=current_levels,
                    table_path=current_table_path,
                    wrapper_path=current_wrapper_path,
                    trace_path=branch / "seed_psl_trace.json",
                    split_label="validation_screen",
                    profile_path=branch / "seed_psl_profile.json",
                    profile_bins=args.profile_bins,
                )
                clock.note(seed_probe["seconds"], 8)
                parent_occupancy_seeded = occupancy_of(
                    seed_probe, current_levels, protected_sl)
                print(
                    f"[V19 SEED PSL] target={target:.2f} "
                    f"{protected_sl_parent} -> {protected_sl} "
                    f"cost {parent_occupancy['cost']:.4f} -> "
                    f"{parent_occupancy_seeded['cost']:.4f}", flush=True)
            if not (args.seed_protected_sl
                    and protected_sl != protected_sl_parent):
                # keep the re-pinned seed deployment when psl was changed
                current_table_path = root_table_path
                current_wrapper_path = root_wrapper
            current_occupancy = (
                parent_occupancy_seeded if args.seed_protected_sl
                and protected_sl != protected_sl_parent else parent_occupancy)
            current_profile_path = (
                branch / "seed_psl_profile.json"
                if args.seed_protected_sl
                and protected_sl != protected_sl_parent else probe_profile)
            projection_history = []
            for correction in range(args.budget_corrections + 1):
                if in_band(current_occupancy["cost"], target):
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
                levels = [int(x) for x in projection["levels"]]
                if levels == current_levels:
                    break
                action = {
                    "type": "initial_budget_projection",
                    "predicted_cost": projection["predicted_cost"],
                    "projection": _short_action(projection),
                }
                table = make_v16_table(
                    current_table, levels,
                    root_parent_table=root_table_path,
                    target_cost=target,
                    action=action,
                    command=command,
                )
                stem = f"budget{correction:02d}_" + "-".join(map(str, levels))
                table_path, wrapper_path = deploy(branch, stem, table, levels)
                result = eval_windows(
                    model, enc_val, subset0["A"] + subset0["B"],
                    ctx=args.ctx,
                    levels=levels,
                    table_path=table_path,
                    wrapper_path=wrapper_path,
                    trace_path=branch / f"{stem}_trace.json",
                    split_label="validation_screen",
                    profile_path=branch / f"{stem}_profile.json",
                    profile_bins=args.profile_bins,
                )
                clock.note(result["seconds"], 8)
                occupancy = occupancy_of(result, levels)
                projection_history.append({
                    "stage": "initial_budget_projection",
                    "cost": occupancy["cost"],
                    "levels": levels,
                })
                closer = (abs(occupancy["cost"] - target)
                          < abs(current_occupancy["cost"] - target))
                if not closer:
                    break
                current_levels = levels
                current_table = table
                current_table_path = table_path
                current_wrapper_path = wrapper_path
                current_occupancy = occupancy
                current_profile_path = branch / f"{stem}_profile.json"
            if not in_band(current_occupancy["cost"], target):
                summary = {
                    "schema": V16_SCHEMA,
                    "status": "budget_projection_failed",
                    "target": target,
                    "parent_levels": parent_levels,
                    "parent_cost": parent_cost,
                    "attempts": projection_history,
                    "command": command,
                }
                _atomic_json(summary_path, summary)
                summaries.append(summary)
                continue

            if args.ladder_groups != "global":
                # Split the in-band flat ladder into per-group ladders, all
                # seeded IDENTICAL, so this deployment is the same
                # configuration expressed per group -- the groups only start
                # diverging once a measured move is accepted.
                current_levels = group_state_from_levels(
                    current_levels, args.ladder_groups)
                assert_no_pin_collision(current_levels, protected_sl)
                current_table_path, current_wrapper_path = deploy(
                    branch, "group00_seed", current_table, current_levels)
                seed_eval = eval_windows(
                    model, enc_val, subset0["A"] + subset0["B"],
                    ctx=args.ctx,
                    levels=current_levels,
                    table_path=current_table_path,
                    wrapper_path=current_wrapper_path,
                    trace_path=branch / "group00_seed_trace.json",
                    split_label="validation_screen",
                    profile_path=branch / "group00_seed_profile.json",
                    profile_bins=args.profile_bins,
                )
                clock.note(seed_eval["seconds"], 8)
                current_occupancy = occupancy_of(seed_eval, current_levels)
                current_profile_path = branch / "group00_seed_profile.json"
                print(
                    f"[V19 GROUP SEED] target={target:.2f} "
                    f"mode={args.ladder_groups} "
                    f"groups={len(current_levels)} "
                    f"cost={current_occupancy['cost']:.4f} "
                    f"shares={ {k: round(v, 4) for k, v in current_occupancy['group_shares'].items()} }",
                    flush=True)

            confirm_result = eval_windows(
                model, enc_val, partition["confirm_windows"],
                ctx=args.ctx,
                levels=current_levels,
                table_path=current_table_path,
                wrapper_path=current_wrapper_path,
                trace_path=branch / "incumbent_confirm_trace.json",
                split_label="validation_confirm",
            )
            clock.note(confirm_result["seconds"],
                       len(partition["confirm_windows"]))
            confirm_occ = occupancy_of(confirm_result, current_levels)
            state = {
                "schema": V16_SCHEMA,
                "run_fingerprint": run_fingerprint["sha256"],
                "sweep_completed": 0,
                "stalled": 0,
                "stop_reason": None,
                "accepted_moves": [],
                "prev_pair_moves": [],
                "tier2": {"armed": True, "last_check_nll": None, "checks": []},
                "partition": partition,
                "incumbent": {
                    "levels": current_levels,
                    "protected_sl": protected_sl,
                    "table": str(current_table_path),
                    "wrapper": str(current_wrapper_path),
                    "profile": str(current_profile_path),
                    "confirm_cache": {
                        str(w): v
                        for w, v in confirm_result["window_nll"].items()},
                    "confirm_cost": float(confirm_occ["cost"]),
                    "confirm_ppl": float(confirm_result["ppl"]),
                },
                "full_test": None,
            }
            _atomic_json(checkpoint_path, state)

        inc = state["incumbent"]
        sigma_w = float(args.sigma_fallback)

        # ------------------------------------------------------------------
        # sweeps
        # ------------------------------------------------------------------
        sweep_open = False
        for sweep in range(int(state["sweep_completed"]) + 1,
                           args.max_sweeps + 1):
            if sweep_open:            # the previous iteration completed
                clock.end_sweep()
                sweep_open = False
            n_cand_est = float(args.guard_cand_est)
            wrapup_windows = (
                (enc_test.numel() // args.ctx if enc_test is not None else 0)
                + len(partition["confirm_windows"])
                + (len(partition["holdout_windows"]) if args.holdout else 0))
            # A-priori guess only until a sweep has actually been measured.
            fallback_windows = (n_cand_est + args.advance_total + 6) * 4 + 16
            sweep_windows = clock.sweep_windows(fallback_windows)
            need = (clock.estimate(sweep_windows)
                    + clock.estimate(wrapup_windows))
            if clock.remaining() < need:
                state["stop_reason"] = state["stop_reason"] or "walltime_guard"
                print(
                    f"[V16 WALLTIME] target={target} stopping before sweep "
                    f"{sweep}: {clock.remaining()/3600:.2f}h left, "
                    f"need {need/3600:.2f}h (sweep {sweep_windows:.0f}w"
                    f"{'' if clock.sweep_windows_seen else ' est'}"
                    f" + wrapup {wrapup_windows:.0f}w, "
                    f"slack {clock.slack:g})", flush=True)
                break
            clock.begin_sweep()
            sweep_open = True
            subset = partition["subsets"][(sweep - 1)
                                          % len(partition["subsets"])]
            inc_levels = _ladder_state(inc["levels"])
            inc_psl = inc.get("protected_sl") or protected_sl
            inc_eval = eval_windows(
                model, enc_val, subset["A"] + subset["B"],
                ctx=args.ctx,
                levels=_runtime_levels(inc_levels, Path(inc["table"])),
                table_path=Path(inc["table"]),
                wrapper_path=Path(inc["wrapper"]),
                trace_path=branch / f"s{sweep:02d}_incumbent_trace.json",
                split_label="validation_screen",
                profile_path=branch / f"s{sweep:02d}_incumbent_profile.json",
                profile_bins=args.profile_bins,
            )
            clock.note(inc_eval["seconds"], 8)
            inc_occ = occupancy_of(inc_eval, inc_levels, inc_psl)
            with (branch / f"s{sweep:02d}_incumbent_profile.json").open() as f:
                inc_profile = json.load(f)
            with Path(inc["table"]).open() as f:
                inc_table = json.load(f)

            accepted_this_sweep = False
            next_pair_moves: list[tuple[int, int]] = []
            phases = ("precision",) + (
                ("threshold",) if args.threshold_phase else ())
            for phase in phases:
                if phase == "threshold" and accepted_this_sweep:
                    continue
                if args.ladder_groups != "global":
                    proposals = build_group_proposals(
                        inc_table, inc_occ, inc_levels,
                        target=target,
                        table_levels=parent_levels,
                        root_parent_table=root_table_path,
                        command=command,
                        protected_sl=inc_psl or protected_sl,
                        min_level=args.min_level,
                        max_level=args.max_level,
                        pc_lengths=pc_lengths,
                    )
                    print(
                        f"[V19 GROUP] target={target:.2f} sweep={sweep} "
                        f"mode={args.ladder_groups} "
                        f"groups={len(inc_levels)} "
                        f"candidates={len(proposals)}", flush=True)
                else:
                    proposals = generate_v16_candidates(
                        inc_table, inc_profile, inc_occ,
                        target=target,
                        root_parent_table=root_table_path,
                        command=command,
                        phase=phase,
                        threshold_mass_steps=threshold_steps,
                        split_fractions=split_fractions,
                        min_classes=args.min_classes,
                        max_classes=args.max_classes,
                        min_level=args.min_level,
                        max_level=args.max_level,
                        allow_removal=args.allow_removal,
                        structured_mass_steps=structured_steps,
                        prev_pair_moves=state.get("prev_pair_moves") or [],
                        pc_lengths=pc_lengths,
                    )
                # The surrogate's features are flat-ladder shape descriptors
                # (floor, ceiling, gaps); they carry no meaning once each
                # group has its own ladder, so it is off in group mode.
                if (phase == "precision" and args.surrogate_proposals > 0
                        and args.ladder_groups == "global"):
                    history_records = []
                    if history_path.exists():
                        with history_path.open() as f:
                            history_records = [
                                json.loads(line) for line in f if line.strip()]
                    surrogate = fit_surrogate(
                        history_records,
                        target=target,
                        min_history=args.surrogate_min_history,
                        min_transfer=args.surrogate_min_transfer,
                    )
                    _append_history(history_path, {
                        "sweep": sweep, "phase": phase,
                        "stage": "surrogate_fit",
                        "ok": surrogate["ok"], "r": surrogate.get("r"),
                        "n_a": surrogate["n_a"], "n_b": surrogate["n_b"],
                    })
                    if surrogate["ok"]:
                        seen = {tuple(int(x) for x in r["levels"])
                                for r in history_records
                                if r.get("stage") == "A" and r.get("levels")}
                        seen.add(tuple(inc_levels))
                        seen |= {tuple(p["levels"]) for p in proposals}
                        extra = surrogate_candidates(
                            inc_table, inc_occ, surrogate,
                            target=target,
                            root_parent_table=root_table_path,
                            command=command,
                            seen_levels=seen,
                            k=args.surrogate_proposals,
                            min_level=args.min_level,
                            max_level=args.max_level,
                        )
                        proposals.extend(extra)
                        print(
                            f"[V16.1 SURROGATE] target={target:.2f} "
                            f"sweep={sweep} r={surrogate['r']:+.3f} "
                            f"proposals={len(extra)}", flush=True)
                    else:
                        print(
                            f"[V16.1 SURROGATE] target={target:.2f} "
                            f"sweep={sweep} inactive "
                            f"(n_a={surrogate['n_a']} n_b={surrogate['n_b']} "
                            f"r={surrogate.get('r')})", flush=True)
                n_cand_est = 0.7 * n_cand_est + 0.3 * len(proposals)
                print(
                    f"[V16 SWEEP] target={target:.2f} sweep={sweep} "
                    f"phase={phase} candidates={len(proposals)}", flush=True)
                stage_a = []
                delta_vectors = []
                for idx, proposal in enumerate(proposals):
                    stem = (f"s{sweep:02d}_{phase}_{idx:03d}_"
                            f"{proposal['name']}")
                    table_path, wrapper_path = deploy(
                        branch, stem, proposal["table"], proposal["levels"])
                    result = eval_windows(
                        model, enc_val, subset["A"],
                        ctx=args.ctx,
                        levels=proposal["levels"],
                        table_path=table_path,
                        wrapper_path=wrapper_path,
                        trace_path=branch / f"{stem}_traceA.json",
                        split_label="validation_screen_A",
                    )
                    clock.note(result["seconds"], len(subset["A"]))
                    cand_psl = proposal.get("protected_sl") or inc_psl
                    occ = occupancy_of(result, proposal["levels"], cand_psl)
                    deltas_a = paired_deltas(
                        result["window_nll"], inc_eval["window_nll"],
                        subset["A"])
                    delta_vectors.append(deltas_a)
                    eligible = in_band(occ["cost"], target)
                    record = {
                        "sweep": sweep, "phase": phase, "stage": "A",
                        "candidate": proposal["name"],
                        "family": proposal["family"],
                        "levels": proposal["levels"],
                        "protected_sl": proposal.get("protected_sl"),
                        "cost": occ["cost"],
                        "eligible": eligible,
                        "mean_delta_a": _mean(deltas_a),
                    }
                    _append_history(history_path, record)
                    if eligible:
                        stage_a.append({
                            "proposal": proposal,
                            "stem": stem,
                            "table_path": table_path,
                            "wrapper_path": wrapper_path,
                            "deltas_a": deltas_a,
                            "cost_a": occ["cost"],
                            "name": proposal["name"],
                            "family": proposal["family"],
                        })
                sigma_w = pooled_sigma_w(
                    delta_vectors, fallback=args.sigma_fallback)
                ranked = sorted(stage_a, key=lambda x: _mean(x["deltas_a"]))
                advance = select_advance(
                    ranked, quotas=quotas, total=args.advance_total)
                stage_a_best = advance[0] if advance else None
                # Predeclare the only candidates that may reach C in A-rank
                # order.  B can veto them but cannot promote a candidate from
                # outside this set, so selection and replication stay split.
                confirm_tested = advance[:args.confirm_top_k]
                for item in advance:
                    result_b = eval_windows(
                        model, enc_val, subset["B"],
                        ctx=args.ctx,
                        levels=item["proposal"]["levels"],
                        table_path=item["table_path"],
                        wrapper_path=item["wrapper_path"],
                        trace_path=branch / f"{item['stem']}_traceB.json",
                        split_label="validation_screen_B",
                    )
                    clock.note(result_b["seconds"], len(subset["B"]))
                    deltas_b = paired_deltas(
                        result_b["window_nll"], inc_eval["window_nll"],
                        subset["B"])
                    gate = replication_accept_gate(
                        item["deltas_a"], deltas_b,
                        sigma_w=sigma_w,
                        accept_floor=args.accept_floor,
                        se_mult=args.confirm_se_mult,
                    )
                    item.update({"deltas_b": deltas_b, "gate": gate})
                    _append_history(history_path, {
                        "sweep": sweep, "phase": phase, "stage": "B",
                        "candidate": item["name"],
                        "family": item["family"],
                        "mean_delta_b": gate["mean_b"],
                        "mean_delta_ab": gate["mean_ab"],
                        "sign_replicates": gate["sign_replicates"],
                        "passes": gate["accept"],
                        "stage_a_winner": item is stage_a_best,
                        "confirm_eligible": item in confirm_tested,
                        "sigma_w": sigma_w,
                    })
                replicating = [it for it in advance if it["gate"]["accept"]]
                for item in sorted(
                        replicating, key=lambda x: x["gate"]["mean_ab"])[:2]:
                    action = item["proposal"]["action"]
                    if (item["family"] == "value_move"
                            and "index" in action):
                        next_pair_moves.append(
                            (int(action["index"]),
                             int(action["requested_level"])))
                if stage_a_best is None:
                    continue
                print(
                    f"[V20 A->B] target={target:.2f} sweep={sweep} "
                    f"phase={phase} A_best={stage_a_best['name']} "
                    f"mean_b={stage_a_best['gate']['mean_b']:+.6f} "
                    f"mean_ab={stage_a_best['gate']['mean_ab']:+.6f} "
                    f"thr={stage_a_best['gate']['threshold']:+.6f} "
                    f"replicates={stage_a_best['gate']['accept']} "
                    f"confirm_top_k={len(confirm_tested)}", flush=True)

                confirm_pool = [
                    item for item in confirm_tested
                    if item["gate"]["accept"]
                ]
                adjusted_mult = _bonferroni_se_mult(
                    args.confirm_se_mult, len(confirm_tested))
                winner = None
                winner_psl = None
                confirm_eval = None
                confirm_occ = None
                deltas_c = None
                veto = None
                for confirm_rank, candidate in enumerate(confirm_pool, 1):
                    candidate_psl = (
                        candidate["proposal"].get("protected_sl") or inc_psl)
                    candidate_confirm = eval_windows(
                        model, enc_val, partition["confirm_windows"],
                        ctx=args.ctx,
                        levels=candidate["proposal"]["levels"],
                        table_path=candidate["table_path"],
                        wrapper_path=candidate["wrapper_path"],
                        trace_path=(branch
                                    / f"{candidate['stem']}_confirm.json"),
                        split_label="validation_confirm",
                    )
                    clock.note(candidate_confirm["seconds"],
                               len(partition["confirm_windows"]))
                    candidate_occ = occupancy_of(
                        candidate_confirm,
                        candidate["proposal"]["levels"], candidate_psl)
                    candidate_deltas_c = paired_deltas(
                        candidate_confirm["window_nll"],
                        {int(k): v for k, v in inc["confirm_cache"].items()},
                        partition["confirm_windows"])
                    candidate_veto = confirm_veto_gate(
                        candidate["deltas_a"], candidate["deltas_b"],
                        candidate_deltas_c,
                        sigma_w=sigma_w,
                        accept_floor=args.accept_floor,
                        se_mult=adjusted_mult,
                    )
                    cost_ok = in_band(candidate_occ["cost"], target)
                    keep = bool(cost_ok and (
                        candidate_veto["keep"] or not args.confirm_veto))
                    _append_history(history_path, {
                        "sweep": sweep, "phase": phase,
                        "stage": "confirm_veto",
                        "candidate": candidate["name"],
                        "family": candidate["family"],
                        "levels": candidate["proposal"]["levels"],
                        "confirm_rank": confirm_rank,
                        "confirm_tests_planned": len(confirm_tested),
                        "base_se_mult": args.confirm_se_mult,
                        "adjusted_se_mult": adjusted_mult,
                        "confirm_ppl": candidate_confirm["ppl"],
                        "confirm_cost": candidate_occ["cost"],
                        "confirm_cost_in_band": cost_ok,
                        "veto": candidate_veto,
                        "accepted": keep,
                    })
                    if not keep:
                        print(
                            f"[V20 CONFIRM-VETO] target={target:.2f} "
                            f"sweep={sweep} phase={phase} "
                            f"candidate={candidate['name']} "
                            f"mean_abc={candidate_veto['mean_abc']:+.6f} "
                            f"thr={candidate_veto['threshold']:+.6f} "
                            f"cost_ok={cost_ok} REVERTED", flush=True)
                        continue
                    winner = candidate
                    winner_psl = candidate_psl
                    confirm_eval = candidate_confirm
                    confirm_occ = candidate_occ
                    deltas_c = candidate_deltas_c
                    veto = candidate_veto
                    break

                if winner is None:
                    _append_history(history_path, {
                        "sweep": sweep, "phase": phase,
                        "stage": "accept_decision",
                        "candidate": stage_a_best["name"],
                        "family": stage_a_best["family"],
                        "levels": stage_a_best["proposal"]["levels"],
                        "gate": stage_a_best["gate"],
                        "accepted": False,
                        "reason": ("no_A_ranked_candidate_replicated"
                                   if not confirm_pool
                                   else "all_confirm_candidates_vetoed"),
                    })
                    continue

                # STOP-only diagnostics run only for the candidate that has
                # cleared A, B, the multiplicity-adjusted C veto, and cost.
                stop_check = confirm_stop_check(
                    deltas_c,
                    sigma_w=sigma_w,
                    se_mult=args.confirm_se_mult,
                    regress_window_nats=args.regress_window_nats,
                    regress_window_max=args.regress_window_max,
                )
                _append_history(history_path, {
                    "sweep": sweep, "phase": phase, "stage": "confirm_stop",
                    "candidate": winner["name"],
                    "family": winner["family"],
                    "levels": winner["proposal"]["levels"],
                    "confirm_ppl": confirm_eval["ppl"],
                    "confirm_cost": confirm_occ["cost"],
                    "confirm_cost_in_band": True,
                    "stop_check": stop_check,
                })
                _append_history(history_path, {
                    "sweep": sweep, "phase": phase,
                    "stage": "accept_decision",
                    "candidate": winner["name"],
                    "family": winner["family"],
                    "levels": winner["proposal"]["levels"],
                    "gate": winner["gate"],
                    "confirm_veto": veto,
                    "accepted": True,
                })
                print(
                    f"[V20 ACCEPT] target={target:.2f} sweep={sweep} "
                    f"phase={phase} candidate={winner['name']} "
                    f"mean_abc={veto['mean_abc']:+.6f} "
                    f"thr={veto['threshold']:+.6f} "
                    f"confirm_ppl={confirm_eval['ppl']:.6f}", flush=True)
                accepted_this_sweep = True
                state["accepted_moves"].append({
                    "sweep": sweep,
                    "phase": phase,
                    "name": winner["name"],
                    "family": winner["family"],
                    "levels": winner["proposal"]["levels"],
                    "protected_sl": winner_psl,
                    "ab_mean_delta": winner["gate"]["mean_ab"],
                    "ab_threshold": winner["gate"]["threshold"],
                    "confirm_mean_delta": stop_check["mean"],
                    "confirm_ppl": confirm_eval["ppl"],
                    "confirm_cost": confirm_occ["cost"],
                })
                inc = state["incumbent"] = {
                    "levels": winner["proposal"]["levels"],
                    "protected_sl": winner_psl,
                    "table": str(winner["table_path"]),
                    "wrapper": str(winner["wrapper_path"]),
                    "profile": None,
                    "confirm_cache": {
                        str(w): v
                        for w, v in confirm_eval["window_nll"].items()},
                    "confirm_cost": float(confirm_occ["cost"]),
                    "confirm_ppl": float(confirm_eval["ppl"]),
                }
                inc_psl = winner_psl
                if stop_check["stop"]:
                    state["stop_reason"] = (state["stop_reason"]
                                            or "confirm_regression")
                    break

            state["prev_pair_moves"] = [list(m) for m in next_pair_moves]
            state["sweep_completed"] = sweep
            state["stalled"] = 0 if accepted_this_sweep else \
                int(state["stalled"]) + 1
            if int(state["stalled"]) >= args.patience:
                state["stop_reason"] = state["stop_reason"] or "patience"
            _atomic_json(checkpoint_path, state)

            # ------- two-tier FP16 stop check (stop-only, never selects) ---
            if (accepted_this_sweep and fp16_confirm_ppl is not None
                    and args.fp16_test_ppl is not None
                    and enc_test is not None):
                tier1 = (args.tier1_ratio_scale * args.ppl_fp16_ratio
                         * fp16_confirm_ppl)
                conf_nll = math.log(float(inc["confirm_ppl"]))
                tier2 = state["tier2"]
                if float(inc["confirm_ppl"]) <= tier1:
                    rearm_gain = 2.0 * sigma_w / 4.0
                    if (not tier2["armed"]
                            and tier2["last_check_nll"] is not None
                            and conf_nll
                            < tier2["last_check_nll"] - rearm_gain):
                        tier2["armed"] = True
                    if tier2["armed"]:
                        test_eval = _eval_once(
                            model, enc_test,
                            ctx=args.ctx,
                            levels=_runtime_levels(inc["levels"], Path(inc["table"])),
                            table_path=Path(inc["table"]),
                            wrapper_path=Path(inc["wrapper"]),
                            trace_path=branch
                            / f"s{sweep:02d}_tier2_test_trace.json",
                            split="test",
                        )
                        test_occ = occupancy_of(
                            test_eval, _ladder_state(inc["levels"]),
                            inc.get("protected_sl") or protected_sl)
                        check = {
                            "sweep": sweep,
                            "test_ppl": test_eval["ppl"],
                            "test_cost": test_occ["cost"],
                            "threshold": (args.ppl_fp16_ratio
                                          * args.fp16_test_ppl),
                        }
                        tier2["checks"].append(check)
                        tier2["armed"] = False
                        tier2["last_check_nll"] = conf_nll
                        state["full_test"] = {
                            "table": inc["table"],
                            "ppl": test_eval["ppl"],
                            "nll": test_eval["nll"],
                            "tokens": test_eval["tokens"],
                            "cost": test_occ["cost"],
                            "trace": test_eval["trace"],
                        }
                        _atomic_json(checkpoint_path, state)
                        print(
                            f"[V16 TIER2] target={target:.2f} "
                            f"test_ppl={test_eval['ppl']:.6f} "
                            f"threshold={check['threshold']:.6f}", flush=True)
                        if test_eval["ppl"] <= check["threshold"]:
                            state["stop_reason"] = \
                                "full_test_within_fp16_ratio"
                            _atomic_json(checkpoint_path, state)
                            break

            if state["stop_reason"]:
                break
        else:
            if args.max_sweeps > 0 and state["stop_reason"] is None:
                state["stop_reason"] = "max_sweeps"

        # ------------------------------------------------------------------
        # wrap-up: frozen winner -> best wrapper, holdout, full test
        # ------------------------------------------------------------------
        inc = state["incumbent"]
        inc_levels = _ladder_state(inc["levels"])
        final_psl = inc.get("protected_sl") or protected_sl
        best_wrapper = branch / "best_wrapper.json"
        # In group mode the per-bucket ladders already live in the table the
        # incumbent points at; the wrapper's global list is only the fallback
        # AdaptiveMPConfig validates, so it carries the parent ladder.
        _write_wrapper(
            best_wrapper,
            parent_levels if isinstance(inc_levels, dict) else inc_levels,
            Path(inc["table"]))
        holdout_result = None
        if args.holdout and partition["holdout_windows"]:
            holdout_result = eval_windows(
                model, enc_val, partition["holdout_windows"],
                ctx=args.ctx,
                levels=_runtime_levels(inc_levels, Path(inc["table"])),
                table_path=Path(inc["table"]),
                wrapper_path=best_wrapper,
                trace_path=branch / "holdout_trace.json",
                split_label="validation_holdout",
            )
            clock.note(holdout_result["seconds"],
                       len(partition["holdout_windows"]))
        final_test = None
        final_test_cost = None
        if enc_test is not None:
            cached = state.get("full_test")
            if cached and cached.get("table") == inc["table"]:
                final_test = cached
                final_test_cost = cached["cost"]
            else:
                test_eval = _eval_once(
                    model, enc_test,
                    ctx=args.ctx,
                    levels=_runtime_levels(inc_levels, Path(inc["table"])),
                    table_path=Path(inc["table"]),
                    wrapper_path=best_wrapper,
                    trace_path=branch / "final_test_trace.json",
                    split="test",
                )
                test_occ = occupancy_of(test_eval, inc_levels, final_psl)
                final_test = {
                    "table": inc["table"],
                    "ppl": test_eval["ppl"],
                    "nll": test_eval["nll"],
                    "tokens": test_eval["tokens"],
                    "cost": test_occ["cost"],
                    "trace": test_eval["trace"],
                }
                final_test_cost = test_occ["cost"]

        fp16_holdout_ppl = None
        if fp16_ref is not None and holdout_result is not None:
            fp16_holdout_ppl = math.exp(_mean(
                float(fp16_ref["window_nll"][str(w)])
                for w in partition["holdout_windows"]))
        summary = {
            "schema": V16_SCHEMA,
            "search_version": "v16.1",
            "threshold_phase": bool(args.threshold_phase),
            "pc_lengths": pc_lengths,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "ok",
            "target": target,
            "model": args.model_path,
            "root_parent_wrapper": str(root_wrapper),
            "root_parent_table": str(root_table_path),
            "parent_levels": parent_levels,
            "parent_cost": parent_cost,
            "best_levels": inc_levels,
            "best_class_count": len(inc_levels),
            "best_protected_stoc_len": final_psl,
            "best_table": inc["table"],
            "best_wrapper": str(best_wrapper),
            "confirm_ppl": inc["confirm_ppl"],
            "confirm_cost": inc["confirm_cost"],
            "holdout_ppl": (None if holdout_result is None
                            else holdout_result["ppl"]),
            "holdout_windows": len(partition["holdout_windows"]),
            "final_test_ppl": None if final_test is None
            else final_test["ppl"],
            "final_test_cost": final_test_cost,
            "final_test_tokens": (None if final_test is None
                                  else final_test["tokens"]),
            "final_test_trace": (None if final_test is None
                                 else final_test["trace"]),
            "fp16_ref": (None if fp16_ref is None
                         else str(args.fp16_ref)),
            "fp16_confirm_ppl": fp16_confirm_ppl,
            "fp16_holdout_ppl": fp16_holdout_ppl,
            "fp16_test_ppl": args.fp16_test_ppl,
            "ppl_fp16_ratio": args.ppl_fp16_ratio,
            "tier2_checks": state["tier2"]["checks"],
            "accepted_moves": state["accepted_moves"],
            "sweeps_run": state["sweep_completed"],
            "stop_reason": state["stop_reason"],
            "sigma_w": sigma_w,
            "partition": {
                "n_windows": n_windows,
                "screen_block_ids": partition["screen_block_ids"],
                "confirm_block_ids": partition["confirm_block_ids"],
                "holdout_windows": len(partition["holdout_windows"]),
            },
            "history": str(history_path),
            "command": command,
        }
        _atomic_json(summary_path, summary)
        summaries.append(summary)
        print(
            f"[V16 TARGET RESULT] target={target:.2f} levels={inc_levels} "
            f"accepted={len(state['accepted_moves'])} "
            f"confirm_ppl={inc['confirm_ppl']:.6f}@{inc['confirm_cost']:.4f} "
            f"final_test_ppl="
            f"{(final_test['ppl'] if final_test else math.nan):.6f} "
            f"stop={state['stop_reason']}", flush=True)

    _atomic_json(outdir / "summary.json", {
        "schema": V16_SCHEMA + "-wave",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model_path,
        "targets": targets,
        "branches": summaries,
        "command": command,
    })
    with (outdir / "results.tsv").open("w") as f:
        f.write("target\tstatus\tlevels\taccepted\tconfirm_ppl\tconfirm_cost\t"
                "holdout_ppl\tfinal_test_ppl\tfinal_test_cost\tstop_reason\t"
                "best_wrapper\n")
        for s in summaries:
            f.write(
                f"{s['target']:.6f}\t{s['status']}\t"
                f"{','.join(map(str, s.get('best_levels', [])))}\t"
                f"{len(s.get('accepted_moves', []))}\t"
                f"{s.get('confirm_ppl', math.nan):.8f}\t"
                f"{s.get('confirm_cost', math.nan):.6f}\t"
                f"{(s.get('holdout_ppl') or math.nan):.8f}\t"
                f"{(s.get('final_test_ppl') or math.nan):.8f}\t"
                f"{(s.get('final_test_cost') or math.nan):.6f}\t"
                f"{s.get('stop_reason')}\t{s.get('best_wrapper', '')}\n")
    print(f"[V16 DONE] summary={outdir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
