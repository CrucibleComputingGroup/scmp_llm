"""Per-(operator-group x layer-band) ladders.

WHY
---
The calibrated table has always carried 36 buckets (9 ops x 4 layer quartiles)
whose THRESHOLDS are fitted per bucket, but a single LADDER -- one list of rung
values -- shared by all of them.  Measured rung occupancy says that wastes
roughly half the ladder.  On 14B t32, ladder [97, 64, 49, 32, 24, 20]:

    qk:l1        [0.970, 0.029, 0.0,   0.0,   0.0,  0.0 ]   top 2 rungs only
    down_proj:l1 [0.0,   0.0,   0.008, 0.982, 0.009, 0.0 ]  never touches 97/64
    MLP mean     [0.0,   0.0,   ~0.02, 0.28,  0.33,  0.40]

The score ops sit at the top of the ladder and the MLP sits at the bottom, so
each population gets about half the available resolution.  Splitting the ladder
by group gives each its own rungs at the same rung count.

Layer bands matter too: 14B MLP puts 0.619 of its MAC on the cheapest rung in
the early band but only 0.282 in the next one, and the only knob that could
express that today (threshold moves) measured ZERO signal -- 0/165 accepts.
That is the tell that the binding constraint is rung VALUES, not mass split.

WHAT IS AND IS NOT DECIDED HERE
-------------------------------
Budget projection holds each group's share of the cost budget FIXED (see
project_group_ladders_to_budget).  So the initial per-group state reproduces
today's allocation exactly, and any budget moved between attention and MLP, or
between early and late layers, is an explicit proposal that has to earn its
acceptance -- never a side effect of the solver.
"""

from __future__ import annotations

from typing import Iterable, Optional

from benchmark.ppl.mp_joint_refine import project_levels_to_budget
from benchmark.ppl.mp_ladder_refine import weighted_cost

GROUP_MODES = ("global", "op", "layer", "op:layer")

# Operator groups, cut where the measured occupancy actually separates:
# score ops pin to the ceiling, MLP lives on the floor, and the attention
# linears spread across the middle.
SCORE_OPS = ("qk", "av")
ATTN_LINEAR_OPS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_OPS = ("gate_proj", "up_proj", "down_proj")
OP_GROUP_OF = {
    **{op: "score" for op in SCORE_OPS},
    **{op: "attnlin" for op in ATTN_LINEAR_OPS},
    **{op: "mlp" for op in MLP_OPS},
}

LAYER_BANDS = ("early", "middle", "late")


DEFAULT_LAYER_BUCKETS = 4


def band_of_bucket(l_bucket: int, layer_buckets: int) -> str:
    """Coarsen a layer BUCKET index onto early/middle/late.

    Bands must be a coarsening of the table's layer buckets, never an
    independent partition of the layer index.  A bucket is the finest unit
    that can carry its own ladder (the table stores one ladder per
    (op, t_bucket, l_bucket)), so if a band boundary fell inside a bucket,
    that bucket's blocks would be attributed to two different ladders while
    only one ladder could be written -- realized cost then fails to
    reconcile.  With 4 buckets and 3 bands the split is l0 | l1,l2 | l3.
    """
    if layer_buckets <= 1:
        return LAYER_BANDS[0]
    if l_bucket <= 0:
        return LAYER_BANDS[0]
    if l_bucket >= layer_buckets - 1:
        return LAYER_BANDS[-1]
    return LAYER_BANDS[1]


def layer_band(block_idx: int, total_blocks: int,
               layer_buckets: int = DEFAULT_LAYER_BUCKETS) -> str:
    """Map an absolute layer index onto early/middle/late, via its bucket.

    The block -> bucket step uses the same floor(idx * n / total) arithmetic
    as the runtime's _bucket_index, so a band boundary can never fall in a
    different place from the table's own layer bucketing.
    """
    if total_blocks <= 1:
        return LAYER_BANDS[0]
    l_bucket = int(block_idx) * int(layer_buckets) // int(total_blocks)
    return band_of_bucket(l_bucket, int(layer_buckets))


def group_key(op: str, block_idx: int, total_blocks: int,
              mode: str = "op:layer",
              layer_buckets: int = DEFAULT_LAYER_BUCKETS) -> str:
    """Ladder-group identity for one (operator, layer)."""
    if mode not in GROUP_MODES:
        raise ValueError(f"unknown ladder-group mode {mode!r}; "
                         f"valid: {', '.join(GROUP_MODES)}")
    if mode == "global":
        return "global"
    og = OP_GROUP_OF.get(op)
    if og is None:
        raise ValueError(
            f"operator {op!r} has no ladder group; add it to OP_GROUP_OF "
            "rather than letting it silently share the global ladder")
    if mode == "op":
        return og
    band = layer_band(block_idx, total_blocks, layer_buckets)
    if mode == "layer":
        return band
    return f"{og}:{band}"


def group_keys_for_mode(mode: str) -> list[str]:
    """Every group key a mode can produce, in a stable order."""
    if mode == "global":
        return ["global"]
    ogs = ["score", "attnlin", "mlp"]
    if mode == "op":
        return ogs
    if mode == "layer":
        return list(LAYER_BANDS)
    return [f"{og}:{band}" for og in ogs for band in LAYER_BANDS]


def aggregate_trace_weights_grouped(
    trace: dict,
    group_ladders: dict[str, list[int]],
    *,
    mode: str,
    total_blocks: Optional[int] = None,
    layer_buckets: int = DEFAULT_LAYER_BUCKETS,
    protected_stoc_len: Optional[int] = None,
    protected_weight_hint: float = 0.0,
) -> dict:
    """Per-group MAC occupancy and the (ladder-independent) realized cost.

    The single-ladder aggregate_trace_weights RAISES on any stream length
    outside its one ladder, so it cannot read a grouped trace at all -- with
    per-group ladders the valid set is the union over groups, and the same
    length may be a different rung index in two groups.

    Realized cost does NOT depend on any ladder: it is sum(stoc_len * MAC
    share) over whatever actually ran, plus the protected pins.  Ladders are
    needed only to attribute each group's shares to rung indices, which is what
    the proposal machinery consumes.
    """
    groups = trace.get("groups") or []
    if total_blocks is None:
        total_blocks = max((int(g.get("block", 0)) for g in groups),
                           default=0) + 1

    psl = None if protected_stoc_len is None else int(protected_stoc_len)
    total = 0.0
    # group key -> {stoc_len: macs}; protected work is split out globally
    by_group: dict[str, dict[int, float]] = {}
    protected_macs = 0.0
    cost_macs = 0.0

    for g in groups:
        macs = float(g.get("macs", 0.0))
        if macs <= 0.0:
            continue
        sl = int(g["stoc_len"])
        total += macs
        cost_macs += sl * macs
        key = group_key(str(g["op"]), int(g.get("block", 0)),
                        int(total_blocks), mode, int(layer_buckets))
        ladder = group_ladders.get(key)
        if ladder is None:
            raise ValueError(
                f"trace group {g.get('op')}@block{g.get('block')} maps to "
                f"ladder group {key!r}, which has no ladder")
        if psl is not None and sl == psl and sl not in ladder:
            # protected pins sit outside every adaptive ladder (the V9 design);
            # they are priced globally, not as a rung of any group
            protected_macs += macs
            continue
        by_group.setdefault(key, {})
        by_group[key][sl] = by_group[key].get(sl, 0.0) + macs

    if total <= 0.0:
        raise ValueError("trace contains no positive SC MAC counts")

    out_weights: dict[str, list[float]] = {}
    unexpected: dict[str, dict[int, float]] = {}
    for key, ladder in group_ladders.items():
        shares = {sl: m / total for sl, m in by_group.get(key, {}).items()}
        out_weights[key] = [shares.pop(sl, 0.0) for sl in ladder]
        left = {sl: w for sl, w in shares.items() if w > 1e-9}
        if left:
            unexpected[key] = left
    if unexpected:
        raise ValueError(
            "trace contains SC lengths outside the adaptive/protected set: "
            + "; ".join(
                f"{k}: " + ", ".join(f"{sl}:{w:.6f}"
                                     for sl, w in sorted(v.items()))
                for k, v in sorted(unexpected.items())))

    protected_weight = protected_macs / total
    if psl is not None and protected_weight == 0.0 and protected_weight_hint:
        # Legacy tables where the pin length COLLIDES with an adaptive rung:
        # the pins' MACs are already counted in that rung, so the hint has to
        # be carved OUT of it, never added on top -- adding would break the
        # sum-to-1 invariant and silently inflate realized cost.  If the pin
        # length is not a rung anywhere, the trace genuinely contains no
        # protected work and the weight is 0.
        hint = max(float(protected_weight_hint), 0.0)
        for key, ladder in group_ladders.items():
            if psl in ladder:
                idx = ladder.index(psl)
                take = min(hint, out_weights[key][idx])
                out_weights[key][idx] -= take
                protected_weight += take
                break

    observed = sum(sum(v) for v in out_weights.values()) + protected_weight
    if abs(observed - 1.0) > 1e-6:
        raise ValueError(
            f"trace MAC shares sum to {observed:.9f}, expected 1")

    return {
        "group_level_weights": out_weights,
        "protected_weight": protected_weight,
        # ladder-independent: sum(stoc_len * share) over everything that ran
        "cost": cost_macs / total,
        "total_macs": total,
        "group_shares": {k: sum(v) for k, v in out_weights.items()},
    }


def group_cost_contributions(
    group_ladders: dict[str, list[int]],
    group_level_weights: dict[str, list[float]],
) -> dict[str, float]:
    """Each group's contribution to the global MAC-weighted cost."""
    return {
        key: weighted_cost(group_ladders[key], group_level_weights[key])
        for key in group_ladders
    }


def project_group_ladders_to_budget(
    group_ladders: dict[str, list[int]],
    group_level_weights: dict[str, list[float]],
    *,
    target_cost: float,
    protected_weight: float = 0.0,
    protected_stoc_len: int = 0,
    min_gap: int = 1,
    min_level: int = 1,
    max_level: int = 128,
) -> dict:
    """Project every group's ladder, holding each group's COST SHARE fixed.

    The adaptive budget is target_cost minus the protected pins.  Each group
    keeps its current fraction of that budget and its ladder is projected to
    the resulting per-group target with the existing, tested single-ladder
    projector.  Consequences, both deliberate:

      * with one group this is exactly the old behaviour, so `global` mode is
        a true regression control;
      * the solver NEVER moves budget between attention and MLP or between
        layer bands on its own -- that has to come from a proposal that is
        measured and accepted like any other move.
    """
    contributions = group_cost_contributions(
        group_ladders, group_level_weights)
    adaptive_now = sum(contributions.values())
    adaptive_target = float(target_cost) - (
        float(protected_weight) * float(protected_stoc_len))
    if adaptive_now <= 0.0:
        raise ValueError("no adaptive MAC mass to project")
    if adaptive_target <= 0.0:
        raise ValueError(
            f"target_cost {target_cost} is fully consumed by protected pins "
            f"({protected_weight:.6f} x {protected_stoc_len})")
    scale = adaptive_target / adaptive_now

    out_ladders: dict[str, list[int]] = {}
    per_group: dict[str, dict] = {}
    for key, ladder in group_ladders.items():
        weights = group_level_weights[key]
        share = sum(weights)
        if share <= 0.0:
            # group carries no MAC mass this trace: leave its ladder alone
            out_ladders[key] = [int(x) for x in ladder]
            per_group[key] = {"skipped": "no MAC mass"}
            continue
        # per-unit-MAC cost inside the group, so the projector sees a
        # normalized problem identical in shape to the single-ladder case
        cond = [w / share for w in weights]
        res = project_levels_to_budget(
            ladder, cond,
            target_cost=(contributions[key] / share) * scale,
            protected_weight=0.0,
            protected_stoc_len=0,
            min_gap=min_gap,
            min_level=min_level,
            max_level=max_level,
        )
        out_ladders[key] = [int(x) for x in res["levels"]]
        per_group[key] = res

    projected = weighted_cost(
        [1], [0.0], protected_weight, protected_stoc_len) + sum(
            weighted_cost(out_ladders[k], group_level_weights[k])
            for k in out_ladders)
    return {
        "group_ladders": out_ladders,
        "cost": projected,
        "scale": scale,
        "per_group": per_group,
    }


def assert_no_pin_collision(
    group_ladders: dict[str, list[int]],
    protected_stoc_len: Optional[int],
) -> None:
    """Refuse a ladder set whose rungs collide with the protected pins.

    aggregate_trace_weights cannot separate protected work that lands on an
    adaptive rung -- that is why LIFT_PSL is 95 and never 96.  With up to nine
    ladders in flight the collision odds rise sharply, and a collision
    corrupts the realized COST silently, which then corrupts every accept
    decision because candidates are filtered on realized cost.  Fail loudly
    at construction instead.
    """
    if protected_stoc_len is None:
        return
    psl = int(protected_stoc_len)
    hits = sorted(k for k, ladder in group_ladders.items() if psl in ladder)
    if hits:
        raise ValueError(
            f"protected stoc_len {psl} collides with a rung in ladder "
            f"group(s) {', '.join(hits)}; realized cost could not be "
            "attributed. Move the pin length or the rung.")


def apply_group_ladders_to_table(
    table: dict,
    group_ladders: dict[str, list[int]],
    *,
    mode: str,
    total_blocks: int,
    layer_buckets: int,
) -> dict:
    """Write per-bucket stoc_len_levels into a table's buckets.

    The runtime reads bucket["stoc_len_levels"] when present and falls back to
    the table-level ladder when absent (AdaptiveMPConfig.get_levels), so a
    table this function never touched stays on the byte-identical path.
    """
    import copy
    out = copy.deepcopy(table)
    if mode == "global":
        return out
    buckets = out.get("buckets") or {}
    for bucket_key, payload in buckets.items():
        op, _t_bucket, l_bucket = bucket_key.split(":")[0], None, None
        parts = bucket_key.split(":")
        if len(parts) != 3:
            continue
        op = parts[0]
        l_bucket = int(parts[2].lstrip("l"))
        if mode == "global":
            continue
        og = OP_GROUP_OF.get(op)
        if og is None:
            continue
        band = band_of_bucket(l_bucket, layer_buckets)
        key = (og if mode == "op"
               else band if mode == "layer" else f"{og}:{band}")
        ladder = group_ladders.get(key)
        if ladder is None:
            continue
        payload["stoc_len_levels"] = [int(x) for x in ladder]
    return out
