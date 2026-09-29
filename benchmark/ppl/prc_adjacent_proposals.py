"""Adjacent-rung, cost-matched allocation proposals; no execution changes.

Every proposal edits one boundary in each of two distinct buckets. Neighbor
constraints preserve monotonicity and bound the rung displacement for *every*
float32 normalized metric, including exact threshold ties. Histogram replay
bounds both the number of groups changed and their MACs/cycles. Candidate
trajectories must be profiled and audited again by the evaluator.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict

import numpy as np

from benchmark.ppl.prc_local_proposals import (
    LINEARS, MLP, PROJECTIONS, ProfileCollector as _OriginalCollector,
    _entry, _group_cost, _lengths, baseline_cost, exact_profile_cost,
)


class AdjacentProfileCollector(_OriginalCollector):
    """Observe actual group counts and MACs, with both tables' threshold edges."""

    def __init__(self, model, bins=8192, reference_table=None):
        super().__init__(model, bins=bins)
        self.reference_table = copy.deepcopy(reference_table)

    def _group(self, cfg, op, block, total, kind, levels, thresholds, device):
        import torch
        group = super()._group(cfg, op, block, total, kind, levels, thresholds, device)
        if "count_hist" not in group:
            if self.reference_table is not None:
                old = _entry(self.reference_table, group)["thresholds"]
                edges = np.unique(np.r_[group["edges"], old].astype(np.float32))
                group["edges"] = edges
                group["reps"] = (np.r_[edges, np.float32(1)] if kind == "prc"
                                 else np.r_[np.float32(0), edges])
                group["edges_gpu"] = torch.as_tensor(edges, device=device)
                group["hist"] = torch.zeros(len(edges) + 1, dtype=torch.float64, device=device)
            group["count_hist"] = torch.zeros_like(group["hist"])
        return group

    def _add(self, group, metric, weights, actual_lengths):
        import torch
        super()._add(group, metric, weights, actual_lengths)
        index = torch.bucketize(metric.reshape(-1).contiguous(), group["edges_gpu"],
                                right=group["kind"] == "attention")
        group["count_hist"].add_(torch.bincount(index, minlength=group["hist"].numel()))

    def snapshot(self):
        result = super().snapshot()
        result["version"] = 2
        result["count_source"] = "actual adjustable dispatched groups; escaped attention excluded"
        for tag, group in result["groups"].items():
            internal = self._groups[tag]
            keep = internal["hist"].detach().cpu().numpy() > 0
            count = internal["count_hist"].detach().cpu().numpy()[keep]
            group["count"] = count.astype(float).tolist()
            group["total_groups"] = float(count.sum())
        return result


ProfileCollector = AdjacentProfileCollector


def _indices(group, thresholds, metric=None):
    metric = np.asarray(group["metric"] if metric is None else metric, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if group["kind"] == "prc":
        return np.searchsorted(thresholds, metric, side="left")
    index = np.full(metric.shape, len(group["levels"]) - 1, dtype=np.int64)
    for i in range(len(thresholds) - 1, -1, -1):
        index[metric >= thresholds[i]] = i
    return index


def _universal_displacement(group, before, after):
    """Check every threshold equality and every intervening float32 interval."""
    edges = np.unique(np.r_[0., 1., before, after].astype(np.float32))
    probes = np.unique(np.clip(np.r_[edges,
        np.nextafter(edges, np.float32(-np.inf)),
        np.nextafter(edges, np.float32(np.inf))], 0, 1))
    return int(np.max(np.abs(_indices(group, before, probes) -
                                 _indices(group, after, probes))))


def _arrays(group):
    metric = np.asarray(group["metric"], dtype=np.float32)
    mac = np.asarray(group["mac"], dtype=np.float64)
    count = np.asarray(group.get("count", []), dtype=np.float64)
    if (metric.shape != mac.shape or metric.shape != count.shape or metric.ndim != 1
            or not metric.size or np.any(metric[1:] < metric[:-1])
            or not np.all(np.isfinite(metric)) or not np.all(np.isfinite(mac))
            or not np.all(np.isfinite(count)) or np.any(mac <= 0) or np.any(count <= 0)):
        raise ValueError(f"invalid count/MAC histogram: {group['key']}")
    return metric, mac, count


def _limits(operator, bucket, groups, mac):
    values = (operator, bucket, groups, mac)
    if not all(np.isfinite(v) and 0 < v <= 1 for v in values):
        raise ValueError("trust-region fractions must be finite and in (0,1]")


def audit_transition(base_table, candidate_table, profile,
                     max_operator_cost_fraction=.02, max_bucket_cost_fraction=.05,
                     max_changed_group_fraction=.05, max_changed_mac_fraction=.05):
    """Audit both tables on the SAME inputs, including a candidate trajectory.

    Operator caps bound gross moved cycles, so opposing changes cannot cancel
    the cap. Costs include fixed SC work in the total but use adjustable cycles
    as the conservative denominator for per-bucket/per-operator caps.
    """
    _limits(max_operator_cost_fraction, max_bucket_cost_fraction,
            max_changed_group_fraction, max_changed_mac_fraction)
    reasons, edits = [], []
    normalized = copy.deepcopy(candidate_table)
    for section, allowed in (("per_row_chunk", LINEARS), (None, {"qk", "av"})):
        before = (base_table.get(section, {}) if section else base_table).get("buckets", {})
        after = (candidate_table.get(section, {}) if section else candidate_table).get("buckets", {})
        normalized_buckets = (normalized.get(section, {}) if section else normalized).get("buckets", {})
        if before.keys() != after.keys():
            reasons.append("bucket keys changed")
            continue
        for key, entry in before.items():
            if key.split(":")[0] not in allowed:
                continue
            a, b = entry.get("thresholds"), after[key].get("thresholds")
            if a == b:
                continue
            if (not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b)
                    or not all(isinstance(v, (int, float)) and np.isfinite(v) and 0 <= v <= 1 for v in b)):
                reasons.append(f"invalid thresholds: {key}")
                continue
            if b != sorted(b, reverse=section is None):
                reasons.append(f"nonmonotone thresholds: {key}")
            changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            if len(changed) != 1:
                reasons.append(f"not exactly one boundary: {key}")
            kind = "prc" if section else "attention"
            tag = kind + "|" + key
            if tag not in profile["groups"]:
                reasons.append(f"edited bucket was not observed: {tag}")
            edits.append((tag, changed))
            normalized_buckets[key]["thresholds"] = copy.deepcopy(a)
    if normalized != base_table:
        reasons.append("fields other than allowed thresholds changed")
    if base_table != candidate_table and len(edits) != 2:
        reasons.append("expected two distinct edited buckets")
    if reasons:
        return dict(feasible=False, reasons=reasons, edited_buckets=len(edits))

    base_cycles = new_cycles = float(profile["fixed_cycle_macs"])
    op_base, op_net, op_gross = defaultdict(float), defaultdict(float), defaultdict(float)
    details, max_rung = [], 0
    for tag, group in profile["groups"].items():
        metric, mac, count = _arrays(group)
        old = _entry(base_table, group)["thresholds"]
        new = _entry(candidate_table, group)["thresholds"]
        # Exact CDF replay requires all comparison boundaries to be histogram edges.
        edges = np.asarray(group["edges"], dtype=np.float32)
        if not np.all(np.isin(np.asarray(old + new, np.float32), edges)):
            reasons.append(f"profile lacks comparison threshold edges: {tag}")
        old_length, new_length = _lengths(group, old), _lengths(group, new)
        cost0, cost1 = float(np.dot(mac, old_length)), float(np.dot(mac, new_length))
        base_cycles += cost0
        new_cycles += cost1
        op = group["op"]
        op_base[op] += cost0
        op_net[op] += cost1 - cost0
        op_gross[op] += abs(cost1 - cost0)
        if old == new:
            continue
        rung_delta = _universal_displacement(group, old, new)
        max_rung = max(max_rung, rung_delta)
        changed = _indices(group, old) != _indices(group, new)
        fraction_count = float(count[changed].sum() / count.sum())
        fraction_mac = float(mac[changed].sum() / mac.sum())
        fraction_cost = abs(cost1 - cost0) / cost0
        boundary = next(i for i, (x, y) in enumerate(zip(old, new)) if x != y)
        item = dict(tag=tag, kind=group["kind"], key=group["key"], op=op,
                    boundary=boundary, old_threshold=old[boundary], new_threshold=new[boundary],
                    adjacent_levels=[group["levels"][boundary], group["levels"][boundary + 1]],
                    cycle_change=cost1 - cost0, base_cycle_macs=cost0,
                    changed_groups=float(count[changed].sum()), total_groups=float(count.sum()),
                    changed_group_fraction=fraction_count, changed_mac_fraction=fraction_mac,
                    bucket_cost_fraction=fraction_cost, max_rung_displacement=rung_delta)
        details.append(item)
        if rung_delta > 1:
            reasons.append(f"multi-rung displacement: {tag}")
        for value, limit, label in ((fraction_count, max_changed_group_fraction, "group fraction"),
                                   (fraction_mac, max_changed_mac_fraction, "MAC fraction"),
                                   (fraction_cost, max_bucket_cost_fraction, "bucket cycles")):
            if value > limit + 1e-12:
                reasons.append(f"{label} exceeds cap: {tag}")
        if not np.any(changed):
            reasons.append(f"threshold edit changes no observed groups: {tag}")
    operators = {op: dict(base_cycle_macs=cost, net_cost_fraction=op_net[op] / cost,
                         gross_cost_fraction=op_gross[op] / cost)
                 for op, cost in op_base.items() if cost > 0}
    for op, item in operators.items():
        if item["gross_cost_fraction"] > max_operator_cost_fraction + 1e-12:
            reasons.append(f"operator gross cycles exceed cap: {op}")
    positive = sum(max(d["cycle_change"], 0) for d in details)
    negative = sum(max(-d["cycle_change"], 0) for d in details)
    if edits and (positive <= 0 or negative <= 0):
        reasons.append("edits must include a donor and a recipient")
    return dict(feasible=not reasons, reasons=reasons, edited_buckets=len(edits),
                max_rung_displacement=max_rung, moves=details, operators=operators,
                base_cycle_macs=base_cycles, candidate_cycle_macs=new_cycles,
                base_cost=base_cycles / profile["total_macs"],
                candidate_cost=new_cycles / profile["total_macs"],
                predicted_cost_ratio=new_cycles / base_cycles,
                transferred_fraction=min(positive, negative) / base_cycles,
                transfer_mismatch_fraction=abs(positive - negative) / base_cycles)


def _curves(group, total_cycles, max_transfer, op_cost, caps, rejected):
    """All bounded single-boundary moves, as compact monotone CDF curves."""
    metric, mac, count = _arrays(group)
    thresholds = np.asarray(group["thresholds"], dtype=np.float32)
    edges = np.asarray(group["edges"], dtype=np.float32)
    levels = np.asarray(group["levels"], dtype=np.float64)
    if len(thresholds) != len(levels) - 1:
        raise ValueError("threshold/ladder mismatch")
    if ((group["kind"] == "prc" and np.any(np.diff(levels) <= 0)) or
            (group["kind"] == "attention" and np.any(np.diff(levels) >= 0))):
        raise ValueError("strictly monotone deployed ladders required")
    side = "right" if group["kind"] == "prc" else "left"
    positions = np.searchsorted(metric, edges, side=side)
    prefix_mac, prefix_count = np.r_[0., np.cumsum(mac)], np.r_[0., np.cumsum(count)]
    cm, cn = prefix_mac[positions], prefix_count[positions]
    base_cost = _group_cost(group, thresholds)
    cycle_cap = min(max_transfer * total_cycles, caps[0] * op_cost, caps[1] * base_cost)
    curves = []
    for boundary, old in enumerate(thresholds):
        if group["kind"] == "prc":
            low = thresholds[boundary - 1] if boundary else 0.
            high = thresholds[boundary + 1] if boundary + 1 < len(thresholds) else 1.
            factor = levels[boundary] - levels[boundary + 1]
        else:
            low = thresholds[boundary + 1] if boundary + 1 < len(thresholds) else 0.
            high = thresholds[boundary - 1] if boundary else 1.
            factor = levels[boundary + 1] - levels[boundary]
        old_pos = np.searchsorted(metric, old, side=side)
        moved_mac = np.abs(cm - prefix_mac[old_pos])
        moved_count = np.abs(cn - prefix_count[old_pos])
        delta = (cm - prefix_mac[old_pos]) * factor
        valid = ((edges >= low) & (edges <= high) & (edges != old)
                 & (np.abs(delta) <= cycle_cap * (1 + 1e-12))
                 & (moved_mac <= caps[3] * mac.sum() * (1 + 1e-12))
                 & (moved_count <= caps[2] * count.sum() * (1 + 1e-12)))
        for promote in (True, False):
            eligible = np.flatnonzero(valid & ((delta > 0) if promote else (delta < 0)))
            if not eligible.size:
                rejected["boundary_direction_no_bounded_capacity"] += 1
                continue
            cost = np.abs(delta[eligible])
            order = np.lexsort((np.abs(edges[eligible] - old), cost))
            eligible, cost = eligible[order], cost[order]
            # Flat CDF ranges: use the closest threshold for the same transferred cost.
            unique = np.r_[True, cost[1:] != cost[:-1]]
            curves.append(dict(group=group, boundary=boundary, promote=promote,
                               thresholds=edges[eligible[unique]], costs=cost[unique]))
    return curves


def _category(recipient, donor):
    r, d = recipient["group"]["op"], donor["group"]["op"]
    if r in ("qk", "av") and d in LINEARS:
        return r + "_from_linears"
    if d in ("qk", "av") and r in LINEARS:
        return "linears_from_" + d
    if {r, d} == {"qk", "av"}:
        return r + "_from_" + d
    if r in PROJECTIONS and d in MLP:
        return "projections_from_mlp"
    if r in MLP and d in PROJECTIONS:
        return "mlp_from_projections"
    if r == d:
        return "within_" + r
    return "within_projections" if r in PROJECTIONS else "within_mlp"


def _match(recipient, donor, limit, total):
    r = recipient["costs"]
    d = donor["costs"]
    nr, nd = np.searchsorted(r, limit, side="right"), np.searchsorted(d, limit, side="right")
    if not nr or not nd:
        return None
    r, d = r[:nr], d[:nd]
    insertion = np.searchsorted(d, r)
    best = None
    for offset in (-1, 0):
        j = np.clip(insertion + offset, 0, nd - 1)
        transferred = np.minimum(r, d[j])
        error = np.abs(r - d[j])
        ok = ((error <= total * 1e-4 * (1 + 1e-12))
              & (error <= transferred * .10 * (1 + 1e-12))
              & (transferred > total * 1e-8))
        hits = np.flatnonzero(ok)
        if not hits.size:
            continue
        # Both CDF costs are sorted, so transferred cost is non-decreasing.
        # Resolve the maximum-cost plateau in NumPy rather than a Python loop
        # over thousands of histogram edges for every boundary pair.
        top = hits[transferred[hits] == transferred[hits[-1]]]
        i = int(top[np.argmin(error[top])])
        candidate = (float(transferred[i]), -float(error[i]), int(i), int(j[i]))
        if best is None or candidate > best:
            best = candidate
    return best


def propose(table, profile, transfer_fractions=(.0025, .005), max_candidates=12,
            max_operator_cost_fraction=.02, max_bucket_cost_fraction=.05,
            max_changed_group_fraction=.05, max_changed_mac_fraction=.05):
    """Generate diverse complete adjacent-boundary exchanges around the incumbent.

    Transfer fractions are ceilings, not required amounts. Proposal construction
    uses 80% of the local caps, leaving margin for candidate-trajectory changes;
    final auditing uses the caller's full caps. A quantized histogram may produce
    smaller transfers or no feasible proposal.
    ``propose.last_diagnostics`` also reports empty/rejected proposal pools.
    """
    caps = (max_operator_cost_fraction, max_bucket_cost_fraction,
            max_changed_group_fraction, max_changed_mac_fraction)
    _limits(*caps)
    planning_margin = .8
    planning_caps = tuple(planning_margin * cap for cap in caps)
    fractions = sorted(set(float(x) for x in transfer_fractions))
    if (not fractions or not all(np.isfinite(x) and 0 < x <= .01 for x in fractions)
            or not isinstance(max_candidates, int) or not 0 < max_candidates <= 64):
        raise ValueError("invalid proposal limits")
    replay, actual = baseline_cost(table, profile), exact_profile_cost(profile)
    if abs(replay - actual) > max(1e-9, abs(actual) * 1e-10):
        raise ValueError("incumbent table does not replay observed baseline")
    op_cost = defaultdict(float)
    for group in profile["groups"].values():
        if not np.array_equal(np.asarray(_entry(table, group)["thresholds"], np.float32),
                              np.asarray(group["thresholds"], np.float32)):
            raise ValueError("proposals require the exact profiled incumbent")
        op_cost[group["op"]] += _group_cost(group, group["thresholds"])
    total = float(profile["total_cycle_macs"])
    rejected = Counter()
    curves = [curve for group in profile["groups"].values()
              for curve in _curves(group, total, max(fractions), op_cost[group["op"]], planning_caps, rejected)]
    recipients = [c for c in curves if c["promote"]]
    donors = [c for c in curves if not c["promote"]]
    pools, seen = defaultdict(list), set()
    for recipient in recipients:
        for donor in donors:
            rg, dg = recipient["group"], donor["group"]
            if (rg["kind"], rg["key"]) == (dg["kind"], dg["key"]):
                rejected["same_bucket_pair"] += 1
                continue
            for fraction in fractions:
                limit = fraction * total
                if rg["op"] == dg["op"]:
                    limit = min(limit, .5 * planning_caps[0] * op_cost[rg["op"]])
                match = _match(recipient, donor, limit, total)
                if match is None:
                    rejected["no_costmatched_pair"] += 1
                    continue
                moved, neg_error, ri, di = match
                rt, dt = float(recipient["thresholds"][ri]), float(donor["thresholds"][di])
                signature = (rg["kind"], rg["key"], recipient["boundary"], rt,
                             dg["kind"], dg["key"], donor["boundary"], dt)
                if signature in seen:
                    rejected["duplicate_pair"] += 1
                    continue
                seen.add(signature)
                category = _category(recipient, donor)
                pools[category].append(dict(recipient=recipient, donor=donor, rt=rt, dt=dt,
                                            fraction=fraction, moved=moved,
                                            error=-neg_error, signature=signature))
    priority = ["qk_from_linears", "linears_from_qk", "av_from_linears", "linears_from_av",
                "qk_from_av", "av_from_qk", "projections_from_mlp", "mlp_from_projections",
                "within_projections", "within_mlp"]
    categories = [c for c in priority if pools.get(c)] + sorted(set(pools) - set(priority))
    out, usage = [], Counter()
    while len(out) < max_candidates and any(pools.values()):
        for category in categories:
            if not pools[category] or len(out) >= max_candidates:
                continue
            pool = pools[category]
            def rank(item):
                r, d = item["recipient"], item["donor"]
                tags = (r["group"]["key"], d["group"]["key"])
                return (sum(usage[t] for t in tags), -item["moved"], item["error"], item["signature"])
            item = pool.pop(min(range(len(pool)), key=lambda i: rank(pool[i])))
            candidate = copy.deepcopy(table)
            for role, threshold in (("recipient", item["rt"]), ("donor", item["dt"])):
                move = item[role]
                _entry(candidate, move["group"])["thresholds"][move["boundary"]] = threshold
            audit = audit_transition(table, candidate, profile, *caps)
            if not audit["feasible"]:
                rejected["final_audit_rejected"] += 1
                continue
            for role in ("recipient", "donor"):
                usage[item[role]["group"]["key"]] += 1
            audit.update(category=category, requested_transfer_fraction=item["fraction"],
                         transfer_limit_is_ceiling=True,
                         planning_cap_fraction=planning_margin,
                         max_operator_cost_fraction=caps[0], max_bucket_cost_fraction=caps[1],
                         max_changed_group_fraction=caps[2], max_changed_mac_fraction=caps[3])
            out.append(dict(name=f"adj_{category}_{len(out):02d}", table=candidate, diagnostics=audit))
    summary = dict(single_boundary_curves=len(curves), recipient_curves=len(recipients),
                   donor_curves=len(donors), unique_matched_pairs=len(seen),
                   emitted_candidates=len(out), rejection_counts=dict(rejected),
                   planning_cap_fraction=planning_margin, categories=categories)
    propose.last_diagnostics = summary
    for result in out:
        result["diagnostics"]["proposal_pool"] = copy.deepcopy(summary)
    return out


propose.last_diagnostics = {}
