"""Observe deployed PRC dispatch and propose small, iso-cost threshold transfers.

This module never changes ladders, escape gates, protected channels, the INT mask,
weights, or SC execution.  The collector wraps dispatch functions only while its
context is active; it returns their original results without modification.  Costs
are actual MACs times deployed lengths, including short final chunks and every
executed MoE expert.  INT/FP operations are outside the SC cost denominator.

Histogram boundaries include incumbent thresholds. Proposed thresholds snap to
the same boundaries, so replay costs are exact on the profiled input trajectory,
not a bin-center approximation. Actual candidate trajectories must still be run.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np


LINEARS = frozenset(("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"))
PROJECTIONS = frozenset(("q_proj", "k_proj", "v_proj", "o_proj"))
MLP = frozenset(("gate_proj", "up_proj", "down_proj"))


def _entry(table: dict, group: dict) -> dict:
    section = (table.get("per_row_chunk") or {}) if group["kind"] == "prc" else table
    payload = (section.get("buckets") or {}).get(group["key"])
    if payload is None:
        raise ValueError(f"profiled bucket missing from table: {group['kind']}:{group['key']}")
    return payload


def _lengths(group: dict, thresholds) -> np.ndarray:
    metric = np.asarray(group["metric"], dtype=np.float32)
    th = np.asarray(thresholds, dtype=np.float32)
    levels = np.asarray(group["levels"], dtype=np.float64)
    if len(th) != len(levels) - 1:
        raise ValueError(f"threshold/ladder mismatch: {group['key']}")
    if group["kind"] == "prc":
        if np.any(th[1:] < th[:-1]):
            raise ValueError("PRC thresholds must be ascending")
        return levels[np.searchsorted(th, metric, side="left")]
    if np.any(th[1:] > th[:-1]):
        raise ValueError("attention thresholds must be descending")
    idx = np.full(metric.shape, len(levels) - 1, dtype=np.int64)
    for i in range(len(th) - 1, -1, -1):
        idx[metric >= th[i]] = i
    return levels[idx]


def _group_cost(group: dict, thresholds) -> float:
    return float(np.dot(np.asarray(group["mac"], dtype=np.float64),
                        _lengths(group, thresholds)))


def exact_profile_cost(profile: dict) -> float:
    """Actual SC MAC-weighted length, including all fixed SC work."""
    if profile["total_macs"] <= 0:
        raise ValueError("empty SC profile")
    return float(profile["total_cycle_macs"] / profile["total_macs"])


def baseline_cost(table: dict, profile: dict) -> float:
    """Replay a table on the observed inputs; lengths/escape configuration stay fixed."""
    cost = float(profile["fixed_cycle_macs"])
    for group in profile["groups"].values():
        payload = _entry(table, group)
        levels = (payload.get("levels") if group["kind"] == "prc" else
                  payload.get("stoc_len_levels", table.get("stoc_len_levels")))
        if list(levels) != group["levels"]:
            raise ValueError(f"ladder changed for {group['key']}")
        cost += _group_cost(group, payload["thresholds"])
    return cost / float(profile["total_macs"])


class ProfileCollector:
    """Context-managed, observe-only profiler for a loaded SC model.

    Run ordinary model forwards inside the context, then call ``snapshot()``.
    A fresh instance is required per table; changing config during collection is
    rejected. The compact snapshot is JSON serializable and CPU-solver friendly.
    """

    def __init__(self, model, bins: int = 8192):
        if bins < 32:
            raise ValueError("at least 32 histogram bins required")
        self.model, self.bins = model, int(bins)
        self._groups: dict[str, dict[str, Any]] = {}
        self._linear = None
        self._attention = None
        self._fixed_macs = 0.0
        self._fixed_cycles = 0.0
        self._fixed_device = []
        self._entered = False

    def _bucket(self, cfg, op, block, total, kind):
        from scmp_kernels.mp.config import _bucket_index
        n = ((getattr(cfg, "prc_layer_buckets", 0) or cfg.layer_buckets)
             if kind == "prc" else cfg.layer_buckets)
        return f"{op}:t0:l{_bucket_index(int(block), int(total), int(n))}"

    def _group(self, cfg, op, block, total, kind, levels, thresholds, device):
        import torch
        key = self._bucket(cfg, op, block, total, kind)
        tag = f"{kind}|{key}"
        levels, thresholds = list(map(int, levels)), list(map(float, thresholds))
        if tag not in self._groups:
            # Float32 is the dispatch metric/threshold dtype in sc_common.
            edges = np.unique(np.r_[np.linspace(0, 1, self.bins + 1),
                                    thresholds].astype(np.float32))
            # PRC intervals are (a,b]; attention intervals are [a,b).
            reps = (np.r_[edges, np.float32(1)] if kind == "prc"
                    else np.r_[np.float32(0), edges])
            self._groups[tag] = dict(
                kind=kind, op=op, key=key, levels=levels, thresholds=thresholds,
                edges=edges, reps=reps,
                edges_gpu=torch.as_tensor(edges, device=device),
                hist=torch.zeros(len(edges) + 1, dtype=torch.float64, device=device),
                actual=torch.zeros((), dtype=torch.float64, device=device),
                calls=0,
            )
        g = self._groups[tag]
        if g["levels"] != levels or g["thresholds"] != thresholds:
            raise ValueError(f"allocation changed while profiling {tag}")
        if g["hist"].device != device:
            raise ValueError("multi-device dispatch profiling is not supported")
        g["calls"] += 1
        return g

    def _add(self, group, metric, weights, actual_lengths):
        import torch
        idx = torch.bucketize(metric.reshape(-1).contiguous(), group["edges_gpu"],
                              right=group["kind"] == "attention")
        w = weights.reshape(-1).to(dtype=torch.float64)
        group["hist"].add_(torch.bincount(idx, weights=w, minlength=group["hist"].numel()))
        group["actual"].add_((w * actual_lengths.reshape(-1).to(torch.float64)).sum())

    def _observe_prc(self, x, chunk_d, levels, thresholds, assignment):
        import torch
        if self._linear is None or thresholds is None:
            raise ValueError("PRC dispatch observed outside a supported SCLinear call")
        mod, cfg, op, block, total, seen = self._linear
        if seen:
            raise ValueError("multiple PRC dispatches in one SCLinear call")
        self._linear[-1] = True
        n, d = x.shape
        if n == 0:
            return
        nch = (d + chunk_d - 1) // chunk_d
        xa = x.abs()
        if nch * chunk_d != d:
            xa = torch.nn.functional.pad(xa, (0, nch * chunk_d - d))
        metric = xa.view(n, nch, chunk_d).amax(-1)
        lo, hi = metric.min(), metric.max()
        metric = (metric - lo) / (hi - lo).clamp_min(1e-8)
        widths = torch.full((nch,), int(chunk_d), dtype=torch.float64, device=x.device)
        widths[-1] = d - (nch - 1) * chunk_d
        weights = (widths * mod.out_features)[None, :].expand(n, -1)
        actual = torch.as_tensor(levels, device=x.device)[assignment.long()]
        group = self._group(cfg, op, block, total, "prc", levels, thresholds, x.device)
        self._add(group, metric, weights, actual)

    def _observe_attention(self, metric, cfg, assignment, kwargs):
        import torch
        if self._attention is None:
            return
        a, b, kw, seen = self._attention
        op, block, total = kw.get("operator"), kw.get("block_idx"), kw.get("total_blocks")
        if op not in ("qk", "av") or kwargs.get("operator") != op:
            raise ValueError("unrecognized attention dispatch context")
        if seen:
            raise ValueError("multiple classifiers in one attention product")
        self._attention[-1] = True
        levels = cfg.get_levels(operator=op, block_idx=block, total_blocks=total)
        thresholds = cfg.get_thresholds(0, 1, operator=op, block_idx=block, total_blocks=total)
        if thresholds is None or cfg.target_fractions is not None:
            raise ValueError("only calibrated attention thresholds are supported")
        values = cfg.classify_level_values(operator=op, block_idx=block, total_blocks=total)
        actual = torch.as_tensor(values, device=metric.device)[assignment.row_levels.long()]
        mac = int(a.shape[-1]) * int(b.shape[-2])
        span = metric.max() - metric.min()
        if float(span.item()) < 1e-8:
            # Runtime assigns the longest rung before threshold/escape handling.
            self._fixed_macs += metric.numel() * mac
            self._fixed_cycles += metric.numel() * mac * int(levels[0])
            return
        mn = (metric - metric.min()) / span
        esc = cfg.get_escape_threshold(0, 1, operator=op, block_idx=block, total_blocks=total)
        esc_len = int(getattr(cfg, "escape_stoc_len", 0) or 0)
        escaped = (mn > mn.new_tensor(esc)) if esc is not None and esc < 1 and esc_len else torch.zeros_like(mn, dtype=torch.bool)
        # Keep these device scalars until snapshot, avoiding an extra host sync.
        self._fixed_device.append((escaped.sum() * mac, (actual[escaped].double() * mac).sum()))
        group = self._group(cfg, op, block, total, "attention", levels, thresholds, metric.device)
        keep = ~escaped
        weights = torch.full((int(keep.sum().item()),), mac, dtype=torch.float64, device=metric.device)
        self._add(group, mn[keep], weights, actual[keep])

    def __enter__(self):
        import os
        import model.sc_common as scm
        if self._entered or getattr(scm, "_prc_local_collector", None) is not None:
            raise RuntimeError("ProfileCollector contexts cannot overlap")
        if os.environ.get("SC_PRC_ROWSHARED", "0") != "0":
            raise ValueError("row-shared ablation is not a per-group allocation")
        self._scm = scm
        self._orig = (scm.SCLinear.forward, scm.per_row_chunk_rungs,
                      scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows)
        collector = self
        old_linear, old_prc, old_attention, old_classify = self._orig

        def linear(mod, x):
            cfg = mod._sc_config
            op, block = getattr(mod, "_sc_op_name", None), getattr(mod, "_sc_block_idx", None)
            if not getattr(cfg, "use_sc_linear", True) or scm._hybrid_backend(cfg, op, block) != "sc":
                return old_linear(mod, x)
            if x.numel() == 0:
                return old_linear(mod, x)
            if op not in LINEARS or block is None:
                raise ValueError(f"unrecognized SC linear {op}/{block}")
            mp = getattr(cfg, "sc_mp_config", None)
            total = getattr(cfg, "_sc_total_blocks", None) or getattr(cfg, "num_hidden_layers", None)
            if mp is None or total is None or getattr(cfg, "sc_group_stoclen", None) is not None or getattr(cfg, "sc_ste_grad", False):
                raise ValueError("collector requires deployed adaptive PRC linears")
            unit = getattr(mod, "_sc_unit_idx", None)
            protected = mp.get_protected_channels(operator=op, block_idx=block, unit_idx=unit)
            count = int(scm._channel_index_tensor(protected, x.shape[-1], x.device).numel())
            rows = x.numel() // x.shape[-1]
            fixed_mac = rows * count * mod.out_features
            fixed_len = int(getattr(mp, "protected_channel_stoc_len", None) or max(mp.stoc_len_levels))
            previous = collector._linear
            collector._linear = [mod, mp, op, block, total, False]
            try:
                result = old_linear(mod, x)
                if count < x.shape[-1] and not collector._linear[-1]:
                    raise ValueError(f"SC linear has no observed PRC dispatch: {op}/{block}")
                collector._fixed_macs += fixed_mac
                collector._fixed_cycles += fixed_mac * fixed_len
                return result
            finally:
                collector._linear = previous

        def prc(x, chunk_d, levels, thresholds=None, target=0.0):
            result = old_prc(x, chunk_d, levels, thresholds=thresholds, target=target)
            collector._observe_prc(x, chunk_d, levels, thresholds, result)
            return result

        def attention(a, b, **kw):
            previous = collector._attention
            collector._attention = [a, b, kw, False]
            try:
                result = old_attention(a, b, **kw)
                if not collector._attention[-1]:
                    raise ValueError("SC attention did not use adaptive row classification")
                return result
            finally:
                collector._attention = previous

        def classify(metric, config, *args, **kw):
            result = old_classify(metric, config, *args, **kw)
            if collector._attention is not None:
                collector._observe_attention(metric, config, result, kw)
            return result

        scm._prc_local_collector = self
        scm.SCLinear.forward, scm.per_row_chunk_rungs = linear, prc
        scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows = attention, classify
        self._entered = True
        return self

    def __exit__(self, *exc):
        scm = self._scm
        (scm.SCLinear.forward, scm.per_row_chunk_rungs,
         scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows) = self._orig
        del scm._prc_local_collector
        self._entered = False

    def snapshot(self) -> dict:
        groups = {}
        fixed_mac, fixed_cycles = self._fixed_macs, self._fixed_cycles
        for mac, cycles in self._fixed_device:
            fixed_mac += float(mac.item())
            fixed_cycles += float(cycles.item())
        total_mac, total_cycles = fixed_mac, fixed_cycles
        for tag, g in self._groups.items():
            hist = g["hist"].detach().cpu().numpy()
            nonzero = hist > 0
            group = {k: g[k] for k in ("kind", "key", "op", "levels", "thresholds", "calls")}
            group.update(metric=g["reps"][nonzero].astype(float).tolist(),
                         mac=hist[nonzero].tolist(), edges=g["edges"].astype(float).tolist(),
                         actual_cycle_macs=float(g["actual"].item()))
            replay = _group_cost(group, group["thresholds"])
            if abs(replay - group["actual_cycle_macs"]) > max(1.0, abs(replay) * 1e-10):
                raise ValueError(f"baseline dispatch replay mismatch for {tag}: {replay} vs {group['actual_cycle_macs']}")
            groups[tag] = group
            total_mac += float(hist.sum())
            total_cycles += group["actual_cycle_macs"]
        out = dict(version=1, bins=self.bins, groups=groups, fixed_macs=fixed_mac,
                   fixed_cycle_macs=fixed_cycles, total_macs=total_mac,
                   total_cycle_macs=total_cycles)
        out["actual_mean_length"] = exact_profile_cost(out)
        return out


def _shift(group: dict, delta: float) -> list[float]:
    """Lower thresholds promote for both ascending PRC and descending attention."""
    th = np.asarray(group["thresholds"], dtype=np.float32)
    if delta == 0:
        return list(group["thresholds"])
    edges = np.asarray(group["edges"], dtype=np.float32)
    wanted = np.clip(th.astype(np.float64) + delta, 0, 1)
    idx = np.searchsorted(edges, wanted, side="left").clip(0, len(edges) - 1)
    left = np.maximum(idx - 1, 0)
    idx = np.where(np.abs(edges[left] - wanted) < np.abs(edges[idx] - wanted), left, idx)
    return edges[idx].astype(float).tolist()


def _shift_to_cost(groups: list[dict], target: float, promote: bool):
    base = sum(_group_cost(g, g["thresholds"]) for g in groups)
    best = (abs(base - target), base, {g["key"]: list(g["thresholds"]) for g in groups})
    lo, hi = 0.0, 1.0
    for step in range(40):
        distance = 1.0 if step == 0 else (lo + hi) / 2
        delta = -distance if promote else distance
        th = {g["key"]: _shift(g, delta) for g in groups}
        cost = sum(_group_cost(g, th[g["key"]]) for g in groups)
        candidate = (abs(cost - target), cost, th)
        # Neither side may exceed its transfer cap by one histogram bin.
        bounded = cost <= target if promote else cost >= target
        if candidate[0] < best[0] and bounded:
            best = candidate
        if step == 0:
            continue
        if (promote and cost < target) or (not promote and cost > target):
            lo = distance
        else:
            hi = distance
    return best[1], best[2]


def propose(table: dict, profile: dict, transfer_fraction: float = .02,
            max_candidates: int = 8) -> list[dict]:
    """Return up to eight bounded, bidirectional iso-cost threshold transfers.

    Each direction transfers at most ``transfer_fraction`` of TOTAL SC cycles.
    If a population is pinned, its missing capacity limits/skips that direction.
    Replay cost error must be <=0.1% of total SC cycles. This is only proposal
    cost; the caller must reject actual-trajectory cost drift during evaluation.
    """
    if not 0 < transfer_fraction <= .10:
        raise ValueError("transfer_fraction must be in (0, 0.10]")
    actual = exact_profile_cost(profile)
    replay = baseline_cost(table, profile)
    if abs(replay - actual) > max(1e-9, actual * 1e-10):
        raise ValueError(f"incumbent table does not replay observed baseline: {replay} vs {actual}")
    groups = list(profile["groups"].values())
    for group in groups:
        payload = _entry(table, group)
        if not np.array_equal(np.asarray(payload["thresholds"], np.float32),
                              np.asarray(group["thresholds"], np.float32)):
            raise ValueError("proposals must start at the exact profiled thresholds")
    total = float(profile["total_cycle_macs"])
    schemes = [("qk", frozenset(("qk",)), "linears", LINEARS),
               ("av", frozenset(("av",)), "linears", LINEARS),
               ("qk", frozenset(("qk",)), "av", frozenset(("av",))),
               ("projections", PROJECTIONS, "mlp", MLP)]
    out, seen = [], set()
    for an, aset, bn, bset in schemes:
        for rn, rset, dn, dset in ((an, aset, bn, bset), (bn, bset, an, aset)):
            receivers = [g for g in groups if g["op"] in rset]
            donors = [g for g in groups if g["op"] in dset]
            if not receivers or not donors:
                continue
            rc0 = sum(_group_cost(g, g["thresholds"]) for g in receivers)
            dc0 = sum(_group_cost(g, g["thresholds"]) for g in donors)
            rmax = sum(_group_cost(g, _shift(g, -1)) for g in receivers)
            dmin = sum(_group_cost(g, _shift(g, 1)) for g in donors)
            transfer = min(transfer_fraction * total, rmax - rc0, dc0 - dmin)
            if transfer <= total * 1e-6:
                continue
            rc, rt = _shift_to_cost(receivers, rc0 + transfer, True)
            gain = rc - rc0
            if gain <= total * 1e-6 or gain > transfer_fraction * total * (1 + 1e-6):
                continue
            dc, dt = _shift_to_cost(donors, dc0 - gain, False)
            if dc >= dc0 or abs((rc - rc0) + (dc - dc0)) > total * .001:
                continue
            cand = copy.deepcopy(table)
            for group in receivers + donors:
                _entry(cand, group)["thresholds"] = (rt if group in receivers else dt)[group["key"]]
            import json
            fingerprint = json.dumps(cand, sort_keys=True, separators=(",", ":"))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            cost = baseline_cost(cand, profile)
            out.append(dict(name=f"{rn}_from_{dn}", table=cand, diagnostics=dict(
                receiver=rn, donor=dn, transfer_fraction_requested=transfer_fraction,
                transferred_cycle_macs=rc - rc0, transferred_fraction=(rc - rc0) / total,
                donor_freed_fraction=(dc0 - dc) / total, baseline_mean_length=actual,
                predicted_mean_length=cost, predicted_cost_ratio=cost / actual,
                receiver_capacity_fraction=(rmax - rc0) / total,
                donor_capacity_fraction=(dc0 - dmin) / total)))
            if len(out) >= max_candidates:
                return out
    return out
