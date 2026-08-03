#!/usr/bin/env python
"""FINAL SC mixed-precision algorithm (V17) — best-possible configuration.

Consolidates every V17 lesson into one parameterized, gated pipeline that emits
a deployable table + wrapper. Objective is BEST QUALITY at honestly-reported
realized cost (not iso-cost vs V9): the escape gate is purely additive and the
cost floats up, which the (PPL, realized-energy) reporting accounts for.

    Stage 0  Hybrid INT mask, dose 20%  (mask-blind calibration: the MP table is
             byte-identical across doses, PROVEN, so no recalibration is needed
             — only the hybrid config swaps at eval time).
    Stage 1  sigma dispatch + per-(op,layer-bucket) allocation = the V9 parent
             (act_global_v9). Unchanged: sigma is a good WITHIN-group row
             ranker; it is only the cross-group PRICE that is wrong.
    Stage 2  Deterministic structural recipe (measured-validated):
             a. protected-channel trim to ~80 + MLP-floor reinvest to iso-cost.
                The pins were ~2x over-provisioned; biggest single lever
                (4B -3.11%), plateaus below 80. psl must stay OFF-ladder.
             b. attention -> ceiling lift IF affordable at iso-cost (llama8B
                yes, stacks to -4.96%; 4B/30B infeasible -> skipped).
             c. OPTIONAL per-layer measured re-pricing: promote layers whose
                MEASURED marginal value dNLL/dcycle is positive, fund from those
                where it is negative. Requires per-layer resolution: adjacent
                layers carry OPPOSITE-sign marginals (L0 -0.60 vs L2 +0.46), so
                4-bucket aggregation cancels the signal.
    Stage 3  Additive absolute escape gate, k>=2.0: rows above mu_b + k*sigma_b
             escape to 128. NOTHING is decreased elsewhere, so realized cost
             floats up (+~1.7 cyc at k=2.0) and is reported.

EXCLUDED by evidence (do not re-add):
  * threshold-nudge moves           - zero measured signal (r~0.03)
  * floor raises funded by the CEILING - hurt (+0.68%); pins are the slack,
                                      the ceiling is useful capacity
  * per-op / per-layer moves priced by the PRE-MASK sensitivity file - the mask
    consumes that sensitivity, so it anti-predicts (early_protect hurt on 4/4
    models). Only DEPLOYED-config measured prices are valid.

Emits table.json + wrapper.json + build_report.json, gated by:
topology validation, iso-cost tolerance, occupancy re-derivation, wrapper
resolve through the real eval load path, and a runtime-loader smoke.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "kernels"))
from benchmark.ppl.mp_ladder_refine import (  # noqa: E402
    _atomic_json, _protected_info, _write_wrapper, aggregate_trace_weights,
    resolve_parent_table, weighted_cost, wrapper_escape_gate,
)
from benchmark.ppl.mp_joint_refine import (  # noqa: E402
    OP_GROUPS, _hist_values, _shift_one_boundary, profile_level_weights,
)
from benchmark.ppl.mp_v16_refine import make_v16_table  # noqa: E402
from scmp_kernels.mp.config import AdaptiveMPConfig  # noqa: E402

MLP = OP_GROUPS["mlp"]
ATTN = OP_GROUPS["attn_matmul"]
TOL = 0.05
DEFAULT_PSL = 80          # measured optimum; plateaus below, never a ladder rung
DEFAULT_GATE_K = 2.0      # additive escape gate


# ----------------------------------------------------------------- helpers
def expected_cost(table: dict, profile: dict) -> float:
    psl, pw = _protected_info(table)
    lw = profile_level_weights(table, profile, protected_weight=pw)
    return weighted_cost(table["stoc_len_levels"], lw, pw, psl or 0)


def _shift(table, profile, keyfilter, mass_delta, boundary=-1):
    """Shift the floor boundary for buckets matching keyfilter(op, l_bucket)."""
    out = copy.deepcopy(table)
    values = _hist_values(profile)
    n = 0
    for key, group in (profile.get("groups") or {}).items():
        op, lb = str(group["op"]), group.get("l_bucket")
        if not keyfilter(op, lb):
            continue
        payload = (out.get("buckets") or {}).get(key)
        if payload is None:
            continue
        hist = [float(x) for x in group["mac_weighted_hist"]]
        shifted, mv = _shift_one_boundary(
            [float(x) for x in payload["thresholds"]], hist, values,
            boundary=boundary, mass_delta=mass_delta)
        payload["thresholds"] = shifted
        n += 1 if mv.get("changed") else 0
    return out, n


def _fill_to_cost(table, profile, target, ops=MLP):
    """Spend/recover budget on MLP floors (cascade -1,-2,-3) to hit target."""
    need = target - expected_cost(table, profile)
    if abs(need) <= TOL:
        return table, expected_cost(table, profile), []
    base, used = table, []
    for b in (-1, -2, -3):
        sat, _ = _shift(base, profile, lambda o, l: o in ops, 1.0, boundary=b)
        if expected_cost(sat, profile) >= target - TOL:
            lo, hi, best = 0.0, 1.0, (None, 1e9, None)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                cand, _n = _shift(base, profile, lambda o, l: o in ops, mid,
                                  boundary=b)
                c = expected_cost(cand, profile)
                if abs(c - target) < best[1]:
                    best = (cand, abs(c - target), c)
                if c < target:
                    lo = mid
                else:
                    hi = mid
            return best[0], best[2], used + [str(b)]
        base, used = sat, used + [str(b)]
    return base, expected_cost(base, profile), used + ["SATURATED"]


# ------------------------------------------------------------------ stages
def stage2a_psl_trim(table, profile, target, new_psl, rep):
    """Trim the over-provisioned protected pins. Does NOT spend the freed budget.

    ORDER MATTERS: the freed pin budget must be offered to the attention lift
    (2b) FIRST, and only the residual reinvested into MLP floors (2d). Spending
    it all on floors here silently starves the lift — that is exactly the
    winning llama8B recipe (trim -> lift -> reinvest residual)."""
    levels = [int(x) for x in table["stoc_len_levels"]]
    if new_psl in levels:
        raise ValueError(f"psl {new_psl} collides with a ladder rung {levels}")
    t = copy.deepcopy(table)
    old = t["protected_channels"]["stoc_len"]
    t["protected_channels"]["stoc_len"] = new_psl
    freed = expected_cost(table, profile) - expected_cost(t, profile)
    rep["stage2a"] = {"psl_from": old, "psl_to": new_psl,
                      "freed_avgL": freed, "cost_after_trim": expected_cost(t, profile)}
    return t


def stage2b_attention_lift(table, profile, target, rep):
    """Lift attention to the ceiling IFF the freed pin budget covers it.

    Applied BEFORE the MLP reinvest, so it competes for the freed budget rather
    than being starved by it. llama8B affords it (stacks to -4.96%); 4B/30B do
    not (lift costs ~1.64 vs ~0.89 freed) and correctly skip."""
    levels = [int(x) for x in table["stoc_len_levels"]]
    t = copy.deepcopy(table)
    for key in [f"{op}:t0:l{i}" for op in ATTN for i in range(4)]:
        p = t["buckets"].get(key)
        if p is None:
            continue
        # topology_insert/remove tables archive occupancy as pass1_*; read and
        # write back whichever convention this table uses.
        ckey = "counts" if "counts" in p else "pass1_counts"
        fkey = "fractions" if "fractions" in p else "pass1_fractions"
        if p.get(ckey) is None:
            continue
        tot = int(sum(int(x) for x in p[ckey]))
        p["thresholds"] = [0.0] * (len(levels) - 1)      # sign-safe sentinel
        p[ckey] = [tot] + [0] * (len(levels) - 1)
        p[fkey] = [1.0] + [0.0] * (len(levels) - 1)
    lift_cost = expected_cost(t, profile) - expected_cost(table, profile)
    over = expected_cost(t, profile) - target
    if over > TOL:                        # freed budget does not cover the lift
        rep["stage2b"] = {"applied": False, "lift_cost_avgL": lift_cost,
                          "overshoot_avgL": over,
                          "reason": "lift exceeds the freed pin budget"}
        return table
    rep["stage2b"] = {"applied": True, "lift_cost_avgL": lift_cost,
                      "cost_after_lift": expected_cost(t, profile)}
    return t


def stage2c_measured_reprice(table, profile, target, prices, rep,
                             promote_scale=0.35, min_dcost=0.005):
    """Per-LAYER reallocation using MEASURED marginal values (deployed config).

    prices: [{"layer": int, "marginal_value": float, "d_cost": float}, ...]
    Promote layers whose measured marginal value is positive; fund by demoting
    those where it is negative. Only per-layer resolution can express this."""
    usable = [p for p in prices if p.get("marginal_value") is not None
              and p.get("d_cost", 0) >= min_dcost]
    pos = {p["layer"]: p["marginal_value"] for p in usable if p["marginal_value"] > 0}
    neg = {p["layer"]: p["marginal_value"] for p in usable if p["marginal_value"] < 0}
    if not pos or not neg:
        rep["stage2c"] = {"applied": False, "reason": "need both signs"}
        return table
    pmax = max(pos.values()); nmax = max(abs(v) for v in neg.values())
    t, _ = _shift(table, profile,
                  lambda o, l: o in MLP and l in pos,
                  0.0)                                     # placeholder, per-layer below
    # promote each positive layer proportional to its measured value
    t = copy.deepcopy(table)
    for L, v in pos.items():
        t, _ = _shift(t, profile, lambda o, l, L=L: o in MLP and l == L,
                      promote_scale * (v / pmax))
    lo, hi, best = 0.0, 1.0, (None, 1e9, None)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cand = t
        for L, v in neg.items():
            cand, _n = _shift(cand, profile,
                              lambda o, l, L=L: o in MLP and l == L,
                              -mid * (abs(v) / nmax))
        c = expected_cost(cand, profile)
        if abs(c - target) < best[1]:
            best = (cand, abs(c - target), c)
        if c > target:
            lo = mid
        else:
            hi = mid
    rep["stage2c"] = {"applied": True, "promote_layers": sorted(pos),
                      "demote_layers": sorted(neg), "cost": best[2]}
    return best[0]


# -------------------------------------------------------------------- main
def build(parent_wrapper, profile_path, parent_trace, outdir, *,
          psl=DEFAULT_PSL, gate_k=DEFAULT_GATE_K, attention_lift=True,
          prices_path=None, target_cost=None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rep = {"algorithm": "final_mp_v17", "status": "failed",
           "params": {"psl": psl, "gate_k": gate_k,
                      "attention_lift": attention_lift}}
    wrapper, ptp, parent = resolve_parent_table(Path(parent_wrapper))
    levels = [int(x) for x in parent["stoc_len_levels"]]
    psl0, pw = _protected_info(parent)
    profile = json.load(open(profile_path))
    if target_cost is None:
        agg = aggregate_trace_weights(json.load(open(parent_trace)), levels,
                                      protected_stoc_len=psl0)
        target_cost = float(agg["cost"])
    rep["target_cost"] = target_cost

    # 2a trim pins -> 2b offer the freed budget to the attention lift ->
    # 2c optional measured re-price -> 2d reinvest the RESIDUAL to iso-cost.
    t = stage2a_psl_trim(parent, profile, target_cost, psl, rep)
    if attention_lift:
        t = stage2b_attention_lift(t, profile, target_cost, rep)
    if prices_path:
        t = stage2c_measured_reprice(
            t, profile, target_cost,
            json.load(open(prices_path)).get("results", []), rep)
    t, cost_filled, used = _fill_to_cost(t, profile, target_cost)
    rep["stage2d_reinvest"] = {"boundaries": used, "cost": cost_filled}

    final_cost = expected_cost(t, profile)
    rep["final_cost"] = final_cost
    rep["cost_err"] = final_cost - target_cost
    if abs(final_cost - target_cost) > TOL:
        rep["notes"] = "iso-cost gate failed"
        _atomic_json(outdir / "build_report.json", rep)
        raise SystemExit(f"iso-cost miss {final_cost:.4f} vs {target_cost:.4f}")

    psl_f, pw_f = _protected_info(t)
    lw = profile_level_weights(t, profile, protected_weight=pw_f)
    assert abs(sum(lw) + pw_f - 1.0) < 1e-9, "occupancy must sum to 1"
    assert psl_f == psl and psl not in levels
    assert min(levels) >= 16, "no rung below 16"

    ft = make_v16_table(t, levels, root_parent_table=ptp,
                        target_cost=target_cost,
                        action={"type": "final_mp_v17", **rep["params"],
                                "predicted_cost": final_cost},
                        command="final_mp_algorithm.py")
    tp = outdir / "table.json"; wp = outdir / "wrapper.json"
    _atomic_json(tp, ft)
    _write_wrapper(wp, levels, tp)
    # Stage 3: additive escape gate (table untouched; cost floats up by design)
    if gate_k is not None:
        w = json.load(open(wp))
        w["escape_gate_k"] = float(gate_k)
        w["escape_stoc_len"] = 128
        _atomic_json(wp, w)
    rw, rtp, rt = resolve_parent_table(wp)
    gate = wrapper_escape_gate(rw)
    assert (gate.get("escape_gate_k") == gate_k) if gate_k else (gate == {})
    AdaptiveMPConfig(stoc_len_levels=list(levels), threshold_table_path=str(tp))
    rep.update({"status": "ok", "table": str(tp), "wrapper": str(wp),
                "escape_gate": gate})
    _atomic_json(outdir / "build_report.json", rep)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-wrapper", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--parent-trace", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--psl", type=int, default=DEFAULT_PSL)
    ap.add_argument("--gate-k", type=float, default=DEFAULT_GATE_K)
    ap.add_argument("--no-attention-lift", action="store_true")
    ap.add_argument("--prices", default=None,
                    help="per-layer measured marginal prices (stage 2c)")
    a = ap.parse_args()
    rep = build(a.parent_wrapper, a.profile, a.parent_trace, a.outdir,
                psl=a.psl, gate_k=a.gate_k,
                attention_lift=not a.no_attention_lift, prices_path=a.prices)
    print(json.dumps({k: rep[k] for k in
                      ("status", "target_cost", "final_cost", "cost_err",
                       "escape_gate")}, indent=1))
    for s in ("stage2a", "stage2b", "stage2c"):
        if s in rep:
            print(f"{s}: {json.dumps(rep[s])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
