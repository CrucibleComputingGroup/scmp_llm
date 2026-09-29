#!/usr/bin/env python
"""Rotation-gate verdict: compare rotated-model MP calibration tables against
the trusted unrotated V9 baseline table.  CPU-only, no torch needed.

For each rotated table (one per rotation seed) vs the baseline:
  * per-op-bucket sigma(L): level_mean_error at every level, per bucket and
    num_units-weighted mean over the 4 layer buckets, with % deltas
  * headline d_sigma@{32,48,96} (halved levels) percent per op
  * dispatch_metrics comparison (chosen metric, sign, rho_signed, rho_amax)
  * allocation shift (per-bucket avg_stoc_len, level fractions, global
    expected_avg_stoc_len / expected_flop_avg_stoc_len / global_lambda)
  * CONTROL: sigma movement split into R1/R2-REACHED ops
    [q/k/v/gate/up/o_proj, av] vs NOT-REACHED ops [qk, down_proj] — the
    not-reached inputs are mathematically unchanged by the fold, so material
    movement there means a fold bug or an indirect SmoothQuant-recalibration
    effect; flagged.

Emits <out-prefix>.json (full detail) and <out-prefix>.md (summary).
"""
import argparse
import json
import os
import statistics
import sys

REACHED_OPS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
               "o_proj", "av"]
NOT_REACHED_OPS = ["qk", "down_proj"]

DEFAULT_BASELINE = ("/nfs/turbo/coe-nbleier/allenjin/hpca/"
                    "mp_calib_mp_v9_int6_20260716_001908/"
                    "meta-llama_Llama-3.1-8B-Instruct__mp_avg32_v9__"
                    "act_global_v9_pc0.01x112_act_collapse_comp_pcov"
                    "down_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.10.json")


def load_table(path):
    with open(path) as f:
        return json.load(f)


def pct(new, old):
    if old == 0:
        return None
    return (new - old) / abs(old) * 100.0


def weighted_sigma(table, op):
    """num_units-weighted mean level_mean_error over the op's layer buckets.
    Returns (levels-aligned list, total_units, per_bucket dict)."""
    buckets = table["buckets"]
    n_levels = len(table["stoc_len_levels"])
    acc = [0.0] * n_levels
    tot = 0
    per_bucket = {}
    for key, b in buckets.items():
        if key.split(":")[0] != op:
            continue
        nu = b["num_units"]
        lme = b["level_mean_error"]
        per_bucket[key] = {
            "num_units": nu,
            "level_mean_error": lme,
            "avg_stoc_len": b.get("avg_stoc_len"),
            "avg_error": b.get("avg_error"),
            "fractions": b.get("fractions"),
            "metric_mean": b.get("metric_mean"),
            "metric_std": b.get("metric_std"),
        }
        for i in range(n_levels):
            acc[i] += nu * lme[i]
        tot += nu
    if tot == 0:
        return None, 0, {}
    return [a / tot for a in acc], tot, per_bucket


def compare_one(base, rot, rot_path, interest_levels):
    levels = base["stoc_len_levels"]
    assert rot["stoc_len_levels"] == levels, \
        f"{rot_path}: stoc_len_levels differ ({rot['stoc_len_levels']} vs {levels})"
    assert rot.get("layer_buckets") == base.get("layer_buckets"), rot_path
    ops = base["operators"]
    assert rot["operators"] == ops, rot_path

    out = {"rotated_table": rot_path,
           "model_path": rot.get("model_path"),
           "levels": levels,
           "per_op": {}, "dispatch": {}, "global": {}, "control": {}}

    for op in ops:
        ws_b, nu_b, pb_b = weighted_sigma(base, op)
        ws_r, nu_r, pb_r = weighted_sigma(rot, op)
        if ws_b is None or ws_r is None:
            out["per_op"][op] = {"missing": True}
            continue
        d = {"num_units_base": nu_b, "num_units_rot": nu_r,
             "sigma_base": ws_b, "sigma_rot": ws_r,
             "sigma_delta_pct": [pct(r, b) for r, b in zip(ws_r, ws_b)],
             "at_interest": {}}
        for L in interest_levels:
            if L in levels:
                i = levels.index(L)
                d["at_interest"][str(L)] = {
                    "base": ws_b[i], "rot": ws_r[i],
                    "delta_pct": pct(ws_r[i], ws_b[i])}
        # per-bucket detail (matched keys only; report mismatches)
        pb = {}
        for key in sorted(set(pb_b) | set(pb_r)):
            if key not in pb_b or key not in pb_r:
                pb[key] = {"missing_in": "rot" if key not in pb_r else "base"}
                continue
            b_, r_ = pb_b[key], pb_r[key]
            pb[key] = {
                "num_units": [b_["num_units"], r_["num_units"]],
                "sigma_delta_pct": [pct(r, b) for r, b in
                                    zip(r_["level_mean_error"],
                                        b_["level_mean_error"])],
                "avg_stoc_len": [b_["avg_stoc_len"], r_["avg_stoc_len"]],
                "avg_error": [b_["avg_error"], r_["avg_error"]],
                "fractions_base": b_["fractions"],
                "fractions_rot": r_["fractions"],
                "metric_mean": [b_["metric_mean"], r_["metric_mean"]],
                "metric_std": [b_["metric_std"], r_["metric_std"]],
            }
        d["per_bucket"] = pb
        out["per_op"][op] = d

    # dispatch metrics
    dm_b = base.get("dispatch_metrics", {}) or {}
    dm_r = rot.get("dispatch_metrics", {}) or {}
    for op in ops:
        b_ = dm_b.get(op)
        r_ = dm_r.get(op)
        out["dispatch"][op] = {
            "base": b_, "rot": r_,
            "metric_changed": (b_ or {}).get("metric") != (r_ or {}).get("metric"),
            "sign_changed": (b_ or {}).get("sign") != (r_ or {}).get("sign"),
            "rho_signed_delta": (None if not (b_ and r_) else
                                 r_["rho_signed"] - b_["rho_signed"]),
            "rho_amax_delta": (None if not (b_ and r_) else
                               r_["rho_amax"] - b_["rho_amax"]),
        }

    for k in ("expected_avg_stoc_len", "expected_flop_avg_stoc_len",
              "global_lambda", "budget_ratio", "method", "seed"):
        out["global"][k] = {"base": base.get(k), "rot": rot.get(k)}
    pc_b = base.get("protected_channels") or {}
    pc_r = rot.get("protected_channels") or {}

    def _pc_count(pc):
        try:
            return sum(len(v) if hasattr(v, "__len__") else 1
                       for v in pc.values())
        except Exception:
            return None
    out["global"]["protected_channels_entries"] = {
        "base": _pc_count(pc_b), "rot": _pc_count(pc_r)}

    # control: reached vs not-reached mean |delta sigma %| over buckets+levels
    def mean_abs(op_set):
        vals = []
        for op in op_set:
            d = out["per_op"].get(op) or {}
            for key, pbv in (d.get("per_bucket") or {}).items():
                for v in pbv.get("sigma_delta_pct", []):
                    if v is not None:
                        vals.append(abs(v))
        return statistics.fmean(vals) if vals else None

    def mean_signed(op_set):
        vals = []
        for op in op_set:
            d = out["per_op"].get(op) or {}
            for v in (d.get("sigma_delta_pct") or []):
                if v is not None:
                    vals.append(v)
        return statistics.fmean(vals) if vals else None

    reached_abs = mean_abs(REACHED_OPS)
    notreached_abs = mean_abs(NOT_REACHED_OPS)
    out["control"] = {
        "reached_ops": REACHED_OPS,
        "not_reached_ops": NOT_REACHED_OPS,
        "reached_mean_abs_dsigma_pct": reached_abs,
        "not_reached_mean_abs_dsigma_pct": notreached_abs,
        "reached_mean_signed_dsigma_pct": mean_signed(REACHED_OPS),
        "not_reached_mean_signed_dsigma_pct": mean_signed(NOT_REACHED_OPS),
    }
    flag = (notreached_abs is not None and notreached_abs > 2.0)
    out["control"]["flag_not_reached_moved"] = bool(flag)
    out["control"]["flag_rule"] = ("FLAG if not-reached mean |dsigma%| > 2.0 "
                                   "(fold bug or indirect SQ-recalibration "
                                   "effect suspected)")
    return out


def render_md(report, interest_levels):
    L = []
    L.append("# Rotation gate report — offline R1+R2 vs unrotated V9 "
             "(llama8B)\n")
    L.append(f"Baseline: `{report['baseline_table']}`\n")
    L.append("Verdict axis: does the QuaRot-style offline rotation COMPRESS "
             "the SC per-level reconstruction error sigma that the MP "
             "calibrator measures?  Negative dsigma% = compression (good). "
             "No PPL is evaluated here.\n")
    for seed_rep in report["seeds"]:
        L.append(f"\n## {os.path.basename(seed_rep['rotated_table'])}\n")
        c = seed_rep["control"]
        flag = ("**FLAGGED: not-reached ops moved materially — suspect fold "
                "bug or indirect SQ effect**" if
                c["flag_not_reached_moved"] else "control clean")
        ra = c['reached_mean_abs_dsigma_pct']
        na = c['not_reached_mean_abs_dsigma_pct']
        rs = c['reached_mean_signed_dsigma_pct']
        ns = c['not_reached_mean_signed_dsigma_pct']

        def _f(x):
            return "n/a" if x is None else f"{x:+.2f}%"
        L.append(f"- reached ops mean signed dsigma: {_f(rs)} "
                 f"(mean |dsigma| {_f(ra)})")
        L.append(f"- NOT-reached ops (qk, down_proj) mean signed dsigma: "
                 f"{_f(ns)} (mean |dsigma| {_f(na)}) — {flag}\n")
        hdr = "| op | reached |"
        sep = "|---|---|"
        for lev in interest_levels:
            hdr += f" dsigma@{lev} |"
            sep += "---|"
        hdr += " rho_signed base->rot | metric base->rot |"
        sep += "---|---|"
        L.append(hdr)
        L.append(sep)
        for op in seed_rep["per_op"]:
            d = seed_rep["per_op"][op]
            if d.get("missing"):
                continue
            reached = "yes" if op in REACHED_OPS else "NO (control)"
            row = f"| {op} | {reached} |"
            for lev in interest_levels:
                ai = d["at_interest"].get(str(lev))
                row += (f" {ai['delta_pct']:+.2f}% |" if ai and
                        ai["delta_pct"] is not None else " n/a |")
            dm = seed_rep["dispatch"][op]
            b_, r_ = dm["base"], dm["rot"]
            if b_ and r_:
                row += (f" {b_['rho_signed']:.3f} -> {r_['rho_signed']:.3f} |"
                        f" {b_['metric']}({b_['sign']:+.0f}) -> "
                        f"{r_['metric']}({r_['sign']:+.0f}) |")
            else:
                row += " n/a | n/a |"
            L.append(row)
        g = seed_rep["global"]
        L.append("")
        L.append(f"- expected_avg_stoc_len: "
                 f"{g['expected_avg_stoc_len']['base']} -> "
                 f"{g['expected_avg_stoc_len']['rot']}; "
                 f"expected_flop_avg_stoc_len: "
                 f"{g['expected_flop_avg_stoc_len']['base']} -> "
                 f"{g['expected_flop_avg_stoc_len']['rot']}; "
                 f"global_lambda: {g['global_lambda']['base']} -> "
                 f"{g['global_lambda']['rot']}")
    xs = report.get("cross_seed")
    if xs:
        L.append("\n## Cross-seed consistency (weighted per-op dsigma%, "
                 "sign agreement)\n")
        L.append("| op | level | mean dsigma% | per-seed | consistent sign |")
        L.append("|---|---|---|---|---|")
        for row in xs:
            L.append(f"| {row['op']} | {row['level']} | "
                     f"{row['mean_delta_pct']:+.2f}% | "
                     f"{', '.join('%+.2f' % v for v in row['per_seed'])} | "
                     f"{'yes' if row['sign_consistent'] else 'NO'} |")
    L.append("\n*(sigma values are the calibrator's per-level mean relative "
             "reconstruction errors; levels are HALVED cycle counts — "
             "nominal = 2x.)*\n")
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=DEFAULT_BASELINE)
    p.add_argument("--rotated", nargs="+", required=True,
                   help="one or more rotated-model calibration tables")
    p.add_argument("--out-prefix", required=True,
                   help="writes <prefix>.json and <prefix>.md")
    p.add_argument("--interest-levels", default="96,48,32",
                   help="HALVED levels for the headline delta (default "
                        "96,48,32 = nominal 192,96,64... quoted as halved)")
    args = p.parse_args()

    interest = [int(x) for x in args.interest_levels.split(",")]
    base = load_table(args.baseline)
    report = {"baseline_table": args.baseline,
              "interest_levels_halved": interest,
              "seeds": []}
    for rp in args.rotated:
        rot = load_table(rp)
        report["seeds"].append(compare_one(base, rot, rp, interest))

    # cross-seed consistency on the weighted per-op deltas
    cross = []
    if len(report["seeds"]) > 1:
        ops = list(report["seeds"][0]["per_op"].keys())
        for op in ops:
            for lev in interest:
                vals = []
                for s in report["seeds"]:
                    ai = s["per_op"].get(op, {}).get("at_interest", {}) \
                        .get(str(lev))
                    if ai and ai["delta_pct"] is not None:
                        vals.append(ai["delta_pct"])
                if len(vals) > 1:
                    cross.append({
                        "op": op, "level": lev, "per_seed": vals,
                        "mean_delta_pct": statistics.fmean(vals),
                        "sign_consistent": all(v < 0 for v in vals) or
                                           all(v > 0 for v in vals)})
        report["cross_seed"] = cross

    with open(args.out_prefix + ".json", "w") as f:
        json.dump(report, f, indent=1)
    with open(args.out_prefix + ".md", "w") as f:
        f.write(render_md(report, interest))
    print(f"[report] wrote {args.out_prefix}.json and .md "
          f"({len(report['seeds'])} rotated table(s) vs baseline)")
    for s in report["seeds"]:
        c = s["control"]
        print(f"[report] {os.path.basename(s['rotated_table'])}: reached "
              f"signed {c['reached_mean_signed_dsigma_pct']} | not-reached "
              f"signed {c['not_reached_mean_signed_dsigma_pct']} | "
              f"flag={c['flag_not_reached_moved']}")


if __name__ == "__main__":
    main()
