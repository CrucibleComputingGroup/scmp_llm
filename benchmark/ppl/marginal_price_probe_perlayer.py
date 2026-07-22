#!/usr/bin/env python
"""Per-LAYER-INDEX marginal-value probe (V17) — refinement of the 4-bucket probe.

The 4-bucket probe resolves l0..l3 (9 layers each) — too coarse to trust an
"early vs late" conclusion. This one resolves EVERY layer. layer_buckets=36 is a
clean bijection (L->L), so we:
  1. un-bucket V9's 4-layer-bucket table into a 36-layer table (each layer copies
     its parent 4-bucket's thresholds) — behaviorally IDENTICAL to V9;
  2. re-profile it (eval writes per-layer metric histograms) + get baseline NLL;
  3. for each layer L, promote layer L's MLP off the floor by a small mass and
     measure the paired dNLL / d_cost = the MEASURED marginal value of a cycle
     spent at that specific layer.
Output: a per-layer marginal-value curve. Answers whether it's a smooth
early->late gradient, specific hot layers, or noise — at full layer resolution.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path("/home/allenjin/Projects/scmp_llm")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "kernels"))
from scmp_kernels.mp.config import _bucket_index  # noqa: E402
from benchmark.ppl.mp_v16_refine import (  # noqa: E402
    _build_parent_model, _load_eval_stream, eval_windows, paired_deltas,
)
from benchmark.ppl.mp_ladder_refine import (  # noqa: E402
    _atomic_json, _protected_info, _write_wrapper, weighted_cost,
)
from benchmark.ppl.mp_joint_refine import (  # noqa: E402
    _hist_values, _shift_one_boundary, profile_level_weights,
)

MLP = ("gate_proj", "up_proj", "down_proj")


def exp_cost(table, profile):
    psl, pw = _protected_info(table)
    lw = profile_level_weights(table, profile, protected_weight=pw)
    return weighted_cost(table["stoc_len_levels"], lw, pw, psl or 0)


def unbucket_to_per_layer(table, n_layers):
    """4-layer-bucket table -> per-layer (n_layers) table, behavior-identical.

    Each layer L copies the thresholds of its parent 4-bucket
    _bucket_index(L, n_layers, 4). New layer_buckets = n_layers, so the runtime
    maps L->L (bijection) and every layer keeps its original dispatch."""
    old_lb = int(table.get("layer_buckets", 4))
    out = copy.deepcopy(table)
    out["layer_buckets"] = n_layers
    new_buckets = {}
    # collect ops/timesteps present
    ops = sorted({k.split(":")[0] for k in table["buckets"]})
    tbs = sorted({k.split(":")[1] for k in table["buckets"]})
    for op in ops:
        for tb in tbs:
            for L in range(n_layers):
                parent_lb = _bucket_index(L, n_layers, old_lb)
                src = table["buckets"].get(f"{op}:{tb}:l{parent_lb}")
                if src is None:
                    continue
                new_buckets[f"{op}:{tb}:l{L}"] = copy.deepcopy(src)
    out["buckets"] = new_buckets
    return out


def promote_layer_mlp(table, profile, layer, mass_delta):
    out = copy.deepcopy(table)
    values = _hist_values(profile)
    changed = False
    for key, group in (profile.get("groups") or {}).items():
        if str(group["op"]) not in MLP or group.get("l_bucket") != layer:
            continue
        payload = (out.get("buckets") or {}).get(key)
        if payload is None:
            continue
        hist = [float(x) for x in group["mac_weighted_hist"]]
        shifted, mv = _shift_one_boundary(
            [float(x) for x in payload["thresholds"]], hist, values,
            boundary=-1, mass_delta=+mass_delta)
        payload["thresholds"] = shifted
        changed = changed or mv.get("changed", False)
    return out, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--mass-delta", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    v9 = json.load(open(args.table))
    levels = [int(x) for x in v9["stoc_len_levels"]]
    perlayer = unbucket_to_per_layer(v9, args.n_layers)
    base_tab = outdir / "perlayer_table.json"
    base_wrap = outdir / "perlayer_wrapper.json"
    _atomic_json(base_tab, perlayer)
    _write_wrapper(base_wrap, levels, base_tab)

    model, tok = _build_parent_model(args.model_path, base_wrap, args.alpha)
    enc = _load_eval_stream(tok, split="validation",
                            max_tokens=args.ctx * (args.n_windows + 2),
                            ctx=args.ctx)
    windows = list(range(args.n_windows))

    # baseline eval on the un-bucketed table WRITES the per-layer profile
    prof_path = outdir / "perlayer_profile.json"
    base = eval_windows(model, enc, windows, ctx=args.ctx, levels=levels,
                        table_path=base_tab, wrapper_path=base_wrap,
                        trace_path=outdir / "base_trace.json",
                        split_label="perlayer_baseline",
                        profile_path=prof_path)
    base_nll = base["window_nll"]
    profile = json.load(open(prof_path))
    base_cost = exp_cost(perlayer, profile)
    print(f"[perlayer] un-bucketed baseline ppl={base['ppl']:.5f} "
          f"cost={base_cost:.4f} (must match V9 4B ~13.78/31.31)", flush=True)

    results = []
    for L in range(args.n_layers):
        cand, changed = promote_layer_mlp(perlayer, profile, L, args.mass_delta)
        if not changed:
            results.append({"layer": L, "changed": False, "marginal_value": None})
            print(f"[perlayer] L{L:02d}: no floor mass", flush=True)
            continue
        ct = outdir / f"cand_L{L}.json"; cw = outdir / f"cand_L{L}_wrapper.json"
        _atomic_json(ct, cand); _write_wrapper(cw, levels, ct)
        d_cost = exp_cost(cand, profile) - base_cost
        ev = eval_windows(model, enc, windows, ctx=args.ctx, levels=levels,
                          table_path=ct, wrapper_path=cw,
                          trace_path=outdir / f"cand_L{L}_trace.json",
                          split_label=f"perlayer_L{L}")
        deltas = paired_deltas(ev["window_nll"], base_nll, windows)
        mean_d = sum(deltas) / len(deltas)
        marginal = (-mean_d / d_cost) if d_cost > 1e-9 else None
        results.append({"layer": L, "changed": True, "mean_dNLL": mean_d,
                        "d_cost": d_cost, "marginal_value": marginal})
        print(f"[perlayer] L{L:02d}: dNLL={mean_d:+.6f} dcost={d_cost:+.4f} "
              f"marginal={marginal if marginal is None else round(marginal,6)}",
              flush=True)
        for p in (ct, cw):
            try: p.unlink()
            except OSError: pass

    _atomic_json(outdir / "marginal_price_perlayer.json",
                 {"model": args.model_path, "base_ppl": base["ppl"],
                  "base_cost": base_cost, "n_layers": args.n_layers,
                  "mass_delta": args.mass_delta, "results": results})

    valid = [r for r in results if r.get("marginal_value") is not None]
    print("\n[perlayer] per-layer MLP marginal value (high = cycle here buys more NLL):")
    for r in valid:
        bar = "#" * max(0, int(r["marginal_value"] * 2000))
        print(f"  L{r['layer']:02d} {r['marginal_value']:+.6f} {bar}")
    half = args.n_layers // 2
    early = [r["marginal_value"] for r in valid if r["layer"] < half]
    late = [r["marginal_value"] for r in valid if r["layer"] >= half]
    e = sum(early) / len(early) if early else 0
    la = sum(late) / len(late) if late else 0
    print(f"\n[perlayer] early(<{half}) mean={e:+.6f}  late(>={half}) mean={la:+.6f}")
    print(f"[perlayer] {'LATE>early (matches ground truth)' if la > e else 'early>=late (does NOT match)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
