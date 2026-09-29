"""Paired held-out TRAIN-set NLL of several MP tables on one loaded SC model.

A CALIBRATION-TIME selection signal, never a reported PPL: wikitext-2 TRAIN windows
drawn stratified with a seed different from the calibration windows (overlaps with the
calibration set are dropped), ctx 2048, identical windows for every table (paired).
Tables are swapped by reloading the MP config onto the same model
(loader.apply_mp_config_from_env), so model load, INT mask, front end and numerics are
shared and only the allocation differs. Evals are bit-deterministic, so the paired
per-window differences carry no run-to-run noise -- only window-sampling spread.

  python benchmark/ppl/heldout_nll.py --model_path <hf> --hybrid <hybrid_config.json> \
      --tables parent=<wrapper>,c17=<wrapper>,g5=<wrapper> --windows 16 --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")
REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--hybrid", required=True, help="parent hybrid_config.json (INT mask)")
    ap.add_argument("--tables", required=True, help="name=wrapper,name=wrapper,... "
                    "(first = reference for paired differences)")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--calib-n", default="8",
                    help="calibration window counts (comma list; each drawn stratified with "
                         "seed --calib-seed, as the calibrators do) whose starts are excluded")
    ap.add_argument("--frontend", default="awq")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tables = [t.split("=", 1) for t in args.tables.split(",")]
    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = tables[0][1]
    os.environ["SC_HYBRID_CONFIG_JSON"] = args.hybrid
    os.environ["FRONTEND"] = args.frontend
    os.environ.setdefault(
        "ACT_SCALES_DIR",
        "/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales")

    from benchmark.quant.eval_quant import build_sc_model
    from benchmark.ppl.calibrate_mp_thresholds import _select_int_swap_windows
    from loader import apply_mp_config_from_env
    from datasets import load_dataset

    model, tok = build_sc_model(args.model_path, "mp", mp_table=tables[0][1])
    model.eval()
    dev = next(model.parameters()).device
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    calib_set = set()
    for n_ in [int(v) for v in str(args.calib_n).split(",") if v]:
        _, calib_starts = _select_int_swap_windows(enc, 2048, n_, sampling="stratified",
                                                   seed=args.calib_seed)
        calib_set |= set(int(s) for s in calib_starts)
    wins, starts = _select_int_swap_windows(enc, 2048, args.windows + len(calib_set),
                                            sampling="stratified", seed=args.seed)
    keep = [(w, s) for w, s in zip(wins, starts) if int(s) not in calib_set][:args.windows]
    print(f"[hnll] {len(keep)} held-out train windows (excluded {len(calib_set)} calib starts)")

    res = {"windows": [int(s) for _, s in keep], "tables": {}}
    for name, wrapper in tables:
        os.environ["MP_CONFIG_JSON"] = wrapper
        model.config.sc_mp_config = None
        apply_mp_config_from_env(model)
        if getattr(model.config, "sc_mp_config", None) is None:
            raise SystemExit(f"[hnll] table did not load: {wrapper}")
        from model.sc_common import SCLinear
        m0 = next(m for m in model.modules() if isinstance(m, SCLinear))
        if getattr(m0._sc_config, "sc_mp_config", None) is not model.config.sc_mp_config:
            raise SystemExit("[hnll] SCLinear does not see the swapped table")
        nprc = len(getattr(model.config.sc_mp_config, "per_row_chunk", {}) or {})
        print(f"[hnll] loaded {name}: per_row_chunk buckets {nprc}")
        from model.sc_common import mp_tracker_reset, mp_tracker_flop_avg_stoc_len
        mp_tracker_reset()
        losses = []
        with torch.no_grad():
            for w, _s in keep:
                ids = w.unsqueeze(0).to(dev)
                losses.append(float(model(input_ids=ids, labels=ids).loss))
        cost = float(mp_tracker_flop_avg_stoc_len())
        res["tables"][name] = {"wrapper": wrapper, "window_nll": losses,
                               "mean_nll": sum(losses) / len(losses), "cost": cost}
        print(f"[hnll] {name:10s} mean NLL {res['tables'][name]['mean_nll']:.5f} "
              f"(PPL {math.exp(res['tables'][name]['mean_nll']):.4f})  cost {cost:.3f}")
    ref = tables[0][0]
    rl = res["tables"][ref]["window_nll"]
    for name, _ in tables[1:]:
        d = [a - b for a, b in zip(res["tables"][name]["window_nll"], rl)]
        m = sum(d) / len(d)
        sd = (sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5
        se = sd / math.sqrt(len(d))
        res["tables"][name]["paired_vs_" + ref] = {
            "mean_dnll": m, "se": se, "z": m / se if se > 0 else float("inf"),
            "dppl_pct": (math.exp(m) - 1) * 100}
        print(f"[hnll] {name:10s} vs {ref}: dNLL {m:+.5f} ± {se:.5f} (z {m / se if se else 0:+.1f}), "
              f"≈ {(math.exp(m) - 1) * 100:+.2f}% PPL")
    names = [n for n, _ in tables]
    for i in range(1, len(names)):
        for j in range(i + 1, len(names)):
            a, b = res["tables"][names[j]]["window_nll"], res["tables"][names[i]]["window_nll"]
            d = [x - y for x, y in zip(a, b)]
            m = sum(d) / len(d)
            sd = (sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5
            se = sd / math.sqrt(len(d))
            res.setdefault("pairs", {})[f"{names[j]}-{names[i]}"] = {"mean_dnll": m, "se": se,
                                                                     "z": m / se if se > 0 else 0.0}
            print(f"[hnll] {names[j]} vs {names[i]}: dNLL {m:+.5f} ± {se:.5f}")
    Path(args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
