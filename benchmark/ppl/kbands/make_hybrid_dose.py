"""Emit a higher-dose hybrid INT mask that is a strict NESTED SUPERSET of a
cell's deployed mask.

WHY NESTED. The int_ablation dose curve is only interpretable because its masks
satisfy 5% subset 10% subset 20% off one sensitivity source. Re-ranking from
scratch at 30% would change WHICH entries are masked as well as how many, and
any PPL move would be unattributable. So we keep every entry the deployed 20%
mask already has and only ADD.

WHY WE DO NOT REPRODUCE THE ORIGINAL RANKING. The deployed method
(`top_fraction_by_bucket_worst_delta_loss`) is stratified; a global top-N on the
obvious scores reproduces only 72% of it, so guessing it would silently perturb
the base set. Taking the archived mask verbatim as the base sidesteps that
entirely -- we never need to know how it was ranked.

SCORING FOR THE ADDITIONS. `level_mean_error[-1] - level_mean_error[0]` from the
int8 measured-curve sensitivity: the extra reconstruction error an entry incurs
when its stream is shortened, i.e. how much it wants to be INT. Note a MIXED
BASIS on llama8B, whose deployed mask came from the `int_swap` source while its
additions are scored here on measured-curve; that is deliberate -- int_swap was
measured and does NOT beat measured_curve (2026-07-24), and measured_curve is
the only basis available for all four models. Recorded in `selection`.

  python make_hybrid_dose.py --model 4B --target 32 --fraction 0.30 --out <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RES = Path("/home/allenjin/Projects/hpca_results/llm")
SENS_DIR = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/"
                "_hpca_sens_layer_int8_20260709_012921")


def score_table(model: str) -> dict:
    """(op, layer) -> marginal error from shortening. Higher = wants INT more."""
    p = SENS_DIR / f"{model}_sc_int8_measured_curve.json"
    if not p.is_file():
        raise SystemExit(f"[dose] missing sensitivity source {p}")
    out = {}
    for k, e in json.load(open(p))["buckets"].items():
        op, _t, l = k.split(":")
        lme = e.get("level_mean_error") or []
        if len(lme) >= 2:
            out[(op, int(l[1:]))] = float(lme[-1]) - float(lme[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base_p = RES / "ppl" / "mp_best" / "configs" / a.model / f"target{a.target}" / "hybrid_config.json"
    hc = json.load(open(base_p))
    sched = {op: list(v) for op, v in hc["schedule"].items()}
    total = sum(len(v) for v in sched.values())
    base = {(op, i) for op, v in sched.items() for i, x in enumerate(v) if x != "sc"}
    want = int(round(a.fraction * total))
    if want <= len(base):
        raise SystemExit(f"[dose] target {want} <= deployed {len(base)}; this tool "
                         "only ADDS (nesting is the point)")

    scores = score_table(a.model)
    missing = [k for k in ((op, i) for op, v in sched.items() for i in range(len(v)))
               if k not in scores]
    if missing:
        # A silent partial score table would bias additions toward whatever the
        # sensitivity file happens to cover. Refuse.
        raise SystemExit(f"[dose] {len(missing)} (op,layer) entries have no score "
                         f"(e.g. {missing[:4]}); refusing to add on partial data")

    cands = sorted((k for k in scores if k not in base),
                   key=lambda k: -scores[k])[:want - len(base)]
    mark = hc.get("int_label") or "int7"
    for op, i in cands:
        sched[op][i] = mark
    got = {(op, i) for op, v in sched.items() for i, x in enumerate(v) if x != "sc"}
    assert base <= got, "additions must be a superset of the deployed mask"

    hc["schedule"] = sched
    hc["selection"] = {
        **hc.get("selection", {}),
        "fraction": a.fraction,
        "method": "deployed_base_plus_top_marginal_error",
        "base_from": str(base_p),
        "base_entries": len(base),
        "added_entries": len(cands),
        "selected_entries": len(got),
        "selected_fraction": len(got) / total,
        "total_entries": total,
        "addition_score": "level_mean_error[-1]-level_mean_error[0]",
        "addition_source": str(SENS_DIR / f"{a.model}_sc_int8_measured_curve.json"),
        "nested_superset_of_deployed": True,
        "mixed_basis_note": ("base mask ranking is whatever the deployed cell used "
                             "(llama8B = int_swap); additions are measured_curve"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(hc, open(a.out, "w"), indent=1)
    print(f"[dose] {a.model} t{a.target}: {len(base)} -> {len(got)} / {total} "
          f"({len(got)/total*100:.1f}%)  +{len(cands)}  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
