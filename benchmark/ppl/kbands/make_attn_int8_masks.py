#!/usr/bin/env python3
"""qk/av -> INT8, every weight-bearing matmul -> SC.

A RULE, not a calibrated selection: the two attention products are the only
operators with no weights, hence the only ones no PTQ front-end can reach
(SmoothQuant and AWQ both stop at SCLinear) -- they run dynamic per-row absmax
with zero calibrated component, which is exactly why SC handles them worst
(measured: qk/av carry ~64% of llama8B's excess from 7.3% of its MACs).

Dose: 2 of 9 operators = 22.2% of matmuls, ~5-23% of MACs by model, so SC keeps
77-95% of the arithmetic -- MORE than the deployed mask's ~80%.
INT8 (not the deployed INT7) because these two operators are where the INT side
should be strongest and they are cheap in MACs.

  python make_attn_int8_masks.py <model> <target>
"""
import json, pathlib, sys

ARCH = pathlib.Path("/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best_after_hpca_3/configs")
OUT = pathlib.Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/attnint8")

def main():
    model, target = sys.argv[1], sys.argv[2]
    src = ARCH / model / f"target{target}" / "hybrid_config.json"
    cfg = json.loads(src.read_text())
    sched = {op: ["sc"] * len(v) for op, v in cfg["schedule"].items()}
    for op in ("qk", "av"):
        sched[op] = ["int8"] * len(sched[op])
    cfg["schedule"] = sched
    cfg["int_bits"] = 8
    n = sum(1 for op in sched for x in sched[op] if x != "sc")
    T = sum(len(v) for v in sched.values())
    cfg["selection"] = {"method": "attn_int8_rule_20260809", "base": str(src),
                        "fraction": round(n / T, 4), "entries": n, "total": T,
                        "note": "qk+av on INT8 (weightless A x A products, "
                                "unreachable by any PTQ front-end); all "
                                "weight-bearing matmuls on SC."}
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{model}_t{target}_attnint8.json"
    dst.write_text(json.dumps(cfg, indent=1))
    print(f"{dst.name}: qk+av = {n}/{T} matmuls ({100*n/T:.1f}%) at int8; rest SC")


if __name__ == "__main__":
    main()
