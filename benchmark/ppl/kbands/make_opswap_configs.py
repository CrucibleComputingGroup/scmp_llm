#!/usr/bin/env python3
"""Per-op INT7-swap diagnostic configs (2026-08-07, user-approved wave).

For one archived cell's hybrid config, emit one variant per operator with that
operator's ENTIRE schedule routed to int7 (all blocks), everything else
untouched. Running the winner arm with such a config measures, in LOSS units,
how much of the cell's PPL excess that operator's SC error carries — the
attribution sigma cannot give (llama8B converts sigma to loss ~20x the Qwens).

NOT a deployable cell: swapping an op to INT7 adds INT MACs outside the SC
budget (energy axis). Diagnostic only; tag carries `swap_<op>` so the archive
builder never picks these up (job names must not start with prcppl_/grid_).

Usage: python make_opswap_configs.py <model> <target> [ops...]
Writes: $TURBO/kbands/kbands_20260801/opswap/<model>_t<target>_swap_<op>.json
"""
import json, sys, pathlib

ARCH = pathlib.Path("/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best_after_hpca_3/configs")
OUT = pathlib.Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/opswap")
ALL_OPS = ["av", "qk", "down_proj", "up_proj", "gate_proj", "o_proj",
           "q_proj", "k_proj", "v_proj"]

def main():
    model, target = sys.argv[1], sys.argv[2]
    ops = sys.argv[3:] or ALL_OPS
    src = ARCH / model / f"target{target}" / "hybrid_config.json"
    base = json.loads(src.read_text())
    int_label = f"int{base['int_bits']}"
    OUT.mkdir(parents=True, exist_ok=True)
    for op in ops:
        cfg = json.loads(src.read_text())
        n = len(cfg["schedule"][op])
        already = sum(1 for s in cfg["schedule"][op] if s != "sc")
        cfg["schedule"][op] = [int_label] * n
        cfg["selection"] = {
            "method": f"opswap_diagnostic({op})", "base": str(src),
            "note": f"op {op}: {already}/{n} blocks were already {int_label}; "
                    f"now {n}/{n}. All other ops untouched.",
        }
        dst = OUT / f"{model}_t{target}_swap_{op}.json"
        dst.write_text(json.dumps(cfg, indent=1))
        print(f"{dst.name}: {op} {already}/{n} -> {n}/{n} {int_label}")

if __name__ == "__main__":
    main()
