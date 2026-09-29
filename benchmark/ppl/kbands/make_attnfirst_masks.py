#!/usr/bin/env python3
"""Loss-priced hybrid masks: attention-first, ENTRY-matched to the archive mask.

Motivation (measured 2026-08-07, llama8B t96 op-swap wave): qk+av carry ~64%
of the PPL excess from 7.3% of MACs, while the deployed mask spends only 12/58
entries there (its selector ranks by an SC-fragility proxy). Recompose the
mask by measured loss instead.

DOSE INVARIANT = ENTRY COUNT (user decision 2026-08-08). The fixed 20% dose is
the fraction of (operator, block) MATMULS routed to INT -- what
`selection.fraction` has always recorded. Emits `<model>_t<T>_attnfirst_em.json`.
An earlier MAC-matched variant (`_mm`, which produced mp_best_after_hpca_4) held
the INT *MAC* share fixed instead and let the entry share drift to 27-33%; those
files are kept for provenance but the entry-matched form is the standard, and it
also leaves far MORE work on SC (~93.5% of MACs vs the archive's ~80%) because
attention matmuls are individually cheap.

Usage: python make_attnfirst_masks.py <model> <target> [--trace <t96 trace>]
Writes $TURBO/kbands/kbands_20260801/attnmask/<model>_t<target>_attnfirst.json
"""
import argparse, collections, json, pathlib

ARCH = pathlib.Path("/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best_after_hpca_3")
OUT = pathlib.Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/attnmask")
TRACES = {"4B": "4B_t96_prcqk_grid_trace.json", "llama8B": "llama8B_t96_prcqk_trace.json",
          "14B": "14B_t96_parent_grid_trace.json", "30B": "30B_t96_prcqk_trace.json"}


def op_macs_per_block(model):
    """MACs per block for each op, from a deployed trace (masked blocks are
    absent from the trace, so average over the blocks that DID run SC)."""
    t = json.load(open(ARCH / "traces" / TRACES[model]))
    nb = int(t["header"]["total_blocks"])
    tot = collections.defaultdict(float)
    blks = collections.defaultdict(set)
    for g in t["groups"]:
        tot[g["op"]] += g["macs"]
        blks[g["op"]].add(g["block"])
    return {op: tot[op] / max(len(blks[op]), 1) for op in tot}, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("target")
    a = ap.parse_args()
    src = ARCH / "configs" / a.model / f"target{a.target}" / "hybrid_config.json"
    cfg = json.loads(src.read_text())
    int_label = f"int{cfg['int_bits']}"
    arch_sched = {op: list(v) for op, v in cfg["schedule"].items()}
    avg, nb = op_macs_per_block(a.model)
    total = sum(avg.get(op, 0.0) * len(arch_sched[op]) for op in arch_sched)

    def share(sched):
        return sum(avg.get(op, 0.0) for op in sched
                   for x in sched[op] if x != "sc") / total

    target_share = share(arch_sched)
    sched = {op: ["sc"] * len(v) for op, v in arch_sched.items()}
    order = [("qk", b) for b in range(len(arch_sched["qk"]))]
    order += [("av", b) for b in reversed(range(len(arch_sched["av"])))]
    # then the archive's own remaining picks, in block order, MLP/proj only
    order += [(op, b) for op in arch_sched if op not in ("qk", "av")
              for b, x in enumerate(arch_sched[op]) if x != "sc"]

    # DOSE INVARIANT = ENTRY COUNT (user decision 2026-08-08): the fixed 20%
    # dose is defined as the fraction of (operator, block) matmuls routed to
    # INT, exactly as `selection.fraction` records it. MAC-matching instead
    # (the first version of this script) preserved INT *MACs* but let the entry
    # share drift to 27-33%, i.e. a third of matmuls bypassing SC instead of a
    # fifth. Entry-matching keeps the canonical dose AND leaves far MORE work on
    # SC (~93.5% of MACs vs the archive's ~80%), because attention matmuls are
    # individually cheap.
    n_entries = sum(1 for op in arch_sched for x in arch_sched[op] if x != "sc")
    used = 0
    for op, b in order:
        if used >= n_entries:
            break
        if sched[op][b] != "sc":
            continue
        sched[op][b] = int_label
        used += 1

    cfg["schedule"] = sched
    got = share(sched)
    cfg["selection"] = {
        "method": "attnfirst_entrymatched_20260808", "base": str(src),
        "fraction": round(used / sum(len(v) for v in sched.values()), 4),
        "entries": used, "archive_entries": sum(1 for op in arch_sched
                                                for x in arch_sched[op] if x != "sc"),
        "int_mac_share": round(got, 5), "archive_int_mac_share": round(target_share, 5),
        "note": "all qk, then av (late blocks first), then the archive's own "
                "remaining picks until the archive's INT MAC share is matched "
                "without exceeding it. Ranking motivated by the llama8B t96 "
                "op-swap wave (qk -1.75%, av -1.51%, projections ~0).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{a.model}_t{a.target}_attnfirst_em.json"
    dst.write_text(json.dumps(cfg, indent=1))
    qk_n = sum(1 for x in sched["qk"] if x != "sc")
    av_n = sum(1 for x in sched["av"] if x != "sc")
    print(f"{dst.name}: qk {qk_n}/{nb} av {av_n}/{nb} entries {used} "
          f"(archive {cfg['selection']['archive_entries']}) "
          f"INT MAC {100*got:.2f}% vs archive {100*target_share:.2f}%")


if __name__ == "__main__":
    main()
