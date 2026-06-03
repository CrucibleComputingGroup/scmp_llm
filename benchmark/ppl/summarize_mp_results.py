"""Summarize a run_mp_sweep.sh output directory into a comparison table.

Parses every ``Qwen_*<prec>[_<method>].log`` in the given sweep outdir, extracts
the FP16 baseline, the int8 uniform ceiling, and each method's SC PPL @ realized
avg_sl, and prints a per-model table (methods × precisions) with the best method
per cell marked. Pure stdlib; no GPU needed.

Usage:
    python benchmark/ppl/summarize_mp_results.py <outdir>
    python benchmark/ppl/summarize_mp_results.py benchmark/ppl/_mp_overnight_<tag>
"""
import os
import re
import sys
import glob

KNOWN_PRECS = ["int8", "len192", "int7", "len96"]   # check len192 before len96
KNOWN_METHODS = ["act_global", "grad_global", "measured", "act"]  # longest first
PREC_TARGET = {"int8": 128, "len192": 91, "int7": 64, "len96": 48}


def _parse_name(base):
    """Qwen_<model>_<prec>[_<method>] -> (model, prec, method|None)."""
    name = base[len("Qwen_"):] if base.startswith("Qwen_") else base
    for prec in KNOWN_PRECS:
        marker = "_" + prec
        idx = name.find(marker)
        if idx < 0:
            continue
        model = name[:idx]
        rest = name[idx + len(marker):]
        method = rest[1:] if rest.startswith("_") else None
        # normalize model label
        model = model.replace("Qwen3-", "").replace("-Instruct-2507", "")
        return model, prec, method
    return None, None, None


def _parse_log(path):
    txt = open(path).read()
    fp = re.search(r"FP16 baseline\s+([\d.]+)", txt)
    fp = fp.group(1) if fp else None
    # SC line: "...avg_sl=NN.N (halved)   PPL ... (×R vs fp16)"  (MP)
    #       or "SC sl=128 (halved) ...    PPL ... (×R vs fp16)"  (uniform int8)
    ppl = avg = None
    for ln in txt.splitlines():
        if "vs fp16" not in ln or "SC " not in ln:
            continue
        a = re.search(r"avg_sl=([\d.]+)", ln)
        p = re.search(r"\(halved\)[^0-9]*([\d.]+)", ln)
        if p:
            ppl = p.group(1)
            avg = a.group(1) if a else "128"   # uniform int8 has no avg_sl field
        break
    return fp, ppl, avg


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    outdir = sys.argv[1]
    logs = glob.glob(os.path.join(outdir, "Qwen_*.log"))
    if not logs:
        print(f"No Qwen_*.log files in {outdir}")
        sys.exit(1)

    data = {}     # (model, prec, method) -> (ppl, avg)
    fp16 = {}     # model -> fp16
    models, precs, methods = [], [], []
    for L in sorted(logs):
        base = os.path.basename(L)[:-4]
        model, prec, method = _parse_name(base)
        if model is None:
            continue
        fp, ppl, avg = _parse_log(L)
        if model not in models:
            models.append(model)
        if prec not in precs:
            precs.append(prec)
        if method and method not in methods:
            methods.append(method)
        if fp:
            fp16[model] = fp
        data[(model, prec, method)] = (ppl or "RUN", avg or "?")

    # canonical ordering
    models = [m for m in ["4B", "8B", "14B", "30B-A3B", "32B"] if m in models] + \
             [m for m in models if m not in ["4B", "8B", "14B", "30B-A3B", "32B"]]
    precs = [p for p in ["len192", "int7", "len96"] if p in precs]
    methods = [m for m in ["act", "act_global", "grad_global", "measured"] if m in methods]

    def fval(s):
        try:
            return float(s)
        except Exception:
            return float("inf")

    for model in models:
        int8 = data.get((model, "int8", None)) or data.get((model, "int8", "act"))
        ceil = f", int8={int8[0]}" if int8 else ""
        print(f"\n===== Qwen3-{model}  (FP16={fp16.get(model, '?')}{ceil}) =====")
        head = f"{'prec(target)':14s} " + " ".join(f"{m:>15s}" for m in methods)
        print(head)
        print("-" * len(head))
        for prec in precs:
            cells, best, bestv = [], None, float("inf")
            for meth in methods:
                d = data.get((model, prec, meth))
                cells.append(d)
                if d and fval(d[0]) < bestv:
                    bestv, best = fval(d[0]), meth
            row = f"{prec+'('+str(PREC_TARGET[prec])+')':14s} "
            for meth, d in zip(methods, cells):
                if not d:
                    txt = "-"
                else:
                    mark = "*" if meth == best else " "
                    txt = f"{mark}{d[0]}@{d[1]}"
                row += f"{txt:>16s}"
            print(row)
    print("\n  * = best method in that cell.  PPL @ realized avg_sl (halved cycles).")


if __name__ == "__main__":
    main()
