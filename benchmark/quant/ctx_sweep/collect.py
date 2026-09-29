#!/usr/bin/env python3
"""Collect the AWQ INT/fp16 context sweep into hpca_results/llm/ctx_sweep/.

Reads results_ctxsweep_awq_ctx{2048,4096,8192}_<DATE>.tsv from Turbo and emits
per-model CSVs plus SUMMARY.md.

Two things this enforces rather than assumes:

1. **ctx-2048 reproduction gate.** Eval is bit-deterministic, so every ctx-2048
   cell must reproduce a frozen reference exactly: fp16 against
   hpca_results/llm/int/<model>.csv, and INT8/INT6 -- taken as the LOWER of the
   symmetric and asymmetric run, which is model_quality.tex's convention --
   against the awq_ppl column of frontend_awq/int_awq_vs_smoothquant.csv.
   A mismatch means the harness or the AWQ scale cache moved, so this RAISES
   instead of writing an unverified table.
2. **No cross-ctx PPL deltas.** PPL is not comparable across context lengths
   (longer windows mechanically lower it, and eval_quant.py:505 truncates to
   (N//ctx)*ctx so the token set differs). The only derived quantity emitted is
   ratio-to-fp16 AT THE SAME ctx.

Usage:  python collect.py [--date 20260730] [--allow-partial]
"""
import argparse
import csv
import pathlib
import sys

TURBO = pathlib.Path("/nfs/turbo/coe-nbleier/allenjin/hpca/results")
REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO.parent / "hpca_results" / "llm" / "ctx_sweep"
REF = REPO.parent / "hpca_results" / "llm" / "int"
AWQ_REF = (REPO.parent / "hpca_results" / "llm" / "frontend_awq"
           / "int_awq_vs_smoothquant.csv")

MODELS = ["4B", "llama8B", "14B", "30B"]
CONFIGS = ["fp16", "W8A8_symm", "W8A8_asymm", "W6A6_symm", "W6A6_asymm"]
# Paper convention (model_quality.tex): the INT row at a given precision is the
# LOWER of the symmetric and asymmetric run.
INT_PRECS = [8, 6]
CTXS = [2048, 4096, 8192]
TOL = 5e-5


def load_tsv(path):
    """(model, config) -> value. Skips header rows and FAILED cells."""
    out, failed = {}, []
    if not path.exists():
        return out, failed
    for line in path.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 4 or parts[0] == "model":
            continue
        model, config, metric, value = parts
        if metric != "ppl":
            continue
        if value.startswith("FAILED"):
            failed.append((model, config, value))
            continue
        out[(model, config)] = float(value)
    return out, failed


def load_fp16_ref():
    """model -> frozen fp16 PPL from hpca_results/llm/int/."""
    out = {}
    for model in MODELS:
        path = REF / f"{model}.csv"
        if not path.exists():
            continue
        for r in csv.DictReader(path.open()):
            if r["config"] == "fp16" and r["metric"] == "ppl":
                out[model] = float(r["value"])
    return out


def load_awq_ref():
    """(model, prec) -> frozen AWQ PPL (already the lower of symm/asymm)."""
    if not AWQ_REF.exists():
        return {}
    return {(r["model"], int(r["int_prec"])): float(r["awq_ppl"])
            for r in csv.DictReader(AWQ_REF.open())}


def int_row(cells, model, prec):
    """Lower of symm/asymm at this precision, or None if neither ran."""
    vals = [cells.get((model, f"W{prec}A{prec}_{v}")) for v in ("symm", "asymm")]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260730")
    ap.add_argument("--tag-prefix", default="ctxsweep_awq")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write what exists instead of requiring all 60 cells")
    args = ap.parse_args()

    data, failures = {}, []
    for ctx in CTXS:
        path = TURBO / f"results_{args.tag_prefix}_ctx{ctx}_{args.date}.tsv"
        data[ctx], failed = load_tsv(path)
        failures += [(ctx, *f) for f in failed]

    total = len(MODELS) * len(CONFIGS) * len(CTXS)
    missing = [(m, c, ctx) for ctx in CTXS for m in MODELS for c in CONFIGS
               if (m, c) not in data[ctx]]
    if failures:
        print("[collect] FAILED cells:", file=sys.stderr)
        for ctx, m, c, why in failures:
            print(f"  ctx={ctx} {m}/{c}: {why}", file=sys.stderr)
    if missing and not args.allow_partial:
        raise SystemExit(
            f"[collect] {len(missing)}/{total} cells missing (jobs still "
            f"running?). First: {missing[:5]}. Re-run with --allow-partial to "
            "write an incomplete table.")

    # --- gate: ctx-2048 must reproduce the frozen references exactly ----------
    fp16_ref, awq_ref = load_fp16_ref(), load_awq_ref()
    drift, checked = [], 0
    for model in MODELS:
        got, want = data[2048].get((model, "fp16")), fp16_ref.get(model)
        if got is not None and want is not None:
            checked += 1
            if abs(got - want) > TOL:
                drift.append((model, "fp16", got, want, "llm/int"))
        for prec in INT_PRECS:
            got, want = int_row(data[2048], model, prec), awq_ref.get((model, prec))
            if got is None or want is None:
                continue
            checked += 1
            if abs(got - want) > TOL:
                drift.append((model, f"INT{prec}", got, want,
                              "frontend_awq/int_awq_vs_smoothquant.csv"))
    if drift:
        for model, label, got, want, src in drift:
            print(f"[collect] DRIFT {model}/{label}: ctx2048 got {got:.4f}, "
                  f"{src} has {want:.4f}", file=sys.stderr)
        raise SystemExit(
            "[collect] ctx-2048 cells do not reproduce the frozen references. "
            "Eval is bit-deterministic, so this means the harness or the AWQ "
            "scale cache changed -- the sweep is not comparable to the paper's "
            "INT anchors. Refusing to write.")
    print(f"[collect] ctx-2048 reproduction gate: {checked} references match "
          f"exactly (fp16 vs llm/int, INT8/6 vs AWQ reference)")

    # --- per-model CSVs: every raw cell, plus the derived paper-convention row -
    OUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        rows = []
        for ctx in CTXS:
            fp16 = data[ctx].get((model, "fp16"))
            for config in CONFIGS:
                ppl = data[ctx].get((model, config))
                if ppl is None:
                    continue
                rows.append({
                    "model": model, "config": config, "ctx": ctx,
                    "ppl": f"{ppl:.4f}",
                    # ratio is AT THE SAME ctx -- the only comparison this
                    # table supports.
                    "ratio_to_fp16_same_ctx":
                        f"{ppl / fp16:.4f}" if fp16 else "",
                })
            for prec in INT_PRECS:
                best = int_row(data[ctx], model, prec)
                if best is None:
                    continue
                rows.append({
                    "model": model, "config": f"INT{prec}", "ctx": ctx,
                    "ppl": f"{best:.4f}",
                    "ratio_to_fp16_same_ctx":
                        f"{best / fp16:.4f}" if fp16 else "",
                })
        if not rows:
            continue
        with (OUT / f"{model}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[collect] wrote {OUT / f'{model}.csv'} ({len(rows)} rows)")

    lines = [
        "# INT / FP16 perplexity vs. evaluation context length",
        "",
        f"Tag `{args.tag_prefix}_ctx{{2048,4096,8192}}_{args.date}`. Full "
        "WikiText-2 test, stride = ctx, `PPL_MAX_TOKENS=0`, **AWQ front-end**.",
        "",
        "**Read this table DOWN columns, never across rows.** Perplexity is not "
        "comparable across context lengths: longer windows mechanically lower "
        "PPL (more conditioning per token), and `eval_quant.py:505` truncates "
        "the stream to `(N//ctx)*ctx`, so the evaluated token set differs by "
        "ctx (146 windows at 2048 vs. 36 at 8192). The comparable quantity is "
        "`ratio_to_fp16_same_ctx`, shown in parentheses.",
        "",
        "INT rows are the **lower of the symmetric and asymmetric** run at each "
        "precision, under the AWQ front-end -- `model_quality.tex`'s convention, "
        "so these cells are front-end- and variant-matched to the paper's INT "
        "anchors. Per-variant numbers are in the per-model CSVs.",
        "",
        "AWQ scales are loaded from `act_scales/awq_scales/*_b4.pt`; they are "
        "calibration-derived and therefore identical at every eval ctx. fp16 is "
        "front-end-invariant (`eval_quant.py` returns plain HF before the "
        "SmoothQuant/AWQ branch). All cells run eager attention "
        "(`BASELINE_ATTN`), which INT requires so QK/AV are fake-quantized.",
        "",
        "| model | row | " + " | ".join(f"ctx {c}" for c in CTXS) + " |",
        "|---|---|" + "---|" * len(CTXS),
    ]
    for model in MODELS:
        for label in ["fp16"] + [f"INT{p}" for p in INT_PRECS]:
            cells = []
            for ctx in CTXS:
                fp16 = data[ctx].get((model, "fp16"))
                ppl = (fp16 if label == "fp16"
                       else int_row(data[ctx], model, int(label[3:])))
                if ppl is None:
                    cells.append("--")
                elif label == "fp16" or not fp16:
                    cells.append(f"{ppl:.4f}")
                else:
                    cells.append(f"{ppl:.4f} ({ppl / fp16:.3f}x)")
            lines.append(f"| {model} | {label} | " + " | ".join(cells) + " |")
    if missing:
        lines += ["", f"INCOMPLETE: {len(missing)}/{total} cells missing at "
                      "write time."]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"[collect] wrote {OUT / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
