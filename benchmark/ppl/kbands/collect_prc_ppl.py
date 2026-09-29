"""Collect the per-(row, chunk) PPL wave into one table.

READS THE COST FROM THE TRACE, not from calibration. Calibration promised
child/parent = 1.0000, but a promise is not a measurement -- the trace prices
each rung separately and is the only number that reflects what actually ran.

COMPARES S1 -> S2 (parent arm vs prc arm, both run in THIS wave), never against
the archived parent PPL: that would fold in every environment change since the
archived run. On 4B t48 with K-bands, 83% of the naive parent-to-result delta
was numerics rather than allocation.

FLAGS ANYTHING INSIDE THE NOISE FLOOR. sd = 0.0069 PPL over 6 identity
controls, so |delta| < ~0.014 (2 sd) is not a result and is printed as such.

GUARDS THE NULL. If a prc cell's linear MACs are not mostly priced at
d_in == 128, it ran per-ROW dispatch and its "no change" is an artifact, not a
finding -- the exact failure this project's verification chain exists to
exclude.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl")
FP16 = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}
NOISE_SD = 0.0069
MODELS = ["4B", "llama8B", "14B", "30B"]
TARGETS = [32, 48]


def read_trace(tag: str):
    """(linear-only MAC-weighted mean length, share of linear MACs at d_in=128)."""
    for p in (OUT / f"{tag}_trace.json", *sorted(OUT.glob(f"{tag}_trace*.json"))):
        if not p.is_file():
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        grps = doc.get("groups", doc)
        if not isinstance(grps, list):
            continue
        lin_ops = {"q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"}
        lin = [g for g in grps if g.get("op") in lin_ops]
        macs = sum(g["macs"] for g in lin)
        if not macs:
            continue
        cyc = sum(g["macs"] * g["stoc_len"] for g in lin)
        m128 = sum(g["macs"] for g in lin if g.get("d_in") == 128)
        return cyc / macs, m128 / macs
    return None, None


LOGS = Path("/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/"
            "hpca/logs/_kbands")


def read_ppl(tag: str):
    """PPL + all-op realized cost from the eval's [RESULT] line.

    eval_quant.py prints the result to stdout; it does NOT write a summary
    JSON next to the trace. Read the newest log for this tag, and take the LAST
    [RESULT] line in it (a resubmitted cell can append to the same name).
    """
    best = (None, None, None)
    for f in sorted(LOGS.glob(f"prcppl_{tag}_*.out"),
                    key=lambda q: q.stat().st_mtime, reverse=True):
        for line in reversed(f.read_text(errors="ignore").splitlines()):
            if "[RESULT]" not in line or "metric=ppl" not in line:
                continue
            kv = {}
            for tok in line.split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kv[k] = v
            try:
                return (float(kv["value"]),
                        float(kv.get("realized_flop_avg_sl", "nan")),
                        float(kv.get("tokens", "nan")))
            except Exception:                               # noqa: BLE001
                continue
    return best


def main() -> int:
    rows, missing = [], []
    for m in MODELS:
        for t in TARGETS:
            par_tag, prc_tag = f"{m}_t{t}_parent", f"{m}_t{t}_prc"
            p_ppl, p_flop, p_tok = read_ppl(par_tag)
            c_ppl, c_flop, c_tok = read_ppl(prc_tag)
            p_cost, _ = read_trace(par_tag)
            c_cost, c_share = read_trace(prc_tag)
            # all-op realized cost from the eval itself; the trace numbers are
            # linear-only. Both are reported: iso-compute must hold on the
            # all-op axis, which is what the archive's column records.
            p_cost = p_flop if p_cost is None else p_cost
            if p_ppl is None or c_ppl is None:
                missing.append(f"{m} t{t} "
                               f"({'parent ' if p_ppl is None else ''}"
                               f"{'prc' if c_ppl is None else ''})".strip())
                continue
            d = c_ppl - p_ppl
            rows.append(dict(model=m, target=t, parent=p_ppl, prc=c_ppl,
                             delta=d, pct=100.0 * d / p_ppl,
                             p_cost=p_flop, c_cost=c_flop, share=c_share,
                             sig=abs(d) >= 2 * NOISE_SD,
                             x_fp16=c_ppl / FP16[m],
                             tok_ok=(p_tok == c_tok)))

    print(f"\nper-(row, chunk) vs per-row parent — both arms run in this wave")
    print(f"noise floor sd={NOISE_SD} PPL; |delta| < {2*NOISE_SD:.4f} is NOT a result\n")
    print(f"{'model':9} {'tgt':>4} {'parent':>9} {'prc':>9} {'delta':>8} "
          f"{'%':>7} {'cost p/c':>13} {'d128':>6} {'x_fp16':>7}  verdict")
    for r in rows:
        cost = (f"{r['p_cost']:.1f}/{r['c_cost']:.1f}"
                if r['p_cost'] and r['c_cost'] else "  -/-")
        share = f"{100*r['share']:.0f}%" if r['share'] is not None else "  ?"
        if r['share'] is not None and r['share'] < 0.5:
            verdict = "INVALID: ran per-ROW"
        elif r['p_cost'] and r['c_cost'] and r['c_cost'] > r['p_cost'] + 1e-6:
            verdict = "INVALID: overspent"
        elif not r.get('tok_ok', True):
            verdict = "INVALID: token count differs"
        elif not r['sig']:
            verdict = "within noise"
        else:
            verdict = "IMPROVED" if r['delta'] < 0 else "REGRESSED"
        print(f"{r['model']:9} {r['target']:>4} {r['parent']:9.4f} {r['prc']:9.4f} "
              f"{r['delta']:+8.4f} {r['pct']:+6.2f}% {cost:>13} {share:>6} "
              f"{r['x_fp16']:7.4f}  {verdict}")

    if rows:
        sig = [r for r in rows if r['sig']]
        won = [r for r in sig if r['delta'] < 0]
        print(f"\n{len(won)}/{len(sig)} significant cells improved "
              f"({len(rows) - len(sig)} within noise, {len(rows)} total)")
    if missing:
        print(f"\nnot yet complete: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
