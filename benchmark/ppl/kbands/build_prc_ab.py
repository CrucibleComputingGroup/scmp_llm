"""Build the per-row (parent) vs per-(row,chunk) A/B table from [RESULT] lines.

Pure-allocation comparison: both arms share wrapper, INT mask, attention
thresholds and AWQ front end; only the linear projections' length assignment
differs. Reports raw dPPL, realized SC-cost change, x_fp16, and the share of
the parent's excess over fp16 that the child removes.

  python build_prc_ab.py --harvest <harvested.txt> [--child-tag prc] [--extra <glob of .out>]
"""
import argparse, glob, re, statistics as st
FP16 = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}
RX = re.compile(r"(?:^|/)(?:prcppl|[A-Za-z0-9]+)_(4B|llama8B|14B|30B)_t(\d+)_(\w+?)_\d+\.out"
                r".*?value=([\d.]+).*?realized_flop_avg_sl=([\d.]+)")

def parse(lines):
    d = {}
    for l in lines:
        m = RX.search(l)
        if m:
            mod, t, arm, v, c = m.groups()
            d[(mod, int(t), arm)] = (float(v), float(c))
    return d

ap = argparse.ArgumentParser()
ap.add_argument("--harvest", required=True)
ap.add_argument("--child", default="prc")
ap.add_argument("--parent", default="parent")
ap.add_argument("--extra", default="")
a = ap.parse_args()
lines = open(a.harvest).read().splitlines()
if a.extra:
    for f in sorted(glob.glob(a.extra)):
        for l in open(f):
            if "[RESULT]" in l:
                lines.append(f + "|" + l.strip())
d = parse(lines)
rows = []
print(f"| model | T | parent PPL @cost | {a.child} PPL @cost | dPPL | dcost | x_fp16 par->child | excess removed |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for mod in ("4B", "llama8B", "14B", "30B"):
    for t in (32, 40, 48, 64, 96):
        p, c = d.get((mod, t, a.parent)), d.get((mod, t, a.child))
        if not (p and c):
            continue
        dp = (c[0] / p[0] - 1) * 100
        dc = (c[1] / p[1] - 1) * 100
        xp, xc = p[0] / FP16[mod], c[0] / FP16[mod]
        exr = (p[0] - c[0]) / (p[0] - FP16[mod]) * 100 if p[0] > FP16[mod] else float("nan")
        rows.append((mod, t, dp, dc, exr))
        print(f"| {mod} | {t} | {p[0]:.4f} @{p[1]:.2f} | {c[0]:.4f} @{c[1]:.2f} | {dp:+.2f}% | {dc:+.1f}% | {xp:.4f}->{xc:.4f} | {exr:.1f}% |")
if rows:
    dps = [r[2] for r in rows]
    print(f"\n{len(rows)} cells: mean dPPL {st.mean(dps):+.2f}%, median {st.median(dps):+.2f}%, "
          f"improved {sum(x < 0 for x in dps)}/{len(dps)}; mean dcost {st.mean(r[3] for r in rows):+.2f}%")
