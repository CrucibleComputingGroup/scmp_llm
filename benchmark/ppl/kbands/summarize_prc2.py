"""One table for the prc2 study: parent (per-row, submitted) vs v7 prc vs prc2 arms.

Reads [RESULT] lines (harvested August file + scratch logs of p2_/p3_/p2ppl_ jobs) and
the prc2 held-out diag JSONs. Columns: PPL@realized SC cost, dPPL, dcost, share of the
parent's fp16 excess removed, held-out linear err ratio, and the linear budget scale at
which prc2 matches the parent's held-out error (interpolated from the sweep).
"""
import glob, json, re, os
FP16 = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}
HARV = "/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/harvested_results_20260920.txt"
LOGS = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands"
P2 = "/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2"
RX = re.compile(r"value=([\d.]+).*?realized_flop_avg_sl=([\d.]+)")
res = {}
for l in open(HARV):
    m = re.match(r"prcppl_(4B|llama8B|14B|30B)_t(\d+)_(parent|prc)_\d+\.out", l)
    r = RX.search(l)
    if m and r:
        res[(m[1], int(m[2]), "v7" if m[3] == "prc" else "parent")] = (float(r[1]), float(r[2]))
for f in glob.glob(f"{LOGS}/p*_*.out"):
    b = os.path.basename(f)
    m = re.match(r"(p2|p3|p2ppl)_(4B|llama8B|14B|30B)_t(\d+)_(\w+?)_\d+\.out", b)
    if not m:
        continue
    for l in open(f):
        if "[RESULT]" in l:
            r = RX.search(l)
            res[(m[2], int(m[3]), m[4])] = (float(r[1]), float(r[2]))

def iso_scale(diag):
    sw = diag.get("sweep_total") or {}
    pts = sorted((float(k), v["err_over_par"], v["L_over_par"]) for k, v in sw.items())
    for (s0, e0, l0), (s1, e1, l1) in zip(pts, pts[1:]):
        if (e0 - 1) * (e1 - 1) <= 0 and e0 != e1:
            w = (1 - e0) / (e1 - e0)
            return l0 + w * (l1 - l0)
    return None

arms = sorted({k[2] for k in res} - {"parent"})
print("| model | T | parent | " + " | ".join(arms) + " |")
print("|---|---:|---:|" + "---:|" * len(arms))
for mdl in ("4B", "llama8B", "14B", "30B"):
    for t in (32, 40, 48, 64, 96):
        p = res.get((mdl, t, "parent"))
        if not p:
            continue
        cells = []
        for a in arms:
            c = res.get((mdl, t, a))
            if not c:
                cells.append("")
                continue
            dp = (c[0] / p[0] - 1) * 100
            dc = (c[1] / p[1] - 1) * 100
            ex = (p[0] - c[0]) / (p[0] - FP16[mdl]) * 100 if p[0] > FP16[mdl] else float("nan")
            cells.append(f"{c[0]:.4f} ({dp:+.2f}%, cost {dc:+.1f}%, excess {ex:.0f}%)")
        print(f"| {mdl} | {t} | {p[0]:.4f} @{p[1]:.2f} | " + " | ".join(cells) + " |")
print()
print("held-out pre-flight (prc2 c17): err new/par at iso-cost, linear budget scale for iso-error")
for f in sorted(glob.glob(f"{P2}/*_c17_diag.json")):
    d = json.load(open(f))
    ops = [k for k in d["ops"] if not k.startswith("total_")]
    en = sum(d["ops"][o]["new_err"] for o in ops); ep = sum(d["ops"][o]["par_err"] for o in ops)
    s = iso_scale(d)
    print(f"  {os.path.basename(f)[:-14]:14s} L new/par {d['ops']['total_new_L']/d['ops']['total_par_L']:.3f}  "
          f"err new/par {en/ep:.3f}  iso-error linear budget {'%.2f' % s if s else 'n/a'}x")
