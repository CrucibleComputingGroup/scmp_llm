"""CPU tests for the round-7 calibration side (calib10_r7 + prc_r7_solve), 2026-09-28.

  cd scmp_llm && PYTHONPATH=$PWD/kernels <annstention python> -m unittest \
      benchmark.ppl.test_prc_r7_calib_20260928 -v

What is proven here without a GPU:
  * calib10 == calib9 at its defaults: calib9's module-level code is imported by identity, and
    main()'s diff vs calib9 is only `# r7`-tagged lines (+ the two-line per-head loop header they
    replace), every one of which is exercised at its default below;
  * the prefix-split capture is calib7's 128-row capture: a synthetic 512-row capture whose
    rank<128 rows are a "calib7" 128-row sample reproduces, through prc_r7_solve (identity),
    the calib7 joint solve (verbatim reference code) and calib9's held-out predictions exactly;
  * kappa = 1 is calib9's plain joint lambda exactly (K/J arms == calib9 resolve_tables gfisla);
  * emitted round-7 tables (per-bucket attention ladders up to 128) are lineage-clean and load
    exactly as the runtime resolves them (get_levels / classify_level_values / thresholds /
    escape / synthetic dispatch / per_row_chunk); tampered tables are refused.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import sys
import tempfile
import unittest
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from benchmark.ppl import mp_per_row_chunk_calib9_r6 as c9  # noqa: E402
from benchmark.ppl import mp_per_row_chunk_calib10_r7 as c10  # noqa: E402
from benchmark.ppl import prc_r7_solve as S  # noqa: E402
from benchmark.ppl.mp_per_row_chunk_calib5 import (  # noqa: E402
    DENSE_LADDER, bucket_of, deploy_rungs)
from benchmark.ppl.test_prc_r6_retarget import (  # noqa: E402
    CALIB7_global, CALIB7_joint, synth_bins)

CALIB9 = REPO / "benchmark/ppl/mp_per_row_chunk_calib9_r6.py"
CALIB10 = REPO / "benchmark/ppl/mp_per_row_chunk_calib10_r7.py"
CALIB9_SHA = "4561bc370eda34696fd6516ce0a6e7cee1c8a9dc51a0dd490ad1b970b978a892"
MPB = REPO.parent / "hpca_results/llm/ppl/mp_best/configs"
PARENT_4B_T40 = MPB / "4B/target40"
PARENT_30B_T40 = MPB / "30B/target40"
PRC2 = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2")
# The two calib9 lines calib10 replaces (by one tagged line, r7_head_selections at prefix 0).
ALLOWED_DELETIONS = [
    "        for h in torch.unique(bh).tolist():",
    "            sel = (bh == h).nonzero(as_tuple=True)[0]",
]


# ---------------------------------------------------------------------------------------------
# verbatim reference code (calib9 main() inline blocks, closure variables made arguments)
# ---------------------------------------------------------------------------------------------
def CALIB9_abins(att_calib_parts, akeys, att_ladder, bins):
    """calib9 main() lines 1044-1062 (the attention bins of the joint solve), verbatim."""
    def cat(parts, f):
        return np.concatenate([p[f] for p in parts])
    abins, fixed, par_att = {}, 0.0, 0.0
    acur = "fis"
    for k in akeys:
        lad, t_esc, esc_len, lv_meas, _ = att_ladder[k]
        parts = att_calib_parts[k]
        mn, cur, mac, parL = cat(parts, "mn"), cat(parts, acur), cat(parts, "mac"), cat(parts, "parL")
        col = {int(L): i for i, L in enumerate(parts[0]["lv"])}
        esc = (mn > t_esc) if (t_esc is not None and t_esc < 1.0 and esc_len) else np.zeros(mn.shape, bool)
        A = np.asarray(lad, dtype=np.float64)
        ok = ~esc
        m_ok = mn[ok]
        order = np.argsort(m_ok, kind="stable")
        Bn = int(min(bins, max(m_ok.size, 1)))
        edges = np.linspace(0, m_ok.size, Bn + 1).astype(np.int64)
        E = np.add.reduceat(cur[ok][order][:, [col[int(L)] for L in lad]], edges[:-1], axis=0)
        C = np.add.reduceat(mac[ok][order], edges[:-1])
        abins[k] = (E, C, m_ok[order][edges[:-1]], A)
        if esc.any():
            fixed += float((mac[esc] * esc_len).sum())
        par_att += float((mac * parL).sum())
    return abins, fixed, par_att


def CALIB9_deploy_att(mn, th_desc, lad_asc, t_esc, esc_len):
    D = list(reversed(lad_asc))
    lvl = np.full(mn.shape, len(D) - 1)
    for i in range(len(th_desc) - 1, -1, -1):
        lvl[mn >= th_desc[i]] = i
    L = np.asarray(D, dtype=np.float64)[lvl]
    if t_esc is not None and t_esc < 1.0 and esc_len:
        L[mn > t_esc] = esc_len
    return L


def CALIB9_HJ(store_hold, att_hold, keys, akeys, att_ladder, tables, att_th_by, Lv, li, top_col,
              n_tok):
    """calib9 main() lines 1097-1132 (joint held-out Fisher prediction), verbatim."""
    def cat(parts, f):
        return np.concatenate([p[f] for p in parts])

    def cur_of(parts, name):
        e = cat(parts, name)
        return e - e[:, [top_col]]
    acur = "fis"
    HJ = {}
    for tname in ["parent", "gfis", "gfisla"]:
        tot_f = tot_c = 0.0
        for k in keys:
            hp = store_hold.get(k)
            if not hp:
                continue
            mac, hm, parL = cat(hp, "mac"), cat(hp, "mn"), cat(hp, "parL")
            if tname == "parent" or tname not in tables:
                Lr = parL
            else:
                Lr = Lv[deploy_rungs(torch.from_numpy(hm), tables[tname][k]).numpy()]
            cols = np.asarray([li[int(v)] for v in Lr])
            tot_f += float(cur_of(hp, acur)[np.arange(mac.size), cols].sum())
            tot_c += float((mac * Lr).sum())
        for k in akeys:
            lad, t_esc, esc_len, lv_meas, th_par = att_ladder[k]
            for p_ in att_hold.get(k, []):
                col = {int(L): i for i, L in enumerate(p_["lv"])}
                if c9.base_table_name(tname) == "gfisla":
                    Lr = CALIB9_deploy_att(p_["mn"], att_th_by[tname][k], lad, t_esc, esc_len)
                else:
                    Lr = p_["parL"]
                cols = np.asarray([col[int(v)] for v in Lr])
                tot_f += float(p_[acur][np.arange(Lr.size), cols].sum())
                tot_c += float((p_["mac"] * Lr).sum())
        HJ[tname] = (tot_f, tot_c)
    return {t: {"pred_dnll": 0.5 * (f - HJ["parent"][0]) / n_tok,
                "cost_over_parent": c / HJ["parent"][1]} for t, (f, c) in HJ.items()}


def CALIB9_diag_lin(store_hold, keys, tables_lin, levels_by_key, ladder, li, top_col, n_tok):
    """calib9 main() held-out scoring of linear tables (diag['tables'][name]['pred_dnll_fis'])."""
    def cat(parts, f):
        return np.concatenate([p[f] for p in parts])

    def cur_of(parts, name):
        e = cat(parts, name)
        return e - e[:, [top_col]]
    hkeys = [k for k in keys if store_hold.get(k)]
    acc = {}
    for tname in ["parent"] + list(tables_lin):
        a = 0.0
        for k in hkeys:
            hp = store_hold[k]
            mac, hm, parL = cat(hp, "mac"), cat(hp, "mn"), cat(hp, "parL")
            ar = np.arange(mac.size)
            if tname == "parent":
                Lrow = parL
            else:
                th = tables_lin[tname].get(k)
                lv_ = levels_by_key.get(tname, {}).get(k, ladder)
                Lrow = np.asarray(lv_, dtype=np.float64)[deploy_rungs(torch.from_numpy(hm), th).numpy()]
            cols = np.asarray([li[int(v)] for v in Lrow])
            a += float(cur_of(hp, "fis")[ar, cols].sum())
        acc[tname] = a
    return {t: 0.5 * (v - acc["parent"]) / n_tok for t, v in acc.items()}


# ---------------------------------------------------------------------------------------------
# synthetic capture: a 512-row prefix-split capture whose rank<128 rows ARE a 128-row sample
# ---------------------------------------------------------------------------------------------
TOTAL_ROWS = 65536          # B*H*N for 32 heads x 2048 tokens
K_HEAD, M_KEYS = 128, 2048
EXTRA = (16, 20, 24, 28, 40, 48, 56, 80, 112)


def _att_record(rng, lv, lad, th_par, t_esc, rows, blk, win, rep, is_qk):
    """One per-call record at `rows` sample rows; returns (record, base) with fis = base*rep."""
    mn = rng.uniform(0, 1, size=rows).astype(np.float32) ** (2 if is_qk else 0.5)
    mn = mn.astype(np.float32)
    parL = CALIB9_deploy_att(mn, th_par, lad, t_esc, 128).astype(np.float32)
    Larr = np.asarray(lv, dtype=np.float64)
    scale = (rng.lognormal(0, 1, size=rows) * (0.2 + mn)).astype(np.float32)
    base = (scale[:, None] / (Larr[None, :] ** 2) * rng.uniform(0.9, 1.1, size=(rows, len(lv))))
    base = base.astype(np.float32)
    raw_base = (base * np.float32(3.0)).astype(np.float32)
    rec = dict(mn=mn, raw=(raw_base * np.float32(rep)).astype(np.float32),
               mac=np.full(rows, float(K_HEAD * M_KEYS) * rep), parL=parL, lv=np.asarray(lv),
               fis=(base * np.float32(rep)).astype(np.float32),
               fis_emp=(base * np.float32(2 * rep)).astype(np.float32),
               blk=np.full(rows, blk, dtype=np.int16),
               bh=rng.integers(0, 32, size=rows).astype(np.int32),
               pos=rng.integers(0, 2048, size=rows).astype(np.int32),
               rank=np.arange(rows, dtype=np.int32), win=np.full(rows, win, np.int16))
    return rec


def _prefix7(rec, P, factor):
    """The calib7 (P-row) record hidden in a calib10 record: first P rows, x factor."""
    out = {f: (v[:P] if f != "lv" else v) for f, v in rec.items()}
    for f in ("mac", "raw", "fis", "fis_emp"):
        out[f] = rec[f][:P] * factor
    return out


def _lin_record(rng, measured, parL_vals, rows, blk):
    n = rows
    mn = rng.uniform(0, 1, size=n).astype(np.float32)
    Larr = np.asarray(measured, dtype=np.float64)
    base = (rng.lognormal(0, 1, size=n)[:, None] / Larr[None, :] ** 2).astype(np.float32)
    return dict(mn=mn, mac=np.full(n, 4096.0, dtype=np.float32),
                parL=rng.choice(parL_vals, size=n).astype(np.float32),
                blk=np.full(n, blk, np.int16), cidx=np.tile(np.arange(4, dtype=np.int16), n // 4),
                rpos=np.repeat(np.arange(n // 4, dtype=np.int32), 4),
                expert=np.zeros(n, bool), fis=base, fis_emp=(base * 2).astype(np.float32),
                raw=base, rowmax=mn, yE=np.ones(n, np.float32))


def build_capture(d: Path, parent_dir: Path, *, seed=0, total_blocks=36, n_calib=3, n_hold=2,
                  att_rows=512, prefix=128, inc_relation="identical", bins=256):
    """Write a synthetic calib10 capture + its calib7 references into directory d.

    Returns dict of paths. References (c7_gfis, c7_gfisla, c7 diag) are made by verbatim
    calib7/calib9 code from the 128-row sample hidden in the 512-row records."""
    rng = np.random.default_rng(seed)
    table = json.loads((parent_dir / "table.json").read_text())
    wrapper = json.loads((parent_dir / "wrapper.json").read_text())
    lb = int(table["layer_buckets"])
    glob = sorted(int(v) for v in table["stoc_len_levels"])
    k_esc = float(wrapper["escape_gate_k"])
    ladder = list(DENSE_LADDER)
    Lv = np.asarray(ladder, dtype=np.float64)
    lbins, budget, _ = synth_bins(seed + 11, n_bins=64)
    keys = sorted(lbins)
    measured = sorted(set(ladder) | set(glob))
    li = {L: i for i, L in enumerate(measured)}
    top_col = li[128]
    rep10, rep7 = TOTAL_ROWS / att_rows, TOTAL_ROWS / prefix
    factor = float(att_rows // prefix)
    att_ladder, akeys = {}, []
    att10 = {"calib": defaultdict(list), "hold": defaultdict(list)}
    att7 = {"calib": defaultdict(list), "hold": defaultdict(list)}
    blocks_of = defaultdict(list)
    for blk in range(total_blocks):
        blocks_of[bucket_of(blk, total_blocks, lb)].append(blk)
    wins = list(range(n_calib + n_hold))
    for op in ("av", "qk"):
        for b in range(lb):
            k = (op, b)
            akeys.append(k)
            pay = table["buckets"][f"{op}:t0:l{b}"]
            t_esc = float(pay["metric_mean"]) + k_esc * float(pay["metric_std"])
            lv = sorted(set(glob) | {128} | set(EXTRA))
            th_par = [float(v) for v in pay["thresholds"]]
            att_ladder[k] = (glob, t_esc, 128, lv, th_par)
            for w in wins:
                ph = "hold" if w >= n_calib else "calib"
                for blk in blocks_of[b][:2]:
                    rec = _att_record(rng, lv, glob, th_par, t_esc, att_rows, blk, w, rep10,
                                      op == "qk")
                    att10[ph][k].append(rec)
                    att7[ph][k].append(_prefix7(rec, prefix, factor))
    # sanity: the prefix rescale is exact (x4 of rep 128 == rep 512)
    assert rep10 * factor == rep7
    # linear hold store
    hold_store = defaultdict(list)
    for k in keys:
        for w in range(n_hold):
            hold_store[k].append(_lin_record(rng, measured, glob, 64, bucket_of(0, 1, 1)))
    # ---- calib10-side state (512 rows, in-process bins) ----
    ab10, fixed10, par_att10 = CALIB9_abins(att10["calib"], akeys, att_ladder, bins)
    par_lin = sum(budget[k] * float(lbins[k][1].sum()) for k in keys)
    stem = d / "cell_r7.json"
    calib_record = {"parent": str(parent_dir), "att_rows_per_call": att_rows, "out": str(stem)}
    c9rec = {"calibrator": "mp_per_row_chunk_calib10_r7.py", "r7": {"att_prefix_split": prefix}}
    # in-process gfis (== calib7's gfis thresholds: linears are identical)
    rs7, th_g = CALIB7_global(lbins, keys, budget, Lv)
    c9.emit_table(stem, "gfis", th_g, None, table, wrapper, ladder,
                  dict(calib_record, table="gfis"), dict(c9rec, table="gfis"))
    capd = d / "cell_r7_diag.json"
    meta = {"schema": c10.R7_STATE_SCHEMA, "ladder": ladder, "measured": measured,
            "keys": [[k[0], k[1]] for k in keys], "akeys": [[k[0], k[1]] for k in akeys],
            "budget": {f"{k[0]}|{k[1]}": budget[k] for k in keys},
            "target0_gfis": float(par_lin), "target0_gfisla": float(par_lin + par_att10),
            "par_lin": par_lin, "par_att": par_att10, "fixed": float(fixed10),
            "att_ladder": {f"{k[0]}|{k[1]}": [list(map(int, v[0])), v[1], v[2],
                                              list(map(int, v[3])), v[4]]
                           for k, v in att_ladder.items()},
            "calib_record": calib_record, "c9_record": c9rec, "solves": {},
            "parent_table_sha256": hashlib.sha256((parent_dir / "table.json").read_bytes()).hexdigest(),
            "parent_wrapper_sha256": hashlib.sha256((parent_dir / "wrapper.json").read_bytes()).hexdigest()}
    import argparse
    ns = argparse.Namespace(att_rows_per_call=att_rows, att_prefix_split=prefix,
                            att_measure_extra=",".join(map(str, EXTRA)), calib_windows=n_calib,
                            holdout_windows=n_hold, bins=bins, att_dump=str(d / "cell_r7_att.npz"),
                            r7_hold_dump=str(d / "cell_r7_hold.npz"), diag=str(capd), out=str(stem))
    meta.update(c10.r7_state_meta(args=ns, att_store=att10, hold_store=hold_store,
                                  measured=measured, ladder=ladder, starts=wins,
                                  hold_idx=range(n_calib, n_calib + n_hold), prot_levels=[64, 128],
                                  total_blocks=total_blocks, n_lb=lb, n_units=0, top_k=1,
                                  currencies=["fis"], cap=128))
    c9.save_solve_state(d / "cell_r7_state.npz", meta, lbins, ab10)
    # att dump (calib9 --att-dump format)
    out_ = {"measured": np.asarray(measured), "ladder": np.asarray(ladder)}
    for ph in ("calib", "hold"):
        for k, parts in att10[ph].items():
            tag = f"att|{ph}|{k[0]}|{k[1]}"
            for f_ in ("mn", "mac", "parL", "blk", "bh", "pos", "rank", "win"):
                out_[f"{tag}|{f_}"] = np.concatenate([p[f_] for p in parts])
            for f_ in ("raw", "fis", "fis_emp"):
                out_[f"{tag}|{f_}"] = np.concatenate([p[f_] for p in parts]).astype(np.float32)
            out_[f"{tag}|lv"] = np.asarray(parts[0]["lv"])
    np.savez_compressed(d / "cell_r7_att.npz", **out_)
    prot = {"calib": defaultdict(list), "hold": defaultdict(list)}
    prot["hold"][("q_proj", 0)].append(dict(mac=np.full(8, 128.0), raw=np.ones((8, 2), np.float32),
                                            blk=np.zeros(8, np.int16), rpos=np.arange(8, dtype=np.int32),
                                            expert=np.zeros(8, bool), fis=np.ones((8, 2), np.float32)))
    c10.r7_write_hold_dump(d / "cell_r7_hold.npz", hold_store, prot, measured, [64, 128])
    # ---- calib7 references from the hidden 128-row sample (verbatim code) ----
    ab7, fixed7, par_att7 = CALIB9_abins(att7["calib"], akeys, att_ladder, bins)
    rj7, lin7, att_th7, _ = CALIB7_joint(lbins, ab7, keys, akeys, budget, Lv, fixed7, par_att7)
    refd = d / "ref"
    refd.mkdir()
    rec7 = {"parent": str(parent_dir), "att_rows_per_call": prefix, "out": "c7"}
    c9.emit_table(refd / "c7.json", "gfis", th_g, None, table, wrapper, ladder,
                  dict(rec7, table="gfis"))
    c9.emit_table(refd / "c7.json", "gfisla", lin7, att_th7, table, wrapper, ladder,
                  dict(rec7, table="gfisla"))
    hj = CALIB9_HJ(hold_store, att7["hold"], keys, akeys, att_ladder,
                   {"gfis": th_g, "gfisla": lin7}, {"gfisla": att_th7}, Lv, li, top_col,
                   n_hold * 2047)
    c17 = {k: sorted(float(np.float32(x)) for x in rng.uniform(0, 1, len(ladder) - 1)) for k in keys}
    dl = CALIB9_diag_lin(hold_store, keys, {"gfis": th_g, "c17": c17}, {}, ladder, li, top_col,
                         n_hold * 2047)
    diag7 = {"joint": hj, "tables": {"gfis": {"pred_dnll_fis": dl["gfis"]},
                                     "c17": {"pred_dnll_fis": dl["c17"]}},
             "att_replay_agreement": 1.0}
    (refd / "c7_diag.json").write_text(json.dumps(diag7, indent=1))
    inc = refd / "c7_gfisla_table.json"
    if inc_relation == "prc_thresholds_only":
        t = json.loads(inc.read_text())
        kk = next(iter(t["per_row_chunk"]["buckets"]))
        th = t["per_row_chunk"]["buckets"][kk]["thresholds"]
        th[0] = th[0] * 0.5
        inc = refd / "candidate_table.json"
        inc.write_text(json.dumps(t, indent=1))
    # ---- the capture's OWN in-process 512-row joint table (J512 at scale 1.0), its --score-tables
    #      (c17, c17_s80, inc) and its own diag, by verbatim calib9 code on the 512-row records ----
    rj10, lin10, att_th10, _ = CALIB7_joint(lbins, ab10, keys, akeys, budget, Lv, fixed10, par_att10)
    c9.emit_table(stem, "gfisla", lin10, att_th10, table, wrapper, ladder,
                  dict(calib_record, table="gfisla"), dict(c9rec, table="gfisla"))
    c17s80 = {k: sorted(float(np.float32(x)) for x in rng.uniform(0, 1, len(ladder) - 1)) for k in keys}
    scd = d / "score"
    scd.mkdir()
    c9.emit_table(scd / "s.json", "c17", c17, None, table, wrapper, ladder, {"table": "c17"})
    c9.emit_table(scd / "s.json", "c17_s80", c17s80, None, table, wrapper, ladder,
                  {"table": "c17_s80"})
    inc_t = json.loads(Path(inc).read_text())
    inc_lin = {(kk.split(":")[0], int(kk.split(":l")[1])): e["thresholds"]
               for kk, e in inc_t["per_row_chunk"]["buckets"].items()}
    hj10 = CALIB9_HJ(hold_store, att10["hold"], keys, akeys, att_ladder,
                     {"gfis": th_g, "gfisla": lin10}, {"gfisla": att_th10}, Lv, li, top_col,
                     n_hold * 2047)
    dl10 = CALIB9_diag_lin(hold_store, keys, {"gfis": th_g, "gfisla": lin10, "c17": c17,
                                              "c17_s80": c17s80, "inc": inc_lin},
                           {}, ladder, li, top_col, n_hold * 2047)
    capd.write_text(json.dumps({"att_replay_agreement": 1.0, "joint": hj10,
                                "tables": {n: {"pred_dnll_fis": v} for n, v in dl10.items()
                                           if n != "parent"}}, indent=1))
    score_tables = {"c17": scd / "s_c17_table.json", "c17_s80": scd / "s_c17_s80_table.json",
                    "inc": Path(inc)}
    joint_line = S.joint_line({"lin_L_parent": par_lin / sum(float(lbins[k][1].sum()) for k in keys),
                               "lin_L": sum(float((lbins[k][1] * Lv[rj7[0][k]]).sum()) for k in keys)
                               / sum(float(lbins[k][1].sum()) for k in keys),
                               "att_L_parent": par_att7 / sum(float(np.concatenate(
                                   [p["mac"] for p in att7["calib"][k]]).sum()) for k in akeys),
                               "att_L": (sum(float((ab7[k][1] * ab7[k][3][rj7[1][k]]).sum())
                                             for k in akeys) + fixed7) / sum(float(np.concatenate(
                                                 [p["mac"] for p in att7["calib"][k]]).sum())
                                                 for k in akeys)})
    return {"stem": stem, "state": d / "cell_r7_state.npz", "att": d / "cell_r7_att.npz",
            "hold": d / "cell_r7_hold.npz", "ref_gfis": refd / "c7_gfis_table.json",
            "ref_gfisla": refd / "c7_gfisla_table.json", "ref_diag": refd / "c7_diag.json",
            "inc": inc, "joint_line": joint_line, "lbins": lbins, "keys": keys, "akeys": akeys,
            "att7": att7, "att10": att10, "budget": budget, "table": table, "wrapper": wrapper,
            "score_tables": score_tables, "capd": capd}


def load_capture(paths):
    return S.Capture(paths["state"], paths["att"], paths["hold"])


# ---------------------------------------------------------------------------------------------
# calib10 == calib9 at the defaults
# ---------------------------------------------------------------------------------------------
class TestCalib10Identity(unittest.TestCase):
    def test_calib9_is_the_frozen_file(self):
        self.assertEqual(hashlib.sha256(CALIB9.read_bytes()).hexdigest(), CALIB9_SHA)

    def test_shared_code_is_calib9_by_identity(self):
        for name in ("WEIGHT_TAGS", "SCALE_NAME_RE", "scale_suffix", "base_table_name",
                     "emitted_table_names", "parse_scales", "scaled_target", "th_from_bins",
                     "bisect_lambda", "att_th_from", "solve_global", "joint_cost", "solve_joint",
                     "emit_table", "save_solve_state", "load_solve_state", "resolve_tables",
                     "error_curves_w", "GradPass"):
            self.assertIs(getattr(c10, name), getattr(c9, name), name)
        for name in ("staircase_dp", "deploy_rungs", "call_metric", "bucket_of", "solve_bucket"):
            self.assertIs(getattr(c10, name), getattr(c9, name), name)

    def test_main_diff_is_only_tagged_r7_lines(self):
        a = CALIB9.read_text()
        b = CALIB10.read_text()
        a = a[a.index("def main() -> int:"):].splitlines()
        b = b[b.index("def main() -> int:"):].splitlines()
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        deleted, added = [], []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            deleted += a[i1:i2]
            added += b[j1:j2]
        self.assertEqual(deleted, ALLOWED_DELETIONS)
        untagged = [ln for ln in added if "# r7" not in ln]
        self.assertEqual(untagged, [], "every calib10-only line in main() must carry '# r7'")
        self.assertGreater(len(added), 10)

    def test_r7_defaults_are_off(self):
        import argparse
        ns = argparse.Namespace(**{a: c10.R7_DEFAULTS[a] for a in c10.R7_NEW_ARGS},
                                att_rows_per_call=128)
        self.assertFalse(c10.r7_active(ns))
        view = {"parent": "p", "att_rows_per_call": 128}
        rec = {"calibrator": "mp_per_row_chunk_calib9_r6.py", "target_scales": [1.0]}
        v2, r2, r7 = c10.r7_adjust_records(ns, view, rec, [1.0], ())
        self.assertIs(v2, view)
        self.assertIs(r2, rec)
        self.assertIsNone(r7)
        v3, r3, r7b = c10.r7_adjust_records(ns, view, None, [1.0], ())
        self.assertIsNone(r3)
        self.assertEqual(c10.r7_parse_lengths("", 128), [])
        self.assertEqual(c10.r7_check_prefix(0, 128), 0)

    def test_r7_active_records(self):
        import argparse
        ns = argparse.Namespace(att_prefix_split=128, r7_hold_dump="h.npz", prot_measure="64,128",
                                att_rows_per_call=512, target_scales="1.0", solve_state="s.npz",
                                att_measure_extra="112", att_dump="a.npz")
        view = {"parent": "p", "att_prefix_split": 128, "r7_hold_dump": "h.npz",
                "prot_measure": "64,128", "att_rows_per_call": 512}
        v, rec, r7 = c10.r7_adjust_records(ns, view, None, [1.0],
                                           ("target_scales", "solve_state", "att_measure_extra",
                                            "att_dump"))
        self.assertEqual(v, {"parent": "p", "att_rows_per_call": 512})
        self.assertEqual(rec["calibrator"], c10.R7_CALIBRATOR)
        self.assertEqual(rec["r7"]["att_prefix_split"], 128)
        self.assertEqual(rec["att_dump"], "a.npz")
        self.assertEqual(r7["att_rows_per_call"], 512)

    def test_prefix_and_length_parsing(self):
        self.assertEqual(c10.r7_check_prefix(128, 512), 128)
        for bad in ((96, 512), (512, 512), (-1, 512), (128, 384)):
            with self.assertRaises(SystemExit):
                c10.r7_check_prefix(*bad)
        self.assertEqual(c10.r7_parse_lengths("128,64,64", 128), [64, 128])
        with self.assertRaises(SystemExit):
            c10.r7_parse_lengths("129", 128)

    def test_head_selections_default_is_calib9_loop(self):
        g = torch.Generator().manual_seed(3)
        for R in (128, 512, 37):
            bh = torch.randint(0, 32, (R,), generator=g)
            ref = [(h, (bh == h).nonzero(as_tuple=True)[0]) for h in torch.unique(bh).tolist()]
            got = c10.r7_head_selections(bh, 0)
            self.assertEqual([h for h, _ in got], [h for h, _ in ref])
            for (_, a), (_, b) in zip(got, ref):
                self.assertTrue(torch.equal(a, b))

    def test_prefix_split_reproduces_calib7_batches(self):
        """The 512 sample's first 128 ranks are calib7's 128 sample (same crc32 randperm), and
        each head's rank<128 batch equals calib7's per-head batch in content and order."""
        for op, blk, win in (("qk", 3, 0), ("av", 40, 5)):
            def sel_att(R):
                g = torch.Generator(device="cpu")
                g.manual_seed(zlib.crc32(f"att:0:{op}:{blk}:{win}".encode()))
                return torch.randperm(TOTAL_ROWS, generator=g)[:min(R, TOTAL_ROWS)]
            i7, i10 = sel_att(128), sel_att(512)
            self.assertTrue(torch.equal(i10[:128], i7))
            N = 2048
            bh7, bh10 = i7 // N, i10 // N
            parts = c10.r7_head_selections(bh10, 128)
            cal7 = {h: (bh7 == h).nonzero(as_tuple=True)[0] for h in torch.unique(bh7).tolist()}
            seen = torch.zeros(512, dtype=torch.long)
            for h, sel in parts:
                self.assertTrue(bool((bh10[sel] == h).all()))
                seen[sel] += 1
                if bool((sel < 128).all()):
                    self.assertTrue(torch.equal(i10[sel], i7[cal7[h]]), (op, h))
                else:
                    self.assertTrue(bool((sel >= 128).all()))
            self.assertTrue(bool((seen == 1).all()))
            self.assertEqual(sorted(h for h, s in parts if bool((s < 128).all())), sorted(cal7))

    def test_calib10_parser_skips_without_cuda(self):
        import os
        import subprocess
        argv = ["--parent", "/p", "--model_path", "m", "--out", "/o/x.json", "--diag", "/o/d.json",
                "--att-rows-per-call", "512", "--att-prefix-split", "128", "--prot-measure",
                "64,128", "--r7-hold-dump", "/o/h.npz", "--att-measure-extra", "112",
                "--att-dump", "/o/a.npz", "--solve-state", "/o/s.npz"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        p = subprocess.run([sys.executable, str(CALIB10)] + argv, capture_output=True, text=True,
                           env=env, cwd=str(REPO))
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        self.assertIn("SKIP: needs CUDA", p.stdout)


class TestCalib10Helpers(unittest.TestCase):
    def test_protected_record_mirrors_error_curves(self):
        calls = []

        def fake_sc(x, w, L, smooth):
            calls.append((tuple(x.shape), int(L)))
            xs = x if smooth is None else x / smooth
            return (xs * (1.0 + 1.0 / L)) @ (w if smooth is None else w * smooth).t()
        old = c9._sc
        c9._sc = fake_sc
        try:
            g = torch.Generator().manual_seed(0)
            R, D, O = 6, 300, 5
            x = torch.randn(R, D, generator=g)
            w = torch.randn(O, D, generator=g)
            sm = torch.rand(D, generator=g) + 0.5
            pidx = torch.tensor([3, 7, 150, 151, 299] + list(range(10, 140)))
            W = {"fis": torch.rand(R, O, generator=g)}
            rows = torch.arange(R)
            rec = c10.r7_protected_record(x, w, sm, pidx, [64, 128], W, 2.0, 5, rows, True)
            raw, wt, widths = c9.error_curves_w(x[:, pidx], w[:, pidx], sm[pidx], [64, 128], W)
            self.assertTrue(np.allclose(rec["raw"], (raw.sum(1) * 2.0).numpy()))
            self.assertTrue(np.allclose(rec["fis"], (wt["fis"].sum(1) * 2.0).numpy()))
            self.assertEqual(rec["mac"][0], float(len(pidx) * O) * 2.0)
            self.assertEqual(rec["fis"].shape, (R, 2))
            self.assertTrue(rec["expert"].all())
            self.assertEqual(sorted({c[0][1] for c in calls}), [7, 128])   # 135 cols = 128 + 7
        finally:
            c9._sc = old

    def test_hold_dump_roundtrip(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d_:
            hs = defaultdict(list)
            meas = [8, 16, 64, 128]
            for k in (("q_proj", 0), ("down_proj", 3)):
                for _ in range(2):
                    hs[k].append(_lin_record(rng, meas, [16, 64], 8, 0))
            prot = {"calib": defaultdict(list), "hold": defaultdict(list)}
            path = Path(d_) / "h.npz"
            names = c10.r7_write_hold_dump(path, hs, prot, meas, [])
            with np.load(path) as z:
                for k, parts in hs.items():
                    for f in ("mn", "mac", "parL", "fis", "fis_emp", "blk", "cidx", "rpos", "expert"):
                        a = z[f"lin|hold|{k[0]}|{k[1]}|{f}"]
                        b = np.concatenate([p[f] for p in parts])
                        self.assertEqual(a.dtype, b.dtype, f)
                        self.assertTrue(np.array_equal(a, b), f)
            self.assertIn("measured", names)


# ---------------------------------------------------------------------------------------------
# solver pieces
# ---------------------------------------------------------------------------------------------
class TestSolverPieces(unittest.TestCase):
    def test_ladder_policies_on_real_ladders(self):
        meas = sorted({16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128} |
                      {30, 31, 47, 49, 65, 45, 46, 66, 109})
        t32 = [16, 24, 32, 48, 64, 96]
        t40 = [30, 31, 32, 47, 49, 65, 96]
        t48 = [32, 45, 46, 47, 48, 66, 109]
        self.assertEqual(S.ladder_for("inherit", t40, meas), t40)
        self.assertEqual(S.ladder_for("up", t32, meas), t32 + [112, 128])
        self.assertEqual(S.ladder_for("up", t40, meas), t40 + [112, 128])
        self.assertEqual(S.ladder_for("up", t48, meas), t48 + [128])      # 112 < 1.1 * 109
        self.assertEqual(S.ladder_for("up", [48, 60, 105], meas + [60, 105]), [48, 60, 105, 128])
        dense = S.ladder_for("dense", t40, meas)
        self.assertEqual(min(dense), 30)
        self.assertTrue({40, 56, 80, 112, 128} <= set(dense))
        full = S.ladder_for("full", t40, meas)
        self.assertTrue({16, 20, 24, 28} <= set(full))
        with self.assertRaises(ValueError):
            S.ladder_for("up", t32, [16, 24, 32, 48, 64, 96, 128])       # 112 unmeasured
        with self.assertRaises(ValueError):
            S.ladder_for("nope", t32, meas)

    def test_kappa_bins_identity_and_scaling(self):
        E = np.arange(12, dtype=np.float32).reshape(4, 3)
        ab = {("qk", 0): (E, np.ones(4), np.zeros(4, np.float32), np.asarray([1.0, 2.0, 3.0]))}
        self.assertIs(S.kappa_bins(ab, 1.0), ab)
        sc = S.kappa_bins(ab, 0.42)
        self.assertEqual(sc[("qk", 0)][0].dtype, np.float32)
        self.assertTrue(np.array_equal(sc[("qk", 0)][0], E * 0.42))
        self.assertIs(sc[("qk", 0)][1], ab[("qk", 0)][1])
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                S.kappa_bins(ab, bad)

    def test_deploy_rules_agree_on_float32_thresholds(self):
        rng = np.random.default_rng(2)
        mn = rng.uniform(0, 1, 5000).astype(np.float32)
        th = sorted((float(np.float32(x)) for x in rng.uniform(0, 1, 6)), reverse=True)
        mn[:6] = np.asarray(th, np.float32)
        lad = [16, 24, 32, 48, 64, 96, 128]
        a = S.deploy_att(mn, th, lad, 0.8, 128)
        b = S.deploy_att_runtime(mn, th, lad, 0.8, 128)
        c = CALIB9_deploy_att(mn, th, lad, 0.8, 128)
        self.assertTrue(np.array_equal(a, c))
        self.assertLessEqual(int((a != b).sum()), 2)     # only rows within 1 ulp of mu+2tau

    def test_fixed_point(self):
        raw, s = S.fixed_point_scale(1.0043, 42.40, 42.10, 1.59)
        self.assertAlmostEqual(raw, 1.0043 * (42.40 - 1.59) / (42.10 - 1.59))
        self.assertEqual(s, round(raw, 4))
        with self.assertRaises(ValueError):
            S.fixed_point_scale(1.0, 50.0, 40.0, 1.0)

    def test_kappa_rule_is_one_sided(self):
        R = S.KAPPA_RULE
        self.assertEqual((R["kappa_att"], R["kappa_att_se"]), (0.83, 0.07))
        keep, rev = S.KAPPA_PINS["keep_0.42"], S.KAPPA_PINS["revert_to_kappa_1"]
        # clearly above the band -> keep, currency 0.83/1.96
        d = S.kappa_decision_core(2.0, 0.2)
        self.assertEqual((d["decision"], d["ratio"], d["kappa_pins"]), ("keep_0.42", 0.42, keep))
        self.assertEqual(d["currency_ratio"], 0.83 / 1.96)
        # inside the band -> revert
        d = S.kappa_decision_core(0.95, 0.2)
        self.assertEqual((d["decision"], d["ratio"], d["kappa_pins"]), ("revert_to_kappa_1", 1.0, rev))
        self.assertEqual(d["currency_ratio"], 1.0)
        # reviewer adversarial A1: kappa_lin_bs 0.30 +- 0.06 FAR BELOW kappa_att -> revert
        # (the v1 two-sided rule kept 0.42 here)
        d = S.kappa_decision_core(0.30, 0.06)
        self.assertEqual(d["decision"], "revert_to_kappa_1")
        self.assertIn("direction_contradicted", d["flags"])
        # the band edge is not a confirmation
        band = math.sqrt(0.1 ** 2 + 0.07 ** 2)
        self.assertEqual(S.kappa_decision_core(0.83 + band - 1e-9, 0.1)["decision"], "revert_to_kappa_1")
        self.assertEqual(S.kappa_decision_core(0.83 + band + 1e-9, 0.1)["decision"], "keep_0.42")
        # gap case: kept (one pin pair per branch), flagged
        d = S.kappa_decision_core(1.3, 0.2)
        self.assertEqual((d["decision"], d["kappa_pins"]), ("keep_0.42", keep))
        self.assertIn("gap_case", d["flags"])
        # undefined -> revert + user review
        d = S.kappa_decision_core(None, None)
        self.assertEqual((d["decision"], d["requires_user_review"]), ("revert_to_kappa_1", True))

    def test_kappa_pins_equal_the_shared_registration(self):
        try:
            from benchmark.ppl import prc_r7_prereg_20260928 as PR
        except ImportError:
            self.skipTest("prc_r7_prereg_20260928 not present")
        self.assertEqual(json.loads(json.dumps(PR.KAPPA_PINS)), S.KAPPA_PINS)
        self.assertEqual(PR.KAPPA_DECISION_RATIO, S.KAPPA_DECISION_RATIO)

    def test_kappa_pooled_statistic(self):
        rng = np.random.default_rng(4)
        w = list(range(16))
        d32 = list(rng.normal(0.020, 0.006, 16))
        d40 = list(rng.normal(0.012, 0.005, 16))
        meas = {"30B_t32": {"window_dnll": d32, "windows": w, "ok": True},
                "30B_t40": {"window_dnll": d40, "windows": w, "ok": True}}
        preds = {"30B_t32": 0.009, "30B_t40": 0.006}
        d = S.kappa_decision_pooled(meas, preds, {"30B_t32": True, "30B_t40": True})
        sums = [a + b for a, b in zip(d32, d40)]
        M = sum(sums) / 16
        sd = math.sqrt(sum((x - M) ** 2 for x in sums) / 15)
        self.assertAlmostEqual(d["kappa_lin_bs"], M / 0.015, places=12)
        self.assertAlmostEqual(d["se_kappa_lin_bs"], sd / 4 / 0.015, places=12)
        self.assertEqual(d["decision"], "keep_0.42" if d["kappa_lin_bs"] - 0.83 > d["band"]
                         else "revert_to_kappa_1")
        # undefined cases all revert with a review flag
        for bad_meas, bad_preds, ident in (
                ({"30B_t32": meas["30B_t32"]}, preds, None),                          # cell missing
                (dict(meas, **{"30B_t40": dict(meas["30B_t40"], ok=False)}), preds, None),
                (dict(meas, **{"30B_t40": dict(meas["30B_t40"], simulated=True)}), preds, None),
                (meas, {"30B_t32": 0.009}, None),                                      # p missing
                (meas, {"30B_t32": 0.009, "30B_t40": -0.02}, None),                    # sum p <= 0
                (meas, preds, {"30B_t32": True, "30B_t40": False}),                    # identity
                (dict(meas, **{"30B_t40": dict(meas["30B_t40"], windows=w[::-1])}), preds, None)):
            d = S.kappa_decision_pooled(bad_meas, bad_preds, ident)
            self.assertEqual((d["decision"], d["requires_user_review"]), ("revert_to_kappa_1", True),
                             d["undefined_reasons"])
            self.assertTrue(d["undefined_reasons"])

    def test_kappa_for_cell_and_eval_side_records(self):
        dense = S.kappa_for_cell({"family_kappa": False})
        self.assertEqual((dense["branch"], dense["kappa_lin"], dense["currency_ratio"]), ("dense", 1.0, 1.0))
        with self.assertRaises(ValueError):
            S.kappa_for_cell({"family_kappa": False}, None, {"kappa_lin": 1.96, "kappa_att": 0.83})
        with self.assertRaises(ValueError):
            S.kappa_for_cell({"family_kappa": True}, None)
        own = json.loads(json.dumps(S.kappa_decision_core(2.0, 0.2)))        # as read from a file
        k = S.kappa_for_cell({"family_kappa": True}, own, {"kappa_lin": 1.96, "kappa_att": 0.83})
        self.assertEqual((k["branch"], k["currency_ratio"]), ("keep_0.42", 0.83 / 1.96))
        with self.assertRaises(ValueError):                                   # wrong pins
            S.kappa_for_cell({"family_kappa": True}, own, {"kappa_lin": 0.83, "kappa_att": 0.83})
        with self.assertRaises(ValueError):                                   # tampered own record
            S.kappa_for_cell({"family_kappa": True}, dict(own, decision="revert_to_kappa_1"))
        with self.assertRaises(ValueError):
            S.kappa_for_cell({"family_kappa": True}, {"schema": "x"})
        # an eval-side (two-sided v1) record: the branch is RE-DERIVED one-sidedly
        ev = {"schema": S.EVAL_KAPPA_SCHEMA, "decision": "keep_0.42", "ratio": 0.42,
              "kappa_lin_bs": 0.30, "se_kappa_lin_bs": 0.06, "flags": ["direction_contradicted"]}
        k = S.kappa_for_cell({"family_kappa": True}, ev)
        self.assertEqual(k["branch"], "revert_to_kappa_1")
        self.assertFalse(k["decision_info"]["record_agrees"])
        with self.assertRaises(ValueError) as cm:                             # its keep pins refused
            S.kappa_for_cell({"family_kappa": True}, ev, {"kappa_lin": 1.96, "kappa_att": 0.83})
        self.assertIn("reconcile", str(cm.exception))
        ev2 = dict(ev, kappa_lin_bs=2.1, se_kappa_lin_bs=0.3, flags=[])
        self.assertEqual(S.kappa_for_cell({"family_kappa": True}, ev2)["branch"], "keep_0.42")
        ev3 = dict(ev, kappa_lin_bs=None, se_kappa_lin_bs=None,
                   flags=["undefined: missing/failed cells ['30B_t40']"])
        k3 = S.kappa_for_cell({"family_kappa": True}, ev3)
        self.assertEqual((k3["branch"], k3["requires_user_review"]), ("revert_to_kappa_1", True))

    def test_check_prereg(self):
        good = {"primary": "UK", "arms": ["UK", "K", "U"], "ladder_policy": "up", "mde_nats": 0.0116,
                "s0": 1.0037, "family_kappa": True}
        self.assertIs(S.check_prereg(good), good)
        for bad in (dict(good, ladder_policy="dense"), dict(good, ladder_policy="full"),
                    dict(good, primary="J"), dict(good, arms=["UK", "J"]), dict(good, primary="X"),
                    dict(good, mde_nats=0), dict(good, mde_nats=None),
                    {k: v for k, v in good.items() if k != "mde_nats"}):
            with self.assertRaises((ValueError, TypeError, SystemExit)):
                S.check_prereg(bad)

    def test_arm_names(self):
        self.assertEqual(S.arm_table_name("UK", 0.42, 1.0043), "r7UK_k0420_s10043")
        self.assertEqual(S.arm_table_name("U", 1.0, 1.0), "r7U_k1000_s10000")
        with self.assertRaises(ValueError):
            S.arm_table_name("X", 1, 1)


@unittest.skipUnless((PARENT_4B_T40 / "table.json").is_file(), "4B t40 parent config missing")
class TestEmissionAndValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.table = json.loads((PARENT_4B_T40 / "table.json").read_text())
        self.wrapper = json.loads((PARENT_4B_T40 / "wrapper.json").read_text())
        rng = np.random.default_rng(5)
        self.glob = sorted(int(v) for v in self.table["stoc_len_levels"])
        self.lin = {(op, b): sorted(float(np.float32(x)) for x in rng.uniform(0, 1, 16))
                    for op in S.LINEAR_OPS for b in range(4)}
        self.A = {(op, b): sorted(set(self.glob) | {112, 128}) for op in S.ATT_OPS for b in range(4)}
        self.att = {k: sorted((float(np.float32(x)) for x in rng.uniform(0, 1, len(A) - 1)),
                              reverse=True) for k, A in self.A.items()}
        self.att_glob = {k: sorted((float(np.float32(x)) for x in rng.uniform(0, 1, len(self.glob) - 1)),
                                   reverse=True) for k in self.A}

    def tearDown(self):
        self.tmp.cleanup()

    def test_emit_equals_calib9_bytes_without_ladders(self):
        rec = {"x": 1, "table": "gfisla"}
        for prov, extra in (({}, None), ({"prc9_r6": {"a": 1}}, {"a": 1})):
            d9, d7 = self.d / f"c9{bool(prov)}", self.d / f"r7{bool(prov)}"
            d9.mkdir()
            d7.mkdir()
            o9, t9, _ = c9.emit_table(d9 / "c.json", "gfisla", self.lin, self.att_glob, self.table,
                                      self.wrapper, DENSE_LADDER, rec, extra)
            o7, t7 = S.emit_r7_table(d7 / "c.json", "gfisla", self.lin, self.att_glob,
                                     {k: self.glob for k in self.A}, self.table, self.wrapper,
                                     DENSE_LADDER, rec, prov)
            self.assertEqual(t9.read_bytes(), t7.read_bytes())

    def test_extended_ladders_load_validate_and_sweep(self):
        o, t = S.emit_r7_table(self.d / "c.json", "r7UK_k0420_s10000", self.lin, self.att, self.A,
                               self.table, self.wrapper, DENSE_LADDER, {"x": 1},
                               {S.R7_KEY: {"arm": "UK"}})
        cand = json.loads(t.read_text())
        for k, A in self.A.items():
            b = cand["buckets"][f"{k[0]}:t0:l{k[1]}"]
            self.assertEqual(b["stoc_len_levels"], sorted(A, reverse=True))
            self.assertEqual(len(b["thresholds"]), len(A) - 1)
        self.assertEqual(S.validate_lineage(self.table, cand, prc_ladder=DENSE_LADDER), [])
        rep = S.resolver_sweep(o, t, PARENT_4B_T40, 36)
        self.assertEqual(rep["attention_op_blocks"], 2 * 36)
        self.assertEqual(rep["max_attention_length"], 128)
        self.assertEqual(rep["linear_op_blocks"], 7 * 36)
        self.assertGreater(rep["planted_rows"], 0)
        self.assertEqual(rep["calib_rule_vs_runtime_mirror_rows"], 0)

    def test_lineage_rejects_non_allocation_edits(self):
        _, t = S.emit_r7_table(self.d / "c.json", "r7UK_k0420_s10000", self.lin, self.att, self.A,
                               self.table, self.wrapper, DENSE_LADDER, {"x": 1}, {})
        good = json.loads(t.read_text())
        self.assertEqual(S.validate_lineage(self.table, good), [])
        self.assertTrue(S.validate_lineage(self.table, good, allow_bucket_ladders=False))

        def bad(fn):
            c = copy.deepcopy(good)
            fn(c)
            return S.validate_lineage(self.table, c)
        self.assertTrue(bad(lambda c: c["buckets"]["qk:t0:l0"].__setitem__(
            "stoc_len_levels", [256] + c["buckets"]["qk:t0:l0"]["stoc_len_levels"][1:])))
        self.assertTrue(bad(lambda c: c["buckets"]["qk:t0:l0"].__setitem__(
            "stoc_len_levels", list(reversed(c["buckets"]["qk:t0:l0"]["stoc_len_levels"])))))
        self.assertTrue(bad(lambda c: c["buckets"]["qk:t0:l0"]["thresholds"].pop()))
        self.assertTrue(bad(lambda c: c["buckets"]["av:t0:l2"].__setitem__("metric_mean", 0.5)))
        self.assertTrue(bad(lambda c: c["protected_channels"].__setitem__("stoc_len", 96)))
        self.assertTrue(bad(lambda c: c.__setitem__("stoc_len_levels", [128] + c["stoc_len_levels"][1:])))
        self.assertTrue(bad(lambda c: c["buckets"]["q_proj:t0:l1"]["thresholds"].__setitem__(0, 0.123)))
        self.assertTrue(bad(lambda c: c["per_row_chunk"].__setitem__("layer_buckets", 36)))
        self.assertTrue(bad(lambda c: c["per_row_chunk"]["buckets"]["q_proj:t0:l0"].__setitem__(
            "levels", [8, 256])))
        self.assertTrue(bad(lambda c: c.__setitem__("dispatch_metrics", {})))

    def test_emit_refuses_bad_ladders(self):
        for A in ([30, 40, 129], [40, 40, 64], [64]):
            bad = dict(self.A)
            bad[("qk", 0)] = A
            att = dict(self.att)
            att[("qk", 0)] = [0.5] * max(len(A) - 1, 0)
            with self.assertRaises(SystemExit):
                S.emit_r7_table(self.d / f"b{len(A)}{A[-1]}.json", "x", self.lin, att, bad, self.table,
                                self.wrapper, DENSE_LADDER, {}, {})

    def test_sweep_rejects_wrapper_change(self):
        o, t = S.emit_r7_table(self.d / "c.json", "x", self.lin, self.att, self.A, self.table,
                               self.wrapper, DENSE_LADDER, {}, {})
        w = json.loads(o.read_text())
        w["escape_stoc_len"] = 64
        o.write_text(json.dumps(w))
        with self.assertRaises(ValueError):
            S.resolver_sweep(o, t, PARENT_4B_T40, 36)


@unittest.skipUnless((PARENT_4B_T40 / "table.json").is_file(), "4B t40 parent config missing")
class TestSyntheticCaptureEndToEnd(unittest.TestCase):
    """A synthetic 512-row prefix-split capture on the real 4B t40 parent table."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.p = build_capture(cls.d, PARENT_4B_T40)
        cls.cap = load_capture(cls.p)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def identity(self, cap=None, **kw):
        p = self.p
        return S.run_identity(capture=cap or self.cap, stem=p["stem"], parent_dir=PARENT_4B_T40,
                              ref_gfis=p["ref_gfis"], ref_gfisla=p["ref_gfisla"],
                              ref_diag=p["ref_diag"], inc_table=p["inc"], prefix=128,
                              expect_joint_line=p["joint_line"],
                              capture_score_tables=p["score_tables"], **kw)

    def write_identity(self, d, cap=None):
        rep = self.identity(cap)
        rep["cell"] = "4B_t40"
        path = Path(d) / "identity.json"
        S.write_json(path, rep, exclusive=True)
        return path

    def test_identity_exact(self):
        rep = self.identity()
        self.assertTrue(rep["ok"], json.dumps(rep["checks"], indent=1))
        self.assertTrue(rep["identity_ok"])
        self.assertEqual(rep["identity_level"], "exact")
        self.assertTrue(all(rep["checks"]["exact"].values()))
        own = rep["notes"]["own_table_scores"]
        self.assertTrue(own["ok"], own)
        self.assertEqual(sorted(own["equal"]), sorted(["joint.gfis", "joint.gfisla", "tables.gfis",
                                                       "tables.gfisla", "tables.c17",
                                                       "tables.c17_s80", "tables.inc"]))

    def test_identity_detects_a_tampered_non_prefix_hold_row(self):
        """Check (7): a hold attention row OUTSIDE the 128 prefix (rank >= 128) is invisible to the
        c7 comparisons but must break the capture's own-diag re-score (cpu, exact even for near)."""
        with tempfile.TemporaryDirectory() as e_:
            z = dict(np.load(self.p["att"]))
            key = next(f for f in z if f.startswith("att|hold|av|") and f.endswith("|fis"))
            rank = z[key[:-len("fis")] + "rank"]
            row = int(np.nonzero(rank >= 128)[0][3])
            z[key] = z[key].copy()
            z[key][row, :] *= np.float32(1.5)
            bad = Path(e_) / "att.npz"
            np.savez_compressed(bad, **z)
            cap = S.Capture(self.p["state"], bad, self.p["hold"])
            rep = self.identity(cap, allow_near=True)
            self.assertTrue(rep["checks"]["exact"]["prefix_preds_equal_c7_diag"])   # prefix blind
            self.assertFalse(rep["checks"]["cpu_exact"]["own_tables_rescored_equal_capture_diag"])
            self.assertEqual(rep["identity_level"], "fail")
            self.assertFalse(rep["identity_ok"])

    def test_prefix_view_is_the_128_row_sample(self):
        for k in self.cap.akeys:
            for ph in ("calib", "hold"):
                got = self.cap.att_parts(ph, k, 128)
                want = self.p["att7"][ph][k]
                self.assertEqual(len(got), len(want))
                for a, b in zip(got, want):
                    for f in ("mn", "mac", "fis", "raw", "parL"):
                        self.assertEqual(a[f].dtype, b[f].dtype)
                        self.assertTrue(np.array_equal(a[f], b[f]), (k, ph, f))

    def test_identity_detects_a_tampered_dump(self):
        with tempfile.TemporaryDirectory() as e_:
            z = dict(np.load(self.p["att"]))
            key = next(f for f in z if f.startswith("att|calib|qk|") and f.endswith("|fis"))
            lv = [int(v) for v in z[key[:-len("fis")] + "lv"]]
            col = lv.index(int(self.cap.att_ladder[self.cap.akeys[0]][0][-1]))   # a parent rung
            z[key] = z[key].copy()
            z[key][5, col] *= np.float32(1.5)                                    # rank 5: prefix row
            bad = Path(e_) / "att.npz"
            np.savez_compressed(bad, **z)
            cap = S.Capture(self.p["state"], bad, self.p["hold"])
            rep = self.identity(cap)
            self.assertFalse(rep["ok"])
            self.assertEqual(rep["identity_level"], "fail")
            self.assertFalse(rep["checks"]["cpu_exact"]["att_dump_bins_equal_state"])

    def test_identity_detects_wrong_reference(self):
        with tempfile.TemporaryDirectory() as e_:
            t = json.loads(Path(self.p["ref_gfisla"]).read_text())
            kk = "qk:t0:l1"
            t["buckets"][kk]["thresholds"][0] = min(1.0, t["buckets"][kk]["thresholds"][0] + 0.01)
            bad = Path(e_) / "g.json"
            bad.write_text(json.dumps(t))
            rep = S.run_identity(capture=self.cap, stem=self.p["stem"], parent_dir=PARENT_4B_T40,
                                 ref_gfis=self.p["ref_gfis"], ref_gfisla=bad,
                                 ref_diag=self.p["ref_diag"], inc_table=bad, prefix=128)
            self.assertFalse(rep["checks"]["exact"]["prefix_joint_equals_c7_gfisla"])
            self.assertFalse(rep["ok"])

    def test_near_level_requires_permission(self):
        with tempfile.TemporaryDirectory() as e_:
            diag = json.loads(Path(self.p["ref_diag"]).read_text())
            diag["joint"]["gfisla"]["pred_dnll"] += 1e-9       # 'last-bit' difference
            bad = Path(e_) / "diag.json"
            bad.write_text(json.dumps(diag))
            kw = dict(capture=self.cap, stem=self.p["stem"], parent_dir=PARENT_4B_T40,
                      ref_gfis=self.p["ref_gfis"], ref_gfisla=self.p["ref_gfisla"], ref_diag=bad,
                      inc_table=self.p["inc"], prefix=128, expect_joint_line=self.p["joint_line"])
            rep = S.run_identity(**kw)
            self.assertEqual(rep["identity_level"], "near")
            self.assertFalse(rep["ok"])
            rep = S.run_identity(allow_near=True, **kw)
            self.assertTrue(rep["ok"])
            diag["joint"]["gfisla"]["pred_dnll"] += 1e-3        # a real difference
            bad.write_text(json.dumps(diag))
            rep = S.run_identity(allow_near=True, **kw)
            self.assertEqual(rep["identity_level"], "fail")

    def test_inc_relation_prc_only(self):
        with tempfile.TemporaryDirectory() as e_:
            p = build_capture(Path(e_), PARENT_4B_T40, seed=3, inc_relation="prc_thresholds_only")
            cap = load_capture(p)
            rep = S.run_identity(capture=cap, stem=p["stem"], parent_dir=PARENT_4B_T40,
                                 ref_gfis=p["ref_gfis"], ref_gfisla=p["ref_gfisla"],
                                 ref_diag=p["ref_diag"], inc_table=p["inc"], prefix=128,
                                 inc_relation="prc_thresholds_only")
            self.assertTrue(rep["ok"], rep["checks"])

    def test_kappa1_inherit_is_calib9_gfisla(self):
        """J (and K at kappa_att == kappa_lin) == calib9's resolve_tables gfisla on the state."""
        cap = self.cap
        for s in (1.0, 1.0043):
            res = c9.resolve_tables(cap.meta, cap.keys, cap.akeys, cap.lbins, cap.abins_state,
                                    [s], ["gfisla"])
            lin, att, _ = res[f"gfisla{c9.scale_suffix(s)}"]
            sol = S.solve_arm(cap, S.arm_ladders(cap, "J", "up"), 1.0, s)
            self.assertEqual(sol["lin_th"], lin)
            self.assertEqual(sol["att_th"], att)

    def test_kappa_currency_equals_scaled_bins(self):
        cap = self.cap
        lads = S.arm_ladders(cap, "UK", "up")
        sol = S.solve_arm(cap, lads, 0.42, 1.0)
        ab, fixed, par_att = cap.abins_for(lads)
        scaled = {k: (E * 0.42, C, b, A) for k, (E, C, b, A) in ab.items()}
        rj = c9.solve_joint(cap.lbins, scaled, cap.keys, cap.akeys, cap.Lv, fixed,
                            cap.par_lin + par_att)
        for k in cap.akeys:
            self.assertTrue(np.array_equal(rj[1][k], sol["rj"][1][k]))
        sol1 = S.solve_arm(cap, lads, 1.0, 1.0)
        self.assertLessEqual(sol["att_L"], sol1["att_L"] + 1e-9)       # cheaper attention error
        self.assertLessEqual(sol["calib_cost"], sol["target"])
        self.assertGreater(sol["calib_cost_over_target"], 0.99)

    def test_scorer_matches_calib9_hj(self):
        cap = self.cap
        rj7 = json.loads(Path(self.p["ref_gfisla"]).read_text())
        g7 = json.loads(Path(self.p["ref_gfis"]).read_text())
        sc = S.score_tables(cap, {"parent": None, "gfis": g7, "gfisla": rj7}, prefix=128)
        pr = S.preds(sc, "parent", cap.n_tok)
        diag = json.loads(Path(self.p["ref_diag"]).read_text())
        self.assertEqual(pr["gfisla"]["pred_dnll"], diag["joint"]["gfisla"]["pred_dnll"])
        self.assertEqual(pr["gfis"]["pred_dnll"], diag["joint"]["gfis"]["pred_dnll"])
        self.assertAlmostEqual(pr["gfisla"]["pred_dnll_lin"] + pr["gfisla"]["pred_dnll_att"],
                               pr["gfisla"]["pred_dnll"], places=12)

    PREREG = {"primary": "UK", "arms": ["UK", "K", "U"], "ladder_policy": "up",
              "mde_nats": 0.006, "s0": 1.0043, "family_kappa": True}

    def test_solve_arms_emit_validate_score(self):
        dec = S.kappa_decision_core(2.0, 0.2)                              # KEEP
        with tempfile.TemporaryDirectory() as e_:
            ident = self.write_identity(e_)
            out = Path(e_) / "solve"
            s = S.run_solve(capture=self.cap, parent_dir=PARENT_4B_T40, inc_table=self.p["inc"],
                            out_dir=out, cell="4B_t40", prereg=self.PREREG, identity_path=ident,
                            kappa_decision=dec, sensitivity_ratios=[0.35, 0.61],
                            sensitivity_policies=["dense"])
            names = sorted(s["arms"])
            r = 0.83 / 1.96
            self.assertEqual(names, sorted([S.arm_table_name("UK", r, 1.0043),
                                            S.arm_table_name("K", r, 1.0043),
                                            S.arm_table_name("U", 1.0, 1.0043)]))
            self.assertEqual(s["primary"], S.arm_table_name("UK", r, 1.0043))
            self.assertEqual(s["mde_nats"], 0.006)
            self.assertEqual(s["identity"]["identity_level"], "exact")
            self.assertEqual((s["kappa_lin"], s["kappa_att"], s["kappa_branch"]), (1.96, 0.83, "keep_0.42"))
            for n, rec in s["arms"].items():
                t = json.loads(Path(rec["table_path"]).read_text())
                self.assertEqual(t[S.R7_KEY]["table"], n)
                self.assertLessEqual(rec["resolver_sweep"]["max_attention_length"], 128)
                hv = rec["heldout_vs_inc"]          # the eval side's exact recomputation
                self.assertEqual(rec["pred_true_vs_inc"],
                                 1.96 * float(hv["pred_dnll_lin"]) + 0.83 * float(hv["pred_dnll_att"]))
                self.assertEqual(rec["kappa_ratio_used"], 1.0 if rec["arm"] == "U" else 0.83 / 1.96)
                a_ = rec["attribution_pred_true"]
                self.assertAlmostEqual(a_["allocation_vs_J_same_scale"] + a_["budget_J_scale_vs_J_1"]
                                       + a_["resample_J_1_vs_INC"], rec["pred_true_vs_inc"], places=12)
                self.assertEqual(rec["eligible"], rec["pred_true_vs_inc"] <= -0.006)
                kinds = {b.get("stoc_len_levels") is not None for kk, b in t["buckets"].items()
                         if kk.split(":")[0] in S.ATT_OPS}
                self.assertEqual(kinds, {rec["ladder_policy"] != "inherit"})
                self.assertLessEqual(rec["calib_cost_over_target"], 1.0)
                self.assertIn("allocation_vs_J_same_scale", rec["attribution_pred_true"])
            # J controls at s0 and 1.0: in memory only, never emitted, never arms
            self.assertEqual(sorted(c["target_scale"] for c in s["controls"].values()), [1.0, 1.0043])
            self.assertFalse(any("J" in p_.name for p_ in out.iterdir()))
            sens = s["sensitivity_not_preregistered"]
            self.assertTrue({S.arm_table_name(a, rr, 1.0043) for a in ("UK", "K")
                             for rr in (0.35, 0.61)} <= set(sens))
            self.assertIn(S.arm_table_name("UK", r, 1.0043) + "_dense", sens)
            self.assertEqual(sorted(p_.name for p_ in out.iterdir() if p_.name.endswith("_table.json")),
                             sorted(f"4B_t40_{n}_table.json" for n in names))
            self.assertTrue((out / "solve_summary.json").is_file())
            with self.assertRaises(FileExistsError):                  # never overwrites a solve
                S.run_solve(capture=self.cap, parent_dir=PARENT_4B_T40, inc_table=self.p["inc"],
                            out_dir=out, cell="4B_t40", prereg=self.PREREG, identity_path=ident,
                            kappa_decision=dec)

    def test_solve_revert_and_refusal(self):
        dec = S.kappa_decision_core(0.30, 0.06)                            # REVERT (A1)
        with tempfile.TemporaryDirectory() as e_:
            ident = self.write_identity(e_)
            s = S.run_solve(capture=self.cap, parent_dir=PARENT_4B_T40, inc_table=self.p["inc"],
                            out_dir=Path(e_) / "o", cell="4B_t40",
                            prereg=dict(self.PREREG, mde_nats=10.0), identity_path=ident,
                            kappa_decision=dec, emit=False)
            self.assertEqual(sorted(s["arms"]), sorted([S.arm_table_name("UK", 1.0, 1.0043),
                                                        S.arm_table_name("K", 1.0, 1.0043)]))
            self.assertIn("U", s["omitted_arms"])
            self.assertEqual(s["kappa_currency_ratio"], 1.0)
            self.assertEqual((s["kappa_lin"], s["kappa_att"]), (0.83, 0.83))
            self.assertFalse(any(r["eligible"] for r in s["arms"].values()))     # MDE 10 nats
            k = s["arms"][S.arm_table_name("K", 1.0, 1.0043)]
            j = s["controls"]["control_" + S.arm_table_name("J", 1.0, 1.0043)]
            self.assertEqual(k["heldout_vs_inc"], j["heldout_vs_inc"])        # K at kappa 1 == J

    def test_solve_fail_closed_and_prereg_enforced(self):
        dec = S.kappa_decision_core(2.0, 0.2)
        kw = dict(capture=self.cap, parent_dir=PARENT_4B_T40, inc_table=self.p["inc"], cell="4B_t40",
                  kappa_decision=dec, emit=False)
        with tempfile.TemporaryDirectory() as e_:
            with self.assertRaises(SystemExit):                           # no identity record
                S.run_solve(out_dir=Path(e_) / "a", prereg=self.PREREG,
                            identity_path=Path(e_) / "none.json", **kw)
            ident = self.write_identity(e_)
            rep = json.loads(ident.read_text())
            for mut in (lambda r: r.__setitem__("identity_ok", False),
                        lambda r: r.__setitem__("cell", "30B_t32"),
                        lambda r: r["files"]["state"].__setitem__("sha256", "0" * 64)):
                r2 = copy.deepcopy(rep)
                mut(r2)
                bad = Path(e_) / "bad_identity.json"
                bad.write_text(json.dumps(r2))
                with self.assertRaises(SystemExit):
                    S.run_solve(out_dir=Path(e_) / "b", prereg=self.PREREG, identity_path=bad, **kw)
            for pre in (dict(self.PREREG, ladder_policy="full"), dict(self.PREREG, primary="J")):
                with self.assertRaises(ValueError):
                    S.run_solve(out_dir=Path(e_) / "c", prereg=pre, identity_path=ident, **kw)
            with self.assertRaises(SystemExit):                           # arm not registered
                S.run_solve(out_dir=Path(e_) / "d", prereg=dict(self.PREREG, arms=["UK"]),
                            identity_path=ident, arms=["UK", "K"], **kw)
            with self.assertRaises(SystemExit):                           # initial solve w/o primary
                S.run_solve(out_dir=Path(e_) / "e", prereg=self.PREREG, identity_path=ident,
                            arms=["K"], **kw)
            with self.assertRaises(ValueError):                           # family cell w/o decision
                S.run_solve(out_dir=Path(e_) / "f", prereg=self.PREREG, identity_path=ident,
                            **dict(kw, kappa_decision=None))

    def test_solve_fixed_point_record(self):
        """Non-s0 scales only from the registered (eval-side) fixed-point record, each arm at its
        OWN s1; the record must match the cell, s0 and the kappa pins."""
        dec = S.kappa_decision_core(2.0, 0.2)
        with tempfile.TemporaryDirectory() as e_:
            ident = self.write_identity(e_)
            fp = {"cell": "4B_t40", "s0": 1.0043, "simulated": False,
                  "kappa_pins": {"kappa_lin": 1.96, "kappa_att": 0.83, "branch": "keep_0.42"},
                  "arms": {"UK": {"status": "ok", "s1": 1.0117}, "K": {"status": "ok", "s1": 1.0102},
                           "U": {"status": "refused: out of band", "s1": None}}}
            kw = dict(capture=self.cap, parent_dir=PARENT_4B_T40, inc_table=self.p["inc"],
                      cell="4B_t40", prereg=self.PREREG, identity_path=ident, kappa_decision=dec,
                      emit=False)
            s = S.run_solve(out_dir=Path(e_) / "o", scales=[1.0117, 1.0102], fixed_point_record=fp, **kw)
            r = 0.83 / 1.96
            self.assertEqual(sorted(s["arms"]), sorted([S.arm_table_name("UK", r, 1.0117),
                                                        S.arm_table_name("K", r, 1.0102)]))
            self.assertEqual(s["stage"], "fixed_point_s1")
            self.assertIn("U", s["omitted_arms"])
            for bad_kw in (dict(scales=[1.0200], fixed_point_record=fp),          # not registered
                           dict(scales=[1.0117]),                                 # no record
                           dict(scales=[1.0117], fixed_point_record=dict(fp, cell="30B_t32")),
                           dict(scales=[1.0117], fixed_point_record=dict(fp, s0=1.0037)),
                           dict(scales=[1.0117], fixed_point_record=dict(fp, simulated=True)),
                           dict(scales=[1.0117], fixed_point_record=dict(
                               fp, kappa_pins={"kappa_lin": 0.83, "kappa_att": 0.83}))):
                with self.assertRaises(SystemExit):
                    S.run_solve(out_dir=Path(e_) / "x", **bad_kw, **kw)

    def test_kappa_att_A_is_descriptive(self):
        """The round-6 A table analogue (qk buckets on [128] + g[1:]) is scored vs INC and gives a
        non-binding kappa_att(A); nothing else in the summary depends on it."""
        dec = S.kappa_decision_core(2.0, 0.2)
        with tempfile.TemporaryDirectory() as e_:
            ident = self.write_identity(e_)
            inc = json.loads(Path(self.p["inc"]).read_text())
            glob = sorted(int(v) for v in inc["stoc_len_levels"])
            A = sorted([128] + glob[:-1], reverse=True)
            for b in range(4):
                inc["buckets"][f"qk:t0:l{b}"]["stoc_len_levels"] = A
            at = Path(e_) / "A_table.json"
            at.write_text(json.dumps(inc))
            r6a = {"table": str(at), "sha256": S.sha256(at), "measured_dnll": -0.02,
                   "measured_se": 0.004}
            kw = dict(capture=self.cap, parent_dir=PARENT_4B_T40, inc_table=self.p["inc"],
                      cell="4B_t40", prereg=self.PREREG, identity_path=ident, kappa_decision=dec,
                      emit=False)
            s = S.run_solve(out_dir=Path(e_) / "o", r6_arm_a=r6a, **kw)
            s0 = S.run_solve(out_dir=Path(e_) / "o2", **kw)
            ka = s["kappa_att_A"]
            self.assertFalse(ka["binding"])
            self.assertEqual(ka["pred_vs_inc"]["pred_dnll_lin"], 0.0)
            if ka["pred_vs_inc"]["pred_dnll_att"] < 0:
                self.assertAlmostEqual(ka["kappa_att_A"], -0.02 / ka["pred_vs_inc"]["pred_dnll_att"])
            self.assertEqual({n: r["pred_true_vs_inc"] for n, r in s["arms"].items()},
                             {n: r["pred_true_vs_inc"] for n, r in s0["arms"].items()})

    def test_pipeline_check(self):
        with tempfile.TemporaryDirectory() as e_:
            rep = S.pipeline_check(self.cap, PARENT_4B_T40, Path(e_) / "pc")
            self.assertTrue(rep["ok"], rep["lineage_problems"])
            self.assertEqual(rep["resolver_sweep"]["max_attention_length"], 128)
            esc = rep["escape"]
            self.assertTrue(all(v["has_128_rung"] for v in esc.values()))
            old = S.pipeline_check
            S.pipeline_check = lambda *a_, **k_: (_ for _ in ()).throw(ValueError("boom"))
            try:
                g = S.guarded_pipeline_check(self.cap, PARENT_4B_T40, Path(e_) / "pc2")
            finally:
                S.pipeline_check = old
            self.assertFalse(g["ok"])
            self.assertIn("ValueError: boom", g["error"])

    def test_driver_identity_always_written(self):
        """prc_r7_capture run_identity: identity.json is written even when the pipeline check
        raises (exit 4, capture stays valid); a --tag re-run writes new files."""
        from benchmark.ppl import prc_r7_capture_20260928 as D
        p = self.p
        with tempfile.TemporaryDirectory() as e_:
            cell = {"id": "4B_t40",
                    "paths": {"state": str(p["state"]), "att_dump": str(p["att"]),
                              "hold_dump": str(p["hold"]), "stem": str(p["stem"]),
                              "calib_log": str(Path(e_) / "calib10.log"),
                              "identity": str(Path(e_) / "identity.json"),
                              "pipeline_dir": str(Path(e_) / "pipeline_check")},
                    "refs": {"c7_gfis_table": str(p["ref_gfis"]), "c7_gfisla_table": str(p["ref_gfisla"]),
                             "c7_diag": str(p["ref_diag"]), "inc_table": str(p["inc"]),
                             "expect_joint_line": p["joint_line"], "expect_global_line": "x",
                             "inc_relation": "identical"},
                    "r7": {"att_prefix_split": 128, "att_rows_per_call": 512,
                           "score_tables": {k: str(v) for k, v in p["score_tables"].items()}},
                    "calib_args": {"parent": str(PARENT_4B_T40)}}
            mf = Path(e_) / "m.json"
            mf.write_text("{}")
            man = {"preregistered_rules": {"identity_allow_near": False}, "_path": str(mf)}
            old = S.pipeline_check
            S.pipeline_check = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("emitter bug"))
            try:
                rc = D.run_identity(man, cell)
            finally:
                S.pipeline_check = old
            self.assertEqual(rc, 4)
            rep = json.loads((Path(e_) / "identity.json").read_text())
            self.assertTrue(rep["identity_ok"])
            self.assertFalse(rep["ok"])
            self.assertIn("emitter bug", rep["pipeline_check"]["error"])
            with self.assertRaises(SystemExit):
                D.run_identity(man, cell)                                 # never overwrites
            self.assertEqual(D.run_identity(man, cell, tag="rerun1"), 0)
            rep2 = json.loads((Path(e_) / "identity_rerun1.json").read_text())
            self.assertTrue(rep2["ok"] and rep2["pipeline_check"]["ok"])
            self.assertTrue((Path(e_) / "pipeline_check_rerun1").is_dir())

    def test_score_cli_and_identity_cli(self):
        from benchmark.ppl.prc_r7_solve import main as smain
        with tempfile.TemporaryDirectory() as e_:
            out = Path(e_) / "id.json"
            st = ",".join(f"{k}={v}" for k, v in self.p["score_tables"].items())
            rc = smain(["identity", "--stem", str(self.p["stem"]), "--parent", str(PARENT_4B_T40),
                        "--ref-gfis", str(self.p["ref_gfis"]), "--ref-gfisla", str(self.p["ref_gfisla"]),
                        "--ref-diag", str(self.p["ref_diag"]), "--inc-table", str(self.p["inc"]),
                        "--expect-joint-line", self.p["joint_line"], "--out", str(out),
                        "--score-tables", st, "--pipeline-dir", str(Path(e_) / "pc")])
            self.assertEqual(rc, 0)
            rep = json.loads(out.read_text())
            self.assertTrue(rep["ok"] and rep["identity_ok"])
            self.assertTrue(rep["pipeline_check"]["ok"])
            self.assertIn("tables.c17_s80", rep["notes"]["own_table_scores"]["equal"])
            rc = smain(["score", "--stem", str(self.p["stem"]), "--parent", str(PARENT_4B_T40),
                        "--tables", f"inc={self.p['inc']}", "--out", str(Path(e_) / "sc.json")])
            self.assertEqual(rc, 0)

    def test_solve_and_kappa_decision_clis(self):
        """solve reads everything registered from the manifest; kappa-decision computes the pooled
        one-sided rule from step0_measured.json + capture diags + identity records."""
        from benchmark.ppl.prc_r7_solve import main as smain
        p = self.p
        with tempfile.TemporaryDirectory() as e_:
            e = Path(e_)
            ident = self.write_identity(e_)
            rng = np.random.default_rng(8)
            cells = []
            for cid, pc in (("30B_t32", 0.009), ("30B_t40", 0.006)):
                cd = e / cid
                cd.mkdir()
                diag = cd / "diag.json"
                diag.write_text(json.dumps({"tables": {"c17": {"pred_dnll_fis": -0.001},
                                                       "c17_s80": {"pred_dnll_fis": -0.001 + pc}}}))
                (cd / "identity.json").write_text(json.dumps({"identity_ok": True}))
                cells.append({"id": cid, "enabled": True,
                              "inputs": {"c17_table": {"sha256": S.sha256(p["score_tables"]["c17"])},
                                         "c17_s80_table": {"sha256": S.sha256(p["score_tables"]["c17_s80"])}},
                              "paths": {"diag": str(diag), "identity": str(cd / "identity.json")}})
            cells.append({"id": "4B_t40", "enabled": True,
                          "paths": {"state": str(p["state"]), "att_dump": str(p["att"]),
                                    "hold_dump": str(p["hold"]), "identity": str(ident),
                                    "stem": str(p["stem"])},
                          "calib_args": {"parent": str(PARENT_4B_T40)},
                          "refs": {"inc_table": str(p["inc"])},
                          "prereg": dict(self.PREREG, family_kappa=True)})
            man = e / "manifest.json"
            man.write_text(json.dumps({"preregistered_rules": {"kappa_rule": S.KAPPA_RULE},
                                       "output_root": str(e / "out"), "cells": cells}))
            s0 = []
            w = list(range(16))
            for cid, mu in (("30B_t32", 0.030), ("30B_t40", 0.020)):
                f = e / f"{cid}_step0.json"
                f.write_text(json.dumps({"ok": True, "simulated": False, "windows": w,
                                         "c17": {"table": str(p["score_tables"]["c17"])},
                                         "s80": {"table": str(p["score_tables"]["c17_s80"])},
                                         "pair": {"window_dnll": list(rng.normal(mu, 0.004, 16))}}))
                s0 += ["--step0-measured", f"{cid}={f}"]
            kd = e / "kappa_decision.json"
            self.assertEqual(smain(["kappa-decision", "--manifest", str(man), "--out", str(kd)] + s0), 0)
            dec = json.loads(kd.read_text())
            self.assertTrue(dec["authoritative"])
            self.assertEqual(dec["decision"], "keep_0.42")                    # ~3.3 >> 0.83 + band
            self.assertEqual(dec["rule_id"], S.KAPPA_RULE["id"])
            rc = smain(["solve", "--manifest", str(man), "--cell", "4B_t40", "--out-dir",
                        str(e / "solve"), "--kappa-decision", str(kd)])
            self.assertEqual(rc, 0)
            sm = json.loads((e / "solve" / "solve_summary.json").read_text())
            self.assertEqual(sm["kappa_currency_ratio"], 0.83 / 1.96)
            self.assertEqual(sm["mde_nats"], self.PREREG["mde_nats"])
            # the eval side's explicit form (prc_fixedpoint_r7 resolve_command): values are CHECKED
            explicit = ["solve", "--manifest", str(man), "--cell", "4B_t40", "--kappa-decision", str(kd),
                        "--stem", str(p["stem"]), "--parent", str(PARENT_4B_T40),
                        "--inc-table", str(p["inc"]), "--arms", "UK,K,U", "--kappa-lin", "1.96",
                        "--kappa-att", "0.83", "--target-scales", "1.0043", "--ladder-policy", "up",
                        "--mde-nats", str(self.PREREG["mde_nats"])]
            self.assertEqual(smain(explicit + ["--out-dir", str(e / "solve2")]), 0)
            for i_, bad_val in ((explicit.index("--mde-nats") + 1, "0.001"),
                                (explicit.index("--ladder-policy") + 1, "full"),
                                (explicit.index("--kappa-lin") + 1, "0.83"),
                                (explicit.index("--target-scales") + 1, "1.0100")):
                bad_argv = list(explicit)
                bad_argv[i_] = bad_val
                with self.assertRaises((SystemExit, ValueError, FileNotFoundError)):
                    smain(bad_argv + ["--out-dir", str(e / f"solve_bad{i_}")])
            # a changed step-0 table breaks the decision (undefined -> revert + review)
            f = e / "30B_t40_step0.json"
            r = json.loads(f.read_text())
            r["s80"]["table"] = str(p["score_tables"]["c17"])
            f.write_text(json.dumps(r))
            kd2 = e / "kappa_decision2.json"
            smain(["kappa-decision", "--manifest", str(man), "--out", str(kd2)] + s0)
            dec2 = json.loads(kd2.read_text())
            self.assertEqual((dec2["ratio"], dec2["requires_user_review"]), (1.0, True))
            # a manifest whose rule differs from the code's is refused
            bad = e / "bad_manifest.json"
            bad.write_text(json.dumps({"preregistered_rules": {"kappa_rule": dict(S.KAPPA_RULE,
                                                                                  kappa_att=0.82)},
                                       "cells": cells}))
            with self.assertRaises(SystemExit):
                smain(["kappa-decision", "--manifest", str(bad)] + s0)


MANIFEST = REPO / "benchmark/ppl/kbands/prc_r7_capture_20260928.json"


@unittest.skipUnless(MANIFEST.is_file(), "round-7 capture manifest not built")
class TestDriverAndManifest(unittest.TestCase):
    def setUp(self):
        from benchmark.ppl import prc_r7_capture_20260928 as D
        self.D = D
        self.m = json.loads(MANIFEST.read_text())

    def test_cells_and_gpu_plan(self):
        self.assertEqual([c["id"] for c in self.m["cells"]],
                         ["30B_t32", "30B_t40", "30B_t48", "4B_t40", "4B_t64"])
        self.assertEqual([c["enabled"] for c in self.m["cells"]], [True, True, True, False, False])
        self.assertEqual(self.m["max_total_gpus"], 4)
        sub = self.m["gpu_plan"]["submit"]
        self.assertEqual(len(sub), 2)
        self.assertIn("R7C_MANIFEST_SHA256=<SHA> sbatch --parsable --array=0-1%2 benchmark", sub[0])
        self.assertIn("--array=2 --dependency=afterany:62208440", sub[1])
        self.assertIn("--array=0-2%1", self.m["gpu_plan"]["conservative_alternative"])
        self.assertEqual(self.m["gpu_plan"]["wall_time_request"], "12:00:00")
        sb = (REPO / "benchmark/ppl/kbands/run_prc_r7_capture_20260928.sbatch").read_text()
        self.assertIn("#SBATCH --array=0-2%1\n", sb)                  # a bare sbatch holds <= 1 GPU
        self.assertIn("#SBATCH --time=12:00:00\n", sb)
        self.assertIn("verify-pin", sb)

    def test_preregistration_is_frozen_in_the_manifest(self):
        rules = self.m["preregistered_rules"]
        self.assertEqual(rules["kappa_rule"], S.KAPPA_RULE)
        self.assertIs(rules["identity_allow_near"], False)
        for key in ("refusal", "mde", "target_scale", "one_full_test_per_cell", "non_candidates",
                    "arms", "ladder_policy", "fail_closed", "realized_length_audit", "kappa_att_A",
                    "disclosure", "4B"):
            self.assertIn(key, rules)
        # one source of truth with the eval side: its FROZEN registration record, hashed here
        rp = Path(self.m["shared_registration"]["record"])
        self.assertEqual(self.m["shared_registration"]["sha256"],
                         hashlib.sha256(rp.read_bytes()).hexdigest())
        self.assertEqual(self.m["frozen_manifests"][str(rp)]["sha256"],
                         self.m["shared_registration"]["sha256"])
        rec = json.loads(rp.read_text())
        self.assertEqual(rec["kappa"]["pins"], S.KAPPA_PINS)
        want = {c: (rec["mde"]["nats"][c], rec["target_scale"]["s0"][c])
                for c in ("30B_t32", "30B_t40", "30B_t48")}
        self.assertEqual(want, {"30B_t32": (0.0129, 1.0037), "30B_t40": (0.0092, 1.0043),
                                "30B_t48": (0.0091, 1.011)})
        for c in self.m["cells"]:
            pr = c["prereg"]
            self.assertIs(S.check_prereg(pr), pr)
            if c["model"] == "30B":
                self.assertEqual((pr["primary"], pr["arms"], pr["ladder_policy"]),
                                 ("UK", ["UK", "K", "U"], "up"))
                self.assertEqual((pr["mde_nats"], pr["s0"]), want[c["id"]])
                self.assertTrue(pr["family_kappa"])
                self.assertTrue(pr["round6_gate"]["passed"])
                self.assertEqual(pr["round6_gate"]["cell"], "30B_t40" if c["id"] == "30B_t48" else c["id"])
                self.assertEqual("r6_arm_A" in pr, c["id"] != "30B_t48")
            else:
                self.assertFalse(c["enabled"])
                self.assertFalse(pr["family_kappa"])
                self.assertIn("not_run_reason", pr)
        g40 = next(c for c in self.m["cells"] if c["id"] == "4B_t40")["prereg"]["round6_gate"]
        self.assertFalse(g40["passed"])

    def test_pin_and_verify(self):
        import os
        with tempfile.TemporaryDirectory() as d_:
            mp = Path(d_) / "prc_r7_capture_20260928.json"
            mp.write_bytes(MANIFEST.read_bytes())
            m = json.loads(mp.read_text())
            old = os.environ.pop("R7C_MANIFEST_SHA256", None)
            try:
                self.assertTrue(self.D.verify_pin(mp))                      # no pin, no env
                pin = self.D.write_pin(m, mp)
                rec = json.loads(pin.read_text())
                sha = self.D.sha256(mp)
                self.assertEqual(rec["manifest_sha256"], sha)
                self.assertTrue(all(sha in c for c in rec["submit"]))
                with self.assertRaises(FileExistsError):
                    self.D.write_pin(m, mp)
                os.environ["R7C_MANIFEST_SHA256"] = sha
                self.assertEqual(self.D.verify_pin(mp), [])
                mp.write_text(mp.read_text() + " ")                         # a rebuild after submit
                self.assertTrue(self.D.verify_pin(mp))
            finally:
                os.environ.pop("R7C_MANIFEST_SHA256", None)
                if old is not None:
                    os.environ["R7C_MANIFEST_SHA256"] = old

    def test_argv_is_incumbent_recipe_plus_r7(self):
        import argparse
        for c in self.m["cells"]:
            argv = self.D.calib_argv(c)
            ap = argparse.ArgumentParser()
            for opt in ("--parent", "--model_path", "--out", "--diag", "--ladder", "--calib-windows",
                        "--holdout-windows", "--rows-per-call", "--expert-calls-per-block", "--bins",
                        "--seed", "--frontend", "--currencies", "--att-rows-per-call", "--dump",
                        "--grad-mode", "--score-tables", "--target-scales", "--solve-state",
                        "--att-measure-extra", "--att-dump", "--att-prefix-split",
                        "--r7-hold-dump", "--prot-measure"):
                ap.add_argument(opt, default=None)
            ns = vars(ap.parse_args(argv))
            rec = json.loads(Path(c["refs"]["c7_gfisla_table"]).read_text())["prc6_calib"]
            for k in ("parent", "model_path", "ladder", "calib_windows", "holdout_windows",
                      "rows_per_call", "expert_calls_per_block", "bins", "seed", "frontend",
                      "currencies", "grad_mode"):
                self.assertEqual(ns[k], str(rec[k]), (c["id"], k))
            self.assertEqual(ns["dump"], "")
            self.assertEqual(ns["att_rows_per_call"], "512")
            self.assertEqual(ns["att_prefix_split"], str(rec["att_rows_per_call"]))
            self.assertEqual(ns["target_scales"], "1.0")
            st = dict(x.split("=", 1) for x in ns["score_tables"].split(","))
            self.assertEqual(sorted(st), ["c17", "c17_s80", "inc"])
            inc_c17 = rec["score_tables"].split("=", 1)[1]
            self.assertEqual(st["c17"], inc_c17, c["id"])          # the incumbent's own c17
            lad = c["parent_ladder"]
            meas = set(lad) | {int(v) for v in ns["att_measure_extra"].split(",")} | {128}
            self.assertTrue({112, 128} <= meas or {128} <= meas)
            self.assertEqual(S.ladder_for("up", lad, meas)[-1], 128)
            for p in (ns["out"], ns["diag"], ns["solve_state"], ns["att_dump"], ns["r7_hold_dump"]):
                self.assertTrue(p.startswith(c["paths"]["capture_dir"] + "/"))

    def test_preflight(self):
        c = self.m["cells"][1]
        self.assertEqual(self.D.preflight(self.m, c, require_fresh=False), [])
        bad = self.D.preflight(self.m, self.m["cells"][3], require_fresh=False)
        self.assertTrue(any("disabled" in b for b in bad))
        m2 = copy.deepcopy(self.m)
        first = next(iter(m2["code_hashes"]))
        m2["code_hashes"][first] = "0" * 64
        self.assertTrue(any("source changed" in b for b in self.D.preflight(m2, c, require_fresh=False)))
        m3 = copy.deepcopy(self.m)
        m3["cells"][1]["r7"]["att_measure_extra"] = [129]
        self.assertTrue(self.D.preflight(m3, m3["cells"][1], require_fresh=False))
        for mut in (lambda m: m["preregistered_rules"]["kappa_rule"].__setitem__("kappa_att", 0.82),
                    lambda m: m["preregistered_rules"].__setitem__("identity_allow_near", True),
                    lambda m: m["cells"][1]["prereg"].__setitem__("ladder_policy", "full"),
                    lambda m: m["cells"][1]["prereg"].__setitem__("mde_nats", None),
                    lambda m: m["cells"][1]["prereg"].__setitem__("family_kappa", False)):
            m4 = copy.deepcopy(self.m)
            mut(m4)
            self.assertTrue(any("pre-registration" in b or "kappa_rule" in b or "allow_near" in b
                                or "family_kappa" in b
                                for b in self.D.preflight(m4, m4["cells"][1], require_fresh=False)))


@unittest.skipUnless((PARENT_30B_T40 / "table.json").is_file(), "30B t40 parent config missing")
class TestReal30BTables(unittest.TestCase):
    def test_incumbents_are_lineage_clean_and_sweep(self):
        parent = json.loads((PARENT_30B_T40 / "table.json").read_text())
        for name in ("30B_t40_c7_gfisla_table.json", "30B_t40_c7_gfis_table.json"):
            p = PRC2 / name
            if not p.is_file():
                self.skipTest("prc2 tables missing")
            t = json.loads(p.read_text())
            self.assertEqual(S.validate_lineage(parent, t, allow_bucket_ladders=False,
                                                prc_ladder=DENSE_LADDER), [], name)
        cand07 = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_local_20260926/30B_t40/"
                      "candidate07_mlp_from_projections_table.json")
        if cand07.is_file():
            self.assertEqual(S.validate_lineage(parent, json.loads(cand07.read_text()),
                                                allow_bucket_ladders=False), [])

    def test_extended_30B_table_sweep(self):
        parent = json.loads((PARENT_30B_T40 / "table.json").read_text())
        wrapper = json.loads((PARENT_30B_T40 / "wrapper.json").read_text())
        glob = sorted(int(v) for v in parent["stoc_len_levels"])
        rng = np.random.default_rng(9)
        A = {(op, b): sorted(set(glob) | {112, 128}) for op in S.ATT_OPS for b in range(4)}
        att = {k: sorted((float(np.float32(x)) for x in rng.uniform(0, 1, len(v) - 1)), reverse=True)
               for k, v in A.items()}
        lin = {(op, b): sorted(float(np.float32(x)) for x in rng.uniform(0, 1, 16))
               for op in S.LINEAR_OPS for b in range(4)}
        with tempfile.TemporaryDirectory() as d_:
            o, t = S.emit_r7_table(Path(d_) / "c.json", "r7UK_k0420_s10043", lin, att, A, parent,
                                   wrapper, DENSE_LADDER, {}, {S.R7_KEY: {}})
            cand = json.loads(t.read_text())
            self.assertEqual(S.validate_lineage(parent, cand, prc_ladder=DENSE_LADDER), [])
            rep = S.resolver_sweep(o, t, PARENT_30B_T40, 48)
            self.assertEqual(rep["attention_op_blocks"], 96)
            self.assertEqual(rep["max_attention_length"], 128)


if __name__ == "__main__":
    unittest.main()
