"""Round-6 BUDGET RE-TARGET driver (2026-09-28): 14B t32 c7gfisla, 30B t64 c7gfis.

Why: both incumbents were solved to the parent's cost on the calibration pairs but deploy
on their own trajectory 1.34% / 0.64% under the parent's test cost (investigation RC6), and
sit 0.07% / 0.05% above <= 1.05x fp16. Pure allocation fix: re-solve the SAME calib7 recipe
(same seed, windows, currency, ladder, rule; calib9_r6 = calib7 + --target-scales) at the
parent's cost as measured on held-out TRAIN windows, then run one full-protocol test.

Stages (each writes JSON into the cell dir; the launcher chains them):
  preflight       CPU. Code + frozen-input hashes, fresh output paths. No writes.
  stage-a         GPU. Parent and incumbent on the SAME 16 held-out TRAIN windows the
                  incumbent's heldout_nll.json used (seed 101, calib starts excluded), with
                  exact SC traces. Identity: every window NLL must equal the stored one bit for
                  bit. Then the pre-registered one-pass fixed point
                      s = (P - U) / (C1 - U),  rounded to 1e-4,
                  P = parent trace cost, C1 = incumbent trace cost, U = the incumbent's
                  UNCONTROLLED cycles per MAC (protected slice; + qk/av for a gfis arm). This
                  is the held-out-train child-cost ratio referred to the population the solve
                  controls (derivation: C(s) = U + s (C1 - U) = P).
  calib-argv      CPU. Prints the calib9 argv = the incumbent's recorded prc6_calib args with
                  new out/diag + --target-scales 1.0,<s> --solve-state.
  calib-identity  CPU. calib9's s=1.0 table must equal the incumbent table in every field but
                  provenance; the s table may differ from the incumbent only in allocation
                  thresholds; CPU re-solve from the solve state must reproduce both tables.
  stage-b         GPU. Incumbent window-0 identity in this process, then the s table on the
                  16 held-out windows with a trace. GATE: |C(s)/P - 1| <= 0.3% (P from
                  stage-a); sanity abort if its mean held-out NLL is > +0.02 nats worse than the
                  incumbent's (bug guard, ~+2% PPL, far outside any budget effect). No
                  NLL-based selection. Writes selected.json.
  finalize        CPU. Verify the full-test trace protocol + realized-length audit, write
                  full_test_result.json (a best(all) candidate, disclosed as a budget correction,
                  whatever its sign).
Units: every stream length / cost here is HALVED (nominal = 2x); max 128.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LINEAR_OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATT_OPS = ("qk", "av")
PROVENANCE_KEYS = ("prc6_calib", "prc9_r6")
SCHEMA = "prc-r6-retarget-v1"


# ----------------------------------------------------------------------------- utilities
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    payload = json.dumps(value, indent=1, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as f:
            f.write(payload)
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload)
        tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_manifest(path, index):
    m = read_json(path)
    if m.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema {m.get('schema')!r} != {SCHEMA}")
    if not 0 <= index < len(m["cells"]):
        raise ValueError(f"cell index {index} outside manifest")
    cell = m["cells"][index]
    if cell["index"] != index:
        raise ValueError("manifest cell index mismatch")
    return m, cell


def preflight(manifest, cell, *, require_fresh=True, allow_unfrozen=False):
    """Hashes of code and frozen inputs; incumbent wrapper->table link; fresh outputs."""
    code = manifest.get("code_hashes") or {}
    if not code:
        if not allow_unfrozen:
            raise ValueError("manifest has no code_hashes (not frozen)")
    else:
        if set(code) != set(manifest["source_files"]):
            raise ValueError("incomplete source freeze")
        for p, h in code.items():
            if sha256(p) != h:
                raise ValueError(f"experiment source changed: {p}")
    for p, h in cell["hashes"].items():
        if sha256(p) != h:
            raise ValueError(f"frozen input changed: {p}")
    w = read_json(cell["incumbent_wrapper"])
    if w["threshold_table_path"] != cell["incumbent_table"]:
        raise ValueError("incumbent wrapper does not point at the frozen incumbent table")
    t = read_json(cell["incumbent_table"])
    if not (t.get("per_row_chunk") or {}).get("buckets"):
        raise ValueError("incumbent has no per_row_chunk buckets")
    if t["model_path"] != cell["model_path"]:
        raise ValueError("incumbent model mismatch")
    pt = read_json(Path(cell["parent"]) / "table.json")
    if pt["model_path"] != cell["model_path"]:
        raise ValueError("parent model mismatch")
    rec = t.get("prc6_calib") or {}
    for k, v in cell["calib_args"].items():
        if rec.get(k) != v:
            raise ValueError(f"incumbent prc6_calib[{k}]={rec.get(k)!r} != manifest {v!r}")
    if require_fresh:
        if Path(cell["cell_dir"]).exists():
            raise ValueError(f"refusing to reuse cell dir {cell['cell_dir']}")
        if Path(cell["test_trace"]).exists():
            raise ValueError(f"refusing to overwrite test trace {cell['test_trace']}")
    return w, t


def configure_environment(manifest, cell):
    for key in ("SC_RNG_GRID", "SC_RNG_GRID_ATTN", "SC_RNG_GRID_QK", "SC_RNG_GRID_AV",
                "SC_ATTN_SMOOTH_JSON", "SC_HYBRID_FORCE_INT_BITS", "SC_MP_TRACE"):
        if os.environ.get(key):
            raise ValueError(f"forbidden inherited variant: {key}={os.environ[key]!r}")
    if os.environ.get("SC_PRC_ROWSHARED", "0") != "0":
        raise ValueError("SC_PRC_ROWSHARED ablation is forbidden")
    if os.environ.get("SC_HW_MAX_MASKS", "64") not in ("", "64"):
        raise ValueError("SC_HW_MAX_MASKS != 64 is simulation-only")
    for key, value in (("SC_OWEN_MODE", "bitrev"), ("SC_SCRAMBLE_MASKS", "64")):
        if os.environ.get(key, value) != value:
            raise ValueError(f"wrong protocol: {key}")
        os.environ[key] = value
    act = str(REPO.parent / "hpca_results/llm/ppl/mp_best/act_scales")
    os.environ.update(QUANT_CONFIG="mp", MP_CONFIG_JSON=str(Path(cell["parent"]) / "wrapper.json"),
                      SC_HYBRID_CONFIG_JSON=cell["hybrid_config"], FRONTEND="awq", CTX="2048",
                      STRIDE="2048", PPL_WINDOW_BATCH_SIZE="1", PPL_MAX_TOKENS="0",
                      SQ_ALPHA="0.5", ACT_SCALES_DIR=act,
                      AWQ_SCALES_DIR=str(Path(act) / "awq_scales"), AWQ_OBJ_BITS="4")
    if not Path(cell["awq_cache"]).is_file():
        raise ValueError(f"frozen AWQ cache missing: {cell['awq_cache']}")


# ----------------------------------------------------------------------------- MP config
def mp_from_wrapper(wrapper_path):
    """AdaptiveMPConfig exactly as loader.apply_mp_config_from_env builds it (CPU only)."""
    from scmp_kernels.mp import AdaptiveMPConfig
    spec = read_json(wrapper_path)
    if spec.get("type") != "AdaptiveMPConfig":
        raise ValueError(f"{wrapper_path}: not an AdaptiveMPConfig wrapper")
    tp = spec["threshold_table_path"]
    if not os.path.isabs(tp):
        tp = os.path.join(os.path.dirname(os.path.abspath(wrapper_path)), tp)
    esc_k = spec.get("escape_gate_k")
    return AdaptiveMPConfig(stoc_len_levels=[int(x) for x in spec["stoc_len_levels"]],
                            threshold_table_path=tp,
                            escape_gate_k=None if esc_k is None else float(esc_k),
                            escape_stoc_len=int(spec.get("escape_stoc_len", 128)))


def protected_widths(table_payload):
    out = {}
    pc = table_payload.get("protected_channels") or {}
    for key, idx in (pc.get("indices") or {}).items():
        parts = key.split(":")
        op, blk = parts[0], int(parts[1][1:])
        unit = int(parts[2][1:]) if len(parts) == 3 else None
        out[(op, blk, unit)] = len(idx)
    psl = int(pc["stoc_len"]) if pc.get("indices") else None
    return out, psl


# ----------------------------------------------------------------------------- trace analysis
def analyze_trace(trace_payload, wrapper_path, *, total_blocks=None, arm="gfisla"):
    """Exact SC cost + uncontrolled share + realized-length audit (pitfall 1: every realized
    stream length must be a rung the runtime resolver (get_levels / classify_level_values /
    get_per_row_chunk) assigns to that (op, bucket), or the protected-slice length)."""
    from scmp_kernels.mp.config import _bucket_index
    cfg = mp_from_wrapper(wrapper_path)
    table = read_json(cfg.threshold_table_path)
    widths, psl = protected_widths(table)
    hdr = trace_payload.get("header", {})
    nb = int(total_blocks or hdr.get("total_blocks") or 0)
    if nb <= 0:
        raise ValueError("trace analysis needs total_blocks")
    groups = trace_payload["groups"]
    if not groups:
        raise ValueError("empty trace")
    tot_m = tot_c = 0.0
    cyc = defaultdict(float)
    mac = defaultdict(float)
    hist = defaultdict(lambda: defaultdict(float))
    violations, numerics, ambiguous = [], [], []
    n_lb = int(table.get("layer_buckets", 4) or 4)
    for g in groups:
        op, blk, L, m = g["op"], g["block"], int(g["stoc_len"]), float(g["macs"])
        if not (g.get("rng_levels") == 128 and g.get("sc_prec") == 8 and g.get("halve") is True):
            numerics.append({k: g.get(k) for k in ("op", "block", "stoc_len", "rng_levels",
                                                    "sc_prec", "halve")})
        if not (1 <= L <= 128):
            violations.append({"op": op, "block": blk, "stoc_len": L, "why": "outside [1,128]"})
        tot_m += m
        tot_c += m * L
        if op in ATT_OPS:
            cls = op
            allowed = set(int(v) for v in cfg.classify_level_values(
                operator=op, block_idx=blk, total_blocks=nb))
        elif op in LINEAR_OPS:
            w = widths.get((op, blk, g.get("unit"))) or widths.get((op, blk, None))
            is_prot = (w is not None and psl is not None and int(g["d_in"]) == w and L == psl)
            if w is not None and int(g["d_in"]) == w and L != psl:
                ambiguous.append({"op": op, "block": blk, "unit": g.get("unit"), "d_in": w,
                                  "stoc_len": L})
            cls = "prot" if is_prot else "lin"
            prc = cfg.get_per_row_chunk(op, blk, nb)
            if prc is not None:
                allowed = set(int(v) for v in prc[0])
            else:
                allowed = set(int(v) for v in cfg.classify_level_values(
                    operator=op, block_idx=blk, total_blocks=nb))
            if psl is not None:
                allowed.add(psl)
        else:
            cls = "other"
            allowed = set()
        if L not in allowed:
            violations.append({"op": op, "block": blk, "unit": g.get("unit"), "stoc_len": L,
                               "allowed": sorted(allowed)})
        cyc[cls] += m * L
        mac[cls] += m
        hist[f"{op}:l{_bucket_index(blk, nb, n_lb)}"][str(L)] += m
    if tot_m <= 0:
        raise ValueError("trace has no MACs")
    uncontrolled = ("prot",) if arm == "gfisla" else ("prot",) + ATT_OPS
    U = sum(cyc[c] for c in uncontrolled) / tot_m
    return {"cost": tot_c / tot_m, "total_macs": tot_m, "total_cycle_macs": tot_c,
            "U": U, "U_prot": cyc["prot"] / tot_m, "uncontrolled_classes": list(uncontrolled),
            "cycle_share": {c: cyc[c] / tot_c for c in sorted(cyc)},
            "mac_share": {c: mac[c] / tot_m for c in sorted(mac)},
            "mean_L": {c: cyc[c] / mac[c] for c in sorted(mac) if mac[c] > 0},
            "protected_stoc_len": psl, "audit_ok": not violations and not numerics
            and not mac.get("other"),
            "violations": violations[:50], "n_violations": len(violations),
            "numerics_violations": numerics[:20], "protected_ambiguous": ambiguous[:20],
            "histogram_macs": {k: dict(sorted(v.items(), key=lambda kv: -int(kv[0])))
                               for k, v in sorted(hist.items())}}


def retarget_scale(P, C1, U, digits=4):
    """s such that U + s (C1 - U) = P (one fixed-point pass), rounded."""
    if not (P > U and C1 > U):
        raise ValueError(f"degenerate costs P={P} C1={C1} U={U}")
    raw = (P - U) / (C1 - U)
    return raw, round(raw, digits)


def paired_stats(cand, ref):
    if len(cand) != len(ref) or len(cand) < 2:
        raise ValueError("paired losses need equal counts >= 2")
    d = [a - b for a, b in zip(cand, ref)]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    return {"mean_dnll": m, "se": se, "z": (m / se) if se > 0 else None,
            "dppl_pct": math.expm1(m) * 100}


# ----------------------------------------------------------------------------- table checks
def strip_provenance(t):
    t = copy.deepcopy(t)
    for k in PROVENANCE_KEYS:
        t.pop(k, None)
    return t


def allocation_only_diff(base, cand, *, allow_attention):
    """Fields that differ between two tables beyond per_row_chunk thresholds and (if allowed)
    qk/av bucket thresholds. Empty list = pure allocation change."""
    a, b = strip_provenance(base), strip_provenance(cand)
    bad = []
    pa, pb = (a.get("per_row_chunk") or {}).get("buckets", {}), (b.get("per_row_chunk") or {}).get("buckets", {})
    if set(pa) != set(pb):
        bad.append("per_row_chunk bucket keys differ")
    for k in set(pa) & set(pb):
        if pa[k].get("levels") != pb[k].get("levels"):
            bad.append(f"per_row_chunk {k} levels differ")
        tb = pb[k].get("thresholds")
        if len(tb) != len(pa[k].get("thresholds")) or list(tb) != sorted(tb):
            bad.append(f"per_row_chunk {k} thresholds malformed")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in tb):
            bad.append(f"per_row_chunk {k} thresholds out of [0,1]")
    na, nb_ = copy.deepcopy(a), copy.deepcopy(b)
    for t in (na, nb_):
        for k, v in ((t.get("per_row_chunk") or {}).get("buckets") or {}).items():
            v["thresholds"] = None
    for k, v in (nb_.get("buckets") or {}).items():
        if k.split(":")[0] in ATT_OPS:
            ta, tb = (na["buckets"].get(k) or {}).get("thresholds"), v.get("thresholds")
            if ta is None or tb is None or len(ta) != len(tb):
                bad.append(f"attention {k} threshold shape changed")
                continue
            if list(tb) != sorted(tb, reverse=True):
                bad.append(f"attention {k} thresholds not descending")
            if ta != tb and not allow_attention:
                bad.append(f"attention {k} thresholds changed on a linear-only arm")
            v["thresholds"] = None
            na["buckets"][k]["thresholds"] = None
    if na != nb_:
        diff = sorted(k for k in set(na) | set(nb_) if na.get(k) != nb_.get(k))
        bad.append(f"non-allocation fields differ: {diff[:10]}")
    return bad


# ----------------------------------------------------------------------------- GPU stages
class HeldoutRunner:
    """heldout_nll.py's exact procedure (model built on the parent wrapper, tables swapped via
    loader.apply_mp_config_from_env, loss = model(ids, labels=ids).loss) + exact traces."""

    def __init__(self, manifest, cell):
        import torch
        configure_environment(manifest, cell)
        from benchmark.quant.eval_quant import build_sc_model
        from benchmark.ppl.calibrate_mp_thresholds import _select_int_swap_windows
        from datasets import load_dataset
        self.torch = torch
        self.cell = cell
        parent_wrapper = str(Path(cell["parent"]) / "wrapper.json")
        self.model, tok = build_sc_model(cell["model_path"], "mp", mp_table=parent_wrapper)
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        if self.dev.type != "cuda":
            raise ValueError("GPU stage must run on its Slurm GPU")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        H = manifest["protocol"]["heldout"]
        calib_set = set()
        for n_ in H["calib_n"]:
            _, cs = _select_int_swap_windows(enc, 2048, n_, sampling="stratified",
                                             seed=H["calib_seed"])
            calib_set |= set(int(s) for s in cs)
        wins, starts = _select_int_swap_windows(enc, 2048, H["windows"] + len(calib_set),
                                                sampling="stratified", seed=H["seed"])
        self.keep = [(w, s) for w, s in zip(wins, starts) if int(s) not in calib_set][:H["windows"]]
        self.starts = [int(s) for _, s in self.keep]
        stored = read_json(cell["incumbent_heldout"])["windows"]
        if self.starts != stored:
            raise ValueError(f"held-out windows {self.starts} != stored {stored}")
        self.total_blocks = int(self.model.config.num_hidden_layers)

    def evaluate(self, name, wrapper, out_dir, *, n_windows=None, trace_on=True):
        from loader import apply_mp_config_from_env
        from model.sc_common import SCLinear, mp_tracker_reset, mp_tracker_flop_avg_stoc_len
        from scmp_kernels import trace
        torch = self.torch
        os.environ["MP_CONFIG_JSON"] = str(wrapper)
        self.model.config.sc_mp_config = None
        apply_mp_config_from_env(self.model)
        mp = self.model.config.sc_mp_config
        mods = [m for m in self.model.modules() if isinstance(m, SCLinear)]
        if mp is None or not mods or any(m._sc_config.sc_mp_config is not mp for m in mods):
            raise ValueError(f"table swap did not reach every SCLinear: {wrapper}")
        keep = self.keep[:n_windows] if n_windows else self.keep
        tpath = Path(out_dir) / f"{name}_heldout_trace.json"
        if trace_on:
            trace.reset()
            trace.enable(str(tpath), mode="summary")
        mp_tracker_reset()
        t0 = time.time()
        losses = []
        with torch.no_grad():
            for i, (w, _s) in enumerate(keep):
                ids = w.unsqueeze(0).to(self.dev)
                losses.append(float(self.model(input_ids=ids, labels=ids).loss))
                if not math.isfinite(losses[-1]):
                    raise ValueError("nonfinite held-out NLL")
                print(f"[r6] {name} window {i + 1}/{len(keep)} NLL={losses[-1]:.7f}", flush=True)
        tracker = float(mp_tracker_flop_avg_stoc_len())
        res = {"name": name, "wrapper": str(wrapper), "window_nll": losses,
               "mean_nll": sum(losses) / len(losses), "tracker_cost": tracker,
               "windows": [int(s) for _, s in keep], "seconds": time.time() - t0}
        if trace_on:
            trace.flush(header_extra={"stage": name, "split": "train", "windows": res["windows"],
                                      "mp_config_json": str(wrapper),
                                      "total_blocks": self.total_blocks,
                                      "model": self.cell["model_path"]}, reset_after=True)
            trace.disable()
            res["trace"] = str(tpath)
        return res


def stage_a(manifest, cell, out):
    preflight(manifest, cell, require_fresh=False)
    stored = read_json(cell["incumbent_heldout"])["tables"]
    R = HeldoutRunner(manifest, cell)
    res = {"stage": "a", "cell": cell["id"], "windows": R.starts, "tables": {}}
    checks = {}
    for name, wrapper, sname in (("parent", str(Path(cell["parent"]) / "wrapper.json"), "parent"),
                                 ("incumbent", cell["incumbent_wrapper"], cell["incumbent_heldout_name"])):
        r = R.evaluate(name, wrapper, out)
        a = analyze_trace(read_json(r["trace"]), wrapper, total_blocks=R.total_blocks,
                          arm=cell["arm"])
        r["analysis"] = {k: v for k, v in a.items() if k != "histogram_macs"}
        write_json(Path(out) / f"{name}_heldout_histogram.json", a["histogram_macs"])
        r["identity_window_nll_exact"] = (r["window_nll"] == stored[sname]["window_nll"])
        r["stored_tracker_cost"] = stored[sname]["cost"]
        r["tracker_cost_equal_stored"] = (r["tracker_cost"] == stored[sname]["cost"])
        checks[f"{name}_identity"] = r["identity_window_nll_exact"]
        checks[f"{name}_audit"] = a["audit_ok"]
        res["tables"][name] = r
        write_json(Path(out) / "stage_a.json", res)
    P = res["tables"]["parent"]["analysis"]["cost"]
    C1 = res["tables"]["incumbent"]["analysis"]["cost"]
    U = res["tables"]["incumbent"]["analysis"]["U"]
    s_raw, s = retarget_scale(P, C1, U)
    lo, hi = manifest["protocol"]["scale_band"]
    checks["scale_in_band"] = (lo < s <= hi)
    up, ui = (res["tables"]["parent"]["analysis"]["U_prot"],
              res["tables"]["incumbent"]["analysis"]["U_prot"])
    checks["protected_classification_consistent"] = (ui > 0 and abs(up / ui - 1) <= 0.05)
    res.update(P_parent_trace_cost=P, C1_incumbent_trace_cost=C1, U_incumbent=U,
               U_parent=res["tables"]["parent"]["analysis"]["U"], r_trace=C1 / P,
               f_controlled_parent=1 - res["tables"]["parent"]["analysis"]["U"] / P,
               s_naive_inverse_ratio=P / C1, s_raw=s_raw, target_scale=s,
               predicted_cost_ratio_at_s=(U + s * (C1 - U)) / P, checks=checks,
               ok=all(checks.values()))
    write_json(Path(out) / "stage_a.json", res)
    print(f"[r6] stage-a {cell['id']}: P {P:.5f} C1 {C1:.5f} (r {C1 / P:.6f}) U {U:.4f} "
          f"-> s {s_raw:.6f} ~ {s:.4f}; checks {checks}", flush=True)
    return 0 if res["ok"] else 5


def stage_b(manifest, cell, out):
    preflight(manifest, cell, require_fresh=False)
    A = read_json(Path(out) / "stage_a.json")
    I = read_json(Path(out) / "calib_identity.json")
    if not (A.get("ok") and I.get("ok")):
        raise ValueError("stage-b requires passing stage-a and calib-identity")
    stored = read_json(cell["incumbent_heldout"])["tables"]
    wrapper = I["scaled_wrapper"]
    R = HeldoutRunner(manifest, cell)
    res = {"stage": "b", "cell": cell["id"], "candidate_wrapper": wrapper,
           "target_scale": A["target_scale"]}
    idw = R.evaluate("identity_incumbent_w0", cell["incumbent_wrapper"], out, n_windows=1,
                     trace_on=False)
    res["identity_incumbent_w0"] = idw
    ident = idw["window_nll"][0] == stored[cell["incumbent_heldout_name"]]["window_nll"][0]
    r = R.evaluate("candidate", wrapper, out)
    a = analyze_trace(read_json(r["trace"]), wrapper, total_blocks=R.total_blocks, arm=cell["arm"])
    r["analysis"] = {k: v for k, v in a.items() if k != "histogram_macs"}
    write_json(Path(out) / "candidate_heldout_histogram.json", a["histogram_macs"])
    res["candidate"] = r
    P = A["P_parent_trace_cost"]
    ratio = a["cost"] / P
    inc_nll = stored[cell["incumbent_heldout_name"]]["window_nll"]
    par_nll = stored["parent"]["window_nll"]
    res["paired_vs_incumbent"] = paired_stats(r["window_nll"], inc_nll)
    res["paired_vs_parent"] = paired_stats(r["window_nll"], par_nll)
    tol = manifest["protocol"]["cost_gate_tolerance"]
    sanity = manifest["protocol"]["sanity_abort_dnll"]
    checks = {"incumbent_identity_this_process": ident, "candidate_audit": a["audit_ok"],
              "cost_gate": abs(ratio - 1) <= tol + 1e-12,
              "sanity_not_catastrophic": res["paired_vs_incumbent"]["mean_dnll"] <= sanity}
    res.update(cost_ratio_vs_parent=ratio, cost_ratio_vs_incumbent=a["cost"] / A["C1_incumbent_trace_cost"],
               predicted_cost_ratio=A["predicted_cost_ratio_at_s"], checks=checks,
               ok=all(checks.values()))
    write_json(Path(out) / "stage_b.json", res)
    sel = {"evaluate": res["ok"], "wrapper": wrapper if res["ok"] else None,
           "reason": ("held-out-train cost within +-%.2f%% of the parent; full test follows"
                      % (100 * tol)) if res["ok"] else
           "stopped before test: " + ", ".join(k for k, v in checks.items() if not v),
           "cost_ratio_vs_parent": ratio, "tolerance": tol,
           "reporting_rule": manifest["preregistered_rules"]["best_all_entry"]}
    write_json(Path(out) / "selected.json", sel, exclusive=True)
    print(f"[r6] stage-b {cell['id']}: cost ratio {ratio:.6f} (tol {tol}); held-out dNLL vs "
          f"incumbent {res['paired_vs_incumbent']['mean_dnll']:+.5f} "
          f"(z {res['paired_vs_incumbent']['z']}); checks {checks} -> evaluate={sel['evaluate']}",
          flush=True)
    return 0


# ----------------------------------------------------------------------------- CPU stages
def calib_argv(manifest, cell, out, scale):
    """calib9 argv: the incumbent's recorded calib7 arguments, new out/diag, + r6 flags."""
    a = dict(cell["calib_args"])
    stem = Path(out) / f"{cell['id']}_r6.json"
    argv = []
    order = ("parent", "model_path", "ladder", "calib_windows", "holdout_windows",
             "rows_per_call", "expert_calls_per_block", "bins", "seed", "frontend", "currencies",
             "att_rows_per_call", "dump", "grad_mode", "score_tables")
    if set(order) != set(a):
        raise ValueError(f"calib_args keys {sorted(a)} != expected {sorted(order)}")
    argv += ["--out", str(stem), "--diag", str(Path(out) / f"{cell['id']}_r6_diag.json")]
    for k in order:
        v = a[k]
        if k == "dump":
            if v != "":
                raise ValueError("the incumbent recipe had no dump")
            continue
        argv += [f"--{k.replace('_', '-') if k not in ('model_path',) else k}", str(v)]
    s = float(scale)
    if not (s > 1.0):
        raise ValueError(f"target scale {s} must exceed 1.0 for a re-target run")
    argv += ["--target-scales", f"1.0,{s:.4f}",
             "--solve-state", str(Path(out) / f"{cell['id']}_r6_state.npz")]
    return argv


def calib_identity(manifest, cell, out):
    from benchmark.ppl.mp_per_row_chunk_calib9_r6 import (load_solve_state, resolve_tables,
                                                          scale_suffix)
    from benchmark.ppl.prc_resolve_r6 import compare_thresholds
    A = read_json(Path(out) / "stage_a.json")
    s = float(A["target_scale"])
    arm = cell["arm"]
    stem = Path(out) / f"{cell['id']}_r6.json"
    w1 = stem.with_name(f"{stem.stem}_{arm}.json")
    ws = stem.with_name(f"{stem.stem}_{arm}{scale_suffix(s)}.json")
    t1 = read_json(read_json(w1)["threshold_table_path"])
    ts = read_json(read_json(ws)["threshold_table_path"])
    inc = read_json(cell["incumbent_table"])
    res = {"stage": "calib-identity", "cell": cell["id"], "arm": arm, "target_scale": s,
           "s1_wrapper": str(w1), "scaled_wrapper": str(ws)}
    checks = {}
    # (1) scale 1.0 reproduces the incumbent exactly (all non-provenance fields)
    checks["s1_equals_incumbent"] = strip_provenance(t1) == strip_provenance(inc)
    p1, pi = dict(t1.get("prc6_calib") or {}), dict(inc.get("prc6_calib") or {})
    for k in ("out", "diag"):
        p1.pop(k, None)
        pi.pop(k, None)
    checks["s1_calib_args_equal_incumbent"] = p1 == pi
    w1p, wip = read_json(w1), read_json(cell["incumbent_wrapper"])
    checks["s1_wrapper_equal_modulo_path"] = (
        {k: v for k, v in w1p.items() if k != "threshold_table_path"}
        == {k: v for k, v in wip.items() if k != "threshold_table_path"})
    # (2) the scaled table is a pure allocation change of the incumbent
    bad = allocation_only_diff(inc, ts, allow_attention=(arm == "gfisla"))
    res["scaled_allocation_violations"] = bad
    checks["scaled_allocation_only"] = not bad
    wsp = read_json(ws)
    checks["scaled_wrapper_equal_modulo_path"] = (
        {k: v for k, v in wsp.items() if k != "threshold_table_path"}
        == {k: v for k, v in wip.items() if k != "threshold_table_path"})
    rec = (ts.get("prc9_r6") or {}).get("solve") or {}
    checks["scaled_record_scale"] = rec.get("target_scale") == s
    res["scaled_solve_record"] = rec
    # (3) the solve state reproduces both tables on CPU
    meta, keys, akeys, lbins, abins = load_solve_state(Path(out) / f"{cell['id']}_r6_state.npz")
    rs = resolve_tables(meta, keys, akeys, lbins, abins, [1.0, s], [arm])
    mism = []
    for name, payload in ((arm, t1), (f"{arm}{scale_suffix(s)}", ts)):
        lin, att, _ = rs[name]
        mism += compare_thresholds(name, lin, att, payload)
    res["state_roundtrip_mismatches"] = mism[:20]
    checks["state_roundtrip_exact"] = not mism
    # (4) the scaled table actually spends more on the calibration sample
    checks["scaled_calib_cost_above_s1"] = (
        rec.get("calib_cost", 0) > ((t1.get("prc9_r6") or {}).get("solve") or {}).get("calib_cost", float("inf")))
    # diag agreement at 1.0 (reported, not binding)
    try:
        d9 = read_json(Path(out) / f"{cell['id']}_r6_diag.json")
        d7 = read_json(cell["incumbent_diag"])
        res["diag_joint_equal"] = {k: d9.get("joint", {}).get(k) == d7.get("joint", {}).get(k)
                                   for k in ("parent", "gfis", "gfisla", "att_th")}
        res["diag_tables_equal"] = {k: d9["tables"].get(k) == v for k, v in d7["tables"].items()}
        res["diag_att_replay"] = [d9.get("att_replay_agreement"), d7.get("att_replay_agreement")]
    except Exception as exc:  # reported only
        res["diag_compare_error"] = repr(exc)
    res["checks"] = checks
    res["ok"] = all(checks.values())
    write_json(Path(out) / "calib_identity.json", res, exclusive=True)
    print(f"[r6] calib-identity {cell['id']}: {checks}", flush=True)
    return 0 if res["ok"] else 6


def finalize(manifest, cell, out, trace_path):
    sel = read_json(Path(out) / "selected.json")
    if not sel.get("evaluate"):
        raise ValueError("finalize called without a passing selection")
    B = read_json(Path(out) / "stage_b.json")
    A = read_json(Path(out) / "stage_a.json")
    tr = read_json(trace_path)
    h = tr["header"]
    failed = []
    exp_tok = 288627 if cell["model"] == "llama8B" else 298862
    for cond, msg in ((h.get("eval_tokens") == exp_tok, "eval_tokens"),
                      (h.get("ctx") == 2048 and h.get("stride") == 2048, "ctx/stride"),
                      (h.get("ppl_max_tokens") == 0, "ppl_max_tokens"),
                      (h.get("ppl_window_batch_size") == 1, "ppl_window_batch_size"),
                      (h.get("owen_mode") == "bitrev", "owen_mode"),
                      (str(h.get("scramble_masks")) == "64", "scramble_masks"),
                      (h.get("sc_prec") == 8 and h.get("sc_halve") is True, "sc_prec/halve"),
                      (h.get("model") == cell["model_path"], "model"),
                      (h.get("mp_config_json") == sel["wrapper"], "mp_config_json"),
                      (isinstance(h.get("ppl"), (int, float)) and math.isfinite(h["ppl"])
                       and h["ppl"] > 0, "ppl")):
        if not cond:
            failed.append(msg)
    a = analyze_trace(tr, sel["wrapper"], arm=cell["arm"])
    if not a["audit_ok"]:
        failed.append("realized-length/numerics audit")
    write_json(Path(out) / "test_histogram.json", a["histogram_macs"])
    ppl = float(h["ppl"])
    fp16 = cell["fp16_ppl"]
    thr = cell["flip_thresholds"]
    copy_to = Path(out) / "test_trace.json"
    shutil.copyfile(trace_path, copy_to)
    wrapper = Path(sel["wrapper"])
    table = Path(read_json(wrapper)["threshold_table_path"])
    arm_name = f"r6ts_{cell['incumbent_arm']}_s{int(round(A['target_scale'] * 1e4)):05d}"
    rec = {
        "arm": arm_name, "ppl": ppl, "cost": a["cost"], "protocol_ok": not failed,
        "failed_checks": failed, "model_header": h.get("model"), "wrapper": str(wrapper),
        "trace": str(trace_path), "trace_copy": str(copy_to),
        "trace_sha256": sha256(trace_path),
        "candidate_hashes": {str(p): sha256(p) for p in (wrapper, table)},
        "target_scale": A["target_scale"], "target_scale_raw": A["s_raw"],
        "heldout_cost_ratio_vs_parent": B["cost_ratio_vs_parent"],
        "test_cost_vs_parent": a["cost"] / cell["parent_test_cost"] - 1,
        "test_cost_vs_incumbent": a["cost"] / cell["incumbent_test_cost"] - 1,
        "ppl_vs_incumbent_pct": (ppl / cell["incumbent_ppl"] - 1) * 100,
        "ppl_vs_parent_pct": (ppl / cell["parent_ppl"] - 1) * 100,
        "x_fp16": ppl / fp16, "fp16_ppl": fp16, "flip_thresholds": thr,
        "passes_1p05x_fp16": ppl <= thr["1.05x"], "passes_1p10x_fp16": ppl <= thr["1.10x"],
        "incumbent": {"arm": cell["incumbent_arm"], "ppl": cell["incumbent_ppl"],
                      "cost": cell["incumbent_test_cost"], "trace": cell["incumbent_trace"]},
        "best_all": {"previous_best_ppl": cell["incumbent_ppl"],
                     "new_best_ppl": min(ppl, cell["incumbent_ppl"]),
                     "new_best_arm": arm_name if ppl < cell["incumbent_ppl"] else cell["incumbent_arm"],
                     "entry_rule": manifest["preregistered_rules"]["best_all_entry"]},
        "disclosure": manifest["preregistered_rules"]["disclosure"],
        "cost_shares_test": a["cycle_share"], "mean_L_test": a["mean_L"],
        "trace_header": h,
    }
    write_json(Path(out) / "full_test_result.json", rec, exclusive=True)
    print(f"[r6] finalize {cell['id']}: PPL {ppl:.6f} ({rec['x_fp16']:.5f}x fp16; 1.05x "
          f"{'PASS' if rec['passes_1p05x_fp16'] else 'fail'}) vs incumbent "
          f"{rec['ppl_vs_incumbent_pct']:+.3f}%; test cost vs parent "
          f"{100 * rec['test_cost_vs_parent']:+.3f}%; protocol_ok {rec['protocol_ok']}", flush=True)
    return 0 if not failed else 7


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("preflight", "stage-a", "calib-argv", "calib-identity",
                                      "stage-b", "finalize"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--cell-dir", default="")
    ap.add_argument("--trace", default="")
    ap.add_argument("--allow-unfrozen", action="store_true",
                    help="preflight only: accept a manifest without code_hashes (dry runs)")
    ap.add_argument("--dry-scale", type=float, default=None,
                    help="calib-argv only (launcher dry run): use this scale, no stage_a.json")
    args = ap.parse_args()
    manifest, cell = load_manifest(args.manifest, args.index)
    out = Path(args.cell_dir or cell["cell_dir"])
    if args.stage == "preflight":
        preflight(manifest, cell, require_fresh=True, allow_unfrozen=args.allow_unfrozen)
        print(f"[r6] preflight PASS {cell['id']}", flush=True)
        return 0
    if args.stage == "calib-argv":
        if args.dry_scale is not None:
            scale = args.dry_scale
        else:
            A = read_json(out / "stage_a.json")
            if not A.get("ok"):
                raise ValueError("stage-a did not pass")
            scale = A["target_scale"]
        sys.stdout.write("\0".join(calib_argv(manifest, cell, out, scale)) + "\0")
        return 0
    try:
        if args.stage == "stage-a":
            return stage_a(manifest, cell, out)
        if args.stage == "calib-identity":
            return calib_identity(manifest, cell, out)
        if args.stage == "stage-b":
            return stage_b(manifest, cell, out)
        return finalize(manifest, cell, out, args.trace)
    except BaseException as exc:
        write_json(out / f"failure_{args.stage}.json",
                   {"type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
