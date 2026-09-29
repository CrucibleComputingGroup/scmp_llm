"""Round-7 EVAL library (2026-09-28): generic table/trace audits, windows, statistics,
the pre-registered kappa rule, and the GPU / CPU-simulation evaluators.

"Generic" = valid for tables that are NOT a known edit of the incumbent (INC): step-0's
c17 / c17_s80 tables and every round-7 re-solved table (UK / K / U). The round-6
arm-A-specific helpers (upstream-record identity, first-edit transform, INC-edit length
maps, allocation_only_diff) are deliberately NOT used. What is kept and generalized:

  * lineage validator vs INC: only allocation fields may differ (PRC thresholds, attention
    thresholds, attention per-bucket ladders <= 128, provenance keys; the protected-slice
    length or the table layer bucketing only with an explicit allowance);
  * runtime resolver sweep (CPU, torch): every (op, block) resolves through the RUNTIME
    (get_levels / classify_level_values / get_thresholds / get_escape_threshold /
    get_per_row_chunk) to exactly the JSON table's own ladder/thresholds, escape constants
    equal INC's, and a synthetic adaptive_classify_rows dispatch equals a pure-numpy
    dispatch computed from the JSON (the resolver-bug family guard);
  * realized-length audit of every trace against the table's OWN ladders, twice
    (runtime resolver via prc_r6_retarget.analyze_trace, and the JSON mirror here),
    plus same SC (op, block) set / equal total and per-(op, block) MACs vs INC on the
    same windows, fixed SC numerics, no length > 128.

Units: stream lengths and costs are HALVED code units (nominal = 2x); code cap 128.
Frozen round-3/4/6 sources are imported, never edited. Module import is torch-free.
"""
from __future__ import annotations

import copy
import gc
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "kernels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.ppl import prc_r6_attn_diag_arms as R  # noqa: E402  (round-6 frozen: import only)
from benchmark.ppl.prc_r6_retarget_20260928 import analyze_trace, mp_from_wrapper  # noqa: E402,F401
from benchmark.ppl.prc_local_refine import choose_disjoint_starts  # noqa: E402  (round-3 frozen)

CAP = 128
ATTN_OPS = R.ATTN_OPS
LINEAR_OPS = R.LINEAR_OPS
ALL_OPS = ATTN_OPS + LINEAR_OPS
KB = Path("/nfs/turbo/coe-nbleier/allenjin/hpca/kbands")
R7_ROOT = KB / "prc_r7_20260928"
SCMP = REPO.parent
MP_BEST = SCMP / "hpca_results/llm/ppl/mp_best"
BEST_ALL = SCMP / "hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.json"
R6_DIAG_MANIFEST = REPO / "benchmark/ppl/kbands/prc_r6_attn_diag_20260928.json"
R6_DIAG_ROOT = KB / "prc_r6_20260928"
ENV_PY = "/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python"

QWEN_TOKEN_SHA = "9bdd7deff48783430543edbc3bc2f12ecd2be45526f8a5a52b12ce8a9915ddde"
QWEN_NTOK = 2518423
CTX = 2048
WINDOW_DRAW_SEED = 260928
PREFIX_GUARD = (0, 65536)  # parent per-row calibrator's stripped-join prefix (scout 3b)

FIXED_PROVENANCE_KEYS = ("prc2_calib", "prc6_calib", "prc9_r6")
R7_PROVENANCE = re.compile(r"^(prc\d+_r7|r7_[a-z0-9_]+)$")
# Per-rung DESCRIPTIVE fields of a bucket payload (never read by the runtime; verified by
# the resolver sweep). They may be re-emitted when a bucket's ladder changes.
DESCRIPTIVE_BUCKET_KEYS = ("counts", "fractions", "avg_stoc_len", "pass1_counts", "pass1_fractions",
                           "pass1_level_mean_error", "pass1_avg_error")
RUNTIME_BUCKET_KEYS = ("thresholds", "stoc_len_levels", "metric_mean", "metric_std")
COST_GATE_TOL = 0.01
CONFIRM_Z = -2.0
# Profile/trace totals are integer MAC and cycle-MAC counts. Below 2**53 float64 holds them
# exactly (the collector sums integer-valued float64), so they must agree EXACTLY; above it the
# relative tolerance applies (13/13 round-6 GPU profiles agreed bit for bit).
EXACT_INT_LIMIT = 2 ** 53
PROFILE_TOL = 1e-12
FP16_PPL = {"4B": 10.0445, "llama8B": 7.2130, "14B": 8.6383, "30B": 7.2613}
MODEL_PATHS = {"30B": "Qwen/Qwen3-30B-A3B-Instruct-2507", "4B": "Qwen/Qwen3-4B-Instruct-2507"}
TOTAL_BLOCKS = {"30B": 48, "4B": 36}
SIM_LABEL = "SIMULATION (CPU dry-run from archived traces; synthetic NLLs; NOT a measurement)"


class IdentityError(RuntimeError):
    """An incumbent / reference re-evaluation did not reproduce a recorded result exactly."""


class AuditError(RuntimeError):
    """A table, trace or profile is not what the frozen manifest says it must be."""


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
sha256_file = R.sha256_file
canonical = R.canonical


def read_json(path):
    return json.loads(Path(path).read_text())


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


def resolve_table_path(wrapper_path) -> Path:
    """Table path exactly as loader.apply_mp_config_from_env resolves it."""
    spec = read_json(wrapper_path)
    tp = spec["threshold_table_path"]
    if not os.path.isabs(tp):
        tp = os.path.join(os.path.dirname(os.path.abspath(str(wrapper_path))), tp)
    return Path(tp)


def loader_fields(wrapper: dict) -> dict:
    """The wrapper fields loader.apply_mp_config_from_env reads (extra keys are ignored)."""
    esc_k = wrapper.get("escape_gate_k")
    return {"type": wrapper.get("type"),
            "stoc_len_levels": [int(v) for v in wrapper["stoc_len_levels"]],
            "escape_gate_k": None if esc_k is None else float(esc_k),
            "escape_stoc_len": int(wrapper.get("escape_stoc_len", 128))}


def check_wrapper_matches_inc(wrapper: dict, inc_wrapper: dict, name: str) -> None:
    a, b = loader_fields(wrapper), loader_fields(inc_wrapper)
    if a != b:
        raise AuditError(f"{name}: wrapper loader fields {a} != incumbent {b}")
    if a["type"] != "AdaptiveMPConfig" or a["escape_gate_k"] != 2.0 or a["escape_stoc_len"] != CAP:
        raise AuditError(f"{name}: expected AdaptiveMPConfig with the mu+2tau escape gate at 128, got {a}")


def bucket_index(block, total_blocks, num_buckets):
    return R.bucket_index(int(block), int(total_blocks), int(num_buckets))


def global_ladder(table: dict) -> list:
    return R.global_ladder(table)


# ---------------------------------------------------------------------------
# JSON mirror of the runtime resolvers (independent of scmp_kernels)
# ---------------------------------------------------------------------------
def _layer_buckets(table):
    return int(table.get("layer_buckets", 1) or 1)


def json_bucket_payload(table, op, block, nb):
    key = f"{op}:t0:l{bucket_index(block, nb, _layer_buckets(table))}"
    return key, (table.get("buckets") or {}).get(key)


def json_ladder(table, op, block, nb) -> list:
    """get_levels mirror: bucket ladder, else operator-default ladder, else global."""
    _key, payload = json_bucket_payload(table, op, block, nb)
    if isinstance(payload, dict) and payload.get("stoc_len_levels") is not None:
        return [int(v) for v in payload["stoc_len_levels"]]
    od = (table.get("operator_defaults") or {}).get(op)
    if isinstance(od, dict) and od.get("stoc_len_levels") is not None:
        return [int(v) for v in od["stoc_len_levels"]]
    return global_ladder(table)


def json_thresholds(table, op, block, nb):
    _key, payload = json_bucket_payload(table, op, block, nb)
    if isinstance(payload, dict) and payload.get("thresholds") is not None:
        return [float(v) for v in payload["thresholds"]]
    od = (table.get("operator_defaults") or {}).get(op)
    if isinstance(od, dict) and od.get("thresholds") is not None:
        return [float(v) for v in od["thresholds"]]
    return None


def json_escape_threshold(table, wrapper, op, block, nb):
    """get_escape_threshold mirror: mu + k sigma from the bucket, else operator default."""
    k = loader_fields(wrapper)["escape_gate_k"]
    if k is None:
        return None
    _key, payload = json_bucket_payload(table, op, block, nb)
    for p in (payload, (table.get("operator_defaults") or {}).get(op)):
        if isinstance(p, dict) and p.get("metric_mean") is not None and p.get("metric_std") is not None:
            return float(p["metric_mean"]) + k * float(p["metric_std"])
    return None


def json_values(table, wrapper, op, block, nb) -> list:
    """classify_level_values mirror: ladder (+ escape length appended when the gate is on
    and the escape length is not already a rung)."""
    ladder = json_ladder(table, op, block, nb)
    f = loader_fields(wrapper)
    if f["escape_gate_k"] is None or f["escape_stoc_len"] in ladder:
        return ladder
    return ladder + [f["escape_stoc_len"]]


def json_realizable(table, wrapper, op, block, nb) -> set:
    """Lengths a per-row-dispatched (op, block) can actually realize: the ladder, plus the
    escape length only if the gate can fire (t_esc < 1; the normalized metric max is 1)."""
    ladder = json_ladder(table, op, block, nb)
    out = set(ladder)
    f = loader_fields(wrapper)
    t_esc = json_escape_threshold(table, wrapper, op, block, nb)
    if f["escape_gate_k"] is not None and t_esc is not None and t_esc < 1.0:
        out.add(f["escape_stoc_len"])
    return out


def json_prc_entry(table, op, block, nb):
    prc = table.get("per_row_chunk") or {}
    if not prc.get("buckets"):
        return None
    n_lb = int(prc.get("layer_buckets", 0) or 0) or _layer_buckets(table)
    entry = prc["buckets"].get(f"{op}:t0:l{bucket_index(block, nb, n_lb)}")
    if entry is None:
        return None
    return [int(v) for v in entry["levels"]], [float(v) for v in entry["thresholds"]]


def protected_widths(table):
    out = {}
    pc = table.get("protected_channels") or {}
    for key, idx in (pc.get("indices") or {}).items():
        parts = key.split(":")
        op, blk = parts[0], int(parts[1][1:])
        unit = int(parts[2][1:]) if len(parts) == 3 else None
        out[(op, blk, unit)] = len(idx)
    psl = int(pc["stoc_len"]) if pc.get("indices") else None
    return out, psl


def numpy_dispatch(metric, thresholds, ladder, *, t_esc, esc_len, gate_on):
    """Pure-numpy mirror of adaptive_classify_rows' calibrated path + _apply_escape_gate,
    mapped to stream lengths. ``metric`` float32 1-D."""
    import numpy as np
    m = np.asarray(metric, dtype=np.float32)
    ladder = [int(v) for v in ladder]
    values = ladder if (not gate_on or esc_len in ladder) else ladder + [int(esc_len)]
    if m.size == 0:
        return np.zeros(0, dtype=np.int64)
    lo, hi = m.min(), m.max()
    if float(hi - lo) < 1e-8:
        return np.full(m.shape, ladder[0], dtype=np.int64)
    norm = (m - lo) / (hi - lo)
    idx = np.full(m.shape, len(ladder) - 1, dtype=np.int64)
    for i, th in enumerate(thresholds):
        lower = np.float32(th)
        if i == 0:
            mask = norm >= lower
        else:
            mask = (norm >= lower) & (norm < np.float32(thresholds[i - 1]))
        idx[mask] = i
    if gate_on and t_esc is not None and t_esc < 1.0:
        esc = norm > np.float32(t_esc)
        idx[esc] = ladder.index(int(esc_len)) if int(esc_len) in ladder else len(ladder)
    return np.asarray(values, dtype=np.int64)[idx]


# ---------------------------------------------------------------------------
# lineage validator (pure python)
# ---------------------------------------------------------------------------
def _is_provenance_key(k):
    return k in FIXED_PROVENANCE_KEYS or bool(R7_PROVENANCE.match(k))


def _check_attention_bucket(key, payload, failures):
    ladder = payload.get("stoc_len_levels")
    th = payload.get("thresholds")
    if ladder is not None:
        ladder = [int(v) for v in ladder]
        if len(ladder) < 2 or any(b >= a for a, b in zip(ladder, ladder[1:])):
            failures.append(f"bucket ladder not strictly descending with >=2 rungs: {key} {ladder}")
        if max(ladder) > CAP or min(ladder) < 1:
            failures.append(f"bucket ladder outside [1, {CAP}]: {key} {ladder}")
    return ladder, th


def validate_lineage(inc: dict, cand: dict, *, total_blocks: int, allow_psl_change=False,
                     allow_layer_rekey=False) -> dict:
    """``cand`` may differ from INC ONLY in allocation fields (see module doc).

    Returns a report; ``report['ok']`` is False with ``failures`` listing every violation."""
    failures, notes = [], []
    nb = int(total_blocks)
    for t, name in ((inc, "INC"), (cand, "candidate")):
        if int(t.get("sc_prec", 8)) != 8 or not bool(t.get("halve_bipolar_stoc_len", True)):
            failures.append(f"{name}: sc_prec/halve are not 8/True (cap {CAP})")
        if int(t.get("timestep_buckets", 1)) != 1:
            failures.append(f"{name}: timestep bucketing is not supported")
    if [int(v) for v in cand.get("stoc_len_levels", [])] != global_ladder(inc):
        failures.append("top-level stoc_len_levels changed (runtime requires wrapper == table ladder)")
    lb_inc, lb_c = _layer_buckets(inc), _layer_buckets(cand)
    rekey = lb_inc != lb_c
    if rekey and not allow_layer_rekey:
        failures.append(f"layer_buckets changed {lb_inc} -> {lb_c} without an explicit allowance")
    # ---- per-(op, block) resolution of every bucket (works for equal and re-keyed tables)
    att_th_changed, att_ladder_changed, desc_changed = set(), {}, set()
    seen_c = set()
    for op in ALL_OPS:
        for block in range(nb):
            ki, pi = json_bucket_payload(inc, op, block, nb)
            kc, pc = json_bucket_payload(cand, op, block, nb)
            seen_c.add(kc)
            if (pi is None) != (pc is None):
                failures.append(f"bucket presence differs at ({op}, {block}): {ki} vs {kc}")
                continue
            if pi is None:
                continue
            for f in ("metric_mean", "metric_std"):
                if pi.get(f) != pc.get(f):
                    failures.append(f"escape constant {f} changed at ({op}, {block}) {ki}->{kc}")
            if op in ATTN_OPS:
                ladder, th = _check_attention_bucket(kc, pc, failures)
                n_lv = len(ladder) if ladder is not None else len(global_ladder(cand))
                if th is None or len(th) != n_lv - 1:
                    failures.append(f"attention thresholds shape wrong at {kc}")
                else:
                    th = [float(v) for v in th]
                    if any(not (0.0 <= v <= 1.0) or not math.isfinite(v) for v in th):
                        failures.append(f"attention thresholds outside [0, 1] at {kc}")
                    if any(b > a + 1e-6 for a, b in zip(th, th[1:])):
                        failures.append(f"attention thresholds not non-increasing at {kc}")
                if [float(v) for v in (pc.get("thresholds") or [])] != [float(v) for v in (pi.get("thresholds") or [])]:
                    att_th_changed.add(kc)
                if pc.get("stoc_len_levels") != pi.get("stoc_len_levels"):
                    att_ladder_changed[kc] = [int(v) for v in pc["stoc_len_levels"]] \
                        if pc.get("stoc_len_levels") is not None else None
                extra = set(pc) - set(pi) - {"stoc_len_levels"} - set(DESCRIPTIVE_BUCKET_KEYS)
                if extra:
                    failures.append(f"new non-allocation fields in attention bucket {kc}: {sorted(extra)}")
                a = {k: v for k, v in pi.items() if k not in RUNTIME_BUCKET_KEYS + DESCRIPTIVE_BUCKET_KEYS}
                b = {k: v for k, v in pc.items() if k not in RUNTIME_BUCKET_KEYS + DESCRIPTIVE_BUCKET_KEYS}
                if canonical(a) != canonical(b):
                    failures.append(f"non-allocation fields changed in attention bucket {kc}")
                if any(canonical(pi.get(k)) != canonical(pc.get(k)) for k in DESCRIPTIVE_BUCKET_KEYS):
                    desc_changed.add(kc)
            else:
                if pc.get("stoc_len_levels") is not None:
                    failures.append(f"linear per-row bucket {kc} carries a ladder")
                if canonical(pi) != canonical(pc):
                    failures.append(f"linear per-row bucket payload changed at ({op}, {block}) {ki}->{kc}")
    stray = set((cand.get("buckets") or {})) - seen_c
    if stray:
        failures.append(f"buckets never resolved by any (op, block): {sorted(stray)[:5]}")
    # ---- per_row_chunk: keys / levels / non-bucket fields fixed; thresholds may move
    prc_i, prc_c = inc.get("per_row_chunk") or {}, cand.get("per_row_chunk") or {}
    if not prc_c.get("buckets"):
        failures.append("candidate has no per_row_chunk buckets")
    eff_i = int(prc_i.get("layer_buckets", 0) or 0) or lb_inc
    eff_c = int(prc_c.get("layer_buckets", 0) or 0) or lb_c
    if eff_i != eff_c:
        failures.append(f"per_row_chunk effective layer buckets changed {eff_i} -> {eff_c} "
                        "(a re-keyed table must pin per_row_chunk.layer_buckets explicitly)")
    if set(prc_i.get("buckets", {})) != set(prc_c.get("buckets", {})):
        failures.append("per_row_chunk bucket keys changed")
    prc_th_changed = set()
    for k in set(prc_i.get("buckets", {})) & set(prc_c.get("buckets", {})):
        ei, ec = prc_i["buckets"][k], prc_c["buckets"][k]
        if set(ec) != {"levels", "thresholds"} or set(ei) != {"levels", "thresholds"}:
            failures.append(f"per_row_chunk {k} carries unexpected fields")
        if [int(v) for v in ec["levels"]] != [int(v) for v in ei["levels"]]:
            failures.append(f"per_row_chunk {k} levels changed")
        lv = [int(v) for v in ec["levels"]]
        if lv != sorted(lv) or max(lv) > CAP or min(lv) < 1 or len(lv) < 2:
            failures.append(f"per_row_chunk {k} ladder invalid {lv}")
        th = [float(v) for v in ec["thresholds"]]
        if len(th) != len(lv) - 1 or th != sorted(th) or any(not (0.0 <= v <= 1.0) for v in th):
            failures.append(f"per_row_chunk {k} thresholds malformed")
        if th != [float(v) for v in ei["thresholds"]]:
            prc_th_changed.add(k)
    oi = {k: v for k, v in prc_i.items() if k not in ("buckets", "layer_buckets")}
    oc = {k: v for k, v in prc_c.items() if k not in ("buckets", "layer_buckets")}
    if canonical(oi) != canonical(oc):
        failures.append("per_row_chunk non-bucket fields changed")
    # ---- protected channels
    pi_, pc_ = copy.deepcopy(inc.get("protected_channels") or {}), copy.deepcopy(cand.get("protected_channels") or {})
    psl_i, psl_c = pi_.pop("stoc_len", None), pc_.pop("stoc_len", None)
    if canonical(pi_) != canonical(pc_):
        failures.append("protected channel indices / fields changed")
    if psl_i != psl_c:
        if not allow_psl_change:
            failures.append(f"protected-slice length changed {psl_i} -> {psl_c} without an explicit allowance")
        elif not (isinstance(psl_c, int) and 1 <= psl_c <= CAP):
            failures.append(f"protected-slice length {psl_c} outside [1, {CAP}]")
    # ---- everything else at top level
    skip = {"buckets", "per_row_chunk", "protected_channels"} | ({"layer_buckets"} if rekey else set())
    keys = set(inc) | set(cand)
    prov = sorted(k for k in keys if _is_provenance_key(k))
    for k in sorted(keys - skip - set(prov)):
        if canonical(inc.get(k)) != canonical(cand.get(k)):
            failures.append(f"top-level field changed: {k}")
    if canonical(strip_provenance(inc)) == canonical(strip_provenance(cand)):
        notes.append("candidate is allocation-identical to INC")
    return {"ok": not failures, "failures": failures[:50], "n_failures": len(failures),
            "prc_threshold_keys_changed": sorted(prc_th_changed),
            "attention_threshold_keys_changed": sorted(att_th_changed),
            "attention_ladders_changed": dict(sorted(att_ladder_changed.items())),
            "descriptive_bucket_fields_changed": sorted(desc_changed),
            "provenance_keys": prov, "layer_rekey": rekey,
            "protected_stoc_len": [psl_i, psl_c], "notes": notes}


def strip_provenance(t: dict) -> dict:
    return {k: v for k, v in t.items() if not _is_provenance_key(k)}


def allocation_signature(t: dict) -> str:
    """Canonical content of a table minus provenance (dedup of identical arms)."""
    return hashlib.sha256(canonical(strip_provenance(t)).encode()).hexdigest()


def escape_disclosure(table: dict, wrapper: dict, total_blocks: int) -> dict:
    """Per attention bucket whose ladder contains the escape length: whether the mu+2tau
    gate can still fire (t_esc < 1: escaped rows fold onto the 128 rung; the gate adds
    nothing where the calibrated 128 threshold lies at or below t_esc) or never fires."""
    out = {}
    f = loader_fields(wrapper)
    nb = int(total_blocks)
    for op in ATTN_OPS:
        for block in range(nb):
            key, payload = json_bucket_payload(table, op, block, nb)
            if key in out or payload is None:
                continue
            ladder = json_ladder(table, op, block, nb)
            if f["escape_stoc_len"] not in ladder:
                continue
            t_esc = json_escape_threshold(table, wrapper, op, block, nb)
            th = json_thresholds(table, op, block, nb) or []
            th128 = th[ladder.index(f["escape_stoc_len"])] if ladder.index(f["escape_stoc_len"]) < len(th) else None
            if t_esc is None or t_esc >= 1.0:
                status = "gate never fires (t_esc >= 1); 128 is a genuinely new rung"
            elif th128 is not None and th128 <= t_esc:
                status = "gate redundant: the calibrated 128 threshold lies at or below mu+2tau"
            else:
                status = "gate still the only route to 128 above its threshold; escaped rows fold onto the 128 rung"
            out[key] = {"ladder": ladder, "t_esc": t_esc, "threshold_128": th128, "status": status}
    return out


# ---------------------------------------------------------------------------
# runtime resolver sweep (CPU; imports torch + scmp_kernels)
# ---------------------------------------------------------------------------
def resolver_sweep(wrapper_path, inc_wrapper_path, total_blocks, *, allow_psl_change=False,
                   synthetic_rows=2048, seed=20260928) -> dict:
    """Load the candidate EXACTLY as the runtime does and check every (op, block):
    get_levels == JSON ladder, classify_level_values == JSON values, get_thresholds == JSON,
    get_escape_threshold == INC's == JSON mirror, get_per_row_chunk == JSON, and a synthetic
    adaptive_classify_rows dispatch == the pure-numpy dispatch from the JSON (including
    escape folding onto a 128 rung and the constant-metric branch)."""
    import numpy as np
    import torch
    from scmp_kernels.mp.config import _bucket_index, adaptive_classify_rows
    cfg = mp_from_wrapper(str(wrapper_path))
    inc = mp_from_wrapper(str(inc_wrapper_path))
    wrapper = read_json(wrapper_path)
    table = read_json(resolve_table_path(wrapper_path))
    inc_table = read_json(resolve_table_path(inc_wrapper_path))
    nb = int(total_blocks)
    gen = torch.Generator().manual_seed(int(seed))
    f = loader_fields(wrapper)
    checked = {"attention": 0, "prc": 0, "linear_escape": 0}
    rows_by_len = defaultdict(int)
    for n_lb in {_layer_buckets(table), int((table.get("per_row_chunk") or {}).get("layer_buckets", 0) or 0)} - {0}:
        for b in range(nb):
            if bucket_index(b, nb, n_lb) != _bucket_index(b, nb, n_lb):
                raise AuditError("bucket_index mirror disagrees with the runtime")
    for op in ALL_OPS:
        for block in range(nb):
            e_rt = cfg.get_escape_threshold(0, 1, operator=op, block_idx=block, total_blocks=nb)
            e_inc = inc.get_escape_threshold(0, 1, operator=op, block_idx=block, total_blocks=nb)
            e_js = json_escape_threshold(table, wrapper, op, block, nb)
            if e_rt != e_inc or e_rt != e_js:
                raise AuditError(f"escape threshold ({op}, {block}): runtime {e_rt} INC {e_inc} JSON {e_js}")
            if op not in ATTN_OPS:
                checked["linear_escape"] += 1
                continue
            want = json_ladder(table, op, block, nb)
            got = cfg.get_levels(operator=op, block_idx=block, total_blocks=nb)
            if list(got) != want:
                raise AuditError(f"get_levels({op}, {block}) = {got}, JSON {want}")
            if max(got) > CAP:
                raise AuditError(f"ladder above {CAP} at ({op}, {block}): {got}")
            vals = cfg.classify_level_values(operator=op, block_idx=block, total_blocks=nb)
            if list(vals) != json_values(table, wrapper, op, block, nb):
                raise AuditError(f"classify_level_values({op}, {block}) = {vals}")
            th = cfg.get_thresholds(0, 1, operator=op, block_idx=block, total_blocks=nb)
            jth = json_thresholds(table, op, block, nb)
            if th is None or [float(v) for v in th] != jth:
                raise AuditError(f"get_thresholds({op}, {block}) = {th}, JSON {jth}")
            for trial in range(3):
                if trial == 2:
                    metric = torch.full((64,), 0.37, dtype=torch.float32)  # constant-metric branch
                else:
                    metric = (torch.rand(synthetic_rows, generator=gen) ** (3 if trial == 0 else 0.5)).float()
                a = adaptive_classify_rows(metric, cfg, operator=op, block_idx=block, total_blocks=nb)
                L_rt = torch.tensor(vals)[a.row_levels].numpy()
                L_np = numpy_dispatch(metric.numpy(), jth, want, t_esc=e_js, esc_len=f["escape_stoc_len"],
                                      gate_on=f["escape_gate_k"] is not None)
                if not np.array_equal(L_rt, L_np):
                    raise AuditError(f"synthetic dispatch ({op}, {block}) runtime != JSON mirror "
                                     f"({int((L_rt != L_np).sum())} rows)")
                for v, rows in a.level_row_indices.items():
                    want_rows = np.nonzero(L_rt == int(v))[0]
                    if not np.array_equal(np.sort(rows.numpy()), want_rows):
                        raise AuditError(f"level_row_indices[{v}] disagrees with row_levels ({op}, {block})")
                covered = sum(int(r.numel()) for r in a.level_row_indices.values())
                if covered != metric.numel():
                    raise AuditError(f"level_row_indices do not partition the rows ({op}, {block})")
                for L in L_rt.tolist():
                    rows_by_len[int(L)] += 1
            checked["attention"] += 1
    for op in LINEAR_OPS:
        for block in range(nb):
            got = cfg.get_per_row_chunk(op, block, nb)
            want = json_prc_entry(table, op, block, nb)
            gi = inc.get_per_row_chunk(op, block, nb)
            if (got is None) != (want is None) or (got is None) != (gi is None):
                raise AuditError(f"get_per_row_chunk({op}, {block}) presence: runtime {got is not None}, "
                                 f"JSON {want is not None}, INC {gi is not None}")
            if got is None:
                continue
            if (list(got[0]), [float(v) for v in got[1]]) != want:
                raise AuditError(f"get_per_row_chunk({op}, {block}) = {got}, JSON {want}")
            if list(gi[0]) != list(got[0]):
                raise AuditError(f"PRC ladder differs from INC at ({op}, {block})")
            checked["prc"] += 1
    if cfg.protected_channel_indices != inc.protected_channel_indices:
        raise AuditError("protected channel indices changed")
    if cfg.protected_channel_stoc_len != inc.protected_channel_stoc_len and not allow_psl_change:
        raise AuditError(f"protected-slice length {cfg.protected_channel_stoc_len} != INC "
                         f"{inc.protected_channel_stoc_len}")
    if (cfg.protected_channel_stoc_len or 0) > CAP:
        raise AuditError("protected-slice length above the cap")
    if [int(v) for v in cfg.stoc_len_levels] != global_ladder(inc_table) or cfg.dispatch_metrics != inc.dispatch_metrics:
        raise AuditError("runtime global ladder or dispatch metrics changed")
    return {"ok": True, "checked": checked, "protected_stoc_len": cfg.protected_channel_stoc_len,
            "synthetic_rows_by_len": dict(sorted(rows_by_len.items()))}


# ---------------------------------------------------------------------------
# trace audits
# ---------------------------------------------------------------------------
def json_trace_audit(payload: dict, wrapper: dict, table: dict, total_blocks: int) -> dict:
    """Realized-length audit against the table's OWN JSON ladders (independent of the
    runtime resolver): attention -> ladder (+ escape if it can fire); linear main slice ->
    PRC levels (per-row values when a table has no PRC entry); protected slice -> psl."""
    widths, psl = protected_widths(table)
    nb = int(total_blocks)
    violations, n = [], 0
    cache = {}
    for g in payload["groups"]:
        op, blk, L = g["op"], g["block"], int(g["stoc_len"])
        n += 1
        if blk is None or not (1 <= L <= CAP):
            violations.append({"op": op, "block": blk, "stoc_len": L, "why": "no block or L outside [1,128]"})
            continue
        blk = int(blk)
        if op in ATTN_OPS:
            key = ("att", op, blk)
            if key not in cache:
                cache[key] = json_realizable(table, wrapper, op, blk, nb)
            allowed = cache[key]
        elif op in LINEAR_OPS:
            w = widths.get((op, blk, g.get("unit"))) or widths.get((op, blk, None))
            if w is not None and psl is not None and int(g["d_in"]) == w and L == psl:
                continue
            key = ("lin", op, blk)
            if key not in cache:
                prc = json_prc_entry(table, op, blk, nb)
                cache[key] = set(prc[0]) if prc is not None else json_realizable(table, wrapper, op, blk, nb)
            allowed = cache[key]
        else:
            violations.append({"op": op, "block": blk, "why": "unknown op"})
            continue
        if L not in allowed:
            violations.append({"op": op, "block": blk, "unit": g.get("unit"), "stoc_len": L,
                               "allowed": sorted(allowed)})
    return {"ok": not violations, "n_groups": n, "n_violations": len(violations), "violations": violations[:30]}


def class_costs(agg: "R.TraceAgg") -> dict:
    cyc, mac = defaultdict(int), defaultdict(int)
    for (kind, _op, _q), hist in agg.bucket_hist.items():
        for L, m in hist.items():
            cyc[kind] += L * m
            mac[kind] += m
    tot = agg.total_macs
    return {"cycles_per_sc_mac": {k: cyc[k] / tot for k in sorted(cyc)},
            "mac_share": {k: mac[k] / tot for k in sorted(mac)},
            "mean_len": {k: cyc[k] / mac[k] for k in sorted(mac) if mac[k]}}


def bucket_delta_vs_inc(agg, inc_agg, inc_table, inc_wrapper) -> dict:
    """Descriptive per-bucket delta cycles (pp of INC cycles) and the MAC share an arm puts
    on lengths INC's own ladder for that bucket could not realize (e.g. a new 128 rung)."""
    out = {}
    nb = agg.total_blocks
    for bkey in sorted(set(agg.bucket_hist) | set(inc_agg.bucket_hist)):
        kind, op, q = bkey
        ha, hi = agg.bucket_hist.get(bkey, {}), inc_agg.bucket_hist.get(bkey, {})
        ca, ci = sum(L * m for L, m in ha.items()), sum(L * m for L, m in hi.items())
        ma = sum(ha.values())
        entry = {"kind": kind, "delta_cycles_pct_of_inc": 100.0 * (ca - ci) / inc_agg.total_cycle_macs,
                 "mean_len": ca / ma if ma else None, "inc_mean_len": ci / sum(hi.values()) if hi else None,
                 "share_by_len": {str(L): m / ma for L, m in sorted(ha.items())} if ma else {}}
        if kind == "attention" and ma:
            blocks = [b for b in range(nb) if bucket_index(b, nb, agg.layer_buckets) == q]
            inc_ok = set()
            for b in blocks:
                inc_ok |= json_realizable(inc_table, inc_wrapper, op, b, nb)
            mi = sum(hi.values())
            # 128 is realizable by INC only through the escape gate; report both shares explicitly
            entry["share_outside_inc_realizable"] = sum(m for L, m in ha.items() if L not in inc_ok) / ma
            entry["share_at_128"] = ha.get(CAP, 0) / ma
            entry["inc_share_at_128"] = (hi.get(CAP, 0) / mi) if mi else None
        out[f"{op}:l{q}"] = entry
    return out


def audit_trace(payload: dict, *, name: str, wrapper_path, inc_agg, inc_table: dict, inc_wrapper: dict,
                hybrid: dict, total_blocks: int, layer_buckets: int, expected_windows=None,
                expected_wrapper=None, compare_macs=True) -> tuple[dict, "R.TraceAgg"]:
    """Every generic check for one arm trace; returns (report, TraceAgg)."""
    wrapper = read_json(wrapper_path)
    table = read_json(resolve_table_path(wrapper_path))
    agg = R.TraceAgg(payload, total_blocks, layer_buckets)
    failures = R.audit_trace_basics(agg, name, hybrid)
    hdr = agg.header
    if expected_windows is not None and hdr.get("windows") != [int(s) for s in expected_windows]:
        failures.append(f"{name}: trace header windows differ from the evaluated windows")
    if expected_wrapper is not None and hdr.get("mp_config_json") != str(expected_wrapper):
        failures.append(f"{name}: trace header mp_config_json is not the arm wrapper")
    if inc_agg is not None and compare_macs:
        if agg.op_blocks() != inc_agg.op_blocks():
            failures.append(f"{name}: SC (op, block) set differs from INC")
        if agg.total_macs != inc_agg.total_macs:
            failures.append(f"{name}: total MACs {agg.total_macs} != INC {inc_agg.total_macs}")
        bad = [k for k in inc_agg.op_block_macs if inc_agg.op_block_macs[k] != agg.op_block_macs.get(k)]
        if bad:
            failures.append(f"{name}: per-(op, block) MACs differ at {len(bad)} positions, e.g. {sorted(bad)[:3]}")
    rt = analyze_trace(payload, str(wrapper_path), total_blocks=total_blocks, arm="gfisla")
    if not rt["audit_ok"]:
        failures.append(f"{name}: runtime-resolver realized-length audit failed "
                        f"({rt['n_violations']} violations, e.g. {rt['violations'][:2]})")
    js = json_trace_audit(payload, wrapper, table, total_blocks)
    if not js["ok"]:
        failures.append(f"{name}: JSON-ladder realized-length audit failed "
                        f"({js['n_violations']} violations, e.g. {js['violations'][:2]})")
    max_len = max(int(g["stoc_len"]) for g in payload["groups"])
    if max_len > CAP:
        failures.append(f"{name}: realized length {max_len} > {CAP}")
    report = {"ok": not failures, "failures": failures, "name": name, "wrapper": str(wrapper_path),
              "cost": agg.cost, "total_macs": agg.total_macs, "total_cycle_macs": agg.total_cycle_macs,
              "max_len": max_len, "runtime_audit": {k: rt[k] for k in ("audit_ok", "n_violations", "cost", "U_prot",
                                                                          "cycle_share", "mac_share", "mean_L")},
              "json_audit": {k: js[k] for k in ("ok", "n_groups", "n_violations")},
              "class_costs": class_costs(agg),
              "histogram_macs": rt["histogram_macs"]}
    if abs(rt["cost"] / agg.cost - 1) > 1e-9:
        report["ok"] = False
        report["failures"].append(f"{name}: analyze_trace cost {rt['cost']} != TraceAgg cost {agg.cost}")
    if inc_agg is not None:
        report["buckets_vs_inc"] = bucket_delta_vs_inc(agg, inc_agg, inc_table, read_json_cached_wrapper(inc_wrapper))
    return report, agg


def read_json_cached_wrapper(w):
    return w if isinstance(w, dict) else read_json(w)


def load_trace(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# statistics and pre-registered rules
# ---------------------------------------------------------------------------
paired = R.paired
arm_metrics = R.arm_metrics


def cost_gate(C, P, tol=COST_GATE_TOL) -> dict:
    ratio = C / P
    return {"C": C, "P": P, "ratio": ratio, "tolerance": tol, "ok": bool(abs(ratio - 1.0) <= tol + 1e-12)}


def screen_proceeds(primary: dict) -> dict:
    """Pre-registered screen gate for the PRIMARY arm (secondaries: information only)."""
    conds = {"identity_and_audits_ok": bool(primary.get("audit_ok")),
             "cost_gate_ok": bool(primary.get("cost_gate", {}).get("ok")),
             "screen_mean_dnll_negative": primary["metrics"]["dnll"] < 0.0}
    return {"proceed_to_confirm": all(conds.values()), "conditions": conds}


def confirm_passes(metrics: dict, audits_ok: bool) -> dict:
    z = metrics.get("z")
    ok = bool(audits_ok and z is not None and z < CONFIRM_Z)
    return {"full_test": ok, "z": z, "threshold": CONFIRM_Z, "audits_ok": bool(audits_ok)}


def pred_passes_mde(pred_true, mde) -> bool:
    """Refuse arms whose predicted effect is below the MDE or of the wrong sign:
    require pred_true <= -MDE (nats; negative = improvement)."""
    return (isinstance(pred_true, (int, float)) and isinstance(mde, (int, float)) and mde > 0
            and math.isfinite(pred_true) and pred_true <= -mde)


KAPPA_RULE = {
    "primary_ratio": 0.42,
    "kappa_att": 0.83, "kappa_att_se": 0.07,
    "kappa_att_source": "CRIT section 1 / scout_calib section 3: pooled held-out kappa_att over 30B "
                        "t32/t40/t48 (0.81-0.83 +- 0.07); primary ratio 0.42 = kappa_att/kappa_lin fitted on "
                        "held-out only",
    "cells": ["30B_t32", "30B_t40"],
    "pred_key": "pred_dnll_fis",
    "score_names": {"c17": "c17", "s80": "c17_s80"},
    "statistic": "kappa_lin_bs = sum_c m_c / sum_c p_c over the step-0 cells; m_c = mean over the 16 paired "
                 "TRAIN windows of NLL(c17_s80) - NLL(c17) (GPU, step 0); p_c = pred_dnll_fis(c17_s80) - "
                 "pred_dnll_fis(c17) from the round-7 capture diag of cell c (MC-Fisher, capture hold windows; "
                 "both tables score linear thresholds with parent attention, so p_c is a pure linear "
                 "budget-shift prediction)",
    "se": "SE_bs = sd_w(sum_c d_cw) / sqrt(n_w) / |sum_c p_c| (cells share the same windows, so the "
          "per-window sum carries their covariance); prediction noise excluded (disclosed)",
    "decision": "ONE-SIDED (plan section 4: the family currency is live only 'provided the step-0 check confirms "
                "kappa_lin > kappa_att'): keep ratio 0.42 iff kappa_lin_bs - kappa_att > sqrt(SE_bs^2 + "
                "kappa_att_se^2); otherwise revert to kappa = 1 (ratio 1.0): inside the band (the task's 'within "
                "1 SE' case), below it (direction_contradicted) or undefined (sum p_c <= 0, a cell missing / not "
                "ok, a capture whose identity is not ok, different window lists). Undefined also sets "
                "requires_user_review.",
    "aligned_with": "prc_r7_solve.KAPPA_RULE id prc-r7-kappa-rule-v2-20260928 (kappa_decision_core): the same "
                    "branch function of (kappa_lin_bs, SE_bs, undefined); cross-checked in "
                    "test_prc_screen_r7_20260928",
    "flags_nonbinding": "direction_contradicted (kappa_lin_bs < kappa_att - band: linears measured cheaper "
                        "than attention) and gap_case (keep, but implied ratio kappa_att/kappa_lin_bs > 0.55, "
                        "where 0.42 may over-correct) are reported and set requires_user_review; they do not "
                        "change the branch.",
    "per_cell_sign_check": "a -20% linear budget cut must be predicted AND measured to raise NLL: any cell with "
                           "p_c <= 0 or m_c <= 0 is flagged cell_sign_inconsistent (broken score or broken "
                           "measurement the pooled ratio would hide) and sets requires_user_review; it does not "
                           "change the branch.",
    "step0_cell_integrity": "the kappa stage flags (requires_user_review) any rule cell whose step-0 job wrote "
                            "failure.json or has no complete step0_summary.json after step0_measured.json, and "
                            "treats a cell whose round-7 capture identity.json is missing or not identity_ok as "
                            "missing (no Fisher prediction from a capture that failed identity -> undefined).",
    "authority": "ONE kappa rule for round 7 (kappa_att 0.83 +- 0.07, pooled over 30B t32 + t40, keep ratio 0.42, "
                 "one-sided), implemented identically on both sides. The eval-side kappa_decision.json written by "
                 "prc_step0_r7_20260928.py kappa (schema prc-step0-r7-v1-kappa) is the step-0 decision RECORD: the "
                 "screen builder requires it, and the calib-side solver re-derives the same branch from its "
                 "statistic (prc_r7_solve.kappa_branch_from_record) and refuses pins that disagree. The v1 "
                 "calib-side constants (0.82 +- 0.073, two-sided) are superseded. Absolute (kappa_lin, kappa_att) "
                 "per branch: prc_r7_prereg_20260928.KAPPA_PINS (keep 1.96/0.83; revert 0.83/0.83; dense 1/1).",
}


def kappa_decision(measured: dict, preds: dict, rule: dict = KAPPA_RULE) -> dict:
    """``measured[cell] = {"window_dnll": [...], "windows": [...], "ok": bool}`` (s80 - c17);
    ``preds[cell] = p_c``. Returns the decision record (see KAPPA_RULE)."""
    cells = list(rule["cells"])
    out = {"rule": rule, "per_cell": {}, "requires_user_review": False, "flags": []}
    missing = [c for c in cells if c not in measured or c not in preds or not measured[c].get("ok")]
    for c in cells:
        if c in measured and c in preds and measured[c].get("ok"):
            d = measured[c]["window_dnll"]
            st = paired([x for x in d], [0.0] * len(d))
            p = float(preds[c])
            out["per_cell"][c] = {"m": st["mean_dnll"], "se_m": st["se"], "p": p,
                                  "kappa_lin_bs": (st["mean_dnll"] / p) if p > 0 else None,
                                  "se_kappa": (st["se"] / abs(p)) if p else None}
            if not (p > 0.0) or not (st["mean_dnll"] > 0.0):
                out["flags"].append(f"cell_sign_inconsistent:{c} (p_c {p:+.6g}, m_c {st['mean_dnll']:+.6g}; a "
                                    "-20% linear budget cut must be predicted and measured to raise NLL)")
                out["requires_user_review"] = True
    # one-sided: revert unless the statistic is defined AND exceeds kappa_att by more than the band
    ratio, decision, band, k_bs, se_bs = 1.0, "revert_to_kappa_1", None, None, None
    if missing:
        out["requires_user_review"] = True
        out["flags"].append(f"undefined: missing/failed cells {missing}")
    else:
        wins = [tuple(measured[c]["windows"]) for c in cells]
        n = len(wins[0])
        if len(set(wins)) != 1 or any(len(measured[c]["window_dnll"]) != n for c in cells) or n < 2:
            out["requires_user_review"] = True
            out["flags"].append("undefined: step-0 cells do not share one window list of >= 2 windows")
            n = 0
    if not missing and n:
        sums = [sum(measured[c]["window_dnll"][i] for c in cells) for i in range(n)]
        P = sum(float(preds[c]) for c in cells)
        M = sum(sums) / n
        sd = math.sqrt(sum((x - M) ** 2 for x in sums) / (n - 1))
        if P <= 0:
            out["requires_user_review"] = True
            out["flags"].append(f"undefined: pooled prediction sum p = {P} <= 0")
        else:
            k_bs, se_bs = M / P, sd / math.sqrt(n) / P
            band = math.sqrt(se_bs ** 2 + rule["kappa_att_se"] ** 2)
            if (k_bs - rule["kappa_att"]) > band:
                ratio, decision = rule["primary_ratio"], "keep_0.42"
            if k_bs < rule["kappa_att"] - band:
                out["flags"].append("direction_contradicted")
                out["requires_user_review"] = True
            if decision == "keep_0.42" and rule["kappa_att"] / k_bs > 0.55:
                out["flags"].append("gap_case")
                out["requires_user_review"] = True
            out["sensitivity_se_bs_only"] = {"band": se_bs,
                                             "would_keep": (k_bs - rule["kappa_att"]) > se_bs}
        out["pooled"] = {"sum_m_per_cell_mean": M, "sum_p": P, "n_windows": n, "sd_window_sum": sd}
    out.update(kappa_lin_bs=k_bs, se_kappa_lin_bs=se_bs, band=band, decision=decision, ratio=ratio,
               implied_ratio_measured=(rule["kappa_att"] / k_bs) if k_bs else None)
    return out


# ---------------------------------------------------------------------------
# windows (Qwen WikiText-2 TRAIN stream)
# ---------------------------------------------------------------------------
def stratified_starts(n, seed, ntok=QWEN_NTOK, ctx=CTX) -> list:
    """Mirror of calibrate_mp_thresholds._select_int_swap_windows(sampling='stratified')."""
    import numpy as np
    total_blocks = int(ntok) // int(ctx)
    take = min(int(n), total_blocks)
    rng = np.random.default_rng(int(seed))
    edges = np.linspace(0, total_blocks, take + 1, dtype=np.int64)
    return [int(rng.integers(int(edges[i]), int(edges[i + 1]))) * int(ctx) for i in range(take)]


def overlaps(a, b, ctx=CTX) -> list:
    bs = sorted(int(x) for x in b)
    return sorted({int(s) for s in a for o in bs if not (s + ctx <= o or o + ctx <= s)})


def round7_exclusions() -> dict:
    """Every Qwen-stream TRAIN window any earlier round touched (scout 3b), by source."""
    src = {}

    def add(name, starts):
        src[name] = sorted(int(s) for s in starts)
    for f in sorted(glob.glob(str(KB / "prc_local_20260926/*/windows.json"))
                    + glob.glob(str(KB / "prc_adjacent_20260927/*/windows.json"))):
        w = read_json(f)
        if w.get("token_ids_sha256") != QWEN_TOKEN_SHA:
            continue
        rel = f.replace(str(KB) + "/", "")
        add(rel + ":search+confirm", w["starts"]["search"] + w["starts"]["confirm"])
        add(rel + ":excluded", w["excluded"])
    for f in sorted(glob.glob(str(KB / "prc2/*_heldout_nll.json"))):
        if Path(f).name.startswith("llama8B"):
            continue  # Llama tokenizer: a different token stream
        add("prc2 heldout " + Path(f).name, read_json(f)["windows"])
    for n in (6, 8, 12, 14, 16, 24):
        add(f"stratified seed0 n={n}", stratified_starts(n, 0))
    add("heldout seed101 superset n=24", stratified_starts(24, 101))
    m6 = read_json(R6_DIAG_MANIFEST)
    for cid, c in m6["cells"].items():
        add(f"r6 diag {cid}", c["windows"]["starts"])
        if c["identity"]["type"] == "probe":
            add(f"r6 probe {cid}", [c["identity"]["start"]])
    return src


def derive_round7_windows() -> dict:
    """Deterministic round-7 window registry: exclusion set + prefix guard, then
    choose_disjoint_starts(2518423, 2048, 16, 32, excl, seed=260928): 16 fresh screen
    windows (30B t48) and 32 fresh confirmation windows (all cells)."""
    src = round7_exclusions()
    excl = sorted(set(s for v in src.values() for s in v))
    guard = list(range(PREFIX_GUARD[0], PREFIX_GUARD[1], CTX))
    screen, confirm = choose_disjoint_starts(QWEN_NTOK, CTX, 16, 32, sorted(set(excl) | set(guard)),
                                             WINDOW_DRAW_SEED)
    if overlaps(screen + confirm, excl + guard):
        raise AssertionError("drawn windows overlap the exclusion set")
    m6 = read_json(R6_DIAG_MANIFEST)
    return {"schema": "prc-windows-r7-v1", "token_ids_sha256": QWEN_TOKEN_SHA, "n_tokens": QWEN_NTOK,
            "ctx": CTX, "split": "train", "exclusion_sources": {k: len(v) for k, v in src.items()},
            "excluded": excl, "prefix_guard": list(PREFIX_GUARD), "draw_seed": WINDOW_DRAW_SEED,
            "draw_call": "prc_local_refine.choose_disjoint_starts(2518423, 2048, 16, 32, excluded U "
                         "prefix-guard starts, 260928)",
            "screen16_fresh_t48": screen, "confirm32": confirm,
            "r6_diag_windows_30B": m6["cells"]["30B_t32"]["windows"]["starts"],
            "r6_diag_windows_4B": m6["cells"]["4B_t40"]["windows"]["starts"],
            "capture_windows_expected": sorted(stratified_starts(8, 0)),
            "note": "The round-7 capture (stratified seed 0, 6 calib + 2 hold = n 8) is inside the "
                    "exclusion set; any other capture window set must be checked with overlaps()."}


# ---------------------------------------------------------------------------
# evaluators
# ---------------------------------------------------------------------------
class Evaluator:
    """GPU evaluator: one model load per cell (built on ``build_wrapper``), tables swapped
    via loader.apply_mp_config_from_env, loss = model(ids, labels=ids).loss on enc[s:s+2048]
    of the WikiText-2 TRAIN join (same path as rounds 4/6), exact summary traces."""
    simulated = False

    def __init__(self, cell: dict, out, *, build_wrapper, tag="r7"):
        self.cell, self.out, self.build_wrapper, self.tag = cell, Path(out), str(build_wrapper), tag

    def build(self):
        from benchmark.ppl.prc_local_refine import configure_environment
        import torch
        from datasets import load_dataset
        from benchmark.quant.eval_quant import build_sc_model
        configure_environment({"incumbent_wrapper": self.build_wrapper,
                               "hybrid_config": self.cell["hybrid_config"],
                               "model_path": self.cell["model_path"]})
        for key in ("SC_MP_TRACE", "SC_MP_TRACE_MODE"):
            if os.environ.get(key):
                raise ValueError(f"{key} must be unset; the driver owns tracing")
        self.torch = torch
        self.model, tok = build_sc_model(self.cell["model_path"], "mp", mp_table=self.build_wrapper)
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        if self.dev.type != "cuda":
            raise ValueError("round-7 evaluation must run on its assigned Slurm GPU")
        cfg = self.model.config
        total = getattr(cfg, "_sc_total_blocks", None) or getattr(cfg, "num_hidden_layers", None)
        if int(total) != int(self.cell["total_blocks"]):
            raise ValueError(f"model has {total} blocks, manifest says {self.cell['total_blocks']}")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        return hashlib.sha256(self.enc.numpy().tobytes()).hexdigest(), int(self.enc.numel())

    def swap(self, wrapper_path):
        from loader import apply_mp_config_from_env
        from model.sc_common import SCLinear
        os.environ["MP_CONFIG_JSON"] = str(wrapper_path)
        self.model.config.sc_mp_config = None
        apply_mp_config_from_env(self.model)
        mp = self.model.config.sc_mp_config
        mods = [m for m in self.model.modules() if isinstance(m, SCLinear)]
        if mp is None or not mods or any(m._sc_config.sc_mp_config is not mp for m in mods):
            raise ValueError(f"table swap did not reach every SCLinear: {wrapper_path}")
        if Path(mp.threshold_table_path).resolve() != resolve_table_path(wrapper_path).resolve():
            raise ValueError(f"loaded MP config does not point at the wrapper's table: {wrapper_path}")

    def evaluate(self, name, wrapper_path, starts, *, profile="none", expected=None, header_extra=None):
        """Returns {window_nll, seconds, snapshot, first_trace, trace}. ``profile`` in
        {"none", "first", "all"}; ``expected`` = per-window NLLs that must match bit for bit."""
        torch = self.torch
        from benchmark.ppl.prc_local_proposals import ProfileCollector
        from model.sc_common import mp_tracker_reset
        from scmp_kernels import trace
        if profile not in ("none", "first", "all"):
            raise ValueError(profile)
        self.swap(wrapper_path)
        trace_path = self.out / f"{name}_trace.json"
        if trace_path.exists():
            raise ValueError(f"refusing to overwrite {trace_path}")
        hdr = dict(header_extra or {})
        trace.reset()
        trace.enable(str(trace_path), mode="summary")
        mp_tracker_reset()
        losses, snapshot, first_trace = [], None, None
        collector_all = ProfileCollector(self.model) if profile == "all" else None
        if collector_all is not None:
            collector_all.__enter__()
        started = time.time()
        try:
            for i, start in enumerate(starts):
                ids = self.enc[int(start):int(start) + CTX].unsqueeze(0).to(self.dev)
                if ids.numel() != CTX:
                    raise ValueError(f"window {start} is shorter than ctx")
                if profile == "first" and i == 0:
                    collector = ProfileCollector(self.model)
                    with collector:
                        with torch.no_grad():
                            loss = float(self.model(input_ids=ids, labels=ids).loss)
                    snapshot = collector.snapshot()
                    first_trace = self.out / f"{name}_first_window_trace.json"
                    trace.flush(path=str(first_trace), reset_after=False,
                                header_extra={**hdr, "stage": name + "_first_window", "windows": [int(start)]})
                else:
                    with torch.no_grad():
                        loss = float(self.model(input_ids=ids, labels=ids).loss)
                if not math.isfinite(loss):
                    raise ValueError(f"nonfinite NLL in {name} window {start}")
                losses.append(loss)
                print(f"[{self.tag}] {self.cell['id']} {name} window {i + 1}/{len(starts)} "
                      f"start={start} NLL={loss!r}", flush=True)
                if expected is not None and loss != expected[i]:
                    raise IdentityError(f"{self.cell['id']} {name} window {start}: NLL {loss!r} != "
                                        f"recorded {expected[i]!r}")
        finally:
            if collector_all is not None:
                collector_all.__exit__(None, None, None)
        if collector_all is not None:
            snapshot = collector_all.snapshot()
        trace.flush(header_extra={**hdr, "stage": name, "windows": [int(s) for s in starts]}, reset_after=True)
        trace.disable()
        return {"window_nll": losses, "seconds": time.time() - started, "snapshot": snapshot,
                "first_trace": first_trace, "trace": trace_path}

    def close(self):
        for attr in ("model", "enc"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def synthesize_trace(inc_payload: dict, inc_table: dict, inc_wrapper: dict, arm_table: dict,
                     arm_wrapper: dict, total_blocks: int, *, prc_shift=0, header=None) -> dict:
    """SIMULATION ONLY: re-map an INC summary trace onto an arm's OWN ladders (rung index
    preserved, clamped; PRC rung index shifted down by ``prc_shift``; protected slice ->
    arm psl). MACs are preserved exactly; every length is realizable under the arm's table.
    Never a measurement."""
    nb = int(total_blocks)
    wi, psl_i = protected_widths(inc_table)
    _wa, psl_a = protected_widths(arm_table)
    fa = loader_fields(arm_wrapper)
    merged = {}
    cache = {}
    for grp in inc_payload["groups"]:
        g = dict(grp)
        op, blk, L = g["op"], g["block"], int(g["stoc_len"])
        if blk is not None and op in ATTN_OPS:
            key = ("a", op, int(blk))
            if key not in cache:
                cache[key] = (json_values(inc_table, inc_wrapper, op, int(blk), nb),
                              json_ladder(inc_table, op, int(blk), nb),
                              json_ladder(arm_table, op, int(blk), nb),
                              json_realizable(arm_table, arm_wrapper, op, int(blk), nb))
            vi, li, la, real = cache[key]
            idx = vi.index(L)  # an INC length outside INC's own values = invalid stand-in
            if idx < len(li):
                newL = la[min(idx, len(la) - 1)]
            else:
                newL = fa["escape_stoc_len"] if fa["escape_stoc_len"] in real else la[0]
            g["stoc_len"] = int(newL)
        elif blk is not None and op in LINEAR_OPS:
            w = wi.get((op, int(blk), g.get("unit"))) or wi.get((op, int(blk), None))
            if w is not None and psl_i is not None and int(g["d_in"]) == w and L == psl_i:
                g["stoc_len"] = int(psl_a)
            else:
                pi = json_prc_entry(inc_table, op, int(blk), nb)
                pa = json_prc_entry(arm_table, op, int(blk), nb)
                if pa is not None and pi is not None and L in pi[0]:
                    r = max(0, min(len(pa[0]) - 1, pi[0].index(L) - int(prc_shift)))
                    g["stoc_len"] = int(pa[0][r])
                else:
                    real = sorted(json_realizable(arm_table, arm_wrapper, op, int(blk), nb))
                    g["stoc_len"] = int(min(real, key=lambda v: (abs(v - L), v)))
        g["row_cycles"] = int(g["rows"]) * int(g["stoc_len"])
        k = tuple(g[x] for x in ("block", "op", "unit", "stoc_len", "sc_prec", "halve", "mode",
                                 "granularity", "rng_levels", "chunk_d", "smoothed", "d_in", "d_out"))
        if k in merged:
            for x in ("calls", "rows", "macs", "row_cycles"):
                merged[k][x] += g[x]
        else:
            merged[k] = g
    hdr = dict(inc_payload.get("header", {}))
    hdr.update(header or {})
    hdr["simulated_from"] = "synthesize_trace (INC trace re-mapped onto the arm ladders; NOT a measurement)"
    return {"schema": inc_payload.get("schema", "scmp-trace-summary-v1"), "header": hdr,
            "groups": list(merged.values())}


class SimEvaluator:
    """CPU dry-run evaluator: archived INC / parent traces stand in for GPU traces, other
    arms are synthesized onto their own ladders, NLLs are synthetic. Never writes under
    the Turbo round-7 root."""
    simulated = True

    def __init__(self, cell: dict, out, *, inc_wrapper, inc_standin, nll_ref: dict, token_sha,
                 parent_wrapper=None, parent_standin=None, deltas=None, prc_shift=None, nll_exact=None):
        self.cell, self.out = cell, Path(out)
        if str(self.out.resolve()).startswith(str(R7_ROOT)):
            raise ValueError("simulation must not write under the Turbo round-7 root")
        self.inc_wrapper, self.parent_wrapper = str(inc_wrapper), str(parent_wrapper) if parent_wrapper else None
        self.inc_standin, self.parent_standin = inc_standin, parent_standin
        self.nll_ref, self.token_sha = dict(nll_ref), token_sha
        self.deltas, self.prc_shift = dict(deltas or {}), dict(prc_shift or {})
        # (wrapper, start) -> exact NLL (identity probes of non-INC tables)
        self.nll_exact = {(str(w), int(s)): v for (w, s), v in (nll_exact or {}).items()}
        self._payloads = {}

    def build(self):
        self.inc_table = read_json(resolve_table_path(self.inc_wrapper))
        self.inc_w = read_json(self.inc_wrapper)
        return self.token_sha, None

    def _payload(self, path):
        if path not in self._payloads:
            self._payloads[path] = load_trace(path)
        return self._payloads[path]

    def evaluate(self, name, wrapper_path, starts, *, profile="none", expected=None, header_extra=None):
        trace_path = self.out / f"{name}_trace.json"
        if trace_path.exists():
            raise ValueError(f"refusing to overwrite {trace_path}")
        hdr = {**(header_extra or {}), "stage": name, "windows": [int(s) for s in starts]}
        w = str(wrapper_path)
        if w == self.inc_wrapper:
            src = self._payload(self.inc_standin)
            payload = {"schema": src.get("schema"), "header": {**src["header"], **hdr}, "groups": src["groups"]}
        elif self.parent_wrapper and w == self.parent_wrapper and self.parent_standin:
            src = self._payload(self.parent_standin)
            payload = {"schema": src.get("schema"), "header": {**src["header"], **hdr}, "groups": src["groups"]}
        else:
            payload = synthesize_trace(self._payload(self.inc_standin), self.inc_table, self.inc_w,
                                       read_json(resolve_table_path(w)), read_json(w),
                                       self.cell["total_blocks"], prc_shift=self.prc_shift.get(w, 0), header=hdr)
        with trace_path.open("w") as f:
            json.dump(payload, f, separators=(",", ":"))
        first_trace = None
        if profile == "first":
            first_trace = self.out / f"{name}_first_window_trace.json"
            p1 = dict(payload)
            p1["header"] = {**payload["header"], "stage": name + "_first_window", "windows": [int(starts[0])]}
            with first_trace.open("w") as f:
                json.dump(p1, f, separators=(",", ":"))
        snapshot = None
        if profile in ("first", "all"):
            tm = sum(int(g["macs"]) for g in payload["groups"])
            tc = sum(int(g["macs"]) * int(g["stoc_len"]) for g in payload["groups"])
            snapshot = {"version": 1, "bins": 0, "groups": {}, "fixed_macs": float(tm),
                        "fixed_cycle_macs": float(tc), "total_macs": float(tm), "total_cycle_macs": float(tc),
                        "actual_mean_length": tc / tm, "simulated": True}
        losses = []
        delta = self.deltas.get(w, 0.0)
        for i, s in enumerate(starts):
            if (w, int(s)) in self.nll_exact:
                loss = self.nll_exact[(w, int(s))]
            else:
                base = self.nll_ref.get(int(s), 2.0 + 0.1 * math.sin(int(s) / 7919.0))
                loss = base + delta * (1.0 + 0.3 * math.cos(int(s) / 104729.0))
            losses.append(loss)
            if expected is not None and loss != expected[i]:
                raise IdentityError(f"simulated {name} window {s}: {loss!r} != {expected[i]!r}")
        del payload
        gc.collect()
        return {"window_nll": losses, "seconds": 0.0, "snapshot": snapshot, "first_trace": first_trace,
                "trace": trace_path}

    def close(self):
        self._payloads.clear()
        gc.collect()


def profile_checks(snapshot, first_trace_payload_or_totals, table) -> dict:
    """Profiler totals == trace totals and the table's own ladders/thresholds replay the
    dispatched lengths exactly (baseline_cost uses the JSON per-bucket ladder)."""
    from benchmark.ppl.prc_local_proposals import baseline_cost, exact_profile_cost
    tm, tc = first_trace_payload_or_totals
    replay = baseline_cost(table, snapshot)
    exact = exact_profile_cost(snapshot)
    checks = {"trace_total_macs": tm, "trace_total_cycle_macs": tc,
              "profile_total_macs": snapshot["total_macs"], "profile_total_cycle_macs": snapshot["total_cycle_macs"],
              "profile_exact_cost": exact, "profile_table_replay_cost": replay,
              "exact_integer_comparison": bool(max(tm, tc) < EXACT_INT_LIMIT)}
    if not (_totals_agree(snapshot["total_macs"], tm) and _totals_agree(snapshot["total_cycle_macs"], tc)):
        raise AuditError(f"profiler totals disagree with the trace: {checks}")
    if max(tm, tc) < EXACT_INT_LIMIT:
        ok = replay == exact
    else:
        ok = abs(replay / exact - 1) <= PROFILE_TOL
    if not ok:
        raise AuditError(f"table ladders/thresholds do not replay the dispatched lengths: {checks}")
    return checks


def _totals_agree(profile_total, trace_total) -> bool:
    """Integer totals: exact below 2**53 (float64 holds them exactly), else relative PROFILE_TOL."""
    if max(abs(float(profile_total)), abs(float(trace_total))) < EXACT_INT_LIMIT:
        return float(profile_total) == float(trace_total)
    return abs(float(profile_total) / float(trace_total) - 1) <= PROFILE_TOL


def trace_totals(path_or_payload):
    payload = load_trace(path_or_payload) if not isinstance(path_or_payload, dict) else path_or_payload
    tm = sum(int(g["macs"]) for g in payload["groups"])
    tc = sum(int(g["macs"]) * int(g["stoc_len"]) for g in payload["groups"])
    return tm, tc


def configure_protocol_env_check() -> dict:
    """Refuse forbidden inherited variants (RNG grid, QK smooth, row-shared, >64 masks)."""
    bad = {k: os.environ[k] for k in ("SC_RNG_GRID", "SC_RNG_GRID_ATTN", "SC_RNG_GRID_QK", "SC_RNG_GRID_AV",
                                     "SC_ATTN_SMOOTH_JSON", "SC_HYBRID_FORCE_INT_BITS") if os.environ.get(k)}
    if os.environ.get("SC_PRC_ROWSHARED", "0") != "0":
        bad["SC_PRC_ROWSHARED"] = os.environ["SC_PRC_ROWSHARED"]
    if os.environ.get("SC_HW_MAX_MASKS", "64") not in ("", "64"):
        bad["SC_HW_MAX_MASKS"] = os.environ["SC_HW_MAX_MASKS"]
    for key, value in (("SC_OWEN_MODE", "bitrev"), ("SC_SCRAMBLE_MASKS", "64")):
        if os.environ.get(key, value) != value:
            bad[key] = os.environ[key]
    if bad:
        raise ValueError(f"forbidden numerical-protocol variants in the environment: {bad}")
    return {k: os.environ.get(k) for k in ("SC_OWEN_MODE", "SC_SCRAMBLE_MASKS", "SC_HW_MAX_MASKS")}


PROTOCOL = {"frontend": "awq", "owen_mode": "bitrev", "scramble_masks": 64, "hw_max_masks": 64,
            "ctx": 2048, "stride": 2048, "split": "train", "sc_prec": 8, "sc_halve": True,
            "qk_rebalance": False, "rng_grid": "fixed128", "awq_obj_bits": 4, "sq_alpha": 0.5,
            "escape": "mu+2tau gate (escape_gate_k 2.0, escape_stoc_len 128) unchanged in every arm",
            "max_code_length": CAP}


def check_protocol(p: dict) -> None:
    for k, v in PROTOCOL.items():
        if p.get(k) != v or type(p.get(k)) is not type(v):
            raise ValueError(f"fixed protocol mismatch in manifest: {k}")
