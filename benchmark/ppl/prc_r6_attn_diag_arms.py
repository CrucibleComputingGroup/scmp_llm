"""Round-6 attention diagnostic: arm tables, validator, trace audit, decision rule.

All stream lengths are HALVED code units (nominal = 2x); the code cap is 128
(sc_prec=8, halve on). Arms are pure table edits inside existing runtime
support, applied to the exact best(all) incumbent of a cell:

  INC     byte-identical copy of the incumbent table.
  A       the listed qk/av runtime buckets get a per-bucket ladder
          [128] + global[1:] (the top rung raised to the cap). Thresholds,
          escape constants, linears and protected channels are unchanged, so
          every non-escaped group keeps its rung index and only index-0 rows
          change length; escaped rows already ran at 128 and fold onto index 0.
  ATT128  every qk/av bucket gets ladder [128] + global[1:] and all-zero
          thresholds, so every SC attention row runs at 128. Attribution only.
  LIN128  every per_row_chunk ladder becomes [128]*n (thresholds unchanged)
          and the protected-slice length becomes 128. Attribution only.

No RNG, QK operand, smooth-scale, escape-semantics, INT-mask, AWQ or runtime
metadata change. The top-level ``stoc_len_levels`` (which must equal the
wrapper's) is never edited. Module import is torch-free; only
``resolution_sweep`` loads the runtime config class.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ATTN_OPS = ("qk", "av")
LINEAR_OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Dataflow stage inside one decoder block. Inputs of every op at (block, stage)
# depend only on positions strictly before it, so records upstream of an arm's
# first edited position must be identical to the incumbent's.
STAGE = {"q_proj": 0, "k_proj": 0, "v_proj": 0, "qk": 1, "av": 2, "o_proj": 3,
         "gate_proj": 4, "up_proj": 4, "down_proj": 5}
ARMS = ("INC", "A", "ATT128", "LIN128")
EDIT_ARMS = ("A", "ATT128", "LIN128")
ATTRIBUTION_ONLY_ARMS = ("ATT128", "LIN128")
META_KEY = "r6_attn_diag_arm"
ROUND_TAG = "prc_r6_attn_diag_20260928"
CAP = 128
FIXED_NUMERICS = {"sc_prec": 8, "halve": True, "rng_levels": 128,
                  "mode": "bipolar", "granularity": "per_row"}

# Pre-registered Round-7 gate constants (see round7_gate).
GATE_MARGINAL_FRACTION = 0.6
GATE_SE_MULTIPLIER = 1.5
GATE_GROSS_DNLL = {"30B": -0.01, "4B": -0.004}
BUDGETS = (32, 40, 48, 64, 96)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj) -> str:
    """Order-independent serialization; NaN-safe equality for table payloads."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def bucket_index(value: int, total: int, num_buckets: int) -> int:
    """Mirror of scmp_kernels.mp.config._bucket_index (unit-tested for equality)."""
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


def parse_bucket_key(key: str):
    op, t_part, l_part = key.split(":")
    if not (t_part.startswith("t") and l_part.startswith("l")):
        raise ValueError(f"bad bucket key {key!r}")
    return op, int(t_part[1:]), int(l_part[1:])


def global_ladder(table: dict) -> list:
    levels = [int(v) for v in table["stoc_len_levels"]]
    if len(levels) < 2 or any(b >= a for a, b in zip(levels, levels[1:])):
        raise ValueError(f"global ladder must be strictly descending with >=2 rungs: {levels}")
    return levels


def table_cap(table: dict) -> int:
    sc_prec = int(table.get("sc_prec", 8))
    halve = bool(table.get("halve_bipolar_stoc_len", True))
    cap = 2 ** (sc_prec - 1) if halve else 2 ** sc_prec
    if cap != CAP:
        raise ValueError(f"round-6 arms assume sc_prec=8 with halving (cap {CAP}); got cap {cap}")
    return cap


def raised_ladder(table: dict) -> list:
    g = global_ladder(table)
    if g[0] >= CAP:
        raise ValueError(f"global top {g[0]} is already at the {CAP} cap; raising it is a no-op")
    return [CAP] + g[1:]


def attention_bucket_keys(table: dict) -> list:
    return sorted(k for k in table.get("buckets", {}) if k.split(":")[0] in ATTN_OPS)


def check_incumbent_shape(table: dict) -> None:
    """Preconditions the arm definitions rely on (verified on all four incumbents)."""
    table_cap(table)
    global_ladder(table)
    lb = int(table.get("layer_buckets", 1))
    if int(table.get("timestep_buckets", 1)) != 1:
        raise ValueError("timestep bucketing is not supported by the round-6 arms")
    expected = sorted(f"{op}:t0:l{q}" for op in ATTN_OPS for q in range(lb))
    if attention_bucket_keys(table) != expected:
        raise ValueError(f"attention buckets {attention_bucket_keys(table)} != {expected}")
    for key, payload in table["buckets"].items():
        if "stoc_len_levels" in payload:
            raise ValueError(f"incumbent already has a per-bucket ladder: {key}")
    for key, payload in (table.get("operator_defaults") or {}).items():
        if "stoc_len_levels" in payload:
            raise ValueError(f"incumbent has an operator-default ladder: {key}")
    prc = table.get("per_row_chunk") or {}
    if not prc.get("buckets"):
        raise ValueError("incumbent has no per_row_chunk section")
    if int(prc.get("layer_buckets", 0) or 0) not in (0, lb):
        raise ValueError("per_row_chunk layer_buckets override is not supported here")
    for key, entry in prc["buckets"].items():
        levels = [int(v) for v in entry["levels"]]
        if levels != sorted(levels) or max(levels) > CAP or min(levels) <= 0:
            raise ValueError(f"unexpected incumbent PRC ladder at {key}: {levels}")
    if not (table.get("protected_channels") or {}).get("indices"):
        raise ValueError("incumbent has no protected channels")
    if META_KEY in table:
        raise ValueError("the base table is itself a round-6 arm table")


# ---------------------------------------------------------------------------
# arm construction and validation
# ---------------------------------------------------------------------------
def arm_spec_ok(arm: str, spec: dict, table: dict) -> None:
    if arm not in EDIT_ARMS:
        raise ValueError(f"unknown edit arm {arm!r}")
    if arm == "A":
        buckets = spec.get("buckets")
        if not isinstance(buckets, list) or not buckets or len(set(buckets)) != len(buckets):
            raise ValueError("arm A needs a non-empty explicit list of distinct attention buckets")
        known = set(attention_bucket_keys(table))
        for key in buckets:
            if key not in known:
                raise ValueError(f"arm A bucket {key!r} is not an attention bucket of this table")
        if set(spec) - {"buckets"}:
            raise ValueError(f"unexpected arm A fields: {sorted(set(spec) - {'buckets'})}")
    elif arm == "ATT128":
        if spec:
            raise ValueError("ATT128 takes no parameters")
    elif arm == "LIN128":
        if spec != {"protected": True}:
            raise ValueError("LIN128 is defined as all PRC ladders AND the protected slice at 128")


def build_arm_table(base: dict, arm: str, spec: dict, provenance: dict) -> dict:
    """Return the arm's table: a deep copy of ``base`` with exactly the arm's edits."""
    check_incumbent_shape(base)
    arm_spec_ok(arm, spec, base)
    table = copy.deepcopy(base)
    top = raised_ladder(base)
    n = len(top)
    if arm == "A":
        for key in spec["buckets"]:
            table["buckets"][key]["stoc_len_levels"] = list(top)
    elif arm == "ATT128":
        for key in attention_bucket_keys(base):
            table["buckets"][key]["stoc_len_levels"] = list(top)
            table["buckets"][key]["thresholds"] = [0.0] * (n - 1)
    elif arm == "LIN128":
        for entry in table["per_row_chunk"]["buckets"].values():
            entry["levels"] = [CAP] * len(entry["levels"])
        table["protected_channels"]["stoc_len"] = CAP
    table[META_KEY] = {"round": ROUND_TAG, "arm": arm, "spec": copy.deepcopy(spec),
                       **copy.deepcopy(provenance)}
    validate_arm_table(base, table, arm, spec)
    return table


def validate_arm_table(base: dict, cand: dict, arm: str, spec: dict) -> dict:
    """Candidate must equal ``base`` after undoing exactly the arm's edits.

    Also enforces: top-level ladder unchanged, every ladder <= 128, bucket
    ladders strictly descending with >= 2 rungs, PRC ladders ascending
    (non-decreasing), threshold shapes/ranges valid. Returns a small report.
    """
    check_incumbent_shape(base)
    arm_spec_ok(arm, spec, base)
    if META_KEY not in cand or cand[META_KEY].get("arm") != arm \
            or cand[META_KEY].get("spec") != spec:
        raise ValueError(f"arm metadata missing or inconsistent for {arm}")
    if [int(v) for v in cand["stoc_len_levels"]] != global_ladder(base):
        raise ValueError("top-level stoc_len_levels changed")
    top = raised_ladder(base)
    g = global_ladder(base)
    norm = copy.deepcopy(cand)
    norm.pop(META_KEY)
    edited_att = (set(spec["buckets"]) if arm == "A" else
                  set(attention_bucket_keys(base)) if arm == "ATT128" else set())
    for key, payload in norm.get("buckets", {}).items():
        ladder = payload.get("stoc_len_levels")
        if key in edited_att:
            if ladder != top:
                raise ValueError(f"{arm}: bucket {key} ladder {ladder} != {top}")
            if arm == "ATT128":
                th = payload.get("thresholds")
                if th != [0.0] * (len(top) - 1):
                    raise ValueError(f"ATT128: bucket {key} thresholds must be all zero")
                payload["thresholds"] = copy.deepcopy(base["buckets"][key]["thresholds"])
            del payload["stoc_len_levels"]
        elif ladder is not None:
            raise ValueError(f"{arm}: unlisted bucket {key} carries a ladder {ladder}")
    for key, payload in cand.get("buckets", {}).items():
        ladder = payload.get("stoc_len_levels")
        if ladder is not None:
            ladder = [int(v) for v in ladder]
            if len(ladder) < 2 or any(b >= a for a, b in zip(ladder, ladder[1:])):
                raise ValueError(f"bucket ladder not strictly descending: {key} {ladder}")
            if max(ladder) > CAP:
                raise ValueError(f"bucket ladder exceeds the {CAP} cap: {key} {ladder}")
            th = [float(v) for v in payload["thresholds"]]
            if len(th) != len(ladder) - 1 or any(not 0.0 <= v <= 1.0 for v in th) \
                    or any(b > a + 1e-6 for a, b in zip(th, th[1:])):
                raise ValueError(f"invalid thresholds for bucket ladder: {key}")
    prc_after = norm["per_row_chunk"]["buckets"]
    if prc_after.keys() != base["per_row_chunk"]["buckets"].keys():
        raise ValueError("per_row_chunk bucket keys changed")
    for key, entry in cand["per_row_chunk"]["buckets"].items():
        levels = [int(v) for v in entry["levels"]]
        if levels != sorted(levels) or max(levels) > CAP or min(levels) <= 0 or len(levels) < 2:
            raise ValueError(f"invalid PRC ladder at {key}: {levels}")
    if arm == "LIN128":
        for key, entry in prc_after.items():
            if entry["levels"] != [CAP] * len(base["per_row_chunk"]["buckets"][key]["levels"]):
                raise ValueError(f"LIN128: PRC ladder {key} is not all-{CAP}")
            entry["levels"] = copy.deepcopy(base["per_row_chunk"]["buckets"][key]["levels"])
        if norm["protected_channels"].get("stoc_len") != CAP:
            raise ValueError("LIN128: protected slice length must be 128")
        norm["protected_channels"]["stoc_len"] = base["protected_channels"]["stoc_len"]
    if canonical(norm) != canonical(base):
        raise ValueError(f"{arm}: table differs from the incumbent beyond the arm's edits")
    if canonical({k: v for k, v in cand.items() if k != META_KEY}) == canonical(base):
        raise ValueError(f"{arm}: table is unchanged")
    return {"arm": arm, "edited_attention_buckets": sorted(edited_att),
            "top_before": g[0], "top_after": top[0] if edited_att else g[0],
            "lin128": arm == "LIN128"}


def expected_attention_ladder(arm, spec, table, key):
    g = global_ladder(table)
    if arm == "ATT128" or (arm == "A" and key in spec["buckets"]):
        return [CAP] + g[1:]
    return g


def _runtime_cfg(wrapper: dict, table_path):
    from scmp_kernels.mp.config import AdaptiveMPConfig
    esc_k = wrapper.get("escape_gate_k")
    return AdaptiveMPConfig(stoc_len_levels=[int(v) for v in wrapper["stoc_len_levels"]],
                            threshold_table_path=str(table_path),
                            escape_gate_k=None if esc_k is None else float(esc_k),
                            escape_stoc_len=int(wrapper.get("escape_stoc_len", 128)))


def resolution_sweep(wrapper: dict, table_path, arm: str, spec: dict, base: dict,
                     base_table_path, total_blocks: int, synthetic_rows: int = 2048) -> dict:
    """Load the arm exactly as the runtime does and check every (op, block) resolves
    to the intended ladder/thresholds/escape through BOTH resolvers
    (get_levels and classify_level_values), plus a synthetic dispatch check against
    the incumbent loaded from ``base_table_path``. Imports torch (CPU only)."""
    import torch
    from scmp_kernels.mp.config import _bucket_index, adaptive_classify_rows
    cfg = _runtime_cfg(wrapper, table_path)
    inc_cfg = _runtime_cfg(wrapper, base_table_path)
    g = global_ladder(base)
    lb = int(base.get("layer_buckets", 1))
    gen = torch.Generator().manual_seed(20260928)
    checked, changed_rows = 0, 0
    for op in ATTN_OPS:
        for block in range(total_blocks):
            q = bucket_index(block, total_blocks, lb)
            if q != _bucket_index(block, total_blocks, lb):
                raise AssertionError("bucket_index mirror disagrees with runtime")
            key = f"{op}:t0:l{q}"
            want = expected_attention_ladder(arm, spec, base, key)
            got = cfg.get_levels(operator=op, block_idx=block, total_blocks=total_blocks)
            if got != want:
                raise ValueError(f"get_levels({op}, {block}) = {got}, expected {want}")
            values = cfg.classify_level_values(operator=op, block_idx=block, total_blocks=total_blocks)
            want_values = list(want) if (cfg.escape_gate_k is None or int(cfg.escape_stoc_len) in want) \
                else list(want) + [int(cfg.escape_stoc_len)]
            if values != want_values:
                raise ValueError(f"classify_level_values({op}, {block}) = {values}, expected {want_values}")
            th = cfg.get_thresholds(0, 1, operator=op, block_idx=block, total_blocks=total_blocks)
            want_th = ([0.0] * (len(want) - 1) if arm == "ATT128"
                       else [float(v) for v in base["buckets"][key]["thresholds"]])
            if [float(v) for v in th] != want_th:
                raise ValueError(f"thresholds changed for {key} under {arm}")
            esc = cfg.get_escape_threshold(0, 1, operator=op, block_idx=block, total_blocks=total_blocks)
            esc_inc = inc_cfg.get_escape_threshold(0, 1, operator=op, block_idx=block,
                                                   total_blocks=total_blocks)
            if esc != esc_inc:
                raise ValueError(f"escape threshold changed for {key}: {esc} vs {esc_inc}")
            # Synthetic dispatch: identical indices except escaped rows (folded to
            # index 0 at the same length 128); ATT128 puts every row at 128.
            metric = torch.rand(synthetic_rows, generator=gen) ** 3
            a_inc = adaptive_classify_rows(metric, inc_cfg, operator=op, block_idx=block,
                                           total_blocks=total_blocks)
            a_arm = adaptive_classify_rows(metric, cfg, operator=op, block_idx=block,
                                           total_blocks=total_blocks)
            inc_vals = inc_cfg.classify_level_values(operator=op, block_idx=block,
                                                     total_blocks=total_blocks)
            L_inc = torch.tensor(inc_vals)[a_inc.row_levels]
            L_arm = torch.tensor(values)[a_arm.row_levels]
            if arm == "ATT128":
                if not bool((L_arm == CAP).all()):
                    raise ValueError(f"ATT128 synthetic dispatch not all at 128 ({op}, {block})")
            else:
                mapped = L_inc.clone()
                if want[0] == CAP and g[0] != CAP:
                    mapped[L_inc == g[0]] = CAP
                if not torch.equal(mapped, L_arm):
                    raise ValueError(f"{arm} synthetic dispatch is not the intended length map ({op}, {block})")
            changed_rows += int((L_arm != L_inc).sum())
            checked += 1
    prc_checked = 0
    for op in LINEAR_OPS:
        for block in range(total_blocks):
            key = f"{op}:t0:l{bucket_index(block, total_blocks, lb)}"
            got = cfg.get_per_row_chunk(op, block, total_blocks)
            entry = base["per_row_chunk"]["buckets"].get(key)
            if entry is None:
                if got is not None:
                    raise ValueError(f"unexpected PRC entry for {key}")
                continue
            want_levels = ([CAP] * len(entry["levels"]) if arm == "LIN128"
                           else [int(v) for v in entry["levels"]])
            want = (want_levels, [float(v) for v in entry["thresholds"]])
            if got is None or (list(got[0]), [float(v) for v in got[1]]) != want:
                raise ValueError(f"get_per_row_chunk({op}, {block}) = {got}, expected {want}")
            if inc_cfg.get_per_row_chunk(op, block, total_blocks)[1] != got[1]:
                raise ValueError(f"PRC thresholds differ from the incumbent at {key}")
            prc_checked += 1
    psl = cfg.protected_channel_stoc_len
    want_psl = CAP if arm == "LIN128" else int(base["protected_channels"]["stoc_len"])
    if psl != want_psl or inc_cfg.protected_channel_stoc_len != int(base["protected_channels"]["stoc_len"]):
        raise ValueError(f"protected slice length {psl} != {want_psl}")
    if cfg.protected_channel_indices != inc_cfg.protected_channel_indices:
        raise ValueError("protected channel indices changed")
    if [int(v) for v in cfg.stoc_len_levels] != g or cfg.dispatch_metrics != inc_cfg.dispatch_metrics:
        raise ValueError("runtime global ladder or dispatch metrics changed")
    return {"arm": arm, "attention_op_blocks_checked": checked,
            "prc_op_blocks_checked": prc_checked, "protected_stoc_len": psl,
            "synthetic_rows_changed": changed_rows}


# ---------------------------------------------------------------------------
# trace aggregation and realized-length audit
# ---------------------------------------------------------------------------
class TraceAgg:
    """Exact integer aggregates of one scmp summary trace."""

    def __init__(self, payload: dict, total_blocks: int, layer_buckets: int, source=None):
        self.source = str(source) if source is not None else None
        self.header = payload.get("header", {})
        self.total_blocks, self.layer_buckets = int(total_blocks), int(layer_buckets)
        self.records = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.bucket_hist = defaultdict(lambda: defaultdict(int))
        self.op_block_macs = defaultdict(int)
        self.numerics = set()
        self.dims_vary = 0
        self.unknown_ops = set()
        self.total_macs = 0
        self.total_cycle_macs = 0
        for g in payload["groups"]:
            op, block = g["op"], g["block"]
            if op not in STAGE or block is None:
                self.unknown_ops.add(str(op))
                continue
            block, L, macs, rows = int(block), int(g["stoc_len"]), int(g["macs"]), int(g["rows"])
            self.numerics.add((int(g["sc_prec"]), bool(g["halve"]), int(g["rng_levels"]),
                               str(g["mode"]), str(g["granularity"])))
            if g.get("dims_vary"):
                self.dims_vary += 1
            key = (block, op, g.get("unit"), int(g["d_in"]), int(g["d_out"]))
            rec = self.records[key][L]
            rec[0] += rows
            rec[1] += macs
            kind = "attention" if op in ATTN_OPS else "linear"
            q = bucket_index(block, self.total_blocks, self.layer_buckets)
            self.bucket_hist[(kind, op, q)][L] += macs
            self.op_block_macs[(op, block)] += macs
            self.total_macs += macs
            self.total_cycle_macs += macs * L

    @classmethod
    def load(cls, path, total_blocks, layer_buckets):
        return cls(json.loads(Path(path).read_text()), total_blocks, layer_buckets, source=path)

    @property
    def cost(self) -> float:
        return self.total_cycle_macs / self.total_macs

    def op_blocks(self):
        return set(self.op_block_macs)

    def bucket_shares(self):
        out = {}
        for (kind, op, q), hist in sorted(self.bucket_hist.items()):
            tot = sum(hist.values())
            out[f"{op}:l{q}"] = {"kind": kind, "macs": tot,
                                 "share_by_len": {str(L): m / tot for L, m in sorted(hist.items())},
                                 "mean_len": sum(L * m for L, m in hist.items()) / tot}
        return out


def _edited(arm, spec, kind, op, q):
    if arm == "A":
        return kind == "attention" and f"{op}:t0:l{q}" in spec["buckets"]
    if arm == "ATT128":
        return kind == "attention"
    if arm == "LIN128":
        return kind == "linear"
    return False


def _length_map(arm, g_top):
    if arm == "A":
        return lambda L: CAP if L == g_top else L
    return lambda L: CAP


def first_edit_position(inc: TraceAgg, arm: str, spec: dict):
    best = None
    for (block, op, _unit, _di, _do) in inc.records:
        kind = "attention" if op in ATTN_OPS else "linear"
        q = bucket_index(block, inc.total_blocks, inc.layer_buckets)
        if _edited(arm, spec, kind, op, q):
            pos = (block, STAGE[op])
            best = pos if best is None or pos < best else best
    return best


def hybrid_sc_op_blocks(hybrid: dict, total_blocks: int):
    schedule = hybrid.get("schedule") or {}
    default = str(hybrid.get("default", "sc")).lower()
    out = set()
    for op in STAGE:
        row = schedule.get(op)
        for block in range(total_blocks):
            backend = (str(row[block]).lower() if row is not None and block < len(row) else default)
            if backend == "sc":
                out.add((op, block))
    return out


FIXED_NUMERICS_TUPLE = tuple(FIXED_NUMERICS[k] for k in
                             ("sc_prec", "halve", "rng_levels", "mode", "granularity"))


def audit_trace_basics(agg: TraceAgg, name: str, hybrid: dict | None = None) -> list:
    """Fixed SC numerics, no folded dims, known ops only, SC set == hybrid mask's SC set."""
    failures = []
    if agg.numerics != {FIXED_NUMERICS_TUPLE}:
        failures.append(f"{name} trace numerics {sorted(agg.numerics)} differ from the fixed protocol")
    if agg.dims_vary:
        failures.append(f"{name} trace has {agg.dims_vary} folded dims_vary groups")
    if agg.unknown_ops:
        failures.append(f"{name} trace has unknown ops {sorted(agg.unknown_ops)}")
    if agg.total_macs <= 0:
        failures.append(f"{name} trace is empty")
    if hybrid is not None and agg.op_blocks() != hybrid_sc_op_blocks(hybrid, agg.total_blocks):
        failures.append(f"{name} SC (op, block) set differs from the hybrid INT mask's SC entries")
    return failures


def allowed_lengths(arm: str, spec: dict, table: dict, kind: str, op: str, q: int,
                    escape_len) -> set:
    """Realizable lengths of one runtime bucket under ``arm`` (``table`` = incumbent)."""
    g = global_ladder(table)
    if kind == "attention":
        if arm == "ATT128":
            return {CAP}
        if _edited(arm, spec, kind, op, q):  # arm A listed bucket: [128] + g[1:]
            return {CAP} | set(g[1:])
        return set(g) | ({int(escape_len)} if escape_len else set())
    if arm == "LIN128":
        return {CAP}
    entry = table["per_row_chunk"]["buckets"].get(f"{op}:t0:l{q}")
    levels = set(int(v) for v in entry["levels"]) if entry else set()
    return levels | {int(table["protected_channels"]["stoc_len"])}


def audit_lengths(agg: TraceAgg, arm: str, spec: dict, table: dict, escape_len) -> list:
    failures = []
    for (kind, op, q), hist in sorted(agg.bucket_hist.items()):
        lengths = {L for L, m in hist.items() if m}
        extra = sorted(lengths - allowed_lengths(arm, spec, table, kind, op, q, escape_len))
        if extra:
            failures.append(f"{arm}: {op}:l{q} realized lengths {extra} outside the allowed set")
    return failures


def audit_arm_trace(inc: TraceAgg, arm_agg: TraceAgg, arm: str, spec: dict, table: dict,
                    *, escape_len, hybrid: dict | None = None, expected_windows=None,
                    expected_wrapper=None) -> dict:
    """Realized-length audit of one arm trace against the incumbent trace on the
    same windows. ``table`` is the INCUMBENT table (ladders, psl, PRC levels);
    ``escape_len`` is the wrapper's escape length (None when the gate is off).

    Checks (every failure is listed; ``ok`` is their conjunction):
      * fixed SC numerics (sc_prec 8, halve, 128-level grid, bipolar, per_row), no
        folded dims, no unknown ops, SC (op, block) set == the hybrid mask's;
      * same SC (op, block) set as the incumbent, equal per-(op, block) and total MACs;
      * allowed lengths per bucket (A: listed buckets on [128]+g[1:], hence zero MACs
        on the old top; unlisted as incumbent; ATT128: attention only at 128;
        LIN128: every linear record at 128; linears otherwise within PRC ladder U
        {psl}; attention otherwise within g U {escape});
      * records upstream of the first edited (block, stage) are identical;
      * at the first edited position, the arm histogram equals the incumbent
        histogram pushed through the arm's length map, per record key;
      * A: every unlisted bucket that held >= 1% of its MACs on the old top still
        has MACs on it (guards against a ladder applied to the wrong bucket).
    """
    if arm not in EDIT_ARMS:
        raise ValueError(f"audit_arm_trace needs an edit arm, got {arm!r}")
    g = global_ladder(table)
    g_top = g[0]
    failures = audit_trace_basics(inc, "incumbent", hybrid) + audit_trace_basics(arm_agg, "arm", hybrid)
    failures += audit_lengths(inc, "INC", {}, table, escape_len)
    failures += audit_lengths(arm_agg, arm, spec, table, escape_len)
    if expected_windows is not None and arm_agg.header.get("windows") != list(expected_windows):
        failures.append("arm trace header windows differ from the diagnostic windows")
    if expected_wrapper is not None and arm_agg.header.get("mp_config_json") != str(expected_wrapper):
        failures.append("arm trace header mp_config_json is not the arm wrapper")
    if inc.op_blocks() != arm_agg.op_blocks():
        failures.append("SC (op, block) set differs from the incumbent")
    if inc.total_macs != arm_agg.total_macs:
        failures.append(f"total MACs differ: {arm_agg.total_macs} vs {inc.total_macs}")
    bad_ob = [k for k in inc.op_block_macs if inc.op_block_macs[k] != arm_agg.op_block_macs.get(k)]
    if bad_ob:
        failures.append(f"per-(op, block) MACs differ at {len(bad_ob)} positions, e.g. {sorted(bad_ob)[:3]}")

    top_share_inc, top_share_arm = {}, {}
    if arm == "A":
        for (kind, op, q), hist in sorted(arm_agg.bucket_hist.items()):
            if kind != "attention":
                continue
            inc_hist = inc.bucket_hist.get((kind, op, q), {})
            tot_i, tot_a = sum(inc_hist.values()), sum(hist.values())
            share_i = inc_hist.get(g_top, 0) / tot_i if tot_i else 0.0
            top_share_inc[f"{op}:l{q}"] = share_i
            top_share_arm[f"{op}:l{q}"] = hist.get(g_top, 0) / tot_a if tot_a else 0.0
            if not _edited(arm, spec, kind, op, q) and share_i >= 0.01 and not hist.get(g_top, 0):
                failures.append(f"A: unlisted bucket {op}:l{q} lost every MAC on the old top {g_top}")

    # upstream identity and exact first-edit transform
    pstar = first_edit_position(inc, arm, spec)
    upstream_keys = first_keys = 0
    fmap = _length_map(arm, g_top)
    if pstar is None:
        failures.append(f"{arm}: no edited SC operator present in the incumbent trace")
    else:
        for key in set(inc.records) | set(arm_agg.records):
            block, op = key[0], key[1]
            pos = (block, STAGE[op])
            if pos > pstar:
                continue
            a = {L: tuple(v) for L, v in arm_agg.records.get(key, {}).items()}
            i = {L: tuple(v) for L, v in inc.records.get(key, {}).items()}
            if pos < pstar:
                upstream_keys += 1
                if a != i:
                    failures.append(f"upstream record changed at {key}")
                continue
            first_keys += 1
            kind = "attention" if op in ATTN_OPS else "linear"
            q = bucket_index(block, inc.total_blocks, inc.layer_buckets)
            if _edited(arm, spec, kind, op, q):
                pushed = defaultdict(lambda: [0, 0])
                for L, (rows, macs) in i.items():
                    pushed[fmap(L)][0] += rows
                    pushed[fmap(L)][1] += macs
                if a != {L: tuple(v) for L, v in pushed.items()}:
                    failures.append(f"first edited record {key} is not the intended length map of the incumbent")
            elif a != i:
                failures.append(f"unedited record at the first edited position changed: {key}")

    # cycle decomposition (pp of incumbent cost); direct + cascade == extra exactly
    decomposition = {}
    direct = cascade = predicted_direct = 0
    for bkey in sorted(set(inc.bucket_hist) | set(arm_agg.bucket_hist)):
        kind, op, q = bkey
        hi, ha = inc.bucket_hist.get(bkey, {}), arm_agg.bucket_hist.get(bkey, {})
        ci = sum(L * m for L, m in hi.items())
        ca = sum(L * m for L, m in ha.items())
        edited = _edited(arm, spec, kind, op, q)
        if edited:
            direct += ca - ci
            predicted_direct += sum((fmap(L) - L) * m for L, m in hi.items())
        else:
            cascade += ca - ci
        mi, ma = sum(hi.values()), sum(ha.values())
        decomposition[f"{op}:l{q}"] = {
            "kind": kind, "edited": edited, "macs": ma,
            "delta_cycles_pct_of_incumbent": 100.0 * (ca - ci) / inc.total_cycle_macs,
            "incumbent_mean_len": ci / mi if mi else None,
            "arm_mean_len": ca / ma if ma else None,
            "incumbent_share_by_len": {str(L): m / mi for L, m in sorted(hi.items())} if mi else {},
            "arm_share_by_len": {str(L): m / ma for L, m in sorted(ha.items())} if ma else {},
        }
    return {
        "ok": not failures, "failures": failures, "arm": arm,
        "first_edit_position": list(pstar) if pstar else None,
        "upstream_record_keys_checked": upstream_keys,
        "first_edit_record_keys_checked": first_keys,
        "incumbent_cost": inc.cost, "arm_cost": arm_agg.cost,
        "total_macs": arm_agg.total_macs,
        "incumbent_total_cycle_macs": inc.total_cycle_macs,
        "arm_total_cycle_macs": arm_agg.total_cycle_macs,
        "extra_cycles_pct": 100.0 * (arm_agg.total_cycle_macs - inc.total_cycle_macs) / inc.total_cycle_macs,
        "direct_cycles_pct": 100.0 * direct / inc.total_cycle_macs,
        "cascade_cycles_pct": 100.0 * cascade / inc.total_cycle_macs,
        "predicted_first_order_direct_pct": 100.0 * predicted_direct / inc.total_cycle_macs,
        "A_old_top_share": {"incumbent": top_share_inc, "arm": top_share_arm} if arm == "A" else None,
        "buckets": decomposition,
    }


def compare_trace_to_reference(agg: TraceAgg, ref: TraceAgg) -> list:
    """Exact equality of two traces' record aggregates (identity checks)."""
    failures = []
    if agg.total_macs != ref.total_macs or agg.total_cycle_macs != ref.total_cycle_macs:
        failures.append(f"totals differ: ({agg.total_macs}, {agg.total_cycle_macs}) vs "
                        f"({ref.total_macs}, {ref.total_cycle_macs})")
    a = {k: {L: tuple(v) for L, v in d.items()} for k, d in agg.records.items()}
    b = {k: {L: tuple(v) for L, v in d.items()} for k, d in ref.records.items()}
    if a != b:
        diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
        failures.append(f"{len(diff)} record keys differ, e.g. {sorted(diff, key=str)[:3]}")
    return failures


# ---------------------------------------------------------------------------
# statistics and the pre-registered Round-7 gate
# ---------------------------------------------------------------------------
def paired(candidate, reference) -> dict:
    """Paired window statistics (same formula as prc_local_refine.paired_stats)."""
    if len(candidate) != len(reference) or len(candidate) < 2:
        raise ValueError("paired losses require equal counts >= 2")
    if not all(math.isfinite(x) for x in list(candidate) + list(reference)):
        raise ValueError("nonfinite loss")
    diffs = [a - b for a, b in zip(candidate, reference)]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs))
    return {"mean_dnll": mean, "se": se,
            "z": mean / se if se else (0.0 if mean == 0 else None),
            "dppl_pct": math.expm1(mean) * 100}


def arm_metrics(arm_nll, inc_nll, arm_cost, inc_cost) -> dict:
    st = paired(arm_nll, inc_nll)
    se_pct = 100.0 * math.exp(st["mean_dnll"]) * st["se"]  # delta method for 100*expm1
    extra = 100.0 * (arm_cost / inc_cost - 1.0)
    return {"dnll": st["mean_dnll"], "se_dnll": st["se"], "z": st["z"],
            "dppl_pct": st["dppl_pct"], "se_pct": se_pct,
            "extra_cycles_pct": extra,
            "dppl_pct_per_1pct_extra_cycles": (st["dppl_pct"] / extra) if extra else None,
            "n_windows": len(arm_nll)}


def chord_table(best_rows: list, trace_costs: dict) -> dict:
    """%PPL reduction per 1% extra SC cycles between adjacent best(all) budgets.

    chord(m, t -> t') = [100*(PPL_t - PPL_t')/PPL_t] / [100*(C_t'/C_t - 1)],
    positive when the higher budget is better; C = full-test trace cost."""
    rows = {(r["model"], int(r["target"])): r for r in best_rows}
    out = {}
    for model in sorted({m for m, _ in rows}):
        for lo, hi in zip(BUDGETS, BUDGETS[1:]):
            if (model, lo) not in rows or (model, hi) not in rows:
                continue
            a, b = rows[(model, lo)], rows[(model, hi)]
            ca, cb = trace_costs[(model, lo)], trace_costs[(model, hi)]
            dppl = 100.0 * (a["best_ppl"] - b["best_ppl"]) / a["best_ppl"]
            dcyc = 100.0 * (cb / ca - 1.0)
            out[f"{model}:t{lo}->t{hi}"] = {
                "model": model, "from_target": lo, "to_target": hi,
                "from_ppl": a["best_ppl"], "to_ppl": b["best_ppl"],
                "from_cost": ca, "to_cost": cb, "dppl_pct": dppl, "dcycles_pct": dcyc,
                "chord_pct_ppl_per_1pct_cycles": dppl / dcyc,
                "log_chord": math.log(a["best_ppl"] / b["best_ppl"]) / math.log(cb / ca),
                "from_arm": a.get("winning_arm"), "to_arm": b.get("winning_arm"),
            }
    return out


def binding_chord(chords: dict, model: str, target: int) -> dict:
    """Pre-registered binding chord: the interval from the cell's budget to the next
    higher budget (the direction arm A moves along the cost axis)."""
    idx = BUDGETS.index(int(target))
    up = chords[f"{model}:t{BUDGETS[idx]}->t{BUDGETS[idx + 1]}"]
    down = chords.get(f"{model}:t{BUDGETS[idx - 1]}->t{BUDGETS[idx]}") if idx > 0 else None
    return {"binding": "upward", "chord": up["chord_pct_ppl_per_1pct_cycles"], "upward": up,
            "downward_sensitivity": down}


def round7_gate(model: str, a_metrics: dict, chord: float, verification_ok: bool) -> dict:
    """Pre-registered rule (manifest ``preregistered_rules.round7_gate``).

    proceed iff verification_ok
            and dPPL(A) < -(0.6 * chord * extra_cycles%) - 1.5 * SE%
            and gross dNLL(A) <= -0.01 (30B) / -0.004 (4B).
    dPPL(A) = 100*expm1(mean paired dNLL); SE% = 100*exp(mean)*paired SE;
    extra_cycles% = 100*(cost_A/cost_INC - 1), both costs trace-exact on the
    same 16 windows; chord = binding upward chord in %PPL per 1% cycles."""
    if model not in GATE_GROSS_DNLL:
        raise ValueError(f"no gross-dNLL threshold pre-registered for {model}")
    threshold = -(GATE_MARGINAL_FRACTION * chord * a_metrics["extra_cycles_pct"]) \
        - GATE_SE_MULTIPLIER * a_metrics["se_pct"]
    cond_budget = a_metrics["dppl_pct"] < threshold
    cond_gross = a_metrics["dnll"] <= GATE_GROSS_DNLL[model]
    proceed = bool(verification_ok and cond_budget and cond_gross)
    return {"proceed_round7_attention_ladder_solve": proceed,
            "verification_ok": bool(verification_ok),
            "cond_beats_budget_marginal": bool(cond_budget),
            "cond_gross_dnll": bool(cond_gross),
            "dppl_pct_A": a_metrics["dppl_pct"], "threshold_dppl_pct": threshold,
            "budget_term_pct": -(GATE_MARGINAL_FRACTION * chord * a_metrics["extra_cycles_pct"]),
            "se_term_pct": -GATE_SE_MULTIPLIER * a_metrics["se_pct"],
            "dnll_A": a_metrics["dnll"], "gross_dnll_threshold": GATE_GROSS_DNLL[model],
            "chord": chord, "extra_cycles_pct": a_metrics["extra_cycles_pct"],
            "constants": {"marginal_fraction": GATE_MARGINAL_FRACTION,
                          "se_multiplier": GATE_SE_MULTIPLIER}}


PREREGISTERED_RULES = {
    "round7_gate": (
        "For each cell, the Round-7 attention-ladder joint solve proceeds ONLY IF (i) every identity "
        "and realized-length audit of the diagnostic passed, AND (ii) dPPL(A) < -(0.6 x chord x "
        "extra_cycles(A)%) - 1.5 x SE(A)%, AND (iii) gross dNLL(A) <= -0.01 nats on 30B cells "
        "(<= -0.004 on 4B cells). dNLL/dPPL/SE are paired over the 16 fixed TRAIN windows "
        "(dPPL = 100*expm1(mean dNLL), SE% = 100*exp(mean dNLL)*paired SE); extra_cycles(A)% = "
        "100*(cost_A/cost_INC - 1) from the two arms' own SC traces on those windows; chord = "
        "%PPL reduction per 1% SC cycles between the cell's budget and the model's next higher "
        "budget on the best(all) curve (BEST_ALL_VS_SUBMITTED_20260927.json PPL, full-test trace "
        "cost). The downward chord is recorded as a non-binding sensitivity only."),
    "attribution_only": (
        "ATT128 and LIN128 are attribution-only arms (Q9 split of the residual between attention "
        "and linears/routing). They are never candidates for best(all), never full-tested from "
        "this diagnostic, and never enter the Round-7 gate. Arm A is also not a best(all) "
        "candidate as-is: it is not iso-cost and is measured on TRAIN windows only."),
    "fail_closed": (
        "Any identity mismatch (30B: incumbent window NLLs and trace totals vs round-4 "
        "confirm_incumbent; 4B: held-out probe window vs prc2 c7 held-out file; all cells: "
        "unprofiled vs profiled vs restored incumbent) or realized-length audit failure stops "
        "the cell with failure.json; no decision is emitted for it."),
    "reporting": (
        "Report A as per-row attention reallocation, not a T1 granularity effect. Final "
        "reporting remains best full-test PPL across comparable allocation rounds."),
}


def synthesize_arm_trace(payload: dict, arm: str, spec: dict, table: dict, total_blocks: int,
                         header_extra: dict | None = None) -> dict:
    """SIMULATION ONLY (CPU tests / driver dry-run): apply an arm's intended length
    map to every edited record of an incumbent summary trace, with no cascade.
    Never a measurement; the GPU driver audits real traces."""
    g_top = global_ladder(table)[0]
    fmap = _length_map(arm, g_top) if arm in EDIT_ARMS else (lambda L: L)
    lb = int(table.get("layer_buckets", 1))
    merged = {}
    for grp in payload["groups"]:
        grp = dict(grp)
        op, block = grp["op"], grp["block"]
        kind = "attention" if op in ATTN_OPS else "linear"
        if arm in EDIT_ARMS and op in STAGE and block is not None \
                and _edited(arm, spec, kind, op, bucket_index(int(block), total_blocks, lb)):
            grp["stoc_len"] = fmap(int(grp["stoc_len"]))
            grp["row_cycles"] = int(grp["rows"]) * grp["stoc_len"]
        key = tuple(grp[k] for k in ("block", "op", "unit", "stoc_len", "sc_prec", "halve", "mode",
                                     "granularity", "rng_levels", "chunk_d", "smoothed", "d_in", "d_out"))
        if key in merged:
            for k in ("calls", "rows", "macs", "row_cycles"):
                merged[key][k] += grp[k]
        else:
            merged[key] = grp
    header = dict(payload.get("header", {}))
    header.update(header_extra or {})
    header["simulated_from"] = "synthesize_arm_trace (no cascade; not a measurement)"
    return {"schema": payload.get("schema", "scmp-trace-summary-v1"), "header": header,
            "groups": list(merged.values())}
