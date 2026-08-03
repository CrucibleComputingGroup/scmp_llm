"""Frozen-threshold, validation-loss refinement of an SC-MP precision ladder.

This is a second pass after ``calibrate_mp_thresholds.py``:

1. Load a completed AdaptiveMPConfig table and freeze every dispatch threshold,
   dispatch metric, and protected-channel decision.
2. Raise the lowest adaptive rung by a small step.
3. Lower higher rungs, using measured MAC occupancy, until the original
   deployment budget is restored.
4. Accept the candidate only when held-out validation NLL improves at the same
   realized MAC-weighted stream length. Repeat until the next move is rejected.

The model is loaded once and its runtime MP config is swapped between candidate
tables. This makes a multi-round search much cheaper than launching one model
load per candidate. The final test PPL remains a separate evaluation; this
script defaults to the WikiText-2 validation split to avoid tuning on test.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def weighted_cost(
    levels: Iterable[int],
    level_weights: Iterable[float],
    protected_weight: float = 0.0,
    protected_stoc_len: int = 0,
) -> float:
    """MAC-weighted stream length for adaptive + fixed protected work."""
    return (
        sum(float(sl) * float(w) for sl, w in zip(levels, level_weights))
        + float(protected_weight) * float(protected_stoc_len)
    )


def aggregate_trace_weights(
    trace: dict,
    levels: Iterable[int],
    *,
    protected_stoc_len: int | None,
    protected_weight_hint: float = 0.0,
    escape_stoc_len: int | None = None,
) -> dict:
    """Extract normalized MAC occupancy for each adaptive ladder index.

    V9 deliberately uses protected@112 outside its adaptive ladder, so the
    protected share is directly observable. For older tables where protected
    work collides with an adaptive level, ``protected_weight_hint`` separates
    the two approximately; v9 does not need that fallback.

    ``escape_stoc_len`` (escape gate, R7): when the parent wrapper carries an
    escape gate, escaped rows run at a stream length outside the adaptive
    ladder (e.g. 128 over a 96-ceiling ladder). Their MAC share is split out
    as ``escape_weight`` and priced into ``cost`` at ``escape_stoc_len`` —
    the same path as every other level, so realized cost stays exact. If the
    escape length collides with a ladder rung (or the protected length) its
    share is already priced there at the identical stream length, so cost is
    unaffected and ``escape_weight`` stays 0.
    """
    levels = [int(x) for x in levels]
    by_sl: dict[int, float] = {}
    total = 0.0
    for group in trace.get("groups") or []:
        macs = float(group.get("macs", 0.0))
        if macs <= 0.0:
            continue
        sl = int(group["stoc_len"])
        by_sl[sl] = by_sl.get(sl, 0.0) + macs
        total += macs
    if total <= 0.0:
        raise ValueError("trace contains no positive SC MAC counts")
    shares = {sl: macs / total for sl, macs in by_sl.items()}
    protected_weight = 0.0
    if protected_stoc_len is not None:
        psl = int(protected_stoc_len)
        if psl in levels:
            protected_weight = min(
                max(float(protected_weight_hint), 0.0), shares.get(psl, 0.0))
        else:
            protected_weight = shares.pop(psl, 0.0)
    escape_weight = 0.0
    if escape_stoc_len is not None:
        esl = int(escape_stoc_len)
        if esl not in levels and (protected_stoc_len is None
                                  or esl != int(protected_stoc_len)):
            escape_weight = shares.pop(esl, 0.0)
    level_weights = [shares.pop(sl, 0.0) for sl in levels]
    if protected_stoc_len is not None and int(protected_stoc_len) in levels:
        idx = levels.index(int(protected_stoc_len))
        level_weights[idx] = max(level_weights[idx] - protected_weight, 0.0)
        shares.pop(int(protected_stoc_len), None)
    unexpected = {sl: w for sl, w in shares.items() if w > 1e-9}
    if unexpected:
        raise ValueError(
            "trace contains SC lengths outside the adaptive/protected set: "
            + ", ".join(f"{sl}:{w:.6f}" for sl, w in sorted(unexpected.items()))
        )
    observed = sum(level_weights) + protected_weight + escape_weight
    if abs(observed - 1.0) > 1e-6:
        raise ValueError(f"trace MAC shares sum to {observed:.9f}, expected 1")
    return {
        "level_weights": level_weights,
        "protected_weight": protected_weight,
        "escape_weight": escape_weight,
        "cost": weighted_cost(
            levels, level_weights, protected_weight,
            int(protected_stoc_len or 0),
        ) + escape_weight * float(escape_stoc_len or 0),
        "total_macs": total,
    }


def propose_floor_exchange(
    levels: Iterable[int],
    level_weights: Iterable[float],
    *,
    target_cost: float,
    protected_weight: float = 0.0,
    protected_stoc_len: int = 0,
    floor_step: int = 4,
    donor_step: int = 1,
    min_gap: int = 1,
    extra_cost: float = 0.0,
) -> dict:
    """Raise the floor, then greedily project higher rungs back to budget.

    Raising the floor can collide with the rung above it after several rounds.
    In that case the receiver-side increase propagates upward just enough to
    keep the ladder strictly descending. Donor reductions then spend from the
    higher rungs while preserving the same ordering constraint.

    Every intermediate donor state is considered and the state closest to the
    target is returned. Occupancy is frozen for the projection; the candidate's
    actual trace is measured after evaluation and is the acceptance authority.
    """
    old = [int(x) for x in levels]
    weights = [float(x) for x in level_weights]
    if len(old) < 2 or len(old) != len(weights):
        raise ValueError("levels and level_weights must have equal length >= 2")
    if floor_step <= 0 or donor_step <= 0 or min_gap <= 0:
        raise ValueError("floor_step, donor_step, and min_gap must be positive")
    if any(old[i] <= old[i + 1] for i in range(len(old) - 1)):
        raise ValueError(f"levels must be strictly descending, got {old}")

    candidate = old.copy()
    candidate[-1] += int(floor_step)
    # Receiver propagation: preserve the class count/threshold topology when a
    # raised floor catches the rung immediately above it.
    for i in range(len(candidate) - 2, -1, -1):
        candidate[i] = max(candidate[i], candidate[i + 1] + int(min_gap))

    def cost(xs: list[int]) -> float:
        # extra_cost: constant non-ladder work (escape-gate MAC share at its
        # fixed escape length) — priced like the protected term.
        return weighted_cost(
            xs, weights, protected_weight, protected_stoc_len) + float(extra_cost)

    receiver_levels = candidate.copy()
    receiver_cost = cost(candidate)
    best = candidate.copy()
    best_cost = receiver_cost
    path: list[dict] = []

    # At most a few hundred one-cycle moves for current ladders. Zero-occupancy
    # rungs are allowed to move because doing so can unlock the donor above.
    for _ in range(10000):
        moves = []
        for i in range(len(candidate) - 1):
            lowered = candidate[i] - int(donor_step)
            if lowered < candidate[i + 1] + int(min_gap):
                continue
            trial = candidate.copy()
            trial[i] = lowered
            trial_cost = cost(trial)
            moves.append((trial_cost, i, trial))
        if not moves:
            break
        # Prefer not to undershoot and spend from the highest rung first, which
        # is the user's proposed floor-for-ceiling exchange. If every available
        # one-cycle move would cross the target, take the smallest overshoot.
        feasible = [m for m in moves if m[0] >= target_cost]
        if feasible:
            trial_cost, donor_idx, trial = min(feasible, key=lambda m: m[1])
        else:
            trial_cost, donor_idx, trial = min(
                moves, key=lambda m: (abs(m[0] - target_cost), m[1]))
        candidate = trial
        path.append({
            "donor_index": int(donor_idx),
            "donor_level": int(candidate[donor_idx]),
            "predicted_cost": float(trial_cost),
        })
        if abs(trial_cost - target_cost) < abs(best_cost - target_cost):
            best = candidate.copy()
            best_cost = trial_cost
        if trial_cost <= target_cost:
            break

    if best == old:
        raise ValueError("floor exchange produced no ladder change")
    return {
        "old_levels": old,
        "receiver_levels": receiver_levels,
        "levels": best,
        "target_cost": float(target_cost),
        "receiver_cost": float(receiver_cost),
        "predicted_cost": float(best_cost),
        "predicted_budget_error": float(best_cost - target_cost),
        "floor_step": int(floor_step),
        "donor_step": int(donor_step),
        "min_gap": int(min_gap),
        "donor_moves": path,
    }


def resolve_parent_table(wrapper_path: Path) -> tuple[dict, Path, dict]:
    wrapper_path = wrapper_path.resolve()
    with wrapper_path.open() as f:
        wrapper = json.load(f)
    if wrapper.get("type") != "AdaptiveMPConfig":
        raise ValueError(f"expected AdaptiveMPConfig wrapper: {wrapper_path}")
    table_path = Path(wrapper["threshold_table_path"])
    if not table_path.is_absolute():
        table_path = wrapper_path.parent / table_path
    table_path = table_path.resolve()
    with table_path.open() as f:
        table = json.load(f)
    wrapper_levels = [int(x) for x in wrapper["stoc_len_levels"]]
    table_levels = [int(x) for x in table["stoc_len_levels"]]
    if wrapper_levels != table_levels:
        raise ValueError(
            f"parent wrapper/table level mismatch: {wrapper_levels} vs {table_levels}")
    return wrapper, table_path, table


def wrapper_escape_gate(wrapper: dict) -> dict:
    """Escape-gate keys carried by an AdaptiveMPConfig wrapper.

    Empty dict = gate off (absent or null ``escape_gate_k``) — every caller
    then behaves byte-identically to the pre-gate code."""
    if wrapper.get("escape_gate_k") is None:
        return {}
    return {
        "escape_gate_k": float(wrapper["escape_gate_k"]),
        "escape_stoc_len": int(wrapper.get("escape_stoc_len", 128)),
    }


def make_candidate_table(
    parent_table: dict,
    levels: Iterable[int],
    *,
    parent_table_path: Path,
    round_index: int,
    target_cost: float,
    proposal: dict,
    history: list[dict],
    command: str,
) -> dict:
    """Clone a table while changing only level values and diagnostics."""
    levels = [int(x) for x in levels]
    out = copy.deepcopy(parent_table)
    old_levels = [int(x) for x in parent_table["stoc_len_levels"]]
    if len(levels) != len(old_levels):
        raise ValueError("pass 2 must preserve the number of threshold classes")
    out["stoc_len_levels"] = levels
    for section in ("operator_defaults", "buckets"):
        for payload in (out.get(section) or {}).values():
            counts = [int(x) for x in payload.get("counts") or []]
            if len(counts) == len(levels) and sum(counts) > 0:
                payload["avg_stoc_len"] = (
                    sum(c * sl for c, sl in zip(counts, levels)) / sum(counts))
            # These error measurements belong to the parent levels. Keep them
            # explicitly as pass-1 diagnostics rather than mislabeling them as
            # measurements at the refined ladder.
            if "level_mean_error" in payload:
                payload["pass1_level_mean_error"] = payload.pop("level_mean_error")
            if "avg_error" in payload:
                payload["pass1_avg_error"] = payload.pop("avg_error")
    out.pop("expected_avg_stoc_len", None)
    out["expected_flop_avg_stoc_len"] = float(proposal["predicted_cost"])
    parent_method = str(parent_table.get("method", "adaptive_mp"))
    if not parent_method.endswith("_ppl_ladder_refine"):
        out["method"] = parent_method + "_ppl_ladder_refine"
    out["pass2_ladder_refine"] = {
        "type": "frozen_threshold_validation_nll",
        "parent_table": str(parent_table_path),
        "parent_levels": old_levels,
        "thresholds_frozen": True,
        "dispatch_metrics_frozen": True,
        "protected_channels_frozen": True,
        "round": int(round_index),
        "target_flop_avg_stoc_len": float(target_cost),
        "proposal": proposal,
        "history": history,
        "command": command,
    }
    return out


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def _write_wrapper(path: Path, levels: Iterable[int], table_path: Path,
                   *, escape_gate: dict | None = None) -> None:
    payload = {
        "type": "AdaptiveMPConfig",
        "stoc_len_levels": [int(x) for x in levels],
        "threshold_table_path": str(table_path.resolve()),
    }
    if escape_gate:
        # Escape gate (R7): carry the parent wrapper's gate keys onto every
        # wrapper this pass writes, so candidate/best wrappers reproduce the
        # evaluated configuration instead of silently dropping the gate.
        payload.update(escape_gate)
    _atomic_json(path, payload)


def _load_trace(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _protected_info(table: dict) -> tuple[int | None, float]:
    protected = table.get("protected_channels") or {}
    if not protected.get("indices"):
        return None, 0.0
    return (
        int(protected.get("stoc_len", 128)),
        float(protected.get("global_mac_weighted_frac", 0.0)),
    )


def nominal_target_cost(table: dict) -> float:
    """Recover the original all-SC target, including protected work.

    Protected-channel compensation rewrites the top-level ``budget_ratio`` to
    the lower residual target. The original ratio is retained inside the
    protected payload and is the correct deployment constraint for pass 2.
    """
    protected = table.get("protected_channels") or {}
    ratio = protected.get("original_budget_ratio")
    if ratio is None:
        ratio = table.get("budget_ratio")
    ref = table.get("budget_ref_stoc_len")
    if ratio is None or ref is None:
        raise ValueError(
            "parent table does not record budget_ratio/budget_ref_stoc_len")
    target = float(ratio) * float(ref)
    if target <= 0.0:
        raise ValueError(f"invalid nominal target cost {target}")
    return target


def _install_runtime_table(model, levels: list[int], table_path: Path,
                           escape_gate: dict | None = None):
    from scmp_kernels.mp import AdaptiveMPConfig

    mp = AdaptiveMPConfig(
        stoc_len_levels=levels,
        threshold_table_path=str(table_path),
        **(escape_gate or {}),
    )
    # SCLinear stores the shared HF config in ``_sc_config`` while attention
    # modules expose it as ``config``. Update every distinct config object so
    # model-family details cannot leave part of the graph on the prior ladder.
    configs = {id(model.config): model.config}
    for module in model.modules():
        for attr in ("config", "_sc_config"):
            cfg = getattr(module, attr, None)
            if cfg is not None:
                configs[id(cfg)] = cfg
    for cfg in configs.values():
        setattr(cfg, "sc_mp_config", mp)
    return mp


def _eval_once(
    model,
    enc,
    *,
    ctx: int,
    levels: list[int],
    table_path: Path,
    wrapper_path: Path,
    trace_path: Path,
    split: str,
    metric_profile_path: Path | None = None,
    metric_profile_bins: int = 257,
    window_losses: list | None = None,
    escape_gate: dict | None = None,
) -> dict:
    from benchmark.quant.eval_quant import compute_ppl
    from model.sc_common import (
        mp_tracker_avg_stoc_len,
        mp_tracker_flop_avg_stoc_len,
        mp_metric_profile_disable,
        mp_metric_profile_reset,
        mp_metric_profile_snapshot,
        mp_tracker_reset,
    )
    from scmp_kernels import trace as sc_trace

    print(
        f"[pass2 eval] split={split} levels={levels} table={table_path.name}",
        flush=True,
    )
    _install_runtime_table(model, levels, table_path, escape_gate=escape_gate)
    os.environ["MP_CONFIG_JSON"] = str(wrapper_path)
    sc_trace.reset()
    sc_trace.enable(str(trace_path), mode="summary")
    mp_tracker_reset()
    if metric_profile_path is not None:
        mp_metric_profile_reset(enabled=True, bins=metric_profile_bins)
    else:
        mp_metric_profile_disable()
    ppl, tokens, seconds = compute_ppl(
        model, enc, ctx, ctx, window_losses=window_losses)
    metric_profile = None
    if metric_profile_path is not None:
        metric_profile = mp_metric_profile_snapshot()
        mp_metric_profile_disable()
        _atomic_json(metric_profile_path, metric_profile)
    row_cost = mp_tracker_avg_stoc_len()
    tracker_cost = mp_tracker_flop_avg_stoc_len()
    sc_trace.flush(
        str(trace_path),
        header_extra={
            "purpose": "frozen-threshold pass2 ladder refinement",
            "dataset_split": split,
            "levels": levels,
            "threshold_table": str(table_path),
            "ppl": ppl,
            "nll": math.log(ppl),
            "eval_tokens": tokens,
            "eval_seconds": seconds,
            "ppl_window_batch_size": int(
                os.environ.get("PPL_WINDOW_BATCH_SIZE", "1")),
            "realized_avg_sl": row_cost,
            "realized_flop_avg_sl": tracker_cost,
        },
    )
    print(
        f"[pass2 eval result] levels={levels} ppl={ppl:.6f} "
        f"nll={math.log(ppl):.8f} tokens={tokens} sec={seconds:.1f} "
        f"tracker_flop_avg_sl={tracker_cost:.4f}",
        flush=True,
    )
    result = {
        "ppl": float(ppl),
        "nll": float(math.log(ppl)),
        "tokens": int(tokens),
        "seconds": float(seconds),
        "row_avg_stoc_len": float(row_cost),
        "tracker_flop_avg_stoc_len": float(tracker_cost),
        "trace": str(trace_path),
        "table": str(table_path),
        "wrapper": str(wrapper_path),
        "levels": levels,
    }
    if metric_profile_path is not None:
        result["metric_profile"] = str(metric_profile_path)
    return result


def _load_eval_stream(tokenizer, *, split: str, max_tokens: int, ctx: int):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    if max_tokens > 0:
        enc = enc[:max_tokens]
    enc = enc[: (enc.shape[0] // ctx) * ctx]
    if enc.numel() < ctx:
        raise ValueError(
            f"{split} stream has only {enc.numel()} usable tokens for ctx={ctx}")
    return enc


def _build_parent_model(model_path: str, wrapper_path: Path, alpha: float):
    """Build through eval_quant's public env-driven MP entry point."""
    from benchmark.quant.eval_quant import build_model

    os.environ["MP_CONFIG_JSON"] = str(wrapper_path.resolve())
    return build_model(model_path, "mp", alpha=alpha)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-wrapper", required=True, type=Path)
    p.add_argument("--model-path", default="")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--split", choices=["train", "validation", "test"],
                   default="validation")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--floor-step", type=int, default=4)
    p.add_argument("--donor-step", type=int, default=1)
    p.add_argument("--min-gap", type=int, default=1)
    p.add_argument("--budget-tol", type=float, default=0.35)
    p.add_argument("--min-nll-improvement", type=float, default=0.0)
    p.add_argument("--target-cost", type=float, default=None,
                   help="Default: use the parent table's original compensated "
                        "target (e.g. avg32 -> 32), correcting runtime drift.")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--proposal-only", action="store_true",
                   help="CPU-only: print one proposal from --baseline-trace.")
    p.add_argument("--baseline-trace", type=Path, default=None,
                   help="Reuse a completed pass2 parent trace instead of "
                        "rerunning the baseline; also used by --proposal-only.")
    return p


def _validate_split_rounds(split: str, max_rounds: int) -> None:
    """Keep the held-out test split out of the ladder search loop."""
    if split == "test":
        if max_rounds != 0:
            raise SystemExit(
                "--split test is evaluation-only and requires --max-rounds 0")
    elif max_rounds < 1:
        raise SystemExit("--max-rounds must be >= 1 for ladder search")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    wrapper, parent_table_path, parent_table = resolve_parent_table(
        args.parent_wrapper)
    parent_levels = [int(x) for x in wrapper["stoc_len_levels"]]
    protected_sl, protected_hint = _protected_info(parent_table)
    escape_gate = wrapper_escape_gate(wrapper)
    escape_sl = escape_gate.get("escape_stoc_len")
    if escape_gate:
        print(f"[pass2] escape gate ON: k={escape_gate['escape_gate_k']:g} "
              f"escape_stoc_len={escape_sl}", flush=True)

    if args.proposal_only:
        if args.baseline_trace is None:
            raise SystemExit("--proposal-only requires --baseline-trace")
        occupancy = aggregate_trace_weights(
            _load_trace(args.baseline_trace), parent_levels,
            protected_stoc_len=protected_sl,
            protected_weight_hint=protected_hint,
            escape_stoc_len=escape_sl,
        )
        target = float(
            args.target_cost
            if args.target_cost is not None
            else nominal_target_cost(parent_table))
        proposal = propose_floor_exchange(
            parent_levels, occupancy["level_weights"],
            target_cost=target,
            protected_weight=occupancy["protected_weight"],
            protected_stoc_len=int(protected_sl or 0),
            floor_step=args.floor_step,
            donor_step=args.donor_step,
            min_gap=args.min_gap,
            extra_cost=occupancy.get("escape_weight", 0.0) * float(escape_sl or 0),
        )
        print(json.dumps({"occupancy": occupancy, "proposal": proposal}, indent=2))
        return 0

    if not args.model_path:
        raise SystemExit("--model-path is required unless --proposal-only is used")
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --proposal-only is used")
    _validate_split_rounds(args.split, args.max_rounds)
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)

    # Build once with the parent wrapper; hybrid/SQ environment is supplied by
    # the launcher exactly as in pass 1.
    model, tokenizer = _build_parent_model(
        args.model_path, args.parent_wrapper, args.alpha)
    enc = _load_eval_stream(
        tokenizer, split=args.split, max_tokens=args.max_tokens, ctx=args.ctx)

    baseline_trace = (
        args.baseline_trace.resolve()
        if args.baseline_trace is not None
        else outdir / "round00_parent_trace.json")
    if args.baseline_trace is not None:
        trace_payload = _load_trace(baseline_trace)
        header = trace_payload.get("header") or {}
        if header.get("dataset_split") != args.split:
            raise SystemExit(
                f"baseline trace split {header.get('dataset_split')!r} does not "
                f"match requested {args.split!r}")
        if "ppl" not in header or "nll" not in header:
            raise SystemExit(
                f"baseline trace lacks pass2 ppl/nll metadata: {baseline_trace}")
        baseline = {
            "ppl": float(header["ppl"]),
            "nll": float(header["nll"]),
            "tokens": int(header.get("eval_tokens", 0)),
            "seconds": float(header.get("eval_seconds", 0.0)),
            "row_avg_stoc_len": float(header.get("realized_avg_sl", 0.0)),
            "tracker_flop_avg_stoc_len": float(
                header.get("realized_flop_avg_sl", 0.0)),
            "trace": str(baseline_trace),
            "table": str(parent_table_path),
            "wrapper": str(args.parent_wrapper.resolve()),
            "levels": parent_levels,
        }
        print(f"[pass2] reusing baseline trace: {baseline_trace}", flush=True)
    else:
        baseline = _eval_once(
            model, enc, ctx=args.ctx,
            levels=parent_levels,
            table_path=parent_table_path,
            wrapper_path=args.parent_wrapper.resolve(),
            trace_path=baseline_trace,
            split=args.split,
            escape_gate=escape_gate,
        )
    occupancy = aggregate_trace_weights(
        _load_trace(baseline_trace), parent_levels,
        protected_stoc_len=protected_sl,
        protected_weight_hint=protected_hint,
        escape_stoc_len=escape_sl,
    )
    baseline["trace_flop_avg_stoc_len"] = occupancy["cost"]
    target_cost = float(
        args.target_cost
        if args.target_cost is not None
        else nominal_target_cost(parent_table))
    baseline_budget_error = float(occupancy["cost"] - target_cost)
    history: list[dict] = [{
        "round": 0,
        "accepted": True,
        "reason": "parent",
        "target_flop_avg_stoc_len": target_cost,
        "budget_error": baseline_budget_error,
        **baseline,
    }]
    best = baseline
    current_levels = parent_levels
    current_occupancy = occupancy
    best_table_path = parent_table_path

    for round_idx in range(1, args.max_rounds + 1):
        try:
            proposal = propose_floor_exchange(
                current_levels,
                current_occupancy["level_weights"],
                target_cost=target_cost,
                protected_weight=current_occupancy["protected_weight"],
                protected_stoc_len=int(protected_sl or 0),
                floor_step=args.floor_step,
                donor_step=args.donor_step,
                min_gap=args.min_gap,
                extra_cost=(current_occupancy.get("escape_weight", 0.0)
                            * float(escape_sl or 0)),
            )
        except ValueError as exc:
            history.append({
                "round": round_idx, "accepted": False,
                "reason": f"no_feasible_proposal: {exc}",
            })
            break
        levels = proposal["levels"]
        print(
            f"[pass2 proposal] round={round_idx} old={current_levels} "
            f"new={levels} target={target_cost:.4f} "
            f"predicted={proposal['predicted_cost']:.4f}",
            flush=True,
        )
        stem = f"round{round_idx:02d}_levels" + "-".join(map(str, levels))
        table_path = outdir / f"{stem}.json"
        wrapper_path = outdir / f"{stem}_wrapper.json"
        trace_path = outdir / f"{stem}_trace.json"
        candidate_table = make_candidate_table(
            parent_table, levels,
            parent_table_path=parent_table_path,
            round_index=round_idx,
            target_cost=target_cost,
            proposal=proposal,
            history=history,
            command=command,
        )
        _atomic_json(table_path, candidate_table)
        _write_wrapper(wrapper_path, levels, table_path,
                       escape_gate=escape_gate)
        result = _eval_once(
            model, enc, ctx=args.ctx,
            levels=levels,
            table_path=table_path,
            wrapper_path=wrapper_path,
            trace_path=trace_path,
            split=args.split,
            escape_gate=escape_gate,
        )
        candidate_occupancy = aggregate_trace_weights(
            _load_trace(trace_path), levels,
            protected_stoc_len=protected_sl,
            protected_weight_hint=protected_hint,
            escape_stoc_len=escape_sl,
        )
        actual_cost = float(candidate_occupancy["cost"])
        result["trace_flop_avg_stoc_len"] = actual_cost
        result["predicted_flop_avg_stoc_len"] = float(
            proposal["predicted_cost"])
        budget_error = actual_cost - target_cost
        nll_improvement = float(best["nll"] - result["nll"])
        budget_ok = abs(budget_error) <= args.budget_tol
        loss_ok = nll_improvement > args.min_nll_improvement
        accepted = bool(budget_ok and loss_ok)
        if not budget_ok:
            reason = f"budget_error={budget_error:+.4f}"
        elif not loss_ok:
            reason = f"nll_improvement={nll_improvement:+.8f}"
        else:
            reason = "validation_nll_improved_at_iso_budget"
        record = {
            "round": round_idx,
            "accepted": accepted,
            "reason": reason,
            "target_flop_avg_stoc_len": target_cost,
            "budget_error": budget_error,
            "nll_improvement": nll_improvement,
            "proposal": proposal,
            **result,
        }
        history.append(record)
        candidate_table["expected_flop_avg_stoc_len"] = actual_cost
        candidate_table["pass2_ladder_refine"]["evaluation"] = record
        _atomic_json(table_path, candidate_table)
        if not accepted:
            break
        best = result
        best_table_path = table_path
        current_levels = levels
        current_occupancy = candidate_occupancy

    best_wrapper = outdir / "best_wrapper.json"
    _write_wrapper(best_wrapper, best["levels"], best_table_path,
                   escape_gate=escape_gate)
    summary = {
        "schema": "scmp-frozen-threshold-ladder-refine-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model_path,
        "dataset_split": args.split,
        "max_tokens": args.max_tokens,
        "ctx": args.ctx,
        "parent_wrapper": str(args.parent_wrapper.resolve()),
        "parent_table": str(parent_table_path),
        "parent_levels": parent_levels,
        "protected_stoc_len": protected_sl,
        "escape_gate": escape_gate or None,
        "target_flop_avg_stoc_len": target_cost,
        "best_levels": best["levels"],
        "best_ppl": best["ppl"],
        "best_nll": best["nll"],
        "best_trace_flop_avg_stoc_len": best["trace_flop_avg_stoc_len"],
        "best_wrapper": str(best_wrapper),
        "history": history,
        "command": command,
    }
    _atomic_json(outdir / "summary.json", summary)
    with (outdir / "results.tsv").open("w") as f:
        f.write("round\taccepted\tlevels\tppl\tnll\tflop_avg_sl\tbudget_error\treason\n")
        for rec in history:
            if "levels" not in rec:
                continue
            f.write(
                f"{rec['round']}\t{int(bool(rec['accepted']))}\t"
                f"{','.join(map(str, rec['levels']))}\t{rec['ppl']:.8f}\t"
                f"{rec['nll']:.10f}\t{rec['trace_flop_avg_stoc_len']:.6f}\t"
                f"{float(rec.get('budget_error', 0.0)):+.6f}\t{rec['reason']}\n")
    print(
        "[PASS2 RESULT] "
        f"model={args.model_path} parent_levels={parent_levels} "
        f"best_levels={best['levels']} parent_ppl={baseline['ppl']:.6f} "
        f"best_ppl={best['ppl']:.6f} target={target_cost:.4f} "
        f"realized={best['trace_flop_avg_stoc_len']:.4f} "
        f"summary={outdir / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
