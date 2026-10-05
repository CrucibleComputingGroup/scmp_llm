"""Per-group precision trace: the stream length of EVERY computation group, in execution
order, as input for the energy/latency simulator.

The simulator's input today is the precision trace of scmp_kernels/trace.py
(scmp-trace-summary-v1): calls, rows, MACs and row-cycles summed per (block, op, unit,
stoc_len, d_in, d_out, rng_levels, smoothed, ...) over the whole eval. It says how many
groups ran at each length, never which ones -- a linear's (token row, 128-chunk) group or an
attention (head, query) row is gone once it is counted, so tile occupancy (a tile runs at
its longest row), scheduling and per-group control cannot be modelled from it. This tool
keeps every group.

Output (GTRACE_OUT is a directory), for the first GTRACE_WINDOWS eval windows (a window is
one ctx-token prefill; 0 = every window):

  group_trace.npz
    w<w>/b<block>/<op>[/e<expert>]   uint8 stream length of every group of one call
        linears  [rows, n_chunks]. Row = token; for a MoE expert, its routed tokens in the
                 HF 4.51 gather order: top-k slot major, then token (group_trace_check.
                 expert_tokens rebuilds it from router_selected). Chunk = 128 consecutive
                 columns of the DISPATCHED input, i.e. the input without its protected
                 columns, columns in ascending order, the short tail chunk last. Group MACs =
                 chunk width * d_out.
        qk, av   [batch*heads, rows], row index = b*heads + h: one length per (query head,
                 query row). K/V heads are shared by attention_heads / kv_heads query heads
                 (GQA, header). qk contracts over head_dim; av over the keys (one whole
                 probability row, unchunked). Group MACs = d_in * d_out, DENSE: the emulator
                 multiplies every key and applies the causal mask afterwards (a causal array
                 would do (pos+1) * head_dim).
    protected/b<block>/<op>[/e<expert>]   int32 protected input columns of an SC linear.
                 They run at protected_stoc_len for every row: rows * n_protected * d_out
                 MACs per call. Dispatched columns = the sorted complement.
    w<w>/token_ids, w<w>/b<block>/router_selected|router_weights (MoE)
  group_trace.json
    header  provenance, units, rng_levels, head counts, the coverage note
    ops     static description per (op, block, unit): backend, d_in, d_out, chunk widths,
            ladder (index -> length map the dispatch used), smoothed (AWQ scales folded in
            at run time), protected columns and length
    calls   per window, every operator call of the SC-MP datapath in execution order;
            INT-masked calls (hybrid schedule) carry their shapes and no lengths

Lengths are SC cycles in the code's halved space, the unit of stoc_len in the existing
trace (nominal = 2x). FP16 work (lm_head, embeddings, norms, softmax, MoE router) is absent,
as in the existing trace.

Two modes. GTRACE_FP_MATMUL=0 (default) runs the real SC eval and stops after the last
requested window; the eval is bit-deterministic and its windows are independent, so these
are exactly the first windows of the deployed full-test eval, at ~28 s per window on 4B
(the full test is what is slow: 4068 s). GTRACE_CHECK_TRACE=<path> also writes the
scmp-trace-summary of the captured windows, which group_trace_check.py reproduces from the
groups as an identity check. GTRACE_FP_MATMUL=1 needs no SC inference: every SC matmul
becomes an exact FP matmul (AWQ scales cancel) while the deployed dispatch code still runs
on every call -- on the FP trajectory (4B full test 191 s). Every group's length then comes
from FP activations, which miss the SC error of the same block and of every block upstream:
on 4B t48 (4 windows) only 69% of groups (67% of MACs) keep the SC length, 90% within one
ladder rung, per-op trace cost -2.3...+0.8%, 32-row tile-max avg_sl -1.3%. On MoE the routing
also follows the FP trajectory. Use it for length statistics, not as a per-group record.
The hook points are tile_cost.py's; nothing here touches dispatch or numerics.

Usage -- the environment of kbands/run_prc_ppl.sbatch, plus
    GTRACE_OUT=/dir [GTRACE_WINDOWS=4] [GTRACE_FP_MATMUL=0] [GTRACE_CHECK_TRACE=/path.json]
    python -u benchmark/ppl/group_trace.py
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SCHEMA = "scmp-group-trace-v1"
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTENTION = ("qk", "av")
COVERAGE = ("SC-MP datapath operator calls only: SC calls with per-group lengths, INT-masked "
            "calls with shapes -- FP16 ops (lm_head, embeddings, MoE router gate, norms, "
            "softmax) are NOT recorded")
SEMANTICS = {
    "stoc_len": "SC cycles per group, halved (sign-magnitude) space = stoc_len of "
                "scmp-trace-summary-v1; nominal = 2x",
    "linear_group": "(row, 128-chunk of the dispatched input); MACs = chunk width * d_out",
    "linear_rows": "dense: token position; MoE expert: routed tokens in HF gather order, top-k "
                   "slot major then token (group_trace_check.expert_tokens)",
    "protected": "every row's protected columns run at protected_stoc_len; "
                 "MACs = rows * n_protected * d_out",
    "attention_group": "(query head, query row), row index b*heads + h; MACs = d_in * d_out "
                       "(dense); K/V shared by attention_heads/kv_heads query heads",
    "int_call": "whole (op, block) on the INT backend, no SC groups",
}
TRAJECTORY = {
    False: "exact: the deployed SC run (windows 0.. of the full-test eval)",
    True: "fp: SC matmuls replaced by exact FP matmuls; every length (and MoE routing) comes "
          "from the FP trajectory, not the deployed SC run -- a statistical stand-in, not a "
          "per-group record",
}


class _CaptureDone(BaseException):
    """Raised on the first call of the window after the last requested one."""


def _suffix(unit):
    return "" if unit is None else f"/e{unit}"


class GroupTraceCollector:
    def __init__(self, n_windows):
        self.n_windows = int(n_windows)     # <= 0: every window
        self.window = -1          # incremented on every block-0 q_proj call (any backend)
        self.maps = {}            # npz key -> uint8 lengths of one call's groups
        self.protected = {}       # npz key -> int32 protected input columns
        self.ops = {}             # (op, block, unit) -> static description
        self.calls = {}           # window -> operator call records, execution order
        self.routing = {}         # (window, block) -> (selected int16, weights float16)
        self.tokens = {}          # window -> int32 token ids
        self._linear = None       # SCLinear call in flight
        self._attention = None    # SC attention call in flight
        self._block = None        # block and SC config of the latest linear call: an INT
        self._config = None       # attention call carries neither
        self._attn_seen = {}      # (window, block) -> attention ops called so far

    def _keep(self):
        return self.window >= 0 and (self.n_windows <= 0 or self.window < self.n_windows)

    def _op(self, key, **desc):
        if self.ops.setdefault(key, desc) != desc:
            raise ValueError(f"static description of {key} changed between calls")

    def _call(self, **rec):
        calls = self.calls.setdefault(self.window, [])
        calls.append(dict(seq=len(calls), **rec))

    def _store(self, key, lengths):
        import torch
        if lengths.numel() and (int(lengths.max()) > 255 or int(lengths.min()) <= 0):
            raise ValueError(f"stream length outside uint8 range at {key}")
        if key in self.maps:
            raise ValueError(f"two dispatches for one group map: {key}")
        self.maps[key] = lengths.to(torch.uint8).cpu().numpy()

    def _attention_call(self, op, block, backend, a, b, key, ladder, smoothed):
        seen = self._attn_seen.setdefault((self.window, block), [])
        if op in seen:
            raise ValueError(f"{op} called twice in block {block} of window {self.window}")
        seen.append(op)
        bsz, heads, n, k = (int(v) for v in a.shape)
        m = int(b.shape[-2])
        self._op((op, block, None), backend=backend, heads=heads, d_in=k, d_out=m,
                 ladder=ladder, smoothed=smoothed)
        self._call(block=block, op=op, unit=None, backend=backend, batch=bsz, heads=heads,
                   rows=n, d_in=k, d_out=m, groups=key)

    @contextmanager
    def installed(self, scm, moe_block_cls=None, embedding_cls=None):
        import torch
        orig = (scm.SCLinear.forward, scm.per_row_chunk_rungs, scm._sc_attention_matmul_ab_t,
                scm.adaptive_classify_rows, scm._int_attention_matmul_ab_t)
        old_linear, old_prc, old_attention, old_classify, old_int_attention = orig
        old_moe = moe_block_cls.forward if moe_block_cls is not None else None
        old_emb = embedding_cls.forward if embedding_cls is not None else None
        col = self

        def linear(mod, x):
            op = getattr(mod, "_sc_op_name", None)
            block = getattr(mod, "_sc_block_idx", None)
            unit = getattr(mod, "_sc_unit_idx", None)
            if op == "q_proj" and block == 0:
                col.window += 1
                if col.n_windows > 0 and col.window >= col.n_windows:
                    raise _CaptureDone()
            cfg = mod._sc_config
            col._block, col._config = block, cfg
            if not getattr(cfg, "use_sc_linear", True) or x.numel() == 0:
                return old_linear(mod, x)
            rows, d_in, d_out = x.numel() // x.shape[-1], int(x.shape[-1]), int(mod.out_features)
            backend = scm._hybrid_backend(cfg, op, block)
            if backend != "sc":
                if col._keep():
                    col._op((op, block, unit), backend=backend, d_in=d_in, d_out=d_out)
                    col._call(block=block, op=op, unit=unit, backend=backend, rows=rows,
                              d_in=d_in, d_out=d_out, groups=None)
                return old_linear(mod, x)
            mp = getattr(cfg, "sc_mp_config", None)
            if (op not in LINEARS or mp is None
                    or getattr(cfg, "sc_group_stoclen", None) is not None
                    or getattr(cfg, "sc_ste_grad", False)):
                raise ValueError(f"group trace needs deployed adaptive PRC linears ({op}/{block})")
            protected = scm._channel_index_tensor(
                mp.get_protected_channels(operator=op, block_idx=block, unit_idx=unit),
                d_in, x.device)
            previous = col._linear
            ctx = col._linear = dict(op=op, block=block, unit=unit, seen=False, key=None,
                                     chunk_d=None, chunk_widths=[], ladder=None)
            try:
                result = old_linear(mod, x)
                n_prot = int(protected.numel())
                if n_prot < d_in and not ctx["seen"]:
                    raise ValueError(f"SC linear without a PRC dispatch: {op}/{block}")
                if sum(ctx["chunk_widths"]) != d_in - n_prot:
                    raise ValueError(f"chunks of {op}/{block} do not tile the dispatched input")
                if col._keep():
                    pkey = None
                    if n_prot:
                        pkey = f"protected/b{block}/{op}{_suffix(unit)}"
                        if pkey not in col.protected:
                            col.protected[pkey] = protected.to(torch.int32).cpu().numpy()
                    col._op((op, block, unit), backend="sc", d_in=d_in, d_out=d_out,
                            chunk_d=ctx["chunk_d"], chunk_widths=ctx["chunk_widths"],
                            ladder=ctx["ladder"],
                            smoothed=getattr(mod, "smooth_scales", None) is not None,
                            protected_cols=n_prot, protected_key=pkey,
                            protected_stoc_len=(int(getattr(mp, "protected_channel_stoc_len", None)
                                                    or max(mp.stoc_len_levels))
                                                if n_prot else None))
                    col._call(block=block, op=op, unit=unit, backend="sc", rows=rows,
                              d_in=d_in, d_out=d_out, groups=ctx["key"])
                return result
            finally:
                col._linear = previous

        def prc(x, chunk_d, levels, thresholds=None, target=0.0):
            result = old_prc(x, chunk_d, levels, thresholds=thresholds, target=target)
            ctx = col._linear
            if ctx is None:
                raise ValueError("PRC dispatch outside an SCLinear call")
            if ctx["seen"]:
                raise ValueError("multiple PRC dispatches in one SCLinear call")
            ctx["seen"] = True
            d, nch = x.shape[1], result.shape[1]
            ctx.update(chunk_d=int(chunk_d), ladder=[int(v) for v in levels],
                       chunk_widths=[int(chunk_d)] * (nch - 1) + [int(d - (nch - 1) * chunk_d)])
            if col._keep():
                ctx["key"] = f"w{col.window}/b{ctx['block']}/{ctx['op']}{_suffix(ctx['unit'])}"
                col._store(ctx["key"], torch.as_tensor(list(levels), device=x.device)[result.long()])
            return result

        def attention(a, b, **kw):
            previous = col._attention
            ctx = col._attention = dict(op=kw.get("operator"), block=kw.get("block_idx"),
                                        shape=tuple(a.shape), seen=False, key=None, ladder=None)
            try:
                result = old_attention(a, b, **kw)
                if not ctx["seen"]:
                    raise ValueError("SC attention without adaptive row classification")
                if col._keep():
                    col._attention_call(ctx["op"], ctx["block"], "sc", a, b, ctx["key"],
                                        ctx["ladder"], kw.get("smooth_scales") is not None)
                return result
            finally:
                col._attention = previous

        def classify(metric, config, *args, **kw):
            result = old_classify(metric, config, *args, **kw)
            ctx = col._attention
            if ctx is None:
                return result          # the linears' per-row assignment, unused under PRC
            if (ctx["op"] not in ATTENTION or kw.get("operator") != ctx["op"]
                    or kw.get("block_idx") != ctx["block"] or ctx["seen"]):
                raise ValueError("unrecognized attention dispatch context")
            ctx["seen"] = True
            # The index -> length map the runtime executes (escape slot included).
            values = config.classify_level_values(
                operator=ctx["op"], block_idx=kw.get("block_idx"),
                total_blocks=kw.get("total_blocks"))
            ctx["ladder"] = [int(v) for v in values]
            if col._keep():
                bsz, heads, n, _ = ctx["shape"]
                ctx["key"] = f"w{col.window}/b{ctx['block']}/{ctx['op']}"
                lengths = torch.as_tensor(list(values), device=metric.device)[
                    result.row_levels.long()]
                col._store(ctx["key"], lengths.view(bsz * heads, n))
            return result

        def int_attention(a, b, **kw):
            result = old_int_attention(a, b, **kw)
            if col._keep():
                # qk always precedes av in a block; the schedule must agree.
                block = col._block
                op = "av" if "qk" in col._attn_seen.get((col.window, block), []) else "qk"
                backend = scm._hybrid_backend(col._config, op, block)
                if not backend.startswith("int"):
                    raise ValueError(f"INT attention call but the schedule says {backend} "
                                     f"for {op}/{block}")
                col._attention_call(op, block, backend, a, b, None, None, False)
            return result

        def moe_forward(blk, hidden_states):
            if col._keep():
                with torch.no_grad():
                    hs = hidden_states.reshape(-1, hidden_states.shape[-1])
                    w = torch.softmax(blk.gate(hs), dim=1, dtype=torch.float)
                    w, sel = torch.topk(w, blk.top_k, dim=-1)
                    if blk.norm_topk_prob:
                        w = w / w.sum(dim=-1, keepdim=True)
                block = getattr(blk.experts[0].gate_proj, "_sc_block_idx", None)
                col.routing[(col.window, block)] = (
                    sel.to(torch.int16).cpu().numpy(), w.to(torch.float16).cpu().numpy())
            return old_moe(blk, hidden_states)

        def embedding(emb, ids):
            nxt = col.window + 1
            if (ids.dim() == 2 and ids.shape[0] == 1 and emb.num_embeddings > 50000
                    and nxt >= 0 and (col.n_windows <= 0 or nxt < col.n_windows)):
                # called before block 0 of the NEXT window
                col.tokens[nxt] = ids[0].to(torch.int32).cpu().numpy()
            return old_emb(emb, ids)

        (scm.SCLinear.forward, scm.per_row_chunk_rungs, scm._sc_attention_matmul_ab_t,
         scm.adaptive_classify_rows, scm._int_attention_matmul_ab_t) = (
            linear, prc, attention, classify, int_attention)
        if moe_block_cls is not None:
            moe_block_cls.forward = moe_forward
        if embedding_cls is not None:
            embedding_cls.forward = embedding
        try:
            yield self
        finally:
            (scm.SCLinear.forward, scm.per_row_chunk_rungs, scm._sc_attention_matmul_ab_t,
             scm.adaptive_classify_rows, scm._int_attention_matmul_ab_t) = orig
            if moe_block_cls is not None:
                moe_block_cls.forward = old_moe
            if embedding_cls is not None:
                embedding_cls.forward = old_emb

    def save(self, out_dir, header):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = dict(self.maps)
        arrays.update(self.protected)
        for (w, block), (sel, wt) in self.routing.items():
            arrays[f"w{w}/b{block}/router_selected"] = sel
            arrays[f"w{w}/b{block}/router_weights"] = wt
        for w, ids in self.tokens.items():
            arrays[f"w{w}/token_ids"] = ids
        np.savez_compressed(out_dir / "group_trace.npz", **arrays)
        windows = sorted(self.calls)
        ops = [dict(op=op, block=b, unit=u, **d) for (op, b, u), d in sorted(
            self.ops.items(), key=lambda t: (t[0][1], t[0][0], -1 if t[0][2] is None else t[0][2]))]
        meta = dict(schema=SCHEMA,
                    header=dict(header, windows=windows, n_maps=len(self.maps),
                                n_groups=int(sum(v.size for v in self.maps.values()))),
                    ops=ops, calls={str(w): self.calls[w] for w in windows})
        (out_dir / "group_trace.json").write_text(json.dumps(meta, indent=1))


def _sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def _git(repo):
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain")
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(status) if status is not None else None)


def provenance(repo, config):
    """Where the capture came from: config files, SC knobs, code state."""
    wrapper = os.environ.get("MP_CONFIG_JSON")
    table = None
    try:
        table = json.loads(Path(wrapper).read_text()).get("threshold_table_path")
    except (OSError, TypeError, ValueError):
        pass
    sources = ("model/sc_common.py", "kernels/scmp_kernels/mp/config.py",
               "kernels/scmp_kernels/sc/matmul.py", "benchmark/quant/eval_quant.py",
               "benchmark/ppl/group_trace.py")
    return dict(
        model=os.environ.get("MODEL_PATH"), cell=os.environ.get("GTRACE_CELL"),
        arm=os.environ.get("GTRACE_ARM"), ctx=int(os.environ.get("CTX", "2048")),
        frontend=os.environ.get("FRONTEND"), mp_config_json=wrapper, threshold_table=table,
        hybrid_config_json=os.environ.get("SC_HYBRID_CONFIG_JSON"),
        sha256={k: _sha256(v) for k, v in (("mp_config_json", wrapper), ("threshold_table", table),
                                           ("hybrid_config_json",
                                            os.environ.get("SC_HYBRID_CONFIG_JSON")))},
        attention_heads=getattr(config, "num_attention_heads", None),
        kv_heads=getattr(config, "num_key_value_heads", None),
        sc=dict(sc_prec=getattr(config, "sc_prec", None),
                halve=getattr(config, "sc_halve_bipolar_stoc_len", None),
                chunk_d=getattr(config, "sc_linear_chunk_d", None),
                # no enable-grid policy is active (main checks), so every SC call runs the
                # kernel default grid -- the rng_levels field of scmp-trace-summary-v1
                rng_levels=(2 ** (int(getattr(config, "sc_prec", 8)) - 1)
                            if getattr(config, "sc_halve_bipolar_stoc_len", False)
                            else 2 ** int(getattr(config, "sc_prec", 8))),
                **{k: os.environ.get(k) for k in (
                    "SC_OWEN_MODE", "SC_SCRAMBLE_MASKS", "SC_RNG_GRID", "SC_RNG_GRID_ATTN",
                    "SC_RNG_GRID_QK", "SC_RNG_GRID_AV")}),
        code=dict(scmp_llm=_git(repo), kernels=_git(repo / "kernels"),
                  sha256={s: _sha256(repo / s) for s in sources}),
    )


def main() -> int:
    out = os.environ.get("GTRACE_OUT")
    if not out:
        sys.exit("[gtrace] GTRACE_OUT (an output directory) is required")
    n_windows = int(os.environ.get("GTRACE_WINDOWS", "4"))
    fp = os.environ.get("GTRACE_FP_MATMUL", "0") == "1"
    check_trace = os.environ.get("GTRACE_CHECK_TRACE")
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "kernels")]
    import torch
    import model.sc_common as scm
    if os.environ.get("SC_MP_TRACE"):
        sys.exit("[gtrace] unset SC_MP_TRACE: a truncated capture must not write an eval "
                 "trace (GTRACE_CHECK_TRACE writes the capture's own)")
    if scm._SC_RNG_GRID or scm._SC_RNG_GRID_QK or scm._SC_RNG_GRID_AV:
        sys.exit("[gtrace] SC_RNG_GRID* policies are unsupported: the trace records one "
                 "rng_levels for every call")
    if fp and check_trace:
        sys.exit("[gtrace] GTRACE_CHECK_TRACE needs GTRACE_FP_MATMUL=0 (FP matmuls record "
                 "nothing)")
    if fp:
        def fp_matmul(a, b, **_):
            # (a/s)(b*s)^T == a b^T exactly, so AWQ smooth_scales need no handling.
            return a.float() @ b.float().t()
        scm._sc_matmul = fp_matmul
    sc_trace = None
    if check_trace:
        from scmp_kernels import trace as sc_trace
        sc_trace.enable(check_trace, mode="summary")   # eval_quant resets it before the eval
    moe_cls = None
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe as qm
        moe_cls = qm.Qwen3MoeSparseMoeBlock
    except Exception:  # dense-only transformers build
        pass
    col = GroupTraceCollector(n_windows)
    script = repo / "benchmark" / "quant" / "eval_quant.py"
    finished = "eval_completed"
    t0 = time.time()
    with col.installed(scm, moe_block_cls=moe_cls, embedding_cls=torch.nn.Embedding):
        sys.argv = [str(script)]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except _CaptureDone:
            finished = f"stopped_after_{n_windows}_windows"
        except SystemExit as e:  # eval_quant exits only on errors; a finished eval returns
            if e.code not in (None, 0):
                raise
            finished = f"eval_exit_{e.code}"
    secs = time.time() - t0
    windows = sorted(col.calls)
    if sc_trace is not None:
        path = sc_trace.flush(check_trace, header_extra=dict(
            truncated_capture=True, windows=windows, finished=finished,
            mp_config_json=os.environ.get("MP_CONFIG_JSON"),
            note="scmp-trace-summary of the group-trace capture windows only; NOT a "
                 "full-test eval trace"))
        sc_trace.disable()
        print(f"[gtrace] check trace -> {path}")
    header = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), finished=finished,
                  fp_matmul=fp, trajectory=TRAJECTORY[fp], n_windows_requested=n_windows,
                  capture_seconds=round(secs, 1),
                  coverage=COVERAGE, semantics=SEMANTICS, check_trace=check_trace,
                  **provenance(repo, col._config))
    col.save(out, header)
    n_groups = sum(v.size for v in col.maps.values())
    print(f"[gtrace] {finished} in {secs:.0f}s (fp_matmul={fp}): windows {windows}, "
          f"{len(col.maps)} group maps, {n_groups} groups, "
          f"{sum(len(c) for c in col.calls.values())} calls -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
