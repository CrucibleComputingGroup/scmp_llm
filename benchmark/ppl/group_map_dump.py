"""Per-group stream-length maps: observe-only capture of the DEPLOYED dispatch.

For the first few evaluation windows, record the stream length the deployed code assigns
to every computation group, at every layer (T7 figure):

  linears     one length per (token row, 128-chunk) of A = the operator input, protected
              columns excluded (they run at one fixed length). For a MoE expert the rows
              are that expert's routed tokens, in the order the HF block gathers them.
  attention   one length per (head, query row) for QK^T and AV.
  MoE routing each block's top-k expert selection, so expert rows map back to token
              positions (HF 4.51 Qwen3MoeSparseMoeBlock: rows = torch.where(mask[e]),
              i.e. ordered by top-k slot, then token).

The four hook points and the index->length maps are the ones tile_cost.py validated
(flat avg_sl == the archived trace cost). Nothing touches dispatch or numerics.

This is a capture, not an evaluation: it stops once the last requested window has been
dispatched and reports no perplexity. DUMP_FP_MATMUL=1 replaces SC matmuls by exact FP
matmuls (dispatch still runs, on the FP trajectory) and is off by default.

Usage -- the environment of kbands/run_prc_ppl.sbatch, plus
    DUMP_OUT=/path/group_map.npz  [DUMP_WINDOWS=4]  [DUMP_FP_MATMUL=0]
    python -u benchmark/ppl/group_map_dump.py
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTENTION = ("qk", "av")


class _DumpDone(BaseException):
    """Raised on the first call of the window after the last requested one."""


class GroupMapCollector:
    def __init__(self, n_windows):
        self.n_windows = int(n_windows)
        self.window = -1          # incremented on every block-0 q_proj call (any backend)
        self.maps = {}            # key string -> uint8 array
        self.meta = {}            # (op, block, unit) -> chunk widths / protected columns
        self.routing = {}         # (window, block) -> (selected int16, weights float16)
        self.tokens = {}          # window -> int32 token ids
        self._linear = None
        self._attention = None

    def _keep(self):
        return 0 <= self.window < self.n_windows

    @staticmethod
    def _key(w, block, op, unit):
        return f"w{w}/b{block}/{op}" + ("" if unit is None else f"/e{unit}")

    @contextmanager
    def installed(self, scm, moe_block_cls=None, embedding_cls=None):
        import torch
        orig = (scm.SCLinear.forward, scm.per_row_chunk_rungs,
                scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows)
        old_linear, old_prc, old_attention, old_classify = orig
        old_moe = moe_block_cls.forward if moe_block_cls is not None else None
        old_emb = embedding_cls.forward if embedding_cls is not None else None
        col = self

        def linear(mod, x):
            op = getattr(mod, "_sc_op_name", None)
            block = getattr(mod, "_sc_block_idx", None)
            unit = getattr(mod, "_sc_unit_idx", None)
            if op == "q_proj" and block == 0:
                col.window += 1
                if col.window >= col.n_windows:
                    raise _DumpDone()
            cfg = mod._sc_config
            if (not getattr(cfg, "use_sc_linear", True) or x.numel() == 0
                    or scm._hybrid_backend(cfg, op, block) != "sc"):
                return old_linear(mod, x)
            mp = getattr(cfg, "sc_mp_config", None)
            if (op not in LINEARS or mp is None
                    or getattr(cfg, "sc_group_stoclen", None) is not None
                    or getattr(cfg, "sc_ste_grad", False)):
                raise ValueError(f"group map needs deployed adaptive PRC linears ({op}/{block})")
            protected = mp.get_protected_channels(operator=op, block_idx=block, unit_idx=unit)
            count = int(scm._channel_index_tensor(protected, x.shape[-1], x.device).numel())
            previous = col._linear
            col._linear = [mod, op, block, unit, count, False]
            try:
                result = old_linear(mod, x)
                if count < x.shape[-1] and not col._linear[-1]:
                    raise ValueError(f"SC linear without a PRC dispatch: {op}/{block}")
                return result
            finally:
                col._linear = previous

        def prc(x, chunk_d, levels, thresholds=None, target=0.0):
            result = old_prc(x, chunk_d, levels, thresholds=thresholds, target=target)
            if col._linear is None:
                raise ValueError("PRC dispatch outside an SCLinear call")
            mod, op, block, unit, count, seen = col._linear
            if seen:
                raise ValueError("multiple PRC dispatches in one SCLinear call")
            col._linear[-1] = True
            n, d = x.shape
            if n and col._keep():
                lengths = torch.as_tensor(list(levels), device=x.device)[result.long()]
                if int(lengths.max()) > 255 or int(lengths.min()) <= 0:
                    raise ValueError(f"stream length outside uint8 range at {op}/{block}")
                col.maps[col._key(col.window, block, op, unit)] = (
                    lengths.to(torch.uint8).cpu().numpy())
                mk = (op, block, unit)
                if mk not in col.meta:
                    nch = result.shape[1]
                    widths = [chunk_d] * (nch - 1) + [d - (nch - 1) * chunk_d]
                    fixed = int(getattr(mod._sc_config.sc_mp_config,
                                        "protected_channel_stoc_len", None)
                                or max(mod._sc_config.sc_mp_config.stoc_len_levels))
                    col.meta[mk] = dict(chunk_widths=widths, protected_columns=count,
                                        protected_len=fixed if count else None,
                                        out_features=int(mod.out_features))
            return result

        def attention(a, b, **kw):
            previous = col._attention
            col._attention = [tuple(a.shape), tuple(b.shape), kw, False]
            try:
                result = old_attention(a, b, **kw)
                if not col._attention[-1]:
                    raise ValueError("SC attention without adaptive row classification")
                return result
            finally:
                col._attention = previous

        def classify(metric, config, *args, **kw):
            result = old_classify(metric, config, *args, **kw)
            if col._attention is None:
                return result          # the linears' unused per-row assignment
            a_shape, b_shape, akw, seen = col._attention
            op = akw.get("operator")
            if op not in ATTENTION or kw.get("operator") != op or seen:
                raise ValueError("unrecognized attention dispatch context")
            col._attention[-1] = True
            if col._keep():
                values = config.classify_level_values(
                    operator=op, block_idx=akw.get("block_idx"),
                    total_blocks=akw.get("total_blocks"))
                lengths = torch.as_tensor(list(values), device=metric.device)[
                    result.row_levels.long()]
                bsz, heads, n, _ = a_shape
                col.maps[col._key(col.window, akw.get("block_idx"), op, None)] = (
                    lengths.view(bsz * heads, n).to(torch.uint8).cpu().numpy())
            return result

        def moe_forward(blk, hidden_states):
            if col._keep():
                with torch.no_grad():
                    hs = hidden_states.reshape(-1, hidden_states.shape[-1])
                    logits = blk.gate(hs)
                    w = torch.softmax(logits, dim=1, dtype=torch.float)
                    w, sel = torch.topk(w, blk.top_k, dim=-1)
                    if blk.norm_topk_prob:
                        w = w / w.sum(dim=-1, keepdim=True)
                block = getattr(blk.experts[0].gate_proj, "_sc_block_idx", None)
                col.routing[(col.window, block)] = (
                    sel.to(torch.int16).cpu().numpy(), w.to(torch.float16).cpu().numpy())
            return old_moe(blk, hidden_states)

        def embedding(emb, ids):
            if (ids.dim() == 2 and ids.shape[0] == 1 and emb.num_embeddings > 50000
                    and 0 <= col.window + 1 < col.n_windows):
                # called before block 0 of the NEXT window
                col.tokens[col.window + 1] = ids[0].to(torch.int32).cpu().numpy()
            return old_emb(emb, ids)

        (scm.SCLinear.forward, scm.per_row_chunk_rungs,
         scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows) = (
            linear, prc, attention, classify)
        if moe_block_cls is not None:
            moe_block_cls.forward = moe_forward
        if embedding_cls is not None:
            embedding_cls.forward = embedding
        try:
            yield self
        finally:
            (scm.SCLinear.forward, scm.per_row_chunk_rungs,
             scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows) = orig
            if moe_block_cls is not None:
                moe_block_cls.forward = old_moe
            if embedding_cls is not None:
                embedding_cls.forward = old_emb

    def save(self, out, extra):
        arrays = dict(self.maps)
        for (w, block), (sel, wt) in self.routing.items():
            arrays[f"w{w}/b{block}/router_selected"] = sel
            arrays[f"w{w}/b{block}/router_weights"] = wt
        for w, ids in self.tokens.items():
            arrays[f"w{w}/token_ids"] = ids
        np.savez_compressed(out, **arrays)
        meta = dict(extra, windows_captured=sorted({int(k.split("/")[0][1:]) for k in self.maps}),
                    n_maps=len(self.maps),
                    groups={f"{op}/b{b}" + ("" if u is None else f"/e{u}"): v
                            for (op, b, u), v in sorted(self.meta.items(),
                                                        key=lambda t: (t[0][1], t[0][0],
                                                                       -1 if t[0][2] is None
                                                                       else t[0][2]))})
        Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))


def main() -> int:
    out = os.environ.get("DUMP_OUT")
    if not out:
        sys.exit("[gmap] DUMP_OUT is required")
    n_windows = int(os.environ.get("DUMP_WINDOWS", "4"))
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "kernels")]
    import torch
    import model.sc_common as scm
    fp = os.environ.get("DUMP_FP_MATMUL", "0") == "1"
    if os.environ.get("SC_MP_TRACE"):
        sys.exit("[gmap] unset SC_MP_TRACE: a truncated capture must not write a trace")
    if fp:
        def fp_matmul(a, b, **_):
            # (a/s)(b*s)^T == a b^T exactly, so AWQ smooth_scales need no handling.
            return a.float() @ b.float().t()
        scm._sc_matmul = fp_matmul
    moe_cls = None
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe as qm
        moe_cls = qm.Qwen3MoeSparseMoeBlock
    except Exception:  # dense-only transformers build
        pass
    col = GroupMapCollector(n_windows)
    script = repo / "benchmark" / "quant" / "eval_quant.py"
    finished = "eval_completed"
    with col.installed(scm, moe_block_cls=moe_cls, embedding_cls=torch.nn.Embedding):
        sys.argv = [str(script)]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except _DumpDone:
            finished = f"stopped_after_{n_windows}_windows"
        except SystemExit as e:  # eval_quant ran to the end (fewer windows than asked)
            finished = f"eval_exit_{e.code}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    col.save(out, dict(finished=finished, fp_matmul=fp, n_windows_requested=n_windows,
                       model=os.environ.get("MODEL_PATH"),
                       mp_config_json=os.environ.get("MP_CONFIG_JSON"),
                       hybrid_config_json=os.environ.get("SC_HYBRID_CONFIG_JSON"),
                       frontend=os.environ.get("FRONTEND"), ctx=os.environ.get("CTX")))
    print(f"[gmap] {finished}; {len(col.maps)} maps, {len(col.routing)} routing tables, "
          f"{len(col.tokens)} token windows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
