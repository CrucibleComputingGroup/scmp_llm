"""CPU checks for the per-group precision trace (no model/GPU required).

    python -m unittest benchmark.ppl.test_group_trace -v      (from scmp_llm/)

The real SCLinear / per-(row,chunk) rungs / attention classifier / escape gate / hybrid INT
backend and the real sc_matmul entry (validation + scmp_kernels.trace recording) run; only
the Triton kernels behind sc_matmul are replaced by FP matmuls that record what they
executed.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class GroupTraceTests(unittest.TestCase):
    def test_capture_matches_kernels_and_trace(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        import model.sc_common as scm
        import scmp_kernels.sc.matmul as scmm
        from scmp_kernels import trace as sc_trace
        from scmp_kernels.mp.config import AdaptiveMPConfig
        from benchmark.ppl.group_trace import GroupTraceCollector, _CaptureDone
        from benchmark.ppl import group_trace_check as chk

        mp = AdaptiveMPConfig([32, 16, 8])
        mp.per_row_chunk = {}
        mp.bucket_thresholds = {("q_proj", 0, 0): [.75, .25], ("qk", 0, 0): [.75, .25]}
        mp._load_per_row_chunk({"buckets": {"q_proj:t0:l0": {
            "levels": [8, 16, 32], "thresholds": [.25, .75]}}})
        mp.protected_channel_indices = {("q_proj", 0, None): [0, 1, 2, 3]}
        mp.protected_channel_stoc_len = 64
        mp.escape_gate_k = 2.0
        mp.escape_stoc_len = 64
        mp.bucket_escape_thresholds = {("qk", 0, 0): .8}
        cfg = SimpleNamespace(sc_mp_config=mp, use_sc_linear=True, _sc_total_blocks=2,
                              sc_linear_chunk_d=128, sc_halve_bipolar_stoc_len=True,
                              sc_hybrid_schedule={("k_proj", 0): "int7", ("av", 0): "int7"},
                              sc_hybrid_default="sc", sc_hybrid_int_bits=7)
        q = scm.SCLinear(261, 5, bias=False, sc_config=cfg)    # residual 257 = 128+128+1
        q._sc_op_name, q._sc_block_idx, q._sc_unit_idx = "q_proj", 0, None
        k = scm.SCLinear(261, 3, bias=False, sc_config=cfg)    # INT7 by the schedule
        k._sc_op_name, k._sc_block_idx, k._sc_unit_idx = "k_proj", 0, None
        emb = torch.nn.Embedding(50001, 2)
        g = torch.Generator().manual_seed(0)
        windows = []
        for _ in range(3):
            windows.append(dict(
                ids=torch.randint(0, 50001, (1, 6), generator=g),
                x=torch.randn(6, 261, generator=g) * torch.linspace(0.2, 3.0, 261),
                a=torch.randn(1, 2, 5, 17, generator=g) * torch.tensor([.2, 1., 3., .5, 2.])[:, None],
                b=torch.randn(1, 2, 7, 17, generator=g),
                p=torch.softmax(torch.randn(1, 2, 5, 7, generator=g), -1),
                v=torch.randn(1, 2, 17, 7, generator=g)))
        kw = dict(mode="bipolar", sc_prec=8, stoc_len=32, halve_bipolar_stoc_len=True,
                  mp_config=mp, operator="qk", block_idx=0, total_blocks=2)

        executed = []               # (kind, left, right, lengths or stoc_len)

        def mlp_kernel(a, b, **o):
            if o.get("rung_table") is not None:
                lens = torch.tensor(o["level_lens"])[o["rung_table"].long()]
                executed.append(("prc", a.clone(), b.clone(), lens))
            else:
                executed.append(("fixed", a.clone(), b.clone(), o["stoc_len"]))
            return a @ b.T

        def row_kernel(a, b, **o):
            executed.append(("row", a.clone(), b.clone(), o["stoc_len"]))
            return a @ b.T

        def run(n):
            outs = []
            for w in windows[:n]:
                emb(w["ids"])
                outs += [q(w["x"]), k(w["x"]), scm._sc_attention_matmul_ab_t(w["a"], w["b"], **kw),
                         scm._int_attention_matmul_ab_t(w["p"], w["v"], bits=7, sym=True,
                                                        chunk_size=128)]
            return outs

        hooks = lambda: (scm.SCLinear.forward, scm.per_row_chunk_rungs,
                         scm._sc_attention_matmul_ab_t, scm.adaptive_classify_rows,
                         scm._int_attention_matmul_ab_t, torch.nn.Embedding.forward)
        old = hooks()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scm, "_HAS_SC", True), \
                patch.object(scmm, "_sc_matmul_per_row_mlp", mlp_kernel), \
                patch.object(scmm, "_sc_matmul_per_row", row_kernel), torch.no_grad():
            plain = run(2)
            executed.clear()
            state = torch.random.get_rng_state().clone()
            col = GroupTraceCollector(2)
            trace_path = str(Path(tmp) / "sc_trace.json")
            sc_trace.reset()
            sc_trace.enable(trace_path, mode="summary")
            try:
                with col.installed(scm, embedding_cls=torch.nn.Embedding):
                    hooked = run(2)
                    n_exec = len(executed)
                    with self.assertRaises(_CaptureDone):
                        run(1)      # the next window's first q_proj call stops the capture
                    self.assertEqual(len(executed), n_exec, "no kernel may run after the stop")
            finally:
                sc_trace.flush(trace_path)
                sc_trace.disable()
            self.assertEqual(old, hooks())
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertTrue(all(torch.equal(p, h) for p, h in zip(plain, hooked)))

            # Linear maps == the rung tables the kernel executed (window order).
            prc = [e for e in executed if e[0] == "prc"]
            self.assertEqual(len(prc), 2)
            for w in range(2):
                got = torch.as_tensor(col.maps[f"w{w}/b0/q_proj"]).long()
                self.assertTrue(torch.equal(got, prc[w][3]))
            # Attention maps == the per-level kernel calls, rows identified by exact match.
            for w in range(2):
                a, b = windows[w]["a"].reshape(2, 5, 17), windows[w]["b"].reshape(2, 7, 17)
                want = torch.zeros(2, 5, dtype=torch.long)
                for kind, left, right, sl in executed:
                    if kind != "row":
                        continue
                    for h in range(2):
                        if not torch.equal(right, b[h]):
                            continue
                        for row in left:
                            j = next((j for j in range(5) if torch.equal(row, a[h, j])), None)
                            if j is not None:
                                want[h, j] = sl
                self.assertTrue(bool((want > 0).all()))
                self.assertTrue(torch.equal(torch.as_tensor(col.maps[f"w{w}/b0/qk"]).long(), want))
            self.assertGreater(len({int(v) for v in col.maps["w0/b0/qk"].reshape(-1)}), 1,
                               "fixture must mix attention lengths")

            # Calls: execution order, INT calls with shapes and no groups.
            for w in range(2):
                calls = col.calls[w]
                self.assertEqual([(c["seq"], c["op"], c["backend"], c["groups"]) for c in calls],
                                 [(0, "q_proj", "sc", f"w{w}/b0/q_proj"),
                                  (1, "k_proj", "int7", None),
                                  (2, "qk", "sc", f"w{w}/b0/qk"),
                                  (3, "av", "int7", None)])
                self.assertEqual((calls[3]["heads"], calls[3]["rows"], calls[3]["d_in"],
                                  calls[3]["d_out"]), (2, 5, 7, 17))
                self.assertTrue(torch.equal(torch.as_tensor(col.tokens[w]).long(),
                                            windows[w]["ids"][0]))
            desc = col.ops[("q_proj", 0, None)]
            self.assertEqual((desc["chunk_widths"], desc["protected_cols"],
                              desc["protected_stoc_len"], desc["smoothed"]),
                             ([128, 128, 1], 4, 64, False))
            self.assertEqual(col.ops[("qk", 0, None)]["ladder"], [32, 16, 8, 64])

            # Save / load round trip, then the identity against the real trace summary.
            col.save(Path(tmp) / "gt", dict(model="stub", ctx=6, sc=dict(rng_levels=128)))
            meta, arrays, ops = chk.load_group_trace(Path(tmp) / "gt")
            groups = chk.summary_from_groups(meta, arrays, ops)
            _, trace = chk.summary_from_trace(trace_path)
            ident = chk.identity(groups, trace)
            self.assertTrue(ident["ok"], json.dumps(ident, indent=1, default=str))
            self.assertGreater(ident["n_keys"], 5)
            # Flat tile avg_sl == the MAC-weighted mean of the summary.
            flat = chk.tile_avg_sl(meta, arrays, ops)["1"]["all"]
            self.assertAlmostEqual(flat, chk.per_op_avg_sl(groups)["all"], places=9)
            # A capture agrees with itself everywhere.
            cmp = chk.compare((meta, arrays, ops), (meta, arrays, ops))
            self.assertTrue(cmp["dense_call_structure_identical"])
            self.assertEqual(cmp["per_op"]["all"]["same_frac"], 1.0)
            arrays.close()


class CompareTests(unittest.TestCase):
    def test_rung_distance_and_mac_weights(self):
        import numpy as np
        from benchmark.ppl import group_trace_check as chk
        ops = {("q_proj", 0, None): dict(chunk_widths=[128, 64], chunk_d=128,
                                         ladder=[8, 16, 32, 64], protected_cols=0)}
        call = dict(seq=0, block=0, op="q_proj", unit=None, backend="sc", rows=2, d_in=192,
                    d_out=10, groups="w0/b0/q_proj")
        meta = dict(calls={"0": [call]})
        e = {"w0/b0/q_proj": np.array([[8, 16], [32, 64]], np.uint8)}
        f = {"w0/b0/q_proj": np.array([[8, 32], [8, 64]], np.uint8)}

        class A(dict):
            files = property(lambda self: list(self))
        cmp = chk.compare((meta, A(f), ops), (meta, A(e), ops))["per_op"]["q_proj"]
        self.assertEqual(cmp["same_frac"], 0.5)
        self.assertEqual(cmp["within_1_rung_frac"], 0.75)     # 16->32 is one rung, 32->8 two
        macs = np.array([[1280, 640], [1280, 640]], float)
        self.assertAlmostEqual(cmp["mac_same_frac"], (1280 + 640) / macs.sum())
        self.assertAlmostEqual(cmp["dispatched_avg_sl_exact"],
                               (8 * 1280 + 16 * 640 + 32 * 1280 + 64 * 640) / macs.sum())
        self.assertAlmostEqual(cmp["dispatched_avg_sl_fp"],
                               (8 * 1280 + 32 * 640 + 8 * 1280 + 64 * 640) / macs.sum())

    def test_moe_rows_join_on_token(self):
        """Expert rows follow each capture's routing (HF gather order: top-k slot major,
        then token), so the comparison must join them on token, not on position."""
        import numpy as np
        from benchmark.ppl import group_trace_check as chk

        class A(dict):
            files = property(lambda self: list(self))
        ops = {("up_proj", 0, e): dict(chunk_widths=[128], chunk_d=128, ladder=[8, 16, 32],
                                       protected_cols=0) for e in range(3)}
        per_token = np.array([[8], [16], [32], [16]], np.uint8)   # same in both captures
        sel_e = np.array([[0, 1], [1, 0], [0, 2], [2, 1]], np.int16)
        sel_f = np.array([[1, 0], [1, 0], [0, 2], [2, 0]], np.int16)   # token 3: 1 -> 0

        def capture(sel):
            arrays, calls = {"w0/b0/router_selected": sel}, []
            for e in range(3):
                tok = chk.expert_tokens(sel, e)
                key = f"w0/b0/up_proj/e{e}"
                arrays[key] = per_token[tok]
                calls.append(dict(seq=e, block=0, op="up_proj", unit=e, backend="sc",
                                  rows=len(tok), d_in=128, d_out=4, groups=key))
            return dict(calls={"0": calls}), A(arrays), ops
        self.assertEqual(chk.expert_tokens(sel_e, 0).tolist(), [0, 2, 1])
        self.assertEqual(chk.expert_tokens(sel_f, 0).tolist(), [2, 0, 1, 3])
        cmp = chk.compare(capture(sel_f), capture(sel_e))
        self.assertTrue(cmp["dense_call_structure_identical"])
        self.assertEqual(cmp["per_op"]["up_proj"]["same_frac"], 1.0)
        self.assertEqual(cmp["unmatched"], {"exact_only_groups": 1, "fp_only_groups": 1})
        self.assertAlmostEqual(cmp["routing_same_set_frac"], 0.75)


if __name__ == "__main__":
    unittest.main()
