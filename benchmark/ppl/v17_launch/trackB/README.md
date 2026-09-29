# V17 Track B — fallback measured search, fixed acceptance (launch-ready)

Stage-2 engine = the V16.1 paired-window search with the V17 acceptance
protocol. Code was already present in `benchmark/ppl/mp_v16_refine.py`
(applied by a prior session; verified here, not re-implemented).

## The three V17 changes (verified in code + tests)
1. **Accept on A->B replication.** `replication_accept_gate` (docstring "Sole
   acceptance authority (V17)") accepts iff the stage-A winner's sign replicates
   on disjoint half B (mean_b < 0) AND the pooled A+B mean clears the
   pooled-sigma significance threshold. In the sweep loop the winner is accepted
   (`accepted_this_sweep = True`) BEFORE the confirm eval; `if not accept:
   continue` short-circuits without touching the confirm windows.
2. **16-window confirm is STOP-ONLY.** `confirm_stop_check` has no "accept" key;
   it can only set `stop_reason = "confirm_regression"`. Proven by
   `test_confirm_stop_check_cannot_accept`. `--patience` default raised 2 -> 3;
   branches pass `--patience 4`. Resume from checkpoint verifies a 1-window
   incumbent-NLL bit-identity probe; walltime_guard summaries continue (not
   skipped).
3. **New `lift_compound` macro family** (`propose_lift_compound`, quota 3):
   emits the measured-direction compound as single candidates —
   `lift_full` ({8 attention buckets -> top rung 96} x {psl -> 95} x {MLP floor
   raises to iso-cost}) and `lift_psl_reinvest` (psl 95 + MLP reinvest). Offline
   smoke vs the V9 4B table: both iso-cost 31.24 (profile currency ~= parent
   realized 31.31), levels unchanged [96,64,48,32,24,16], psl=95 (never 96), no
   rung < 16. Threshold family OFF by default; surrogate ON.

## Tests (all green)
`python -m unittest benchmark.ppl.test_mp_v16_refine benchmark.ppl.test_mp_v16_refine_flow`
— 33 tests incl. `test_replication_gate_{accepts_on_sign_replication,
rejects_on_b_sign_flip,rejects_below_pooled_significance}`,
`test_confirm_stop_check_{cannot_accept,stop_semantics}`. Full ppl suite: 77 pass.

## Branches (1 GPU, 24h, tag mp_v17_fbsearch_20260719_033500, --patience 4)
Each seeds from its V9 avg32 parent (lane glob; NOT a transplant cell).
| script | model | target | parent (V9 avg32) |
|---|---|---|---|
| fb_4B_t32.sbatch  | 4B      | 32 | Qwen_Qwen3-4B-Instruct-2507 |
| fb_4B_t36.sbatch  | 4B      | 36 | Qwen_Qwen3-4B-Instruct-2507 |
| fb_l8B_t40.sbatch | llama8B | 40 | meta-llama_Llama-3.1-8B-Instruct |
| fb_14B_t32.sbatch | 14B     | 32 | Qwen_Qwen3-14B |

Auto-resubmit: the internal 22.5h walltime guard writes a resumable
`walltime_guard` summary before the 24h SBATCH limit; the wrapper then
resubmits itself from checkpoint (bounded to 3 continuations). Per-target
summary.json is authoritative (shared results.tsv can be clobbered by parallel
jobs). Full-test backfill on selected branch ends stays enabled (lane
`--test-tokens 0`). Compare winners vs V9 finals at realized cost
(V9: 4B 13.784@31.31, llama8B 10.488@32.04, 14B 9.837@32.15).

Scripts: `/nfs/turbo/coe-nbleier/allenjin/hpca/mp_v17_fbsearch_20260719_033500/sbatch/`
Outputs: `.../mp_v16_refine_mp_v17_fbsearch_20260719_033500/<model>/target<N>p000/`
