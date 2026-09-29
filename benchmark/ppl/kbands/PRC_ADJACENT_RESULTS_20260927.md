# Round 4 (+ 30B companion) adjacent-rung allocation: RESULTS

Written 2026-09-28 from the finished artifacts. Protocol: [`PRC_ADJACENT_20260927.md`](PRC_ADJACENT_20260927.md) (unchanged).
Diagnosis of why this round could not have found anything: [`investigations/T1_STALL_INVESTIGATION_20260928.md`](investigations/T1_STALL_INVESTIGATION_20260928.md) §2.

**Outcome: best(all) is unchanged in all four cells.** The only confirmed candidate (llama8B t32) lost on the full test.

## Runs

| array | tasks | cells | finished (EDT, 2026-09-27) |
|---|---|---|---|
| **62104342** (round 4) | 0, 1 | `4B_t32`, `llama8B_t32` | 15:20, 16:10 (the latter includes the full test) |
| **62105189** (round 4's 30B companion) | 0, 1 | `30B_t32`, `30B_t40` | 17:22, 17:36 |

Some notes call array 62105189 "round 5". Its manifest (`prc_adjacent_30b_20260927.json`, `scope_extension`) files it as a
round-4 extension. Same driver, generator, gates and source hashes as 62104342. Nothing else is labelled round 5.

## Per-cell outcome

Paired ΔPPL = 100·expm1(mean ΔNLL) of candidate minus incumbent on the same windows.
"Moved" is the proposal's `transferred_fraction` of total SC cycles. Every one of the 48 proposals requested 0.25%; the 0.5%
variants were de-duplicated.

| cell | incumbent (full-test PPL) | selected candidate | moved | search, 6 TRAIN windows | confirmation, 16 fresh windows | full test | outcome |
|---|---|---|---:|---|---|---|---|
| 4B t32 | `c6_gfis` (11.189097) | `candidate01_adj_linears_from_qk_01` | 0.061% | −0.446%, z −1.23 | **+0.138%**, z +0.48 | not run | **incumbent retained** |
| llama8B t32 | `c6_gfis` (8.279485) | `candidate09_adj_within_o_proj_09` | 0.019% | −0.749%, z −2.19 | −0.398%, z −1.98 (passed z < −1.5) | **8.285860 vs 8.279485 = +0.077%** | **confirmed, then lost on test; incumbent retained** |
| 30B t32 | `c7_gfisla` (8.581178) | `candidate11_adj_within_down_proj_11` | 0.020% | −1.138%, z −2.53 | −0.329%, z −0.68 | not run | **incumbent retained** |
| 30B t40 | round-3 `candidate07_mlp_from_projections` (8.085611) | `candidate07_adj_mlp_from_projections_07` | 0.024% | −0.644%, z −2.00 | −0.079%, z −0.27 | not run | **incumbent retained** |

**Checks that passed in every cell:**
- The no-op, profiled and restored incumbent evaluations matched exactly in NLL and traced cost (`identity.json`, `exact_match: true`).
- `identity_before` and `identity_after` gave equal window NLLs.
- Cost and locality were feasible at confirmation. Exact-trace cost change for the confirmed candidate vs the incumbent:
  −0.052% (4B), −0.037% (llama8B), +0.013% (30B t32), −0.013% (30B t40).

**Full test of the llama8B t32 candidate:**
- Full WikiText-2 test, ctx 2048, stride 2048, `ppl_max_tokens=0`, batch 1, 288,627 tokens, bitrev, 64 masks, sc_prec 8,
  halved. The run took 3,336 s.
- Candidate cost 33.849339 vs incumbent 33.863100 (−0.041%). The driver chose the incumbent
  (`candidate_beats_incumbent: false`).

## Round-level numbers

These come from the investigation (§2b–§2c), re-checked against the files below.
- **Search.** 39 of 48 candidates had a negative 6-window search delta. Realized moves were 0.0048–0.0685% of SC cycles.
- **Why the search "wins" are noise.**
  - None of the 48 falls below the null best-of-12 5th percentile.
  - Confirmations kept on average about 16% of the search effect.
  - The expected effect of a move this size is ≤0.02%, against a confirmation MDE80 of 0.47–1.14%.
- **Cascade.** Each threshold edit re-dispatches the downstream per-call min–max normalized buckets. The median gross cycle change
  was 3–15× the planned move.
- **Best(all) file.** `hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.{json,md,csv}` (dated 2026-09-27 16:00 UTC)
  predates these completions but needs no change: no round-4 candidate beat its incumbent on test. The llama8B t32 candidate is a
  comparable pure-allocation full-test row (8.285860). It loses to the incumbent and does not enter the headline.

## Sources

- Per-cell outputs: `/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_adjacent_20260927/<cell>/`:
  - `selected.json` (search + confirmation stats, gate booleans, reason);
  - `search_results.json`;
  - `search_*_nll.json`, `confirm_{incumbent,candidate}_nll.json`;
  - `identity*.json`, `windows.json`, `provenance.json`, `job.log`.
- llama8B t32 full test:
  - result: `prc_adjacent_20260927/llama8B_t32/full_test_result.json`;
  - trace: `/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl/llama8B_t32_prc_p4adjacent20260927_trace.json`.
- Slurm logs in `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/`:
  - `p4adjacent20260927_62104342_{0,1}.out`;
  - `p4adj30b20260927_62105189_{0,1}.out`.
- Manifests and submission records, all unchanged:
  - `prc_adjacent_20260927{,_submission}.json`;
  - `prc_adjacent_30b_20260927{,_submission}.json`.
- Frozen round-4 sources: `benchmark/ppl/prc_adjacent_{refine,proposals}.py` and calib7. **Do not edit.** Later rounds copy or
  import from them.
