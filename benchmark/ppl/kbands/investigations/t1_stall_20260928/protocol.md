# T1 per-group allocation: search/selection protocol audit of rounds 3–5

Date 2026-09-27. Read-only, CPU-only analysis of existing JSON/trace artifacts. No jobs, no model loads.
Helper scripts and intermediate outputs are in this directory (`trace_headers.py`, `heldout_vs_test.py`,
`trace_moves.py`, `trace_macmoves.py`, and `*.json`). All %PPL values are `100*ΔNLL` or `100*ln(PPL_a/PPL_b)`.
They are equal to relative PPL change to within rounding at these sizes.

## 0. Which runs are "rounds 3, 4, 5"

| label used here | run | Slurm | cells | incumbent |
|---|---|---|---|---|
| R3 (coarse shift) | `prc_local_20260926` | 62032412 | 4B_t32, llama8B_t32, 30B_t40 | c6gfis / c6gfis / c7gfisla |
| R4 (adjacent, dense) | `prc_adjacent_20260927` | 62104342 | 4B_t32, llama8B_t32 | c6gfis |
| R5 (adjacent, 30B) | `prc_adjacent_30b_20260927` | 62105189 | 30B_t32, 30B_t40 | c7gfisla / R3 cand07 |

The docs call the 30B companion part of "round 4". Treating it as the user's "round 5" is my inference. No
other post-round-2 job exists: the newest `_kbands` logs are `p4adj30b20260927_62105189_{0,1}.out`
(17:22/17:36 today). "Claude's round 3" (c8 robust/per-block/re-linearized/attention-ceiling,
PRC2_OVERNIGHT.md:328-435) is older and was screened on held-out windows only, never on full test.

## 1. Bottom line

1. **R4/R5 could not have produced or detected a >=1% gain.** Their candidates moved **0.005–0.068% of total
   SC cycles net (<=0.14% gross), and changed the stream length of only 0.04–0.53% of MACs**. That is about 35–1000x
   smaller than every historical >=1% difference. The largest plausible true effect is about 0.01% PPL, and even
   under extreme (20x) mispricing it is only about 0.1–0.4%. The confirmation MDE was about 0.6–0.9%, so the
   search was a lottery over null candidates. All 48 search results match the null winner's-curse expectation.
   The one "confirmed" win (llama t32, −0.40%, z=−1.98) came out **+0.077% on full test**.
2. **R3 had big enough steps (about 2% of cycles) but no direction information.** Its candidates were
   whole-population common threshold offsets, nonlocal enough to skip rungs. 19/20 made loss worse, by up to +17.6%.
   R3 proves that moves of this size have measurable effects (about 0.5–2% PPL). It also shows that the
   generator chose directions blind.
3. **The evaluator is not the bottleneck.** Earlier rounds show that 16 paired TRAIN windows predict full-test
   deltas well. Over 49 per-group-vs-per-group pairs: r=0.96, slope 0.94, sign agreement 46/49, and 28/28 when
   |z|>=1.5. The linear-Fisher surrogate also predicts test deltas: over 24 pairs, r=0.76, slope 0.96, sign 22/24.
   Rounds 3–5 did not fail because selection is unreliable. They failed because no candidate had a real effect
   of >=1%, and the generator used no loss model to find one.
4. **Neither proposal generator used any loss, error or Fisher information.** Both were purely
   histogram/cost-driven. R4/R5 took 12 candidates from pools of 54k–91k cost-matched pairs, ranked by
   (bucket reuse, **largest moved cycles**, cost-mismatch, lexicographic signature).
5. **The 1% exact-trace cost guard never bound** (max |cost ratio−1| = 0.54% in R3 and 0.35% in R4/R5).
   Every candidate was a net-zero exchange, so the user's "~1% cost doesn't matter" allowance was never used.

## 2. The designs as run (verified from code and artifacts)

| | R3 | R4/R5 |
|---|---|---|
| generator | `prc_local_proposals.py:342-407`: 4 fixed schemes × 2 directions (qk↔linears, av↔linears, qk↔av, projections↔MLP); the **same additive offset on every threshold of every bucket** of a population, bisected to cost (`_shift_to_cost`, :319) | `prc_adjacent_proposals.py:319-419`: one boundary in each of 2 buckets; neighbor-bounded (<=1 rung); category round-robin in fixed priority (:378); rank key `(usage, -moved, error, signature)` (:388-391) |
| loss model in proposal | none | none |
| step | ceiling 2% of total cycles; realized 0.91–2.00% (`search_results.json` `proposal.transferred_fraction`) | ceilings 0.25%/0.5% of total; **realized net 0.0047–0.068%**; gross (both sides) <=0.138%; changed MACs 0.04–0.53% of total |
| binding cap | population capacity | 45/96 moves at the 80%×5% group/MAC cap (>=3.9%); the rest limited by the cost-matching partner (`_match`, 10% mismatch, :300) or neighbor thresholds. Bucket-cost cap (max 1.87%) and operator-gross cap (max 0.86%) never bound; the 0.25%/0.5% ceilings never bound |
| windows | 6 TRAIN search + 16 fresh TRAIN confirm (disjoint), paired | same |
| selection | min search mean → confirm → full test **regardless** | min search mean among negative → confirm; full test only if `mean<0 and z<-1.5` (`prc_adjacent_refine.py:109,284`) |
| cost guard | exact trace within 1% (`within_cost`, `prc_local_refine.py:112`) | same |
| candidates | 20 (8/4/8) | 48 (12×4) |

Pool sizes, from `proposal_summary.json` `pool` in each R4/R5 cell:
- 4B: 628 single-boundary curves, 90,622 matched pairs.
- llama8B: 490 curves, 54,292 pairs.
- 30B t32: 524 curves, 64,740 pairs.
- 30B t40: 551 curves, 68,558 pairs.

12 candidates per cell is a 0.013–0.022% sample, chosen for size and diversity rather than predicted value.
Every category is also proposed in both directions (e.g. `qk_from_linears` and `linears_from_qk`). Near a
locally balanced incumbent, at most one direction of a pair can help to first order.

## 3. Noise model: what one paired window can resolve

The eval is bit-deterministic. Identity checks pass exactly: `identity.json` `exact_match` is true, and
`identity_before` equals `identity_after`. "Noise" is therefore the window-to-window heterogeneity of a
candidate's effect. Per-window SD of the paired ΔNLL (σ_w):

| source (move size) | 4B | llama8B | 14B | 30B |
|---|---|---|---|---|
| R4/R5 search, median over 12 (tiny moves, 6 windows) | 0.0100 | 0.0080 | — | 0.0114 / 0.0091 |
| R4/R5 confirm (16 windows) | 0.0116 | 0.0080 | — | 0.0195 / 0.0119 |
| R3 search median / confirm (2% moves) | 0.0164 / 0.0143 | 0.0189 / 0.0139 | — | 0.0242 / 0.0119 |
| earlier held-out, per-group pairs, median (Fisher re-solves; 9–52% of MACs change L), 16 windows | 0.0149 | 0.0113 | 0.0147 | 0.0133 |

Sources: `prc_adjacent_20260927/*/search_results.json`, `.../selected.json` (`confirmation`),
`prc_local_20260926/*/...`, and `kbands/prc2/*_heldout_nll.json` (computed in `heldout_vs_test.py`).

**Key fact: σ_w is almost independent of move size.** A move that changes 0.2% of MACs by one rung yields the
same ~0.01 per-window scatter as a Fisher re-solve that changes 30% of MACs. The effect grows with move size,
but the noise does not. So SNR is proportional to move size, and micro-moves are undetectable by construction.
The likely mechanism is that any change of the rounding pattern re-draws each window's SC error. This is an
inference: the magnitudes are verified, the mechanism is not.

Minimum detectable effect (80% power, one-sided z<−1.5 gate), in %PPL, where MDE = (1.5+0.84)·σ_w/√n:

| σ_w | n=6 | n=16 | n=32 | n=64 | full test n≈146 |
|---|---|---|---|---|---|
| 0.008 | 0.76 | 0.47 | 0.33 | 0.23 | 0.16 |
| 0.010 | 0.96 | 0.59 | 0.41 | 0.29 | 0.19 |
| 0.012 | 1.15 | 0.70 | 0.50 | 0.35 | 0.23 |
| 0.015 | 1.43 | 0.88 | 0.62 | 0.44 | 0.29 |

- Power for a **true 1%** effect at n=16 is 0.97–0.99 (σ_w 0.010–0.012). The 16-window confirmation is
  adequate for 1% effects.
- Power for a true 0.3% effect at n=16 is only 0.31–0.38.
- The full-test paired SE is about σ_w/√146 ≈ 0.08–0.10%. This matches `scmp_llm/CLAUDE.md:75`, which says
  "noise floor sd 0.0069" (≈0.083% at PPL 8.3). Test windows: 298,862 tokens for Qwen and 288,627 for Llama
  (trace headers), i.e. 146/141 windows of 2048.

## 4. Could the step sizes move PPL by 1% at all?

**Budget elasticity** e = −dlnPPL/dln(cost), measured between adjacent budgets of the best(all) arms
(`hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.json`, `best_ppl`/`best_cost`):

| model | t32→t40 | t40→t48 | t48→t64 | t64→t96 | uniform extra cycles ≡ 1% PPL at t32 |
|---|---|---|---|---|---|
| 4B | 0.183 | 0.159 | 0.058 | 0.030 | 5.5% |
| llama8B | 0.152 | 0.171 | 0.069 | 0.024 | 6.6% |
| 14B | 0.089 | 0.073 | 0.040 | 0.017 | 11% |
| 30B | 0.265 | 0.164 | 0.105 | 0.036 | 3.8% |

An iso-cost exchange of fraction f of total cycles, from a donor with zero marginal value to a recipient worth
k× the average cycle, yields at most k·e·f.

- **R4/R5.** f_net = 0.005–0.068%. With k=1: <=0.018%. With k=20 (the upper end of the historical 5–20×
  attention mispricing, PREREG_LOSS_WEIGHTED_OBJECTIVE.md §2): <=0.36% (30B t32, best case). Typical
  candidates (f≈0.02–0.03%) are <=0.1%.
- **Sigma-to-PPL cross-check.** At 0.33–0.36 %PPL per %σ (PREREG table; LOOP_STATE.md:2321), a typical move
  changes about 0.2% of MACs by ΔL/L≈15%. That is about 0.03% of σ per side, or about 0.01% PPL. For a 1% gain
  from R4/R5-sized moves you would need k = 1%/(e·f) ≈ 80–700.
- **R3.** f≈2%, so 1% needs k·e >= 0.5 (k≈2–5). That is possible in principle, but only for a well-aimed
  direction. The R3 directions were fixed a priori, and the realized moves were nonlocal (ROUND3_POSTMORTEM:
  llama QK own budget −20%, 25% of QK MACs jumping 96→19 deployed). Their measured true effects were large and
  harmful (+0.5% to +17.6%).
- **History.** Across 79 comparable test-arm pairs (`trace_moves.py`, `trace_macmoves.py`), every pair with
  |test ΔPPL| >= 1% (15 pairs) moved **>=5.8% of per-(block,op) SC cycles and >=18.6% of MACs changed stream
  length**. Examples: llama8B t32 c17→c6gfis −1.32% (5.79%, 28%); 30B t32 c7gfis→c7gfisla −1.56% (6.86%,
  18.6%); 30B t40 −2.84% (7.68%, 31.5%). All of these came from loss-model re-solves. Spearman(|ΔPPL|,
  moved cycles) = 0.39.
- **High budgets.** At t64/t96, 1% PPL is worth 30–60% extra uniform cycles. A >=1% pure-allocation gain is
  therefore implausible there under any protocol. Target t32–t48, where e = 0.07–0.27.

## 5. Search, confirmation and test agreement

**R4/R5 search results are exactly what 48 null candidates would produce.**
- Null best-of-12 simulation, using each candidate's own σ_w over 6 windows: E[min] = −0.99/−0.71/−0.69/−0.64%
  (30B t32 / 30B t40 / 4B / llama).
- Observed minima: −1.14/−0.65/−0.45/−0.75%.
- 39/48 search deltas are negative, with cell means of −0.45/−0.18/−0.16/−0.22%. The average pairwise
  correlation of the window-difference vectors is +0.32/+0.17/+0.01/+0.17. This is consistent with a
  common-mode term: the incumbent's own per-window idiosyncrasy enters every paired difference with the same
  sign. It is not evidence of a broad improvement direction.

Selected candidates (search → confirm → test, %PPL):

| round | cell | search (6w) | confirm (16w, z) | full test |
|---|---|---|---|---|
| R3 | 4B t32 | +0.545 | +0.431 (+1.21) | +0.212 |
| R3 | llama8B t32 | +0.665 | +0.354 (+1.02) | +0.342 |
| R3 | 30B t40 | −0.052 | −0.045 (−0.15) | −0.030 |
| R4 | 4B t32 | −0.447 | +0.138 (+0.48) | not run |
| R4 | llama8B t32 | −0.752 | −0.398 (−1.98) | **+0.077** (8.285860 vs 8.279485) |
| R5 | 30B t32 | −1.145 | −0.329 (−0.68) | not run |
| R5 | 30B t40 | −0.646 | −0.079 (−0.27) | not run |

Search/confirm sign agreement is 6/7, with r=0.90 across all 7. That correlation is driven by R3's genuinely
harmful candidates. In R4/R5 the confirmation keeps on average **16%** of the search effect (pure shrinkage).

The llama false positive is expected at this rate. Under the null, a single confirmation passes z<−1.5 with
p=0.067 per cell, so P(>=1 of 4 cells passes) ≈ 24%. If the −0.40% had been real, the test result (SE≈0.07%)
would have landed near −0.40%. It landed at +0.077%. Source: `prc_adjacent_20260927/llama8B_t32/full_test_result.json`.

**Earlier rounds: held-out (16 windows, TRAIN seed 101, `heldout_nll.py`) against full test.**

| pair set | n | r | slope | sign agree | |z|>=1.5 sign agree | rms(test−ho) |
|---|---|---|---|---|---|---|
| all incl. parent | 112 | 0.990 | 0.895 | 106/112 | 79/79 | 0.40% |
| per-group vs per-group | 49 | 0.962 | 0.941 | 46/49 | 28/28 | 0.31% |
| per-group, |ho|<1% | 35 | 0.680 | 0.655 | 32/35 | 14/14 | 0.32% |

Source: `kbands/prc2/*_heldout_nll.json` joined with trace-header PPLs in
`kbands_20260801/ppl/*_trace.json` (all `ppl_max_tokens=0`). Parent test values are the AWQ+h20
`submitted_ppl`. 14B t64/t96 non-h20 parents are excluded.

**Fisher surrogate against test.** Linear-Fisher `pred_dnll_fis` differences between arms (`*_c6/c7_diag.json`)
against test: n=24, r=0.76, slope 0.96, sign 22/24. The joint-attention prediction (`joint.gfisla -
joint.gfis`, 30B) is right when it is large: t32 −2.6% predicted vs −1.56% test, t40 −3.1% vs −2.84%,
t96 +0.88% vs +0.74%. It is wrong when it is small: t64 −0.03% vs +0.27%. This agrees with
ROUND3_INVESTIGATION item 3.

A candidate generator therefore **exists** that predicts the sign of about 1%-sized effects. Rounds 3–5 did
not use it.

## 6. Cost guard

R4/R5 search cost ratios were 0.9989–1.0030. R3's ranged 0.9970–1.0054. All confirmation and test costs were
within 0.22%. The +-1% guard never rejected anything. It is not a cause of the failure. It is an unused
resource: the user's rule allows about +1% cycles. Spent uniformly, that buys only e×1% ≈ 0.15–0.27% at t32.
Spent on groups worth k≈3–5× average, it could buy about 0.5–1%. Any such result must be reported with its
cost.

## 7. Docs that disagree with the artifacts

- `scmp_llm/CLAUDE.md` STATUS says "No round-4 PPL results yet". That is stale: all four R4/R5 cells finished
  between 15:20 and 17:36 on 2026-09-27, and llama t32 has a full test (+0.077%, incumbent retained).
  `PRC_ADJACENT_20260927.md` has no results section.
- `PRC_ADJACENT_20260927.md` presents "transfers of 0.25% and 0.5% of total SC cycles" as ceilings. Realized
  net transfers were **0.0047–0.068%**. The per-bucket 4% group/MAC planning cap, not the transfer ceiling, set
  the step. The doc says moves "may be smaller" but never reports that they were 4–100× smaller.
- The best(all) 30B t40 "winner" is `round3_local` at −0.030% vs c7gfisla, which is below the test noise floor
  (~0.08–0.10%). `BEST_ALL_VS_SUBMITTED_20260927.md` does caveat this.
- Checked and consistent: ROUND3_POSTMORTEM's "19/20 worsened" and its search/confirm/test table match the
  artifacts.

## 8. Conclusion and a protocol that could reach 1% within 4 GPUs

- **R3:** could not realistically produce >=1%. Its steps were big enough but blind to direction and nonlocal,
  and the operator-level budget split it perturbed had already been balanced by the round-1/2 Fisher λ solves.
- **R4/R5:** no, by construction. The expected effect was O(0.01%) against an MDE of O(0.7%).
- **The evaluation half of the protocol works.** The paired 16-window TRAIN estimate predicts test with r≈0.96.
- **What has to change is candidate generation and step size.** Requirements for a round that can reach 1%:
1. **Loss-guided, coherent, large moves.** Move 3–10% of SC cycles, the regime of every historical >=1%
   effect. Keep each group within <=1 rung, but move many boundaries at once. This fixes R3's coincident-threshold
   jumps. Parameterize candidates as a small 1-D or 3-D family from the existing joint Fisher DP, e.g. price
   multipliers on {qk, av, linears} or per-layer-quarter multipliers, plus one "+<=1% cost" promotion variant.
   Pre-screen with the Fisher-predicted ΔNLL, which is sign-reliable for moves of about 0.5% or more. Keep only
   candidates predicted at <=−0.5%.
2. **Search:** <=4–6 candidates plus the exact incumbent on 16 shared paired windows (SE≈0.25–0.3%).
   Advance <=2 with z<−2.
3. **Confirm:** on 32 fresh windows (MDE80≈0.4–0.5%), with z<−2 for multiplicity. Run a full test for any
   candidate that passes. By the user's best(all) rule, optionally also test the family's best, since dense tests
   are cheap.
4. **Scope:** t32–t48 only, where e = 0.07–0.27. Skip t64/t96.
5. **GPU budget,** measured from `seconds` fields and `[RESULT] sec=` in the logs. Per-window cost is about
   28 s for 4B, 24 s for llama8B, 50 s for 14B and 90–95 s for 30B. A full test takes 0.8–1.1 h (4B),
   0.9–1.3 h (llama8B), 2.0–2.2 h (14B) and 2.8–4.2 h (30B). One cell costs about 2.3 GPU-h for 4B or llama8B,
   about 4.5 GPU-h for 14B and about 8–9 GPU-h for 30B, excluding Fisher/gradient capture. One night on 4 GPUs
   (about 48 GPU-h) covers 4B and llama8B t32/t40/t48, 14B t32/t40/t48 and 30B t32/t40 (about 45 GPU-h).
   Paired sequential testing (16 windows, extend to 32 only when |z| is between 1 and 3) cuts this further.
6. **Pre-register** an MDE and predicted effect for each candidate. Refuse to run any candidate whose predicted
   |Δ| is below the confirmation MDE.

## 9. Suspected causes (protocol angle), open questions

These appear in the structured output.
