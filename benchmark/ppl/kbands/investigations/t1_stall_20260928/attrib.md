# T1 per-group allocation: where the remaining PPL loss sits, and how much allocation can still recover

Read-only investigation, 2026-09-27. No GPU work, no repo files touched. All new numbers come from
CPU scripts in this directory (`attrib_*.py`) run on existing Turbo artifacts. Units: stream lengths
are HALVED code units (nominal = 2x). "MC-Fisher" = the calibrators' `fis` currency (diagonal Fisher,
MC labels). "Parent" = the submitted per-row cell (AWQ, h=20% INT7 mask; 14B t64/t96 use the h20 parents).

## Bottom line

1. **Attention or linears? Linears at tight budgets, the SC ceiling at loose ones, attention only on 30B.**
   - Dense models at t32 to t48: most of the remaining gap is budget-limited linear error above L=128,
     mostly in the MLP. By the Fisher extrapolation it is 46-86% of the round-1 gap for 4B/llama8B and
     77-109% for 14B. gate/up/down carry 79% (4B) and 85% (llama8B) of the linear excess.
   - Attention's excess over its own ceiling is small or already priced at parity with linears. Round 3c
     implies attention 96->128 would buy about 0.012-0.024 nats on 4B/llama8B t32, but it costs 7-7.5% of
     total cycles. At iso-cost it was neutral or worse.
   - t64/t96: the gap is dominated by the **SC ceiling**, i.e. every SC group at L=128 with an h20 mask:
     4B +0.74%, llama8B +4.37%, 14B +0.38%, 30B +3.05% over fp16. No allocation can touch it.
     The best t96 cells already sit +0.59-0.68% above that ceiling, and 14B is 0.38% *below* it.
   - 30B t32 to t48 is the exception. After linear excess and ceiling are removed, a residual of 0.055-0.076
     nats remains in round 1. Round 2 recovered 0.016-0.028 nats of it by moving qk up and av down.
     About 0.03-0.05 nats is still unexplained; it is either attention or an MoE/interaction effect.
2. **Irreducible or misallocated?** At t32 the linear floor at L=128 is small: about 5% of the parent's linear
   Fisher loss (0.0049 vs 0.100 MC-nats on 4B t32; 0.0052 vs 0.091 on llama8B t32). So linears are
   budget-limited, not floor-limited.
   - Against a perfect-information Fisher oracle, the deployed rule still leaves 38-42% of the parent's linear
     excess on the table. That is roughly 5.6-6.6% PPL at t32.
   - **About 90% of that oracle gap is per-token loss sensitivity**, which the runtime cannot see. An oracle that
     knows each group's exact SC error but only a static per-(op, block) sensitivity reaches 0.561 (4B) and
     0.596 (llama8B). The deployed rule is at 0.597 and 0.645. The full oracle is at 0.215 and 0.222.
   - Row sensitivity is essentially uncorrelated with activation magnitude (Spearman +0.04 / -0.03) and only
     weakly correlated with position (-0.23 / -0.14).
3. **What is recoverable with deployable information** (upper bounds, t32, measured-PPL equivalents; section 3):
   - **(i) Within-operator group selection:** re-fitting thresholds is worth at most 0.17-0.46%. A perfect
     SC-error statistic is worth at most 0.42-0.58%. Exact per-group SC error plus static sensitivity is worth
     at most 0.38-0.64%. Static chunk-class or position keys add nothing: 0.563 vs 0.561 on 4B, 0.598 vs 0.596
     on llama8B.
   - **(ii) Cross-operator/layer budget:** even the Fisher oracle gains only 0.02-0.03 of parent excess from
     crossing bucket boundaries, i.e. at most about 0.5%.
     - Linear-to-attention re-splitting was harvested by round 2 on 30B and is neutral on dense cells.
     - Round 3's 2% coarse transfers lost in BOTH directions on 4B, which means the incumbent sits at a broad
       optimum for those moves. Round 4's adjacent moves were 5-50x smaller than their stated ceiling and
       below detectability.
   - **(iii) Ladder shape:** for linears, the 17-rung ladder is within about 0.001 of an all-lengths oracle.
     The deployable rule puts only 0.4-4% of MACs on rung 8 and about 0.1% on 128, so the ladder ends do not
     bind. For attention, the ladder top is below 128 in 13/20 cells, and attention is pinned at it in several.
     That is the one ladder lever left untested outside 4B/llama8B t32.

   **Net:** at dense t32 to t48, deployable pure-allocation headroom beyond best(all) is about 0.2-0.7% per cell,
   and about 0.1% or less at t64+. The only place with plausibly ≥1% is 30B t32 to t48 attention/MoE, plus a
   smaller amount on 4B t40/t64, where the parent under-funds attention.

## 0. Rounds 4 and 5 finished; the docs lag behind

`scmp_llm/CLAUDE.md` STATUS and `PRC_ADJACENT_20260927.md` still say "No round-4 PPL results yet". The
artifacts show that array 62104342 (4B_t32, llama8B_t32) and array 62105189 (30B_t32, 30B_t40; the
"round 5" companion) all finished between 15:20 and 17:36 on 09-27. Sources are Turbo
`prc_adjacent_20260927/<cell>/{selected,search_results,confirm_*_nll,full_test_result}.json` and the logs
`p4adjacent20260927_62104342_{0,1}.out` and `p4adj30b20260927_62105189_{0,1}.out`.

| cell | selected candidate | search ΔPPL (z) | confirm ΔPPL (z), 16 windows | full test |
|---|---|---:|---:|---|
| 4B t32 | adj_linears_from_qk_01 | -0.446% (-1.23) | +0.138% (+0.48) | not run: failed gate |
| llama8B t32 | adj_within_o_proj_09 | -0.749% (-2.19) | -0.398% (-1.98) | **8.28586 vs 8.27948, +0.077%: loses** |
| 30B t32 | adj_within_down_proj_11 | -1.138% (-2.53) | -0.329% (-0.68) | not run |
| 30B t40 | adj_mlp_from_projections_07 | -0.644% (-2.00) | -0.079% (-0.27) | not run |

- **Moves were tiny.** Every candidate requested 0.25% of total SC cycles (`proposal.requested_transfer_fraction`
  = 0.0025), but the transferred fraction was only 0.005-0.068% because the locality caps bound first. The
  selected moves were 0.048%, 0.018%, 0.020% and 0.024%.
- **The expected effect was undetectable.** At the within-cell elasticity (d ln PPL / d ln cost = -0.44 on 4B t32,
  SUMMARY §3), moving 0.05% of cycles is worth at most about 0.02% PPL, even at a several-fold price gap. The
  confirmation standard error was 0.20-0.49% PPL.
- **The search "wins" were a shared-baseline artifact.** 39 of 48 search deltas were negative: 11/12, 8/12, 10/12
  and 10/12 by cell. Opposite-direction transfers both "won", for example 4B qk_from_linears -0.23% and
  linears_from_qk -0.45%. Real improvements from a smooth objective cannot behave that way; this is the
  common incumbent's noise on those 6 windows, not signal.
- **Round 3 on 4B t32 lost in both directions at 2%.** Every transfer got worse: qk+ +1.40%, qk- +2.25%, av+ +0.55%,
  av- +1.02%, projections+ +1.85% (z 6.2), MLP+ +1.04% (`prc_local_20260926/4B_t32/search_results.json`). The
  incumbent is at a broad optimum for population transfers.

## 1. Where SC cycles go (best(all) full-test traces)

Script `attrib_att_share.py`, reading `groups[].macs × stoc_len` of each winning trace (`BEST_ALL_VS_SUBMITTED_20260927.json`
`rows[].trace`). SC matmuls only; INT-masked blocks are not recorded.

| cell | attention MAC% | attention cycle% | L_qk | L_av | L_lin | attention ladder top (parent) |
|---|---:|---:|---:|---:|---:|---:|
| 4B t32 / t40 / t48 / t64 / t96 | 13.1 | 32.8 / 29.1 / 26.0 / 20.8 / 17.3 | 83.6 / 94.9 / 98.4 / 104.5 / 127.9 | 85.6 / 89.9 / 99.6 / 102.6 / 126.7 | 26.1 / 33.8 / 42.4 / 59.2 / 91.6 | 97 / 96 / 111 / 105 / 128 |
| llama8B t32 / t40 / t48 / t64 / t96 | 7.3-7.5 | 21.4 / 17.2 / 18.7 / 14.3 / 9.3 | 97.1 / 99.0 / 128 / 128 / 127.9 | 96.1 / 98.2 / 128 / 128 / 118.0 | 28.8 / 37.1 / 43.4 / 60.2 / 93.4 | 96 / 98 / 128 / 128 / 128 |
| 14B t32 / t40 / t48 / t64(h20) / t96(h20) | 5.5 | 14.9 / 11.2 / 12.6 / 7.5 / 7.3 | 93.1 / 85.5 / 113.4 / 87.9 / 127.7 | 87.4 / 83.8 / 112.7 / 87.5 / 127.1 | 29.8 / 38.7 / 45.4 / 62.2 / 94.1 | 96 / **86** / 128 / **89** / 128 |
| 30B t32 / t40 / t48 / t64 / t96 | 23.3 | 43.2 / 41.1 / 39.8 / 40.6 / 29.9 | 88.1 / 92.8 / 105.7 / 117.3 / 127.5 | 44.5 / 61.8 / 72.3 / 117.0 / 118.3 | 25.1 / 32.6 / 39.7 / 51.9 / 87.0 | 96 / 96 / 109 / 117 / 128 |

Ladder tops come from `hpca_results/llm/ppl/mp_best/manifest.json` `levels`, and for 14B t64 from
`prc2/parents_h20/14B/target64/wrapper.json` (levels [89..58]).

- Attention is 5.5-23% of SC MACs but 7-43% of SC cycles, because it already runs 2-4x longer than the linears.
- The dense t32/t40 cells are pinned at the top ordinary rung, 96-97, with escape to 128. 14B t40 and 14B t64 are
  pinned at compressed tops of 86 and 89, which the per-row search inherited.
- Attention length is non-monotone in budget in several parents:
  - 4B t40 (82.7/84.7) is at the t32 level (83.4/85.5).
  - 4B t64 (89/93) is below t48 (98/100).
  - 14B t40 (85.5) is below t32 (96.8).
  - 14B t64 (88) is below t48 (113).
  - llama8B t96 av (117) is below t48/t64 (128).
- Round 2 raised attention in 4B t40/t64 and won -0.94% and -0.57% vs round 1.

## 2. Per-cell gap decomposition (round-1 linear-only Fisher cell)

Script `attrib_decomp.py`; κ table in `attrib_kappa.py`.

**How each term is estimated:**
- **Linear excess.** E_par is the parent's linear Fisher excess over L=128 in MC-nats. It is recovered from each
  diag as `-pred_dnll_fis / (1 - fis_over_parent)` for the `gfis` table (`prc2/<cell>_{c6,c6h20,c7}_diag.json`;
  c7 for 30B).
- **κ calibration.** κ = measured test Δln PPL(gfis vs parent) / predicted Δ. It is consistent per cell:
  4B t32 c17 1.68 and gfis 1.67; median about 1.4.
- **Independent check of κ.** The true-label Fisher on llama8B t32 predicts the measured test change almost
  exactly (κ_emp 1.02 for c17, 1.05 for gfis). κ_emp·E_emp ≈ 0.135 nats, which matches κ_MC·E_MC ≈ 0.13.
- **Ceiling.** C = ln(uniform L=128 + h20 INT8 mask / fp16), from `hpca_results/llm/frontend_awq/awq_mask_mp_table.csv`
  (4B 10.1186, llama8B 7.5282, 14B 8.6709, 30B 7.4826).
- **Rest.** rest = gap − C − κ·E_par·ratio_r1. It holds attention excess, interactions and mask effects.

| cell | gap(round-1, nats) | linear excess | ceiling | rest | best(all) vs ceiling |
|---|---:|---:|---:|---:|---:|
| 4B t32 | 0.108 | 0.093 (86%) | 0.007 | 0.008 | +10.58% |
| 4B t40 | 0.079 | 0.039 (49%) | 0.007 | 0.033 | +6.47% |
| 4B t48 | 0.041 | 0.025 (61%) | 0.007 | 0.009 | +3.44% |
| 4B t64 | 0.031 | 0.011 (35%) | 0.007 | 0.013 | +1.85% |
| 4B t96 | 0.015 | 0.001 | 0.007 | 0.007 | +0.63% |
| llama8B t32 | 0.138 | 0.087 (63%) | 0.043 (31%) | 0.008 | +9.98% |
| llama8B t40 | 0.107 | 0.050 (46%) | 0.043 | 0.015 | +6.62% |
| llama8B t48 | 0.077 | 0.038 | 0.043 | -0.004 | +3.45% |
| llama8B t64 | 0.058 | 0.017 | 0.043 (74%) | -0.002 | +1.53% |
| llama8B t96 | 0.050 | 0.003 | 0.043 (85%) | 0.004 | +0.59% |
| 14B t32 | 0.053 | 0.044 (83%) | 0.004 | 0.005 | +4.68% |
| 14B t40 | 0.030 | 0.023 (77%) | 0.004 | 0.003 | +2.68% |
| 14B t48 | 0.017 | 0.019 | 0.004 | -0.005 | +1.37% |
| 14B t64 (h20) | 0.007 | 0.010 | 0.004 | -0.007 | +0.33% |
| 14B t96 (h20) | 0.001 | ~0 | 0.004 | — | -0.38% |
| 30B t32 | 0.183 | 0.085 (46%) | 0.030 | **0.068** | +14.68% |
| 30B t40 | 0.136 | 0.030 | 0.030 | **0.076** | +8.06% |
| 30B t48 | 0.102 | 0.017 | 0.030 | **0.055** | +5.00% |
| 30B t64 | 0.049 | 0.012 | 0.030 (61%) | 0.008 | +1.94% |
| 30B t96 | 0.037 | 0.003 | 0.030 (82%) | 0.004 | +0.68% |

**Round 2 recovered part of the 30B rest.** Measured c7gfisla vs c7gfis: -0.0156, -0.0284 and -0.0232 nats at
t32/t40/t48. That leaves roughly 0.05, 0.05 and 0.03 nats unexplained after the best cells.

**Caveats:**
- **Linear extrapolation.** κ is measured on a 34-50% reduction of linear Fisher error, so extrapolating it to
  100% is approximate. Values above 100%, and negative rests on 14B t48/t64 and llama8B t48/t64, show an
  error bar of about ±0.01-0.02 nats.
- **Independent round-3c check.** Attention top rung 96/97 -> 128 at iso-cost on t32 (`PRC2_OVERNIGHT.md:381-433`;
  att128 vs the reproduction: 4B +0.0122 nats, llama8B +0.0012). Scale the linear cut (0.876x / 0.904x) with the
  c17 vs c17_s80 slope (0.0028 / 0.0026 nats per 1% of linear budget). The implied attention gain from 96 to 128
  is ≤0.022 nats on 4B and ≤0.024 nats on llama8B, an upper bound because loss is convex in budget.
  - This is larger than the llama8B "rest" (0.008), so the linear share there is probably overstated by up to
    about 0.016 nats.
  - Either way, attention's value per cycle equals the linears' at the margin, so there is no iso-cost gain.

**Per-operator attribution inside the linears.** Parent Fisher excess on the hold windows (`attrib_h2_*_fis.json`
`A_per_op_hold`):

| model (t32) | down_proj | up_proj | gate_proj | q_proj | o_proj | k_proj | v_proj |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 33.8% | 25.3% | 19.7% | 8.5% | 7.4% | 3.3% | 1.9% |
| llama8B | 31.5% | 32.8% | 20.7% | 6.4% | 4.6% | 2.9% | 1.2% |

down_proj alone holds about half of the L=128 floor (51% and 49%).

**Older op-swap probes point the same way, on different baselines.** These are op -> INT7 swaps on the
`_3` baseline, which includes qk rebalance, so attention is understated.
- 4B t32 swaps (`LOOP_STATE.md:3226`): up -3.73%, down -3.67%, gate -1.57%, o -1.42%, av -1.22%, v -0.77%,
  q -0.68%, k -0.36%, qk -0.24%.
- llama8B t96 (`LOOP_STATE.md:2751`): attention is 64% of the deficit.
- 14B t96 (`LOOP_STATE.md:2831`): down_proj carries the loss, and the av swap makes it worse.

## 3. How much of the linear headroom a deployable rule can reach

Scripts `attrib_headroom2.py`, `attrib_chunkclass.py` and `attrib_cross.py`, run on the clean six-window dumps
`prc2/{4B,llama8B}_t32_c8w6.npz`. Setup:
- 6 calib windows and 2 hold windows (windows 3 and 7); linears only.
- All numbers are hold-window error divided by the parent's error at the parent's linear cost.
- My re-implementation of the deployed rule reproduces the diag's `gfis` value: 4B 0.5969 vs 0.5972, llama8B
  0.6452 vs 0.6454.
- PPL equivalents use κ·E_par = 0.168 nats (4B, MC), 0.130 (llama8B, MC) and 0.142 (llama8B, empirical-label
  currency with κ≈1.05).

| allocation (same cost) | what it knows | 4B MC | llama8B MC | llama8B emp | ≈ gain vs deployed (PPL) |
|---|---|---:|---:|---:|---:|
| rowmax staircase (per-row rule) | row absmax | 0.790 | 0.782 | 0.810 | worse |
| **mn staircase fit on calib, scored on hold (≈ deployed gfis)** | today's rule | **0.597** | **0.645** | **0.651** | — |
| mn staircase fit in-sample on hold | best thresholds for today's statistic | 0.581 | 0.632 | 0.619 | 0.27% / 0.17% / 0.46% |
| staircase on a perfect SC-error statistic (raw error at L=24 per MAC) | exact SC error ranking | 0.572 | 0.606 | 0.610 | 0.42% / 0.50% / 0.58% |
| free per-group choice: exact raw error × static per-(op, block) sensitivity | exact SC error | 0.561 | 0.596 | 0.624 | 0.61% / 0.64% / 0.38% |
| same, sensitivity static per (op, block, chunk index) | + static chunk classes | 0.563 | 0.598 | — | no extra gain |
| same, sensitivity static per (op, block, 256-token position bin) | + position | 0.571 | 0.627 | — | worse |
| free choice, exact raw error × true per-ROW sensitivity | per-token loss sensitivity | 0.244 | 0.242 | 0.200 | 6.1% / 5.4% / 6.6% (not deployable) |
| Fisher oracle, one L per row | per-token, row granularity | 0.330 | 0.294 | 0.243 | — |
| Fisher oracle, per group (17 rungs) | everything | 0.215 | 0.222 | 0.183 | 6.6% / 5.6% / 6.9% (not deployable) |
| every linear at L=128 (about 4-5x linear cycles) | — | 0 | 0 | 0 | 10.5% / 8.7% / 9.7% |

**Robustness of the per-token signal.** An oracle allocated with MC-Fisher and scored with true-label Fisher still
reaches 0.363 (llama8B). Row sensitivities from the two label sources have Spearman 0.91
(`attrib_cross_llama8B.json`). So the per-token signal is real and stable, not label noise. It is simply unobservable
at dispatch.

**Diag-vs-dump difference.** The diag oracles quoted in `PRC2_OVERNIGHT.md:259` (0.251/0.221) are from the c6 run;
the c7/c8w6 values are 0.244/0.215.

## 4. Does the runtime statistic pick the right groups?

`attrib_h2_*.json` `D_statistic_quality`; Spearman values are within-key medians on the hold windows.

- **For SC error, yes.** mn vs the marginal raw-error gain (L 24->32 per MAC): 0.66 on 4B (range 0.25-0.93) and
  0.79 on llama8B (0.37-0.93). A perfect error statistic buys at most 0.009-0.026 of parent excess inside the same
  staircase family.
- **For loss, weakly.** mn vs the marginal Fisher gain: 0.40 on 4B, 0.25 on llama8B.
  - 69% (4B) and 83-84% (llama8B) of the log-variance of the per-group Fisher marginal value comes from the
    sensitivity factor (Fisher/raw), not from SC error.
  - Row sensitivity vs rowmax: +0.04 on 4B, -0.03 / -0.01 on llama8B. Row sensitivity vs position: -0.23 on 4B,
    -0.14 / -0.17 on llama8B.
- **The loss is heavy-tailed by token.** The top 1% of rows carry 33% (4B), 37% (llama8B MC) and 44% (llama8B emp)
  of the parent's linear Fisher excess. Only 19-22% of those rows are in their key's top-10% rowmax.
  This is the same phenomenon that wrecked the 12-window calib8 run (`PRC2_OVERNIGHT.md:342-376`).

## 5. Cross-operator and cross-layer reallocation and attention: what is already measured

- **Bucket boundary.** Fisher oracle within vs global is 0.244 vs 0.215 on 4B and 0.244 vs 0.222 on llama8B
  (c8w6 diag), so crossing (op, layer-quartile) boundaries is worth ≤0.03 of parent excess even with perfect
  information.
  - The deployed gfis already uses one global λ.
  - Per-block keys: predicted 0.572 vs 0.597 (`c8w6_diag.json round3.blk`), measured a tie (+0.0011 NLL,
    `PRC2_OVERNIGHT.md:361-366`).
- **Linear<->attention (round 2).** It wins on 30B t32/t40/t48 (-1.55/-2.80/-2.29%) and on 4B t40/t64/t96,
  llama8B t96 and 14B t32/t96 (0.08-0.94%). It loses on 4B t48, llama8B t32-t64, 14B t40/t48/t64 and 30B t64/t96
  (`BEST_ALL_VS_SUBMITTED_20260927.json` candidates).
- **Attention ceiling at iso-cost (round 3c).** 4B t32 worse, llama8B t32 neutral. Not tested on 30B or on the
  cells with compressed attention tops (14B t40/t64, 4B t40/t64, 30B t48/t64).
- **Population transfers.** Round 3 at 2%: 19/20 worse, and both directions lose on 4B. Round 4 at 0.005-0.07%:
  nothing confirmed (section 0). Both families live inside the ≤0.03 bucket-boundary bound above, so their failure
  was expected from the attribution.
- **Loss-weighted currencies.** P1 pooled loss-weighting was refuted (`LOOP_STATE.md:2371-2419`); global σ loses on
  4B; robust/clipped Fisher ties (`PRC2_OVERNIGHT.md:361-366`).

## 6. Ladder shape

- **Linears.** The Fisher oracle on the 17-rung ladder vs all 20 measured lengths: 0.2441 vs 0.2427 (4B) and 0.2441
  vs 0.2427 (llama8B), from `c8w6_diag.json` `fis/ladder` vs `fis/all`.
  - The deployable rule puts 0.4-3.8% of MACs on L=8 and about 0.1% on L=128. Only the non-deployable oracle wants
    13-19% on L=8.
  - So extending the linear ladder below 8 (halved), or adding rungs, buys nothing for a deployable rule.
- **Attention.** The ladder top is below 128 in 13/20 cells (section 1), and attention is pinned at it in 14B t40
  (86) and t64 (89), 4B t64 (105) and 30B t64 (117). Only 4B t32 and llama8B t32 have been tested with a 128 top.

## 7. Implications for the ≤3 new rounds (attribution angle only)

- **Expected size.** Pure allocation cannot deliver ≥1% on dense cells at t48+. At dense t32/t40 the deployable
  bound is about 0.2-0.7%, which is at or below the 16-window detection limit (SE 0.2-0.5% PPL). Anything aimed
  there needs ≥48 confirmation windows or an effect predicted at ≥1%.
- **Where the rest of the gap sits.** Either per-token sensitivity (not observable) or the SC ceiling
  (llama8B +4.4%, 30B +3.1%). The ceiling is RNG/representation/QK territory and outside the rules.
- **Candidate rounds, ranked by attributed headroom:**
  1. **30B t32/t40/t48 attention allocation with an attention ladder top of 128**, especially for qk, which sits at
     88-106 against a top of 96-109. Joint Fisher re-solve at iso-cost against the incumbent; no escape, RNG or QK
     transform change.
     - Evidence: 0.03-0.05 nats of unexplained rest; round 2's qk-up / av-down moves won 1.5-2.8%. Fisher
       over-predicted those wins by 1.1-1.7x.
     - Size the lever first with an over-budget diagnostic on the 16 confirmation windows: attention forced to
       L=128, incumbent linears. That is 1 GPU for about 25 min per cell.
     - Kill the round if even that diagnostic recovers <0.01 nats.
  2. **Dense cells whose attention is pinned at a compressed top or is non-monotone in budget:** 4B t40 and t64
     (rest 0.024 and 0.007 nats after round 2), then 14B t40 (86) and t64-h20 (89). Same joint re-solve with top
     128. Expected ≤0.5-1% (4B t40 is the best bet); for 14B, expect ~0 (rest ≈ 0).
  3. **Only if a linear round is required:** a calibration-transfer fix, i.e. more calibration windows with per-row
     Fisher-mass clipping and the same staircase/λ. It targets only the calib->hold gap, 0.013-0.032 of parent
     excess (≤0.2-0.5%). The prior robust-Fisher arm tied, so this is a low-value round.
- **Do not repeat** population transfers, one-rung threshold nudges, static chunk classes, position keys or denser
  linear ladders. The bounds in sections 3 and 5 cap all of them at ≤0.3%.

## 8. Docs that disagree with the artifacts

1. **Round 4 status.** CLAUDE.md STATUS and PRC_ADJACENT_20260927.md say no round-4 results exist. All four cells
   are complete (section 0), and llama8B t32 was full-test evaluated and lost (+0.077%). Nothing enters best(all).
2. **Round-4 transfer size.** PRC_ADJACENT says "transfers of 0.25% and 0.5% of total SC cycles are ceilings".
   Every one of the 48 proposals requested 0.25% and none 0.5%. Realized transfers were 0.005-0.068%, so the
   pilot tested moves 4-50x smaller than the text implies.
3. **The Fisher oracle gap.** ROUND3_INVESTIGATION calls the remaining t32 gap "3-4 percentage points". That is in
   MC-Fisher-predicted units (4B: oracle -7.58% vs gfis -3.97%). With κ≈1.4-1.7 it corresponds to about 5.6-6.6%
   measured-PPL-equivalent, but about 90% of it requires per-token sensitivity, so it is not a deployable target.
4. **LOOP_STATE P1.** Its note "allocation is ~96% exhausted (deployable within 0.8-1.6pp of the oracle)" was
   per-row-era σ-currency. In Fisher currency the per-group oracle gap is large, but it is non-deployable.
   The two statements measure different things.

## 9. Evidence that is missing

- **No attention error dumps.** The c8w6 npz holds linears only, so attention's absolute Fisher share is inferred
  (section 2 rest, round-3c back-out), not measured. No dump exists for 14B or 30B at all.
- **No direct attribution runs** exist on the current baseline: attention forced to L=128 with incumbent linears,
  and linears forced to 128 with incumbent attention. They would replace the κ extrapolation.
- **The 30B rest is not attributed.** It is unknown whether it is attention precision or MoE-routing/interaction
  loss; the router is fp16, but SC error upstream can flip top-k, as seen in the STE bug.
- **Only two hold windows** back the section-3 in-sample bounds, and they are optimistic. The 4B true-label
  (`fis_emp`) currency is broken in the c7/c8w6 dumps (absurd predictions), so only MC is available on 4B.
- **Op-swap attribution** exists only on older `_3` baselines that include qk rebalance.

## Reproduction

All scripts are in this directory and are CPU-only; each takes under 3 minutes and under 0.9 GB RSS.

- `attrib_att_share.py` (+ `attrib_trace_agg.py`) -> `attrib_att_share.json`
- `attrib_kappa.py` -> `attrib_kappa.json`
- `attrib_decomp.py` -> `attrib_decomp.json`
- `attrib_headroom2.py <npz> {fis|fis_emp}` -> `attrib_h2_*.json`
- `attrib_chunkclass.py` -> `attrib_cc_*.json`
- `attrib_cross.py` -> `attrib_cross_llama8B.json`
- `attrib_r4.py <cell dir>` summarizes rounds 3 and 4.

The staircase DP is copied verbatim from `mp_per_row_chunk_calib2.py:108`. The user slice is capped at 4 GB and
shared with other agents; a first, non-streaming version OOM-ed on llama8B (`attrib_h1_llama8B_fis_EMPTY_oom.json`).
