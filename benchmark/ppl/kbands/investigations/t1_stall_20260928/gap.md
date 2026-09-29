# T1 per-group allocation: how big the gap is, all 20 LLM cells

Date: 2026-09-27. Read-only investigation. No GPU jobs were run and no existing files were changed.
Scratch sources: `gap_work/gap.py` (per-cell metrics, written to `gap_rows.json` / `gap_table.csv`),
`gap_work/opsplit.py` (per-operator SC work from the full-test traces, written to `opsplit.json`), and inline
round-3/4 statistics computed from Turbo `search_results.json` / `selected.json`.

## 0. Bottom line

1. **The claim itself holds, but the gain is modest past 6.58 bits.** Best(all) improves 20/20 cells, with a mean change of
   −2.53% against the per-row AWQ+h20 parent. By budget the mean is −5.77 / −3.12 / −2.16 / −1.12 / −0.46% (6 / 6.32 / 6.58 / 7 / 7.58 bits).
   The direct T1 correction on its own (`c17`: per-group, reconstruction-calibrated) averages −1.51%. By budget that is
   −3.91 / −1.63 / −1.13 / −0.69 / −0.18%, with 19/20 cells improved. 14B t96 gets worse, +0.03%.
2. **Past round 2, the extra rounds added essentially nothing.** Best(all) = −2.526% mean and min(round 1, round 2) = −2.523%.
   Codex's rounds 3–5 add a mean of −0.0015 pp. Only one cell moved: 30B t40, −0.030%, which is 0.36 noise-sd.
   Round 4 fully tested one candidate (llama8B t32), and it came out +0.077% worse.
3. **At 7 and 7.58 bits the gap to fp16 mostly cannot be fixed by allocation.** The SC ceiling is uniform full-length
   (nominal 256) SC with the same h=20% mask under AWQ. That ceiling is already 1.0074 / 1.0437 / 1.0038 / 1.0305 × fp16
   (4B / 8B / 14B / 30B). At 7.58 bits best(all) sits only +0.63 / +0.59 / −0.38 / +0.68% above the ceiling.
   At 7 bits it sits +1.85 / +1.53 / +0.33 / +1.94% above it. So llama8B t96, at 1.0498× fp16, is within 0.6% of the best any
   pure allocation could reach.
4. **As a share of the addressable headroom, per-group closes about 36–49% of the parent's excess over the ceiling at every budget.**
   The per-budget means are 39.6 / 36.4 / 41.0 / 48.7 / 47.7%. Gains "shrink with budget" because the parent's own excess shrinks,
   from +7…+26% at t32 to +1.0…+1.6% at t96. The algorithm is not getting weaker.
5. **The headline mixes in effects that are not granularity.** Rounds 1 and 2 (Fisher pricing, and on 30B the re-split of the per-row
   attention thresholds) provide 15–81% of each cell's log-gain. On 30B t40/t48, 74% and 81% of the gain comes from round 2 moving
   per-row attention budget. Attention is still per-row, so that part is not a granularity effect. Pure granularity (the
   same-calibration per-row control `r17` against `c17`) is measured in only 3 cells: −4.33%, −2.05% and −1.63%.
6. **How far the paper's pass/fail thresholds are.** The ≤1.10× fp16 counts are 13 (parent) → 14 (c17) → 15/20 (best).
   The ≤1.05× counts are 7 → 8 → 9/20. Three cells are each about 1.2% from lowering a model's cheapest passing budget one step:
   4B t32 needs −1.25%, llama8B t40 −1.15%, 30B t40 −1.21%. Two ≤1.05× misses are within noise: 14B t32 needs −0.07% and 30B t64 −0.05%.
7. **Against the levers the user excluded, pure allocation looks worse.** `_4` has 12/20 ≤1.05× and 19/20 ≤1.10×. `_3` has 10/20 and 17/20.
   Best(all) is 0.1–10.6% above `_4` in 16/20 cells, with 30B worst at +3.3…+10.6%. It beats `_4` on 14B at t32/t40/t48/t64.
   `_4` 30B t96 (7.2922) is 2.5% below the pure-allocation ceiling (7.4826), which only a change to the substrate
   (the QK rebalance) can do. So that gap is out of scope.
8. **INT is out of reach at every matched width.** INT6 under AWQ is at 1.001–1.013× fp16. The SC ceiling itself is worse than
   INT6 on llama8B (+3.07%) and 30B (+2.72%). This comparison can never read as "good enough" through allocation; the paper
   already makes SC's case on energy.

## 1. What is compared (provenance)

- **Parent** = the submitted per-row allocation re-measured under AWQ with h=20%. These are the values in the
  submitted `tab:quality-llm` (Overleaf commit `4290310`, 2026-08-01, e.g. 4B 6-bit 11.99). Source:
  `hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.json` `rows[].submitted_ppl`. 14B t64/t96 use the corrected
  h20 parents, 8.7758 / 8.6418.
- **c17** (per-group reconstruction; the Sep 23 prc2 recipe), **r1** (`c6gfis` / `c7gfis` / `c6h20gfis`: linear-only global
  Fisher; `c6gfis_emp` where it was tested), **r2** (`c7gfisla` / `c7h20gfisla`: joint linear + attention-threshold λ),
  **r3** (`round3_local`) and **g5** all come from `rows[].candidates` of the same JSON.
- **r4/r5**: Turbo `prc_adjacent_20260927/<cell>/{search_results,selected,full_test_result}.json`, arrays 62104342 (4B/llama8B t32)
  and 62105189 (30B t32/t40). **Round naming:** no document uses "round 5". The 30B array is documented as the
  four-GPU expansion of round 4 (`PRC_ADJACENT_20260927.md` §"Expansion to four GPUs"). I treat the user's rounds
  3/4/5 as local (62032412) / adjacent dense (62104342) / adjacent 30B (62105189). Separately, Claude ran an earlier "round 3"
  on Sep 24–25 (calib8 robust Fisher / per-block / re-linearize / attention ceiling; `PRC2_OVERNIGHT.md:328-441`). It was
  held-out only and produced no test winner.
- **fp16 / INT**: `hpca_results/llm/int/<m>.csv` (SmoothQuant per-channel RTN; INT7 exists only here). AWQ INT8/6/4 come from
  `hpca_results/llm/frontend_awq/int_awq_vs_smoothquant.csv`. The paper's comment says there is no AWQ INT7.
- **AWQ-uniform** (uniform SC with the cell's deployed mask, allocator off): `frontend_awq/mp_vs_uniform_under_awq.md`.
  14B t64/t96 are replaced by the h20 values 8.9559 / 8.6846 from `frontend_awq/awq_mask_mp_table.csv`, because the .md holds
  h10 values there. There is no t40 uniform run.
- **SC ceiling** = `awq_mask_mp_table.csv` "Uniform, h=20%" at nominal 256 (AWQ). This is approximate. It uses an INT8 mask
  where the MP cells use INT7; `int_ablation` puts that width effect at 0.13–0.23%, so the true ceiling is slightly worse
  and the headroom slightly smaller. The mask composition may also differ. MP cells also carry protected channels, and
  14B t96 in fact beats this ceiling by 0.38%.
- **uniform_hybrid** (`hpca_results/llm/uniform_hybrid/uniform_hybrid_all.csv`) is SmoothQuant with an h=10% INT8-ranked mask.
  It is the comparator CLAUDE.md designates, but it matches neither the front end nor the mask of these cells. It is shown for completeness only.
- **_3 / _4**: `hpca_results/llm/ppl/mp_best_after_hpca_{3,4}/manifest.json` (`ppl`, `winner`, `realized_flop_avg_sl`).
  These are NOT comparable: they bundle the QK rebalance, the "grid", the v7 tables and (in `_4`) attnfirst masks that run attention entirely
  on INT7. `_4` 14B t64/t96 are also h10 cells.
- Noise floor: sd 0.0069 PPL, a dispatcher-reshuffle spread (`prc2/README.md` §Protocol), about 0.07–0.09% relative.

## 2. Per-cell tables

### Table A — per-cell PPL chain (full WikiText-2 test, ctx 2048; cost = trace MAC-weighted halved L)

| cell | bits | fp16 | parent (submitted, AWQ+h20) @cost | c17 per-group recon | r1 lin-Fisher | r2 joint | r3/r4 | **best(all)** @cost | arm | best ×fp16 | best vs parent | c17 vs parent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 4B_t32 | 6 | 10.0445 | 11.9886 @33.72 | 11.3305 | 11.1891 | 11.1953 | r3 11.2128 | **11.1891** @33.73 | c6gfis | 1.1140 | -6.67% | -5.49% |
| 4B_t40 | 6.32 | 10.0445 | 11.1440 @41.54 | 10.8766 | 10.8751 | 10.7729 | — | **10.7729** @41.50 | c7gfisla | 1.0725 | -3.33% | -2.40% |
| 4B_t48 | 6.58 | 10.0445 | 10.6491 @49.62 | 10.5030 | 10.4665 | 10.5532 | — | **10.4665** @49.77 | c6gfis | 1.0420 | -1.71% | -1.37% |
| 4B_t64 | 7 | 10.0445 | 10.4749 @65.05 | 10.4085 | 10.3645 | 10.3056 | — | **10.3056** @64.95 | c7gfisla | 1.0260 | -1.62% | -0.63% |
| 4B_t96 | 7.58 | 10.0445 | 10.2225 @95.97 | 10.1993 | 10.1995 | 10.1827 | — | **10.1827** @96.27 | c7gfisla | 1.0138 | -0.39% | -0.23% |
| llama8B_t32 | 6 | 7.2130 | 8.6758 @34.03 | 8.3899 | 8.2795 | 8.3108 | r3 8.3078; r4 8.2859 | **8.2795** @33.86 | c6gfis | 1.1479 | -4.57% | -3.30% |
| llama8B_t40 | 6.32 | 7.2130 | 8.2093 @41.66 | 8.0857 | 8.0300 / emp 8.0266 | 8.1023 | — | **8.0266** @41.54 | c6gfis_emp | 1.1128 | -2.23% | -1.51% |
| llama8B_t48 | 6.58 | 7.2130 | 7.9200 @49.63 | 7.8151 | 7.7878 | 7.7888 | — | **7.7878** @49.55 | c6gfis | 1.0797 | -1.67% | -1.32% |
| llama8B_t64 | 7 | 7.2130 | 7.7109 @65.07 | 7.6584 | 7.6435 | 7.6487 | — | **7.6435** @65.09 | c6gfis | 1.0597 | -0.87% | -0.68% |
| llama8B_t96 | 7.58 | 7.2130 | 7.6145 @95.29 | 7.5987 | 7.5855 | 7.5723 | — | **7.5723** @95.52 | c7gfisla | 1.0498 | -0.55% | -0.21% |
| 14B_t32 | 6 | 8.6383 | 9.3097 @33.56 | 9.1672 | 9.1041 | 9.0764 | — | **9.0764** @33.11 | c7gfisla | 1.0507 | -2.51% | -1.53% |
| 14B_t40 | 6.32 | 8.6383 | 9.0598 @41.57 | 8.9478 | 8.9029 / emp 8.9163 | 8.9076 | — | **8.9029** @41.18 | c6gfis | 1.0306 | -1.73% | -1.24% |
| 14B_t48 | 6.58 | 8.6383 | 8.9072 @49.46 | 8.8142 | 8.7894 | 8.8073 | — | **8.7894** @49.08 | c6gfis | 1.0175 | -1.32% | -1.04% |
| 14B_t64 | 7 | 8.6383 | 8.7758 @64.01 | 8.7110 | 8.6993 | 8.7658 | — | **8.6993** @63.59 | c6h20gfis | 1.0071 | -0.87% | -0.74% |
| 14B_t96 | 7.58 | 8.6383 | 8.6418 @96.14 | 8.6441 | 8.6455 | 8.6383 | — | **8.6383** @95.89 | c7h20gfisla | 1.0000 | -0.04% | +0.03% |
| 30B_t32 | 6 | 7.2613 | 9.4647 @34.12 | 8.9599 | 8.7163 | 8.5812 | — | **8.5812** @33.92 | c7gfisla | 1.1818 | -9.33% | -5.33% |
| 30B_t40 | 6.32 | 7.2613 | 8.5294 @42.65 | 8.4123 | 8.3210 | 8.0880 | r3 8.0856 | **8.0856** @42.46 | round3_local | 1.1135 | -5.20% | -1.37% |
| 30B_t48 | 6.58 | 7.2613 | 8.1784 @51.12 | 8.1159 | 8.0414 | 7.8571 | — | **7.8571** @50.60 | c7gfisla | 1.0820 | -3.93% | -0.76% |
| 30B_t64 | 7 | 7.2613 | 7.7146 @67.53 | 7.6601 | 7.6281 | 7.6483 | — | **7.6281** @67.10 | c7gfis | 1.0505 | -1.12% | -0.71% |
| 30B_t96 | 7.58 | 7.2613 | 7.5985 @95.58 | 7.5751 | 7.5336 | 7.5898 | — | **7.5336** @95.20 | c7gfis | 1.0375 | -0.85% | -0.31% |

### Table B — gap closure and remaining gap

| cell | parent ×fp16 | best ×fp16 | fp16-gap closed c17 / best | SC ceiling (uniform-256, h20, AWQ) | parent over ceiling | best over ceiling | ceiling-gap closed | PPL change needed for ≤1.10× | for ≤1.05× |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4B_t32 | 1.1935 | 1.1140 | 33.9% / 41.1% | 10.1186 | +18.48% | +10.58% | 42.8% | -1.25% | -5.74% |
| 4B_t40 | 1.1095 | 1.0725 | 24.3% / 33.8% | 10.1186 | +10.13% | +6.47% | 36.2% | pass | -2.10% |
| 4B_t48 | 1.0602 | 1.0420 | 24.2% / 30.2% | 10.1186 | +5.24% | +3.44% | 34.4% | pass | pass |
| 4B_t64 | 1.0428 | 1.0260 | 15.4% / 39.3% | 10.1186 | +3.52% | +1.85% | 47.5% | pass | pass |
| 4B_t96 | 1.0177 | 1.0138 | 13.0% / 22.4% | 10.1186 | +1.03% | +0.63% | 38.3% | pass | pass |
| llama8B_t32 | 1.2028 | 1.1479 | 19.5% / 27.1% | 7.5282 | +15.24% | +9.98% | 34.5% | -4.17% | -8.53% |
| llama8B_t40 | 1.1381 | 1.1128 | 12.4% / 18.3% | 7.5282 | +9.05% | +6.62% | 26.8% | -1.15% | -5.64% |
| llama8B_t48 | 1.0980 | 1.0797 | 14.8% / 18.7% | 7.5282 | +5.20% | +3.45% | 33.7% | pass | -2.75% |
| llama8B_t64 | 1.0690 | 1.0597 | 10.5% / 13.5% | 7.5282 | +2.43% | +1.53% | 36.9% | pass | -0.91% |
| llama8B_t96 | 1.0557 | 1.0498 | 3.9% / 10.5% | 7.5282 | +1.15% | +0.59% | 48.9% | pass | pass (margin 0.02%) |
| 14B_t32 | 1.0777 | 1.0507 | 21.2% / 34.7% | 8.6709 | +7.37% | +4.68% | 36.5% | pass | -0.07% |
| 14B_t40 | 1.0488 | 1.0306 | 26.6% / 37.2% | 8.6709 | +4.49% | +2.68% | 40.3% | pass | pass |
| 14B_t48 | 1.0311 | 1.0175 | 34.6% / 43.8% | 8.6709 | +2.73% | +1.37% | 49.8% | pass | pass |
| 14B_t64 | 1.0159 | 1.0071 | 47.2% / 55.6% | 8.6709 | +1.21% | +0.33% | 72.9% | pass | pass |
| 14B_t96 | 1.0004 | 1.0000 | —% / —% | 8.6709 | -0.34% | -0.38% | —% | pass | pass |
| 30B_t32 | 1.3034 | 1.1818 | 22.9% / 40.1% | 7.4826 | +26.49% | +14.68% | 44.6% | -6.92% | -11.15% |
| 30B_t40 | 1.1746 | 1.1135 | 9.2% / 35.0% | 7.4826 | +13.99% | +8.06% | 42.4% | -1.21% | -5.70% |
| 30B_t48 | 1.1263 | 1.0820 | 6.8% / 35.0% | 7.4826 | +9.30% | +5.00% | 46.2% | pass | -2.96% |
| 30B_t64 | 1.0624 | 1.0505 | 12.0% / 19.1% | 7.4826 | +3.10% | +1.94% | 37.3% | pass | -0.05% |
| 30B_t96 | 1.0464 | 1.0375 | 7.0% / 19.2% | 7.4826 | +1.55% | +0.68% | 56.0% | pass | pass |

### Table C — comparators at matched budget (NOT all comparable; see notes)

| cell | best(all) | INT ref | best vs INT | AWQ-uniform + deployed h20 mask | best vs AWQ-uniform | uniform_hybrid (SQ, h10) | _3 | _4 @cost [winner] | best vs _4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4B_t32 | 11.1891 | 10.0503 (INT6-AWQ) | +11.33% | 16.7650 | -33.26% | 56.5943 | 11.0860 | 10.9927 @26.8 [attnfirst] | +1.79% |
| 4B_t40 | 10.7729 | 10.0503 (INT6-AWQ) | +7.19% | — | — | — | 10.7868 | 10.6667 @33.7 [attnfirst] | +1.00% |
| 4B_t48 | 10.4665 | 10.0503 (INT6-AWQ) | +4.14% | 12.0231 | -12.95% | 13.0657 | 10.3783 | 10.3783 @— [a3:prevv1:qk+kband] | +0.85% |
| 4B_t64 | 10.3056 | 9.9754 (INT7-SQ) | +3.31% | 11.1020 | -7.17% | 11.6299 | 10.1974 | 10.1803 @57.3 [attnfirst] | +1.23% |
| 4B_t96 | 10.1827 | 9.9754 (INT7-SQ) | +2.08% | 10.3228 | -1.36% | 10.6188 | 10.0702 | 10.0702 @92.3 [a3:prcqk+grid] | +1.12% |
| llama8B_t32 | 8.2795 | 7.3039 (INT6-AWQ) | +13.36% | 14.9109 | -44.47% | 25.6495 | 8.4864 | 8.1354 @28.7 [attnfirst] | +1.77% |
| llama8B_t40 | 8.0266 | 7.3039 (INT6-AWQ) | +9.89% | — | — | — | 8.1302 | 7.9144 @32.4 [attnfirst] | +1.42% |
| llama8B_t48 | 7.7878 | 7.3039 (INT6-AWQ) | +6.63% | 9.7795 | -20.37% | 10.9030 | 7.8777 | 7.6776 @40.5 [attnfirst] | +1.44% |
| llama8B_t64 | 7.6435 | 7.2457 (INT7-SQ) | +5.49% | 8.4933 | -10.01% | 8.9866 | 7.6912 | 7.4621 @57.8 [attnfirst] | +2.43% |
| llama8B_t96 | 7.5723 | 7.2457 (INT7-SQ) | +4.51% | 7.7000 | -1.66% | 7.8630 | 7.5992 | 7.5647 @88.0 [qk-a0.5(prcqk)] | +0.10% |
| 14B_t32 | 9.0764 | 8.6773 (INT6-AWQ) | +4.60% | 10.5818 | -14.23% | 12.3157 | 9.1581 | 9.1482 @31.8 [attnfirst] | -0.78% |
| 14B_t40 | 8.9029 | 8.6773 (INT6-AWQ) | +2.60% | — | — | — | 9.0522 | 9.0183 @36.1 [attnfirst] | -1.28% |
| 14B_t48 | 8.7894 | 8.6773 (INT6-AWQ) | +1.29% | 9.1975 | -4.44% | 9.7016 | 8.8571 | 8.8571 @50.8 [a3:prc] | -0.76% |
| 14B_t64 | 8.6993 | 8.6283 (INT7-SQ) | +0.82% | 8.9559 | -2.87% | 9.1859 | 8.7060 | 8.7060 @63.9 [a3:parent+grid] | -0.08% |
| 14B_t96 | 8.6383 | 8.6283 (INT7-SQ) | +0.12% | 8.6846 | -0.53% | 8.7531 | 8.6102 | 8.6102 @95.7 [a3:parent+grid] | +0.33% |
| 30B_t32 | 8.5812 | 7.2846 (INT6-AWQ) | +17.80% | 12.7425 | -32.66% | 14.6014 | 7.9073 | 7.7566 @33.7 [attnfirst] | +10.63% |
| 30B_t40 | 8.0856 | 7.2846 (INT6-AWQ) | +11.00% | — | — | — | 7.8016 | 7.7125 @35.5 [attnfirst] | +4.84% |
| 30B_t48 | 7.8571 | 7.2846 (INT6-AWQ) | +7.86% | 9.0604 | -13.28% | 9.5098 | 7.5663 | 7.4702 @49.0 [attnfirst] | +5.18% |
| 30B_t64 | 7.6281 | 7.2325 (INT7-SQ) | +5.47% | 8.3856 | -9.03% | 8.6731 | 7.3792 | 7.3792 @69.2 [a3:prcqk] | +3.37% |
| 30B_t96 | 7.5336 | 7.2325 (INT7-SQ) | +4.16% | 7.6576 | -1.62% | 7.7771 | 7.2922 | 7.2922 @94.0 [a3:prcqk] | +3.31% |

### Table D — per-round increments on the best-so-far (negative = better)

| cell | c17→+r1 | →+r2 | →+r3 (Codex) | →+r4/r5 (Codex) | spread of comparable candidates |
|---|---:|---:|---:|---:|---:|
| 4B_t32 | -1.248% | +0.000% | +0.000% | no test (confirm +0.14%) | 1.26% |
| 4B_t40 | -0.014% | -0.940% | — | — | 0.96% |
| 4B_t48 | -0.347% | +0.000% | — | — | 0.83% |
| 4B_t64 | -0.423% | -0.568% | — | — | 1.00% |
| 4B_t96 | +0.000% | -0.162% | — | — | 0.16% |
| llama8B_t32 | -1.316% | +0.000% | +0.000% | test +0.077% vs incumbent (not adopted) | 1.33% |
| llama8B_t40 | -0.730% | +0.000% | — | — | 0.94% |
| llama8B_t48 | -0.349% | +0.000% | — | — | 0.35% |
| llama8B_t64 | -0.195% | +0.000% | — | — | 0.20% |
| llama8B_t96 | -0.174% | -0.174% | — | — | 0.35% |
| 14B_t32 | -0.688% | -0.304% | — | — | 1.00% |
| 14B_t40 | -0.502% | +0.000% | — | — | 0.50% |
| 14B_t48 | -0.280% | +0.000% | — | — | 0.28% |
| 14B_t64 | -0.134% | +0.000% | — | — | 0.76% |
| 14B_t96 | +0.000% | -0.067% | — | — | 0.08% |
| 30B_t32 | -2.719% | -1.550% | — | no test (confirm −0.33%, z −0.68) | 4.41% |
| 30B_t40 | -1.085% | -2.800% | -0.030% | no test (confirm −0.08%, z −0.27) | 4.04% |
| 30B_t48 | -0.918% | -2.293% | — | — | 3.29% |
| 30B_t64 | -0.418% | +0.000% | — | — | 0.42% |
| 30B_t96 | -0.547% | +0.000% | — | — | 0.75% |

### Table E — SC work split in full-test traces (halved L; MAC-weighted)

| cell | linear MAC share | attention (qk+av) MAC share | parent L: lin / qk / av | best L: lin / qk / av | attention share of best SC cycles |
|---|---:|---:|---:|---:|---:|
| 4B_t32 | 0.869 | 0.131 | 26.1 / 83.4 / 85.5 | 26.1 / 83.6 / 85.6 | 0.328 |
| 4B_t40 | 0.869 | 0.131 | 35.2 / 82.7 / 84.7 | 33.8 / 94.9 / 89.9 | 0.291 |
| 4B_t48 | 0.869 | 0.131 | 42.2 / 98.3 / 99.6 | 42.4 / 98.4 / 99.6 | 0.260 |
| 4B_t64 | 0.869 | 0.131 | 61.1 / 89.4 / 93.2 | 59.2 / 104.5 / 102.6 | 0.208 |
| 4B_t96 | 0.869 | 0.131 | 91.9 / 127.5 / 118.7 | 91.6 / 127.9 / 126.7 | 0.173 |
| llama8B_t32 | 0.925 | 0.075 | 28.9 / 97.1 / 96.1 | 28.8 / 97.1 / 96.1 | 0.214 |
| llama8B_t40 | 0.927 | 0.073 | 37.2 / 99.0 / 98.2 | 37.1 / 99.0 / 98.2 | 0.172 |
| llama8B_t48 | 0.927 | 0.073 | 43.5 / 128.0 / 128.0 | 43.4 / 128.0 / 128.0 | 0.187 |
| llama8B_t64 | 0.927 | 0.073 | 60.1 / 128.0 / 128.0 | 60.2 / 128.0 / 128.0 | 0.143 |
| llama8B_t96 | 0.927 | 0.073 | 93.2 / 126.2 / 117.2 | 93.4 / 127.9 / 118.0 | 0.093 |
| 14B_t32 | 0.945 | 0.055 | 29.9 / 96.8 / 96.0 | 29.8 / 93.1 / 87.4 | 0.149 |
| 14B_t40 | 0.945 | 0.055 | 39.1 / 85.5 / 83.8 | 38.7 / 85.5 / 83.8 | 0.112 |
| 14B_t48 | 0.945 | 0.055 | 45.8 / 113.4 / 112.6 | 45.4 / 113.4 / 112.7 | 0.126 |
| 14B_t64 | 0.945 | 0.055 | 62.6 / 87.9 / 87.5 | 62.2 / 87.9 / 87.5 | 0.075 |
| 14B_t96 | 0.945 | 0.055 | 94.5 / 127.4 / 121.7 | 94.1 / 127.7 / 127.1 | 0.073 |
| 30B_t32 | 0.767 | 0.233 | 25.7 / 65.1 / 59.5 | 25.1 / 88.1 / 44.5 | 0.432 |
| 30B_t40 | 0.767 | 0.233 | 36.0 / 67.6 / 62.6 | 32.6 / 92.8 / 61.8 | 0.411 |
| 30B_t48 | 0.767 | 0.233 | 45.3 / 72.3 / 68.8 | 39.7 / 105.7 / 72.3 | 0.398 |
| 30B_t64 | 0.767 | 0.233 | 52.5 / 117.3 / 117.0 | 51.9 / 117.3 / 117.0 | 0.406 |
| 30B_t96 | 0.767 | 0.233 | 87.5 / 127.5 / 118.2 | 87.0 / 127.5 / 118.3 | 0.299 |

### Table F — best(all) versus the fixed weight-bearing partition (response.md Table R2; NOT iso-cost, different mask)

| cell | fixed partition (7 linears on SC, qk/av on INT8; v7 tables) PPL @SC cost | best(all) PPL @SC cost | best vs fixed |
|---|---:|---:|---:|
| 4B_t32 | 11.0955 @26.99 | 11.1891 @33.73 | +0.84% |
| 4B_t48 | 10.4264 @40.09 | 10.4665 @49.77 | +0.38% |
| 4B_t64 | 10.2014 @57.37 | 10.3056 @64.95 | +1.02% |
| llama8B_t32 | 8.4343 @28.90 | 8.2795 @33.86 | -1.84% |
| llama8B_t48 | 7.7849 @41.15 | 7.7878 @49.55 | +0.04% |
| llama8B_t64 | 7.5295 @57.89 | 7.6435 @65.09 | +1.51% |
| 14B_t32 | 9.2258 @31.33 | 9.0764 @33.11 | -1.62% |
| 14B_t48 | 8.9124 @46.76 | 8.7894 @49.08 | -1.38% |
| 14B_t64 | 8.8394 @51.77 | 8.6993 @63.59 | -1.59% |
| 30B_t32 | 7.7413 @32.94 | 8.5812 @33.92 | +10.85% |
| 30B_t48 | 7.4551 @48.44 | 7.8571 @50.60 | +5.39% |
| 30B_t64 | 7.4116 @54.63 | 7.6281 @67.10 | +2.92% |

Source: `hpca_results/llm/weight_bearing_sc/results.csv` (`ppl`, `realized_sc_samples`). The fixed partition is better in 8/12 cells.

Notes on the tables above:
- The cost in Table A is the MAC-weighted mean of halved L over every SC group in the full-test trace (the `build_prc2_archive.trace_cost`
  definition, recomputed here). Best(all) costs sit −1.34…+0.31% from the parent (mean −0.37%), so the cost drift is
  conservative on average.
- In Table B, "fp16-gap closed" = (parent − best) / (parent − fp16). It is undefined for 14B t96, whose parent is already at 1.0004× fp16.
  "Ceiling-gap closed" = (parent − best) / (parent − ceiling).
- In Table C, "INT ref" is the exact width only at t32 (INT6) and t64 (INT7). The INT7 values are SmoothQuant, so they do not match
  the front end. The `_4` 4B t48 cost is "—" in `_4/SUMMARY.md`.

## 3. Counts and per-budget summaries

| archive / arm | ≤1.05× fp16 | ≤1.10× fp16 | comparable to T1? |
|---|---:|---:|---|
| parent (submitted, AWQ+h20) | 7/20 | 13/20 | baseline |
| c17 (per-group reconstruction) | 8/20 | 14/20 | yes |
| **best(all)** | **9/20** | **15/20** | yes (test-selected) |
| `_3` (QK + grid + v7) | 10/20 | 17/20 | no |
| `_4` (loss-priced mask; attnfirst runs attention on INT7) | 12/20 | 19/20 | no |
| AWQ-uniform + deployed mask (16 cells, no t40) | 3/16 | 6/16 | allocator-off comparator |

| bits (target) | geomean ×fp16 parent / c17 / best / `_4` | mean ΔPPL vs parent: c17 / best | best ≤1.05× / ≤1.10× | mean ceiling-gap closed |
|---|---|---|---:|---:|
| 6 (t32) | 1.1917 / 1.1449 / 1.1225 / 1.0871 | −3.91% / −5.77% | 0/4, 1/4 | 39.6% |
| 6.32 (t40) | 1.1168 / 1.0986 / 1.0818 / 1.0662 | −1.63% / −3.12% | 1/4, 2/4 | 36.4% |
| 6.58 (t48) | 1.0783 / 1.0662 / 1.0550 / 1.0378 | −1.13% / −2.16% | 2/4, 4/4 | 41.0% |
| 7 (t64) | 1.0474 / 1.0401 / 1.0356 / 1.0180 | −0.69% / −1.12% | 2/4, 4/4 | 48.7% |
| 7.58 (t96) | 1.0298 / 1.0280 / 1.0251 / 1.0129 | −0.18% / −0.46% | 4/4, 4/4 | 47.7% (n=3) |

Per model, mean ΔPPL of best (c17): 4B −2.74% (−2.02%), llama8B −1.98% (−1.40%), 14B −1.29% (−0.90%), 30B −4.09% (−1.70%).
Mean ceiling-gap closed: 4B 39.8%, llama8B 36.2%, 14B 49.9%, 30B 45.3%. **llama8B is the weakest model.**
It closes only 18.3% of its parent's fp16 gap at t40, 13.5% at t64 and 10.5% at t96.

Fixed-rule versus test-selected (mean ΔPPL vs parent over 20 cells): always c17 −1.51%, always r1 −2.09%, always r2 −2.31%,
min(r1,r2) −2.52%, best(all) −2.53%. Picking per cell between r1 and r2 on the test set is worth about 0.2–0.4 pp of the headline.
Rounds 3–5 are worth 0.0015 pp.

## 4. The pure T1 (granularity) effect

Same calibrator, windows, AWQ, dense ladder, per-bucket budget and solver; only the row-shared switch differs
(`PAPER_METRICS_20260927.md` "Same-ladder granularity controls"; `prc2/SUMMARY.md` §3):

| cell | parent | per-row same-calib (r17/r17m) @cost | c17 @cost | granularity only | recalibration only | granularity share of c17 / of best(all) log-gain |
|---|---:|---:|---:|---:|---:|---:|
| 4B t32 (r17m, cost-matched) | 11.9886 | 11.8429 @33.52 | 11.3305 @33.67 | −4.33% (+0.45% cost) | −1.22% | 78% / 64% |
| llama8B t32 | 8.6758 | 8.5659 @33.61 | 8.3899 @33.73 | −2.05% (+0.36%) | −1.27% | 62% / 44% |
| 4B t48 | 10.6491 | 10.6772 @49.11 | 10.5030 @49.57 | −1.63% (+0.92%) | +0.26% | 119% / 95% |

No granularity control exists for 14B, 30B, or any t40/t64/t96 cell. No per-row Fisher (r1/r2-style) control exists either.
Without one, the −0.5…−2.8% that rounds 1–2 add on top of c17 cannot be credited to granularity: the same loss-aware
re-pricing might also help a per-row allocation. On 30B t32/t40/t48, round 2's gain (−1.55 / −2.80 / −2.29%) comes with a large
re-split of per-row attention. Trace mean L for qk goes 65.1→88.1 / 67.6→92.8 / 72.3→105.7, and for av 59.5→44.5 at t32 (Table E).
**That is an attention-budget effect with nothing to do with granularity.**

Linear squared error versus PPL: the held-out pre-flight (`prc2/SUMMARY.md` §4) shows c17 cuts linear-projection squared error
by 40/39/37/34/21% on 4B (t32…t96), 21→9% on llama8B, 22–27% then 16% on 14B, and 27→8% on 30B. The PPL gained per 10 pp of error
removed falls from −1.37 / −1.57 / −0.68 / −1.99% at t32 to −0.11 / −0.22 / +0.02 / −0.37% at t96 (4B / 8B / 14B / 30B).
Caveat: 14B t64/t96 are h10 diag rows, and 30B t32 is c17 rather than c17e32. **At high budgets the linear SC error is no longer
what limits PPL.** What remains is the per-row attention (7–43% of SC cycles; mean L 83–128 on the dense models and 44–118 on 30B, against a cap of 128; at 8B t48/t64 it is
already pinned at 128.00), the INT7 mask, and the substrate floor (Table B ceiling).

## 5. Why Codex's rounds 3–5 could not move the numbers

| round | cells | cycles moved per candidate (share of total SC cycles) | 6-window search result | confirmation (16 windows) | full test |
|---|---|---|---|---|---|
| r3 local (62032412) | 4B t32, 8B t32, 30B t40 | ≈2.0% (0.9% for two 30B QK moves) | 19/20 worse, in BOTH directions; mean +1.97%, median +0.89%; 8B linears←qk +17.6% | +0.43% / +0.35% / −0.045% | +0.212% / +0.342% / −0.030% |
| r4 adjacent (62104342) | 4B t32, 8B t32 | 0.005–0.068% (median 0.028% over 48 candidates); requested caps 0.25% / 0.5% | best −0.45% (z −1.23) / −0.75% (z −2.19) | +0.14% (z +0.48) / −0.40% (z −1.98, passed) | not run / **+0.077%** |
| r5 adjacent 30B (62105189) | 30B t32, t40 | 0.009–0.068% | best −1.14% (z −2.53) / −0.64% (z −2.00) | −0.33% (z −0.68) / −0.08% (z −0.27) | not run |

- **Round 3 shows the operator-family split is at a local optimum.** A 2% move either way (qk↔linears, av↔linears, qk↔av,
  projections↔MLP) makes loss worse. The postmortem's own diagnosis (`ROUND3_POSTMORTEM_20260927.md`) is that
  coincident zero thresholds sent QK rows straight from the longest rung to the shortest. It does not explain the
  symmetric worsening on 4B, where 8/8 candidates came out +0.55…+2.25%.
- **Round 4/5 was underpowered by design.** The locality caps (≤5% of groups / MACs per bucket, 80% headroom) moved 4–90× less
  work than the requested 0.25–0.5% transfer. Take the measured within-cell elasticity d lnPPL / d ln cost = −0.44
  (4B t32, `prc2/SUMMARY.md` §3). A 0.03% transfer can then change PPL by at most about 0.01%. The median paired se on 6
  windows was 0.40% PPL (range 0.14–1.10%), so SNR was ≲0.03. The winners match the expected minimum of 12 null draws:
  about −1.63 × 0.40% ≈ −0.65%, against −0.45 / −0.75 / −1.14 / −0.64% observed. 39/48 search deltas were negative
  (mean −0.25%), which points to a baseline-draw offset shared across candidates rather than a real effect. The confirmations shrank
  toward 0, and the one full test went the wrong way.
- **The measurement resolution sets a floor on any new round.** The full test is about 146 (Qwen) / 141 (Llama) windows of 2048 tokens,
  with a noise floor of about 0.07–0.09%. A 16-window confirmation has se ≈ 0.2–0.5% PPL. Effects below about 0.5% per cell
  therefore cannot be validated before the full test. A round aimed at sub-0.3% gains can only be judged on the full test, and a
  best-of-k selection there carries about 1 sd (≈0.07%) of optimism.

## 6. Rankings

**By remaining allocation-addressable gap (best over the SC ceiling; the upper bound on what pure allocation could recover):**
30B t32 +14.68% > 4B t32 +10.58% > 8B t32 +9.98% > 30B t40 +8.06% > 8B t40 +6.62% > 4B t40 +6.47% > 30B t48 +5.00% >
14B t32 +4.68% > 8B t48 +3.45% ≈ 4B t48 +3.44% > 14B t40 +2.68% > 30B t64 +1.94% > 4B t64 +1.85% > 8B t64 +1.53% >
14B t48 +1.37% > 30B t96 +0.68% > 4B t96 +0.63% > 8B t96 +0.59% > 14B t64 +0.33% > 14B t96 −0.38%.
(At t32/t40 the ceiling is far from reachable at that budget. There the within-bucket oracle is the realistic bound; see §8.)

**By remaining gap to fp16 (best ×fp16):** 30B t32 1.182, 8B t32 1.148, 4B t32 1.114, 30B t40 1.114, 8B t40 1.113,
30B t48 1.082, 8B t48 1.080, 4B t40 1.073, 8B t64 1.060, 14B t32 1.051, 30B t64 1.051, 8B t96 1.050, 4B t48 1.042,
30B t96 1.038, 14B t40 1.031, 4B t64 1.026, 14B t48 1.018, 4B t96 1.014, 14B t64 1.007, 14B t96 1.000.

**By weakness of the T1 story** (small direct c17 gain, a large share of the gain from non-granularity rounds, or a gain at noise level):
1. 14B t96: c17 +0.03% (worse), best −0.04% = 0.0035 PPL = 0.5 noise-sd. There is no headroom; the parent was already 1.0004× fp16.
2. 8B t96 (c17 −0.21%, 63% of best from rounds), 4B t96 (−0.23%, 42%), 30B t96 (−0.31%, 64%): about 6–9 noise-sd, but under 0.9%.
3. 30B t48 (c17 −0.76% but best −3.93%, 81% from the round-2 attention re-split) and 30B t40 (c17 −1.37% versus −5.20%, 74%).
   The headline gains here are not granularity.
4. The t64 cells: c17 −0.63 / −0.68 / −0.74 / −0.71%. 4B t64 gets 61% of its best from rounds.
5. The strongest T1 cells are 4B t32 (c17 −5.49%, granularity −4.33% measured), 30B t32 (c17 −5.33%), 8B t32 (−3.30%, granularity −2.05%)
   and 4B t40 (−2.40%).

## 7. Interpretations of "not good enough", quantified

| yardstick | where per-group stands | gap still open | reachable by pure allocation? |
|---|---|---|---|
| ≤1.10× fp16 (the paper's quality bound; "score = lowest passing budget") | 15/20; lowest passing budget per model: 4B t40, 8B t48, 14B t32, 30B t48 (parent: t48/t48/t32/t64) | 4B t32 −1.25%, 8B t40 −1.15%, 30B t40 −1.21% would each lower a model by one budget step; 8B t32 −4.17%, 30B t32 −6.92% | the ~1.2% targets are plausible at t32/t40 (headroom over the ceiling is 6–11%); 30B t32 to 1.10× is doubtful |
| ≤1.05× fp16 | 9/20 (parent 7) | 14B t32 −0.07%, 30B t64 −0.05% (inside noise); 8B t64 −0.91%; 4B t40 −2.10%; 8B t48 −2.75%; 30B t48 −2.96%; 8B t96 passes by only 0.02% | only in the near-miss cells, and those are at noise level |
| INT at matched bits | +11.3 / +13.4 / +4.6 / +17.8% vs INT6 at 6 bits; +0.8…+5.5% vs INT7 at 7 bits | the SC ceiling is already +3.07% (8B) / +2.72% (30B) vs INT6 | **No.** The paper argues SC on energy |
| the `_3` / `_4` levers (QK rebalance, grid, attnfirst masks) | 16/20 cells behind `_4` (+0.10…+10.63%); ahead on 14B t32/t40/t48 (−0.78 / −1.28 / −0.76%) | 30B +3.3…+10.6%; 4B/8B +0.9…+2.4% | **No** for most of it: `_4` 30B t96 is below the pure-allocation ceiling |
| the claimed granularity benefit ("submitted MP is pessimistic") | pessimism = −4.33 / −2.05 / −1.63% (3 controls); c17 −0.18% mean at 7.58 bits | a reviewer can call the fix a no-op at ≥7 bits (c17 −0.18…−0.74%, 14B t96 +0.03%) | the magnitude at high budget is bounded by the ceiling (≤0.7% at t96) |
| the fixed weight-bearing partition in response.md Table R2 (qk/av on INT8, v7 tables) | PaYN is worse in 8/12: 30B +10.85 / +5.39 / +2.92%, 4B +0.84 / +0.38 / +1.02%, 8B t48/t64 +0.04 / +1.51% | not iso-cost (the fixed arm spends 3–20% fewer SC samples) and a different mask/width | not by allocation (it changes the INT mask); flag for T3 |

## 8. Implications for ≤3 new pure-allocation rounds (gap view only)

- **Where to spend.** Allocation headroom is concentrated at t32/t40, plus 30B/4B t48. The threshold-crossing targets are
  4B t32, llama8B t40 and 30B t40, each about 1.2–1.3%. Cells at t64 and above have ≤1.9% headroom (t96 ≤0.7%, 14B t96 none), so a round
  targeting them cannot produce a visible change in the paper.
- **The effect size a round needs.** Aim for ≥0.5–1% per cell, i.e. well above the 16-window se of 0.2–0.5%. Round 4's 0.03% moves could not be seen.
- **A rough within-bucket bound.** Held-out linear error for today's statistic versus the unconstrained oracle, at the parent's per-bucket
  budgets (`prc2/SUMMARY.md` §5), is 0.600→0.518 (4B t32), 0.790→0.748 (8B t32), 0.748→0.635 (14B t48) and 0.762→0.620
  (30B t48). Linearly extrapolating with the §4 slopes gives roughly −1.1 / −0.7 / −0.5 / −0.4% PPL. That is at most one threshold
  crossing. The oracle is also not deployable with the fixed absmax statistic.
- **Operator-family budget re-splits look exhausted at ±2% steps on 4B/8B** (round 3), and on dense cells in round 2. The one
  large lever left inside the rules is attention's per-row budget on 30B. There round 2 won −1.55…−2.80% at t32–t48, and
  attention is 23% of MACs and 30–43% of cycles. 30B t64/t96 round 2 lost (+0.27% / +0.75%).

## 9. What the rebuttal needs from the T1/T2 numbers

- `rebuttal_task.md` T1: admit that dispatch was per-row while quantization was per group, and that the submitted MP accuracy is therefore *pessimistic*.
  T2: report new data, "at least LLM and ViT". Both LLM requirements are already met by the numbers above: c17 19/20, best(all) 20/20,
  and 3 granularity controls. The ViT reruns are still outstanding (`response.md` header).
- `response.md` T1/T2 numbers were checked against the artifacts and match: controls −1.63…−4.33% at +0.36…+0.92% cost; c17 19/20 at
  −1.51%; best −2.53%, t32 −5.77%, max −9.33% (30B t32), min −0.04% (14B t96); Table R1 geomeans 1.192/1.145/1.123 …
  1.030/1.028/1.025; cost range −1.34…+0.31%; 4/4 models ≤1.10× at 6.58 bits versus 3/4 submitted.
- The rebuttal's weak spots, from the gap view, are the following. Granularity is isolated in only 3 cells, none of them 14B/30B or ≥7 bits.
  Best(all) folds in loss-aware re-pricing and a per-row attention re-split that have no per-row control, plus test selection
  worth about 0.2–0.4 pp. And Table R2 (T3) shows PaYN losing to the fixed INT8-attention partition in 8/12 cells.

## 10. Places where the documents disagree with the artifacts

1. **`SCMP/CLAUDE.md` says the submitted paper cites `mp_best/` (SmoothQuant).** The submitted `tab:quality-llm` (Overleaf `4290310`)
   actually prints the AWQ re-measures: 4B 6-bit 11.99 = AWQ 11.9886, whereas SmoothQuant `mp_best_all.csv` gives 12.5715. 14B 7 / 7.58 bits
   are the h20 configs, 8.78 / 8.64. BEST_ALL's "submitted" = AWQ+h20 is the correct reading.
2. **`SCMP/CLAUDE.md` (prc2 paragraph) and `prc2/SUMMARY.md` are stale.** They show 14B t64/t96 against h10 parents (8.7341 / 8.6150) and
   say "every finished cell improves". With h20 parents, c17 is 19/20 and 14B t96 is +0.03%. SUMMARY §1b still shows 4B t40 round 2 as "—",
   but `c7gfisla` = 10.7729 exists and is the best(all) winner. `response.md` already flags SUMMARY as stale, yet `SCMP/CLAUDE.md` still calls it "source of truth".
3. **`scmp_llm/CLAUDE.md` STATUS says "No round-4 PPL results yet".** All four round-4/5 tasks finished (logs end 15:20–17:36 EDT, Sep 27).
   No round-4 results section or doc exists. Outcome: no confirmed winner, and the llama8B t32 test came out +0.077% (8.285860 vs 8.279485).
4. **`PRC_ADJACENT_20260927.md` calls 0.25% / 0.5% transfers "ceilings".** The actual moves were 0.005–0.068% of cycles (`proposal.transferred_fraction`),
   4–90× smaller. The doc does not quantify this.
5. **Trace headers read `use_smoothquant: "1"`, `smoothquant_alpha: 0.5` even for AWQ runs** (e.g. the 4B t32 c6gfis trace). The prc2
   README says this field is inert. Do not infer the front end from trace headers, and that includes the `weight_bearing_sc` baseline.
6. **In `frontend_awq/mp_vs_uniform_under_awq.md`, 14B t64/t96 are h10 (MP 8.7341 / 8.6150, uniform 9.0653 / 8.6707).** The paper's comparison
   uses h20 (uniform 8.9559 / 8.6846), per `PAPER_METRICS_20260927.md`.
