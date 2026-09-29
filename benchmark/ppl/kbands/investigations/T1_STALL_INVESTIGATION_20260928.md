> **READ-ONLY INVESTIGATION (2026-09-28).** No GPU work, no model loads, and no repo files were changed to produce it. Copied
> verbatim into the repo on 2026-09-28 from the session scratch directory; only this header block was added.
>
> **Path resolution.** In the text below, `inv/` means the session scratch directory
> `/tmp/claude-114365137/-home-allenjin-Projects-SCMP/aad3505a-096e-4395-aba0-71871b9d6077/scratchpad/inv/`.
> - The five mapper reports `inv/{rounds,gap,algo,attrib,protocol}.md` are copied byte-for-byte to
>   [`t1_stall_20260928/`](t1_stall_20260928/) next to this file (sha256 in its `README.md`).
> - Every other `inv/…` path (the H1–H7 lens dirs `inv/h*mech/`, `inv/h*skeptic/`, `inv/h5/`, `inv/synth/`, `inv/final/`,
>   `inv/gap_work/`, and the loose `inv/*.py` / `inv/*.json`) exists **only** in that scratch dir, which is not durable. Treat
>   those citations as provenance notes, not as reproducible repo artifacts.
> - Turbo (`/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/…`), `hpca_results/…`, and repo paths are durable and unchanged.
>
> **Follow-ups:** round-4/5 outcomes → [`../PRC_ADJACENT_RESULTS_20260927.md`](../PRC_ADJACENT_RESULTS_20260927.md);
> rounds 6–8 plan (incl. the round-6 critic's corrections to §5b/§6) → [`../ROUNDS_6_8_PLAN_20260928.md`](../ROUNDS_6_8_PLAN_20260928.md).

# T1 per-group allocation: why the PPL is not better, and what ≤3 pure-allocation rounds can still buy

2026-09-28. Read-only consolidation of five mapper reports (`inv/{rounds,gap,algo,attrib,protocol}.md`),
seven hypotheses (H1–H7), each checked by a data lens and a code lens, and my own re-checks where
the two lenses disagreed. No GPU work, no model loads, no repo files changed. `inv/` =
`/tmp/claude-114365137/-home-allenjin-Projects-SCMP/aad3505a-096e-4395-aba0-71871b9d6077/scratchpad/inv/`.
Units: code/trace stream lengths are HALVED (nominal = 2×). %PPL ≈ 100·ΔNLL at these sizes.
"Parent" = the submitted per-row cell re-measured under AWQ + h=20% INT7 mask (14B t64/t96 use the
requested h20 parents). All 20 cells are full protocol (full WikiText-2, ctx 2048, `ppl_max_tokens=0`).

## Bottom line

1. **The per-group fix works, but the remaining gap is mostly outside what allocation can reach.**
   - Best(all) improves 20/20 cells, mean −2.53% vs parent.
   - By round the mean splits as follows (`inv/rounds.md` §3, `inv/decomp.py`):
     - c17 per-group recalibration: −1.51%;
     - global Fisher λ (round 1): −0.60%;
     - joint linear+attention λ (round 2): −0.44%;
     - everything after round 2: −0.002%.
   - Dense t32–t48: most of the remaining linear loss is per-token loss sensitivity. No runtime-visible
     statistic sees it, so deployable headroom there is ≤0.25–0.5% at t32.
   - t96 is within 0.35–0.67% of the SC ceiling (uniform 256 nominal + the same mask).
   - The one place a ≥0.5% allocation lever plausibly remains is 30B t32–t48 attention: qk sits at the
     inherited ladder top, and 0.03–0.05 nats are still unexplained after round 2.
2. **Rounds 3–5 could not have found anything.** No proposal carried loss information.
   - Round 3 moved about 2% of cycles with non-local, rung-skipping offsets. Its candidates were
     genuinely harmful: 19/20 worse.
   - Rounds 4/5 moved 0.005–0.068% of cycles. That is below both the per-call re-dispatch cascade each
     edit triggers (median 3–15× the planned move) and the 0.5–1.1% confirmation MDE.
   - The one confirmation "pass" (llama8B t32) lost on test: +0.077%.
3. **Realistic value of ≤3 new pure-allocation rounds: about −0.05 to −0.3 pp on the 20-cell mean**, not
   the −0.2 to −0.5 pp in the diagnoser's draft. The gain comes from:
   - 30B t32–t48 attention: 0–0.8% per cell, unmeasured;
   - budget re-targeting on 14B t32: 0.12–0.26%, likely enough to flip it to ≤1.05× (9 → 10/20 cells);
   - small change elsewhere.
   - No identified lever flips any ≤1.10× cell: 4B t32 needs −1.25%, llama8B t40 −1.15%, 30B t40 −1.21%.
4. **The numbers the user may be comparing against use excluded levers.** `_4` (12/20 ≤1.05×) and `_3`
   rely on qk rebalance, the enable grid, and attnfirst masks that run attention on INT7. Best(all)
   trails `_4` by 0.1–10.6% in 16/20 cells, worst on 30B. `_4`'s 30B t96 is 2.5% *below* the
   pure-allocation ceiling.

---

## 1. The gap in numbers

### 1a. Per-cell table

Sources:
- PPL/cost: `hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.json` (`rows[].submitted_ppl`,
  `best_ppl`, `winning_arm`, `delta_cost_pct_approx`, `candidates[]`).
- fp16: `hpca_results/llm/int/<m>.csv`.
- SC ceiling: `hpca_results/llm/frontend_awq/awq_mask_mp_table.csv`, row "Uniform, h=20%" at 256 nominal
  (4B 10.1186, llama8B 7.5282, 14B 8.6709, 30B 7.4826).
- Recomputed by `inv/final/cell_table.py`.
- "c17 share" = ln(c17/parent) / ln(best/parent).

| cell | bits | parent | c17 | best(all) | arm | best vs parent | c17 share | best ×fp16 | to ≤1.10× | to ≤1.05× | best over SC ceiling | cost vs parent |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4B t32 | 6 | 11.9886 | 11.3305 | **11.1891** | c6gfis | -6.67% | 82% | 1.1140 | -1.25% | -5.74% | +10.58% | +0.04% |
| 4B t40 | 6.32 | 11.1440 | 10.8766 | **10.7729** | c7gfisla | -3.33% | 72% | 1.0725 | pass | -2.10% | +6.47% | -0.10% |
| 4B t48 | 6.58 | 10.6491 | 10.5030 | **10.4665** | c6gfis | -1.71% | 80% | 1.0420 | pass | pass | +3.44% | +0.30% |
| 4B t64 | 7 | 10.4749 | 10.4085 | **10.3056** | c7gfisla | -1.62% | 39% | 1.0260 | pass | pass | +1.85% | -0.15% |
| 4B t96 | 7.58 | 10.2225 | 10.1993 | **10.1827** | c7gfisla | -0.39% | 58% | 1.0138 | pass | pass | +0.63% | +0.31% |
| llama8B t32 | 6 | 8.6758 | 8.3899 | **8.2795** | c6gfis | -4.57% | 72% | 1.1479 | -4.17% | -8.53% | +9.98% | -0.49% |
| llama8B t40 | 6.32 | 8.2093 | 8.0857 | **8.0266** | c6gfis_emp | -2.23% | 67% | 1.1128 | -1.15% | -5.64% | +6.62% | -0.30% |
| llama8B t48 | 6.58 | 7.9200 | 7.8151 | **7.7878** | c6gfis | -1.67% | 79% | 1.0797 | pass | -2.75% | +3.45% | -0.16% |
| llama8B t64 | 7 | 7.7109 | 7.6584 | **7.6435** | c6gfis | -0.87% | 78% | 1.0597 | pass | -0.91% | +1.53% | +0.03% |
| llama8B t96 | 7.58 | 7.6145 | 7.5987 | **7.5723** | c7gfisla | -0.55% | 37% | 1.0498 | pass | pass | +0.59% | +0.24% |
| 14B t32 | 6 | 9.3097 | 9.1672 | **9.0764** | c7gfisla | -2.51% | 61% | 1.0507 | pass | **-0.07%** | +4.68% | **-1.34%** |
| 14B t40 | 6.32 | 9.0598 | 8.9478 | **8.9029** | c6gfis | -1.73% | 71% | 1.0306 | pass | pass | +2.68% | -0.94% |
| 14B t48 | 6.58 | 8.9072 | 8.8142 | **8.7894** | c6gfis | -1.32% | 79% | 1.0175 | pass | pass | +1.37% | -0.76% |
| 14B t64 h20 | 7 | 8.7758 | 8.7110 | **8.6993** | c6h20gfis | -0.87% | 85% | 1.0071 | pass | pass | +0.33% | -0.66% |
| 14B t96 h20 | 7.58 | 8.6418 | 8.6441 | **8.6383** | c7h20gfisla | -0.04% | (c17 worse) | 1.0000 | pass | pass | -0.38% | -0.26% |
| 30B t32 | 6 | 9.4647 | 8.9599 (e32) | **8.5812** | c7gfisla | -9.33% | 56% | 1.1818 | -6.92% | -11.15% | +14.68% | -0.60% |
| 30B t40 | 6.32 | 8.5294 | 8.4123 | **8.0856** | round3_local | -5.20% | 26% | 1.1135 | -1.21% | -5.70% | +8.06% | -0.44% |
| 30B t48 | 6.58 | 8.1784 | 8.1159 | **7.8571** | c7gfisla | -3.93% | 19% | 1.0820 | pass | -2.96% | +5.00% | -1.02% |
| 30B t64 | 7 | 7.7146 | 7.6601 | **7.6281** | c7gfis | -1.12% | 63% | 1.0505 | pass | **-0.05%** | +1.94% | -0.64% |
| 30B t96 | 7.58 | 7.5985 | 7.5751 | **7.5336** | c7gfis | -0.85% | 36% | 1.0375 | pass | pass | +0.68% | -0.40% |

### 1b. Summary

- **Counts.**
  - ≤1.10× fp16: parent 13, c17 14, best(all) **15/20**.
  - ≤1.05×: 7, 8, **9/20**.
  - Excluded-lever archives: `_3` 17 / 10 and `_4` 19 / 12 (`inv/gap.md` §3).
- **By budget, mean best vs parent**: t32 −5.77, t40 −3.12, t48 −2.16, t64 −1.12, t96 −0.46%
  (`BEST_ALL…json` `summary.by_budget`).
- **Near misses.**
  - ≤1.05×: 14B t32 (needs −0.07%) and 30B t64 (−0.05%). llama8B t96 passes by only 0.02%.
  - ≤1.10×: 4B t32 (−1.25%), llama8B t40 (−1.15%), 30B t40 (−1.21%).
  - Far: llama8B t32 (−4.17%) and 30B t32 (−6.92%).
- **Round decomposition** (`inv/rounds.md` §3, per-budget t32/t40/t48/t64/t96):
  - c17: −3.91 / −1.63 / −1.13 / −0.69 / −0.18%.
  - The round-1 Fisher and round-2 joint λ add the rest.
  - Rounds 3–5 add −0.0015 pp; the only moved cell is 30B t40 at −0.030%, 0.36 noise-sd.
- **Test selection is part of the headline.**
  - Always-c17 −1.51%, always-r1 −2.09%, always-r2 −2.31%, min(r1, r2) −2.52%, best(all) −2.53%
    (`inv/gap.md` §3).
  - Per-cell choice between r1 and r2 on test is therefore worth about 0.2–0.4 pp.
- **Non-granularity content.**
  - On 30B t40/t48, 74%/81% of the log-gain comes from round 2 re-splitting per-row attention (c17 share
    26%/19% in the table).
  - Pure granularity is isolated in only 3 cells (`PAPER_METRICS_20260927.md:63-66`):
    - 4B t32 r17m→c17: −4.33% at +0.45% cost;
    - llama8B t32: −2.05%;
    - 4B t48: −1.63%.
  - On 14B/30B t48 there is only an error-space granularity control. Held-out linear error per-row
    same-calibration → per-group mn is 0.939→0.748 (14B) and 0.896→0.762 (30B)
    (Turbo `prc2/{14B,30B}_t48_a3_diag.json` `statistic_analysis.row_same_calib` vs `mn`).
  - No per-row Fisher control (r1/r2 recipe with `SC_PRC_ROWSHARED=1`) exists.
- **Provenance note.** Eight best(all) dense tables were built by pre-STE-fix calib6 code:
  7 `c6gfis` plus llama8B t40 `c6gfis_emp`. They predate the 09-24 13:20 bit-exact STE fix
  (PRC2_OVERNIGHT.md 13:20 entry). They are valid full-protocol results, but the code that made them no
  longer exists; calib6 was overwritten 09-24 13:13 (`inv/h6skeptic`). The provenance should say so.

---

## 2. What rounds 3–5 did and why each failed

### 2a. Naming

- No artifact is labelled "round 5". The Codex post-round-2 jobs are, in order:
  - `prc_local` 62032412 (round 3);
  - `prc_adjacent` 62104342 (round 4; 4B/llama8B t32);
  - 30B companion 62105189 (30B t32/t40). The docs file it under round 4 (`prc_adjacent_30b_20260927.json`
    `scope_extension`), so reading it as "round 5" is an inference.
- PRC2_OVERNIGHT also has an earlier Claude "round 3/3b/3c" (calib8 robust Fisher, per-block keys,
  re-linearization, attention ceiling; held-out only, no test). Below: R3-Claude, R3, R4, R5.
- No post-round-2 round touched 14B or any t48–t96 cell (`inv/rounds.md` §2).

### 2b. Ledger

| round | generator (code) | step (share of total SC cycles) | result | why it failed |
|---|---|---|---|---|
| R3-Claude (09-24/25) | calib8: 12-window raw MC Fisher, clipped/kraw, per-block keys, re-linearized; 3c: attention top relabelled 96/97→112/128, **linears Fisher re-solved at target×s** (`prc_offline_solve.py:213-233,268-277`) | large | held-out only. 12-window raw Fisher **+32.5% vs parent**, from a section-break spike in one window. Robust/per-block ≈ tie (+0.11…+0.45% vs c6gfis, z ≤1.2). att128: 4B **+0.0178 NLL (z +6.0)**, llama8B −0.0015 (z −0.49) | currency spikes; the 3c raise was non-selective (every top-rung row moved) and was tried only on dense t32 |
| R3 `prc_local` | `prc_local_proposals.py:306-316,364-407`: one additive offset on **every** threshold of a population; 4 fixed schemes × 2 directions; no loss signal | 0.91–2.00% | 19/20 worse on search, up to +17.6% (llama8B linears←qk, z +12.9). Tests: +0.212%, +0.342%, −0.030% | non-local and rung-skipping. Coincident zero thresholds sent about 25% of llama QK MACs 96→19 (ROUND3_POSTMORTEM:37-46). 4.4–25.5% of all SC MACs changed rung, up to 13 rungs; down_proj took −8.0% of its own budget vs −1…−1.8% for the other linears (`inv/h2mech/r3_rungs_*.json`). Even-order penalty +0.28…+1.81% per 2% move vs odd (directional) terms ≤0.42% (`inv/h2skeptic/r3pairs.json`) |
| R4 `prc_adjacent` dense | `prc_adjacent_proposals.py:378-392`: category round-robin ranked by (bucket usage, −moved, **cost-match** error, signature); no loss signal; 12 of 54k–91k pairs | **0.0048–0.0685%** (all 48 requested 0.25%; the 0.5% variants were all de-duplicated) | 39/48 search deltas negative. Confirmations: 4B +0.138% (z +0.48); llama8B −0.398% (z −1.98) → **test +0.077%** (8.285860 vs 8.279485, `prc_adjacent_20260927/llama8B_t32/full_test_result.json`) | expected effect ≤0.02% vs confirmation MDE80 0.47–1.14%. Selection = best-of-12 noise |
| R5 (30B companion) | same generator | 0.009–0.068% | confirmations −0.329% (z −0.68), −0.079% (z −0.27); no test | same |

### 2c. Why the step sizes failed

The data and code lenses agree on the mechanism:

- **Per-call re-dispatch cascade.**
  - Dispatch is min–max normalized per call, so a threshold edit changes the inputs of every downstream
    call.
  - Upstream buckets were byte-identical in 48/48 candidates, so the pairing is valid.
  - Unedited downstream buckets re-dispatch anyway. Their gross cycle change is a median
    4.9× / 14.9× / 3.4× / 4.3× the planned move (range 0.2–32×). Between 0.02% and 1.20% of all MACs
    change rung, and realized cost moves by up to 0.30% against a planned ~0.0005%
    (`inv/h2mech/cascade_*.log`).
  - Per-window noise σ_w tracks this realized drift: within-cell r(σ_w, realized total variation) =
    0.32/0.84/0.73/0.62; pooled r = 0.53, p < 0.001 (`inv/h2skeptic/realcorr.py`).
  - It does not track the planned move (r −0.24…+0.29). So `inv/protocol.md` §3's "σ_w is independent
    of move size" holds only for the *planned* size.
- **Power.**
  - Null best-of-12: 0/48 candidates fall below the null 5th percentile. The observed minima sit at
    null quantiles 0.85/0.30/0.32/0.54 (`inv/h2skeptic/nulls_r4.json`).
  - False-pass risk with 4 cells is about 24–28%.
  - Confirmations kept on average 16% of the search effect.
  - The most potent move ever observed was 2.53 %PPL per % of cycles, the 14B t64 h20 round-2
    attention cut (`inv/h2skeptic/potency.json`). Even at that potency, R4/R5 planned moves bound at
    ≤0.17–0.26%.
- **The shared negative offset is unexplained.** Most candidates "beat" the incumbent on the same 6
  windows. Its significance is borderline: combined sign-flip p = 0.051 (data lens); exchangeability
  p = 0.025, Stouffer z −2.48 (code lens). The incumbent was chosen on test, not on these windows, so
  selection cannot explain it.
- **The evaluator is not the problem.**
  - 16-window paired held-out NLL vs full test over 49 per-group pairs: r = 0.962, slope 0.94, sign
    46/49, 28/28 when |z| ≥ 1.5 (`inv/protocol.md` §5).
  - Round-2 held-out vs test over 20 cells: r = 0.973 (`inv/rounds.md` §5).
  - Resolution degrades below |effect| ≈ 0.5%: r = 0.57–0.68, rms 0.28–0.32%.
- **Round 3's steps were detectable. Its move type was the problem.**
  - An efficient per-family-λ re-solve with the same per-family cost change does 1.7–4.6× less
    surrogate damage (`inv/h3mech/r3eff_*.json`).
  - R3 therefore does **not** show that the family split is optimal. The 95% upper bound on a clean
    first-order family gain is about 1–1.8% (`inv/h2skeptic/r3pairs.json`).
- **The cost guard never bound.** Max |cost/incumbent − 1| was 0.54% against a 1% tolerance
  (`search_results.json`). The user's ~1% allowance went unused.

---

## 3. Verified root causes, ranked

Ranking: first by how much of the "not good enough" gap a cause explains, then by live recoverable
value. "Surviving" = neither lens refuted it with solid evidence. Where the lenses disagreed, I re-checked
the point myself; see "Re-check" in each item.

| # | root cause (restated after verification) | status | est. recoverable under constraints | lever |
|---|---|---|---|---|
| RC1 | Dense t32–t48 linear loss is per-token sensitivity the runtime statistic cannot see; the in-family threshold headroom is at or below the calibrator's own table-to-table noise (H3, restated) | supported on 4B/llama8B t32 (MC-Fisher surrogate); 14B/30B untested | ≤0.5% (4B t32), ≤0.25% (llama8B t32), ≤0.2–0.3% t48 and ≤0.1% t64 by κE scaling | none worth a round; at most a fit8/calibration-transfer check |
| RC2 | t96 sits at the SC ceiling; t64 is budget-limited with small deployable headroom (H4, restated) | t96 supported; t64 as originally stated refuted | t96 ~0.1–0.3%; t64 ≤0.3–0.5% (4B t64 via attention) | none for t96; 4B t64 as a Round A check |
| RC3 | 30B t32–t48: qk (and av q3) pinned at the inherited per-row ladder top 96/96/109; 0.03–0.05 nats still unexplained after round 2 (H1 leg (a), narrowed) | supported as a necessary condition; the gain is unmeasured | 30B t32/t40/t48 0–0.8% each; 4B t40/t64 0–0.4%; others ≈0. Mean ≈0.05–0.2 pp | Round A (per-bucket qk/av ladders to 128 in a joint λ), gated by a diagnostic |
| RC4 | Rounds 3–5 used blind generators at wrong step sizes; each edit also triggers an uncontrolled re-dispatch cascade (H2, restated) | supported | 0 directly; gates every other lever | protocol redesign (§5c) |
| RC5 | T1's lever reaches only the dispatched linears, 51–77% of SC cycles at t32; attention stays per-row and the protected slice has one fixed length | verified in code | n/a (structural) | explains why granularity gains cannot exceed the linear share |
| RC6 | Deployed tables under-spend the parent's cost by 0.6–1.34% on 14B/30B, from trajectory shift between calibration and deployment (H6 cost leg) | supported; variance leg refuted | 14B t32 0.12–0.26% (flips ≤1.05×); 14B t40 ~0.1%; 30B t32/t48 ~0.1–0.16%; mean 0.03–0.05 pp | Round B: `--target-scale`, set from held-out-train child cost |
| RC7 | Expectation set by `_3`/`_4`, whose wins come from excluded levers | verified | 0 | framing only |
| RC8 | The protected-slice length is frozen from the per-row era, but it is not a root cause (H5) | descriptive facts verified; causal claim refuted | ≤0.1–0.15%/cell; llama8B t40 maybe ~0.35% (1.5 SE) | two-sided add-on arms only |

### RC1. Dense linears: the per-token sensitivity limit

**Mechanism, verified in code.**
- The runtime rule is the per-(row, 128-chunk) absmax of the unsmoothed residual `x_dispatch`, min–max
  normalized over the call (`model/sc_common.py:505-515`), applied via `bucketize` (:530).
- Tables are keyed by (op, t, layer_bucket) = 28 staircases on one 17-rung ladder
  (`mp/config.py:795-811`).
- One global Fisher λ is solved at the parent's linear calibration cost (`calib6:533-537`).
- There is no linear escape.

**Evidence** (c8w6 dumps, 6 calibration + 2 hold windows; `inv/attrib.md` §3, `inv/h3skeptic/`):
- Deployed-rule hold error, relative to the parent: 0.597 (4B) and 0.645 (llama8B). The Fisher oracle
  reaches 0.215 / 0.222.
- 86% (4B, range 0.81–0.92 over 28 rotations) and 88% (llama8B median) of that gap is per-token
  sensitivity.
- Per-token sensitivity is uncorrelated with activation magnitude: Spearman +0.03/+0.04 (4B) and
  −0.02 (llama8B). It is not concentrated at early positions (rows with rpos < 16 hold 1.7–2.9% of the
  excess), yet the top 1% of rows carry 33.6% / 37.9% of the excess.
- The only within-key correlate above 0.3 is the static block index (4B −0.31). Per-block keys were
  already measured and tied: kraw:blk +0.0011 NLL, z 0.25 (`prc2/4B_t32_c8o_select_heldout_nll.json`).

**Deployable bound.** The 6→8-window learning curve (`inv/h3skeptic/learn_*.json`) compares fit8
in-sample against fit6 out-of-sample on the hold pairs:
- 4B: 0.598 vs 0.628, a gap of 0.030 of parent excess.
- llama8B: 0.643 vs 0.662, a gap of 0.019.
- At κ·E_par = 0.168 / 0.130 nats this is **≤0.50% / ≤0.25% PPL**. Re-checked here from
  `log_learn_*.txt`.
- Cross-label scoring keeps only about 40–60% of the oracle's advantage (`inv/attrib_cross_llama8B.json`),
  so even the "if observable" figure is about 2–4%, not 5.4–6.6%.

**Re-check (the lenses disagreed on noise).** The code lens called the c8w6o re-solve's +0.0056 NLL
(4B, z +2.1) "calibration redraw noise". I verified that `c7_gfis`, `c8w6_gfis` and `c8w6o_fis` are
**identical on 28/28 keys**, and that each differs from `c6_gfis` on 28/28 keys (max |Δth| 0.39 on 4B,
0.38 on llama8B; Turbo `prc2/{4B,llama8B}_t32_*_table.json`, script inline in this session).

So the +0.56–0.72% is the pre- vs post-STE-fix code difference, not a redraw. It is still informative:
the surrogate scores the two tables nearly equal (hold 0.6002 vs 0.5972), yet they differ by +0.62%
(pooled over 40 windows, z ≈ 3) in measured NLL. **The Fisher surrogate cannot rank linear tables that
differ by ≤~0.5%.** Any linear lever of that size must be measured, not predicted.

**Scope limits.**
- The bucket-crossing oracle is 0.029/0.022 on dense t32 but 0.092/0.080 on 30B t32/t40 (`*_c7_diag.json`).
  The global λ already harvests that; it is not remaining headroom, but "≤0.03" does not generalize.
- The protected slice sits outside every dump.
- 14B/30B dumps do not exist.

### RC2. High budgets: the SC ceiling at t96, limited headroom at t64

**Ceiling numbers, verified.**
- Uniform-256 with h20 is 1.0074 / 1.0437 / 1.0038 / 1.0305× fp16.
- Best(all) over the ceiling:
  - t96: +0.63 / +0.59 / −0.38 / +0.68%;
  - t64: +1.85 / +1.53 / +0.33 / +1.94%.
- The mask sets are identical (65/58/72/87 (op, block) entries); only the width differs, INT8 in the
  ceiling vs INT7 in the MP cells (`inv/h4skeptic/maskcmp.py`, `inv/h4mechsk/maskcmp.py`).
- With the INT7 width correction (int_ablation README §4) the t96 gaps become +0.48 / +0.35 / −0.25 / +0.67%.

**The ceiling is not a strict bound.** MP beats uniform-256 in two places:
- 14B at t96: 8.6383 < 8.6709;
- 4B under SmoothQuant with the identical mask: 10.2084 < 10.2187.

The mechanism is the protected-channel split in the MP path (`sc_common.py:804-866`), which the uniform
path lacks. So realistic t96 headroom is about 0.3–1.0%, of which ≈0.1–0.3% is reachable.

**At t64, H4 as first stated is wrong.**
- 26–72% of the fp16 gap sits *above* the ceiling.
- Attention is not at 128: mean L is 104.5/102.6 (4B), 87.9/87.5 (14B) and 117.3/117.0 (30B); only
  llama8B is at 128.
- Post-round-1 linear excess is 0.010–0.017 nats (`inv/attrib_decomp.json`).
- Deployable gain is ≤0.3–0.5%. The only identified lever is 4B t64 attention: round 2 already bought
  −0.57% there, and 95% of qk sits on the 105 top.
- At t64–t96, 1% PPL costs 32–79% more cycles (e = 0.017–0.036).

### RC3. 30B t32–t48 attention, and the one live lever

**Facts reproduced by both lenses.**
- The joint λ solves attention only on the parent's ladder: `lad = get_levels(...)`, `lv_meas = lad ∪
  {esc}` (`mp_per_row_chunk_calib7.py:422-427`, verified). The DP runs on `A = lad`, escape rows are
  fixed cost, target = `par_lin + par_att` (:777-796, verified).
- Attention sampling is `--att-rows-per-call 128` (:183, verified) out of B·H·N = 65,536 rows. That is
  about 9 blocks × 6 windows × 128 = 6,912 rows over 2,048 bins per bucket, vs 64 of 2,048 rows for
  linears.
- `--dump` writes only linear `store` arrays (:896-906, verified).
- No incumbent sets `bucket_stoc_len_levels`. The runtime supports per-bucket ladders
  (`mp/config.py:469-490, 841-878`), and escaped rows fold into a 128 rung.
- Attention RNG is unaffected: `_attn_grid_for` returns None unless `SC_RNG_GRID_*` is set
  (`sc_common.py:166-196`; `run_prc_ppl.sbatch:139-147` leaves them empty).

**Top-rung binding in the round-2 winners** (`inv/synth/v_topbind.py`, re-run here):

| cell | top | qk q0–q3, share on top | av q0–q3, share on top |
|---|---|---|---|
| 30B t32 | 96 | 90/74/73/71% | 0–1% |
| 30B t40 | 96 | 81/66/89/99% | 1/1/7/**85%** |
| 30B t48 | 109 | 84/81/94/92% | 14/1/7/**90%** |
| 4B t40 | 96 | 89–96% | 40/97/94/99% |
| 4B t64 | 105 | 91–97% | 65/100/96/100% |

**Re-check (the lenses disagreed on qk-only vs qk+av).** Per the table above, av q3 is also bound on
30B t40/t48, and av q1–q3 on 4B. So the right design is *separate per-bucket qk and av ladders*, with a
128 rung added only where the bucket sits at ≥85% on its top. That means:
- 30B: qk q0–q3 plus av q3;
- 4B: qk plus av q1–q3.

A qk-only change would leave the bound av q3 buckets untouched.

**What does not survive.**
- **Leg (b), "Fisher under-prices attention cuts", is refuted.**
  - The residual (held-out minus predicted) is +0.27 pp in raise cells and +0.23 pp in cut cells;
    permutation p = 0.57.
  - The pooled fit is measured = 0.166 pp + 0.82 × predicted, r = 0.92 (`inv/h1skeptic/resid_test.json`,
    `inv/h1mech/resid.json`).
  - The cut cells lost because the solve predicted almost no gain there (8/11 had |pred| ≤ 0.18%).
    Where it predicted a loss of ≥ +0.4% (llama8B t40, 14B t64-h20, 30B t96), the loss happened
    (`inv/synth/v_joint_pred.py`).
- **The dense "direction rule" is half confounded.** For dense cells, r1 = `c6gfis` (pre-fix) and
  r2 = `c7gfisla`. I re-derived the clean joint term (`c7gfisla − c7gfis`, same run, same 16 windows;
  `inv/h1skeptic/decomp_r2.json`):
  - 4B t40: −0.61% (z −1.16), not the −1.34% total;
  - 4B t64: −0.51% (z −1.52), not −0.95%;
  - 4B t32, which cut attention by 13%: a clean **−0.82% (z −1.84) win**.
  - The recalibration term correlates −0.859 with direction.
  - On the clean term: raises win 8/9 on held-out, cuts win 2/11. The rule survives as an association,
    but it is mediated by the predicted size of the gain, not by direction.
- **The no-cut variant is dropped.** It would have blocked 30B t32's round-2 win, which came from
  qk 65.5→88.1 with av 59.6→44.5.
- **The pinned dense cells are dropped** (llama8B t32/t40, 14B t32/t40/t64-h20, 30B t64). There the
  currency itself cut attention (all 6 were cut; `inv/synth/v_r2dir.py`), and the one direct test
  (R3-Claude 3c on llama8B t32) was neutral.

**Size, still unmeasured.**
- Raising qk top-rung rows to 128 costs +6.94 / +6.33 / +3.29% of total SC cycles at 30B t32/t40/t48,
  and qk+av costs +4.7+4.3% (4B t40) and +2.2+2.1% (4B t64) (`inv/h1skeptic/topcost.py`, re-run here).
- The linear dump shows error ∝ ~1/L² (squared error 1.73× at 96 and 3.72× at 64, relative to 128).
  So the 96→128 step buys ≈0.36–0.39× the per-cycle error reduction of the 64→96 steps that round 2
  bought.
- Fisher over-predicted the 30B raises: realized/predicted = 0.64 / 0.95 / 0.61.
- Combined estimate: **0–0.8% per 30B cell, 0–0.4% on 4B t40/t64, ≈0.05–0.2 pp on the 20-cell mean.**
  This is not enough to lift 30B t40 to ≤1.10× (needs −1.21%) on its own.
- The 30B residual may instead be MoE routing flips, which a diagonal STE Fisher cannot see
  (`inv/attrib.md` §9).

**Framing.** This is per-ROW attention reallocation, not granularity. Report it under best(all), never
as a T1 effect.

### RC4. Protocol failure of rounds 3–5

See §2. What is established:
- R4/R5 were undetectable by construction: expected effect O(0.01%), MDE O(0.5–1.1%), and the
  re-dispatch cascade was larger than the move.
- R3 was detectable but its move type was lossy (the even-order penalty dominates).
- Neither generator used Fisher, error or loss information. A loss-model generator already existed:
  linear Fisher prediction vs test has r = 0.76, sign 22/24, and the joint prediction is sign-reliable
  when |pred| ≥ 0.4% (`inv/protocol.md` §5).

**Correction to the diagnoser's lever.** "Pre-screen multiplier perturbations on Fisher-predicted ΔNLL"
cannot work as written. calib7 solves one Lagrangian λ with an exact per-key staircase DP (:614-653,
:796-803), so any iso-cost price-multiplier deviation at a fixed ladder is predicted ΔNLL ≥ 0 *on the
calibration windows*.

Pre-screening is meaningful only:
- on the hold-window prediction, or
- when the feasible set changes (new rungs, new budget).

### RC5. Structural reach of T1

- Per-(row, chunk) dispatch exists only in `SCLinear` (`sc_common.py:904`).
- Attention runs `_sc_matmul(granularity="per_row")` with no `chunk_d` and no `rung_table`
  (sc_common 1287-1288):
  - qk contracts over head_dim = 128, one chunk, so per-row already is per-group for qk;
  - av is quantized per whole row, so per-chunk av would change quantization, which is an execution
    change.
- The protected slice runs at one table-wide `protected_channels.stoc_len` (`mp/config.py:515-518`,
  `sc_common.py:845-866`).
- Shares of SC cycles at t32 (`inv/algo.md` §3):

  | cell | per-group linears | attention | protected |
  |---|---:|---:|---:|
  | 4B t32 | 60.3% | 32.8% | 7.0% |
  | llama8B t32 | 72.3% | 21.4% | 6.2% |
  | 14B t32 | 77.3% | 14.9% | 7.8% |
  | 30B t32 | 51.3% | 43.2% | 5.5% |

- Attention MACs are counted **dense**: `mac = K·M` per row (`calib7` rec, :457-461). Causal hardware
  would halve them. So attention is priced about 2× its causal cost in every budget, both parent and
  child. That is an accounting convention, not a lever (see §5d).

### RC6. Deployment under-spend (H6 cost leg; the variance leg is refuted)

**The shortfall.**
- The bisect hits the parent's cost on the 6 calibration pairs (calib6 L476/534, :497-512; calib7
  :794-803).
- Deployed on the child's own trajectory, the tables under-spend:
  - 14B t32 −1.34%; 30B t48 −1.02%; 14B t40 −0.94%; 14B t48 −0.76%; 14B t64-h20 −0.66%; 30B t64 −0.64%;
    30B t32 −0.60% (§1a);
  - 4B t48/t96 over-spend by +0.30/+0.31%, and llama8B t96 by +0.24%.
- The shortfall is a stable per-cell bias, not noise. On 14B t32: c17 −1.12%, c6gfis −1.28%,
  c7gfisla −1.34%.
- It is visible **before test**: held-out-train child cost matches test cost within a mean |Δ| of
  0.25 pp (14B t32 0.9868 vs 0.9872; `inv/h6mech/heldout_vs_test_cost.json`).
- In linear-only arms the shortfall sits in the dispatched linears. In 14B t32's joint solve, 1.02 of
  the 1.34 pp is attention (`inv/h6mech/costdecomp.json`).

**Value.**
- Elasticity for 14B t32 is 0.19 (log-log) to 0.21 (c17 9.1672 @33.19 vs s80 9.4615 @28.19;
  `inv/rounds.md` §9). The marginal is about 0.6× the chord.
- So 14B t32 recovers ≈0.12–0.26%, against the 0.067% it needs.
- 30B t64 needs 0.049% and would get ≈0.03–0.06%: a coin flip, below the 0.09% reshuffle sd.

**Variance leg, refuted by both lenses.**
- Leave-one-window-out iso-cost spread is 0.0020 (4B) and 0.0049 (llama8B) of parent excess, i.e.
  0.03–0.07% PPL (`inv/h6skeptic/lowo_*`).
- Tables made by the same code are bit-identical (verified above).
- The 9 STE-fix-only pairs give Σz² = 9.38 on 9 df (p = 0.40).
- The 12-window spike came from the data (a section break), not from MC labels.
- Multi-label or clipped Fisher is therefore not a lever. Robust 12-window pricing already tied or lost.

**Re-check (the lenses disagreed on the re-target source).** Bisecting to *full-test* trace cost leaks
test data into calibration. Use the held-out train windows' child-trajectory cost with one fixed-point
pass, target × parent/child. It predicts test cost to about 0.25 pp and needs no test data.

**Blocker.** calib6/7 have no budget-scale flag (only calib3 has `--budget-scale`, :300, :560; verified).
The 8 pre-fix `c6gfis`/`c6gfis_emp` tables cannot be regenerated exactly, so re-targeting them means
re-solving with calib7 code.

### RC7. The expectation is set by excluded levers

- `_4` has 12/20 cells ≤1.05× and 19/20 ≤1.10×. It wins via attnfirst (attention entirely on INT7),
  `prcqk` (QK rebalance) and `grid` (`hpca_results/llm/ppl/mp_best_after_hpca_4/manifest.json`
  `winner`; `inv/gap.md` Table C).
- Best(all) is behind `_4` by 0.1–10.6% in 16/20 cells, with 30B at +3.3…+10.6%. It is *ahead* on 14B
  t32/t40/t48/t64 (−0.78 / −1.28 / −0.76 / −0.08%).
- `_4` 30B t96 = 7.2922 (via `a3:prcqk`, not attnfirst) is 2.5% below the pure-allocation ceiling of
  7.4826. No allocation can match it.
- vs INT: the SC ceiling itself is already +3.07% (llama8B) / +2.72% (30B) over INT6-AWQ. The paper has
  to argue SC on energy (`inv/gap.md` §7).

### RC8. Protected-slice length (H5): real, frozen, not a root cause

**Descriptive facts, verified by both lenses.**
- One global psl, reproduced per cell (`inv/h5mech/psl_best.json`), t32…t96:
  - 4B 84/68/84/112/128;
  - llama8B 68/95/95/112/128;
  - 14B 80/95/112/112/128;
  - 30B 80/68/112/95/128.
- It takes 3.14–7.80% of SC cycles on 2.34–3.23% of MACs.
- It is excluded from every per-group DP: calib2 :425-434, calib7 :360-366.
- It was never varied in the per-group era (`inv/h5mech/psl_rounds.json`).

**Why it is not a root cause.**
- psl *was* tuned per-row. The v20 searches offered iso-cost `pc_len` moves in every sweep and accepted
  `pc_len_68` at 4B t40 and 30B t40 (`mp_best/manifest.json`).
- The landscape around the incumbents is flat or mixed. Two confirm-stage candidates were vetoed
  (`inv/h5/pcl_center.json`, `inv/h5mech/pcl_center_mine.json`).
- The V17 4B t32 curve is flat below 80.
- Per-group lowers the marginal value of linear cycles: 16/16 chords at ratio 0.55–0.99. First order,
  psl should therefore go **up** about 14–22%, not down to free cycles.
- Estimated value: ≤0.1–0.15% per cell. The exception is llama8B t40, where the per-row `pc_len_72`
  signal (−0.44%, vetoed at 1.5 SE) suggests ≈0.35%.

---

## 4. Refuted or unsupported explanations

| claim | verdict | evidence |
|---|---|---|
| Fisher systematically under-prices attention *cuts* (H1 leg b) | refuted | residual +0.27 (raise) vs +0.23 pp (cut), p = 0.57; generic optimism, slope 0.82, r 0.92 (`inv/h1skeptic/resid_test.json`) |
| Dense round-2 wins prove attention under-funding | half confounded | clean joint term 4B t40 −0.61%, t64 −0.51% (not −1.34/−0.95%); recalibration term r = −0.859 with direction (`inv/h1skeptic/decomp_r2.json`) |
| "Never cut attention" constraint | refuted | 30B t32's −1.55% came from av −25%, qk +35% |
| Selective raise to 128 would help 14B t40/t64-h20 and llama8B t32/t40 | unsupported | currency cut attention in all 6 pinned cells; 3c on llama8B t32 neutral (−0.0015, z −0.49) |
| Round 3c cut linears by a uniform factor | incorrect description (mappers `algo.md` §0 and the H1 statement) | `prc_offline_solve.py:213-233` re-solves linears with the global Fisher λ at `target*lin_scale`; the *attention* raise was the non-selective part (:268-277) |
| Rounds 3–5 show the operator-family split is already optimal | refuted | R3's move type is 1.7–4.6× lossier than an efficient per-family move; R4/R5 were underpowered; clean family gain ≤~1–1.8% not excluded (95% UB) |
| σ_w is independent of move size (`inv/protocol.md` §3) | refuted as stated | it scales with realized re-dispatch (pooled r = 0.53), not the planned move |
| Calibration solves are noisy (±0.7%); multi-label / more-window Fisher will help (H6 variance leg; diagnoser's "Round C lower-variance Fisher") | refuted | LOWO iso-cost spread 0.03–0.07% PPL; same-code tables identical 28/28 keys; the ±0.7% is the pre/post-STE-fix code change (Σz² = 9.38/9 df for fix-only pairs); 12-window robust Fisher tied or lost |
| "Crossing buckets is worth ≤0.03" in general | dense-t32 oracle only | 0.092/0.080 on 30B t32/t40, 0.034–0.039 on 14B, already harvested by the global λ |
| Uniform-256 h20 is a hard ceiling; 0% recoverable at t64/t96 (H4 as stated) | refuted | MP < ceiling on 14B (AWQ) and 4B (SQ, same mask); t64 has 1.30–1.93% above the INT7-adjusted ceiling |
| psl should be cut to free cycles; psl is a first-order lever (H5) | refuted | per-row v20 tuned it; flat landscape; per-group ⇒ psl should move *up*; ≤0.1–0.15% |
| Per-call min–max normalization is the largest remaining linear misallocation (H7) | refuted as a loss claim | the abs-threshold arm cuts **raw** held-out error 10.6/13.4% (14B/30B t48, cost-adjusted), but a table-only per-block free arm cuts raw error *more* (4B 0.539 vs 0.571; llama8B 0.762 vs 0.785) while Fisher error gets *worse* (0.674 vs 0.658; 0.753 vs 0.737; `inv/h7mechsk/`, `inv/h7skeptic/pb*`); the raw currency is 77–89% last-quartile |
| The dispatch statistic is "free" / equals the kernel's group scale | false under AWQ | stat on unsmoothed `x_dispatch`; kernel quantizes after `apply_smoothing` (`sc/matmul.py:192-193`); smoothed stat is worse (0.631 vs 0.600 on 4B t32) |
| Fisher pre-screen of iso-cost multiplier families | invalid | Lagrangian optimality ⇒ predicted ΔNLL ≥ 0 on calibration windows at a fixed ladder |
| R4/R5 search "wins" (−0.45…−1.14%) | noise | 0/48 below the null 5th percentile; confirmations kept 16%; test +0.077% |
| The "rest" in dense cells is attention | unsupported | 4B t32 has lower attention L than t40 yet a smaller rest (0.008 vs 0.033 nats); 3c back-out gives ≤0.022–0.024 nats gross for 96→128 at 7–7.5% of cycles |

---

## 5. Unexplored pure-allocation degrees of freedom

### 5a. Inventory

Only table-level changes; no RNG, QK operand, mask, AWQ, chunking or new metadata type. The class
labels follow `inv/algo.md` §7.

| DOF | status | expected value (this investigation) | GPU needed |
|---|---|---|---|
| Per-bucket attention ladders (`bucket_stoc_len_levels`), separate qk/av, rung 128 (+112 fill) where ≥85% sits on top; joint λ | never tried; only the non-selective relabel (3c, dense t32) | 30B t32/t40/t48 0–0.8%; 4B t40/t64 0–0.4% | yes: one new calib7 capture per cell (needs attention error at the new rungs) |
| Attention sampling density (`--att-rows-per-call` 128 → 512) and an attention dump for offline solves | untested | variance reduction on the attention staircase (~3.4 rows/bin today); size unknown | folds into the same capture |
| Budget exactness: `--target-scale` from held-out child cost | untested | 14B t32 0.12–0.26% (≤1.05× flip); 14B t40 ~0.1%; 30B t32/t48 ~0.1–0.16% | re-solve + full test per cell |
| Hold-prediction gate for choosing r1 vs r2 per cell | not used | would have separated round 2's wins from losses (post hoc); also removes test-selection from the headline | none (CPU on existing diags) |
| Protected-slice length psl, two-sided (parent ± one step, and 128) | frozen since per-row v20 | ≤0.1–0.15%/cell; llama8B t40 ≈0.35% at 1.5 SE | re-solve at ±Δ budget + full test |
| Efficient per-family-λ transfers (±5/±10% of linear cycles, cost-matched, CPU-solved from dumps) | never done in true NLL (R3 used lossy offsets) | ≤~0.2% in the surrogate (broad optimum); true-NLL gain not excluded up to ~1% | eval only (held-out) |
| Selection among surrogate-equivalent tables by measured NLL (LOWO / seed variants) | untested | unknown: same-code between-table τ ≈ 0 in 9 fix-only pairs, but the proxy was blind to a real 0.62% difference on 4B t32 | eval only |
| Per-block linear keys inside the global Fisher λ on 14B/30B (the allowed analogue of H7) | tied on 4B t32; untested on 14B/30B | ≈0–0.3% | dump + eval |
| Linear ladder floor <8 | unexplored | ≤~0.3% bound; the deployed rule puts ≤3.8% of MACs at L=8, so it barely binds | not worth it |
| Round-2 joint λ on cells it never improved (14B t40/t48, llama8B) | done | no | — |
| Grey (user decision): per-op psl (code change), escape k, attention statistic, inverted linear direction, absolute thresholds (H7) | grey | H7 ≲0.3% at best, sign uncertain (§4) | — |

### 5b. Revised mapping to ≤3 rounds

Input for the planner, not a launch plan. It replaces the diagnoser's draft.

- **Round A: attention ladder extension (RC3). Gated.**
  - Step 0, diagnostic (§6 Q1), on 30B t40 and 30B t32. Kill the round unless it passes.
  - Then one calib7 capture per cell on 30B t32/t40/t48, plus 4B t64 as the dense check (4B t40
    optional). The capture adds:
    - per-bucket qk/av ladders `{…, 96, 112, 128}` (30B t48: `{…, 109, 128}`) only on buckets with
      ≥85% on top;
    - `--att-rows-per-call 512`;
    - an attention dump.
  - Solve the joint λ **unconstrained**, at the incumbent's total cost, with escape μ+2τ semantics
    unchanged.
  - Gate: hold prediction ≤ −0.5% after a 0.6× realization discount; then 16-window paired z < −1.5
    and ΔPPL ≤ −0.3%; then full test.
  - No no-cut variant, no κ_att multiplier.
- **Round B: budget exactness plus a two-sided psl add-on (RC6 + RC8).**
  - Add `--target-scale` to calib6/7.
  - Re-solve 14B t32 (c7gfisla recipe, target × 1.0134 from held-out-train child cost), 14B t40, and
    the 30B Round-A solves.
  - psl ± one step on llama8B t40 and 14B t48.
  - Everything disclosed as a budget correction.
- **Round C: conditional, measurement-first.** Run the cheap discriminating evals from §6 (Q4 LOWO
  table spread, Q5 fit8, Q6 efficient family transfers) on 4B/llama8B t32.
  - Only an arm that reaches z < −2 on 32 fresh windows becomes a full test.
  - If none does, spend Round C's GPU budget on the per-row Fisher control (§6 Q10). It adds no PPL but
    is what makes the T1 claim honest.
- **Do not repeat:**
  - family offsets, adjacent micro-swaps;
  - raw 12-window Fisher, multi-label/clipped Fisher;
  - per-block keys on dense cells, re-linearization;
  - a non-selective attention-top relabel;
  - the σ-currency global λ;
  - a Fisher pre-screen of multiplier families at a fixed ladder;
  - any t96 round.

### 5c. Protocol every round must follow (from RC4)

1. **Loss-model-generated candidates** (joint-λ re-solves with an enlarged feasible set or a changed
   budget) that move ≥3–10% of cycles at ≤1 rung per group.
2. **Record and control the realized cascade**: the per-bucket rung total variation on the candidate's
   own trajectory, and the child cost on held-out train windows.
3. **Search, confirm, test.**
   - Search: ≤4–6 candidates plus the re-evaluated incumbent on 16 shared paired windows.
   - Confirm: 32 fresh windows at z < −2. MDE80 ≈ 0.5–0.85% at σ_w 0.012–0.024 for large moves.
   - Full test for passers. Under best(all), the family best may also be tested, but label it.
4. **Pre-register** the predicted effect and the MDE. Refuse any candidate whose |pred| < MDE.
5. **Scope and budget**: t32–t48 only, ≤4 GPUs.
   - Per-window cost is ~28 s (4B), 24 s (llama8B), 50 s (14B), 90–95 s (30B).
   - A full test is 0.8–1.1 / 0.9–1.3 / 2.0–2.2 / 2.8–4.2 GPU-h (`inv/protocol.md` §8).

### 5d. User decisions needed

1. **Raising qk/av stream lengths above the inherited ladder top.** It is table-only, the runtime
   supports it, and RNG is unchanged. Round 2 already changed qk lengths via thresholds, and the rung
   set is listed as allocation. Please confirm this counts as "same QK behavior".
2. **Deliberately spending up to +1% cycles.** At t32, +1% of cycles is worth ≈0.1–0.3% PPL (marginal
   e ≈ 0.6× chord). It flips no ≤1.10× cell; 14B t32 flips ≤1.05× at 0% extra anyway. Recommend
   re-targeting to exactly parent cost, with disclosure.
3. **Absolute thresholds (H7).** Recommend not asking: the loss-currency evidence is negative.
4. **The dense attention cost model** (`mac = K·M`) prices attention ≈2× its causal cost for parent and
   child alike. Changing it is an accounting change, not allocation. Flag it for the paper's cost
   statement only.
5. **"Round 5" = array 62105189** is inferred; please confirm.

---

## 6. Open questions that need a GPU

All respect the 4-GPU cap and keep mask, AWQ, RNG, QK operands and escape semantics.

| # | question | test (cells, arms) | cost | decision rule |
|---|---|---|---|---|
| Q1 | Can a 128 attention rung pay at iso-cost on 30B? (gates Round A) | 30B t40 (incumbent `prc_local…/candidate07`) and 30B t32 (`c7gfisla`); 16 existing confirmation windows. (A) per-bucket qk + av-q3 top → 128 via `bucket_stoc_len_levels`, thresholds unchanged (+6.3% / +6.9% cycles). (B, needs `--target-scale`) linears re-solved with the same λ at the same extra cycles. Optionally 4B t64 | ~30–60 GPU-min per 30B variant | Kill Round A if gross ΔNLL(A) > −0.01, or if A is not better than B by ≥1.5 paired SE (~0.004–0.005 nats) |
| Q2 | True MP-at-max ceiling at t96 | incumbent t96 wrapper with every rung and psl forced to 128 (single-rung ladder), 4B and llama8B (or 30B) | ~1–4 GPU-h each, full test | ≥0.4% below uniform-256 ⇒ t96 headroom ≈1% (revisit); within ±0.2% ⇒ no t96 round |
| Q3 | Does re-targeting to parent cost flip 14B t32? | 14B t32 c7gfisla, target × 1.0134 (held-out-train child cost), same seed and windows; plus a full test of the existing `30B_t48_c17_s80` table for the 30B elasticity | ~1–2 h re-solve + 1.7 h test; 30B ~4 h | cost within ±0.3% of 33.56 and PPL ≤ 9.0702 ⇒ confirmed; ΔPPL < 0.07% at matched cost ⇒ refuted |
| Q4 | Is there proxy-invisible table variance worth harvesting? | 3 LOWO 4B t32 tables (5–16% of MACs differ, proxy-equal within 0.002) on 24 held-out windows | ~30 min | sd ≥ 0.3% ⇒ select among solves by measured NLL (with winner's-curse correction); < 0.1% ⇒ drop |
| Q5 | Deployable linear calibration bound in true NLL | fit8 4B t32 table (offline, `prc_offline_solve.py`) vs `c6gfis`, 24 held-out windows | ~30–40 min | no ≥0.3% gain at z < −1.5 ⇒ linear-threshold calibration dead on dense cells |
| Q6 | Is the operator-family split optimal in true NLL? | efficient per-family-λ proj↔MLP transfers ±5/±10% of linear cycles (CPU-solved from c8w6 dumps), 4B/llama8B t32, 24 held-out windows | ~4 tables × 25 min per cell | none at z < −1.5 ⇒ split optimal; otherwise a legitimate Round-C lever |
| Q7 | Do the H3 bounds hold on 14B/30B? | calib8-style dump for 14B t32 (full mode) and 30B t32 (block mode), with attention and raw absmax; CPU `rotate/learn/attrib_headroom2` | 1–2 h each, no eval | fit6→fit8 bound > 0.06 of parent excess on 30B ⇒ ~1% linear headroom there |
| Q8 | psl direction under per-group | llama8B t40 (psl 95→72 and →128), 14B t48 (112→84 and →128), linear λ re-solved at ∓Δ | 4 full tests | all arms within ±0.2% ⇒ drop psl |
| Q9 | Is the 30B residual attention or MoE routing? | 30B t40: attention forced to 128 with incumbent linears, and linears forced to 128 with incumbent attention, on 16 windows | ~1 GPU-h | splits the 0.03–0.05 nats between attention and interaction/routing |
| Q10 | How much of best(all) is granularity? (T1 framing) | r1/r2 recipe with `SC_PRC_ROWSHARED=1` on 4B t32 and 30B t40 | calib + full test per cell | needed before crediting the Fisher/joint gains to per-group |
| Q11 | (only if the user wants H7) abs vs per-block-free vs deployed in **Fisher** currency on 14B/30B t48 | calib7-style pass storing raw absmax + Fisher | 1–2 h, no eval | abs must beat per-block-free by ≥0.02 of parent excess in Fisher currency before any rule change |

---

## Appendix A. Doc vs artifact discrepancies (verified)

1. `scmp_llm/CLAUDE.md:26` STATUS says "No round-4 PPL results yet". All 4 round-4/5 tasks finished
   09-27 15:20–17:36. llama8B t32 test: 8.285860 vs 8.279485 (+0.077%). PRC_ADJACENT_20260927.md has
   no results section.
2. `SCMP/CLAUDE.md` calls `prc2/SUMMARY.md` "source of truth". SUMMARY is stale: pre-finish finals
   (4B t40 10.8751, 14B t32 9.1041, llama8B t96 7.5855) and h10 14B t64/t96 rows. The current numbers
   are `BEST_ALL_VS_SUBMITTED_20260927.*`. `SCMP/CLAUDE.md` also still says the 14B t64/t96 parents are
   h10.
3. PRC_ADJACENT_20260927.md describes 0.25%/0.5% transfer ceilings. All 48 candidates requested 0.25%
   (the 0.5% ones were de-duplicated); realized 0.0048–0.0685%.
4. `sc_common.py:480-483` docstring: the statistic is "FREE … exactly the group quantization scale".
   False under AWQ; the dispatch absmax is on the unsmoothed `x_dispatch`.
5. 14B t64-h20 wrapper: `stoc_len_levels [89,72,66,63,61,58]`, **no `escape_gate_k`** (verified). The
   paper's μ+2τ statement does not cover this cell. `SEARCH_SPACE_MAP.md:8` ("escape k=2.0 … in every
   parent") is false for it and for the t96 cells, where 128 is an ordinary rung.
6. 9/20 incumbents re-solved attention thresholds (7 `c7gfisla`, 1 `c7h20gfisla`, 1 `round3_local`),
   although docs say attention uses the parent's thresholds.
7. The "±0.7% re-run variance" (c7gfis vs c6gfis) is the pre/post-STE-fix code change, not re-run
   noise. Same-code tables are identical 28/28 keys (verified here).
8. prc2 `README.md:131-132` caveat 8 says no `*_a3_diag.json` exists, but
   `prc2/{14B,30B}_t48_a3_diag.json` exist (09-23 16:03/16:32, verified). README:103's "~83% of the
   oracle" holds only for 4B/llama8B; on 14B/30B it is 69%/63%. The README omits the 14B/30B absolute-
   threshold result (raw 0.748→0.663, 0.762→0.680).
9. llama8B t32/t40/t48 `c7gfis − c6gfis` held-out means are all +0.00142 (t64 −0.00145), with different
   per-window vectors. Coincidence, not duplication.
10. `SEARCH_SPACE_MAP.md:9` "protected channels 1% @128 in every parent". Artifacts: psl 68–128 (128 only
    at t96 and the 14B h10 t64 parent); protected MAC share 2.3–3.2% (4B widths ~1% q/k/v/o, 3%
    gate/up, 6% down).
11. `inv/gap.md` §1 says "the mask composition may also differ" for the ceiling. It is identical
    65/58/72/87; only the width differs (INT8 vs INT7).
12. Mapper-internal: `inv/attrib.md` gives the selected R4 move as "0.048%"; 4B cand01 is 0.0614%.
    `inv/synth/v_r2dir.py` reports the "top" as the largest rung <128, which is wrong for cells where 128
    is a real rung (14B t48, the t96 cells). Use `v_topbind.py`/`topcost.py`, which use
    max(`stoc_len_levels`).
13. PRC2_OVERNIGHT's 4B t32 "gfisla vs gfis −0.82% (z −1.8)" is vs the regenerated `c7_gfis`; vs the
    actual incumbent `c6_gfis` it is −0.11% (z −0.22). Both are correct for their comparator. The
    −0.82% is the clean joint term (§3 RC3).

## Appendix B. T1 framing caveat for the rebuttal

- Report best(all) −2.53% as three parts:
  - c17 granularity recalibration: −1.51%;
  - allocation algorithm (global Fisher −0.60%, joint attention λ −0.44%);
  - test selection between r1 and r2: ~0.2–0.4 pp of the total.
- Pure granularity at iso-calibration: 4B t32 −4.33%, llama8B t32 −2.05%, 4B t48 −1.63% in PPL.
  14B/30B t48 exist in held-out error only (0.939→0.748, 0.896→0.762).
- On 30B t40/t48, 74–81% of the gain is per-row attention re-splitting. Round A would add more
  non-granularity gain.
- 14B t96: c17 is +0.03% (worse); best −0.04% is 0.5 noise-sd.
- 8 best(all) dense tables were produced by pre-STE-fix calib6 code (valid, but say so).

## Appendix C. Sources and scripts

- Mapper reports: `inv/{rounds,gap,algo,attrib,protocol}.md`.
- Lens scripts and outputs: `inv/{h1mech,h1skeptic,h2mech,h2skeptic,h3mech,h3skeptic,h4mechsk,h4skeptic,h5,h5mech,h6mech,h6skeptic,h7mechsk,h7skeptic,synth}/`.
- Re-checks run for this document:
  - `inv/final/cell_table.py` (per-cell table);
  - `inv/synth/v_topbind.py`, `v_joint_pred.py`, `v_r2dir.py`, `inv/h1skeptic/topcost.py`
    (all re-run);
  - inline table-identity check over `prc2/{4B,llama8B}_t32_{c6_gfis,c7_gfis,c8w6_gfis,c8w6o_fis}_table.json`;
  - reads of `mp_per_row_chunk_calib7.py:178-190,415-432,452-466,775-805,870-910` and
    `prc_offline_solve.py:1-60,180-277`;
  - wrapper fields of the round-2 t96/14B t48/14B t64 wrappers;
  - `prc2/{14B,30B}_t48_a3_diag.json` `statistic_analysis`;
  - `PAPER_METRICS_20260927.md:59-68`; `hpca_results/llm/int/*.csv`;
    `frontend_awq/awq_mask_mp_table.csv`.
- Artifacts: `hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.{json,md,csv}`;
  Turbo `/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/{prc2,prc_local_20260926,prc_adjacent_20260927}/`;
  Slurm logs `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/`.
