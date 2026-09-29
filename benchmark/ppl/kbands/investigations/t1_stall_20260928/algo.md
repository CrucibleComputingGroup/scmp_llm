# T1 per-group allocation: algorithm and degrees of freedom

Date: 2026-09-27. Read-only investigation: no GPU work, no models loaded, no existing files changed.
Scope: how the current best per-group allocation works end to end (checked in code), which
allocation degrees of freedom (DOFs) exist, which ones have been tried, and why rounds 3–5 found nothing.
Units: code/trace stream lengths are HALVED (nominal = 2×). All numbers come from the files cited in each
section. Helper scripts and JSON outputs are in this directory: `trace_occ2.py` → `occ_all.json`,
`att_occ.py`, `floor.py`, `shapes.py`, `r2_att.py`.

---

## 0. Bottom line

1. **Per-group only affects 51–77% of the SC cycle budget at t32.** The rest goes to per-row attention
   (15–43% of cycles) and a protected-channel slice held at one fixed length (5–8% of cycles). The
   protected slice was carried over from the per-row parent and has **never been re-optimised**. (§3)
2. **Attention is stuck at the top rung of its ladder, and that ladder stops below 128.** Attention
   uses the parent's per-row ladder, and its top rung is below 128 (86–117) in 13 of 20 cells. In 10
   of those 13 cells, ≥85% of qk MACs sit on that top rung (for example, llama8B t32: qk 96% and av
   100% at 96). Only the μ+2τ escape reaches 128. In 14B t64-h20 the ladder tops out at 89 and there is
   no escape, so attention never goes above 89. (§3)
3. **New finding: round 2 won when it raised attention.** Across all 20 round-2 cells, round 2 beat
   round 1 on test in **8 of 9** cells where the joint λ *raised* attention's mean length (mean
   −0.93%). It won in only **1 of 11** cells where it *cut* attention (mean +0.34%). The one-sided
   Fisher exact p is 0.0006, but the grouping was chosen after the fact. In the cells that won, the
   raise was stopped by the ladder top (30B qk buckets at 86–96 of a 96 ceiling; 4B t64 qk at 104.5
   of 105). Every ladder edit tried so far lacked at least one of three features: (a) a solved
   *selective* raise above the old top, (b) no cuts to attention, (c) separate qk/av ladders. The one
   prior "ceiling" test moved *all* top-rung rows up together and paid for it by scaling the linears
   uniformly, and only on 4B/llama8B t32. (§4)
4. **Within the linear representation, allocation is essentially used up.** The only runtime signal is
   the per-call-normalised absmax of each (row, chunk), with one threshold vector per operator and
   layer quartile. Round 1 already solves that global problem in the Fisher surrogate. What is left:
   crossing bucket boundaries is worth ~0.3% in the oracle, a denser ladder ~0.01 pp, per-block keys
   tie, and robust Fisher ties. The remaining gap to the Fisher oracle is per-token sensitivity, and
   no absmax-based rule can see it. That would need a new runtime signal or metadata, which is outside
   the user's constraints. (§5)
5. **Rounds 3–5 perturbed a used-up DOF, at step sizes that could not work.**
   - Round 3 moved 0.9–2% of all cycles, with rung-skipping jumps. 19 of 20 candidates got worse, in
     both directions of every population pair (the one exception is a tie on 30B t40), so the
     incumbent is locally optimal at that scale.
   - Rounds 4–5 moved only **0.005–0.068%** of cycles. With the measured elasticities
     (|dlnPPL/dlnCost| ≤ 0.44), no plausible effect clears the 6-window paired SE (0.35–0.51% PPL).
     39 of 48 search deltas were negative, which fits shared-baseline noise plus a winner's curse, and
     all four winners shrank at confirmation. (§6)

---

## 1. End-to-end map of the deployed per-group allocation (code-verified)

### 1a. Runtime rule for the 7 linear projections (`model/sc_common.py`)
- **Protected slice first** (`SCLinear.forward`, lines ~805–866).
  - The table lists protected input channels per (op, block, unit).
  - They are removed from the operand and run as one separate `_sc_matmul` at the single table-wide
    `protected_channels.stoc_len` (line 855).
  - No rung table is used for that slice, so every protected chunk gets the same length.
  - The residual `x_dispatch` holds the remaining columns (line 866).
- **Per-(row,chunk) dispatch** (`per_row_chunk_rungs`, lines 462–561; called at line 904).
  - Statistic: `m = |x_dispatch|.amax` per (token row, 128-column chunk of the *residual*), with the
    tail chunk at its true width (line 511).
  - Normalisation: min–max over **all (row, chunk) pairs of that call** (line 513). For MoE this is
    one expert's routed tokens.
  - Rung: `torch.bucketize(mn, thresholds)` (line 530), so the rule is **monotone increasing in
    absmax only**. Levels and thresholds must be ascending (`mp/config.py:_load_per_row_chunk`,
    729–793). No sign or metric option exists.
  - **No escape gate on linears.**
- **Quantisation is per (row, chunk) on both operands** (`sc/kernels.py:1622–1668`,
  `fused_quantize_bipolar_perrow` per chunk).
  - The kernel quantises the *smoothed* operand: `apply_smoothing` runs first at `sc/matmul.py:193`.
  - The dispatch statistic is computed on the *unsmoothed* `x_dispatch`. Under AWQ the two differ (see §9).
  - One RNG / cum_indicator table is shared by every chunk (`sc/kernels.py:1576`).
- **Table key.** `get_per_row_chunk(op, block)` (config.py 795–811) resolves `(op, t0, layer_bucket)`.
  - The layer axis is the table's `layer_buckets`, which is 4 (quartiles) in every incumbent.
  - An optional `per_row_chunk.layer_buckets` field (line 749) allows **per-block keys without new
    metadata types**. No incumbent uses it.

### 1b. Runtime rule for attention (qk, av) — per ROW, unchanged by T1
- `_sc_attention_matmul_ab_t` (sc_common.py 1171–1350) computes one metric per (head, query) row over
  all B·H·N rows (line 1240):
  - metric from the table's `dispatch_metrics`, amax/l2/crest with a sign;
  - global min–max normalisation over the call;
  - descending thresholds per (op, layer quartile) on the bucket ladder (`get_levels`, config.py
    841–878);
  - then the **escape gate**: normalised metric > μ_b+2τ_b goes to `escape_stoc_len` = 128 (halved)
    (config.py 880–918, 1111–1180; μ_b/τ_b are stored per bucket as `metric_mean`/`metric_std`).
- Each row is then run through `_sc_matmul(..., granularity="per_row")` with **no `chunk_d` and no
  `rung_table`** (lines 1287–1288). That means one quantisation scale and one length per whole row.
- **Per-chunk attention lengths are impossible without changing execution.** `rung_table` requires the
  chunked bipolar path (`matmul.py` gate ~199; the 3D batched path rejects it at `kernels.py ~1779`).
  Using it for av would also change av's quantisation from per-row to per-128-key-chunk scales.

### 1c. Table format actually consumed (all 20 best(all) incumbents; wrapper + table JSON)
| element | granularity | incumbent value |
|---|---|---|
| linear thresholds | per (op, layer quartile): 7×4 = 28 vectors of 16 thresholds | solved |
| linear ladder | per bucket allowed; **all 28 buckets use the same 17-rung** `[8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128]` | frozen since c17 |
| attention thresholds | per (op, layer quartile): 2×4 = 8 vectors | parent's (c17/gfis) or re-solved (gfisla) |
| attention ladder | per bucket allowed (`bucket_stoc_len_levels`); **no incumbent sets one**, so qk and av share the parent's global `stoc_len_levels` (5–8 rungs) | frozen (parent) |
| attention statistic | per operator (amax/l2/crest, ±) | parent's |
| escape | `escape_gate_k`=2.0, `escape_stoc_len`=128, per-bucket μ/τ | parent's; **absent** in all t96 cells and 14B t64-h20 |
| protected slice | channel set per (op, block, unit); ONE global `stoc_len` | parent's (psl = 68/80/84/95/112/128) |
| per_row_chunk.layer_buckets | optional finer layer axis | unused (None) everywhere |

(Tabulated by the inline script in this session from each incumbent's wrapper/table; see §3.)

### 1d. Calibrators (how the tables were produced)

**c17: `mp_per_row_chunk_calib2.py` (raw squared error, within-bucket).**
- Hooks every SC linear on the per-row **parent's** trajectory (L409–470).
- Takes the metric on the full call (L436) and samples 64 rows per call.
- Error curves: `‖SC(x_c,w_c,L) − x_c w_cᵀ‖²` summed over outputs, per chunk, with AWQ smoothing and
  the true tail width (L218–235).
- Exact monotone-staircase DP over 2048 metric-sorted bins at a bisected λ (L108–187).
- **The budget is pinned per (op, quartile)** to the parent's MAC-weighted L on the same calibration
  pairs (L490). Linears never trade budget with each other, with other layers, or with attention.

**c6 `gfis` / `gfis_emp`: `mp_per_row_chunk_calib6.py` (round 1).**
- Currency is `e_c(L) = Σ_j g²_{row,j}·ε²_{c,j}(L) − (same at L=128)` (docstring L9; `cur_of` L462).
- g = dLoss/dy_op on the parent SC trajectory through an exact straight-through gradient.
- `fis` draws **one** MC label per position (L397/439); `fis_emp` uses the true next token.
- **One global λ across all 28 linear keys** at the parent's total linear calibration cost (L533).
  This redistributes budget across operators and quartiles. The within-bucket version (`wfis`) was
  worse: on 4B t32 held-out, gfis − wfis = −1.66%, z −4.4 (PRC2_OVERNIGHT.md, 18:30 entry).
- Attention stays at the parent. The protected slice is excluded from both the budget and the DP.
- "h20" (c6h20gfis) means the same algorithm on the h=20% 14B t64/t96 parents; it is not a different
  currency.

**c7 `gfisla`: `mp_per_row_chunk_calib7.py` (round 2).**
- Adds attention rows to the **same λ**.
- Attention MAC per row = K·M, counted dense (L460).
- Attention thresholds are re-solved **on the parent's own bucket ladder** (`lad = get_levels(...)`, L422).
- Escape rows are fixed cost (L791). Total target = parent linear + attention cost (L794).
- Attention sampling is `--att-rows-per-call 128` out of B·H·N = 65,536 rows per call (4B), i.e.
  0.2% of rows. Linears sample 64 of 2048 rows (3.1%). So attention's Fisher estimate is far sparser.

**Rounds 3–5: `prc_local_*`, `prc_adjacent_*`.** These edit thresholds only, on frozen incumbents, and
select by measured paired NLL. Ladders, statistic, protected channels and escape are frozen by manifest
(`prc_adjacent_20260927.json` → `scope`).

---

## 2. Operand shapes and chunks per row (checked in trace `d_in`/`d_out` + HF config.json)
Residual width = in_features − protected width. Chunk count = ceil(residual/128). Tail widths match the traces.

| model | q/k/v | o | gate/up | down | qk (K, out) | av (K, out) |
|---|---|---|---|---|---|---|
| 4B (2560/9728, 32 H) | 2534 → 20 (tail 102) | 4055 → 32 (87) | 2483 → 20 (51) | 9144 → 72 (56) | 128, 2048 | 2048 keys, 128 |
| llama8B (4096/14336) | 4055 → 32 (87) | 4055 → 32 (87) | 3973 → 32 (**5**) | 13475 → 106 (35) | 128, 2048 | 2048, 128 |
| 14B (5120/17408, 40 H) | 5068 → 40 (76) | 5068 → 40 (76) | 4966 → 39 (102) | 16363 → 128 (107) | 128, 2048 | 2048, 128 |
| 30B-A3B (2048; experts 768) | 2027 → 16 (107) | 4055 → 32 (87) | experts 1986 → 16 (66) | experts 721 → 6 (81) | 128, 2048 | 2048, 128 |

- qk contracts over head_dim = 128, which is exactly one chunk. For qk, per-row already *is* per-group.
- av contracts over 2048 keys (16 possible chunks), but it is quantised and dispatched per whole row.
- Sources: `shapes.py` output. Configs are under `/nfs/turbo/coe-nbleier/allenjin/hf_cache/.../config.json`
  (Llama head_dim = 4096/32). The 30B tables use `Qwen/Qwen3-30B-A3B-Instruct-2507`.

---

## 3. Where the cycle budget goes in the best(all) incumbents (full-test traces)
Source: `trace_occ2.py` → `occ_all.json`. Protected slice = trace groups whose `d_in` equals the protected width.

| cell | linear share of SC cycles (per-group) | protected share (len) | attention share (mean L) |
|---|---:|---:|---:|
| 4B t32 | 60.3% | 7.0% (84) | 32.8% (84.6) |
| llama8B t32 | 72.3% | 6.2% (68) | 21.4% (96.6) |
| 14B t32 | 77.3% | 7.8% (80) | 14.9% (90.1) |
| 30B t32 | 51.3% | 5.5% (80) | 43.2% (63.0) |
| 30B t48 | 55.1% | 5.2% (112) | 39.8% (86.4) |

Attention MAC share: 4B 13.1%, llama8B 7.3–7.5%, 14B 5.5%, 30B 23.3%.

**Attention rows sit on a sub-128 top rung** (`att_occ.py`; % of MACs on the top ordinary rung, escape
at 128 separate):

| cell | top rung | qk | av |
|---|---:|---:|---:|
| 4B t32 | 97 | 58% | 67% |
| 4B t40 | 96 | 93% | 86% |
| 4B t48 | 111 | 58% | 67% |
| 4B t64 | 105 | 95% | 92% |
| llama8B t32 | 96 | 96% | 100% |
| llama8B t40 | 98 | 97% | 99% |
| 14B t32 | 96 | 88% | 80% |
| 14B t40 | 86 | 91% | 92% |
| 14B t64-h20 | 89 (no escape) | 93% | 92% |
| 30B t32 | 96 | 75% | 1% |
| 30B t40 | 96 | 85% | 23% |
| 30B t48 | 109 | 89% | 28% |
| 30B t64 | 117 | 97% | 100% |

In llama8B t48/t64, 14B t48 and all t96 cells the top rung is 128.

**Linear floor and cap** (`floor.py`):
- MACs at L=8 are ≤4.6% (14B t32), which is ≤1.4% of linear cycles, so the floor barely binds.
- At t96, 22–35% of linear MACs are at ≥112, so the 128 cap (an SC invariant) binds.

---

## 4. Round 2's result depends on which way it moved attention (new, `r2_att.py`)
Source: `prc_round2_status_20260927.json` (r1/r2 test PPL + traces) plus each trace's attention mean L.

| group | n | round 2 beats round 1 on test | mean (r2 − r1) |
|---|---:|---:|---:|
| joint λ raised attention mean L (>+0.5%) | 9 | **8** | **−0.93%** |
| joint λ cut attention mean L (<−0.5%) | 11 | 1 | +0.34% |

Examples:
- 30B t40: qk 67.6→92.8, test −2.80%.
- 4B t40: qk 82.8→94.9, av 84.7→89.9, −0.94%.
- 4B t64: qk 89.4→104.5, −0.57%.
- 14B t64-h20: qk 87.9→79.2, +0.76%.
- 30B t96: qk 127.5→101.5, +0.75%.
- llama8B t40: cut, +0.90%.

Exceptions: 14B t48 raised and lost (+0.20%); 14B t32 cut and won (−0.30%). One-sided Fisher exact
p = 0.0006. The grouping is post hoc, so treat it as a lead, not a result.

Solve logs show the raise hitting the ceiling:
- `p7_30B_t32_c7_61848816.out`: qk buckets solve to L 85.9–92.3 on the ladder `[16,24,32,48,64,96]`.
- `p7_30B_t40_c7_*`: qk l3 at 95.8 of 96.
- `p7_30B_t48_c7_*`: qk l2 at 108.5 of 109.
- 4B t64 winner: qk 104.5 of 105.

Reading: in this surrogate (the MC diagonal Fisher on attention outputs), the cut direction is
unreliable, while the raise direction is productive but capped by the inherited ladder. This matches
the note already in `PRC2_OVERNIGHT.md` (round-2 lever map), now confirmed on all 20 cells.

---

## 5. How used up is the linear representation? (existing diags)
From `4B/llama8B/14B/30B_t32_c7_diag.json`, held-out, parent = 1:

| cell | deployable gfis, Fisher error | global Fisher oracle | within-bucket oracle | share of oracle reduction captured |
|---|---:|---:|---:|---:|
| 4B t32 | 0.597 | 0.215 | 0.244 | 51% |
| llama8B t32 | 0.645 | 0.222 | — | 46% |
| 14B t32 | 0.668 | 0.211 | — | 42% |
| 30B t32 | 0.507 | 0.135 | — | 57% |

- `fis/all` equals `fis/ladder` to 3 decimals, so for L ≥ 8 more rungs buy nothing. The measured set
  never went below 8.
- Crossing bucket boundaries: 4B t32 within-bucket 0.244 vs global 0.215, about −0.3% predicted PPL.
- The oracles are per pair and see each token's Fisher weight, which a deployable rule cannot. The
  offline robust re-solve found the useful part of Fisher is mostly per-(op, block) sensitivity:
  `kraw:q` ≈ `fis:q` (PRC2_OVERNIGHT.md, 2026-09-24 late).
- Per-token Fisher mass is extremely heavy-tailed: in the 12-window dump, the top 1% of rows carry
  99.2% of the mass.
- So the remaining linear headroom is representational (it needs a new runtime signal), not a search
  problem.

---

## 6. Rounds 3–5: why nothing was found
Sources: Turbo `prc_local_20260926/<cell>/search_results.json`, `selected.json`;
`prc_adjacent_20260927/<cell>/…`; logs `p3local20260926_*`, `p4adjacent20260927_62104342_{0,1}.out`,
`p4adj30b20260927_62105189_{0,1}.out`.

**Round 3 (coarse, 0.91–2.0% of cycles moved).**
- 19 of 20 candidates were worse on search. Both directions of each population pair were worse
  (e.g. 4B t32: qk←linears +1.40%, linears←qk +2.25%), except one tie: 30B t40 mlp←projections
  −0.05%, z −0.07.
- Test: 4B +0.212%, llama8B +0.342%, 30B t40 −0.030% (the only "win", z −0.15).
- **"Round 5" = the 30B companion array 62105189** (30B t32/t40): same driver, protocol and gate,
  separate manifest `prc_adjacent_30b_20260927.json`. Rounds 4 and 5 are summarised together below.

**Round 4 (dense 4B/llama8B t32) and round 5 (30B t32/t40), adjacent-rung moves.**
- Realised transfers were 0.0048–0.0685% of total SC cycles across all 48 candidates. The requested
  "ceilings" were 0.25–0.5%, so the moves were 4–50× smaller.
- 39 of 48 search deltas were negative. Per-candidate paired SE on 6 windows was 0.0035–0.0051 NLL
  (≈0.35–0.51% PPL).
- Best candidates, search → confirmation:

| cell | search | confirmation |
|---|---|---|
| 4B t32 | −0.446% | +0.138% |
| llama8B t32 | −0.749% (z −2.19) | −0.398% (z −1.98); test 8.28586 vs incumbent 8.27948 = **+0.077%** |
| 30B t32 | −1.138% (z −2.53) | −0.329% (z −0.68) |
| 30B t40 | −0.644% | −0.079% |

**Power check.**
- Within-cell elasticity d lnPPL / d lnCost, from SUMMARY §2a (c17 vs the s80 arm): 4B t32 −0.442,
  llama8B t32 −0.342, 14B t32 −0.194, 4B t48 −0.109.
- Moving δ = 0.02–0.07% of cycles, the effect is about k·|e|·δ, where k is the recipient/donor
  marginal-value ratio.
- Reproducing the search deltas would need k ≈ 16 (4B), ≈ 110 (llama8B) and ≈ 140 (30B t32). The
  30B figure assumes |e| ≈ 0.4, since no 30B within-cell elasticity has been measured.
- At a realistic k the true effect is ≪ 0.1% PPL, below the SE. The frequent negative deltas fit a
  shared incumbent baseline plus picking the best of 12. The confirmation gate did its job.

---

## 7. Every allocation DOF, classified

Classes:
- **(a)** pure allocation, explored (result given)
- **(b)** pure allocation, frozen or unexplored
- **(c)** grey: a runtime-rule or statistic choice among *existing* mechanisms; needs user sign-off
- **(d)** forbidden by the user's constraints

"Extra reduction" means an additional per-group reduction at runtime.

| # | DOF | where set | class | status / evidence | runtime cost note |
|---|---|---|---|---|---|
| 1 | linear granularity (row, 128-chunk of residual) | sc_common 462–561 | (a) | T1 itself: r17→c17 −4.42% raw on 4B t32, −2.05% llama8B t32 (SUMMARY §3) | — |
| 2 | linear thresholds, 28 vectors (op × quartile) | table `per_row_chunk.buckets` | (a) | c17 within-bucket; gfis global λ (headline); rounds 3–5 local edits: no gain | none |
| 3 | budget split across linear ops/quartiles | calibrator | (a) | c17 pinned: 20/20 improve vs parent, mean −1.49%; gfis global λ adds more, e.g. −1.25% vs c17 on 4B t32, −1.32% on llama8B t32 | none |
| 4 | budget split linears ↔ attention | calib7 joint λ | (a), partly | gfisla: wins 30B t32–t48 (−1.55/−2.80/−2.29% vs r1); dense mixed; **depends on direction (§4)** | none |
| 5 | linear ladder density (17 rungs, 8..128) | table `levels` | (a) | "all measured lengths" oracle = 17-rung oracle (Δ≈0.01 pp); pow2-only costs +0.9–5% (per-row era) | k_table slice per rung, cached |
| 6 | linear ladder floor < 8 (L = 1..7) | table `levels` | **(b)** unexplored | L=8 holds ≤4.6% of linear MACs, ≤1.4% of linear cycles, so upside ≤~0.3% PPL | none |
| 7 | linear ladder cap > 128 | SC invariant | (d) | `stoc_len ≤ 2^(sc_prec−1)` halved; cap binds at t96 (22–35% of MACs ≥112) | — |
| 8 | per-block linear keys (`per_row_chunk.layer_buckets`) | config.py 749 | (a) | c8 robust blk candidates tie (kraw:blk +0.0011 NLL, z +0.25); calib3 held-out 0.593 vs 0.600 (4B t32) | more table rows, same format |
| 9 | calibration currency: raw / σ / Fisher-MC / Fisher-emp / clipped / kraw | calibrator | (a) | MC Fisher best; emp wins only llama8B t40; global σ lost 4B (+1.3%); clip/kraw tie; raw 12-window Fisher catastrophic (+32.5% held-out) | — |
| 10 | Fisher estimator variance (1 MC sample/token, 6 windows, 64 lin rows, **128 att rows/call**) | calib6/7 args | **(b)** partly | more windows raw → spikes; clipped tie; **multi-sample MC and denser attention sampling untested** | offline only |
| 11 | expansion point (parent vs child trajectory) | calib8 `--expand-at` | (a) | re-linearised on round-1: worse (fis:q +0.0306 NLL, z +7.5) | offline |
| 12 | total budget vs parent (cost drift, user allows ~1%) | bisect target | **(b)** | incumbents are −1.34% (14B t32), −1.02% (30B t48), −0.94% (14B t40) vs submitted cost; at e = −0.19…−0.44, re-targeting to +0% is worth ≈0.1–0.4% | none (disclose) |
| 13 | **attention ladder** (top rung, rung set, separate qk/av, per quartile) | wrapper `stoc_len_levels` + bucket `stoc_len_levels` | **(b)** mostly unexplored | only test: uniform top-rung raise to 112/128 with thresholds untouched and linears scaled uniformly, 4B t32 (worse, +0.0178 NLL) and llama8B t32 (neutral). **Never** a joint-λ selective raise, separate qk/av ladders, or any 30B/14B cell | none (table) |
| 14 | attention thresholds (8 vectors) | table buckets | (a) | gfisla; rounds 3–5 local edits | none |
| 15 | direction constraint on attention in the joint solve (no cut / price multiplier κ_att) | calibrator | **(b)** unexplored | §4: cuts lose 10/11, raises win 8/9 | offline |
| 16 | per-block attention thresholds/ladders | table `layer_buckets` = n_blocks (also re-keys the unused per-row linear buckets) | **(b)** unexplored | linear analogue tied, so low expected value | more table rows |
| 17 | attention sampling density in calibration | calib7 `--att-rows-per-call` | **(b)** | 0.2% of attention rows vs 3.1% of linear rows | offline |
| 18 | **protected-slice length psl** (one global scalar) | table `protected_channels.stoc_len` | **(b)** never re-optimised in the per-group era | 5–8% of SC cycles at t32 (psl 68–84 vs linear mean 23–28); per-row-era V17: "psl~80 is the biggest cheap lever" | none |
| 19 | per-op protected length | runtime reads one global value | (c) | needs a small code change | — |
| 20 | protected channel set / fraction | table indices | (c) | an existing static per-column mechanism; changing it changes residual chunking and quantisation; not re-tuned since the parent | — |
| 21 | escape gate k (μ+kτ), escape length | wrapper | (c) | paper states μ+2τ, 256 nominal; absent in 5 incumbents already | — |
| 22 | attention dispatch statistic (amax/l2/crest ±) | table `dispatch_metrics` | (c) | parent's ρ-selection; μ/τ must be recomputed if changed | l2 and crest need an extra reduction (crest needs both) |
| 23 | linear statistic (raw absmax, smoothed absmax, l2, crest) | hard-coded in `per_row_chunk_rungs` | (c) code | calib3 held-out: smoothed worse (0.631 vs 0.600 on 4B t32; 0.778 vs 0.748 on 14B t48); composite ≤0.3 pp | smoothed absmax is the only literally free one |
| 24 | normalisation: absolute thresholds (drop per-call min–max) | runtime rule | (c) | calib3 held-out raw error: 4B t32 0.571 vs 0.600, 4B t48 0.592 vs 0.634, **14B t48 0.663 vs 0.748, 30B t48 0.680 vs 0.762**, llama8B ≈ 0; the largest measured error lever left, but it changes the paper's "per-call relative" rule | removes a reduction |
| 25 | monotone direction for linears (inverted / non-monotone) | `_load_per_row_chunk` enforces ascending | (c) code | per-row parents used inverted metrics (v_proj l2−, up_proj crest−); untested for per-group | — |
| 26 | static chunk classes (chunk→class map) | new table field + lookup | (d) | new runtime metadata; CPU signal exists (rank corr 0.69–0.81, ROUND3_INVESTIGATION) | new metadata |
| 27 | per-expert (unit) thresholds for MoE | not keyed by unit | (d) | new keys + code | new metadata |
| 28 | av per-128-key-chunk dispatch | attention path | (d) | changes quantisation granularity + kernel path | execution change |
| 29 | chunk size `sc_linear_chunk_d` | config | (d) | changes quantisation | — |
| 30 | INT mask / dose | hybrid_config | (d) | user | — |
| 31 | RNG / enable grid (`rng_levels`, `SC_RNG_GRID`, attention grid policy), Owen, masks | env/kernel | (d) | user | — |
| 32 | qk rebalance (`sc_attn_smooth`), AWQ scales, asymmetric SC, halving | — | (d) | user | — |

---

## 8. Why per-group PPL is not better — from the algorithm / DOF angle
1. **Structural reach.** The T1 fix touches only the linear dispatched groups: 51–77% of SC cycles at
   t32, 55–88% overall. Attention (qk is a single chunk; av is not chunked) and the protected slice
   are unaffected by granularity.
2. **Inherited, frozen ceilings.** Attention runs on the per-row parent's compressed ladder (tops
   86–117), and most attention MACs sit on that top rung. Every per-group round has frozen it. Where
   round 2 could raise attention it won 8 of 9 times; where the ladder only allowed a cut it lost 10
   of 11 times.
3. **Frozen protected slice.** 5–8% of t32 cycles run at the per-row-era psl and sit outside every DP.
4. **The linear rule is near its representational optimum.** It sees only normalised absmax per
   (row, chunk). The Fisher oracle gap is per-token sensitivity, which cannot be reached without new
   metadata or hardware. Ladder density, bucket boundaries and per-block keys are ≈0; robust Fisher ties.
5. **Surrogate fragility.** One MC sample per token gives a heavy-tailed Fisher, and attention is
   sampled 16× more sparsely than linears. Fisher predictions are good for large moves (30B) but
   unreliable for cutting attention and for small moves (ROUND3_INVESTIGATION item 3).
6. **Diminishing returns with budget.** Parent linear error falls ~1/L². At t96 about ⅓ of linear
   MACs sit at the 128 cap (SC invariant), and escape/attention are also at 128.
7. **Rounds 3–5 were aimed at DOF #2, which is already used up**, with non-local moves (round 3) or
   moves too small to measure (rounds 4–5).

---

## 9. Mismatches between docs and artifacts
1. **The "free statistic" docstring is wrong under AWQ.** `sc_common.per_row_chunk_rungs` claims "the
   statistic is FREE … exactly the group quantization scale the kernel computes". The dispatch absmax
   is computed on the unsmoothed `x_dispatch` (line 904 input), while the kernel scale is on `x/s`
   after `apply_smoothing` (matmul.py:193). The CLAUDE.md phrase "maximum magnitude can reuse the value
   already computed for the group scale" is therefore not true of the implementation as deployed
   (TWO_PHASE_DESCRIPTION item 6 agrees). Switching to the smoothed statistic is worse in held-out
   error (DOF #23).
2. **Escape is not in every parent.** SEARCH_SPACE_MAP says escape k=2.0 is "in every parent". Five
   incumbents have no `escape_gate_k`: all four t96 cells (ladder top 128, so escape is moot) and
   **14B t64-h20, whose attention is capped at 89 with no escape**. The paper's μ+2τ statement does
   not cover that cell.
3. **Attention thresholds are not always the parent's.** CLAUDE.md/README say "attention keeps the
   parent's per-row thresholds". That holds for c17/gfis, but **9 of 20 best(all) cells** (7 c7gfisla,
   1 c7h20gfisla, 1 round3_local derived from gfisla) re-solved attention thresholds.
4. **The status block is stale.** `scmp_llm/CLAUDE.md` still says "No round-4 PPL results yet". All
   four tasks (arrays 62104342, 62105189) completed on 2026-09-27 between 15:20 and 17:36. Only
   llama8B t32 reached full test, at 8.28586 (not better).
5. **Round-4 transfers were far below the stated sizes.** PRC_ADJACENT's "requested transfers of
   0.25% and 0.5%" became realised transfers ≤0.0685% in every candidate.
6. **"Only cut" applies to pinned cells only.** PRC2_OVERNIGHT says "attention ladders top out at 96 …
   so the joint λ can only CUT attention". That is true only for pinned cells; in 9 of 20 cells round 2
   raised attention.

---

## 10. Candidate rounds from this angle (input for synthesis, not a launch plan)
All three change only the table. No RNG, QK, mask, AWQ, chunking or metadata-type changes.

- **R-A: attention unpinning in the joint λ.**
  - Cells: attention top < 128. Priority 30B t32/t40/t48, then 4B t40/t64, 14B t40/t64-h20,
    llama8B t32/t40.
  - Give qk and av separate per-quartile ladders (bucket `stoc_len_levels`) that reach 128, e.g. the
    parent's rungs ∪ {112, 128} or a dense set.
  - Measure attention error at those lengths (the escape length 128 is already in `lv_meas` when the
    gate is on).
  - Raise `--att-rows-per-call` 128→512.
  - Solve two variants: an unconstrained joint λ, and one with no attention cuts (each attention
    bucket's mean L ≥ the parent's).
  - Keep μ+2τ escape semantics (the escape length becomes the top rung).
  - Evidence: §3, §4.
- **R-B: protected-slice length inside λ.**
  - Measure the protected slice's Fisher-weighted error at psl ∈ {parent, 48, 64, 80, 96, 112, 128}.
  - Do a 1-D outer search over psl, re-solving λ for the remaining budget each time; iso-cost.
  - All tight cells. Evidence: §3, DOF #18.
- **R-C (small add-on): budget exactness plus lower-variance Fisher.**
  - Target realised test cost = parent cost. Seven incumbents are 0.60–1.34% under (14B t32/t40/t48/t64,
    30B t32/t48/t64).
  - Multi-sample MC labels (e.g. 8 per position) to damp the heavy tail.
  - Fold into R-A/R-B rather than running it alone.
- **Needs user approval (grey):** absolute thresholds for linears (DOF #24). This is the largest
  measured error lever left for 14B and 30B, but it is a runtime-rule change.

Selection design lessons from rounds 3–5:
- Moves must be large enough for their expected effect to exceed the 6-window SE (~0.4% PPL).
- Compare each candidate against a *re-evaluated* incumbent, or use a two-stage design with fresh
  windows (as rounds 4–5 did), and account for picking the best of N.
