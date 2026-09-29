# prc2 — making per-group (per-(row,chunk)) allocation pay (started 2026-09-22 night)

Goal (user, 2026-09-22): a LARGE, paper-reportable gain from per-group allocation, from
**pure better allocation only** (ladders / thresholds / budget distribution / their
calibration) — no qk rebalance, no enable-grid, no mask changes, no new modules.

## The comparison
- **parent** = submitted per-row `mp_best` allocation, AWQ-deployed, same INT7 mask
  (`KB_ARM=parent`, `prcppl_<m>_t<T>_parent_*`, re-run 2026-08-03 in the same env; the
  env reproduced August to the digit on 2026-09-20, `plawctl_llama8B_t32`).
- **v7 prc** (the −1.04% result) = same wrapper + per_row_chunk section from
  `mp_per_row_chunk_calib.py` (`prcppl_<m>_t<T>_prc_*`).
- **prc2** (this work) = same wrapper + per_row_chunk section from
  `mp_per_row_chunk_calib2.py`. Attention thresholds, INT mask, front end unchanged.
  Each (op, layer-bucket) is budgeted at the parent's TRACED dispatched cost.

Baseline A/B (v7 vs parent), from `build_prc_ab.py`: 20 cells, mean −1.04%, median
−0.38%, 16/20 improve, costs differ −11.6%…+18.1% (not iso-cost).

## Why v7 under-delivered (diagnosed tonight; all are calibration/table defects)
1. thresholds cut from `x[:256]` of one window, but runtime min-max-normalizes over the
   full 2048-row call → deployed per-op mean L −6…−24% vs intended (down_proj to −47%);
2. only the first 4 calls per (op, bucket) — the first ~4 blocks of each layer bucket;
3. the hook sampled INT-masked modules (27/112 calls on 4B);
4. error curves measured WITHOUT the module's AWQ `smooth_scales` (runtime applies them);
5. tail chunk dropped;
6. ladder = the parent's per-ROW ladder (e.g. 4B t64 [48,60,61,62,63,64,105,128]):
   43–84% of linear MACs sit on the floor at t40–t64, 4–16% of SC cycles movable;
7. histogram imposed from the unconstrained oracle instead of solved for the
   monotone staircase the runtime can express;
8. budgets from the 256-row sample (+9…23% at t32) or trace incl. protected slice.

## prc2 calibrator (`benchmark/ppl/mp_per_row_chunk_calib2.py`, table-only)
Fixes 1–8: full-call normalization; all blocks; SC modules only; smooth_scales applied;
tail chunk included; dense ladder [8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128]
(halved; 128 kept); Lagrangian DP over monotone staircases on metric-sorted bins (exact
per multiplier, verified = brute force on 300 random problems; multiplier bisected to
the budget). Pre-flight on held-out windows: realized L and squared error for parent /
v7 / prc2.
- **`d17` (first attempt, 02:11–03:35, NOT used for any cited number):** prefix train
  windows (3 calib + 1 held-out), 128 rows/call, per-bucket budget = parent TEST trace
  (dispatched groups only).
- **`c17` (all cited numbers):** `--budget-source calib` (per-bucket budget = the parent's
  own MAC-weighted L on the same calibration pairs — no test information), 6 calib + 2
  held-out windows stratified over the train stream (seed 0), 64 rows/call, 8 expert
  calls per MoE block and window (30B t32 retry `c17e32`: 32).

## Corrections applied 2026-09-23 afternoon (after an adversarial audit)
- Cost now comes from each full-eval TRACE (Σ macs·L / Σ macs over all SC groups). The
  [RESULT] tracker prices each (row,chunk) pair at 1/n_chunks of the row and reads
  per-group arms up to ~0.4% cheap (llama8B t96 c17: tracker −0.4%, trace +0.04%).
  Numbers below that quote tracker costs are superseded by `hpca_results/llm/ppl/prc2/SUMMARY.md`.
- On the linears the per-(row,chunk) path does NOT run the parent's μ+2τ escape gate
  (its dense ladder reaches 128); attention keeps it. "Same wrapper … escape gate" was
  wrong for the linears.
- Granularity −4.42% is RAW; r17 spent 1.1% less than c17 (≈−4.0% cost-adjusted,
  inferred). Cost-matched control `r17m` queued. Per the user's rule (~1% cost doesn't
  matter, 1% PPL does) the raw number is the one to quote.
- The 09:18 30B t32 gate failure was NOT "expert-sampling noise": dense q_proj/o_proj
  over-spent and the gate's aggregate under-weighted MoE experts (~11% vs ~67% of linear
  MACs) because sampled expert pairs were not re-weighted by 1/(sampling rate). True-MAC
  ratio 1.007. Measurement fixed in calib2/3 before any remaining 30B c17 result.
- The calib3 analysis jobs crashed on a float32 JSON dump after printing their tables
  (no `*_a3_diag.json`); numbers live in the logs, now copied to Turbo
  `/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2/logs/`.
- 14B t64/t96 parents run an h=10% INT mask (paper table cells are h=20%).
- "Same env": parents ran 2026-08-03, kernels.py changed 08-07 (uncommitted); a post-move
  parent re-run (`ctl_4B_t32_parent_postmove`) is queued to confirm the baseline.
- The 0.8× run saves 20% of DISPATCHED linear cycles (17.9% incl. the unscaled protected
  slice; 12.0% of total SC cycles by trace).


## MORNING SUMMARY (as of 09:25, 2026-09-23; later cells still running — see Log)

**Pure-allocation per-(row,chunk) vs the submitted per-row parent, full protocol, AWQ,
same INT mask / attention / front end; costs are REALIZED SC cost from each trace.**

| cell | parent | prc2 (c17) iso-cost | v7 (old) | fp16 excess removed |
|---|---:|---:|---:|---:|
| 4B t32 | 11.9886 | **11.3305 (−5.49%, cost −0.3%)** | −5.01% @ +3.7% cost | 34% |
| 4B t40 | 11.1440 | **10.8766 (−2.40%, −0.3%)** | −0.60% | 24% |
| 4B t48 | 10.6491 | **10.5030 (−1.37%, −0.3%)** | −0.37% | 24% |
| llama8B t32 | 8.6758 | **8.3899 (−3.30%, −1.3%)** | −2.18% @ +0.4% | 20% |
| llama8B t40 | 8.2093 | **8.0857 (−1.51%, −0.8%)** | +0.68% | 12% |
| llama8B t48 | 7.9200 | **7.8151 (−1.32%, −0.8%)** | −0.24% | 15% |
| 14B t32 | 9.3097 | **9.1672 (−1.53%, −1.1%)** | −1.63% @ +6.6% | 21% |

* **Iso-PPL:** 4B t32 at 0.8× linear budget = 11.9836 vs parent 11.9886 → **same PPL with
  12.1% fewer total SC cycles** (tracker; 12.0% by trace; 20% fewer dispatched linear cycles). Held-out predicts iso-error at
  0.75–0.78× linear budget on 4B, 0.85–0.87× on 14B/30B, 0.87–0.91× on llama8B.
* **Granularity vs recalibration (4B t32):** identically-calibrated per-ROW control =
  11.8549 (−1.12% vs parent) ⇒ **granularity alone = −4.42%** (≈80% of the total).
  Error axis: per-row-same-calib 0.848, per-(row,chunk) 0.600, oracle 0.518 (4B t32);
  0.885 / 0.790 / 0.748 (llama8B t32). The deployable rule captures 83% of the oracle.
* Every cell is iso-cost within 1.3% (the gate enforced [−3%,+2%] on held-out train data).
* Headroom is model-dependent: per-(row,chunk) helps most where chunk magnitudes are
  heterogeneous (4B q/k/o/down); llama8B's oracle ceiling itself is only 0.75× error.
* NOT used anywhere: qk rebalance, enable-grid, loss-priced mask, new runtime modules.
  The only runtime edit is the default-off `SC_PRC_ROWSHARED` ablation switch.

**Caveats to state honestly:** (1) the parent is SmoothQuant-calibrated/AWQ-deployed; the
row control shows most of the gain is granularity, not recalibration, on 4B t32 — llama8B
t32 and 4B t48 controls are queued; (2) t64/t96 cells will show small numbers (little
fp16 excess left); (3) 30B t32 failed the cost gate (+2.8%) and is being recalibrated with
denser expert sampling — the gate was not relaxed.

## Log
- 2026-09-22 ~23:00 — partition saturated (lab interactive sessions hold the GPUs).
  Replaced separate calib/eval jobs with ONE chained job per cell
  (`run_prc2_calib.sbatch`, P2_EVAL=1): calibrate (128 rows/call, 3 calib + 1 held-out
  window) → pre-registered gate (held-out linear L within [−3%,+2%] of parent AND
  held-out squared error < parent) → full-protocol eval via run_prc_ppl.sbatch
  (trace tag `<m>_t<T>_prc_p2d17`). 20 jobs `p2_<m>_t<T>_d17`, ids in
  scratchpad p2_jobs.txt (61759485…61759505).
- 2026-09-23 02:11–03:35 — all 20 `d17` calibrations ran (7–24 min each). Every eval
  was skipped or crashed, for two reasons:
  * **ENV BUG (affects every eval in `annstention`, not just this work):** the editable
    install `site-packages/__editable___scmp_kernels_0_1_0_finder.py` still maps
    `scmp_kernels -> /home/allenjin/Projects/scmp_llm/kernels/scmp_kernels`, which is gone
    since the 2026-09-21 SCMP/ move ⇒ `ModuleNotFoundError: scmp_kernels` inside
    eval_quant. Worked around with `PYTHONPATH=.../SCMP/scmp_llm/kernels` in
    run_prc2_calib.sbatch; the env itself is NOT modified. Permanent fix (user's call):
    edit that MAPPING path, or `pip install -e SCMP/scmp_llm/kernels` again.
  * **gate failed on 4B/llama8B/14B** because budgets came from the TEST-set parent
    trace while calibration used the first train windows (one article), where the
    per-row parent itself spends +22% on down_proj vs its test trace; new tables were
    matched to the trace, so on held-out they looked 5–10% cheaper than the parent.
    30B matched (1.003) and passed.
  * d17 held-out pre-flight (linear squared error vs parent; v7 for reference):

    | cell | L new/par | L v7/par | err new/par | err v7/par |
    |---|---:|---:|---:|---:|
    | 4B t32/40/48/64/96 | .93/.95/.98/.98/.97 | 1.12/.95/1.04/1.00/.98 | .63/.65/.66/.69/.87 | .49/.75/.64/.69/.90 |
    | llama8B | .94/.97/.97/.98/.98 | 1.08/.95/1.04/1.03/.98 | .79/.85/.85/.91/.96 | .72/.90/.85/.89/1.00 |
    | 14B | .90/.92/.95/.96/.96 | 1.15/.90/1.07/.97/.98 | .82/.77/.74/.86/.94 | .66/.87/.72/.81/.90 |
    | 30B | 1.00/1.00/.99/.99/1.00 | 1.24/.98/1.06/1.02/1.01 | .74/.75/.76/.74/.87 | .43/.81/.59/.66/.83 |

    Where v7 is near iso-cost (t40) prc2 beats it on all 4 models. v7's lower error
    at t32/t48 is bought with +4…+24% cost on this data.
- 04:55 — `c17` = calib budgets (parent's own L on the SAME calibration pairs; no test
  information) + stratified windows (6 calib + 2 held-out, spread over the train stream)
  + 64 rows/call; emits an extra table at 0.8× budget (`_s80`). 20 chained jobs,
  priority order 4B/llama8B t32–48 → 14B/30B t32,48 → rest (ids in c17_jobs.txt).
- 05:05 — analysis job `calib3` (4B t32, llama8B t32, 14B t48, 30B t48, 4B t64): the
  staircase on today's statistic reaches only 1/1.3–1/1.6 of the oracle's error on 4B
  down_proj, so the dispatch STATISTIC is the next suspect. calib3 scores, on the same
  held-out data and budget: unsmoothed-normalized (today), smoothed-normalized (= the
  group's actual quantization scale, which the paper says is reused), absolute
  thresholds (no per-call min/max), per-block thresholds, and the oracle.
- 06:04 — **first full-protocol c17 result: 4B t32 = 11.3305 @ 33.62 vs parent 11.9886 @
  33.72 → −5.49% PPL at −0.3% cost (iso-cost), 33.9% of the fp16 excess removed**
  (v7: −5.01% but at +3.7% cost). Held-out pre-flight had predicted linear error 0.60×
  parent at 1.009× L. 4B t40 pre-flight: 0.61× at 1.012×. Sweep says prc2 matches the
  parent's linear error at ~0.75× the linear budget (err/par 0.88 @0.8×, 1.11 @0.7×);
  the 0.8× table eval for 4B t32 is queued (p2ppl_4B_t32_c17_s80).
- 06:17 — 4B t40 c17 = 10.8766 @ 41.41 vs parent 11.1440 @ 41.54 → −2.40% at −0.3% cost (v7: −0.60%); 24.3% of fp16 excess removed.
- 07:15 — llama8B t32 c17 = 8.3899 @ 33.60 vs parent 8.6758 @ 34.03 → −3.30% at −1.3% cost (v7: −2.18% at +0.4%); 19.5% of excess removed.
- 07:20 — 4B t48 c17 = 10.5030 @ 49.49 vs parent 10.6491 @ 49.62 → −1.37% at −0.3% cost (v7: −0.37%); 24.2% of excess removed.
- 07:22 — llama8B t40 c17 = 8.0857 @ 41.33 vs parent 8.2093 @ 41.66 → −1.51% at −0.8% cost (v7: +0.68%); 12.4% of excess removed.
- 07:32 — llama8B t48 c17 = 7.8151 @ 49.23 vs parent 7.9200 @ 49.63 → −1.32% at −0.8% cost (v7: −0.24%); 14.8% of excess removed.
- 07:35 — **statistic / granularity analysis (calib3, 4B t32, held-out linear error at
  iso-cost, parent = 1.000):** today's statistic (mn) 0.600; smoothed-operand scale
  (mn_sm) 0.631 (worse); absolute thresholds (raw) 0.571; raw_sm 0.586; per-block
  thresholds 0.593; **per-ROW control calibrated identically (same windows, AWQ, dense
  ladder, budget, solver) 0.848**; unconstrained oracle 0.518. ⇒ the deploy rule already
  captures 83% of the oracle's gain; the statistic is not a lever (≤5%); per-block ≈ 0.
  Granularity itself is the bulk: 0.848 → 0.600 (−29% vs the identically-calibrated
  per-row control); recalibration alone −15%.
- 07:50 — to decompose the PPL gain the same way: env-gated `SC_PRC_ROWSHARED=1` in
  `model/sc_common.per_row_chunk_rungs` (default off ⇒ byte-identical; CPU-verified:
  default output unchanged, rows constant, rung = row max) + `calib3 --row-shared`
  (row-aggregated staircase). Control cells `p3_<m>_t<T>_r17` for 4B t32, llama8B t32,
  4B t48 (chained calib → gate → full eval).
- 08:44 — **iso-PPL: 4B t32 prc2 at 0.8× linear budget (c17_s80) = 11.9836 @ 29.63 vs parent 11.9886 @ 33.72 → same PPL (−0.04%) at −12.1% total SC cost** (−20% linear cycles). Held-out had predicted iso-error at 0.75× linear.
- 09:02 — **decomposition, 4B t32:** per-ROW control calibrated identically (r17, SC_PRC_ROWSHARED=1) = 11.8549 @ 33.31 → recalibration alone −1.12% vs parent; per-(row,chunk) c17 11.3305 @ 33.62 → **granularity alone −4.42%** vs the identically-calibrated per-row control (≈80% of the −5.49% total).
- 09:18 — 30B t32 c17 GATE FAILED (held-out L new/par 1.028 > 1.02; err 0.732). [attribution corrected in the "Corrections" section: dense q/o over-spend + expert under-weighting in the gate aggregate, true-MAC ratio 1.007.] Re-queued as c17e32 (32 expert calls/block-window). Gate NOT relaxed.
- 09:20 — 14B t32 c17 = 9.1672 @ 33.18 vs parent 9.3097 @ 33.56 → −1.53% at −1.1% cost (v7: −1.63% but at +6.6% cost); 21.2% of excess removed.
- 10:33 — 14B t48 c17 = 8.8142 @ 49.10 vs parent 8.9072 @ 49.46 → −1.04% at −0.7% cost (v7: −0.56% @ +2.7%); 34.6% of excess removed.
- 10:33 — 4B t96 c17 = 10.1993 @ 96.05 vs parent 10.2225 @ 95.97 → −0.23% at +0.1% cost (v7: −0.09%); 13.0% of excess removed.
- 10:47 — 4B t64 c17 = 10.4085 @ 64.88 vs parent 10.4749 @ 65.05 → −0.63% at −0.3% cost (v7: −0.40% @ −4.7%); 15.4% of excess removed. 4B complete: −5.49/−2.40/−1.37/−0.63/−0.23 (mean −2.02%).
- 12:17 — llama8B t96 c17 = 7.5987 @ 94.91 vs parent 7.6145 @ 95.29 → −0.21% at −0.4% (v7: +0.04%). llama8B complete: −3.30/−1.51/−1.32/−0.68/−0.21 (mean −1.40%).

## Investigation 2026-09-23 afternoon (workflow wf_0b3b032d-84f; archive README/SUMMARY updated)
- **Why the gain shrinks with budget:** the per-row parent's own held-out linear error
  falls ~1/L² (E_par·L² roughly constant per model); per-group removes a roughly constant
  fraction (4B 34–40% through t64), so the absolute error removed — and its PPL cost —
  shrinks. At t96 ~⅓ of linear MACs sit at the 128 cap in both arms. Attention's share of
  cost and staircase expressiveness are ruled out; "the fp16 gap shrinks" is only partial
  (it breaks at 14B t96 where the parent is already below fp16).
- **Headroom (narrowed after user challenge):** only WITHIN-bucket headroom is shown to be
  small — at the parent's per-(op, layer-quartile) budgets, in summed squared error, the
  rule gets 83% of an unconstrained per-pair oracle (4B/llama8B t32). NOT bounded: the
  per-bucket budget split itself (inherited from the SmoothQuant-era per-row parent, never
  re-optimized for per-group; per-op error ratios 0.44–0.87 and MLP error ≈4× costlier per
  unit suggest it is off), linear↔attention re-split, loss-aware objectives, and the
  protected-slice length. The earlier cross-op refutations were on the per-row allocator.
- **Reporting:** headline per-budget iso-cost ΔPPL (t32–t48 strongest); the 20-cell mean
  is projected ≈ −1.5…−1.7% (30B decides) — don't headline a pooled mean; cycles saved only
  where measured within one cell (SUMMARY §2a).
- **User rule (2026-09-23):** a ~1% cost increase doesn't matter; a 1% PPL reduction matters
  much more ⇒ quote raw ΔPPL; treat ≤1–2% cost drift as iso-cost.
- 13:38 — 14B t40 c17 = 8.9478 vs parent 9.0598 → −1.24% (v7: −0.08%).
- 14:05 — 30B t48 c17 = 8.1159 vs parent 8.1784 → −0.76% (v7: −1.04% @ +5.1% cost).

## Global per-group allocation — Stage 1 (user, 2026-09-23: "per-bucket budget split should not exist; linear and attention -> sure")
- `benchmark/ppl/mp_per_row_chunk_calib5.py` = calib2 measurement + ONE global λ across every
  linear (row,chunk) group of every op and layer (the paper's Phase-1 solve), each
  (op, layer-quartile) key keeping only its deployable monotone staircase (budget floats).
  Currency = paper's σ pricing made separable: e_c(L) = (‖ε_c(L)‖² − ‖ε_c(128)‖²)₊ / ‖y_row‖²
  (row's full FP output energy) ⇒ a row's chunks sum to its Δσ²; comparable across ops/layers.
  Total linear budget = parent's own linear cost on the calibration pairs. Attention = parent (stage 1).
- Selection signal before any test eval: `benchmark/ppl/heldout_nll.py` — paired NLL on 16
  held-out wikitext-2 TRAIN windows (stratified seed 101, calibration windows excluded), one
  model load, tables swapped (parent / c17 / g5). NOT a reported PPL.
- Gate (`run_prc5.sbatch`): g5 held-out NLL < c17 AND < parent, and held-out linear L within
  [−3%,+2%] of parent ⇒ full-protocol test eval (trace tag `<m>_t<T>_prc_p2g5`).
- Queued 14:30: 4B t32/t48, llama8B t32/t48, 14B t32 (`p5_<m>_t<T>_g5`).
- Stage 2 (attention in the same λ) after Stage 1 reads out.
- 14:45 — user: focus t32–t48, t64/t96 may be delayed. Deferred (nice 200): 30B t64 c17 (cancelled after 4 min, requeued), 30B t96 c17, 4B t64 s80, p3ana 4B t64. Stage 1 extended to 4B t40, llama8B t40, 14B t40/t48, 30B t32/t40/t48 (30B: 16 expert calls).
- 15:05 — 14B t64 c17 = 8.7115 vs parent 8.7341 → −0.26% (h=10% mask cell).
- 15:22 — llama8B t32 per-ROW control r17 = 8.5659 vs parent 8.6758 (−1.27%, recalibration) → c17 8.3899 ⇒ granularity alone −2.05% (≈60% of the −3.30% total; 4B t32 was ≈80%).
- 15:40 — user: "enhance the oracle to break the bucket boundary — that is the actual oracle".
  calib5 now reports, on held-out pairs at the parent's cost, in BOTH currencies (rel = σ
  pricing, raw = squared error): within-bucket oracle and GLOBAL oracle (one λ over every
  (row,chunk) of every op/layer). CPU check: global ≤ within at equal cost ✓. The 83% quoted
  earlier was vs the WITHIN-bucket oracle only. Caveat: an oracle is only as real as its
  currency — a loss-aware (Fisher-weighted) currency is being designed (workflow
  wf_e60bef66-56c) together with an adversarial review of calib5 + the held-out NLL gate.
  4B t32 / llama8B t32 g5 started before this change → will be re-run for the oracle numbers.
- 15:37 — 14B t96 c17 = 8.6102 vs parent 8.6150 → −0.06% (both below fp16; within noise).
- 15:45 — **Stage 1 (global λ, σ currency) 4B t32 LOST — gate stopped it.** Held-out TRAIN NLL
  (16 windows, paired): parent PPL 11.879; c17 11.116 (−6.43%, z −12.4); g5 11.263 (−5.19%);
  g5 vs c17 +1.3% (dNLL +0.0132 ± 0.0060). The held-out signal tracks test (c17 test −5.49%).
  g5 moved budget gate 20→15, q 21→18, up 18→22, v 39→53: lower σ-error (0.606 vs 0.691 within)
  but higher loss ⇒ the σ (relative-error) currency misprices ACROSS operators. Removing the
  bucket boundary needs a loss-aware currency (Fisher; design in progress) or a measured one.
- 15:58 — **Stage 1 llama8B t32: global σ table WINS on held-out** (g5 −4.05% vs parent, c17 −3.36%;
  g5 vs c17 −0.72%, z −2.4) → test eval running. Held-out tracks test well (c17 held-out −3.36% vs
  test −3.30%; 4B −6.43% vs −5.49%). ⇒ σ currency helps llama8B, hurts 4B: model-dependent sign.
- 16:20 — review workflow (wf_e60bef66-56c): math correct; majors: (i) within vs global confounded
  currency (c17 raw, g5 σ) → add within-σ; (ii) σ over-weights MoE expert rows ~top_k (8×);
  (iii) gate ignored SE and allowed +2% spend; (iv) run_prc5 not fail-closed. Fisher design adopted.
- 16:40 — `mp_per_row_chunk_calib6.py`: straight-through gradient pass on the deployed SC trajectory
  (SCLinear + attention SC/INT matmuls), per-row per-output g² for the sampled rows (MC-label Fisher
  'fis' + true-label 'fis_emp'), Fisher-weighted error curves from the same SC measurements; emits
  within/global × σ(MoE-corrected)/Fisher tables; held-out scoring in every currency + Fisher-
  PREDICTED ΔPPL (validate vs measured held-out NLL); oracles (within/global, ladder/all lengths).
  `run_prc6.sbatch` (fail-closed): calib6 → heldout_nll(parent,c17,wrel,grel,wfis,gfis; per-table
  cost) → best candidate must beat c17 by ≥2 paired SE and the parent → test eval.
  heldout_nll now records realized tracker cost per table. Cancelled the pending σ-only p5 jobs.
  Queued p6 for 4B t32, llama8B t32, 4B t48.
- 16:53 — llama8B t32 global-σ (g5) TEST = 8.3710 vs c17 8.3899 (−0.23%) / parent 8.6758 (−3.51%). Held-out had predicted −0.72% vs c17. σ-global: small gain on llama8B, loss on 4B (held-out +1.3%).
- 17:27 — llama8B t32 c17_s80 = 8.8363 @ ~28.9 vs parent 8.6758 (+1.85% at −15% cost) → brackets the iso-PPL point with c17 (≈10% SC cycles saved).
- 17:45 — **Fisher calib (calib6) 4B t32, held-out (2 windows):** Fisher-PREDICTED ΔPPL vs parent —
  c17: true-label Fisher −6.15% (MC −3.30%) vs measured held-out NLL −6.43% (test −5.49%);
  g5 (global σ): −5.37% (MC −3.14%) vs measured −5.19%. ⇒ the TRUE-LABEL (empirical) Fisher
  predicts measured held-out NLL well and ranks c17 > g5 correctly; the MC-label Fisher
  under-predicts ~2×. gfis (global, MC currency) predicted −6.41% (true-label) at L/par 1.017.
  ORACLES (held-out err / parent at parent cost): raw within 0.518 / GLOBAL 0.283; σ 0.608 / 0.537;
  Fisher-MC 0.251 / 0.221 ⇒ predicted ΔPPL within −7.02% / GLOBAL −7.30% (MC units; likely ~2×
  understated). ⇒ in a loss-aware currency, crossing the bucket boundary is worth only ~0.3% in
  the oracle; the large gap is oracle (per-pair, sees per-token loss sensitivity) vs any
  deployable absmax staircase (−3.3…−4.1% in the same MC units).
- 17:55 — added the true-label Fisher as an allocation currency (tables w/gfis_emp); llama8B t32
  and 4B t48 calib6 jobs resubmitted with currencies rel,fis,fis_emp (first submission had the
  comma-list truncated by sbatch --export; cancelled before start).
- 18:30 — **4B t32 held-out NLL (16 windows, paired), all at the same cost (tracker 33.45–33.56):**
  c17 (within, raw) −6.43% vs parent; wrel (within σ) −5.90%; grel (GLOBAL σ) −5.19%;
  wfis (within Fisher-MC) −6.31%; **gfis (GLOBAL Fisher-MC) −7.85% — beats c17 by −1.53%
  (dNLL −0.0153 ± 0.0049, z −3.1)**; gfis vs wfis −1.66% (z −4.4) ⇒ removing the bucket
  boundary PAYS once the currency is loss-aware (it lost with σ). Gate passed → full test eval
  of gfis running (trace tag 4B_t32_prc_p2c6gfis).
- 20:25 — **4B t32 GLOBAL Fisher (gfis) TEST = 11.1891 @ 33.67 vs parent 11.9886 @ 33.72 (−6.67%) and c17 11.3305 (−1.25%).** llama8B t32 held-out: gfis −1.8% vs c17 (z −6.5), gfis_emp −1.0% (MC-label beats true-label for allocation); gate passed → test running. 4B t48 held-out running. Queued global-Fisher for 4B t40, llama8B t40/t48, 14B t32/t40/t48 (currencies fis,fis_emp); 30B needs block-sequential backward (later).
- 20:45 — **llama8B t32 GLOBAL Fisher (gfis) TEST = 8.2795 @ 33.74 vs parent 8.6758 @ 34.03 (−4.57%) and c17 8.3899 (−1.32%).** 4B t48 c17_s80 = 10.6761 @ 42.6 vs parent 10.6491 (+0.25% at −14% cost; ≈12% SC cycles saved at parent PPL, bracketed).
- 21:10 — user: global Fisher is the headline allocation (paper §3 pricing will be updated);
  headline numbers first, ablations last. Added `--grad-mode block` to calib6 (exact block-
  sequential backward: no-grad forward stores each decoder layer's input; loss grads at the final
  hidden state; back-propagate one layer at a time with the STE wrappers; MC labels now seeded per
  window). Validation job `blkcheck_4B_t32` (block vs full on 4B t32, calibration only). 30B t32/
  t40/t48 global-Fisher jobs queued in block mode (16 expert calls per block-window).
  Headline queue: 4B t40/t48, llama8B t40/t48, 14B t32/t40/t48 (full mode), 30B t32/t40/t48 (block).
- 21:08 — 4B t48 global Fisher held-out: gfis −0.75% vs c17 (z −2.7), −2.0% vs parent → test eval running; gfis_emp +0.5% vs c17 (MC-label Fisher is consistently the better allocation currency).
- 21:10 — 30B t32 c17e32 (32 expert calls; corrected MoE gate weighting: 1.0106 pass) TEST = 8.9599 @ 33.96 vs parent 9.4647 @ 34.12 → −5.33%. c17 now complete on all t32–t48 cells.
- 21:09 — llama8B t48 c17iso_s90 = 7.8937 @ ~45.5 vs parent 7.9200 @ 49.63 → better PPL at −8.3% cost (≥8% SC cycles saved, measured).
- 21:29 — llama8B t40 calib6 pre-flight (emp-Fisher units): c17 −1.96% (test −1.51%), gfis −2.40%, gfis_emp −2.47%; ORACLE emp-Fisher within −6.79% / GLOBAL −6.89% ⇒ bucket boundary ~0.1% even in the oracle; large gap deployable-vs-oracle (oracle sees per-token loss sensitivity).
- 21:40 — user to bed: 'after all experiments are done, update response.md; leave non-LLM part to later'. Queued global-Fisher t64/t96 for all 4 models at nice 100 (14B/30B block mode). Note: 30B t64/t96 c17 arms are still pending at nice 200.
- 22:01 — 4B t40 global Fisher held-out: gfis −0.05% vs c17 (z −0.09), −2.2% vs parent ⇒ GATE FAILED (no gain over c17 here). Decision: headline = ONE algorithm (global Fisher) on every cell ⇒ run the gfis TEST eval even when the gate fails (p6ev_<m>_t<T>_c6gfis); gate kept as a held-out report. Launched 4B t40.
- 22:12 — llama8B t40 held-out: gfis −0.74% vs c17 (z −2.2), gfis_emp −0.94% (z −2.9) → job evaluates gfis_emp; llama8B t48: gfis −0.32% (z −1.3) → gate failed. Launched p6ev gfis test evals for llama8B t40 and t48 (one algorithm everywhere).
- 22:56 — 14B t48 c17iso_s85 = 8.8934 @ ~43.3 vs parent 8.9072 @ 49.46 → better PPL at −12.4% cost (≥12% SC cycles saved, measured).
- 22:56 — **4B t48 global Fisher TEST = 10.4665 @ 49.66 vs parent 10.6491 (−1.71%), c17 10.5030 (−0.35%).**
- 2026-09-24 13:00 — overnight (monitoring lapsed after 23:17; no crashes, all FAILED states are gate
  decisions except 30B): c17 complete 20/20 (30B t64 −0.71%, t96 −0.31%). Global-Fisher (gfis) TEST:
  4B t40 −2.41% (= c17), llama8B t40 −2.23% (gfis_emp; −0.73% vs c17), llama8B t48 −1.67% (−0.35% vs
  c17), 14B t40 −1.58% (gfis_emp; −0.35% vs c17). Held-out gates failed (gfis ≈ c17) at 14B t32/t48 and
  all t64/t96 cells. **30B: Fisher currency SKIPPED — no gradients captured for the MoE expert
  projections (12 expert keys) in block mode ⇒ 30B gfis not yet built (bug under investigation).**
  block-mode check (4B t32): MC-label 'fis' matches full mode (pred −3.97% vs −4.06%); TRUE-label
  'fis_emp' is broken in block mode (absurd predictions) ⇒ never use block-mode fis_emp.
  Parent post-move re-run = 11.9886 @33.72 (bit-identical ⇒ env verified). Cost-matched per-row
  control r17m = 11.8429 @33.52 ⇒ granularity −4.33% at matched cost (4B t32). s80 4B t64: 13.2% cycles
  saved at parent PPL. Queued gfis TEST evals (headline = one algorithm everywhere): 14B t32/t40/t48
  (nice 0), 4B/llama8B/14B t64/t96 (nice 50).
- 13:20 — 30B bug diagnosed: STE forward y_fp+(y_sc−y_fp) is not bit-identical to y_sc in bf16 ⇒ MoE router top-k flips ⇒ expert row counts differ between gradient and measurement passes ⇒ all expert keys dropped. Fixed with exact custom-autograd STE (forward = SC value bit-exact; backward = FP gradient; CPU-verified). run_prc6 now ALWAYS test-evaluates gfis (gate reported, not blocking). Relaunched 30B t32/t40/t48 (block, fis, 16 experts).
- 13:45 — response.md drafted (LLM parts; user outline kept at top). User: after round-2 results (hopefully strictly better), update paper and response.md again. ROUND 2 launched: calib7 (global Fisher over linears AND attention in one λ; attention = parent per-row rule with re-solved thresholds, escape rows fixed) → held-out (parent, c17, gfis, gfisla) → test eval of gfisla. 12 cells t32–t48 (30B block mode, 16 experts); 4B t32 at nice 0 as the canary. Cancelled the pending 30B p6 jobs (p7 also emits gfis).
- 14:22 — ROUND-2 canary 4B t32: attention replay of parent thresholds agreement 1.00000 (120 calls) ⇒ emitted semantics correct. JOINT λ: linear L 24.12→25.76, attention L 83.49→72.90 (av l0 → 33; qk ~83–91). Pred dPPL (MC-fis, held-out): gfis −3.97% @1.010, gfisla −4.14% @1.015. Held-out NLL running.
- 15:45 — 14B t32 global Fisher TEST = 9.1041 @ 33.12 vs parent 9.3097 (−2.21%), c17 9.1672 (−0.69%).
- 16:43 — ROUND-2 4B t32: held-out gfisla −7.95% vs parent (gfis −7.19%; gfisla vs gfis −0.82%, z −1.8; vs c17 −1.64%, z −3.4); TEST gfisla = 11.1953 @ 33.57 vs round-1 gfis 11.1891 @ 33.73 ⇒ TIE (+0.06%), −6.62% vs parent. Attention↔linear re-split: neutral on 4B t32 at test.
- 17:00 — user: let round 2 finish; if not good enough, run round 3; use all GPUs (asked whether to use other accounts — pending answer; staying on the lab allocation). Round-3 prep: calib7 --dump (per-pair mn, cidx, rpos, blk, rowmax, raw/fis/fis_emp curves) → job dump_4B_t32 to analyze what the Fisher oracle keys on (the deployable-vs-oracle gap is the largest remaining headroom).
- 17:28 — ROUND-2 4B t48: joint λ barely moves (attn 98.1→95.5, lin 40.8→41.0); held-out gfisla +0.17% vs c17 (gfis −0.60%) ⇒ round 2 neutral/negative on 4B; test eval running for completeness.
- 18:20 — 14B t48 global Fisher TEST = 8.7894 @ 49.07 vs parent 8.9072 (−1.32%), c17 8.8142 (−0.28%).
- 19:50 — ROUND-2 4B t48 TEST gfisla = 10.5532 vs parent 10.6491 (−0.90%) but vs round-1 gfis 10.4665 (+0.83%) and c17 10.5030 (+0.48%) ⇒ round 2 WORSE on 4B t48 (tie at t32): Fisher appears to underprice attention error (softmax nonlinearity). 4B t64 round-1 gfis TEST = 10.3645 (−1.05% vs parent, −0.42% vs c17). Queue reshuffle: kept round-2 llama8B t32/t40/t48 (decisive); cancelled round-2 4B t40 and 14B t32/t40/t48; 30B resubmitted as calib7 with P6_EVAL=gfis (headline linears-only global Fisher).
- 20:44 — 14B t40 global Fisher TEST = 8.9029 vs parent 9.0598 (−1.73%), c17 8.9478 (−0.50%). Headline (round-1 gfis) complete for 4B/8B/14B at 6/6.32/6.58 bits.
- 20:51 — 4B t96 global Fisher TEST = 10.1995 (−0.22% vs parent; = c17 10.1993).

## 2026-09-24 evening — round 2 closed, round 3 launched

**Round 2 (calib7, attention in the joint Fisher λ, `gfisla`) does not beat round 1 (`gfis`).**
| cell | round-1 test | round-2 test | held-out gfisla − gfis (dNLL) | joint λ moved attention L |
|---|---|---|---|---|
| 4B t32 | 11.1891 | 11.1953 | −0.00825 ± 0.00448 | 83.5 → 72.9 |
| 4B t48 | 10.4665 | 10.5532 | +0.00772 ± 0.00388 | 98.1 → 95.5 |
| llama8B t32 | 8.28 (c6gfis) | eval running | +0.00275 ± 0.00298 | 96.6 → 94.3 |
The Fisher-predicted dPPL rates round 2 equal to round 1 (e.g. llama8B −3.19 vs −3.18%), but it
loses on measured NLL: the diagonal Fisher under-prices attention error. Attention ladders top out
at 96 (halved; 128 only via the escape gate), so the joint λ can only CUT attention. Response
rule (user 2026-09-24): "final" = per-cell better of round 1 / round 2; today round 1 everywhere.

**Round 3 = calib8 (`benchmark/ppl/mp_per_row_chunk_calib8.py`, `run_prc8.sbatch`).** Linears
only (attention = parent), one global Fisher λ at the parent's cost, 12 calibration windows
(round 1: 6). Emits `<m>_t<T>_c8_gfis` (quartile keys) and `_c8_gfisblk` (one threshold vector
per (op, BLOCK); runtime rule unchanged, the table carries `per_row_chunk.layer_buckets =
n_blocks` — new default-off field in `scmp_kernels/mp/config.py`, attention buckets keep the
table's quartiles). Also prints Fisher-predicted held-out dPPL for static chunk-class keys
(q_cc2/4, blk_cc2/4; analysis only, not deployable without a chunk→class lookup). Held-out: 24
windows excluding BOTH calibration sets (`heldout_nll.py --calib-n 8,14`); test eval of the best
candidate only if it beats the round-1 headline on held-out. CPU-verified on synthetic data
(finer keys lower Fisher error at iso-cost; per-block lookup round-trips; attention unaffected).
Jobs: p8_4B_t32 (with per-pair dump; replaces the cancelled dump_4B_t32), p8_llama8B_t32,
p8_14B_t32, p8_4B_t48, p8_llama8B_t48. Queue order: p8 t32 > p7_30B_t32 > p8 rest > 30B t40/48 >
round-2 llama8B t40/48 > t64/t96 evals.

### 2026-09-24 late — round 2 final: 0/3; round 3 diagnosis = heavy-tailed Fisher
- llama8B t32 round-2 test **8.3108** vs round 1 8.2795 ⇒ round 2 loses 3/3 on test; remaining
  round-2 cells (llama8B t40/t48) cancelled (lab has ONE free GPU: 8/10 held by interactive jobs).
- **calib8 on 4B t32 with 12 windows produced a BAD gfis** (Fisher-predicted held-out +17.66%;
  global λ moved gate/up 20→14, o_proj 33→41 — opposite of round 1). Dump diagnosis
  (`4B_t32_c8.npz`): calibration window 5 carries 80–99% of every operator's Fisher mass; ~20
  token rows at positions 875–951 carry 83% of ALL mass; top 1% of rows carry 99.2%. The span is
  a wikitext section break ("... years old . \n\n\n\n\n = = Rediscovery of the crypt = = \n\n\n\n\n
  It had ...") — delimiter tokens acting as secondary attention sinks. The single-sample MC Fisher
  per token is extremely heavy-tailed; round 1's 6-window draw happened to have no such spike.
- Fix = ROBUST pricing (pure calibration, same runtime rule): `benchmark/ppl/prc_offline_solve.py`
  re-solves the global λ from the dump with `fisclipC` (each token row's Fisher mass capped at C×
  the median row mass of its (op, block)) or `kraw` (raw error × per-(op, block) sensitivity =
  MEDIAN over windows of Σfis/Σraw), keyed per quartile (q) or per block (blk). 4B t32
  Fisher-predicted held-out dPPL: fis:q +17.66%, fisclip10:q −5.37, fisclip100:q −5.56, kraw:q
  −5.54, fisclip10:blk −5.66, kraw:blk −5.94 (round-1 c6gfis on the same 2 windows −6.14 — within
  noise); robust variants recover round 1's op split (o_proj ≈ 25–27, up ≈ 23).
- Jobs: `ps_4B_t32_c8o` (run_prc_select.sbatch: 24 held-out windows excluding both calibration
  sets vs parent/c6gfis/c17 → test eval of the best if it beats c6gfis), `p9_llama8B_t32`
  (run_prc9.sbatch: calib8 dump → offline robust solve → select → eval). Superseded/cancelled:
  p8_{llama8B_t32,14B_t32,4B_t48,llama8B_t48}, dump_4B_t32.
- p8_4B_t32 held-out (24 windows, both calibration sets excluded): parent 12.83, c6gfis 11.88
  (−7.45%), c17 12.12, **c8gfis 17.00 (+32.5% vs parent)**, c8gfisblk 13.96 (+8.8%) — the spiky
  12-window Fisher is catastrophic on measured NLL, confirming the diagnosis. No test eval.
- ps_4B_t32_c8o (24 held-out windows): robust 12-window candidates vs round-1 c6gfis dNLL —
  fisclip10:q +0.0044 (z +1.2), fisclip100:q +0.0039, kraw:q +0.0045, fisclip10:blk +0.0015
  (z +0.4), kraw:blk +0.0011 (z +0.25). **None beats round 1**; robust pricing = a safeguard (it
  turns the +32% spike table into round-1 quality), not a gain. Per-block keys ≈ −0.003 NLL vs
  quartile keys at equal currency. No test eval.
- Next (round 3b): second, RE-LINEARIZED pass — calib8 `--expand-at <round-1 gfis wrapper>`
  (Fisher and activations taken on the round-1 table's own SC trajectory; budgets still the
  parent's), 6 windows = the round-1 window set, then offline fis/robust × q/blk → select.
  Job p9x_4B_t32 (run_prc9.sbatch P9_EXPAND=1 P8_CW=6). Memory request cut to 64G (the lab's
  interactive jobs leave ~130G of the group memory cap; 150G jobs cannot start).
- p9x_4B_t32 (second, re-linearized pass on the round-1 trajectory, 6 round-1 windows): vs c6gfis
  fis:q +0.0306 (z +7.5), fis:blk +0.0254, fisclip100:q +0.0044, fisclip100:blk +0.0062, kraw:blk
  +0.0057 ⇒ re-linearization HURTS; round 1 stays best. Round 3 (robust / per-block / Phase-2
  re-linearization) is negative on 4B t32.

### 2026-09-25 03:30 — round 3c: the ATTENTION ladder ceiling (iso-cost)
The submitted per-row search compressed several tight-budget ladders to a top rung < 128, and
attention is pinned on it: llama8B t32 (top 96) and t40 (98), 14B t32 (96) put 98–100% of
qk/av rows on the top rung (thresholds all 0.0); 4B t32 (97) 62%. The per-group linears reach
128; attention cannot (only μ+2τ escape rows do). Round 2's measured loss from CUTTING attention
2.4% on llama8B (+0.38% PPL) says attention precision is valuable there. Test: raise attention's
top rung to 112 / 128 (table + wrapper stoc_len_levels; thresholds untouched) and solve the linears
(round-1 Fisher currency, 6 round-1 windows) at s × the parent's per-group budget with s =
1 − Δcost_att / C_lin (from the round-1 trace, protected slices excluded) ⇒ iso total cost:
| cell | att MAC share | top → 112: +cost, s | top → 128: +cost, s |
|---|---|---|---|
| 4B t32 | 0.131 (62% at top 97) | +3.6%, 0.9399 | +7.5%, 0.8759 |
| llama8B t32 | 0.075 (98%) | +3.5%, 0.9519 | +7.0%, 0.9038 |
| llama8B t40 | 0.073 (98%, top 98) | +2.4%, 0.9683 | +5.1%, 0.9321 |
| 14B t32 | 0.055 (99%) | +2.6%, 0.9658 | +5.2%, 0.9316 |
Jobs p9a_{llama8B_t32,14B_t32,llama8B_t40,4B_t32} (run_prc9.sbatch P9_SFX=c8w6 P8_CW=6; variants
fis:q [round-1 reproduction], fis:q:s=…:att=112, fis:q:s=…:att=128) → held-out vs c6gfis → test eval
of a winner. Solver: prc_offline_solve.py `s=`/`att=` variant fields (tested on a synthetic dump).
- **30B t32 round 2 WINS held-out** (p7_30B_t32_c7, 24 windows): parent 9.151, c17 8.606, gfis 8.373
  (−8.5%), **gfisla 8.233 (−10.0%)**; gfisla − gfis = −0.0169 ± 0.0052 (z −3.3). On 30B attention is
  NOT pinned (av buckets at L 28–60, qk 86–92); the joint λ moves attention 61.8 → 62.9 (av down,
  qk up) and linears 23.9 → 23.5. The p7 job test-evaluates gfis (P6_EVAL=gfis); round-2 test eval
  submitted separately (p7ev_30B_t32_c7gfisla, job 61867949) for the per-cell better-of rule.
  Same check pending for 30B t40/t48 when their held-out lands.
- **30B t40 round 2 WINS held-out by more**: gfis 7.993 vs **gfisla 7.760** (−0.0297 ± 0.0040, z −7.5);
  the joint λ moves attention 64.7 → 74.9 and linears 34.8 → 31.7 — 30B's parent under-funds
  attention (not pinned: its ladder top is 96 but av sits far below it). Round-2 test eval
  submitted (p7ev_30B_t40_c7gfisla).
- **p9a_llama8B_t32 (attention ceiling, iso-cost): NEUTRAL.** vs round-1 c6gfis (24 held-out windows):
  round-1 reproduction (c8w6o_fis) −0.0027 (z −1.3), att112 + linears 0.9519× −0.0027 (z −0.95),
  att128 + linears 0.9038× −0.0015 (z −0.49). The attention gain from a higher ceiling is paid back
  by the linear cut at iso-cost. Test eval (of the reproduction) cancelled; selection rule tightened to
  z < −1.5 before any test eval.
- Round-2 lever map: the joint λ can only CUT attention where the parent pins every attention row
  on a top rung (dense llama8B t32–t64, 14B t32) — there round 2 loses; it can move budget INTO
  attention where rows sit below the top (30B everywhere; 14B t40 top 86 / t48 top 128 with
  nonzero thresholds). Launched round 2 for 14B t40/t48 (`p7_14B_t{40,48}_c7`,
  P7_ONLY_IF_BETTER=1: test-eval gfisla only if it beats gfis on held-out at z < −1.5).
  Cancelled p9a_llama8B_t40 (pinned like llama8B t32, whose ceiling test was neutral).
- **30B t48 round 2 also WINS held-out**: gfis 7.713 vs **gfisla 7.544** (−0.0221 ± 0.0034, z −6.5);
  attention 70.2 → 86.4, linears 43.1 → 37.6. ⇒ 30B round 2 wins held-out at t32/t40/t48.
- The first two standalone gfisla evals (61867949, 61868826) died in 5 s: ModuleNotFoundError
  scmp_kernels (standalone run_prc_ppl.sbatch did not export PYTHONPATH). run_prc_ppl.sbatch now
  exports it itself. Resubmitted: p7ev_30B_t{32,40,48}_c7gfisla (61869786/87/88).
- 14B t40 round 2: joint λ moves attention 84.4 → 82.7 (a cut, like the other dense cells); held-out
  gfisla − gfis +0.0015 ± 0.0033 ⇒ no test eval. Round 2 helps where the joint λ moves budget INTO
  attention (30B), not where it cuts attention (every dense cell so far).
- p9a_4B_t32 (attention ceiling): vs c6gfis — reproduction +0.0056 (z +2.1), att112 + linears
  0.9399× +0.0087, att128 + 0.8759× +0.0178 (z +6.0) ⇒ WORSE. With llama8B t32 neutral, round 3c
  (attention ceiling at iso-cost) is closed; p9a_14B_t32 cancelled.
- **30B t32 round 2 on TEST: 8.5812** (vs round 1 8.7163, −1.55%; held-out predicted −1.7%) ⇒ final
  30B t32 = round 2, −9.33% vs parent 9.4647. 30B t40/t48 round-2 evals running.
- **Round-3 verdict:** nothing beats round 1 on the dense models (robust Fisher = tie, per-block ≈
  tie, re-linearized 2nd pass worse, attention ceiling neutral/worse). The only lever beyond round 1
  is round 2 on 30B (attention under-funded there, not pinned).
- **30B t40 round 2 on TEST: 8.0880** (vs round 1 8.3210, −2.80%; held-out −2.9%) ⇒ final 30B t40 = round 2,
  −5.18% vs parent 8.5294.

### 2026-09-25 12:45 — 14B t64/t96 switched to h=20% (user), 30B t64/t96 finals
- 14B t48 round 2: joint λ moved attention UP (111.8 → 115.8) yet held-out gfisla − gfis +0.0055
  (z +2.0) ⇒ no test eval; round 1 final.
- User: "for 14B switch it to h=20%". The paper's 14B 7/7.58-bit MP cells are NOT the mp_best
  configs with a bigger mask; they are separate configs: t64 = `awq14bt64_55367988` (8.7758; wrapper
  results/awq_14bt64_v20_20pct/14B_t64_v20_ungated_wrapper.json, levels 89…58, no gate; mask
  _hpca_mp_v9_hybdose_all…/14B_mp_avg32_v9_…top0p20), t96 = `mba_14B_t96_54972413` (8.6418; wrapper
  mp_calib_mp_avg96_hyb20_20260724/…hyb0.20_wrapper.json; mask _hpca_mp_avg96_hyb20…top0p20).
  Parent dirs built at `/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2/parents_h20/14B/target{64,96}`
  (wrapper/table/hybrid_config + SOURCE.json; t64 parent_trace.json = that run's AWQ trace; the t96
  trace file was later OVERWRITTEN by an h=10% run (header ppl 8.6150, mp_best wrapper) ⇒ no t96
  parent trace; the parent is re-evaluated).
- Scripts: `KB_PARENT` (run_prc_ppl/run_prc6/run_prc2_calib), `KB_PTRACE` + optional parent trace
  (run_prc2_calib), `P6_C17` (run_prc6) — defaults unchanged.
- Jobs: p2_14B_t{64,96}_c17h20 (per-group c17 recipe: rows 64, 6+2 windows, emit 0.8, gate→eval),
  p6_14B_t{64,96}_c6h20 (round 1: fis,fis_emp, full grad → held-out → eval gfis),
  prcppl_14B_t96_parent_h20 (parent re-eval in this env), p7_30B_t{64,96}_c7 (fis, block grad,
  16 experts; eval gfis; round-2 eval to follow if held-out prefers it). The h=10% 14B t96 final eval
  was cancelled; the h=10% 14B t64 final (8.6763) is superseded for the paper table.
- **30B t48 round 2 on TEST: 7.8571** (vs round 1 8.0414, −2.29%; held-out −2.2%) ⇒ final 30B t48 = round 2,
  −3.93% vs parent 8.1784. 30B round 2 wins test at 3/3 tight budgets; held-out predicted each.

### 2026-09-26 — complete round 2, then write the first rebuttal response

User decisions: final numbers are **best(all comparable allocation rounds)** on full-test
PPL, retaining provenance and realized cost. Held-out scores remain diagnostics and do not
suppress test evaluation. Limit our concurrent GPU occupation to **4**. Work must improve
allocation / mixed precision only: keep RNG, QK behavior, per-cell INT masks and front end
fixed. After round 2, prepare the first rebuttal response before considering another round 3.

Submitted Slurm array **61982983**, `0-13%4`, one GPU per task, 120G RAM, four CPUs, 18h/task.
At submission there were no other user jobs. Slurm confirmed `ArrayTaskThrottle=4` and
`gres/gpu=1`; all tasks were initially pending for priority.

- Launcher: `run_prc7_finish_20260926.sbatch`; exact task mapping:
  `prc7_finish_20260926.tsv`; submission/protocol record and script hashes:
  `prc7_finish_20260926_submission.json`.
- Ten calibration → 16-window held-out → full-test jobs: 4B t40/t64/t96;
  llama8B t40/t48/t64/t96; 14B t32/t64/t96.
- Four evaluation-only jobs reuse existing round-2 tables: 14B t40/t48, 30B t64/t96.
- Six round-2 tests already complete: 4B t32/t48, llama8B t32, 30B t32/t40/t48.
- 14B t64/t96 use `parents_h20/14B/target{64,96}`, c17h20 controls and distinct c7h20
  output tags. `run_prc7.sbatch` now honors and forwards KB_PARENT/P6_C17 correctly.
- All new jobs set `P6_EVAL=gfisla`, `P7_ONLY_IF_BETTER=0`, full WikiText-2,
  ctx/stride=2048, batch=1. No QK rebalance or RNG variant is enabled.
- Per-cell logs retain the builder's `p7_..._<job>.out` / `p7ev_..._<job>.out` naming;
  scheduler logs are `p7finish_20260926_61982983_<task>.out` in the same `_kbands` directory.

Status audited before submission: all 20 c17 and all 20 linear-only Fisher tests complete.
Late results: 14B h20 t64 c17 8.7110 / Fisher 8.6993; t96 parent 8.6418 / c17 8.6441 /
Fisher 8.6455. 30B t64/t96 linear-only Fisher 7.6281/7.5336. Their round-2 held-out scores
were worse, but both now get test evaluation under the user's exhaustive best-round rule.

**Before refreshing the archive/response:** `build_prc2_archive.py` still needs h20-aware
parent/c17/round selection (`parent_h20`, `c17h20`, `c6h20gfis`, `c7h20gfisla`); a naive
rebuild keeps stale h10 comparisons. Also replace its held-out-gated selection wording with
the user's best-across-rounds rule. Do not combine different-budget ablations or per-row
controls with the comparable per-group rounds. SUMMARY/manifest/response currently predate
the late results. The paper's Phase-2 draft is local; runtime and result tables remain to edit.

### 2026-09-26 17:15 — pending queue diagnosis / reservation correction

Array 61982983 had not started (all14 pending Priority, no logs) at17:14.
The submission omitted `--reservation=rtx6000_arph_nodes`, which the successful
preceding jobs used. Added ReservationName to the existing pending array with
`scontrol update` (no cancellation/resubmission); verified all tasks have it and
ArrayTaskThrottle remains4. Added the directive to the launcher for future use.
At17:15, nb1 had10 allocated GPUs: vballoli4 (61756857), ruijieg6
(61728406/61649198/61727105,2 each). Our array was still pending immediately
after correction; no GPU progress claimed.


### 2026-09-26 23:21 EDT — authorized round-3 local pilot queued

User explicitly authorized code and overnight queueing after the round-3 investigation.
Array **62032412** (0–2%3), dependency **afterany:61982983**, waits for the entire
round-2 array to terminate. Round 2 still occupies four GPUs; pilot then uses at
most three. Account nbleier_owned1, reservation rtx6000_arph_nodes, 1 GPU / 4 CPUs /
120G / 24h per cell. Slurm dependency and resources verified after submission.

Cells: 4B t32, llama8B t32, 30B t40; exact best comparable archived incumbents
c6gfis / c6gfis / c7gfisla. Pure threshold-allocation changes, fixed ladders,
escape/protected channels, masks, AWQ caches, RNG, and QK operands. Up to eight
2%-cycle bidirectional transfers per cell; exact-trace cost matching, six TRAIN
search windows, sixteen disjoint confirmation windows, and one automatic full-test
of the best new candidate if confirmation cost remains within 1%. Confirmation
quality is recorded independently; final numbers remain best(all comparable rounds).

All 23 CPU tests and three launch dry runs passed. Runtime GPU no-op/replay checks
run first inside each job. No paper files edited or pushed. Static chunk classes
remain a backup proposal, not part of this pilot. Prepare the first rebuttal
response from completed round-2 evidence as previously requested.

Details: PRC_LOCAL_20260926.md, prc_local_20260926.json,
prc_local_20260926_submission.json. Outputs under Turbo
hpca/kbands/prc_local_20260926/<model>_t<target>/.
