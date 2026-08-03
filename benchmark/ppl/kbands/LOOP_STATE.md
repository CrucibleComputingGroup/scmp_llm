# ⇢ START HERE — Phase 3 / SC-MP loop state

**THIS FILE IS THE ENTRY POINT.** Read it top to bottom before doing anything.
Path: `scmp_llm/benchmark/ppl/kbands/LOOP_STATE.md`

## ⚡ GPU UTILIZATION RULE (owned hardware — idle is the biggest waste)

**Keep at least 4 GPUs busy at ALL times.** Check `squeue -u allenjin -h | wc -l`
at the START and END of every loop iteration. If the count is below 4:

    bash benchmark/ppl/kbands/fill_queue.sh          # KB_MIN_JOBS defaults to 4

That script builds any missing child bundles and submits from a standing
backlog of genuinely open questions, newest-value first. It is idempotent —
safe to run every iteration.

Rules that go with it:
* **Never end an iteration with the queue below 4.** Waiting on results is not
  a reason to idle; there is always backlog.
* **Do not submit filler.** Every cell must answer an open question. Burning
  GPU-hours on a known-outcome cell costs the same as idling and also pollutes
  the ledger. If the backlog is genuinely exhausted, ADD to it (below) rather
  than repeating a settled cell.
* A cell is ~1.2 h (4B/llama8B), ~2.7 h (14B), ~4.6 h (30B). Prefer launching
  long 30B cells FIRST when the queue is empty so they overlap everything else.
* The 10-concurrent cap is lab-shared; 4 is the floor we own, not the ceiling.
  Fill above 4 when the backlog supports it.

## 📋 STANDING BACKLOG (keep this stocked; add as questions arise)
1. per-(row,chunk) dispatch — the −44.8% axis, UNBUILT, top priority
2. verify the per-group ceiling on REAL captured activations (all four models)
3. MoE band support — key the band map by (op, block, expert); 30B currently
   CANNOT use bands at all, and it is the model qk helps most
4. what predicts qk benefit? 14B is the lone regression and the closest parent
   to fp16 — test the headroom hypothesis on all four models
5. asymmetric SC (zero-point) + non-pow2 grid — the representation fix; the
   error budget says it is the only route to −12% at t32
6. rewrite the outlier-clip diagnostic (returned −255%; the clip destroyed the
   signal, so that row means nothing)

---

# Phase 3 (per-group / K-band MP) — autonomous loop state

**Mandate (user, 2026-08-02):** iterate overnight/through the week to find a
good algorithm for **per-group mixed precision under the same SC budget**.
User checks back in ~1 week. GPU capacity: 10 concurrent (lab-shared).

> **READ THIS FIRST each loop iteration.** Context gets summarized; this file
> is the durable memory. Update the LEDGER and NEXT after every wave.

---

## The idea

SC quantization is already per-**group** (128-wide chunks along the contraction
dim K, each with its own scale) but stream-length allocation is per-**row** —
every chunk in a row shares one `stoc_len`. Phase 3 spends that unused axis.

Row dispatch is UNCHANGED (same metric, thresholds, escape gate, same rung
index `k` per row). What changes: the residual contraction axis is partitioned
into bands of whole chunks, and band `b` runs rung `k` at its own length
`L[b][k]`.

**Why it cannot lose:** the per-row parent is inside the space at
`L[b][k] = L_parent[k]`. **Iso-compute is an identity, not a tolerance:**

    Σ_b (w_b / R) · L[b][k]  ==  L_parent[k]        for every rung k

MACs are linear in the contraction dim, so a column fraction IS the MAC
fraction; the identity holds for any input, any rung occupancy, any escape-gate
firing. Enforced at table **load**, so a cell that starts is already in budget.

## Hard invariants (violating any of these invalidates a cell)

1. **Bands are WHOLE 128-chunks.** Proven on GPU (job 55934465): chunk-aligned
   splits differ by `rel ~1e-7` (fp32 accumulation regrouping only); random
   channel splits differ by `rel ~1.3-1.9e-1` — six orders larger. The scramble
   is indexed by WITHIN-chunk offset, so relocating a whole chunk is free and
   relocating arbitrary channels is not.
2. **Never exceed the halved cap 128** (`2**(sc_prec-1)`). Above it the stream
   wraps; results are meaningless, not merely worse. Enforced in the loader and
   clamped in the allocator grid.
3. **Every band ≥ 2 chunks.** A band ≤ `chunk_d` wide falls off the chunked
   kernel path (`D > chunk_d` gate) onto a different quantization impl.
4. **Band widths constant per operator** across blocks — ladders are per
   (op, layer-bucket) but membership is per (op, block), so drifting widths
   would make the identity unsatisfiable. The ragged tail chunk is pinned to
   the last band for this reason.
5. **Full protocol only** (`PPL_MAX_TOKENS=0`, ctx 2048, B=1). No truncated
   evals, ever — smokes included.
6. **Model scope is exactly** 4B / llama8B / 14B / 30B. Never swap a model
   inside a comparison.
7. **Attention is out of scope.** `qk` contracts over head_dim=128 = one
   chunk (nothing to split, confirmed in every trace). `av` contracts over
   seq len but runs unchunked, so banding it changes baseline numerics.
   Attention keeps the parent allocation byte-for-byte — that is also what
   makes the budget identity exact model-wide.
   Reachable MAC share: 4B 86.9% / llama8B 92.5% / 14B 94.5% / 30B 76.7%.
8. **MoE (30B) is dense-only so far** — the allocator raises on a unit index.
   Band map key must become (op, block, expert) before 30B runs.

## Code map

| file | role |
|---|---|
| `kernels/scmp_kernels/mp/config.py` | `k_bands` schema, loader + all validation, `get_k_bands` |
| `model/sc_common.py` | runtime band dispatch in SCLinear + per-band `mac_scale` |
| `benchmark/ppl/mp_kbands.py` | build child bundles (`identity` / `apply`) |
| `benchmark/ppl/mp_kbands_calib.py` | the allocator (band assignment + per-rung solve) |
| `benchmark/ppl/kbands/run_kband_cell.sbatch` | full-protocol eval of one bundle |
| `benchmark/ppl/kbands/run_kband_tool.sbatch` | `KB_JOB=equiv` / `KB_JOB=alloc` |
| `benchmark/ppl/kbands/test_kband_equivalence.py` | GPU equivalence proof |
| `kernels/tests/test_k_bands.py` | 22 schema/resolver tests |

Artifacts: `/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/`
Logs: `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/`

## Parents (mp_best deployed winners, the things we refine)

| model | t32 PPL / realized | t48 PPL / realized | fp16 |
|---|---|---|---|
| 4B | 12.571527 / 33.73 | 10.774525 / 49.66 | 10.0445 |
| llama8B | 9.157662 / 34.00 | 8.061970 / 49.62 | 7.2130 |
| 14B | 9.550835 / 33.61 | 9.017426 / 49.49 | 8.6383 |
| 30B | 9.624090 / 34.09 | 8.224459 / 51.09 | 7.2613 |

All 8: `escape_gate_k=2.0`, INT7 hybrid mask ~20%, `layer_buckets=4`.

---

## LEDGER (append every result; never delete)

### 2026-08-01/02 — foundations
- **Escape-gate ladder bug FIXED** (`_apply_escape_gate` rebuilt against the
  GLOBAL ladder while classify used the per-bucket one → 96 assignments for 64
  rows, double-dispatch). Dormant in all 20 deployed cells (zero per-bucket
  ladders) so **no archived result is affected**; fires the moment per-band
  ladders exist. 4th bug in the "two resolvers disagree about the ladder"
  family — `classify_level_values` and my own dispatch loop had it too.
- **GPU equivalence PROVEN** (job 55934465): see invariant 1. Also test C —
  one outlier chunk at 128, rest at 56, mean 65.0 vs uniform 64 — cut
  reconstruction error **−47.8%**. Synthetic tensor with a planted 30× outlier
  and 1.6% over budget, so NOT a PPL prediction, but the mechanism has real
  leverage on the structure LLM activations actually have.
- **S1 v1 FAILED** (IndexError, o_proj): sized the dispatch loop with
  `classify_level_values`, which appends the escape length as an extra index.
  Fixed to `get_levels` + a loud length check + 3 regression tests.
- **Allocator v1 FAILED**: `load_sc_model` returns the model, not a tuple.
  Switched to `build_sc_model` (the deployed eval's own loader) so the
  allocator measures the model the runtime actually executes.
- Grid clamped to ≤128 (had reached 135 — would have wrapped).

### 2026-08-02 — S0 PASSED (gate fix proven a no-op)
Job 55931422, 4B t48 parent, full protocol, 1h23m:

    ppl = 10.77452518005564   (archived: 10.77452518005564)  EXACT, 16 digits
    tokens = 298,862          realized_flop_avg_sl = 49.66 (archived 49.657)

Read the exact value from the TRACE HEADER, not the `[RESULT]` line — the
latter prints `{ppl:.4f}` and can only ever confirm ~1e-4. **Always check
`traces/*.json` header `ppl` for exact comparisons.**
Consequence: the escape-gate ladder fix does not move any deployed cell, so
every archived mp_best number stands and the harness reproduces bit-for-bit.
S0 is therefore a reusable control — no need to re-run it per experiment.

### 2026-08-02 — allocator v2 WORKS; solver bias found and fixed
Allocator v1 (job 55935019) completed in **8 min** and solved all 28 buckets
(7 linears × 4 layer buckets) with `predicted_err_ratio < 1.0` EVERYWHERE:
k_proj best (0.76–0.87), then q/v_proj (0.83–0.97), o/down/gate/up (0.89–0.98).

**But it was cheating.** 83 of 140 rung allocations OVERSPENT (mean +0.0405
cycles, +0.062% of budget; max +0.63%). Cause: feasibility was two-sided
(|residue| ≤ tol) and a longer stream always lowers error, so the argmin
systematically preferred candidates that happened to round UP — the optimizer
was buying its win with compute instead of allocation.

FIX: `solve_rung` → `rung_candidates`, feasibility now ONE-SIDED
`-tol ≤ residue ≤ 0` (never overspend), yielding every integer payback
rounding so the search still has room. Ties break toward giving budget BACK.
The parent (residue exactly 0) is always admissible, so the search never
returns empty. Verified: 118 candidates generated, 0 overspending, all
residues in [-0.25, 0], cap 128 respected.
**Consequence: any Phase-3 win is now at NO MORE than parent MAC cost, so
"iso-compute" needs no asterisk.** Re-running all allocations under v2.

Band geometry note: importance-ranked membership makes bands UNEVEN, e.g.
down_proj [4480, 4664] rather than the identity control's [4608, 4536] — band 0
holds the top-importance chunks, band 1 the rest plus the ragged tail.

### 2026-08-02 — allocator v3: error-curve SNAPPING was hiding improvements
v2 ran clean (0 overspend everywhere, confirmed on 4 cells) but produced a
nonsense spread in how much it found:

    14B t32     n_improved =   0/168   pred_err mean 1.0000
    4B t48      n_improved = 114/140   pred_err mean 0.9416
    llama8B t32 n_improved =  79/168
    llama8B t48 n_improved =  58/140

NOT a model difference — a **measurement-resolution artifact**. `err(b,L)`
snapped to the nearest MEASURED grid point (spacing 4). 14B t32's ladder
[96,64,48,32,24,16] is all multiples of 8, so every ±1 candidate snapped onto
the SAME point as the parent, tied, and reported "no improvement". 4B t48's
irregular [111,85,49,48,33] makes offsets from five different rungs overlap
into an effectively denser grid, so it could resolve them. **`n_improved` was
measuring LADDER REGULARITY, not allocation quality, and was not comparable
across cells.**

FIX: `err()` now does LOG-LOG interpolation between bracketing measured points.
Log-log is the right space — SC error falls as roughly A/L^p, a straight line
in (log L, log E) — so it is near-exact, not merely smooth. Verified on the
exact failing case (14B t32 ladder, spacing-4 grid, synthetic A/L curves with a
10× band asymmetry): **6/6 rungs improved**, budget flows to the higher-error
band, still zero overspend.

LESSON for future waves: any statistic that varies with the LADDER rather than
the model is suspect. Compare `n_improved`/`pred_err_ratio` across cells as a
sanity check every time.

v2 outputs preserved at `alloc_v2_snapbug/` for comparison; v3 relaunched on
all 6 dense cells (55935797-802).

### 2026-08-02 — the argmin exploits EVERY source of slack (3 instances)
A pattern worth internalizing: three separate bugs here were all the same
failure mode — the search finds and exploits whatever slack the objective
leaves it.
  1. two-sided residue tolerance  -> bought wins with compute
  2. nearest-grid-point snapping  -> bought wins with quantization noise
  3. single-sample error curves   -> can buy wins with sampling noise
(3) is now instrumented rather than assumed away: `band_error_curve` returns
curves on two DISJOINT row halves (parity split, one SC pass), the solve uses
half A, and `score_on` scores that SAME choice on half B. A held-out ratio near
1.0 while the solve ratio is well under it = the gain was noise.
**Check `heldout_err_ratio` before believing any `predicted_err_ratio`.**

### 2026-08-02 — band ASYMMETRY is the real lever (not the ladder solve)
With interpolation fixed, 14B t32 still improves only 6/168 (mean 0.9999)
while 4B t48 gets 114/140 (mean 0.9518). That is NOT a bug — it is the 50/50
importance split diluting outlier chunks with ordinary ones until both bands
have near-equal error density, at which point water-filling correctly declines
to move anything. My synthetic test improved 6/6 on 14B t32's exact ladder only
because I handed it a 10x band asymmetry.

Added `--hot-frac` (fraction of chunks in band 0) and a
`band_err_density_asymmetry` diagnostic = max/min of E_b(L_parent)/w_b at each
parent rung. Asymmetry is the number that predicts whether ANY redistribution
can pay; reporting it makes a null result diagnosable (bad split) rather than
merely disappointing. Sweeping hot_frac ∈ {0.125, 0.25, 0.5} on 4B t48.
Trade-off to watch: a narrow hot band concentrates outliers (more asymmetry)
but each cycle moved there covers fewer channels.

### 2026-08-02 — v3 predicted-gain table, all 6 dense cells (hot_frac=0.5)
Solve-half predicted error ratio (lower = more headroom found). NOT held-out,
NOT PPL — treat as an upper bound on what the allocation can deliver.

| cell | rungs improved | mean ratio | best rung |
|---|---|---|---|
| 4B t48      | 114/140 | 0.9518 | 0.7879 |
| 4B t32      | 101/168 | 0.9615 | 0.7619 |
| 14B t48     |  69/196 | 0.9871 | 0.9065 |
| llama8B t48 |  62/140 | 0.9888 | 0.8697 |
| llama8B t32 |  60/168 | 0.9822 | 0.8032 |
| 14B t32     |   6/168 | 0.9999 | 0.9940 |

Pattern: **4B ≫ llama8B ≈ 14B t48 ≫ 14B t32.** The smallest model has by far
the most exploitable band asymmetry; 14B t32 has essentially none at this
split. Note this does NOT follow the AWQ front-end pattern (where margins grew
as the budget tightened) — here 4B t48 beats 4B t32, i.e. the LOOSER budget
has more headroom. Do not assume budget-tightness monotonicity for K-bands.

⚠ Even the best cell is a ~5% predicted RECONSTRUCTION-error reduction.
Reconstruction-error gains historically translate weakly to PPL, so the honest
prior is that PPL moves much less than 5%, if at all. The S2 cells decide.

### 2026-08-02 — HELD-OUT CHECK PASSES: the gains are not fitted noise
Solve on row-half A, score the SAME allocation on disjoint row-half B:

| cell | solve mean | held-out mean | gain surviving |
|---|---|---|---|
| 4B t48      | 0.9518 | 0.9518 | 100% |
| llama8B t32 | 0.9814 | 0.9821 |  97% |

Per-bucket agreement is tight everywhere (e.g. k_proj:l2 0.8211 vs 0.8216;
down_proj:l3 0.9409 vs 0.9326). So slack source (3) — the argmin selecting
sampling noise — is REFUTED: 512 rows is ample and the allocation generalizes
across rows. The predicted reconstruction-error reductions are real.

⚠ SCOPE OF THIS CLAIM. The split is by row PARITY within the same
(block, window) sample, so it tests only that we are not fitting row-level
sampling noise. It does NOT test transfer across blocks, across calibration
windows, or to the wikitext-2 TEST split. Those remain open and are exactly
what the S2 full-protocol PPL measures.

Standing conclusion: the allocator is sound and its numbers can be trusted AS
RECONSTRUCTION-ERROR predictions. The open question is now purely the
**error → PPL transfer**, not the allocator.

### 2026-08-02 — hot_frac sweep: narrowing the hot band HELPS, modestly
4B t48, same parent, only the band-size split varies:

| hot_frac | band asymmetry | predicted err | held-out | down_proj widths |
|---|---|---|---|---|
| 0.5   | (pre-diagnostic) | 0.9518 | 0.9518 | [4480, 4664] |
| 0.125 | 3.48x            | 0.9442 | 0.9435 | [1152, 7992] |

Concentrating the top-importance chunks into a narrow band raises error-density
asymmetry to 3.48x and improves the predicted gain 4.8% -> 5.6%. Directionally
confirms that ASYMMETRY, not the ladder solve, is the lever — but it is a 0.8pp
move, not a step change, so per-group MP looks like a modest-gain mechanism on
real activations rather than the ~48% seen on a synthetic planted-outlier
tensor. Held-out still tracks solve (0.9435 vs 0.9442), so this is real.

**UPDATE — the sweep is NON-MONOTONIC; the optimum is INTERIOR at ~0.25:**

| hot_frac | asymmetry | predicted err | held-out | down_proj widths |
|---|---|---|---|---|
| 0.5    | (pre-diag) | 0.9518 | 0.9518 | [4480, 4664] |
| **0.25** | 2.79x    | **0.9369** | 0.9365 | [2304, 6840] |
| 0.125  | 3.48x      | 0.9442 | 0.9435 | [1152, 7992] |

Asymmetry rises monotonically as the band narrows, but the GAIN peaks at 0.25.
This is the predicted trade-off made concrete: a narrower hot band concentrates
outliers (more asymmetry) but covers fewer channels, so each cycle moved there
buys less total error reduction. **Do not tune on asymmetry — it is a
diagnostic, not the objective.** Best so far: 6.3% predicted error reduction.

NOTE the floor: bands need >= 2 chunks, so narrow ops (q/k/v_proj, 19 full
chunks) bottom out at 2 chunks around hot_frac ~0.1 and stop responding to
further narrowing; only down_proj (71 chunks) has real room. That is likely
WHY 0.125 regresses — the small ops are pinned while down_proj over-narrows.
A per-operator hot_frac would decouple them; untested.

**SWEEP COMPLETE — interior optimum at hot_frac=0.25:**

| hot_frac | asymmetry | predicted err | held-out | down_proj widths |
|---|---|---|---|---|
| 0.5    | (pre-diag) | 0.9518 | 0.9518 | [4480, 4664] |
| **0.25** | 2.79x    | **0.9369** | 0.9365 | [2304, 6840] |
| 0.125  | 3.48x      | 0.9442 | 0.9435 | [1152, 7992] |
| 0.0625 | 3.59x      | 0.9444 | 0.9449 | [512, 8632]  |

Asymmetry SATURATES (3.48 -> 3.59) while the gain plateaus at ~0.944 — past
0.125 the top chunks are already isolated and further narrowing concentrates
nothing new. **OPERATING POINT: hot_frac=0.25, 6.3% predicted error reduction,
held-out confirmed (0.9365).**

Three S2 PPL points now queued from the SAME parent, differing only in
allocation quality — 0.9518 / 0.9442 / 0.9369 — which measures the
error -> PPL transfer SLOPE rather than a single anecdote. hf=0.25 allocations
prepped for the other 5 dense cells (55938278-82) so a fan-out is one command
whichever way the transfer goes; the predicted-error table at the optimum is
evidence for the writeup even if PPL does not move.

### 2026-08-02 — ★ KEY FINDING: band asymmetry is a MODEL property and it
### predicts whether per-group MP can pay at all

hot_frac=0.25 (the 4B optimum) applied to other cells:

| cell | hf=0.5 | hf=0.25 | delta | band asymmetry |
|---|---|---|---|---|
| 4B t48      | 0.9518 | 0.9369 | **-0.0149** | **2.79x** |
| 4B t32      | 0.9617 | 0.9416 | **-0.0201** | **2.79x** |
| llama8B t48 | 0.9887 | 0.9874 | -0.0013 | 1.29x |
| llama8B t32 | 0.9814 | 0.9859 | **+0.0045** | 1.29x |

**Asymmetry is a property of the MODEL, not of the tuning.** 4B's channel
importance is strongly skewed (2.79x error-density gap between bands) and
responds to band-geometry tuning. llama8B's is nearly flat (1.29x): both bands
carry the same error per unit width, water-filling has essentially nothing to
move, and re-tuning hot_frac is noise — it made llama8B t32 WORSE (+0.0045,
possible because changing hot_frac changes the PARTITION, so it is a different
problem, not a worse solve of the same one; the parent is still in the space of
each individual solve).

This gives the earlier ranking (4B >> llama8B ~ 14B t48 >> 14B t32) a
MECHANISM rather than leaving it an observation, and it yields a cheap
predictive rule: **measure band error-density asymmetry first; if it is near
1.0 there is no headroom and per-group MP cannot help that model, whatever the
band geometry.** Asymmetry costs one FP + a few SC calls to measure — far
cheaper than a full-protocol cell.

Practical consequence: hot_frac must be tuned PER MODEL (4B wants 0.25,
llama8B is indifferent-to-worse), and a per-OPERATOR hot_frac is the untested
next refinement (small ops floor at 2 chunks while down_proj over-narrows).

### 2026-08-02 — ★★ S1 FAILED THE STATED BAR, and the reason changes the
### METHODOLOGY: the MP dispatcher amplifies numerical noise into ~0.01 PPL

    S0 parent          ppl = 10.774525180055640   flop_avg_sl = 49.657398
    S1 identity child  ppl = 10.764676976522695   flop_avg_sl = 49.636094
    delta              ppl = -0.009848 (-0.0914%)  cost = -0.021305

My stated bar was |dPPL| < 1e-3. The observed delta is **10x that**. Reporting
it as a FAIL rather than widening the bar.

DIAGNOSIS — it is not an implementation bug. Trace comparison shows total MACs
identical to 9 decimals (ratio 1.000000000) and the same 38 (op, len) groups,
but the length DISTRIBUTION shifted — **including in `qk` and `av`, which the
K-band code never touches.** That is only possible if the perturbation reaches
attention through the activations. The chain:

  band split changes fp32 accumulation ORDER in linears (~1e-7, as measured by
  the equivalence test) -> perturbed activations reach the next layer -> the MP
  dispatch metric (amax/l2/crest, min-max normalized PER CALL) shifts slightly
  -> rows sitting near a threshold BOUNDARY flip to a different rung -> different
  stream lengths -> different MACs and different PPL.

**The dispatcher is a step function, so an infinitesimal input change produces
a DISCRETE reallocation.** Bit-determinism (see
[[project_scmp_eval_determinism_and_test_selection]]) holds only for a
bit-identical computation; any mathematically-equivalent regrouping escapes it.

### CONSEQUENCES — read before interpreting ANY Phase 3 result
1. **There is a NOISE FLOOR of order 0.01 PPL (~0.09%) on any comparison
   between configs that are not bit-identical.** A Phase-3 "win" smaller than
   that is indistinguishable from dispatch-boundary reshuffling.
2. **S2 must be compared against S1 (the identity child), NOT against S0.**
   S1 isolates the band-split numerics; only S1->S2 is attributable to the
   ALLOCATION. Comparing S2 to S0 would credit the allocator with a numerical
   accident. (S1 happened to land -0.0098 BETTER, so an S0 comparison would
   flatter Phase 3 by that much.)
3. Iso-compute is also perturbed: S1 spent 0.021 cycles LESS than the parent
   without being asked to. The per-rung identity still holds for what the
   allocator ASSIGNS; what drifts is which rows the dispatcher assigns.
4. Estimating the floor: identity variants with n_bands = 2 / 3 / 4 are all
   mathematically the parent but numerically distinct. The SPREAD of their PPLs
   IS the noise floor. n_bands=3 (55938877) and 4 (55938880) launched.

This is the single most important methodological result of the run so far. It
likely applies to prior MP waves too — any V17/V19/V20 comparison between
configs that changed the computation numerically carries the same floor.

### 2026-08-02 — asymmetry PREDICTS headroom; only 4B has any

At the optimum hot_frac=0.25, across every dense cell measured:

| cell | predicted err | held-out | band asymmetry | headroom |
|---|---|---|---|---|
| 4B t48      | 0.9369 | 0.9365 | **2.79x** | yes |
| 4B t32      | 0.9416 | 0.9417 | **2.79x** | yes |
| llama8B t32 | 0.9859 | 0.9862 | 1.29x | no |
| llama8B t48 | 0.9874 | 0.9871 | 1.29x | no |
| 14B t48     | 0.9935 | 0.9938 | 1.26x | no |
| 14B t32     | 0.9999 | 0.9999 | 1.24x | no |

COMPLETE, 6/6 cells, and the ordering is monotone in asymmetry with no
exceptions: 2.79x -> 6.3/5.8% gain; 1.29x -> 1.4%; 1.26x -> 0.65%;
1.24x -> 0.01%. One cheap statistic, measured in ~8 min, orders every cell.

Asymmetry tracks headroom EXACTLY: 2.79x -> ~6% predicted gain; 1.24-1.29x ->
0.01-1.4%. Mechanism: the protected-channel pin already removes the top outlier
channels, and what remains is homogeneous for most models. **Qwen3-4B is the
exception — its channel importance stays skewed even after the pin.**

### ⚠ PROJECTION, stated before the PPL lands so it cannot be retrofitted
Combining this with the measured ~0.01 PPL noise floor:
- llama8B / 14B predict 0.01-1.4% RECONSTRUCTION-error reduction. After the
  (historically weak) error -> PPL transfer, that is far below the floor.
  **These cells cannot show a detectable Phase-3 effect. Do not run more of
  them hoping otherwise — that would be a known-outcome trial.**
- 4B predicts ~6%. It is the ONLY cell where a detectable effect is plausible,
  and even there it is not assured.

So the live question has narrowed to exactly one: **does 4B's 6% predicted
reconstruction gain clear the 0.01 PPL floor at iso-compute?** Three 4B t48 S2
cells (hf 0.5 / 0.125 / 0.25) plus the n_bands=2/3/4 identity floor samples
answer it directly.

### 2026-08-02 — FIRST S2 PPL: llama8B t32. Looks like a win. IS NOT ONE.

    parent (mp_best)  ppl = 9.157661848077   flop_avg_sl = 34.0014
    S2 phase3 hf=0.5  ppl = 9.147687864525   flop_avg_sl = 33.98
    delta                  -0.009974 (-0.1089%)

**-0.11% at iso-compute would be a reportable Phase-3 win if taken at face
value. It is not attributable.** The delta is 0.0100 against a measured noise
floor of 0.0098 — the two are the same number. And this cell's predicted
reconstruction gain was only 1.9%, exactly the regime the pre-registered
projection said could not produce a detectable effect. The projection held.

This is the concrete payoff of the S1 control: without it the honest-looking
move is to report "Phase 3 improves llama8B t32 by 0.11% at iso-compute",
which would have been spurious. **Any Phase-3 delta near 0.01 PPL is
dispatch-boundary reshuffling until proven otherwise by a matched identity
control on the SAME model.**

Launched the llama8B t32 identity control (55939846) — not a known-outcome
trial: it measures llama8B's own noise floor, which is unknown and is required
to interpret a number already in hand.

### 2026-08-02 — llama8B t48 confirms: also below the floor

    parent 8.061970470653  ->  S2 8.056088635906   delta -0.005882 (-0.073%)
    cost 49.6178 -> 49.61  |  predicted recon gain 1.1%  |  floor ~0.0098

Both llama8B cells land under the floor, as projected (t32 -0.0100 @1.9%
predicted, t48 -0.0059 @1.1% predicted). The ordering is consistent with
predicted gain but BOTH are inside noise, so that ordering is not evidence.
**llama8B: no detectable Phase-3 effect at iso-compute. As predicted from
asymmetry 1.29x, before the runs.**

### 2026-08-02 — ★★★ DECISIVE: 4B t48, the BEST cell. Allocation contributes
### ~0.002 PPL — five times BELOW the noise floor.

    S0 parent    10.774525180056   flop 49.6574
    S1 identity  10.764676976523   flop 49.6361
    S2 phase3    10.762692315317   flop 49.59     (hf=0.5, 4.8% predicted gain)

    NAIVE   S0 -> S2 = -0.011833 (-0.110%)   <- would have been reported as a win
    CORRECT S1 -> S2 = -0.001985 (-0.018%)   <- the ALLOCATION's contribution

**83% of the naive win is the band-split NUMERICS, not the allocation.**
Allocation contribution 0.0020 vs floor 0.0098 => NOT ATTRIBUTABLE, on the cell
with the most exploitable structure of any in scope.

### ⚠ RETRACTED (same day, by the very next cell): "transfer ratio ~0.4%"
I wrote that 4.8% predicted recon gain -> 0.018% PPL implies a transfer ratio
of ~0.004, and flagged it as one point. The next cell shows the ratio is not a
stable quantity at all:

| variant | predicted recon gain | S1 -> S2 PPL | implied transfer |
|---|---|---|---|
| hf=0.5   | 4.8% | -0.00198 (-0.018%) | 0.4% |
| hf=0.125 | 5.6% | **-0.02625 (-0.244%)** | **4.4%** |

Nearly identical predicted gains; **13x different PPL effect; 10x different
implied transfer ratio.** So "the transfer ratio is ~0.4%" is WRONG as a
general statement — do not carry it forward. What survives is weaker and still
useful: **the sigma model does not ORDER band geometries by their PPL effect.**
hf=0.125 has a worse predicted recon gain than hf=0.25 and barely beats hf=0.5,
yet moves PPL 13x more than hf=0.5.

### 2026-08-02 — ★★★ ALL THREE 4B t48 VARIANTS IN, AND THEY ORDER MONOTONICALLY

| variant | predicted recon gain | S1 -> S2 PPL | x floor | realized flop |
|---|---|---|---|---|
| hf=0.5   | 4.8% | -0.00198 | 0.2x | 49.59 |
| hf=0.125 | 5.6% | -0.02625 | 2.7x | 49.59 |
| **hf=0.25** | **6.3%** | **-0.03501** | **3.6x** | **49.54** |

S0 parent 10.774525180 (flop 49.6574) -> hf=0.25 **10.729662434 (flop 49.54)**:
-0.0448 PPL (-0.42%) at LOWER cost. Allocation-only contribution (S1 -> S2) is
-0.03501 (-0.33%). vs fp16 10.0445: parent 1.0727x -> hf=0.25 1.0682x.
**No overspend anywhere** — all three are at or under the parent's realized
cost, and the best is 0.12 cycles UNDER.

### ⚠ SECOND RETRACTION, same hour: "sigma does not ORDER band geometries"
I wrote that after two points. With the third it is **false** — the ordering is
perfectly monotone (4.8/5.6/6.3% -> 0.00198/0.02625/0.03501). What is true is
that the relationship is strongly NONLINEAR/convex: +0.8pp predicted gain
(4.8->5.6) multiplies the PPL effect 13x, while the next +0.7pp adds only 1.3x.
Possibly a threshold — hf=0.5 has near-equal band widths and may simply not
concentrate outliers enough to matter.

**SELF-CAUTION for future iterations: I generalized prematurely TWICE in one
hour off partial data (the "0.4% transfer ratio", then "sigma does not order").
Both were stated on 1-2 points and overturned by the next cell. With ~1.4h per
cell it is cheap to WAIT for the full sweep before writing a law.**

### ★★★★ 2026-08-02 — RESOLVED: PHASE 3 WINS ON 4B t48, 6.3 sd OUTSIDE THE
### NOISE FLOOR, AT LOWER COST.

Floor measured properly (n=4 identity configs, n_bands = 1/2/3/4, ALL
mathematically the parent, differing only in fp32 accumulation grouping):

    mean 10.772893042   sd 0.006916   span 0.016590
    deviations: +0.0016, -0.0082, -0.0018, +0.0084   <- BOTH directions, as
    a noise distribution must (nb4 landed ABOVE the parent). Hypothesis (b)
    from the previous entry is refuted: the floor is symmetric noise, and the
    Phase-3 effects are not draws from it.

| config | ppl | vs floor mean | z | verdict |
|---|---|---|---|---|
| hf=0.5   | 10.762692315 | -0.01020 | -1.5 | not significant |
| hf=0.125 | 10.738425706 | -0.03447 | **-5.0** | significant |
| **hf=0.25** | **10.729662434** | **-0.04323** | **-6.3** | **strongly significant** |

**Headline: 4B t48, Phase 3 (hf=0.25) = 10.7297 vs deployed parent 10.7745.
-0.42% PPL at 49.54 realized cycles vs the parent's 49.6574 — i.e. BETTER
QUALITY AT LOWER COMPUTE.** vs fp16 10.0445: 1.0727x -> 1.0682x.

Hypothesis (a) confirmed: the effect is real. And the equal-split hf=0.5 being
insignificant (-1.5 sd) while the concentrated splits are 5-6 sd is exactly the
predicted mechanism — asymmetry is what makes per-group allocation pay.

### CAVEATS — do not overstate this
1. **ONE model, ONE target.** 4B t48 only. 4B t32 has the same 2.79x asymmetry
   and should replicate; llama8B/14B (1.24-1.29x) predicted and DELIVERED
   nothing, so this is not a general win — it is a win where asymmetry exists.
2. **n=4 floor samples**, so sd=0.0069 is imprecise. Even a 2x underestimate
   leaves hf=0.25 at >3 sd, so the conclusion is robust to that, but a wider
   floor sample would tighten it.
3. The sigma model ORDERS the three variants correctly but the relationship is
   convex, not linear — do not use predicted gain to extrapolate PPL.

### 2026-08-02 — llama8B t32 now has its control; confirms the null
    S0 parent   9.157661848
    S1 identity 9.155652776   noise-only  -0.002009
    S2 phase3   9.147687865   ALLOCATION  -0.007965   (~1.2x 4B's floor sd)

Not significant. Contrast 4B t48 hf=0.25 at -0.03501 (-6.3 sd) — a **4.4x
larger allocation effect on the model with 2.2x the band asymmetry**, using the
same code and the same objective. The asymmetry statistic predicted both
outcomes before either cell ran.

(Only ONE llama8B identity sample, so no z-score for it; the comparison uses
4B's floor sd as a stand-in. That is a stated approximation, not a measurement.)

### THE RESULT IN ONE LINE
Per-group (K-band) MP beats the deployed per-row parent by **-0.42% PPL at
LOWER compute on 4B t48 (6.3 sd)**, does nothing on llama8B/14B, and **band
error-density asymmetry predicts which in ~8 min instead of ~1.4 h.**

### ★★★★ 2026-08-02 — NEW GOAL (user): 1.05x fp16 on all 4 models.
### And the 2-band implementation was leaving 2-3x ON THE TABLE.

Gap to 1.05x, and the lowest budget that meets it TODAY:

| model | lowest budget @1.05x | x_fp16 @t48 | gap |
|---|---|---|---|
| 4B      | t64 (1.0471) | 1.0727 | -2.1% |
| llama8B | **NONE — t128 is 1.0574** | 1.1177 | -6.0% |
| 14B     | t48 (1.0439) OK | 1.0439 | met |
| 30B     | t96 (1.0483) | 1.1326 | -7.3% |

**llama8B cannot reach 1.05x by ALLOCATION at any budget** — at t128 the
allocator has nothing left to allocate, so 1.0574 is the SC quality FLOOR.
INT8 reaches 1.0002, so the headroom is real but lives in the SC grid /
front-end / mask dose, not the allocator. Separate track.

### ORACLE: how much does BAND COUNT buy? (closed form, no GPU)
Fix one population of 72 per-chunk importances, group into n bands by
importance quantile. For E_b(L)=A_b/L at iso-cost the optimum is
L_b ~ sqrt(A_b/w_b), so E*/E_unif = (sum sqrt(A_b w_b))^2/(W*sum A_b) — exact.

| chunk-importance distribution | 2 | 4 | 8 | 16 | 72 (per-chunk) |
|---|---|---|---|---|---|
| lognormal s=1 (15x spread)  | 16.2% | 22.3% | 24.8% | 25.7% | 26.2% |
| lognormal s=2 (107x spread) | 35.1% | 50.2% | 56.1% | 58.1% | 59.0% |
| **5 outliers @50x (LLM-like)** | **18.3%** | 33.9% | 46.2% | 50.0% | **54.1%** |
| near-flat (llama8B-like)    |  0.1% |  0.2% |  0.2% |  0.2% |  0.2% |

**In the outlier-cluster regime, 2 bands captures only 34% of the available
gain.** Granularity IS the lever, as the user argued. Extrapolating the
measured 4B t48 result (-0.42% PPL at 2 bands): per-chunk could plausibly give
~3x that, ~-1.2%, versus the -2.1% needed for 1.05x — reachable when combined
with another lever (AWQ front-end is worth -1.9% on uniform).
Also note the near-flat row: no band count rescues llama8B. Consistent with
its measured 1.29x asymmetry and its null PPL result.

⚠ A FIRST version of this test was WRONG and said the opposite (more bands =
worse). It built A_b as a geometric ramp with fixed endpoints, so raising n
SMOOTHED the distribution rather than resolving it. Holding the chunk
population FIXED is what makes the comparison mean anything. Kept as a warning:
synthetic tests can encode the answer you feed them.

### SOLVER REPLACED: hot-band sweep -> exact Lagrangian water-filling
The old move set perturbs ONE band with proportional payback — fine at 2 bands,
hopeless as bands multiply (reachable set O(B*max_delta) in a B-dim space).
`solve_bucket_waterfill` solves min sum_b E_b(L_b) s.t. sum_b w_b L_b <= budget
by bisecting the Lagrange multiplier (tight: E_b measured, decreasing, convex),
then greedily spending leftover slack. One-sided budget, so still never
overspends. Verified against the closed-form sqrt(A/w) optimum.
`solve_bucket` (hot-band) is KEPT so the published -0.42% result stays
reproducible.

Launched: 4B t48 allocations at n_bands = 4 / 8 / 16 (55943173-5).

### ★★★★★ 2026-08-02 — 4B t32 REPLICATES AND IS 4x BIGGER THAN t48

    parent    12.571526971939   flop 33.7306
    identity  12.559569448484   (noise control)
    phase3    12.409696318296   flop 33.59      <- UNDER the parent's budget
    ALLOCATION delta (S1 -> S2) = -0.14987  ~= -22 sd   (floor sd 0.0068, n=5)

**-1.19% at t32 vs -0.33% at t48.** Per-group MP pays ~4x more at the TIGHT
budget — which is the regime that matters, since the goal is to lower the
passing rung. Replicates the 4B t48 win independently (different target, own
identity control) and confirms the asymmetry rule predicts across targets.

⚠ AND IT INVERTS sigma's prediction: sigma said t48 had MORE headroom
(0.9369 vs 0.9416). PPL says t32 benefits 4x more. Third strike for sigma as a
PPL proxy — weak, nonlinear, and now non-monotone ACROSS budgets. Stop using
predicted_err_ratio to choose which CELL to work on; it is only usable to rank
variants within one cell.

### ⚠ CORRECTION — "llama8B NEVER reaches 1.05x" was WRONG
That read the SmoothQuant mp_best table. The AWQ front-end is already measured
full-protocol on the exact deployed bundles
(`hpca_results/llm/frontend_awq/mpbest_awq_vs_smoothquant.csv`, 23/24 cells
improve, bundle unchanged, front-end the only swap). Re-baselining costs ZERO
GPU:

| model | AWQ x_fp16 |
|---|---|
| 4B      | t48 1.0602, t64 1.0428 |
| llama8B | t96 1.0557, **t128 1.0481 PASSES** |
| 14B     | **t40 1.0488 PASSES**, t48 1.0311 |
| 30B     | **t96 1.0464 PASSES**, t128 1.0369 |

Also `ppl/mp_best/rebuild.py:172-185` hard-codes the 10% mask, so the llama8B
t128 row is MIS-SELECTED — a better 20% cell exists.

**RESTATED GOAL: lower each model's passing rung by one step.** Gaps in nats
(1.05x == 0.04879 nats excess over fp16):
  4B t48      gap 0.00967  REACHABLE (~55%)
  llama8B t96 gap 0.00540  REACHABLE (~75%)   | t64 gap 0.01794 (~35%)
  30B t64     gap 0.01175  REACHABLE (~45%)
  14B t32     gap 0.02604  NOT reachable — **stop tuning 14B**

### THE CHEAP SEARCH OBJECTIVE (use this, do not invent another)
    e(c) = mean over windows W of [ NLL_c(w) - NLL_fp16(w) ]   nats, lower better
paired per window, wikitext-2 VALIDATION, ctx 2048, 4-window blocks.
Subtracting the per-window fp16 reference is load-bearing — it is what makes
the val->test slope 1.00 (r=0.999, residual 0.0036 nats) at 32 windows.

REUSE (import, do not rewrite): `mp_v16_refine.py:1725 eval_windows`,
`:234 paired_deltas`, `:242 pooled_sigma_w`, `:145 build_window_partition`,
`:262 replication_accept_gate`, `:323 confirm_veto_gate`.

COST (warm process, model load amortised) s/window: 28.7 (4B) 27.1 (llama8B)
65.7 (14B) 63.2 (30B). 32 windows = 15.3 / 14.5 / 35.0 / 33.7 min, i.e. 12-22%
of a full run. Do NOT use PPL_MAX_TOKENS truncation — same cost AND it burns
the test split.

NOISE: sigma_w = 0.0204 (4B) 0.0234 (llama8B) 0.0212 (14B) 0.0168 (30B) nats,
empirical over 1541 records. **The code's `DEFAULT_SIGMA_W = 0.015`
(mp_v16_refine.py:118) UNDER-estimates by ~35%** and is used whenever a sweep
has <5 candidates — so past accept gates were too permissive. Pass 0.021.

**EFFICIENCY CORRECTION to my own practice:** at n<=32 windows the sampling SE
(0.0035 nats) exceeds the dispatch-reshuffle floor (0.00064 nats) by 6x, so
identity controls are WASTED GPU during a search. They are mandatory ONLY for
full-protocol comparisons, where sampling SE ~ 0 and the floor dominates.

### NEXT SEARCH VARIABLE: qk/av OPERAND CONDITIONING (not more allocator work)
The allocator variable is near its optimum (<0.5% range left; V20 regressed at
t32 everywhere, threshold moves 0/165 accepts). The unspent variable is operand
conditioning on qk/av — **~90% of rows, and NEITHER front-end touches it**
(SmoothQuant and AWQ both reach SCLinear only; qk/av run dynamic symmetric
absmax with zero calibrated component).

Exact, score-invariant, runtime-free:  scores = sum_d (Q_d s_d)(K_d / s_d)

**CORRECTNESS FIX to the lead recorded in CLAUDE.md:** that lead says scale
W_q out-channels by s and W_k by 1/s. **INVALID on Qwen3 (4B/14B/30B)** —
per-head q_norm/k_norm RMSNorms sit on the q_proj/k_proj OUTPUTS and
renormalise s away. Correct fold: `q_norm.weight *= s`, `k_norm.weight /= s`
(a diagonal over head_dim, cleaner). Llama-3.1-8B has NO QK-norm, so it folds
into W_q/W_k rows. Constraints: s constant within each RoPE rotate_half pair
(s[d]==s[d+head_dim/2]) and within each GQA KV group (one K head serves several
Q heads via `_repeat_kv`, sc_common.py:1008).

SPACE: do not black-box raw s (73k params on 4B). Use the SmoothQuant migration
form s_d = mq_d^alpha / mk_d^(1-alpha) with mq/mk the post-RoPE pair-max-pooled
per-dim maxima; search alpha x pooling granularity in
{per-kv-group, per-layer, per-layer-band} — 14-20 named candidates per cell.
MOVES: none. Archive measurement says single-coordinate moves near the
incumbent replicate at ~0 reliability while multi-coordinate ones replicate at
+0.87/+0.56/+0.37 — local search on this objective measures noise.

### ★★★ 2026-08-02 — GRANULARITY CONFIRMED: more bands = monotonically better
Water-filling solver, 4B t48. `down_proj` is the only op with enough chunks to
span the whole range, so compare that column like-for-like:

| n_bands | down_proj predicted gain | all-ops | note |
|---|---|---|---|
| 2  | 2.3% | 6.3% | |
| 4  | 3.1% | 7.7% | |
| 8  | 5.1% | **9.4%** | best all-ops |
| 16 | **6.6%** | — | **covers down_proj ONLY** |

Held-out tracks the solve exactly (8 bands: 0.9056 vs 0.9057). 8 bands is 1.5x
the predicted gain of 2 bands across all ops; down_proj keeps improving to 16.
4B **t32** at 8 bands: solve 0.9151 vs 0.9416 at 2 bands — and t32 is the
budget where 2 bands already delivered -1.19% PPL, so this is the highest-
leverage cell in the programme.

**STRUCTURAL LIMIT — the next refinement.** Bands need >= 2 chunks. Narrow ops
(q/k/v_proj ~19 full chunks, gate/up ~19, o_proj ~31) cap at 8-9 bands, while
down_proj has 71. So `--n-bands 16` SILENTLY dropped to down_proj alone (4 of
28 buckets). A PER-OPERATOR band count is the obvious next move: down_proj 16+,
narrow ops 8. Currently `--n-bands` is global.

### TWO SOLVER FIXES, both surfaced by validation rather than by a crash
1. **Iso-cost check was two-sided; the solver is one-sided.** Overspend stays a
   hard error (breaks the iso-compute claim). UNDERspend cannot break it — the
   cell simply used less compute than its parent, making a win STRONGER — so it
   is now allowed up to `K_BAND_MAX_UNDERSPEND = 2.0` cycles, beyond which it
   is reported as a solver bug rather than tolerated. The old check was
   rejecting valid allocations (v_proj:t0:l1 realized 47.70 vs parent 48).
2. **That guard then caught a real solver bug.** 4-band q_proj left 3.07 cycles
   unspent. Cause: measured E(L) has noisy UPTICKS, and the greedy
   slack-spender needs a positive marginal gain to take a step, so it stalls at
   the first one. Fix: `monotonize()` projects each curve onto the monotone
   cone by running-minimum before solving. SC error genuinely falls with L, so
   any rise is noise; running-min never INCREASES a measured value, so it
   cannot manufacture an optimistic curve.

### 2026-08-02 — PER-OPERATOR band counts, and the bug that hid inside them
`--n-bands` is now a MAXIMUM, clamped per operator. Runtime reads the count off
each operator's own ladder list (`len(band_ladders)`), the loader validates per
operator, and the allocator clamps per operator.

**BUG (found because a "per-operator" run still covered down_proj alone):** the
cap was computed from `n_chunks` but the split sliced `full` (= n_chunks minus
the pinned tail). q_proj: 20 chunks -> cap 10 -> per_band = 19//10 = **1**, so
nine SINGLETON bands, and the >=2-chunks guard then dropped every narrow op.
Also the remainder was dumped on the last band, giving down_proj
[512 x 15, 1464] — a final band 3x wider than the rest holding the 11 LEAST
important chunks, which wastes exactly the resolution more bands were meant to
buy (that partition scored 0.9816, far worse than 8 balanced bands' 0.9151).

FIX: derive the cap from `_n_full`, and slice `b*n_full//B : (b+1)*n_full//B`
so widths stay balanced. Verified across all 7 real 4B geometries at caps
8/16/32: every band >= 2 chunks, every band wider than chunk_d, widths sum to R,
max/min width ratio 1.36-1.90 (was 3x).

Resulting per-op band counts at cap 16: down_proj 16, o_proj 15, q/k/v_proj 9,
gate/up_proj 9 — all 7 ops covered, versus 1 before.

### ★★★★ 2026-08-02 — qk REBALANCE: recon done, and there is a BETTER anchor
### than folding into q_norm

VERIFIED FACTS (Qwen3, transformers modeling_qwen3.py in the annstention env):
- Order is q_proj -> view(B,N,H,128) -> **q_norm (RMSNorm over head_dim)** ->
  transpose -> **RoPE** -> sc_eager_attention_forward -> SC qk. Norm BEFORE RoPE.
- `q_norm.weight` / `k_norm.weight` are shape **(head_dim,) = (128,)**, ONE
  vector per decoder layer, shared across all heads (norm runs on the (B,N,H,128)
  view). Qwen3-4B: 32 Q heads / 8 KV heads.
- RMSNorm's weight is the FINAL elementwise multiply and the variance comes from
  the PRE-weight tensor (modeling_qwen3.py:71-76), so `q_norm.weight *= s` is
  exactly `s (*) q` with no feedback. **This confirms CLAUDE.md's recorded lead
  (scale W_q output channels) is NOT exact on Qwen3** — q_norm would renormalise
  it away.
- RoPE is rotate_half: it couples d with **d + head_dim/2** (d <-> d+64), and
  cos/sin are duplicated so both members share an angle.

**HARD CONSTRAINT for the q_norm fold:** a diagonal S commutes with the 2x2
rotation in plane (d, d+64) ONLY if s[d] == s[d+64]. So folding into q_norm/
k_norm gives **64 usable DOF per layer, not 128**. Numerically verified
(D=8, m=3, n=7): pair-constant s reproduces the score to err 0.0; a free
per-dim s errs 11%. GQA adds NO further constraint (the norms are head-shared).

### ★ THE BETTER ANCHOR — `smooth_scales`, already in the kernel, unused on qk
`sc_matmul` already takes `smooth_scales` and applies `(a/s, b*s)` along the
CONTRACTED dim INSIDE the matmul (matmul.py:179-181, quant/smoothquant.py:124).
Because it is applied inside the matmul, it sits AFTER RoPE — so it admits a
**FULL 128-dim s with NO pair constraint**, and
sum_d (a_d/s_d)(b_d s_d) == sum_d a_d b_d is exactly score-invariant.

It is currently passed ONLY from SCLinear; the three attention call sites
(sc_common.py:923, 953, 977) pass nothing. **That IS the "attention has never
had a front-end" gap, and closing it is a few lines rather than a weight
surgery.**

Trade-off to record: the q_norm fold is zero-runtime but 64 DOF; smooth_scales
is 128 DOF at the cost of one elementwise scale on Q and K per call — the SAME
operation class SCLinear already runs in deployment, so it is not new hardware.
START WITH smooth_scales (more DOF, less code, reuses tested plumbing); the
q_norm fold remains available as the zero-cost variant if it pays.

Both operands are quantized SYMMETRIC PER-ROW with absmax over head_dim
(kernels.py:2003-2005, grouped.py:153-156) — exactly the axis s reshapes, which
is the mechanism this exploits.

### ★★★★ 2026-08-02 — qk REBALANCE IMPLEMENTED AND TESTED

GPU test (job 55944491, `test_qk_rebalance.py`) — all PASS:
- **Exact FP invariance**: rel 7.4e-08 (float tolerance) for every alpha in
  {0, .25, .5, .75, 1}.
- **No RoPE pair constraint**: a deliberately pair-INCONSISTENT s (s[d] !=
  s[d+64]) is equally exact, because `smooth_scales` is applied INSIDE
  sc_matmul, i.e. after RoPE. A q_norm fold could not express that vector.
- **SC error falls 72.8-75.3%** at every stream length (128/64/48/32), best
  alpha 0.4-0.5.
  ⚠ BUT that was synthetic operands with |Q| spread 648x, |K| 364x. The gain is
  a direct function of that imbalance, so it is an upper bound on a favourable
  case, NOT a prediction.

### ★ REAL post-RoPE per-dim spread (job 55944501/2) — decides the payoff

| model | mean \|Q\| spread | mean \|K\| spread | layers captured |
|---|---|---|---|
| **4B**  | **36.8x** | **81.0x** | 26 |
| llama8B | 4.5x | 4.1x | 25 |

**4B has real, large imbalance; llama8B is nearly balanced.** Far below the
synthetic 648x/364x, so expect much less than 75% — but 4B has genuine
structure and llama8B has almost none.

This MIRRORS the band-asymmetry split (4B 2.79x vs llama8B 1.29x). Consistent
story: **4B has exploitable structure on every axis measured; llama8B is
diffuse on every axis.**

⚠ **This REFUTES the survey's pre-registered prediction** that llama8B is "the
model where selection-based levers must fail and only GLOBAL operand transforms
(front-end, qk rebalance) can pay". The qk rebalance IS a global operand
transform, and llama8B's Q/K are already balanced, so there is nothing to
migrate. Record it as refuted, not quietly dropped.

(Only 26/36 and 25/32 layers captured: the hybrid INT mask routes ~20% of qk
slots to INT, which never reach the SC path. Expected, not a bug.)

### IMPLEMENTATION (all landed)
- `sc_common.py`: `sc_attn_smooth` config default (None = byte-identical),
  `_qk_smooth_scales()` resolver with length/positivity validation, and the
  vector threaded through ALL THREE attention SC call sites (adaptive-MP,
  legacy-MP, uniform).
- `loader.py`: `apply_attn_smooth_from_env` (SC_ATTN_SMOOTH_JSON), called from
  `eval_quant.py` right after the hybrid config.
- `benchmark/ppl/kbands/qk_calib.py`: measures post-RoPE per-dim maxima at the
  exact point the SC product sees them (post-q_norm, post-RoPE, post-GQA
  repeat) and emits s_d = mq^alpha / mk^(1-alpha) per layer.
- `run_kband_cell.sbatch` takes `KB_QK_SMOOTH`.

First PPL cell: 4B t48 + qk rebalance alpha=0.5 (55944646).

### NEXT (highest value first)
1. **Replicate on 4B t32** (same 2.79x asymmetry) with its own identity
   control. If it reproduces, the asymmetry rule has predictive force for
   BOTH signs and the finding is solid.
2. Floor samples are cheap and reusable — add 2-3 more identity variants to
   tighten sd.
3. hot_frac between 0.125 and 0.25 is unexplored; the optimum may be sharper.
4. 30B is untouched (MoE band keying) and has the 2nd-highest attention share;
   its asymmetry is UNKNOWN and worth measuring — it is the only remaining
   model that could plausibly have 4B-like structure.

### 2026-08-02 — hf=0.125 is 2.7x the floor. Two live hypotheses (RESOLVED above).
    hf=0.125:  S1 -> S2 = -0.02625  (2.7x the single-sample floor estimate)

(a) REAL: band geometry affects PPL through a channel sigma cannot see, and
    per-group MP works but its objective is mis-specified.
(b) NOISE: the floor is a ONE-SAMPLE estimate. A single draw says nothing about
    spread; if the dispatch-reshuffle distribution has sd ~0.01, -0.026 is a
    ~2.6 sigma excursion — suggestive, not conclusive.

**Cannot distinguish these until the n_bands=3/4 identity samples land** (both
~53min in). Those are mathematically the parent and differ only numerically, so
their spread IS the floor distribution. Resolving (a) vs (b) is now the single
most important open question in the run — it decides whether Phase 3 is a
negative result or a mis-specified-objective result.

DO NOT report hf=0.125 as a win until the floor spread is known.

### PRE-REGISTERED PREDICTION (hf=0.25 cell still running)
hf=0.25 predicts 6.3% vs hf=0.5's 4.8%. If transfer is ~linear, expect
S1 -> S2 ~= -0.0026, still ~4x below the floor. **I expect it NOT to clear the
floor.** Recording this before the cell lands.

### STANDING RULE for every future cell
Report `S1 -> S2`, never `S0 -> S2`. A cell without a matched identity control
on the same model+target is UNINTERPRETABLE and must not be tabled.

### IN FLIGHT
- 55934467 S1 4B t48 identity — must match S0 to |ΔPPL| < 1e-3, realized
  cost < 0.01 cycles apart. NOT bit-identical (fp32 regrouping).
- Allocator v3 × 6 dense cells: 55935797 (4B t32), 55935798 (4B t48),
  55935799/800 (llama8B t32/t48), 55935801/2 (14B t32/t48).
  30B blocked on MoE expert keying.

---

## NEXT (rewrite each iteration)

1. Confirm S0 exact + S1 within tolerance. **If S0 moves at all, STOP** — the
   gate fix was not a no-op and nothing downstream is interpretable.
2. Inspect `4B_t48_alloc.json`: band widths, `predicted_err_ratio` per rung,
   per-rung `residue`. Sanity: residues ≪ 0.25, ratios ≤ 1.0.
3. S2 = apply the allocation, full-protocol 4B t48. **This is the headline
   number** — Phase 3 vs 10.774525 at iso-compute.
4. If S2 wins: fan out to the other 7 (model × target) cells.
   If S2 is flat: the 2-band near-even split (down_proj 4608/4536) has little
   leverage — try (a) more bands, (b) deliberately UNEVEN widths so the hot
   band is narrow and can be pushed far, (c) importance-ranked membership
   rather than the current top-m split.
   If S2 loses: suspect the correlation caveat (per-band errors are positively
   correlated because chunks share one RNG prefix, so Σ_b E_b under-estimates
   row error) — switch the solve to a measured joint objective.

## Open questions / threads not being chased
- The deployed **protected-channel path gathers arbitrary channels**, so by the
  equivalence test it runs a ~13–19%-different computation than an unsplit
  reference. Not wrong, but never validated. Separate thread.
- Per-band error curves are measured on ONE (block, window) sample per
  (op, layer-bucket) — thin. Widen if results look noisy.
- Correlation across bands (shared `cum_indicator`/RNG prefix) makes the
  additive error model optimistic. It is a ranking signal; PPL is the arbiter.

---

## ★★★★★ 2026-08-02 — qk REBALANCE IS THE BIG LEVER: −2.05% AT IDENTICAL COST

    4B t48   fp16 10.0445   1.05x boundary 10.546725
    floor: mean 10.772111  sd 0.006264  (n=6 identity configs)

| config | PPL | x_fp16 | z | realized flop |
|---|---|---|---|---|
| parent (deployed)      | 10.774525 | 1.0727 | +0.4 | 49.657 |
| best K-band (2 bands)  | 10.729662 | 1.0682 | −6.8 | 49.54 |
| **qk rebalance α=0.5** | **10.553872** | **1.0507** | **−34.8** | 49.65 |

**−2.05% at IDENTICAL compute** (the transform is score-invariant, so it does
not touch the allocation at all). Five times larger than anything the allocator
produced, and this is α=0.5 — first guess, no search.

Still 0.0071 PPL ABOVE the 1.05x boundary, so it does NOT pass yet.

WHY THIS WORKS, and why nobody found it: qk/av is ~90% of dispatch rows and the
ONLY operator class no PTQ front-end reaches — SmoothQuant and AWQ both stop at
SCLinear, so attention has always run on dynamic symmetric absmax with zero
calibrated component. 4B's post-RoPE per-dim spread is 36.8x (Q) / 81.0x (K), so
a contracted-dim rebalance has a great deal to migrate.

### TWO ROUTES TO CLEAR 1.05x, both launched
1. **COMPOSITION (55946398).** qk acts on ATTENTION, K-bands on LINEARS —
   disjoint operators. If the −0.0449 band gain composes: 10.5090 = **1.0462x**,
   comfortably under. NOT assumed: this project has measured ZERO additivity
   when two levers hit one bottleneck (the v6 stack test), so it needs its own
   cell. These two genuinely do not share a bottleneck, which is the reason to
   expect composition here.
2. **α SWEEP (55946399-402: 0.25/0.4/0.6/0.75).** Synthetic preferred 0.4-0.5.
   Pooling granularity (per-layer / per-kv-group / per-layer-band) is a second
   free axis, untouched.
Plus qk at t32 (55946403), the budget where every lever has paid more.

### BAND COUNT x BUDGET — granularity pays at TIGHT budgets only

| | 2 bands | 8 bands | best |
|---|---|---|---|
| t48 | −0.42% | −0.35% | 2 bands |
| t32 | −1.29% | **−1.55%** | **8 bands** |

At t32 (short streams, large error to redistribute) 8 bands beats 2 by 1.22x.
At t48 the extra fragmentation costs more than the finer allocation buys.
Consistent with per-group MP paying 4x more at t32 overall.

**RETRACTED: "8 bands = 1.6x what 2 bands extracted."** That extrapolated
predicted reconstruction gain across a family σ does not order. At t48 8 bands
was WORSE than 2; at t32 it was 1.22x, not 1.6x.

### σ → PPL, four data points
- within one geometric parameter (hot_frac): **monotone**
- across band counts: **not** monotone, and budget-dependent
- across budgets: **not** monotone (σ said t48 > t32; PPL said t32 4x bigger)
- across models: **held** (asymmetry 2.79x vs 1.29x predicted 4B >> llama8B)

**Rule: σ ranks variants differing in ONE parameter. Every cross-family claim
needs its own PPL cell** — which is the argument for wiring the 32-window
validation objective (15 min, r=0.999 vs full test) instead of spending 1.4h
cells on questions σ cannot answer.

### ★★ 2026-08-02 — σ is PERFECTLY INVERSE to PPL across band count at t48

| bands | predicted err | PPL | rank by pred | rank by PPL |
|---|---|---|---|---|
| 2 (hf0.25)  | 0.9369 | **10.729662** | 3 (worst) | **1 (best)** |
| 8           | 0.9057 | 10.736730 | 2 | 2 |
| per-op <=16 | 0.8978 | 10.756253 | **1 (best)** | 3 (worst) |

Not merely "not monotone" — **exactly reversed**, three for three. Meanwhile at
t32 the SAME ladder runs the RIGHT way (8 bands beat 2 by 1.22x). So the sign
of the granularity effect DEPENDS ON THE BUDGET.

Mechanism (consistent with both): more bands buys finer allocation but costs
fragmentation — more distinct stream lengths perturbing the dispatch metric,
more gathers restarting the fp32 accumulator, and a summed-E_b model that grows
more optimistic as cross-band correlation compounds. At t48 there is little
error left to redistribute, so fragmentation cost dominates. At t32 streams are
short, the error pool is large, and allocation benefit wins.

**PRACTICAL RULE: 2 bands at loose budgets, ~8 at tight ones. Do NOT push band
count on the assumption that finer is better — it is only better where there is
error to redistribute.**

This also means the oracle bound (2 bands captures ~34% of the per-chunk
optimum) describes RECONSTRUCTION headroom that does NOT survive transfer to
PPL at loose budgets. Keep the oracle as an upper bound on the reconstruction
axis only.

## ★★★★★ 2026-08-02 — 4B t48 PASSES 1.05x fp16

    fp16 10.0445    1.05x boundary 10.546725    floor sd 0.006264 (n=6)

| config | PPL | x_fp16 | z | realized flop | |
|---|---|---|---|---|---|
| parent (deployed)  | 10.774525 | 1.0727 | +0.4 | 49.657 | |
| K-bands only (2)   | 10.729662 | 1.0682 | −6.8 | 49.54 | |
| qk only (α=0.5)    | 10.553872 | 1.0507 | −34.8 | 49.65 | 0.007 over |
| **qk + K-bands**   | **10.503276** | **1.0457** | **−42.9** | **49.54** | **PASSES** |

**−2.52% vs the deployed parent at LOWER realized compute (49.54 vs 49.657).**
1.05x was previously reachable only at t64, so this lowers 4B's passing rung a
full step — the restated goal.

### THE LEVERS ARE ADDITIVE (measured, not assumed)
    qk gain    0.220654
    band gain  0.044863
    sum        0.265516  -> predicted 10.509009
    ACTUAL     0.271249  -> measured  10.503276
    excess +0.005733 vs floor sd 0.006264  => ADDITIVE WITHIN NOISE

This matters because the v6 stack test measured ZERO additivity for two levers
hitting one bottleneck. These compose because they act on DISJOINT operator
classes: qk on attention, K-bands on linears. Neither alone clears 1.05x.

### α SWEEP (4B t48) — shallow, monotone, optimum >= 0.75
| α | 0.25 | 0.4 | 0.5 | 0.6 | 0.75 |
|---|---|---|---|---|---|
| PPL | 10.5731 | 10.5570 | 10.5539 | 10.5496 | **10.5480** |

Range 0.0251 across the sweep = 4x the floor sd, so worth tuning — but the
LEVER is worth 0.2207. **The win is the lever, not its hyperparameter.**
Still descending at 0.75; α=0.85/1.0 launched.

### WHY THIS WAS AVAILABLE
qk/av is ~90% of dispatch rows and the ONLY operator class no PTQ front-end
reaches — SmoothQuant and AWQ both stop at SCLinear (verified in the AWQ
ablation notes and again in this recon). Attention has always run on dynamic
symmetric absmax with zero calibrated component. 4B's post-RoPE per-dim spread
is 36.8x (Q) / 81.0x (K): a great deal to migrate, and nothing migrating it.

The allocator axis was near its ceiling (<0.5% remaining range, V20 regressed at
t32 everywhere, threshold moves 0/165 accepts). **The win came from changing the
search VARIABLE, not the search over it.**

---

## ★★★★★ 2026-08-03 — THE PER-ROW AXIS IS NEARLY EMPTY; PER-GROUP IS 24-34x BIGGER

Measured per-(row, chunk) error curves (14 lengths x 72 chunks, one SC call per
(chunk, length) gives every row at once) + EXACT Lagrangian water-fill.
All policies at the SAME mean cost of 32 cycles:

| policy | decisions | total sq err | vs uniform |
|---|---|---|---|
| 0. uniform (one L everywhere) | 1 | 4.9135e7 | 0.0% |
| 1. **per-row — what MP does today** | 512 | 4.8482e7 | **−1.3%** |
| 2. per-chunk | 72 | 3.3698e7 | **−31.4%** |
| 3. **per-(row, chunk) — full space** | 36,864 | 2.7101e7 | **−44.8%** |

**The entire per-row MP program (V10→V20, every generation of threshold/ladder
search) is worth 1.3% squared-error reduction at t32. Per-chunk is worth 31.4%
— 24x more. The full per-group space is worth 44.8%.**

Increment of the full space over today's per-row dispatch: **−44.1%**.
Increment over per-chunk: −19.6%, so ROW-ADAPTIVITY INSIDE A CHUNK still matters.

### ⚠ RETRACTION: "per-group granularity has a hard ceiling of 3.1% PPL"
That came from an oracle that (a) assigned ONE length per chunk shared across
all 512 rows, collapsing a 512x72 space to 72; (b) assumed E_c(L)=A_c/L, a model
fitted from a SINGLE measured length, rather than measured curves; (c) was one
synthetic draw. It bounded one restricted policy class under a fitted model and
I reported it as the ceiling of the idea. **The user pushed back; the user was
right.** Measured properly the axis is an order of magnitude larger.

### WHAT THIS MEANS FOR THE ALGORITHM
My deployed K-bands give −1.73% PPL at t32 while the per-chunk oracle is −31.4%
squared error, so the IMPLEMENTATION is the limit, not the idea. K-bands assign
static per-(op, layer-bucket, band) ladders; the oracle allocates every chunk
independently and adapts per row. **Build per-(row, chunk) dispatch** — design
"B" deferred on day one. It is runtime-feasible: the per-chunk amax is ALREADY
computed as the group quantization scale, so the dispatch statistic is free.

CAVEAT held: these are squared-error reductions on SC partial products, and the
error→PPL transfer has been weak and non-monotone across families. But −44.8% is
an order of magnitude above anything measured so far, so even weak transfer is
material.

## 2026-08-03 — AWQ-BASELINED STACK (the correct reference; controls exact)

    t48 AWQ parent  10.649096  x_fp16 1.0602   (archived 10.6491 — EXACT)
    t32 AWQ parent  11.988601  x_fp16 1.1935   (archived 11.9886 — EXACT)

| cell | PPL | x_fp16 | |
|---|---|---|---|
| t48 AWQ parent | 10.649096 | 1.0602 | |
| t48 + qk α=1.0 | 10.425032 | **1.0379** | passes 1.05x |
| **t48 + qk + K-band** | **10.378313** | **1.0332** | **best; −2.54%** |
| t32 AWQ parent | 11.988601 | 1.1935 | |
| t32 + per-op16 band | 11.816135 | 1.1764 | −1.44% |

**The levers still compose on AWQ, and the result is BETTER than on SmoothQuant
(1.0332 vs 1.0438).** 4B t48 clears 1.05x with margin on the correct baseline.

30B post-RoPE spread at t32 = **48.3x |Q| / 119.1x |K|** — the largest of any
model, and 30B has the worst t32 gap (30.9%). qk cells launched.

## 2026-08-03 — t32 FULL STACK on AWQ: -4.28%, a third of what t32 needs

| 4B t32 (AWQ baseline) | PPL | x_fp16 | vs parent |
|---|---|---|---|
| AWQ parent | 11.988601 | 1.1935 | — |
| + per-op16 K-bands | 11.816135 | 1.1764 | −1.44% |
| **+ qk + K-bands** | **11.475783** | **1.1425** | **−4.28%** |

Both levers compose at t32 as they do at t48 (bands −1.44%, qk −2.84% implied).
**But t32 needs −12.0% and both levers are now spent and swept.** −4.28% is a
third of the way, and there is no remaining tuning on either axis.

This is exactly what the error budget predicted, and it is why the ceiling
measurement is the load-bearing result: today's per-row dispatch sits on a
−1.3% axis while per-(row,chunk) is −44.8%. The K-band implementation converts
only a sliver of the per-chunk axis (−1.73% PPL against a −31.4% squared-error
oracle). **Per-(row,chunk) dispatch is the only lever left with enough headroom
for t32.**

### STATUS OF THE 1.05x GOAL
| model | budget | best so far | passes 1.05x? |
|---|---|---|---|
| 4B | t48 | **1.0332** (AWQ + qk + bands) | **YES** — rung lowered t64 → t48 |
| 4B | t32 | 1.1425 | no; needs −8% more |
| 14B | t48 | 1.0311 (AWQ parent alone) | YES already |
| 30B | t32/t64 | qk cells running (spread 48x/119x, worst gap 30.9%) | — |
| llama8B | any | blocked: no structure on any axis measured | — |

## ⚠ 2026-08-03 — THE SPREAD SCREEN IS REFUTED

| model | post-RoPE Q/K spread | qk effect on AWQ |
|---|---|---|
| 4B  | 36.8x / 81.0x | **−2.10%** (helps) |
| 14B | 39.0x / 62.6x | **+0.91%** (HURTS) |

14B t48: AWQ parent 8.9072 → AWQ + qk α=1.0 **8.987851** (1.0311 → 1.0405).
Nearly identical spread to 4B, OPPOSITE SIGN. Confirmed twice (α=0.5 on
SmoothQuant, α=1.0 on AWQ), so it is not an α artifact.

**I presented that screen as "correctly calling every outcome so far". It had
n=2 and got the third wrong. Retract it as a predictor.**

CONSEQUENCE: llama8B was written off for qk on the strength of this screen
(4.5x/4.1x) WITHOUT ever running a cell. That dismissal is unsupported and is
now being tested (calib launched).

Open question worth a diagnostic rather than another guess: what distinguishes
4B from 14B? Candidates — how close the parent already is to fp16 (14B 1.0311
vs 4B 1.0602, so less headroom and more room to be hurt); how much of qk the
INT mask already routes away (14B captured 29 layers vs 4B 26); GQA group
structure. Do NOT propose another screen without validating it on all four
models first.

## 2026-08-03 — qk TRANSFER, all models on AWQ. The spread screen fails BOTH ways.

| model | budget | AWQ parent | + qk α=1.0 | effect | Q/K spread |
|---|---|---|---|---|---|
| 4B | t48 | 10.6491 | **10.4250** | **−2.10%** | 36.8x / 81.0x |
| 14B | t48 | 8.9072 | 8.9879 | **+0.91%** | 39.0x / 62.6x |
| llama8B | t48 | 7.9200 | 7.9032 | −0.21% | 4.5x / 4.1x |
| 30B | t32/t64 | — | running | — | 48.3x / 119.1x |

**HIGH spread can HURT (14B); LOW spread can HELP (llama8B). The screen has no
predictive signal.** It drove two wrong decisions: llama8B was skipped entirely
on its basis (it gains, modestly), and 14B was predicted to gain (it regresses).

llama8B caveat: −0.21% is ~3x the 4B noise floor but there is NO llama8B AWQ
identity control, so treat as suggestive, not established.

### WHAT ACTUALLY PREDICTS qk BENEFIT — unknown, and worth ONE diagnostic
Candidates, none tested: headroom of the parent (14B is already 1.0311, the
closest to fp16 of any cell, and the only one that regressed); how many qk
slots the INT mask routes away (14B 29 layers captured vs 4B 26); GQA group
count; whether attention or the linears dominate that model's residual error.
**Rule now in force: no screen gets proposed as a tool until validated on all
four models.**

## ★★★★★ 2026-08-03 — 30B t64 PASSES 1.05x. SECOND rung lowered.

    30B t64   AWQ parent 7.7146    x_fp16 1.0624
              + qk α=1.0 7.424072  x_fp16 **1.0224**   −3.77%
              1.05x boundary 7.6244 -> PASSES, with margin

Largest qk gain of any model, from the qk lever ALONE (no bands yet).
**30B's passing rung goes t96 -> t64.**

### qk transfer, all four models, AWQ baseline — FINAL
| model | qk effect | Q/K spread | result |
|---|---|---|---|
| **30B** | **−3.77%** | 48.3x/119.1x | t64 = **1.0224**, rung t96→t64 |
| **4B** | **−2.10%** | 36.8x/81.0x | t48 = 1.0379; 1.0332 with bands, rung t64→t48 |
| llama8B | −0.21% | 4.5x/4.1x | marginal, no rung change |
| 14B | **+0.91%** | 39.0x/62.6x | REGRESSES |

The spread ordering (48.3 → −3.77, 39.0 → **+0.91**, 36.8 → −2.10, 4.5 → −0.21)
puts the SECOND-HIGHEST spread as the only regression. Screen explains nothing.

### GOAL STATUS: two of four models have their rung lowered a full step
| model | rung before | rung now | x_fp16 |
|---|---|---|---|
| 4B | t64 | **t48** | 1.0332 |
| 30B | t96 | **t64** | 1.0224 |
| 14B | t48 | t48 (unchanged; qk hurts it) | 1.0311 |
| llama8B | t128 | t128 (marginal only) | 1.0957 @t48 |

## 2026-08-03 — 30B qk across ALL THREE budgets: monotone in budget tightness

| budget | AWQ parent | + qk α=1.0 | gain | |
|---|---|---|---|---|
| t32 | 1.3034 | 1.1879 | **−8.87%** | |
| t48 | 1.1263 | 1.0634 | **−5.58%** | misses 1.05x by 0.097 PPL |
| t64 | 1.0624 | **1.0224** | −3.77% | **PASSES** |

**Tighter budget → larger qk gain, monotone on three points.** Exactly opposite
to the allocation levers, which pay LEAST where the gap is worst. That is why qk
is the lever for the hard cases and bands are not.

### MoE BAND SUPPORT IS NOW A SIZED, NOT SPECULATIVE, TASK
30B t48 needs another ~0.9% to pass 1.05x. Bands gave −1.44% on 4B t32 — enough
in principle — but **30B CANNOT use bands at all**: it is MoE and the allocator
raises on the expert index by design (`_sc_unit_idx` guard in
mp_kbands_calib.py). Both 30B results are qk ALONE.

Work required: key the band map by (op, block, EXPERT) instead of (op, block).
The protected-channel table already uses that grammar (`op:b<N>:u<M>`), so the
schema supports it; what is missing is the allocator measuring per-expert (each
expert sees a different token subset, so averaging across experts would band
them all off one expert's statistics — which is exactly what the guard prevents).
**Promoted to backlog item 2.** It is the difference between 30B passing at t64
and possibly at t48.

## 2026-08-03 — llama8B: THREE statistics agree it has nothing to exploit

| lever | statistic | measured effect |
|---|---|---|
| K-bands (t32) | band asymmetry **1.29x** (lowest) | **−0.16%** |
| qk rebalance (t48) | Q/K spread **4.5x/4.1x** (lowest) | −0.21% |
| INT mask dose | dose response 4-9x FLATTER than any other model | (recorded in int_ablation) |

Three independent structural statistics agree: llama8B's sensitivity is DIFFUSE,
so every SELECTION-based lever finds nothing to select. Note this is a
within-model consistency, NOT a revival of the refuted cross-model spread screen
— that failed at PREDICTING across models (14B high spread regressed), whereas
this is three different measurements of the same model agreeing.

**Consequence: llama8B needs the REPRESENTATION fix (asymmetric SC + non-pow2
grid), not allocation or operand conditioning.** Do not spend further cells on
selection levers for llama8B.

## ★★ 2026-08-03 — BAND COUNT HAS SATURATED. Row-adaptivity is the missing piece.

4B t32, AWQ baseline:

| bands | PPL | vs parent |
|---|---|---|
| per-op ≤16 | 11.816135 | −1.44% |
| **max 35 (finest legal)** | 11.816309 | −1.44% |

Doubling 16 → 35 moves PPL by **0.00017**, far under the ~0.006 noise floor.
**The band-count axis is EXHAUSTED at t32.**

WHY THIS MATTERS — it isolates the missing ingredient. The ladder was still
climbing earlier (2→8→16 gave −1.19/−1.55/−1.73% on SmoothQuant), which is what
motivated going to 35. It has now flattened while still sitting far below the
per-(row,chunk) oracle (−44.8% squared error). So the gap to the oracle is NOT
about how FINELY the contraction axis is cut — it is about WHAT VARIES:

* K-bands assign STATIC per-(op, layer-bucket) ladders, shared by every row.
* The oracle assigns per-(ROW, chunk) — it adapts to the INPUT.

Saturating on band count while far below the oracle means **row-adaptivity
inside the chunk is the whole remaining gap**. That is precisely what
per-(row,chunk) dispatch adds and what nothing built so far does.

**Do not spend further cells raising band count.** Backlog item 1 (per-(row,
chunk) dispatch) is now the ONLY open route on the allocation axis.

## 2026-08-03 — 4B t32 FULLY CHARACTERIZED; exploratory phase CLOSED

| config (AWQ baseline) | x_fp16 | vs parent |
|---|---|---|
| AWQ parent | 1.1935 | — |
| + bands (per-op ≤16) | 1.1764 | −1.44% |
| + bands (max35) | 1.1764 | −1.44% |
| **+ qk + bands (≤16)** | **1.1425** | **−4.28%  ← best** |
| + qk + bands (max35) | 1.1455 | −4.03% |

max35 is slightly WORSE than ≤16 once qk is present (−4.03 vs −4.28, a 0.03 PPL
gap ≈ 5x the noise floor), so past the optimum finer bands cost a little —
consistent with the fragmentation effect measured at t48.

**−4.28% is the ceiling of everything currently built at t32, against −12%.**
Both levers swept, band count saturated, α swept to its optimum. There is no
remaining TUNING anywhere; every open route is a BUILD:

1. **per-(row,chunk) dispatch** — bands saturated at −1.44% while the oracle is
   −44.8%, and finer cutting has stopped helping, so ROW-ADAPTIVITY is provably
   the entire remaining allocation gap.
2. **MoE band support** — 30B cannot use bands at all; worth ~0.9%, which is the
   difference between 30B passing 1.05x at t64 vs t48.
3. **asymmetric SC + non-pow2 grid** — the error budget's only measured route to
   −12% at t32 (INT5 at MATCHED level count closes 53.8% of the SC-INT gap; a
   zero-point adds 12.9 more).

## 2026-08-03 — 14B REGRESSES ON BOTH LEVERS

| lever | budget | effect |
|---|---|---|
| qk rebalance | t48 | **+0.91%** (worse) |
| K-bands | t32 | **+0.12%** (worse, ~2x noise floor — closer to neutral) |

    14B t32  AWQ parent 9.3097 (1.0777) -> + bands 9.321255 (1.0791)

**14B is the one model where NOTHING built this week helps.** It is also the
model whose parent sits closest to fp16 (1.0311 @t48, 1.0777 @t32).

⚠ I initially wrote "+0.12% helps" — WRONG, a higher PPL is worse. Caught on
re-read. Sign errors on small deltas are easy here; always state the direction
explicitly, not just the signed number.

Headroom is the obvious hypothesis (least room to gain, most room to be hurt by
a perturbation), but it is a hypothesis on n=1 and this run has already refuted
two screens built that way. **Do not act on it without testing all four models.**

Practical consequence: 14B already passes 1.05x at t48 unaided (1.0311), so it
needs nothing from us. Deprioritise it and spend cells on 30B/4B t32, where the
gaps are real.
