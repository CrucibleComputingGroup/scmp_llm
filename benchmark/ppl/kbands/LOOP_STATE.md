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

## ★★★★★ 2026-08-03 — REAL-ACTIVATION CEILING: the synthetic OVERSTATED per-chunk

Same exact water-fill, but on activations captured from the deployed model
(down_proj block 12, via build_sc_model so smoothing/mask/protected match):

| policy | 4B real | 14B real | 4B SYNTHETIC (old) |
|---|---|---|---|
| uniform | 0% | 0% | 0% |
| per-row (today's MP) | −5.1% | −5.2% | −1.3% |
| **per-chunk (what K-bands do)** | **−6.5%** | **−3.9%** | −31.4% |
| **per-(row, chunk)** | **−36.8%** | **−37.6%** | −44.8% |

**The synthetic overstated per-chunk by ~5x** (−31.4% vs −6.5%) because it was
built with strong STATIC channel structure. Real activations have weak
static-in-K structure; nearly all the value is ROW-ADAPTIVITY inside the chunk.
per-(row,chunk) survives the reality check (−37% vs −45%), per-chunk does not.

### THIS EXPLAINS THE 14B REGRESSION
On 14B, per-chunk (−3.9%) is WORSE than per-row (−5.2%). A static chunk
allocation is actively harmful there — which is exactly why K-bands regress 14B
(+0.12% t32, +0.33% t48). Not a mystery any more, and not a tuning failure.

### AND IT SIZES THE REMAINING WORK
K-bands deliver −1.44% PPL because they are a STATIC per-chunk method and
per-chunk is worth only −4 to −6% on real data. The unbuilt input-adaptive
per-(row,chunk) policy is worth −37%, about **6x more**. The search space swept
so far (~30 configs of alpha x band-count x hot_frac) is not too small so much
as the WRONG SPACE: every configuration in it is static.

**Next build is unambiguous: per-(row,chunk) dispatch.** The statistic it needs
(per-chunk amax) is already computed as the group quantization scale, so it is
free at runtime.

## ★★★★★ 2026-08-03 — the GPU kernel ALREADY makes per-(row,chunk) free

I had estimated per-(row,chunk) dispatch would cost ~5x kernel launches. WRONG —
that assumed a per-cycle loop this kernel does not have.

`build_cum_indicator_kernel`: `cum[d,k,v] = |{i<k : rng_b[d,i] <= v}|` — a PREFIX
SUM over the cycle axis. `enable_matmul_tiled_kernel` does ONE O(1) lookup per
(row,col,d); its inner loop runs over D and **never over stoc_len**. Runtime is
independent of stream length (CLAUDE.md: 256->16 changes wall-clock ~4%).

### What that buys
1. **cum table: NO CHANGE.** Prefix sums nest, so the length-L table is the first
   L rows of the L_max=128 table. Verified the one transform that could break
   nesting: `_owen_scramble` is a per-dim XOR mask (position-independent) and the
   rescale is elementwise. Deployed config (halve=1 => grid_levels=128 !=
   base=256) always takes the rescale path, so nesting holds uniformly.
   This SIMPLIFIES current code, which LRU-caches one cum per stoc_len.
2. **k_table (D,V) -> (R,D,V)**, one slice per rung. 5 x 64KB = 320KB, fits L2.
   `k_table[d,ba]` -> `k_table[rung[m],d,ba]`: same load count, per-row offset.
3. **scale scalar -> per-row vector**, `acc *= scale_m[:,None]`.
4. **Per-CHUNK variation is free**: the D-chunk loop is ALREADY host-side
   (`for d_start in range(0,D,chunk_d)` in `_sc_matmul_bipolar_mlp_chunked`),
   one launch per chunk — 76 for 4B down_proj, not 1. Per-chunk rungs are just a
   different slice per iteration.
5. **The dispatch statistic is already computed in the loop**: `scale_a` from
   `fused_quantize_bipolar_perrow(a_chunk,...)` IS the per-(row,chunk) amax.

No extra launches, no atomics, no row sorting, no extra FLOPs. => there is no
wall-clock reason to restrict per-(row,chunk) to selected operators.

## ★★★★★ 2026-08-03 — TWO OPERATOR CLASSES (prediction made, then confirmed)

Predicted from the q_proj outlier: post-LN ops (q/k/v/gate/up) take the
hidden state as input and should carry PERSISTENT CHANNEL outliers (what
SmoothQuant/AWQ target) => static per-chunk structure. Intermediate ops
(o_proj, down_proj) take post-SiLU / post-attention inputs whose outliers are
TOKEN-dependent => row structure. Measured on real activations, t32:

| op (4B) | per-row | per-chunk | per-(row,chunk) | chunk captures |
|---|---|---|---|---|
| k_proj    | -6.6% | -72.2% | -74.5% | **97%** |
| q_proj    | -7.2% | -68.4% | -70.8% | **97%** |
| v_proj    | -3.6% | -55.6% | -59.0% | **94%** |
| gate_proj | -3.5% | -31.9% | -36.8% | **87%** |
| up_proj   | -0.7% | -23.3% | -28.5% | **82%** |
| o_proj    | -2.0% | -15.2% | -40.3% | 38% |
| down_proj | -5.1% |  -6.5% | -36.8% | **18%** |

14B q_proj transfers: -1.2 / -39.2 / -47.4 (chunk captures 83%).
CONFIRMED: post-LN 82-97%, intermediate 10-38%. Clean separation.

### The uncomfortable part
**Per-row is the WRONG AXIS for 5 of the 7 linear operators.** On up_proj it
buys -0.7% where per-chunk buys -23.3%. Today's deployed MP dispatches per-ROW
on all of them. per-(row,chunk) / per-row ratio runs 7x (down_proj) to 41x
(up_proj) -- EVERY linear operator, not just the one bands were tuned on.

### Why K-bands did not collect this
Bands ARE static per-chunk, so they should have paid on q/k/v/gate/up. Open
question, two candidates: (a) band COUNT too coarse -- q_proj has 20 chunks and
the deployed cap gave it far fewer bands than the 20 the oracle used; (b) the
band error curves pool rows, so the allocator may never see channel structure.
Worth checking before assuming per-(row,chunk) is the only route.

## OPERATIONAL HAZARD — editing kernels while jobs run (2026-08-03)

Job 56007115 died with `dynamic_func() missing 2 required positional arguments:
'BLOCK_N' and 'BLOCK_K'` and it was NOT a code bug -- all three launch sites
were consistently patched. Cause: `scmp_kernels` is an EDITABLE install and
`@triton.jit` reads the kernel source LAZILY via inspect at FIRST LAUNCH, not
at import. A job that imported the old caller into memory, then first launched
the kernel after a signature edit landed on disk, binds the NEW signature to
the OLD call. Any kernel-signature edit therefore kills in-flight jobs at an
arbitrary later moment, with an error that reads like a real bug.

Rule: before editing a kernel SIGNATURE, either drain the queue or accept that
running cells must be re-run. Adding kwargs with defaults is safe; adding
POSITIONAL params is not.

## 2026-08-03 — per-(row,chunk) KERNEL BUILT AND VERIFIED (uncommitted)

15/15 tests pass (`benchmark/ppl/kbands/test_per_row_chunk_len.py`, job 56007665).

### Changes
* `scmp_kernels/sc/kernels.py` — `enable_matmul_tiled_kernel` gains
  `rung_ptr`/`row_scale_ptr` + `PER_ROW_LEN: tl.constexpr` (default False, so
  the existing path compiles unchanged); `k_table` becomes an (R,D,V) stack via
  `_get_cached_k_table_stack`; `_sc_matmul_bipolar_mlp_chunked` takes
  `rung_table` (N, n_chunks) + `level_lens`.
* `scmp_kernels/sc/matmul.py` — `rung_table`/`level_lens` on the public
  `sc_matmul`; rejected on every path that would ignore them; **trace emits one
  record per rung** instead of one call-level `stoc_len`.
* `scmp_kernels/mp/config.py` — `per_row_chunk` section + `get_per_row_chunk`.
* `model/sc_common.py` — `per_row_chunk_rungs()` + the SCLinear branch.

### What the tests establish
* Uniform rung r is **BIT-IDENTICAL** to a plain call at L_r, for all 5 rungs
  => the prefix-nesting argument is empirically confirmed, not just argued.
* Mixed per-row is bit-identical to a row-wise reference; per-chunk matches
  summed single-chunk references.
* **Trace prices the allocation**: mean L traced 55.500 == intended 55.500
  (L_max 128). Without this a t32 cell would have been billed at 128.
* 4 guards fire: chunk_d=0, out-of-range rung, wrong-shape table, and the
  unscrambled-L_max case where cum would stop nesting.

### Dispatch is ONE call, not K
The per-row path gathers rows and calls sc_matmul once per rung. The
per-(row,chunk) path passes a rung table and calls ONCE, so it issues FEWER
launches than today's MP while allocating on a 76x finer grid.

### OPEN — the gate on all of this
Policies 1-3 in the ceiling are ORACLES (water-fill on measured error, needs
the FP reference at runtime). Deployment sees only the per-(row,chunk) absmax.
Policy 4 (`amax THRESHOLD`) was added to measure that gap; jobs 56007743/44/55/56
on 4B down/q/up_proj + 14B down_proj. If the proxy captures little of the -37%,
the statistic is wrong and the ladder needs a different one BEFORE any
calibration pipeline is built on it.

## ⚠ 2026-08-03 — the operator-class split is QWEN3-ONLY, not universal

llama8B breaks it. Fraction of the per-(row,chunk) gain that STATIC per-chunk
allocation captures, post-LN ops:

| cell | per-chunk | per-(row,chunk) | chunk captures |
|---|---|---|---|
| 4B q_proj      | -68.4% | -70.8% | 97% |
| 4B k_proj      | -72.2% | -74.5% | 97% |
| 4B v_proj      | -55.6% | -59.0% | 94% |
| 4B gate_proj   | -31.9% | -36.8% | 87% |
| 14B q_proj     | -39.2% | -47.4% | 83% |
| **llama8B q_proj**    | **-9.8%** | -26.1% | **38%** |
| **llama8B gate_proj** | **-3.4%** | -15.5% | **22%** |

So "post-LN ops carry persistent channel outliers" is a **Qwen3-family**
property. Llama-3.1-8B's post-LN ops behave like INTERMEDIATE ops -- their
structure is in rows, not channels. Do NOT write the operator-class rule as
universal.

**This independently explains llama8B's history**: it is the model where every
lever so far came out marginal (qk -0.21%, bands nil). Both levers allocate on
axes llama8B does not carry its structure on. per-(row,chunk) is the first
lever that reaches llama8B's actual axis -- and it is still the weakest model
(-15.5 to -26.1% vs 4B's -28.5 to -74.5%), so expect a smaller win there.

## ⚠ 2026-08-03 — CONFOUND: the ceiling was measured on SMOOTHQUANT activations

`capture_real` loads the mp_best parents, which are SmoothQuant. The DEPLOYED
baseline is **AWQ + INT7 20%**. AWQ exists to migrate per-channel activation
outliers into the weights -- exactly the structure the post-LN per-chunk gain
feeds on. So the -55% to -72% per-chunk numbers on q/k/v may be materially
OVERSTATED on the baseline that actually matters, and the operator-class split
could be partly an artifact of the weaker front-end.

Same failure class as the synthetic overstating per-chunk 5x: measure on the
thing you will deploy on. `--frontend awq` added (KB_FRONTEND=awq); the
post-LN cells must be re-measured there before the operator map is trusted.

Prediction to test: under AWQ the per-CHUNK column shrinks a lot on q/k/v/gate/up
and much less on down_proj/o_proj (whose outliers are token-dependent, so AWQ's
per-channel scales cannot reach them). If per-(row,chunk) holds up while
per-chunk collapses, the case for ROW-adaptivity gets STRONGER, not weaker.

### full operator x model map (SmoothQuant capture, t32) — per-(row,chunk) vs uniform
| | 4B | 14B | llama8B | 30B |
|---|---|---|---|---|
| q_proj    | -70.8% | -47.4% | -26.1% | -26.0% |
| k_proj    | -74.5% | | | |
| v_proj    | -59.0% | | | |
| gate_proj | -36.8% | | -15.5% | -27.8% |
| up_proj   | -28.5% | | | |
| o_proj    | -40.3% | (rerun) | | |
| down_proj | -36.8% | -37.6% | -21.4% | -19.2% |
Per-row (today) on the same cells: -0.3% to -7.2%. Every cell favours
per-(row,chunk); the SPREAD across models is 15-75%, so gains will be uneven.

## ★★★★★ 2026-08-03 — GATE PASSED: the FREE statistic captures 94-99% of the oracle

Policies 1-3 are oracles (water-fill on MEASURED error => needs the FP16
reference at runtime). Policy 4 is what can actually be deployed: a threshold
on the per-(row,chunk) absmax, min-max normalized per call. Rank-matched to the
oracle's length histogram, so cost is identical and only ORDERING is tested.

| cell (t32) | per-row | per-(row,chunk) ORACLE | amax THRESHOLD | captures |
|---|---|---|---|---|
| 4B q_proj      | -7.2% | -70.8% | **-70.1%** | **99%** |
| 14B down_proj  | -5.2% | -37.6% | **-36.5%** | **97%** |
| 4B down_proj   | -5.1% | -36.8% | **-35.2%** | **96%** |
| 4B up_proj     | -0.7% | -28.5% | **-26.8%** | **94%** |

The per-chunk absmax IS the group quantization scale, so this needs NO extra
reduction -- unlike l2 or crest. The oracle is therefore nearly REACHABLE, and
the calibrator's job reduces to picking thresholds that reproduce the
water-filled length histogram (the same counts->thresholds step the per-row
calibrator already does).

Also in: 14B o_proj -42.7% per-(row,chunk) vs -2.5% per-row (17x).

⚠ BUG in those logs: the "captures N%" line prints garbage (e.g.
-35182316308920%) because `max(t3/base-1, 1e-12)` takes the max of a NEGATIVE
ratio and 1e-12, so it divides by 1e-12. Fixed to `max(abs(...), 1e-12)` in the
current script. The raw error columns are unaffected and are what the table
above is computed from.

## ★★★ 2026-08-03 — AWQ confound REFUTED + the simplest statistic WINS

### (a) AWQ vs SmoothQuant: no material difference (my concern was wrong)
| cell | SQ per-chunk | AWQ per-chunk | SQ per-(row,chunk) | AWQ per-(row,chunk) |
|---|---|---|---|---|
| 4B q_proj    | -68.4% | -68.6% | -70.8% | -70.6% |
| 14B q_proj   | -39.2% | -39.0% | -47.4% | -47.3% |
| 4B down_proj |  -6.5% |  -6.6% | -36.8% | -37.4% |
Within 0.8pp everywhere. The channel structure per-chunk allocation exploits is
NOT the structure AWQ removes. The ceiling and the operator map both hold on the
deployed baseline. Concern raised, tested, retracted.

### (b) DEPLOYABLE-RULE BAKE-OFF — plain amax wins, "compose both" is REFUTED
Fraction of the per-group oracle captured (7 cells):

| cell | amax | amax x \|\|w_c\|\| | static chunk + row shift |
|---|---|---|---|
| 4B q_proj       | **99%** | 99% | 98% |
| 14B q_proj      | **98%** | 98% | 90% |
| 4B down_proj    | **96%** | 96% | **41%** |
| 14B o_proj      | **94%** | 95% | **47%** |
| llama8B q_proj  | **94%** | 94% | **57%** |
| llama8B down    | **92%** | 92% | **44%** |
| 4B o_proj       | **91%** | 92% | **45%** |

* **amax alone is the answer.** 91-99% everywhere including llama8B.
* The static weight norm adds NOTHING (<=0.3pp) -- drop it, keep the statistic
  free.
* **My "static chunk base + row shift" design (policy 6) is REFUTED.** It was
  motivated by the operator map (compose the channel and row structures) and it
  collapses to 41-47% on the intermediate ops. Composing the two axes with a
  fixed form is WORSE than letting one metric rank all pairs freely. Do not
  revisit without new evidence.

=> statistic question CLOSED: threshold on the per-(row,chunk) absmax, min-max
normalized per call. Free at runtime (it IS the group quantization scale).

## PLANNED (needs OK) — first full-protocol per-(row,chunk) PPL wave

Everything so far is SQUARED ERROR on partial products. That is an upper bound
on the allocation axis and this ledger already records four claims that died
exactly at the error->PPL step. Nothing is claimed until these run.

Proposed wave (8 cells, full protocol, PPL_MAX_TOKENS=0, ctx 2048, B=1):

| cell | model | target | config | control |
|---|---|---|---|---|
| 1-4 | 4B / llama8B / 14B / 30B | t32 | prc thresholds, AWQ + INT7 20% | AWQ parent |
| 5-8 | 4B / llama8B / 14B / 30B | t48 | same | AWQ parent |

* Baseline: **AWQ + INT7 20% mask** (the deployed one), parent PPL from
  `hpca_results/llm/frontend_awq/mpbest_awq_vs_smoothquant.csv`.
* Compare S1->S2 (identity child vs allocated child), NOT S0->S2. On 4B t48,
  83% of the naive parent-to-result delta was numerics, not allocation.
* Noise floor sd 0.0069 PPL (n=6 identity controls); a delta under ~0.014 is
  not a result.
* 30B needs MoE support first (expert-keyed groups) or it drops to 6 cells.
* Cost check: every cell must report realized MAC-weighted mean length <= its
  target, from the TRACE (which now prices per rung, not at L_max).

### Known inefficiency in the calibrator (not a correctness bug)
`--windows` caps captures PER (op, bucket) KEY, but a single forward already
supplies ~9 blocks per bucket, so keys fill during forward 1 and the remaining
forwards do no work. Wastes wall-clock, does not bias the result beyond
sampling the first ~4 blocks of each bucket. Fix with an early break when every
key is full.

## ⚠ 2026-08-03 — the GLOBAL water-fill confounds the experiment; parent budgets are the default now

The first calibrator run hit its budget exactly (32.00 vs 32.0, 28 buckets) but
its allocation is not the experiment I want to run:

  o_proj    l0  hist=[32768,0,0,0,0,0]  mean_L=18.00   <- ENTIRE bucket at floor
  k_proj    l0  hist=[20062,338,80,0,0,0] mean_L=18.17
  q_proj    l3  hist=[72,7,1988,9542,5540,3331] mean_L=58.63
  down_proj l3  mean_L=42.16

One global multiplier over RAW squared error re-decides the CROSS-LAYER split
at the same time as the granularity. Raw squared error is not comparable across
operators -- a layer whose output has larger magnitude shows larger error
regardless of sensitivity -- so this starves early layers and floods late ones.
That is the same failure mode the sigma/measured cross-layer history already
hit, and it would confound the first PPL cell: a loss could not be attributed to
granularity.

**FIX: `--budget-mode parent` (now the DEFAULT).** Each (op, bucket) is held at
the PARENT's realized mean length, read from the DEPLOYED resolver
(`adaptive_classify_rows` + `_mp_dispatch_metric`, never a reimplementation --
two resolvers disagreeing is a four-instance bug family here), and the
water-fill redistributes only WITHIN the group. Then:
  * granularity is the ONLY variable that changes;
  * the parent is INSIDE the search space (per-row == per-(row,chunk) with all
    of a row's chunks sharing its rung), so the refinement cannot lose by
    construction -- the property that made K-bands safe.
`--budget-mode global` is retained for a later, separate cross-layer study.
It now REFUSES to fall back silently: no sc_mp_config => hard error.

## ★★ 2026-08-03 — BUDGET DEPENDENCE: per-(row,chunk) pays at EVERY budget

| cell | t32 | t48 | t64 | t96 |
|---|---|---|---|---|
| 4B down_proj ORACLE      | -36.8% | -39.3% | -38.5% | -29.4% |
| 4B down_proj DEPLOYABLE  | -35.2% | -37.8% | — | -28.6% |
| 14B down_proj ORACLE     | -37.6% | — | -39.7% | — |
| 14B down_proj DEPLOYABLE | -36.5% | — | -38.9% | — |
| 4B q_proj ORACLE         | -70.8% | — | (t64 running) | — |

Roughly FLAT at -29% to -40% across a 3x budget range, peaking mid-budget and
easing slightly at t96. Deployable tracks the oracle to within 0.8-1.6pp at
every budget.

**This profile differs from BOTH earlier levers** and that is what makes it
useful: qk pays MORE at tight budgets (30B -8.87/-5.58/-3.77% at t32/48/64),
the allocation levers pay LESS, and this one pays everywhere. The four models
need help at different rungs (4B t48, llama8B t128, 14B t40, 30B t64), so a
lever that only worked at one end could never move all of them.

Policy 6 (static chunk + row shift) stays refuted at every budget
(-18.7% to -25.4% vs deployable amax's -28.6% to -37.8%).

## 2026-08-03 — parent-budget calibration WORKS; two caveats recorded

4B t32 with `--budget-mode parent` (job 56009322):
  down_proj l0 hist=[42357,9788,10179,7843,4806,2851] mean_L=29.47
  gate_proj l1 hist=[19372,478,445,168,17,0]          mean_L=18.75
  o_proj    l3 hist=[4091,5382,6671,5550,4616,6458]   mean_L=49.13
Mean lengths span 18.75-49.13 and every bucket keeps a spread across rungs,
versus the global solve's 16.00-86.70 with three buckets pinned entirely at the
floor. This is the allocation to test.

### CAVEAT 1 — it UNDERSPENDS: MAC-weighted mean 28.54 vs the parent's 32
The per-group water-fill takes the largest allocation with mean <= the parent's
own budget, and it refuses to buy length where a pair's error curve has gone
flat. The 3.46 unused cycles cannot be redistributed without reopening the
cross-layer question this mode exists to hold fixed. For the first experiment
this is FAVOURABLE (a win at 28.54 vs a parent at 32 is a win at LOWER cost),
but the realized number must be read from the TRACE at eval time, never assumed.

### CAVEAT 2 — ESCAPE-GATE PARITY (found before it could bite)
`adaptive_classify_rows` applies `_apply_escape_gate` INTERNALLY, so the
per-group budgets read from it already include gate spending. The child branch
has no gate, so it would have dropped a deployed feature (every parent runs
k=2.0, len=128) AND undershot the budget it was matched to -- surfacing as
"cheaper and worse" rather than as a bug. Fixed by appending escape_stoc_len as
a top rung when the parent has a gate, so the child can reach the same length
through the ordinary threshold mechanism with no new runtime machinery.
NOTE: the gate params live in wrapper.json, NOT table.json -- the calibrator
reads `table`, so this must be re-pointed before the parity fix actually fires.

## ══ CONSOLIDATED OPERATOR MAP (20 cells, t32, real activations) ══

per-(row,chunk) vs uniform at equal mean cost; (deployable amax rule in parens);
per-row = what deployed MP does today.

| op | 4B | 14B | llama8B | 30B |
|---|---|---|---|---|
| k_proj    | -74.5% | | | |
| q_proj    | -70.8% (-70.0) | -47.4% (-46.4) | -26.1% (-24.6) | -26.0% |
| v_proj    | -59.0% | | | |
| o_proj    | -40.3% (-36.8) | -42.7% (-40.1) | -21.8% (-18.6) | -45.2% (-41.8) |
| down_proj | -36.8% (-35.2) | -37.6% (-36.5) | -21.4% (-19.7) | -19.2% (-17.2) |
| gate_proj | -36.8% | | -15.5% | -27.8% |
| up_proj   | -28.5% (-26.8) | | -15.3% (-13.7) | -28.0% (-25.7) |
| **per-row (today), same cells** | **-0.7 to -7.2%** | **-1.2 to -5.2%** | **-0.1 to -2.4%** | **-0.1 to -8.3%** |

**20/20 cells favour per-(row,chunk).** Deployable rule captures 85-99%
(median ~93%). Ratio to per-row runs 5x to 41x.

Model ordering is consistent: 4B > 14B ~ 30B > llama8B. llama8B has both the
weakest gain AND the flattest per-row baseline -- it is the model to watch.

### STATUS OF THE BUILD
DONE: kernel (15/15 tests), sc_matmul API, mp config `per_row_chunk`, SCLinear
branch, calibrator with parent budgets + escape-gate parity.
NOT DONE: any PPL measurement. MoE (30B) still unsupported by the calibrator.

## ⚠⚠ 2026-08-03 — CALIBRATOR BUG: three cells came out ABOVE the parent's cost

Parent realized_flop_avg_sl (mp_best_all.csv) vs the child's calibrated mean:

| cell | parent | child (par2) | |
|---|---|---|---|
| 4B t32      | 33.73 | 28.54 | under ✓ |
| llama8B t32 | 34.00 | **38.09** | **OVER 12%** |
| llama8B t48 | 49.62 | **52.03** | OVER 4.9% |
| 14B t32     | 33.61 | **34.68** | OVER 3.2% |

The per-group cap should make an overspend impossible, so the accounting was
wrong. TWO causes, both mine:

1. **Protected channels.** SCLinear dispatches on the RESIDUAL -- protected
   channels are split off and run at protected_stoc_len OUTSIDE the MP ladder.
   The calibrator captured the FULL input, so (a) `_mp_dispatch_metric` saw a
   different vector than the deployed path and produced a different parent
   assignment, and (b) the allocator priced channels that are not its to spend.
   FIXED: the hook now removes protected channels before metric and error
   curves, exactly as the runtime does.
2. **Not comparable anyway.** The printed mean covers LINEARS ONLY; the
   parent's realized_flop_avg_sl also includes qk/av. Now labelled as such.

Added a hard guard: any group whose allocated mean exceeds its parent budget
aborts the calibration rather than emitting a non-iso-cost table.

**The binding cost check remains the child's TRACE at eval time vs the
parent's** -- the trace now prices per rung, so it cannot flatter itself.
Had the wave launched on the par2 tables, llama8B t32 would have run 12% over
budget and any win would have been unattributable.

## ★ 2026-08-03 — calibration is now ISO-COST BY CONSTRUCTION (ratio 1.0000)

The "14B overspends 16%" reading was MY ACCOUNTING, not the allocation. The
child's mean covers LINEARS ONLY; mp_best's realized_flop_avg_sl also covers
qk/av, and the allocator deliberately runs attention short — so a compliant
child looks like a large overspend against that column.

Fix: aggregate the PARENT's own per-group means with the SAME mac x rep
weights and compare on that basis. Result:

  4B t48      child 46.38  parent 46.38  -> 1.0000
  llama8B t48 child 47.73  parent 47.73  -> 1.0000

The per-group water-fill realizes each group's parent budget to 4 dp (the mean
is near-continuous in lambda with 10k-70k pairs per group), so the child is
iso-cost BY CONSTRUCTION, not by luck. `achieved` is computed from the ACTUAL
allocation and `parent_lin` from the targets — two independent computations, so
1.0000 is a real check, not a tautology.

Hard guard added: child > parent on the same basis => abort, no table emitted.

### RULE FOR ANY FUTURE COST CLAIM
Never compare a linear-only number against `realized_flop_avg_sl`. Either
aggregate both on the same operator set, or read both from the TRACE (which now
prices per rung and cannot flatter a per-(row,chunk) cell).

## ★★ 2026-08-03 — MoE IS UNBLOCKED by per-(row,chunk) (K-bands never were)

30B calibration RUNS and emits 28 buckets (jobs 56010567/68). K-bands could not
touch MoE at all -- `get_k_bands` raises on the expert index -- and that is why
every 30B post-HPCA cell is qk-only.

per-(row,chunk) is not blocked because it does NOT key on expert: all experts of
a block share the (op, layer-bucket) entry, which is also the right granularity
(128 experts x 36 blocks of separate ladders would be absurd, and the runtime
resolver `get_per_row_chunk(op, block, total_blocks)` has no unit dimension).
The one MoE-specific hazard -- an expert receiving ZERO tokens under sparse
top-k, where min()/max() over an empty dim raises, the same routing case that
broke the per-row calibrator -- is guarded in `per_row_chunk_rungs`.

This matters beyond 30B: it means the wave can be 8 cells, and it removes the
standing "30B cannot use per-group allocation" limitation entirely.

## 2026-08-03 — ALL DENSE CALIBRATIONS ISO-COST; the 14B "overspend" was accounting

| cell | child linear-only | parent linear-only | ratio |
|---|---|---|---|
| 4B t32      | 31.96 | 31.96 | 1.0000 |
| 4B t48      | 46.38 | 46.38 | 1.0000 |
| llama8B t32 | 33.38 | 33.38 | 1.0000 |
| llama8B t48 | 47.73 | 47.73 | 1.0000 |
| 14B t32     | 38.97 | 38.97 | 1.0000 |
| 14B t48     | 52.47 | 52.47 | 1.0000 |

14B's parent linear-only mean is **38.97**, not the 33.61 in
`realized_flop_avg_sl` (which includes qk/av, run short by the allocator). The
"16% overspend" was entirely the wrong denominator. Tables at
`$TURBO/kbands/kbands_20260801/prc/*_par4_prc.json`.

### READY, AWAITING OK — 8-cell full-protocol wave
4B / llama8B / 14B / 30B x {t32, t48}, PPL_MAX_TOKENS=0, ctx 2048, B=1,
AWQ + INT7 20% parent, S1->S2, noise floor sd 0.0069, realized cost from the
TRACE. ~20-30 GPU-hours. t64 tables calibrating now so the wave can widen
without re-deriving anything (30B's current passing rung is t64).

## ⚠⚠ 2026-08-03 — the WIRING CHECK caught a bug that would have killed all 8 cells

`wrapper.json` stores `threshold_table_path` RELATIVE ("table.json"). The
emitted per-(row,chunk) table lives in a different directory
($TURBO/kbands/.../prc/), so the loader resolved it against the WRONG dir:

  FileNotFoundError: .../kbands_20260801/prc/table.json

Every wave cell would have died at load. FIXED: the calibrator now rewrites
`threshold_table_path` ABSOLUTE against the parent bundle and verifies the file
exists before emitting; 23 already-written tables were patched.

Note this was NOT the failure the wiring check was designed for (that was the
silent per-row fallback). It paid for itself on a different bug in its first
run -- which is the argument for running a cheap end-to-end load test before any
multi-hour wave, not just unit tests of the pieces.

## ⚠⚠⚠ 2026-08-03 — ROOT CAUSE: per_row_chunk was written to the WRONG FILE

The wiring check reported exactly the failure it was built for:
  FAIL  AdaptiveMPConfig parsed a per_row_chunk section  -- 0 entries
  FAIL  no SCLinear falls back to per-row dispatch -- 0 resolved, 224 MISSED

`_load_per_row_chunk` is called from `AdaptiveMPConfig.load_threshold_table(path)`,
so its payload is **table.json**, NOT wrapper.json. The calibrator wrote the
section into the WRAPPER. `k_bands` lives in the table for the same reason —
the precedent was right there and I did not follow it.

Both earlier symptoms share this one root cause. The "fix" of pointing
`threshold_table_path` at the PARENT table made it strictly worse: that table
has no per_row_chunk, so the config loaded cleanly with zero entries.

**What this would have produced without the wiring check:** 8 cells running to
completion, at the parent's exact cost, reporting the parent's PPL +/- noise.
A clean, plausible, completely null table -- which I would have read as
"per-(row,chunk) does not transfer to PPL" and used to retire a direction that
20 measured cells support. This is the single most dangerous failure mode in
this whole line of work, and it was silent by construction.

FIX: emit TWO files — `<name>_v5_prc_table.json` (parent table + per_row_chunk)
and `<name>_v5_prc.json` (wrapper with an ABSOLUTE threshold_table_path). The
calibrator now LOADS THE EMITTED TABLE BACK through the deployed parser and
aborts unless the bucket count round-trips.

RULE: never trust that an emitted config is read. Load it back through the
deployed loader and assert the feature is live before spending GPU-hours.

## 2026-08-03 — the config round-trip, and why it took four attempts

The round-trip verification (load the emitted table back through the DEPLOYED
parser, abort unless the buckets survive) found three further problems, all in
the emit/verify path, none in the runtime:

1. **Section in the wrong file** (the real one). `_load_per_row_chunk` runs
   inside `load_threshold_table(path)`, so it parses table.json. The calibrator
   wrote the section into wrapper.json => 0 entries parsed, 224 SCLinears
   falling back to per-row, a perfectly plausible NULL result. `k_bands` lives
   in the table for exactly this reason.
2. **Validator constructor**: `AdaptiveMPConfig(stoc_len_levels)` is required
   and takes the ladder DESCENDING; the per_row_chunk ladder is ASCENDING by
   design (rung index IS the position, and the runtime maps bucketize() output
   straight onto it). Two lists, two conventions — do not harmonise them.
3. **Validator ladder**: constructing with the per-BUCKET ladder trips
   `table levels != runtime levels`, because the per_row_chunk ladder may be
   LONGER (it appends the escape length as a top rung). Verified the runtime is
   unaffected: wrapper and table both carry the parent's 6-rung
   `stoc_len_levels`, they agree at load, and each bucket's 7-rung ladder is
   validated on its own by `_load_per_row_chunk` (cap, ascending, threshold
   count) and never cross-checked against `stoc_len_levels`.

llama8B t48 passed all along because its parent ladder already contained 128,
so no escape rung was appended and 6 == 6 — a reminder that a single passing
cell proves nothing about the others.

## ★★ 2026-08-03 — WIRING GATE PASSES: the dispatch is LIVE in a real model

  4B t32      28 buckets parsed, **252/252 SCLinear resolved, 0 fallbacks**
  llama8B t32 28 buckets parsed, **224/224 SCLinear resolved, 0 fallbacks**
  logits finite; 8 distinct stream lengths in use (ladder + escape rung 128)
  traced linear-only mean: 4B 28.81 (budget 31.96), llama8B 32.00 (budget 33.38)

Both UNDER budget — the water-fill declines to buy length where a pair's error
curve has gone flat, and the protected slices run at their own length.

The one FAIL was MY TEST, not the deployment: it asserted every linear record
carries d_in == 128. Legitimate other widths exist — the residual TAIL chunk
(residual width is not a multiple of 128 once protected channels are removed)
and the PROTECTED slices (which run at protected_stoc_len outside the MP ladder
entirely). Replaced with a MAC-SHARE assertion (>50% of linear MACs at
d_in=128), which is what "per-(row,chunk) is live" actually means. The printed
d_in list was also truncated to the 6 smallest values, so it could not have
shown 128 even when present — a misleading diagnostic on top of a wrong test.

### ALL 8 TABLES READY
$TURBO/kbands/kbands_20260801/prc/{4B,llama8B,14B,30B}_t{32,48}_v7_prc.json
(+ _prc_table.json). Every one: iso-cost child/parent = 1.0000, round-trip
verified through the deployed parser, escape-gate parity, protected channels
excluded from the allocation.

## ══ FINAL PRE-WAVE STATE (2026-08-03) ══

### Wiring gate — dispatch is LIVE, zero fallbacks, every model
| cell | SCLinear resolved | traced linear-only mean | budget |
|---|---|---|---|
| 4B t32      | 252/252 | 28.81 | 31.96 |
| 4B t48      | 252/252 | 42.02 | 46.38 |
| llama8B t32 | 224/224 | 32.00 | 33.38 |
| 14B t32     | 280/280 | 36.30 | 38.97 |
| 14B t48     | 280/280 | — | 52.47 |
| **30B t32** | **18624/18624** | — | — |

llama8B t32 returns ALL PASS incl. **96.0% of linear MACs priced at d_in=128**.
Every cell is UNDER budget (the water-fill declines to buy length where a
pair's error curve is flat; protected slices run at their own length).
30B's 18,624 resolved modules are every expert projection across 128 experts —
MoE is not merely calibrating, it is fully DISPATCHING.

### Wave launcher: benchmark/ppl/kbands/run_prc_ppl.sbatch
* Runs BOTH arms (KB_ARM=prc | parent) so the comparison is S1->S2 in ONE
  environment. Comparing against the ARCHIVED parent number would fold in every
  environment change since that run.
* REFUSES a `prc` cell whose table has zero per_row_chunk buckets — the exact
  bug that would record a per-ROW run under the child's name.
* Environment mirrors the archived runtime.json exactly (bitrev / masks 64 /
  sc_prec 8 / halve / INT7 sym / chunk 128) + FRONTEND=awq + PPL_MAX_TOKENS=0.

### THE ONLY UNTESTED CLAIM
Whether a 5-41x improvement in allocation quality moves PPL. Everything above is
squared error on partial products; this ledger records FOUR earlier claims that
died at exactly that transfer. Needs an explicit OK: 8 cells + 8 parent
controls, ~40-60 GPU-hours.

## ══ 8/8 WIRING GATE: ALL PASS (2026-08-03) ══

| cell | MACs at d_in=128 | traced mean | budget |
|---|---|---|---|
| 4B t32      | 94.8% | 28.81 | 31.96 |
| 4B t48      | 94.8% | 42.02 | 46.38 |
| llama8B t32 | 96.0% | 32.00 | 33.38 |
| llama8B t48 | 96.1% | 45.43 | 47.73 |
| 14B t32     | 95.0% | 36.30 | 38.97 |
| 14B t48     | 95.0% | 50.07 | 52.47 |
| 30B t32     | 92.0% | 35.78 | (MoE) |
| 30B t48     | 92.0% | 52.66 | (MoE) |

Zero fallbacks on any model. Every cell UNDER budget. 30B dispatches
18,624/18,624 expert projections. The pipeline is DONE up to the PPL question.

### The full self-correction ledger for this build
REFUTED (mine): AWQ confound (<=0.8pp), amax x ||w_c|| statistic (<=0.3pp),
static-chunk + row-shift composite (41-47% vs amax 91-96%).
BUGS FOUND (all looked like plausible numbers with satisfied budgets):
 1. global water-fill silently re-deciding the cross-layer split
 2. escape-gate omission (parent budgets include gate spend)
 3. protected channels priced by the allocator + wrong dispatch metric
 4. linear-only vs all-op cost basis (phantom 16% overspend)
 5. relative threshold_table_path (would have killed all 8 cells at load)
 6. **per_row_chunk written to the WRAPPER; the parser reads the TABLE**
 7. round-trip validator constructor arity
 8. round-trip validator ladder (per-bucket vs stoc_len_levels)
 9. wiring-test d_in assertion too strict (tail chunks + protected slices)
10. wiring-test diagnostic read the wrapper, and truncated to 6 values
Every one was caught by checking against something INDEPENDENTLY RECORDED —
the archived parent cost, the deployed parser, the trace. #6 is the one that
mattered: it would have produced a flawless NULL 8-cell table.

## ══ BUILD COMPLETE — BLOCKED ON APPROVAL (2026-08-03) ══

16 tables ready: {4B, llama8B, 14B, 30B} x t{32,48,64,96}, at
$TURBO/kbands/kbands_20260801/prc/<model>_t<target>_v7_prc.json
Every one: iso-cost child/parent = 1.0000, round-trip verified through the
deployed parser, escape-gate parity, protected channels excluded.

12/12 gated cells ALL PASS (t32/t48 all four models, plus t64). 92-96% of
linear MACs priced per 128-chunk, zero per-row fallbacks anywhere, every cell
under budget, 30B dispatching 18,624/18,624 expert projections.

### THE LOOP HAS NO SUBSTANTIVE WORK LEFT WITHOUT THE WAVE
Everything measurable short of PPL is measured. Further gating of tables that
may never be used would be manufacturing activity, not doing work — the GPU
floor rule exists to avoid wasting capacity, not to consume it. Leaving GPUs
idle is the correct state here.

NEXT (needs explicit OK): benchmark/ppl/kbands/run_prc_ppl.sbatch
  for m in 4B llama8B 14B 30B; do for t in 32 48; do
    for arm in parent prc; do
      KB_MODEL=$m KB_TARGET=$t KB_ARM=$arm sbatch run_prc_ppl.sbatch
  done; done; done
16 jobs, ~40-60 GPU-hours. Read PPL from eval_summary, cost from the trace,
compare S1->S2 against sd 0.0069.

## ══ PPL WAVE LAUNCHED 2026-08-03 (user OK) ══

16 jobs 56015451-56015466: {4B, llama8B, 14B, 30B} x t{32,48} x {parent, prc}.
Full protocol (PPL_MAX_TOKENS=0, ctx 2048, B=1), AWQ + INT7 20%, tables
`<model>_t<target>_v7_prc.json`. Both arms run in the SAME environment so the
comparison is S1->S2; the archived parent number is NOT the control.

### How to read the result when it lands
* PPL from each cell's eval_summary; realized cost from its TRACE (which prices
  per rung, so a per-(row,chunk) cell cannot flatter itself).
* Noise floor sd 0.0069 PPL (n=6 identity controls). A delta under ~0.014 is
  NOT a result.
* A prc cell must also come in at or under its parent arm's traced cost. All 8
  tables calibrated to child/parent = 1.0000 and every wiring gate traced UNDER
  budget, so an overspend would mean something changed at eval time.
* **If the deltas are ~0**: check the trace for d_in==128 share before
  concluding anything. A silent per-row fallback produces exactly that, and it
  is the failure this whole verification chain exists to exclude. The launcher
  refuses a table with 0 buckets, but check the trace anyway.

## ⚠⚠⚠ 2026-08-03 — PPL WAVE: the lever TRANSFERS, but half the cells OVERSPEND

4/4 completed pairs improved, all far outside the noise floor. But the cost
check splits them:

| cell | PPL delta | parent LIN | child LIN | ratio | verdict |
|---|---|---|---|---|---|
| 4B t48      | -0.37% | 42.20 | 40.45 | **0.959** | valid: better AND 4% cheaper |
| llama8B t48 | -0.24% | 43.49 | 41.12 | **0.945** | valid: better AND 5% cheaper |
| llama8B t32 | -2.18% | 28.95 | 29.20 | 1.009 | INVALID: +0.9% compute |
| 14B t32     | -1.63% | 29.93 | 32.26 | **1.078** | INVALID: +7.8% compute |

**The two LARGE wins are partly PURCHASED. The two ISO-COST wins are SMALL.**

### ROOT CAUSE (mine): the parent budget came from a 256-row SAMPLE
Calibration's per-group parent target vs the parent's TRUE realized linear cost:
  4B t32   31.96 vs 26.09  (-18.4%)   4B t48   46.38 vs 42.20  (-9.0%)
  llama8B t32 33.38 vs 28.95 (-13.3%) llama8B t48 47.73 vs 43.49 (-8.9%)
  14B t32  38.97 vs 29.93  (-23.2%)
The sample OVERESTIMATES the parent by 9-23%, so every child was calibrated
against a too-generous budget and at t32 (where the parent runs leanest) it
spent the difference. I applied "read the cost from the TRACE, not from a
promise" everywhere except to the budget itself.

### FIX (parent traces now exist for these cells)
Recalibrate per-group targets from the PARENT ARM's TRACE, not from a
calibration sample, then re-run the affected cells. Expect the -2.18% and
-1.63% to SHRINK — they were partly bought.

### vs the ARCHIVE — still behind, as expected without qk
4B t48: archive qk+kband **10.3783 (-2.54%)** vs prc-only 10.6093 (-0.37%).
llama8B t48: archive qk 7.9032 (-0.21%) vs prc 7.9011 (-0.24%) — TIED
(0.0021 gap, noise floor 0.0138).
Control: the parent arm reproduced the archived AWQ parent EXACTLY on both
cells (10.6491, 7.9200), so the environment is right and the comparison sound.

### AND: on 4B t48, prc did NOT beat static K-BANDS
bands -0.44% vs prc -0.37% — indistinguishable. The 6x error-axis advantage of
per-(row,chunk) over per-chunk did NOT transfer on that cell. The error metric
overstates what the allocation is worth in PPL, which is the fifth instance of
that pattern in this ledger.

## ★★★★★ 2026-08-04 — PPL RESULT, cost-adjusted and CROSS-VALIDATED

The binary iso-cost gate was DISCARDING REAL SIGNAL (user's call: +0.4% cost
"doesn't matter at all", and the archive itself tolerates drift — 30B t48
realizes 51.1 against a nominal 48). Attribute the gain instead, using measured
local PPL-vs-compute sensitivity (llama8B has two prc runs at different cost,
giving the slope directly: 0.28 %PPL per %compute; wide-range parent slopes are
1.47x flatter, so scale the other models').

### ALLOCATION VALUE of per-(row,chunk) vs the deployed per-row MP
| cell | raw PPL | cost | compute part | **ALLOCATION** |
|---|---|---|---|---|
| 4B t32      | -5.01% | +3.7% | -1.29% | **-3.29%** (2 runs) |
| llama8B t32 | -2.18% | +0.4% | -0.12% | **-2.00%** (2 runs) |
| llama8B t48 | -0.24% | -4.6% | +1.28% | **-1.51%** |
| 4B t48      | -0.37% | -3.2% | +1.11% | **-1.48%** |
| 14B t32     | -1.63% | +6.6% | -0.88% | -0.75% |
| 14B t48     | -0.56% | +2.7% | -0.37% | -0.20% |

**The t48 cells were misread in the OPPOSITE direction**: they ran 3-5% CHEAPER,
so their raw -0.3% UNDERSTATES the allocation effect, which is ~-1.5%.

### CROSS-VALIDATION — two runs ~10pp apart in compute AGREE
  llama8B t32: -2.06% (at +0.4% cost) vs -1.93% (at -7.4% cost) — 0.13pp apart
  4B t32:      -3.72% (at +3.7% cost) vs -2.87% (at -6.2% cost) — 0.85pp apart
Not a single-point estimate that could be a compute artifact.

### vs the ARCHIVE (mp_best_after_hpca) — WINS 3/5, TIES 1, LOSES 1
| cell | archive | mine | |
|---|---|---|---|
| **14B t48** | 8.9072 (parent) | **8.8571** | **-0.56%, x_fp16 1.0253 — PASSES 1.05** |
| 14B t32 | 9.3097 (parent) | 9.1581 | -1.63% |
| 4B t32  | 11.4758 (qk+kband) | 11.3881 | -0.76% |
| llama8B t48 | 7.9032 (qk) | 7.9011 | -0.03% (tie) |
| 4B t48  | 10.3783 (qk+kband) | 10.6093 | +2.23% (archive wins — it has qk) |

**14B t48 is the first cell under 1.05x fp16**, on the model where qk regressed
(+0.91%) and K-bands regressed (+0.33%). Nothing else moved 14B.

### IN FLIGHT: combined qk + per-(row,chunk)
Disjoint operators (attention vs linears); qk+bands previously composed
additively. 4B t48 is the likeliest next 1.05 pass (qk alone gave -2.10% on 4B,
prc gives -1.48% => ~1.034x). Plus qk-only CONTROLS on 14B, where the archive
has no qk data at all.

## ══ FINAL 2026-08-04 — mp_best_after_hpca_2 SHIPPED ══

7 of 8 cells come from this wave; only 4B t48 is kept from the predecessor.
Archive: hpca_results/llm/ppl/mp_best_after_hpca_2/ (SUMMARY + manifest +
HANDOFF + configs/prc_tables/qk_scales). Builder is idempotent:
`python benchmark/ppl/kbands/build_mp_best_after_hpca_2.py`.

### 30B was the last flip and the biggest
Its prc-only cells LOST to the archive; the COMBINED arm won by a wide margin:
  30B t32  archive 8.6255 -> 8.0212  (-7.01% vs archive, -15.25% vs per-row)
  30B t48  archive 7.7217 -> 7.6394  (-1.07% vs archive,  -6.59% vs per-row)
30B is a qk model AND has real per-(row,chunk) value; neither lever alone shows
it. This is the clearest evidence the two compose on disjoint operators.

### qk composition, measured WITH controls
helps: 4B t32 (-2.97%), 4B t48 (-2.08%), llama8B t48 (-0.30%), 30B (large)
hurts: 14B t32 (qk 9.3525 vs parent 9.3097), 14B t48 (prcqk 8.9423 vs prc
       8.8571, qk-only 8.9879), llama8B t32 (+0.56%)
=> per-cell lever selection, never uniform application.

### Closest to the 1.05 goal
30B t48 at 1.0521 (from 1.0634). Not a pass. No NEW cell crosses 1.05 — the
predecessor already had 3 (30B t64, 14B t48, 4B t48) and this archive improves
two of them without adding a fourth.

## ══ 2026-08-04 — COVERAGE FIX + a REPRODUCIBILITY GAP in the v7 tables ══

### Correction to the section above
"this archive improves two of them" is WRONG — it improves ONE. Of the
predecessor's three 1.05x passers: 14B t48 improved (8.9072 -> 8.8571, 1.0311
-> 1.0253); **4B t48 is byte-identical and merely CARRIED** (delta_vs_prev
exactly 0.00%, source `mp_best_after_hpca`); 30B t64 was never run in the wave.
Also note "7 of 8" is the `source: this wave` count, NOT an improvement count —
6 cells improved, 1 was carried, and llama8B t32 had no predecessor cell to
improve on.

### The t32/t48 grid silently DROPPED a passing cell
The builder's `TARGETS = [32, 48]` meant the archive did not contain 30B t64
(1.0224x, a PASS), 4B t40, or llama8B t96 at all. Quoting "2 passing" from this
grid against the predecessor's "3 passing" from its own 10-cell grid reads as a
regression that never happened. FIXED: `TARGETS = [32, 40, 48, 64, 96]`. Cells
with no wave run and no predecessor entry fall out as `pending`; cells with only
a predecessor entry are carried. Archive is now **11 cells, 3 passing 1.05x**
(14B t48 1.0253, 30B t64 1.0224, 4B t48 1.0332), 7 from this wave, 4 carried.
Sensitivities are anchored on (32,48) explicitly, so widening is inert for the
cost model.

### ⚠⚠ THE DEPLOYED v7 TABLES CANNOT BE REGENERATED FROM THIS REPO
A reproduction control (recalibrate 4B t32 via `KB_JOB=prccalib`, diff against
the archived table) FAILED to reproduce v7:

  4B_t32_v5 == v6 == v7   sha256 09d6d6a5...   (the 8 deployed cells use this)
  4B_t32_repro == iso2    sha256 dc3ce76c...   (what prccalib emits TODAY)
  0 of 28 buckets match between them.

`benchmark/ppl/mp_per_row_chunk_calib.py` is **UNTRACKED — not in git**, so the
pre-iso2 solve has no history to recover. The current file also REFUSES by
construction to emit a non-iso-cost table (the guards near lines 491/499/514),
so it *structurally cannot* produce a v7-style table. The v7 tables on Turbo
remain valid and verified; they are simply not reproducible from source.

**iso2 is the LOSING variant, measured with controls** — cheaper but worse on
both axes wherever both ran:

| 4B t32 arm | PPL | cost | raw vs parent | cost-adjusted |
|---|---|---|---|---|
| v7 prc   | 11.3881 | 34.97 | -5.01% | **-3.72%** |
| iso2 prc | 11.9019 | 31.64 | -0.72% | -2.87% |

Also llama8B t32 (8.4864 v7 vs 8.6863 iso2) and 14B t32 (9.1581 vs 9.4675).
So v7 does NOT win merely by overspending — it wins after the compute term is
subtracted. Do not "fix" the archive by swapping in iso2 tables.

### 4B t40 DROPPED from the wave (user decision)
It could only be given an iso2-style table, which would be a method
inconsistency inside a comparison. Coverage is already closed by the TARGETS fix
(carried at 1.0801; it was never a passing cell). The mislabeled table this
session emitted was renamed `4B_t40_iso2style_prc*.json` — it was written as
`4B_t40_v7_prc.json`, and `run_prc_ppl.sbatch` defaults `KB_TBL=v7`, so any
future 4B t40 run would have loaded the losing variant silently.

### IN FLIGHT — 8 jobs, 56141489-56141497
{30B t64, llama8B t96} x {parent, prc, qk, prcqk}, full protocol, v7 tables
(pre-existing, wire-gated ALL PASS: 30B t64 92.0% of linear MACs at d_in=128
traced 59.94; llama8B t96 96.1% traced 90.31). Both cells are `prev:qk` today,
so the question is whether per-(row,chunk) beats qk-alone at the looser budgets.
Rebuild with `build_mp_best_after_hpca_2.py` when they land.

## ══ 2026-08-04 OVERNIGHT — P1 pooled loss-weighted objective QUEUED ══

Pre-registration (written BEFORE launch, refutation criteria fixed):
`benchmark/ppl/kbands/PREREG_LOSS_WEIGHTED_OBJECTIVE.md`. Launcher:
`benchmark/ppl/kbands/run_lossw.sbatch`.

### Why: the objective, not the allocation
Allocation is ~96% exhausted (deployable tracks the per-group oracle within
0.8-1.6pp) yet sigma converts to PPL at 0.33-0.36 %/% and at 0.00 past the knee.
All 20 deployed mp_best cells run `cross_layer_weight: uniform` — an UNWEIGHTED
sigma — while the knock-down probe measures attention as 6.39x more
loss-sensitive than linears (forensics band 5-20x).

### Two of my own hypotheses died first (do not re-derive)
* A barrier/penalty term is the WRONG fix: sigma is ALREADY convex at the floor
  (last step 2.3-2.7x the mean prior step, all 36 buckets of 4B t32).
* The `measured` probe is NOT blind: it reports the 6.39x. The ESTIMATOR fails —
  22% of raw dLoss entries are NEGATIVE (impossible in expectation).
* NEW REFUTATION: `measured_marg` is dead. 69% sign violations, attention/linear
  ratio -0.86x (wrong sign), operator ranking inverts. Never use it again.

### The lever: pool before flooring
`--measure-pool {none|operator|attn_linear}` added to
`calibrate_mp_thresholds.py` (`_pool_group_weights`, pools RAW dLoss then hands
the existing floor/normalize path a pooled dict; `none` is EXACT identity so no
existing behaviour changes). Unit-tested offline against the stored 4B probe:
pooling also recovers signal the floor was eating — normalized attention/linear
is 5.21x at `none` but 6.39x at `operator`/`attn_linear`, i.e. flooring noisy
negatives attenuates the signal ~18%.

### Wave: 16 jobs 56164166-56164199, t32, 4 models x {control, p1}
calib -> eval chained with `--dependency=afterok`, so a failed calibration
cannot burn eval hours. Each calib passes a GATE that refuses to hand the eval a
table whose weighting no-opped (asserts distinct-weight count 2/9/36 by arm and
attention/linear > 1). control = `--cross-layer-weight uniform`, fresh from the
same code path (the deployed mp_best tables went through v16/v20 search and are
NOT a valid control).

**FRONT-END: calibration SmoothQuant, eval AWQ.** `calibrate_mp_thresholds.py`
has NO AWQ path at all (only `--calib-smoothquant`); the prc calibrator does.
This split is not introduced here — every deployed cell is a SQ-calibrated table
measured under AWQ. Both arms share it, so it cannot move the delta. Deliberately
did NOT add AWQ calibration, as that would change a second variable.

### Reading it in the morning
Winner is the LOWER full-protocol PPL, judged COST-ADJUSTED (sensitivities 14B
0.134, 30B 0.401, 4B 0.348, llama8B 0.279); noise floor sd 0.0069, |delta| <
~0.014 is not a result. Check prediction 1 FIRST from the trace: if allocation
does not shift from gate/up/q_proj toward qk/av, the mechanism is wrong
regardless of PPL. Expect the largest gain on 30B (attention 22.4% of true MACs)
and the smallest on 14B (6.0%) — 14B has regressed on every lever so far and is
the likeliest way P1 fails the all-four-models bar.

### Caveat
These cells are NOT comparable to the archive: fresh plain calibration (no
v16/v20 search), so absolute PPL will be worse than mp_best. Only the
control-vs-p1 delta is meaningful.

## ══ 2026-08-04 — P1 REFUTED; the PROBE is the constraint, not the objective ══

Full results in `PREREG_LOSS_WEIGHTED_OBJECTIVE.md` (post-hoc section).

* **P1 refuted by its pre-registered bar.** 4B cost-adjusted **+1.18%** (loses),
  llama8B **-0.54%** (wins); both clear the 0.014 noise floor. 14B and 30B arms
  are INVALID (see below), so P1 cannot reach the required 3-of-4.
* **The knock-down probe does not scale.** Negative raw dLoss fraction — a
  direct noise read-out, since cutting precision cannot reduce loss — rises
  **8% / 31% / 39% / 56%** for 4B / llama8B / 14B / 30B. On 30B the pooled
  LINEAR mean is negative, i.e. no signal at all.
* **GATE BUG (mine), now fixed.** The gate judged the ratio AFTER
  `_normalize_group_weights`, which floors at `0.1*mean_pos`. On 30B that turned
  a raw **-22.68x** into a normalized **+45.00x** — the floor manufactured the
  weight and the gate passed where it should have refused. `run_lossw.sbatch`
  now judges the pooled RAW ratio, refuses a non-positive linear mean, and
  refuses `raw_negative_frac > 35%`. Replayed against the emitted tables it
  correctly passes 4B/llama8B and refuses 14B/30B.
* **Do NOT run P2/P3 against this probe.** They differ from P1 only in
  resolution; resolution is not what failed. Fix the estimator first (more
  windows, paired/common-random-number probing, or a variance-reduced
  sensitivity estimator).
* 14B/30B p1 evals were left running: their arms are invalid as a test of P1,
  but they still document what a noise-manufactured weight does. Cancel if the
  GPUs are wanted.

### P1 FINAL — all 16 jobs landed (2026-08-04)

| model | control | p1 | COST-ADJ | arm valid? |
|---|---|---|---|---|
| 4B | 12.4549 @31.51 | 12.6051 @31.49 | **+1.18%** | yes (8% noise) |
| llama8B | 8.9695 @32.86 | 8.9220 @32.85 | **-0.54%** | yes (31%) |
| 14B | 9.4063 @31.85 | 9.4285 @31.86 | +0.24% | no (39%) |
| 30B | 9.6473 @31.70 | 10.2434 @32.20 | **+6.81%** | no (56%) |

**30B is a positive control for the gate bug**: its 45x attention weight was
manufactured by the floor from a NEGATIVE pooled linear mean, and deploying it
cost +6.18% raw PPL. The corrected gate refuses it — worth ~6.2 GPU-hours on
this wave alone.

**CAUTION — do not over-read "fix the probe".** 4B has the cleanest probe in the
set (8% negatives, only single-digit one) and P1 lost there by the largest valid
margin (+1.18%); the lone win was llama8B at 31%. If variance were the whole
story 4B should have won. Two separable hypotheses remain: **H-probe**
(variance alone) vs **H-sign** (cross-operator loss weighting has a
model-dependent sign, like qk). Cheapest discriminator = re-probe 4B with
paired / common-random-number draws + more windows; if the ratio stabilises and
P1 STILL loses on 4B, H-sign wins and this whole direction is dead.

## ══ 2026-08-04 RETRACTION — the "calibrator blocker" was my error ══

I claimed (in PREREG section 8, HANDOFF pitfall 8, and the scmp_llm CLAUDE.md
status block) that the deployed v7 tables "cannot be regenerated from source"
and that this BLOCKED new algorithm work. **Both halves are wrong.**

* **v7 is reproducible**: omit `--parent-trace`. `mp_per_row_chunk_calib.py`
  (~line 448) then takes `par_targets` from the calibration-sample means, which
  is exactly how v7 was built. I inferred "refuses by construction" from the
  iso-cost guard's wording without reading the budget path above it.
* **`iso2` is the BUG FIX, not a handicap.** The 256-row sample OVERESTIMATES
  the parent budget by 9-23% (4B t32 31.96 vs true 26.09; 14B t32 38.97 vs
  29.93), so v7 children got a too-generous budget. That is pitfall 1 and this
  ledger's own "the two LARGE wins are partly PURCHASED".
* The residual cost-adjusted gap (v7 -3.72% vs iso2 -2.87% on 4B t32) is a
  2-point sensitivity extrapolated over a ~10% cost swing — inside its own
  uncertainty, NOT evidence that v7 allocates better.
* The blocker never applied to P1 regardless: P1 used
  `calibrate_mp_thresholds.py`, not `prccalib`.

**Standing guidance:** calibrate new levers WITH `--parent-trace`, judge
cost-adjusted, and do not restore the sample-budget path.

**What remains TRUE:** the archive's per-(row,chunk) cells are sample-budgeted
and overspend (4B t32 34.97 vs parent 33.72). The manifest discloses it in
`realized_flop_avg_sl` / `cost_adjusted_value_pct` — the deliberate call to
attribute rather than gate. Anyone quoting the raw `vs parent` column without
the `value` column is over-claiming. If the archive needs to be defensible on a
strict iso-cost basis for the paper, the prc cells must be recalibrated with
`--parent-trace` and re-evaluated; expect the headline wins to SHRINK.

## ══ 2026-08-05 ★ SC ENABLE-GRID MATCHING — a NEW 1.05x pass, runtime-free ══

`sc_matmul` defaults `rng_levels = 2**(sc_prec-1) = 128` for EVERY `stoc_len`,
and `model/sc_common.py` never overrode it. A group at the ladder floor was
therefore representing a 128-level grid with ~18 stochastic samples: a stream of
L cycles resolves ~L magnitudes, so everything finer is SAMPLING NOISE, not
quantization error. On 4B t32, 44.1% of all MACs sit at that floor.

### The grid must be a POWER OF TWO
First sweep (grid=L): mean rel-L2 change at L<=32 only **-2.34%**, and it won
ONLY at L in {16,32,64} -- exactly the powers of two -- doing nothing at
18/24/48/96. The Owen/bit-reversal scramble is a pow2 construction
(mask = bit_reverse(d mod M)), so a non-pow2 grid breaks its structure.
Second sweep (grid = largest pow2 <= L): **-4.21%**, winning at every rung
through 64, and largest on q/k/v -- the attention feeders sigma misprices.
Probe: `benchmark/ppl/kbands/probe_rng_grid.py`.

### Deployed as `SC_RNG_GRID=pow2` (OFF by default, byte-identical when unset)
`model/sc_common.py` wraps `_sc_matmul` and injects
`rng_levels = 1 << (L.bit_length()-1)` for 2 <= L < 96 (128 measured better at
L>=96). Same cycle count, same 20% mask, same budget -- NO new hardware.

### PPL, 11 cells, identity control EXACT
`gridctl_30B_t48_prcqk` with the flag UNSET reproduced the archived 7.6394
EXACTLY, so every delta below is the grid and not code drift.

| cell | archive | grid | delta | cost | x_fp16 |
|---|---|---|---|---|---|
| **30B t48** | 7.6394 | **7.5663** | **-0.96%** | 53.60 (was 53.82) | **1.0420** <- 1.0521 **NEW PASS** |
| **30B t32** | 8.0212 | **7.9073** | **-1.42%** | 40.10 (was 40.4) | 1.0890 <- 1.1047 |
| 4B t40 | 10.8491 | 10.8099 | -0.36% | 41.49 | 1.0762 <- 1.0801 |
| llama8B t96/t48/t32, 30B t64 | — | — | +0.03..+0.12% | — | BELOW noise floor, not results |
| 14B t32 / 4B t32 / 4B t48 | — | — | +0.51/+0.37/+0.53% | — | small REAL regressions |

Both 30B wins are cheaper AND better, so neither is purchased. Archive is now
**4 of 11 passing 1.05x** (was 3) and 30B's passing rung drops **t64 -> t48**.
Per-cell like qk: 30B gains most (largest attention MAC share, 22.4%), llama8B
flat, 4B/14B t32 slightly worse. Builder folds it via `read_grid_result`.

### Why this matters for the PAPER's weak half
It gives "finer-grained than fixed-point" a MECHANISM instead of an option
count: **SC decouples the quantization grid from the cost.** In fixed-point, b
bits fixes both the 2^b grid AND the cost. SC runs a 16-level grid at 18, 24 or
32 cycles -- same resolution, different variance, different cost. That is a
2-D per-group precision space; fixed-point has 1-D. It also survives the ladder
ablation (7 rungs ~ 95-99% of a 14-rung oracle) that undercuts "128 options".

### DEAD: the loss-weighting family (P1 / B1 / B2)
B2 (normalized block-local Jacobian) lost on 4/4: +0.50 / +1.33 / +0.93 / +2.55%.
B1 (raw per-row multiply) was catastrophic: 4B +123%, llama8B +489% -- the
documented `grad` cliff blowup, since Gauss-Newton squares a CV-9..26 tail.
Per-row loss-relevance structure is REAL (CV 9-26 within k_proj/v_proj, stable
to ~2%, and down_proj is exactly 0.00 on dense models / 0.47 on MoE, which is
the residual-path identity showing up correctly) but reweighting sigma by it
does NOT convert. Sixth instance of a proxy-axis gain failing to transfer.
Do not revisit without a different consumption mechanism than reweighting.

## ══ 2026-08-05 ★★★ POW2-RESTRICTED LADDER — the missing half-2 experiment ══

**The thesis half "SC is finer-grained than fixed-point" finally has a direct
measurement, and it is POSITIVE on all four models.**

### Design
Same allocator, same budget, same 20% mask, same protocol, same front-end.
ONLY the ladder's rung PLACEMENT changes:
  calibrated  96,64,48,32,24,16   (96/48/24 are NOT powers of two)
  pow2-only  128,64,32,16         (the ONLY rungs in range a fixed-point
                                   datapath can express)
Controls are the in-wave `control` arms. Gate refuses the arm if any non-pow2
rung survives. Emitted ladders verified `[128,64,32,16]` on all four models.

| model | calibrated | pow2-only | penalty | cost | COST-ADJ |
|---|---|---|---|---|---|
| 4B | 12.4549@31.51 | 13.1061@31.32 | +5.23% | -0.60% | **+5.02%** |
| 14B | 9.4063@31.85 | 9.6476@31.49 | +2.57% | -1.13% | **+2.41%** |
| 30B | 9.6473@31.70 | 10.1227@31.56 | +4.93% | -0.44% | **+4.75%** |
| llama8B | 8.9695@32.86 | 9.0421@32.98 | +0.81% | +0.37% | **+0.91%** |

**Mean cost-adjusted penalty 3.27%, 4/4 models.** pow2-only ran CHEAPER on 3 of
4 and still lost, so this is not a budget artifact. All deltas (0.65 / 0.24 /
0.48 / 0.07 PPL) are far above the sd-0.0069 noise floor.

### Why this is the right operationalization
The old framing ("128 options vs BitMoD's 4") is undercut by our own ladder-size
ablation (7 rungs ~ 95-99% of a 14-rung oracle). The correct claim is not how
many options coexist but that **the ladder is CALIBRATED from a continuum**:
`mp_best` uses **15 distinct ladders across 20 cells** and **43 distinct rung
values, 39 of them non-powers-of-two** (97, 85, 74, 111, 117, 109, 66, 63, 61,
49, 47, 45, 42, 39, 35, 33, 25, 19, 18 ...). The rungs cluster where a cell needs
resolution -- 30B t48 packs four into 45-48, 4B t64 packs five into 60-64 --
exactly where fixed-point's neighbours are 32 and 64, a 2x gap with nothing
between. So 7 rungs suffice BECAUSE they are the right 7, and choosing them
requires the continuum.

NOTE the rung COUNT and PLACEMENT are confounded here (4 pow2 rungs vs 6
calibrated) -- but that confound IS the finding: in the deployed range [16,128]
there are exactly four powers of two, so a fixed-point ladder cannot have six.

### ⚠ CORRECTS a standing claim
Memory `project_scmp_mp_beat_actglobal_study` records "pow2 levels beat
non-pow2". That does NOT hold for ladder rung placement at matched budget --
here non-pow2 wins by 0.9-5.0% on every model. Whatever that earlier result
compared, it must not be cited as evidence against fine rung spacing.

## ══ 2026-08-07 — DIRECTION 1 RESOLVED BY FREE ANALYSIS: llama8B's floor is
## MODEL FRAGILITY at the SC ceiling, not error, not a mask, not the front-end ══

Per NEXT_SESSION_PROMPT: no GPU spent; everything below is traces + tables +
existing CSVs. Session artifacts: t96 decomposition script + JSON in the session
scratchpad; map deliverable = `SEARCH_SPACE_MAP.md` (this dir).

### 1. The deficit is a CEILING property (allocator exonerated, again)
AWQ t128 uniform ceiling (mpbest_awq_vs_smoothquant.csv): 4B 1.0131,
llama8B **1.0481**, 14B 0.9920, 30B 1.0369. llama8B t96 archive = 1.0535, so the
128→96 budget cut costs only ~0.5pp; ~4.8pp is the ceiling itself. INT W8A8
through the same driver = 1.0002 ⇒ protocol/fp16 reference sane (lead 5 CLOSED).

### 2. Lead 1 (mask source) is t32-ONLY — cannot explain t96
Verified all 24 mp_best hybrid_configs: llama8B t40–t96 use the SAME
measured_curve sensitivity + same selection method as every Qwen cell. Only
llama8B t32 uses int_swap (`v20_gate2_intswap_top20`). Lead 1 survives only as a
cheap t32 experiment. (Also: 14B t64/t96 masks are fraction 0.1, not 0.2.)

### 3. Lead 4 (front-end) CLOSED — llama8B's scales are the TAMEST of 4 models
SQ act_scales down_proj max 537 vs 4B's 4552 (order of magnitude tamer);
AWQ picks the exact identity s≡1 on 9 late-layer llama8B modules (0 on any
Qwen) — AWQ itself reports nothing to smooth. dispatch ρ_amax ≈ 0.02–0.05 on
llama8B (weakest of any model). ⚠ MEMORY CORRECTED: "llama8B down_proj = amax
INVERTED" was v2-era; deployed llama8B down_proj = l2/+ everywhere; the only
amax/INV cells are 14B o_proj t64/96. Residual untested front-end lever: SQ
α=0.8–0.9 for Llama (repo docstring's own guidance; weak — AWQ is α-free and
reproduces the same floor).

### 4. ★ t96 PER-OPERATOR σ DECOMPOSITION INVERTS THE PREMISE
Trace-MAC currency (never num_units), σ from each cell's SQ-parent table
buckets[op:t0:l*].level_mean_error; 30B qk σ@128 = 0.226 reproduces the
forensics 0.243 ✓. Global MAC-weighted σ:

| model | mwσ(assigned) | mwσ@128 | x_fp16 t96 | excess-nats per unit σ |
|---|---|---|---|---|
| 4B | 0.0788 | 0.0641 | 1.0026 | 0.03 |
| llama8B | 0.0776 | 0.0583 | 1.0535 | **0.67** |
| 14B | 0.0661 | 0.0555 | 0.9967 | ≤0 |
| 30B | 0.1041 | 0.0833 | 1.0043 | 0.04 |

**llama8B carries LESS SC error than 4B/30B and converts it to loss ~15–20×
more efficiently.** Not explained by fp16 level (30B fp16 7.26 ≈ llama 7.21).
Per-op σ@128: llama8B linears ≈ 14B to 3 decimals; qk anomalously GOOD (0.045);
av anomalously BAD (0.361 vs 0.22–0.26; bucket l0 0.358 vs Qwen 0.14–0.18 = 2.5×,
l2 0.455) — av is 3.8% of MACs, ~19–23% of MACσ. σ caveat: SQ-parent tables
don't see qk-rebalance/grid, so 30B/4B true σ is lower than tabled — corrects
their damage-per-σ UP, does not close the order-of-magnitude gap.

### 5. ★ INT LADDER = independent fragility proof + a RULER for SC noise
llama8B is the only model above parity at W7A7 (1.0045); at W6A6 asymm its
excess (1.44pp) is ~5× any Qwen (0.25–0.30pp). Same early-onset smooth noise
sensitivity, different quantization mechanism ⇒ MODEL property. Interpolating
each model's own INT-asymm curve at its SC-ceiling excess: SC@L128 ≈ **W5.2
(llama8B) / W5.2 (30B) / W5.7 (4B) / ≥W8 (14B)-equivalent damage** — SC ceiling
noise is ~W5.5-equivalent everywhere it can be measured; each model's deficit is
its own sensitivity at that noise level. 30B recovers via qk rebalance (its
noise was concentrated/reducible); llama8B's operands are already balanced —
nothing left to condition, only the grid itself can go finer.

### 6. Diffuse-error tally now SIX independent statistics
band asym 1.29× · Q/K spread 4.5×/4.1× · ~linear INT-dose response · mask
recovers 0.5pp at ceiling (vs 7.6pp on 4B, uniform_hybrid vs uniform) · per-op σ
flat/normal · dispatch ρ_amax ≈ 0.02–0.05. Selection levers are CORRECTLY
exhausted on this model.

### CONSEQUENCE for the 1.05× goal on llama8B
Its INT parity point is ~W7A7, its SC ceiling sits at ~W5.2-equivalent ⇒ the
ceiling noise must drop ~2–4× to pass at ANY budget. The only open lever class
matching every statistic is the REPRESENTATION fix — asymmetric SC / zero-point
(INT asymm buys llama8B 0.96pp @W6, 4.3pp @W5; enable-grid alone was neutral on
it). Definitive loss-unit attribution available cheaply: per-op INT7-swap wave
via hybrid_config editing (config-only; av is masked in only 5/32 blocks).

### housekeeping
* `recreate_all.py --grid-policy main` now grids parent too (handoff-directed
  revert; parent+grid wins 3 cells and was silently dropped).
* t32 archive tables (v17/v20-era) carry NO level_mean_error — a t32 σ
  decomposition needs sub-cliff extrapolation; not done, stated here so the gap
  is a decision, not an oversight.
* IN FLIGHT: `grid_30B_t96_prcqk` (56726323, pre-session); Qwen-shaped-code
  audit of the llama SC path (results to be appended below when complete).

### 2026-08-07 addendum — Qwen-shaped-code audit COMPLETE: no correctness bug;
### four mis-tuned llama defaults + one full-length RNG artifact

**Deployment path CLEAN** (lead 3 CLOSED): `_patch_attention_for_calibration`
covers all three modeling families symmetrically; llama runs the adapter path
with eager forced; SC attention PROVEN live from trace row counts (qk rows =
(32−7 masked)×32 heads×288,768 tokens exactly); GQA repeat/rows/mac_per_row all
exact (k/v_proj priced at 4096×1024 — GQA-aware); no module-name/36-block/
head_dim hardcoding in the active path; hybrid mask keys are (op,b,u) — no
name parsing. ⚠ Stale docs: CLAUDE.md still claims llama-sdpa-only smoke and
per_head attention granularity — both false since PR#3 / 2026-07-03.

**FINDINGS (none overturns fragility; ranked):**
1. **`mp_best/rebuild.py:171-183` hard-codes the 10% mask for EVERY t128 cell.**
   llama8B is the model that pays: int_dose shows 20% → 1.050 vs shipped 1.057.
   Free ~0.7pp at the ceiling, already-measured data. (Known in this ledger
   since 2026-08-02; code still unchanged. Fix needs OK — mp_best is frozen.)
2. **qk rebalance is structurally unable to see llama:** (A) `qk_calib.py:78-79`
   head-POOLS the per-dim maxima — valid for Qwen (head-shared q/k_norm gains),
   meaningless for llama (per-head W_q structure, no QK-norm); a (H,head_dim)
   table is a small change and the only route to llama's qk structure.
   (B) deployed α is pinned 1.0 (`fill_queue.sh:99,112`) = the WORST end when
   |Q|≈|K| spread (llama 4.6×/4.1×); α=0.5 is one cheap cell. (C) qk σ-curves +
   dispatch metric are computed on the UNSMOOTHED operand
   (`_sc_attn_matmul_at_level` has no smooth_scales; metric on `a3` pre-scale)
   while deployment quantizes `a/s` — llama and 30B use `crest`, which is
   scale-free and maximally perturbed by a per-dim rescale. Consistent with
   llama t96 arms all inside the noise floor.
3. **SQ α=0.5 for llama** where `smoothquant.py:69-71`'s own docstring says
   0.8–0.9 for Llama. Never overridden per-model. SmoothQuant column only
   (AWQ is α-free and floors the same), so bounded small.
4. **Calib front-end ≠ deploy front-end** (tables SQ-fit, cells AWQ-run) —
   model-agnostic, but llama's MLP share (75.2% of MACs vs 63.4% on 4B) is
   where the two diverge most.
5. **Full-length RNG artifact, model-agnostic:** Owen masks assign `d mod 64`
   with M=64 while qk's D=head_dim=128 ⇒ dims d and d+64 — exactly the RoPE
   rotate_half pair — share ONE scramble mask (perfectly correlated SC noise on
   the most correlated coordinate pair). And for `av` (D=K_seq), position 0 =
   the attention sink gets the IDENTITY mask (bit_reverse(0)=0). Identical on
   all four models so it does NOT explain llama-vs-Qwen; cheap simulation
   ablation = SC_SCRAMBLE_MASKS=128 with the HW_MAX_MASKS=64 cap relaxed.
6. Latent landmines (no current effect): llama adapter lacks the post-load
   `_attn_implementation="eager"` belt Qwen has (a checkpoint config could
   silently drop qk/av to FP16-sdpa — would show as BETTER PPL + missing trace
   groups); `mp_per_row_chunk_calib.py:80` floors n_chunks vs sc_common ceil —
   diverges only on non-128-divisible widths (none in scope).

Audit scope: read-only; nothing modified.

## ══ 2026-08-07 — USER OK: diagnostics wave LAUNCHED + the next ALGORITHM ══

User approved all four proposals + directed algorithm-level research toward
"lowest bitstream at similar PPL". Design doc: `PLAN_2D_PRECISION.md` (rungs
become (L, grid, mode) attributes; space.tex utilization verdict inside).

### Launched (22 jobs in system, 10 GPUs saturated)
* **Wave A — per-op INT7-swap** 56733300–311: llama8B t96 prcqk ×9 ops +
  14B t96 parent+grid ×{av,down,up}. Configs `opswap/` (make_opswap_configs.py);
  masks are the archive mask with ONE op fully int7. Baselines: 7.5992 / 8.6102.
  DIAGNOSTIC ONLY (energy axis changes); tags `*_swap_<op>`, job names outside
  prcppl_/grid_ so rebuilds can never ingest them.
* **Wave B** 56733370 sqa085 (llama8B sc_int8 pure SQ α=0.85 vs 7.6628);
  56733375→56733376 qk α=0.5 calib→prcqk eval (llama8B t96, vs α=1.0 7.5992).
* **Wave C** 56733371/72 m128 (llama8B/4B sc_int8 pure, SC_SCRAMBLE_MASKS=128
  SC_HW_MAX_MASKS=128 — SIMULATION-ONLY, tests RoPE-pair mask-reuse noise; vs
  7.6628 / 11.0893). Kernel change: `SC_HW_MAX_MASKS` env override in
  kernels.py `_scramble_mask_count` (byte-identical unset).
* **Wave D** 56733373/74 mcmask (llama8B t32 prc+parent with the measured_curve
  t48/t96 mask — t48≡t96 schedules verified — vs int_swap-mask 8.4864/8.6758).
  Caveat: prc table was calibrated under the int_swap mask environment.
* **Asym probe** 56733701/02: probe_asym_real.py llama8B+4B t96 — REAL-operand
  bipolar vs unipolar(zp) at matched cycles, per op, + INT sym/asym refs.
  Gate for PLAN_2D_PRECISION stage 1.
* ⚠ CANCELLED 56733606/07 (t32_error_budget on llama8B/30B): that probe is
  SYNTHETIC — `--model` is parsed but N/D/M and tensors are hard-coded
  4B-down_proj-shaped, so per-model runs are a known-outcome no-op. Its
  docstring says "on real activations"; it is not. Do not re-launch per-model.

### Code landed this session
* kernels.py: SC_HW_MAX_MASKS override (simulation-only, default identical).
* recreate_all.py: `--grid-policy main` grids parent again (handoff revert).
* mp_best/rebuild.py: t128 uniform candidate now enumerates BOTH mask doses
  (10% + 20%) and select_winners picks per model — fixes the hard-coded 10%
  that mis-selected llama8B (7.5714@20% vs 7.6268@10%), also 4B/30B.
  **rebuild.py NOT yet run** — deferred until the queue drains (it rewrites
  config bundles that starting jobs read; llama8B t128 will improve to 1.0500).
* NEW: make_opswap_configs.py, probe_asym_real.py, PLAN_2D_PRECISION.md.

### How to read Wave A when it lands
Recovery_op = base_ppl − swap_ppl, in PPL units, minus the (small) INT7 error
the swap adds. Rank ops by recovery on llama8B; compare against the σ shares
(down 31 / up 22 / av 19 / gate 17%). If recovery is FLAT across ops at ~zero,
the 5% is genuinely diffuse and only representation-level levers remain. If av
recovers disproportionately vs its 3.8% MAC share, stage 1 of the 2-D plan
(unipolar av) is the direct fix. 14B controls calibrate the method (its ceiling
excess is ~0, so its recoveries should all be ≈0 — any big 14B recovery means
the swap methodology is confounded).

### 2026-08-07 — ⛔ ASYMMETRIC SC KILLED BY USER; program narrows to (L × grid)
User: "the point of using sc is to save energy, using asymmetric increases
cycles, which increases energy." Bipolar sign-magnitude halving IS the energy
advantage; unipolar has no halve trick (grid parity ⇒ ~2× cycles), and a
dual-mode PE is new runtime hardware. Actions: asym probes 56733701/02
CANCELLED unrun; PLAN_2D_PRECISION.md rewritten to rungs = (L, g) only —
calibrated per-rung pow2 enable-grid, cycle-neutral, existing rng_levels
selector; SEARCH_SPACE_MAP A3 closed (do not revisit); the sc_vs_int_gap
memory's "asymmetric SC" half is superseded, its grid half remains live.
The queue keeps 20 jobs: op-swap ×12, sqa085, m128 ×2, mcmask ×2,
qkcal05→qka05, grid_30B_t96 — none touch the dead axis.

## ══ 2026-08-08 OVERNIGHT WAVE E — two loss-priced levers at t32 (user OK) ══

### What tonight's diagnostics established (inputs to this wave)
1. **op-swap, llama8B t96** (base 7.5992): qk **-1.75%**, av **-1.51%**,
   down -0.44, gate -0.36, up -0.19, v -0.14, o/q/k ~floor. **Attention = 64%
   of the deficit from 7.3% of MACs**; recoveries sum to ~86% of the excess
   (near-additive). sigma prices qk at 2.0% of error mass => **~17x
   mispricing**; MLPs (75% of MACs) are over-priced ~3.5x.
2. **Scramble-mask count M=64->128, pure uniform sc_int8 ceiling**:
   4B 11.0893 -> **10.7981 (-2.63%)**, llama8B 7.6628 -> **7.6425 (-0.26%)**.
   Cycle-neutral (identical streams; only the Owen mask population changes) =>
   a chunk of the ceiling floor is RNG STRUCTURE, not irreducible sampling.
   Pays where raw SC error is large (4B), not where loss-sensitivity is the
   problem (llama8B). SIMULATION-ONLY at M=128 (HW selector caps at 64) —
   label any M>64 row as an ideal, never a deployed cell.
3. CLOSED: SQ alpha=0.85 llama8B **7.8603 vs 7.6628 = +2.58% WORSE** (kernel
   docstring's Llama guidance does not apply here; alpha stays 0.5).
   CLOSED: llama8B t32 measured_curve mask is WORSE on BOTH arms (prc 8.5301
   vs 8.4864 +0.51%; parent 8.7724 vs 8.6758 +1.11%) => the int_swap mask is a
   genuine per-cell win, lead 1 dead, archive stands.
   NEUTRAL: 30B t96 prcqk + grid 7.2951 vs 7.2922 (under floor).

### Wave E — 12 cells, t32 first (biggest headroom), winning arm per cell
E1 `attnmask_*_t32` 56743785-88 — attention-first mask, **MAC-MATCHED** to the
   archive mask (all qk, then av late-first, then the archive's own remaining
   picks until the archive's INT MAC share is hit, never exceeded). Entry count
   rises (58->98 on llama8B) but INT MAC share is equal to 2 dp on all four
   models, so this is a pure COMPOSITION test at the same energy operating
   point. Builder `make_attnfirst_masks.py` (entry-matched was REJECTED: it
   moves only 6.5% of llama8B MACs to INT vs 20.0% = a different operating
   point, not a better mask).
E2 `m128mp_*_t32` 56743789-92 — M=128 under the DEPLOYED MP config, i.e. does
   the ceiling finding transfer to a tight budget with allocation active.
   Caveat: tables were calibrated at M=64; thresholds key on ACTIVATION amax
   (input-derived, M-independent) so they transfer, but the sigma CURVES that
   set the allocation were measured at M=64 — if M=128 lowers error unevenly
   across ops the allocation is slightly stale. Read as a lower bound.
E3 `combo_{llama8B,4B}_t32` 56743794-95 — E1+E2 together. Disjoint mechanisms
   (mask composition vs RNG structure) so they should compose additively; this
   is the cell with a chance at a large move.
E4 `attnmask_{llama8B,30B}_t48` 56743796-97 — second budget.
Launcher gained `KB_MASKS` (default 64 = byte-identical). Startup verified:
E2 prints SC_SCRAMBLE_MASKS=128 HW_MAX=128; E1 prints 64 + the attnfirst mask.

### Baselines (mp_best_after_hpca_3, AWQ + INT7 20%, full protocol)
t32: 4B 11.0860 | llama8B 8.4864 | 14B 9.1581 | 30B 7.9073
t48: llama8B 7.8777 | 30B 7.5663.  fp16: 10.0445 / 7.2130 / 8.6383 / 7.2613.
Noise floor sd 0.0069; |delta| < ~0.014 is not a result.

### How to read it in the morning
* Winner = lower full-protocol PPL, judged COST-ADJUSTED from each run's own
  TRACE (sensitivities 4B 0.348, llama8B 0.279, 14B 0.134, 30B 0.401).
* E1 changes WHICH ops are INT at equal INT-MAC share, so its SC pool changes
  composition: realized_flop_avg_sl will move even at equal energy. Read the
  trace, do not assume.
* If E1 wins, the mask SELECTOR should be rebuilt on measured swap loss for
  all four models (~18 more cells) — that is the loss-priced-allocation
  program's first deployable piece.
* If E2 transfers, M is an ARCHITECTURE parameter question (7-bit mask
  selector, cycle-neutral) — price it in the energy model before claiming it.
* STILL RUNNING from the diagnostics wave: 14B op-swap controls (av/down/up)
  and qka05 (qk alpha=0.5, llama8B t96, vs alpha=1.0 7.5992).
* rebuild.py t128 dose fix is landed but NOT yet run (deferred while jobs read
  the config bundles); run it once the queue drains.

### ★★ 2026-08-08 — qk ALPHA=0.5 WINS ON llama8B, and the 14B controls REFUTE
### "attention carries the loss" as a UNIVERSAL claim

**qk alpha=0.5, llama8B t96 prcqk: 7.5647 vs alpha=1.0's 7.5992 = -0.45%**
(~5x the noise floor, cycle-neutral, score-invariant). The audit's Finding B
predicted exactly this: alpha=1.0 flattens Q completely and dumps ALL imbalance
into K, which is right when |K| spread >> |Q| (4B 81 vs 37, 30B 117 vs 47) and
WRONG when they are equal (llama8B 4.1 vs 4.6). The deployed alpha was pinned
1.0 by `fill_queue.sh`, never swept per model. Launched alpha=0.5 at llama8B
t32/t48 (calib 56744123/25 -> eval 56744130/31): t32 is where alpha=1.0
REGRESSED llama8B, so the sign may flip there.
⚠ LAUNCHER BUG CAUGHT PRE-RUN: `run_prc_ppl.sbatch` HARDCODED the
`_qk_alpha1.0.json` filename, so `KB_QKALPHA=0.5` was silently ignored and the
cell would have run alpha=1.0 UNDER THE alpha=0.5 LABEL — the same
named-but-missing/silent-fallback class this file already records twice. Fixed
(`QKA="${KB_QKALPHA:-1.0}"`); the two evals submitted before the patch were
CANCELLED and resubmitted, because sbatch snapshots the script at submit time.

### 14B op-swap controls (base 8.6102) — the control did its job
| op -> INT7 | 14B | llama8B (same test) |
|---|---|---|
| av        | **8.6246 (+0.17%, WORSE)** | 7.4841 (-1.51%) |
| down_proj | **8.5655 (-0.52%)** | 7.5656 (-0.44%) |
| up_proj   | 8.6055 (-0.05%, floor) | 7.5847 (-0.19%) |

**14B's SC damage is in down_proj, NOT attention** — its av swap gives nothing
(slightly worse). So "attention carries the loss" is a llama8B property, not a
law. This is INTERNALLY CONSISTENT with the whole 14B record: qk rebalance
REGRESSES 14B (+0.91%), K-bands regress it, and per-chunk allocation is worse
than per-row on it. Two different models, two different loss-carrying operator
classes — which is precisely the argument FOR a measured (loss-priced) mask and
allocation and AGAINST any single hand-picked operator prior.
Methodology check passes: 14B is BELOW fp16 at t96 (0.9967x) so a "~0 recovery"
expectation was too strong -- the swap measures "does INT7 beat SC on this op
here", which is a real per-op question even for a below-fp16 cell.

### ⚠ PRE-REGISTERED PREDICTION for Wave E1 (written before the cells land)
The attention-first mask is built from llama8B's swap ranking. Given the 14B
control I predict, per cell: **llama8B HELPS** (attention = 64% of its loss),
**4B/30B likely help** (largest attention MAC share 14.3%/22.9%, and qk
rebalance pays on both), **14B REGRESSES** (its loss is in down_proj, and the
attention-first mask moves down_proj OFF int7 and onto SC). If 14B regresses
and the others gain, that is not a failure of the lever — it is the per-cell
selection this project already applies to qk and grid, and the strongest
argument yet that the mask SELECTOR must be rebuilt on measured per-op loss
per model (~18 cells, the loss-priced program's first deployable piece).

## ★★★★★ 2026-08-08 — LOSS-PRICED (ATTENTION-FIRST) MASK: the biggest single
## lever this project has measured. BETTER PPL **AND** 16-23% CHEAPER.

Same INT MAC share as the archive mask (MAC-matched to 2dp), same table, same
front-end, same everything else. ONLY the mask COMPOSITION changes: which 20%
of MACs run INT7. Ranked by the measured op-swap loss instead of the deployed
SC-fragility proxy.

| cell | PPL | vs archive | realized SC flop | vs archive | cost-adj* |
|---|---|---|---|---|---|
| **llama8B t32** | **8.1354** | **-4.14%** | **28.72** (was 34.18) | **-16.0%** | **~-8.6%** |
| **4B t32** | **10.9927** | **-0.84%** | **26.84** (was 34.95) | **-23.2%** | **~-8.9%** |
*sensitivities llama8B 0.279 / 4B 0.348, extrapolated past their fitted ~10%
range, so read the RAW columns as the result and the cost-adj as indicative.

x_fp16: llama8B 1.1765 -> **1.1279**; 4B 1.1037 -> **1.0944**.

### Mechanism (verified in the traces, not inferred)
Attention is now 100% INT7 (`qk`/`av` absent from both SC traces; SC pool is
pure linears). Attention rows carry the HIGHEST stream lengths (llama8B t96
qk 126 / av 117 vs a global 88), so moving them out of the SC pool removes
expensive cycles, while the MLP blocks that come back INTO SC run at ~20-30.
Hence PPL improves (SC was bad at attention -- the op-swap wave measured
qk -1.75% / av -1.51% on llama8B) AND the SC average length collapses.
llama8B SC pool now: up 32.97% @23.81, down 24.72% @29.55, gate 21.63% @29.96,
o 8.83% @43.34, q 7.65% @21.61, v 2.28% @51.28, k 1.91% @27.34; global 28.81.

### Why this was available
The mask SELECTOR ranks by `top_fraction_by_bucket_worst_delta_loss` -- an
SC-fragility proxy computed per (op, block) bucket. The op-swap wave measured
the actual loss carried per operator and the ranking disagrees violently:
llama8B's proxy mask spends 12/58 entries on attention which carries 64% of the
loss. This is the first DEPLOYED consequence of the loss-priced program.

### THE CELLS ARE 16-23% UNDER BUDGET => REINVESTMENT IS FREE HEADROOM
Launched E5 `reinv_*_t40` 56744772-75 (4 models, attnfirst mask + the t40
table): t40 with attention removed from SC should realize ~34 = the t32
baseline's cost, making it a clean ISO-COST comparison against t32. If it holds
the PPL gain at equal cost, the combined move is worth far more than either
half. (llama8B t40 archive = 8.1302 @41.67, so the reinvested cell has to beat
8.1354 while costing ~34 rather than 41.67.)

### CAVEATS, stated before the morning read
1. **The MP table is mask-blind** (recorded in mp_v8_forensics/verdict_B): the
   allocator never sees the hybrid mask, so these cells run an allocation
   calibrated for a DIFFERENT mask. That is why they underspend -- and it means
   the gain is NOT from better allocation, it is from better mask composition.
   A mask-aware recalibration is the obvious follow-up and is un-run.
2. **Energy axis, not iso-total-compute.** INT MAC share is matched to 2dp, so
   the INT side is fair, and the SC side is strictly cheaper -- but the INT7
   MACs are still priced outside the SC budget, so the energy model must
   arbitrate the final number.
3. **Thesis tension to resolve with the user:** attention now runs entirely on
   INT7, i.e. SC does none of it. The 20% dose is fixed by decree and we only
   chose WHICH 20%, so this is inside the rules -- but "SC for attention" as a
   story point is affected, and the paper's operator coverage claim needs
   restating. FLAGGED, not decided.
4. 4B's combo (attnfirst + M=128) is 11.0250, WORSE than attnfirst alone
   (10.9927), so M=128 does not transfer additively under MP on 4B -- the
   m128mp singles will confirm.

### ★★ 2026-08-08 — llama8B t32 FULL DECOMPOSITION: the levers COMPOSE
| arm | PPL | vs archive | SC flop | x_fp16 |
|---|---|---|---|---|
| archive (prc)        | 8.4864 | —      | 34.18 | 1.1765 |
| M=128 only           | 8.4161 | -0.83% | 34.11 (ISO) | 1.1668 |
| attention-first only | 8.1354 | -4.14% | 28.72 | 1.1279 |
| **both**             | **8.0954** | **-4.61%** | **28.71** | **1.1223** |

* **M=128 TRANSFERS under MP on llama8B** (-0.83% at iso-cost, ~10x the noise
  floor) — the ceiling ablation was not a uniform-only artifact there.
* Near-additive: sum of parts -4.97%, measured -4.61% (93%). Disjoint
  mechanisms (mask composition vs RNG mask population), as predicted.
* **4B is the opposite on M=128**: combo 11.0250 is WORSE than attnfirst alone
  10.9927 (+0.29%), despite M=128 winning -2.63% at 4B's uniform CEILING. So
  the ceiling win does NOT transfer to 4B under MP. Most likely the registered
  caveat: the prc table's sigma curves were measured at M=64, so at M=128 the
  ALLOCATION is stale, and 4B (the model with the most exploitable structure,
  band asym 2.79x) has the most allocation to get stale. llama8B, whose
  structure is diffuse, loses nothing from a stale allocation. TESTABLE:
  recalibrate a prc table at M=128 and re-run 4B — un-run, and it would decide
  whether M=128 is model-dependent or merely calibration-stale.
* Launched the maximum-value cells: `reinvcombo_llama8B_t{40,48}` 56744845/46 =
  attention-first + M=128 + a LOOSER table, i.e. spend the freed 16% back.
  llama8B t40 archive is 8.1302 @41.67; a reinvested cell should beat that at
  ~34 cost, and if it beats 8.0954 it is the new best llama8B t32-cost cell.

### ⚠ CORRECTION (same hour): M=128 DOES work on 4B alone — the levers
### ANTI-COMPOSE on 4B while they COMPOSE on llama8B

| 4B t32 | PPL | vs archive | SC flop |
|---|---|---|---|
| archive (prcqk)      | 11.0860 | —      | 34.95 |
| **M=128 only**       | **10.9783** | **-0.97%** | 34.89 (ISO) |
| attention-first only | 10.9927 | -0.84% | 26.84 (-23.2%) |
| both                 | 11.0250 | -0.55% | 26.82 |

I wrote "M=128 does not transfer to 4B under MP" off the COMBO cell alone. That
was wrong on n=1: M=128 alone is 4B's best raw-PPL arm at t32 (-0.97%, iso-cost,
~15x the noise floor). What is true is narrower and more interesting:
**on 4B the two levers ANTI-compose** (combo is worse than either alone, +0.29%
vs attnfirst, +0.43% vs M128), while **on llama8B they compose near-additively**
(-4.61% vs -4.14/-0.83 singles). Same code, same protocol, opposite interaction
sign — a fourth instance in this project of a lever whose SIGN is model-
dependent (qk, grid, K-bands, now lever-composition itself).
Working hypothesis (untested): both levers perturb the SC pool that the
M=64-calibrated prc table was fitted to — attnfirst by removing attention
entirely, M=128 by lowering SC error across the board — and 4B, which has by
far the most exploitable structure (band asymmetry 2.79x vs llama8B's 1.29x),
has the most allocation quality to lose when its calibration goes stale.
llama8B's diffuse structure means a stale allocation costs it nothing.
DISCRIMINATOR (un-run, one cell): recalibrate a 4B prc table AT M=128 and
re-run the combo. If the anti-composition disappears, it is calibration
staleness, not a real interaction.

Per-model best at t32 so far:
* llama8B **8.0954** (attnfirst + M128) = 1.1223x fp16, at 28.71 vs 34.18 cost.
* 4B raw-PPL best **10.9783** (M128, iso-cost); cost-adjusted best is attnfirst
  (-0.84% at -23.2% cost). The t40 reinvestment cell decides which to deploy.

## ★★★★★ 2026-08-08 — llama8B t48 + attention-first DOMINATES THE ARCHIVE'S
## t64 CELL: better PPL at 36% LESS COMPUTE. The rung goal, achieved.

    new t48 attnfirst   7.6776  x_fp16 1.0644  @ 40.50
    archive t64         7.6912  x_fp16 1.0663  @ 63.20   <- beaten on BOTH axes
    archive t48         7.8777  x_fp16 1.0922  @ 47.36   (-2.54% PPL, -14.5% cost)
    archive t96         7.5992  x_fp16 1.0535  @ 88.03   (still 1.0% better, at
                                                          2.2x the compute)

**This is the "same quality, lower bitstream" claim demonstrated end-to-end**,
and on llama8B — the model that had resisted every allocation lever for a month
(prc -0.2..-2.2%, qk -0.21%, bands nil, grid nil). It did not come from the
allocator at all: it came from pricing the INT mask by MEASURED per-operator
loss instead of an SC-fragility proxy.

### llama8B curve, old vs new (all full protocol, AWQ + 20% INT dose)
| budget | archive PPL @ cost | NEW PPL @ cost | note |
|---|---|---|---|
| t32 | 8.4864 @ 34.18 | **8.0954 @ 28.71** | attnfirst + M=128, -4.61% |
| t48 | 7.8777 @ 47.36 | **7.6776 @ 40.50** | attnfirst, -2.54%, beats archive t64 |
| t64 | 7.6912 @ 63.20 | (dominated by the new t48) | |
| t96 | 7.5992 @ 88.03 | (qk alpha=0.5: **7.5647**, -0.45%) | |

Reinvestment cells in flight will say whether spending the freed 14-16% back
buys more quality still (reinv_llama8B_t40, reinvcombo_llama8B_t40/t48).

### ⚠ 2026-08-08 — MY PRE-REGISTERED 14B PREDICTION WAS WRONG (direction)
I predicted "14B REGRESSES on the attention-first mask" because its op-swap
control put the SC damage in down_proj, not attention, and the new mask moves
down_proj OFF int7 and back onto SC. Measured:

    14B t32 attnfirst  9.1482 @ 31.84   vs archive 9.1581 @ 35.76
    dPPL -0.11% (0.0099 < the 0.014 floor => NOT a PPL result)
    dcost -11.0%  =>  cost-adjusted ~-1.58% (sensitivity 0.134)

So: not a regression, not a PPL win either — **PPL-neutral and meaningfully
cheaper.** The mechanism half of the reasoning survives (14B gains no PPL from
evicting attention, exactly as its op-swap control said), but the predicted
DAMAGE from putting down_proj back on SC did not materialize. Recording the
miss rather than reframing it: the prediction was directional and it was wrong.

### THE GENERAL SHAPE, across three models now
| model | dPPL | dcost | reading |
|---|---|---|---|
| llama8B t32 | **-4.14%** | -16.0% | attention carries its loss (swap: 64%) |
| 4B t32      | -0.84% | -23.2% | mild PPL gain, biggest cost cut |
| 14B t32     | -0.11% (floor) | -11.0% | no PPL effect; pure cost |

**The cost reduction is UNIVERSAL and structural** — attention rows carry the
longest streams on every model, so evicting them from the SC pool always cuts
the MAC-weighted mean length (11-23% here). **The PPL effect is model-dependent
and tracks the op-swap attribution**, which is exactly what a loss-priced
selector is supposed to do: it finds nothing to win where there is nothing to
win, and it still returns the compute.
Consequence for the algorithm: the mask selector should be calibrated per model
on measured swap loss (llama8B: attention-heavy; 14B: down_proj-heavy), not
given one hand-picked operator prior. 30B t32 (22.9% attention MAC share, the
model qk helps most) is the outstanding test.

## ★★★★★ 2026-08-08 — REINVESTMENT: spending the freed budget back is where
## the big numbers are. -6.74% (llama8B) / -3.78% (4B) AT LOWER COST THAN t32.

The attention-first cells came in 11-23% under budget because the MP table is
mask-blind. Reinvestment = run the SAME attention-first mask against a LOOSER
table (t40) so the realized cost lands back at ~the t32 baseline's.

| cell (vs its t32 ARCHIVE baseline) | PPL | x_fp16 | cost | dPPL | dcost |
|---|---|---|---|---|---|
| llama8B archive t32 | 8.4864 | 1.1765 | 34.18 | — | — |
| **llama8B reinv t40+attnfirst** | **7.9144** | **1.0972** | **32.35** | **-6.74%** | **-5.4%** |
| 4B archive t32 | 11.0860 | 1.1037 | 34.95 | — | — |
| **4B reinv t40+attnfirst** | **10.6667** | **1.0619** | **33.74** | **-3.78%** | **-3.5%** |

STRICT DOMINATION on both models: better PPL AND lower realized compute than
the deployed t32 cell. 4B's cell also beats the ARCHIVE'S OWN t40 cell
(10.7868 @ 40.38) by -1.11% at -16.4% cost.

### Where the session now stands per model (best cell at ~t32 cost)
| model | archive t32 | new best | improvement |
|---|---|---|---|
| llama8B | 8.4864 (1.1765) | **7.9144 (1.0972)** | **-6.74%** |
| 4B | 11.0860 (1.1037) | **10.6667 (1.0619)** | **-3.78%** |
| 14B | 9.1581 (1.0602) | 9.1482 @ -11% cost (PPL-neutral) | cost only |
| 30B | 7.9073 (1.0890) | attnfirst cell still running | — |

### Also landed
* **qk alpha=0.5 helps llama8B at t32 too**: 8.4087 @ 34.18 = -0.92% ISO-COST
  vs the archive prc cell, on the budget where alpha=1.0 REGRESSED it. The
  pinned alpha=1.0 was wrong for this model at every budget measured (t96
  -0.45%, t32 -0.92%). alpha belongs in the per-model calibration.
* **M=128 hurts 14B** (9.1876 vs 9.1581, +0.32%) while helping llama8B
  (-0.83%) and 4B (-0.97%). Third lever with a model-dependent sign.

### E6 launched (56746513-15): reinvest one rung further
4B t48 + attnfirst and 14B t48 + attnfirst (should realize ~40 = t48-baseline
cost, the configuration that already made llama8B's t48 beat the archive t64),
plus 4B t40 + attnfirst + M=128 (4B's best single arm stacked onto its
reinvestment).

### 2026-08-08 — 4B reinvestment holds at the NEXT operating point too
`reinv_4B_t48` (t48 table + attention-first mask) = **10.4013 @ 39.89**.
Two readings, both fair:
* vs the archive's **t40** cell 10.7868 @ 40.38 => **-3.57% at ISO-COST**.
* vs the archive's **t48** cell 10.3783 (cost not recorded — that cell has no
  trace in the archive, `realized_flop_avg_sl` is null) => +0.22% PPL at ~40 vs
  a nominal ~47-49, i.e. archive-t48 quality for ~15% less compute.
So 4B improves ~3.6-3.8% at BOTH the t32-cost and t40-cost operating points —
the curve shifts, it is not a single lucky cell.
`reinvcombo_4B_t40` 10.7166 @ 33.75 is WORSE than `reinv_4B_t40` 10.6667 @
33.74 => 4B's anti-composition with M=128 replicates a THIRD time (t32 combo,
t40 reinvcombo). On 4B, use attention-first WITHOUT M=128; on llama8B, use both.
Launched `reinv_{4B,llama8B}_t64` 56747911/12 to test the next rung.

## ★★★★★ 2026-08-08 — llama8B PASSES 1.05x fp16. A STANDING "IMPOSSIBLE"
## CONCLUSION IN THIS LEDGER IS NOW OVERTURNED.

    reinv_llama8B_t64 (t64 table + attention-first mask)
        7.4621 @ 57.76   x_fp16 = 7.4621/7.2130 = **1.0345**   PASSES

vs archive t64 7.6912 @ 63.20 (1.0663): **-2.98% at -8.6% cost.**
vs archive t96 7.5992 @ 88.03 (1.0535): **-1.80% at -34.4% cost** — the t64-cost
cell beats the archive's t96 cell outright.

### What this overturns
This ledger recorded (2026-08-02, and repeated since): "**llama8B cannot reach
1.05x by ALLOCATION at any budget** — at t128 the allocator has nothing left to
allocate, so 1.0574 is the SC quality FLOOR", later revised to 1.0481 on the
AWQ front-end, and the model was written off for selection levers entirely.
**That floor was measured with SC EXECUTING ATTENTION.** The attention-first
mask routes qk/av to INT7, and llama8B's deficit was 64% attention (op-swap,
measured yesterday) — so the floor was never a property of SC, it was a property
of SC-on-attention for this model. Correct statement going forward: llama8B's
SC floor is 1.048 *when SC runs attention*; with attention on the INT side of
the fixed 20% dose it reaches **1.0345 at t64**.

### llama8B, complete new curve
| cost | archive | NEW | x_fp16 |
|---|---|---|---|
| ~32 | 8.4864 @ 34.18 | **7.8581 @ 32.34** | 1.1765 -> **1.0894** |
| ~40 | 7.8777 @ 47.36 (t48) | **7.6353 @ 40.49** | 1.0922 -> **1.0585** |
| ~58 | 7.6912 @ 63.20 (t64) | **7.4621 @ 57.76** | 1.0663 -> **1.0345 PASS** |
| ~88 | 7.5992 @ 88.03 (t96) | (dominated by the t64 cell) | 1.0535 |

Launched `reinvcombo_llama8B_t64` (adds M=128, which composes on llama8B),
`reinv_30B_t48`, `reinv_14B_t64` (56749223-25).

### ⚠ THE CAVEAT THAT MUST TRAVEL WITH THIS NUMBER
Attention now runs entirely on INT7, i.e. **SC executes no attention at all** in
these cells. The 20% INT dose is fixed by decree and only its COMPOSITION
changed (MAC-matched to 2dp against the archive mask), so this is inside the
project's rules and the energy comparison is fair on the INT side — but any
claim of the form "SC reaches 1.05x on llama8B" must state that attention is on
the INT path. USER DECISION REQUIRED on whether the paper takes this trade.

### 2026-08-08 — 4B op-swap wave at t32 (56749278-86): the calibration input
### the loss-priced mask actually needs

`make_attnfirst_masks.py` currently HARDCODES the order qk -> av -> the
archive's own picks. That order came from llama8B's t96 swap table, and the 14B
control proved the loss-carrying operator class is model-dependent (14B:
down_proj, not attention). So the deployed version must rank by each model's
OWN measured dL per INT MAC. We have swap tables for llama8B (9 ops, t96) and
14B (3 ops, t96); 4B and 30B have none.

Launched all 9 ops for **4B at t32** — the budget we actually deploy at, which
is the right place to measure the prices (the t96 swaps were near-ceiling and
cleaner, but composition is decided at the tight budget). 9 cells x ~28 min.
Baseline: 11.0860 @ 34.95. Also fills GPUs that Wave E is releasing.
Not a known-outcome trial: 4B's per-op loss attribution is unmeasured, and its
attention MAC share (14.3%) sits between llama8B's 7.1% and 30B's 22.9%.
30B's swap wave is the remaining gap (~9 x 3h = expensive; decide after 4B).

Also in: `reinv_14B_t48` 8.9002 @ 47.02 vs archive t48 8.8571 = +0.49% PPL at
-7.5% cost (cost-adjusted -0.51%) => 14B reinvestment is neutral at t48, having
been -1.53% at t32-cost. Consistent with 14B having the least headroom (its
archive cells already sit at 1.0253-1.0602x fp16).

### 2026-08-08 — 30B lands: attention-first pays there too, at BOTH budgets
    attnmask_30B_t32  7.7566 @ 33.74  vs archive 7.9073 @ 40.10  -1.91% at -15.9% cost  (1.0890 -> 1.0682)
    attnmask_30B_t48  7.4702 @ 48.98  vs archive 7.5663 @ 53.60  -1.27% at  -8.6% cost  (1.0420 -> **1.0288**)
    m128mp_30B_t32    7.8697 @ 40.12  ISO-COST                   -0.48%
30B t48 improves to 1.0288x fp16 (already a 1.05 passer, now with margin) and
its t32 cell drops to 1.0682. Its reinvestment cells are still running.

### M=64 -> 128 scramble masks, ALL FOUR MODELS under deployed MP at t32 (iso-cost)
| model | dPPL | |
|---|---|---|
| 4B | **-0.97%** | helps |
| llama8B | **-0.83%** | helps |
| 30B | **-0.48%** | helps |
| 14B | **+0.32%** | HURTS |
Helps 3 of 4, and the one it hurts is the model that regresses on every other
lever too (qk +0.91%, K-bands +0.12/+0.33%, per-chunk worse than per-row).
Reminder: M=128 is SIMULATION-ONLY (HW_MAX_MASKS=64 is the silicon cap); it is
an ARCHITECTURE-PARAMETER result (a 7-bit mask selector), cycle-neutral, and
must never be tabled as a deployed cell at the current hardware spec.

## ★★★ 2026-08-08 — 4B's SWAP TABLE FIXES THE RANKING RULE:
## rank by dL PER UNIT OF COMPUTE FREED, not by raw dL.

4B t32 op-swaps (baseline 11.0860 @ 34.95):
| op -> INT7 | raw dPPL | dcost | cost-adjusted VALUE |
|---|---|---|---|
| up_proj   | **-3.73%** | +10.50% | -0.07% |
| down_proj | **-3.67%** |  +3.66% | -2.40% |
| av        | -1.22% | **-10.24%** | **-4.78%** |
| qk        | -0.24% |  **-9.67%** | **-3.60%** |
(remaining 5 ops still running)

**On RAW PPL, 4B's loss is in the MLP — the OPPOSITE of llama8B (attention) and
the same as 14B (down_proj).** Had I ranked by raw dL, 4B's mask would have gone
MLP-first. But masking an MLP op COSTS compute (it removes SHORT-stream rows
from the SC pool, RAISING the mean length) while masking attention FREES it
(long-stream rows leave). Since reinvestment converts freed compute back into
quality, the correct objective is dL per unit of compute freed — and on that
axis attention still wins on 4B (-4.78 / -3.60 vs -2.40 / -0.07).

This retro-explains the whole wave: 4B's attention-first cell was only -0.84%
raw but -23.2% cost, and reinvesting that gave **-3.78%** at t32-cost. The
mask's job is not only to remove loss, it is to remove loss CHEAPLY so the
allocator can spend the difference.

**RULE for the deployable selector (supersedes "attention-first"):**
    score(op) = measured dNLL(op -> INT) / (SC cycles freed by masking op)
    subject to the fixed INT MAC dose; solve as a knapsack.
`make_attnfirst_masks.py` currently hardcodes qk -> av -> archive order, which
happens to be near-optimal on this score for 3 of 4 models but is NOT the rule.
Rebuild it to consume the measured swap table per model.

Also in: `reinv_30B_t40` 7.7125 @ 35.52 = **-2.46% vs the archive t32 cell
(7.9073 @ 40.10) at -11.4% cost**, x_fp16 1.0890 -> 1.0621.

## ══ 2026-08-08 — WAVE COMPLETE (queue drained). Two housekeeping items. ══

### ⚠ MY ERROR: `reinv_30B_t48` was a DUPLICATE of `attnmask_30B_t48`
Both were launched with KB_TARGET=48, KB_ARM=prcqk, SC_RNG_GRID=pow2 and the
SAME `30B_t48_attnfirst.json` mask — only KB_TAGSUFFIX differed. I intended the
second as "reinvestment" but reinvestment means a LOOSER TABLE than the target
being compared against, and at t48 vs t48 there is nothing to reinvest. Result:
identical numbers to 4 dp (7.4702 @ 48.98 both), ~3 GPU-hours wasted on 30B.
Silver lining, worth keeping: two INDEPENDENT job submissions of the same config
returned bit-identical PPL and cost, which is a free confirmation of
[[project_scmp_eval_determinism_and_test_selection]] across processes/nodes.
Lesson: a "reinvestment" cell is defined by (table target) > (comparison
target); encode that in the launcher rather than in the tag.

### 4B t32 op-swap table COMPLETE (9/9 ops, baseline 11.0860 @ 34.95)
| op -> INT7 | raw dPPL | dcost | VALUE (dL per compute freed) |
|---|---|---|---|
| **av** | -1.22% | -10.24% | **-4.78%** |
| **qk** | -0.24% | -9.67% | **-3.60%** |
| down_proj | **-3.67%** | +3.66% | -2.40% |
| **o_proj** | -1.42% | 0.00% | **-1.42%** |
| v_proj | -0.77% | -0.17% | -0.83% |
| k_proj | -0.36% | +0.26% | -0.27% |
| up_proj | **-3.73%** | +10.50% | -0.07% |
| gate_proj | -1.57% | +5.29% | +0.27% |
| q_proj | -0.68% | +3.12% | +0.41% |

**The two ops with the SMALLEST raw effect (av, qk) are the two BEST masking
targets; the two with the LARGEST raw effect (up/down_proj) are worthless once
their compute is priced.** Ranking on raw dL would have built exactly the wrong
mask for 4B. This is the cleanest possible demonstration of the corrected
objective. Note `o_proj` (-1.42% at EXACTLY iso-cost) is 4B's 3rd-best target
and is NOT in the hardcoded attention-first order — another concrete way the
measured selector differs from the heuristic.

### ⏸ rebuild.py NOT RUN — needs explicit OK (it overwrites the frozen archive)
The two-dose fix is landed in `mp_best/rebuild.py`. Previewed against the
measured `int_dose_all.csv`, it would change the t128 ceiling row on 3 of 4
models: 4B 10.3302 -> **10.2187** (1.028 -> 1.017), llama8B 7.6268 -> **7.5714**
(1.057 -> **1.050**), 30B 7.5546 -> **7.5154** (1.040 -> 1.035); 14B correctly
stays at 10% (its 20% cell regresses). Running it rewrites `mp_best/configs/**`
and `traces/**`, which the paper cites, so per the launch-discipline rule it
waits for the user rather than being done autonomously overnight.

## ⛔ 2026-08-08 — HARD CONSTRAINT (user): MP IS SC-DOMAIN ONLY.
## "INT7 as a rung in the per-group allocator" is REJECTED as unbuildable.

User: "mp should happen only at sc domain. within one layer, you cannot mix int
and sc." A PE array runs SC streams or INT MACs for a given GEMM; routing some
128-element groups of one matmul to INT and others to SC is not a hardware you
can build. The per-group MP knob is the STREAM LENGTH (and enable-grid) inside
SC. Recorded durably as [[feedback_mp_sc_domain_only]].

**RETRACTED (mine, same day): "unify the mask into the allocator — INT7 as a
rung, not a separate stage."** It would have made the INT-vs-SC choice
per-GROUP inside a matmul. Dead; do not revisit.

**The overnight results are UNAFFECTED — verified, not assumed.** Every
attention-first mask assigns `sc` or `int7` per (operator, block), i.e. per
whole matmul, exactly like the deployed hybrid schedule; distinct values in the
file are exactly {sc, int7} with no sub-matmul entries. llama8B -7.40%, the
1.05x pass, and all four models' gains stand as hardware-legal configurations.

### What remains OPEN for allocation, all pure-SC
1. **MASK-AWARE ALLOCATION.** The allocator does not know which whole matmuls
   the mask removed from its pool, so it underspends 11-23% — that underspend
   is exactly what the reinvestment hack recovered by hand. Making the budget
   accounting mask-aware is budget bookkeeping, NOT datapath mixing, so it is
   legal, and it would subsume reinvestment into the algorithm. Highest-value
   allocation item on the board.
2. **Loss-priced CROSS-OPERATOR stream-length pricing.** All 20 deployed cells
   run `cross_layer_weight: uniform` on a pure sigma objective (verified in the
   tables), while sigma misprices qk ~17x. The swap table is the first
   estimator with the SNR to fix it. Tier 1 = measured per-operator prices set
   the cross-operator split; tier 2 = sigma keeps shaping WITHIN an operator.
3. **Calibrated per-rung enable-grid** (PLAN_2D_PRECISION.md).
The MASK itself stays what it is today: a whole-matmul, offline, calibration-
time backend choice. What this session changed is only HOW it is ranked
(measured loss per unit of compute freed) — which is a calibration decision,
not a hardware one.

## ══ 2026-08-08 — DOSE INVARIANT SETTLED: 20% BY OPERATOR COUNT, not by MACs ══

User, on being shown that the attention-first masks preserved INT *MACs* but let
the *entry* share drift: **"we should keep the By operator count 20%."**

The two definitions and why they diverge — attention matmuls are individually
CHEAP in MACs, so many more of them fit inside the same MAC budget:

| model | INT entries archive -> attnfirst(mm) | INT MACs archive -> attnfirst(mm) |
|---|---|---|
| 4B | 20.1% -> **27.2%** | 21.25% -> 21.21% |
| llama8B | 20.1% -> **33.3%** | 20.03% -> 20.03% |
| 14B | 20.0% -> **32.2%** | 18.06% -> 18.06% |
| 30B | 20.1% -> 21.1% | 20.21% -> 20.21% |

**ENTRY-matched is now the standard**, and it is strictly better for the thesis:
it keeps the canonical dose AND leaves far more work on SC — llama8B 93.5% of
MACs on SC vs the archive's 80% (INT MAC share 6.47%), 14B 94.6%, 4B 87.1%,
30B 79.3%. It also still masks all of qk plus most of av, i.e. the operators the
swap table prices highest, so most of the measured gain should survive.

### ⚠ PROVENANCE HAZARD I CREATED AND FIXED
Regenerating the masks OVERWROTE the MAC-matched files that produced every
number in `mp_best_after_hpca_4`. Recovered from the archive's own copies
(`_4/masks/`, written at build time) and the two variants are now separate:
  `<model>_t<T>_attnfirst_mm.json`  MAC-matched — PRODUCED _4's results
  `<model>_t<T>_attnfirst_em.json`  ENTRY-matched — the new standard
`build_mp_best_after_hpca_4.py` is PINNED to `_mm` so a rebuild cannot pair new
masks with old numbers; `make_attnfirst_masks.py` now emits `_em` only.
Lesson: a generator that overwrites its own output destroys the provenance of
every result already derived from it — version the filename, not just the
content.

### Wave: 12 entry-matched cells (t40/t48/t64 x 4 models)
Winning arm per model (4B/llama8B/30B prcqk, 14B prc; 30B keeps grid), tag
`attnfirst_em`, job prefix `emask_`. These re-measure the headline
configurations under the correct dose invariant. Expect realized SC cost to be
HIGHER than the `_mm` cells (a bigger SC pool: 93.5% vs 80% of MACs on llama8B),
so the comparison must be read on TOTAL SC cycle-MACs + INT MACs, not on mean
stream length alone — the mean is over a different pool.
**`mp_best_after_hpca_4` stands as measured but its cells are MAC-matched; once
these land, the entry-matched set is what the paper should carry.**

## ★★★★★ 2026-08-09 — MARGINAL PRICES MEASURED: the deployed allocation is
## PROVABLY OFF-OPTIMUM by 5-24x. First clean algorithm-level result.

### The instrument (new): one-rung perturbation, staying in SC
For each operator, shift ONLY that operator's per-(row,chunk) thresholds one rung
shorter and run full protocol. dPPL / (compute freed) IS the Lagrangian's price
dLoss/dCycles at the deployed allocation. Never routes to INT (constraint: MP is
SC-domain only). Generator `make_marginal_tables.py` (round-trips every emitted
table through the DEPLOYED parser before any GPU is spent).
Distinct from the op-swap wave, which measured the INTEGRAL (remove an operator's
SC error entirely); the allocator needs the DERIVATIVE, and they diverge at the
cliff.

### ESTIMATOR QUALITY — this is why six earlier attempts failed and this did not
26/28 cells have the physically correct sign (shortening a stream raises loss);
the 2 negatives are noise-scale. The knock-down probe that killed `measured`,
`measured_marg`, `grad*`, `fisher`, `P1`, `B1/B2` had 8/31/39/56% impossible
signs by model. The perturbation estimator is ~7% and noise-scale.

### PRICES at t32 (dPPL% per 1% of TOTAL compute freed)
| model | dearest to starve | cheapest | SPREAD |
|---|---|---|---|
| 4B | down_proj 0.78 | gate_proj 0.14 | 5.6x |
| llama8B | down_proj 0.83, k_proj 0.78 | o_proj 0.17 | 4.9x |
| 14B | v_proj 0.86 | o_proj 0.11 | 7.8x |
| 30B | v_proj 2.85, k_proj 2.31 | up_proj 0.12 | **24x** |

**At a Lagrangian optimum every price is EQUAL (KKT). They differ by 5-24x, so
the deployed allocation is not optimal — and the direction of the fix is
MEASURED, not guessed.** Note the cross-model pattern: the projections that feed
attention (v/k/q_proj) are dear on the big models while the MLP trio (up/gate)
is cheap everywhere except 4B — the opposite of where sigma sends the budget
(sigma's MAC-weighted mass is ~70% MLP).
Caveat on the largest prices: 30B v/k_proj have tiny denominators (0.27/0.20% of
total cost), so their ratios carry wide error bars; the ORDERING is robust, the
magnitude is not.

### THE MOVE (launched): equalize prices at <= control cost
`make_equalize_tables.py` lengthens the dear operators one rung and shortens the
cheap ones to pay for it, greedy on price, never exceeding the control's cost.
Predictions stated BEFORE the runs, as sums of measured terms:
| model | lengthen | shorten | predicted dPPL | cost slack |
|---|---|---|---|---|
| llama8B | down, k, up | o, gate, v, q | **-2.41%** | +2.69% |
| 4B | down, v | k, q, gate, o | **-1.48%** | +3.52% |
| 14B | v, down, q, k | o, up, gate | -0.98% | +10.01% |
| 30B | v, k, q, o | up, gate, down | -0.49% | +10.47% |
If these land near prediction, the sum-of-measured-terms model is validated and
the next step is a full price-weighted re-solve rather than one-rung moves.
If they undershoot, the prices are not additive across operators and that is
itself the finding.

### ⚠ BUG CAUGHT PRE-GPU (same silent-fallback family as the qk-alpha one)
The first submission passed `KB_PRC_OVERRIDE`, which `run_prc_ppl.sbatch` did
not support — all 28 cells would have run the DEPLOYED table under a perturbed
label, and the launcher's 0-bucket guard CANNOT catch it because the deployed
table has buckets. A flawless null result. Cancelled, launcher patched to honour
`KB_PRC_OVERRIDE`, resubmitted, and verified from the log that the perturbed
file is the one loaded.

### IN FLIGHT (36 cells, GPUs saturated)
4 equalization cells at t32 + the full t48 marginal wave (4 controls + 28 ops) —
prices may reorder at a looser budget, and t48 is where the deployed cells sit.

## ══ 2026-09-20 — THE 2026-08-09/10 WAVE WAS NEVER HARVESTED. Collected here. ══

This ledger stopped at "IN FLIGHT (36 cells, GPUs saturated)". The cells ran.
84 `[RESULT]` lines dated 2026-08-09/10 sat unread for six weeks. All 680 logs
in `_kbands` are now copied to
`/nfs/turbo/.../kbands_20260801/logs_backup_20260920/` and the parsed lines to
`harvested_results_20260920.txt` — scratch purges at ~60 days and these were at
day 42.

Validity: every `margctl_*` control reproduces its `mp_best_after_hpca_3`
baseline to the digit (4B t32 11.0860@34.95, llama8B 8.4864@34.18,
14B 9.1581@35.76, 30B 7.9073@40.10). Token counts are a single value per model
across all 349 harvested runs (298,862 Qwen / 288,627 Llama).

### ★ THE EQUALIZATION MOVE FAILED ITS PRE-REGISTRATION. Prices are NOT additive.

The wave pre-registered predicted dPPL as sums of measured marginal terms, and
stated the falsification criterion itself: "If they undershoot, the prices are
not additive across operators and that is itself the finding." They undershot,
and two reversed sign.

| model | predicted dPPL | actual (`equal`, ≤ ctl cost) | actual dcost |
|---|---:|---:|---:|
| llama8B | **−2.41%** | −0.65% | +1.6% |
| 4B | **−1.48%** | −0.06% (inside noise) | −1.9% |
| 14B | **−0.98%** | **+0.68%** (sign flip) | −8.4% |
| 30B | **−0.49%** | **+1.13%** (sign flip) | −9.6% |

`equal2` overspends its control (+5…+13% cost) and loses on all four models
once cost-adjusted. **Consequence: the proposed next step in the previous entry
— "a full price-weighted re-solve rather than one-rung moves" — rests on a
premise this wave refuted. Do not build it.** The one-rung prices remain valid
as a DIAGNOSTIC of where the allocation is off-optimum; they are not a
composable objective.

### The t48 marginal table is noise-dominated — the instrument does not scale
At t48 one rung frees only 0.1–0.9% of total compute (vs 0.2–6.5% at t32), so
the price denominator collapses: **8/28 cells have a physically impossible
negative price at t48 versus 2/28 at t32.** The 2026-08-09 entry's claim that
the perturbation estimator is ~7% impossible-signed is a **t32-only** property.
Quote it that way. Prices at t48 (4B up_proj −2.13, llama8B q_proj −0.94,
14B q_proj −0.18, 30B k_proj −0.35) are not measurements.

### `aint8` (qk+av → INT8, every weight-bearing matmul → SC) — NOT dose-matched
Four cells beat the `_4` archive cost-adjusted: 30B t64 −8.05%, 30B t32 −1.22%,
14B t64 −1.02%, 30B t48 −0.71% (the two 30B t32/t48 by strict dominance —
better PPL AND lower SC cost). **But the arm is confounded on two axes at once:**
it runs **INT8** where the archive runs **INT7**, and it routes a **larger** INT
side (qk+av entire = 96/432 entries) than the archive's `attnfirst`
(91/432, MAC-matched to 20.2%). `realized_flop_avg_sl` prices only SC work, so
neither the extra bit nor the extra entries are charged. The win is not bankable
as measured. The decisive control is the same rule at **INT7** (`aint7`), which
isolates the bit from the rule.

### ★★★★★ 2026-09-20 — THE PAPER'S LLM TABLE IS FOUR GENERATIONS STALE

`Overleaf/.../src/evaluation/model_quality.tex` `tab:quality-llm` sources its
PaYN SC-MP rows from `frontend_awq/mpbest_awq_vs_smoothquant.csv`, i.e. **the
frozen `mp_best` allocation, calibrated under SmoothQuant and deployed under AWQ
with the thresholds never re-fit** (the file's own note 1). Everything since —
per-(row,chunk) dispatch, qk rebalance, enable-grid, loss-priced mask — is
absent from the submitted paper. All 20 cells improve, same front-end (AWQ),
same dose (h=20%), same full protocol:

| | mean cost-adjusted gain vs the paper's MP cells | cells ≤1.05x fp16 |
|---|---:|---:|
| paper as submitted | — | 7/20 |
| `mp_best_after_hpca_3` (attention stays on SC) | **−3.22%** | 10/20 |
| `mp_best_after_hpca_4` (+loss-priced mask, attention→INT7) | **−6.79%** | 12/20 |

Largest single cell: 30B @ 6 bits **9.4647 → 7.7566** (−18.0%) at lower realized
cost. Every one of the 20 cells improves under both archives; no cell regresses.
All deployed winners in both archives carry `hw_realizable: true` (the
simulation-only M=128 arms win 5 cells in `_4` but are correctly not selected).

**This is the same edit as rebuttal weakness T1.** The group-vs-row granularity
error the rebuttal must admit IS what per-(row,chunk) dispatch fixes, so the
admission and the improved table are one change, not two.

⚠ **FAIRNESS BLOCKER before this is tabled.** The table's `Uniform SC` rows come
from `mp_vs_uniform_under_awq.md` = each cell's exact deployed hybrid mask with
**the allocator disabled**. `_3`/`_4` cells additionally carry qk operand
rebalance and the enable-grid, which are NOT allocation — a uniform comparator
that lacks them would credit the allocator with their work, the same error
`uniform/` vs `uniform_hybrid/` already cost this project once. Updating the MP
row REQUIRES re-running the uniform row with qk+grid and the same mask, or
restricting the update to cells whose winner uses neither.
⚠ `_4`'s attnfirst cells realize far under their target label (4B t32 realizes
26.84 against a t32 label) because the mask evicted attention from the SC pool;
the bits label prices SC only. Do not table the label as the cost.

## ══ 2026-09-20 — ALLOCATION SHAPE: the L(m) relationship was never optimized ══

### ⛔ FIRST: the compute environment was DEAD and is now on Turbo
`annstention` lived at `/scratch/.../shared_data/envs/annstention` and the scratch
purge reaped its Python **stdlib** — 3 of 153 top-level `.py` files survived, so
the interpreter could not boot (`init_fs_encoding: ... no codec search functions`).
site-packages (torch, transformers) was intact. `sc_llm` has no torch; `vit_sc`
lacks typing_extensions. **No SC experiment could run at all.**
Fixed by rsyncing the env to **`/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention`**
and restoring the stdlib from the cached `python-3.10.20-h741d88c_0` package — the
EXACT build hash the env was created with, so it is a file restore, no dependency
resolution, no ABI risk. `conda config --prepend envs_dirs /nfs/turbo/.../conda-envs`
makes `conda activate annstention` resolve to the Turbo copy, so every existing
sbatch script works unchanged. **Do not recreate envs on scratch.**
Verified byte-faithful: `plawctl_4B_t32` reproduces the August `prcppl_4B_t32_prc`
number to the digit (11.3881 @ 34.97).

### The gap: ranking was validated, SHAPE never was
`mp_per_row_chunk_calib.py:179-205` (`thresholds_from_alloc`) sorts pairs by the
proxy metric, takes `counts = bincount(alloc)` from the oracle's allocation, and
cuts thresholds at those cumulative counts. **The oracle's length HISTOGRAM is
imposed exactly and only the proxy's RANKING is validated** (the 94-99% figure).
Sharper still: that histogram was optimized for the UNCONSTRAINED oracle, which
the deployed monotone-in-m rule cannot realize — a histogram optimal for a free
assignment is not optimal under a monotone constraint. Nothing anywhere re-solves
the histogram under that constraint, so **the shape of L(m) has never been an
optimization variable.**

### The closed form the shape should have
For one output element the chunk errors add with |eps_j| ~ m_j/sqrt(L_j):
  independent chunk errors  -> minimize sum m_j^2/L_j  s.t. sum L_j = B  =>  L ~ m
  perfectly correlated      -> minimize sum m_j/sqrt(L_j)                =>  L ~ m^(2/3)
The array shares two RNG banks across the contraction, so the correlated case is
the physical one. **Measured (probe_alloc_shape.py, 4B, 14 real operands, iso-cost):
p=2/3 wins on 11/14, mean squared error -11.53%, closing 61% of the gap to an
oracle that needs an FP16 reference.** The exponent the physics predicts is the
exponent the data picks. ⚠ That probe's baseline was an equal-rung split at
meanL 47.3 while the deployed cell runs at 27.6 — right direction, wrong
operating point; the end-to-end wave is the number that counts.

### ★ STRANDED RUNGS — confirmed in the prc tables, worst where budget is tightest
`thresholds_from_alloc:195-198` emits threshold **1.0** when the water-fill gives a
rung zero pairs, and `bucketize` can never exceed 1.0, so that rung is UNREACHABLE.
Buckets carrying >=1 dead rung: **30B t32 25/28, 30B t40 25/28, 4B t40 17/28,
llama8B t32 16/28 (49 of 168 thresholds), 14B t40 16/28** — but only 1-4/28 at t96.
The effect concentrates exactly where allocation matters most. Realized usage
agrees: 30-72% of MACs sit on the FLOOR rung, 0.1-1.9% on the top.
⚠ An earlier note in this session put these numbers on the per-ROW `buckets`
section; that was the wrong table, then over-corrected to "not real". Both wrong:
the phenomenon is real in the per-(row,chunk) tables, at the counts above.

### ⚠ NEW BUG: calibration and deployment normalize over different populations
Normalization is per-call min-max over ALL (row, chunk) pairs. The calibrator cuts
thresholds from a **256-row** sample (`mp_per_row_chunk_calib.py:222,336-337`)
while deployment runs **ctx 2048** (`run_prc_ppl.sbatch:147`). A max over 8x more
heavy-tailed samples is systematically larger, so every runtime `mn` is biased
DOWN relative to calibration and mass slides toward the cheap rungs. Same family
as the already-recorded 9-23% parent-budget overestimate. Unfixed.

### Also found (not yet acted on)
* **Min-max is the least robust normalizer possible** — `hi` is one order statistic
  over ~8e4 heavy-tailed values. Median consecutive-threshold ratio is 1.05-1.71,
  so a 5% change in a call's single largest chunk demotes EVERY group in the
  14B t40 cell one full rung.
* **Degenerate-call resolver disagreement** — a constant-metric call gets the
  SHORTEST rung under prc (`sc_common.py`, levels ascending) and the LONGEST under
  the per-row resolver (`config.py:1054-1066`). The two-resolvers bug family again.
* **MoE**: rows/call is min 1, median 81, max 2802. At N=1 the allocation is
  decided entirely by within-row chunk RANKING, independent of absolute magnitude.
* **Nothing limits the rung count** — not the hardware, kernel, or loader. Extra
  rungs cost one k_table slice each (R x 64KB, LRU-cached) and **zero extra kernel
  launches**; the D-chunk loop already launches once per chunk.

### The wave (launched, 8 cells, t32, full protocol, AWQ, h=20% unchanged)
`make_powerlaw_prc.py` re-emits each bucket as 16 rungs whose thresholds place
L = kappa * m^(2/3), with kappa bisected per bucket so the induced mean length
matches the parent staircase's. **Iso-cost holds per bucket to max 0.003%** across
all four models, so the cross-layer split is untouched. NO code change — the power
law is expressible in the deployed (levels asc, thresholds asc) format, so this is
a table swap. Every table round-tripped through the DEPLOYED parser
(AdaptiveMPConfig) before writing: 28/28 buckets accepted on all four models.
Arms: `plawctl_*` (deployed v7 table re-run in this env) vs `plaw23_*`.
Jobs 61606068-71, 61606093-96.

## 2026-09-23 — prc2: per-(row,chunk) allocation recalibrated correctly (T1/T2)
Archive `hpca_results/llm/ppl/prc2/` (README/SUMMARY/manifest; builder
`build_prc2_archive.py`); narrative + run log `PRC2_OVERNIGHT.md`.
- v7 prc under-delivered because its CALIBRATOR was wrong, not because granularity is
  worth ~1%: thresholds cut on x[:256] but runtime normalizes over the 2048-row call
  (deployed/intended L −6…−24% per op); first 4 blocks per bucket only; INT-masked calls
  sampled; error curves without AWQ smooth_scales; tail chunk dropped; per-ROW ladder
  reused (floor held 43–84% of linear MACs at t40–t64); unconstrained-oracle histogram
  imposed on a monotone rule; budgets from a biased sample.
- `mp_per_row_chunk_calib2.py` fixes all eight (see archive README). Budgets = the
  parent's own L on the same stratified TRAIN windows (the first train windows make the
  parent overspend down_proj +22% vs its test trace — prefix calibration windows are a
  trap). Pre-registered held-out gate before every chained c17/r17 eval. 30B t32 read
  1.028 and failed — cause (corrected): dense q_proj/o_proj over-spend plus a MEASUREMENT
  bug: sampled expert pairs were not re-weighted by 1/(sampling rate), so the gate's
  aggregate weighted experts at ~11% of linear MACs vs ~67% true (true-MAC ratio 1.007).
  Fixed in calib2/3 on 2026-09-23 before any remaining 30B c17 result existed; gate
  thresholds unchanged; the c17e32 retry (32 expert calls) was already queued.
- Iso-cost vs parent (full protocol, same wrapper/mask/attention/AWQ): 4B
  −5.49/−2.40/−1.37/−0.63/−0.23%; llama8B −3.30/−1.51/−1.32/−0.68% (t32–t64); 14B
  −1.53/−1.04% (t32/t48); llama8B t96 −0.21%. Every finished cell improves; trace cost
  within 1.1% of the parent (the [RESULT] tracker reads per-group arms ≤0.4% cheap —
  archive uses trace cost). SUMMARY.md is authoritative for counts/means.
- DECOMPOSITION (4B t32): identically-calibrated per-ROW control (`SC_PRC_ROWSHARED=1`)
  −1.12% ⇒ granularity alone −4.42% raw (control spent 1.1% less; ≈−4.0% cost-adjusted;
  cost-matched control r17m queued). Held-out error: per-row-same-calib 0.848,
  per-(row,chunk) 0.600, oracle 0.518 (4B t32); 0.885/0.790/0.748 (llama8B t32).
- NOT PURSUED (held-out, same budget): smoothed-operand statistic (0.631 vs 0.600 on
  4B t32, worse; ≈equal on llama8B), absolute thresholds (0.571 on 4B t32 but ≈0 on
  llama8B, and a runtime-rule change), per-block thresholds (0.593 / 0.791). The deployed
  statistic captures ~83% of the oracle's gain on both models tested.
- WHY THE GAIN SHRINKS WITH BUDGET (investigation 2026-09-23): the per-row parent's own
  held-out linear error falls ~1/L² (E_par·L² roughly constant per model) while per-group
  removes a ~constant fraction of it (4B 34–40% through t64); at t96 ~⅓ of linear MACs
  already sit at the 128 cap in both arms (k/v/o pinned). Attention's cost share and
  staircase expressiveness are ruled out. WITHIN-BUCKET headroom is small: at the parent's
  per-(op,layer-quartile) budgets and in summed squared error, the rule gets 83% of an
  unconstrained per-pair oracle (4B/llama8B t32). This does NOT bound cross-bucket,
  cross-op or linear↔attention re-splits, nor a loss-aware objective — the per-bucket
  budgets are inherited from the SmoothQuant-era per-row parent and were never
  re-optimized for per-group (unexplored headroom).
- NOTE: on the linears the per-(row,chunk) path has no μ+2τ escape gate (its dense
  ladder reaches 128); attention keeps the parent's escape.
- Iso-PPL: 4B t32 at 0.8× linear budget = 11.9836 vs parent 11.9886 at −12.1% SC cycles.
