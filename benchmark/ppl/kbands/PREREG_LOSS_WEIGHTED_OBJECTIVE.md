# Pre-registration — pooled loss-weighted sigma objective

Written 2026-08-04, BEFORE any run. Everything below comes from artifacts
already on disk; no GPU was used to produce it. Refutation criteria are fixed
here so the result cannot be re-interpreted after the fact.

## 1. The question

Per-(row, chunk) allocation captures ~96% of a per-group oracle on the error
axis (deployable tracks oracle within 0.8-1.6pp at every budget), yet realises
only -6.26% cost-adjusted PPL against -70.8% oracle error headroom on 4B q_proj.
Where does the other order of magnitude go?

## 2. Answer: the objective's CURRENCY, not the allocation

sigma -> PPL leverage, measured (v8 forensics, currency-corrected):

| cell | MAC-wt sigma gain | PPL gain | PPL% per sigma% |
|---|---|---|---|
| 4B (L2) | -4.21% | -1.53% | 0.36 |
| 14B avg48 | -4.77% | -1.57% | 0.33 |
| 4B (L2->L3) | -6.37% (**better**) | ~0.00% (**flat**) | **0.00** |

llama8B bought near-identical sigma at two budgets (-0.00571 @64, -0.00551 @48)
for a **2.4x** different PPL effect. The exchange rate is neither efficient nor
stable.

**Mechanism.** sigma is a *local, per-operator, relative* reconstruction error.
The solver compares sigma across operators as if a unit costs the same NLL
everywhere. It does not: attention is mispriced **5-20x** because k/v error
propagates into scores -> softmax and compounds downstream, which a local
reconstruction error cannot see. Forensics: "solver sells attention first, PPL
punishes it."

Verified here across the whole deployed archive: **all 20 `mp_best` cells run
`cross_layer_weight: uniform` and `loss_weight_by_grad: False`.** Every shipped
cell optimises an unweighted sigma.

## 3. What is NOT the problem — do not re-derive

* **sigma is not structurally cliff-blind.** Its curve is already convex at the
  floor: the last step is uniformly **2.3-2.7x** the mean prior step across all
  36 buckets of 4B t32. A barrier/penalty term is therefore the WRONG fix —
  sigma sees the floor, it underprices it *relative to other operators*.
* **The `measured` probe is not blind either.** It detects the attention gap at
  **6.39x** (qk mean dLoss 0.2145 vs linear 0.0229), inside the forensics band.
  The idea works; the ESTIMATOR is what fails.
* **Allocation-side levers are ~exhausted**: ladder resolution dead (7 rungs
  capture 95-99% of a 14-rung oracle), band count saturates from 8 up, amax is
  already near-optimal (composite metrics much worse).

## 4. NEW: `measured_marg` is dead — mark it refuted

The marginal probe was introduced as FIX 1 for `measured`. On 4B it is noise and
should not be used again:

| probe | negative dLoss entries | attention/linear ratio | qk abs(mean)/sd |
|---|---|---|---|
| `measured` (to floor) | 8/36 (**22%**) | **+6.39x** | 1.03 |
| `measured_marg` | 25/36 (**69%**) | **-0.86x** (wrong sign) | 0.72 |

A negative dLoss means "cut precision, loss improves" — impossible in
expectation, so the rate is a direct noise read-out. 69% sign violations makes
`measured_marg` uninformative, and its operator ranking INVERTS (o_proj,
up_proj, gate_proj come out as the *least* sensitive). This explains its known
int7/len96 inconsistency.

## 5. The proposal — POOL the weights, do not re-probe

`measured` estimates **36** weights (9 ops x 4 layer buckets) at per-cell
SNR ~ 1. That is why "measured ~ act_global": the operator-level signal is real
but is drowned by per-bucket noise at full resolution.

Pooling is the standard rescue for an estimator with stable group means and
noisy cells. Ladder, cheapest first:

* **P1 — 1 scalar.** One weight, attention vs linear: `w_attn/w_lin ~ 6.4`,
  pooled over 8 buckets. Highest SNR, nearly free to implement.
* **P2 — 9 weights.** One per operator, pooled over its 4 layer buckets
  (SNR ~x2 vs per-bucket).
* **P3 — 36 weights.** Per (op, layer). **Already effectively refuted** by the
  22% sign-violation rate; included only to show where resolution breaks.

Objective becomes `sum_g w_op(g) * n_g * sigma_{g,k}` in place of
`sum_g n_g * sigma_{g,k}`. No runtime change: weights are consumed offline by
the solver and the deployed artifact still receives only `L_g`. Satisfies the
runtime-free constraint.

## 6. Pre-registered predictions

Relative to the uniform-sigma parent at matched cost:

1. P1/P2 shift budget **from gate/up/q_proj toward qk/av**. In 4B t32 the solver
   currently floors gate_proj (99.99% of rows), q_proj (96.52%) and up_proj
   (92.61%) — 51.2% of true MACs — while attention (13.07% of true MACs, and the
   highest sigma of any operator at 0.24-0.33) is never floored. The shift IS the
   mechanism; if allocation does not move, nothing below is meaningful.
2. Gain is **largest on 30B** (attention 22.4% of true MACs) and **smallest on
   14B** (6.0%). Genuine risk: 14B regresses on every lever tried so far.
3. P2 >= P1, but by less than P1 >= parent (diminishing returns from resolution).
4. P3 <= P2 (noise dominates at full resolution).

## 7. Refutation criteria — fixed in advance

REFUTED if any of:

* Allocation does not measurably shift toward attention under P1 (prediction 1
  fails) — mechanism wrong regardless of PPL.
* P1 fails to beat its in-wave parent on **cost-adjusted** PPL on >=3 of 4
  models, using archive sensitivities (14B 0.134, 30B 0.401, 4B 0.348,
  llama8B 0.279).
* P3 > P2 (would mean per-bucket noise is signal, invalidating section 5).
* Any delta below the noise floor: **sd 0.0069 PPL; abs(delta) < ~0.014 is not
  a result.**

Project rule: a lever must show genuine upside on all four models to ship; a
single-model win goes inside a larger algorithm, not into the archive.

## 8. BLOCKER — resolve before running this

`prccalib` today emits the `iso2` variant, measurably worse than the deployed v7
(4B t32 cost-adjusted -2.87% vs -3.72%). Any new lever calibrated now starts
from a handicapped parent and cannot be cleanly attributed. The v7 solve is not
reproducible from source (`mp_per_row_chunk_calib.py` is UNTRACKED; the current
file refuses by construction to emit a non-iso-cost table). Fix the calibrator
baseline first, or run P1 strictly against an in-wave parent calibrated the same
way.

## 9. Cost

Zero-GPU work is DONE (this document). Remaining:
* P1: 1 calibration + 4 models x {parent, P1} full-protocol PPL ~ 8 cells.
* P2: +4 cells, only if P1 clears its refutation bar.
* ~20-30 GPU-hours for P1 on the t32/t48 grid (30B ~4-6h/arm, 14B ~2-3h,
  4B/llama8B ~1h, per LOOP_STATE timings).

Needs explicit OK before launch.

---

# RESULTS (post-hoc; sections 1-9 above are unamended as pre-registered)

## A gate bug, found by the results

The first gate judged the attention/linear ratio on the NORMALIZED weights.
`_normalize_group_weights` floors at `0.1 * mean_pos`, so a **negative** pooled
linear mean gets clipped upward and reappears as a large positive ratio. On
30B t32 the pooled RAW ratio was **-22.68x** and the normalized one **+45.00x**:
the floor MANUFACTURED the weight and the gate passed exactly where it should
have failed. Fixed in `run_lossw.sbatch` — judge the pooled RAW ratio, refuse a
non-positive linear mean, and refuse `raw_negative_frac > 35%`.

## The probe does not scale — this is the real finding

Fraction of raw dLoss entries that are NEGATIVE (impossible in expectation, so a
direct noise read-out), by model:

| model | negative frac | pooled RAW attn/linear | corrected gate |
|---|---|---|---|
| 4B | **8%** | 3.51x | PASS |
| llama8B | **31%** | 6.80x | PASS |
| 14B | **39%** | 9.34x | REFUSE (>35%) |
| 30B | **56%** | **-22.68x** | REFUSE (linear mean <= 0) |

Noise rises monotonically with model size and by 30B the signal is gone. Only
2 of 4 models can be validly measured at all. Note also that 4B reports 3.51x
here against 6.39x from the stored `len192` probe — the ratio is not stable
across probe configurations either.

## P1 outcome: REFUTED by its own criteria

Cost-adjusted vs its in-wave control (sensitivities as pre-registered):

| model | control | p1 | raw | cost | COST-ADJ | arm valid? |
|---|---|---|---|---|---|---|
| 4B | 12.4549 @31.51 | 12.6051 @31.49 | +1.21% | -0.06% | **+1.18%** | yes |
| llama8B | 8.9695 @32.86 | 8.9220 @32.85 | -0.53% | -0.03% | **-0.54%** | yes |
| 14B | 9.4063 @31.85 | 9.4285 @31.86 | +0.24% | +0.03% | **+0.24%** | no (39% noise) |
| 30B | 9.6473 @31.70 | 10.2434 @32.20 | +6.18% | +1.58% | **+6.81%** | no (56% noise) |

**30B is a positive control for the gate bug.** Its 45x attention weight was
manufactured by the floor from a negative pooled linear mean, and deploying it
cost **+6.18% raw PPL** — a catastrophic regression, exactly what an
over-weighted attention term starving the linears should do. The corrected gate
refuses that cell, which would have saved ~6.2 GPU-hours (2:14 calib + 4:00
eval) on this wave alone.

Refutation criterion was "fails to beat its in-wave control on cost-adjusted PPL
on >=3 of 4 models". With 1 win, 1 loss and 2 unmeasurable, P1 cannot reach 3/4.
**REFUTED.** Both valid deltas clear the 0.014 noise floor (4B 0.150,
llama8B 0.048), so neither is noise.

## What this redirects to — and one caution against over-reading it

Probe variance is clearly ONE binding constraint: the estimator is unusable
above ~8B, and pooling cannot rescue group means that have the wrong sign.

**But do not conclude "fix the probe and P1 works."** The cleanest probe in the
set is 4B (8% negatives, the only model in single digits) and P1 still LOST
there by +1.18% — the largest valid regression. Meanwhile the single win came
from llama8B at 31% negatives. If probe noise were the whole story, 4B should
have been the win. It was not.

The defensible reading is therefore weaker than "the objective is right, the
estimator is broken": up-weighting attention has a **model-dependent sign**,
exactly like the qk lever (which helps 4B/llama8B/30B but regresses 14B). A
better estimator would sharpen the measurement; it would not obviously flip 4B.

So there are two hypotheses left, and they are separable:

* **H-probe** — variance alone. Test by re-probing 4B with paired /
  common-random-number draws (baseline and probe sharing SC draws) and more
  windows. If 4B's ratio stabilises near the stored 6.39x AND P1 then wins on
  4B, H-probe survives.
* **H-sign** — cross-operator loss weighting is genuinely not a universal
  direction. Supported if 4B's ratio stabilises and P1 still loses there.

Cheapest discriminator is the 4B re-probe: one model, one calibration, the
cleanest signal in the set. Do NOT re-run P2/P3 against the current probe —
they differ from P1 only in resolution, and resolution is not what failed.
