# Two-phase framework description: code audit and proposed wording

Date: 2026-09-27. This is a recommendation, not a paper edit or a new experiment.
Scope: the current LLM allocation candidates and the requested best(all) results.

## Recommendation

Keep **Preparation → Phase 1: Initial Allocation → Phase 2: Loss-Aware Refinement → Runtime Dispatch**.
The Fisher solve belongs inside Phase 2: it changes the allocation objective using
end-to-end sensitivity. The subsequent bounded threshold experiments, if mentioned,
also belong inside Phase 2. Neither operation requires a third phase.

Do not present the current best(all) table as the output of one validation-selected
configuration. Candidate calibration uses training data, but the reported cell is
selected by full-test PPL across comparable allocation variants, as the user requested.
The evaluation section must disclose that distinction. A candidate being frozen
before its own test evaluation does not make the final comparison test-independent.

Suggested short overview:

> Calibration has two phases. Phase 1 constructs an initial allocation from local
> reconstruction errors. Phase 2 refines this allocation using end-to-end loss
> information: it re-prices linear computation groups with an output-sensitivity
> estimate and solves their allocation jointly under the compute budget. Attention
> can enter the same budget solve while retaining its per-row dispatch rule. The
> resulting tables map input statistics to stream lengths at inference.

Suggested evaluation sentence, required with the current table:

> We report the lowest full-test perplexity among the comparable allocation
> candidates evaluated for each model and budget, retaining the same frontend,
> integer mask, protected channels, and SC numerics. Calibration-window losses are
> reported as diagnostics; the final per-cell selection uses the test results.

## What the measured pipeline actually did

The existing coarse initializer and structural loss search produced the archived
per-row parent configurations. The corrected per-group reconstruction solve and
the Fisher solves then **each loaded that parent**, rather than running in a chain
where c17 becomes the Fisher expansion point.

* `mp_per_row_chunk_calib2.py` (c17): measures raw squared errors of linear
  `(token row, 128-element residual chunk)` partial products. It solves a monotone
  staircase for each operator/layer-quarter at that parent's own sampled cost.
  Attention, protected channels, and the integer mask stay fixed.
* `mp_per_row_chunk_calib6.py`: captures output gradients and measures errors on
  the parent SC trajectory. One multiplier reallocates the total linear budget
  across all linear threshold sets. Attention stays fixed.
* `mp_per_row_chunk_calib7.py`: also measures attention and solves linear groups
  and attention rows together. Attention's existing ladder and escape gate stay
  fixed; its thresholds change. It additionally emits a linear-only control.
* `prc_local_proposals.py` / `prc_local_refine.py`: start from frozen winning
  tables and test a bounded set of cost-matched threshold changes. They do not
  capture a new Fisher estimate or re-solve a Fisher dynamic program.

Accordingly, saying “Phase 1 is raw-error PRC DP, followed by Fisher evaluated on
that Phase-1 table” would describe a sensible future integrated implementation,
but **not the expansion point used for the present results**. A minimally invasive
description can retain the existing reconstruction-based coarse initialization,
put the correction/global solve inside Phase 2, and explain the candidate families
in evaluation. If Phase 1 is rewritten around corrected PRC DP, label the old
parent as the fixed calibration reference instead of silently implying a new
serial run.

## Fisher objective: exact meanings and restrained wording

For linear group `g=(r,c)`, let `delta_g,j(L)` be its SC partial-product output
minus its floating-point partial-product output at output channel `j`. The
remaining input columns, weights, and AWQ scales are the same as deployment.
The measured surrogate is

```
Ehat_g(L) = sum_j G[r,j]^2 * delta_g,j(L)^2.
```

`G` is the derivative of the **sum** of next-token cross-entropies in a window
with respect to the operator output. The principal currency draws one target
token at each position from that reference SC model's softmax, using the existing
teacher-forced input context. It does not generate a new autoregressive input
sequence. A second, empirical variant uses the observed next-token labels.

Use “a diagonal Fisher surrogate for loss sensitivity.” Do not call it the exact
loss increase, a measured Hessian, or a full second-order model. SC forward
values are preserved and backward propagation uses floating-point matmul
derivatives through custom straight-through functions. Gradients are taken on
the **reference SC trajectory**, not necessarily the subsequently selected
candidate's trajectory. Historical early c6 runs preceded the bit-exact custom
STE correction; candidate provenance must not be erased by a universal claim
that every archived gradient pass was byte-identical to the parent.

The code squares residual components and gradient components before summing.
It omits output-channel covariance, cross-chunk error cancellation, and the
interactions of simultaneous operator changes. These limitations motivate
checking actual loss; they do not prove that a particular softmax effect caused
an observed regression.

`cur_of` subtracts `Ehat_g(Lmax)` from every rung without clipping the Fisher
residual. This is a rung-independent constant and leaves the minimizer unchanged.
The factor `1/2` and division by the held-out token count appear in diagnostic
predicted dNLL, not in the optimization objective. Thus there is no need to add
a new normalization parameter to the paper's allocation equation.

Sampling multiplies each row's error and MAC weight by `N/R`; sampled experts
also receive inverse inclusion-probability weights. A group's cost is its true
residual width times the number of output channels times its stream length.
Protected slices are fixed outside the variable linear budget. In the joint
solve, attention escape rows contribute fixed cost.

Code anchors:

* `mp_per_row_chunk_calib7.py:70–89`: partial products and weighted squared errors.
* `:92–164`: straight-through forward/backward definitions.
* `:305–340`: squared output gradients.
* `:344–403`: residual channels, row/expert sampling, MAC weights.
* `:472–552`: block/full backward, summed CE, sampled/true labels.
* `:579–591`: raw/relative/Fisher currencies and longest-rung subtraction.
* `:636–655`: within-key and global linear solves.
* `:775–803`: variable joint budget and fixed attention escapes.

## Deployable solve

The DP operates on metric-sorted bins, not arbitrary independent group choices.
For a given multiplier, it minimizes the sum of surrogate error and MAC-cycle
price subject to a non-decreasing sequence of rung indices. The recurrence is
exact for the binned monotone problem at that multiplier. Bisection searches for
a feasible multiplier; do not claim an exact unconstrained combinatorial optimum
or exact test-set cost equality. Costs on new inputs are measured from traces.

The linear ladder has 17 entries, `[8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128]`
in the implementation's **halved/bipolar cycle units**. The paper must convert
these consistently if its equations use nominal cycles. The 128 endpoint is
the implementation's longest supported stream under this convention.

For linears, the monotone rule is fixed to increasing full-call-normalized
group absmax, with one threshold set per operator type and layer quarter.
Attention retains its calibrated per-row statistic/direction, its own existing
ladder, and its escape rule. One global multiplier does not imply one global
statistic, ladder, or threshold vector.

## Minimal pseudocode preserving two phases

This version returns a candidate family because that accurately reflects the
reported best(all) protocol. A final deployment table is selected from that
family; the evaluation section must state the actual selection data.

```
Preparation
    Fix the frontend, integer mask, protected channels, and SC numerics.

Phase 1: Initial allocation
    Measure operator-local reconstruction errors on calibration inputs.
    Construct a compute-feasible initial allocation and its dispatch tables.

Phase 2: Loss-aware refinement
    Refine the allocation structure with paired end-to-end calibration losses.
    Fix the reference configuration and its calibration MAC-cycle budget.
    Capture output sensitivities and measure linear group errors by rung.
    Bisect one MAC-cycle price; solve each metric-ordered threshold set by DP.
    Form linear-only and joint linear/attention allocation candidates.
    Optionally evaluate bounded threshold-refinement candidates.
    Measure candidate calibration losses and realized costs; retain provenance.

Return the feasible candidate tables and their measured diagnostics.
```

The optional sentence describes the tested bounded search family, not a proven
quality improvement or a guarantee that each group only moves one rung. The
current local proposals can skip rungs when thresholds coincide. They should
not become a prominent main-method claim on the basis of a 0.03% test change.

## Required corrections to the current prose

1. `framework/refinement.tex` currently says attention joins “when held-out
   calibration windows prefer it.” That is inconsistent with best(all) test
   selection. Describe two feasible candidate solves; put selection in evaluation.
   Do not substitute “30B only”: current winners include joint-attention tables
   on 4B, 8B, and 14B as well.
2. The final table is not universally MC-Fisher: the 8B 6.32-bit winner is the
   empirical-Fisher candidate. State the alternative label source in methodology
   or candidate provenance. The 30B 6.32-bit winner is the tiny local-refinement
   change to its joint-attention table. Neither exception should be hidden.
3. `framework/pricing.tex` currently defines a relative-norm objective and
   converts an unconstrained allocation histogram to thresholds. These are legacy
   initializer semantics. Corrected PRC calibration uses raw squared partial
   errors and directly optimizes the monotone rule. Do not conflate their equations.
4. `framework/runtime.tex` currently selects among absmax/L2/crest and applies an
   escape gate generally. For final PRC linears, the statistic is absmax with
   increasing rungs, the ladder is the dense 17-rung ladder, and there is **no
   linear escape gate**. The selected-statistic and escape language belongs to
   attention/legacy initialization.
5. Runtime normalization pools all groups in the same call. The decision is
   therefore not a function of `x_g` alone unless the equation explicitly includes
   the call's minimum and maximum. For MoE experts, the call is that expert's
   routed token population.
6. Avoid claiming that the implemented linear statistic is literally the free
   quantization scale. `sc_common.py:462–526` computes raw residual absmax before
   passing `smooth_scales` into the matmul; `sc/matmul.py:192–193` then applies
   smoothing before quantization. Under AWQ these scales can differ. Safe wording:
   “maximum magnitude of the dispatched computation group.” A hardware reuse
   claim requires checking where that reduction occurs in the hardware design.
7. `framework/refinement.tex`'s final “table regenerated from the skeleton's
   level occupancies” sentence is stale for the direct monotone solve: the DP's
   boundary locations are the emitted thresholds. Do not describe a second
   histogram-imposition step after DP.
8. The existing calibration-time macro for the Fisher pass does not automatically
   cover all candidate sweeps, 20 round-2 evaluations, or the local pilot. Keep
   per-candidate calibration cost separate from total experimental search cost.

Current numerical provenance is in
`hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.{md,csv,json}`.
The architecture and non-LLM generality claims need a separate review before
extending these LLM-specific gradient and runtime statements to every workload.
