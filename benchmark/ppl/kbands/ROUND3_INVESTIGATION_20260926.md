# Investigation for a possible further allocation round

Date: 2026-09-26. Investigation only: no new GPU jobs or allocation changes launched.
Round 2 is still running. Finish it and prepare the first rebuttal response before
deciding whether to implement this proposal. User constraints: at most four GPUs;
fixed RNG, QK operands/rebalancing, frontend, masks, protected-channel choices and
per-cell parent; final reporting is best(full-test results from comparable rounds).

**Recommendation:** first test a small, cost-matched refinement of the exact winning
per-group allocation, accepting complete candidate tables by measured paired NLL.
Treat QK and AV as separate recipients/donors. This is a bounded pilot, not a proposal
to launch another full 20-cell sweep. A two-class static chunk extension is a credible
backup, supported by new CPU-only diagnostics but not yet by an end-to-end result.

## Evidence that changes the decision

1. The useful round-2 changes are selective. 30B test PPL improves by 1.55/2.80/2.29%
   at 6/6.32/6.58 bits, but worsens by 0.266/0.746% at 7/7.58 bits. Dense-model completed
   tests have no win so far. All these are allocation changes at essentially the same cost.
2. QK and AV should not be pooled into one attention knob. In the winning 30B t32
   full traces, round 1 -> round 2 changes mean QK length 65.50 -> 88.13, AV 59.56 ->
   44.55, and linear length 25.48 -> 25.10. These are DEPLOYED code cycles (double them
   for nominal cycles). At t40/t48, QK also gains substantially. This concerns allocated
   stream lengths, not any QK operand transform.
3. Large Fisher predictions are useful; tiny predictions are unreliable. For 30B
   t32/t40/t48, predicted round-2-vs-regenerated-round-1 PPL changes are -2.595/-3.077/
   -3.561%, compared with measured paired held-out -1.671/-2.923/-2.189% and test
   -1.550/-2.800/-2.292%. At 30B t64 the prediction is only -0.0248%, while held-out/test
   worsen +0.200/+0.266%. For 4B t48 the prediction is -0.0182%, but held-out worsens
   +0.775%. This does not establish that all attention error is systematically underpriced.
   In particular, the 30B t96 +0.879% prediction is from the held-out Fisher surrogate:
   it already warns of regression before measured NLL confirms it. Calibration-to-held-out
   transfer, not only the form of the curvature approximation, needs attention.

Sources: `/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2/*_c7_diag.json`,
`*_c7_heldout_nll.json`, and full traces under
`/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl/`.

## Comparator correction

Some early dense round-2 held-out files compare against a newly generated `c7_gfis`
linear table, whereas round-1 TEST numbers are from `c6_gfis`. The tables differ.
The comparison must use exact wrapper/table identities, not just the label `gfis`.

For 4B t32, the c6 and c7 held-out files have exactly the same 16 windows and
bit-identical parent losses. Against regenerated c7_gfis, gfisla appears to improve
held-out PPL by 0.821% (z=-1.843). Against the actual c6_gfis test incumbent it improves
only 0.109% (z=-0.219), already a tie. The +0.055% test change is consistent with that.
Thus the earlier narrative of a significant held-out win reversing on test is misleading.

Against the actual old c6 incumbent on the same windows, round-2 changes are:

| Cell | Held-out PPL change | Paired z |
|---|---:|---:|
| 4B t32 | -0.109% | -0.219 |
| 4B t48 | +0.921% | +2.314 |
| llama8B t32 | +0.418% | +1.602 |
| 14B t40 | -0.023% | -0.048 |
| 14B t48 | +0.271% | +0.938 |

14B h20 c6/c7 linear tables match. The recorded historical STE fix is a plausible cause
of earlier differences, but no original pre-fix source snapshot was established, so do
not present that causal explanation as proven. The full-test wins on 30B remain valid.

## What was already tried

Claude ran an earlier round 3 on September 24-25. Repeating its broad variants is not
well justified by the evidence:

| Earlier variant | Result on tested cases |
|---|---|
| Twelve calibration windows, raw Fisher | 4B t32 +32.47% held-out PPL vs parent; one window dominated Fisher mass |
| Robust/clipped Fisher | Recovered round-1 quality; best quartile candidate +0.00392 NLL vs round 1 |
| Finer per-block thresholds | Best robust candidate +0.00111 NLL, z=+0.25: tie |
| Re-linearize on the round-1 trajectory | Raw +0.03056 NLL; best clipped +0.00442: no win |
| Attention ceiling 128, paying with linear budget | 4B worse (+0.01779 NLL); Llama neutral (-0.00151, z=-0.49) |

The 17-rung oracle is also extremely close to the oracle over all MEASURED lengths
(the union of this ladder and parent lengths, not an exhaustive 128-length oracle).
On clean six-window dumps the additional predicted gain is about 0.01 percentage points.
More rungs are not a priority. The remaining 3-4 percentage-point Fisher oracle gap at
t32 is a surrogate gap using unavailable per-token sensitivities, not a promised PPL gain.

Sources: `PRC2_OVERNIGHT.md:328-435`; Turbo `4B_t32_c8o_select_heldout_nll.json`,
`4B_t32_c8xo_select_heldout_nll.json`, and `*_c8w6*_heldout_nll.json`.

## Proposed first pilot: refine the deployed table using measured loss

1. Freeze the exact incumbent wrapper/table and its realized cost. Include the unchanged
   table explicitly. A newly solved Fisher table with unit multipliers is not guaranteed
   to reproduce this incumbent.
2. Profile MAC-weighted rung occupancies and full-call normalized metrics on TRAIN
   windows under that incumbent. Generate a small number of complete candidate tables
   that exchange roughly 1-3% of total SC cycles between populations. Use both directions
   and keep QK/AV separate; also allow a few linear-to-linear transfers. Retain current
   ladders, escape behavior, masks and runtime statistics initially.
3. Bound how many assignments/cycles can move. Match the incumbent's total cost rather
   than the nominal target label. Check realized cost after the candidate runs, since
   its upstream activations can move the threshold occupancies.
4. Evaluate each complete candidate on identical TRAIN search windows. Confirm the best
   on a fresh, disjoint set, including the exact incumbent in the same evaluator. Avoid
   recycling the heavily queried seed-101 windows as a fresh confirmation set.
5. Pilot tight 4B and Llama, plus one 30B cell (for example t40). Cap the candidate and
   iteration count. Look for a repeatable improvement of order 0.5-1% PPL at approximately
   equal compute before broadening. This is a decision threshold, not a performance forecast.
   Full-test results remain subject to the user's best-across-rounds reporting rule.

This is different from the older failed `measured_marg`/pooled-sigma experiments: it
evaluates complete, paired, iso-cost assignments around the corrected per-group incumbent,
rather than using noisy single-boundary derivatives to rescale the whole objective.
The old failures and V20 search overfitting still argue for a small search and fresh confirmation.

Implementation detail that matters: the old refinement generators mostly edit legacy
`table['buckets']`; current linear dispatch reads `per_row_chunk`. Reusing those generators
unchanged would not refine the linear groups. Reuse `heldout_nll.py` for evaluation, but
build a PRC-aware proposal and cost path. Linears use ascending thresholds; attention uses
descending thresholds and fixed escape rows. DP with a few operator-specific price factors
is another proposal mechanism, but existing dumps do not contain attention arrays, so it
would require fresh profiling. Direct bounded threshold transfers are the cheaper first probe.

Sources: `mp_v16_refine.py:999-1026`, `mp_joint_refine.py:190-228`,
`marginal_price_probe.py:63`, `mp_per_row_chunk_calib7.py:741-815,896-908`,
`model/sc_common.py:462-530`.

## Why the current proxy could miss such improvements

`mp_per_row_chunk_calib7.py:70-89` squares each chunk residual before aggregating.
Gradient capture at lines 305-340 keeps only squared gradients. The attention path
at lines 408-463 uses the same diagonal treatment. Consequently it drops signed
cross-chunk cancellation and output-channel covariance. In general,

    sum_c sum_j F_j * delta[c,j]^2

is not equal to

    sum_j F_j * (sum_c delta[c,j])^2.

The first is a separable proxy; the second includes interactions among chunks even
with a diagonal output metric. A full projected Fisher would additionally retain output
covariance. These are concrete omissions, but their practical magnitude is not yet measured.
It is premature to claim that softmax or any particular omitted term causes the losses.

If local measured-loss proposals expose repeatable headroom while Fisher misranks them,
a subsequent objective study could score aggregate attention/block outputs or signed
projected residuals. This needs new measurements: squared-error dumps cannot recover signs.
It also loses simple per-group separability, so it should not precede the cheap pilot.

Relevant primary research: [BRECQ](https://arxiv.org/abs/2102.05426) motivates block-level
reconstruction to account for dependencies; [APTQ](https://arxiv.org/html/2402.14866v1)
and [BoA](https://arxiv.org/html/2406.13474v3) account for nonlinear attention/dependencies
in quantization objectives. Their weight-optimization methods are not being proposed here:
only the principle of evaluating allocation errors after their interactions is relevant.

## Backup: a small static chunk-class dispatch extension

New CPU-only analysis used the existing clean six-window dumps
`{4B,llama8B}_t32_c8w6.npz`. For each operator/block, static chunks were ranked on
calibration by excess Fisher error at deployed L=24 divided by summed squared normalized
absmax. That same ranking was assessed on two held-out measurement windows.

| Projection | 4B median rank correlation | 4B held-out high/low score | Llama correlation | Llama high/low |
|---|---:|---:|---:|---:|
| o_proj | 0.787 | 2.31x | 0.771 | 2.46x |
| q_proj | 0.686 | 1.36x | 0.706 | 1.43x |
| v_proj | 0.786 | 1.77x | 0.805 | 1.76x |

High/low compares the training-ranked highest and lowest chunk quartiles. Much of the
signal reflects reconstruction/error-shape heterogeneity; dividing Fisher by raw error
weakens it. This is evidence of predictable information beyond activation magnitude,
not evidence of PPL gain. Only two held-out windows and one diagnostic length were checked.

A modest next discriminator would be two sensitivity classes for o/q/v, tested by the same
global cost-constrained solve and then actual NLL. The earlier class diagnostics ran only
on the corrupted 12-window Fisher sample; clean six-window jobs disabled class variants.
Deployment would require a static chunk-to-class map and two threshold vectors, so storage
and lookup cost need accounting. It changes the dispatch representation and is therefore a
larger commitment than the first pilot, though still purely an allocation mechanism.

Reproduction script: `investigations/round3_chunk_signal_20260926.py`.

The immediate workflow remains: finish round 2, refresh the h20-aware results and draft
the first rebuttal response, then decide whether the measured gain/headroom warrants this work.
