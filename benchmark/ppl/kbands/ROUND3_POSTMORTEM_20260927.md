# Round-3 postmortem — September 27

Recommendation: use the completed LLM results for the first rebuttal response now.
Do not expand the coarse threshold-shift sweep. A subsequent small diagnostic of
genuinely adjacent-rung exchanges could repair the design weakness below; it is
not yet evidence of achievable improvement. No round-4 jobs have been launched.

## Results and prediction consistency

Twenty complete candidate tables were tested on six paired TRAIN search windows
(8 for 4B, 4 for Llama, 8 for 30B). Nineteen worsened measured loss. The best new
candidate in each cell was then measured on sixteen fresh confirmation windows
and the full test set; the incumbent remained eligible for final best(all).

| Cell | Search ΔPPL | Confirmation ΔPPL | Full-test ΔPPL |
|---|---:|---:|---:|
| 4B, 6 bits | +0.545% | +0.431% | +0.212% |
| Llama 8B, 6 bits | +0.665% | +0.354% | +0.342% |
| 30B, 6.32 bits | −0.052% | −0.045% | −0.030% |

Thus there was no strong measured quality win that reversed on test. The two
dense-model test candidates were the least bad new candidates, deliberately run
as diagnostic full tests. All three confirmation effects failed the predeclared
z < −1.5 criterion. The 30B effect is effectively flat (confirmation z=−0.15).

Baseline no-op/profile/restore identity checks passed. Predicted compute was
accurate: largest actual-versus-predicted relative cost discrepancy was about
0.345%, 0.537%, and 0.066% for the three cells. There was no Fisher prediction of
candidate NLL in this pilot: proposal generation used dispatch histograms for
cost, followed by direct measured NLL.

## Design weakness: global budget locality did not imply group locality

The implementation adds one common offset to all thresholds of a population.
It bounds net transferred SC cycles, but does not bound a group's rung distance,
the fraction of changed groups, or each affected operator's relative budget.

Llama QK occupied 10.192% of baseline SC cycles. The nominal 2%-of-total QK-to-
linears transfer cut QK's own realized cycle budget by 20.016%. Every QK bucket
started with five thresholds equal to zero. Moving all five together preserved
their coincidence, sending affected rows directly from the longest ordinary
rung to the shortest. About 25.30% of actual QK MACs landed at the shortest rung,
with none on intermediate rungs. In nominal paper units this is a jump from
192 to 38 cycles; code uses deployed lengths 96 to 19. Search PPL worsened 17.60%.

The AV-to-linears move similarly cut AV's own cycles by 16.96% and worsened
search PPL by 4.10%. These are large interventions within the affected operator.
This is a concrete flaw in describing the pilot as a small local refinement.
It is a plausible explanation for the large regressions, not a measured causal
decomposition separating attention damage from the simultaneous linear changes.

The original judgment also overestimated remaining opportunity in broad
operator-family transfers. The known Fisher oracle gap mostly concerns which
individual groups deserve precision inside a bucket. Common threshold shifts
do not add that information, and the pilot did not use marginal Fisher errors
to choose boundaries. It tested one transfer scale and a narrow family of
coarse proposals, not the full space of per-group allocation refinements.

## Implication for another round

The failure refutes these coarse directions at this step size. It does not prove
allocation is exhausted. If another pilot is warranted after the first response,
the cleanest test is a single-cell adjacent-rung proposal: move one boundary at
a time, cap changed-group fraction and per-operator budget change, use smaller
total transfers, and confirm an actual loss improvement before broadening.
Fixed RNG, masks, frontend, QK operands, ladders and deployed representation
remain mandatory. Static chunk classes are a later backup because they require
additional dispatch metadata and still have no end-to-end win.

Evidence: `prc_local_20260926_results.json`; Turbo
`prc_local_20260926/<cell>/{profile,search_results,selected}.json`, candidate tables
and exact search traces; implementation `../prc_local_proposals.py`.
