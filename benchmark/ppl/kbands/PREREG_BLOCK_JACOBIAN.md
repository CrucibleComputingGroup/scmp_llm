# Pre-registration — per-row block-local Jacobian weighting (B1)

Written 2026-08-04 BEFORE any run. Criteria fixed here.
Fixed constraints: **INT mask stays at 20%**, budget unchanged, no runtime
hardware. This is an ALGORITHM change to per-group allocation only.

## 1. The defect, in one line of code

`calibrate_mp_thresholds.py:1547`

    level_errors.append(_relative_l2_rows(sc_out, teacher))

σ is the relative L2 error of **the operator's own output**. It is purely local:
it measures how wrong a group is, never how much that wrongness matters
downstream. The allocator then compares σ across operators as if a unit of error
costs the same everywhere.

Consequences, all measured:

* σ → PPL leverage 0.36 (4B), 0.33 (14B), and **0.00** past the knee — more σ
  bought, zero PPL.
* Same σ buys a **2.4×** different PPL at two budgets on llama8B.
* Attention mispriced **5-20×**; "solver sells attention first, PPL punishes it."
* The allocator already captures **~96%** of the per-group oracle ON THIS METRIC
  (deployable within 0.8-1.6pp). We are solving the wrong problem well.

## 2. Why granularity is collapsing where we need it

From the deployed traces, MAC-weighted stream-length distribution:

| cell | MACs at ladder FLOOR | length entropy |
|---|---|---|
| 4B t32 | **44.1%** | 2.34 / 3.00 bits |
| 30B t32 | 13.5% | 2.56 / 3.00 |

On the tightest cell, 44% of compute sits at a single length — the per-group
granularity that is the contribution is degenerating exactly at the budgets we
care about, and t24 will be worse.

## 3. Why P1 failed and why this is not P1 again

P1 fixed the CURRENCY but at operator granularity, using an END-TO-END ΔLoss
knock-down probe. That probe's noise scales with depth: negative (impossible)
entries **8 / 31 / 39 / 56%** for 4B / llama8B / 14B / 30B. Unusable above ~8B.

B1 keeps per-ROW granularity and replaces the estimator:

    d_row(L) = || d(h_block · u) / d y_row ||  x  sigma_row(L)

where `h_block` is the enclosing transformer block's output and `u` a fixed
probe vector. The Jacobian is taken **one block deep**, not through 36 blocks
and the loss, so it is far lower variance. It trades a little bias (block-local,
not task-loss) for a lot of variance — the correct trade given the numbers above.

Prior art: block-wise reconstruction (BRECQ-style) beats layer-wise objectives
for exactly this reason — local error does not predict task loss. Our five
recorded instances of error-axis gains failing to transfer are the same effect.

## 4. Implementation (no runtime change)

Reuses the existing per-row gradient plumbing verbatim: `PendingMerger.record`
already stashes each call's output tensor and reduces a backward signal to one
`g_row` per calibrator row, with attention-specific reduces that keep qk/av
aligned. Only the SOURCE of that backward changes.

`--grad-source {loss,block}` (new, default `loss` = current behaviour, must be
byte-identical). With `block`: capture each decoder block's output, then per
block run `torch.autograd.grad((h_b*u).sum(), inputs=<op outputs in block b>,
retain_graph=True)`. Because every `y_row` is inside block b and `h_b` is that
block's output, the chain rule path is confined to block b automatically — no
contamination from other blocks, by construction rather than by masking.
Weights are consumed offline; the deployed artifact still receives only `L_g`.

## 5. Pre-registered predictions

1. **Allocation shifts toward attention and toward high-gain rows**, and the
   floor mass on 4B t32 FALLS from 44.1%. If the allocation does not move, the
   mechanism is wrong regardless of PPL.
2. Gain is largest where σ misprices most: 30B and 4B at tight budgets.
3. B1 beats the in-wave uniform-σ control on cost-adjusted PPL.
4. The probe is LOW variance: the per-row gains must be **non-negative by
   construction** (they are norms), so the sign-violation failure mode that
   killed P1 cannot occur. Report the gain distribution's spread instead — if
   gains are near-constant across rows within an operator, B1 degenerates to a
   per-operator rescale and should be expected to behave like P1.

## 6. Refutation criteria — fixed in advance

REFUTED if any of:

* Floor mass and allocation do not move vs the control (prediction 1).
* Per-row gains are near-constant within operators (coefficient of variation
  < 0.1) — then this is P1 in disguise and carries P1's verdict.
* Fails to beat its in-wave control on cost-adjusted PPL on >=3 of 4 models
  (sensitivities 14B 0.134, 30B 0.401, 4B 0.348, llama8B 0.279).
* Any delta below the noise floor: sd 0.0069 PPL, |delta| < ~0.014 is not a
  result.

## 7. Controls

* `--grad-source loss` must remain byte-identical to today (asserted).
* Control arm = same code path, `--cross-layer-weight uniform`, no gradient
  weighting, calibrated in the SAME wave. The deployed mp_best tables went
  through v16/v20 search and are NOT a valid control.
* Mask fixed at the deployed 20%; budget unchanged. Any PPL move is the
  allocation.
