# (L × grid) per-group precision — the next algorithm (design, 2026-08-07)

User direction: "use the MP algorithm to reach similar PPL at the lowest
possible bitstream."

> ⛔ **ASYMMETRIC / UNIPOLAR SC IS REJECTED (user decision, 2026-08-07).**
> SC's point is energy; bipolar sign-magnitude HALVES the cycle count and that
> halving IS the energy advantage. Unipolar has no halve trick — matching the
> bipolar grid costs ~2× cycles ⇒ more energy — and a dual-mode PE is new
> runtime hardware (violates [[feedback_runtime_free_no_new_hardware]]).
> An earlier draft of this file proposed rungs = (L, grid, mode); the mode
> axis is dead. The probe (probe_asym_real.py) was cancelled before running;
> do not re-launch it or re-derive the unipolar path for the LLM. The kernel's
> mode="unipolar" stays as-is for other apps; it is not an MP lever here.

## 1. Verdict: is the space of `src/framework/space.tex` fully utilized?

The paper defines the space as: each of the G computation groups (128-element
quantization groups) takes any of 128 stream lengths; |Ω| = 128^G.

What the deployed algorithm actually realizes, with measured utilization:

| projection of the space | measured coverage | verdict |
|---|---|---|
| lengths per cell: ≤7 calibrated rungs (of 128) | 7 rungs ≈ 95–99% of a 14-rung oracle; pow2-only ladders cost +3.27% ⇒ the *choice from the continuum* is load-bearing, the *count* is not | exhausted |
| assignment = monotone threshold on per-group amax | amax captures 91–99% of the measured-error water-fill oracle, 7/7 cells | exhausted (within σ) |
| per-(row,chunk) granularity | deployed; allocation ~96% of per-group oracle | exhausted (within σ) |
| cross-layer split | frozen at per-row-era parent means | open, but every objective tried failed (probe noise 31–56%) |

**The LENGTH axis, under the σ objective, is used up.** The remaining room is
in what the formulation leaves out:

1. **The grid axis** — `rng_levels` decouples grid resolution from cycle
   count; fixed-point cannot (b bits fixes both). Deployed only as the fixed
   policy `pow2 ≤ L`: it moved 30B a full rung (t64→t48) yet HURT 4B/14B t32 —
   direct evidence a CALIBRATED per-rung grid beats any fixed policy, and the
   axis is cycle-neutral: same stream length, same datapath, the grid selector
   already exists in the kernel/architecture. This is the surviving 2-D
   program.
2. **The objective** — allocation is ~optimal in σ while damage-per-σ varies
   ~20× across models (llama8B). The grid axis dodges this by improving the
   error curves themselves rather than re-weighting σ.

## 2. The proposal: rungs become (L, g) — calibrated grid per rung

Keep dispatch untouched (per-(row,chunk) amax thresholds → rung index).
A rung of (op, layer-bucket) becomes

    rung k  =  (L_k, g_k)
      L_k : stream length in cycles — the ONLY cost carrier, as today
      g_k : enable-grid levels ≤ 128, pow2 (Owen scramble is a pow2
            construction — non-pow2 grids measured DEAD: won only at
            pow2 L, nothing at 18/24/48/96)

Why this is cheap and safe:
* **Cost stays 1-D.** Cycles depend only on L; the water-fill and the
  iso-compute identity are untouched. g_k is a per-rung argmin over MEASURED
  error at that L — separable, no combinatorial growth, no new runtime
  statistic (the grid is resolved at table load, per rung).
* **Energy-neutral by construction.** Same cycles; the grid changes WHICH
  quantization levels the same stream samples. No new hardware: `rng_levels`
  is an existing kernel argument and the trace already records it.
* **Subsumes the deployed policy.** SC_RNG_GRID=pow2 is the special case
  g_k = largest pow2 ≤ L_k for every rung. The measured per-cell split
  (30B −1.42/−0.96%, 4B/14B t32 +0.4/+0.5%) is exactly what a calibrated
  per-(op,bucket,rung) choice would arbitrate instead of a global env flag.
* **Where it should pay:** floor rungs (L ≤ 32) at t24–t32, where a 128-level
  grid under an 18–32-cycle stream is mostly sampling noise (measured:
  −4.21% rel-L2 at L ≤ 64 with the pow2 policy; largest on q/k/v — the
  attention feeders σ misprices), and wherever the escape gate pins 128-length
  streams whose grid could stay fine while FLOOR rungs coarsen.

## 3. Why this serves "lowest bitstream at similar PPL"

At t24–t32 the mass sits on floor rungs, where allocation is exhausted (bands
saturated, prc deployed, V20 regressed) — the only lever that still moves the
error there without buying cycles is matching the grid to the stream. 30B
already moved a passing rung on the uncalibrated policy alone. The calibrated
version is the natural candidate to move the next rung, and it is the paper's
sharpest "finer than fixed-point" statement: fixed-point picks one number b;
we pick (length, grid) per group with cost tied only to length.

## 4. Build order (each stage gated by measured error BEFORE any PPL cell)

1. **E(L, g) surfaces per (op, layer-bucket)** — extend `probe_rng_grid.py`
   from its policy sweep to the full grid {8,16,32,64,128} × ladder-L on REAL
   residual operands (prc-calibrator capture machinery: protected channels
   removed, AWQ applied). Output: per-(op,bucket,rung) best-g table + the
   error margin over g=128 and over pow2≤L. Gate: calibrated g beats the pow2
   policy by ≥10% rel-L2 on cells where the policy REGRESSED (4B/14B t32) —
   i.e. the win must come from arbitration, not from rediscovering the policy.
2. **Table plumbing** — `per_rung_grid` section in table.json (loader
   validation round-tripped through the DEPLOYED parser — the four-instance
   two-resolvers bug family), `sc_common` passes rng_levels per rung; replaces
   the SC_RNG_GRID env policy. Byte-identical when the section is absent.
3. **PPL wave** at t32 (4 models × {parent-grid-policy, calibrated-grid}) with
   identity controls; then **t24 calibration wave** (no parents exist) where
   the lever should pay most.

## 5. Honesty ledger

* Six prior instances of error-axis gains failing to transfer to PPL. Stage 1
  is an error-axis gate; PPL cells decide. Identity controls per model+target.
* Grid choices must stay pow2 (scramble structure); never sweep rng_levels
  above 2^sc_prec or above what the stream can express meaningfully.
* The M=64 scramble-mask cap interacts with this axis (mask reuse across the
  RoPE pair at head_dim 128): the m128 ceiling ablation (jobs 56733371/72)
  prices that separately; if it matters, it is an architecture parameter
  question (mask-selector width), not an allocation lever.
* Per-cell arbitration is calibration OUTPUT (like qk/grid today), never
  hand-tuning, per the paper's one-algorithm framing.
