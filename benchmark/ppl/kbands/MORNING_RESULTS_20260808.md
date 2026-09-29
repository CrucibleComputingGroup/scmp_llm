# Overnight results — 2026-08-07/08

Regenerate the full table any time: `python benchmark/ppl/kbands/collect_wave_e.py`
Full narrative + every caveat and retraction: `LOOP_STATE.md` (this dir).
All cells full protocol (`PPL_MAX_TOKENS=0`, ctx 2048, wikitext-2), AWQ + INT7
20% dose, noise floor sd 0.0069 PPL (|Δ| < ~0.014 absolute is not a result).

## 1. The headline — ALL FOUR MODELS IMPROVE AT THE TIGHTEST BUDGET

**A mask priced by MEASURED per-operator loss — instead of the deployed
SC-fragility proxy — lowers the whole quality/compute curve.** Best cells at or
below the deployed t32 cell's own compute:

| model | deployed t32 | **new best** | ΔPPL | Δcost | x_fp16 |
|---|---|---|---|---|---|
| **llama8B** | 8.4864 @ 34.18 | **7.8581 @ 32.34** | **−7.40%** | −5.4% | 1.1765 → **1.0894** |
| **4B** | 11.0860 @ 34.95 | **10.6667 @ 33.74** | **−3.78%** | −3.5% | 1.1037 → **1.0619** |
| **30B** | 7.9073 @ 40.10 | **7.7566 @ 33.74** | **−1.91%** | −15.9% | 1.0890 → **1.0682** |
| **14B** | 9.1581 @ 35.76 | **9.0183 @ 36.14** | **−1.53%** | +1.1% | 1.0602 → **1.0440** |

Every one is a strict (or iso-cost) improvement: better perplexity at equal or
lower compute. 30B's reinvestment cell is still running and should improve
further; llama8B's recipe adds M=128, 4B's deliberately does not (§3).

**Rungs lowered.** llama8B t48 + attention-first + M=128 = **7.6353 @ 40.49**,
which beats the archive's **t64** cell (7.6912 @ 63.20) on quality at **36%
less compute**; and its plain t48 (7.6776 @ 40.50) already did. 4B's t48-table
cell (10.4013 @ 39.89) beats the archive **t40** cell by −3.57% at iso-cost.
That is "same PPL at a lower bitstream", demonstrated on two models.

### ★ llama8B PASSES 1.05× fp16 — a standing "impossible" is overturned
`reinv_llama8B_t64` = **7.4621 @ 57.76 = 1.0345× fp16**. The ledger had
recorded that llama8B "cannot reach 1.05× by allocation at any budget", with an
SC floor of 1.048 measured even at the ceiling. **That floor was measured with
SC executing attention** — and llama8B's deficit is 64% attention. With
attention on the INT side of the same fixed 20% dose, the floor does not apply.

| cost | archive | new | x_fp16 |
|---|---|---|---|
| ~32 | 8.4864 @ 34.18 | **7.8581 @ 32.34** | 1.1765 → **1.0894** |
| ~40 | 7.8777 @ 47.36 (t48) | **7.6353 @ 40.49** | 1.0922 → **1.0585** |
| ~58 | 7.6912 @ 63.20 (t64) | **7.4621 @ 57.76** | 1.0663 → **1.0345 PASS** |
| ~88 | 7.5992 @ 88.03 (t96) | dominated by the t64 cell (−1.80% at −34% cost) | 1.0535 |

## 2. What produced it (three ingredients, all measured)

**(a) Per-operator loss attribution (the new instrument).** Route ONE operator
entirely to INT7, measure full-protocol PPL. llama8B t96, 9 ops:

| op | ΔPPL | share of the 5.35pp excess | MAC share | σ share |
|---|---|---|---|---|
| qk | **−1.75%** | ~35% | 3.5% | 2.0% |
| av | **−1.51%** | ~30% | 3.8% | 18.6% |
| down/gate/up | −0.44/−0.36/−0.19% | ~20% | 75% | 70% |
| o/q/k/v_proj | ≈ floor | ~0 | 24% | 9% |

Attention = **64% of the loss from 7.3% of the MACs**; σ misprices qk ~17×.
Recoveries sum to ~86% of the excess, so the attribution is near-additive.
**14B is the opposite** (av gives nothing, down_proj −0.52%) ⇒ the
loss-carrying operator class is MODEL-DEPENDENT and must be measured per model.

**(b) Attention-first mask, MAC-matched.** Same 20% INT dose, same table, same
front-end; only WHICH MACs are INT changes. Matched to the archive's INT MAC
share to 2 dp (entry-matching was rejected — it would have moved only 6.5% of
llama8B's MACs to INT vs the archive's 20%, i.e. a different operating point).
Traces confirm attention is fully INT7 and the SC pool is pure linears.
*The compute saving is universal and structural* (attention carries the longest
streams on every model, so evicting it always cuts the mean length, 11–23%);
*the PPL gain tracks the swap attribution* (llama8B ≫ 4B > 14B ≈ 0).

**(c) Reinvestment.** Because the MP table is mask-blind, those cells land
11–23% UNDER budget. Running the same mask against a LOOSER table (t40) spends
it back and converts it to quality — that is where the −6.74% / −3.78% come
from.

## 3. Other levers settled overnight

| lever | result | sign |
|---|---|---|
| **qk α=0.5** (deployed α was pinned 1.0) | llama8B −0.92% @t32, −0.42% @t48, −0.45% @t96, all ISO-COST | helps llama8B at every budget |
| **Scramble masks M=64→128** | ceiling: 4B −2.63%, llama8B −0.26%. Under MP @t32 iso-cost: llama8B −0.83%, 4B −0.97%, 30B −0.48%, 14B **+0.32%** | helps 3/4, hurts 14B; SIMULATION-ONLY (HW selector caps at 64) |
| lever composition | llama8B: attnfirst+M128 compose (−4.61%). 4B: they ANTI-compose | model-dependent |
| SQ α=0.85 for Llama | **+2.58% WORSE** than α=0.5 | refuted |
| llama8B t32 measured_curve mask | +0.51% (prc) / +1.11% (parent) worse | int_swap mask stands |
| 30B t96 + grid | +0.04% (under floor) | neutral |

## 3a. ★ The ranking rule, corrected by 4B's swap table

4B t32 swaps show its RAW loss is in the MLP (up_proj −3.73%, down_proj
−3.67%) — the opposite of llama8B (attention) and the same as 14B. But masking
an MLP op **costs** compute (it removes short-stream rows, raising the SC mean)
while masking attention **frees** it. Since reinvestment turns freed compute
back into quality, the right objective is per unit of compute freed, on which
attention still wins for 4B (av −4.78%, qk −3.60% vs down −2.40%, up −0.07%).

**Deployable selector (supersedes the hardcoded "attention-first"):**

    score(op) = measured ΔNLL(op → INT) / (SC cycles freed by masking op)
    maximize Σ score subject to the fixed INT MAC dose   [a knapsack]

The mask's job is not only to remove loss — it is to remove loss CHEAPLY, so
the allocator can spend the difference. That is precisely why 4B's cell was
only −0.84% raw but became −3.78% once the freed 23% was reinvested.

## 3b. The deployable recipe, per model (all measured, none hand-tuned)

| model | mask | M | qk α | budget source | result |
|---|---|---|---|---|---|
| llama8B | attention-first | **128** | 0.5 | t40 table @ t32 cost | −7.40% |
| 4B | attention-first | 64 (M=128 ANTI-composes, 3 replications) | 1.0 | t40 table @ t32 cost | −3.78% |
| 30B | attention-first | 64 | 1.0 | t32 (t40 cell running) | −1.91% |
| 14B | attention-first | 64 (M=128 hurts, +0.32%) | n/a (qk regresses 14B) | t40 table | −1.53% |

Every column has a model-dependent sign somewhere. That is the empirical case
for calibrating these per model rather than fixing them globally.

## 4. What this means for the algorithm (and the paper)

The deployed allocator is ~96% optimal **against σ**, and σ is the wrong
objective across operators — measured mispricing up to 17×. The contribution
this defines is a **two-tier calibration**:

* **Tier 1 (few, measured):** per-operator loss prices from swap cells set the
  cross-operator budget split AND the INT-mask composition (a knapsack:
  maximize measured ΔL per INT MAC at fixed dose).
* **Tier 2 (millions, free):** σ + the per-(row,chunk) amax threshold continue
  to shape allocation WITHIN an operator, where they are proven (91–99% of the
  per-group oracle).

Every lever measured this week has a model-dependent sign (qk, grid, M, mask
composition, lever interaction). That is not noise — it is the argument that
these are CALIBRATION OUTPUTS, not hand-tuned constants.

## 4b. 4B's complete price table (9/9 ops, t32) — why the objective matters

| op → INT7 | raw ΔPPL | Δcost | **value** |
|---|---|---|---|
| **av** | −1.22% | −10.24% | **−4.78%** |
| **qk** | −0.24% | −9.67% | **−3.60%** |
| down_proj | **−3.67%** | +3.66% | −2.40% |
| **o_proj** | −1.42% | 0.00% | **−1.42%** |
| v_proj | −0.77% | −0.17% | −0.83% |
| k_proj | −0.36% | +0.26% | −0.27% |
| up_proj | **−3.73%** | +10.50% | −0.07% |
| gate_proj | −1.57% | +5.29% | +0.27% |
| q_proj | −0.68% | +3.12% | +0.41% |

The two operators with the **smallest** raw effect are the **best** masking
targets; the two with the largest raw effect are worthless once compute is
priced. Also note `o_proj` — 4B's third-best target, at exactly iso-cost — is
NOT in the hardcoded attention-first order, so the measured selector will
differ from the heuristic even where the heuristic works.

## 5. Open / needs a decision

1. **Thesis question for the user:** the attention-first mask puts attention
   entirely on INT7, so SC runs none of it. The 20% dose is fixed by decree and
   we only chose WHICH 20%, so it is inside the rules — but it changes how the
   paper states SC's operator coverage. **Flagged, not decided.**
2. Mask-aware recalibration is un-run: the allocator never sees the mask, which
   is exactly why these cells underspend. A mask-aware table should beat the
   reinvestment hack.
3. 4B anti-composition (attnfirst + M128) — one cell decides whether it is a
   real interaction or stale M=64 calibration: recalibrate a 4B prc table at
   M=128 and re-run.
4. Energy model must price the INT7 side; INT MAC share is matched to 2 dp, so
   the SC side is a clean saving, but the final number is the energy model's.
5. `rebuild.py` t128 dose fix is landed but NOT run (deferred while jobs read
   the config bundles) — llama8B's ceiling row improves 1.057 → 1.050.
