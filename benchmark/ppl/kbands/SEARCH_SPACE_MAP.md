# SEARCH-SPACE MAP — what is deployed, refuted, and open (2026-08-07)

Deliverable for NEXT_SESSION_PROMPT Direction 2. Evidence pointers are to
`LOOP_STATE.md` (LS), `scmp_llm/CLAUDE.md` ⚑ (CM), `hpca_results` dirs, or this
session's analysis (S, in `mp_best_after_hpca_3`-era artifacts). Written to be
sufficient to (a) pick the next lever and (b) fix the paper's design-space section.

## 0. The axes

Per computation group the deployed system (or a future version) can choose:

| # | axis | cardinality per group | status |
|---|---|---|---|
| A1 | stream length L | 128 | deployed (the paper's only counted axis) |
| A2 | quantization grid (rng_levels) | ≤ log2 choices tested; in principle ≤128 | partially deployed (pow2≤L policy) |
| A3 | grid symmetry / zero-point | 2 (symm/asymm) + offset continuum | UNTOUCHED |
| A4 | backend (SC vs INT7) | 2 | fixed 20% dose by decree (out of scope) |
| A5 | operand conditioning (front-end / qk / av scales) | continuum, offline | linears + qk done; av NONE |
| A6 | dispatch statistic | 3 (amax/l2/crest) × sign | per-row: ρ-selected; prc: amax only |
| A7 | granularity of A1 | row → (row,chunk) | deployed at (row,chunk); per-element absurd |
| A8 | cross-layer budget split | continuum | pinned to per-row-era parents (open) |

The paper counts only A1 (|Ω| = 128^G). The honest deployed space is at least
(A1 × A2) per group; the paper fix is §4 below.

## 1. Deployed (all full-protocol, archive = `mp_best_after_hpca_3`, 20 cells)

| lever | measured value | scope caveat |
|---|---|---|
| per-row length MP (v9→v20 heritage) | baseline for everything below | per-row axis is worth only −0.7…−7.2% squared error of the −15…−75% available (LS operator map) |
| per-(row,chunk) dispatch | −3.35% mean PPL era-2; core of 16/20 winners | archive cells are sample-budgeted (overspend disclosed in manifest `cost_adjusted_value_pct`); iso-strict variant (`--parent-trace`) shrinks wins |
| qk operand rebalance α=1.0 | 30B −8.87/−5.58/−3.77% @t32/48/64; 4B −2.1% | PER-CELL: regresses 14B (+0.91%) and llama8B t32; helps most where Q/K spread is large, but spread is NOT predictive cross-model (refuted screen) |
| SC enable-grid pow2≤L | 30B t48 −0.96%, t32 −1.42% (cheaper AND better); 4B t40 −0.36% | neutral llama8B; small real regressions 4B/14B t32 → per-cell |
| calibrated (non-pow2) ladders | pow2-restriction costs +0.91…+5.02% cost-adj on 4/4 | THE "finer than fixed-point" measurement; 43 rung values, 39 non-pow2 |
| AWQ front-end | 12/12 wins, mean −1.93%, grows as budget tightens | breaks front-end-match to INT baselines; never table vs INT |
| hybrid INT mask 20% | dose curve in `int_ablation/`; 5% captures 64% of 0→20% gain | priced on ENERGY axis, not iso-compute; llama8B dose response ~linear (diffuse) |
| escape gate k=2.0, len 128(halved) | in every parent | threshold = μ+2τ on min–max-NORMALIZED per-call stat |
| protected channels 1% @128 | in every parent | gathers arbitrary channels ⇒ numerics never validated vs unsplit (LS open thread) |

## 2. Refuted / dead — with the killing evidence (do not redo)

| axis / idea | killed by |
|---|---|
| static K-bands (per-chunk, static) | superseded by prc; on real activations per-chunk is WORSE than per-row on 14B (−3.9% vs −5.2%); explains 14B band regression |
| band count > ~8–16 | 16→35 bands moves PPL 0.00017 (25× under noise floor) |
| σ cross-layer re-weighting: measured, measured_marg, grad, grad_group, grad_sc, fisher, P1 pooled, B1/B2 Jacobian | 6+ recorded instances of proxy-gain non-transfer; probe negative-ΔLoss fraction 8/31/39/56% (4B/8B/14B/30B); B1 catastrophic (+123/+489%) |
| barrier / cliff-penalty objective | σ already 2.3–2.7× convex at the floor — wrong premise |
| composite dispatch statistics | amax×‖w‖ adds ≤0.3pp; static-chunk+row-shift collapses to 41–47% of oracle |
| Q/K-spread screen as cross-model predictor | 14B (2nd-highest spread) is the lone qk regression; llama8B (lowest) gains |
| hot_frac tuning on asymmetry | asymmetry is a diagnostic, not the objective; optimum interior (0.25) |
| rotation (orthogonal operand maps) | σ rises on reached ops in every geometry (o_proj lone exception) |
| int_swap mask ranking | loses to measured_curve 2/3, tie 1/3 (frozen-wrapper caveat); ROBUST version staged, unlaunched |
| ladder SIZE beyond ~7 rungs | 7 rungs ≈ 95–99% of 14-rung oracle |
| per-inference coexistence COUNT (paper) | escape+protected falsify any fixed count; per-unit options is the honest axis |
| "llama8B down_proj = amax INVERTED" (memory) | WRONG vs deployed configs: llama8B down_proj = l2/+; the amax/INV cells are 14B o_proj t64/96 (S; memory corrected 2026-08-07) |

## 3. Open axes, ranked by expected value

EV = (mechanism-level headroom) × (probability of PPL transfer, priced by this
project's history: error-axis→PPL transfer has failed ~6 times, succeeded ~3
— prc, qk, grid — always when the lever changed WHAT is represented, not how the
budget is shuffled).

1. ~~A3 asymmetric SC / zero-point~~ — **CLOSED BY USER DECISION 2026-08-07.**
   SC's point is energy: bipolar sign-magnitude HALVES cycles and that halving
   is the energy advantage; unipolar has no halve trick, so matching the grid
   costs ~2× cycles ⇒ more energy, and a dual-mode PE is new runtime hardware
   ([[feedback_runtime_free_no_new_hardware]]). The error-budget zp numbers
   (+12.9pp, INT asymm gains) remain true but are not reachable inside SC's
   energy story. Do not revisit. (probe_asym_real.py cancelled unrun;
   mode="unipolar" in the kernel stays for non-LLM apps only.)
2. **A2 calibrated per-rung grid** — NOW THE PRIMARY ALGORITHM LEVER
   (design: `PLAN_2D_PRECISION.md`). Cycle-neutral (same L, coarser enable
   grid, existing rng_levels selector = no new hardware). Evidence the policy
   version leaves value on the table: pow2≤L wins 30B a rung (t64→t48) yet
   REGRESSES 4B/14B t32 — a calibrated per-(op,bucket,rung) choice arbitrates
   this per cell. Also the sharpest "finer than fixed-point" paper claim
   (fixed-point b bits fixes grid AND cost; SC picks (L,g) with cost = L only).
3. **A5 av conditioning** — MEDIUM EV, blocked on a mechanism. av σ@128 is the
   worst per-unit error of any operator on every model (0.22–0.36) and llama8B's
   is the worst of all (0.361) (S, t96 decomposition). But: softmax rows are
   simplex-valued (non-negative, sum 1, near-one-hot) — a contracted-dim scale
   s_d on (A/s, V·s) IS available (same identity as qk; V-side folds into
   nothing — it's activations both sides). Any proposal must handle
   length-dependence (contraction over seq positions). No cell until a
   score-invariant, runtime-free form is written down and error-probed.
4. **A8 joint cross-layer × prc** — MEDIUM-LOW EV. Biggest untested ALLOCATION
   axis, but every cross-layer objective tried has failed to transfer, and
   allocation is ~96% exhausted vs the per-group oracle within groups. Needs a
   sensitivity-weighted objective that does not exist yet; the probe that would
   supply it is the one that fails at scale (negatives 39–56% on 14B/30B).
   Discriminator first: paired/common-random-number re-probe on 4B (cheap); if
   the ratio stabilizes and reweighting still loses, close A8 permanently.
5. **budget underspend redistribution** — LOW-MEDIUM EV, cheap and mechanical:
   water-fill leaves 5–15% unspent (LS); a second pass converts to length. But
   the parent-budget mode exists precisely to hold the cross-layer split fixed —
   redistribution reopens it; do it WITHIN operators first.
6. **A6 per-op dispatch metrics for prc** — LOW EV: amax already captures
   91–99% of the per-group oracle on 7/7 cells; the residual is ≤9pp of an axis
   whose PPL transfer is sub-linear. Only worth bundling into some other wave.
7. **t24 budget point** — not a lever, a NEW OPERATING POINT. Needs a
   calibration wave (no parents). Grid lever should pay more there (deeper
   floor regime). Schedule after the next lever lands, not before.
8. **SQ α=0.8–0.9 for Llama** (S, front-end audit: repo docstring recommends it
   for Llama, code ships 0.5 for all) — LOW EV: AWQ (α-free, own search)
   reproduces the same llama8B floor within 1pp, and llama8B's scales are the
   tamest of all models. One cheap cell would close it; do not expect >1pp.

### 3b. Additions from the 2026-08-07 code audit (see LOOP_STATE addendum)

9. **t128 mask-dose fix** — `mp_best/rebuild.py:171-183` hard-codes 10% for
   every model's ceiling cell; llama8B's measured 20% cell is 1.050 vs shipped
   1.057. FREE (already-measured data), but touches the frozen archive ⇒ OK
   required.
10. **qk hygiene bundle** (cheap, mostly llama/30B): calibrate σ + dispatch
   metric WITH the deployed smooth_scales (`_sc_attn_matmul_at_level` lacks the
   param; crest is scale-free ⇒ max perturbation); sweep α per model instead of
   pinning 1.0 (wrong end when |Q|≈|K|, llama's exact case); optional
   (H, head_dim) per-head scale table — the only way llama's qk structure is
   expressible (head-pooling erases it; Qwen tolerates pooling only because
   q/k_norm gains are head-shared). Low EV for llama PPL (its qk σ is already
   the best of all models) — file under correctness, not headroom.
11. **Scramble-mask / RoPE-pair resonance probe** — M=64 masks over
   D=head_dim=128 puts dims (d, d+64) — exactly the rotate_half pair — on ONE
   Owen mask (correlated noise on the most correlated coordinates), and av's
   position 0 (the attention sink) gets the identity mask. Model-agnostic
   full-length noise source ⇒ cannot explain llama-vs-Qwen, but it is a
   candidate component of the universal ~W5.5-equivalent ceiling noise (S).
   One simulation-only ablation (SC_SCRAMBLE_MASKS=128, HW_MAX_MASKS cap
   relaxed) prices it. If it pays, it interacts with A2 (grid) and the paper's
   ceiling story.

## 4. Paper design-space section — required fixes

1. |Ω| = 128^G counts ONE axis. With the deployed enable-grid the per-group
   choice is (L, grid) — grid ∈ {pow2 ≤ L} adds ~log2(L) options per group.
   RECOMMENDED FRAMING: keep the per-unit table at 128 length options (honest,
   already verified) and add grid decoupling as a MECHANISM claim, not a
   cardinality multiplier: "fixed-point b bits fixes grid AND cost; SC chooses
   them independently (a 16-level grid at 18, 24, or 32 cycles)".
2. The 7-rung-sufficiency ablation MUST be framed as "the right 7, chosen from
   a continuum": pow2-restricted ladders cost +0.91–5.02% cost-adjusted on 4/4
   (mean 3.27%) at equal-or-lower cost. The rungs cluster (30B t48: four in
   45–48) exactly where fixed-point offers nothing between 32 and 64.
3. Per-cell lever selection (qk, grid) is part of the deployed algorithm and
   must be described as calibration OUTPUT (per-model configuration), never as
   hand-tuning; 14B regressing on qk is the evidence the calibrator must gate
   levers per model.
4. The archive's prc cells are sample-budgeted (manifest discloses
   cost_adjusted_value_pct); if the paper needs strict iso-cost, recalibrate
   with `--parent-trace` and expect headline wins to shrink (CM ⚑ 2026-08-04).

## 5. llama8B (Direction 1) — resolution as of this session

See `LOOP_STATE.md` entry 2026-08-07 (this session) / session report. One line:
llama8B's ~5% is a CEILING property (present at uniform L=128 with no
allocator), its SC reconstruction error is NORMAL-to-LOW (mwσ@128 0.058 vs 4B
0.064 / 30B 0.083), and its own INT curve independently shows early-onset noise
sensitivity (only model above parity at W7A7; 5× Qwen excess at W6A6). SC at
ceiling ≈ W5.2-equivalent noise for it. Diffuse on SIX statistics ⇒ selection
levers are correctly exhausted. With A3 closed by user decision (energy), the
in-scope levers for llama8B are: whatever the op-swap wave localizes (loss-unit
attribution, in flight), the calibrated grid (A2 — but the pow2 policy was
within noise on llama8B, so expect little), the t128 mask-dose fix (+0.7pp,
free), and the m128 scramble ablation outcome. If those exhaust, llama8B is
reported as the noise-fragile case at its measured floor — a model property,
not an algorithm failure.
