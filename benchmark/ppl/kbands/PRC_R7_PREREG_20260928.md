# Round 7 pre-registration: family-κ joint λ over attention ladders up to 128 (2026-09-28)

**Status: BUILT, NOT SUBMITTED.** Nothing has run, and no round-7 output directory exists
(`/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r7_20260928/` is absent). There is no submission record and the capture
manifest is not pinned. Submitting needs the user's OK (§10 lists what to confirm first). At 16:41 EDT, `squeue -u allenjin`
was empty. Round 6 is complete (both 62208439 and 62208440 COMPLETED; record [`PRC_R6_RESULTS_20260928.md`](PRC_R6_RESULTS_20260928.md)),
so the ≤4-GPU cap applies to round 7 alone.

Plan context: [`ROUNDS_6_8_PLAN_20260928.md`](ROUNDS_6_8_PLAN_20260928.md) §4. Evidence:
[`investigations/T1_STALL_INVESTIGATION_20260928.md`](investigations/T1_STALL_INVESTIGATION_20260928.md) (RC3, RC6, §5c) and
the critic [`investigations/t1_stall_20260928/r6_critic_20260928.md`](investigations/t1_stall_20260928/r6_critic_20260928.md)
(L1, L3, L6/L7).

**Authority.** This document summarizes the frozen JSON records. Where it differs from them, the JSON governs.

| record | sha256 | frozen (UTC) | role |
|---|---|---|---|
| `kbands/prc_r7_prereg_20260928.json` | `c7fcfaec091ce01a…` | 19:40:51, before any round-7 output existed | MDE, κ pins, s0, fixed-point rule, arms, ladder policy, best(all) bound. Written once; `load_verified` passes (checked 16:41 EDT). |
| `benchmark/ppl/prc_r7_prereg_20260928.py` | `80ec74e5…` | — | Constants module. The screen builder refuses if it no longer equals the JSON record. |
| `kbands/prc_step0_r7_20260928.json` | `cdbfdbad5e3f168c…` | 19:48:53 | Step-0 manifest: 17 hashed sources, all verified unchanged. κ rule text included. |
| `kbands/prc_r7_capture_20260928.json` | `7a7d8121ffee4594…` | v2, before any capture | Capture manifest: per-cell `prereg` blocks, the κ rule verbatim, 28 hashed sources (all verified unchanged). Hashes the prereg JSON. |
| `kbands/prc_windows_r7_20260928.json` | `b06d64d9…` | — | Window registry: screen, confirm and exclusion sets. |

Units: stream lengths and costs are HALVED code units (nominal = 2×), with a code maximum of 128. %PPL ≈ 100·ΔNLL (nats).

## 0. What round 7 is

**One algorithm update:** a joint linear + attention λ, re-solved offline at parent cost, with four pure-allocation changes:
- **(a) Attention ladders reach 128.** Every qk/av bucket is offered ladder policy `up`: the parent ladder, plus 112 if
  112 ≥ 1.1·top, plus 128. This uses the existing `bucket_stoc_len_levels` runtime support.
- **(b) Family-calibrated currency.** The objective is κ_lin·ΔF_lin + κ_att·ΔF_att + λ·cost, so attention error is scaled
  by r = κ_att/κ_lin (CRIT L1).
- **(c) 512 attention rows per call** in the capture (was 128).
- **(d) One fixed-point target scale** so that the table spends the parent's cost (§4).

**What a table may change** (enforced by `validate_lineage`): per_row_chunk thresholds, attention thresholds, and attention
bucket ladders, with max 128.

**Held fixed:**
- the baseline lineage and INC (best(all));
- the h20 INT mask;
- the AWQ front end;
- SC RNG: bitrev, 64 masks, the fixed 128-level grid;
- no qk rebalance and no smooth-scale change;
- escape μ+2τ (`escape_gate_k` 2.0, code length 128);
- psl and the global ladder.

Also excluded: an INT rung, asymmetric SC, and any new runtime metadata.

**Escape disclosure.** Where 128 is a bucket rung, escape is a no-op on that bucket. On av (μ+2τ ≥ 1) the gate never fires,
so 128 is a genuinely new rung there. Escape status is reported per bucket, never as a blanket "no-op". On av buckets with a
128 rung and th[0] = 1.0, each call's max-metric row runs at 128 (pre-existing deploy semantics). Attention cost stays dense
(`mac = K·M`).

**Not in round 7:**
- **CRIT L6** (per-block attention keys) and **L7** (psl inside the joint λ). No fixed point or MDE path is registered for
  them, the screen builder refuses fold-in arms, and lineage forbids psl changes.
- **Standalone re-targets.** Round 6 refuted re-targeting on its best case (14B t32 full test +0.089%, no flip). So the
  plan-§4 fallback "re-target 30B t48 alone" is void. The s0/s1 scale inside round 7 is part of solving at parent cost; its
  share of any gain is disclosed through the J controls (§1).

## 1. Cells, the round-6 gate, and arms

**Primary rule** (registered for any gate outcome):
- 30B cell whose round-6 gate passed → primary **UK**.
- 30B cell whose gate failed or is missing → primary **K**.
- 30B t48 was not in the diagnostic and follows t40's gate.
- Dense cells → primary **U** at κ = 1. UK on a dense cell is the same allocation as U, so it is refused.

**Actual outcomes** (PRC_R6_RESULTS):

| cell | round-6 gate: ΔPPL(A) vs threshold | primary | arms | MDE, nats (≈%PPL) | s0 | status |
|---|---|---|---|---:|---:|---|
| 30B t32 | PASS: −1.981% vs −1.526% (z −5.2) | UK | UK, K, U | 0.0129 (1.29%) | 1.0037 | enabled |
| 30B t40 | PASS: −2.356% vs −1.252% (z −7.2) | UK | UK, K, U | 0.0092 (0.92%) | 1.0043 | enabled |
| 30B t48 | not measured; follows t40 → PASS | UK | UK, K, U | 0.0091 (0.91%) | 1.0110 | enabled |
| 4B t40 | FAIL: −0.602% vs −1.222% | — | — | 0.0148 | 1.0010 | not run |
| 4B t64 | marginal PASS: −0.440% vs −0.419% (z −1.9) | U (κ = 1) | U | 0.0073 (0.73%) | 1.0015 | **disabled by default; user decision** (§10 item 6) |

**Arms** (one capture per cell; offline CPU solves):
- **UK:** `up` ladders, κ-corrected currency, r = κ_att/κ_lin of the branch pins (§2).
- **K:** inherited parent ladders, κ-corrected.
- **U:** `up` ladders, κ = 1 (r = 1).
- Under the **revert** branch:
  - U equals UK and is not solved.
  - K runs at κ = 1, which equals the J control at s0 (disclosed).
- **J** (CPU control, never an arm): inherited ladders, κ = 1, 512 rows. J at s0 and at 1.0 gives an additive attribution
  of any gain into allocation, budget and resample.

**Screen composition.** The primary plus at most 2 secondaries (≤3 arms) plus the re-evaluated INC. Allocation-duplicates
of an earlier arm are dropped.

**Never candidates:**
- the capture's in-process tables;
- the pipeline-check table;
- J;
- `sensitivity_not_preregistered` entries (the `dense`/`full` ladder policies).

## 2. Step 0 and the κ rule (ONE rule, registered on both sides)

**What step 0 measures** (CRIT L3, `prc_step0_r7_20260928.py`, GPU, TRAIN windows):
- On 30B t40 and t32: the existing `c17` and `c17_s80` tables, a pure −20% linear-budget shift. t32 uses the `c17e32` pair.
  They run on the round-6 diagnostic windows, with INC bit identity before, between and after.
- On every screen cell: an INC 16-window profile (INC16) and the parent's exact cost on the same windows (PARENT16 → P).
- Task 2 (30B t48) is a reference-only cell on the 16 fresh `screen16_fresh_t48` windows. Its INC16 becomes t48's screen
  identity reference.
- Nothing from step 0 is citable or a best(all) candidate.

**Rule** (`prc_r7_solve.KAPPA_RULE` = `prc_eval_r7_20260928.KAPPA_RULE`, id `prc-r7-kappa-rule-v2-20260928`):
- **κ_att = 0.83 ± 0.07**: pooled held-out 30B value (CRIT §1; scout 0.8305 ± 0.0728).
- **Statistic.** κ_lin_bs = Σ_c m_c / Σ_c p_c over c ∈ {30B t32, 30B t40}.
  - m_c = step-0 16-window paired mean NLL(c17_s80) − NLL(c17).
  - p_c = pred_dnll_fis(c17_s80) − pred_dnll_fis(c17), the MC-Fisher prediction from the same cell's round-7 capture diag.
  - SE_bs = sd_w(Σ_c d_cw) / √16 / |Σ_c p_c|. Prediction noise is excluded (disclosed).
- **Keep** ratio 0.42 **iff** κ_lin_bs − 0.83 > √(SE_bs² + 0.07²).
- Otherwise **revert** to κ = 1. That covers three cases: inside the band; below it (`direction_contradicted`); or undefined
  (Σp ≤ 0, a cell missing or not ok, or a capture that failed identity). Undefined also sets `requires_user_review`.
- **Pins used in pred_true.** The measurement picks the branch; it does not set the value.

  | branch | κ_lin | κ_att | r |
  |---|---:|---:|---:|
  | keep | 1.96 | 0.83 | 0.4235 |
  | revert | 0.83 | 0.83 | 1 |
  | dense | 1 | 1 | 1 |

- 30B t48 takes the pooled decision.
- **Non-binding flags.** These set `requires_user_review` but never change the branch:
  - `direction_contradicted`;
  - `gap_case`: keep, but κ_att/κ_lin_bs > 0.55, i.e. κ_lin_bs < 1.51, where 0.42 may over-correct;
  - `cell_sign_inconsistent`: p_c ≤ 0 or m_c ≤ 0;
  - a step-0 cell that failed after `step0_measured.json`.
- **Decision record:** `$R7/step0/kappa_decision.json`, written by `prc_step0_r7_20260928.py kappa`. The screen builder
  requires it. The solver re-derives the branch from the record's statistic and refuses pins that disagree. The two sides
  agreed on 400 randomized trials plus 99 cross-checked cases.
- **Disclosed.** κ_lin_bs is fitted on the same 16 windows that the t32/t40 screens use. The confirmation windows are fresh.

**Departures from the task wording** ("keep 0.42 unless κ_lin_bs is within 1 SE of κ_att, then revert to κ = 1"). These
need the user's OK:
- (i) The band uses the combined SE, including κ_att's ±0.07, not SE_bs alone.
- (ii) The rule is one-sided. A κ_lin_bs clearly **below** κ_att reverts, where the literal wording would keep 0.42.
- (iii) Undefined reverts.
- (iv) "κ = 1" is pinned as 0.83/0.83, not 1/1. Under revert, pred_true is therefore 0.83 × the raw Fisher prediction,
  which makes refusal more likely than raw κ = 1 would.

## 3. Prediction, MDE, refusal

- **Prediction.** pred_true(arm vs INC) = κ_lin·ΔL + κ_att·ΔA.
  - ΔL and ΔA are held-out MC-Fisher predictions on the capture's hold windows (full-table scorer, split by family).
  - The branch pins are used, and they are the same for every arm of a cell.
- **MDE80** = (2 + z_0.80)/√32 · σ_w = 0.50233·σ_w, rounded up to 1e-4 nats. It is sized for the confirmation test: 32
  paired windows, one-sided z < −2, power 0.80.
  - σ_w is the largest paired per-window SD among the cell's pre-existing 16-window allocation pairs: prc2 `c7_heldout`
    gfisla−gfis, gfisla−parent, gfis−parent and c17−parent, plus round-6 A−INC. ATT128/LIN128 are excluded.
  - Values are in §1; the derivation is in the prereg JSON `mde_derivation`.
  - There is no override: `--mde` and `CELL=PATH@SCALE` are refused.
- **Refusal.** An arm is screened only if **pred_true ≤ −MDE**. This is signed, so it is stricter than "|pred_true| ≥ MDE":
  it also refuses predicted losses.
  - It applies to the s1 table that would be screened.
  - **A refused primary ends the cell:** no screen, no confirmation, no full test. Secondaries are never screened alone.
- **Descriptive only: κ_att(A)** = round-6 measured dNLL(A) / the capture's Fisher prediction of the round-6 A table vs INC.
  - If κ_att(A) − 2 SE > 0.83, the solver reports `qk_raise_underpriced` to the user.
  - It changes neither the currency, the refusal, nor the primary.

**Honest expectation.**
- The registered MDEs (0.91–1.29%) exceed the plan's §2 priors: 0.3–0.8% at t32 and 0.1–0.4% at t40/t48. **Refusal of every
  cell is plausible.**
- The only stronger signal is round 6's gross TRAIN reading. Arm A net of the gate's 0.6×chord funding term was:
  - ≈ −1.0 pp on t32: −1.981 + 0.6·0.2292·7.00;
  - ≈ −1.6 pp on t40: −2.356 + 0.6·0.1475·8.66.

  This is rough arithmetic, not iso-cost and not a pred_true.
- If every primary is refused, round 7 spends step 0 plus the captures (≈11–12.5 GPU-h) and adds no candidate. Its only
  outputs are then κ_lin_bs and κ_att(A), neither citable.

## 4. Target scale: one fixed-point pass (`prc_fixedpoint_r7_20260928.py`)

1. **s0 solve.** Solve each registered arm once at s0 (§1). s0 is the scout's held-out, trace-equivalent INC re-target to
   parent cost; the 4B values are approximate.
2. **Replay.** Compute C(s0) by replaying the arm's s0 table on the step-0 INC16 profile. Every profiled row is re-dispatched
   through the arm's **own** ladders and thresholds; escaped attention rows and the protected slice stay fixed.
   - P = step-0 PARENT16 exact trace cost on the same windows.
   - U = protected-slice cycles per SC MAC of the INC16 trace.
3. **Scale.** s1 = round(s0·(P − U)/(C(s0) − U), 1e-4). It is refused outside [0.97, 1.05].
4. **Re-solve once.** Each arm is re-solved once at its own s1. Only that table is screened. There is no iteration, no
   second re-solve and no re-screen.
5. **Recording.** The replayed C(s1)/P is recorded and flagged when it is beyond 0.25%; this is non-binding. **The binding
   check is the measured screen cost gate (§5).**
6. **Validation.** Replaying round-6 arm A on INC's profile predicted A's measured cost within 0.01–0.12% on 30B and within
   0.27–0.59% on 4B.
7. **Missing reference.** A cell without a complete, non-simulated step-0 reference is disabled.

## 5. Screen → confirm → full test (`prc_screen_r7_20260928.py`, one GPU and one model load per cell)

**Windows.** All windows are TRAIN, ctx 2048, and the token-stream sha must equal `9bdd7def…`.

| cell | screen windows (16, paired) | INC identity reference |
|---|---|---|
| 30B t32, t40 | round-6 diagnostic windows (= round-4 confirmation windows) | round-4 `confirm_incumbent`, NLLs and trace records bit for bit |
| 30B t48 | `screen16_fresh_t48` | step-0 INC16 plus the held-out probe on window 30720 |

- **Confirmation** uses `confirm32` for every cell: 32 fresh windows, disjoint from every earlier window set including
  prc2 held-out, rounds 3/4/6 and the capture windows (seed 260928).
- **P** comes from the step-0 `parent_reference.json`. The driver re-derives it from the reference trace.

**Screen gate** (primary only). All must hold:
- identity and every audit pass;
- the cost gate **|C/P − 1| ≤ 1%**, measured on the screen windows;
- the screen mean ΔNLL vs INC is **< 0**.

Secondaries are information only.

**Confirm.** Evaluate INC and the primary on `confirm32`. Proceed to the full test iff the paired **z < −2** and the audits
pass.

**Full test.** Only the pre-registered primary is tested: full protocol via `run_prc_ppl.sbatch` (full WikiText-2 test,
ctx 2048, `PPL_MAX_TOKENS=0`, batch 1, same settings as the incumbent). **One full test per cell at most.** Secondaries are
never confirmed or tested; this avoids CRIT's 0.1–0.15% selection inflation.

**Finalize checks:**
- the full-protocol header;
- a realized-length audit of the TEST trace;
- the run environment, parsed from `run_prc_ppl`'s own log: hybrid mask, `FRONTEND=awq`, 64 masks / HW_MAX 64, deployed RNG
  grid, attention grids unset, no qk scales.

**best(all) entry:**
- A confirmed primary's full-test PPL enters **whatever its sign**, provided every check passed and
  |C_test/P_test − 1| ≤ 1.5%.
- Beyond 1% it is flagged.
- Beyond 1.5% it is recorded but not entered (user decision).
- The final number stays the best full-test result across comparable rounds, with provenance. Any update goes to a **new
  dated** best(all) file.

**Audits on every arm trace:**
- realized lengths inside the arm's own ladders, both through the runtime resolver (`get_levels`) and through the JSON
  mirror;
- no length > 128;
- the same SC (op, block) set and per-(op, block) MACs as INC;
- an exact first-window profile replay.

A failed primary audit stops the cell. **Gap:** see §10 item 8.

## 6. Framing

- Round-7 gains are **per-row attention reallocation plus a calibration-objective change** (plus the s0/s1 budget
  correction, attributed via J). They are **not a T1 granularity effect**; report them under best(all) with provenance.
- The one-algorithm description is "joint λ with a family-calibrated currency over ladders that reach L_max".
  - As built, the offline solve exists only for 30B t32/t40/t48.
  - The other 17 cells are not evaluated in round 7: dense κ = 1, the round-6 gate failed or was not run, and CRIT verdict 5
    found no ≥0.5% lever on llama8B or 14B.
- The first rebuttal response does not wait for round 7.

## 7. GPU schedule (≤4 GPUs total; round 6 finished, current occupancy 0)

| phase | jobs | concurrent GPUs | GPU-h | wall clock |
|---|---|---:|---:|---|
| night 1 | step 0 `--array=0-2%1` (30B_t40 → 30B_t32 → 30B_t48) | 1 | ≈4.7 (+0.35 if 4B t64 task 3) | ≈4.7 h, sequential |
| night 1 | captures (A) `0-1%2` + (B) `2` | 3 | 6.0–7.8 (2.0–2.6 per cell) | ≈2.0–2.6 h |
| CPU | κ decision, s0 solves, fixed point, s1 re-solves, screen build | 0 | — | not timed on real captures |
| night 2 | screens `%K`, ≤3 30B cells (screen + confirm + full test in one job, 16 h wall) | K ≤ 4 − other running | 1.7–2.0 per cell; +≈4 per confirmed cell | ≈2 h, or ≈6 h with a full test |

- **Totals:** ≈10.7–12.5 GPU-h if every primary is refused at the solve; up to ≈30 GPU-h if all three cells reach a full test.
- **Lab-wide cap.** `nbleier_owned1` was at its lab-wide 10/10 cap during review, so jobs may pend.
- **Duplicated INC.** The t32/t40 screens re-run INC on the same 16 windows as step-0 INC16, about 0.45 GPU-h per cell. This
  is kept for the in-job identity check.

## 8. Submit commands (NOT executed; need the user's OK)

⚠ Array indices differ by launcher:
- step 0: 0 = 30B_t40, 1 = 30B_t32, 2 = 30B_t48, 3 = 4B_t64;
- capture and screen: 0 = 30B_t32, 1 = 30B_t40, 2 = 30B_t48, 3 = 4B_t40, 4 = 4B_t64.

```bash
# 0. Pre-checks (login node). Every GPU job of ours counts toward the 4.
squeue -u allenjin
cd /home/allenjin/Projects/SCMP/scmp_llm
PY=/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python
R7=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r7_20260928
sha256sum benchmark/ppl/kbands/prc_step0_r7_20260928.json    # expect cdbfdbad5e3f168cd3ce8192c0af2b1c...
sha256sum benchmark/ppl/kbands/prc_r7_capture_20260928.json  # expect 7a7d8121ffee45947ee24528fd30c5e9...
sha256sum benchmark/ppl/kbands/prc_r7_prereg_20260928.json   # expect c7fcfaec091ce01a2d06808f36d12ca6...

# 1. Step 0: one GPU, tasks sequential. (Only if the user enables 4B t64: --array=0-3%1.)
S0=$(sbatch --parsable --array=0-2%1 benchmark/ppl/kbands/run_prc_step0_r7_20260928.sbatch)

# 2. Captures: pin once (the capture manifest can never be rebuilt afterwards), then 2 + 1 GPUs.
PYTHONPATH=$PWD/kernels $PY benchmark/ppl/prc_r7_capture_20260928.py pin --manifest benchmark/ppl/kbands/prc_r7_capture_20260928.json
SHA=$(sha256sum benchmark/ppl/kbands/prc_r7_capture_20260928.json | cut -c1-64)   # still 7a7d8121...
CA=$(R7C_MANIFEST_SHA256=$SHA sbatch --parsable --array=0-1%2 benchmark/ppl/kbands/run_prc_r7_capture_20260928.sbatch)
CB=$(R7C_MANIFEST_SHA256=$SHA sbatch --parsable --array=2 benchmark/ppl/kbands/run_prc_r7_capture_20260928.sbatch)
#   The calib side's (B) line carried --dependency=afterany:62208440. That job COMPLETED at 15:45, so the
#   dependency is dropped here: sbatch can fail with "Job dependency problem" once a finished job is purged.
#   If any other GPU job of ours is running, use the conservative single array instead:
#   CA=$(R7C_MANIFEST_SHA256=$SHA sbatch --parsable --array=0-2%1 benchmark/ppl/kbands/run_prc_r7_capture_20260928.sbatch)
#   Then record $S0, $CA, $CB in NEW files kbands/prc_step0_r7_20260928_submission.json and
#   kbands/prc_r7_capture_20260928_submission.json.

# 3. ONE COMMAND (2026-09-28) runs everything below in order, checks the preconditions, stops loudly, prints the
#    per-cell summary and the throttled screen sbatch line (never calls sbatch; safe to re-run):
#      bash benchmark/ppl/kbands/run_r7_stage2_cpu_20260928.sh     # exit 0 done / 2 not ready / 3 ask user / 4 step failed
#    The individual registered steps it runs:
# 3. CPU chain, only after step 0 AND every capture finished (sacct COMPLETED; each $R7/<cell>/capture/identity.json
#    has identity_ok = true; exit 3 = identity fail/'near' -> STOP and ask the user; exit 4 = capture valid, pipeline check failed).
PYTHONPATH=$PWD/kernels $PY benchmark/ppl/prc_step0_r7_20260928.py kappa \
    --manifest benchmark/ppl/kbands/prc_step0_r7_20260928.json \
    --capture-manifest benchmark/ppl/kbands/prc_r7_capture_20260928.json          # -> $R7/step0/kappa_decision.json
for C in 30B_t32 30B_t40 30B_t48; do
  PYTHONPATH=$PWD/kernels $PY benchmark/ppl/prc_r7_solve.py solve \
      --manifest benchmark/ppl/kbands/prc_r7_capture_20260928.json --cell $C --out-dir $R7/$C/solve_s0
  PYTHONPATH=$PWD/kernels $PY benchmark/ppl/prc_fixedpoint_r7_20260928.py compute --cell $C \
      --solve-summary $R7/$C/solve_s0/solve_summary.json                          # -> $R7/fixed_point/${C}_fixed_point.json
  # It prints the ONE re-solve command (out-dir $R7/$C/solve_fixedpoint); run it verbatim, once.
done
PYTHONPATH=$PWD/kernels $PY benchmark/ppl/kbands/build_prc_screen_r7_20260928.py \
    --from-solve 30B_t32=$R7/30B_t32/solve_fixedpoint/solve_summary.json \
    --from-solve 30B_t40=$R7/30B_t40/solve_fixedpoint/solve_summary.json \
    --from-solve 30B_t48=$R7/30B_t48/solve_fixedpoint/solve_summary.json \
    --fixed-point 30B_t32=$R7/fixed_point/30B_t32_fixed_point.json \
    --fixed-point 30B_t40=$R7/fixed_point/30B_t40_fixed_point.json \
    --fixed-point 30B_t48=$R7/fixed_point/30B_t48_fixed_point.json \
    --kappa-decision $R7/step0/kappa_decision.json                                # -> kbands/prc_screen_r7_20260928.json
#   Omit a cell whose fixed point refused every arm. Cells whose s1 primary is refused come out disabled.
#   --accept-kappa-review only after the user has reviewed a flagged κ decision.

# 4. Screens: one GPU per ENABLED cell, K + every other running GPU job of ours <= 4.
#    No --dependency is needed: the manifest cannot exist before step 0 and the captures finish.
sbatch --parsable --array=<enabled task ids, e.g. 0-2>%K benchmark/ppl/kbands/run_prc_screen_r7_20260928.sbatch
#   Record the job ids in NEW kbands/prc_screen_r7_20260928_submission.json.
```

## 9. Hash locks and file rules

- **Round 6 and 4.** Never edit anything hashed by the round-6 manifests (`prc_r6_attn_diag_20260928.json`,
  `prc_r6_retarget_20260928.json`) or the round-4 manifests (`prc_adjacent_*`, calib7). All were re-verified unchanged at
  16:41 EDT (70/50/30/28 hashed paths).
- **Step-0 lock.** Once step 0 is submitted, do not edit `prc_eval_r7_20260928.py`, `prc_step0_r7_20260928.py`,
  `test_prc_eval_r7_20260928.py`, `build_prc_step0_r7_20260928.py` or `run_prc_step0_r7_20260928.sbatch`.
- **Capture lock.** After the pin, do not edit `prc_r7_solve.py`, `mp_per_row_chunk_calib10_r7.py`,
  `prc_r7_capture_20260928.py`, its tests, builder or launcher, or the prereg JSON.
  - Any edit to `prc_r7_prereg_20260928.py` makes the screen builder refuse, by design.
- **Editable until the screen manifest is frozen:** the screen driver, the screen builder, `prc_fixedpoint_r7_20260928.py`,
  `test_prc_screen_r7_20260928.py` and the screen launcher. None of them is hashed by step 0 or the capture.
- **Every job exports** `PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels`.
- **Tests at build time** (logs in `$SCRATCH/r7/`):
  - eval side: 47 OK (26 `test_prc_eval_r7` + 21 `test_prc_screen_r7`), `verify2/test_eval.log` at 16:06;
  - calib side: 52 OK, `verify3_calib_stdout.log` at 16:31;
  - dry runs of every launcher are in `fix_dryrun/` and `dry_*.log`.

## 10. Open issues (unresolved; the user decides)

**Before step 0 / the captures:**
1. **The κ rule departs from the task wording** (§2 items i–iv): one-sided, a combined-SE band, undefined reverts, and revert
   pins of 0.83/0.83.
2. **Conservative MDE and known-outcome risk.** The MDE sits at the upper end of the scout ranges (§3). Refusal on every cell
   is plausible, but it is only knowable after ≈11–12.5 GPU-h of step 0 plus captures (`feedback_no_known_outcome_trials`).
3. **best(all) test-cost bound of ±1.5%.** The task specified only the held-out |C/P − 1| ≤ 1% gate.
4. **Still open from review:**
   - pooling t32 + t40 for κ_lin_bs;
   - the screen futility clause "mean ΔNLL < 0" (not in the task's screen → confirm protocol);
   - 30B t48 inheriting t40's gate without its own diagnostic.
5. **Scope addition.** Step-0 task 2 (30B t48 reference, ≈+1.0 GPU-h) is not in the task's step-0 definition. It is needed
   for t48's fixed point and its identity reference.
6. **4B t64** passed its round-6 gate marginally, so the task's rule would include it.
   - The calib side registered it as *not run*: a known outcome, 0–0.3% plausible against an MDE of 0.73%. It stays disabled
     (fail-closed).
   - Enabling it needs, in order:
     - the user's OK;
     - step-0 task 3 (`--array=0-3%1`, +0.35 GPU-h);
     - a separate capture manifest (`build_prc_r7_capture_20260928.py --enable-4b --out <other>`);
     - a **new** launcher that reads that manifest (the capture sbatch hard-codes its manifest path).

**Before the screen:**

7. **30B MoE calib10 block-mode bit identity has not been shown on GPU.** An `identity_level` of `near` is
   stop-and-report (exit 3) for a user decision. The screen builder's "near is flagged" path cannot be reached under the v2
   capture manifest, because `near` sets `identity_ok = false`.
8. **Realized-length audit: IMPLEMENTED 2026-09-28 (screen side only; no hashed file edited).** The capture manifest's
   `realized_length_audit` (each new 112/128 rung's realized MAC share within ±0.10 absolute of the solve's hold-window
   `att_mac_hist_hold`) is now enforced by `prc_screen_r7_20260928.py`:
   - the builder stores each arm's `realized_length_prediction` from its sha-locked s1 solve summary; the driver
     re-derives it at preflight and refuses any mismatch or stand-in (dry-run manifests only);
   - every screened arm's GPU trace passes the standard audits plus the share check BEFORE any candidate NLL is
     released (the evaluator's per-window `NLL=` log lines are masked meanwhile);
   - any screened arm's failure (primary or secondary) stops the cell with `<arm>_screen_audit_failure.json` and no
     decision, as registered ("any failure stops the cell"); the check is descriptive on confirm / full-test traces.
   - Tests: `benchmark/ppl/test_prc_screen_audit_r7_20260928.py` (real round-6 4B t64 / 30B t40 arm-A traces + mutations).
9. **c7 logs.** The c7 calibration logs the capture identity step reproduces exist only on `/scratch` until the capture job
   copies them (sha-checked) into `capture/inputs`.
10. **First-order replay.** The fixed-point replay follows INC's trajectory (validation in §4). The measured 1% screen gate
    stays binding.
11. **Cross-agent edits.** The calib side edited `prc_r7_solve.py` concurrently (last rebuild 15:46). No peer session
    confirmed it had finished. Re-run `test_prc_screen_r7_20260928` if the solver changes before the pin.
12. **Stale adversarial tests.** The reviewer tests in `$SCRATCH/r7/review/test_adv_r7_eval.py` call pre-fix APIs and will
    error. Each defect they demonstrated is covered by a new test.

`$SCRATCH` = `/tmp/claude-114365137/-home-allenjin-Projects-SCMP/aad3505a-096e-4395-aba0-71871b9d6077/scratchpad`.

## 11. Files (all new for round 7)

| kind | files |
|---|---|
| prereg | `benchmark/ppl/prc_r7_prereg_20260928.py`, `kbands/prc_r7_prereg_20260928.json` |
| step 0 | `benchmark/ppl/prc_eval_r7_20260928.py`, `benchmark/ppl/prc_step0_r7_20260928.py`, `benchmark/ppl/test_prc_eval_r7_20260928.py`, `kbands/build_prc_step0_r7_20260928.py`, `kbands/run_prc_step0_r7_20260928.sbatch`, `kbands/prc_step0_r7_20260928.json`, `kbands/prc_windows_r7_20260928.json` |
| capture and solve | `benchmark/ppl/mp_per_row_chunk_calib10_r7.py`, `benchmark/ppl/prc_r7_capture_20260928.py`, `benchmark/ppl/prc_r7_solve.py`, `benchmark/ppl/test_prc_r7_calib_20260928.py`, `kbands/build_prc_r7_capture_20260928.py`, `kbands/run_prc_r7_capture_20260928.sbatch`, `kbands/prc_r7_capture_20260928.json` |
| fixed point and screen | `benchmark/ppl/prc_fixedpoint_r7_20260928.py`, `benchmark/ppl/prc_screen_r7_20260928.py`, `benchmark/ppl/test_prc_screen_r7_20260928.py`, `kbands/build_prc_screen_r7_20260928.py`, `kbands/run_prc_screen_r7_20260928.sbatch` (the manifest `kbands/prc_screen_r7_20260928.json` does not exist yet); stage-2 chain `kbands/run_r7_stage2_cpu_20260928.{sh,py}`; tests `benchmark/ppl/test_prc_screen_audit_r7_20260928.py`, `benchmark/ppl/test_prc_r7_stage2_cpu_20260928.py` (all hashed into the screen manifest when it is built) |
| outputs (future) | `$R7/step0/`, `$R7/<cell>/capture/`, `$R7/<cell>/solve_s0/`, `$R7/fixed_point/`, `$R7/<cell>/solve_fixedpoint/`, `$R7/screen/` |
