# scmp_llm — reproduction guide

> ═══════════════════════════════════════════════════════════════════════════
> ## ⚑ CURRENT STATUS / SESSION HANDOFF — last updated 2026-07-19
>
> ### 2026-07-19 — V17 pivot: measured-damage currency. READ `~/Projects/MP_V17_HANDOFF_20260719.md` FIRST
> Oracle-cell night settled the phase-2 design: σ may not price cross-op/
> cross-length trades (4B floor-down cells lost +6.5%/+8.4% at iso-compute;
> σ-said-cheap), measured-direction compound WON on "saturated" 14B
> (attn→96 + psl 112→95 + floors raised: **9.7378@32.08 vs 9.8366@32.15**),
> rotation (R1/R2-only QuaRot) REFUTED on llama8B (10.5532@32.12 vs
> 10.4879@32.04; autopsy `~/Projects/papers/ROT_FAILURE_ANALYSIS.md`).
> V16.1 finals all in (llama8B t40 8.986@39.91 w/ 4 accepts incl. first
> surrogate compound; 14B 0 accepts at ALL targets — now known to be
> search-blindness, not saturation). Three user-approved tracks were specced
> but their build agents died with a session restart (zero partial work):
> A transplant wave (14B winner → 4B/llama8B/30B), B Stage-2 tonight =
> FALLBACK measured search (user decision: V16.1 + fixed acceptance
> [accept-on-A→B-replication, confirm stop-only, patience 3-4] + new
> `lift_compound` macro family; branches 4B t32/t36, llama8B t40, 14B t32;
> damage-model d_b(L) fit runs as background analysis only),
> C absolute escape gate (first runtime change, sc_common.py). Full specs,
> lessons, gotchas, artifact paths: `~/Projects/MP_V17_HANDOFF_20260719.md`
> + `~/Projects/MP_ORACLE_REPORT_20260718.md`. Paper story (user-approved):
> ONE algorithm, two-stage calibration — σ for dispatch/init (zero evals),
> measured NLL for allocation (small eval budget).
>
> ### 2026-07-17 — V14 forensics verdict + V16 paired-window refinement launched
> **V14 post-mortem (5-agent audit, all conclusions from primary artifacts):**
> the `--fp16-ppl` gate compared 8k VALIDATION-window PPL against a FULL-TEST
> FP16 threshold; validation windows run 21–25% easier (parent val/test ratio
> 0.75–0.79), so the parent was already under threshold on 4B/14B/30B and
> **9/10 completed V14 branches stopped at their first accepted candidate in
> sweep 1 — the threshold phase never ran on any gated branch.** llama8B (parent
> above threshold) is the control: 3 genuine sweeps, 2 accepted moves ≈ −1.6%
> val. The 12.19→13.50 confirm→test gap is ~95% DISTRIBUTION SHIFT (a
> never-selected control shows the same 0.90 ratio), ≤5% selection overfit.
> Eval parity 13.5048 vs 13.7840 vs fp16 10.0445 CONFIRMED identical protocol
> (same compute_ppl/build_model, 298,862 loss tokens); BUT the 4B t32 "win" is
> not iso-compute (realized 31.998 vs V9's 31.31; local slope −0.42 PPL/cycle
> prices the gap at ≈ the whole gain). Measured noise: single-8k-window
> candidate ΔNLL σ≈0.0076 nats ⇒ V14's 0.002 accept gates were ~0.3σ;
> shortlist-by-screen bias +0.0069 nats. Details in the session memory
> (`project_scmp_phase2_v14_gate_artifact`).
>
> **V16 (`benchmark/ppl/mp_v16_refine.py` + `mp_v16_lane.sh`)** replaces the
> V12/V14 protocol: whole-validation-split block partition (32-window rotating
> screen pool ×4 subsets split A/B, 16-window confirm set spread over the split,
> ~69-window holdout), PAIRED per-window ΔNLL vs incumbent everywhere
> (`compute_ppl(..., window_losses=)` opt-in, default bit-identical), pooled-σ
> gates (select on A → replicate on B → ONE confirm eval that GATES but never
> chooses), macro-moves (floor-exchange via propose_floor_exchange,
> ceiling-compress/raise, joint-pair of prev sweep's best value moves), family
> quotas, removal allowed but confirm-gated, per-sweep checkpoints with
> bit-identity resume probe, walltime guard. FP16 stop is two-tier per user
> decision: tier-1 = confirm-set PPL vs 0.90×1.1×fp16-val (measured on the SAME
> windows by `fp16_val_reference.py`, refs in Turbo `fp16_val_refs/`) only
> TRIGGERS tier-2 = a REAL full-test eval; stop only if test ≤ 1.1×fp16-test
> (capped once/target, stop-only, never selects). Budget band is ASYMMETRIC
> [target−0.35, target+1.0] (user: +1 cycle is fine if PPL improves; energy
> table will report realized costs). Tests: 20 v16 unit+mocked-flow tests
> (incl. full main() dry-run, resume, corrupt-checkpoint) + all 29 prior ppl
> tests pass.
>
> **Live V16 wave:** tag `mp_v16_paired_20260717_165536`, 12 cells = {4B,
> llama8B, 14B, 30B} × targets {32,36,40}, one GPU each, 24h, jobs
> `53835227-38` (submitted 16:55 ET; account cap 10 shared — cells start as
> GPUs free). Outputs `$TURBO/mp_v16_refine_mp_v16_paired_20260717_165536/`;
> logs scratch `logs/_mp_v16_refine_mp_v16_paired_20260717_165536/`. Compare
> winners vs V9 full-test at the QUOTED realized costs (V9 4B 13.7840@31.31,
> llama8B 10.4879@32.04, 14B 9.8366@32.15, 30B 10.2951@31.54).
> Confirmed sweep-1/2 accepts (16-window confirm, paired vs incumbent):
> 4B t32 insert-16 −1.78%, t36 insert-28 −1.41%, t40 remove_i3_up −1.84%;
> llama8B t32 none. All accepts are ladder-SHAPE moves; threshold moves 0/165.
>
> ### 2026-07-18 — hybrid-mask dose-response: the biggest single-knob lever
> 4B avg32 full-test PPL vs INT-mask fraction (same V9 pipeline, only
> `--hybrid-int-frac` changed; tag `mp_v9_hybdose_4B_20260718_143110`):
> 10% 13.784@31.31 → 15% 13.584@31.17 (−1.45%) → **20% 13.190@31.52 (−4.31%)**
> — larger than any phase-2 search gain, at ~unchanged SC cycle cost. Caveats:
> more INT7 compute (energy axis prices it; "use less SC" identity risk for the
> thesis) and the calibrator is mask-blind, so this is the lower bound of a
> mask-aware calibration. Full-matrix dose wave launched 18:30 ET, tag
> `mp_v9_hybdose_all_20260718_183035`, jobs `53927482-511`: 30 cells =
> {4B,llama8B,14B,30B} × {avg32,avg36,avg40} × {10,15,20}% (minus existing
> avg32@10% anchors + today's two 4B cells). New driver configs
> `mp_avg36_v9`/`mp_avg40_v9` (same ladder, ratios 0.28125/0.3125); the
> act_global_v9 guard now accepts mp_avg{32,36,40}_v9.
>
> **v16/v16.1 full-test scoreboard so far** (vs V9 4B 13.784@31.31, 8B
> 10.488@32.04, 14B 9.837@32.15, 30B 10.295@31.54): 4B t32 13.589@32.04 (V14's
> 13.505 still best-at-32), t36 12.435@35.98, t40 11.756@40.02; 8B t32 null
> (=V9), t36 9.542@35.92 ([96,50,31,27,24] via v16.1 removal); 14B t32 null
> (=V9, 1.139×fp16); 30B t32 10.196@32.07 (budget-projection only,
> walltime_guard). In flight: 14B t36/t40, 30B t36/t40, 8B t40 (holds the
> first surrogate-proposed accept, surr_i5-4_i6-2 confirm −1.58%).
>
> ### 2026-07-18 — V16.1 (user-approved; no QuaRot) + trace-mining verdicts
> 3-agent mining of the v16 histories (~400 candidates): threshold family
> carries ZERO signal (variance at/below noise floor, both confirms
> sign-flipped — ~30% of search compute wasted); compound moves replicate
> (A→B r +0.87/+0.56/+0.37) while single moves don't; paired noise is 2-4.5×
> smaller than pooled σ_w (74-93% of in-region variance explainable);
> protected 112-cycle pins hold ~3% of MACs but ~10% of the t32 cycle budget
> and were never ablated. Full verdicts in session memory
> (`project_scmp_v17_method_mining`); dense-grid Lloyd-Max ladder design +
> fresh anchor-σ measurement identified as the v17 lever (pass1 per-level
> error vectors in v14/v16 tables are position-copied stale V9 values — DATA
> BUG, do not trust them).
>
> **V16.1 changes in `mp_v16_refine.py` (defaults; queued jobs pick them up
> automatically at start):** (1) threshold phase OFF by default
> (`--threshold-phase` to re-enable); (2) surrogate-guided compound proposals
> — pure-python ridge on the branch's own history.jsonl (era-centered paired
> deltas, ladder-shape features), gated on B-stage transfer r≥0.2 and ≥40
> history rows, nominates ≤5 two-coordinate ladders/sweep (family
> `surrogate`); (3) protected-channel LENGTH move family (`pc_length`,
> default 96:88:80:72:64) — table-driven stoc_len rewrite + budget
> reprojection, per-candidate psl threaded through occupancy/state/summary
> (`best_protected_stoc_len`). Family quotas now
> macro=3,value=3,insert=3,remove=1,surrogate=4,pc_length=2. 56 tests pass
> (incl. surrogate fit/transfer-gate/proposal validity, pc-family, mocked
> end-to-end flow). Running cells (4B×3, 8B t32) finish under v16 semantics
> (in-memory code); pending cells (8B t36/40, 14B×3, 30B×3) run v16.1.
> A v16.1 resume wave over the finished v16 branches (starts from their
> winners via checkpoint/parent chaining) needs explicit approval to launch.
>
> **V15 jobs left running:** 53824168 (14B t32 full-test backfill, done ~17:45
> ET), 53823059 (8B resume + full test, done ~18:00–21:00 ET). 4B makeup done:
> full-test 13.504833@31.998. V14 14B t32.141/36/40 still lack full-test
> backfills; no 30B V15 exists.
>
> **Read this block first.** BIG RESULT 2026-07-05: the MP budget was **ROW-weighted
> (a bug)**. Fixing it to **MAC/FLOP-weighted** (`--budget-weight macs`) makes
> iso-compute MP **beat uniform EVERYWHERE** and makes simple `act_global` ≈
> `measured`. Every lower §*Findings* / root-cause section that ranks `measured ≫
> act_global`, ships row-weighted `act_global` config C, or reports MP *losing* on
> llama8B is a **SUPERSEDED row-weighted artifact** — see "Status of the algorithm"
> below. (The pre-Jul-2 kernel-confound caveat on those tables still also applies.)
>
> ### 2026-07-16 — act_global_v9 implemented + launched (int6/avg32 only)
> V9 is the low-precision-only experiment requested after inspecting the
> `mp_final` traces. `hpca --mp-method act_global_v9` is paired exclusively with
> `--configs mp_avg32_v9`: levels `{96,64,48,32,24,16}`, target avg_sl 32, v3
> MAC pricing + auto metric + residual fill, and the new objective
> `max(0, sigma(level)-sigma(96))^2` (`--objective delta_sigma2`). All allocator
> stages now use the same objective transform (global lambda, argmin, metric
> selection, and residual refinement). This also fixes a discovered confound in
> the old sigma2 experiment: refinement reconstructed its assignment with raw
> sigma after lambda had been solved with sigma2, so prior v3s logs do not cleanly
> evaluate sigma2. V9 protected channels are fixed at **112 cycles** (a multiple
> of 16), with the existing compensated fractions/act_collapse recipe. Legacy
> methods retain their 128-cycle default. Lightweight objective tests, pycompile,
> shell syntax, dry-run wiring, and `git diff --check` pass.
>
> **Live v9 PPL wave:** tag `mp_v9_int6_20260716_001908`, launched 00:19 ET
> outside the sandbox after a real four-model preflight. Jobs: `53670488`
> (`v9_4B_l8_i6`, 4B then llama8B, gl1802), `53670489` (`v9_14B_i6`, gl1803),
> `53670490` (`v9_30B_i6`, gl1803). All three entered RUNNING and their startup
> logs confirm levels `[96,64,48,32,24,16]`, ratio 0.25, `act_global_v9`, and
> protected `pc0.01x112_act_collapse_comp` with MLP overrides
> `down=0.06,up=0.03,gate=0.03`. Results:
> `/nfs/turbo/coe-nbleier/allenjin/hpca/results/results_mp_v9_int6_20260716_001908.tsv`;
> tables: `/nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_mp_v9_int6_20260716_001908/`;
> cell logs: scratch `logs/_hpca_mp_v9_int6_20260716_001908/`.
>
> ### 2026-07-16 — v10 frozen-threshold PPL ladder refinement: completed
> New `benchmark/ppl/mp_ladder_refine.py` is a **second pass**, not a replacement
> allocator. It freezes a completed AdaptiveMPConfig's thresholds, signed
> dispatch metrics, class count/topology, hybrid mask, and protected channels;
> raises the lowest adaptive rung by 4 cycles; then lowers higher adaptive rungs
> one cycle at a time using measured MAC occupancy until the parent deployment
> cost is restored. Each candidate is evaluated on a fixed 32k-token WikiText-2
> **validation** stream in one model load. A round is accepted only if validation
> NLL improves and realized trace cost is within 0.35 cycles of the baseline;
> stop at the first rejection (max four rounds). Candidate tables retain parent
> provenance and the full move/evaluation history. The test split is not used for
> search. Pure tests cover protected/level trace separation, exact budget
> projection, ordering propagation, frozen thresholds, table provenance, and the
> public eval-model entry point. The completed 4B `mp_final` result validates the
> floor-for-ceiling idea. Strict parent-cost branch: levels
> `{128,64,48,32,16}` -> `{55,51,44,32,24}`, 32k validation PPL
> `16.0295 -> 13.0764`, realized cost `30.7260 -> 30.7405`. Nominal-32 branch:
> `{65,63,47,32,24}`, validation PPL `12.6769`, cost `32.0207`. A third move
> toward near-uniform was worse and correctly rejected in each branch. Outputs:
> `mp_ladder_refine_mp_v10_{ppl_refine_20260716_004007,iso_parent_20260716_011224}/4B/`.
> Repro launcher: `benchmark/ppl/mp_ladder_refine_lane.sh`.
>
> ### 2026-07-16 — v11 joint ladder + threshold refinement: completed
> `benchmark/ppl/mp_joint_refine.py` starts directly from the completed
> `mp_final` avg32 wrapper; it does **not** depend on v9/v10. It alternates
> MAC-budget-projected integer ladder moves with end-to-end validation-NLL
> threshold candidates. Runtime profiling records GPU-resident histograms of
> normalized dispatch metrics per operator/layer bucket, allowing threshold
> changes to be expressed as small profiled-MAC-mass moves. Tested candidates:
> promote MLP floor, promote attention-linear floor, demote qk/av top, and a
> combined attention-to-MLP transfer. Signed dispatch metrics, protected channel
> indices/length, class topology, and hybrid INT mask stay frozen. `sc_prec=8`
> and maximum stream length **128** are enforced; a briefly submitted max-256
> experiment was cancelled at user request and all cap-extension code removed.
> 19 objective/v10/v11 unit tests, pycompile, shell syntax, and diff checks pass.
>
> Search protocol: fixed 8,188-token WikiText-2 validation prefix, followed by
> 32,752-token **validation confirmation** for each selected branch. These are
> not test-split numbers; run one frozen test evaluation before using them as
> final paper PPL. All 20 branches completed and wrote wrappers. Confirmed
> validation PPL / realized cost / levels for targets `{32,36,40,48}`:
>
> ```text
> model     32                         36                         40                         48
> 4B        12.9006/31.9759/57-56-42-32-25  11.8839/35.9895/89-69-47-33-26  11.0825/39.9835/89-66-47-33-31  10.4534/48.0139/123-63-48-39-37
> llama8B   10.1432/31.9047/86-63-48-30-20   9.0980/35.9464/88-61-47-34-25    8.6171/39.8899/128-64-48-34-27   8.0751/48.0291/128-64-47-39-38
> 14B        9.4217/31.9407/82-63-48-31-20   9.0353/36.0268/85-63-48-32-26    8.8465/40.0205/87-61-48-33-32    8.7574/47.9928/128-65-48-41-38
> 30B        9.7789/32.0158/49-48-47-31-25   8.9264/36.0279/82-64-48-30-26    8.4672/40.0703/126-65-51-31-27   8.1405/48.0546/128-68-48-40-37
> ```
>
> Threshold refinement was the selected 8k winner in 13/16 budget branches;
> the exact selected move varies by model/budget, so do not hard-code “MLP
> promotion always wins.” It is **not yet proven to generalize**: on 4B the
> 32k-confirmed frozen-threshold v10 remains better than v11 both at strict iso
> cost (`13.0764 < 13.2174`) and nominal 32 (`12.6769 < 12.9006`). Use 32k or
> multiple validation prefixes for threshold selection before adopting v11 over
> v10. The monotonic 32k curves strongly support increasing budget, while 14B
> begins saturating around 40-48. Strict realized-parent-cost
> confirmations also completed: 4B `13.2174@30.7378`, llama8B
> `10.0254@32.3286`, 14B `9.5209@32.3316`, 30B `9.7444@31.5189`.
>
> Outputs: budget sweep
> `/nfs/turbo/coe-nbleier/allenjin/hpca/mp_joint_refine_mp_v11_budget_mpfinal_20260716_025421/`;
> strict parent-cost ablations
> `/nfs/turbo/coe-nbleier/allenjin/hpca/mp_joint_refine_mp_v11_joint_mpfinal_20260716_025421/`.
> Completed jobs: strict parent `53675955-58`; `{32,36,40,48}` sweep
> `53676395-98`. Launcher: `benchmark/ppl/mp_joint_refine_lane.sh`.

### 2026-07-16 — V12 generalized precision-topology pass launched
V12 (`benchmark/ppl/mp_systematic_refine.py`) is a second pass directly over
the completed `mp_final` avg32 wrappers.  The ladder cardinality is searchable:
the pass can move any rung, insert a new level at every internal gap or either
endpoint (including retaining a lower top and adding 112/128), remove a rung,
and perturb every threshold boundary.  Insertions split the profiled MAC mass
of the affected class; all candidates are reprojected to the requested
MAC-weighted budget.  Protected-channel lengths are not newly introduced as
adaptive rungs, pass-1 counts/fractions are preserved under diagnostic names,
and guard plus confirmation costs are checked against budget tolerance.

The initial four-job wave (`53718948-51`) was stopped at the user's request
after being capped at two sweeps.  It was relaunched outside the sandbox at
12:38 ET with tag `mp_v12_topology_mpfinal_conv_20260716_123829`:
`53719771` (4B), `53719772` (llama8B), `53719773` (14B), and `53719774`
(30B), all RUNNING on gl1802/gl1805.  This wave searches targets
`parent:32:36:40:48`, allows up to 8 sweeps, and stops after 2 consecutive
rejected sweeps; it tests two candidate insertion values per gap.  Outputs:
`/nfs/turbo/coe-nbleier/allenjin/hpca/mp_systematic_refine_mp_v12_topology_mpfinal_conv_20260716_123829/`;
logs: `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_mp_systematic_refine_mp_v12_topology_mpfinal_conv_20260716_123829/`.
Scheduler commands were intentionally run outside the Codex sandbox per the
execution-environment decision above.

### 2026-07-16 — 4B V13 focused follow-up
The first V12 4B branch selected a short-window-favorable four-level removal.
To test the proposed correction without interrupting the four-model V12 wave,
the launcher now supports insertion-only topology (`--no-allow-removal`),
family-specific guard shortlists, and a nonzero minimum mean-NLL improvement.
Job `53767030` (tag `mp_v13_4b_insertonly_mpfinal_182418`) is RUNNING on gl1802
with targets `parent:32`, 8k screen and guard windows, up to 8 sweeps,
two-rejection patience, three guard candidates per family, and no rung removal.
Its outputs are under
`/nfs/turbo/coe-nbleier/allenjin/hpca/mp_systematic_refine_mp_v13_4b_insertonly_mpfinal_182418/`.

### 2026-07-16 — V9 + phase-2 cardinality branches and FP16 gate
The next phase-2 wave starts from the stronger `mp_v9_int6_20260716_001908`
parent rather than `mp_final`: V9's six-level ladder and protected 112-cycle
channels remain intact, while phase 2 re-optimizes end-to-end PPL.  The lane now
supports an explicit `--fixed-class-count` filter.  The fixed-six branch disables
insertions/removals, so it cannot erase the V9 topology; the insertion-only branch
allows up to eight adaptive levels but never removes a V9 rung.  These are
separate, auditable cardinality branches rather than one greedy topology path.

The search also accepts `--fp16-ppl` and stops a target early when an accepted
candidate is below `1.1 * PPL_FP16` on both screen and guard windows.  The mandatory
32k validation confirmation is still run and recorded separately; because the
gate uses full-test FP16 references against validation windows, it is an early
stopping heuristic, not a substitute for the final frozen test evaluation.
The V9 branch uses tag `mp_v14_v9_insertonly_20260716` under
`/nfs/turbo/coe-nbleier/allenjin/hpca/`; the four fixed-six jobs were cancelled
before useful evaluation at the user's request. Slurm commands were issued
outside the Codex sandbox per the execution-environment requirement above.
Existing V12/V13 jobs were left running.

### 2026-07-17 — automatic full-test reporting and makeup wave
`benchmark/ppl/mp_systematic_refine.py` now evaluates each selected frozen
branch on the complete WikiText-2 `test` split after validation confirmation
(`--test-tokens 0`, about 298k tokens).  It records `final_test_ppl`, realized
test cost, token count, and `final_test_trace.json`; when resuming an older
summary it backfills the final test without repeating the search.  The test
split is never used for candidate selection.

The focused V15 jobs are `53823059` (8B resume from the accepted partial
eight-level candidate, two fixed-topology sweeps), `53823060` (4B makeup
final-test evaluation from the selected V14 target-32 wrapper), and `53824168`
(14B equivalent, relaunched after the old 12-hour job `53823061` was
cancelled).  Outputs are under
`/nfs/turbo/coe-nbleier/allenjin/hpca/mp_v15_v9_*_20260717*/`.

**Scheduler policy:** systematic-refinement and full-test jobs use a 24-hour
wall-clock limit (`--time=24:00:00`); do not use the old 12-hour limit.
>
> The active 30B MP-RULER jobs can be cancelled and resumed: `call_api.py` reads
> existing prediction JSONL indices, appends only missing samples, and the lane
> also skips completed summary cells. At inspection, avg64f/avg48f/avg32 had
> 15/14/14 of 50 samples saved. Cancelling loses at most the in-flight sample and
> model-load time; relaunch with the same tag/budget. Do not delete prediction
> directories. With explicit user approval, jobs `53553136` (avg64f), `53553137`
> (avg48f), and `53553138` (avg32) were cancelled to launch v9. Resume each with
> its original `SubmitLine`/budget and tag `mp_ruler_ctx4096_20260714_021537`.
>
> **Execution-environment requirement (user decision, 2026-07-16):** Slurm
> scheduler operations must be run **outside the Codex sandbox**. This includes
> scheduler inspection (`squeue`, `scontrol`, `sacct`) as well as mutations
> (`scancel`, `sbatch`, `srun`). Sandboxed scheduler calls can hang or fail to
> reach Slurm services and must not be treated as authoritative. Request/use
> elevated unsandboxed execution for these commands, while still obtaining the
> user's explicit approval before cancelling jobs or submitting a new wave.
>
> ### 2026-07-06 — PIPELINE v2: SQ-calibration fix (+3 more calibrator fixes)
> **Model scope is now EXACTLY 4B / llama8B / 14B / 30B** (user decision; 1.7B+32B
> dropped — do not spend cells on them). Four calibrator fixes landed:
> 1. **`--calib-smoothquant` was itself broken** (silent no-op unless a separate
>    `USE_SMOOTHQUANT`/`SMOOTHQUANT_SCALES` env pair was ALSO set): every fw_* table
>    measured σ on UNSMOOTHED activations while eval deploys SQ α=0.5. Now
>    self-sufficient (resolves the same `ACT_SCALES_DIR` file + `SQ_ALPHA` as eval,
>    fails loudly on 0 matches) and **default-ON in overnight_mp.sh (`CALIB_SQ=1`)**.
>    First smoke (4B int7 act_global_fw_sq, truncated 20k eval): PPL 11.06 vs 12.77
>    unsmoothed — allocation shifts attention→~125 cyc, linears→~62; ρ under the
>    deployed geometry drops for linears (down 0.375→−0.05, gate 0.34→−0.05) ⇒ amax
>    metric even weaker than thought (dispatch-metric work = top lever).
> 2. **"measured OVERSPENDS" RETRACTED — summarizer artifact.** fw_summary's old
>    flop_avg_sl came from the never-exercised `operator_defaults` fallback (mean-W_g
>    mispricing), not the deployed buckets. Deployed allocations: act_global/measured/
>    fisher ALL exactly 64.00/48.00; eval drift ≤0.75%. **measured's llama8B edge
>    (8.31 vs 8.42 int7; 9.80 vs 9.97 len96) is genuine iso-FLOP** — loss-aware W_g
>    helps llama8B; act_global still ties/wins 4B. The REAL bug was **measured_curve
>    UNDERSPENDING 2.7–7%** (feasible-side λ + coarse discrete cost) — fixed by the
>    MAC-priced residual greedy fill (auto-enabled for the curve method; payload
>    `refine_mode=curve_residual_fill`). measured_curve rows quarantined from fw_*.
> 3. Calibrator now exports **`expected_flop_avg_stoc_len`** (the true iso-compute
>    check; smoke: 63.9999) and **ρ is computed BEFORE the curve override** (old
>    measured_curve ρ values were tie-breaking noise — ignore them).
> 4. Summaries: `benchmark/ppl/fw_summarize.py` (buckets-sourced flop_avg, uniform
>    twins, x_fp16) — fw_summary.tsv regenerated (12 iso-FLOP rows).
> **Results namespaces:** `hpca_results/llm/mp/fw_*` = 2026-07-05 unsmoothed-calib run
> (PPLs valid, superseded by...) → `sq_results.tsv`/`sq_manifest.tsv` + Turbo
> `mp_calib_sq/` = the v2 (SQ-calibrated) sweep, in flight on 3 GPUs (gl1802 smoke +
> gl1804 4B + gl1807 llama8B via nbleier0; owned 2-GPU backfill queued behind the
> lab 10-GPU cap). Row-weighted-era files moved to `mp/_superseded_rowweighted/`;
> under-budget measured_curve rows in `mp/_quarantine_underbudget/`.
> Driver hardening: cross-node cell locks (`$TABLES/.lock_<tag>`), `MODELS=` filter,
> `SMOKE_METHODS=`, sweep sbatch `benchmark/ppl/sbatch_mp_sq.sbatch` (gated on
> `~/sq_smoke_passed`).
>
> ### 2026-07-09 — MP calibration cache quarantined; current best direction
> Old MP calibration data is suspected bad enough to be unsafe for automatic
> reuse. The reusable Turbo cache dirs were moved out of the script-default paths:
>
> ```text
> /nfs/turbo/coe-nbleier/allenjin/hpca/_quarantine_bad_mp_calib_20260709_203642/
> ```
>
> Contents:
> `mp_calib_fw`, `mp_calib_overnight`, `mp_calib_protected`,
> `mp_calib_protected_gn_20260707_121507`, `mp_calib_sq`. Future MP launches
> should use fresh `TABLES`, `RESULTS`, and `MANIFEST` paths so no stale wrapper
> or skip row is reused.
>
> Current best algorithmic direction is **hybrid INT + protected-channel
> `act_global` MP**, not `measured_curve` as a default. Evidence: protected
> channel helped the aggressive low-average MP point at roughly the same
> compensated FLOP budget (4B 14.8210→13.6191, llama8B 9.3799→8.8165, 14B
> 10.0332→9.5976). `measured_curve` has mixed results and prior under-budget /
> artifact caveats; do not call it SOTA without a fresh fixed-budget run.
>
> Use the first-class `hpca` MP config names:
> `mp_avg96_burst128`, `mp_avg64_burst128`, `mp_avg48_burst128`. The average
> number is the target MAC-weighted SC stream length after the hybrid INT mask
> is applied; `burst128` means the MP levels still include 128 and protected
> channels are pinned to 128 by default. Current launch shape:
>
> ```bash
> bash hpca --models 4B --configs mp_avg96_burst128 --metrics ppl \
>   --sc-backend hybrid --hybrid-int-frac 0.10 \
>   --hybrid-sensitivity-dir /nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921 \
>   --hybrid-sensitivity-config sc_int8 --tag <tag>
> ```
>
> Correct MP+hybrid calibration order:
> 1. Apply the per-operator-per-layer hybrid mask first (top 10% sensitive
>    entries go INT).
> 2. Calibrate MP only on the remaining SC graph.
> 3. Use `act_global` with MAC/FLOP-weighted budget (`--budget-weight macs`) and
>    protected channels at 128 cycles with compensation
>    (`--protect-channel-frac ... --protect-channel-stoc-len 128
>    --protect-compensate-budget`).
> 4. SC budget is the budget for the remaining SC computation only; INT layers
>    are outside that SC budget.
>
> ### 2026-07-10 — act_global_v2 launched (new MP algorithm, full 4-model sweep)
> New method `act_global_v2` = act_global + two flag-gated allocator changes
> (defaults off ⇒ byte-identical act_global):
> 1. **`--argmin-pricing macs`** — KKT-consistent per-row cycle price. The old
>    argmin priced cycles by `rep_g = mac_per_row × true/stored`; the subsample
>    ratio (attention ~128× vs dense linears ~4×, experts 1×) cancels between a
>    stored row's benefit and cost in the true KKT solution, so pricing by
>    `mac_per_row` ONLY fixes a mispricing that affects every model (dense: 4B
>    avg48 gave v_proj@115cyc σ0.13 while down_proj@40 σ0.27; MoE: floored
>    qk/av below uniform → the 30B MP<uniform catastrophe). Budget stays
>    R_g-weighted (iso-compute unchanged; `expected_flop_avg_stoc_len` still
>    the check). Guards: requires budget-weight macs + global scope, no refine.
> 2. **`--metric-select auto`** — per-operator SIGNED dispatch-metric selection.
>    Calibration captures candidates (amax / l2 / crest=amax÷l2) per row, picks
>    per op the candidate with best |Spearman ρ| vs true σ-benefit (negative ρ
>    ⇒ deploy inverted; margin 0.05 vs amax; partial-capture ⇒ keep amax),
>    builds thresholds on the winner, exports `dispatch_metrics` in the table;
>    runtime (`_mp_dispatch_metric` in sc_common.py + AdaptiveMPConfig) computes
>    the chosen metric — all O(D), runtime-free. Old tables ⇒ amax (back-compat).
>    Motivation: ρ(amax) ≈ 0/negative on down/o/up_proj on ALL models.
>    Unit-tested: formula/ranking parity calib↔runtime, sign parity
>    (−metric ≡ 1−m), JSON round-trip, selection end-to-end (5/5 pass).
> **Launched 2026-07-10 ~20:14 ET** (queued behind RULER jobs, 10-GPU cap full):
> smoke job 53296574 (4B avg64, 65k-token gate: flop-avg 64±2 + PPL<14 + logs
> per-op allocation vs v1 refs qk111/av111/v128/down63) → touches
> `~/mp_v2_smoke_passed`; lanes 53296575-78 (4B/llama8B/14B/30B ×
> {avg96,avg64,avg48}, full PPL, `--dependency=afterok` + marker check,
> kill-on-invalid-dep). Tags: smoke `mp_v2_smoke_20260710_201348` (non-citable),
> full `mp_v2_hyb_pc_20260710_201348`. Scripts `benchmark/ppl/mp_v2_{smoke,lane}.sh`;
> results `/nfs/turbo/coe-nbleier/allenjin/hpca/results/results_mp_v2_hyb_pc_20260710_201348.tsv`;
> tables `mp_calib_mp_v2_hyb_pc_20260710_201348` (fresh dirs per quarantine rule).
> **Baseline to beat (v1, same hybrid+pc pipeline, `results_mp_hyb_pc_20260709_205822.tsv`):**
> 4B 10.34/10.83/11.62, llama8B 7.71/8.18/8.89, 14B 8.70/9.00/9.28,
> 30B 8.09/10.20/12.23 (avg96/avg64/avg48). 30B is the expected big win
> (allocation inversion fix); dense direction supported by the SQ-calib smoke
> (attention-heavy shift 12.77→11.06) but magnitude genuinely open.
>
> ### 2026-07-07 — Protected-channel GN selector launched
> First protected-channel full run improved aggressive avg96/len96 but not enough
> for the paper target: 4B `burst_act_global` 14.8210 →
> `pc1_act_weight_comp` 13.6191; llama8B 9.3799 → 8.8165; 14B 10.0332 → 9.5976.
> New algorithm implemented: `--protect-channel-metric act_grad_weight`, an
> offline diagonal Gauss-Newton selector for protected input channels:
> `score[j] = E[x_j^2] * Σ_o W[o,j]^2 * E[(dL/dy_o)^2]`. It runs one FP
> backward prepass per calibration window, then exports only channel indices;
> eval/runtime overhead is identical to the existing split-channel PC path.
> New launcher methods in `benchmark/ppl/protected_channel_mp.sh`:
> `pc1_act_grad_weight_comp` and `pc2_act_grad_weight_comp` (1%/2% protected at
> 128 cycles, strict compensated FLOP budget). GPU smoke passed on 4B:
> method `act_global_fw_sq_pc0.01x128_act_grad_weight_comp`,
> expected_flop_avg_stoc_len=48.1107, 252 protected modules.
> Full-protocol 8h run launched 2026-07-07 12:15 ET:
> results `hpca_results/llm/mp/protected_gn_full_20260707_121507_results.tsv`,
> manifest `protected_gn_full_20260707_121507_manifest.tsv`, tables
> `/nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_protected_gn_20260707_121507`,
> logs
> `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_mp_protected_gn_full_20260707_121507/`.
> Active lanes at launch: 4B on gl1807, llama8B on gl1808, 30B on existing
> gl1802 allocation step `52953626.6`; 14B queued as `mp_gn_14B` behind
> `AssocGrpMemLimit` and should start when memory budget frees.
> Post-launch corrections: standalone `srun` jobs created idle allocations, so
> 4B/llama8B/14B were relaunched via `sbatch` using
> `hpca_results/llm/mp/run_gn_protected_lane_20260707_121507.sh`. 14B must use
> `Qwen/Qwen3-14B` (the `Qwen/Qwen3-14B-Instruct-2507` id is invalid). 30B OOMed
> during the full-ctx GN backward prepass, so `--protect-channel-stat-ctx` /
> `--protect-channel-stat-sequences` were added; 30B was relaunched on gl1802
> step `52953626.7` with stat windows `4 × 512`, while main threshold calibration
> and PPL eval remain `CTX=2048`, `PPL_MAX_TOKENS=0`.
> 2026-07-08 correction: the missing llama8B GN row is a launcher bug only. The
> run used `meta-llama/Llama-3.1-8B`, but the canonical HPCA model and available
> SmoothQuant cache are `meta-llama/Llama-3.1-8B-Instruct`
> (`act_scales_meta-llama_Llama-3.1-8B-Instruct.pt`). Patch/rerun llama8B with
> `-Instruct` before judging whether GN helps or hurts that model.
>
> ### What we're doing (the thesis)
> Argue that **stochastic computing (SC) enables finer-grained mixed precision
> than fixed-point, and that finer MP beats uniform precision at the same
> compute budget.** SC precision = `stoc_len` (stream cycle count), a *per-call
> runtime knob* — any integer cycle count works (incl. non-pow2 like 96/48) via
> early termination of one `sc_prec=8` Sobol stream, no datapath change. Compute
> budget = average `stoc_len` (cycles); iso-budget = MP@avg_sl N vs uniform@N.
> Target venue: HPCA. The claim has **two halves** — track both:
>   1. **"MP beats uniform @ iso budget"** — partially tested (see gaps).
>   2. **"SC is finer-grained than fixed-point"** — **NO experiment yet.**
>
> ### Units convention — DISCUSS BUDGETS IN NOMINAL (before-halving) CYCLES
> `halve_bipolar_stoc_len=1` runs bipolar streams at HALF the nominal length
> (uSystolic/HUB sign-magnitude trick, no accuracy loss), so every budget has two
> numbers that differ by 2×: **nominal** (before halving = the config name = the
> 2^prec "int-N" stream / the `avgN` name) and **halved** (after halving = the
> actual kernel cycle count). **Convention: in conversation, notes, plots, and the
> paper we quote the NOMINAL number** (it matches the config names + the E(stoc_len)
> energy/compute-budget story). CODE ARTIFACTS STAY HALVED — `--mp_levels`,
> `--budget_ref_stoc_len`, `stoc_len` in JSON tables, `trace.py`, and the
> `cycles`/`realized_avg_sl` columns in `hpca_results/*.csv` are all halved.
> Convert with `nominal = 2 × halved`.
>
> | config | nominal (TALK IN THIS) | halved (code/CSV/JSON) |
> |---|---|---|
> | int8   | 256 | 128 |
> | avg192 | 192 | 96  |
> | int7   | 128 | 64  |
> | avg96  | 96  | 48  |
> | int6   | 64  | 32  |
>
> So "run MP at 128 and 96" (nominal) = the int7 and avg96 budgets = halved 64/48.
> The `hpca_results/mp/` cells (halved `budget`/`realized_avg_sl` ≈ 96) are the
> **nominal-192 (avg192)** budget — the mildest MP point; the untested aggressive
> budgets are **nominal 128 and 96**.
>
> ### Status of the algorithm — iso-FLOP (MAC-weighted budget) is the fix
> Per-row stream-length MP. Offline calibration (FP teacher over a few wikitext2
> windows) measures each row's activation metric `|x|.amax(-1)` + per-level SC
> reconstruction error σ; a cross-layer Lagrangian (one shared λ over 9-op × 4-layer
> groups) picks each row's level; counts → metric thresholds → JSON table → runtime
> per-row dispatch (`model/sc_common.py`).
>
> **ROOT-CAUSE FIX (2026-07-05): the budget must be FLOP-weighted, not row-weighted.**
> `_global_lambda` priced cost as `Σ R_g·L_g` (rows). But attention (av+qk) is ~90%
> of ROWS yet only ~7-14% of FLOPs (MACs/row varies ~100-290× av-vs-linear). The
> row-budget therefore (a) made "iso-budget" NOT iso-compute (act_global@row-64 =
> FLOP-118 ≈ 1.85× uniform's compute) and (b) STARVED qk despite qk having the
> HIGHEST σ (0.58 vs linears 0.08-0.23) — its 90%-of-rows made it "expensive".
> FIX: `--budget-weight macs --mac-weights-trace <sc_int7 trace>` prices cost by
> MACs (per-op MACs/row read from a trace) → iso-budget = iso-compute (linears
> expensive, attention cheap). Table `method` gets a `_fw` tag. Row-avg and FLOP-avg
> now swap roles: a FLOP-64 allocation reads row-avg ≈ 116 in `realized_avg_sl`.
>
> **iso-FLOP results** (`hpca_results/llm/mp/fw_summary.tsv`; PPL vs uniform at
> matched FLOP-avg 64=int7 / 48=len96; fp16 4B=10.04, llama8B=7.21):
>
> | model | budget | act_global | measured | measured_curve | uniform | best vs uni |
> |---|---|---|---|---|---|---|
> | 4B      | int7  | 12.77 | 12.82  | 12.63 | 16.94 | −25% |
> | 4B      | len96 | **20.14** | 21.60 | 20.64 | 33.13 | −39% |
> | llama8B | int7  | 8.42  | 8.31\* | 8.51  | 9.24  | −10% |
> | llama8B | len96 | 9.97  | 9.80\* | 10.31 | 11.64 | −16% |
>
> **Conclusions:** (1) iso-compute MP beats uniform EVERYWHERE (−8 to −39%), incl.
> "robust" llama8B that *lost* under row-weighting → the model-dependent-loss story
> was a budget artifact. (2) Simple `act_global` (recon-σ) ≈ `measured` /
> `measured_curve` at true iso-FLOP → the expensive ΔLoss probing was only
> compensating for the budget bug on 4B; on llama8B `measured` keeps a real
> −1.3/−1.7% edge (see the 2026-07-06 block). (3) ~~measured OVERSPENDS~~
> **RETRACTED 2026-07-06** — the 74/54 numbers were a summarizer artifact
> (operator_defaults, not the deployed buckets); all three methods are exactly
> on budget. Every method improved 15-30% from the row→MAC budget fix.
> Harness: `benchmark/ppl/overnight_mp.{sh,sbatch}`; index `fw_manifest.tsv`
> (grep → wrapper for reuse, `calib_command` for repro). `measured_curve` = allocate
> by a probed per-level ΔLoss CURVE (not W_g×σ); competitive but not needed.
>
> ### ONE PPL protocol (arch_impl `ppl.py` REMOVED 2026-07-04)
> Everything — INT baselines, SC uniform, AND per-row MP — now runs through the
> **HPCA protocol**: `benchmark/quant/eval_quant.py` (full wikitext-2 ~298k tok,
> ctx 2048, SmoothQuant α=0.5), driven by `./hpca`. fp16 14B = 8.64.
> For paper/citable PPL rows, keep `PPL_MAX_TOKENS=0` (or unset). Any
> `PPL_MAX_TOKENS>0` run is a smoke/screening run only and must be labeled
> non-citable; do not compare truncated rows against INT/uniform CSVs.
> **MP in the HPCA protocol:** set `QUANT_CONFIG=mp` + `MP_CONFIG_JSON=<wrapper.json>`
> (a calibrate_mp_thresholds.py table) — eval_quant loads it into `cfg.sc_mp_config`
> and dispatches per-row MP through the SAME path as the uniform `sc_*` cells, so
> MP is finally apples-to-apples with uniform + INT. The old `benchmark/ppl/ppl.py`
> (65k tok, ctx 1024) + `tests/run_mp_sweep.sh` + `summarize_mp_results.py` are
> DELETED — they were a second incompatible PPL scale that only polluted
> comparisons. **All pre-2026-07-04 arch_impl PPL numbers below (Findings tables,
> the 15.49/17.10 4B int7 figures, etc.) are in the removed protocol — do NOT cite
> them; re-run in the HPCA protocol.** Calibration still uses
> `calibrate_mp_thresholds.py` (now `--ctx_len 2048` to match deploy).
>
> ### Experiments so far
> **CURRENT (2026-07-05, iso-FLOP / MAC-weighted, HPCA protocol) — the citable MP
> result:** `hpca_results/llm/mp/fw_summary.tsv` — 4B + llama8B × {int7,len96} ×
> {act_global, measured, measured_curve, fisher}, ALL beating uniform (table above).
> 14B/1.7B breadth was in flight when this session ended (partial in `fw_results.tsv`).
>
> **SUPERSEDED (row-weighted budget — do NOT cite):** every `_mp_overnight_*`
> arch_impl run — the 4B int7 act_global 15.49 / measured 17.96 numbers, the full
> Jun 2–4 matrix, the §Findings tables below — used the row-weighted budget bug.
> Their RANKINGS (measured ≫ act_global; MP loses on llama8B) are ARTIFACTS; the
> iso-FLOP result inverts them (act_global ≈ measured; MP wins everywhere).
>
> **HPCA baselines (Turbo `/nfs/turbo/coe-nbleier/allenjin/hpca/results/`):**
> - INT: `results_bitmod_protocol.tsv` (Jul 3 18:12, DONE) — fp16→W4A4, all 4
>   models. asymm beats symm below W6A6.
> - SC uniform: `results_sc_uniform.tsv` (tag sc_uniform, launched Jul 3 ~19:00)
>   — **BROKEN/in-flight.** Only completed cell 1.7B sc_int8 = **PPL 61,116**
>   (should be ~1.0–1.1× fp16). Harness bug — likely uniform-under-halve path.
>
> ### What's citable
> Nothing before **2026-07-02**. Pre-06-03 = SC_SCRAMBLE_MASKS=256; a kernel edit
> landed ~Jul 2 23:52 (`arch_impl/logs/_scramble_ref_preedit.pt`) between the
> `_mp_overnight_m1fix_spot` (superseded, worse) and `_mp_overnight_m1fix_global`
> (clean) runs. All clean cells are single-run (no seed variance) at realized
> avg_sl 62–66 vs target 64.
>
> ### HPCA plan
> - **Phase 1 — INT quant baselines** (`./hpca` default configs): DONE, canonical
>   in `results_bitmod_protocol.tsv`. fp16 + {W8..W4}×{symm,asymm}, SmoothQuant
>   α=0.5 + RTN fake-quant, full protocol. (LongBench/RULER **silently never
>   record** — known bug; SC cells skip them by design.)
> - **Phase 2 — SC uniform baselines** (`./hpca --configs sc_int8,sc_avg192,
>   sc_int7,sc_avg96,sc_int6 --metrics ppl`, tag sc_uniform): 20 cells =
>   {1.7B,14B,30B,llama8B} × {128,96,64,48,32 cycles}. IN FLIGHT + first cell
>   broken. This supplies uniform PPL-vs-cycles curves + energy/latency traces.
>
> ### Open issues / immediate next actions (priority order)
> 1. **Diagnose the sc_uniform harness bug** (1.7B sc_int8 = 61,116). ~~Suspect the
>    uniform-under-halve path~~ **RULED OUT (session-2, 20:30 ET, diag on gl1808 +
>    trace forensics):** the trace shows all 252 groups at (stoc_len=128,
>    rng_levels=128), rows conserve exactly, linears smoothed=true. Bisection
>    (`~/hpca_diag_sc.py` on 1.7B, log `~/hpca_diag_sc_gl1808.log`): fp16 20.5,
>    LINEAR-only SC 21.3 (fine!), **ATTN-only SC 17,125 (the bug)**; plain(None)
>    path == explicit{} path bit-identical (23175.380 both) so the harness wiring
>    is NOT the cause; noSQ still broken. **The bug is SC per_head attention
>    itself.** RESOLVED (session-2, ~21:00 ET, `~/hpca_confirm_perrow.py`,
>    `~/hpca_localize.py`, 6-window ctx-1024 slices): the broken uniform cells
>    routed attention through the **per_head** kernel, while the shipped MP
>    runs use **per_row** (`_sc_attention_matmul_ab_t` hardcodes per_row in the
>    mp branch). **FIX APPLIED:** attention is now per_row-only in `model/
>    sc_common.py` (`_sc_attention_matmul_ab_t` uniform branch + `_sc_attn_ab_t_at_
>    stoc_len`; default `sc_granularity=per_row`; per_head removed). per_head was
>    size-dependent: 14B 1.39×, 4B 2.6×, **1.7B 830×** (17125). per_row fixes the
>    catastrophe: 4B 1.13× (14.77), 14B ~lossless. **Masks 256 REFUTED** (user
>    asked; tested 1.7B+4B): no help — 1.7B qk-only 64.5@m64 vs 71.0@m256, 4B qk
>    ~14 at both. **Residual is 1.7B-specific Q·Kᵀ in EARLY layers** (per_row
>    1.7B: qk-only 64.5 ≈ full 66.5, av-only 20.9 ≈ fp16 20.5, layer0-alone 36.7;
>    4B qk-only 13.9 ≈ fp16 13.1). qk already at 128 cyc = SC max, so SC cannot
>    resolve 1.7B early-layer attention scores — an intrinsic SC limit on that
>    model, NOT masks/scramble/per_head. OPEN: why 1.7B early-qk (Q/K outliers?);
>    options = drop 1.7B, or keep early-layer qk fp16 for a usable 1.7B SC number.
>    Broken sc_uniform streams killed; sc_uniform tsv rows 1.7B sc_int8..avg96 are
>    garbage — delete + RE-RUN with the per_row fix before reuse. (Discriminator
>    `~/hpca_diag_sc2.py` tests masks/owen — do NOT run: user set masks=64 fixed,
>    no other scrambling.)
> 2. **The iso-budget comparator does not exist and sc_uniform can't supply it:**
>    clean MP cells are **4B/32B on arch_impl protocol**; sc_uniform is
>    **1.7B/14B/30B/llama8B on HPCA protocol** — zero model overlap, incompatible
>    protocols. Must run **uniform SC {128,96,64,48} on 4B+32B in the arch_impl
>    protocol** (explicit-stoc_len path), pinned single GPU, to get "MP@64 vs
>    uniform@64."
> 3. **The "finer than fixed-point" half needs an experiment:** flagship int7
>    levels [128,64,32] are all pow2 → don't even exercise fine granularity. Add
>    a pow2-restricted vs fine-level MP ablation at matched avg_sl, and ideally a
>    fixed-point-MP baseline + an E(stoc_len) energy table (GPU wall-clock does
>    NOT track stoc_len, ~4% over 256→16 — cost story rests on the cycle/trace
>    model, `scmp_kernels/trace.py`).
> 4. **Extend clean config-C MP** to ≥3 models × ≥2 budgets (currently 2×1) and
>    **replicate** act_global-C 2–3× (seed variance unknown).
> 5. ~~Calibration runs WITHOUT SmoothQuant while eval applies α=0.5~~ **FIXED
>    2026-07-06** (`--calib-smoothquant` made self-sufficient + default-on; see the
>    PIPELINE v2 block). Legacy unsmoothed tables = Turbo `mp_calib_fw/`.
>
> ### Live runs (as of 2026-07-03 ~19:15 ET — verify before trusting)
> - gl1802 (job 52830052, 9d left): stream A = 14B sc_uniform — SAFE.
> - gl1808 (job 52794122, **~3h35m left, dies ~22:51 ET**): stream B = 1.7B/
>   llama8B/30B sc_uniform (15 cells) — **will be walltime-killed mid-run.**
> - Monitor: `column -t -s$'\t' /nfs/turbo/coe-nbleier/allenjin/hpca/results/results_sc_uniform.tsv`
> - done markers `~/hpca_sc_{a,b}.done` (touch is UNCONDITIONAL — grep log for
>   `!!! FAILED`); logs `~/hpca_sc_{a,b}.log`.
>
> ### Key parameters (canonical operating point)
> `sc_prec=8`, `halve_bipolar_stoc_len=1` (cap = 128; a level value **is** the
> halved cycle count), `SC_OWEN_MODE=bitrev`, `SC_SCRAMBLE_MASKS=64`,
> SmoothQuant α=0.5, budget_ratio=0.5, 36 groups (9 ops × 4 layer buckets, 1
> timestep bucket). MP levels: int8=[128] (uniform ceiling), len192=[128,96,64]
> (avg 91), int7=[128,64,32] (avg 64), len96=[64,48,32] (avg 48). Generation
> cliff ≈ stoc_len 48 (level 32 is sub-cliff). Env: conda `annstention`,
> `HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache`.
>
> ### Commands (ONE protocol — HPCA/eval_quant.py)
> ```bash
> # 1) Calibrate an MP table (ctx 2048 to match deploy; produces <table>.json).
> #    Near-lossless WF-RQ = --objective sigma2; act_global baseline omits it.
> python benchmark/ppl/calibrate_mp_thresholds.py --model_path <hf> \
>   --mp_levels 128,96,64 --budget_ratio 0.67 --budget_ref_stoc_len 128 \
>   --sc_prec 8 --halve 1 --ctx_len 2048 --budget-scope global \
>   --objective sigma2 --output_json benchmark/ppl/mp_calib/<safe>__wfrq.json
> #    then write <safe>__wfrq_wrapper.json = {"type":"AdaptiveMPConfig",
> #    "stoc_len_levels":[128,96,64],"threshold_table_path":"<safe>__wfrq.json"}
>
> # 2) Evaluate MP in the HPCA protocol (same driver as INT + SC-uniform):
> MODEL_PATH=<hf> QUANT_CONFIG=mp CTX=2048 SQ_ALPHA=0.5 \
>   MP_CONFIG_JSON=benchmark/ppl/mp_calib/<safe>__wfrq_wrapper.json \
>   ACT_SCALES_DIR=benchmark/ppl python -u benchmark/quant/eval_quant.py
>
> # INT baselines / SC uniform baselines (unchanged):
> bash hpca --metrics ppl                                    # INT baselines
> bash hpca --tag sc_uniform --configs sc_int8,sc_avg192,sc_int7,sc_avg96,sc_int6 \
>   --metrics ppl --models 14B                               # SC uniform
> ```
>
> ### Repo state
> Branch `feat/mp-config-wiring`, HEAD `fc57656` (mp/trace wiring, M1 global qk
> classify, marg/grad_group calibrators). The calibrator IS committed here — the
> old "269-line uncommitted diff" note is stale. Per-session memory index:
> `~/.claude/projects/-home-allenjin-Projects/memory/MEMORY.md`
> ([[scmp-mp-root-cause]], [[scmp-claim-evidence-audit]], [[hpca-sc-uniform-launch]],
> [[hpca-baseline-fairness-audit]]).
> ═══════════════════════════════════════════════════════════════════════════

This repo swaps every `torch.matmul` / `nn.Linear` inside Llama-3.1-8B with
stochastic-computing (SC) kernels from `scmp_kernels`. The goal is to measure
quality and speed of an LLM running entirely on SC matmul, and to localize
where SC quantization error concentrates.

## Layout

```
.
├── model/
│   └── llama_sc.py          # HF Llama modeling with SCLinear + sc_matmul attention
├── kernels/                 # git submodule -> scmp_kernels package
├── test.py                  # generation smoke test (env-driven)
├── check_mse.py             # end-to-end logit MSE: SC vs fp16
└── check_perlayer_mse.py    # per-matmul MSE via forward hooks
```

The submodule directory is named `kernels/` (not `scmp_kernels/`) so it doesn't
shadow the installed `scmp_kernels` Python package via PEP 420 namespace
fallback. Don't rename it back.

## Environment

Tested on Great Lakes node `gl1802` (NVIDIA RTX PRO 6000 Blackwell, 98 GB).
Conda env name in our setup: `annstention`.

### Great Lakes GPU / Slurm rules — persistent reference

Use Slurm, not local `python`, for GPU work. The login/Codex shell has no GPU and
often no `annstention` Python packages. Always activate conda inside the Slurm
payload.

When running Slurm CLI commands from Codex tools, request unsandboxed/escalated
execution. On 2026-07-08, the same `gl-login6` shell could run `squeue` normally,
but sandboxed Codex commands could not contact `glctld` (`Slurmctld(primary) at
glctld is DOWN`, or `squeue` hung until timeout). The same commands worked
immediately with escalation:

```bash
squeue -u allenjin
scontrol ping
```

For persistent experiments launched from Codex, prefer **`sbatch`**. Background
`nohup srun ... &` children from Codex can be reaped before they open stdout. Use
one GPU per job for independent cells; this lets Slurm start as many as the
account cap allows. The known-good resource shape is:

```bash
sbatch --job-name=<name> \
  --account=nbleier_owned1 --reservation=rtx6000_arph_nodes \
  --partition=gpu-rtx6000 --gres=gpu:1 --cpus-per-task=12 --mem=180G \
  --time=24:00:00 \
  --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/<name>_%j.out \
  --wrap='source ~/.bashrc; conda activate annstention; cd /home/allenjin/Projects/scmp_llm; <command>'
```

For interactive diagnostics inside an already-running allocation, use `srun
--jobid=<jobid> --gres=gpu:0 bash -lc '...'` only for CPU-side checks such as
`ps`, `tail`, or reading logs. Do **not** expect it to see a GPU unless requesting
a GPU step or running inside the original batch payload. For GPU Python checks,
submit a small `sbatch` smoke instead.

The older interactive pattern still works from a normal login shell, but is not
the default from Codex:

```bash
nohup srun --account=nbleier_owned1 --reservation=rtx6000_arph_nodes \
  --partition=gpu-rtx6000 --gres=gpu:1 --cpus-per-task=12 --mem=120G \
  --time=40:00 bash -lc 'source ~/.bashrc; conda activate annstention; cd ...;
  python -u ...' > /scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/<run>.out 2>&1 &
```

```bash
conda create -n annstention python=3.10 -y
conda activate annstention

# Pinned versions that work together. Drift from these and things break
# (transformers 5.x removed LossKwargs; accelerate 1.13 needs numpy 2.x).
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
pip install triton==3.6.0
pip install transformers==4.51.3 tokenizers==0.21.1
pip install accelerate==1.13.0 huggingface-hub==0.36.2
pip install numpy==2.0.2 einops==0.8.2 nvtx==0.2.14
```

Clone with submodule and editable-install the kernel package:

```bash
git clone --recurse-submodules https://github.com/CrucibleComputingGroup/scmp_llm.git
cd scmp_llm
pip install -e ./kernels
```

If you cloned without `--recurse-submodules`:
`git submodule update --init --recursive` then the pip install.

## HuggingFace access

Llama-3.1-8B-Instruct is gated. One-time:

```bash
hf auth login --force   # paste a token with model-access permission
```

The model is ~16 GB and downloads on first run to `~/.cache/huggingface`.

## Generation smoke test

```bash
python test.py                              # SC enabled, defaults
DISABLE_SC=1 python test.py                 # fp16 baseline
SC_PREC=8 SC_STOC_LEN=128 python test.py    # override SC params
```

Defaults: prompt `"please explain LLM"`, 128 new tokens, greedy decode,
`sc_prec=8`, `sc_stoc_len=256`, `sc_mode=bipolar`. Attention uses `per_head`
granularity; linear uses `per_row` + `chunk_d=128` (required because Llama
D≥4096 won't fit `cum_indicator` in L2 without chunking).

Reference timings on the test GPU:

| config | ms/tok |
|---|---|
| fp16 baseline | 19 |
| SC sc_prec=8 stoc_len=256 | 1647 |
| SC sc_prec=8 stoc_len=16 | 1583 (gibberish output) |

The ~87× SC slowdown (relative to fp16) is dominated by `cum_indicator` table
build + Triton launch overhead. Sweeping `stoc_len` from 256 down to 16
changes runtime by ~4%; it only affects quality.

## SC knobs (read by `model/llama_sc.py` from `LlamaConfig`)

| attr | default | scope |
|---|---|---|
| `use_sc_attn` | True | enable SC for attention matmuls |
| `use_sc_linear` | True | enable SC for nn.Linear projections |
| `sc_prec` | 8 | quantization grid; 8 → 256 levels |
| `sc_stoc_len` | 256 | stochastic stream length |
| `sc_mode` | `"bipolar"` | locked because `chunk_d` only supports bipolar |
| `sc_granularity` | `"per_head"` | attention path |
| `sc_linear_granularity` | `"per_row"` | linear path |
| `sc_linear_chunk_d` | 128 | linear inner-dim chunking; needed for D≥1024 |

Override at runtime by mutating `model.config.<attr>` after `from_pretrained`,
or via env vars exposed by `test.py` (`DISABLE_SC`, `SC_PREC`, `SC_STOC_LEN`).

## Diagnostic scripts

### `check_mse.py` — end-to-end logit MSE

One forward pass, sweeps `STOC_LENS` env var (default `256,128,96,64,48,32,16`)
against a saved fp16 reference. Reports MSE, max |Δ|, argmax match rate, and
next-token agreement per config.

```bash
python check_mse.py                                # default sweep
STOC_LENS=256,128,64,32 python check_mse.py        # custom sweep
```

Expected: MSE grows monotonically as `stoc_len` shrinks; next-token argmax
stays correct down to `stoc_len=48`; flips at `stoc_len=32` (and full
sequence-wide argmax-match drops from 80% at stoc_len≥96 to 40% at
stoc_len≤64 — single-token agreement degrades earlier than the next-token
prediction does).

### `check_perlayer_mse.py` — per-matmul MSE

Hooks every `SCLinear`, runs one forward pass, and for each matmul compares
the SC output to the cuBLAS reference computed on the same input. Aggregates
by projection type and lists the top-10 worst single matmuls.

> **Scope:** this script captures the Q/K/V/O + MLP **projections** only —
> hooks fire on `SCLinear` modules. The Q·Kᵀ and softmax·V matmuls live
> inside `eager_attention_forward` (not `nn.Linear`) and are **not**
> measured here, even after PR #3. The projection breakdown below is
> therefore independent of the sdpa-fallback bug and still applies.

```bash
python check_perlayer_mse.py
SC_STOC_LEN=128 python check_perlayer_mse.py
```

Expected pattern at `sc_prec=8 stoc_len=256`:

- **`k_proj`** is the worst by *mean* MSE (~1.3e-3) — systematic per-row
  quantization error from 4096-wide outputs feeding into Q·Kᵀ.
- **`down_proj`** has the worst *single* matmul (~4e-3 at the last layer)
  because its input (post-SiLU·up_proj) has sparse outliers up to ~480
  in magnitude. This is the LLM analog of the diffusion FC2 issue: a
  single big value per row collapses the quantization scale.
- `o_proj` is essentially free (MSE ~3.5e-6) — small dynamic range,
  no outliers.

If you need to skip specific layers from SC, the candidates in priority order
are: late-layer `down_proj` first, then all `k_proj`.

## Qwen3-4B-Instruct (`model_qwen4b/`)

A second SC integration, this time as a thin non-invasive adapter rather
than a forked modeling file. Two surfaces are SC-ified:

1. Every `nn.Linear` inside each `Qwen3DecoderLayer` (q/k/v/o + gate/up/down)
   is replaced with `SCLinear`, reusing the loaded weight tensors.
2. `transformers.models.qwen3.modeling_qwen3.eager_attention_forward` is
   module-globally swapped for `sc_eager_attention_forward`, which routes
   Q·Kᵀ and softmax·V through `scmp_kernels.sc_matmul` when `use_sc_attn`.

Unlike `model/llama_sc.py`, `make_qwen3_sc` forces
`attn_implementation="eager"` so the SC attention path actually runs (the
llama smoke test defaults to `sdpa` and so only exercises `SCLinear`).

### Run

```bash
bash model_qwen4b/_run_test.sh fp16  DISABLE_SC=1   # baseline
bash model_qwen4b/_run_test.sh sc256                # SC defaults
SC_PREC=8 SC_STOC_LEN=128 bash model_qwen4b/_run_test.sh sc128  # custom
```

The wrapper sets, before invoking `python test.py`:

- `HF_HOME=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/hf_cache`
  — lab-shared scratch. Model weights must not live in `/home` (quota).
- `QWEN_MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507` (override via env).

Logs land in `model_qwen4b/_run_<tag>.log` with a `_run_<tag>.done` touch
on clean exit. Both are ignored by `.gitignore`.

### Reference timings on PRO 6000 Blackwell (gl1810)

| config | ms/tok |
|---|---|
| fp16 baseline | 19 |
| SC sc_prec=8 stoc_len=256 | 1208 |

### Kernel dependency

The SC attention path stresses the per-head bipolar kernel with
**asymmetric** matmuls (softmax·V where `N_q ≠ K_kv`). Requires
scmp_kernels at the PR-#10 merge or later — the original kernel
hardcoded `M = N` for the symmetric Q·Kᵀ case and crashed on the
softmax·V reshape. Tracked here as a submodule at
`kernels/`; bump with `git submodule update --init --recursive`.

### MoE support (Qwen3 30B-A3B / 235B-A22B)

`qwen3_sc.py` also patches
`transformers.models.qwen3_moe.modeling_qwen3_moe.eager_attention_forward`
when that module is importable, so MoE checkpoints use the same SC
attention path as dense Qwen3. The MoE **router** (the
`Qwen3MoeSparseMoeBlock.gate` Linear that picks top-k experts) is
explicitly excluded from `SCLinear` replacement via
`_SKIP_LINEAR_NAMES = {"gate"}` — quantizing the router corrupts
top-k expert selection and collapses output to gibberish at every
`stoc_len`. `gate_proj` (MLP gate projection, different role) is
**not** skipped.

Run the same wrapper with a MoE model id:

```bash
QWEN_MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  bash model_qwen4b/_run_test.sh sc256_moe
```

### Diagnostic scripts

Three SC-quality diagnostics ported from `scmp_llm/check_*.py`. All take
`QWEN_MODEL_PATH`, `SC_PREC`, and (where relevant) `STOC_LENS` /
`SC_STOC_LEN` from the environment. Assertions enforce
`max(STOC_LENS) <= 2**SC_PREC` — never sweep past the SC RNG grid.

| script | what it measures |
|---|---|
| `check_mse.py` | One forward pass per `stoc_len`. Reports logit MSE, max\|Δ\|, argmax-match %, and next-token (SC vs FP16) per config. Default sweep `256,192,128,96,64,48,32,24,16,12,8`. |
| `check_perlayer_mse.py` | Hooks every `SCLinear`, captures `(x, sc_out)` and computes `F.linear(x, W, b)` as the cuBLAS reference. Aggregates by projection (q/k/v/o + gate/up/down) and lists top-10 worst single matmuls. |
| `check_gen.py` | Decodes `NEW_TOKENS` (default 64) under each `stoc_len`. Use this — not MSE — to see where prose collapses into rubbish; autoregressive feedback amplifies single-token noise. |

Wrapper scripts under `model_qwen4b/` automate common combinations:

```bash
# Single-model overnight quality sweep: logit MSE (Phase 1) +
# per-layer MSE at stoc_len ∈ {256, 64, 16} (Phase 2).
bash model_qwen4b/_run_sweep.sh sweep

# Multi-model run across {4B, 8B, 14B, 30B-A3B MoE}. Each model
# gets its own _run_<tag>.log; per-model failures do not block others.
bash model_qwen4b/_run_sweep_all.sh

# Qualitative generation sweep: decoded text per stoc_len.
bash model_qwen4b/_run_gen.sh gen
```

### Empirical quality floor (Qwen3-4B-Instruct-2507, default prompt)

From `check_gen.py` with `NEW_TOKENS=64`:

- `stoc_len ≥ 96` — indistinguishable from FP16 for practical use.
- `stoc_len = 64` — still coherent English on-topic; minor stylistic drift.
- `stoc_len = 48` — **cliff**. Output starts a sentence then collapses into nonsense within ~10 tokens.
- `stoc_len ≤ 32` — token salad / mojibake.

MSE alone is misleading: MSE 2.4 at `stoc_len=48` looks comparable to MSE 2.8 at `stoc_len=32`, but autoregressive feedback turns the former from one-bad-token-recoverable into total collapse.

## Mixed-precision (MP) calibration — per-token-row SC stream length

> **⚠ BUDGET FIX 2026-07-05 — read the ⚑ STATUS block first.** The mechanics below
> (per-row dispatch, calibrator flags, pipeline) are current, but the budget was
> **row-weighted** (`Σ R_g·L_g`) which is NOT iso-compute. Always pass
> **`--budget-weight macs --mac-weights-trace <sc_int7 trace>`** (FLOP/energy budget,
> iso-compute). The **§Findings (iso-budget…)** and **§Why measured/gradient lose**
> subsections below are **SUPERSEDED row-weighted artifacts** — under the MAC budget
> `act_global ≈ measured` and MP beats uniform everywhere; ignore their rankings.

Instead of one global `stoc_len`, MP assigns each **token row** (for linears /
softmax·V) or **query row** (for Q·Kᵀ) its own `stoc_len` from a small set of
levels, based on a calibrated **threshold on the row's activation magnitude**
(`abs().amax(-1)`, normalized to [0,1] per call). Rows that matter get longer
SC streams; the rest get shorter ones, at a fixed average-`stoc_len` budget.

Pipeline:

```
calibrate_mp_thresholds.py   →  <table>.json (per-(operator,layer-bucket) thresholds)
        ↓ wrapped by
<table>_wrapper.json (MP_CONFIG_JSON)  →  AdaptiveMPConfig.load_threshold_table
        ↓ dispatched at runtime by
model/sc_common.py (SCLinear / sc_eager_attention_forward, per-row level dispatch)
```

Calibration runs an FP teacher forward over a few wikitext2 windows, measures
each row's **per-level SC reconstruction error** σ (relative L2 vs the FP
output) at every `stoc_len` level, then solves a budget-constrained allocation
(Lagrangian on σ) and converts the per-level row counts into thresholds on the
sorted activation metric. Operators calibrated: `q/k/v/o/gate/up/down_proj`,
`qk` (Q·Kᵀ), `av` (softmax·V).

### Importance signals (`--method` in the sweep) — what each is for

All share the same runtime mechanism (threshold on the activation metric); they
differ only in **what objective sets the thresholds**:

| method | objective | what it's testing |
|---|---|---|
| `act` | minimize Σσ (reconstruction error), **per-(op,layer) budget** | baseline — every layer pinned to the same avg stoc_len |
| `grad` | weight each row's σ by its loss sensitivity `‖∂L/∂y_row‖` (one extra backward) | does clean-forward loss-gradient importance help? |
| `grad_sc` | same, but `g` measured on the **SC-noisy** trajectory via a straight-through estimator (`--grad-on-sc`) | does noisy-trajectory gradient help where clean grad fails (high noise)? |
| `act_global` | minimize Σσ with **ONE global budget** (`--budget-scope global`) | cross-layer: let budget flow from insensitive to sensitive layers |
| `grad_global` | gradient-weighted σ, global budget | cross-layer with a gradient signal |
| `measured` | global budget, each (op,layer) group weighted by **measured ΔLoss** from a knock-down probe **to the floor**, `act` quantile within (`--cross-layer-weight measured`) | cross-layer with a ground-truth loss-sensitivity signal |
| `measured_marg` | **FIX 1** for `measured`: probe knocks each group down a *small step* from baseline (`--measure-marg-frac`, not to the floor), so ΔL ≈ ∂L/∂budget stays in the locally-linear regime at every precision (`--cross-layer-weight measured_marg`) | does a *marginal* (non-cliff) probe fix measured's int7/len96 inconsistency? |
| `grad_group` | **FIX 2** for `grad`: aggregate g to ONE per-group `W_g=mean\|∂L/∂y\|` (winsorized), `act` σ within group — no per-row g·σ multiply (`--cross-layer-weight grad_group`) | does using gradient only as a coarse cross-layer weight (averaging out per-row noise) beat act_global without the cliff blowup? |

### Precision configs (`--prec`)

Levels and budgets are in **halved space** (`halve_bipolar_stoc_len=1`, so the
cap is `2**(sc_prec-1)=128`; a level value *is* the cycle count). `int8` is the
uniform no-MP ceiling.

| prec | levels | target avg stoc_len |
|---|---|---|
| `int8` | uniform 128 | 128 |
| `len192` | 128,96,64 | 91 |
| `int7` | 128,64,32 | 64 |
| `len96` | 64,48,32 | 48 |

### How to run

> ⚠ **DEPRECATED (2026-07-04): `run_mp_sweep.sh` / `reproduce_crosslayer.sh` /
> `ppl.py` were DELETED** (they were the arch_impl 65k/ctx-1024 protocol that
> polluted comparisons). Calibrate with `calibrate_mp_thresholds.py --ctx_len
> 2048` then evaluate via `QUANT_CONFIG=mp MP_CONFIG_JSON=<wrapper>
> benchmark/quant/eval_quant.py` (see the ⚑ STATUS "Commands" block). The block
> below is retained only for the calibrator flag reference.

```bash
# (deprecated driver — kept for flag reference only; ppl.py/run_mp_sweep removed)
# bash tests/reproduce_crosslayer.sh
# bash tests/reproduce_crosslayer.sh --max-tokens 4096          # quick smoke

# Or drive the sweep directly (one or more models/precs/methods):
bash tests/run_mp_sweep.sh --models 4B,8B,14B \
  --prec int8,int7,len96,len192 --method act,act_global,grad_global,measured

# Re-print the summary table from any sweep outdir (no GPU needed):
python benchmark/ppl/summarize_mp_results.py benchmark/ppl/_mp_overnight_<tag>

# Calibrate a single table by hand:
python benchmark/ppl/calibrate_mp_thresholds.py --model_path <hf> \
  --mp_levels 128,64,32 --budget_ratio 0.5 --budget_ref_stoc_len 128 \
  --sc_prec 8 --halve 1 --budget-scope global \
  --output_json benchmark/ppl/mp_calib/<safe>__int7_act_global.json
```

The sweep sets `SC_OWEN_MODE=bitrev` (Owen scramble ON, deterministic) and
`SC_SCRAMBLE_MASKS=64`. Calibration and PPL inherit the same Owen mode AND mask
count — they MUST match or thresholds won't transfer. `--recalibrate` forces
fresh tables (needed after changing Owen mode, mask count, or the calibrator).

**Scramble knobs (consolidated 2026-06-03):** valid `SC_OWEN_MODE` values are
`bitrev` (default) / `random` / `off`. Bitrev's mask for dim `d` is
`bit_reverse(d mod M)` with `M = min(SC_SCRAMBLE_MASKS, 2^sc_prec)`; the kernel
default is **64** (6-bit mask pattern). Removed knobs — all fail loudly if set:
`SC_OWEN_MODE=counter`, `SC_DISABLE_OWEN=1` (use `SC_OWEN_MODE=off`), and
`SC_SCRAMBLE_RESCALE` (scramble-before-rescale is now always on; the `=0`
legacy path shared one Sobol trajectory across dims and was known-catastrophic
at short stoc_len). **Comparability warning:** every table produced before
2026-06-03 — including `_mp_overnight_xlayer_fix` — ran at `M=256`; to
reproduce or patch those cells, `export SC_SCRAMBLE_MASKS=256`. PPL at `M=64`
is a different operating point and needs a fresh int8/FP16-relative baseline.

### Findings (iso-budget, wikitext2 65k tokens, ctx 1024, SmoothQuant α=0.5, bitrev)

> ⚠ **These tables are the Jun-2 `_mp_overnight_full_bitrev_qkfix` run — PRE-Jul-2
> kernel and M=256 before 06-03: method *rankings* hold but absolute PPLs are NOT
> citable.** The clean post-kernel-fix numbers (config C) are in the ⚑ STATUS
> block at the top. `uniform sl=128` here is 2× the int7 budget — it is a
> budget-halving ceiling, NOT the iso-budget uniform@64 comparator (still missing).

PPL @ realized avg_sl (halved cycles). **bold** = best method in the row.

**Qwen3-4B** (FP16 11.27, int8 ceiling 12.15)

| prec (target) | act | act_global | grad_global | measured |
|---|---|---|---|---|
| len192 (91) | 13.35 @94 | **13.17** @93 | 14.42 @91 | 13.26 @92 |
| int7 (64)   | 17.02 @66 | **16.87** @66 | 58.52 @62 | 17.95 @64 |
| len96 (48)  | 23.65 @49 | 21.97 @49 | 48.38 @48 | **20.36** @48 |

**Qwen3-8B** (FP16 11.10, int8 ceiling 11.88)

| prec (target) | act | act_global | grad_global | measured |
|---|---|---|---|---|
| len192 (91) | 12.87 @94 | 12.84 @93 | 13.76 @91 | **12.22** @92 |
| int7 (64)   | 15.50 @67 | **14.96** @67 | 19.47 @63 | 15.59 @64 |
| len96 (48)  | 18.56 @49 | **17.21** @49 | 18.91 @48 | 17.40 @48 |

**Qwen3-14B** (FP16 9.85, int8 ceiling 10.03)

| prec (target) | act | act_global | grad_global | measured |
|---|---|---|---|---|
| len192 (91) | 10.34 @95 | 10.35 @94 | 11.27 @92 | **10.21** @91 |
| int7 (64)   | 11.58 @67 | **11.55** @66 | 14.24 @63 | 12.60 @66 |
| len96 (48)  | 12.34 @49 | **12.02** @49 | 13.76 @49 | 12.63 @48 |

Reproduce: `bash tests/reproduce_crosslayer.sh` (table via
`summarize_mp_results.py`). Summary:

- **Cross-layer helps**: `act_global` ≥ uniform `act` in every model×prec cell,
  with real wins in the higher-noise regimes (int7, len96).
- **Reconstruction error is the best signal** — `act_global` is the most
  consistent winner.
- **`measured`** is inconsistent (wins low-noise len192, loses int7).
- **Gradient is the worst** — `grad`, `grad_sc`, and `grad_global` all lose to
  `act`; `grad_global` is catastrophic at high noise (4B int7 PPL ~58). Gradient
  magnitude is a poor precision-importance signal for SC.

### Why measured/gradient lose, and the two fixes (`measured_marg`, `grad_group`)

Root cause (see `arch_impl/DESIGN_measured_grad_fix.md`): every weighted variant
shares the skeleton **global budget · per-group weight W_g · act σ-quantile within
group**, and differs only in how W_g is sourced. The loss signal they add is
lower-fidelity than σ in the tested regimes:

- `measured`'s probe knocks a whole group to `min(levels)` (the floor). At len192
  the floor (64) is mild → ΔL ≈ marginal sensitivity → wins; at int7/len96 the
  floor (32) is at/past the **collapse cliff** → ΔL saturates nonlinearly →
  inconsistent.
- `grad/grad_global` use a per-row `g·σ` objective; g is heavy-tailed, clean-
  trajectory, first-order → concentrates the global budget on a few rows (rest hit
  the cliff), and double-counts σ. "grad also uses σ" does NOT make it ≥ act: grad
  *replaces* act's objective, and a biased weight loses to a uniform one.

Fixes (additive, opt-in; existing methods byte-identical):
- **`measured_marg`** — marginal probe (small step from baseline, grid-independent),
  so ΔL stays locally-linear at every precision.
- **`grad_group`** — gradient aggregated to a per-group W_g (mean |∂L/∂y|,
  winsorized), act σ within group; the per-row noise averages out.

Run all four for the comparison: `--method act_global,measured,measured_marg,grad_group`.
Oracle: the fix's PPL must be ≤ `act_global` per cell, especially at int7/len96.

> **RESOLVED (2026-07-03):** the fixes were run through Phase 2–4 (see ⚑ STATUS
> block). Neither fix beats `act_global` under the clean config C: at 4B int7,
> act_global 15.49 < measured 17.96 < measured_marg 24.90 << grad_group 639.
> **Decision: ship `act_global` config C.** `grad_group` has a structural
> post-softmax scale artifact (all qk buckets clamp to floor → sub-cliff);
> `measured_marg` is probe-noise dominated (ICC 0.46). The 3-GPU fan-out
> (`arch_impl/launch_overnight_xlayer.sh`) also introduced GPU non-reproducibility
> — **run the deciding/paper cells on a single pinned GPU.**

### Two bugs that gated cross-layer (fixed — don't reintroduce)

Cross-layer only works because of two fixes in the **global** solve (per-bucket
`act`/`grad` are unaffected):

1. **rep_g cost pricing.** The global Lagrangian must price each row's cost by
   `rep_g = R_g / n_g` (true per-forward row count / stored subsampled count).
   Without it, attention (many rows) dominates the budget and a single shared λ
   craters the cheap, few-row linear layers — landing *worse than uniform on its
   own objective*. (`_global_lambda` / `_fit_group(rep=...)`.)
2. **qk calibrated per-row, not per-head.** Q·Kᵀ runs **per query row**
   (B·H·N) at runtime, so it must be calibrated per-row too. The old per-head
   metric (H units) under-counted qk ~1000× in the budget, so cross-layer
   over-spent and realized avg_sl drifted far past target (int7 64→92). The
   `qk` branch in `calib_eager` now mirrors the `av` (per-row) branch.

Each calibration table records `expected_avg_stoc_len` (predicted row-weighted
budget) and `global_lambda` — use them to confirm a global table actually holds
budget before trusting its PPL.

### MoE empty-expert calibration bug (fixed — don't reintroduce)

Calibrating a **MoE** model (Qwen3-30B-A3B) was never actually run before — the
findings above are dense-only (4B/8B/14B). The first 30B-A3B calibration crashed in
`_normalize_metric` with `min(): Expected reduction dim ... numel() == 0`: under
sparse top-k routing an expert can receive **zero tokens** in a forward, so its
`gate/up/down_proj` gets an empty `(0, D)` input and `abs().amax(-1).min()` reduces
over an empty dim. Fixed by skipping the SCLinear calibration hook when
`x_flat.shape[0] == 0` and hardening `_normalize_metric` for empty input
(`calibrate_mp_thresholds.py`). This affects **all** methods on MoE, not just the
new ones.

A **second** MoE bug surfaced in the gradient path: `PendingMerger` keyed its
per-row g accumulator by `(operator, block_idx)`, but MoE experts share both the
op-name and the layer index while receiving **different token counts**, so
`g_sum + g` crashed on a size mismatch (e.g. 41 vs 8 rows). Fixed by keying the
accumulator on **call position** (`cid`) — unique per matmul call within a forward
and stable across SC draws (routing is deterministic; the gate is FP/excluded), so
it is byte-identical for dense models and disambiguates MoE experts. This affects
**all** gradient methods on MoE (`grad`/`grad_global`/`grad_sc`/`grad_group`).

### Large-model grad backward

`grad`/`grad_group`/`grad_sc` need a backward pass. Gradient checkpointing now fires
for 14B/30B/32B (was MoE-only) so the backward fits in 98 GB; force with
`CALIB_GRAD_CKPT=1`. Lower `--grad-ctx` (e.g. 512) for extra headroom. SmoothQuant
`act_scales_*.pt` for 30B-A3B and 32B are already calibrated; MP calibration + PPL
for them are wired but only first run in the `xlayer_fix` overnight sweep.

## Precision trace (energy/latency simulator input)

`scmp_kernels/trace.py` logs, for **every `sc_matmul` call** (MP and uniform),
the effective precision (`stoc_len` = true cycle count, post-halving) plus
shape and identity. Zero overhead when off (one bool read); records only
host-side shape metadata — never tensor values, so no device sync.

```bash
SC_MP_TRACE=out.json          python benchmark/ppl/ppl.py   # summary (default)
SC_MP_TRACE=out.json SC_MP_TRACE_MODE=trace ...             # per-call JSONL
```

- **summary** (`scmp-trace-summary-v1`): per-(block, op, unit, stoc_len,
  rng_levels, smoothed) groups with `calls / rows / macs / row_cycles`;
  `d_in`/`d_out` are representative payload (NOT key — attention dims grow
  per decode step; such groups carry `dims_vary: true` while macs/rows stay
  exact). Energy = Σ macs × E(stoc_len). `rng_levels` is the RESOLVED
  enable-grid size (never null). A few hundred KB for PPL runs.
- **trace** (`scmp-trace-v1` JSONL): one ordered record per call (`seq`) —
  latency timeline replay. Spills to disk every 100k records (bounded RAM
  on long decodes); after a spill the flush-path override is ignored.
- `unit` = MoE expert index (from digit-named ModuleList ancestors at
  `replace_linears_with_sc` time) or attention head index in the per-(B,H)
  MP dispatch (uniform attention records are one 3D call, `batch=B*H`,
  `unit=null` — consumers must accept both flavors). The MoE router `gate`
  never appears. **Coverage is SC matmuls only** — lm_head, embeddings,
  norms, softmax run FP16 and are absent (header carries a `coverage` note);
  level-0 (drop) rows issue no matmul and are not recorded.
- `ppl.py` writes one file per sweep config (`<base>_sl<N>.json`) with
  model/config/ppl **and join metadata** (MP_CONFIG_JSON path, total_blocks,
  SmoothQuant setting) in the header; `check_mse.py`/`check_gen.py` are
  wired the same way. An `atexit` hook flushes for apps that never call
  `flush()` (header marked `"atexit": true` — may span multiple configs).
  `calibrate_mp_thresholds.py` **disables** tracing (probe workloads are not
  simulator input). Context is thread-local; accumulation is locked.
- Identity comes from `trace.set_context(...)` — scmp_llm sets it in
  `SCLinear.forward` and every SC attention path (incl. the STE and
  knock-down-probe helpers); other apps (diffusion/ViT) need only that one
  call to adopt.
- Validated: 4B MP run — rows conserve exactly (Σ per-op rows = tokens;
  qk = H×tokens), trace avg_sl == mp_tracker avg_sl, macs == rows·d_in·d_out;
  30B MoE — per-expert rows sum to tokens×top_k (2048 = 256×8), load
  imbalance visible (0–89 rows/expert). Tracing does not change PPL
  (bit-identical reruns). Reviewed by a 3-lens adversarial pass; all
  confirmed findings fixed.

## Common failure modes

- `ImportError: cannot import name 'LossKwargs'` — `transformers>=5` removed
  it. Pin to `4.51.3`.
- `numpy._core.multiarray` AttributeError — `accelerate>=1.13` needs
  `numpy>=2`. Bump numpy to `2.0.2`.
- `scmp_kernels has no attribute 'sc_matmul'` — the local `scmp_kernels/`
  directory is shadowing the installed package as a namespace package.
  Ensure the submodule is at `kernels/`, not `scmp_kernels/`.
- `CUDA error: device unavailable` from `_sc_matmul_per_row_mlp` — usually
  transient; the GPU is in `Exclusive_Process` mode and another context
  briefly held it. Retry.
