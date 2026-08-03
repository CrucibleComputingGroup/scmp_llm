# Track B — V17 fallback-search prep (agent work log)

Work dir: /home/allenjin/Projects/scmp_llm/benchmark/ppl/v17_launch/trackB/
Date: 2026-07-19. NO sbatch submission from this session; scripts are prepared only.

## Task
1. Acceptance protocol fix in mp_v16_refine.py: accept on A->B replication
   (sign replicates on B + pooled A+B significance); 16-window confirm becomes
   STOP-ONLY (cannot accept/select).
2. --patience default 3; sbatch auto-resubmit-from-checkpoint (USR1@900 trap +
   non-converged-at-exit resubmit); make walltime_guard summaries resumable.
3. New macro family lift_compound: {8 attn buckets -> top rung} x {psl -> 95}
   x {MLP floor raises to iso-cost} + partials (attention-only, psl-only+reinvest).
   psl 95 NEVER 96. No rungs < 16 introduced by this family. Threshold family
   stays OFF by default; surrogate stays ON.
4. 4 branch sbatch scripts (WRITE ONLY): 4B t32, 4B t36, llama8B t40, 14B t32,
   parents = V9 avg32 wrappers in mp_calib_mp_v9_int6_20260716_001908, --patience 4.

## Progress
- [x] Read scmp_llm/CLAUDE.md (status block + Slurm rules).
- [x] Read mp_v16_refine.py fully (1774 lines). Key structure noted below.
- [ ] Read mp_v16_lane.sh, mp_joint_refine.py helpers, build_script.py, tests.
- [ ] Code change 1 (acceptance) + tests.
- [ ] Code change 2 (patience/resume) + sbatch resubmit template.
- [ ] Code change 3 (lift_compound) + offline smoke vs V9 4B table.
- [ ] 4 sbatch scripts + README.

## Engine structure notes (mp_v16_refine.py, pre-change)
- Funnel per phase: stage A (subset A, 4 windows, rank all) -> select_advance
  (family quotas, 12) -> stage B (replicate_gate: mean_b<0 and mean_ab<-floor)
  -> finalist = min mean_ab among passers -> ONE confirm eval on 16 windows ->
  confirm_gate (docstring literally: "Sole acceptance authority") ->
  accept = verdict.accept and in_band(confirm_cost).
- 0 accepts on 14B = confirm_gate rejecting everything.
- patience: --patience default 2; state["stalled"] incremented per no-accept sweep.
- Resume: per-branch checkpoint.json written after every sweep; resume verifies
  1-window incumbent NLL bit-identity (GPU probe). Branch skipped if summary.json
  status in (ok, budget_projection_failed) — INCLUDES walltime_guard summaries
  (bug for auto-resubmit: guarded branches would be skipped, must exempt).
- Walltime guard at top of sweep loop writes stop_reason only in memory + final
  summary; on-disk checkpoint keeps stop_reason None -> checkpoint itself is
  resumable.
- pc collision guard exists: candidate psl must not equal an adaptive rung.
