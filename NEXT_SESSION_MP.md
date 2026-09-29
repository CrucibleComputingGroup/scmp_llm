# New-session prompt — improve the SC mixed-precision (MP) algorithm

Paste the block below into a fresh session. It hands off a clean, *working* baseline
(the prior session was mostly debugging) and points at the real headroom.

---

You are continuing work on **`scmp_llm`** (stochastic-computing per-row mixed precision
for LLM inference; HPCA target). **Read `CLAUDE.md` first — especially the ⚑ STATUS
block.** Recent state, established last session:

**What's solved.** The MP *budget* was ROW-weighted (`Σ R_g·L_g`), which is NOT
iso-compute — attention (av+qk) is ~90% of rows but only ~7–14% of FLOPs. Fixing it
to **FLOP/MAC-weighted** (`--budget-weight macs --mac-weights-trace <sc_int7 trace>`)
made **iso-compute MP beat uniform everywhere (−8% to −39%)** and made simple
`act_global` (recon-error σ) ≈ `measured` — the expensive ΔLoss probing was only
compensating for the budget bug. **Ship `act_global --budget-weight macs`.** Results:
`hpca_results/llm/mp/fw_summary.tsv`. Method tables self-document (`calib_command`);
reuse index: `fw_manifest.tsv`.

**Harness (all reused, don't rebuild).**
- Calibrate one table: `benchmark/ppl/calibrate_mp_thresholds.py --model_path <hf>
  --mp_levels <halved,desc> --budget_ratio <r> --budget_ref_stoc_len 128 --sc_prec 8
  --halve 1 --ctx_len 2048 --budget-scope global --cross-layer-weight uniform
  --budget-weight macs --mac-weights-trace hpca_results/llm/uniform/traces/<M>_sc_int7_trace.json
  --output_json <t>.json` → write a `{"type":"AdaptiveMPConfig",...}` wrapper → eval
  via `MODEL_PATH=<hf> QUANT_CONFIG=mp MP_CONFIG_JSON=<wrapper> SQ_ALPHA=0.5
  ACT_SCALES_DIR=/nfs/turbo/coe-nbleier/allenjin/hpca/act_scales python benchmark/quant/eval_quant.py`.
- Sweep many cells on N GPUs: `benchmark/ppl/overnight_mp.{sh,sbatch}`
  (`BUDGET_WEIGHT=macs`, idempotent, writes `fw_results.tsv`/`fw_manifest.tsv`).
- Budgets in NOMINAL (int7=128, len96=96); `--mp_levels` are HALVED (int7=128,64,32).
  Uniform twins for comparison: int7↔sc_int7, len96↔sc_avg96 (weighting-invariant).

**GPU/env.** Great Lakes; account `nbleier_owned1` capped at **10 concurrent GPUs**
(shared with lab — often full; 2-GPU jobs backfill 3-GPU gaps). `--reservation=
rtx6000_arph_nodes`. conda `annstention`, `HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache`.

**Improvement directions (ranked; the win is currently 1.17–2.0× fp16 — there's room):**
1. **Push the win larger with a better per-row DISPATCH METRIC.** Runtime ranks rows
   by `|x|.amax(-1)`; the calibrator's `metric_fidelity_rho` diagnostic shows how well
   that ranks rows by their true σ-benefit (low ρ on av/qk ⇒ a better metric, not more
   levels, is the lever). Try alternative metrics (row L2, post-softmax peak for av).
2. **The "SC is finer-grained than fixed-point" thesis half — STILL NO EXPERIMENT.**
   Now that MP works: pow2-restricted vs fine-level (non-pow2, e.g. 96/48) MP at matched
   FLOP-avg, + a fixed-point-MP baseline. This is a required, missing paper result.
3. **Finish the iso-FLOP result:** confirm across sizes (14B, 1.7B, 30B, 32B — 1.7B has
   no trace yet, generate one) and add **downstream** (RULER MV-NIAH / LongBench at
   iso-FLOP MP via `QUANT_CONFIG=mp`) — PPL-only is thin.
4. **Energy story:** the budget now IS the FLOP/energy proxy — report iso-energy
   MP-vs-uniform + the trace `Σ macs·E(stoc_len)` numbers (`scmp_kernels/trace.py`),
   and compare to the INT baselines (`hpca_results/llm/int/`) at iso-bits/iso-energy.
5. **Fix `measured`'s FLOP-overspend** (it realized FLOP-avg 74/54 vs 64/48 targets) so
   the act_global-vs-measured comparison is airtight; likely a budget-realization /
   threshold-transfer issue that may affect all methods (calib-predicted vs eval-realized).
6. **Per-row within-operator MP under FW** + the av-vs-qk targeting (σ ranks av>qk;
   ΔLoss ranks qk>av) — does a light loss-aware refinement help at the aggressive len96
   budget where margins are thin?
7. **SmoothQuant alignment** (`--calib-smoothquant`): calib runs without SQ, eval applies
   α=0.5 — a known train/deploy mismatch; test if fixing it helps under FW.

**Guardrails (learned the hard way):** always validate new calibrator code with a fast
reduced-settings smoke (check qk allocation) BEFORE an overnight sbatch; sbatch must
`set +u` around `source ~/.bashrc`; never swap 8B Llama↔Qwen3-8B in comparisons; hold
model/config fixed in comparisons (fix or STOP, never silently substitute); `stoc_len ≤
2**sc_prec`. Memory index: `~/.claude/projects/-home-allenjin-Projects/memory/MEMORY.md`
([[project_scmp_mp_overnight_measured_wins]] has the full budget-fix story).
