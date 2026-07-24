# scmp_llm — reproduction guide

> ═══════════════════════════════════════════════════════════════════════════
> ## ⚑ CURRENT STATUS / SESSION HANDOFF — last updated 2026-07-24
>
> ### Frozen deployment archive = `hpca_results/llm/ppl/mp_best/`
> The citable per-cell best MP deployment. 4 models × targets
> {32,40,48,64,96,128}; each cell is a self-contained bundle (wrapper / table /
> hybrid_config / trace / runtime + SHA256SUMS + act_scales). `rebuild.py`
> re-selects the LOWEST full-protocol WikiText-2 test PPL per (model,target)
> across ALL experiment generations (v9 / v11 / v17 / v20 / mp_final / uniform)
> — a deliberate MIX of versions; the winning label + source path per cell is in
> `manifest.json` / `mp_best_all.csv`, human table in `SUMMARY.md`. Selection
> IGNORES cost, so a gated winner can overshoot its nominal target (e.g. 30B t48
> realizes 51.1 cyc); the `realized_flop_avg_sl` column is honest about it. To
> regenerate: `python rebuild.py` (login node, reads Turbo, no GPU). fp16 refs:
> 4B 10.0445 / llama8B 7.2130 / 14B 8.6383 / 30B 7.2613; quality boundary =
> 1.1×fp16.
>
> ### 2026-07-24 — V20 structural search folded into mp_best
> V20 = a fresh fixed-target GLOBAL structural search (`benchmark/ppl/
> wave_v20_staged/run_target.sbatch` → `final_mp_lane.sh`; ≤3 sweeps, k=2 escape
> gate, min level 16) starting each cell from its V18-clean incumbent. 4B/8B/14B
> ran the `structured` lane (tag `mp_v20_structured_{model}_t{t}_20260721`); 30B
> ran the `direct-final hyb20` lane (tag `mp_v20_final_hyb20_30B_t{t}_20260722`;
> B=8 MoE window-batching for the SEARCH only — both citable evals are B=1). All
> 16 cells finished; full-protocol test PPLs live in each branch's `eval_test/`
> (ungated) and `eval_gate2.0/` (k=2) subdirs. **GOTCHA: the `final_test_ppl`
> FIELD in each search `summary.json` is null — the real numbers are in those
> eval subdirs, tokens 298,862 (Qwen) / 288,627 (Llama), ctx 2048, max_tokens 0.**
> Verdict: small wins at mid budgets on the big models, regressions at the
> aggressive t32 everywhere (validation→test overfit — the search compresses the
> ladder to win validation windows and it doesn't transfer), ~neutral on 4B/8B.
> Folded into mp_best 2026-07-24 — 5 improved cells: 14B t40 9.2753→9.2314,
> 14B t48 9.0973→9.0174, 30B t40 8.6856→8.5611, 30B t48 8.4093→8.2245,
> 30B t64 7.8738→7.7372. V20 LOST → incumbent kept: 14B t32/t64, 30B t32.
> 4B/llama8B were already V20 in mp_best. Caveat: 30B t40/48/64 winners use a
> 20% INT mask (was 10% at t48/64) — more INT7 compute, priced on the energy
> axis, NOT iso-total-compute; the ungated variants sit at ~same PPL and closer
> to the nominal budget if a strict iso-budget table is wanted.
>
> ### 2026-07-24 — new int_swap hybrid-mask sweep = NEGATIVE at 20% / t32
> `mp_v20_intswap_mask_eval_20260721` re-evaluated frozen MP+gate wrappers with
> the int_swap-RANKED mask (rank by measured ΔL(INT7)−ΔL(SC) swap gain, instead
> of the current SC-fragility proxy `top_fraction_by_bucket_worst_delta_loss`).
> It does NOT beat the current measured_curve mask: 4B 12.781 vs 12.572 (+1.7%),
> 14B 9.614 vs 9.551 (+0.66%), llama8B 9.1577 identical. Caveat: masks came from
> the INITIAL low-window int_swap sensitivity (`hybrid_configs/
> _hpca_mp_v18_intswap_sens_20260720`); a robust 24-window / stratified / 4-fold
> version (`benchmark/ppl/v19_launch/intswap_robust_30B.sbatch` +
> `verify_30b_intswap_batching.py`) is staged UNCOMMITTED and NOT launched —
> needs approval before it runs. Memory: [[project_scmp_int_swap_mask_ranking]],
> [[project_scmp_hybrid_mask_dose]], [[project_scmp_group_ladders_v19]].
>
> ### Paper story (user-approved, unchanged)
> ONE algorithm vs INT — no variant history in the paper. Two-stage calibration:
> σ for dispatch/init (zero evals) + measured NLL for allocation (small eval
> budget). Uniform comparator for ANY MP claim = `hpca_results/llm/
> uniform_hybrid/` (uniform SC + the SAME hybrid INT mask every mp cell runs),
> NOT pure `uniform/`. Weight-only BitMoD rows labeled separately. Still-open
> half of the thesis: "SC finer-grained than fixed-point" has no dedicated
> experiment yet.
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
> **iso-FLOP takeaways (2026-07-05 `fw_summary` era — superseded by `ppl/mp_best/`
> for citable numbers, but the lessons hold):** (1) iso-compute MP beats uniform
> EVERYWHERE (−8 to −39% then), including "robust" llama8B that *lost* under
> row-weighting → the model-dependent-loss story was a budget artifact. (2) Simple
> `act_global` (recon-σ) ≈ `measured` / `measured_curve` at true iso-FLOP → the
> expensive ΔLoss probing was only compensating for the budget bug (llama8B keeps a
> small ~−1.5% `measured` edge). The old absolute PPL table lived in
> `hpca_results/llm/mp/fw_summary.tsv` (halved-cycle FLOP-avg 64=int7 / 48=len96);
> current per-(model×target) citable results are the `ppl/mp_best/` bundles.
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
> ### Experiments so far — see the mp_best archive
> The citable MP results are the `hpca_results/llm/ppl/mp_best/` bundles (per
> model × target, best-of across generations; see the status block above). Other
> canonical result trees under `hpca_results/llm/`: `int/` (INT W8..W4 baselines,
> from `results_bitmod_protocol.tsv`), `uniform_hybrid/` (**THE** uniform
> comparator — uniform SC + the same hybrid INT mask), `uniform/` (pure-SC mask
> ablation only), `mp_final/`, `mp_v9_hybdose/` (mask dose-response),
> `wonly_bitmod/` (weight-only BitMoD, label separately), `energy/`, `ruler/`.
> **SUPERSEDED — do NOT cite:** every `_mp_overnight_*` arch_impl run and the
> §Findings tables lower in THIS file (65k-token / ctx-1024 removed protocol +
> row-weighted budget bug). Their rankings are artifacts of the row→MAC budget
> fix; the iso-FLOP result inverts them (act_global ≈ measured; MP wins).
>
> ### What's citable
> Only FULL-protocol runs (full wikitext-2 ~298k tok, ctx 2048, `PPL_MAX_TOKENS=0`,
> SmoothQuant α=0.5). No truncated evals ever — smokes included. Nothing before
> 2026-07-02 (pre-06-03 used SC_SCRAMBLE_MASKS=256; masks now fixed at 64).
>
> ### HPCA plan
> - **Phase 1 — INT baselines:** DONE (`results_bitmod_protocol.tsv`, all 4
>   models, fp16 + {W8..W4}×{symm,asymm}). asymm beats symm below W6A6.
> - **Phase 2 — SC uniform + uniform_hybrid baselines:** DONE — per-cycle uniform
>   curves + the hybrid-masked comparator in `hpca_results/llm/uniform_hybrid/`.
>   (LongBench/RULER silently never record via `./hpca` — known harness gap;
>   RULER lives in `hpca_results/llm/ruler/` and cannot carry an MP claim.)
>
> ### Open issues / immediate next actions
> 1. **"SC finer-grained than fixed-point" still has NO dedicated experiment.**
>    The ladders exercise many non-pow2 rungs, but there is no pow2-restricted vs
>    fine-level MP ablation at matched avg_sl and no fixed-point-MP baseline. This
>    is the open half of the thesis (`scmp_kernels/trace.py` supplies the
>    E(stoc_len) energy model; GPU wall-clock does NOT track stoc_len).
> 2. **int_swap robust mask** (24-window/stratified/4-fold) is staged but
>    unlaunched — decide whether the negative quick result (status block) is worth
>    chasing; needs approval to run.
> 3. **Energy-axis pricing of the hybrid INT mask:** 20%-mask cells do more INT7
>    work than 10%-mask cells at the same SC budget, so mp_best's mixed mask
>    fractions are NOT iso-total-compute — the energy table must price the INT
>    layers so cells compare fairly.
>
> **In flight (2026-07-24):** avg96 @ 20% INT mask gap-fill — 4 cells, tag
> `mp_avg96_hyb20_20260724`, jobs 54619966-69, one GPU each. Reproduces the
> deployed 10% avg96 cell EXACTLY (mp_avg96f_burst128, act_global_v3, pc0.01x128
> act_collapse + MLP overrides down0.06/up0.03/gate0.03, INT7 via
> `--hybrid-int-bits-mode legacy`), changing ONLY `--hybrid-int-frac 0.10→0.20`.
> Launcher `benchmark/ppl/avg96_hyb20/run_avg96_hyb20.sbatch`. When done, compare
> full-test PPL vs the deployed 10% t96 cell (4B 10.2986 / llama8B 7.7270 / 14B
> 8.6789 / 30B 7.6510); fold the winners into `ppl/mp_best/` via a new
> `avg96_hyb20_candidate()` in `rebuild.py`. Expect little movement — t96 is
> near-lossless (1.02–1.05×fp16), so 20% has little headroom.
> Scheduler ops (squeue/sbatch/scancel) must run OUTSIDE the Codex sandbox (see
> the Slurm rules below); new waves / deletions need explicit user approval.
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
> Branch `feat/mp-config-wiring` (prior V20 checkpoint `1731c9c`). Latest commit
> adds: this CLAUDE.md refresh through V20, the int_swap Phase-0 calibrator
> changes (`benchmark/ppl/calibrate_mp_thresholds.py`, `test_int_swap.py` +
> `v19_launch/intswap_robust_30B` / `verify_30b_intswap_batching` scripts), the
> V20 14B final-eval script (`wave_v20_staged/eval_14b_final.sbatch`), and the
> 2026-07-24 20% launch scripts (`benchmark/ppl/avg96_hyb20/`). Local-only
> (untracked): `hpca.pre_isoprec_20260721.bak` (hpca driver backup). Per-session
> memory index: `~/.claude/projects/-home-allenjin-Projects/memory/MEMORY.md`.
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

### Historical method comparison (REMOVED protocol — not citable)

The Jun-2 `_mp_overnight_*` cross-layer sweeps once tabled here (65k-token /
ctx-1024 arch_impl protocol, M=256) are superseded and non-citable. Their one
durable takeaway: **reconstruction-error `act_global` was the most consistent MP
importance signal** — beating `measured` (inconsistent at the collapse cliff)
and every gradient variant (worst) — which is why the shipped algorithm
allocates on σ / reconstruction error, not gradient. Current citable results:
`hpca_results/llm/ppl/mp_best/` (see the ⚑ STATUS block).


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
