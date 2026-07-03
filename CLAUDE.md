# scmp_llm — reproduction guide

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

```bash
# One-command reproduction (all models × precs × methods, then prints the table).
# Serial on one GPU (~1 day); see the header for the parallel-per-node form.
bash tests/reproduce_crosslayer.sh
bash tests/reproduce_crosslayer.sh --max-tokens 4096          # quick smoke

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
**Pending overnight results on 4B/8B/14B/30B-A3B/32B** — launch via
`arch_impl/launch_overnight_xlayer.sh` (3-GPU fan-out).

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
