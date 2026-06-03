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
| `measured` | global budget, each (op,layer) group weighted by **measured ΔLoss** from a knock-down probe, `act` quantile within (`--cross-layer-weight measured`) | cross-layer with a ground-truth loss-sensitivity signal |

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

The sweep sets `SC_OWEN_MODE=bitrev`, `SC_SCRAMBLE_RESCALE=1` (Owen scramble ON,
deterministic). Calibration and PPL inherit the same Owen mode — they MUST match
or thresholds won't transfer. `--recalibrate` forces fresh tables (needed after
changing Owen mode or the calibrator).

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
