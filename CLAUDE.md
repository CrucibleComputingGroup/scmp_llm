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
| fp16 baseline | ~18 |
| SC sc_prec=8 stoc_len=256 | ~1440 |
| SC sc_prec=8 stoc_len=16 | ~1416 (gibberish output) |

The 80× SC slowdown is dominated by `cum_indicator` table build + Triton
launch overhead. Sweeping `stoc_len` from 256 down to 16 changes runtime by
<2%; it only affects quality.

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

Expected: MSE grows monotonically as `stoc_len` shrinks; argmax of the
next-token stays correct down to `stoc_len=32`; flips at `stoc_len=16`.

### `check_perlayer_mse.py` — per-matmul MSE

Hooks every `SCLinear`, runs one forward pass, and for each matmul compares
the SC output to the cuBLAS reference computed on the same input. Aggregates
by projection type and lists the top-10 worst single matmuls.

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
**asymmetric** matmuls (softmax·V where `N_q ≠ K_kv`). Requires the
scmp_kernels commit that supports asymmetric M in
`_sc_matmul_per_head_bipolar` — the original kernel hardcoded `M = N`
for the symmetric Q·Kᵀ case and crashed on the softmax·V reshape.

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
