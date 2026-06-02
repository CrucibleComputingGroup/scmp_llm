# scmp_llm

Run an LLM with **every `nn.Linear` and attention matmul simulated in
stochastic computing (SC)** using the Triton kernels from
[`scmp_kernels`](https://github.com/CrucibleComputingGroup/scmp_kernels). The
goal is to measure the quality and speed of an LLM running entirely on SC
matmul, and to localize where SC quantization error concentrates.

Two model integrations are included:

| dir              | model                       | how SC is wired in |
|------------------|-----------------------------|--------------------|
| `model/`         | Llama-3.1-8B-Instruct       | forked HF modeling (`llama_sc.py`): `SCLinear` + `sc_matmul` attention |
| `model_qwen4b/`  | Qwen3-4B/8B/14B + MoE        | non-invasive adapter (`qwen3_sc.py`): swaps `nn.Linear` + monkey-patches `eager_attention_forward` |

> For full reproduction details — pinned dependency versions, reference
> timings, per-layer MSE expectations, the empirical quality floor, and known
> failure modes — see [`CLAUDE.md`](CLAUDE.md). This README is the quickstart.

## Install

The kernel package is a git submodule at `kernels/` (deliberately *not*
`scmp_kernels/`, so it doesn't shadow the installed package via PEP 420).

```bash
git clone --recurse-submodules https://github.com/CrucibleComputingGroup/scmp_llm.git
cd scmp_llm
pip install -e ./kernels          # supplies sc_matmul (Triton, needs a CUDA GPU)
```

If you cloned without `--recurse-submodules`:
`git submodule update --init --recursive`, then the pip install.

Pinned versions that work together (drift breaks things — see CLAUDE.md):

```bash
pip install torch==2.10.0 triton==3.6.0 transformers==4.51.3 \
            accelerate==1.13.0 numpy==2.0.2 einops==0.8.2
```

Llama-3.1-8B-Instruct and the Qwen3 checkpoints are gated; run
`hf auth login --force` once with a token that has model access.

## Quickstart

### Llama (`test.py`)

```bash
python test.py                              # SC enabled, defaults
DISABLE_SC=1 python test.py                 # fp16 baseline
SC_PREC=8 SC_STOC_LEN=128 python test.py    # override SC params
```

Defaults: prompt `"please explain LLM"`, 128 new tokens, greedy decode,
`sc_prec=8`, `sc_stoc_len=256`, bipolar. Attention runs `per_head`; linear runs
`per_row` + `chunk_d=128` (required because Llama D≥4096 won't fit `cum_indicator`
in L2 without chunking).

### Qwen3 (`model_qwen4b/`)

```bash
bash model_qwen4b/_run_test.sh fp16  DISABLE_SC=1   # baseline
bash model_qwen4b/_run_test.sh sc256                # SC defaults
QWEN_MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  bash model_qwen4b/_run_test.sh sc256_moe          # MoE checkpoint
```

The wrapper forces `attn_implementation="eager"` so the SC attention path
actually runs, and points `HF_HOME` at lab-shared scratch (model weights must
not live in `/home` — quota). For MoE, the router `gate` Linear is excluded from
SC replacement; quantizing it collapses top-k expert selection to gibberish.

## SC knobs

Read from `LlamaConfig` / `Qwen3Config`; override by mutating `model.config.<attr>`
after `from_pretrained`, or via the env vars exposed by the run scripts.

| attr | default | scope |
|---|---|---|
| `use_sc_attn` | True | SC for attention matmuls (Q·Kᵀ, softmax·V) |
| `use_sc_linear` | True | SC for `nn.Linear` projections |
| `sc_prec` | 8 | quantization grid; 8 → 256 levels |
| `sc_stoc_len` | 256 | stochastic stream length (quality knob) |
| `sc_granularity` | `per_head` | attention path |
| `sc_linear_granularity` | `per_row` | linear path |
| `sc_linear_chunk_d` | 128 | linear inner-dim chunking; needed for D≥1024 |

## Diagnostics

```bash
python check_mse.py            # end-to-end logit MSE vs fp16, swept over STOC_LENS
python check_perlayer_mse.py   # per-matmul MSE via hooks; ranks worst projections
python check_gen.py            # decoded text per stoc_len (Qwen) — see where prose collapses
```

`check_perlayer_mse.py` covers the Q/K/V/O + MLP **projections** only (hooks fire
on `SCLinear`); the attention-score matmuls live inside
`eager_attention_forward`. Use `check_gen.py`, not MSE, to find the quality
cliff — autoregressive feedback amplifies single-token noise that MSE understates.
