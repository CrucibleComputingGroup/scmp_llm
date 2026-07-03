"""Decode tokens under each stoc_len in a single process.

Loads the model once and sweeps ``STOC_LENS`` to show how output quality
degrades. Auto-dispatches to llama or qwen via ``MODEL_PATH``.

Env vars:
    MODEL_PATH       — HF id (default: llama-3.1-8b)
    PROMPT           — prompt string
    NEW_TOKENS       — generation length (default 128; was 64 in qwen check_gen)
    SC_PREC          — SC precision (default 8)
    STOC_LENS        — comma-separated sweep (default 256,128,96,64,48,32,16)
    SC_ATTN_GRANULARITY — per_head | per_row (applied to whole sweep)
"""
import os
import time

import torch
from transformers import AutoTokenizer

from loader import load_sc_model

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
PROMPT = os.environ.get("PROMPT", "please explain LLM")
NEW_TOKENS = int(os.environ.get("NEW_TOKENS", "128"))
SC_PREC = int(os.environ.get("SC_PREC", "8"))
STOC_LENS = [int(x) for x in os.environ.get(
    "STOC_LENS", "256,128,96,64,48,32,16").split(",")]
SC_ATTN_GRANULARITY = os.environ.get("SC_ATTN_GRANULARITY", "per_head")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = load_sc_model(MODEL_PATH, dtype=torch.float16, device_map="auto")
model.eval()
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)


def _gen(label: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=NEW_TOKENS,
                             do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs.input_ids.shape[1]
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"=== [{label}] {n_new} tok in {elapsed:.1f}s "
          f"= {elapsed / n_new * 1000:.0f} ms/tok ===")
    print(text)
    print()


# fp16 baseline
model.config.use_sc_attn = False
model.config.use_sc_linear = False
_gen(f"FP16 baseline ({MODEL_PATH})")

# SC sweep

try:  # precision trace: one file per sweep config (SC_MP_TRACE=<path>)
    from scmp_kernels import trace as sc_trace
except ImportError:
    sc_trace = None


def _trace_cfg(stoc_len, do_reset=False):
    if sc_trace is None or not sc_trace._ENABLED:
        return
    if do_reset:
        sc_trace.reset()
        return
    import os as _os
    base = _os.environ.get("SC_MP_TRACE", "")
    root, ext = _os.path.splitext(base)
    out = sc_trace.flush(f"{root}_sl{stoc_len}{ext or '.json'}")
    if out:
        print(f"[trace] wrote {out}")

for stoc_len in STOC_LENS:
    _trace_cfg(stoc_len, do_reset=True)
    model.config.use_sc_attn = True
    model.config.use_sc_linear = True
    model.config.sc_prec = SC_PREC
    model.config.sc_stoc_len = stoc_len
    model.config.sc_granularity = SC_ATTN_GRANULARITY
    _gen(f"SC sc_prec={SC_PREC} stoc_len={stoc_len} gran={SC_ATTN_GRANULARITY}")
    _trace_cfg(stoc_len)
