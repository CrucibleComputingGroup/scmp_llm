"""Generation smoke test — Llama or Qwen, auto-dispatched.

Pick model with ``MODEL_PATH`` (HF id or local path). Defaults to llama-3.1-8b.
SC knobs via env var (see ``loader.apply_sc_env_overrides``):

    DISABLE_SC=1                — fp16 baseline
    SC_PREC, SC_STOC_LEN        — SC precision / stream length
    USE_SC_ATTN, USE_SC_LINEAR  — fine-grained 0/1 toggle
    SC_ATTN_GRANULARITY         — per_head | per_row

Prompt / output control:

    PROMPT          — explicit prompt string (default: "please explain LLM")
    PROMPT_TOKENS   — when > 0, build a filler prompt of ~this many tokens
    MAX_NEW_TOKENS  — generation length (default 128)

Examples:

    python test.py                                # llama, full SC default
    DISABLE_SC=1 python test.py                   # llama fp16 baseline
    MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 python test.py   # qwen
    PROMPT_TOKENS=1024 SC_ATTN_GRANULARITY=per_row python test.py
"""
import os
import time

import torch
from transformers import AutoTokenizer

from loader import apply_sc_env_overrides, describe_mode, load_sc_model

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
PROMPT = os.environ.get("PROMPT", "please explain LLM")
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = load_sc_model(MODEL_PATH, dtype=torch.float16, device_map="auto")
apply_sc_env_overrides(model)
model.eval()

if PROMPT_TOKENS > 0:
    filler = ("The quick brown fox jumps over the lazy dog. "
              "She sells seashells by the seashore. ") * 4096
    ids = tokenizer(filler, return_tensors="pt").input_ids[0, :PROMPT_TOKENS]
    prompt = tokenizer.decode(ids, skip_special_tokens=True)
else:
    prompt = PROMPT

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
L_in = inputs.input_ids.shape[1]

with torch.no_grad():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

n_new = outputs.shape[1] - L_in
print(tokenizer.decode(outputs[0, L_in:], skip_special_tokens=True))
peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
print(f"\n[{describe_mode(model)}] model={MODEL_PATH}  prompt={L_in} tok, "
      f"generated {n_new} tokens in {elapsed:.1f}s = "
      f"{elapsed / n_new * 1000:.0f} ms/tok, peak GPU mem = {peak_mb:.0f} MiB")
