"""Qualitative generation sweep: actual decoded text at multiple stoc_len.

Loads the model once, then generates ``NEW_TOKENS`` (default 64) under each
``STOC_LENS`` setting. Print the FP16 baseline first, then each SC sample
prefixed by its config. Lets you eyeball where the output stops being
coherent — MSE alone doesn't answer that.
"""
import os
import time

import torch
from transformers import AutoTokenizer

from qwen3_sc import make_qwen3_sc

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507")
PROMPT = os.environ.get("PROMPT", "please explain LLM")
NEW_TOKENS = int(os.environ.get("NEW_TOKENS", "64"))
SC_PREC = int(os.environ.get("SC_PREC", "8"))
STOC_LENS = [int(x) for x in os.environ.get(
    "STOC_LENS", "256,192,128,96,64,48,32,24,16,12,8").split(",")]
assert max(STOC_LENS) <= 2 ** SC_PREC, (
    f"stoc_len {max(STOC_LENS)} > 2**sc_prec ({2 ** SC_PREC})"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = make_qwen3_sc(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
L_prompt = inputs.input_ids.shape[1]
gen_kwargs = dict(max_new_tokens=NEW_TOKENS, do_sample=False)


def generate(label):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    new_tokens = out.shape[1] - L_prompt
    text = tokenizer.decode(out[0, L_prompt:], skip_special_tokens=True)
    ms_per_tok = elapsed / max(new_tokens, 1) * 1000
    print(f"=== {label} ===")
    print(f"({new_tokens} new tokens, {ms_per_tok:.0f} ms/tok)")
    print(text)
    print()


# FP16 baseline first
model.config.use_sc_attn = False
model.config.use_sc_linear = False
print(f"model = {MODEL_PATH}")
print(f"prompt = {PROMPT!r}   max_new_tokens = {NEW_TOKENS}\n")
generate("FP16 baseline")

# SC sweep
for sl in STOC_LENS:
    model.config.use_sc_attn = True
    model.config.use_sc_linear = True
    model.config.sc_prec = SC_PREC
    model.config.sc_stoc_len = sl
    generate(f"SC sc_prec={SC_PREC} stoc_len={sl}")
