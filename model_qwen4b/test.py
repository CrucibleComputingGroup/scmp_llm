# Generation smoke test for Qwen3-4B with the SC integration.
# Same env-driven interface as scmp_llm/test.py.
import os
import time

import torch
from transformers import AutoTokenizer

from qwen3_sc import make_qwen3_sc

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-4B")
DISABLE_SC = os.environ.get("DISABLE_SC", "0") == "1"
SC_PREC = int(os.environ.get("SC_PREC", "8"))
SC_STOC_LEN = int(os.environ.get("SC_STOC_LEN", "256"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = make_qwen3_sc(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
if DISABLE_SC:
    model.config.use_sc_attn = False
    model.config.use_sc_linear = False
else:
    model.config.sc_prec = SC_PREC
    model.config.sc_stoc_len = SC_STOC_LEN
model.eval()

prompt = "please explain LLM"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

n_new = outputs.shape[1] - inputs.input_ids.shape[1]
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
mode = "FP16 baseline" if DISABLE_SC else f"SC sc_prec={SC_PREC} stoc_len={SC_STOC_LEN}"
print(f"\n[{mode}] generated {n_new} tokens in {elapsed:.1f}s = {elapsed / n_new * 1000:.0f} ms/tok")
print(f"[meta] sc_linear_replacements={getattr(model, '_sc_linear_replacements', 'n/a')}")
