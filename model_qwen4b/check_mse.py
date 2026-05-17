"""Forward-pass MSE: SC logits vs FP16 logits, swept over stoc_len.

Mirrors scmp_llm/check_mse.py but uses the qwen3_sc adapter (make_qwen3_sc)
instead of a forked modeling file. Default STOC_LENS sweep is wider than the
llama version so the morning log shows where quality breaks.
"""
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from qwen3_sc import make_qwen3_sc

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507")
PROMPT = os.environ.get("PROMPT", "please explain LLM")
SC_PREC = int(os.environ.get("SC_PREC", "8"))
# Hard constraint: stoc_len must be <= 2**sc_prec. The RNG grid is sized
# from sc_prec; going above 2**sc_prec wraps the stochastic stream and
# produces garbage logits (worse than smaller stoc_len). Default sweep
# starts at 2**sc_prec and walks down.
STOC_LENS = [int(x) for x in os.environ.get(
    "STOC_LENS", "256,192,128,96,64,48,32,24,16,12,8").split(",")]
assert max(STOC_LENS) <= 2 ** SC_PREC, (
    f"stoc_len {max(STOC_LENS)} > 2**sc_prec ({2 ** SC_PREC}); "
    f"the SC RNG grid is undersized and results are meaningless. "
    f"Either lower stoc_len or raise sc_prec."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = make_qwen3_sc(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
L = inputs.input_ids.shape[1]

with torch.no_grad():
    model.config.use_sc_attn = False
    model.config.use_sc_linear = False
    t0 = time.time()
    ref_logits = model(**inputs).logits.float()
    torch.cuda.synchronize()
    t_ref = time.time() - t0

ref_top1 = ref_logits.argmax(-1)
ref_next = tokenizer.decode(ref_top1[0, -1].item())
print(f"FP16 reference: model={MODEL_PATH}  prompt={PROMPT!r}")
print(f"                L={L}, V={ref_logits.shape[-1]}, forward={t_ref*1000:.0f}ms, "
      f"next-token={ref_next!r}\n")

print(f"{'config':<22}  {'logit MSE':>12}  {'logit max|diff|':>14}  "
      f"{'argmax match':>13}  {'fwd ms':>7}  {'next-token (SC vs ref)':<40}")
print("-" * 120)

for stoc_len in STOC_LENS:
    model.config.use_sc_attn = True
    model.config.use_sc_linear = True
    model.config.sc_prec = SC_PREC
    model.config.sc_stoc_len = stoc_len
    with torch.no_grad():
        t0 = time.time()
        sc_logits = model(**inputs).logits.float()
        torch.cuda.synchronize()
        t_sc = time.time() - t0
    mse = F.mse_loss(sc_logits, ref_logits).item()
    max_abs = (sc_logits - ref_logits).abs().max().item()
    sc_top1 = sc_logits.argmax(-1)
    match = (sc_top1 == ref_top1).float().mean().item()
    sc_next = tokenizer.decode(sc_top1[0, -1].item())
    name = f"sc_prec={SC_PREC} stoc_len={stoc_len}"
    print(f"{name:<22}  {mse:12.4e}  {max_abs:14.4f}  {match*100:12.1f}%  "
          f"{t_sc*1000:7.0f}  sc={sc_next!r}  ref={ref_next!r}")
