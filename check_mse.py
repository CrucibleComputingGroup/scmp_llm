"""Forward-pass MSE: SC logits vs FP16 logits, swept over stoc_len.

Auto-dispatches between Llama and Qwen via ``MODEL_PATH``. Same env-var
contract as ``test.py``. Additional env vars:

    STOC_LENS=256,128,96,64,48,32,16   — comma-separated sweep (default).
    SC_ATTN_GRANULARITY                — per_head | per_row (applies to all
                                          stoc_lens in the sweep).
"""
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from loader import load_sc_model

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
PROMPT = os.environ.get("PROMPT", "please explain LLM")
STOC_LENS = [int(x) for x in os.environ.get("STOC_LENS", "256,128,96,64,48,32,16").split(",")]
SC_PREC = int(os.environ.get("SC_PREC", "8"))
SC_ATTN_GRANULARITY = os.environ.get("SC_ATTN_GRANULARITY", "per_head")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = load_sc_model(MODEL_PATH, dtype=torch.float16, device_map="auto")
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
print(f"FP16 reference: model={MODEL_PATH} prompt={PROMPT!r}")
print(f"  L={L}, V={ref_logits.shape[-1]}, forward={t_ref*1000:.0f}ms, "
      f"next-token={ref_next!r}, sc_granularity={SC_ATTN_GRANULARITY}\n")

print(f"{'config':<22}  {'logit MSE':>12}  {'logit max|diff|':>14}  "
      f"{'argmax match':>13}  {'fwd ms':>7}  {'next-token (SC vs ref)':<30}")
print("-" * 110)


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
    _trace_cfg(stoc_len)
