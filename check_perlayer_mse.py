"""Per-matmul MSE: SC output vs cuBLAS reference, captured during one forward pass.

For every SCLinear in the model we register hooks that, given the same input the
SC kernel saw, compute the F.linear reference and record MSE / max|Δ|.
"""
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.llama_sc import LlamaForCausalLM, SCLinear

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "please explain LLM"
SC_PREC = int(os.environ.get("SC_PREC", "8"))
SC_STOC_LEN = int(os.environ.get("SC_STOC_LEN", "256"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.config.use_sc_attn = True
model.config.use_sc_linear = True
model.config.sc_prec = SC_PREC
model.config.sc_stoc_len = SC_STOC_LEN
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

records = []


def make_hooks(name, weight, bias):
    captured = {}

    def pre(mod, args):
        captured["x"] = args[0].detach()

    def post(mod, args, output):
        x = captured.pop("x")
        with torch.no_grad():
            ref = F.linear(x.to(weight.dtype), weight, bias).float()
            out = output.float()
            mse = F.mse_loss(out, ref).item()
            max_abs = (out - ref).abs().max().item()
            x_absmean = x.abs().mean().item()
            x_absmax = x.abs().max().item()
        records.append(
            dict(name=name, shape=tuple(x.shape), mse=mse,
                 max_abs=max_abs, x_absmean=x_absmean, x_absmax=x_absmax)
        )

    return pre, post


for name, mod in model.named_modules():
    if isinstance(mod, SCLinear):
        pre, post = make_hooks(name, mod.weight, mod.bias)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

with torch.no_grad():
    model(**inputs)

print(f"Captured {len(records)} SC matmul calls. sc_prec={SC_PREC} stoc_len={SC_STOC_LEN}\n")

# Aggregate by projection type (q_proj, k_proj, ..., down_proj)
by_kind = defaultdict(list)
for r in records:
    short = r["name"].rsplit(".", 1)[-1]  # e.g. q_proj, down_proj
    by_kind[short].append(r)

print(f"{'projection':<12}  {'count':>5}  {'mean MSE':>11}  {'max MSE':>11}  "
      f"{'mean max|Δ|':>12}  {'max max|Δ|':>11}  {'mean |x|':>9}  {'max |x|':>9}")
print("-" * 110)
for kind in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
    rs = by_kind.get(kind, [])
    if not rs:
        continue
    import statistics
    mse_list = [r["mse"] for r in rs]
    abs_list = [r["max_abs"] for r in rs]
    xm_list = [r["x_absmean"] for r in rs]
    xM_list = [r["x_absmax"] for r in rs]
    print(f"{kind:<12}  {len(rs):>5}  "
          f"{statistics.mean(mse_list):11.4e}  {max(mse_list):11.4e}  "
          f"{statistics.mean(abs_list):12.4f}  {max(abs_list):11.4f}  "
          f"{statistics.mean(xm_list):9.4f}  {max(xM_list):9.4f}")

# Top-10 worst single matmuls
print("\nTop-10 worst single matmuls by MSE:")
print(f"{'layer.proj':<35}  {'shape':<22}  {'MSE':>11}  {'max|Δ|':>9}  {'|x|max':>8}")
print("-" * 100)
worst = sorted(records, key=lambda r: r["mse"], reverse=True)[:10]
for r in worst:
    print(f"{r['name']:<35}  {str(r['shape']):<22}  {r['mse']:11.4e}  "
          f"{r['max_abs']:9.4f}  {r['x_absmax']:8.4f}")
