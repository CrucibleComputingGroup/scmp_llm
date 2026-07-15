"""Model-level cross-check of the BitMoD port: quantize with the REFERENCE
implementation (BitMoD-HPCA-25 search_datatype, imported from the clone) and
evaluate with the REFERENCE PPL loop (llm_eval_wikitext.py, replicated verbatim
below), then compare against our harness's W4A16_bitmod row.

Three controlled deviations from stock llm_eval_wikitext.py, each aligning a
known protocol difference with OUR harness so the comparison isolates the
quant-math port (unit-tested bit-identical) + integration:
  1. house quant surface: skip lm_head + MoE router `gate` (ptq._SKIP_LINEAR)
  2. fast tokenizer (stock use_fast=False loads the slow Qwen2Tokenizer for
     Qwen3 — a token-stream variant our harness never uses)
  3. attn_implementation="eager" (what our INT/bitmod cells run)
Our W4A16_bitmod cell runs SQ-free by default (A>=16), matching stock BitMoD.
Expected agreement with the harness row: ~0.01 PPL (residual = window layout
and CE-averaging details between the two PPL loops).

Env: MODEL_PATH (default Qwen3-4B), WQ_BITS (4), WQ_DATATYPE (mixed_bitmod),
     WQ_GROUPSIZE (128), BITMOD_REF (path to bitmod_quant clone).
Prints:  [CROSSCHECK] model=... datatype=... ppl=...
"""
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_REF = os.environ.get(
    "BITMOD_REF", "/home/allenjin/Projects/BitMoD-HPCA-25/bitmod_quant")
if _REF not in sys.path:
    sys.path.insert(0, _REF)
from quant_utils.quant_weight import quant_int, quant_int_asym  # noqa: E402
from quant_utils.quant_weight import quant_datatype, search_datatype  # noqa: E402

MODEL = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507")
WQ_BITS = int(os.environ.get("WQ_BITS", "4"))
WQ_DATATYPE = os.environ.get("WQ_DATATYPE", "mixed_bitmod")
WQ_GROUPSIZE = int(os.environ.get("WQ_GROUPSIZE", "128"))
_SKIP = {"gate", "lm_head"}          # house surface, = ptq._SKIP_LINEAR

torch.set_grad_enabled(False)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    device_map="auto", attn_implementation="eager")
n_q = n_skip = 0
for name, mod in model.named_modules():
    for attr, child in list(mod.named_children()):
        if not isinstance(child, torch.nn.Linear):
            continue
        if attr in _SKIP:
            n_skip += 1
            continue
        w = child.weight.data
        if WQ_DATATYPE.startswith("mixed"):
            child.weight.data = search_datatype(
                w, wq_bits=WQ_BITS, datatype=WQ_DATATYPE, group_size=WQ_GROUPSIZE)
        elif WQ_DATATYPE.startswith("int") and "asym" in WQ_DATATYPE:
            child.weight.data = quant_int_asym(
                w, wq_bits=WQ_BITS, group_size=WQ_GROUPSIZE)
        elif WQ_DATATYPE.startswith("int"):
            child.weight.data = quant_int(
                w, wq_bits=WQ_BITS, group_size=WQ_GROUPSIZE)
        else:
            child.weight.data = quant_datatype(
                w, wq_bits=WQ_BITS, datatype=WQ_DATATYPE, group_size=WQ_GROUPSIZE)
        n_q += 1
print(f"[crosscheck] quantized {n_q} Linear layers "
      f"(skipped {n_skip}: {sorted(_SKIP)}) with reference "
      f"{WQ_DATATYPE} {WQ_BITS}b g{WQ_GROUPSIZE}")
model.seqlen = 2048
model = model.eval()

# ---- reference PPL loop (llm_eval_wikitext.py:42-65, fast tokenizer) --------
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
testenc = testenc.input_ids.to(model.device)
nsamples = testenc.numel() // model.seqlen
loss_fct = torch.nn.CrossEntropyLoss()

nlls = []
for i in range(nsamples):
    batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
    lm_logits = model(batch).logits
    shift_logits = lm_logits[:, :-1, :].contiguous().float()
    shift_labels = batch[:, 1:]
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    nlls.append(loss.float() * model.seqlen)
    if (i + 1) % 20 == 0:
        run = torch.exp(torch.stack(nlls).sum() / ((i + 1) * model.seqlen))
        print(f"[crosscheck] window {i + 1}/{nsamples} running_ppl={run.item():.4f}",
              flush=True)

ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
print(f"[CROSSCHECK] model={MODEL} datatype={WQ_DATATYPE} bits={WQ_BITS} "
      f"g={WQ_GROUPSIZE} windows={nsamples} ppl={ppl.item():.6f}")
