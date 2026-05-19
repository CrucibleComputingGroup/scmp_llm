"""Deterministic per-row INT8 PPL — same quant scheme as SC bipolar, no SC.

Monkey-patches ``scmp_kernels.sc_matmul`` with a per-row symmetric int8
fake-quant matmul (round-to-nearest), then runs the same PPL loop as
``ppl.py``. Used to check whether SC at large ``stoc_len`` is hitting the
deterministic INT8 noise floor.
"""
import math
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Monkey-patch BEFORE importing model loader, so SCLinear / sc_attention pick
# up the patched function via `from scmp_kernels import sc_matmul`.
import scmp_kernels  # noqa: E402


@torch.no_grad()
def _fake_int8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    granularity: str = "per_row",
    *,
    mode: str = "bipolar",
    sc_prec: int = 8,
    stoc_len=None,
    chunk_d: int = 0,
    group_a: int = 1,
    group_b: int = 1,
    rng_levels=None,
    config=None,
) -> torch.Tensor:
    """Per-row symmetric INT8 fake-quant `a @ b.T` in fp32.

    Matches SC bipolar/per-row quantization scheme (q_max = 2^(sc_prec-1) - 1,
    per-row abs-max scale) but uses deterministic round-to-nearest instead
    of stochastic sampling. group_a/group_b kept for API parity (group=1 =
    true per-row).
    """
    q_max = 2 ** (sc_prec - 1) - 1
    af = a.to(torch.float32)
    bf = b.to(torch.float32)
    a_abs_max = af.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    b_abs_max = bf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale_a = a_abs_max / q_max
    scale_b = b_abs_max / q_max
    a_int = (af / scale_a).round().clamp(-q_max, q_max)
    b_int = (bf / scale_b).round().clamp(-q_max, q_max)
    a_dq = a_int * scale_a
    b_dq = b_int * scale_b
    return (a_dq @ b_dq.transpose(-2, -1)).to(torch.float32)


scmp_kernels.sc_matmul = _fake_int8_matmul
# Patch the already-imported alias too (model imports `from scmp_kernels import sc_matmul`).
import scmp_kernels.sc.matmul as _sc_mm_mod  # noqa: E402
_sc_mm_mod.sc_matmul = _fake_int8_matmul

from loader import load_sc_model  # noqa: E402

# Re-patch in case loader (or model.sc_common) cached the original at import.
import model.sc_common as _sc_common  # noqa: E402
_sc_common._sc_matmul = _fake_int8_matmul
try:
    import model.llama_sc_legacy as _llsc_legacy  # noqa: E402
    _llsc_legacy._sc_matmul = _fake_int8_matmul
except Exception:
    pass

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
PPL_DATASET = os.environ.get("PPL_DATASET", "wikitext")
PPL_DATASET_CONFIG = os.environ.get("PPL_DATASET_CONFIG", "wikitext-2-raw-v1")
PPL_SPLIT = os.environ.get("PPL_SPLIT", "test")
PPL_MAX_TOKENS = int(os.environ.get("PPL_MAX_TOKENS", "0"))
CTX = int(os.environ.get("CTX", "1024"))
STRIDE = int(os.environ.get("STRIDE", str(CTX // 2)))
SC_PREC = int(os.environ.get("SC_PREC", "8"))
SC_ATTN_GRANULARITY = os.environ.get("SC_ATTN_GRANULARITY", "per_row")


def compute_ppl(model, tokenizer, enc_ids: torch.Tensor):
    device = model.device
    total = enc_ids.shape[0]
    sum_loss = 0.0
    n_loss = 0
    prev_end = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for start in range(0, total - 1, STRIDE):
        end = min(start + CTX, total)
        if end - start < 2:
            break
        input_ids = enc_ids[start:end].unsqueeze(0).to(device)
        labels = input_ids.clone()
        overlap = max(0, prev_end - start)
        if overlap > 0:
            labels[:, :overlap] = -100
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
        valid = (labels[..., 1:] != -100).sum().item()
        if valid > 0:
            sum_loss += float(out.loss) * valid
            n_loss += valid
        prev_end = end
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    secs = time.time() - t0
    ppl = math.exp(sum_loss / n_loss) if n_loss else float("inf")
    return ppl, n_loss, secs


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = load_sc_model(MODEL_PATH, dtype=torch.float16, device_map="auto")
    model.eval()

    ds = load_dataset(PPL_DATASET, PPL_DATASET_CONFIG, split=PPL_SPLIT)
    text = "\n\n".join(d["text"] for d in ds if d.get("text", "").strip())
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    total_tokens = enc.shape[0]
    if PPL_MAX_TOKENS > 0:
        enc = enc[:PPL_MAX_TOKENS]
    eval_tokens = enc.shape[0]
    print(f"model={MODEL_PATH}")
    print(f"data={PPL_DATASET}/{PPL_DATASET_CONFIG}:{PPL_SPLIT} "
          f"— {eval_tokens} of {total_tokens} tokens, ctx={CTX} stride={STRIDE}")
    print(f"backend=DETERMINISTIC INT8 fake-quant (per_row symmetric, sc_prec={SC_PREC})")
    print()

    # FP16 reference
    model.config.use_sc_attn = False
    model.config.use_sc_linear = False
    ppl_fp16, n, secs = compute_ppl(model, tokenizer, enc)
    n_win = max(1, eval_tokens // STRIDE)
    print(f"{'config':<32}  {'PPL':>10}  {'tokens':>8}  {'sec':>7}  {'ms/win':>8}")
    print("-" * 80)
    print(f"{'FP16 baseline':<32}  {ppl_fp16:10.4f}  {n:8d}  {secs:7.1f}  "
          f"{secs * 1000 / n_win:8.0f}")

    # INT8 (via monkey-patched sc_matmul — sc_stoc_len has no effect here)
    model.config.use_sc_attn = True
    model.config.use_sc_linear = True
    model.config.sc_prec = SC_PREC
    model.config.sc_stoc_len = 256  # ignored
    model.config.sc_granularity = SC_ATTN_GRANULARITY
    model.config.sc_linear_granularity = "per_row"
    ppl_int8, n, secs = compute_ppl(model, tokenizer, enc)
    rel = ppl_int8 / ppl_fp16
    name = f"INT8 per_row gran={SC_ATTN_GRANULARITY}"
    print(f"{name:<32}  {ppl_int8:10.4f}  {n:8d}  {secs:7.1f}  "
          f"{secs * 1000 / n_win:8.0f}  (×{rel:.3f} vs fp16)")


if __name__ == "__main__":
    main()
