"""Perplexity sweep on a held-out text dataset.

Loads the model once, computes FP16 reference PPL, then sweeps SC at the
stoc_lens in ``STOC_LENS``. Window-based: concatenate all text, slide a
``CTX``-length window with stride ``STRIDE``, sum the cross-entropy on the
new (non-overlapping) tokens, and report ``exp(sum_loss / sum_tokens)``.

Auto-dispatches llama vs qwen via ``MODEL_PATH``.

Env vars:

    MODEL_PATH       — HF id (default: llama-3.1-8b)
    PPL_DATASET      — HF dataset name (default: ``wikitext``)
    PPL_DATASET_CONFIG
                     — HF dataset config (default: ``wikitext-2-raw-v1``)
    PPL_SPLIT        — split (default: ``test``)
    PPL_MAX_TOKENS   — cap on tokens evaluated (0 = all; default 65536 for
                       a quick first run, override to 0 for the full set)
    CTX              — context window (default 1024)
    STRIDE           — window stride (default = CTX, i.e. non-overlapping)
    SC_PREC          — SC precision (default 8)
    STOC_LENS        — comma-separated SC sweep (default 256,128,64)
    SC_ATTN_GRANULARITY
                     — per_head | per_row (applied to whole sweep)
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
from loader import load_sc_model, apply_mp_config_from_env  # noqa: E402
from model.smoothquant_apply import apply_smoothquant_from_env  # noqa: E402
from model.sc_common import mp_tracker_reset, mp_tracker_avg_stoc_len  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
PPL_DATASET = os.environ.get("PPL_DATASET", "wikitext")
PPL_DATASET_CONFIG = os.environ.get("PPL_DATASET_CONFIG", "wikitext-2-raw-v1")
PPL_SPLIT = os.environ.get("PPL_SPLIT", "test")
PPL_MAX_TOKENS = int(os.environ.get("PPL_MAX_TOKENS", "65536"))
CTX = int(os.environ.get("CTX", "1024"))
STRIDE = int(os.environ.get("STRIDE", str(CTX)))
SC_PREC = int(os.environ.get("SC_PREC", "8"))
STOC_LENS = [int(x) for x in os.environ.get("STOC_LENS", "256,128,64").split(",") if x]
SC_ATTN_GRANULARITY = os.environ.get("SC_ATTN_GRANULARITY", "per_head")
SKIP_FP16 = os.environ.get("SKIP_FP16", "0") == "1"
FP16_REF = float(os.environ.get("FP16_REF", "0"))
SC_HALVE = os.environ.get("SC_HALVE_BIPOLAR_STOC_LEN", "0") == "1"


def compute_ppl(model, tokenizer, enc_ids: torch.Tensor) -> tuple[float, int, float]:
    """Window-slide PPL over a 1-D token id tensor. Returns (ppl, n_tokens, secs)."""
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
        # On overlapping windows, mask tokens we've already scored to avoid
        # double-counting. (No-op when STRIDE == CTX.)
        overlap = max(0, prev_end - start)
        if overlap > 0:
            labels[:, :overlap] = -100

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)

        # HF computes loss as mean cross-entropy over shifted, unmasked positions.
        # Recover the per-window total loss to accumulate correctly.
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
    apply_smoothquant_from_env(model)
    apply_mp_config_from_env(model)
    mp_cfg = getattr(model.config, "sc_mp_config", None)

    # Build a flat 1-D token-id stream.
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
    print()

    print(f"{'config':<28}  {'PPL':>10}  {'tokens':>8}  {'sec':>7}  {'ms/win':>8}")
    print("-" * 76)
    n_win = max(1, eval_tokens // STRIDE)
    if SKIP_FP16:
        if FP16_REF <= 0:
            raise SystemExit("SKIP_FP16=1 requires FP16_REF=<value> to compute ×fp16 ratios")
        ppl_fp16 = FP16_REF
        print(f"{'FP16 baseline (ref)':<28}  {ppl_fp16:10.4f}  {'-':>8}  {'-':>7}  {'-':>8}")
    else:
        # FP16 reference
        model.config.use_sc_attn = False
        model.config.use_sc_linear = False
        ppl_fp16, n, secs = compute_ppl(model, tokenizer, enc)
        print(f"{'FP16 baseline':<28}  {ppl_fp16:10.4f}  {n:8d}  {secs:7.1f}  "
              f"{secs * 1000 / n_win:8.0f}")

    # SC sweep
    for stoc_len in STOC_LENS:
        model.config.use_sc_attn = True
        model.config.use_sc_linear = True
        model.config.sc_prec = SC_PREC
        model.config.sc_stoc_len = stoc_len
        model.config.sc_granularity = SC_ATTN_GRANULARITY
        model.config.sc_halve_bipolar_stoc_len = SC_HALVE
        mp_tracker_reset()
        eff_sl = (2 ** (SC_PREC - 1)) if SC_HALVE else stoc_len
        ppl_sc, n, secs = compute_ppl(model, tokenizer, enc)
        halved_tag = " (halved)" if SC_HALVE else ""
        if mp_cfg is not None:
            avg_sl = mp_tracker_avg_stoc_len()
            name = (f"SC MP levels={mp_cfg.stoc_len_levels} "
                    f"fr={mp_cfg.level_fractions} avg_sl={avg_sl:.1f}"
                    f"{halved_tag}")
        else:
            name = f"SC sl={eff_sl}{halved_tag} gran={SC_ATTN_GRANULARITY}"
        rel = ppl_sc / ppl_fp16
        print(f"{name:<28}  {ppl_sc:10.4f}  {n:8d}  {secs:7.1f}  "
              f"{secs * 1000 / n_win:8.0f}  (×{rel:.3f} vs fp16)")


if __name__ == "__main__":
    main()
