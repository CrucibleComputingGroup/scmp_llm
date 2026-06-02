"""Calibrate SmoothQuant activation scales for an SC model.

Loads the model, runs ``N_SAMPLES`` short sequences from a calibration
dataset (wikitext-2 train by default) with SC disabled, accumulates
per-channel max-abs of every ``SCLinear`` input, and saves the dict
to ``OUTPUT``.

Env vars:
    MODEL_PATH       — HF id (default: Qwen3-4B-Instruct-2507)
    CALIB_DATASET    — HF dataset (default: wikitext)
    CALIB_CONFIG     — dataset config (default: wikitext-2-raw-v1)
    CALIB_SPLIT      — split (default: train)
    N_SAMPLES        — number of sequences to feed (default: 128)
    SEQ_LEN          — tokens per sequence (default: 512)
    OUTPUT           — path to write act_scales dict (default:
                       benchmark/ppl/act_scales_<safe-model-name>.pt)
"""
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO not in sys.path:
    sys.path.append(REPO)

from loader import load_sc_model  # noqa: E402
from model.smoothquant_apply import calibrate_act_scales  # noqa: E402


MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507")
CALIB_DATASET = os.environ.get("CALIB_DATASET", "wikitext")
CALIB_CONFIG = os.environ.get("CALIB_CONFIG", "wikitext-2-raw-v1")
CALIB_SPLIT = os.environ.get("CALIB_SPLIT", "train")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "128"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))

_default_out = os.path.join(
    os.path.dirname(__file__),
    f"act_scales_{MODEL_PATH.replace('/', '_')}.pt",
)
OUTPUT = os.environ.get("OUTPUT", _default_out)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"loading {MODEL_PATH} ...", flush=True)
    t0 = time.time()
    model = load_sc_model(MODEL_PATH, dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    ds = load_dataset(CALIB_DATASET, CALIB_CONFIG, split=CALIB_SPLIT)
    text = "\n\n".join(d["text"] for d in ds if d.get("text", "").strip())
    enc = tok(text, return_tensors="pt").input_ids[0]
    total = enc.shape[0]
    print(f"calib stream: {total} tokens (using {N_SAMPLES} × {SEQ_LEN}-tok windows)", flush=True)

    # Disjoint windows striding through the corpus.
    stride = max(1, (total - SEQ_LEN) // max(1, N_SAMPLES - 1))
    samples = []
    for i in range(N_SAMPLES):
        start = i * stride
        end = start + SEQ_LEN
        if end > total:
            break
        samples.append(enc[start:end].unsqueeze(0))
    print(f"prepared {len(samples)} windows, stride={stride}", flush=True)

    t0 = time.time()
    act_scales = calibrate_act_scales(model, samples)
    secs = time.time() - t0

    # Move to CPU before saving to keep the file device-agnostic.
    act_scales_cpu = {k: v.detach().cpu() for k, v in act_scales.items()}
    print(f"calibrated {len(act_scales_cpu)} layers in {secs:.1f}s", flush=True)

    # Sanity printout: max channel scale per layer-name prefix.
    by_prefix: dict[str, float] = {}
    for name, s in act_scales_cpu.items():
        prefix = name.split(".")[-1]  # e.g. q_proj
        by_prefix[prefix] = max(by_prefix.get(prefix, 0.0), s.max().item())
    print("per-projection max channel max-abs across all layers:")
    for k, v in sorted(by_prefix.items()):
        print(f"  {k:<12} {v:.3f}")

    torch.save(act_scales_cpu, OUTPUT)
    print(f"saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
