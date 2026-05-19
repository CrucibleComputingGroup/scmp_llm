"""LongBench prediction for the SC-llama fork.

Two modes:
* ``--mode fp16`` — vanilla eager attention, no SC.
* ``--mode sc``   — full SC matmul (attention + linear) via
  ``model.llama_sc.LlamaForCausalLM``. Quality knobs:
  ``--sc_prec`` (default 8) and ``--sc_stoc_len`` (default 256).

Outputs land in ``results/pred/<model>/<tag>/<task>.jsonl`` (or
``results/pred_e/...`` with ``--e``), where ``<tag>`` is ``fp16``
or ``sc_prec{P}_stoc{L}``.
"""
import argparse
import json
import os
import random
import sys
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
from transformers import AutoTokenizer  # noqa: E402

from benchmark.config import parse_sc_args  # noqa: E402
from loader import load_sc_model  # noqa: E402

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")

with open(os.path.join(CONFIG_DIR, "model2path.json")) as f:
    model2path = json.load(f)
with open(os.path.join(CONFIG_DIR, "model2maxlen.json")) as f:
    model2maxlen = json.load(f)
with open(os.path.join(CONFIG_DIR, "dataset2prompt.json")) as f:
    dataset2prompt = json.load(f)
with open(os.path.join(CONFIG_DIR, "dataset2maxlen.json")) as f:
    dataset2maxlen = json.load(f)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="llama-3.1-8b",
        choices=sorted(model2path.keys()))
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument(
        "--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument(
        "--task", type=str, required=True,
        help="LongBench task name (e.g. qasper, hotpotqa, ...).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--num_examples", type=int, default=-1,
        help="Cap number of examples per task. -1 = all.")
    parser = parse_sc_args(parser)
    return parser.parse_args(args)


def mode_tag(args: argparse.Namespace) -> str:
    if args.mode == "fp16":
        return "fp16"
    return f"sc_prec{args.sc_prec}_stoc{args.sc_stoc_len}"


def normalize_device_map(device: str) -> Any:
    if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device
    return {"": device}


def load_model(
    model_path: str, dtype: torch.dtype, device: str, args: argparse.Namespace,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device_map = normalize_device_map(device)
    llm = load_sc_model(model_path, dtype=dtype, device_map=device_map)
    if args.mode == "fp16":
        llm.config.use_sc_attn = False
        llm.config.use_sc_linear = False
    else:
        llm.config.use_sc_attn = True
        llm.config.use_sc_linear = True
        llm.config.sc_prec = args.sc_prec
        llm.config.sc_stoc_len = args.sc_stoc_len
    llm.eval()
    return llm, tokenizer


def get_pred(
    llm: Any, tokenizer: Any, data: list[dict[str, object]],
    max_new_tokens: int, prompt_format: str, out_path: str,
) -> None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        model_inputs = tokenizer(
            [prompt], return_tensors="pt", padding=True).to(llm.device)
        print("Input length:", model_inputs.input_ids.shape[1])

        with torch.no_grad():
            generated_ids = llm.generate(
                **model_inputs, max_new_tokens=max_new_tokens, do_sample=False)

        output_ids = generated_ids[:, len(model_inputs.input_ids[0]):].tolist()
        output: list[str] = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        torch.cuda.empty_cache()
        print("Generated output:", output[0][:50])

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": output[0],
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                },
                f, ensure_ascii=False,
            )
            f.write("\n")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model_name = args.model
    model_path = model2path[model_name]
    tag = mode_tag(args)

    llm, tokenizer = load_model(model_path, dtype, args.device, args)

    if args.e:
        datasets = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
            "gov_report", "multi_news", "trec", "triviaqa", "samsum",
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
        ]
        base = f"results/pred_e/{model_name}/{tag}"
    else:
        datasets = [args.task]
        base = f"results/pred/{model_name}/{tag}"

    os.makedirs(base, exist_ok=True)

    for dataset in datasets:
        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
        else:
            data = load_dataset("THUDM/LongBench", dataset, split="test")

        out_path = f"{base}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_new_tokens = dataset2maxlen[dataset]
        data_all = list(data)
        if args.num_examples > 0:
            data_all = data_all[: args.num_examples]

        get_pred(llm, tokenizer, data_all, max_new_tokens, prompt_format, out_path)
