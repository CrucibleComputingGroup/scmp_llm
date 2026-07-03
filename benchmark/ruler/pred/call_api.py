"""RULER prediction entrypoint for local Hugging Face models.

Two modes:
* ``--mode fp16`` — vanilla eager attention, no SC.
* ``--mode sc``   — full SC matmul (attention + linear) via
  ``loader.load_sc_model`` (Llama/Qwen adapters) with ``--sc_prec`` and
  ``--sc_stoc_len``.

Outputs jsonl with field ``pred`` per sample, mirroring upstream:

    {"index": int, "input": str, "outputs": [str], "pred": str, ...}
"""
# ruff: noqa: E402  # allow sys.path modification before imports

import argparse
import importlib
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import load_data

from benchmark.config import parse_sc_args
from loader import load_sc_model


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class HuggingFaceModel:
    def __init__(
        self,
        model_name: str,
        max_new_len: int,
        mode: str,
        sc_prec: int,
        sc_stoc_len: int,
        sc_attn_granularity: str,
        dtype: torch.dtype,
        device: str,
        quant_config: str = "fp16",
    ) -> None:
        self.device = device
        self.max_new_len = max_new_len
        self.mode = mode

        if mode == "quant":
            # Integer PTQ baseline: plain HF + SmoothQuant + fake-quant (no SC).
            from benchmark.quant.eval_quant import build_model
            llm, tokenizer = build_model(model_name, quant_config, device_map=device)
            llm.eval()
            self.llm = llm
            self.tokenizer = tokenizer
            return

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm = load_sc_model(model_name, dtype=dtype, device_map=device)
        if mode == "fp16":
            llm.config.use_sc_attn = False
            llm.config.use_sc_linear = False
        elif mode == "sc_linear":
            llm.config.use_sc_attn = False
            llm.config.use_sc_linear = True
            llm.config.sc_prec = sc_prec
            llm.config.sc_stoc_len = sc_stoc_len
        else:  # "sc"
            llm.config.use_sc_attn = True
            llm.config.use_sc_linear = True
            llm.config.sc_prec = sc_prec
            llm.config.sc_stoc_len = sc_stoc_len
        llm.config.sc_granularity = sc_attn_granularity
        llm.eval()

        self.llm = llm
        self.tokenizer = tokenizer

    def __call__(self, prompt: str, **kwargs: object) -> dict[str, Any]:
        torch.cuda.set_device(self.device)
        generated_text = get_pred(
            self.llm, self.tokenizer,
            input_text=prompt,
            max_new_tokens=self.max_new_len,
        )
        return {"text": [generated_text]}


def get_pred(
    llm: Any, tokenizer: Any, input_text: str, max_new_tokens: int,
) -> str:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_inputs = tokenizer(
        [input_text], return_tensors="pt", padding=True).to(llm.device)

    with torch.no_grad():
        generated_ids = llm.generate(
            **model_inputs, max_new_tokens=max_new_tokens, do_sample=False)

    output_ids = generated_ids[:, len(model_inputs.input_ids[0]):].tolist()
    output: list[str] = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)
    print("Generated output:", output[0])
    return output[0]


def get_output(
    llm: HuggingFaceModel,
    outputs_parallel: dict[int, dict[str, object]],
    idx: int,
    index: int,
    input: str,
    outputs: list[str],
    others: dict[str, object],
    truncation: bool,
    length: int,
) -> None:
    pred = llm(prompt=input)
    if len(pred["text"]) > 0:
        outputs_parallel[idx] = {
            "index": index,
            "pred": pred["text"][0],
            "input": input,
            "outputs": outputs,
            "others": others,
            "truncation": truncation,
            "length": length,
        }


def mode_tag(args: argparse.Namespace) -> str:
    if args.mode == "fp16":
        return "fp16"
    if args.mode == "quant":
        return f"quant_{args.quant_config}"
    if args.mode == "sc_linear":
        return f"sc_linear_prec{args.sc_prec}_stoc{args.sc_stoc_len}"
    return f"sc_prec{args.sc_prec}_stoc{args.sc_stoc_len}"


def main(args: argparse.Namespace) -> None:
    start_time = time.time()

    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")
        return

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml")) as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config["task"]])

    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"

    save_dir = args.save_dir
    if args.chunk_amount > 1:
        pred_file = save_dir / f"{args.task}-{args.chunk_idx}.jsonl"
    else:
        pred_file = save_dir / f"{args.task}.jsonl"

    print(f"Predict {args.task} \nfrom {task_file}\nto {pred_file}")
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(pred_file):
        pred_index = [sample["index"] for sample in load_data(pred_file)]
        data = [
            sample for sample in load_data(task_file)
            if sample["index"] not in pred_index
        ]
    else:
        data = load_data(task_file)

    torch.cuda.set_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    llm = HuggingFaceModel(
        model_name=args.model_name,
        max_new_len=config["tokens_to_generate"],
        mode=args.mode,
        sc_prec=args.sc_prec,
        sc_stoc_len=args.sc_stoc_len,
        sc_attn_granularity=args.sc_attn_granularity,
        dtype=dtype,
        device=args.device,
        quant_config=args.quant_config,
    )

    threads: list[threading.Thread] = []
    outputs_parallel: dict[int, dict[str, object]] = {}
    idx = -1
    with open(pred_file, "a", encoding="utf-8", buffering=1) as fout:
        torch.cuda.set_device(args.device)
        for idx, data_point in tqdm(enumerate(data), total=len(data)):
            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    llm=llm,
                    outputs_parallel=outputs_parallel,
                    idx=idx,
                    index=data_point["index"],
                    input=data_point["input"],
                    outputs=data_point["outputs"],
                    others=data_point.get("others", {}),
                    truncation=data_point.get("truncation", -1),
                    length=data_point.get("length", -1),
                ),
            )
            thread.start()
            threads.append(thread)
            if len(threads) == args.threads:
                for t in threads:
                    t.join()
                threads = []
                for computed_idx in range(idx - args.threads + 1, idx + 1):
                    if computed_idx in outputs_parallel and len(outputs_parallel[computed_idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[computed_idx]) + "\n")

        if len(data) > 0:
            for t in threads:
                t.join()
            for computed_idx in range(idx - len(threads) + 1, idx + 1):
                if computed_idx in outputs_parallel and len(outputs_parallel[computed_idx]) > 0:
                    fout.write(json.dumps(outputs_parallel[computed_idx]) + "\n")

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, default="synthetic")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--subset", type=str, default="validation")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--chunk_amount", type=int, default=1)

    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model id (must be a Llama-family checkpoint).")
    parser.add_argument("--max_len", type=int, default=128000,
                        help="Kept for shell-wrapper compatibility; not consumed directly.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument(
        "--threads", type=int, default=1,
        help="Parallel data-prep thread count. The model itself runs single-threaded on the GPU.")

    # Accept upstream args we don't actually consume (shell-wrapper compat).
    parser.add_argument("--server_type", type=str, default="hf")
    parser.add_argument("--server_host", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=str, default="5000")
    parser.add_argument("--ssh_server", type=str)
    parser.add_argument("--ssh_key_path", type=str)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--sliding_window_size", type=int)
    parser.add_argument("--synthetic_len", type=int, default=0)

    parser = parse_sc_args(parser)
    args = parser.parse_args()

    if args.server_type != "hf":
        raise RuntimeError(
            f"Only --server_type hf is supported in this fork (got '{args.server_type}').")

    print(args)
    seed_everything(2025)
    main(args)
