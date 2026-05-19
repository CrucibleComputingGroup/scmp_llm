import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from metrics import (  # noqa: E402
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    with open(os.path.join(CONFIG_DIR, "model2path.json")) as f:
        model_choices = sorted(json.load(f).keys())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="llama-3.1-8b", choices=model_choices)
    parser.add_argument(
        "--mode", type=str, default="fp16", choices=["fp16", "sc"],
        help="fp16 baseline or full SC matmul.")
    parser.add_argument("--sc_prec", type=int, default=8)
    parser.add_argument("--sc_stoc_len", type=int, default=256)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    return parser.parse_args(args)


def mode_tag(args: argparse.Namespace) -> str:
    if args.mode == "fp16":
        return "fp16"
    return f"sc_prec{args.sc_prec}_stoc{args.sc_stoc_len}"


def scorer_e(
    dataset: str, predictions: list[str], answers: list[list[str]],
    lengths: list[int], all_classes: list[str],
) -> dict[str, float]:
    scores: dict[str, list[float]] = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths, strict=True):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    return {key: round(100 * np.mean(value), 2) for key, value in scores.items()}


def scorer(
    dataset: str, predictions: list[str], answers: list[list[str]],
    all_classes: list[str],
) -> float:
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers, strict=True):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == "__main__":
    args = parse_args()
    tag = mode_tag(args)
    model_name = args.model

    path = (f"results/pred_e/{model_name}/{tag}/" if args.e
            else f"results/pred/{model_name}/{tag}/")

    scores: dict[str, float | dict[str, float]] = {}
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split(".")[0]
        with open(f"{path}{filename}", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        score_result: float | dict[str, float]
        if args.e:
            score_result = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score_result = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score_result

    out_path = f"{path}result.json"

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
