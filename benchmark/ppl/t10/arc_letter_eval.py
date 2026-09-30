"""ARC-Challenge zero-shot letter scoring for rebuttal T10 (reviewer A: reasoning).

The model is built by eval_quant.build_model from the same env as the PPL cells
(QUANT_CONFIG / MP_CONFIG_JSON / SC_HYBRID_CONFIG_JSON / FRONTEND ...), so an MP
row here is the deployed best(all) configuration, unchanged.

One forward pass per question: the prompt lists the lettered choices and ends in
"Answer:"; the prediction is the choice letter with the highest next-token
logit. Batch size is always 1 because SC-MP dispatch min-max normalizes over
every row of a call, so batch-mates would change each other's stream lengths.

Question order is ONE seeded shuffle (ARC_SEED, default 0) shared by every
config, and results stream to a per-question JSONL, so any prefix of the order
is the same fixed random subset for every config. Pre-registered report subset:
the first ARC_REPORT_N (default 500) questions of that order; the full test set
is reported only if every config completes it. A rerun resumes from the JSONL.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from benchmark.quant.eval_quant import build_model  # noqa: E402

try:
    from scmp_kernels import trace as sc_trace
except ImportError:
    sc_trace = None

ARC_FILE = ("/nfs/turbo/coe-nbleier/allenjin/hpca/datasets/arc/"
            "ARC-Challenge-test.parquet")
LETTERS = "ABCDE"


def load_questions(path):
    import pandas as pd
    df = pd.read_parquet(path)
    qs = []
    for r in df.itertuples(index=False):
        labels = [str(x) for x in r.choices["label"]]
        texts = [str(x) for x in r.choices["text"]]
        # 22 questions use numeric labels 1..4; map every label set to A..E by
        # position so the prompt format is identical across questions.
        if str(r.answerKey) not in labels:
            raise SystemExit(f"[arc] {r.id}: answerKey {r.answerKey!r} not in {labels}")
        gold = LETTERS[labels.index(str(r.answerKey))]
        qs.append({"id": str(r.id), "question": str(r.question),
                   "choices": texts, "gold": gold})
    return qs


def prompt_of(q):
    lines = [f"Question: {q['question']}"]
    lines += [f"{LETTERS[i]}. {t}" for i, t in enumerate(q["choices"])]
    lines.append("Answer:")
    return "\n".join(lines)


def letter_ids(tokenizer):
    ids = []
    for c in LETTERS:
        t = tokenizer.encode(" " + c, add_special_tokens=False)
        if len(t) != 1:
            raise SystemExit(f"[arc] ' {c}' is {len(t)} tokens for this tokenizer")
        ids.append(t[0])
    return ids


def main():
    model_path = os.environ["MODEL_PATH"]
    tag = os.environ.get("QUANT_CONFIG", "fp16")
    out_jsonl = os.environ["ARC_OUT"]
    seed = int(os.environ.get("ARC_SEED", "0"))
    report_n = int(os.environ.get("ARC_REPORT_N", "500"))
    qs = load_questions(os.environ.get("ARC_FILE", ARC_FILE))
    order = list(range(len(qs)))
    random.Random(seed).shuffle(order)

    done = {}
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for line in f:
                r = json.loads(line)
                done[r["pos"]] = r
    print(f"[arc] {len(qs)} questions, seed={seed}, resume={len(done)} done")

    model, tokenizer = build_model(model_path, tag,
                                   alpha=float(os.environ.get("SQ_ALPHA", "0.5")))
    lid = letter_ids(tokenizer)
    dev = next(model.parameters()).device
    if sc_trace is not None and sc_trace._ENABLED:
        sc_trace.reset()   # trace covers scoring only, not model build

    t0 = time.time()
    n_new = 0
    with open(out_jsonl, "a") as fout, torch.no_grad():
        for pos, qi in enumerate(order):
            if pos in done:
                continue
            q = qs[qi]
            ids = tokenizer(prompt_of(q), return_tensors="pt").input_ids.to(dev)
            logits = model(input_ids=ids).logits[0, -1].float()
            k = len(q["choices"])
            scores = logits[lid[:k]].tolist()
            pred = LETTERS[max(range(k), key=lambda i: scores[i])]
            rec = {"pos": pos, "id": q["id"], "gold": q["gold"], "pred": pred,
                   "correct": pred == q["gold"], "n_choices": k,
                   "n_tokens": int(ids.shape[1]),
                   "letter_logits": [round(s, 4) for s in scores]}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            done[pos] = rec
            n_new += 1
            if n_new % 25 == 0:
                el = time.time() - t0
                print(f"[arc] {len(done)}/{len(qs)} done, {el / n_new:.2f} s/q, "
                      f"acc so far {sum(r['correct'] for r in done.values()) / len(done):.4f}")

    def acc(n):
        rs = [done[p] for p in range(min(n, len(qs))) if p in done]
        return (sum(r["correct"] for r in rs) / len(rs), len(rs)) if rs else (None, 0)

    a_rep, n_rep = acc(report_n)
    a_full, n_full = acc(len(qs))
    secs = time.time() - t0
    print(f"[RESULT] model={model_path} config={tag} metric=arc_challenge_letter "
          f"acc_first{report_n}={a_rep} n={n_rep} acc_full={a_full} n_full={n_full} "
          f"sec={secs:.1f} s_per_q={secs / max(n_new, 1):.2f}")
    if sc_trace is not None and sc_trace._ENABLED:
        out = sc_trace.flush(header_extra={
            "model": model_path, "config": tag, "task": "arc_challenge_letter",
            "arc_seed": seed, "questions_scored_this_run": n_new,
            "mp_config_json": os.environ.get("MP_CONFIG_JSON", ""),
            "hybrid_config_json": os.environ.get("SC_HYBRID_CONFIG_JSON", ""),
            "frontend": os.environ.get("FRONTEND", "smoothquant"),
            "owen_mode": os.environ.get("SC_OWEN_MODE", "bitrev"),
            "scramble_masks": os.environ.get("SC_SCRAMBLE_MASKS", "64"),
            "sc_prec": 8, "sc_halve": True,
        })
        if out:
            print(f"[trace] wrote {out}")


if __name__ == "__main__":
    main()
