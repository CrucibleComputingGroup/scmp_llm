"""Does matching the SC enable-grid to the stream length reduce error?

MOTIVATION. `sc_matmul` defaults `rng_levels = 2**(sc_prec-1) = 128` regardless
of `stoc_len`, and `model/sc_common.py` never overrides it. So a group running
at the ladder floor represents a 128-level grid with ~18 stochastic samples --
that residual is SAMPLING NOISE, not quantization error. On 4B t32, 44.1% of all
MACs sit at that floor, so if the grid is mispriced there it is mispriced for
nearly half the network.

A stream of L cycles can resolve ~L distinct magnitudes. A grid much finer than
L cannot be represented and only adds variance; a grid much coarser wastes
resolution. This sweeps the two axes against real activations to find whether an
L-matched grid wins, and by how much.

Runtime-free if it works: `rng_levels` is already a per-call kernel argument, so
this is a calibration/dispatch choice, not new hardware. It also makes the
precision space per-group RICHER (stream length x grid), which is the thesis
rather than a workaround.

  KB_MODEL=4B python -m benchmark.ppl.kbands.probe_rng_grid
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

from scmp_kernels import sc_matmul                      # noqa: E402

RES = Path("/home/allenjin/Projects/SCMP/hpca_results/llm")
MODEL = os.environ.get("KB_MODEL", "4B")
TARGET = os.environ.get("KB_TARGET", "32")
SC_PREC = 8
CHUNK_D = 128
# Ladder rungs that actually carry MAC mass at t32, plus the floor.
LENS = [int(v) for v in os.environ.get("KB_LENS", "16,18,24,32,48,64,96").split(",")]


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parent = RES / "ppl" / "mp_best" / "configs" / MODEL / f"target{TARGET}"
    hf = json.load(open(parent / "table.json"))["model_path"]
    print(f"[grid] model {hf}")
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()

    # Capture one real activation per probed projection.
    grabbed: dict = {}
    probes = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj"]
    hooks = []

    def mk(name):
        def _h(mod, inp, _out):
            if name not in grabbed:
                grabbed[name] = (inp[0].detach().reshape(-1, inp[0].shape[-1]),
                                 mod.weight.detach())
        return _h

    layers = model.model.layers
    mid = len(layers) // 2
    for n, m in layers[mid].named_modules():
        base = n.split(".")[-1]
        if base in probes and hasattr(m, "weight"):
            hooks.append(m.register_forward_hook(mk(base)))

    text = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tok(text, return_tensors="pt").input_ids[:, :2048].to("cuda:0")
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()

    print(f"[grid] captured {sorted(grabbed)} from block {mid}\n")
    rows = []
    for name, (x, W) in sorted(grabbed.items()):
        x = x[:512].float()
        Wf = W.float()
        teacher = F.linear(x, Wf)
        print(f"=== {name}  x{tuple(x.shape)} W{tuple(Wf.shape)} ===")
        print(f"{'L':>5s} {'grid=128 (deployed)':>20s} {'grid=L':>10s} "
              f"{'grid=p2<=L':>10s} {'best':>10s} {'gain vs deployed':>17s}")
        for L in LENS:
            out = {}
            # pow2floor: the Owen/bit-reversal scramble is inherently a
            # power-of-two construction (mask = bit_reverse(d mod M), M a power
            # of 2), so a non-pow2 grid breaks its structure. The first sweep
            # showed grid=L winning ONLY at L in {16,32,64} -- exactly the
            # powers of two -- and doing nothing at 18/24/48/96. That predicts
            # the real lever is the largest power of two <= L.
            p2 = 1 << (L.bit_length() - 1)
            for tag, grid in (("128", 128), ("L", L), ("p2<=L", p2)):
                with torch.no_grad():
                    sc = sc_matmul(x, Wf, sc_prec=SC_PREC, stoc_len=L,
                                   rng_levels=grid, mode="bipolar",
                                   granularity="per_row", chunk_d=CHUNK_D,
                                   halve_bipolar_stoc_len=True)
                out[tag] = rel_l2(sc, teacher)
            best_tag = min(out, key=out.get)
            gain = (out[best_tag] - out["128"]) / out["128"] * 100.0
            rows.append({"op": name, "L": L, **out, "best": best_tag, "gain_pct": gain})
            print(f"{L:5d} {out['128']:20.5f} {out['L']:10.5f} {out['p2<=L']:10.5f} "
                  f"{best_tag:>10s} {gain:16.2f}%")
        print()

    outp = Path(os.environ.get(
        "KB_OUT", f"/nfs/turbo/coe-nbleier/allenjin/hpca/gridprobe/{MODEL}_t{TARGET}.json"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(outp, "w"), indent=1)
    print(f"[grid] wrote {outp}")

    short = [r for r in rows if r["L"] <= 32]
    if short:
        avg = sum(r["gain_pct"] for r in short) / len(short)
        print(f"[grid] MEAN error change at L<=32 (the floor regime, 44% of "
              f"4B t32 MACs): {avg:+.2f}%  (negative = grid matching WINS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
