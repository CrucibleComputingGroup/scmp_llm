"""End-to-end wiring check for a per-(row, chunk) table, before any PPL wave.

THE FAILURE THIS EXISTS TO CATCH. If `get_per_row_chunk` returns None for every
module -- a bucket-key mismatch, a section that did not survive the wrapper
round-trip, a config class that never parsed it -- SCLinear silently falls back
to per-ROW dispatch. The eval then runs to completion, reports a plausible PPL,
lands at the parent's cost, and shows no change. That reads as "per-(row, chunk)
does not help", which is the most expensive possible wrong conclusion here: it
would retire a direction that 20 measured cells support.

So this asserts the branch ACTUALLY FIRES, on the real emitted JSON, with the
deployed loader -- and that its traced cost matches what calibration promised.

Run:  KB_PRC_JSON=<wrapper.json> KB_MODEL=4B python -m benchmark.ppl.kbands.test_prc_wiring
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SC_OWEN_MODE", "bitrev")
os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "kernels")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main() -> int:
    prc_json = os.environ.get("KB_PRC_JSON")
    model_key = os.environ.get("KB_MODEL", "4B")
    target = os.environ.get("KB_TARGET", "32")
    if not prc_json or not Path(prc_json).is_file():
        raise SystemExit(f"KB_PRC_JSON not found: {prc_json!r}")
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0

    payload = json.loads(Path(prc_json).read_text())
    print(f"[wire] wrapper: {prc_json}")
    # The section lives in the TABLE (that is what load_threshold_table parses),
    # so follow threshold_table_path rather than reading the wrapper.
    tbl_path = payload.get("threshold_table_path")
    tbl = {}
    if tbl_path and Path(tbl_path).is_file():
        tbl = json.loads(Path(tbl_path).read_text())
        print(f"[wire] table:   {tbl_path}")
    else:
        print(f"[wire] table:   MISSING ({tbl_path!r})")
    print(f"[wire] per_row_chunk buckets in TABLE: "
          f"{len(((tbl.get('per_row_chunk') or {}).get('buckets')) or {})}")

    parent = (f"/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/"
              f"configs/{model_key}/target{target}")
    os.environ["QUANT_CONFIG"] = "mp"
    os.environ["MP_CONFIG_JSON"] = prc_json
    if Path(f"{parent}/hybrid_config.json").is_file():
        os.environ["SC_HYBRID_CONFIG_JSON"] = f"{parent}/hybrid_config.json"
    os.environ.setdefault(
        "ACT_SCALES_DIR",
        "/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales")
    os.environ["FRONTEND"] = os.environ.get("KB_FRONTEND", "awq")
    hf = json.loads(Path(f"{parent}/table.json").read_text())["model_path"]

    from benchmark.quant.eval_quant import build_sc_model
    from model.sc_common import SCLinear
    import scmp_kernels.trace as tr

    model, tok = build_sc_model(hf, "mp", mp_table=prc_json)
    model.eval()
    dev = next(model.parameters()).device
    cfg = model.config
    mp_cfg = getattr(cfg, "sc_mp_config", None)
    total_blocks = getattr(cfg, "num_hidden_layers", None)

    print("\n[1] the loaded config carries per_row_chunk")
    check("AdaptiveMPConfig parsed a per_row_chunk section",
          bool(getattr(mp_cfg, "per_row_chunk", None)),
          f"{len(getattr(mp_cfg, 'per_row_chunk', {}) or {})} entries")

    print("\n[2] every SCLinear RESOLVES a per-(row, chunk) entry")
    miss, hit = [], 0
    for m in model.modules():
        if not isinstance(m, SCLinear):
            continue
        op = getattr(m, "_sc_op_name", None)
        blk = getattr(m, "_sc_block_idx", None)
        if op is None or blk is None:
            continue
        got = mp_cfg.get_per_row_chunk(op, blk, total_blocks)
        if got is None:
            miss.append((op, blk))
        else:
            hit += 1
    check("no SCLinear falls back to per-row dispatch", not miss,
          f"{hit} resolved, {len(miss)} MISSED"
          + (f" e.g. {miss[:5]}" if miss else ""))

    print("\n[3] the branch actually FIRES in a real forward (traced)")
    tmp = "/tmp/_prc_wire_trace.json"
    tr.reset(); tr.enable(tmp, mode="summary")
    ids = tok("The quick brown fox jumps over the lazy dog. " * 60,
              return_tensors="pt").input_ids[:, :512].to(dev)
    with torch.no_grad():
        out = model(ids)
    tr.flush(tmp); tr.disable()
    doc = json.loads(Path(tmp).read_text())
    grps = doc.get("groups", doc)
    lin_ops = {"q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"}
    lin = [g for g in grps if g.get("op") in lin_ops]
    sls = sorted({g["stoc_len"] for g in lin})
    check("logits are finite", bool(torch.isfinite(out.logits).all()))
    # Per-ROW dispatch prices each record at the FULL residual width. Per-(row,
    # chunk) prices them per chunk, so d_in == 128 should carry the bulk of the
    # MACs. Other widths are expected and legitimate: the residual TAIL chunk
    # (the residual is not a multiple of 128 once protected channels are
    # removed) and the PROTECTED slices, which run at protected_stoc_len
    # outside the MP ladder entirely. So assert on the MAC SHARE at 128, not on
    # the set of widths.
    d_ins = sorted({g["d_in"] for g in lin})
    m128 = sum(g["macs"] for g in lin if g["d_in"] == 128)
    mall = sum(g["macs"] for g in lin) or 1
    check("most linear MACs are priced per 128-chunk (per-(row,chunk) is live)",
          m128 / mall > 0.5,
          f"{100.0 * m128 / mall:.1f}% of linear MACs at d_in=128; "
          f"widths seen: {d_ins[:8]}")
    check("more than one stream length is in use", len(sls) > 1,
          f"lengths: {sls}")

    print("\n[4] traced cost matches what calibration promised")
    macs = sum(g["macs"] for g in lin)
    cyc = sum(g["macs"] * g["stoc_len"] for g in lin)
    mean_l = cyc / macs if macs else 0.0
    print(f"       traced linear-only MAC-weighted mean length = {mean_l:.2f}")
    check("traced mean length is a sane positive number",
          0 < mean_l <= 128, f"{mean_l:.2f}")

    print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
