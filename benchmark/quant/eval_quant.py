"""Quant baseline driver: PPL on wikitext-2 + a shared model builder the
accuracy harnesses (LongBench / RULER) import.

Config tags:  fp16 | W8A8_symm | W8A8_asymm | W7A7_symm | ... | W4A4_asymm
              | sc_int8 | sc_avg192 | sc_int7 | sc_avg96 | sc_int6
              | W4A16_symm | W3A16_symm | ...        (weight-only INT, g128, no SQ)
              | W4A16_fp | W3A16_fp                  (BitMoD plain FP4-E2M1 / FP3)
              | W4A16_bitmod | W3A16_bitmod          (BitMoD mixed ER/EA, argmin-MSE)

SmoothQuant smoothing scales depend only on (act_scales, weights, alpha) — NOT
on bit-width — so we calibrate ONE act_scales table per model and reuse it for
every W_xA_x config. FP16 is pure (no smoothing, no quant).

``sc_*`` tags are the SC UNIFORM baselines (see SC_CONFIGS). They run through
the SAME token stream + compute_ppl as the INT cells — the whole point is that
an SC number and an INT number in the results table share one PPL protocol.

Env (PPL mode):
    MODEL_PATH, QUANT_CONFIG (tag), SQ_ALPHA (0.5),
    PPL_MAX_TOKENS (0 = full test set; BitMoD/GPTQ protocol), CTX (2048),
    PPL_WINDOW_BATCH_SIZE (1 = historical path; >1 requires stride=ctx and
    full windows),
    ACT_SCALES_DIR (benchmark/quant/act_scales),
    CALIB_WINDOWS (16), CALIB_CTX (512)
    SC cells also read SC_OWEN_MODE / SC_SCRAMBLE_MASKS (default bitrev / 64).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
from transformers import AutoTokenizer

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from benchmark.quant.ptq import (  # noqa: E402
    QuantConfig, load_plain_model, load_quant_model, apply_ptq,
    calibrate_act_scales, patch_int_attention_once,
)

try:  # precision trace for the energy/latency simulator (SC_MP_TRACE=<path>)
    from scmp_kernels import trace as sc_trace
except ImportError:
    sc_trace = None

ACT_SCALES_DIR = os.environ.get(
    "ACT_SCALES_DIR", os.path.join(_REPO, "benchmark", "quant", "act_scales"))


# ---- SC uniform baseline tags (convention A) --------------------------------
# All five cells run sc_prec=8 + bipolar + halve_bipolar_stoc_len=1, so a value
# below IS the true cycle count (halved space; legacy unhalved stream = 2x).
# The quantization grid stays int8 for every cell — "int7"/"int6" are
# CYCLE-BUDGET labels (the cost of an exact int7/int6 SC unit), not a claim
# that the math runs on an int7/int6 grid: effective resolution is set by the
# stream length; sc_prec only sets the grid and the ceiling.
#   sc_int8   128 cycles — full-length halved stream; the uniform SC ceiling
#   sc_avg192  96 cycles — uniform twin of the MP len192 budget
#   sc_int7    64 cycles — uniform twin of the MP int7 budget
#   sc_avg96   48 cycles — uniform twin of the MP len96 budget
#   sc_int6    32 cycles
SC_CONFIGS = {
    "sc_int8": 128,
    "sc_avg192": 96,
    "sc_int7": 64,
    "sc_avg96": 48,
    "sc_int6": 32,
}

# MP budgets are named by the NOMINAL (pre-halving) convention — the SAME config
# token as the uniform twin — keyed by the calibrated (halved) level set. So an MP
# run and its uniform baseline share the budget token: mp avg192 <-> sc_avg192,
# int7 <-> sc_int7, avg96 <-> sc_avg96. The NAME is nominal (= 2 x halved); the
# levels/cycles in code stay halved. Naming MP by the halved number is WRONG:
# "avg96" for the [128,96,64] / halved-96 run would collide with uniform sc_avg96,
# which is the 2x-smaller nominal-96 (halved-48) budget.
MP_BUDGET_NAMES = {
    (128,): "int8",
    (128, 96, 64): "avg192",
    (128, 64, 32): "int7",
    (64, 48, 32): "avg96",
}


def mp_budget_name(levels) -> str:
    """Pre-halving MP config name (uniform-twin) for a level set."""
    try:
        return MP_BUDGET_NAMES.get(tuple(levels),
                                   "levels" + "-".join(map(str, levels)))
    except TypeError:
        return "levels?"


def parse_config(tag: str):
    """'fp16' -> None (pure fp16). 'W8A8_symm' -> QuantConfig(8,8,True).

    The scheme suffix picks the WEIGHT number format (activations always RTN):
      symm / asymm -> INT RTN (existing baselines)
      fp           -> BitMoD plain FP datatype at w_bits (fp4=E2M1, fp3, ...)
      bitmod       -> BitMoD mixed_bitmod (per-group argmin-MSE over ER/EA)
    With a_bits>=16 any of these is weight-only: activation + attention QK/AV
    fake-quant are bits<16-guarded no-ops, and build_model skips SmoothQuant.
    """
    if tag.lower() in ("fp16", "fp", "baseline"):
        return None
    body, _, scheme = tag.partition("_")
    assert body[0].upper() == "W" and "A" in body, f"bad config tag: {tag}"
    w = int(body[1:body.index("A")])
    a = int(body[body.index("A") + 1:])
    scheme_l = scheme.lower()
    if scheme_l.startswith("sym") or scheme_l.startswith("asym"):
        w_dtype = "int"
    elif scheme_l == "fp":
        w_dtype = f"fp{w}"       # grid resolved/validated by bitmod_dtypes
    elif scheme_l == "bitmod":
        w_dtype = "mixed_bitmod"
    else:
        raise SystemExit(
            f"bad config scheme in tag {tag!r}: expected symm/asymm/fp/bitmod")
    return QuantConfig(
        w_bits=w, a_bits=a, sym=scheme_l.startswith("sym"),
        chunk_size=int(os.environ.get("INT_CHUNK_SIZE", "128")),
        quantize_attention=True, w_dtype=w_dtype,
    )


def _safe(model_path: str) -> str:
    return model_path.replace("/", "_")


def get_act_scales(model_path: str, tokenizer, model, *, recalibrate=False):
    """Load cached per-model SmoothQuant act_scales, else calibrate on wikitext2
    train and cache. Keyed by module name (matches apply_ptq)."""
    os.makedirs(ACT_SCALES_DIR, exist_ok=True)
    path = os.path.join(ACT_SCALES_DIR, f"act_scales_{_safe(model_path)}.pt")
    if os.path.isfile(path) and not recalibrate:
        print(f"[quant] act_scales cache hit: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [d["text"] for d in ds if d.get("text", "").strip()][:2000]
    print(f"[quant] calibrating act_scales ({model_path}) ...")
    scales = calibrate_act_scales(
        model, tokenizer, texts,
        ctx=int(os.environ.get("CALIB_CTX", "512")),
        n_windows=int(os.environ.get("CALIB_WINDOWS", "16")))
    torch.save(scales, path)
    print(f"[quant] wrote {path} ({len(scales)} layers)")
    return scales


def build_sc_model(model_path: str, tag: str, *, device_map="auto",
                   alpha: float = 0.5, mp_table: str = ""):
    """SC model for an ``sc_*`` uniform tag, OR a calibrated per-row MP table.

    When ``mp_table`` (a path to a calibrate_mp_thresholds.py wrapper/table JSON,
    or set via MP_CONFIG_JSON) is given, the model runs per-row mixed-precision
    dispatch (``cfg.sc_mp_config``) instead of a uniform stream length — the SAME
    HPCA protocol (full wikitext-2, ctx 2048, SmoothQuant α) as the uniform sc_*
    and INT baselines, so MP is finally apples-to-apples with them.

    Fairness contract with the INT cells:
      * SAME SmoothQuant act_scales cache (ACT_SCALES_DIR) + same alpha. SC
        applies s at runtime inside sc_matmul ((x/s) @ (s*W)ᵀ via the
        ``smooth_scales`` buffer) — the identical transform apply_ptq folds
        into the weights, applied at a different point.
      * Scrambling pinned to bitrev + 64 masks (the kernel defaults) unless
        the caller exported SC_OWEN_MODE / SC_SCRAMBLE_MASKS themselves.
      * TRUE uniform stream length: ``sc_group_stoclen = {}`` routes every SC
        matmul (linears + Q·Kᵀ + softmax·V) through the explicit-stoc_len
        path with ``sc_stoc_len`` as the per-call cycle count. The plain
        uniform path would silently pin every halved run to 128 cycles
        (sc_common passes stoc_len=None under halve), so it CANNOT express
        the sub-128 cells.
    """
    cycles = SC_CONFIGS.get(tag, 128)   # MP uses 128 as the stream envelope
    os.environ.setdefault("SC_OWEN_MODE", "bitrev")
    os.environ.setdefault("SC_SCRAMBLE_MASKS", "64")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Repo root is already on sys.path (_REPO, inserted above).
    from loader import load_sc_model
    from model.smoothquant_apply import apply_smoothquant_to_model
    from benchmark.quant.ptq import _resolve_device_map
    # Same single-GPU pin as the INT cells: device_map="auto" leaves 30B-MoE
    # weights on `meta`, and the runtime smooth_scales work below (buffer
    # registration against mod.weight) crashes on meta tensors.
    model = load_sc_model(
        model_path, dtype=torch.float16,
        device_map=_resolve_device_map(device_map))
    model.eval()
    # PTQ front-end ablation: FRONTEND=smoothquant (default, byte-identical to
    # the INT cells' scales) | awq (native AWQ, INT-objective scale search; SC
    # stays a pure downstream substrate). Both fold via the same smooth_scales
    # buffer, so only the SCALE VALUE differs.
    frontend = os.environ.get("FRONTEND", "smoothquant").lower()
    if frontend == "awq":
        from model.awq_apply import apply_awq_frontend
        awq_cache = os.environ.get(
            "AWQ_SCALES_DIR", os.path.join(ACT_SCALES_DIR, "awq_scales"))
        n = apply_awq_frontend(model, tokenizer, model_path, cache_dir=awq_cache)
    elif frontend == "smoothquant":
        scales_path = os.path.join(
            ACT_SCALES_DIR, f"act_scales_{_safe(model_path)}.pt")
        if not os.path.isfile(scales_path):
            raise SystemExit(
                f"[sc] missing act_scales cache: {scales_path}\n"
                f"     Run any INT cell for this model first (it calibrates and "
                f"caches) so SC and INT share byte-identical SmoothQuant scales.")
        scales = torch.load(scales_path, map_location="cpu", weights_only=True)
        n = apply_smoothquant_to_model(model, scales, alpha=alpha)
    else:
        raise SystemExit(f"[sc] unknown FRONTEND={frontend!r} (want smoothquant|awq)")
    n_sc = sum(1 for _ in model.modules()
               if type(_).__name__ == "SCLinear") or None
    if n == 0:
        raise SystemExit(
            f"[sc] front-end {frontend!r} matched 0 SCLinear modules — the cell "
            f"would run UNSMOOTHED and be unfair vs INT. Module-name mismatch "
            f"between calibration and the SC model?")
    if n_sc is not None and n < n_sc:
        # Same-key skip as INT's apply_ptq: layers absent from act_scales
        # (e.g. experts that saw zero calib tokens) run unsmoothed in BOTH
        # paths — parity holds, but log it so coverage is auditable.
        print(f"[sc] NOTE: smoothquant covers {n}/{n_sc} SCLinear layers "
              f"(uncovered layers run unsmoothed, same as the INT path)")
    cfg = model.config
    cfg.use_sc_attn = True
    cfg.use_sc_linear = True
    cfg.sc_prec = 8
    cfg.sc_stoc_len = cycles       # halved space: the value IS the cycle count
    cfg.sc_halve_bipolar_stoc_len = True
    from loader import apply_hybrid_config_from_env
    apply_hybrid_config_from_env(model)
    if mp_table:
        # Per-row mixed precision: load the calibrated table into sc_mp_config;
        # sc_common's MP dispatch handles per-row stoc_len (uniform path bypassed).
        os.environ["MP_CONFIG_JSON"] = mp_table
        from loader import apply_mp_config_from_env
        cfg.sc_mp_config = None
        cfg.sc_group_stoclen = None
        apply_mp_config_from_env(model)   # sets cfg.sc_mp_config from MP_CONFIG_JSON
        mp = getattr(cfg, "sc_mp_config", None)
        if mp is None:
            raise SystemExit(f"[sc] MP table did not load: {mp_table}")
        # Feed the runtime tracker the table's per-op MACs/row so it can report
        # the MAC-weighted (iso-compute) realized average alongside the row one.
        # Observe-only; tables without mac_per_row just leave FLOP tracking off.
        try:
            _wr = json.load(open(mp_table))
            # threshold_table_path may be RELATIVE to the wrapper JSON — resolve
            # it against the wrapper's dir exactly as loader.apply_mp_config_from_env
            # does, or the open() below fails and FLOP tracking is silently lost
            # (realized_flop_avg_sl=0.00 recorded as a "successful" result).
            _tbl_rel = _wr.get("threshold_table_path")
            if _tbl_rel and not os.path.isabs(_tbl_rel):
                _tbl_path = os.path.join(os.path.dirname(os.path.abspath(mp_table)), _tbl_rel)
            else:
                _tbl_path = _tbl_rel or mp_table
            _mpr = json.load(open(_tbl_path)).get("mac_per_row") or {}
            from model.sc_common import mp_tracker_set_mac_per_row
            mp_tracker_set_mac_per_row(_mpr)
            if _mpr:
                print(f"[sc] MP tracker: mac_per_row loaded ({len(_mpr)} ops) — "
                      "realized_flop_avg_sl will be reported")
        except Exception as _e:  # never let diagnostics break the eval
            print(f"[sc] MP tracker: mac_per_row unavailable ({_e})")
        levels = getattr(mp, "stoc_len_levels", "?")
        bname = mp_budget_name(levels)
        print(f"[sc] MP (per-row) config=mp_{bname} (nominal name; twin uniform "
              f"sc_{bname}; levels below are halved) "
              f"levels={levels} table={os.path.basename(mp_table)} "
              f"(sc_prec=8, halve=on, owen={os.environ['SC_OWEN_MODE']}, "
              f"masks={os.environ['SC_SCRAMBLE_MASKS']}), "
              f"frontend={frontend} on {n} SCLinear layers (alpha={alpha})")
    else:
        cfg.sc_mp_config = None
        cfg.sc_group_stoclen = {}  # empty map => uniform explicit stoc_len
        print(f"[sc] {tag}: uniform {cycles} cycles "
              f"(sc_prec=8, halve=on, owen={os.environ['SC_OWEN_MODE']}, "
              f"masks={os.environ['SC_SCRAMBLE_MASKS']}), "
              f"frontend={frontend} on {n} SCLinear layers (alpha={alpha})")
    if getattr(cfg, "sc_hybrid_schedule", None) is not None:
        print(f"[hybrid] active schedule={getattr(cfg, 'sc_hybrid_path', '')}")
    return model, tokenizer


def build_model(model_path: str, tag: str, *, device_map="auto",
                alpha: float = 0.5):
    """Return an eval-ready model for a config tag. Used by PPL + LongBench + RULER.

    fp16 -> plain HF. Wx Ax -> plain HF + SmoothQuant + full-matmul fake-quant:
    chunked Linear quant plus INT-patched QK/AV eager attention. act_scales are
    calibrated or cached on the same plain-HF load before Linear replacement.
    sc_* -> SC uniform baseline (see build_sc_model).
    """
    mp_table = os.environ.get("MP_CONFIG_JSON", "").strip()
    if mp_table or tag == "mp" or tag.startswith("mp_"):
        if not mp_table:
            raise SystemExit(
                f"MP tag {tag!r} requires MP_CONFIG_JSON=<calibrated table.json>.")
        return build_sc_model(model_path, tag, device_map=device_map,
                              alpha=alpha, mp_table=mp_table)
    if tag in SC_CONFIGS:
        return build_sc_model(model_path, tag, device_map=device_map, alpha=alpha)
    if tag.lower().startswith("sc_"):
        raise SystemExit(
            f"unknown SC config tag: {tag!r}. Valid: {sorted(SC_CONFIGS)}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    qcfg = parse_config(tag)
    if qcfg is None:
        model = load_plain_model(model_path, device_map=device_map)
        model.eval()
        print(f"[quant] fp16 baseline (no quant) {model_path}")
        return model, tokenizer
    # Load the model ONCE, then (if needed) calibrate act_scales on it, then
    # quantize IN PLACE. Loading a separate calib model + a separate quant model
    # peaks at ~2x the weights, which OOMs a 60GB MoE on a 98GB GPU. Calibration
    # only reads layer INPUTS (forward hooks), never touches weights, so it's safe
    # to run on the same model we then quantize. apply_ptq replaces each Linear in
    # place, freeing the fp16 original as it goes → peak ≈ 1x weights.
    from benchmark.quant.ptq import apply_ptq
    if qcfg.quantize_attention:
        if os.environ.get("BASELINE_ATTN", "eager") != "eager":
            raise SystemExit(
                "INT baselines require BASELINE_ATTN=eager so QK/AV are "
                "fake-quantized; refusing to build a linear-only INT model.")
        n_patched = patch_int_attention_once()
        if n_patched == 0:
            raise SystemExit(
                "INT baseline requested, but no supported HF eager "
                "attention modules were patched.")
        print(f"[ptq] INT attention patch active "
              f"(modules_patched={n_patched}, chunk={qcfg.chunk_size})")
    model = load_plain_model(model_path, device_map=device_map)
    model.eval()
    # act_scales depend only on (model, calib data), NOT bit-width → calibrate
    # once per model and cache; reuse across all W_xA_x configs.
    # WEIGHT-ONLY rows (a_bits>=16) skip SmoothQuant entirely: folding w'=w*s
    # only HARDENS weight quant (outlier channels scale up) while x/s buys
    # nothing at fp16 activations — stock weight-only baselines (BitMoD RTN,
    # GPTQ, AWQ-none) are SQ-free, so an SQ-on W-only row would be a strawman.
    # WONLY_SQ=1 forces SQ back on for a protocol-parity footnote cell.
    # PTQ front-end: FRONTEND=smoothquant (default) | awq. Same switch as the SC
    # path — AWQ reuses the SAME cached per-input-channel scale vectors the SC-AWQ
    # wave built, so an INT baseline is front-end-matched to SC-AWQ. Both fold the
    # X/s,W·s equivalent transform into QuantLinear; only the scale VALUE differs.
    wonly_no_sq = qcfg.a_bits >= 16 and os.environ.get("WONLY_SQ", "0") != "1"
    frontend = os.environ.get("FRONTEND", "smoothquant").lower()
    scales = None
    awq_scales = None
    if frontend == "awq":
        # AWQ is an activation-aware WEIGHT front-end, so it applies even to
        # weight-only configs — that is its home turf (AWQ-BitMoD, AWQ-INT4-wonly).
        # Only SmoothQuant is skipped weight-only (X/s buys nothing at fp16 acts).
        from model.awq_apply import load_awq_scales
        awq_cache = os.environ.get(
            "AWQ_SCALES_DIR", os.path.join(ACT_SCALES_DIR, "awq_scales"))
        awq_scales = load_awq_scales(model_path, awq_cache)
    elif wonly_no_sq:
        print("[quant] weight-only config (A>=16): SmoothQuant skipped "
              "(stock W-only protocol; set WONLY_SQ=1 to force it on)")
    elif frontend == "smoothquant":
        path = os.path.join(ACT_SCALES_DIR, f"act_scales_{_safe(model_path)}.pt")
        if os.path.isfile(path):
            print(f"[quant] act_scales cache hit: {path}")
            scales = torch.load(path, map_location="cpu", weights_only=True)
        else:
            scales = get_act_scales(model_path, tokenizer, model)
    else:
        raise SystemExit(f"[quant] unknown FRONTEND={frontend!r} (want smoothquant|awq)")
    n = apply_ptq(model, qcfg, act_scales=scales, alpha=alpha, awq_scales=awq_scales)
    model.config.use_int_attention = bool(qcfg.quantize_attention)
    model.config.int_attention_bits = int(qcfg.a_bits)
    model.config.int_attention_sym = bool(qcfg.sym)
    model.config.int_chunk_size = int(qcfg.chunk_size)
    torch.cuda.empty_cache()
    _fe = "none(w-only)" if (scales is None and awq_scales is None) else frontend
    print(f"[ptq] {qcfg.tag()}: quantized {n} Linear layers in place "
          f"(frontend={_fe} alpha={alpha}, "
          f"w_dtype={qcfg.w_dtype}, chunk={qcfg.chunk_size}, "
          f"qk_av={'on' if qcfg.quantize_attention else 'off'})")
    return model, tokenizer


@torch.no_grad()
def compute_ppl(
    model,
    enc_ids,
    ctx,
    stride,
    window_losses=None,
    window_batch_size=None,
):
    # window_losses: optional caller-owned list; every scored window appends
    # (start, valid_token_count, mean_loss). The aggregate PPL path is
    # unchanged (sum(loss*valid)/sum(valid) reproduces it exactly).
    if window_batch_size is None:
        window_batch_size = int(os.environ.get("PPL_WINDOW_BATCH_SIZE", "1"))
    window_batch_size = int(window_batch_size)
    if window_batch_size < 1:
        raise ValueError("window_batch_size must be >= 1")

    dev = model.device
    total = enc_ids.shape[0]
    sum_loss, n_loss, prev_end = 0.0, 0, 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    if window_batch_size == 1:
        # Historical path: keep it byte-for-byte separate. This is both the
        # default and the citable/tracing path; batched windows are an opt-in
        # search-throughput optimization only.
        for start in range(0, total - 1, stride):
            end = min(start + ctx, total)
            if end - start < 2:
                break
            ids = enc_ids[start:end].unsqueeze(0).to(dev)
            labels = ids.clone()
            overlap = max(0, prev_end - start)
            if overlap > 0:
                labels[:, :overlap] = -100
            out = model(input_ids=ids, labels=labels)
            valid = (labels[..., 1:] != -100).sum().item()
            if valid > 0:
                loss = float(out.loss)
                sum_loss += loss * valid
                n_loss += valid
                if window_losses is not None:
                    window_losses.append((int(start), int(valid), loss))
            prev_end = end
    else:
        # Independent full windows can share one model forward. This turns a
        # Qwen3-30B-A3B expert's typical 128-row projection into B*128 rows,
        # recovering GPU occupancy without mixing attention across windows.
        # Overlap and padding would change label semantics, so reject them
        # rather than silently compare a different PPL protocol.
        if stride != ctx:
            raise ValueError(
                "window batching requires stride == ctx (independent windows)")
        if total % ctx:
            raise ValueError(
                "window batching requires a token stream of full ctx windows")
        starts = list(range(0, total, ctx))
        for offset in range(0, len(starts), window_batch_size):
            batch_starts = starts[offset:offset + window_batch_size]
            ids = torch.stack(
                [enc_ids[start:start + ctx] for start in batch_starts]
            ).to(dev)
            out = model(input_ids=ids)
            logits = out.logits
            if tuple(logits.shape[:2]) != tuple(ids.shape):
                raise RuntimeError(
                    "batched PPL requires one logit row per input token; got "
                    f"logits={tuple(logits.shape)} ids={tuple(ids.shape)}")
            for row, start in enumerate(batch_starts):
                # Match transformers 4.51 ForCausalLMLoss exactly for one
                # window: upcast all logits, shift by padding labels on the
                # right, retain the final ignore_index row, then mean CE.
                shifted = torch.nn.functional.pad(
                    ids[row], (0, 1), value=-100)[1:].contiguous()
                loss_t = torch.nn.functional.cross_entropy(
                    logits[row].float().contiguous(),
                    shifted,
                    ignore_index=-100,
                    reduction="mean",
                )
                loss = float(loss_t)
                valid = ctx - 1
                sum_loss += loss * valid
                n_loss += valid
                if window_losses is not None:
                    window_losses.append((int(start), int(valid), loss))
            del out, logits, ids
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ppl = math.exp(sum_loss / n_loss) if n_loss else float("inf")
    return ppl, n_loss, time.time() - t0


def main():
    model_path = os.environ["MODEL_PATH"]
    tag = os.environ.get("QUANT_CONFIG", "fp16")
    ctx = int(os.environ.get("CTX", "2048"))
    stride = int(os.environ.get("STRIDE", str(ctx)))
    max_tok = int(os.environ.get("PPL_MAX_TOKENS", "0"))
    alpha = float(os.environ.get("SQ_ALPHA", "0.5"))

    model, tokenizer = build_model(model_path, tag, alpha=alpha)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Canonical GPTQ stream (= BitMoD llm_eval_wikitext.py): join ALL rows,
    # blank lines included, and drop the sub-window tail so every window is
    # full. With stride=ctx this makes our PPL bit-comparable to published
    # GPTQ-protocol tables. NOTE: act_scales calibration (train split) keeps
    # its own join — do not "unify" them, cached scales would be invalidated.
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    if max_tok > 0:
        enc = enc[:max_tok]
    enc = enc[: (enc.shape[0] // ctx) * ctx]
    if sc_trace is not None and sc_trace._ENABLED:
        sc_trace.reset()   # drop anything recorded before the eval proper
    # Realized budget: for MP cells report the DEPLOYMENT row-weighted avg
    # stoc_len (mp_tracker) so PPL is quoted at the ACTUAL budget, not just the
    # calibration target — different metrics/allocations drift differently.
    try:
        from model.sc_common import (mp_tracker_reset, mp_tracker_avg_stoc_len,
                                     mp_tracker_flop_avg_stoc_len)
        mp_tracker_reset()
        _has_mptrack = True
    except Exception:
        _has_mptrack = False
    window_batch_size = int(os.environ.get("PPL_WINDOW_BATCH_SIZE", "1"))
    ppl, n, secs = compute_ppl(
        model, enc, ctx, stride, window_batch_size=window_batch_size)
    realized_sl = mp_tracker_avg_stoc_len() if _has_mptrack else 0.0
    realized_flop_sl = mp_tracker_flop_avg_stoc_len() if _has_mptrack else 0.0
    # realized_flop_avg_sl = MAC-weighted (iso-compute) — the budget's units;
    # realized_avg_sl = row-weighted — threshold-transfer diagnostic only.
    print(f"[RESULT] model={model_path} config={tag} metric=ppl "
          f"value={ppl:.4f} tokens={n} sec={secs:.1f} "
          f"realized_avg_sl={realized_sl:.2f} "
          f"realized_flop_avg_sl={realized_flop_sl:.2f}")
    if sc_trace is not None and sc_trace._ENABLED:
        # Per-call SC precision/shape log -> energy/latency simulator input.
        # Join metadata mirrors benchmark/ppl/ppl.py so simulator tooling can
        # consume hpca traces and MP-sweep traces the same way.
        out = sc_trace.flush(header_extra={
            "model": model_path, "config": tag, "ppl": ppl,
            "eval_tokens": n, "ctx": ctx, "stride": stride,
            "ppl_max_tokens": max_tok,
            "ppl_window_batch_size": window_batch_size,
            "sc_cycles": SC_CONFIGS.get(tag),
            "sc_halve": tag in SC_CONFIGS or bool(os.environ.get("MP_CONFIG_JSON")),
            "sc_prec": 8,
            "mp_config_json": os.environ.get("MP_CONFIG_JSON", ""),
            "total_blocks": getattr(model.config, "_sc_total_blocks", None),
            "attn_granularity": "per_row",  # attention is always per_row (per_head removed 2026-07-03)
            "owen_mode": os.environ.get("SC_OWEN_MODE", "bitrev"),
            "scramble_masks": os.environ.get("SC_SCRAMBLE_MASKS", "64"),
            "use_smoothquant": "1",
            "smoothquant_alpha": alpha,
        })
        if out:
            print(f"[trace] wrote {out}")


if __name__ == "__main__":
    main()
