# Per-row mixed-precision (MP) in scmp_llm

Wires SC mixed precision into the shared SC core. Two MP modes are supported:

- `MPConfig`: fixed-fraction quantile dispatch.
- `AdaptiveMPConfig`: calibrated per-operator/layer threshold tables from
  `benchmark/ppl/calibrate_mp_thresholds.py`.

Each token row in a Linear forward — and each query row in attention QK /
softmax-V — is classified by its row metric and routed to one of the configured
`stoc_len` levels.

## Invocation

Pass a JSON spec via the `MP_CONFIG_JSON` env var. The loader
(`apply_mp_config_from_env` in `loader.py`) attaches an `MPConfig` instance
to `model.config.sc_mp_config`. When unset, every row uses the global
`sc_stoc_len`.

```bash
MP_CONFIG_JSON=benchmark/ppl/mp_m.json \
SC_PREC=8 SC_HALVE_BIPOLAR_STOC_LEN=1 \
SC_ATTN_GRANULARITY=per_row STOC_LENS=128 \
USE_SMOOTHQUANT=1 SMOOTHQUANT_ALPHA=0.5 \
SMOOTHQUANT_SCALES=benchmark/ppl/act_scales_<safe-model>.pt \
python benchmark/ppl/ppl.py
```

Current HPCA runs use `benchmark/quant/eval_quant.py`:

```bash
QUANT_CONFIG=mp \
MP_CONFIG_JSON=benchmark/ppl/mp_calib/<safe>__int7_act_global_wrapper.json \
python benchmark/quant/eval_quant.py
```

With MP enabled, the per-row stoc_len comes from the JSON; the global
`sc_stoc_len` only seeds fallback paths.

## JSON schema

Legacy fixed-fraction MP:

```jsonc
{
  "type": "MPConfig",
  "stoc_len_levels": [128, 64, 32],   // descending; 0 allowed as last (skip)
  "level_fractions": [0.2, 0.5, 0.3]   // sums to 1; omit for equal split
}
```

Calibrated adaptive MP:

```jsonc
{
  "type": "AdaptiveMPConfig",
  "stoc_len_levels": [128, 64, 32],
  "threshold_table_path": "Qwen_Qwen3-4B-Instruct-2507__int7_act_global.json"
}
```

## Hybrid SC/INT schedule

Set `SC_HYBRID_CONFIG_JSON=<path>` (or `HYBRID_CONFIG_JSON=<path>`) alongside
an `sc_*` or `mp` run to route selected `(operator, block)` groups through INT
fake-quant instead of SC. The format intentionally mirrors `scmp_vit`: one row
per operator and one entry per decoder block.

```jsonc
{
  "format": "scmp_llm_hybrid_v1",
  "default": "sc",
  "int_bits": 7,
  "int_sym": true,
  "chunk_size": 128,
  "schedule": {
    "q_proj":    ["sc", "sc", "int7"],
    "down_proj": ["sc", "int7", "int7"],
    "qk":        ["sc", "int7", "sc"],
    "av":        ["sc", "sc", "int7"]
  }
}
```

Entries may be `sc`, `fp`, or `int<N>`; ViT-style `0/1/2` entries are also
accepted as `fp/sc/int<int_bits>`. The INT path reuses the baseline PTQ
fake-quant helpers: linears apply SmoothQuant-compatible chunked weight and
activation RTN, while `qk` and `av` quantize both operands over the reduction
dimension before the fp matmul.

HPCA launcher examples:

```bash
# Sensitivity JSON for later schedule construction.
bash hpca --sensitivity --models 4B --configs sc_avg192 --tag sens_avg192

# Evaluate a hybrid schedule. hpca forces the INT bit-width from the SC budget:
# sc_avg192 -> INT8, sc_avg96 -> INT7.
bash hpca --models 4B --configs sc_avg192 --metrics ppl \
  --sc-backend hybrid \
  --hybrid-config benchmark/ppl/hybrid_schedules/skip_worst_k16.json
```

When launched through `hpca`, `--hybrid-int-bits N` overrides the automatic
budget mapping. The launcher sets `SC_HYBRID_FORCE_INT_BITS=1`, so the chosen
budget bit-width applies even if the schedule entries say `int7`; use plain
`SC_HYBRID_CONFIG_JSON=... python benchmark/quant/eval_quant.py` for fully
literal schedule entries.

The sweep ships three references:

| file | levels | fractions | avg ≈ |
|---|---|---|---|
| `benchmark/ppl/mp_a.json` | [128, 64, 32] | [0.2, 0.5, 0.3] | 67 |
| `benchmark/ppl/mp_m.json` | [128, 96, 64] | [0.34, 0.33, 0.33] | 96 |
| `benchmark/ppl/mp_c.json` | [128, 96] | [0.5, 0.5] | 112 |

## Implementation notes

- `SCLinear.forward` flattens `(B, N, D)` to `(B*N, D)`, computes
  `metric = x.abs().amax(-1)`, classifies, then loops once per `stoc_len`
  level calling `sc_matmul` on the row subset (per_row + chunk_d=128 +
  bipolar). Outputs are scattered back into a pre-allocated buffer.
- `sc_eager_attention_forward` does the same per `(B, H)` slice, using
  `query.abs().amax(-1)` for the Q·Kᵀ classification and
  `attn_weights.abs().amax(-1)` for softmax·V. Costs BH × n_levels kernel
  launches per matmul — slower than the batched 3D path but lets per-row
  stoc_len vary inside a single attention call.
- `model/sc_common.py:mp_tracker_reset()` / `mp_tracker_avg_stoc_len()`
  accumulate the weighted-mean effective stoc_len across one
  `compute_ppl` call. `ppl.py` prints the average alongside the PPL row.
- MoE router `gate` Linear is excluded from `SCLinear` replacement, so it
  is also untouched by MP (same skip set as before).
- `halve_bipolar_stoc_len=True` is still forwarded to `sc_matmul` for
  every level call. The level values in the JSON already live in the
  halved space (≤ `2**(sc_prec-1) = 128` at `sc_prec=8`); the kernel
  treats the explicit `stoc_len` as authoritative and only auto-fills
  `rng_levels` to `2**(sc_prec-1)`.

## Known limitations

- Hybrid INT linears currently fake-quantize the scheduled module's weight at
  forward time rather than prepacking a `QuantLinear` buffer. This is acceptable
  for sparse worst-group sweeps but can be optimized if many groups go INT.
- Attention MP loops over `BH`, which scales with `(batch · num_heads)`.
  For large heads (Qwen3-32B, 30B-A3B-MoE) the per-forward overhead is
  measurable; see `_mp_overnight_<ts>/SUMMARY.txt` for the ratio vs the
  non-MP halved baseline.
