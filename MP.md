# Per-row mixed-precision (MP) in scmp_llm

Wires `scmp_kernels.mp.MPConfig` (fixed-fraction quantile dispatch) into the
SC core. Each token row in a Linear forward — and each query row in
attention QK / softmax-V — is classified by its `abs().amax(-1)` and routed
to one of the configured `stoc_len` levels. The classification uses
`scmp_kernels.mp.classify_rows_by_metric`.

The full per-(operator, layer) `AdaptiveMPConfig` path with calibrated
thresholds is *not* wired yet — only the fixed-fraction `MPConfig`. The
overnight 2026-05-26 MP sweep used this path with the level/fraction
triples agreed with the AdaptiveMPConfig calibration goal.

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

`STOC_LENS` still drives the outer sweep loop in `ppl.py`, but with MP the
per-row stoc_len comes from the JSON; the env value only seeds
`sc_stoc_len` as a fallback for any path that has not been wired.

## JSON schema

```jsonc
{
  "type": "MPConfig",
  "stoc_len_levels": [128, 64, 32],   // descending; 0 allowed as last (skip)
  "level_fractions": [0.2, 0.5, 0.3]   // sums to 1; omit for equal split
}
```

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

- `AdaptiveMPConfig` (timestep-adaptive, per-operator, per-layer
  calibrated thresholds) is recognised in `scmp_kernels.mp` but the
  loader rejects `"type": "AdaptiveMPConfig"` until the threshold-table
  calibration pipeline lands in scmp_llm.
- Attention MP loops over `BH`, which scales with `(batch · num_heads)`.
  For large heads (Qwen3-32B, 30B-A3B-MoE) the per-forward overhead is
  measurable; see `_mp_overnight_<ts>/SUMMARY.txt` for the ratio vs the
  non-MP halved baseline.
