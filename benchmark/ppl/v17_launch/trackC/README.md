# V17 Track C — absolute escape gate A/B (launch-ready)

First runtime-code change of the project. Implemented in `model/sc_common.py`
via `kernels/scmp_kernels/mp/config.py` (`_apply_escape_gate` in the
calibrated-table classify path); wired through `loader.py` (reads wrapper keys)
and `benchmark/ppl/mp_ladder_refine.py` (`wrapper_escape_gate`,
`aggregate_trace_weights` escape pricing).

## What it does
After per-call min-max normalization, a row with normalized metric strictly
above `t_esc_b = mu_b + k*sigma_b` (mu/sigma = per-bucket `metric_mean`/
`metric_std` from the V9 table; k global from the wrapper) escapes to
`escape_stoc_len` (128). Escaped rows flow through the normal tracker, so
realized cost stays exact. Wrapper key `escape_gate_k` absent => gate OFF and
byte-identical (proven by test).

## Safety verification (all green)
`python -m unittest benchmark.ppl.test_mp_escape_gate` — 15 tests:
- byte-identity vs the kernels git-HEAD pre-gate `config.py` (bit-exact row_levels)
- gate-off == raw `_classify_rows_by_thresholds` (addition is inert)
- fire above/below t_esc; both metric sign conventions (positive-scale
  invariance + inverted-convention escapes raw-min rows)
- t_esc >= 1.0 and t_esc == 1.0 never fire (strict compare, norm max 1.0)
- wrapper round-trip incl. legacy wrappers (no gate keys)
- cost accounting: escaped MACs priced at 128; collision-with-rung not
  double-counted; no-escape backward-compatible
Full `benchmark/ppl` suite: 77 tests pass.

## Fire-rate / cost cross-check (independent recompute vs gate_sweep_report)
From the V9 4B table stats + the eval profile
(`mp_v16_refine_.../4B/baseline/parent_probe_profile.json`), MAC-weighted:
- k=3.0: fire 0.2134% of MACs  (report predicts 0.213% — exact match) => cost +0.21 cyc
- k=3.5: fire 0.0688% of MACs                                          => cost +0.07 cyc
Row-% predictions: 0.061% (k=3.0) / 0.014% (k=3.5). All 36 buckets carry mu/sigma.

## Cells (both 4B, wrapper = V9 4B avg32 + gate keys, TABLE UNTOUCHED md5 3296688b)
| script | k | escape_sl | predicted cost delta | predicted fire (rows) |
|---|---|---|---|---|
| gate_k35.sbatch | 3.5 | 128 | +0.07 cyc | 0.014% |
| gate_k30.sbatch | 3.0 | 128 | +0.21 cyc | 0.061% |

Judged on (PPL, realized-energy): trace ON, escaped rows priced at 128, so the
realized flop-avg reflects the (intended, small) budget float. Compare vs V9 4B
final 13.784@31.31.

Wrappers: `/nfs/turbo/coe-nbleier/allenjin/hpca/mp_gate_ab_20260719_032500/`
Outputs: `.../mp_gate_ab_20260719_032500/eval_{k3p5,k3}/`
