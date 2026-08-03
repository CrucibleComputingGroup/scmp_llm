# Track C — absolute escape gate (R7) — agent work log

Work dir: /home/allenjin/Projects/scmp_llm/benchmark/ppl/v17_launch/trackC/
Date: 2026-07-19. NO sbatch submission from this session; scripts are prepared only.

## Task
1. Escape gate in adaptive_classify_rows path: metric_norm > mu_b + k*sigma_b
   => row escapes to escape_stoc_len (128 halved). Wrapper keys escape_gate_k
   (global) + escape_stoc_len (default 128). Key absent => gate OFF,
   byte-identical (must be PROVEN by test).
2. Full test suite (6 items, CPU-only) + full existing ppl suite.
3. 2 A/B wrappers (k=3.5 / 3.0) from the V9 4B avg32 wrapper, table untouched,
   in Turbo mp_gate_ab_<ts>/. Cross-check predicted cost +0.07/+0.21 and fire
   rates 0.014%/0.061% offline vs the actual table.
4. 2 WRITE-ONLY sbatch scripts (gate_k35 / gate_k30) + README here.

## Audit of pre-existing (uncommitted) implementation — 2026-07-19
A prior session already landed a complete implementation (uncommitted):

- kernels/scmp_kernels/mp/config.py (editable-installed package; kernels HEAD
  3ba991b is the PRE-gate version — perfect byte-identity reference):
  * AdaptiveMPConfig fields escape_gate_k (None default) + escape_stoc_len
    (128) + precomputed bucket_escape_thresholds/operator_default_escape_
    thresholds (t_esc = metric_mean + k*metric_std per table payload).
  * load_threshold_table extracts stats; raises if gate on + no stats anywhere.
  * get_escape_threshold mirrors get_thresholds lookup (bucket then op default).
  * classify_level_values(): ladder + appended escape entry (index
    len(stoc_len_levels)) when gate on and 128 not a rung; ladder itself
    otherwise (byte-identical loop).
  * _apply_escape_gate called in adaptive_classify_rows AFTER min-max norm +
    _classify_rows_by_thresholds: strict compare metric_norm > t_esc; t_esc
    None or >= 1.0 => early return (never fires); escaped rows get level index
    len(levels) + level_row_indices[esc_sl] entry; ladder index lists rebuilt.
  * Constant-metric early path (all rows -> level 0) returns BEFORE the gate —
    gate cannot fire there (matches _record_mp_metric_profile's norm=1 case?
    NOTE: profile maps constant to ones, classify maps to level 0 — pre-existing
    asymmetry, not gate-related).
- model/sc_common.py: attention path uses mp_config.classify_level_values()
  (only gate-related change; SCLinear iterates assignment.level_row_indices so
  it needed no change). _record_assignment iterates level_row_indices =>
  tracker sees escaped rows at 128. Trace: escaped rows call _sc_matmul at
  stoc_len=128 under the same op context => trace groups show 128.
- loader.py apply_mp_config_from_env: reads wrapper keys escape_gate_k /
  escape_stoc_len (absent => None/128), passes to AdaptiveMPConfig, prints
  gate note.
- benchmark/ppl/mp_ladder_refine.py: wrapper_escape_gate(), _write_wrapper
  carries gate keys, _install_runtime_table(**escape_gate), _eval_once passes
  gate, aggregate_trace_weights splits escape_weight + prices cost at 128,
  propose_floor_exchange extra_cost. resolve_parent_table returns
  (wrapper, table_path, table); main() unpacks in that order — CORRECT, no
  psl=None mis-unpack.

MISSING (my deliverables): the entire test suite; A/B wrappers; fire-rate
cross-check; sbatch scripts; this README/NOTES.

V9 4B table verified: 36 buckets + 9 operator_defaults, ALL carry
metric_mean/metric_std; levels [96,64,48,32,24,16]; protected stoc_len 112;
mac_per_row present. Wrapper has only type/stoc_len_levels/threshold_table_path.

## Progress
- [x] Read gate_sweep_report.md + scmp_llm/CLAUDE.md + full sc_common.py +
      kernels mp/config.py diff + loader.py diff + mp_ladder_refine.py.
- [x] Audit implementation (above) — complete, no code gaps found so far.
- [ ] Test suite benchmark/ppl/test_mp_escape_gate.py (6 items).
- [ ] Run full existing ppl test suite.
- [ ] Fire-rate/cost cross-check vs table + V16 parent_probe_profile.
- [ ] A/B wrappers on Turbo.
- [ ] sbatch scripts + README.

## Key facts for tests
- Env: conda annstention; NO pytest — unittest style, each test file has
  unittest.main(); run `python -m unittest benchmark.ppl.test_...` or direct.
- _sc_matmul is Triton/GPU — mock it in sc_common for CPU dispatch tests.
- Byte-identity reference: `git -C kernels show HEAD:scmp_kernels/mp/config.py`
  (pre-gate code) imported as a scratch module.
- Table schema minimum: stoc_len_levels, timestep_buckets, layer_buckets,
  operator_defaults{op:{thresholds,[metric_mean,metric_std]}},
  buckets{"op:t0:l0":{...}}; thresholds len = n_levels-1, non-increasing in [0,1].
- V16 4B eval histograms: parent_probe_profile.json under
  mp_v16_refine_mp_v16_paired_20260717_165536 (find exact path).
