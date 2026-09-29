# Round 4: bounded adjacent-rung allocation

Authorized September 27, 2026. The user subsequently raised queue occupancy to
**four GPUs total**. The original pilot contains two cells: Qwen3-4B at nominal 6 bits (`t32`) and
Llama-3.1-8B at nominal 6 bits (`t32`). Both start from their exact current
best(all) `c6_gfis` tables; no Fisher table is regenerated as a substitute.

## Change under test

Round 3 moved all thresholds of a population together. Coincident attention
thresholds could therefore send affected rows across several rungs. This pilot
changes one threshold in each of two distinct buckets: one donor and one
recipient. Thresholds remain between their unchanged neighbors. Float32
threshold equalities and intervening intervals are checked to ensure that no
input metric moves more than one ladder index.

The proposal pool separates QK and AV, includes projection/MLP exchanges and
within-family exchanges, and contains at most 12 complete tables per cell.
Requested transfers of 0.25% and 0.5% of total SC cycles are ceilings; locality
and histogram discreteness may produce smaller moves. Each affected bucket
may change at most 5% of its groups and 5% of its MACs, with at most 5% change
to its own cycle budget. Gross moved cycles for any operator type are capped
at 2% of that operator's adjustable cycle budget; opposite moves do not cancel
this cap. Proposals use 80% of these local limits to leave room for changes in
the observed inputs; audits enforce the full limits. Candidate diagnostics
record the actual amount moved.

The quantization frontend, integer mask, protected channels/lengths, ladders,
dispatch statistics, escape gates, RNG, QK operands, and model weights stay
fixed. Only existing threshold values change. No new runtime metadata is added.

## Evaluation and selection

The manifest, rather than CLI defaults, specifies all experimental parameters.
Each cell uses six fresh WikiText-2 TRAIN search windows and sixteen disjoint
confirmation windows, each 2048 tokens. Window selection excludes earlier
calibration diagnostics and both search and confirmation windows from round 3.
The corpus sampling seed is local to window selection; SC RNG is unchanged.

An observe-only profiler records actual group counts, MACs, and threshold
occupancies, including the true width of tail chunks. It includes both the
incumbent and candidate boundaries so both tables can be replayed exactly on
the same observed inputs. Locality is checked on the incumbent trajectory and
again on each candidate's own trajectory. Exact trace costs, rather than the
approximate tracker alone, must stay within 1% of the incumbent.

No-op, profiled, and restored incumbent evaluations must match exactly in
loss and traced SC work. Only a feasible candidate with lower search NLL
advances to confirmation. Full TEST runs only if fresh confirmation has a
negative paired mean, finite `z < -1.5`, and passes both cost and locality
guards. If no candidate passes, the job finishes with the incumbent retained.
It does not force a test of the least bad candidate.

The launcher verifies the full-test token count, context/stride, batch size,
bit-reversal/mask settings, 8-bit/halve convention, and fixed 128-level SC RNG
grid. Final reporting remains the best full-test PPL across all comparable
allocation rounds, with the existing incumbent always eligible.

## Artifacts and execution

- Driver: `benchmark/ppl/prc_adjacent_refine.py`.
- Proposal generator/profiler: `benchmark/ppl/prc_adjacent_proposals.py`.
- Manifest: `prc_adjacent_20260927.json`; records frozen input and source hashes.
- Launcher: `run_prc_adjacent_20260927.sbatch`, array `0-1%2`, one GPU per task.
- Outputs: `/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_adjacent_20260927/<cell>/`.
- Scheduler logs: `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/p4adjacent20260927_<job>_<task>.out`.
- Each cell records source provenance, exact window starts, identity checks,
  profiles, proposal diagnostics, paired losses, `selected.json`, and, only if
  confirmation passes, a full-test trace and `full_test_result.json`.

Submitted as array **62104342** at 13:38 EDT on September 27, 2026:
task 0 is `4B_t32`, task 1 is `llama8B_t32`. Both initially waited for resources.
The manifest and all 14 source hashes were frozen before submission. Validation
passed 32 CPU tests, both driver preflights, both launcher dry runs, and proposal
generation on two archived profiles (12 candidates each). The archived-profile
smoke used synthetic count=MAC only to exercise the generator; GPU runs collect
actual group counts before proposing any candidate.

Submission details and the latest checked status are recorded in
`prc_adjacent_20260927_submission.json`.

## Expansion to four GPUs

The user increased the limit after the original two-cell submission. A companion
array, **62105189**, adds `30B_t32` and `30B_t40`, each requesting one GPU, with
array concurrency `0-1%2`. The original two-task array **62104342** remains intact;
the two arrays together can occupy at most four GPUs. Both arrays initially
waited for resources. The original manifest retains its two-GPU limit and hashes
as the immutable record of that submission.

The added cells use exactly the same driver, generator, numerical settings,
locality limits, and confirmation gate. Their current best(all) incumbents are
`30B_t32_c7_gfisla` (PPL 8.58117847495862) and round-3
`30B_t40/candidate07_mlp_from_projections` (PPL 8.085610625763966). Both exclude
all recorded 30B round-2 calibration windows and the round-3 t40 search and
confirmation windows. The driver profiles each incumbent afresh.

The companion manifest and submission record are
`prc_adjacent_30b_20260927{,_submission}.json`; its launcher is
`run_prc_adjacent_30b_20260927.sbatch`. Shared source hashes remain identical to
the original submission. Both added cells passed driver preflight, launcher dry
run, and read-only MoE compatibility review. Output cell directories remain under
the same `prc_adjacent_20260927` root. Scheduler logs use prefix
`p4adj30b20260927_<job>_<task>.out`.
