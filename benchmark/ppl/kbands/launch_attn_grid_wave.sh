#!/bin/bash
# PPL wave for the per-operator-family attention enable grid.
#
# DO NOT RUN WITHOUT AN EXPLICIT OK. Launch discipline (~/Projects/CLAUDE.md):
# present the plan and wait for approval before any new sbatch wave.
#
# Every arm re-runs the cell's DEPLOYED WINNER config in THIS environment; the
# control is the same config with no grid override. S1 -> S2, never S0 -> S2:
# comparing against an archived number would fold in every environment
# difference since that run (the environment was rebuilt on Turbo 2026-09-20).
#
# The grid is CYCLE-NEUTRAL -- stoc_len is untouched, only rng_levels changes --
# so realized_flop_avg_sl must match the control to well within the ~3% drift
# allowance. A cost difference would mean something else moved.
#
#   bash launch_attn_grid_wave.sh            # dry run, prints what it would do
#   bash launch_attn_grid_wave.sh --go       # actually submit
set -u
GO=0
[[ "${1:-}" == "--go" ]] && GO=1

SB=/home/allenjin/Projects/SCMP/scmp_llm/benchmark/ppl/kbands/run_prc_ppl.sbatch
LOG=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands

# cell: model target deployed-arm  |  grid arms to test (from the GATE)
#   ARMS is filled in from the gate verdict before this is run.
#   QK=<g>  sets SC_RNG_GRID_QK; AV=<g> sets SC_RNG_GRID_AV; both may appear.
: "${ARMS:?set ARMS, e.g. ARMS='ctl QK=64 AV=64'}"
: "${CELLS:?set CELLS, e.g. CELLS=\"4B:32:prcqk llama8B:32:prc\"}"

n=0
for cell in $CELLS; do
  IFS=: read -r M T A <<<"$cell"
  for arm in $ARMS; do
    env_qk=""; env_av=""; suffix=""
    if [[ "$arm" == "ctl" ]]; then
      suffix="agctl"
    else
      for kv in ${arm//,/ }; do
        k=${kv%%=*}; v=${kv##*=}
        [[ "$k" == "QK" ]] && { env_qk="SC_RNG_GRID_QK=$v"; suffix="${suffix}qk$v"; }
        [[ "$k" == "AV" ]] && { env_av="SC_RNG_GRID_AV=$v"; suffix="${suffix}av$v"; }
      done
      suffix="ag$suffix"
    fi
    name="attngrid_${M}_t${T}_${suffix}"
    n=$((n+1))
    echo "[$n] $name  (KB_ARM=$A  $env_qk $env_av)"
    if [[ $GO -eq 1 ]]; then
      # job name must NOT start with prcppl_ -- the archive builder globs that
      # prefix and would silently ingest a variant as the deployed cell.
      env KB_MODEL=$M KB_TARGET=$T KB_ARM=$A KB_TAGSUFFIX=$suffix \
          ${env_qk:+$env_qk} ${env_av:+$env_av} \
        sbatch --job-name="$name" --output="$LOG/${name}_%j.out" "$SB"
    fi
  done
done
echo
echo "total cells: $n  (hard budget 10)"
[[ $GO -eq 0 ]] && echo "DRY RUN -- pass --go to submit"
