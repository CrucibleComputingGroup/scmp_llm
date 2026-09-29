#!/bin/bash
# Round-7 stage-2 CPU chain (login node, no GPU, never calls sbatch): kappa stage -> s0 solves -> fixed point ->
# s1 re-solves -> screen manifest build -> launcher dry runs -> per-cell summary + the exact screen sbatch command.
# Run once step 0 (62263343) and the captures (62263344) have finished; safe to re-run (reuses complete outputs,
# refuses to overwrite). Exit 0 done / 2 not ready yet / 3 stop and ask the user / 4 a step failed.
#   bash benchmark/ppl/kbands/run_r7_stage2_cpu_20260928.sh [--cells 30B_t32,30B_t40,30B_t48] [--accept-kappa-review]
#   [--accept-identity-recheck]  (user-approved 2026-09-28; see PRC_R7_CAPTURE_IDENTITY_NOTE_20260928.md)
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=""
exec /nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python -u \
  benchmark/ppl/kbands/run_r7_stage2_cpu_20260928.py "$@"
