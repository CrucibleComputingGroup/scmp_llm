#!/bin/bash
# Round-7 CAPTURE (T1 per-group allocation; kbands/ROUNDS_6_8_PLAN_20260928.md section 4, step 1).
# One calib10_r7 capture per cell = the incumbent's exact calib7 recipe + round-7 measurement:
# 512 attention rows/call (rank<128 prefix measured as calib7's own batches), every attention
# bucket measured at its parent rungs + 16..112 + 128, protected-slice Fisher error, held-out
# Fisher scores of c17 / c17_s80 / INC, solve state + attention/hold dumps for the offline
# family-kappa solve (benchmark/ppl/prc_r7_solve.py). Pure measurement: no table from this job is
# a candidate. Then (CPU) the incumbent-identity reproduction BEFORE any extension: in-process
# gfis == c7_gfis, 128-row prefix joint re-solve == c7_gfisla, held-out predictions == c7 diag,
# INC lineage; then one extended-ladder pipeline check through the runtime resolver.
#   array index: 0 = 30B_t32, 1 = 30B_t40, 2 = 30B_t48 (enabled);
#                3 = 4B_t40, 4 = 4B_t64 (disabled, not run in round 7: 4B t40 gate failed; 4B t64 is a
#                known-outcome trial, see the manifest's prereg.not_run_reason)
# Fixed: parent wrapper + h20 INT mask, AWQ cache, SC RNG (bitrev, 64 masks, 128-level grid),
# no QK rebalance / smooth-scale change, escape mu+2tau at 128, psl, dense ladder, seed/windows.
# Pre-registration (kappa rule, primary, MDE, s0, refusal, target-scale rule) is frozen in the
# manifest BEFORE submission; the job refuses to run on any other manifest version (pin).
# GPU cap (user: <= 4 GPUs TOTAL). NOT executed here. Check `squeue -u allenjin` first; then:
#   PY=/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python
#   PYTHONPATH=$PWD/kernels $PY benchmark/ppl/prc_r7_capture_20260928.py pin --manifest benchmark/ppl/kbands/prc_r7_capture_20260928.json
#   SHA=$(sha256sum benchmark/ppl/kbands/prc_r7_capture_20260928.json | cut -c1-64)
#   (A) R7C_MANIFEST_SHA256=$SHA sbatch --parsable --array=0-1%2 benchmark/ppl/kbands/run_prc_r7_capture_20260928.sbatch
#   (B) R7C_MANIFEST_SHA256=$SHA sbatch --parsable --array=2 --dependency=afterany:62208440 benchmark/ppl/kbands/run_prc_r7_capture_20260928.sbatch
# Accounting: round-6 62208439 finished 14:18 EDT; 62208440 (1 GPU) may still run; step 0 = 1 GPU (%1).
#   62208440 + step 0 + (A) = 1 + 1 + 2 = 4; after 62208440: step 0 + (A) + (B) = 4. No screen job
#   while captures run. If anything else of ours holds a GPU, use the conservative single array:
#   R7C_MANIFEST_SHA256=$SHA sbatch --parsable --array=0-2%1 benchmark/ppl/kbands/run_prc_r7_capture_20260928.sbatch
# The header default below is %1 (a bare sbatch holds at most 1 GPU); always pass --array.
# CPU dry run on the login node (no Slurm, no writes, no pin needed):
#   R7C_DRY_RUN=1 R7C_INDEX=<0|1|2> bash benchmark/ppl/kbands/run_prc_r7_capture_20260928.sbatch
#SBATCH --job-name=r7capture20260928
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/r7capture20260928_%A_%a.out
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
export R7C_MANIFEST="$PWD/benchmark/ppl/kbands/prc_r7_capture_20260928.json"
export R7C_INDEX=${SLURM_ARRAY_TASK_ID:-${R7C_INDEX:?array task index required}}
DRIVER=benchmark/ppl/prc_r7_capture_20260928.py
CALIB=benchmark/ppl/mp_per_row_chunk_calib10_r7.py
R7C_ENV_PY=/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python

# Manifest fields (system python3, read-only).
CELL_FIELDS=$(python3 - <<'PY'
import json, os
from pathlib import Path
m = json.loads(Path(os.environ['R7C_MANIFEST']).read_text())
i = int(os.environ['R7C_INDEX'])
assert m['schema'] == 'prc-r7-capture-v1', m['schema']
assert m['max_total_gpus'] == 4 and m['gpus_per_task'] == 1
assert m['output_root'] == '/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r7_20260928'
assert [c['id'] for c in m['cells']] == ['30B_t32', '30B_t40', '30B_t48', '4B_t40', '4B_t64']
assert 0 <= i < len(m['cells']) and m['cells'][i]['index'] == i
c = m['cells'][i]
p = c['paths']
assert p['cell_dir'] == m['output_root'] + '/' + c['id'] and p['capture_dir'] == p['cell_dir'] + '/capture'
f = [c['id'], 'enabled' if c['enabled'] else 'disabled', p['cell_dir'], p['capture_dir'], p['job_log'],
     p['calib_log'], c['inputs']['c7_log']['path'], c['inputs']['c7_log']['sha256']]
assert all('\t' not in x and '\n' not in x and x for x in f)
print('\t'.join(f))
PY
)
IFS=$'\t' read -r CELL_ID CELL_STATE CELL_DIR CAPTURE_DIR JOB_LOG CALIB_LOG C7_LOG C7_LOG_SHA <<< "$CELL_FIELDS"
export CELL_ID CELL_DIR CAPTURE_DIR
echo "[r7c] cell=$CELL_ID ($CELL_STATE) capture=$CAPTURE_DIR"
[[ "$CELL_STATE" == enabled ]] || { echo "[r7c] $CELL_ID is disabled in the manifest; refusing" >&2; exit 2; }

if [[ ${R7C_DRY_RUN:-0} == 1 ]]; then
  # Hash preflight + the exact calib10 argv parsed by calib10's REAL parser with CUDA hidden
  # (it prints "SKIP: needs CUDA" and exits 0).
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels "$R7C_ENV_PY" "$DRIVER" preflight \
    --manifest "$R7C_MANIFEST" --index "$R7C_INDEX"
  mapfile -d '' -t CARGS < <(PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels \
    "$R7C_ENV_PY" "$DRIVER" calib-argv --manifest "$R7C_MANIFEST" --index "$R7C_INDEX")
  echo "[r7c] calib10 argv: ${CARGS[*]}"
  OUTP=$(CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels \
    "$R7C_ENV_PY" "$CALIB" "${CARGS[@]}")
  [[ "$OUTP" == "SKIP: needs CUDA" ]] || { echo "[r7c] calib10 argv check failed: $OUTP" >&2; exit 2; }
  [[ ! -e "$CAPTURE_DIR" ]] || { echo "[r7c] capture dir already exists" >&2; exit 2; }
  echo "[r7c] manifest sha256 $(sha256sum "$R7C_MANIFEST" | cut -c1-64) (a real job needs the pin + R7C_MANIFEST_SHA256)"
  echo "[r7c] DRY RUN PASS $CELL_ID (preflight + calib10 argv parse); nothing written"
  exit 0
fi

set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail

# Freeze the numerical protocol after shell/conda initialization (as run_prc7 / round 6).
while IFS= read -r name; do unset "$name"; done < <(compgen -v SC_ || true)
unset MP_CONFIG_JSON QUANT_CONFIG KB_PRC_OVERRIDE KB_HYBRID KB_QKALPHA KB_TAGSUFFIX SC_MP_TRACE
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64
export CTX=2048 STRIDE=2048 PPL_MAX_TOKENS=0 PPL_WINDOW_BATCH_SIZE=1 SQ_ALPHA=0.5
export FRONTEND=awq
export ACT_SCALES_DIR=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales
export AWQ_SCALES_DIR="$ACT_SCALES_DIR/awq_scales" AWQ_OBJ_BITS=4
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"

python -u "$DRIVER" verify-pin --manifest "$R7C_MANIFEST"   # the manifest version submitted
python -u "$DRIVER" preflight --manifest "$R7C_MANIFEST" --index "$R7C_INDEX"
mkdir -p "$CELL_DIR"
mkdir "$CAPTURE_DIR"   # atomic refusal if this cell was already captured
exec > >(tee "$JOB_LOG") 2>&1
# durable copy of the calib7 log the identity step reproduces (/scratch is purged after ~60 days)
mkdir "$CAPTURE_DIR/inputs"
cp -p "$C7_LOG" "$CAPTURE_DIR/inputs/"
[[ "$(sha256sum "$CAPTURE_DIR/inputs/$(basename "$C7_LOG")" | cut -c1-64)" == "$C7_LOG_SHA" ]] || {
  echo "[r7c] calib7 log copy does not match the manifest sha" >&2; exit 2; }
echo "[r7c] array=${SLURM_ARRAY_JOB_ID:-} task=$R7C_INDEX job=${SLURM_JOB_ID:-} cell=$CELL_ID host=$(hostname)"
echo "[r7c] capture only (no candidate tables); fixed AWQ/mask/RNG/QK/escape/psl; <= 4 GPUs total"
python - <<'PY'
import datetime, hashlib, json, os
from pathlib import Path
mp = Path(os.environ['R7C_MANIFEST'])
m = json.loads(mp.read_text())
code = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in m['source_files']}
assert code == m['code_hashes'], 'experiment source changed after preflight'
env_keys = sorted(k for k in os.environ if k.startswith(('SC_', 'AWQ_', 'CTX', 'STRIDE', 'PPL_', 'SQ_',
                                                         'FRONTEND', 'ACT_SCALES', 'PYTHONPATH', 'HF_')))
msha = hashlib.sha256(mp.read_bytes()).hexdigest()
assert msha == os.environ['R7C_MANIFEST_SHA256'], 'manifest changed after the pin check'
pin = mp.with_name(mp.stem + '_pin.json')
rec = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'manifest': str(mp), 'manifest_sha256': msha,
       'pin': str(pin), 'pin_sha256': hashlib.sha256(pin.read_bytes()).hexdigest(),
       'cell': m['cells'][int(os.environ['R7C_INDEX'])], 'protocol': m['protocol'],
       'preregistered_rules': m['preregistered_rules'], 'code_sha256': code,
       'env': {k: os.environ[k] for k in env_keys},
       'slurm': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID',
                                                'SLURM_ARRAY_TASK_ID', 'SLURMD_NODENAME')}}
with (Path(os.environ['CAPTURE_DIR']) / 'provenance.json').open('x') as f:
    json.dump(rec, f, indent=1)
    f.write('\n')
PY

# (1) calib10_r7 capture (GPU): the incumbent's recorded calib7 args + round-7 measurement.
mapfile -d '' -t CARGS < <(python "$DRIVER" calib-argv --manifest "$R7C_MANIFEST" --index "$R7C_INDEX")
[[ ${#CARGS[@]} -gt 20 ]] || { echo "[r7c] empty calib10 argv" >&2; exit 2; }
echo "[r7c] calib10 argv: ${CARGS[*]}"
# (the pipe into tee returns only after the log is complete; identity reads the calib7-format lines)
srun python -u "$CALIB" "${CARGS[@]}" 2>&1 | tee "$CALIB_LOG"

# (2) Incumbent-identity reproduction + extended-ladder pipeline check (CPU, fail-closed;
#     identity.json is always written).
rc=0
python -u "$DRIVER" identity --manifest "$R7C_MANIFEST" --index "$R7C_INDEX" || rc=$?
if [[ $rc == 3 ]]; then
  echo "[r7c] IDENTITY NOT OK (level fail or near = stop and report): no round-7 arm may be solved from $CAPTURE_DIR" >&2
  exit 3
elif [[ $rc == 4 ]]; then
  echo "[r7c] identity OK but the PIPELINE CHECK failed (emitter/validator): the capture is valid; fix prc_r7_solve and re-run 'identity --tag <t>' on CPU" >&2
  exit 4
elif [[ $rc != 0 ]]; then
  echo "[r7c] identity step crashed (exit $rc)" >&2
  exit $rc
fi
echo "[r7c] capture + identity OK for $CELL_ID; offline solves: benchmark/ppl/prc_r7_solve.py solve --manifest ... --cell $CELL_ID"
