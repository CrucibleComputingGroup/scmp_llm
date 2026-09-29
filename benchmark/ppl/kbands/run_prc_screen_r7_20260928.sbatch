#!/bin/bash
# Round-7 SCREEN -> CONFIRM -> FULL TEST (one GPU per cell). Pure allocation candidates from the round-7
# CPU solves (family-kappa joint lambda, attention ladders up to 128, parent cost); fixed AWQ / INT mask /
# RNG (bitrev, 64 masks, 128-level grid) / QK / escape. Driver: benchmark/ppl/prc_screen_r7_20260928.py.
#   task 0 = 30B_t32   task 1 = 30B_t40   task 2 = 30B_t48   task 3 = 4B_t40   task 4 = 4B_t64
# Disabled tasks (manifest "enabled": false, e.g. 4B unless its round-6 gate passed) exit immediately.
# Per task: [GPU] screen INC + PARENT + primary + <=2 secondaries on 16 TRAIN windows (bit identity,
# audits, cost gate |C/P-1| <= 1%) -> confirm the PRIMARY on 32 fresh windows (z < -2) -> [GPU]
# full-protocol TEST of the primary via run_prc_ppl.sbatch (full WikiText-2 test, ctx 2048,
# PPL_MAX_TOKENS=0, batch 1) -> [CPU] finalize (header + realized lengths + the run environment from
# run_prc_ppl's own log; best(all) entry only within +-1.5% test cost of the parent). Secondaries are
# never confirmed or tested. Every table was solved at its pre-registered fixed-point scale s1.
# GPU cap: submit with --array=<enabled tasks>%K and --dependency so K + every other running GPU job <= 4.
# The manifest (benchmark/ppl/kbands/prc_screen_r7_20260928.json) must be frozen before the job STARTS.
# CPU dry run (no writes, no Slurm; R7SCR_MANIFEST may point at a scratch dry-run manifest):
#   R7SCR_DRY_RUN=1 R7SCR_TASK=<0..4> [R7SCR_MANIFEST=...] bash benchmark/ppl/kbands/run_prc_screen_r7_20260928.sbatch
#SBATCH --job-name=r7screen20260928
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/r7screen20260928_%A_%a.out
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
export R7SCR_MANIFEST="${R7SCR_MANIFEST:-$PWD/benchmark/ppl/kbands/prc_screen_r7_20260928.json}"
export R7SCR_TASK=${SLURM_ARRAY_TASK_ID:-${R7SCR_TASK:?array task index required}}
DRIVER=benchmark/ppl/prc_screen_r7_20260928.py
R7SCR_ENV_PY=/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python

# System-python preflight (no writes).
TASK_FIELDS=$(python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
m = json.loads(Path(os.environ['R7SCR_MANIFEST']).read_text())
i = int(os.environ['R7SCR_TASK'])
dry = os.environ.get('R7SCR_DRY_RUN', '0') == '1'
assert m['schema'] == 'prc-screen-r7-v1', m['schema']
assert m['max_total_gpus'] == 4 and m['gpus_per_task'] == 1
assert [t['id'] for t in m['tasks']] == ['30B_t32', '30B_t40', '30B_t48', '4B_t40', '4B_t64']
assert 0 <= i < len(m['tasks']) and m['tasks'][i]['index'] == i
assert m['output_root'] == '/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r7_20260928/screen'
assert m['protocol']['qk_rebalance'] is False and m['protocol']['rng_grid'] == 'fixed128'
assert dry or m['dry_run'] is False, 'refusing to run a DRY-RUN manifest for real'
assert dry or not m.get('test_build'), 'refusing to run a TEST-BUILD manifest for real'
code = m['code_hashes']
assert code and set(code) == set(m['source_files']), 'incomplete source freeze'
for p, h in code.items():
    assert sha(p) == h, f'experiment source changed: {p}'
t = m['tasks'][i]
c = m['cells'][t['cell']]
if not c['enabled']:
    print('\t'.join([t['id'], 'DISABLED', m['output_root'], '-', '-']))
    raise SystemExit(0)
for p, h in c['hashes'].items():
    assert sha(p) == h, f'frozen input changed: {p}'
out = Path(m['output_root']) / c['id']
assert dry or not out.exists(), f'refusing to reuse an existing cell directory: {out}'
tt = c['arms'][c['primary']]['test_trace']
assert not Path(tt).exists(), f'refusing to overwrite {tt}'
f = [t['id'], 'ENABLED', m['output_root'], c['primary'], tt]
assert all('\t' not in x and '\n' not in x for x in f)
print('\t'.join(f))
PY
)
IFS=$'\t' read -r TASK_ID TASK_STATE R7SCR_ROOT PRIMARY TEST_TRACE <<< "$TASK_FIELDS"
export TASK_ID R7SCR_ROOT PRIMARY TEST_TRACE
if [[ "$TASK_STATE" == "DISABLED" ]]; then
  echo "[r7scr] task $TASK_ID is disabled in the manifest; nothing to do"
  exit 0
fi
echo "[r7scr] verified frozen inputs: task=$TASK_ID primary=$PRIMARY test_trace=$TEST_TRACE root=$R7SCR_ROOT"
if [[ ${R7SCR_DRY_RUN:-0} == 1 ]]; then
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels CUDA_VISIBLE_DEVICES="" "$R7SCR_ENV_PY" "$DRIVER" \
    preflight --manifest "$R7SCR_MANIFEST" --task "$R7SCR_TASK" --deep > /dev/null
  [[ "$TEST_TRACE" == /nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl/*_prc_r7*20260928_trace.json ]] \
    || { echo "[r7scr] test trace name does not follow run_prc_ppl.sbatch naming" >&2; exit 2; }
  echo "[r7scr] DRY RUN PASS task=$TASK_ID (launcher + driver deep preflight); nothing written"
  exit 0
fi

set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail

# Freeze the numerical protocol after shell/conda initialization (as rounds 4 / 6).
while IFS= read -r name; do unset "$name"; done < <(compgen -v SC_ || true)
unset MP_CONFIG_JSON QUANT_CONFIG KB_PRC_OVERRIDE KB_HYBRID KB_QKALPHA KB_TAGSUFFIX KB_TBL KB_PARENT
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 SC_HW_MAX_MASKS=64
export CTX=2048 STRIDE=2048 PPL_MAX_TOKENS=0 PPL_WINDOW_BATCH_SIZE=1 SQ_ALPHA=0.5
export FRONTEND=awq
export ACT_SCALES_DIR=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales
export AWQ_SCALES_DIR="$ACT_SCALES_DIR/awq_scales" AWQ_OBJ_BITS=4
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"

python -u "$DRIVER" preflight --manifest "$R7SCR_MANIFEST" --task "$R7SCR_TASK"
CELL_DIR="$R7SCR_ROOT/$TASK_ID"
mkdir -p "$R7SCR_ROOT" "$R7SCR_ROOT/_logs"
mkdir "$CELL_DIR"   # atomic refusal if the cell already ran
exec > >(tee "$R7SCR_ROOT/_logs/${TASK_ID}_${SLURM_ARRAY_JOB_ID:-nojob}_${R7SCR_TASK}.log") 2>&1
echo "[r7scr] array=${SLURM_ARRAY_JOB_ID:-} task=$R7SCR_TASK job=${SLURM_JOB_ID:-} cell=$TASK_ID host=$(hostname)"
echo "[r7scr] one GPU; pre-registered primary=$PRIMARY; secondaries screened for information only"
export CELL_DIR
python - <<'PY'
import datetime, hashlib, json, os
from pathlib import Path
mp = Path(os.environ['R7SCR_MANIFEST'])
m = json.loads(mp.read_text())
code = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in m['source_files']}
assert code == m['code_hashes'], 'experiment source changed after preflight'
t = m['tasks'][int(os.environ['R7SCR_TASK'])]
env_keys = sorted(k for k in os.environ if k.startswith(('SC_', 'AWQ_', 'CTX', 'STRIDE', 'PPL_', 'SQ_',
                                                         'FRONTEND', 'ACT_SCALES', 'PYTHONPATH', 'HF_')))
rec = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'manifest': str(mp), 'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
       'task': t, 'cell': m['cells'][t['cell']], 'protocol': m['protocol'], 'kappa': m['kappa'],
       'preregistered_rules': m['preregistered_rules'], 'code_sha256': code,
       'env': {k: os.environ[k] for k in env_keys},
       'slurm': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID',
                                                'SLURMD_NODENAME')}}
with (Path(os.environ['CELL_DIR']) / 'provenance.json').open('x') as f:
    json.dump(rec, f, indent=1)
    f.write('\n')
PY

# (1) Screen + confirm (GPU, one model load). Writes selected.json.
srun python -u "$DRIVER" run --manifest "$R7SCR_MANIFEST" --task "$R7SCR_TASK" --out-root "$R7SCR_ROOT"

# (2) Full-test arguments (verified against the pre-registered primary), or stop.
TEST_FIELDS=$(python -u "$DRIVER" test-args --manifest "$R7SCR_MANIFEST" --task "$R7SCR_TASK")
if [[ -z "$TEST_FIELDS" ]]; then
  echo "[r7scr] primary not confirmed (see $CELL_DIR/selected.json); no full test"
  exit 0
fi
IFS=$'\t' read -r T_MODEL T_TARGET T_WRAPPER T_TBL T_TRACE T_PARENT <<< "$TEST_FIELDS"
[[ "$T_TRACE" == "$TEST_TRACE" ]] || { echo "[r7scr] test trace mismatch" >&2; exit 2; }
[[ "$T_TRACE" == "/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl/${T_MODEL}_t${T_TARGET}_prc_${T_TBL}_trace.json" ]] \
  || { echo "[r7scr] manifest test_trace does not match run_prc_ppl.sbatch naming" >&2; exit 2; }
[[ ! -e "$T_TRACE" ]] || { echo "Refusing to overwrite $T_TRACE" >&2; exit 2; }

# (3) Full-protocol TEST of the primary (same eval settings as the incumbent: ctx/stride 2048, batch 1,
#     masks 64, AWQ, hybrid mask from the parent config dir). run_prc_ppl.sbatch is called, never edited.
#     Its own output (the hybrid-mask / FRONTEND / masks / RNG-grid lines the trace header cannot carry)
#     is kept in the cell directory and verified by finalize.
export KB_MODEL="$T_MODEL" KB_TARGET="$T_TARGET" KB_ARM=prc KB_PRC_OVERRIDE="$T_WRAPPER" KB_TBL="$T_TBL" \
       KB_MASKS=64 KB_FRONTEND=awq KB_TAGSUFFIX="" KB_PARENT="$T_PARENT"
RUN_LOG="$CELL_DIR/full_test_run.log"
[[ ! -e "$RUN_LOG" ]] || { echo "Refusing to overwrite $RUN_LOG" >&2; exit 2; }
bash benchmark/ppl/kbands/run_prc_ppl.sbatch 2>&1 | tee "$RUN_LOG"
[[ -f "$T_TRACE" ]] || { echo "[r7scr] test trace missing: $T_TRACE" >&2; exit 3; }

# (4) Protocol verification + result JSON (CPU).
python -u "$DRIVER" finalize --manifest "$R7SCR_MANIFEST" --task "$R7SCR_TASK" --trace "$T_TRACE" --run-log "$RUN_LOG"
echo "[r7scr] task $TASK_ID finished"
