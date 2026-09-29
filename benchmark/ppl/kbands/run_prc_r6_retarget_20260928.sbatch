#!/bin/bash
# Round-6 BUDGET RE-TARGET (investigation RC6 / section 6 Q3). Pure allocation: the incumbent's
# own calib7 recipe (same seed 0, calibration windows, Fisher currency, dense ladder, deploy
# rule, INT mask, AWQ, RNG, attention ladder + mu+2tau escape) re-solved at s x the parent's
# calibration-sample cost, s chosen so the table spends the PARENT's cost on held-out TRAIN
# windows. No QK rebalance, no smooth-scale / escape / ladder / RNG / runtime change.
#   task 0 = 14B_t32  (incumbent c7gfisla, joint linear+attention solve)
#   task 1 = 30B_t64  (incumbent c7gfis, linear-only solve)
# Chain per task (driver = benchmark/ppl/prc_r6_retarget_20260928.py):
#   preflight -> stage-a [GPU: parent + incumbent on the 16 stored held-out TRAIN windows with
#   exact traces; bit-exact NLL identity; s = (P-U)/(C1-U)] -> calib9_r6 [GPU capture, exact
#   incumbent args + --target-scales 1.0,s --solve-state] -> calib-identity [CPU: s=1.0 table
#   == incumbent; s table allocation-only; CPU re-solve reproduces both] -> stage-b [GPU:
#   held-out cost gate |C(s)/P-1| <= 0.3%] -> full-protocol TEST (run_prc_ppl.sbatch; full
#   WikiText-2 test, ctx 2048, PPL_MAX_TOKENS=0, batch 1) -> finalize (result JSON).
# One GPU per task, throttle %2. The round-6 attention-diagnostic array is also capped at two
# GPUs, so the two round-6 arrays together stay within the user's FOUR-GPU limit.
# CPU dry run on the login node (no Slurm, no writes except a scratch argv check):
#   R6TS_DRY_RUN=1 R6TS_INDEX=<0|1> bash benchmark/ppl/kbands/run_prc_r6_retarget_20260928.sbatch
#SBATCH --job-name=r6retarget20260928
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/r6retarget20260928_%A_%a.out
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
export R6TS_MANIFEST="$PWD/benchmark/ppl/kbands/prc_r6_retarget_20260928.json"
export R6TS_INDEX=${SLURM_ARRAY_TASK_ID:-${R6TS_INDEX:?array task index required}}
DRIVER=benchmark/ppl/prc_r6_retarget_20260928.py
R6TS_ENV_PY=/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python

# Manifest fields (system python3, read-only).
CELL_FIELDS=$(python3 - <<'PY'
import json, os
from pathlib import Path
m = json.loads(Path(os.environ['R6TS_MANIFEST']).read_text())
i = int(os.environ['R6TS_INDEX'])
assert m['schema'] == 'prc-r6-retarget-v1', m['schema']
assert m['max_total_gpus'] == 4 and m['array_gpu_cap'] == 2
assert [c['id'] for c in m['cells']] == ['14B_t32', '30B_t64']
assert 0 <= i < len(m['cells']) and m['cells'][i]['index'] == i
c = m['cells'][i]
assert c['cell_dir'] == m['output_root'] + '/' + c['id']
assert m['output_root'] == '/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r6_retarget_20260928'
f = [c['id'], c['model'], str(c['target']), c['parent'], c['cell_dir'], c['test_trace'], c['kb_tbl']]
assert all('\t' not in x and '\n' not in x and x for x in f)
print('\t'.join(f))
PY
)
IFS=$'\t' read -r CELL_ID KB_MODEL KB_TARGET KB_PARENT CELL_DIR TEST_TRACE KB_TBL <<< "$CELL_FIELDS"
export CELL_ID KB_MODEL KB_TARGET KB_PARENT CELL_DIR TEST_TRACE KB_TBL
# run_prc_ppl.sbatch names its trace <OUT>/<model>_t<target>_prc_<KB_TBL>_trace.json; check here.
[[ "$TEST_TRACE" == "/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl/${KB_MODEL}_t${KB_TARGET}_prc_${KB_TBL}_trace.json" ]] \
  || { echo "[r6ts] manifest test_trace does not match run_prc_ppl.sbatch naming" >&2; exit 2; }
echo "[r6ts] cell=$CELL_ID model=$KB_MODEL target=$KB_TARGET out=$CELL_DIR test_trace=$TEST_TRACE"

if [[ ${R6TS_DRY_RUN:-0} == 1 ]]; then
  # Full hash preflight + the exact calib9 argv (at a placeholder scale) parsed by calib9's
  # REAL argument parser with CUDA hidden (it prints "SKIP: needs CUDA" and exits 0).
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels "$R6TS_ENV_PY" "$DRIVER" preflight \
    --manifest "$R6TS_MANIFEST" --index "$R6TS_INDEX"
  mapfile -d '' -t CARGS < <(PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels \
    "$R6TS_ENV_PY" "$DRIVER" calib-argv --manifest "$R6TS_MANIFEST" --index "$R6TS_INDEX" \
    --dry-scale 1.0150)
  echo "[r6ts] calib9 argv: ${CARGS[*]}"
  OUTP=$(CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels \
    "$R6TS_ENV_PY" benchmark/ppl/mp_per_row_chunk_calib9_r6.py "${CARGS[@]}")
  [[ "$OUTP" == "SKIP: needs CUDA" ]] || { echo "[r6ts] calib9 argv check failed: $OUTP" >&2; exit 2; }
  [[ ! -e "$CELL_DIR" && ! -e "$TEST_TRACE" ]] || { echo "[r6ts] outputs already exist" >&2; exit 2; }
  echo "[r6ts] DRY RUN PASS $CELL_ID (preflight + calib9 argv parse); nothing written"
  exit 0
fi

set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail

# Freeze the numerical protocol after shell/conda initialization (as rounds 4 / r6-diag).
while IFS= read -r name; do unset "$name"; done < <(compgen -v SC_ || true)
unset MP_CONFIG_JSON QUANT_CONFIG KB_PRC_OVERRIDE KB_HYBRID KB_QKALPHA KB_TAGSUFFIX
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64
export CTX=2048 STRIDE=2048 PPL_MAX_TOKENS=0 PPL_WINDOW_BATCH_SIZE=1 SQ_ALPHA=0.5
export FRONTEND=awq
export ACT_SCALES_DIR=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales
export AWQ_SCALES_DIR="$ACT_SCALES_DIR/awq_scales" AWQ_OBJ_BITS=4
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"

python -u "$DRIVER" preflight --manifest "$R6TS_MANIFEST" --index "$R6TS_INDEX"
mkdir -p "$(dirname "$CELL_DIR")"
mkdir "$CELL_DIR"   # atomic refusal if the cell already ran
exec > >(tee "$CELL_DIR/job.log") 2>&1
echo "[r6ts] array=${SLURM_ARRAY_JOB_ID:-} task=$R6TS_INDEX job=${SLURM_JOB_ID:-} cell=$CELL_ID host=$(hostname)"
echo "[r6ts] budget re-target only; <=2 GPUs in this array; fixed AWQ/mask/RNG/QK/escape/ladders"
python - <<'PY'
import datetime, hashlib, json, os
from pathlib import Path
mp = Path(os.environ['R6TS_MANIFEST'])
m = json.loads(mp.read_text())
code = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in m['source_files']}
assert code == m['code_hashes'], 'experiment source changed after preflight'
env_keys = sorted(k for k in os.environ if k.startswith(('SC_', 'AWQ_', 'CTX', 'STRIDE', 'PPL_', 'SQ_',
                                                         'FRONTEND', 'ACT_SCALES', 'PYTHONPATH', 'HF_')))
rec = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'manifest': str(mp), 'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
       'cell': m['cells'][int(os.environ['R6TS_INDEX'])], 'protocol': m['protocol'],
       'preregistered_rules': m['preregistered_rules'], 'code_sha256': code,
       'env': {k: os.environ[k] for k in env_keys},
       'slurm': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID',
                                                'SLURM_ARRAY_TASK_ID', 'SLURMD_NODENAME')}}
with (Path(os.environ['CELL_DIR']) / 'provenance.json').open('x') as f:
    json.dump(rec, f, indent=1)
    f.write('\n')
PY

# (1) Held-out TRAIN identity + scale (GPU).
srun python -u "$DRIVER" stage-a --manifest "$R6TS_MANIFEST" --index "$R6TS_INDEX" --cell-dir "$CELL_DIR"

# (2) calib9_r6 capture with the incumbent's exact recorded arguments (GPU).
mapfile -d '' -t CARGS < <(python "$DRIVER" calib-argv --manifest "$R6TS_MANIFEST" \
  --index "$R6TS_INDEX" --cell-dir "$CELL_DIR")
[[ ${#CARGS[@]} -gt 10 ]] || { echo "[r6ts] empty calib9 argv" >&2; exit 2; }
echo "[r6ts] calib9 argv: ${CARGS[*]}"
srun python -u benchmark/ppl/mp_per_row_chunk_calib9_r6.py "${CARGS[@]}"

# (3) Calibration identity + allocation-only + solve-state round trip (CPU).
python -u "$DRIVER" calib-identity --manifest "$R6TS_MANIFEST" --index "$R6TS_INDEX" --cell-dir "$CELL_DIR"

# (4) Held-out TRAIN cost gate (GPU). Writes selected.json.
srun python -u "$DRIVER" stage-b --manifest "$R6TS_MANIFEST" --index "$R6TS_INDEX" --cell-dir "$CELL_DIR"
CANDIDATE=$(python - <<'PY'
import json, os
from pathlib import Path
d = Path(os.environ['CELL_DIR']).resolve()
s = json.loads((d / 'selected.json').read_text())
assert isinstance(s.get('evaluate'), bool)
if not s['evaluate']:
    print('')
else:
    w = Path(s['wrapper']).resolve()
    assert w.is_relative_to(d), 'candidate must be a new round-6 artifact'
    t = Path(json.loads(w.read_text())['threshold_table_path']).resolve()
    assert t.is_relative_to(d) and json.loads(t.read_text())['per_row_chunk']['buckets']
    print(w)
PY
)
if [[ -z "$CANDIDATE" ]]; then
  echo "[r6ts] held-out cost gate not met (see stage_b.json); stopped before the full test"
  exit 0
fi

# (5) Full-protocol TEST (same eval settings as the incumbent: ctx/stride 2048, batch 1, masks 64).
[[ ! -e "$TEST_TRACE" ]] || { echo "Refusing to overwrite $TEST_TRACE" >&2; exit 2; }
export KB_ARM=prc KB_PRC_OVERRIDE="$CANDIDATE" KB_MASKS=64 KB_FRONTEND=awq KB_TAGSUFFIX=""
bash benchmark/ppl/kbands/run_prc_ppl.sbatch
[[ -f "$TEST_TRACE" ]] || { echo "[r6ts] test trace missing: $TEST_TRACE" >&2; exit 3; }

# (6) Protocol verification + result JSON (CPU).
python -u "$DRIVER" finalize --manifest "$R6TS_MANIFEST" --index "$R6TS_INDEX" \
  --cell-dir "$CELL_DIR" --trace "$TEST_TRACE"
echo "[r6ts] task $CELL_ID finished"
