#!/bin/bash
# Round-6 attention diagnostic (investigation section 6 Q1 + Q9). TRAIN windows only;
# no full TEST. Pure allocation arms (table edits); fixed AWQ / INT mask / RNG / QK.
#   task 0 = 30B_t40  [INC, A, ATT128, LIN128]
#   task 1 = 30B_t32  [INC, A, ATT128]
#   task 2 = 4B_dense [4B_t40: INC, A, ATT128; then 4B_t64: INC, A, ATT128]
# One GPU per task; this array is throttled to TWO concurrent GPUs. Another
# round-6 array of at most two GPUs may run at the same time, so the total stays
# within the user's FOUR-GPU limit.
# CPU dry run (no writes, no Slurm):
#   R6DIAG_DRY_RUN=1 R6DIAG_TASK=<0|1|2> bash benchmark/ppl/kbands/run_prc_r6_attn_diag_20260928.sbatch
#SBATCH --job-name=p6diag20260928
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/p6diag20260928_%A_%a.out
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
export R6DIAG_MANIFEST="$PWD/benchmark/ppl/kbands/prc_r6_attn_diag_20260928.json"
export R6DIAG_TASK=${SLURM_ARRAY_TASK_ID:-${R6DIAG_TASK:?array task index required}}
R6DIAG_ENV_PY=/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python

# CPU-only preflight with the system python: frozen sources, frozen inputs, arm
# files, fresh output directories. It performs no writes, including in dry-run mode.
TASK_FIELDS=$(python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
manifest = json.loads(Path(os.environ['R6DIAG_MANIFEST']).read_text())
index = int(os.environ['R6DIAG_TASK'])
assert manifest['schema'] == 'prc-r6-attn-diag-v1', manifest['schema']
assert manifest['max_total_gpus'] == 4 and manifest['array_gpu_cap'] == 2
assert [t['id'] for t in manifest['tasks']] == ['30B_t40', '30B_t32', '4B_dense']
assert 0 <= index < len(manifest['tasks']) and manifest['tasks'][index]['index'] == index
assert manifest['output_root'] == '/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r6_20260928'
assert manifest['protocol']['full_test'] is False and manifest['protocol']['qk_rebalance'] is False
code = manifest['code_hashes']
assert code and set(code) == set(manifest['source_files']), 'incomplete source freeze'
for path, expected in code.items():
    assert sha(path) == expected, f'experiment source changed: {path}'
task = manifest['tasks'][index]
dirs = []
for cid in task['cells']:
    cell = manifest['cells'][cid]
    for path, expected in cell['hashes'].items():
        assert sha(path) == expected, f'frozen input changed: {path}'
    for arm in ['INC'] + cell['arm_order']:
        entry = cell['arms'][arm]
        assert entry['wrapper'] in cell['hashes'] and entry['table'] in cell['hashes'], arm
        assert json.loads(Path(entry['wrapper']).read_text())['threshold_table_path'] == entry['table']
    assert cell['arm_order'][0] == 'A'
    out = Path(manifest['output_root']) / cid
    assert not out.exists(), f'refusing to reuse an existing cell directory: {out}'
    dirs.append(cid)
fields = [task['id'], ' '.join(dirs), manifest['output_root']]
assert all('\t' not in f and '\n' not in f for f in fields)
print('\t'.join(fields))
PY
)
IFS=$'\t' read -r TASK_ID CELL_IDS R6DIAG_ROOT <<< "$TASK_FIELDS"
export TASK_ID CELL_IDS R6DIAG_ROOT
echo "[r6diag] verified frozen inputs: task=$TASK_ID cells=[$CELL_IDS] root=$R6DIAG_ROOT"
if [[ ${R6DIAG_DRY_RUN:-0} == 1 ]]; then
  # Deep CPU preflight in the experiment env (resolver sweep; no model load, no writes).
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels "$R6DIAG_ENV_PY" \
    benchmark/ppl/prc_r6_attn_diag.py --manifest "$R6DIAG_MANIFEST" --task "$R6DIAG_TASK" \
    --preflight --deep > /dev/null
  echo "[r6diag] DRY RUN PASS task=$TASK_ID (launcher + driver deep preflight); nothing written"
  exit 0
fi

set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail

# Freeze the numerical protocol after shell/conda initialization (same as round 4).
while IFS= read -r name; do unset "$name"; done < <(compgen -v SC_ || true)
unset MP_CONFIG_JSON QUANT_CONFIG KB_PRC_OVERRIDE KB_HYBRID KB_QKALPHA
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 SC_HW_MAX_MASKS=64
export CTX=2048 STRIDE=2048 PPL_MAX_TOKENS=0 PPL_WINDOW_BATCH_SIZE=1 SQ_ALPHA=0.5
export FRONTEND=awq
export ACT_SCALES_DIR=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales
export AWQ_SCALES_DIR="$ACT_SCALES_DIR/awq_scales" AWQ_OBJ_BITS=4
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$R6DIAG_ROOT" "$R6DIAG_ROOT/_attn_diag_logs"
for cid in $CELL_IDS; do
  mkdir "$R6DIAG_ROOT/$cid"  # atomic refusal if another task already owns this cell
done
exec > >(tee "$R6DIAG_ROOT/_attn_diag_logs/${TASK_ID}_${SLURM_ARRAY_JOB_ID:-nojob}_${R6DIAG_TASK}.log") 2>&1
echo "[r6diag] array=${SLURM_ARRAY_JOB_ID:-} task=$R6DIAG_TASK job=${SLURM_JOB_ID:-} id=$TASK_ID cells=[$CELL_IDS] host=$(hostname)"
echo "[r6diag] this array <=2 GPUs (throttle %2); aggregate with the other round-6 array <=4"
echo "[r6diag] TRAIN-only diagnostic; ATT128/LIN128 attribution-only; no arm is a best(all) candidate"
python - <<'PY'
import datetime, hashlib, json, os
from pathlib import Path
manifest_path = Path(os.environ['R6DIAG_MANIFEST'])
manifest = json.loads(manifest_path.read_text())
code = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in manifest['source_files']}
assert code == manifest['code_hashes'], 'experiment source changed after preflight'
env_keys = sorted(k for k in os.environ if k.startswith(('SC_', 'AWQ_', 'CTX', 'STRIDE', 'PPL_', 'SQ_',
                                                         'FRONTEND', 'ACT_SCALES', 'PYTHONPATH', 'HF_')))
for cid in os.environ['CELL_IDS'].split():
    record = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'manifest': str(manifest_path),
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'task': manifest['tasks'][int(os.environ['R6DIAG_TASK'])],
              'cell': manifest['cells'][cid], 'protocol': manifest['protocol'],
              'preregistered_rules': manifest['preregistered_rules'],
              'code_sha256': code, 'env': {k: os.environ[k] for k in env_keys},
              'slurm': {k: os.environ.get(k) for k in
                        ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURMD_NODENAME')}}
    with (Path(os.environ['R6DIAG_ROOT']) / cid / 'provenance.json').open('x') as out:
        json.dump(record, out, indent=1)
        out.write('\n')
PY

srun python -u benchmark/ppl/prc_r6_attn_diag.py \
  --manifest "$R6DIAG_MANIFEST" --task "$R6DIAG_TASK" --out-root "$R6DIAG_ROOT"

# Independent post-check of every cell summary (fails the job if anything is missing).
python - <<'PY'
import json, math, os
from pathlib import Path
for cid in os.environ['CELL_IDS'].split():
    s = json.loads((Path(os.environ['R6DIAG_ROOT']) / cid / 'diag_summary.json').read_text())
    assert s['complete'] is True and s['simulated'] is False and s['cell'] == cid
    assert s['identity'].get('restored_incumbent_exact') is True
    for arm, r in s['arms'].items():
        assert r['audit']['ok'] is True and r['best_all_candidate'] is False, arm
        assert all(math.isfinite(r['metrics'][k]) for k in ('dnll', 'se_dnll', 'dppl_pct', 'extra_cycles_pct'))
    g = s['round7_gate']['binding']
    print(f"[r6diag] {cid}: Round-7 proceed={g['proceed_round7_attention_ladder_solve']} "
          f"dPPL(A)={g['dppl_pct_A']:+.3f}% threshold={g['threshold_dppl_pct']:+.3f}% "
          f"dNLL(A)={g['dnll_A']:+.5f} (<= {g['gross_dnll_threshold']}) extra={g['extra_cycles_pct']:+.2f}%")
PY
echo "[r6diag] task $TASK_ID finished"
