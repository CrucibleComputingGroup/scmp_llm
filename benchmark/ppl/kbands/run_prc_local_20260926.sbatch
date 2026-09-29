#!/bin/bash
# Local allocation pilot; starts after every round-2 array task has terminated.
# Three independent one-GPU cells. The dependency prevents overlap with round 2.
#SBATCH --job-name=p3local20260926
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --dependency=afterany:61982983
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/p3local20260926_%A_%a.out
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
export P3LOCAL_MANIFEST="$PWD/benchmark/ppl/kbands/prc_local_20260926.json"
export P3LOCAL_INDEX=${SLURM_ARRAY_TASK_ID:?array task index required}
export P3LOCAL_ROOT=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_local_20260926

# This preflight is CPU-only and performs no writes, including in dry-run mode.
CELL_FIELDS=$(python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
manifest = json.loads(Path(os.environ['P3LOCAL_MANIFEST']).read_text())
index = int(os.environ['P3LOCAL_INDEX'])
assert 0 <= index < len(manifest['cells']) == 3
cell = manifest['cells'][index]
for path, expected in cell['hashes'].items():
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    assert digest.hexdigest() == expected, f'Frozen source changed: {path}'
wrapper = json.loads(Path(cell['incumbent_wrapper']).read_text())
assert wrapper['threshold_table_path'] == cell['incumbent_table']
table = json.loads(Path(cell['incumbent_table']).read_text())
assert table.get('per_row_chunk', {}).get('buckets'), 'Incumbent has no per-group table'
parent = Path(cell['parent'])
assert str(parent / 'hybrid_config.json') == cell['hybrid_config']
assert json.loads((parent / 'table.json').read_text())['model_path'] == cell['model_path']
cell_dir = Path(os.environ['P3LOCAL_ROOT']) / cell['id']
assert not cell_dir.exists(), f'Refusing to overwrite pilot directory: {cell_dir}'
trace = Path('/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl') / (
    cell['id'] + '_prc_p3local20260926_trace.json')
assert not trace.exists(), f'Refusing to overwrite trace: {trace}'
fields = [cell['id'], cell['model'], str(cell['target']), cell['parent'],
          str(cell_dir), str(trace)]
assert all('\t' not in value and '\n' not in value for value in fields)
print('\t'.join(fields))
PY
)
IFS=$'\t' read -r CELL_ID KB_MODEL KB_TARGET KB_PARENT CELL_DIR P3LOCAL_TRACE <<< "$CELL_FIELDS"
export CELL_ID KB_MODEL KB_TARGET KB_PARENT CELL_DIR P3LOCAL_TRACE
echo "[local-pilot] verified frozen inputs: $CELL_ID; output=$CELL_DIR"
[[ ${P3LOCAL_DRY_RUN:-0} == 1 ]] && exit 0

set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail

# Freeze the existing experimental protocol after shell/conda initialization.
while IFS= read -r name; do unset "$name"; done < <(compgen -v SC_ || true)
unset KB_HYBRID KB_QKALPHA KB_PRC_OVERRIDE
export KB_ARM=prc KB_TBL=p3local20260926 KB_TAGSUFFIX=""
export KB_MASKS=64 KB_FRONTEND=awq
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 SC_HW_MAX_MASKS=64
export CTX=2048 STRIDE=2048 PPL_MAX_TOKENS=0 PPL_WINDOW_BATCH_SIZE=1 SQ_ALPHA=0.5
export FRONTEND=awq
export ACT_SCALES_DIR=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales
export AWQ_SCALES_DIR="$ACT_SCALES_DIR/awq_scales" AWQ_OBJ_BITS=4
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$P3LOCAL_ROOT"
mkdir "$CELL_DIR"  # Atomic refusal if another task already owns this cell.
exec > >(tee "$CELL_DIR/job.log") 2>&1
echo "[local-pilot] array=${SLURM_ARRAY_JOB_ID} task=$P3LOCAL_INDEX job=${SLURM_JOB_ID} cell=$CELL_ID"
echo "[local-pilot] afterany round 2; allocation only; fixed AWQ/mask/RNG; no QK transform"
echo "[local-pilot] final choice remains best full-test PPL across comparable rounds"
python - <<'PY'
import hashlib, json, os, datetime
from pathlib import Path
manifest_path = Path(os.environ['P3LOCAL_MANIFEST'])
manifest = json.loads(manifest_path.read_text())
cell = manifest['cells'][int(os.environ['P3LOCAL_INDEX'])]
code = [Path('benchmark/ppl/prc_local_refine.py'),
        Path('benchmark/ppl/prc_local_proposals.py'),
        Path('benchmark/ppl/kbands/run_prc_local_20260926.sbatch'),
        Path('benchmark/ppl/kbands/run_prc_ppl.sbatch')]
record = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'manifest': str(manifest_path),
          'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'cell': cell, 'protocol': manifest['protocol'],
          'code_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
          'slurm': {k: os.environ.get(k) for k in
                    ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID')}}
with (Path(os.environ['CELL_DIR']) / 'provenance.json').open('x') as output:
    json.dump(record, output, indent=2)
    output.write('\n')
PY

srun python -u benchmark/ppl/prc_local_refine.py \
  --manifest "$P3LOCAL_MANIFEST" --index "$P3LOCAL_INDEX" --out-dir "$CELL_DIR"

# The driver decides whether its shortlisted candidate merits a diagnostic test.
# Confirmation quality may be negative; only the explicit evaluate flag controls it.
CANDIDATE=$(python - <<'PY'
import json, os
from pathlib import Path
directory = Path(os.environ['CELL_DIR']).resolve()
selection = json.loads((directory / 'selected.json').read_text())
assert isinstance(selection.get('evaluate'), bool), 'Missing explicit evaluation decision'
if not selection['evaluate']:
    print('')
else:
    wrapper_path = Path(selection['wrapper']).resolve()
    assert wrapper_path.is_relative_to(directory), 'Candidate must be a new pilot artifact'
    wrapper = json.loads(wrapper_path.read_text())
    table_path = Path(wrapper['threshold_table_path']).resolve()
    assert table_path.is_relative_to(directory), 'Candidate table must be a new pilot artifact'
    table = json.loads(table_path.read_text())
    assert table.get('per_row_chunk', {}).get('buckets')
    print(wrapper_path)
PY
)
if [[ -z "$CANDIDATE" ]]; then
  echo "[local-pilot] no candidate requested a diagnostic full test; incumbent retained"
  exit 0
fi
export KB_PRC_OVERRIDE="$CANDIDATE"
[[ ! -e "$P3LOCAL_TRACE" ]] || { echo "Refusing to overwrite $P3LOCAL_TRACE" >&2; exit 2; }
bash benchmark/ppl/kbands/run_prc_ppl.sbatch

python - <<'PY'
import hashlib, json, os, math
from pathlib import Path
manifest = json.loads(Path(os.environ['P3LOCAL_MANIFEST']).read_text())
cell = manifest['cells'][int(os.environ['P3LOCAL_INDEX'])]
trace_path = Path(os.environ['P3LOCAL_TRACE'])
trace = json.loads(trace_path.read_text())
h = trace['header']
assert h['eval_tokens'] == (288627 if cell['model'] == 'llama8B' else 298862), h
assert h['ctx'] == h['stride'] == 2048 and h['ppl_max_tokens'] == 0, h
assert h['ppl_window_batch_size'] == 1 and h['owen_mode'] == 'bitrev', h
assert str(h['scramble_masks']) == '64' and h['sc_prec'] == 8 and h['sc_halve'] is True, h
assert h['mp_config_json'] == os.environ['KB_PRC_OVERRIDE'], h
groups = trace['groups']
macs = sum(float(g['macs']) for g in groups)
assert macs > 0
cost = sum(float(g['macs']) * float(g['stoc_len']) for g in groups) / macs
assert math.isfinite(h['ppl']) and h['ppl'] > 0
cost_change = cost / cell['incumbent_cost'] - 1
eligible = abs(cost_change) <= manifest['protocol']['cost_tolerance_fraction']
better = eligible and h['ppl'] < cell['incumbent_ppl']
wrapper = Path(os.environ['KB_PRC_OVERRIDE'])
table = Path(json.loads(wrapper.read_text())['threshold_table_path'])
record = {'candidate_wrapper': str(wrapper), 'candidate_ppl': h['ppl'],
          'candidate_cost': cost, 'incumbent_ppl': cell['incumbent_ppl'],
          'incumbent_cost': cell['incumbent_cost'], 'cost_change_fraction': cost_change,
          'within_cost_tolerance': eligible, 'candidate_beats_incumbent': better,
          'chosen_wrapper': str(wrapper) if better else cell['incumbent_wrapper'],
          'selection_scope': 'Pilot candidate versus its frozen best comparable incumbent; aggregate all rounds separately.',
          'trace': str(trace_path), 'trace_header': h,
          'candidate_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (wrapper, table)}}
with (Path(os.environ['CELL_DIR']) / 'full_test_result.json').open('x') as output:
    json.dump(record, output, indent=2)
    output.write('\n')
print(f"[local-pilot] VERIFIED PPL={h['ppl']:.8f} cost={cost:.6f}; "
      f"cost change={cost_change:+.3%}; improves eligible incumbent={better}")
PY
