#!/bin/bash
# Round-7 STEP 0 (CRIT L3): measured budget-shift linear marginal on 30B, plus the per-cell screen-window
# references (exact parent cost P + INC 16-window profile) that the pre-registered target-scale fixed
# point replays round-7 tables on. TRAIN windows only; no full TEST; no table is created or edited.
# Kappa cells: INC identity (round-4 / round-6 references), PARENT + C17 probes on window 30720, c17 and
# c17_s80 (a pure -20% linear-budget shift) on the 16 round-6 diagnostic windows, INC identity again.
# Every cell: INC16 (profiled; bit identity vs the stored reference where one exists), PARENT16 (P),
# identity_after. Fixed AWQ / INT mask / RNG (bitrev, 64 masks, 128-level grid) / QK / escape.
# Driver: benchmark/ppl/prc_step0_r7_20260928.py.
#   task 0 = 30B_t40   kappa     (c17 / c17_s80)
#   task 1 = 30B_t32   kappa     (c17e32 / c17e32_s80)
#   task 2 = 30B_t48   reference (fresh t48 screen windows; INC16 becomes the screen reference)
#   task 3 = 4B_t64    reference (OPTIONAL: only with the user's OK to enable 4B t64 -> --array=0-3%1)
# ONE GPU at a time (%1). With 62208440 (1 GPU) and the capture array 0-1%2 (2 GPUs) the total is 4.
# CPU dry run (no writes, no Slurm):
#   R7S0_DRY_RUN=1 R7S0_TASK=<0..3> bash benchmark/ppl/kbands/run_prc_step0_r7_20260928.sbatch
#SBATCH --job-name=r7step0_20260928
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/r7step0_20260928_%A_%a.out
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
export R7S0_MANIFEST="${R7S0_MANIFEST:-$PWD/benchmark/ppl/kbands/prc_step0_r7_20260928.json}"
export R7S0_TASK=${SLURM_ARRAY_TASK_ID:-${R7S0_TASK:?array task index required}}
DRIVER=benchmark/ppl/prc_step0_r7_20260928.py
R7S0_ENV_PY=/nfs/turbo/coe-nbleier/allenjin/conda-envs/annstention/bin/python

# System-python preflight (no writes): frozen sources, frozen inputs, fresh output directories.
TASK_FIELDS=$(python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
m = json.loads(Path(os.environ['R7S0_MANIFEST']).read_text())
i = int(os.environ['R7S0_TASK'])
assert m['schema'] == 'prc-step0-r7-v1', m['schema']
assert m['max_total_gpus'] == 4 and m['array_gpu_cap'] == 1 and m['gpus_per_task'] == 1
assert [t['id'] for t in m['tasks']] == ['30B_t40', '30B_t32', '30B_t48', '4B_t64']
assert 0 <= i < len(m['tasks']) and m['tasks'][i]['index'] == i
assert m['output_root'] == '/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc_r7_20260928/step0'
assert m['protocol']['full_test'] is False and m['protocol']['qk_rebalance'] is False
code = m['code_hashes']
assert code and set(code) == set(m['source_files']), 'incomplete source freeze'
for p, h in code.items():
    assert sha(p) == h, f'experiment source changed: {p}'
assert sha(m['windows_registry']['path']) == m['windows_registry']['sha256'], 'window registry changed'
t = m['tasks'][i]
for cid in t['cells']:
    c = m['cells'][cid]
    for p, h in c['hashes'].items():
        assert sha(p) == h, f'frozen input changed: {p}'
    out = Path(m['output_root']) / cid
    assert not out.exists(), f'refusing to reuse an existing cell directory: {out}'
f = [t['id'], ' '.join(t['cells']), m['output_root']]
assert all('\t' not in x and '\n' not in x for x in f)
print('\t'.join(f))
PY
)
IFS=$'\t' read -r TASK_ID CELL_IDS R7S0_ROOT <<< "$TASK_FIELDS"
export TASK_ID CELL_IDS R7S0_ROOT
echo "[r7s0] verified frozen inputs: task=$TASK_ID cells=[$CELL_IDS] root=$R7S0_ROOT"
if [[ ${R7S0_DRY_RUN:-0} == 1 ]]; then
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels CUDA_VISIBLE_DEVICES="" "$R7S0_ENV_PY" "$DRIVER" \
    preflight --manifest "$R7S0_MANIFEST" --task "$R7S0_TASK" --deep > /dev/null
  echo "[r7s0] DRY RUN PASS task=$TASK_ID (launcher + driver deep preflight); nothing written"
  exit 0
fi

set +u
source ~/.bashrc
conda activate annstention
set -euo pipefail

# Freeze the numerical protocol after shell/conda initialization (as rounds 4 / 6).
while IFS= read -r name; do unset "$name"; done < <(compgen -v SC_ || true)
unset MP_CONFIG_JSON QUANT_CONFIG KB_PRC_OVERRIDE KB_HYBRID KB_QKALPHA KB_TAGSUFFIX
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 SC_HW_MAX_MASKS=64
export CTX=2048 STRIDE=2048 PPL_MAX_TOKENS=0 PPL_WINDOW_BATCH_SIZE=1 SQ_ALPHA=0.5
export FRONTEND=awq
export ACT_SCALES_DIR=/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/act_scales
export AWQ_SCALES_DIR="$ACT_SCALES_DIR/awq_scales" AWQ_OBJ_BITS=4
export PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_llm/kernels${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"

python -u "$DRIVER" preflight --manifest "$R7S0_MANIFEST" --task "$R7S0_TASK"
mkdir -p "$R7S0_ROOT" "$R7S0_ROOT/_logs"
for cid in $CELL_IDS; do
  mkdir "$R7S0_ROOT/$cid"   # atomic refusal if another task already owns this cell
done
exec > >(tee "$R7S0_ROOT/_logs/${TASK_ID}_${SLURM_ARRAY_JOB_ID:-nojob}_${R7S0_TASK}.log") 2>&1
echo "[r7s0] array=${SLURM_ARRAY_JOB_ID:-} task=$R7S0_TASK job=${SLURM_JOB_ID:-} id=$TASK_ID host=$(hostname)"
echo "[r7s0] one GPU (%1); with 62208440 + the capture array the total stays <= 4; TRAIN only, not citable"
python - <<'PY'
import datetime, hashlib, json, os
from pathlib import Path
mp = Path(os.environ['R7S0_MANIFEST'])
m = json.loads(mp.read_text())
code = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in m['source_files']}
assert code == m['code_hashes'], 'experiment source changed after preflight'
env_keys = sorted(k for k in os.environ if k.startswith(('SC_', 'AWQ_', 'CTX', 'STRIDE', 'PPL_', 'SQ_',
                                                         'FRONTEND', 'ACT_SCALES', 'PYTHONPATH', 'HF_')))
for cid in os.environ['CELL_IDS'].split():
    rec = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'manifest': str(mp), 'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
           'task': m['tasks'][int(os.environ['R7S0_TASK'])], 'cell': m['cells'][cid],
           'protocol': m['protocol'], 'kappa_rule': m['kappa_rule'], 'code_sha256': code,
           'env': {k: os.environ[k] for k in env_keys},
           'slurm': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID',
                                                    'SLURMD_NODENAME')}}
    with (Path(os.environ['R7S0_ROOT']) / cid / 'provenance.json').open('x') as f:
        json.dump(rec, f, indent=1)
        f.write('\n')
PY

srun python -u "$DRIVER" run --manifest "$R7S0_MANIFEST" --task "$R7S0_TASK" --out-root "$R7S0_ROOT"

# Independent post-check of every cell (fails the job if anything is missing).
python - <<'PY'
import json, math, os
from pathlib import Path
m = json.loads(Path(os.environ['R7S0_MANIFEST']).read_text())
for cid in os.environ['CELL_IDS'].split():
    d = Path(os.environ['R7S0_ROOT']) / cid
    assert not (d / 'failure.json').exists(), f'{cid}: failure.json present'
    s = json.loads((d / 'step0_summary.json').read_text())
    assert s['complete'] is True and s['simulated'] is False and s['kind'] == m['cells'][cid]['kind']
    assert s['identity'].get('restored_incumbent_exact') is True
    pr = json.loads((d / 'parent_reference.json').read_text())
    assert pr['audit_ok'] is True and pr['simulated'] is False and (d / 'INC16_profile.json').is_file()
    msg = f"[r7s0] {cid}: P={pr['P']:.5f} (audit {pr['audit_ok']}), INC16/P={pr['r_inc_over_parent']:.5f}"
    if m['cells'][cid]['kind'] == 'kappa':
        mm = json.loads((d / 'step0_measured.json').read_text())
        assert mm['ok'] is True and mm['simulated'] is False and mm['identity'].get('identity_mid_exact') is True
        p = mm['pair']
        assert all(math.isfinite(x) for x in p['window_dnll']) and len(p['window_dnll']) == 16
        msg += (f"; dNLL(s80-c17) {p['paired_s80_minus_c17']['mean_dnll']:+.6f} "
                f"(se {p['paired_s80_minus_c17']['se']:.6f}) at {p['dcycles_pct']:+.2f}% cycles; "
                f"{p['dnll_per_1pct_cycles']:+.6f} nats per +1% cycles")
    print(msg)
PY
echo "[r7s0] task $TASK_ID finished; the kappa decision needs the round-7 capture diags (driver stage 'kappa'),"
echo "[r7s0] the fixed point needs the s0 solves (benchmark/ppl/prc_fixedpoint_r7_20260928.py compute)"
