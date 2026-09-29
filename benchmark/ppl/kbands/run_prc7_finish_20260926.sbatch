#!/bin/bash
# Complete all missing round-2 full-test cells. One GPU/task, at most four tasks.
# User rule 2026-09-26: final = best full-test PPL across comparable rounds.
# Held-out scores are retained, but do not suppress any round-2 test evaluation.
#SBATCH --job-name=p7finish_20260926
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=18:00:00
#SBATCH --array=0-13%4
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/p7finish_20260926_%A_%a.out
set -euo pipefail
cd /home/allenjin/Projects/SCMP/scmp_llm
mapfile -t CELLS < benchmark/ppl/kbands/prc7_finish_20260926.tsv
IDX=${SLURM_ARRAY_TASK_ID:?array task index required}
[[ "$IDX" =~ ^[0-9]+$ && "$IDX" -lt "${#CELLS[@]}" ]] || exit 2
IFS=$'\t' read -r MODE KB_MODEL KB_TARGET P6_TAG <<< "${CELLS[$IDX]}"

# Remove inherited experimental variants before setting the established protocol.
while IFS= read -r name; do unset "$name"; done < <(compgen -v SC_ || true)
unset KB_HYBRID
export KB_MODEL KB_TARGET P6_TAG
export P6_CUR=rel,fis P6_GRAD=full P2_EXPERTS=8 P6_HWIN=16
export P6_EVAL=gfisla P7_ONLY_IF_BETTER=0
export KB_MASKS=64 KB_FRONTEND=awq KB_TAGSUFFIX=""
export PPL_WINDOW_BATCH_SIZE=1 STRIDE=2048
OUTD=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2
export KB_PARENT="/home/allenjin/Projects/SCMP/hpca_results/llm/ppl/mp_best/configs/${KB_MODEL}/target${KB_TARGET}"
export P6_C17="$OUTD/${KB_MODEL}_t${KB_TARGET}_c17.json"
if [[ "$P6_TAG" == c7h20 ]]; then
  [[ "$KB_MODEL" == 14B && ( "$KB_TARGET" == 64 || "$KB_TARGET" == 96 ) ]] || exit 2
  export KB_PARENT="$OUTD/parents_h20/14B/target${KB_TARGET}"
  export P6_C17="$OUTD/${KB_MODEL}_t${KB_TARGET}_c17h20.json"
fi
export KB_ARM=prc KB_TBL="p2${P6_TAG}gfisla"
export KB_PRC_OVERRIDE="$OUTD/${KB_MODEL}_t${KB_TARGET}_${P6_TAG}_gfisla.json"
TRACE="/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/kbands_20260801/ppl/${KB_MODEL}_t${KB_TARGET}_prc_${KB_TBL}_trace.json"
export P7_FINISH_MODE="$MODE" P7_FINISH_TRACE="$TRACE"

# Validate inputs without importing GPU packages; also used for login-node dry runs.
python3 - <<'PY'
import json, os
from pathlib import Path
parent = Path(os.environ['KB_PARENT'])
for name in ('wrapper.json', 'table.json', 'hybrid_config.json'):
    json.loads((parent / name).read_text())
for path in [Path(os.environ['P6_C17'])] + (
        [Path(os.environ['KB_PRC_OVERRIDE'])] if os.environ['P7_FINISH_MODE'] == 'eval' else []):
    wrapper = json.loads(path.read_text())
    table = json.loads(Path(wrapper['threshold_table_path']).read_text())
    assert table.get('per_row_chunk', {}).get('buckets'), path
assert not Path(os.environ['P7_FINISH_TRACE']).exists(), 'Refusing to overwrite an existing trace'
if os.environ['P7_FINISH_MODE'] == 'calib':
    wrapper = Path(os.environ['KB_PRC_OVERRIDE'])
    stem = wrapper.name.removesuffix('_gfisla.json')
    assert not list(wrapper.parent.glob(stem + '_*.json')), 'Refusing to overwrite an existing calibration'
else:
    assert os.environ['P7_FINISH_MODE'] == 'eval'
print('[round2] mode={P7_FINISH_MODE} model={KB_MODEL} target={KB_TARGET} tag={P6_TAG} parent={KB_PARENT}'.format(**os.environ))
PY
[[ ${P7_FINISH_DRY_RUN:-0} == 1 ]] && exit 0

# Preserve archive-compatible per-cell logs alongside the scheduler's array logs.
LOGDIR=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands
if [[ "$MODE" == calib ]]; then
  CELLLOG="$LOGDIR/p7_${KB_MODEL}_t${KB_TARGET}_${P6_TAG}_${SLURM_JOB_ID}.out"
else
  CELLLOG="$LOGDIR/p7ev_${KB_MODEL}_t${KB_TARGET}_${P6_TAG}gfisla_${SLURM_JOB_ID}.out"
fi
[[ ! -e "$CELLLOG" ]] || { echo "Refusing to overwrite $CELLLOG" >&2; exit 2; }
exec > >(tee "$CELLLOG") 2>&1
echo "[round2] array=${SLURM_ARRAY_JOB_ID} task=$IDX job=${SLURM_JOB_ID} mode=$MODE model=$KB_MODEL target=$KB_TARGET tag=$P6_TAG"
echo "[round2] final selection: best full-test PPL across comparable rounds; held-out gate disabled"
if [[ "$MODE" == calib ]]; then
  bash benchmark/ppl/kbands/run_prc7.sbatch
else
  bash benchmark/ppl/kbands/run_prc_ppl.sbatch
fi
python3 - <<'PY'
import json, os
from pathlib import Path
h = json.loads(Path(os.environ['P7_FINISH_TRACE']).read_text())['header']
assert h['eval_tokens'] == (288627 if os.environ['KB_MODEL'] == 'llama8B' else 298862), h
assert h['ctx'] == 2048 and h['stride'] == 2048 and h['ppl_max_tokens'] == 0 and h['ppl_window_batch_size'] == 1, h
assert h['mp_config_json'] == os.environ['KB_PRC_OVERRIDE'], h
print(f"[round2] VERIFIED full-test PPL={h['ppl']:.8f} trace={os.environ['P7_FINISH_TRACE']}")
PY
