#!/bin/bash
# Full-protocol PPL for one prc2 table:  launch_prc2_ppl.sh <model> <target> <tag>
# Uses run_prc_ppl.sbatch (KB_ARM=prc) with KB_PRC_OVERRIDE -> the prc2 wrapper.
# Job name p2ppl_<m>_t<T>_<tag> (never prcppl_*, so the archive builders that
# glob prcppl_* are not shadowed); trace tag <m>_t<T>_prc_p2<tag> (no overwrite).
set -euo pipefail
m=$1; t=$2; tag=$3
W=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2/${m}_t${t}_${tag}.json
[[ -f "$W" ]] || { echo "missing $W" >&2; exit 2; }
cd /home/allenjin/Projects/SCMP/scmp_llm
declare -A MEM=([4B]=64G [llama8B]=80G [14B]=120G [30B]=180G)
# editable scmp_kernels install points at the pre-move path; see PRC2_OVERNIGHT.md
PP=/home/allenjin/Projects/SCMP/scmp_llm/kernels
sbatch --mem=${MEM[$m]} --job-name="p2ppl_${m}_t${t}_${tag}" \
  --export=ALL,PYTHONPATH=$PP,KB_MODEL=$m,KB_TARGET=$t,KB_ARM=prc,KB_PRC_OVERRIDE=$W,KB_TBL=p2${tag} \
  benchmark/ppl/kbands/run_prc_ppl.sbatch
