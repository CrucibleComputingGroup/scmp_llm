# Shared environment setup for MP overnight runs.
# Sourced by every per-job launcher.
source ~/.bashrc
conda activate annstention

# Turbo for read-mostly HF cache (10x faster cold reads vs GPFS).
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=/nfs/turbo/coe-nbleier/allenjin/hf_datasets

# Scratch for write-heavy temp (pip, builds, GPU compile cache).
mkdir -p /scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/tmp
export TMPDIR=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/tmp

# SC kernel knobs locked in for the MP sweep.
# (SC_DISABLE_OWEN / SC_SCRAMBLE_RESCALE were removed from the kernel:
#  scramble-before-rescale is always on; SC_OWEN_MODE=off disables scrambling.)
# Bitrev mask count M = min(SC_SCRAMBLE_MASKS, 2^sc_prec); kernel default 64.
# Tables produced before 2026-06-03 (incl. _mp_overnight_xlayer_fix) ran at
# M=256 — export SC_SCRAMBLE_MASKS=256 to reproduce/patch them.

cd /home/allenjin/Projects/scmp_llm
