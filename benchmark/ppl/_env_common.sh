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
unset SC_DISABLE_OWEN          # Owen-in-rescale scramble must be enabled.
export SC_SCRAMBLE_RESCALE=1   # PR #16 path: scramble in rescale, not RNG.

cd /home/allenjin/Projects/scmp_llm
