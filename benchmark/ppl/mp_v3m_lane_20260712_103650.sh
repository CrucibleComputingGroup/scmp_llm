#!/bin/bash
set -o pipefail
set +u; source ~/.bashrc; set -u 2>/dev/null || true
conda activate annstention
cd /home/allenjin/Projects/scmp_llm
bash hpca --models llama8B --configs mp_avg64f_burst128 --metrics ppl \
  --mp-method act_global_v3m --mp-protect-metric act_collapse \
  --sc-backend hybrid --hybrid-int-frac 0.10 \
  --hybrid-sensitivity-dir /nfs/turbo/coe-nbleier/allenjin/hpca/sensitivity/_hpca_sens_layer_int8_20260709_012921 \
  --hybrid-sensitivity-config sc_int8 --tag mp_v3m_llama_20260712_103650
