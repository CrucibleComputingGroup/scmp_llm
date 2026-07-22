#!/bin/bash
#SBATCH --job-name=iswap_sens_14B
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_hpca_mp_v18_intswap_sens_20260720/%x_%j.out

# V18 Phase 1 — int_swap hybrid-mask ranking, 14B. Completes the 4-model set
# (4B + llama8B running, 30B queued).
#
# Per (operator, LAYER), the ΔLoss of putting THAT entry on INT7 instead of SC,
# every other entry on SC at its deployed per-row MP levels (V9 avg32 hyb0.20
# wrapper), mask OFF so every entry is swapped FROM SC.
#
# Windows held at 8, identical to the 4B / llama8B cells, so gains and the
# split-half gate are directly comparable. 360 entries x 8 windows at ~1.44x
# 4B's measured per-forward cost projects to ~14.7 h -- past the 12 h used for
# the small models, hence the 24 h request.
#
# --hybrid-int-bits 7 pinned EXPLICITLY: as of 2026-07-21
# hybrid_int_bits_for_cfg derives the width from the config budget, and
# mp_avg32_v9 now correctly derives int6 (halved 32 -> nominal 64). This wave
# stays at int7 to remain comparable with the 2026-07-18 dose anchors and with
# the other three cells. Without the flag this cell would be measured at a
# different width than its own wave.
#
# No partial output: the ranking JSON is written only at the end of the sweep.

bash /home/allenjin/Projects/scmp_llm/hpca \
  --sensitivity \
  --models 14B \
  --configs mp_avg32_v9 \
  --sens-method int_swap \
  --sens-windows 8 \
  --sens-ctx 512 \
  --hybrid-int-bits 7 \
  --int-swap-ref mp_config \
  --int-swap-mp-config /nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_mp_v9_hybdose_all_20260718_183035/Qwen_Qwen3-14B__mp_avg32_v9__act_global_v9_pc0.01x112_act_collapse_comp_pcovdown_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.20_wrapper.json \
  --tag mp_v18_intswap_sens_20260720
