#!/bin/bash
#SBATCH --job-name=iswap_sens_30B
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_hpca_mp_v18_intswap_sens_20260720/%x_%j.out

# V18 Phase 1 — int_swap hybrid-mask ranking, 30B (Qwen3-30B-A3B, MoE).
#
# Same measurement as the 4B / llama8B cells: per (operator, LAYER), the ΔLoss
# of putting THAT entry on INT7 instead of SC, every other entry on SC at its
# deployed per-row MP levels (V9 avg32 hyb0.20 wrapper), mask OFF.
#
# TWO DELIBERATE DEVIATIONS from the 4B / llama8B cells:
#
# 1. --sens-windows 4 (they use 8). Measured cost on 4B is ~102 s per entry at
#    8 windows; 30B forwards run ~3x slower and it has 432 entries, so 8 windows
#    projects to ~37 h -- past any walltime we can request. 4 windows x 512 ctx
#    projects to ~18.3 h inside a 24 h request. Per-window regime (512 ctx) is
#    held IDENTICAL to the other models so gains stay comparable across models;
#    only the count drops. Consequence: the split-half gate gets 2 windows per
#    half instead of 4, so treat 30B's stability number as weak evidence.
#
# 2. --hybrid-int-bits 7 pinned EXPLICITLY. As of 2026-07-21
#    hybrid_int_bits_for_cfg derives the mask width from the config budget, and
#    mp_avg32_v9 now correctly derives int6 (halved 32 -> nominal 64). This wave
#    must stay at int7 to remain comparable with the 2026-07-18 dose anchors and
#    with the 4B / llama8B cells already running. Without this flag 30B would be
#    measured at a different width than its own wave.
#
# MoE note: the hybrid schedule keys on (op, block) only -- _sc_unit_idx is not
# part of the key -- so one masked MLP entry flips all 128 experts at that
# layer. Activated MACs are still counted per routed (token, expert) pair
# (verified: expert rows = 8x q_proj rows = top-k).
#
# No partial output: the ranking JSON is written only at the end of the sweep.

bash /home/allenjin/Projects/scmp_llm/hpca \
  --sensitivity \
  --models 30B \
  --configs mp_avg32_v9 \
  --sens-method int_swap \
  --sens-windows 4 \
  --sens-ctx 512 \
  --hybrid-int-bits 7 \
  --int-swap-ref mp_config \
  --int-swap-mp-config /nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_mp_v9_hybdose_all_20260718_183035/Qwen_Qwen3-30B-A3B-Instruct-2507__mp_avg32_v9__act_global_v9_pc0.01x112_act_collapse_comp_pcovdown_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.20_wrapper.json \
  --tag mp_v18_intswap_sens_20260720
