#!/bin/bash
#SBATCH --job-name=iswap_sens_4B
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_hpca_mp_v18_intswap_sens_20260720/%x_%j.out

# V18 Phase 1 — int_swap hybrid-mask ranking, 4B.
#
# Measures, per (operator, LAYER), the ΔLoss of putting THAT entry on INT7
# instead of SC, with every other entry on SC at its DEPLOYED calibrated
# per-row MP levels (the V9 avg32 hyb0.20 wrapper), mask OFF so every entry
# is swapped FROM SC. Replaces the shipped
# ranking, which scores SC precision-fragility at nominal 256->128 -- a
# different axis, two octaves above the operating point, and (measured
# 2026-07-20) noise-selected in 49% of the entries it picks.
#
# Produces a RANKING only. No MP table, no eval. 325 forward-groups
# (9 ops x 36 layers + 1 reference) x 8 windows x 512 ctx; est ~4.5h.
#
# GATE: the run prints SPLIT-HALF top-20% set overlap. Below ~70% the ranking
# is noise and Phase 2 must not be launched on it.

bash /home/allenjin/Projects/scmp_llm/hpca \
  --sensitivity \
  --models 4B \
  --configs mp_avg32_v9 \
  --sens-method int_swap \
  --sens-windows 8 \
  --sens-ctx 512 \
  --int-swap-ref mp_config \
  --int-swap-mp-config /nfs/turbo/coe-nbleier/allenjin/hpca/mp_calib_mp_v9_hybdose_4B_20260718_143110/Qwen_Qwen3-4B-Instruct-2507__mp_avg32_v9__act_global_v9_pc0.01x112_act_collapse_comp_pcovdown_proj_0_06_up_proj_0_03_gate_proj_0_03_hyb0.20_wrapper.json \
  --tag mp_v18_intswap_sens_20260720
