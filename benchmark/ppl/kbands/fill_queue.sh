#!/bin/bash
# Keep the owned GPUs busy. Builds any missing child bundles, then submits cells
# until at least KB_MIN_JOBS are queued/running.
#
# Idle GPUs are the biggest waste on owned hardware, so this is the fallback the
# loop calls whenever the queue drops below the floor. Every entry here is a
# genuine open question, NOT filler -- filler would burn the same GPU-hours and
# teach nothing.
cd /home/allenjin/Projects/scmp_llm
# NOTE: source ~/.bashrc with `set -u` OFF -- the module system references unset
# vars and would abort the whole script silently (that is why the first version
# submitted nothing and printed nothing).
set +u
source ~/.bashrc >/dev/null 2>&1
conda activate annstention >/dev/null 2>&1
set -u

D=/nfs/turbo/coe-nbleier/allenjin/hpca/kbands
Q=$D/kbands_20260801/qk
A=$D/kbands_20260801/alloc
MB=/home/allenjin/Projects/hpca_results/llm/ppl/mp_best/configs
MIN=${KB_MIN_JOBS:-4}

busy() { squeue -u allenjin -h 2>/dev/null | wc -l; }

build() {   # build <parent> <alloc> <out>
  [ -f "$3/wrapper.json" ] && return 0
  python -m benchmark.ppl.mp_kbands apply --parent "$1" --alloc "$2" --out "$3" \
    2>&1 | grep -E "wrote child|OVERSPEND|underspend|Error" | head -2
}

submit() {  # submit <name> <model> <label> <bundle> [qk_scales]
  [ "$(busy)" -ge "$MIN" ] && { echo "  [full] skip $1"; return 0; }
  # Never double-submit the same cell: re-running fill_queue while a cell is
  # already queued would burn a GPU on a duplicate, which is worse than idling
  # because it also looks like progress.
  squeue -u allenjin -h -o %j 2>/dev/null | grep -qx "$1" && {
    echo "  [already queued] skip $1"; return 0; }
  # ALSO skip cells that already have a RESULT. Checking only the queue meant a
  # completed cell got resubmitted the moment its job left squeue -- burning a
  # full GPU-hour to recompute a number already in the ledger, which is the
  # "no filler" rule violated by the very script meant to enforce it.
  local tr="$D/kbands_20260801/traces/$2_$3_trace.json"
  [ -f "$tr" ] && { echo "  [already done] skip $1  ($(basename $tr))"; return 0; }
  [ -f "$4/wrapper.json" ] || { echo "  [missing bundle] skip $1"; return 0; }
  # If a qk-scales path is NAMED but missing, REFUSE -- do not silently fall back
  # to a no-qk run. That would execute the plain parent and record it under a
  # "qk" label, i.e. a mislabeled result in the ledger, which is worse than not
  # running at all. (Hit this when a cell was submitted before its calib finished.)
  local qk=""
  if [ $# -ge 5 ]; then
    if [ -f "$5" ]; then qk="$5"
    else echo "  [scales missing] skip $1 -> $5"; return 0; fi
  fi
  # Use `env`, NOT a ${qk:+VAR=val} prefix: bash resolves assignment prefixes at
  # PARSE time, so an expanded one becomes a stray argument to sbatch and the
  # submit fails silently. That cost two cells the first time.
  local j
  if [ -n "$qk" ]; then
    j=$(env KB_MODEL="$2" KB_LABEL="$3" KB_FRONTEND=awq KB_BUNDLE="$4" KB_QK_SMOOTH="$qk" \
        sbatch --job-name="$1" --parsable benchmark/ppl/kbands/run_kband_cell.sbatch 2>&1 | tail -1)
  else
    j=$(env KB_MODEL="$2" KB_LABEL="$3" KB_FRONTEND=awq KB_BUNDLE="$4" \
        sbatch --job-name="$1" --parsable benchmark/ppl/kbands/run_kband_cell.sbatch 2>&1 | tail -1)
  fi
  echo "  submitted $1 -> ${j:-FAILED}"
}

echo "[fill] GPUs busy at start: $(busy)  (floor $MIN)"

# --- build the bundles the backlog needs (idempotent) ------------------------
build $MB/4B/target32      $A/4B_t32_max35_alloc.json      $D/bundles/4B_t32_s2_max35
build $MB/14B/target32     $A/14B_t32_hf0p25_alloc.json    $D/bundles/14B_t32_s2_hf0p25
build $MB/14B/target48     $A/14B_t48_hf0p25_alloc.json    $D/bundles/14B_t48_s2_hf0p25
build $MB/llama8B/target48 $A/llama8B_t48_hf0p25_alloc.json $D/bundles/llama8B_t48_s2_hf0p25
build $MB/llama8B/target32 $A/llama8B_t32_hf0p25_alloc.json $D/bundles/llama8B_t32_s2_hf0p25

# --- BACKLOG, most informative first ----------------------------------------
# finest legal granularity, never PPL-tested
submit kb_4B_awq_max35_t32      4B      awq_max35_t32     $D/bundles/4B_t32_s2_max35
submit kb_4B_awq_qk_max35_t32   4B      awq_qk_max35_t32  $D/bundles/4B_t32_s2_max35   $Q/4B_t32_qk_alpha1.0.json
# 14B: qk REGRESSES it, so bands are its only lever -- untested on AWQ
submit kb_14B_awq_band_t32      14B     awq_band_t32      $D/bundles/14B_t32_s2_hf0p25
submit kb_14B_awq_band_t48      14B     awq_band_t48      $D/bundles/14B_t48_s2_hf0p25
# llama8B: qk marginal alone; never combined with bands
submit kb_llama8B_awq_qk_band   llama8B awq_qk_band_t48   $D/bundles/llama8B_t48_s2_hf0p25 $Q/llama8B_t48_qk_alpha1.0.json
submit kb_llama8B_awq_band_t32  llama8B awq_band_t32      $D/bundles/llama8B_t32_s2_hf0p25
# 4B t32 alternates already built
submit kb_4B_awq_nb8_t32        4B      awq_nb8_t32       $D/bundles/4B_t32_s2_nb8

# --- ADDED 2026-08-03 when the backlog ran dry --------------------------------
# llama8B rung candidate: its t96 AWQ parent is 1.0557 and needs only −0.54% to
# pass 1.05x. qk gave −0.21% at t48; whether it gives more at the looser t96 is
# genuinely unknown (qk has paid MORE at TIGHTER budgets everywhere else, so
# this tests the trend in the direction it predicts should be weakest).
if [ ! -f "$Q/llama8B_t96_qk_alpha1.0.json" ] && [ "$(busy)" -lt "$MIN" ] \
   && ! squeue -u allenjin -h -o %j | grep -qx kb_qkcalib_llama8B_t96; then
  echo "  submitting llama8B t96 qk calib"
  env KB_JOB=qkcalib KB_MODEL=llama8B KB_TARGET=96 KB_ALPHA=1.0 \
    sbatch --job-name=kb_qkcalib_llama8B_t96 --parsable \
    benchmark/ppl/kbands/run_kband_tool.sbatch 2>&1 | tail -1
fi
submit kb_llama8B_awq_qk_t96   llama8B awq_qk_t96 \
  /home/allenjin/Projects/hpca_results/llm/ppl/mp_best/configs/llama8B/target96 \
  $Q/llama8B_t96_qk_alpha1.0.json

# 4B t40: the rung BETWEEN the two measured points. Fills in the rung curve and
# tests whether the stack scales smoothly with budget or has a knee.
if [ ! -f "$Q/4B_t40_qk_alpha1.0.json" ] && [ "$(busy)" -lt "$MIN" ] \
   && ! squeue -u allenjin -h -o %j | grep -qx kb_qkcalib_4B_t40; then
  echo "  submitting 4B t40 qk calib"
  env KB_JOB=qkcalib KB_MODEL=4B KB_TARGET=40 KB_ALPHA=1.0 \
    sbatch --job-name=kb_qkcalib_4B_t40 --parsable \
    benchmark/ppl/kbands/run_kband_tool.sbatch 2>&1 | tail -1
fi
submit kb_4B_awq_qk_t40        4B      awq_qk_t40 \
  /home/allenjin/Projects/hpca_results/llm/ppl/mp_best/configs/4B/target40 \
  $Q/4B_t40_qk_alpha1.0.json

echo "[fill] GPUs busy at end: $(busy)"
