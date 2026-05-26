#!/bin/bash
# Dispatch the 5-model × 3-MP sweep across 5 GPU slots.
#
# Layout (decided 2026-05-26):
#   gl1802 / 50893492 / 128G  → Qwen3-4B-Instruct-2507
#   gl1803 / 50710402 / 512G  → Qwen3-8B
#   gl1806 / 50708292 / 512G  → Qwen3-30B-A3B-Instruct-2507  (bigger CPU side)
#   gl1806 / 50893493 / 128G  → Qwen3-14B
#   gl1809 / 50712265 / 512G  → Qwen3-32B  (starts AFTER 32B SQ calib lands)
#
# Each per-node script (_mp_overnight.sh) runs the 3 MP configs serially.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$HERE/_mp_overnight_${TS}"
mkdir -p "$OUTDIR"
echo "out=$OUTDIR" > "$HERE/_mp_overnight_latest.path"
echo "ts=$TS outdir=$OUTDIR"

launch_one() {
  local node="$1"; local jobid="$2"; local session="$3"; local model="$4"
  local wait_for="${5:-}"
  echo "[$(date)] dispatch  node=$node jobid=$jobid session=$session model=$model wait_for=${wait_for:-(none)}"
  local cmd
  if [[ -n "$wait_for" ]]; then
    cmd="until [[ -f '$wait_for' ]]; do sleep 30; done; bash $HERE/_mp_overnight.sh '$model' '$OUTDIR' '$TS'"
  else
    cmd="bash $HERE/_mp_overnight.sh '$model' '$OUTDIR' '$TS'"
  fi
  # On gl1806 we have two allocations; srun --jobid+--overlap puts the
  # process into the correct cgroup (right GPU UUID). For single-alloc
  # nodes plain ssh+tmux is fine, but srun --jobid is a harmless no-op.
  ssh -o BatchMode=yes "$node" \
    "tmux kill-session -t $session 2>/dev/null; \
     tmux new-session -d -s $session \"srun --jobid=$jobid --overlap bash -lc '$cmd'\""
}

launch_one gl1802 50893492 sweep_4b   Qwen/Qwen3-4B-Instruct-2507
launch_one gl1803 50710402 sweep_8b   Qwen/Qwen3-8B
launch_one gl1806 50708292 sweep_30b  Qwen/Qwen3-30B-A3B-Instruct-2507
launch_one gl1806 50893493 sweep_14b  Qwen/Qwen3-14B

# 32B waits for its SmoothQuant calibration to land act_scales on disk.
SQ32_PATH="$HERE/act_scales_Qwen_Qwen3-32B.pt"
launch_one gl1809 50712265 sweep_32b  Qwen/Qwen3-32B "$SQ32_PATH"

echo "[$(date)] all 5 sweep sessions dispatched."
echo "attach: ssh <node> -t tmux attach -t <session>"
echo "outdir: $OUTDIR"
