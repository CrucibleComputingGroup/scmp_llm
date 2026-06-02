#!/bin/bash
# Aggregate per-(model, MP) log lines into SUMMARY.txt.
# Run after all .done markers land — or anytime as a partial summary.
set -e
OUTDIR="${1:-$(cat /home/allenjin/Projects/scmp_llm/benchmark/ppl/_mp_overnight_latest.path 2>/dev/null | sed 's/^out=//')}"
[[ -z "$OUTDIR" ]] && { echo "Usage: $0 <outdir>"; exit 1; }
[[ ! -d "$OUTDIR" ]] && { echo "no such outdir: $OUTDIR"; exit 1; }

SUMMARY="$OUTDIR/SUMMARY.txt"
{
  echo "MP overnight sweep — $(date)"
  echo "outdir: $OUTDIR"
  echo "knobs:  halved (sl_max=128) + SQ@0.5 + SC_SCRAMBLE_RESCALE=1"
  echo "        per_row attention, sc_prec=8, Owen-in-rescale (PR #16)"
  echo
  printf "%-44s  %-8s  %10s  %10s  %10s  %8s  %s\n" \
    "model" "MP" "FP16" "PPL" "ratio" "avg_sl" "log"
  printf "%-44s  %-8s  %10s  %10s  %10s  %8s  %s\n" \
    "------------------------" "----" "----------" "----------" "----------" "------" "---"

  for log in "$OUTDIR"/*.log; do
    [[ -f "$log" ]] || continue
    base=$(basename "$log" .log)
    model="${base%_mp_*}"
    mp="${base##*_}"
    fp16=$(grep -E "^FP16 baseline " "$log" | awk '{print $3}' | tail -1)
    line=$(grep -E "^SC MP " "$log" | tail -1)
    if [[ -n "$line" ]]; then
      ppl=$(echo "$line" | awk '{
        for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+(\.[0-9]+)?$/) {print $i; exit}
      }')
      ratio=$(echo "$line" | grep -oE '×[0-9.]+ vs fp16' | tr -d '×' | awk '{print $1}')
      avg=$(echo "$line" | grep -oE 'avg_sl=[0-9.]+' | sed 's/avg_sl=//')
      printf "%-44s  %-8s  %10s  %10s  %10s  %8s  %s\n" \
        "$model" "$mp" "${fp16:--}" "${ppl:--}" "${ratio:--}" "${avg:--}" "$(basename "$log")"
    elif grep -qE "FAIL|Traceback" "$log"; then
      printf "%-44s  %-8s  %10s  %10s  %10s  %8s  %s\n" \
        "$model" "$mp" "${fp16:--}" "FAIL" "-" "-" "$(basename "$log")"
    else
      printf "%-44s  %-8s  %10s  %10s  %10s  %8s  %s\n" \
        "$model" "$mp" "${fp16:--}" "RUNNING" "-" "-" "$(basename "$log")"
    fi
  done

  echo
  echo "Done markers:"
  ls "$OUTDIR"/*.done 2>/dev/null | xargs -I{} basename {} || echo "  (none yet)"
} > "$SUMMARY"

cat "$SUMMARY"
