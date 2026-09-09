#!/usr/bin/env bash
# View progress of the detached scripts/gen_defense_data484.py run.
#
#   ./scripts/gen_defense_data484.py.watch.sh        one snapshot
#   ./scripts/gen_defense_data484.py.watch.sh -w     refresh every 30s
#   ./scripts/gen_defense_data484.py.watch.sh -f     tail the raw log
#
# The run writes to datasets_defense_484/run.log, but its live progress bar is
# suppressed when stdout is not a terminal, so the numbers below are derived
# from the on-disk manifests and generated files instead.

set -uo pipefail
cd /root/mats12

OUT=datasets_defense_484
LOG=$OUT/run.log
PIDF=$OUT/run.pid
TOTAL=484

snapshot() {
    local pid status elapsed processed baseline done_now failed low high rate eta remaining

    pid=$(cat "$PIDF" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        status="RUNNING (pid $pid)"
        elapsed=$(ps -p "$pid" -o etimes= 2>/dev/null | tr -d ' ')
    else
        status="NOT RUNNING${pid:+ (last pid $pid)}"
        elapsed=""
    fi

    processed=$(python3 -c "import json;print(len(json.load(open('$OUT/processed_questions.json'))))" 2>/dev/null || echo 0)
    failed=$(python3 -c "import json;print(len(json.load(open('$OUT/failed_questions.json'))))" 2>/dev/null || echo 0)
    low=$(ls "$OUT"/*_gullibility_low.txt 2>/dev/null | wc -l)
    high=$(ls "$OUT"/*_gullibility_high.txt 2>/dev/null | wc -l)

    # questions already done when the current run started, per its own log
    baseline=$(grep -o 'Resuming: [0-9]* question' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9]*')
    baseline=${baseline:-0}
    done_now=$(( processed - baseline ))
    remaining=$(( TOTAL - processed ))

    echo "=============================================================="
    echo " gen_defense_data484   $status   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================="
    python3 - "$processed" "$TOTAL" <<'PY'
import sys
d, t = int(sys.argv[1]), int(sys.argv[2])
w = 40
f = int(w * min(d, t) / t) if t else 0
print(f"  questions  {'#' * f}{'.' * (w - f)}  {d}/{t}  ({100 * d / t:.1f}%)")
PY
    echo "  saved      gullibility low=$low  high=$high  (total $((low + high)) conversations)"
    echo "  failed     $failed question(s) yielded nothing (rescue: --retry_failed)"

    if [[ -n "$elapsed" && "$done_now" -gt 0 ]]; then
        rate=$(python3 -c "print(f'{$done_now / $elapsed * 60:.1f}')")
        eta=$(python3 -c "
r=$done_now/$elapsed
s=int($remaining/r) if r>0 else 0
h,rem=divmod(s,3600); m,_=divmod(rem,60)
print(f'{h}h{m:02d}m')")
        echo "  this run   $done_now question(s) in $((elapsed / 60))m  |  $rate q/min  |  eta $eta"
    elif [[ -n "$elapsed" ]]; then
        echo "  this run   0 question(s) so far in $((elapsed / 60))m  |  eta --"
    fi

    echo "--------------------------------------------------------------"
    echo "  last log lines ($LOG):"
    tail -n 8 "$LOG" 2>/dev/null | sed 's/^/    /'
    echo "=============================================================="
}

case "${1:-}" in
    -f|--follow) exec tail -n 40 -f "$LOG" ;;
    -w|--watch)  while true; do clear; snapshot; sleep "${2:-30}"; done ;;
    *)           snapshot ;;
esac
