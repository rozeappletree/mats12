#!/usr/bin/env bash
# Progress tracker for gen_opus_data100.py.
#
#   scripts/track_data100.sh [output_dir] [per_level]     # one-shot
#   watch -n 30 scripts/track_data100.sh                  # live view
#
# Reads the run log from <output_dir>/run.log, i.e. where the launch command
# in the README/handoff redirects it.

DIR="${1:-datasets_claudeopus_sample2}"
TARGET="${2:-100}"
LOG="$DIR/run.log"

# Levels differ per attribute: certainty_seeking uses "neutral" where the
# others use "medium".
levels_for() {
    case "$1" in
        certainty_seeking) echo "low neutral high" ;;
        *)                 echo "low medium high" ;;
    esac
}

total=0
grand=0
printf '%-20s %-8s %10s\n' ATTRIBUTE LEVEL COUNT
printf '%s\n' '--------------------------------------------'
for attr_dir in "$DIR"/*/; do
    [ -d "$attr_dir" ] || continue
    attr=$(basename "$attr_dir")
    for lv in $(levels_for "$attr"); do
        n=$(find "$attr_dir" -maxdepth 1 -name "conversation_*_${attr}_${lv}.txt" 2>/dev/null | wc -l)
        printf '%-20s %-8s %6d/%-4d\n' "$attr" "$lv" "$n" "$TARGET"
        total=$((total + n))
        grand=$((grand + TARGET))
    done
done
printf '%s\n' '--------------------------------------------'
printf '%-20s %-8s %6d/%-4d' TOTAL '' "$total" "$grand"
[ "$grand" -gt 0 ] && printf '  (%d%%)' $((100 * total / grand))
printf '\n\n'

# The bracket keeps this pgrep from matching its own command line.
pids=$(pgrep -f "[g]en_opus_data100.py")
if [ -n "$pids" ]; then
    echo "status:  RUNNING (pid $(echo "$pids" | tr '\n' ' '))"
else
    echo "status:  not running"
fi

if [ -f "$LOG" ]; then
    fails=$(grep -c -e 'rejected:' -e 'error:' -e 'dropped' "$LOG" 2>/dev/null)
    echo "log:     $LOG  (${fails} reject/drop/error lines)"
    echo "--- last 5 log lines ---"
    tail -n 5 "$LOG"
else
    echo "log:     $LOG (not found)"
fi
