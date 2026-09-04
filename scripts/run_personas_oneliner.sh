#!/usr/bin/env bash
# Launch truthfulqa_personas_oneliner.py detached so it survives SSH disconnects.
# Waits for any in-flight personas run to release the GPU first.
# Any extra args are passed straight through to the python script.
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/personas_oneliner_$(date +%Y%m%d_%H%M%S).log"
setsid nohup bash -c '
  for pidfile in logs/personas.pid logs/personas_oneliner.pid; do
    [ -f "$pidfile" ] || continue
    pid=$(cat "$pidfile")
    # never wait on ourselves (our own pid is written to personas_oneliner.pid)
    [ -z "$pid" ] || [ "$pid" = "$$" ] && continue
    while kill -0 "$pid" 2>/dev/null; do
      echo "[$(date +%H:%M:%S)] waiting for pid $pid ($pidfile) to release the GPU..."
      sleep 20
    done
  done
  echo "[$(date +%H:%M:%S)] GPU free, starting run"
  exec /opt/conda/envs/talktuner-gpu/bin/python -u scripts/truthfulqa_personas_oneliner.py "$@"
' _ "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > logs/personas_oneliner.pid
ln -sfn "$(basename "$LOG")" logs/personas_oneliner.latest.log
echo "started pid $(cat logs/personas_oneliner.pid), logging to $LOG"
echo "follow with: tail -f logs/personas_oneliner.latest.log"
