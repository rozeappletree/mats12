#!/usr/bin/env bash
# Launch truthfulqa_personas.py detached so it survives SSH disconnects.
# Any extra args are passed straight through to the python script.
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/personas_$(date +%Y%m%d_%H%M%S).log"
setsid nohup /opt/conda/envs/talktuner-gpu/bin/python -u scripts/truthfulqa_personas.py "$@" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > logs/personas.pid
ln -sfn "$(basename "$LOG")" logs/personas.latest.log
echo "started pid $(cat logs/personas.pid), logging to $LOG"
echo "follow with: tail -f logs/personas.latest.log"
