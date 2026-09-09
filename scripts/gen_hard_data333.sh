#!/usr/bin/env bash
# Launch the 333-question hard-negatives dataset generation detached, so it
# survives an SSH disconnect. Re-running is safe: the script skips
# already-processed questions via datasets_hard_333/processed_questions.json.
cd /root/mats12
mkdir -p datasets_hard_333
setsid nohup python -u scripts/gen_hard_data333.py \
    >> datasets_hard_333/run.log 2>&1 &
echo $! > datasets_hard_333/run.pid
echo "started pid $(cat datasets_hard_333/run.pid); watch with ./scripts/gen_hard_data333.py.watch.sh"
