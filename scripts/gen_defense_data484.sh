#!/usr/bin/env bash
# Launch the 484-question defense dataset generation detached, so it survives
# an SSH disconnect. Re-running is safe: the script skips already-processed
# questions via datasets_defense_484/processed_questions.json.
cd /root/mats12
mkdir -p datasets_defense_484
setsid nohup python -u scripts/gen_defense_data484.py \
    >> datasets_defense_484/run.log 2>&1 &
echo $! > datasets_defense_484/run.pid
echo "started pid $(cat datasets_defense_484/run.pid); watch with ./scripts/gen_defense_data484.py.watch.sh"
