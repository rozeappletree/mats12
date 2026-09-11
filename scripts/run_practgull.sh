#!/usr/bin/env bash
# PractGULL end-to-end: steer -> generate -> judge -> score.
#
# Stage per input file: src/test_steering.py generates the two steered replies
# per pair on the GPU, then src/calculate_steering_similarity.py scores them
# against the pair's own high-/low-gullibility references with the
# google/gemini-2.5-flash judge. pairs.gemini runs first (small, ~50 min) so a
# complete scored result lands early; pairs.sol follows (~14 h).
#
# Both halves are resumable: an interrupted generation continues from the
# record it stopped on (.partial file), and judgements are cached, so re-running
# this script after a kill picks up where it left off rather than starting over.
#
# See data/PractGULL/README.md for what any of it means.
#
# USAGE
#   bash scripts/run_practgull.sh                 # foreground
#   nohup bash scripts/run_practgull.sh > logs/practgull/pipeline.log 2>&1 &
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
LOGS="$REPO/logs/practgull"
DATA="$REPO/data/PractGULL"
mkdir -p "$LOGS"

CONDA_ENV="${CONDA_ENV:-talktuner-gpu}"
PY=(conda run --no-capture-output -n "$CONDA_ENV" python -u)

# withDefense484Only's summary.json best_layer is 10, but its per-layer accuracy
# is flat at 0.9965 from layer 10 to 40 -- layer 10 is an argmax tie-break, and
# steering there at n_scale=7 destroys the generation (verified: pure token
# garbage). Steer it at the same [12,25) window withRegularGullibility uses, so
# the two checkpoint sets differ only in their training data.
STEER_ARGS=(--layers withDefense484Only=12,25)

stage () {
  local name="$1" src="$DATA/$1"
  local steered="$DATA/steeredoutput.$1"

  echo "=== [$(date -Is)] generating: $name ==="
  "${PY[@]}" src/test_steering.py --input "$src" "${STEER_ARGS[@]}" \
      2>&1 | tee -a "$LOGS/generate.${name%.jsonl}.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! [$(date -Is)] generation FAILED for $name (exit $rc); skipping its scoring"
    return "$rc"
  fi

  echo "=== [$(date -Is)] judging: steeredoutput.$name ==="
  "${PY[@]}" src/calculate_steering_similarity.py --input "$steered" \
      2>&1 | tee -a "$LOGS/similarity.${name%.jsonl}.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && echo "!!! [$(date -Is)] judging FAILED for $name (exit $rc)"
  echo "=== [$(date -Is)] finished: $name ==="
  return "$rc"
}

echo "### PractGULL pipeline started $(date -Is)"
echo "### repo=$REPO env=$CONDA_ENV steer_args=${STEER_ARGS[*]}"

stage pairs.gemini.jsonl
stage pairs.sol.jsonl

echo "### PractGULL pipeline finished $(date -Is)"
