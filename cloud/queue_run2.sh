#!/usr/bin/env bash
# Queue script — auto-launch run #2 the moment run #1 finishes.
#
#   bash queue_run2.sh                # waits, archives, launches v2
#
# Behavior:
#   1. Polls /Users/ishaanaggarwal/humanizer/cloud/eval.json every 60s.
#   2. When it appears (= run #1 finished cleanly), archives run #1's
#      adapter + eval + log into experiments/run-002-pre/.
#   3. Clears the working dir so run #2 doesn't trip over leftover files.
#   4. Launches run #2 via launch.ts with TRAIN_SCRIPT=train_v2.py.
#
# Run this in tmux/nohup so it survives terminal close:
#   nohup bash queue_run2.sh > queue_run2.log 2>&1 & disown
set -euo pipefail

CLOUD_DIR="$HOME/humanizer/cloud"
ARCHIVE_DIR="$HOME/humanizer/experiments/run-002-pre"
RUN_ENV="$HOME/.humanizer-runpod.env"

echo "[queue] $(date)  waiting for run #1 to finish (poll every 60s)..."
echo "[queue] watching $CLOUD_DIR/eval.json"

while [ ! -f "$CLOUD_DIR/eval.json" ]; do
  # Bail if launch.ts isn't running anymore AND there's no eval — run #1 likely failed.
  if ! pgrep -f 'bun run launch' > /dev/null; then
    if [ ! -f "$CLOUD_DIR/eval.json" ]; then
      echo "[queue] $(date)  ERROR: launch.ts exited without producing eval.json — run #1 likely crashed."
      echo "[queue]            tail of training.log:"
      tail -30 "$CLOUD_DIR/training.log" | sed 's/^/[queue]            /'
      echo "[queue]            NOT auto-launching run #2. Investigate first."
      exit 1
    fi
  fi
  sleep 60
done

echo "[queue] $(date)  run #1 done. archiving..."
mkdir -p "$ARCHIVE_DIR"
[ -d "$CLOUD_DIR/adapter" ]      && cp -r "$CLOUD_DIR/adapter"      "$ARCHIVE_DIR/run1_adapter"
[ -f "$CLOUD_DIR/eval.json" ]    && cp    "$CLOUD_DIR/eval.json"    "$ARCHIVE_DIR/run1_eval.json"
[ -f "$CLOUD_DIR/training.log" ] && cp    "$CLOUD_DIR/training.log" "$ARCHIVE_DIR/run1_training.log"
echo "[queue] archived to $ARCHIVE_DIR"

echo "[queue] clearing working dir for run #2..."
rm -rf "$CLOUD_DIR/adapter"
rm -f  "$CLOUD_DIR/eval.json"
mv     "$CLOUD_DIR/training.log" "$CLOUD_DIR/training.log.run1" 2>/dev/null || true

echo "[queue] $(date)  launching run #2 (train_v2.py)..."
cd "$CLOUD_DIR"
# shellcheck source=/dev/null
source "$RUN_ENV"
export RUNPOD_PUBLIC_KEY="$(cat "$HOME/.ssh/id_ed25519.pub")"
export RUNPOD_PRIVATE_KEY="$HOME/.ssh/id_ed25519"
export RUNPOD_CONTAINER_GB="${RUNPOD_CONTAINER_GB:-25}"
export RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-15}"
export RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-ALL}"
export TRAIN_SCRIPT=train_v2.py

nohup "$HOME/.bun/bin/bun" run launch.ts > training.log 2>&1 & disown
echo "[queue] launched. tail $CLOUD_DIR/training.log to watch."
