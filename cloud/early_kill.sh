#!/usr/bin/env bash
# early_kill.sh — kill pod if training stalls (no new step lines for N minutes).
#
# Protects against the run #4 attempt 1 failure mode:
#   - Python OOMs at step 0
#   - launch.ts watch loop's "DEAD" detection silently fails
#   - Pod sits at 0% utilization burning $0.69/hr
#   - Cost cap doesn't fire for hours
#
# This script polls /workspace/output/run.log every 60s. If `step N/...` line
# count hasn't increased for IDLE_MAX_MIN, terminate the pod immediately.
#
#   bash early_kill.sh <pod-id> <ssh-host> <ssh-port> [grace-min] [idle-max-min]
#   bash early_kill.sh foo123 1.2.3.4 12345 5 25
#
# Defaults: 5 min grace (let pod boot + pip install + first step), 25 min idle.
set -euo pipefail

POD_ID="${1:?pod-id required}"
HOST="${2:?ssh-host required}"
PORT="${3:?ssh-port required}"
GRACE_MIN="${4:-5}"
IDLE_MAX_MIN="${5:-25}"
RUN_ENV="$HOME/.humanizer-runpod.env"
CLOUD_DIR="$HOME/humanizer/cloud"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"

echo "[early_kill] $(date)  pod=$POD_ID grace=${GRACE_MIN}m max_idle=${IDLE_MAX_MIN}m"
sleep "$((GRACE_MIN * 60))"

last_step_count=0
last_change_t=$(date +%s)

while true; do
  # If 'done' sentinel appears, training completed successfully — bail clean.
  if ssh -p "$PORT" $SSH_OPTS root@"$HOST" "test -f /workspace/output/done" 2>/dev/null; then
    echo "[early_kill] $(date)  training completed normally — exiting"
    exit 0
  fi

  # Count step lines in the run log on the pod.
  count_raw=$(ssh -p "$PORT" $SSH_OPTS root@"$HOST" \
    "grep -cE '^step ' /workspace/output/run.log 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  count=$(echo "$count_raw" | tr -dc '0-9')
  count=${count:-0}

  if [ "$count" -gt "$last_step_count" ]; then
    last_step_count=$count
    last_change_t=$(date +%s)
  fi

  now=$(date +%s)
  idle_min=$(( (now - last_change_t) / 60 ))

  if [ "$idle_min" -ge "$IDLE_MAX_MIN" ]; then
    echo "[early_kill] $(date)  no step progress for ${idle_min}m (last count=$last_step_count) — TERMINATING POD"
    # Try to save anything that's there (training might have written checkpoints).
    if ssh -p "$PORT" $SSH_OPTS root@"$HOST" "tail -50 /workspace/output/run.log" > "$CLOUD_DIR/early_kill_${POD_ID}_runlog.txt" 2>/dev/null; then
      echo "[early_kill] saved last 50 lines of run.log to early_kill_${POD_ID}_runlog.txt"
    fi
    # shellcheck source=/dev/null
    source "$RUN_ENV"
    export RUNPOD_PUBLIC_KEY="$(cat "$HOME/.ssh/id_ed25519.pub")"
    cd "$CLOUD_DIR"
    "$HOME/.bun/bin/bun" run provision.ts terminate "$POD_ID"
    echo "[early_kill] $(date)  pod terminated."
    exit 1
  fi

  sleep 60
done
