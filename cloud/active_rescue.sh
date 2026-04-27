#!/usr/bin/env bash
# Active rescue: poll pod for /workspace/output/done. When found, SCP
# adapter + eval immediately, regardless of whether launch.ts is hung.
#
#   bash active_rescue.sh <pod-id> <ssh-host> <ssh-port> <adapter-out> <eval-out>
set -euo pipefail
POD_ID="${1:?}"; HOST="${2:?}"; PORT="${3:?}"; ADAPTER="${4:?}"; EVAL="${5:?}"
CLOUD="$HOME/humanizer/cloud"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"

echo "[rescue] $(date)  watching $POD_ID at $HOST:$PORT"
while true; do
  if [ -f "$CLOUD/$EVAL" ] && [ -d "$CLOUD/$ADAPTER" ]; then
    echo "[rescue] artifacts already local — exiting"
    exit 0
  fi
  # Use test exit code, not ls grep — ls's error message contains the path too.
  if ssh -p "$PORT" $SSH_OPTS root@"$HOST" "test -f /workspace/output/done" 2>/dev/null; then
    echo "[rescue] $(date)  done detected — pulling artifacts"
    mkdir -p "$CLOUD/$ADAPTER"
    scp -P "$PORT" $SSH_OPTS root@"$HOST":/workspace/output/eval.json "$CLOUD/$EVAL" 2>&1 | tail -2
    scp -P "$PORT" $SSH_OPTS -r root@"$HOST":/workspace/output/adapter/. "$CLOUD/$ADAPTER/" 2>&1 | tail -2
    echo "[rescue] saved $CLOUD/$ADAPTER + $CLOUD/$EVAL"
    exit 0
  fi
  sleep 300
done
