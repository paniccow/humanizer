#!/usr/bin/env bash
# Active rescue: poll pod for /workspace/output/done. Each detection,
# pull the latest adapter (and eval if it exists). Keep watching after
# pulls — train_v5 writes `done` after every save_every checkpoint, so
# subsequent pulls see fresh adapter state. Exit only when training
# truly finishes (eval.json appears) or after a hard 9-hour deadline.
#
#   bash active_rescue.sh <pod-id> <ssh-host> <ssh-port> <adapter-out> <eval-out>
#
# CHANGED from the early version: no more `set -euo pipefail` because a
# missing eval.json (normal early-checkpoint case) was causing pipefail
# to exit the script BEFORE the adapter scp ran. We now tolerate
# individual scp failures and keep going.
set -uo pipefail   # NB: no -e
POD_ID="${1:?}"; HOST="${2:?}"; PORT="${3:?}"; ADAPTER="${4:?}"; EVAL="${5:?}"
CLOUD="$HOME/humanizer/cloud"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
DEADLINE=$(( $(date +%s) + 32400 ))   # 9 hours

echo "[rescue] $(date)  watching $POD_ID at $HOST:$PORT (continuous)"
while true; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[rescue] $(date)  9-hour deadline reached — exiting"
    exit 1
  fi
  # If we already have BOTH eval.json and a non-empty adapter dir locally,
  # we're done (training completed and artifacts copied).
  if [ -f "$CLOUD/$EVAL" ] && [ -d "$CLOUD/$ADAPTER" ] && [ -n "$(ls -A "$CLOUD/$ADAPTER" 2>/dev/null)" ]; then
    echo "[rescue] $(date)  artifacts already local — exiting"
    exit 0
  fi
  if ssh -p "$PORT" $SSH_OPTS root@"$HOST" "test -f /workspace/output/done" 2>/dev/null; then
    echo "[rescue] $(date)  done detected — pulling latest"
    mkdir -p "$CLOUD/$ADAPTER"
    # Pull adapter dir (may overwrite earlier checkpoint files — that's fine,
    # we want the newest). `|| true` so a transient failure doesn't kill us.
    scp -P "$PORT" $SSH_OPTS -r "root@${HOST}:/workspace/output/adapter/." "$CLOUD/$ADAPTER/" 2>&1 | tail -3 || true
    # Eval may not exist yet (early checkpoints) — just try.
    scp -P "$PORT" $SSH_OPTS "root@${HOST}:/workspace/output/eval.json" "$CLOUD/$EVAL" 2>&1 | tail -2 || true
    if [ -f "$CLOUD/$EVAL" ]; then
      echo "[rescue] $(date)  full artifacts saved (adapter + eval) — exiting"
      exit 0
    else
      echo "[rescue] $(date)  partial pull (adapter only, eval not ready) — keep watching"
    fi
  fi
  sleep 300
done
