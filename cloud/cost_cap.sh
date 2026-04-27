#!/usr/bin/env bash
# Hard cost cap — kill the pod after a max wall-clock time, regardless of
# whether launch.ts is still running. Safety net for "Mac died mid-train".
#
#   bash cost_cap.sh <pod-id> <max-hours>
#
# Example: bash cost_cap.sh 16agfzzootjo72 4
# Run in nohup so it survives terminal close:
#   nohup bash cost_cap.sh 16agfzzootjo72 4 > cost_cap.log 2>&1 & disown
set -euo pipefail

POD_ID="${1:?pod-id required}"
MAX_HOURS="${2:-4}"
RUN_ENV="$HOME/.humanizer-runpod.env"
SLEEP_SEC="$((MAX_HOURS * 3600))"

echo "[cost_cap] $(date)  pod=$POD_ID max=${MAX_HOURS}h"
echo "[cost_cap] sleeping ${SLEEP_SEC}s, will then check & terminate if still up"

sleep "$SLEEP_SEC"

echo "[cost_cap] $(date)  cap reached. checking pod status..."

# shellcheck source=/dev/null
source "$RUN_ENV"
export RUNPOD_PUBLIC_KEY="$(cat "$HOME/.ssh/id_ed25519.pub")"

# Query the pod via the GraphQL API directly to see if it's still alive.
STATUS_JSON=$(curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
  -H 'content-type: application/json' \
  -d "{\"query\":\"query { pod(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}")

if echo "$STATUS_JSON" | grep -q "RUNNING"; then
  echo "[cost_cap] pod still RUNNING — force terminating"
  cd "$HOME/humanizer/cloud"
  "$HOME/.bun/bin/bun" run provision.ts terminate "$POD_ID"
  echo "[cost_cap] $(date)  done."
elif echo "$STATUS_JSON" | grep -q "EXITED"; then
  echo "[cost_cap] pod already EXITED — nothing to do"
else
  echo "[cost_cap] pod state ambiguous, response: $STATUS_JSON"
  echo "[cost_cap] attempting terminate anyway"
  cd "$HOME/humanizer/cloud"
  "$HOME/.bun/bin/bun" run provision.ts terminate "$POD_ID" || true
fi
