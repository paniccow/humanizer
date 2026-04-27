#!/usr/bin/env bash
# Hard cost cap — kill the pod after max wall-clock time. Safety net for
# "launch.ts watch loop silently failed" or "Mac died mid-train".
#
#   bash cost_cap.sh <pod-id> <max-hours> [<adapter-out>] [<eval-out>]
#
# Examples:
#   bash cost_cap.sh oavxljdxye458g 6 adapter-r4 eval-r4.json
#   bash cost_cap.sh 16agfzzootjo72 4
#
# What it does at cap time:
#   1. Look up pod's SSH endpoint via GraphQL
#   2. SSH in, check for /workspace/output/done sentinel
#   3. If 'done' exists AND adapter-out arg given: SCP the adapter + eval
#      locally BEFORE terminating. (Saves work if launch.ts watcher died.)
#   4. Terminate the pod regardless.
set -euo pipefail

POD_ID="${1:?pod-id required}"
MAX_HOURS="${2:-4}"
ADAPTER_OUT="${3:-}"           # optional — if set, try to download before terminate
EVAL_OUT="${4:-}"              # optional — pair with ADAPTER_OUT

RUN_ENV="$HOME/.humanizer-runpod.env"
CLOUD_DIR="$HOME/humanizer/cloud"
SLEEP_SEC="$((MAX_HOURS * 3600))"

echo "[cost_cap] $(date)  pod=$POD_ID max=${MAX_HOURS}h save_to='$ADAPTER_OUT'"
echo "[cost_cap] sleeping ${SLEEP_SEC}s, will then check & terminate if still up"

sleep "$SLEEP_SEC"

echo "[cost_cap] $(date)  cap reached. checking pod status..."

# shellcheck source=/dev/null
source "$RUN_ENV"
export RUNPOD_PUBLIC_KEY="$(cat "$HOME/.ssh/id_ed25519.pub")"

# Query pod status + SSH endpoint via GraphQL.
RESP=$(curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
  -H 'content-type: application/json' \
  -d "{\"query\":\"query { pod(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus runtime { ports { ip isIpPublic publicPort privatePort } } } }\"}")

if echo "$RESP" | grep -q '"desiredStatus":"RUNNING"' && echo "$RESP" | grep -q '"isIpPublic":true'; then
  # Extract SSH endpoint (privatePort=22, isIpPublic=true).
  SSH_HOST=$(echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); ports=d["data"]["pod"]["runtime"]["ports"]; p=next((x for x in ports if x["privatePort"]==22 and x["isIpPublic"]), None); print(p["ip"] if p else "")')
  SSH_PORT=$(echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); ports=d["data"]["pod"]["runtime"]["ports"]; p=next((x for x in ports if x["privatePort"]==22 and x["isIpPublic"]), None); print(p["publicPort"] if p else "")')

  if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && [ -n "$ADAPTER_OUT" ]; then
    echo "[cost_cap] pod RUNNING at $SSH_HOST:$SSH_PORT — checking for done sentinel"
    DONE_OUTPUT=$(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=15 root@"$SSH_HOST" \
      "ls /workspace/output/done /workspace/output/eval.json 2>&1" 2>&1 || echo "FAILED")

    if echo "$DONE_OUTPUT" | grep -q "/workspace/output/done"; then
      echo "[cost_cap] training complete — rescuing adapter + eval before terminate"
      mkdir -p "$CLOUD_DIR/$ADAPTER_OUT"
      scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 \
        root@"$SSH_HOST":/workspace/output/eval.json "$CLOUD_DIR/${EVAL_OUT:-eval.json}" 2>&1 | tail -2
      scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 -r \
        root@"$SSH_HOST":/workspace/output/adapter/. "$CLOUD_DIR/$ADAPTER_OUT/" 2>&1 | tail -2
      echo "[cost_cap] rescue: saved $CLOUD_DIR/$ADAPTER_OUT and $CLOUD_DIR/${EVAL_OUT:-eval.json}"
    else
      echo "[cost_cap] no 'done' sentinel — training didn't complete; nothing to save"
    fi
  fi

  echo "[cost_cap] force terminating pod $POD_ID"
  cd "$CLOUD_DIR"
  "$HOME/.bun/bin/bun" run provision.ts terminate "$POD_ID"
  echo "[cost_cap] $(date)  done."
elif echo "$RESP" | grep -q '"desiredStatus":"EXITED"'; then
  echo "[cost_cap] pod already EXITED — nothing to do"
else
  echo "[cost_cap] pod state ambiguous, response: $RESP"
  echo "[cost_cap] attempting terminate anyway"
  cd "$CLOUD_DIR"
  "$HOME/.bun/bin/bun" run provision.ts terminate "$POD_ID" || true
fi
