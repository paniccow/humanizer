#!/usr/bin/env bash
# Auto-fire held-out detector eval the moment one or more training runs
# produce their eval.json. Designed to run alongside in-flight training.
#
#   bash auto_heldout.sh eval.json eval-r3.json    # watch both
#
# Behavior: poll every 2 min for each input file. When all listed files
# exist, spin up ONE held-out pod and process all of them in a single
# pass (cheaper than per-file pods). Writes <input>.holdout.json next
# to each. Exits.
set -euo pipefail

CLOUD_DIR="$HOME/humanizer/cloud"
RUN_ENV="$HOME/.humanizer-runpod.env"
INPUTS=("$@")
if [ "${#INPUTS[@]}" -eq 0 ]; then
  echo "usage: bash auto_heldout.sh <eval.json> [<eval2.json> ...]"
  exit 1
fi

cd "$CLOUD_DIR"
echo "[auto_heldout] $(date)  watching ${#INPUTS[@]} file(s)"
for f in "${INPUTS[@]}"; do echo "  - $f"; done

# Wait until ALL inputs exist (size > 0).
while true; do
  all_present=1
  for f in "${INPUTS[@]}"; do
    if [ ! -s "$f" ]; then
      all_present=0
      break
    fi
  done
  if [ "$all_present" -eq 1 ]; then break; fi
  sleep 120
done

echo "[auto_heldout] $(date)  all inputs present. launching held-out pod..."

# shellcheck source=/dev/null
source "$RUN_ENV"
export RUNPOD_PUBLIC_KEY="$(cat "$HOME/.ssh/id_ed25519.pub")"
export RUNPOD_PRIVATE_KEY="$HOME/.ssh/id_ed25519"
export RUNPOD_CONTAINER_GB="${RUNPOD_CONTAINER_GB:-25}"
export RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-15}"
export RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-ALL}"

"$HOME/.bun/bin/bun" run run_heldout.ts "${INPUTS[@]}"

echo "[auto_heldout] $(date)  done. holdout files written next to each input."
