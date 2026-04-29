#!/usr/bin/env bash
# End-to-end Run #5 completion watcher.
#
# When Run #5 finishes, active_rescue.sh downloads eval-r5.json and
# adapter-r5/. This watcher waits for those locally, then:
#   1. Re-runs cloud/run_scrub_eval.ts (rents a $0.01 pod for ~1.5 min,
#      scores all 5 runs' base/scrub/trained/scrub variants against
#      roberta-base + roberta-large).
#   2. Runs scripts/bench_rejection.py to project best-of-N reliability.
#   3. Writes a single summary file: cloud/RUN5_RESULT.md
#   4. Posts a macOS notification.
#
# Cost: ~$0.01 (the validation pod). No paid-detector calls.
#
# Usage:
#   cd ~/humanizer/cloud
#   nohup bash watch_r5_complete.sh > watch_r5.log 2>&1 < /dev/null &
#   disown

set -euo pipefail

CLOUD_DIR="$HOME/humanizer/cloud"
EVAL_FILE="$CLOUD_DIR/eval-r5.json"
ADAPTER_DIR="$CLOUD_DIR/adapter-r5"
RESULT_FILE="$CLOUD_DIR/RUN5_RESULT.md"
SCRUB_EVAL_FILE="$CLOUD_DIR/eval-r5.scrub-eval.json"

cd "$CLOUD_DIR"

echo "[watch] $(date)  waiting for $EVAL_FILE + $ADAPTER_DIR/"

# Poll every 60s; safety: bail after 9 hours so this doesn't run forever.
DEADLINE=$(( $(date +%s) + 32400 ))
while true; do
  if [ -f "$EVAL_FILE" ] && [ -d "$ADAPTER_DIR" ]; then
    echo "[watch] $(date)  artifacts received"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[watch] $(date)  9-hour deadline reached without artifacts — exiting"
    exit 1
  fi
  sleep 60
done

# Step 1: validation pod ($0.01)
echo "[watch] $(date)  launching validation pod (cloud/run_scrub_eval.ts)"
if ! ~/.bun/bin/bun run "$CLOUD_DIR/run_scrub_eval.ts"; then
  echo "[watch] $(date)  run_scrub_eval.ts failed — see logs above"
  echo "## Run #5 — validation pod FAILED" > "$RESULT_FILE"
  echo "" >> "$RESULT_FILE"
  echo "Adapter + eval downloaded but the scoring pod errored." >> "$RESULT_FILE"
  echo "Re-run manually: \`cd ~/humanizer/cloud && bun run run_scrub_eval.ts\`" >> "$RESULT_FILE"
  osascript -e 'display notification "validation pod failed — see watch_r5.log" with title "humanizer Run #5"' 2>/dev/null || true
  exit 1
fi

# Step 2: bench projection (pure-python, ~1s)
echo "[watch] $(date)  computing reliability projection"
BENCH_OUT=$(python3 "$HOME/humanizer/scripts/bench_rejection.py" 2>&1 || true)

# Step 3: write summary
{
  echo "# Run #5 — adversarial discriminator GRPO  (auto-generated)"
  echo ""
  echo "**Completed:** $(date)"
  echo ""
  echo "## Artifacts"
  echo ""
  echo '```'
  ls -la "$EVAL_FILE" "$ADAPTER_DIR" 2>&1 | head -20
  echo '```'
  echo ""
  if [ -f "$SCRUB_EVAL_FILE" ]; then
    echo "## Run #5 detector summary (from validation pod)"
    echo ""
    echo '```json'
    python3 -c "
import json
d = json.load(open('$SCRUB_EVAL_FILE'))
print(json.dumps(d.get('summary', {}), indent=2))
" 2>/dev/null || echo "(could not parse summary)"
    echo '```'
    echo ""
  fi
  echo "## Reliability projection (all runs, including #5)"
  echo ""
  echo '```'
  printf '%s\n' "$BENCH_OUT"
  echo '```'
  echo ""
  echo "## Next step"
  echo ""
  echo "If Run #5's conservative best-of-8 (vs hardest detector) ≥ 70%, the"
  echo "rejection-sampling math gets us to ≥ 99.99% with no paid API. If"
  echo "not, plan Run #6 with a real-world detector in the reward loop."
} > "$RESULT_FILE"

echo "[watch] $(date)  wrote $RESULT_FILE"
osascript -e 'display notification "Run #5 validation complete — see cloud/RUN5_RESULT.md" with title "humanizer"' 2>/dev/null || true
