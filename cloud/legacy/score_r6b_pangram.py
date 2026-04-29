"""Run when eval-r6b.json + adapter-r6b/ land. Scores N base + N trained
outputs against Pangram for the head-to-head. Uses 2N credits.

Default: N=10 (20 credits = $1.00). Override with first arg:
  python score_r6b_pangram.py 5    # 5 base + 5 trained = $0.50
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PANGRAM = "https://text.api.pangramlabs.com/v3"
EVAL_FILE = os.path.expanduser("~/humanizer/cloud/eval-r6b.json")


def pangram_score(text):
    headers = {
        "Content-Type": "application/json", "Accept": "application/json",
        "x-api-key": os.environ["PANGRAM_API_KEY"],
    }
    req = urllib.request.Request(
        PANGRAM, data=json.dumps({"text": text}).encode("utf-8"),
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read().decode("utf-8"))
    p = float(payload["fraction_ai"]) + 0.5 * float(payload.get("fraction_ai_assisted", 0.0))
    return p, payload["prediction_short"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    if not os.path.exists(EVAL_FILE):
        print(f"NOT READY: {EVAL_FILE} doesn't exist yet")
        sys.exit(1)

    d = json.load(open(EVAL_FILE))
    base = d["base"]["outputs"][:n]
    trained = d["trained"]["outputs"][:n]

    print(f"=== Run #6 dense (every-2-steps) — Pangram head-to-head ===")
    print(f"Scoring {2*n} outputs ({n} base + {n} trained), {2*n} credits = ${0.05 * 2 * n:.2f}")
    print()

    with ThreadPoolExecutor(max_workers=8) as pool:
        base_results = list(pool.map(pangram_score, base))
        trained_results = list(pool.map(pangram_score, trained))

    print(f"{'idx':>3}  {'BASE p_ai':>10}  {'TRAINED p_ai':>13}  delta   trained-preview")
    print("-" * 100)
    base_scores, trained_scores = [], []
    for i in range(n):
        bp, bcls = base_results[i]
        tp, tcls = trained_results[i]
        base_scores.append(bp); trained_scores.append(tp)
        delta = tp - bp
        prev = (trained[i][:55] + "...") if len(trained[i]) > 55 else trained[i]
        print(f"{i:>3}  {bp:>10.3f}  {tp:>13.3f}  {delta:+.3f}  {prev}")

    print()
    print("=" * 60)
    bm = sum(base_scores) / len(base_scores)
    tm = sum(trained_scores) / len(trained_scores)
    print(f"  BASE    mean p_ai = {bm:.3f}  ASR(<0.5) = {sum(1 for p in base_scores if p < 0.5)}/{n}")
    print(f"  TRAINED mean p_ai = {tm:.3f}  ASR(<0.5) = {sum(1 for p in trained_scores if p < 0.5)}/{n}")
    print(f"  DELTA mean: {tm - bm:+.3f}  (negative = win)")
    print()
    if tm < bm and tm < 0.95:
        print(">>> SIGNAL: trained moved Pangram. May be small — check ASR.")
    elif tm < bm:
        print(">>> Tiny shift. Within noise. Pangram still flags.")
    else:
        print(">>> NO MOVEMENT. Run #6 dense did not beat Pangram.")


if __name__ == "__main__":
    main()
