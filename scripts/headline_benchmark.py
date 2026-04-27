"""Reproduce the README's headline numbers.

Iterates the existing eval JSONs (run #1-#4) and computes the pattern
aggregate before/after the deterministic scrub, plus the per-flag firing
rate. Pure-python — no models, no API key.

  python scripts/headline_benchmark.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# Make humanizer importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanizer.patterns import analyze
from humanizer.pipeline import scrub

EVAL_FILES = [
    ("Run #1 (Qwen-1.5B, simple reward)",  "experiments/run-002-pre/run1_eval.json"),
    ("Run #2 (Qwen-1.5B, multi-obj 2-det)", "cloud/eval.json"),
    ("Run #3 (Qwen-1.5B, multi-obj 4-det)", "cloud/eval-r3.json"),
    ("Run #4 (Qwen-3B, looser gate)",       "cloud/eval-r4.json"),
]


def main():
    print("=" * 88)
    print("HEADLINE BENCHMARK — pattern aggregate (lower = more human, 13 signals)")
    print("=" * 88)
    print(f'{"":<36} {"BASE":>9} {"+SCRUB":>9} {"TRAINED":>9} {"+SCRUB":>9}')
    print("-" * 88)

    best_b = best_bs = best_t = best_ts = 1.0
    flag_counter = Counter()

    for label, path in EVAL_FILES:
        full = Path(path)
        if not full.is_absolute():
            full = Path(__file__).resolve().parents[1] / full
        if not full.exists():
            print(f'{label:<36} (file missing)')
            continue
        d = json.loads(full.read_text())
        base_outs = d["base"]["outputs"]
        trained_outs = d["trained"]["outputs"]
        scrub_base = [scrub(o).text for o in base_outs]
        scrub_trn = [scrub(o).text for o in trained_outs]

        b = statistics.fmean(analyze(o).aggregate for o in base_outs)
        bs = statistics.fmean(analyze(o).aggregate for o in scrub_base)
        t = statistics.fmean(analyze(o).aggregate for o in trained_outs)
        ts = statistics.fmean(analyze(o).aggregate for o in scrub_trn)

        print(f'{label:<36} {b:>9.3f} {bs:>9.3f} {t:>9.3f} {ts:>9.3f}')

        best_b = min(best_b, b)
        best_bs = min(best_bs, bs)
        best_t = min(best_t, t)
        best_ts = min(best_ts, ts)
        for o in scrub_base + scrub_trn:
            for f in analyze(o).flagged:
                flag_counter[f] += 1

    print("-" * 88)
    print(f'{"BEST across runs":<36} {best_b:>9.3f} {best_bs:>9.3f} {best_t:>9.3f} {best_ts:>9.3f}')
    print()

    print("Most-frequent residual flags after scrub (across all 4 runs, ~240 outputs):")
    for sig, count in flag_counter.most_common(13):
        print(f'  {sig:<28} {count}')
    print()

    rel = (best_b - best_bs) / best_b if best_b > 0 else 0
    print(f"Headline: best raw BASE pattern {best_b:.3f}  ->  BASE+SCRUB {best_bs:.3f}  ({rel:.0%} reduction)")


if __name__ == "__main__":
    main()
