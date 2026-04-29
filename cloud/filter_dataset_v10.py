"""Filter Run #9's dataset to keep only pairs where the AI rewrite already
preserved the human's facts. Run #9 trained adapter dropped facts because
many of its training examples had drifted in the AI-rewrite step (gpt-4o-
mini hallucinating instead of preserving) and the model learned to do
the same.

For each (ai, human) pair, compute entity_overlap(human, ai). Keep pairs
above threshold. Free, local, fast.

  python cloud/filter_dataset_v10.py --in cloud/dataset_v9.jsonl --out cloud/dataset_v10.jsonl --threshold 0.7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.expanduser("~/humanizer"))
from humanizer.metrics.facts import entity_overlap, extract_facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="cloud/dataset_v9.jsonl")
    ap.add_argument("--out", dest="out", default="cloud/dataset_v10.jsonl")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Min entity_overlap(human, ai) to keep")
    args = ap.parse_args()

    inp = os.path.expanduser(args.inp)
    out = os.path.expanduser(args.out)

    print(f"Reading {inp}...")
    pairs = []
    with open(inp) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                pairs.append(json.loads(line))
            except Exception:
                continue
    print(f"Loaded {len(pairs)} pairs")

    print(f"Computing entity_overlap on each...")
    overlaps = []
    for p in pairs:
        ov = entity_overlap(p["human"], p["ai"])
        overlaps.append(ov)
        p["_overlap"] = ov

    # Distribution
    bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    counts = Counter()
    for ov in overlaps:
        for i, edge in enumerate(bins[:-1]):
            if edge <= ov < bins[i+1]:
                counts[edge] += 1
                break
        else:
            counts[1.0] += 1
    print("\nDistribution of entity_overlap(human, ai):")
    for edge in bins:
        c = counts[edge]
        bar = "█" * int(c / max(counts.values()) * 40)
        print(f"  [{edge:.1f}+]  {c:>5}  {bar}")
    n = len(pairs)
    mean = sum(overlaps) / n
    print(f"\n  mean overlap: {mean:.3f}")
    print(f"  ≥0.5: {sum(1 for o in overlaps if o >= 0.5)}/{n} = {sum(1 for o in overlaps if o >= 0.5)*100//n}%")
    print(f"  ≥0.7: {sum(1 for o in overlaps if o >= 0.7)}/{n} = {sum(1 for o in overlaps if o >= 0.7)*100//n}%")
    print(f"  ≥0.8: {sum(1 for o in overlaps if o >= 0.8)}/{n} = {sum(1 for o in overlaps if o >= 0.8)*100//n}%")

    # Filter
    kept = [p for p in pairs if p["_overlap"] >= args.threshold]
    print(f"\nKeeping pairs with overlap ≥ {args.threshold}: {len(kept)}/{n} = {len(kept)*100//n}%")

    # By source breakdown
    src_counts_in = Counter(p.get("source", "?") for p in pairs)
    src_counts_kept = Counter(p.get("source", "?") for p in kept)
    print("\nBy source:")
    for src in sorted(src_counts_in):
        cin = src_counts_in[src]; ckept = src_counts_kept[src]
        pct = ckept * 100 // cin if cin else 0
        print(f"  {src:>10}: {ckept:>5}/{cin:<5}  ({pct}% kept)")

    # Write — strip the temporary _overlap field
    with open(out, "w") as f:
        for p in kept:
            p.pop("_overlap", None)
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote {len(kept)} pairs to {out}")
    if len(kept) < 5000:
        print(f"WARN: filtered dataset is smaller than recommended (5000 minimum). "
              f"Consider lowering --threshold.")


if __name__ == "__main__":
    main()
