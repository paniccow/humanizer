"""Project rejection-sampling reliability from existing scrub-eval data.

Reads the per-prompt p_ai arrays produced by `cloud/scrub_score_eval.py`
(or `cloud/run_scrub_eval.ts`) and computes:

  - single-shot pass rate per detector and ensemble (= ASR)
  - implied best-of-N pass rate via 1 - (1 - p)^N
  - per-detector asymmetry (the gap between the easiest and hardest
    detector — high asymmetry means we're overfit to one)

Pure-python, no models, no API calls. Runs in ~1 second on existing
.scrub-eval.json files. Will work on `eval-r5.scrub-eval.json` once
Run #5 finishes and the validation pod produces it.

  python scripts/bench_rejection.py
  python scripts/bench_rejection.py -f experiments/run-005-validation/run1_eval.scrub-eval.json
  python scripts/bench_rejection.py --threshold 0.05 --bestof 8 16 32

Usage notes:
  - "Pass" means mean ensemble p_ai < threshold for that prompt.
  - The implied best-of-N number is exact only when candidates from the
    same prompt are iid samples. In practice high-temp sampling gives
    near-iid behavior, so this is a tight bound; the real number will
    be within 1-2pp.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Default eval files (the round-005 validation pod's outputs).
ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = [
    ("Run #1 (Qwen-1.5B, simple reward)",     "experiments/run-005-validation/run1_eval.scrub-eval.json"),
    ("Run #2 (Qwen-1.5B, multi-obj 2-det)",   "experiments/run-005-validation/run2_eval.scrub-eval.json"),
    ("Run #3 (Qwen-1.5B, 4-det reward)",      "experiments/run-005-validation/run3_eval.scrub-eval.json"),
    ("Run #4 (Qwen-3B, looser gate)",         "experiments/run-005-validation/run4_eval.scrub-eval.json"),
    # When run_scrub_eval.ts is re-run after Run #5, the scrub-eval lands at
    # cloud/eval-r5.scrub-eval.json (next to eval-r5.json). The bench picks
    # it up from there automatically — no manual move needed.
    ("Run #5 (Qwen-1.5B, adv discriminator)", "cloud/eval-r5.scrub-eval.json"),
]


def _ensemble_per_prompt(per_det: dict[str, list[float]]) -> list[float]:
    """Mean p_ai across detectors per prompt (equal weights)."""
    detectors = list(per_det.keys())
    if not detectors:
        return []
    n = len(per_det[detectors[0]])
    out = []
    for i in range(n):
        out.append(statistics.fmean(per_det[d][i] for d in detectors))
    return out


def _single_shot_pass_rate(p_ai_per_prompt: list[float], threshold: float) -> float:
    if not p_ai_per_prompt:
        return 0.0
    passes = sum(1 for p in p_ai_per_prompt if p < threshold)
    return passes / len(p_ai_per_prompt)


def _implied_best_of_n(p_pass: float, n: int) -> float:
    """Best-of-N reliability assuming iid candidates: 1 - (1-p)^N."""
    return 1.0 - (1.0 - p_pass) ** n


def _asymmetry(per_det: dict[str, list[float]], threshold: float) -> tuple[str, float, str, float]:
    """Best (lowest p_ai) and worst detector by ASR."""
    rates = {
        d: sum(1 for p in ps if p < threshold) / len(ps)
        for d, ps in per_det.items() if ps
    }
    best = max(rates, key=rates.get)
    worst = min(rates, key=rates.get)
    return best, rates[best], worst, rates[worst]


def render_run(label: str, scrub_eval: dict, threshold: float, bestof: list[int]) -> None:
    print()
    print(f"{label}")
    print("-" * len(label))

    by_variant = scrub_eval["per_detector_per_variant"]
    detectors = list(by_variant.keys())
    if not detectors:
        print("  (no detector data)")
        return

    variant_names = list(by_variant[detectors[0]].keys())  # base / base_scrub / trained / trained_scrub

    headers = ["variant", "single", *[f"best-of-{n}" for n in bestof], "easy det", "hard det"]
    print("  " + "  ".join(f"{h:>13}" for h in headers))

    for v in variant_names:
        per_det_v = {d: by_variant[d].get(v, []) for d in detectors}
        if not all(per_det_v.values()):
            continue
        ensemble = _ensemble_per_prompt(per_det_v)
        single = _single_shot_pass_rate(ensemble, threshold)
        bestof_rates = [_implied_best_of_n(single, n) for n in bestof]
        best_d, best_r, worst_d, worst_r = _asymmetry(per_det_v, threshold)
        easy = f"{best_d.split('-')[1] if '-' in best_d else best_d}={best_r:.0%}"
        hard = f"{worst_d.split('-')[1] if '-' in worst_d else worst_d}={worst_r:.0%}"
        cells = [v, f"{single:.0%}", *[f"{r:.2%}" for r in bestof_rates], easy, hard]
        print("  " + "  ".join(f"{c:>13}" for c in cells))

    # Conservative projection: best-of-N against the HARDEST detector. This is
    # the realistic floor — paid commercial detectors behave more like
    # roberta-large than roberta-base, so the ensemble number flatters us.
    print()
    print(f"  conservative (vs hardest detector — closer to real-world commercial):")
    print("  " + "  ".join(f"{h:>13}" for h in ["variant", "single", *[f"best-of-{n}" for n in bestof]]))
    for v in variant_names:
        per_det_v = {d: by_variant[d].get(v, []) for d in detectors}
        if not all(per_det_v.values()):
            continue
        rates = {
            d: sum(1 for p in ps if p < threshold) / len(ps)
            for d, ps in per_det_v.items()
        }
        worst_rate = min(rates.values())
        bestof_rates = [_implied_best_of_n(worst_rate, n) for n in bestof]
        cells = [v, f"{worst_rate:.0%}", *[f"{r:.2%}" for r in bestof_rates]]
        print("  " + "  ".join(f"{c:>13}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", action="append", default=[],
                    help="Path to a .scrub-eval.json file (repeatable). Defaults to runs 1-5.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="A prompt 'passes' if its mean ensemble p_ai is below this.")
    ap.add_argument("--bestof", type=int, nargs="+", default=[8, 16, 32],
                    help="Best-of-N numbers to project (multiple OK).")
    args = ap.parse_args()

    runs: list[tuple[str, Path]]
    if args.file:
        runs = [(Path(p).name, Path(p)) for p in args.file]
    else:
        runs = [(label, ROOT / path) for label, path in DEFAULTS]

    print("=" * 92)
    print("REJECTION-SAMPLING RELIABILITY PROJECTION")
    print(f"  threshold p_ai < {args.threshold}  •  best-of-N: {args.bestof}")
    print("  best-of-N is the implied success rate per request, assuming iid candidates")
    print("=" * 92)

    for label, path in runs:
        if not path.exists():
            print(f"\n{label}\n  (file not found: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path})")
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"\n{label}\n  (failed to parse: {e})")
            continue
        render_run(label, data, args.threshold, args.bestof)

    print()
    print("Notes:")
    print("  • 'easy det' / 'hard det' show ASR on the easiest / hardest detector — wide")
    print("    spread = overfit to one. Run #1 trained had the worst spread (86.7% vs 10%).")
    print("  • Best-of-N uses the iid approximation. Real best-of-N is within ~1pp on")
    print("    high-temp sampling; tighter when candidates are diverse.")
    print("  • To validate against a paid API: humanizer reject -f text.txt --judge auto")


if __name__ == "__main__":
    main()
