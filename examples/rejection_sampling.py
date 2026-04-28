"""Rejection-sampling quickstart — the "best-of-N until pass" recipe.

Three modes shown:
  1. Free local validation: --judge roberta. No paid API needed.
  2. Single paid judge: --judge gptzero / originality / pangram.
  3. Ensemble of paid judges: --judge auto picks up whichever keys
     are set in env, builds a weighted EnsembleJudge.

The math: if base humanizer's single-shot pass rate is p, best-of-N
rejection sampling gives 1-(1-p)^N per request. p=0.7 + N=8 -> 99.99%.

Required:
  pip install -e '.[openai]'
  export OPENAI_API_KEY=sk-...   # or sk-or-... for OpenRouter
  # Optional, in any combination, for paid judges:
  export ORIGINALITY_API_KEY=...
  export PANGRAM_API_KEY=...
  export GPTZERO_API_KEY=...

Run:
  python examples/rejection_sampling.py
"""
from humanizer.detectors import judge_from_env
from humanizer.humanizers import (
    PromptHumanizer,
    PromptHumanizerConfig,
    RejectionConfig,
    RejectionSamplingHumanizer,
)


SAMPLE = (
    "In today's rapidly evolving digital landscape, artificial intelligence has "
    "emerged as a transformative force. Furthermore, organizations are leveraging "
    "AI to navigate the intricate complexities of this multifaceted ecosystem. "
    "Moreover, the integration of machine learning is paramount."
)


def main():
    base = PromptHumanizer(PromptHumanizerConfig(model="gpt-4o-mini"))
    judge = judge_from_env()  # picks up paid keys; falls back to roberta-large

    rejection = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=8,
            max_rounds=4,
            p_ai_threshold=0.05,        # strict pass: clearly human
            similarity_threshold=0.78,  # keep meaning intact
        ),
    )

    print(f"judge: {judge.name}")
    print(f"\nINPUT:\n{SAMPLE}\n")

    result = rejection.humanize(SAMPLE)
    meta = result.metadata or {}

    print(f"OUTPUT (passed={meta.get('passed')}, "
          f"p_ai={result.score:.3f}, "
          f"rounds={meta.get('rounds_used')}, "
          f"attempts={result.attempts}):")
    print(result.text)

    if meta.get("per_detector"):
        print("\nper-detector breakdown for chosen candidate:")
        for det, p in meta["per_detector"].items():
            print(f"  {det:>14}: {p:.3f}")


if __name__ == "__main__":
    main()
