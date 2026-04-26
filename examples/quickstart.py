"""Quickstart: humanize a single piece of text and score it.

Requires:
  pip install -e .[openai]
  export OPENAI_API_KEY=sk-...

For a CPU-only / Mac run, use lite=True (loads a single small RoBERTa-base detector).
"""
from humanizer import (
    AdversarialConfig,
    AdversarialHumanizer,
    PromptHumanizer,
    PromptHumanizerConfig,
    default_ensemble,
)


SAMPLE = (
    "In today's rapidly evolving digital landscape, artificial intelligence has "
    "emerged as a transformative force that is reshaping numerous industries. "
    "Furthermore, organizations are increasingly leveraging AI-driven solutions "
    "to enhance operational efficiency, streamline workflows, and deliver "
    "personalized experiences to their customers. Moreover, the integration of "
    "machine learning algorithms with existing infrastructure enables companies "
    "to derive actionable insights from vast amounts of data. Additionally, "
    "this paradigm shift is fundamentally altering the competitive dynamics "
    "across multiple sectors."
)


def main():
    ensemble = default_ensemble(lite=True)

    # Score the original — should look very AI.
    before = ensemble.score(SAMPLE)
    print("BEFORE:", before)

    base = PromptHumanizer(PromptHumanizerConfig(model="gpt-4o-mini"))
    humanizer = AdversarialHumanizer(
        base, ensemble, AdversarialConfig(n_candidates=8, similarity_threshold=0.78)
    )

    result = humanizer.humanize(SAMPLE)
    print(f"\nHUMANIZED (p_ai={result.score:.3f}, sim={result.metadata['similarity']:.3f}):")
    print(result.text)

    after = ensemble.score(result.text)
    print("\nAFTER:", after)


if __name__ == "__main__":
    main()
