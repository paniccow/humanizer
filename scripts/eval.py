"""Quick evaluation: load an HC3 split, humanize, score against the ensemble."""
import json
from pathlib import Path

from humanizer import (
    AdversarialConfig,
    AdversarialHumanizer,
    PromptHumanizer,
    PromptHumanizerConfig,
    default_ensemble,
    evaluate,
)
from humanizer.data import DataConfig, build


def main(n: int = 50, lite: bool = True, out: str = "outputs/eval.jsonl"):
    ds = build(DataConfig(n_examples=n))
    sources = [ex["source"] for ex in ds]

    ensemble = default_ensemble(lite=lite)
    base = PromptHumanizer(PromptHumanizerConfig())
    humanizer = AdversarialHumanizer(base, ensemble, AdversarialConfig(n_candidates=8))

    report = evaluate(humanizer, sources, ensemble)
    print(report.summary())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(json.dumps(s) for s in report.samples))


if __name__ == "__main__":
    main()
