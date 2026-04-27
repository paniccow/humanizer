"""Side-by-side benchmark: every strategy in this repo on the same eval prompts.

Strategies:
  base         - the base instruction model, single sample (no humanization)
  scrub        - deterministic Stage 1 only (no model, no GPU)
  prompt       - PromptHumanizer single shot
  adversarial  - best-of-N with detector ensemble
  pipeline     - scrub -> prompt -> select -> burstiness
  trained      - load LoRA adapter (single sample)
  trained+pipe - adapter via TrainedHumanizer inside the Pipeline

For each strategy, on each of N held-out prompts, report:
  mean detector p_ai (lower = more human)
  mean similarity to source (preserve meaning)
  mean pattern fingerprint aggregate

  python scripts/benchmark.py --eval-file cloud/output/eval_prompts.jsonl --n 20
  python scripts/benchmark.py --strategies base,scrub,trained,trained+pipe --adapter cloud/adapter
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Make the local humanizer package importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class Strategy:
    name: str
    fn: Callable[[str], str]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-file", type=Path, required=False,
                   help="JSONL of {'source': str} rows. If absent, uses HC3 sample.")
    p.add_argument("--n", type=int, default=15, help="Number of eval prompts.")
    p.add_argument("--adapter", type=str, default=None,
                   help="Path to LoRA adapter (enables 'trained' strategies).")
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--strategies", type=str, default="scrub,adversarial,pipeline",
                   help="Comma-separated subset of: base,scrub,prompt,adversarial,pipeline,trained,trained+pipe")
    p.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--lite", action="store_true",
                   help="Single small detector (CPU-friendly).")
    p.add_argument("--out", type=Path, default=None,
                   help="Save full per-strategy outputs as JSON.")
    return p.parse_args()


def load_prompts(args) -> list[str]:
    if args.eval_file and args.eval_file.exists():
        rows = [json.loads(l) for l in args.eval_file.read_text().splitlines() if l.strip()]
        return [r["source"] for r in rows][: args.n]
    # Fall back to HC3 streaming.
    from datasets import load_dataset
    ds = load_dataset("andythetechnerd03/AI-human-text", split="train", streaming=True)
    out = []
    for ex in ds:
        if int(ex.get("generated", 0)) != 1:
            continue
        t = (ex.get("text") or "").strip()
        if 60 <= len(t.split()) <= 200:
            out.append(t)
            if len(out) >= args.n:
                break
    return out


def build_strategies(args) -> list[Strategy]:
    """Construct the strategies the user asked for. Skip silently if their deps fail."""
    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    out: list[Strategy] = []

    if "base" in wanted or "trained" in wanted or "trained+pipe" in wanted:
        from humanizer.humanizers import TrainedHumanizer, TrainedHumanizerConfig
        base_h = TrainedHumanizer(TrainedHumanizerConfig(base_model=args.base, adapter_path=None))
        if "base" in wanted:
            out.append(Strategy("base", lambda t: base_h.humanize(t).text))

    if "scrub" in wanted:
        from humanizer.pipeline import scrub
        out.append(Strategy("scrub", lambda t: scrub(t).text))

    prompt_h = None
    if "prompt" in wanted or "adversarial" in wanted or "pipeline" in wanted:
        from humanizer.humanizers import PromptHumanizer, PromptHumanizerConfig
        prompt_h = PromptHumanizer(PromptHumanizerConfig(model=args.openai_model))
        if "prompt" in wanted:
            out.append(Strategy("prompt", lambda t: prompt_h.humanize(t).text))

    ensemble = None
    if "adversarial" in wanted or "pipeline" in wanted or "trained+pipe" in wanted:
        from humanizer.detectors import default_ensemble
        ensemble = default_ensemble(lite=args.lite)

    if "adversarial" in wanted:
        from humanizer.humanizers import AdversarialHumanizer, AdversarialConfig
        adv = AdversarialHumanizer(prompt_h, ensemble, AdversarialConfig(n_candidates=args.n_candidates))
        out.append(Strategy("adversarial", lambda t: adv.humanize(t).text))

    if "pipeline" in wanted:
        from humanizer.pipeline import Pipeline, PipelineConfig
        pipe = Pipeline(humanizer=prompt_h, detectors=ensemble, config=PipelineConfig(n_candidates=args.n_candidates))
        out.append(Strategy("pipeline", lambda t: pipe.run(t).text))

    if "trained" in wanted:
        if not args.adapter:
            print("[warn] --adapter not provided; skipping 'trained'", file=sys.stderr)
        else:
            from humanizer.humanizers import TrainedHumanizer, TrainedHumanizerConfig
            trained_h = TrainedHumanizer(TrainedHumanizerConfig(base_model=args.base, adapter_path=args.adapter))
            out.append(Strategy("trained", lambda t: trained_h.humanize(t).text))

    if "trained+pipe" in wanted:
        if not args.adapter:
            print("[warn] --adapter not provided; skipping 'trained+pipe'", file=sys.stderr)
        else:
            from humanizer.humanizers import TrainedHumanizer, TrainedHumanizerConfig
            from humanizer.pipeline import Pipeline, PipelineConfig
            trained_h2 = TrainedHumanizer(TrainedHumanizerConfig(base_model=args.base, adapter_path=args.adapter))
            pipe2 = Pipeline(humanizer=trained_h2, detectors=ensemble, config=PipelineConfig(n_candidates=args.n_candidates))
            out.append(Strategy("trained+pipe", lambda t: pipe2.run(t).text))

    return out


def main():
    args = parse_args()
    sources = load_prompts(args)
    print(f"[bench] {len(sources)} prompts")

    strategies = build_strategies(args)
    if not strategies:
        print("[bench] no strategies selected (check --strategies and --adapter).")
        return

    # Lazy: only load detector + similarity if any strategy is going to be scored.
    from humanizer.detectors import default_ensemble
    from humanizer.metrics.semantic import embedding_similarity
    from humanizer.patterns import analyze
    ensemble = default_ensemble(lite=args.lite)

    results: dict[str, dict] = {}
    for strat in strategies:
        print(f"\n[bench] running {strat.name} on {len(sources)} prompts")
        outs: list[str] = []
        t0 = time.time()
        for i, src in enumerate(sources):
            try:
                outs.append(strat.fn(src))
            except Exception as e:  # noqa: BLE001
                print(f"  [{i+1}/{len(sources)}] error: {e}", file=sys.stderr)
                outs.append("")
            if (i + 1) % max(1, len(sources) // 5) == 0:
                print(f"  [{i+1}/{len(sources)}] done")
        elapsed = time.time() - t0

        ens_results = ensemble.score_batch(outs)
        sims = embedding_similarity(sources, outs).tolist()
        pats = [analyze(o).aggregate for o in outs]
        results[strat.name] = {
            "mean_p_ai": statistics.fmean(r.aggregate for r in ens_results),
            "mean_similarity": statistics.fmean(sims),
            "mean_pattern": statistics.fmean(pats),
            "elapsed_s": round(elapsed, 1),
            "outputs": outs,
            "p_ai": [r.aggregate for r in ens_results],
            "similarity": sims,
            "pattern": pats,
        }

    print("\n=== results ===")
    print(f"{'strategy':<18} {'p_ai':>8} {'sim':>8} {'pattern':>8} {'time':>10}")
    for name, r in results.items():
        print(
            f"{name:<18} {r['mean_p_ai']:>8.3f} {r['mean_similarity']:>8.3f} "
            f"{r['mean_pattern']:>8.3f} {r['elapsed_s']:>8.1f}s"
        )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nfull outputs: {args.out}")


if __name__ == "__main__":
    main()
