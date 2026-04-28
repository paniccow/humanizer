"""Command-line entry point.

  humanizer humanize "Some AI text..."         # zero-shot LLM via prompt
  humanizer humanize -f input.txt --adversarial  # best-of-N with detector ensemble
  humanizer detect "Is this AI?"               # run the detector ensemble
  humanizer eval -f sources.jsonl              # full evaluation report
  humanizer prepare-data -n 200                # build dataset, print summary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="Humanize AI-generated text.")
console = Console()


def _read_input(text: Optional[str], file: Optional[Path]) -> str:
    if file:
        return file.read_text()
    if text:
        return text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise typer.BadParameter("Provide TEXT or --file or pipe stdin.")


@app.command()
def humanize(
    text: Optional[str] = typer.Argument(None),
    file: Optional[Path] = typer.Option(None, "-f", "--file"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    adversarial: bool = typer.Option(False, "--adversarial", "-a"),
    n: int = typer.Option(8, "-n", help="Best-of-N candidates (adversarial only)"),
    sim_threshold: float = typer.Option(0.78, "--sim"),
    lite: bool = typer.Option(False, "--lite", help="Use single small detector (CPU/Mac)"),
    apply_burst: bool = typer.Option(False, "--burst", help="Apply burstiness post-processing"),
):
    """Humanize text. Default = single-shot prompt; --adversarial = best-of-N."""
    src = _read_input(text, file)

    from .humanizers import (
        AdversarialConfig,
        AdversarialHumanizer,
        PromptHumanizer,
        PromptHumanizerConfig,
    )

    base = PromptHumanizer(PromptHumanizerConfig(model=model, base_url=base_url))
    if adversarial:
        from .detectors import default_ensemble

        ensemble = default_ensemble(lite=lite)
        h = AdversarialHumanizer(
            base, ensemble, AdversarialConfig(n_candidates=n, similarity_threshold=sim_threshold)
        )
        result = h.humanize(src)
        console.print(f"[dim]p_ai={result.score:.3f}  attempts={result.attempts}[/dim]")
        console.print(f"[dim]similarity={result.metadata['similarity']:.3f}[/dim]\n")
        out = result.text
    else:
        out = base.humanize(src).text

    if apply_burst:
        from .postprocess import apply_burstiness

        out = apply_burstiness(out)

    console.print(out)


@app.command()
def detect(
    text: Optional[str] = typer.Argument(None),
    file: Optional[Path] = typer.Option(None, "-f", "--file"),
    lite: bool = typer.Option(False, "--lite"),
):
    """Run the detector ensemble on a piece of text."""
    src = _read_input(text, file)
    from .detectors import default_ensemble

    ensemble = default_ensemble(lite=lite)
    res = ensemble.score(src)
    table = Table(title=f"AI-detector ensemble — mean p_ai = {res.aggregate:.3f}")
    table.add_column("detector"); table.add_column("p_ai", justify="right")
    for s in res.scores:
        table.add_row(s.name, f"{s.p_ai:.3f}")
    console.print(table)


@app.command()
def scrub(
    text: Optional[str] = typer.Argument(None),
    file: Optional[Path] = typer.Option(None, "-f", "--file"),
    show_edits: bool = typer.Option(False, "--show-edits"),
):
    """Stage 1 deterministic AI-tell scrubber — runs in microseconds, no model needed."""
    src = _read_input(text, file)
    from .pipeline import scrub as _scrub
    from .patterns import analyze

    before = analyze(src)
    out = _scrub(src)
    after = analyze(out.text)
    if show_edits:
        console.print(f"[dim]edits: {out.edits}  by_kind: {out.edits_by_kind}[/dim]")
        console.print(f"[dim]pattern aggregate {before.aggregate:.2f} -> {after.aggregate:.2f}[/dim]\n")
    console.print(out.text)


@app.command()
def patterns(
    text: Optional[str] = typer.Argument(None),
    file: Optional[Path] = typer.Option(None, "-f", "--file"),
):
    """Show the AI-pattern fingerprint for a piece of text — explains *why* it reads as AI."""
    src = _read_input(text, file)
    from .patterns import analyze

    fp = analyze(src)
    console.print(fp.explain())


@app.command()
def pipeline(
    text: Optional[str] = typer.Argument(None),
    file: Optional[Path] = typer.Option(None, "-f", "--file"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    base_url: Optional[str] = typer.Option(
        None, "--base-url",
        help="OpenAI-compatible base URL (OpenRouter: https://openrouter.ai/api/v1)",
    ),
    n: int = typer.Option(16, "-n", help="Best-of-N candidates for selection"),
    refine_passes: int = typer.Option(3, "--refine-passes"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM stages — scrub + post-process only"),
    lite: bool = typer.Option(False, "--lite", help="Use single small detector"),
    show_trace: bool = typer.Option(False, "--trace", help="Print per-stage trace"),
    use_reject: bool = typer.Option(
        False, "--reject",
        help="Replace best-of-N + refine with rejection sampling against a real-world judge",
    ),
    judge: str = typer.Option(
        "auto", "--judge",
        help="Used with --reject. auto | gptzero | originality | pangram | roberta",
    ),
    reject_n: int = typer.Option(8, "--reject-n", help="Candidates per rejection round"),
    reject_rounds: int = typer.Option(4, "--reject-rounds", help="Max rejection escalation rounds"),
    reject_threshold: float = typer.Option(0.05, "--reject-threshold", help="Strict judge pass threshold"),
):
    """Full multi-stage humanization pipeline: scrub → paraphrase → best-of-N
    → iterative refine → burstiness → QA gate.

    Works with any OpenAI-compatible API. For OpenRouter:
        export OPENAI_API_KEY=sk-or-v1-...
        humanizer pipeline -f text.txt \\
          --base-url https://openrouter.ai/api/v1 \\
          --model openai/gpt-4o-mini
    """
    src = _read_input(text, file)
    from .pipeline import Pipeline, PipelineConfig
    from .patterns import analyze

    cfg = PipelineConfig(
        n_candidates=n,
        max_refine_passes=refine_passes,
        do_reject=use_reject,
        reject_candidates=reject_n,
        reject_max_rounds=reject_rounds,
        reject_p_ai_threshold=reject_threshold,
    )

    judge_det = None
    if no_llm:
        # Scrub + burstiness only — no API key needed.
        cfg.do_paraphrase = False
        cfg.do_refine = False
        cfg.do_reject = False
        humanizer = None
        ensemble = None
    else:
        from .humanizers import PromptHumanizer, PromptHumanizerConfig
        from .detectors import default_ensemble
        humanizer = PromptHumanizer(PromptHumanizerConfig(model=model, base_url=base_url))
        ensemble = default_ensemble(lite=lite)
        if use_reject:
            if judge == "auto":
                from .detectors import judge_from_env
                judge_det = judge_from_env()
            elif judge == "gptzero":
                from .detectors import GPTZeroDetector; judge_det = GPTZeroDetector()
            elif judge == "originality":
                from .detectors import OriginalityDetector; judge_det = OriginalityDetector()
            elif judge == "pangram":
                from .detectors import PangramDetector; judge_det = PangramDetector()
            elif judge == "roberta":
                from .detectors import RoBERTaDetector
                judge_det = RoBERTaDetector("roberta-large-openai-detector")
            else:
                raise typer.BadParameter(f"--judge {judge!r} not recognized")

    pipe = Pipeline(humanizer=humanizer, detectors=ensemble, config=cfg, judge=judge_det)
    result = pipe.run(src)

    before_pat = analyze(src).aggregate
    after_pat = result.final_pattern or 0.0
    console.print(
        f"[dim]pattern aggregate {before_pat:.3f} → {after_pat:.3f}  "
        f"(stages: {len(result.stages)})[/dim]\n"
    )
    if show_trace:
        for stage in result.stages:
            console.print(f"[bold]== {stage.name} ==[/bold]")
            console.print(stage.text_after[:200] + ("..." if len(stage.text_after) > 200 else ""))
            if stage.metadata:
                console.print(f"[dim]{stage.metadata}[/dim]\n")
    console.print(result.text)


@app.command()
def reject(
    text: Optional[str] = typer.Argument(None),
    file: Optional[Path] = typer.Option(None, "-f", "--file"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    judge: str = typer.Option(
        "auto", "--judge",
        help="Judge: auto (use whatever paid keys are set, fall back to roberta) | "
             "gptzero | originality | pangram (paid APIs) | roberta (local, free)",
    ),
    n: int = typer.Option(8, "-n", help="Candidates per round"),
    rounds: int = typer.Option(4, "--rounds", help="Max escalation rounds"),
    threshold: float = typer.Option(0.05, "--threshold", help="Strict pass threshold on judge p_ai"),
    sim: float = typer.Option(0.78, "--sim", help="Min cosine similarity vs original"),
    show_trace: bool = typer.Option(False, "--trace", help="Print per-round candidate scores"),
):
    """Rejection-sample candidates against a target detector until one passes.

    Best-of-N with a STRICT threshold against the real judge — the operating
    mode of paid commercial humanizers. Pair with --judge gptzero (set
    GPTZERO_API_KEY) for real-world reliability; --judge roberta is free
    local validation.
    """
    src = _read_input(text, file)
    from .humanizers import (
        PromptHumanizer,
        PromptHumanizerConfig,
        RejectionConfig,
        RejectionSamplingHumanizer,
    )

    if judge == "auto":
        from .detectors import available_paid_detectors, judge_from_env
        avail = [s.name for s in available_paid_detectors()]
        if avail:
            console.print(f"[dim]judge=auto -> ensemble({'+'.join(avail)})[/dim]")
        else:
            console.print(
                "[yellow]judge=auto -> roberta-large fallback "
                "(no paid API keys set; export ORIGINALITY_API_KEY etc. for real-world judging)[/yellow]"
            )
        judge_det = judge_from_env()
    elif judge == "gptzero":
        from .detectors import GPTZeroDetector
        judge_det = GPTZeroDetector()
    elif judge == "originality":
        from .detectors import OriginalityDetector
        judge_det = OriginalityDetector()
    elif judge == "pangram":
        from .detectors import PangramDetector
        judge_det = PangramDetector()
    elif judge == "roberta":
        from .detectors import RoBERTaDetector
        judge_det = RoBERTaDetector("roberta-large-openai-detector")
    else:
        raise typer.BadParameter(
            f"--judge must be one of: auto, gptzero, originality, pangram, roberta — got {judge!r}"
        )

    base = PromptHumanizer(PromptHumanizerConfig(model=model, base_url=base_url))

    traces: list[tuple[int, list[str], list[float]]] = []

    def _trace(round_idx: int, cands: list[str], scores: list[float]) -> None:
        traces.append((round_idx, cands, scores))

    cfg = RejectionConfig(
        candidates_per_round=n,
        max_rounds=rounds,
        p_ai_threshold=threshold,
        similarity_threshold=sim,
    )
    h = RejectionSamplingHumanizer(
        base, judge_det, cfg, on_round=_trace if show_trace else None
    )
    result = h.humanize(src)

    meta = result.metadata or {}
    passed = meta.get("passed", False)
    badge = "[green]PASSED[/green]" if passed else "[yellow]EXHAUSTED[/yellow]"
    console.print(
        f"[dim]{badge}  judge={meta.get('judge')}  "
        f"p_ai={result.score:.3f}  rounds={meta.get('rounds_used')}  "
        f"attempts={result.attempts}  judge_calls={meta.get('judge_calls')}[/dim]\n"
    )
    if show_trace:
        for round_idx, cands, scores in traces:
            console.print(f"[bold]== round {round_idx} ==[/bold]")
            for c, s in zip(cands, scores):
                marker = "[green]✓[/green]" if s < threshold else " "
                snippet = c[:80].replace("\n", " ")
                console.print(f"  {marker} p_ai={s:.3f}  {snippet}...")
        console.print()
    console.print(result.text)


@app.command(name="eval")
def eval_cmd(
    file: Path = typer.Option(..., "-f", "--file", help="JSONL with `source` field"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    adversarial: bool = typer.Option(True, "--adversarial/--no-adversarial"),
    n: int = typer.Option(8, "-n"),
    lite: bool = typer.Option(False, "--lite"),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Save samples JSONL"),
    no_perp: bool = typer.Option(False, "--no-perp", help="Skip perplexity (faster)"),
):
    """Full evaluation: ASR per detector, similarity, perplexity, burstiness."""
    sources = [json.loads(line)["source"] for line in file.read_text().splitlines() if line.strip()]
    from .detectors import default_ensemble
    from .eval import evaluate
    from .humanizers import (
        AdversarialConfig,
        AdversarialHumanizer,
        PromptHumanizer,
        PromptHumanizerConfig,
    )

    base = PromptHumanizer(PromptHumanizerConfig(model=model))
    ensemble = default_ensemble(lite=lite)
    humanizer = AdversarialHumanizer(base, ensemble, AdversarialConfig(n_candidates=n)) if adversarial else base

    report = evaluate(humanizer, sources, ensemble, compute_perplexity=not no_perp)
    console.print(report.summary())
    if out:
        out.write_text("\n".join(json.dumps(s) for s in report.samples))
        console.print(f"\nSamples written to {out}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    workers: int = typer.Option(1, "--workers"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)"),
):
    """Start the FastAPI service. Wraps the rejection sampler.

    Required env: OPENAI_API_KEY (OpenAI or OpenRouter).
    Optional env: ORIGINALITY_API_KEY / PANGRAM_API_KEY / GPTZERO_API_KEY
    for paid-detector judge (default --judge auto picks them up).
    Set HUMANIZER_API_KEY to require bearer-token auth.
    """
    try:
        import uvicorn
    except ImportError:
        raise typer.BadParameter(
            "uvicorn not installed. Install with: pip install -e '.[serve]'"
        )
    uvicorn.run(
        "humanizer.service.app:app",
        host=host, port=port, workers=workers, reload=reload,
    )


@app.command(name="prepare-data")
def prepare_data_cmd(
    n: int = typer.Option(200, "-n"),
    out: Optional[Path] = typer.Option(None, "-o", "--out"),
):
    """Build the HC3-based training dataset, print/save summary."""
    from .data import DataConfig, build

    ds = build(DataConfig(n_examples=n))
    console.print(f"Built dataset with {len(ds)} examples")
    console.print({k: ds[0][k][:120] + "..." if isinstance(ds[0][k], str) and len(ds[0][k]) > 120 else ds[0][k] for k in ds.column_names})
    if out:
        ds.to_json(str(out))
        console.print(f"Wrote {out}")


if __name__ == "__main__":
    app()
