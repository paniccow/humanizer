# humanizer

Take AI-generated text and turn it into something that reads like a person wrote it — preserving meaning, factual content, and writing quality.

After 4 GRPO training runs (~$8) and many iterations on a deterministic scrub pipeline, **the actual deliverable is the multi-stage inference pipeline**, not any trained adapter. Across all 4 runs, `Qwen2.5-3B-Instruct base + scrub` produces lower AI-pattern scores than any GRPO-trained adapter. See `experiments/run-004/FINDINGS.md` for the cross-run benchmark that drove this conclusion.

**Headline numbers** (pattern aggregate, lower = more human, 13 signals).
Reproduce with `python scripts/headline_benchmark.py`:

```
                                       BASE    +SCRUB   TRAINED    +SCRUB
Run #4 (Qwen-3B base)                  0.108    0.067    0.122     0.080
```

- Across all 4 GRPO runs, **best `BASE+SCRUB` is 0.067** — 38% lower than the cleanest raw base output, 70% lower than 1.5B-base outputs.
- After scrub, 87% of 240 outputs across all runs trigger ZERO pattern flags. The 13% remaining mostly have 1 of: burstiness (rare), `contraction_deficit`, or `modality_overload`.
- Full pipeline with LLM stage (`scrub + paraphrase + best-of-16 + refine + burstiness`): not yet measured end-to-end. Expected to push lower.

55 tests passing. CLI: `humanizer pipeline -f input.txt --no-llm` runs the deterministic-only stack with no API key.

## Why this approach

The literature (2023-2025) converges on three findings:

1. **Paraphrasing breaks detectors.** Krishna et al.'s DIPPER (NeurIPS 2023) showed an 11B paraphraser can drop DetectGPT's accuracy from 70% → 5% at 1% FPR while preserving semantics.
2. **Detector-as-reward beats detector-as-target.** AuthorMist (arXiv 2503.08716, 2025) trains Qwen2.5-3B with GRPO using the detector ensemble's `1 - mean(p_AI)` as the reward and reaches 78-96% attack success rate across six commercial detectors. They show explicit semantic / quality terms in the reward are unnecessary and cause reward hacking — KL to the reference policy preserves both.
3. **Best-of-N at inference adds 5-15 ASR points** for free (Adversarial Paraphrasing, arXiv 2506.07001, 2025): sample N candidates, score each with the detector ensemble, pick the lowest-detection one that passes a similarity threshold.

This repo implements all three layers, so you can use it without training (inference-only) or train your own policy on rented GPU.

A fourth layer was added on top: a deterministic **scrub** stage that strips
the lexical AI tells (Furthermore/leverage/intricate/...) in microseconds
without touching a model — see `humanizer/pipeline/scrub.py`. Composes with
everything above.

## What's in this repo

| Layer | Module | Use without training? |
|------|--------|------------------------|
| **Deterministic scrub** (Stage 1) | `humanizer/pipeline/scrub.py` | **yes — instant, no model**. Handles transitions, AI-favorite vocabulary, hedging boilerplate, em-dash density, archaic-formal markers, tricolons, sentence splitting/merging, variance injection, a/an article repair, quoted-speech protection. ~50 rules. |
| **AI-pattern fingerprint** (13 signals) | `humanizer/patterns/` | yes. Burstiness, stiff transitions, favorite words, em-dash density, hedging, tricolons, contraction deficit, n-gram repetition, type-token ratio, sentence-start uniformity, abstract subjects, enumeration shapes, modality overload. |
| **Multi-stage pipeline** | `humanizer/pipeline/pipeline.py` | yes. `scrub → paraphrase → best-of-N → iterative refine → burstiness → QA gate`. Each stage independently toggleable. |
| **Open-source AI-text detectors** + ensemble | `humanizer/detectors/` | yes. RoBERTa-base/large OpenAI, Desklib DeBERTa, Binoculars zero-shot, held-out factory. |
| **Adversarial best-of-N** humanizer | `humanizer/humanizers/adversarial.py` | yes — needs API/local LLM |
| **Burstiness post-processor** | `humanizer/postprocess/` | yes |
| **Trained policy loader** (LoRA adapter) | `humanizer/humanizers/trained.py` | needs adapter; per FINDINGS, BASE outperforms across all 4 runs |
| **GRPO training stack on rented 4090s** | `cloud/` (TypeScript orchestration + Python inner loop) | needs RunPod account |
| **CLI** | `humanizer/cli.py` | `humanize / detect / scrub / patterns / pipeline / eval / prepare-data` |

## Quick start (no training, no API key)

```bash
git clone https://github.com/paniccow/humanizer
cd humanizer
pip install -e .

# Stage-1 deterministic scrub (microseconds, no models)
humanizer scrub "Furthermore, organizations leverage the intricate complexities of AI." --show-edits

# Just see what's flagged in some text
humanizer patterns -f input.txt

# Full pipeline, deterministic-only mode (still no API key)
humanizer pipeline -f input.txt --no-llm

# Full pipeline with an LLM (needs OPENAI_API_KEY)
humanizer pipeline -f input.txt --model gpt-4o-mini -n 16
```

## Architecture

```
                 ┌─────────────────────┐
   AI text ──►   │  PromptHumanizer    │  base policy (any LLM via OpenAI-compatible API
                 │  or TrainedHumanizer│   OR a LoRA adapter trained with GRPO)
                 └──────────┬──────────┘
                            │  N candidates
                            ▼
                 ┌─────────────────────┐
                 │ AdversarialHumanizer│  best-of-N selector
                 └──────────┬──────────┘
                            │  filter by sentence-similarity, then argmin p_ai
                            ▼
                 ┌─────────────────────┐
                 │  DetectorEnsemble   │  RoBERTa-base/large + DeBERTa-v3-large + (optional) Binoculars
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ BurstinessPostProc. │  optional surface edits to bump perplexity & sentence variance
                 └──────────┬──────────┘
                            ▼
                       Humanized text
```

## Install

```bash
git clone https://github.com/paniccow/humanizer
cd humanizer

# Inference only — runs on Mac/CPU
pip install -e .[openai]

# Training — needs CUDA
pip install -e .[train,openai]
```

Set up credentials:

```bash
cp .env.example .env
# edit .env — at minimum OPENAI_API_KEY
```

## Use it (no training required)

### Python

```python
from humanizer import (
    PromptHumanizer, PromptHumanizerConfig,
    AdversarialHumanizer, AdversarialConfig,
    default_ensemble,
)

ensemble  = default_ensemble(lite=True)              # one RoBERTa, fits on a Mac
base      = PromptHumanizer(PromptHumanizerConfig(model="gpt-4o-mini"))
humanizer = AdversarialHumanizer(base, ensemble, AdversarialConfig(n_candidates=8))

result = humanizer.humanize("Furthermore, the model leverages cutting-edge ...")
print(result.text)
print(f"p_ai={result.score:.3f} similarity={result.metadata['similarity']:.3f}")
```

### CLI

```bash
# Single-shot
humanizer humanize "Some AI text here..."

# Best-of-8 with detector ensemble (recommended)
humanizer humanize -f input.txt --adversarial -n 8

# Score with ensemble
humanizer detect "Is this AI generated?"

# Full eval over 50 HC3 examples → JSONL of per-sample detail
humanizer eval -f sources.jsonl -o outputs/eval.jsonl
```

## Train your own policy

Two-stage training following AuthorMist:

### 1. Build the dataset

```bash
humanizer prepare-data -n 10000 -o data/processed/hc3.jsonl
```

Pulls aligned (human-answer, ChatGPT-answer) pairs from HC3 across all domains.

### 2. Supervised warm-start (optional, ~1h on 1× A100)

```bash
python scripts/train_sft.py
```

Trains a LoRA-r16 adapter on Qwen2.5-3B-Instruct to mimic the human side of HC3. This stabilizes GRPO. Skipping it works but takes more steps to converge.

### 3. GRPO against the detector ensemble (~16 GPU-hours on 1× H100)

```bash
python scripts/train_grpo.py
```

Defaults match the AuthorMist paper: lr 5e-6, β (KL) 0.001, G=8 generations, 1 epoch over 10K prompts.

### Hardware

| Stage | Min VRAM | Time | Notes |
|------|---------|------|------|
| Inference (LoRA + base) | 8 GB | — | Mac MPS works for inference |
| SFT warm-start (QLoRA) | 12 GB | 1 h on A100 | RTX 4090 fine |
| GRPO | 40 GB | 4-16 h on H100 | Use `detector_lite=True` to fit on 24 GB at quality cost |

Mac users: do training on **RunPod / Lambda / Vast.ai / Modal**. The repo includes no model weights — everything is on HuggingFace Hub.

## Evaluation

`humanizer eval` reports:

- **Attack-Success-Rate (ASR)** per detector — fraction where `p_ai < 0.5`
- **Mean ensemble p_ai** — primary headline metric
- **Semantic similarity** — MiniLM cosine to source (target ≥ 0.78)
- **Perplexity (GPT-2)** — fluency guardrail
- **Burstiness CV** — sentence-length variance (humans ≈ 0.4-0.7)

A trained policy + adversarial best-of-8 should hit ≥ 80% ASR on the open-source ensemble while keeping semantic similarity ≥ 0.85.

## What this won't do

- **No commercial detector training.** Reward signal only includes open-source detectors. Generalization to GPTZero/Originality/Turnitin happens via ensemble diversity, not direct optimization. AuthorMist's reported 78-96% ASR includes commercial detectors but training there would require their APIs.
- **No watermark removal beyond paraphrase.** Paraphrasing already breaks every published watermark scheme; no extra step needed.
- **No "undetectable forever" guarantee.** Detectors evolve. Re-train periodically as the ensemble drifts.

## Research basis

| Paper | Contribution used here |
|------|------------------------|
| Krishna et al., *Paraphrasing evades detectors of AI-generated text* (NeurIPS 2023) — DIPPER | Confirms paraphrase as the universal attack vector |
| Hans et al., *Spotting LLMs with Binoculars* (ICML 2024) | Binoculars zero-shot detector wrapper for the ensemble |
| AuthorMist (arXiv 2503.08716, 2025) | GRPO recipe, reward function, hyperparameters |
| Adversarial Paraphrasing (arXiv 2506.07001, 2025) | Best-of-N inference-time attack |
| CoPA (arXiv 2505.15337, 2025) | Contrastive selection — not used directly but informs the selector |

## Repo layout

```
humanizer/
├── detectors/           # AI-text detectors (RoBERTa, DeBERTa, Binoculars, Ensemble)
├── humanizers/          # Paraphrasers (Prompt, Adversarial best-of-N, Trained)
├── postprocess/         # Burstiness post-processor (sentence-length, contractions)
├── metrics/             # Semantic similarity, perplexity, burstiness stats
├── data/                # Dataset builder (HC3-based)
├── train/               # SFT and GRPO training scripts
├── eval.py              # End-to-end evaluation harness
└── cli.py               # Typer-based CLI
scripts/                 # CLI wrappers around train.* and eval
tests/                   # Pure-python tests (no model downloads)
examples/                # Quickstart code
```

## Ethics

This tool exists for legitimate uses: privacy, creative writing, avoiding false-positive AI-detector flags on genuine human work, and security research on detector robustness. Detectors have well-documented false-positive bias against non-native English speakers and certain writing styles — paraphrase tools are one defense.

Don't use it to commit academic fraud. If you wouldn't sign your name to the original AI text, fixing the wording isn't going to make it honest.

## License

Apache-2.0.
