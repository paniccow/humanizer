# humanizer

Train an LLM to rewrite AI-generated text so it reads as if a human wrote it — fooling AI-text detectors while preserving meaning, factual content, and writing quality.

This is an ML research / learning project. It implements the current state of the art in detector evasion (as of early 2026): a small instruction-tuned LLM fine-tuned with **GRPO** against an ensemble of open-source AI-text detectors, with **KL regularization** preserving fluency and semantic similarity.

## Why this approach

The literature (2023-2025) converges on three findings:

1. **Paraphrasing breaks detectors.** Krishna et al.'s DIPPER (NeurIPS 2023) showed an 11B paraphraser can drop DetectGPT's accuracy from 70% → 5% at 1% FPR while preserving semantics.
2. **Detector-as-reward beats detector-as-target.** AuthorMist (arXiv 2503.08716, 2025) trains Qwen2.5-3B with GRPO using the detector ensemble's `1 - mean(p_AI)` as the reward and reaches 78-96% attack success rate across six commercial detectors. They show explicit semantic / quality terms in the reward are unnecessary and cause reward hacking — KL to the reference policy preserves both.
3. **Best-of-N at inference adds 5-15 ASR points** for free (Adversarial Paraphrasing, arXiv 2506.07001, 2025): sample N candidates, score each with the detector ensemble, pick the lowest-detection one that passes a similarity threshold.

This repo implements all three layers, so you can use it without training (inference-only) or train your own policy on rented GPU.

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
