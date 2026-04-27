# Run-001 (cloud, RunPod 4090): findings

**Hardware:** SECURE RTX 4090 24GB on RunPod, ~67 minutes wall-clock, **~$0.38 spent**
**Recipe:** Hand-rolled REINFORCE+KL (no TRL — version hell — see commit `3095708`).
Qwen2.5-1.5B-Instruct + LoRA-r16 + 600 prompts × G=4. KL β=0.05. Reward = 1 − mean(p_ai).

## Held-out eval (30 prompts the model never saw during training)

| metric                     |   BASE  |   TRAINED  |  Δ  |
|----------------------------|--------:|-----------:|----:|
| mean p_ai (ensemble)       |  0.503  |    0.498   | −0.005 |
| mean similarity (MiniLM)   |  0.811  |    0.818   | +0.007 |
| ASR (ensemble, p_ai<0.5)   |   70%   |    73%     | +3 pp |
| ASR roberta-base           |  76.7%  |    86.7%   | **+10 pp** |
| ASR roberta-large          |  23.3%  |    13.3%   | **−10 pp** |

**Headline:** trained adapter is ≈ break-even with base on the ensemble headline
metric. **It overfit to roberta-base** (the smaller, more-foolable detector) at
the cost of being *worse* against roberta-large. Net ASR +3 pp is real but
weak; in absolute terms training added effectively nothing on the harder detector.

## Why this happened

1. **Reward was detector-only.** Composite `R = 1 − mean(p_ai)` doesn't penalize
   per-detector divergence. The policy can satisfy the loss by satisfying the
   easier detector, even when the harder one disagrees. Confirmed: roberta-base
   ASR went up while roberta-large went down.
2. **Under-budget.** AuthorMist used 714 × G=8 = 5,712 sampled completions on a
   3B model. We did 600 × G=4 = 2,400 completions on a 1.5B model. ~3-4× short.
3. **No semantic / pattern guardrails.** The trained policy could in principle
   shift outputs toward gibberish-shaped strings the smaller detector likes. KL
   penalty bounded this but didn't prevent the per-detector overfit.

## Stacking the deterministic scrub on top of trained outputs

Pattern aggregate (humanizer/patterns) on the 30 held-out outputs:

| strategy             | mean pattern |  vs base |
|----------------------|-------------:|---------:|
| BASE                 |    0.264     |    —     |
| TRAINED              |    0.263     |   0.000  |
| **TRAINED + SCRUB**  |  **0.158**   | **−0.40 (40% rel)** |

Detector-side numbers for `TRAINED+SCRUB` not yet computed (would require
loading RoBERTa locally; deferring until run #2 finishes so we can do all
detector-bound benchmarks on the pod). But pattern aggregate is a strong
predictor — the scrub is removing the lexical AI tells the trained model
left behind.

## What changes for run #2

Already locked in (`cloud/train_v2.py`, queued via `cloud/queue_run2.sh`):

- **Multi-objective reward.** `R = 1.0·(1−p_ai) + 0.4·sim + 0.3·(1−pattern) + length_penalty`.
- **Hard semantic gate.** Candidates with cos-sim < 0.78 to the source get
  advantage = −1, regardless of detector score. Kills the meaning-tanking
  reward-hack failure mode this run avoided only by accident.
- **Larger budget.** 1200 steps × G=6 = 7,200 sampled completions. Closer to
  AuthorMist's 5,712 baseline, with extra headroom for the constrained reward.
- **Same Qwen2.5-1.5B base** — model size isn't the bottleneck; reward design is.
  Qwen-3B can come in run #3 once we know multi-obj reward converges.

Expected: run #2 should hit ASR > 80% on BOTH detectors (vs run #1's 86.7% / 13.3%
split), with similarity ≥ 0.83 and pattern aggregate ≤ 0.20 — i.e. genuine human-
sounding outputs, not classifier hacks.

## Files

- `run1_adapter/`         — trained LoRA adapter (17 MB safetensors)
- `run1_eval.json`        — full per-example outputs, p_ai, similarity
- `run1_training.log`     — 600 step-by-step lines from training
- `run1_benchmark.json`   — pattern aggregates incl. trained+scrub
- `FINDINGS.md`           — this file
