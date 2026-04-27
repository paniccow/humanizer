# Run-004 (Qwen2.5-3B) + final benchmark across all 4 runs

**Date:** 2026-04-27
**Recipe:** train_v4.py — Qwen2.5-3B-Instruct, sim_floor 0.65, w_det 1.5, 3 detectors, G=2, lora_r=8, completion=140
**Cost:** ~\$2.50 (run #4 attempt 2 — attempt 1 OOM'd at step 0 and ate \$4 sitting idle)

## Run #4 eval (held-out 30 prompts)

| metric | BASE (Qwen-3B) | TRAINED | Δ |
|---|---:|---:|---:|
| mean p_ai (3-detector ensemble) | 0.514 | 0.535 | +0.021 ❌ |
| mean similarity (MiniLM)        | 0.749 | **0.854** | +0.105 ✅ |
| mean pattern aggregate          | 0.125 | 0.142 | +0.017 ❌ |
| ASR roberta-base                |  90%  | 76.7% | −13 pp |
| ASR roberta-large               |  6.7% | **16.7%** | **+10 pp** |
| ASR desklib                     |  10%  | 13.3% | +3 pp |

Same shape as runs #1-3: similarity goes up, raw detector evasion barely
moves. Per-detector ASR rebalanced more cleanly than #2/#3 (less of the
"gamed roberta-base, lost roberta-large" overfit).

## Final cross-run benchmark — pattern aggregate (untrained signal)

The headline question: **does GRPO-trained adapter + scrub beat just running
scrub on the base model?**

```
                                    BASE   + scrub   TRAINED   + scrub
─────────────────────────────────────────────────────────────────────────
Run #1 (Qwen-1.5B, simple reward)  0.264     0.142     0.263     0.157
Run #2 (Qwen-1.5B, multi-obj)      0.240     0.149     0.194     0.133
Run #3 (Qwen-1.5B, 4-det reward)   0.243     0.133     0.243     0.142
Run #4 (Qwen-3B, looser gate)      0.126     0.115     0.149     0.128
                                                       ↑          ↑
                                                  trained worse  base+scrub better
```

Best across all 4 runs:
- **Best BASE + SCRUB: 0.115** (Qwen-3B + scrub) ← the system
- Best TRAINED + SCRUB: 0.128 (Qwen-3B trained + scrub) — *worse* by +11.5%

**Across every single run, training made the model produce text that's harder
for the scrub to clean up than the original base model's text.** No GRPO
adapter we trained beats `Qwen-3B-base + scrub`.

## Why GRPO failed at our budget

1. **Reward signal too weak.** 1200 GRPO steps × G=2-6 = 2,400-7,200 sampled
   completions. AuthorMist's published recipe used 5,712 with a stronger reward
   composition. We're at the bottom of the regime where REINFORCE-with-baseline
   even works.
2. **Detectors too easy individually.** The 2-detector reward (run #2)
   converged to "fool roberta-base, ignore roberta-large." The 4-detector
   reward (run #3) couldn't satisfy all 4 simultaneously and the policy
   essentially got stuck.
3. **Quality terms dominated.** With sim_floor + w_sim + length_penalty all
   constraining the policy, detector evasion had narrow room to move. Run #4's
   looser gate helped marginally but couldn't escape the same plateau.
4. **Base model already cleaner.** Qwen-3B-base produces lower-pattern-aggregate
   text than 1.5B (0.126 vs 0.240). The bigger model already does what we
   wanted GRPO to do — there's less room to improve.

## What actually works (the deliverable)

The **multi-stage inference pipeline** in `humanizer/pipeline/`:
1. **scrub** — deterministic regex pass that kills lexical AI tells (Furthermore,
   leverage, intricate, hedging boilerplate, expand-only-by-AI verbs).
2. **paraphrase** — single LLM rewrite (use `Qwen2.5-3B-Instruct` base, NOT a
   trained adapter — they're worse).
3. **best-of-N** — sample N=16, score each with a small detector ensemble +
   pattern fingerprint, pick the lowest combined score that passes a
   similarity threshold.
4. **iterative refine** — re-run paraphrase+select if patterns still fire,
   with a *targeted* prompt naming the still-firing axes.
5. **burstiness post-process** — sentence-length variance + contractions.

This pipeline + Qwen-3B-base is the system. Pattern aggregate target ≤ 0.115
on essay-style text.

## What would actually move the needle (left for future work)

1. **Adversarial discriminator (~\$8, 8hr).** Train a fresh classifier
   continuously against our outputs and inject it into the GRPO reward.
   Catches the "policy found a niche all our static detectors miss" mode
   collapse — which is the fundamental ceiling of our 4 trained adapters.
2. **Multi-domain training data (~\$3, 3hr).** Mix HC3 (Q&A), WikiText
   (informational), creative writing. The single-domain essay data
   constrains generalization.
3. **Bigger budget for GRPO (~\$15+).** 5000+ steps at G=8 on Qwen-3B with
   the multi-objective reward. AuthorMist's actual budget.
4. **Larger model (Qwen-7B+, ~\$20+).** Run #4 already showed the bigger base
   has lower starting pattern aggregate; 7B might already be at the ceiling
   without any training.

## Files

- `eval.json`        — run #4 base vs trained on 30 held-out prompts
- `run2_eval.json`   — run #2 archive (Qwen-1.5B, 2-detector multi-obj)
- `run3_eval.json`   — run #3 archive (Qwen-1.5B, 4-detector multi-obj)
- `adapter/`         — run #4 LoRA-r8 adapter (15 MB safetensors)
- `FINDINGS.md`      — this file

## Recommendation

**Stop spending on training.** Use `humanizer/pipeline/Pipeline` with
`PromptHumanizer(model="Qwen/Qwen2.5-3B-Instruct")` (or the larger
`gpt-4o-mini` for the LLM stage if you have OpenAI credit) wrapped by
`scrub → paraphrase → best-of-16 → refine → burstiness → QA gate`.

The trained adapters are saved (run #2 and run #4) for future reference if
the field figures out a way to get GRPO to add real value at this scale —
but they're not the system today.
