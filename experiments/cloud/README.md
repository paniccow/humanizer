# Real GPU training on Modal

Runs the full GRPO recipe (AuthorMist-faithful, scaled to fit a $5-8 budget) on
a rented A100. You don't need any local GPU — just the Modal CLI.

## What you get

- **Base:** Qwen2.5-1.5B-Instruct (11× our SmolLM2-135M experiment)
- **Algorithm:** TRL's `GRPOTrainer` — group-relative PPO without a critic
- **Reward:** ensemble of `roberta-base-openai-detector` + `roberta-large-openai-detector`
- **Steps:** 600 GRPO steps, 8 generations per prompt, β_KL = 0.001
- **LoRA:** rank 16 on q/k/v/o (~10M trainable / 1.5B params)
- **Dataset:** 600 AI-labelled prompts from `andythetechnerd03/AI-human-text` (parquet, streams fast)
- **Held-out eval:** 30 prompts the model never sees during training

## Cost

| GPU            | Modal $/hr | Wall-clock | Total      |
|----------------|-----------:|-----------:|-----------:|
| A100-40GB      |     ~$2.50 |     2-3 hr |    $5-8    |
| H100           |     ~$5.50 |   1-1.5 hr |    $6-9    |

**$30 of free Modal credit on signup** — first run is effectively $0.

## Three commands

```bash
# 1. Install Modal (one-time)
pip install modal

# 2. Sign up + auth (one-time)
modal token new        # opens a browser, follow the link, paste the token

# 3. Launch
cd experiments/cloud
modal run modal_train.py::train_main
```

That's it. Modal builds the image (~3-5 min the first time), provisions an A100,
and starts training. You can close your terminal — it keeps running.

Watch progress in the Modal dashboard, or stream logs with:
```bash
modal app logs humanizer-grpo
```

## After training

```bash
# Run the held-out eval (compares BASE vs TRAINED on the same 30 prompts)
modal run modal_train.py::evaluate_main

# Pull the trained LoRA adapter to your machine (~30 MB)
modal run modal_train.py::download
# → experiments/cloud/adapter/
```

The adapter loads via `peft.PeftModel.from_pretrained(base, "experiments/cloud/adapter")`
and works with `humanizer.humanizers.TrainedHumanizer` from the main package.

## Tuning knobs (edit `Cfg` in `modal_train.py`)

| Knob                  | Default | What changes                                        |
|-----------------------|--------:|-----------------------------------------------------|
| `base_model`          | Qwen2.5-1.5B-Instruct | Bigger = better quality, costs more. Try `Qwen2.5-3B-Instruct` on H100. |
| `n_train_prompts`     |   600   | More = better generalization, scales linearly with cost |
| `num_generations` (G) |     8   | Smaller = faster, less stable advantage estimate    |
| `learning_rate`       |  5e-6   | Higher = faster, riskier (KL blows up)              |
| `beta_kl`             | 0.001   | Higher = stays closer to base (safer; less gain)    |
| `lora_r`              |    16   | More capacity, more memory                          |
| `epochs`              |     1   | AuthorMist used 1; more epochs over the same data risks overfit |

## Sanity check the run worked

After `evaluate_main`, look at `eval.json` (downloaded with the adapter):

```json
{
  "delta_p_ai_ensemble": -0.45,    // negative = model fooled detectors more
  "delta_similarity":     0.01,    // ~0 = meaning preserved
  "delta_asr_ensemble":   0.55     // positive = +55pp attack success rate
}
```

Reasonable target for the small budget: `delta_p_ai_ensemble ≤ −0.30` and
`delta_similarity ≥ −0.05`.

## Why Modal vs alternatives

| Option              | Pros                                  | Cons                                |
|--------------------|---------------------------------------|-------------------------------------|
| **Modal** (this)   | Python-native, $30 free credit, no SSH | Per-second billing only on running container |
| RunPod             | Cheapest spot rates                   | Need Docker image, SSH workflow     |
| Lambda             | Best Jupyter UX                       | Hourly billing, no spot             |
| Vast.ai            | Cheapest absolute                     | Shared hosts, variability           |
| Colab Pro          | $10/mo flat                           | A100 not always available; session timeouts |

Switching to another provider is a half-day project — the training script
(`train()` body) is portable; only the `@app.function` decorator and image
need to change.

## Troubleshooting

- **"OOM during generation"**: drop `num_generations` to 4 or `max_completion_length` to 256.
- **"Volume not committing"**: the `volume.commit()` calls inside `train()` are explicit; if you Ctrl-C mid-step, intermediate state may not persist. Restart from a `save_steps` checkpoint.
- **"Reward stays flat"**: check `eval.json` for the held-out p_ai. If it's dropping, the training is fine but variance is high. Re-run with `--n-train-prompts 1200` for a longer run.
