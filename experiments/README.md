# Experiments

Real, executed training + eval runs (not just scripts). Each subdirectory has:

- training/eval logs (raw)
- saved model artifacts (LoRA adapters)
- a `FINDINGS.md` with the honest result

## Runs

- **[run-001/](./run-001/FINDINGS.md)** — SmolLM2-135M, REINFORCE, Mac MPS.
  Outcome: 12-step RL didn't shift the policy meaningfully (p_ai went *up*
  from 0.89 to 0.99 on held-out). But the same model + best-of-6 selection
  dropped p_ai from 0.78 to 0.37 (−52% relative). Confirms the literature:
  best-of-N is the killer recipe at small scale; trained policies need real
  GPU budget.

## Top-level scripts

- `train_rl_smollm.py` — small REINFORCE-with-GRPO-baseline loop, runs on Mac
- `eval_base_vs_rl.py` — apples-to-apples eval of base vs RL-tuned policy
- `eval_best_of_n.py` — training-free best-of-N at inference time
