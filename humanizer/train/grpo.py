"""GRPO training — the main attack.

Faithful reimplementation of the AuthorMist recipe (arXiv:2503.08716, 2025):

  * Base policy: Qwen2.5-3B-Instruct (small enough to RL on a single H100 / A100).
  * Algorithm: Group Relative Policy Optimization (DeepSeek-R1's algorithm).
    Sample G candidates per prompt, compute advantage as (reward - group_mean).
    No critic network needed.
  * Reward: 1 - mean(p_ai) across an ensemble of open-source AI-text detectors.
  * Quality preservation: KL divergence to the SFT-warm-started reference policy.
    AuthorMist showed this is sufficient — explicit semantic terms in the reward
    are unnecessary and can cause reward hacking.

Hardware budget:
  * Production run: 1× H100 80GB, ~16 GPU-hours, ~10K samples.
  * Smoke test:    1× A100 40GB or 1× RTX 4090 (with QLoRA), ~1K samples, 2-3 hours.
  * Mac users: train on rented GPU (RunPod, Lambda, Vast.ai, Modal). Inference
    works on Mac/CPU once the LoRA adapter is downloaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.prepare import DataConfig, build
from ..detectors import default_ensemble
from ..detectors.ensemble import DetectorEnsemble
from ..humanizers.prompt import _SYSTEM_PROMPT


@dataclass
class GRPOConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    sft_adapter: str | None = "checkpoints/sft"      # warm start; None = train from base
    output_dir: str = "checkpoints/grpo"

    # GRPO knobs
    num_generations: int = 8           # G in the paper — candidates per prompt
    max_prompt_length: int = 1024
    max_completion_length: int = 1024
    temperature: float = 0.9
    top_p: float = 0.95
    learning_rate: float = 5e-6
    beta: float = 0.001                # KL penalty coefficient
    epochs: int = 1
    batch_size: int = 1                # one prompt per device, G generations from it
    grad_accum: int = 8

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Detector ensemble — heavier reward = harder to reward-hack
    detector_lite: bool = False
    detector_device: str | None = None

    # Data
    data: DataConfig = field(default_factory=lambda: DataConfig(n_examples=10000))


def _build_prompt(text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Rewrite the following text:\n\n---\n{text}\n---"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _make_reward_fn(ensemble: DetectorEnsemble):
    """Returns a TRL-compatible reward function: (prompts, completions, **) -> list[float]."""

    def reward_fn(prompts, completions, **_):
        # `completions` may be List[List[dict]] (chat) or List[str]; normalize.
        texts: list[str] = []
        for c in completions:
            if isinstance(c, list) and c and isinstance(c[0], dict):
                texts.append(c[0].get("content", ""))
            else:
                texts.append(str(c))
        # Reward per the AuthorMist paper: 1 - mean p_ai across detectors.
        return ensemble.reward_batch(texts)

    return reward_fn


def run(cfg: GRPOConfig | None = None) -> str:
    cfg = cfg or GRPOConfig()
    try:
        from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
        from peft import LoraConfig, PeftModel
    except ImportError as e:
        raise ImportError(
            "GRPO training requires the train extras. Install with: pip install 'humanizer[train]'"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if cfg.sft_adapter:
        model = PeftModel.from_pretrained(model, cfg.sft_adapter, is_trainable=True)

    peft_config = (
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if cfg.use_lora and cfg.sft_adapter is None
        else None
    )

    raw = build(cfg.data)
    ds: Dataset = raw.map(
        lambda e: {"prompt": _build_prompt(e["source"], tokenizer)},
        remove_columns=[c for c in raw.column_names if c != "source"],
    )

    ensemble = default_ensemble(device=cfg.detector_device, lite=cfg.detector_lite)
    reward_fn = _make_reward_fn(ensemble)

    training_args = TRLGRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        beta=cfg.beta,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=["wandb"] if _wandb_available() else [],
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def _wandb_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("wandb") is not None


if __name__ == "__main__":
    run()
