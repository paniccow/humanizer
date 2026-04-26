"""Supervised fine-tuning warm-start.

Runs on the (ai_text, human_text) pairs from `humanizer.data.prepare`. The
goal is NOT to make the model great at humanization on its own — it's to
shift the policy distribution toward producing paraphrase-shaped outputs so
that GRPO converges faster and more stably.

Hardware: ~6 GB VRAM with QLoRA on Qwen2.5-3B (rank 16). Will run on a single
consumer GPU or any Colab/Lambda/RunPod instance. Skip on a Mac — MPS doesn't
play well with TRL/PEFT yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from ..data.prepare import DataConfig, build
from ..humanizers.prompt import _SYSTEM_PROMPT


@dataclass
class SFTConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "checkpoints/sft"
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    bf16: bool = True
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    data: DataConfig = field(default_factory=lambda: DataConfig(n_examples=5000))


def _format(example: dict, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def run(cfg: SFTConfig | None = None) -> str:
    cfg = cfg or SFTConfig()
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
    except ImportError as e:
        raise ImportError(
            "Training requires the train extras. Install with: pip install 'humanizer[train]'"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16 if cfg.bf16 else torch.float16}
    if cfg.use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if cfg.use_qlora:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    raw = build(cfg.data)
    ds: Dataset = raw.map(lambda e: _format(e, tokenizer), remove_columns=raw.column_names)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=10,
        save_strategy="epoch",
        report_to=["wandb"] if _wandb_available() else [],
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=False,
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
