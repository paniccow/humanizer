"""GRPO inner loop — runs ON the rented 4090 pod.

Invoked by launch.ts via SSH after upload. Keeps Python surface area minimal
so the user only ever touches the TS layer.

Recipe (AuthorMist-faithful, scaled to 24GB VRAM):
  Base    : Qwen2.5-1.5B-Instruct + QLoRA-r16 (≈7M trainable / 1.5B)
  Reward  : 1 - mean(p_ai) across roberta-base + roberta-large openai-detectors
  Algo    : TRL GRPOTrainer, G=8, β_KL=0.001, lr=5e-6
  Steps   : 600 (close to AuthorMist's 714)
  Data    : 600 prompts streamed from andythetechnerd03/AI-human-text

Outputs (under /workspace):
  /workspace/output/adapter/        — LoRA adapter + tokenizer
  /workspace/output/eval.json       — base vs trained on 30 held-out prompts
  /workspace/output/training_log.json
  /workspace/output/done            — sentinel file (exit code 0 only)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# All paths are pod-side absolute, configured by launch.ts via env if needed.
WORKSPACE = Path(os.environ.get("HUMANIZER_WORKSPACE", "/workspace"))
OUT = WORKSPACE / "output"
ADAPTER_DIR = OUT / "adapter"
EVAL_PATH = OUT / "eval.json"
LOG_PATH = OUT / "training_log.json"
DONE_PATH = OUT / "done"


@dataclass
class Cfg:
    base_model: str = os.environ.get("HUMANIZER_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
    dataset: str = os.environ.get("HUMANIZER_DATASET", "andythetechnerd03/AI-human-text")
    detector_ids: tuple[str, ...] = (
        "openai-community/roberta-base-openai-detector",
        "openai-community/roberta-large-openai-detector",
    )
    n_train_prompts: int = int(os.environ.get("HUMANIZER_TRAIN_N", 600))
    n_eval_prompts: int = int(os.environ.get("HUMANIZER_EVAL_N", 30))
    min_words: int = 60
    max_words: int = 200

    learning_rate: float = float(os.environ.get("HUMANIZER_LR", 5e-6))
    beta_kl: float = float(os.environ.get("HUMANIZER_BETA", 0.001))
    num_generations: int = int(os.environ.get("HUMANIZER_G", 8))
    max_prompt_length: int = 768
    max_completion_length: int = 384
    temperature: float = 0.9
    per_device_batch: int = 1
    grad_accum: int = 8
    epochs: int = 1
    lora_r: int = int(os.environ.get("HUMANIZER_LORA_R", 16))
    use_qlora: bool = os.environ.get("HUMANIZER_QLORA", "1") == "1"


SYSTEM_PROMPT = (
    "You rewrite AI-generated text so it reads as if a real person wrote it, "
    "while preserving meaning. Vary sentence length aggressively. Use contractions. "
    "Avoid stiff transitional phrases like 'Furthermore' and 'Moreover'. Avoid words "
    "like 'leverage', 'delve', 'intricate', 'multifaceted'. Output ONLY the rewritten text."
)


def format_prompt(tokenizer, source: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Rewrite the following text:\n\n---\n{source}\n---"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_prompts(cfg: Cfg, n: int) -> list[str]:
    from datasets import load_dataset

    print(f"[data] streaming {cfg.dataset}...", flush=True)
    ds = load_dataset(cfg.dataset, split="train", streaming=True)
    out: list[str] = []
    for ex in ds:
        if int(ex.get("generated", 0)) != 1:
            continue
        t = (ex.get("text") or "").strip()
        wc = len(t.split())
        if cfg.min_words <= wc <= cfg.max_words:
            out.append(t)
            if len(out) >= n:
                break
    print(f"[data] got {len(out)} prompts", flush=True)
    return out


def load_detectors(cfg: Cfg):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    detectors = []
    for det_id in cfg.detector_ids:
        print(f"[detector] loading {det_id}", flush=True)
        d = AutoModelForSequenceClassification.from_pretrained(
            det_id, torch_dtype=torch.bfloat16
        ).cuda().eval()
        for p in d.parameters():
            p.requires_grad_(False)
        dt = AutoTokenizer.from_pretrained(det_id)
        id2 = d.config.id2label
        ai_idx = next(
            i for i, l in id2.items()
            if "fake" in str(l).lower() or "label_1" in str(l).lower() or i == d.config.num_labels - 1
        )
        detectors.append((d, dt, int(ai_idx)))
    return detectors


def make_reward_fn(detectors):
    import torch

    @torch.no_grad()
    def p_ai(model, tok, ai_idx, texts):
        enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
        logits = model(**enc).logits.float()
        return torch.softmax(logits, dim=-1)[:, ai_idx].cpu().tolist()

    def reward_fn(prompts, completions, **_):
        texts = []
        for c in completions:
            if isinstance(c, list) and c and isinstance(c[0], dict):
                texts.append(c[0].get("content", ""))
            else:
                texts.append(str(c))
        ensemble = [p_ai(d, tok, idx, texts) for d, tok, idx in detectors]
        n = len(detectors)
        return [1.0 - sum(ensemble[k][i] for k in range(n)) / n for i in range(len(texts))]

    return reward_fn


def train(cfg: Cfg) -> tuple[list[str], list[Any]]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[gpu] {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] loading {cfg.base_model} (qlora={cfg.use_qlora})", flush=True)
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if cfg.use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if cfg.use_qlora:
        base_model = prepare_model_for_kbit_training(base_model)

    detectors = load_detectors(cfg)
    reward_fn = make_reward_fn(detectors)

    prompts = load_prompts(cfg, cfg.n_train_prompts + cfg.n_eval_prompts)
    train_prompts = prompts[: cfg.n_train_prompts]
    eval_prompts = prompts[cfg.n_train_prompts :]
    (OUT / "eval_prompts.jsonl").write_text("\n".join(json.dumps({"source": p}) for p in eval_prompts))

    train_ds = Dataset.from_list([{"prompt": format_prompt(tokenizer, p)} for p in train_prompts])

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=2 * cfg.lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = GRPOConfig(
        output_dir=str(ADAPTER_DIR),
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        beta=cfg.beta_kl,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        report_to=[],
        gradient_checkpointing=True,
        warmup_ratio=0.03,
    )

    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("[train] starting GRPO", flush=True)
    trainer.train()

    print(f"[save] adapter -> {ADAPTER_DIR}", flush=True)
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    LOG_PATH.write_text(json.dumps(trainer.state.log_history, indent=2))

    return eval_prompts, trainer.state.log_history


def evaluate(cfg: Cfg, eval_prompts: list[str]) -> dict:
    import statistics
    import torch
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[eval] starting held-out comparison", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    detectors = load_detectors(cfg)

    @torch.no_grad()
    def score(texts):
        per = []
        for d, dt, idx in detectors:
            enc = dt(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
            logits = d(**enc).logits.float()
            per.append(torch.softmax(logits, dim=-1)[:, idx].cpu().tolist())
        n = len(detectors)
        agg = [sum(per[k][i] for k in range(n)) / n for i in range(len(texts))]
        return agg, {cfg.detector_ids[i].split("/")[-1]: per[i] for i in range(n)}

    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cuda()

    def cosine(a, b):
        ea = emb.encode(a, normalize_embeddings=True, convert_to_tensor=True)
        eb = emb.encode(b, normalize_embeddings=True, convert_to_tensor=True)
        return (ea * eb).sum(-1).cpu().tolist()

    @torch.no_grad()
    def humanize(model, src):
        prompt = format_prompt(tokenizer, src)
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=0.85,
            top_p=0.95,
            max_new_tokens=cfg.max_completion_length,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def evaluate_model(model, label):
        outs = [humanize(model, s) for s in eval_prompts]
        ens, per_d = score(outs)
        sims = cosine(eval_prompts, outs)
        return {
            "label": label,
            "n": len(outs),
            "mean_p_ai_ensemble": statistics.fmean(ens),
            "mean_similarity": statistics.fmean(sims),
            "asr_ensemble": sum(1 for p in ens if p < 0.5) / max(len(ens), 1),
            "asr_per_detector": {
                name: sum(1 for p in scores if p < 0.5) / max(len(scores), 1)
                for name, scores in per_d.items()
            },
            "outputs": outs,
            "p_ai_ensemble": ens,
            "p_ai_per_detector": per_d,
            "similarity": sims,
        }

    print("[eval] BASE", flush=True)
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16).cuda().eval()
    res_base = evaluate_model(base, "BASE")
    del base
    torch.cuda.empty_cache()

    print("[eval] TRAINED", flush=True)
    trained = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16).cuda()
    trained = PeftModel.from_pretrained(trained, str(ADAPTER_DIR)).cuda().eval()
    res_trained = evaluate_model(trained, "TRAINED")

    summary = {
        "n": len(eval_prompts),
        "base":    {k: v for k, v in res_base.items()    if not isinstance(v, (list, dict)) or k.startswith("asr")},
        "trained": {k: v for k, v in res_trained.items() if not isinstance(v, (list, dict)) or k.startswith("asr")},
        "delta_p_ai_ensemble": res_trained["mean_p_ai_ensemble"] - res_base["mean_p_ai_ensemble"],
        "delta_similarity":    res_trained["mean_similarity"]    - res_base["mean_similarity"],
        "delta_asr_ensemble":  res_trained["asr_ensemble"]       - res_base["asr_ensemble"],
    }
    EVAL_PATH.write_text(json.dumps({"summary": summary, "base": res_base, "trained": res_trained}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    cfg = Cfg()
    print(f"[cfg] {cfg}", flush=True)
    eval_prompts, _log = train(cfg)
    evaluate(cfg, eval_prompts)
    DONE_PATH.write_text("ok\n")
    print("[done]", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}", flush=True)
        sys.exit(1)
