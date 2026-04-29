"""GRPO inner loop — hand-rolled, no TRL dependency.

Runs ON the rented 4090 pod. Same algorithm as TRL's GRPOTrainer:
  loss = -E[ A · log π(y|x) ] + β · KL(π || π_ref)
  A    = (R(y) - mean(R(group))) / std(R(group))             — group-relative
  R(y) = 1 - mean(p_ai across detector ensemble)

Memory trick: instead of loading two copies of the policy, we load Qwen-1.5B
ONCE with a LoRA adapter on top. For the *policy* logp we forward as normal;
for the *reference* logp we disable the adapter via `model.disable_adapter()`.
This halves model VRAM vs naive TRL.

Outputs (under /workspace/output):
  /workspace/output/adapter/        — LoRA adapter
  /workspace/output/eval.json       — base vs trained on 30 held-out prompts
  /workspace/output/training_log.json
  /workspace/output/done            — sentinel on success
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    beta_kl: float = float(os.environ.get("HUMANIZER_BETA", 0.05))
    num_generations: int = int(os.environ.get("HUMANIZER_G", 4))
    max_completion_length: int = 200
    temperature: float = 0.95
    save_every: int = 50

    lora_r: int = int(os.environ.get("HUMANIZER_LORA_R", 16))
    use_qlora: bool = os.environ.get("HUMANIZER_QLORA", "0") == "1"


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


def detector_p_ai(detectors, texts):
    import torch
    n = len(detectors)
    per = []
    with torch.no_grad():
        for d, dt, ai_idx in detectors:
            enc = dt(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
            logits = d(**enc).logits.float()
            per.append(torch.softmax(logits, dim=-1)[:, ai_idx].cpu().tolist())
    agg = [sum(per[k][i] for k in range(n)) / n for i in range(len(texts))]
    return agg, per


def compute_logp(model, sequences, prompt_len):
    """Sum log-prob of completion tokens. One sequence at a time to save VRAM."""
    import torch
    import torch.nn.functional as F
    results = []
    for i in range(sequences.shape[0]):
        seq = sequences[i:i+1]
        out = model(seq)
        logits = out.logits[:, :-1, :]
        targets = seq[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        gather = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(gather)
        mask[:, prompt_len - 1:] = 1.0
        results.append((gather * mask).sum(dim=-1))
        del out, logits, log_probs, gather, mask
    return torch.cat(results)


def train(cfg: Cfg) -> tuple[list[str], list[dict]]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    else:
        base_model = base_model.cuda()

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    policy = get_peft_model(base_model, lora_cfg)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"[lora] trainable: {trainable:,} / {total:,} = {trainable/total:.2%}", flush=True)

    detectors = load_detectors(cfg)

    prompts = load_prompts(cfg, cfg.n_train_prompts + cfg.n_eval_prompts)
    train_prompts = prompts[: cfg.n_train_prompts]
    eval_prompts = prompts[cfg.n_train_prompts :]
    (OUT / "eval_prompts.jsonl").write_text("\n".join(json.dumps({"source": p}) for p in eval_prompts))
    print(f"[data] train={len(train_prompts)}  eval={len(eval_prompts)}", flush=True)

    optim = AdamW([p for p in policy.parameters() if p.requires_grad], lr=cfg.learning_rate)

    log_lines: list[dict] = []
    t0 = time.time()
    for step, src in enumerate(train_prompts):
        prompt_text = format_prompt(tokenizer, src)
        enc = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        prompt_len = enc.input_ids.shape[1]

        # Sample G completions
        policy.eval()
        with torch.no_grad():
            seqs = policy.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=0.95,
                max_new_tokens=cfg.max_completion_length,
                num_return_sequences=cfg.num_generations,
                pad_token_id=tokenizer.eos_token_id,
            )
        completions = [tokenizer.decode(s[prompt_len:], skip_special_tokens=True).strip() for s in seqs]
        rewards_p_ai, _ = detector_p_ai(detectors, completions)
        rewards = [1.0 - p for p in rewards_p_ai]
        rewards_t = torch.tensor(rewards, device="cuda")
        baseline = rewards_t.mean()
        std = rewards_t.std() + 1e-6
        advantages = (rewards_t - baseline) / std

        policy.train()
        logp_policy = compute_logp(policy, seqs, prompt_len)
        # Reference policy = same model with LoRA adapter disabled (saves a full model copy).
        with torch.no_grad():
            with policy.disable_adapter():
                logp_ref = compute_logp(policy, seqs, prompt_len)

        n_tokens = max(seqs.shape[1] - prompt_len, 1)
        pg_loss = -(advantages.detach() * (logp_policy / n_tokens)).mean()
        kl = ((logp_policy - logp_ref) / n_tokens).mean()
        loss = pg_loss + cfg.beta_kl * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        optim.step()
        optim.zero_grad()
        torch.cuda.empty_cache()

        line = {
            "step": step + 1,
            "mean_reward": float(rewards_t.mean().item()),
            "max_reward": float(rewards_t.max().item()),
            "mean_p_ai": float(sum(rewards_p_ai) / len(rewards_p_ai)),
            "kl": float(kl.item()),
            "pg_loss": float(pg_loss.item()),
            "loss": float(loss.item()),
            "best_completion": completions[int(rewards_t.argmax())][:200],
            "t": round(time.time() - t0, 1),
        }
        log_lines.append(line)
        print(
            f"step {step+1:4d}/{len(train_prompts)}  "
            f"R̄={line['mean_reward']:.3f}  R_max={line['max_reward']:.3f}  "
            f"p_ai={line['mean_p_ai']:.3f}  KL={line['kl']:+.4f}  "
            f"loss={line['loss']:+.3f}  t={line['t']}s",
            flush=True,
        )

        if (step + 1) % cfg.save_every == 0:
            policy.save_pretrained(str(ADAPTER_DIR))
            LOG_PATH.write_text(json.dumps(log_lines, indent=2))
            print(f"  [checkpoint at step {step+1}]", flush=True)

    policy.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    LOG_PATH.write_text(json.dumps(log_lines, indent=2))
    print(f"[save] adapter -> {ADAPTER_DIR}", flush=True)
    return eval_prompts, log_lines


def evaluate(cfg: Cfg, eval_prompts: list[str]):
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
            do_sample=True, temperature=0.85, top_p=0.95,
            max_new_tokens=cfg.max_completion_length,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def evaluate_model(model, label):
        outs = [humanize(model, s) for s in eval_prompts]
        ens, per_d = detector_p_ai(detectors, outs)
        sims = cosine(eval_prompts, outs)
        return {
            "label": label,
            "n": len(outs),
            "mean_p_ai_ensemble": statistics.fmean(ens),
            "mean_similarity": statistics.fmean(sims),
            "asr_ensemble": sum(1 for p in ens if p < 0.5) / max(len(ens), 1),
            "asr_per_detector": {
                cfg.detector_ids[i].split("/")[-1]: sum(1 for p in per_d[i] if p < 0.5) / max(len(per_d[i]), 1)
                for i in range(len(detectors))
            },
            "outputs": outs,
            "p_ai_ensemble": ens,
            "p_ai_per_detector": {cfg.detector_ids[i].split("/")[-1]: per_d[i] for i in range(len(detectors))},
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


def main():
    cfg = Cfg()
    print(f"[cfg] {cfg}", flush=True)
    eval_prompts, _ = train(cfg)
    evaluate(cfg, eval_prompts)
    DONE_PATH.write_text("ok\n")
    print("[done]", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[error] {e}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
