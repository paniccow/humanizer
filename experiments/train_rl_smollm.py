"""Real REINFORCE-with-baseline RL — direct from base SmolLM2-135M-Instruct.

This is the AuthorMist-style attack scaled down to fit on a Mac. Real gradients,
real reward curve, real adapter saved. The reward combines:
  - 1 - p_ai(detector)               (RoBERTa OpenAI detector — same as eval target)
  - 1 - pattern_aggregate            (the AI-fingerprint module)

Loss:
  L = -E[ A · log π(y|x) ] + β · KL(π || π_ref)
  A = R(y) - mean(R(group))                 (group-relative baseline; GRPO-style)

We sample G=4 completions per prompt, normalize advantages within the group,
and update only LoRA params. KL keeps fluency from collapsing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, "/tmp/humanizer")
from humanizer.patterns import analyze as pattern_analyze  # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
DETECTOR = "openai-community/roberta-base-openai-detector"
OUT = Path("checkpoints/rl")
LOG = Path("logs/rl.jsonl")
EVAL_FILE = Path("data/eval.jsonl")

N_PROMPTS = 12
GROUP_SIZE = 3
LR = 1e-4
KL_BETA = 0.05
W_DETECTOR = 1.0
W_PATTERN = 0.5
MAX_NEW = 80
TEMP = 1.0
LORA_R = 8
SAVE_EVERY = 3  # save adapter every N steps so a stall doesn't lose all progress
MAX_PROMPT_WORDS = 130  # cap input length to keep generation snappy

# MPS is tight on memory holding policy + ref + detector. Disable the upper cap.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

SYSTEM = (
    "You rewrite AI-generated text so it reads as if a real person wrote it, "
    "while preserving meaning. Vary sentence length aggressively. Use contractions. "
    "Avoid stiff transitional phrases like 'Furthermore' and 'Moreover'. Avoid words "
    "like 'leverage', 'delve', 'intricate', 'multifaceted'. Output only the rewritten text."
)


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def fetch_ai_prompts(n: int) -> list[str]:
    print("Streaming andythetechnerd03/AI-human-text (parquet)...")
    ds = load_dataset("andythetechnerd03/AI-human-text", split="train", streaming=True)
    out = []
    for ex in ds:
        if int(ex.get("generated", 0)) != 1:
            continue
        t = (ex.get("text") or "").strip()
        wc = len(t.split())
        if not (60 <= wc <= MAX_PROMPT_WORDS):
            continue
        out.append(t)
        if len(out) >= n:
            break
    return out


def build_prompt(tokenizer, source: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Rewrite the following text:\n\n---\n{source}\n---"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def detector_p_ai(detector, det_tok, texts: list[str], dev: str) -> list[float]:
    enc = det_tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(dev)
    logits = detector(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    id2 = detector.config.id2label
    ai_idx = next(i for i, l in id2.items() if "fake" in str(l).lower() or "label_1" in str(l).lower())
    return probs[:, int(ai_idx)].cpu().tolist()


def reward(detector, det_tok, texts: list[str], dev: str) -> tuple[list[float], list[dict]]:
    p_ais = detector_p_ai(detector, det_tok, texts, dev)
    fps = [pattern_analyze(t) for t in texts]
    rewards = [
        W_DETECTOR * (1.0 - p) + W_PATTERN * (1.0 - fp.aggregate)
        for p, fp in zip(p_ais, fps)
    ]
    detail = [
        {"p_ai": p, "pattern": fp.aggregate, "reward": r}
        for p, fp, r in zip(p_ais, fps, rewards)
    ]
    return rewards, detail


def compute_logp(model, sequences, prompt_len):
    """Sum log-prob of completion tokens. Computes ONE sequence at a time to fit in MPS."""
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
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return torch.cat(results)


def main():
    dev = device()
    print(f"device: {dev}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading tokenizer {BASE}")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading policy {BASE}")
    policy = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to(dev)
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_cfg)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"trainable: {trainable:,} / {total:,} = {trainable/total:.2%}")

    print(f"loading reference (frozen base)")
    ref = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to(dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    print(f"loading detector {DETECTOR}")
    det_tok = AutoTokenizer.from_pretrained(DETECTOR)
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR).to(dev).eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    prompts = fetch_ai_prompts(N_PROMPTS + 10)
    eval_pool = prompts[N_PROMPTS:]
    train_pool = prompts[:N_PROMPTS]
    EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_FILE.write_text("\n".join(json.dumps({"source": s}) for s in eval_pool))
    print(f"train prompts: {len(train_pool)}  eval prompts: {len(eval_pool)}")

    optim = AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)

    log_lines = []
    t0 = time.time()
    for step, src in enumerate(train_pool):
        prompt_text = build_prompt(tokenizer, src)
        enc = tokenizer(prompt_text, return_tensors="pt").to(dev)
        prompt_len = enc.input_ids.shape[1]

        # Sample G completions
        policy.eval()
        with torch.no_grad():
            seqs = policy.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                do_sample=True,
                temperature=TEMP,
                top_p=0.95,
                max_new_tokens=MAX_NEW,
                num_return_sequences=GROUP_SIZE,
                pad_token_id=tokenizer.eos_token_id,
            )
        completions = [tokenizer.decode(s[prompt_len:], skip_special_tokens=True).strip() for s in seqs]

        rewards, detail = reward(detector, det_tok, completions, dev)
        rewards_t = torch.tensor(rewards, device=dev)
        baseline = rewards_t.mean()
        std = rewards_t.std() + 1e-6
        advantages = (rewards_t - baseline) / std

        # log-prob under policy and ref
        policy.train()
        logp_policy = compute_logp(policy, seqs, prompt_len)
        with torch.no_grad():
            # ref is the base model (no adapter); policy uses LoRA on top.
            # For KL we want: KL(policy || base) — penalizes drifting too far from base.
            logp_ref = compute_logp(ref, seqs, prompt_len)

        # Normalize log-probs by completion length so the per-token magnitude is sane.
        n_tokens = (seqs.shape[1] - prompt_len)
        n_tokens = max(n_tokens, 1)

        pg_loss = -(advantages.detach() * (logp_policy / n_tokens)).mean()
        kl = ((logp_policy - logp_ref) / n_tokens).mean()
        loss = pg_loss + KL_BETA * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        optim.step()
        optim.zero_grad()

        line = {
            "step": step + 1,
            "mean_reward": float(rewards_t.mean().item()),
            "max_reward": float(rewards_t.max().item()),
            "mean_p_ai": float(sum(d["p_ai"] for d in detail) / len(detail)),
            "mean_pattern": float(sum(d["pattern"] for d in detail) / len(detail)),
            "kl": float(kl.item()),
            "pg_loss": float(pg_loss.item()),
            "loss": float(loss.item()),
            "best_completion": completions[int(rewards_t.argmax())][:200],
            "t": round(time.time() - t0, 1),
        }
        log_lines.append(line)
        print(
            f"step {step+1:3d}  R̄={line['mean_reward']:.3f}  "
            f"R_max={line['max_reward']:.3f}  p_ai={line['mean_p_ai']:.3f}  "
            f"patt={line['mean_pattern']:.3f}  KL={line['kl']:.4f}  "
            f"loss={line['loss']:+.3f}  t={line['t']}s",
            flush=True,
        )
        # Save incrementally so a stall doesn't lose all progress.
        if (step + 1) % SAVE_EVERY == 0:
            policy.save_pretrained(str(OUT))
            LOG.write_text("\n".join(json.dumps(l) for l in log_lines))
            print(f"  [checkpoint at step {step+1}]", flush=True)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    LOG.write_text("\n".join(json.dumps(l) for l in log_lines))
    policy.save_pretrained(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    print(f"\nRL adapter saved to {OUT}/")
    if log_lines:
        first_n = log_lines[: min(5, len(log_lines))]
        last_n = log_lines[-min(5, len(log_lines)) :]
        print(
            f"reward (first-5 -> last-5 mean):  "
            f"{sum(l['mean_reward'] for l in first_n)/len(first_n):.3f} -> "
            f"{sum(l['mean_reward'] for l in last_n)/len(last_n):.3f}"
        )
        print(
            f"p_ai   (first-5 -> last-5 mean):  "
            f"{sum(l['mean_p_ai'] for l in first_n)/len(first_n):.3f} -> "
            f"{sum(l['mean_p_ai'] for l in last_n)/len(last_n):.3f}"
        )
        print(
            f"pattern(first-5 -> last-5 mean):  "
            f"{sum(l['mean_pattern'] for l in first_n)/len(first_n):.3f} -> "
            f"{sum(l['mean_pattern'] for l in last_n)/len(last_n):.3f}"
        )


if __name__ == "__main__":
    main()
