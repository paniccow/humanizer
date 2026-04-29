"""Phase 2 of Run #8: SFT Qwen-3B + LoRA on (ai_input, human_target) pairs.

This is supervised seq2seq, NOT GRPO. The training signal is dense — every
output token has a real human target. No detector in the loop.

Recipe:
  - Load Qwen2.5-3B-Instruct base in bf16 (~6GB)
  - Add LoRA r=16 adapter (more capacity than GRPO's r=8)
  - Format each pair as a chat completion: system + user(ai) -> assistant(human)
  - Mask everything except the assistant tokens from the loss
  - 2-3 epochs over ~5000 (ai, human) pairs
  - Standard cross-entropy on target tokens
  - LR 1e-4 (10× higher than GRPO; supervised is more forgiving)

VRAM budget on 22GB-effective 4090:
  Qwen-3B bf16:           ~6 GB
  LoRA-r16 + grads + optim: ~1.5 GB
  Activations (batch=4):   ~6 GB
  Total:                   ~14 GB. Plenty of headroom.

Cost: ~2h × $0.69/hr = ~$1.40. No detector calls, no Pangram.

Run on the pod (after dataset is uploaded):
  TRAIN_SCRIPT=train_v8.py ADAPTER_OUT=./adapter-r8 EVAL_OUT=./eval-r8.json bun run launch.ts

Reads the dataset from /workspace/code/dataset_v8.jsonl (uploaded by the
TS launcher). One JSONL line = {"ai": "...", "human": "..."}.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Avoid CUDA frag on long SFT runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WORKSPACE = Path(os.environ.get("HUMANIZER_WORKSPACE", "/workspace"))
OUT = WORKSPACE / "output"
ADAPTER_DIR = OUT / "adapter"
EVAL_PATH = OUT / "eval.json"
LOG_PATH = OUT / "training_log.json"
DONE_PATH = OUT / "done"
DATASET_PATH = WORKSPACE / "code" / "dataset_v8.jsonl"


@dataclass
class Cfg:
    base_model: str = os.environ.get("HUMANIZER_BASE", "Qwen/Qwen2.5-3B-Instruct")
    dataset_path: str = os.environ.get("HUMANIZER_DATASET", str(DATASET_PATH))
    n_eval_holdout: int = int(os.environ.get("HUMANIZER_EVAL_N", 100))
    epochs: int = int(os.environ.get("HUMANIZER_EPOCHS", 2))
    # batch=2 max for Qwen-3B SFT on 24GB. Default attempt at batch=4 OOM'd
    # during cross-entropy because seq2seq activations are hefty: 4 × 512
    # × 3072 hidden × 36 layers × 2 bytes ≈ 9GB just for activations.
    # Use grad_accum=4 to keep effective batch=8.
    batch_size: int = int(os.environ.get("HUMANIZER_BATCH", 2))
    grad_accum: int = int(os.environ.get("HUMANIZER_GRAD_ACCUM", 4))
    learning_rate: float = float(os.environ.get("HUMANIZER_LR", 1e-4))
    lora_r: int = int(os.environ.get("HUMANIZER_LORA_R", 16))
    lora_alpha: int = int(os.environ.get("HUMANIZER_LORA_ALPHA", 32))
    lora_dropout: float = float(os.environ.get("HUMANIZER_LORA_DROPOUT", 0.05))
    warmup_steps: int = int(os.environ.get("HUMANIZER_WARMUP", 50))
    save_every: int = int(os.environ.get("HUMANIZER_SAVE_EVERY", 200))
    log_every: int = int(os.environ.get("HUMANIZER_LOG_EVERY", 10))
    max_seq_len: int = int(os.environ.get("HUMANIZER_MAX_SEQ", 384))   # was 512; tighten to fit
    # Gradient checkpointing trades compute for memory. ~30% slower but
    # frees ~5-7 GB of activations. Required to fit Qwen-3B SFT on 24GB.
    use_grad_checkpointing: bool = os.environ.get("HUMANIZER_GRAD_CKPT", "1") == "1"
    gen_temperature: float = 0.95
    gen_top_p: float = 0.95
    gen_max_new: int = 200
    seed: int = 42


SYSTEM_PROMPT = (
    "You rewrite AI-generated text so it reads as if a real person wrote it. "
    "PRESERVE every fact, number, name, date, place, and entity from the source "
    "exactly. Do not invent new content, do not drop content, do not change "
    "the topic. Match the same content, in a more human style. Output only "
    "the rewrite, no preamble or explanation."
)


def format_pair(tokenizer, ai: str, human: str, max_len: int):
    """Format one (ai, human) pair with the Qwen chat template. Returns
    input_ids and labels (with everything before the assistant response
    masked to -100 so the loss only flows through the human target tokens).

    Renders to string then encodes — bulletproof across transformers
    versions. Some versions of apply_chat_template(tokenize=True) returned
    weird types in our env, hence the explicit detour.
    """
    import torch

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ai},
        {"role": "assistant", "content": human},
    ]
    full_str = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(full_str, add_special_tokens=False)

    # Compute the prompt-prefix length so we can mask labels
    prompt_msgs = msgs[:2]
    prompt_str = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    if len(full_ids) > max_len:
        # Truncate from the END of the assistant response if too long.
        full_ids = full_ids[:max_len]

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[: prompt_len] = -100   # ignore prompt in loss
    return input_ids, labels


def collate(batch, pad_token_id):
    """Pad input_ids + labels to the longest in the batch."""
    import torch
    maxlen = max(b[0].shape[0] for b in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, maxlen), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, maxlen), -100, dtype=torch.long)
    attn_mask = torch.zeros((bsz, maxlen), dtype=torch.long)
    for i, (ids, lbls) in enumerate(batch):
        L = ids.shape[0]
        input_ids[i, :L] = ids
        labels[i, :L] = lbls
        attn_mask[i, :L] = 1
    return input_ids, attn_mask, labels


def load_pairs(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def train(cfg: Cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[cfg] {cfg}", flush=True)
    print(f"[gpu] {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading pairs from {cfg.dataset_path}", flush=True)
    pairs = load_pairs(cfg.dataset_path)
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    eval_pairs = pairs[: cfg.n_eval_holdout]
    train_pairs = pairs[cfg.n_eval_holdout:]
    print(f"[data] train={len(train_pairs)}  eval={len(eval_pairs)}", flush=True)
    (OUT / "eval_pairs.jsonl").write_text("\n".join(json.dumps(p) for p in eval_pairs))

    print(f"[model] loading {cfg.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16,
    ).cuda()

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing — required to fit batch=2+ Qwen-3B SFT on 24GB.
    # Must be enabled AFTER PEFT wrapping; also disable use_cache (incompatible
    # with checkpointing during training).
    if cfg.use_grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        # PEFT requires this so gradients flow through frozen base model
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print(f"[mem] gradient checkpointing enabled (saves ~5-7 GB activations)", flush=True)

    # Preformat all training examples once (saves CPU later)
    print(f"[data] tokenizing pairs (max_seq_len={cfg.max_seq_len})...", flush=True)
    formatted = [
        format_pair(tokenizer, p["ai"], p["human"], cfg.max_seq_len)
        for p in train_pairs
    ]
    # Filter out anything that ended up with no target tokens (rare, can happen
    # if the prompt alone exceeds max_seq_len)
    formatted = [(ids, lbls) for ids, lbls in formatted if (lbls != -100).any()]
    print(f"[data] {len(formatted)} usable pairs after tokenization", flush=True)

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(0.9, 0.999), weight_decay=0.01,
    )

    total_steps = (len(formatted) // cfg.batch_size) * cfg.epochs // cfg.grad_accum
    print(f"[train] total optim steps ≈ {total_steps}", flush=True)

    log_lines: list[dict] = []
    t0 = time.time()
    global_step = 0
    optim_step = 0
    for epoch in range(cfg.epochs):
        rng.shuffle(formatted)
        # Build batches
        for batch_start in range(0, len(formatted), cfg.batch_size):
            batch = formatted[batch_start: batch_start + cfg.batch_size]
            if len(batch) == 0:
                continue
            input_ids, attn_mask, labels = collate(batch, tokenizer.pad_token_id)
            input_ids = input_ids.cuda()
            attn_mask = attn_mask.cuda()
            labels = labels.cuda()

            model.train()
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            global_step += 1

            if global_step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                # Linear warmup
                if optim_step < cfg.warmup_steps:
                    lr = cfg.learning_rate * (optim_step + 1) / cfg.warmup_steps
                    for pg in optim.param_groups:
                        pg["lr"] = lr
                else:
                    for pg in optim.param_groups:
                        pg["lr"] = cfg.learning_rate
                optim.step()
                optim.zero_grad()
                optim_step += 1
                torch.cuda.empty_cache()

            if global_step % cfg.log_every == 0:
                line = {
                    "epoch": epoch + 1, "step": global_step, "optim_step": optim_step,
                    "loss": float(out.loss.detach().item()),
                    "lr": float(optim.param_groups[0]["lr"]),
                    "t": round(time.time() - t0, 1),
                }
                log_lines.append(line)
                print(
                    f"epoch {epoch+1}/{cfg.epochs}  step {global_step}  "
                    f"opt {optim_step}/{total_steps}  loss={line['loss']:.4f}  "
                    f"lr={line['lr']:.2e}  t={line['t']}s",
                    flush=True,
                )

            if optim_step > 0 and optim_step % cfg.save_every == 0 and global_step % cfg.grad_accum == 0:
                model.save_pretrained(str(ADAPTER_DIR))
                tokenizer.save_pretrained(str(ADAPTER_DIR))
                LOG_PATH.write_text(json.dumps(log_lines, indent=2))
                DONE_PATH.touch()
                print(f"  [checkpoint at optim_step {optim_step}; done sentinel touched]", flush=True)

    # Final save
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    LOG_PATH.write_text(json.dumps(log_lines, indent=2))
    print(f"[save] adapter -> {ADAPTER_DIR}", flush=True)
    return model, tokenizer, eval_pairs, log_lines


def evaluate(cfg: Cfg, model, tokenizer, eval_pairs):
    """Generate trained-model output for held-out AI prompts. Save raw output;
    detector scoring happens later (free, locally) via run_eval_detectors.py."""
    import torch
    print(f"[eval] generating outputs for {len(eval_pairs)} held-out pairs", flush=True)
    model.eval()
    base_outputs: list[str] = []
    trained_outputs: list[str] = []
    sources: list[str] = []
    targets: list[str] = []

    for i, p in enumerate(eval_pairs):
        ai = p["ai"]
        human = p["human"]
        sources.append(ai)
        targets.append(human)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ai},
        ]
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        # Trained output (with adapter on)
        with torch.no_grad():
            seq = model.generate(
                input_ids=ids,
                do_sample=True,
                temperature=cfg.gen_temperature,
                top_p=cfg.gen_top_p,
                max_new_tokens=cfg.gen_max_new,
                pad_token_id=tokenizer.eos_token_id,
            )
        trained = tokenizer.decode(seq[0, ids.shape[1]:], skip_special_tokens=True).strip()
        trained_outputs.append(trained)
        # Base output (with adapter disabled)
        with torch.no_grad():
            with model.disable_adapter():
                seq = model.generate(
                    input_ids=ids,
                    do_sample=True,
                    temperature=cfg.gen_temperature,
                    top_p=cfg.gen_top_p,
                    max_new_tokens=cfg.gen_max_new,
                    pad_token_id=tokenizer.eos_token_id,
                )
        base = tokenizer.decode(seq[0, ids.shape[1]:], skip_special_tokens=True).strip()
        base_outputs.append(base)
        if (i + 1) % 10 == 0:
            print(f"  eval {i+1}/{len(eval_pairs)}", flush=True)

    eval_data = {
        "n": len(eval_pairs),
        "sources": sources,
        "targets": targets,
        "base": {"outputs": base_outputs, "label": "BASE"},
        "trained": {"outputs": trained_outputs, "label": "TRAINED"},
    }
    EVAL_PATH.write_text(json.dumps(eval_data, indent=2))
    print(f"[eval] wrote {EVAL_PATH}", flush=True)


def main():
    cfg = Cfg()
    model, tokenizer, eval_pairs, log_lines = train(cfg)
    evaluate(cfg, model, tokenizer, eval_pairs)
    DONE_PATH.touch()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
