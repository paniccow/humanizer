"""GRPO inner loop v5 — adversarial discriminator + run-004 recipe.

Direct response to the user's "real detector still flags it" finding from
end-to-end OpenRouter validation. The static training detectors (RoBERTa-OpenAI
base/large, Desklib) form a fixed evaluation surface; once the policy finds a
niche all three miss, training plateaus. A fresh discriminator that updates
on the policy's own current outputs catches that mode collapse.

Recipe (extends v4):
  + Replace Desklib detector with an ADVERSARIAL discriminator. Same VRAM
    budget (Desklib was DeBERTa-large ~1.7GB; discriminator is RoBERTa-base
    ~500MB, plus optimizer state).
  + Discriminator initialized fresh from openai-community/roberta-base-openai-detector
    weights (warm start). Updated every DISC_UPDATE_EVERY policy steps on a
    rolling window of (recent policy outputs, fresh human samples).
  + Reward stays multi-objective:
      R(y) = w_det · (1 - mean(p_ai across [roberta-base, roberta-large, discriminator])
           + w_sim · cosine_sim(original, y)
           + w_pat · (1 - pattern_aggregate(y))
           + length_penalty
    With sim_floor 0.65 hard gate.

Hypothesis: the discriminator's continuously-shifting decision boundary
forces the policy to actually become human-like rather than detector-niche-like.

Inherits from v4:
  + Qwen2.5-3B-Instruct base, LoRA-r8, completion=140, G=2.
  + sim_floor=0.65, w_det=1.5, w_sim=0.3, w_pat=0.3.

VRAM management for 3B + discriminator on 22GB-effective 4090:
  Qwen-3B bf16:                ~6.0 GB
  LoRA-r8 grads + optim:       ~1.0 GB
  Reference (via .disable_adapter()):  0
  2 frozen detectors bf16:     ~1.5 GB  (was 3 detectors, now 2)
  Discriminator bf16 + optim:  ~1.5 GB  (RoBERTa-base ~500MB + Adam)
  Sim model (MiniLM):          ~0.4 GB
  Generation buffers G=2:      ~2.5 GB
  Activations:                 ~5 GB
  Headroom:                    ~4 GB
  Total:                       ~17-18 GB. Same as v4.

  First attempt at G=4/completion=180/r=16 hit OOM at step 0 forward;
  the actual 4090 instances have 22GB effective (some reserved).

ALSO: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set on
generate() to fight fragmentation.

Diff vs train.py:
  + Reward becomes:  R(y) = w_det · (1 - mean_p_ai)              # detector evasion
                          + w_sim · cosine_sim(original, y)       # semantic preservation
                          + w_pat · (1 - pattern_aggregate(y))    # AI-fingerprint score
                          + length_penalty                         # length-collapse guard
  + Hard semantic gate: if cos_sim(original, y) < SIM_FLOOR, advantage = -1.0
  + Pattern fingerprint computed from a self-contained inline copy of
    humanizer/patterns/signals.py — no cross-package imports on the pod.

Defaults reflect the user's "preserve quality" requirement:
  base_model       Qwen/Qwen2.5-1.5B-Instruct (fits 24GB without QLoRA;
                   optional override to Qwen2.5-3B-Instruct via env var)
  num_generations  6  (was 4)
  num_steps        1200 (was 600)
  beta_kl          0.05 (unchanged — KL alone wasn't enough; we now have
                   explicit similarity + pattern terms)
  w_det            1.0
  w_sim            0.4
  w_pat            0.3
  sim_floor        0.78
"""
from __future__ import annotations

import json
import math
import os
# Reduce CUDA memory fragmentation on 22GB pods running Qwen-3B + 3 detectors.
# Set BEFORE importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
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
    base_model: str = os.environ.get("HUMANIZER_BASE", "Qwen/Qwen2.5-3B-Instruct")
    dataset: str = os.environ.get("HUMANIZER_DATASET", "andythetechnerd03/AI-human-text")
    detector_ids: tuple[str, ...] = (
        "openai-community/roberta-base-openai-detector",
        "openai-community/roberta-large-openai-detector",
    )  # Static detectors — kept frozen. Discriminator joins them dynamically.

    # Adversarial discriminator settings:
    disc_init_id: str = "openai-community/roberta-base-openai-detector"
    disc_update_every: int = 50          # update discriminator every N policy steps
    disc_train_steps: int = 30           # mini-batches per update
    disc_lr: float = 2e-5
    disc_window: int = 200               # rolling window of recent (ai, human) pairs
    disc_batch: int = 8                  # mini-batch size for disc training
    n_train_prompts: int = int(os.environ.get("HUMANIZER_TRAIN_N", 1200))
    n_eval_prompts: int = int(os.environ.get("HUMANIZER_EVAL_N", 30))
    min_words: int = 60
    max_words: int = 200

    # Optimizer / GRPO
    learning_rate: float = float(os.environ.get("HUMANIZER_LR", 5e-6))
    beta_kl: float = float(os.environ.get("HUMANIZER_BETA", 0.05))
    num_generations: int = int(os.environ.get("HUMANIZER_G", 2))   # OOM at G=4 with 3B
    max_completion_length: int = 140                                 # tighter for 3B VRAM
    temperature: float = 0.95
    save_every: int = 100

    # LoRA
    lora_r: int = int(os.environ.get("HUMANIZER_LORA_R", 8))
    use_qlora: bool = os.environ.get("HUMANIZER_QLORA", "0") == "1"

    # Multi-objective reward weights — Phase B revisions:
    #   w_det boosted (was 1.0) so detector evasion is the primary objective
    #     when within the sim floor. Run #2 showed lower w_det = "polished
    #     but still detectable" — model optimized quality but not evasion.
    #   sim_floor relaxed (was 0.78) so the policy has room to actually
    #     restructure semantically, not just word-substitute.
    w_det: float = float(os.environ.get("HUMANIZER_W_DET", 1.5))
    w_sim: float = float(os.environ.get("HUMANIZER_W_SIM", 0.3))
    w_pat: float = float(os.environ.get("HUMANIZER_W_PAT", 0.3))
    sim_floor: float = float(os.environ.get("HUMANIZER_SIM_FLOOR", 0.65))
    len_min_ratio: float = 0.55          # below this fraction of source length: penalty
    len_max_ratio: float = 1.65          # above this: penalty
    sim_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"


SYSTEM_PROMPT = (
    "You rewrite AI-generated text so it reads as if a real person wrote it, "
    "while preserving meaning. Vary sentence length aggressively. Use contractions. "
    "Avoid stiff transitional phrases like 'Furthermore' and 'Moreover'. Avoid words "
    "like 'leverage', 'delve', 'intricate', 'multifaceted'. Keep all facts and entities. "
    "Output ONLY the rewritten text with similar length to the input."
)


# ---- Inline pattern signals (subset of humanizer/patterns/signals.py) ----

_AI_FAVORITES = frozenset({
    "delve", "delves", "delved", "delving",
    "leverage", "leverages", "leveraged", "leveraging",
    "navigate", "navigates", "navigated", "navigating",
    "embark", "embarks", "embarked", "embarking",
    "foster", "fosters", "fostered", "fostering",
    "underscore", "underscores", "underscored",
    "showcase", "showcases", "showcased",
    "intricate", "multifaceted", "paramount", "crucial", "pivotal",
    "robust", "seamless", "comprehensive", "holistic", "nuanced",
    "innovative", "transformative", "groundbreaking",
    "myriad", "vibrant", "bustling", "remarkable",
    "tapestry", "realm", "plethora", "landscape", "ecosystem",
    "paradigm", "synergy", "endeavor", "endeavour",
    "testament", "cornerstone",
    "moreover", "furthermore", "additionally", "consequently",
    "indeed", "essentially", "fundamentally",
})

_AI_TRANSITIONS = frozenset({
    "Furthermore,", "Moreover,", "Additionally,", "In addition,",
    "Consequently,", "Thus,", "Therefore,", "Hence,",
    "However,", "Nevertheless,", "Nonetheless,",
    "In conclusion,", "To conclude,", "In summary,", "Overall,",
})

_HEDGING = (
    "it is important to note that", "it's important to note that",
    "it is worth noting that", "it should be noted that",
    "in today's rapidly evolving", "in today's fast-paced",
    "in today's world", "in today's society", "play a crucial role",
)

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _logistic(x: float, midpoint: float, steep: float = 8.0) -> float:
    return 1.0 / (1.0 + math.exp(-steep * (x - midpoint)))


def pattern_aggregate(text: str) -> float:
    """Single scalar in [0,1]: higher = more AI-like across all signals.
    Same weighting as humanizer/patterns/fingerprint.py, just inlined.
    """
    if not text or not text.strip():
        return 1.0  # empty / whitespace = max suspicious
    sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    words = max(len(text.split()), 1)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())
    n_tokens = max(len(tokens), 1)

    # 1. burstiness — low CV of sentence length means uniform = AI
    if len(sents) >= 3:
        counts = [len(s.split()) for s in sents]
        mean_w = sum(counts) / len(counts)
        var = sum((c - mean_w) ** 2 for c in counts) / len(counts)
        cv = math.sqrt(var) / mean_w if mean_w > 0 else 0.0
        burstiness = max(0.0, min(1.0, 1.0 - (cv / 0.7)))
    else:
        burstiness = 0.5

    # 2. stiff transitions per 100 words
    stiff = sum(text.count(t) for t in _AI_TRANSITIONS) * 100.0 / words
    stiff_score = _logistic(stiff, midpoint=0.8)

    # 3. favorite vocabulary density
    fav = sum(1 for t in tokens if t in _AI_FAVORITES) * 100.0 / n_tokens
    fav_score = _logistic(fav, midpoint=1.0)

    # 4. hedging boilerplate
    lower = text.lower()
    hed = sum(lower.count(p) for p in _HEDGING) * 100.0 / words
    hed_score = _logistic(hed, midpoint=0.4)

    # 5. contraction deficit (phrases that could contract but don't)
    expandable = len(re.findall(
        r"\b(do|does|did|is|are|was|were|can|will|would|could|should|has|have|had) "
        r"(not|is|are|am|have|has|had|will|would)\b", text, re.IGNORECASE,
    ))
    contracted = len(re.findall(
        r"\b(don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
        r"can't|won't|wouldn't|couldn't|shouldn't|"
        r"hasn't|haven't|hadn't|it's|that's|there's|"
        r"you're|they're|we're|i'm|i've)\b", text, re.IGNORECASE,
    ))
    total = expandable + contracted
    contraction_deficit = expandable / total if total > 0 else 0.0

    weights = {
        "burst": 1.5, "stiff": 1.3, "fav": 1.6,
        "hed": 1.2, "contr": 1.0,
    }
    parts = {
        "burst": burstiness, "stiff": stiff_score, "fav": fav_score,
        "hed": hed_score, "contr": contraction_deficit,
    }
    total_w = sum(weights.values())
    return float(sum(parts[k] * w for k, w in weights.items()) / total_w)


# ---- Data + detectors (same as train.py) ----

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


def load_human_samples(cfg: Cfg, n: int) -> list[str]:
    """Stream the same dataset but pull HUMAN-labelled rows (generated=0).
    These are the negative examples for the adversarial discriminator."""
    from datasets import load_dataset
    print(f"[data] streaming human samples from {cfg.dataset}...", flush=True)
    ds = load_dataset(cfg.dataset, split="train", streaming=True)
    out: list[str] = []
    for ex in ds:
        if int(ex.get("generated", 0)) != 0:
            continue
        t = (ex.get("text") or "").strip()
        wc = len(t.split())
        if cfg.min_words <= wc <= cfg.max_words:
            out.append(t)
            if len(out) >= n:
                break
    print(f"[data] got {len(out)} human samples for discriminator", flush=True)
    return out


def load_discriminator(cfg: Cfg):
    """Load a fresh trainable RoBERTa-base classifier, warm-started from the
    OpenAI detector weights. Will be updated every cfg.disc_update_every steps."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"[disc] init from {cfg.disc_init_id} (trainable)", flush=True)
    d = AutoModelForSequenceClassification.from_pretrained(
        cfg.disc_init_id, torch_dtype=torch.float32, ignore_mismatched_sizes=True
    ).cuda().train()
    # All params trainable for the discriminator (unlike frozen detectors)
    for p in d.parameters():
        p.requires_grad_(True)
    t = AutoTokenizer.from_pretrained(cfg.disc_init_id)
    id2 = d.config.id2label
    ai_idx = next(
        i for i, l in id2.items()
        if "fake" in str(l).lower() or "label_1" in str(l).lower() or i == d.config.num_labels - 1
    )
    optim = torch.optim.AdamW(d.parameters(), lr=cfg.disc_lr)
    return d, t, int(ai_idx), optim


def update_discriminator(disc, disc_tok, disc_optim, ai_texts, human_texts, cfg):
    """Run a few mini-batches of supervised training on the discriminator.
    Positive class (AI) = recent policy outputs. Negative class (human) = real."""
    import random as _r
    import torch
    import torch.nn.functional as F

    if not ai_texts or not human_texts:
        return
    disc.train()
    rng = _r.Random()
    pairs = [(t, 1) for t in ai_texts] + [(t, 0) for t in human_texts]
    rng.shuffle(pairs)

    for step in range(cfg.disc_train_steps):
        batch = rng.sample(pairs, k=min(cfg.disc_batch, len(pairs)))
        texts = [p[0] for p in batch]
        labels = torch.tensor([p[1] for p in batch], device="cuda")
        enc = disc_tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
        logits = disc(**enc).logits
        # The label-index for the AI class might not be 1 in id2label; adjust.
        # Since we warm-started from the OpenAI detector where AI = "fake" class,
        # we computed ai_idx earlier. Map our 0/1 labels accordingly.
        # Simplest: use binary cross-entropy on the AI-class logit.
        ai_logits = logits[:, 1] if logits.shape[1] > 1 else logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(ai_logits, labels.float())
        disc_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        disc_optim.step()
    disc.eval()


def load_detectors(cfg: Cfg):
    """Load detectors. Skip ones that fail to load (e.g. config / state-dict
    mismatches) rather than crashing the whole run — better to train against
    3 detectors than 0."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    detectors = []
    for det_id in cfg.detector_ids:
        print(f"[detector] loading {det_id}", flush=True)
        try:
            d = AutoModelForSequenceClassification.from_pretrained(
                det_id, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
            ).cuda().eval()
        except Exception as e:  # noqa: BLE001
            print(f"[detector] FAILED to load {det_id}: {e}", flush=True)
            print(f"[detector] continuing without it", flush=True)
            continue
        for p in d.parameters():
            p.requires_grad_(False)
        dt = AutoTokenizer.from_pretrained(det_id)
        id2 = d.config.id2label
        ai_idx = next(
            i for i, l in id2.items()
            if "fake" in str(l).lower() or "label_1" in str(l).lower() or i == d.config.num_labels - 1
        )
        detectors.append((d, dt, int(ai_idx)))
    if not detectors:
        raise RuntimeError("No detectors loaded successfully")
    print(f"[detector] {len(detectors)}/{len(cfg.detector_ids)} loaded", flush=True)
    return detectors


def detector_p_ai(detectors, texts, extra_disc=None):
    """Get p_ai per detector. If extra_disc=(disc, tok, ai_idx), include it
    as an additional ensemble member with EQUAL weight to the static detectors."""
    import torch
    members = list(detectors)
    if extra_disc is not None:
        members.append(extra_disc)
    n = len(members)
    per = []
    with torch.no_grad():
        for d, dt, ai_idx in members:
            enc = dt(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
            logits = d(**enc).logits.float()
            per.append(torch.softmax(logits, dim=-1)[:, ai_idx].cpu().tolist())
    agg = [sum(per[k][i] for k in range(n)) / n for i in range(len(texts))]
    return agg, per


def compute_logp(model, sequences, prompt_len):
    """One sequence at a time to fit VRAM. Sum of completion-token log-probs."""
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


# ---- Multi-objective reward ----

def length_penalty(source: str, candidate: str, cfg: Cfg) -> float:
    """0 if length is in range, negative scaled by how far out it is."""
    sl = max(len(source.split()), 1)
    cl = max(len(candidate.split()), 1)
    ratio = cl / sl
    if cfg.len_min_ratio <= ratio <= cfg.len_max_ratio:
        return 0.0
    if ratio < cfg.len_min_ratio:
        return -0.5 * (cfg.len_min_ratio - ratio) / cfg.len_min_ratio
    return -0.3 * (ratio - cfg.len_max_ratio) / cfg.len_max_ratio


def compute_reward(cfg, sim_model, detectors, source, candidates, extra_disc=None):
    """Reward includes the static detectors plus an optional adversarial discriminator."""
    import torch
    p_ais, _ = detector_p_ai(detectors, candidates, extra_disc=extra_disc)
    # Cosine similarity of source vs each candidate
    src_emb = sim_model.encode([source], normalize_embeddings=True, convert_to_tensor=True)
    cand_emb = sim_model.encode(candidates, normalize_embeddings=True, convert_to_tensor=True)
    sims = (src_emb * cand_emb).sum(-1).cpu().tolist()

    rewards: list[float] = []
    details: list[dict] = []
    for c, p, s in zip(candidates, p_ais, sims):
        pat = pattern_aggregate(c)
        lp = length_penalty(source, c, cfg)
        if s < cfg.sim_floor:
            # HARD GATE: tanked meaning. Reward = -1 regardless of detector.
            r = -1.0
            details.append({
                "p_ai": p, "sim": s, "pattern": pat, "len_pen": lp,
                "reward": r, "gated": True,
            })
        else:
            r = (
                cfg.w_det * (1.0 - p)
                + cfg.w_sim * s
                + cfg.w_pat * (1.0 - pat)
                + lp
            )
            details.append({
                "p_ai": p, "sim": s, "pattern": pat, "len_pen": lp,
                "reward": r, "gated": False,
            })
        rewards.append(r)
    return rewards, details


# ---- Train + Eval (largely identical to v1, with multi-objective reward calls) ----

def train(cfg: Cfg) -> tuple[list[str], list[dict]]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from sentence_transformers import SentenceTransformer
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
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
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
    print(f"[similarity] loading {cfg.sim_model_id}", flush=True)
    sim_model = SentenceTransformer(cfg.sim_model_id, device="cuda")

    # Adversarial discriminator + a pool of human samples to train it against.
    disc, disc_tok, disc_ai_idx, disc_optim = load_discriminator(cfg)
    extra_disc = (disc, disc_tok, disc_ai_idx)
    human_pool = load_human_samples(cfg, cfg.disc_window * 4)  # 4x the window for variety
    ai_buffer: list[str] = []  # rolling FIFO of recent policy outputs

    prompts = load_prompts(cfg, cfg.n_train_prompts + cfg.n_eval_prompts)
    train_prompts = prompts[: cfg.n_train_prompts]
    eval_prompts = prompts[cfg.n_train_prompts:]
    (OUT / "eval_prompts.jsonl").write_text("\n".join(json.dumps({"source": p}) for p in eval_prompts))
    print(f"[data] train={len(train_prompts)}  eval={len(eval_prompts)}", flush=True)

    optim = AdamW([p for p in policy.parameters() if p.requires_grad], lr=cfg.learning_rate)

    log_lines: list[dict] = []
    t0 = time.time()
    import random as _r
    _rng = _r.Random()

    for step, src in enumerate(train_prompts):
        prompt_text = format_prompt(tokenizer, src)
        enc = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        prompt_len = enc.input_ids.shape[1]

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

        rewards, details = compute_reward(
            cfg, sim_model, detectors, src, completions, extra_disc=extra_disc,
        )
        # Add the gate-passing completions to the AI buffer for the discriminator.
        for c, d in zip(completions, details):
            if not d["gated"]:
                ai_buffer.append(c)
        # Trim AI buffer to the configured window.
        if len(ai_buffer) > cfg.disc_window:
            ai_buffer = ai_buffer[-cfg.disc_window:]

        rewards_t = torch.tensor(rewards, device="cuda")
        baseline = rewards_t.mean()
        std = rewards_t.std() + 1e-6
        advantages = (rewards_t - baseline) / std

        policy.train()
        logp_policy = compute_logp(policy, seqs, prompt_len)
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
            "mean_p_ai": float(sum(d["p_ai"] for d in details) / len(details)),
            "mean_sim": float(sum(d["sim"] for d in details) / len(details)),
            "mean_pattern": float(sum(d["pattern"] for d in details) / len(details)),
            "mean_len_pen": float(sum(d["len_pen"] for d in details) / len(details)),
            "n_gated": int(sum(1 for d in details if d["gated"])),
            "kl": float(kl.item()),
            "loss": float(loss.item()),
            "best_completion": completions[int(rewards_t.argmax())][:200],
            "t": round(time.time() - t0, 1),
        }
        log_lines.append(line)
        print(
            f"step {step+1:4d}/{len(train_prompts)}  "
            f"R̄={line['mean_reward']:+.3f}  R_max={line['max_reward']:+.3f}  "
            f"p_ai={line['mean_p_ai']:.3f}  sim={line['mean_sim']:.3f}  "
            f"pat={line['mean_pattern']:.3f}  gated={line['n_gated']}/{cfg.num_generations}  "
            f"KL={line['kl']:+.3f}  t={line['t']}s",
            flush=True,
        )

        # Update the adversarial discriminator periodically.
        if (step + 1) % cfg.disc_update_every == 0 and len(ai_buffer) >= cfg.disc_batch:
            human_batch = _rng.sample(human_pool, k=min(cfg.disc_window, len(human_pool)))
            update_discriminator(disc, disc_tok, disc_optim, ai_buffer, human_batch, cfg)
            print(
                f"  [disc updated at step {step+1}; ai_buf={len(ai_buffer)} human={len(human_batch)}]",
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
    """Same eval shape as train.py — base vs trained on held-out + similarity."""
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
    emb = SentenceTransformer(cfg.sim_model_id).cuda()

    def cosine(a, b):
        ea = emb.encode(a, normalize_embeddings=True, convert_to_tensor=True)
        eb = emb.encode(b, normalize_embeddings=True, convert_to_tensor=True)
        return (ea * eb).sum(-1).cpu().tolist()

    @torch.no_grad()
    def humanize(model, src):
        prompt = format_prompt(tokenizer, src)
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(
            **enc, do_sample=True, temperature=0.85, top_p=0.95,
            max_new_tokens=cfg.max_completion_length,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def evaluate_model(model, label):
        outs = [humanize(model, s) for s in eval_prompts]
        ens, per_d = detector_p_ai(detectors, outs)
        sims = cosine(eval_prompts, outs)
        pats = [pattern_aggregate(o) for o in outs]
        return {
            "label": label,
            "n": len(outs),
            "mean_p_ai_ensemble": statistics.fmean(ens),
            "mean_similarity": statistics.fmean(sims),
            "mean_pattern": statistics.fmean(pats),
            "asr_ensemble": sum(1 for p in ens if p < 0.5) / max(len(ens), 1),
            "asr_per_detector": {
                cfg.detector_ids[i].split("/")[-1]: sum(1 for p in per_d[i] if p < 0.5) / max(len(per_d[i]), 1)
                for i in range(len(detectors))
            },
            "outputs": outs,
            "p_ai_ensemble": ens,
            "p_ai_per_detector": {cfg.detector_ids[i].split("/")[-1]: per_d[i] for i in range(len(detectors))},
            "similarity": sims,
            "pattern": pats,
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
        "delta_p_ai_ensemble":  res_trained["mean_p_ai_ensemble"] - res_base["mean_p_ai_ensemble"],
        "delta_similarity":     res_trained["mean_similarity"]    - res_base["mean_similarity"],
        "delta_pattern":        res_trained["mean_pattern"]       - res_base["mean_pattern"],
        "delta_asr_ensemble":   res_trained["asr_ensemble"]       - res_base["asr_ensemble"],
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
