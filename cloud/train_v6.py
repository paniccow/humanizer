"""GRPO inner loop v6 — Pangram (real-world API) directly in the reward loop.

The whole point of v6: the user paid for Pangram API credits ($50 = 1000
credits), and v5's adversarial-discriminator was the proxy for "we don't
have a real-world detector." Now we do, so we wire Pangram itself into
the per-step reward computation. The policy learns to fool the actual
target, not a proxy.

Smoke test that motivated this: gpt-4o-mini AND gpt-4o + scrub both
hit fraction_ai=1.000 on every candidate against Pangram. We can't fool
Pangram from the outside; we have to train against it.

Recipe (changes from v5):
  - REMOVE: trainable adversarial discriminator (load_discriminator,
    update_discriminator, ai_buffer, human_pool — none of it).
  - ADD: pangram_score_batch() that POSTs to https://text.api.pangramlabs.com/v3
    with the PANGRAM_API_KEY from the pod's environment. Used as a third
    detector in the ensemble for every step.
  - Reward becomes:
      R(y) = w_det · (1 - mean(p_ai across [roberta-base, roberta-large, pangram]))
           + w_sim · cosine_sim(original, y)
           + w_pat · (1 - pattern_aggregate(y))
           + length_penalty
    With sim_floor 0.65 hard gate.

Cost math (Pangram credits at $0.05/1K words):
  G=2 generations × 1200 steps = 2400 Pangram calls during training
  ~36 words/call × 2400 = 86K words = ~$4.30 in Pangram credits
  Plus held-out eval: ~30 prompts × 4 variants × 36 words = ~$0.20
  Total Pangram cost: ~$4.50 (well under the 1000-credit budget)
  Plus pod time: ~$2-5 secure-cloud 4090 for ~3 hours
  Grand total: ~$10 for v6

Inherits from v4/v5:
  Qwen2.5-3B-Instruct base, LoRA-r8, completion=140, G=2,
  sim_floor=0.65, w_det=1.5, w_sim=0.3, w_pat=0.3.

VRAM management for 3B + 2 frozen detectors (no disc to load now):
  Qwen-3B bf16:                ~6.0 GB
  LoRA-r8 grads + optim:       ~1.0 GB
  Reference (via .disable_adapter()):  0
  2 frozen detectors bf16:     ~1.5 GB
  (no discriminator — Pangram is API-only)
  Sim model (MiniLM):          ~0.4 GB
  Generation buffers G=2:      ~2.5 GB
  Activations:                 ~5 GB
  Headroom:                    ~5 GB
  Total:                       ~16 GB. Comfortable on 22GB.

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
    )  # Local frozen detectors. Pangram joins them via API per step.

    # Pangram API settings (the real-world judge):
    pangram_api_key_env: str = "PANGRAM_API_KEY"
    pangram_endpoint: str = "https://text.api.pangramlabs.com/v3"
    pangram_weight: float = 2.0          # When Pangram fires it dominates the ensemble (heavier than locals)
    pangram_ai_assisted_weight: float = 0.5   # how much of fraction_ai_assisted to count as AI
    pangram_timeout_s: float = 30.0
    pangram_retries: int = 2
    # Sparse Pangram calling: only hit the API every Nth step. Other steps
    # train against local detectors only (free). This is the cost lever.
    # 1   = every step  (~$60 for 400 steps × 2 generations)
    # 25  = every 25th  (~$1.60 for 400 steps; bonus ground-truth signal)
    # 50  = every 50th  (~$0.80)
    # 0   = never        (free, equivalent to v5 without disc)
    # Override via HUMANIZER_PANGRAM_EVERY env var.
    pangram_every: int = int(os.environ.get("HUMANIZER_PANGRAM_EVERY", 25))
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


def pangram_score_batch(texts: list[str], cfg: Cfg, api_key: str) -> list[float]:
    """Hit Pangram's v3 endpoint for each text and return p_ai per text.

    Uses ThreadPoolExecutor for concurrent calls — for G=2 generations a
    single step issues 2 parallel requests, ~250-500ms each, so step
    latency is bounded by the slower one. urllib only (the pod's pip
    deps don't include `requests`).

    p_ai = fraction_ai + 0.5 * fraction_ai_assisted (the conservative
    aggregation we use in the inference-side detector).
    """
    import json as _json
    import time as _time
    import urllib.error
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def _one(text: str) -> float:
        body = _json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            cfg.pangram_endpoint, data=body,
            headers={
                "Content-Type": "application/json", "Accept": "application/json",
                "x-api-key": api_key,
            }, method="POST",
        )
        last_err: Exception | None = None
        for attempt in range(cfg.pangram_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=cfg.pangram_timeout_s) as r:
                    payload = _json.loads(r.read().decode("utf-8"))
                ai = float(payload.get("fraction_ai", 0.0))
                assisted = float(payload.get("fraction_ai_assisted", 0.0))
                return min(1.0, max(0.0, ai + cfg.pangram_ai_assisted_weight * assisted))
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                if attempt < cfg.pangram_retries:
                    _time.sleep(1.5 * (attempt + 1))
        # Don't crash training on a transient API error — degrade gracefully
        # with a neutral 0.5 score and let the local detectors carry the load.
        print(f"[pangram] failed after retries: {last_err}; falling back to 0.5", flush=True)
        return 0.5

    if not texts:
        return []
    workers = min(len(texts), 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, texts))


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


def detector_p_ai(detectors, texts, cfg: Cfg | None = None, pangram_api_key: str | None = None):
    """Get p_ai per text. Local detectors get equal weight (1.0 each); Pangram,
    when present, gets cfg.pangram_weight (default 2.0 — when Pangram fires it
    dominates because it's the actual target).

    `detectors` is the list of (model, tok, ai_idx) tuples for frozen local
    detectors. Pass pangram_api_key=None for the cheap local-only path used
    on most training steps.
    """
    import torch
    members = list(detectors)
    n_local = len(members)
    per: list[list[float]] = []
    with torch.no_grad():
        for d, dt, ai_idx in members:
            enc = dt(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
            logits = d(**enc).logits.float()
            per.append(torch.softmax(logits, dim=-1)[:, ai_idx].cpu().tolist())
    weights = [1.0] * n_local
    if pangram_api_key and cfg is not None:
        pgr = pangram_score_batch(texts, cfg, pangram_api_key)
        per.append(pgr)
        weights.append(cfg.pangram_weight)
    total_w = sum(weights) or 1.0
    agg = [
        sum(per[k][i] * weights[k] for k in range(len(per))) / total_w
        for i in range(len(texts))
    ]
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


def compute_reward(cfg, sim_model, detectors, source, candidates, pangram_api_key=None):
    """Reward includes the local frozen detectors plus (optionally) Pangram via API."""
    import torch
    p_ais, _ = detector_p_ai(detectors, candidates, cfg=cfg, pangram_api_key=pangram_api_key)
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

    # Pangram API: read the key from the pod env. Fail loudly if missing —
    # this run's whole point is training against Pangram.
    pangram_api_key = os.environ.get(cfg.pangram_api_key_env)
    if not pangram_api_key:
        raise RuntimeError(
            f"{cfg.pangram_api_key_env} not set on the pod. Run #6 trains against "
            f"Pangram in the reward loop — it's required."
        )
    print(f"[pangram] api key present, will be queried per step", flush=True)

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

        # Sparse Pangram: only call the API every Nth step. On other steps
        # the reward comes from local detectors only — free, fast, and the
        # primary gradient signal. The periodic Pangram steps act as
        # ground-truth corrections that pull the policy toward the actual
        # real-world target.
        use_pangram_this_step = (
            cfg.pangram_every > 0 and (step % cfg.pangram_every == 0)
        )
        rewards, details = compute_reward(
            cfg, sim_model, detectors, src, completions,
            pangram_api_key=pangram_api_key if use_pangram_this_step else None,
        )

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
            "pangram": use_pangram_this_step,
            "t": round(time.time() - t0, 1),
        }
        log_lines.append(line)
        pgr_marker = " [+pgr]" if use_pangram_this_step else ""
        print(
            f"step {step+1:4d}/{len(train_prompts)}{pgr_marker}  "
            f"R̄={line['mean_reward']:+.3f}  R_max={line['max_reward']:+.3f}  "
            f"p_ai={line['mean_p_ai']:.3f}  sim={line['mean_sim']:.3f}  "
            f"pat={line['mean_pattern']:.3f}  gated={line['n_gated']}/{cfg.num_generations}  "
            f"KL={line['kl']:+.3f}  t={line['t']}s",
            flush=True,
        )

        if (step + 1) % cfg.save_every == 0:
            policy.save_pretrained(str(ADAPTER_DIR))
            tokenizer.save_pretrained(str(ADAPTER_DIR))
            LOG_PATH.write_text(json.dumps(log_lines, indent=2))
            # Touch the `done` sentinel after the first checkpoint so the
            # rescue stack can pull a partial adapter if training dies later.
            # Subsequent active_rescue pulls are no-ops (it exits on first
            # pull), but cost_cap's 7h watchdog will scp the LATEST in-pod
            # adapter at cap time, which by then is the final state.
            DONE_PATH.touch()
            print(f"  [checkpoint at step {step+1}; done sentinel touched]", flush=True)

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
