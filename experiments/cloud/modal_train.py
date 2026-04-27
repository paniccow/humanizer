"""Real GRPO training on a rented GPU via Modal.

Faithful AuthorMist (arXiv:2503.08716) recipe at a budget-conscious scale:

  Base model    : Qwen2.5-1.5B-Instruct (fits on A100-40GB with LoRA + ref + detector)
  Algorithm     : GRPO via TRL — group-relative advantage, no critic
  Reward        : 1 - mean(p_ai) across two RoBERTa detectors
  KL guardrail  : β = 0.001 against frozen reference policy
  Group size    : 8 generations per prompt
  Steps         : 600 (vs AuthorMist's 714 — close to paper budget)
  LoRA          : rank 16 on q/k/v/o, ~10M trainable / 1.5B (~0.7%)

Cost estimate (Modal pricing, 2026):
  A100-40GB at $2.50/hr × 2-3hr ≈ $5-8.
  $30 sign-up credit covers the first run.

Three entry points:
  modal run modal_train.py::train     — run the training (writes adapter to Volume)
  modal run modal_train.py::evaluate  — base vs trained on held-out prompts
  modal run modal_train.py::download  — copy adapter from Volume to local ./adapter
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

# ---------------------------- Modal infrastructure ----------------------------

APP_NAME = "humanizer-grpo"
VOLUME_NAME = "humanizer-artifacts"
ADAPTER_DIR = "/data/grpo_adapter"
HF_CACHE_DIR = "/cache/huggingface"
EVAL_DIR = "/data/eval"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("humanizer-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.0",
        "trl==0.12.0",
        "peft==0.13.2",
        "accelerate==1.1.0",
        "datasets==3.1.0",
        "sentence-transformers==3.3.0",
        "bitsandbytes==0.44.1",
        "numpy<2",
    )
    .env({"HF_HOME": HF_CACHE_DIR, "TRANSFORMERS_CACHE": HF_CACHE_DIR})
)


# ---------------------------- Hyperparameters ---------------------------------

class Cfg:
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    detector_ids = (
        "openai-community/roberta-base-openai-detector",
        "openai-community/roberta-large-openai-detector",
    )
    dataset = "andythetechnerd03/AI-human-text"
    n_train_prompts = 600
    n_eval_prompts = 30
    min_words = 60
    max_words = 200

    # GRPO
    learning_rate = 5e-6
    beta_kl = 0.001
    num_generations = 8
    max_prompt_length = 768
    max_completion_length = 384
    temperature = 0.9
    top_p = 0.95
    per_device_batch = 1                # one prompt per device, G generations from it
    grad_accum = 8

    # LoRA
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    # Schedule
    epochs = 1


SYSTEM_PROMPT = (
    "You rewrite AI-generated text so it reads as if a real person wrote it, "
    "while preserving meaning. Vary sentence length aggressively. Use contractions. "
    "Avoid stiff transitional phrases like 'Furthermore' and 'Moreover'. Avoid words "
    "like 'leverage', 'delve', 'intricate', 'multifaceted'. Output ONLY the rewritten text."
)


def _format_prompt(tokenizer, source: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Rewrite the following text:\n\n---\n{source}\n---"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _load_prompts(n: int):
    """Stream the AI-vs-human dataset; yield AI-labelled rows until we have n."""
    from datasets import load_dataset
    ds = load_dataset(Cfg.dataset, split="train", streaming=True)
    out = []
    for ex in ds:
        if int(ex.get("generated", 0)) != 1:
            continue
        t = (ex.get("text") or "").strip()
        wc = len(t.split())
        if Cfg.min_words <= wc <= Cfg.max_words:
            out.append(t)
            if len(out) >= n:
                break
    return out


# ---------------------------- TRAIN -------------------------------------------

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 4,                # 4-hour cap
    volumes={"/data": volume, "/cache": hf_cache},
)
def train():
    """Run GRPO. Writes the trained LoRA adapter to the Modal Volume."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    from trl import GRPOConfig, GRPOTrainer

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading base model {Cfg.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(Cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        Cfg.base_model, torch_dtype=torch.bfloat16
    )

    print(f"Loading detectors: {Cfg.detector_ids}")
    detectors = []
    det_tokenizers = []
    for det_id in Cfg.detector_ids:
        d = AutoModelForSequenceClassification.from_pretrained(
            det_id, torch_dtype=torch.bfloat16
        ).cuda().eval()
        for p in d.parameters():
            p.requires_grad_(False)
        dt = AutoTokenizer.from_pretrained(det_id)
        # Resolve which class index = "AI" — varies by detector.
        id2 = d.config.id2label
        ai_idx = next(
            i for i, l in id2.items()
            if "fake" in str(l).lower() or "label_1" in str(l).lower() or i == d.config.num_labels - 1
        )
        detectors.append((d, dt, int(ai_idx)))

    @torch.no_grad()
    def detector_p_ai(model, tok, ai_idx, texts):
        enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
        logits = model(**enc).logits.float()
        return torch.softmax(logits, dim=-1)[:, ai_idx].cpu().tolist()

    def reward_fn(prompts, completions, **_):
        """TRL passes lists of completions (chat-formatted). We want plain text."""
        texts = []
        for c in completions:
            if isinstance(c, list) and c and isinstance(c[0], dict):
                texts.append(c[0].get("content", ""))
            else:
                texts.append(str(c))
        ensemble = []
        for det, dtok, ai_idx in detectors:
            ensemble.append(detector_p_ai(det, dtok, ai_idx, texts))
        # Mean p_ai across detectors → reward = 1 - mean.
        n = len(detectors)
        return [1.0 - sum(ensemble[k][i] for k in range(n)) / n for i in range(len(texts))]

    print(f"Loading {Cfg.n_train_prompts} prompts from {Cfg.dataset}")
    prompts = _load_prompts(Cfg.n_train_prompts + Cfg.n_eval_prompts)
    train_prompts = prompts[: Cfg.n_train_prompts]
    eval_prompts = prompts[Cfg.n_train_prompts :]
    print(f"  train={len(train_prompts)}  eval={len(eval_prompts)}")

    # Persist the eval split so the eval entry point uses the SAME held-out set.
    Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
    import json as _json
    Path(f"{EVAL_DIR}/prompts.jsonl").write_text(
        "\n".join(_json.dumps({"source": p}) for p in eval_prompts)
    )
    volume.commit()

    train_ds = Dataset.from_list(
        [{"prompt": _format_prompt(tokenizer, p)} for p in train_prompts]
    )

    peft_config = LoraConfig(
        r=Cfg.lora_r,
        lora_alpha=Cfg.lora_alpha,
        lora_dropout=Cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = GRPOConfig(
        output_dir=ADAPTER_DIR,
        learning_rate=Cfg.learning_rate,
        num_train_epochs=Cfg.epochs,
        per_device_train_batch_size=Cfg.per_device_batch,
        gradient_accumulation_steps=Cfg.grad_accum,
        num_generations=Cfg.num_generations,
        max_prompt_length=Cfg.max_prompt_length,
        max_completion_length=Cfg.max_completion_length,
        temperature=Cfg.temperature,
        beta=Cfg.beta_kl,
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

    print("Starting GRPO training...")
    trainer.train()

    print(f"Saving adapter to {ADAPTER_DIR}")
    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Persist log history for the eval/findings step
    import json
    Path(f"{ADAPTER_DIR}/training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2)
    )
    volume.commit()
    print(f"Done. Adapter at {ADAPTER_DIR}")


# ---------------------------- EVAL --------------------------------------------

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60,
    volumes={"/data": volume, "/cache": hf_cache},
)
def evaluate():
    """Compare BASE vs TRAINED on the held-out eval set. Saves JSON to the Volume."""
    import json
    import statistics
    import torch
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    eval_path = Path(f"{EVAL_DIR}/prompts.jsonl")
    if not eval_path.exists():
        raise FileNotFoundError("Eval prompts missing — run train() first.")

    sources = [json.loads(l)["source"] for l in eval_path.read_text().splitlines()]
    print(f"eval prompts: {len(sources)}")

    tokenizer = AutoTokenizer.from_pretrained(Cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    detectors = []
    for det_id in Cfg.detector_ids:
        d = AutoModelForSequenceClassification.from_pretrained(det_id).cuda().eval()
        dt = AutoTokenizer.from_pretrained(det_id)
        id2 = d.config.id2label
        ai_idx = next(
            i for i, l in id2.items()
            if "fake" in str(l).lower() or "label_1" in str(l).lower() or i == d.config.num_labels - 1
        )
        detectors.append((d, dt, int(ai_idx)))

    @torch.no_grad()
    def score_ensemble(texts):
        per = []
        for d, dt, ai in detectors:
            enc = dt(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
            logits = d(**enc).logits.float()
            per.append(torch.softmax(logits, dim=-1)[:, ai].cpu().tolist())
        n = len(detectors)
        agg = [sum(per[k][i] for k in range(n)) / n for i in range(len(texts))]
        per_named = {det_id.split("/")[-1]: per[i] for i, det_id in enumerate(Cfg.detector_ids)}
        return agg, per_named

    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cuda()

    def cosine(a, b):
        ea = emb.encode(a, normalize_embeddings=True, convert_to_tensor=True)
        eb = emb.encode(b, normalize_embeddings=True, convert_to_tensor=True)
        return (ea * eb).sum(-1).cpu().tolist()

    @torch.no_grad()
    def humanize(model, src):
        prompt = _format_prompt(tokenizer, src)
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=0.85,
            top_p=0.95,
            max_new_tokens=Cfg.max_completion_length,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def evaluate_model(model, label):
        outs = [humanize(model, s) for s in sources]
        ens, per_d = score_ensemble(outs)
        sims = cosine(sources, outs)
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

    print(f"Evaluating BASE...")
    base = AutoModelForCausalLM.from_pretrained(Cfg.base_model, torch_dtype=torch.bfloat16).cuda().eval()
    res_base = evaluate_model(base, "BASE")
    del base
    torch.cuda.empty_cache()

    print(f"Evaluating TRAINED...")
    trained = AutoModelForCausalLM.from_pretrained(Cfg.base_model, torch_dtype=torch.bfloat16).cuda()
    trained = PeftModel.from_pretrained(trained, ADAPTER_DIR).cuda().eval()
    res_trained = evaluate_model(trained, "TRAINED")

    summary = {
        "n": len(sources),
        "base":    {k: v for k, v in res_base.items()    if not isinstance(v, (list, dict)) or k.startswith("asr")},
        "trained": {k: v for k, v in res_trained.items() if not isinstance(v, (list, dict)) or k.startswith("asr")},
        "delta_p_ai_ensemble": res_trained["mean_p_ai_ensemble"] - res_base["mean_p_ai_ensemble"],
        "delta_similarity": res_trained["mean_similarity"] - res_base["mean_similarity"],
        "delta_asr_ensemble": res_trained["asr_ensemble"] - res_base["asr_ensemble"],
    }
    print(json.dumps(summary, indent=2))

    Path(f"{ADAPTER_DIR}/eval.json").write_text(
        json.dumps({"summary": summary, "base": res_base, "trained": res_trained}, indent=2)
    )
    volume.commit()
    return summary


# ---------------------------- DOWNLOAD ----------------------------------------

@app.function(image=image, volumes={"/data": volume})
def _list_adapter():
    """Internal — list adapter files for the local downloader."""
    base = Path(ADAPTER_DIR)
    if not base.exists():
        return []
    return [str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()]


@app.function(image=image, volumes={"/data": volume})
def _read_adapter_file(rel_path: str) -> bytes:
    return (Path(ADAPTER_DIR) / rel_path).read_bytes()


@app.local_entrypoint()
def download(out: str = "experiments/cloud/adapter"):
    """Pull the trained adapter from the Modal Volume to a local directory."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _list_adapter.remote()
    if not files:
        print(f"No adapter found in volume {VOLUME_NAME!r}. Run train() first.")
        return
    print(f"Downloading {len(files)} files to {out_dir}/")
    for rel in files:
        data = _read_adapter_file.remote(rel)
        local = out_dir / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        print(f"  {rel}  ({len(data)/1024:.1f} KB)")
    print("Done.")


@app.local_entrypoint()
def train_main():
    train.remote()


@app.local_entrypoint()
def evaluate_main():
    print(evaluate.remote())
