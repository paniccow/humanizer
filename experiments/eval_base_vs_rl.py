"""Real before/after eval.

Compares the BASE SmolLM2-135M-Instruct with the RL-tuned policy on the
held-out eval prompts. Reports:
  - mean detector p_ai      (lower = better humanization)
  - mean cosine similarity  (preserved meaning)
  - mean pattern aggregate  (AI-fingerprint score from humanizer.patterns)
  - per-example outputs

Real generations, real numbers. Saves JSON results.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, "/tmp/humanizer")
from humanizer.patterns import analyze as pattern_analyze  # noqa: E402

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER = "checkpoints/rl"
DETECTOR = "openai-community/roberta-base-openai-detector"
SIMILARITY = "sentence-transformers/all-MiniLM-L6-v2"
EVAL_FILE = "data/eval.jsonl"
OUT = Path("logs/eval.json")
N = 8
MAX_NEW = 180

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


def build_prompt(tokenizer, source: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Rewrite the following text:\n\n---\n{source}\n---"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tokenizer, source, dev):
    prompt = build_prompt(tokenizer, source)
    enc = tokenizer(prompt, return_tensors="pt").to(dev)
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=0.85,
        top_p=0.95,
        max_new_tokens=MAX_NEW,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion = out[:, enc.input_ids.shape[1]:]
    return tokenizer.decode(completion[0], skip_special_tokens=True).strip()


@torch.no_grad()
def detector_p_ai(detector, det_tok, texts, dev):
    enc = det_tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(dev)
    logits = detector(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    id2 = detector.config.id2label
    ai_idx = next(i for i, l in id2.items() if "fake" in str(l).lower() or "label_1" in str(l).lower())
    return probs[:, int(ai_idx)].cpu().tolist()


def cosine_sim(emb_model, a, b):
    ea = emb_model.encode(a, normalize_embeddings=True, convert_to_tensor=True)
    eb = emb_model.encode(b, normalize_embeddings=True, convert_to_tensor=True)
    return (ea * eb).sum(-1).cpu().tolist()


def evaluate(model, tokenizer, detector, det_tok, emb, sources, dev, label):
    print(f"\n=== Evaluating {label} on {len(sources)} examples ===", flush=True)
    outs = []
    for i, src in enumerate(sources):
        outs.append(generate(model, tokenizer, src, dev))
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  [{i+1}/{len(sources)}] {len(outs[-1].split())} words", flush=True)
    p_ais = detector_p_ai(detector, det_tok, outs, dev)
    sims = cosine_sim(emb, sources, outs)
    pats = [pattern_analyze(t).aggregate for t in outs]
    return {
        "label": label,
        "mean_p_ai": statistics.fmean(p_ais),
        "mean_similarity": statistics.fmean(sims),
        "mean_pattern": statistics.fmean(pats),
        "p_ai": p_ais,
        "similarity": sims,
        "pattern": pats,
        "outputs": outs,
    }


def main():
    dev = device()
    print(f"device: {dev}")
    sources = [json.loads(l)["source"] for l in Path(EVAL_FILE).read_text().splitlines()][:N]
    print(f"eval set: {len(sources)} prompts (held-out, not seen during training)")

    print(f"loading detector {DETECTOR}")
    det_tok = AutoTokenizer.from_pretrained(DETECTOR)
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR).to(dev).eval()

    # baseline: AI sources themselves
    src_p_ais = detector_p_ai(detector, det_tok, sources, dev)
    src_pats = [pattern_analyze(s).aggregate for s in sources]
    print(
        f"raw source AI text — mean p_ai={statistics.fmean(src_p_ais):.3f} "
        f"pattern={statistics.fmean(src_pats):.3f}  (closer to 1 = more AI)"
    )

    print(f"loading similarity {SIMILARITY}")
    emb = SentenceTransformer(SIMILARITY, device="cpu")  # MiniLM on CPU keeps MPS for LLMs

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nloading BASE {BASE}")
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to(dev).eval()
    res_base = evaluate(base, tokenizer, detector, det_tok, emb, sources, dev, "BASE")
    del base
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"\nloading RL adapter {ADAPTER}")
    rl = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to(dev)
    rl = PeftModel.from_pretrained(rl, ADAPTER).to(dev).eval()
    res_rl = evaluate(rl, tokenizer, detector, det_tok, emb, sources, dev, "RL")

    summary = {
        "n": len(sources),
        "raw_source": {
            "mean_p_ai": statistics.fmean(src_p_ais),
            "mean_pattern": statistics.fmean(src_pats),
        },
        "base":  {k: v for k, v in res_base.items() if not isinstance(v, list)},
        "rl":    {k: v for k, v in res_rl.items() if not isinstance(v, list)},
        "delta_p_ai":     res_rl["mean_p_ai"]      - res_base["mean_p_ai"],
        "delta_pattern":  res_rl["mean_pattern"]   - res_base["mean_pattern"],
        "delta_similarity": res_rl["mean_similarity"] - res_base["mean_similarity"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "base": res_base, "rl": res_rl}, indent=2))

    print("\n========== RESULTS ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nFull output: {OUT}")


if __name__ == "__main__":
    main()
