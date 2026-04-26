"""Real best-of-N eval — training-free.

Uses the BASE SmolLM2-135M-Instruct, samples N completions per prompt, picks
the one with the lowest detector p_ai (no training needed). This is the
Adversarial Paraphrasing recipe (arXiv 2506.07001).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, "/tmp/humanizer")
from humanizer.patterns import analyze as pattern_analyze  # noqa: E402

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
DETECTOR = "openai-community/roberta-base-openai-detector"
SIMILARITY = "sentence-transformers/all-MiniLM-L6-v2"
EVAL_FILE = "data/eval.jsonl"
OUT = Path("logs/eval_bestofn.json")
N_PROMPTS = 8
N_CANDIDATES = 6
MAX_NEW = 160
SIM_THRESHOLD = 0.55  # MiniLM cosine, lenient since SmolLM is small

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


def build_prompt(tok, src):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Rewrite the following text:\n\n---\n{src}\n---"},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def sample_n(model, tok, src, dev, n):
    enc = tok(build_prompt(tok, src), return_tensors="pt").to(dev)
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=0.95,
        top_p=0.95,
        max_new_tokens=MAX_NEW,
        num_return_sequences=n,
        pad_token_id=tok.eos_token_id,
    )
    completions = out[:, enc.input_ids.shape[1]:]
    return [tok.decode(c, skip_special_tokens=True).strip() for c in completions]


@torch.no_grad()
def detector_p_ai(detector, det_tok, texts, dev):
    enc = det_tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(dev)
    logits = detector(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    id2 = detector.config.id2label
    ai_idx = next(i for i, l in id2.items() if "fake" in str(l).lower() or "label_1" in str(l).lower())
    return probs[:, int(ai_idx)].cpu().tolist()


def cosine(emb, a, b):
    ea = emb.encode(a, normalize_embeddings=True, convert_to_tensor=True)
    eb = emb.encode(b, normalize_embeddings=True, convert_to_tensor=True)
    return (ea * eb).sum(-1).cpu().tolist()


def main():
    dev = device()
    print(f"device: {dev}")
    sources = [json.loads(l)["source"] for l in Path(EVAL_FILE).read_text().splitlines()][:N_PROMPTS]
    print(f"eval prompts: {len(sources)}")

    print(f"loading detector + similarity")
    det_tok = AutoTokenizer.from_pretrained(DETECTOR)
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR).to(dev).eval()
    emb = SentenceTransformer(SIMILARITY, device="cpu")

    print(f"loading base {BASE}")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to(dev).eval()

    print(f"\n=== Best-of-{N_CANDIDATES} on {N_PROMPTS} prompts ===")
    chosen = []
    single = []  # for comparison: pick the FIRST candidate (no selection)
    for i, src in enumerate(sources):
        cands = sample_n(model, tok, src, dev, N_CANDIDATES)
        p_ais = detector_p_ai(detector, det_tok, cands, dev)
        sims = cosine(emb, [src] * len(cands), cands)
        # filter by sim, then pick lowest p_ai
        kept = [(c, p, s) for c, p, s in zip(cands, p_ais, sims) if s >= SIM_THRESHOLD]
        pool = kept or list(zip(cands, p_ais, sims))
        best = min(pool, key=lambda x: x[1])
        chosen.append({"source": src, "text": best[0], "p_ai": best[1], "similarity": best[2]})
        single.append({"source": src, "text": cands[0], "p_ai": p_ais[0], "similarity": sims[0]})
        print(f"  [{i+1}/{N_PROMPTS}] best p_ai={best[1]:.3f}  (1st sample p_ai={p_ais[0]:.3f})  "
              f"kept={len(kept)}/{N_CANDIDATES}")

    chosen_p = [c["p_ai"] for c in chosen]
    single_p = [c["p_ai"] for c in single]
    chosen_s = [c["similarity"] for c in chosen]
    single_s = [c["similarity"] for c in single]
    chosen_pat = [pattern_analyze(c["text"]).aggregate for c in chosen]
    single_pat = [pattern_analyze(c["text"]).aggregate for c in single]

    summary = {
        "n_prompts": N_PROMPTS,
        "n_candidates": N_CANDIDATES,
        "single_sample": {
            "mean_p_ai": statistics.fmean(single_p),
            "mean_similarity": statistics.fmean(single_s),
            "mean_pattern": statistics.fmean(single_pat),
        },
        "best_of_n": {
            "mean_p_ai": statistics.fmean(chosen_p),
            "mean_similarity": statistics.fmean(chosen_s),
            "mean_pattern": statistics.fmean(chosen_pat),
        },
        "delta_p_ai": statistics.fmean(chosen_p) - statistics.fmean(single_p),
        "delta_pattern": statistics.fmean(chosen_pat) - statistics.fmean(single_pat),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "best_of_n": chosen, "single": single}, indent=2))

    print("\n========== RESULTS ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nFull output: {OUT}")


if __name__ == "__main__":
    main()
