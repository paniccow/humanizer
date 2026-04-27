"""Long-lived Python inference worker — stdin/stdout JSON lines protocol.

Spawned by serve.ts. Loads the base model + LoRA adapter once and keeps them
resident, serving humanize requests one line at a time. Uses CUDA if available,
else MPS (Mac), else CPU.

Protocol (each line a JSON object with trailing newline):

  in   {"type":"humanize", "id": <int>, "text": "...", "n": <int>, "temperature": <float>, "burstiness": <bool>}
  out  {"type":"ready"}                               -- printed once at startup
  out  {"type":"response", "id":<int>, "ok":true,  "data":{"text":..., "pAi":..., "similarity":..., "attempts":...}}
  out  {"type":"response", "id":<int>, "ok":false, "error":"..."}

For best-of-N, the worker uses the loaded RoBERTa-base detector to pick the
candidate with the lowest p_ai (and falling-back if all candidates fail the
similarity threshold). Same logic as humanizer.humanizers.AdversarialHumanizer.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

DETECTOR_ID = "openai-community/roberta-base-openai-detector"
SIM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.6
MAX_NEW = 384

SYSTEM = (
    "You rewrite AI-generated text so it reads as if a real person wrote it, "
    "while preserving meaning. Vary sentence length aggressively. Use contractions. "
    "Avoid stiff transitional phrases like 'Furthermore' and 'Moreover'. Avoid words "
    "like 'leverage', 'delve', 'intricate', 'multifaceted'. Output ONLY the rewritten text."
)


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


class Worker:
    def __init__(self, base_model: str, adapter_path: str | None):
        self.dev = device()
        emit({"type": "log", "msg": f"device={self.dev}"})

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.dev != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(self.dev)
        if adapter_path:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path).to(self.dev)
                emit({"type": "log", "msg": f"loaded adapter from {adapter_path}"})
            except Exception as e:  # noqa: BLE001
                emit({"type": "log", "msg": f"adapter load failed ({e}); using base model only"})
        self.model.eval()

        self.det_tok = AutoTokenizer.from_pretrained(DETECTOR_ID)
        self.detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_ID).to(self.dev).eval()
        id2 = self.detector.config.id2label
        self.ai_idx = next(
            i for i, l in id2.items() if "fake" in str(l).lower() or "label_1" in str(l).lower()
        )

        self.emb = SentenceTransformer(SIM_MODEL, device="cpu")

    def _build_prompt(self, text: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Rewrite the following text:\n\n---\n{text}\n---"},
        ]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def _sample(self, text: str, n: int, temperature: float) -> list[str]:
        prompt = self._build_prompt(text)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.dev)
        out = self.model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=MAX_NEW,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        completions = out[:, enc.input_ids.shape[1] :]
        return [self.tokenizer.decode(c, skip_special_tokens=True).strip() for c in completions]

    @torch.no_grad()
    def _p_ai(self, texts: list[str]) -> list[float]:
        enc = self.det_tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.dev)
        logits = self.detector(**enc).logits.float()
        return torch.softmax(logits, dim=-1)[:, self.ai_idx].cpu().tolist()

    def _sim(self, original: str, candidates: list[str]) -> list[float]:
        ea = self.emb.encode([original] * len(candidates), normalize_embeddings=True, convert_to_tensor=True)
        eb = self.emb.encode(candidates, normalize_embeddings=True, convert_to_tensor=True)
        return (ea * eb).sum(-1).cpu().tolist()

    def humanize(self, text: str, n: int, temperature: float, burstiness: bool) -> dict[str, Any]:
        cands = self._sample(text, n=max(n, 1), temperature=temperature)
        p_ais = self._p_ai(cands)
        sims = self._sim(text, cands)
        scored = list(zip(cands, p_ais, sims))
        kept = [c for c in scored if c[2] >= SIM_THRESHOLD]
        pool = kept or scored
        best = min(pool, key=lambda c: c[1])
        out_text = best[0]
        if burstiness:
            out_text = self._apply_burstiness(out_text)
        return {"text": out_text, "pAi": float(best[1]), "similarity": float(best[2]), "attempts": len(cands)}

    @staticmethod
    def _apply_burstiness(text: str) -> str:
        # Light-touch surface edits: contractions + transition variation.
        # Mirrors humanizer/postprocess/burstiness.py at minimum surface area
        # so the worker has no extra dependencies.
        import re

        contractions = {
            r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't", r"\bdid not\b": "didn't",
            r"\bis not\b": "isn't", r"\bare not\b": "aren't", r"\bwas not\b": "wasn't",
            r"\bcannot\b": "can't", r"\bwill not\b": "won't", r"\bwould not\b": "wouldn't",
            r"\bit is\b": "it's", r"\bthat is\b": "that's", r"\bthere is\b": "there's",
        }
        for pat, repl in contractions.items():
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)
        return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default=None)
    args = p.parse_args()

    adapter = args.adapter if args.adapter and args.adapter not in ("", "none") else None
    try:
        w = Worker(args.base, adapter)
    except Exception as e:  # noqa: BLE001
        emit({"type": "fatal", "error": str(e)})
        sys.exit(1)

    emit({"type": "ready"})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            emit({"type": "response", "id": -1, "ok": False, "error": f"bad json: {e}"})
            continue
        rid = int(req.get("id", -1))
        if req.get("type") != "humanize":
            emit({"type": "response", "id": rid, "ok": False, "error": "unknown type"})
            continue
        try:
            data = w.humanize(
                text=req["text"],
                n=int(req.get("n", 1)),
                temperature=float(req.get("temperature", 0.85)),
                burstiness=bool(req.get("burstiness", False)),
            )
            emit({"type": "response", "id": rid, "ok": True, "data": data})
        except Exception as e:  # noqa: BLE001
            emit({"type": "response", "id": rid, "ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
