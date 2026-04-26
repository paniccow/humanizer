from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import Detector, pick_device


class RoBERTaDetector(Detector):
    """Wraps a RoBERTa/DeBERTa sequence-classification AI-text detector from the Hub.

    Defaults to OpenAI's RoBERTa-large GPT-2 detector. Other compatible IDs:
      - openai-community/roberta-large-openai-detector
      - openai-community/roberta-base-openai-detector
      - desklib/ai-text-detector-v1.01            (DeBERTa v3 large, recent)
      - SuperAnnotate/roberta-large-llm-content-detector
      - fakespot-ai/roberta-base-ai-text-detection-v1
    """

    def __init__(
        self,
        model_id: str = "openai-community/roberta-large-openai-detector",
        device: str | None = None,
        max_length: int = 512,
        ai_label_index: int | None = None,
    ):
        self.name = model_id.split("/")[-1]
        self.model_id = model_id
        self.device = pick_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.ai_label_index = self._resolve_ai_label(ai_label_index)

    def _resolve_ai_label(self, override: int | None) -> int:
        if override is not None:
            return override
        id2label = getattr(self.model.config, "id2label", {}) or {}
        # OpenAI detector: 0 = "Real" (human), 1 = "Fake" (AI)
        for idx, label in id2label.items():
            l = str(label).lower()
            if any(k in l for k in ("fake", "ai", "machine", "generated", "llm", "label_1")):
                return int(idx)
        # Fall back to last index (commonly the "AI" / positive class).
        return self.model.config.num_labels - 1

    @torch.no_grad()
    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    @torch.no_grad()
    def score_batch(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, self.ai_label_index]
        return probs.detach().cpu().tolist()
