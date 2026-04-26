"""Build the (ai_text, human_text) dataset used by SFT + GRPO.

Two sources, both already on the HF Hub — no manual download needed:

  - HC3 (Hello-SimpleAI/HC3): aligned (human_answer, chatgpt_answer) pairs across
    ~37 domains. The cleanest "same-prompt, different-author" dataset around.
  - WikiText-103 + on-the-fly generation: pull human paragraphs, generate AI
    paraphrases via OpenAI/local model. Use this to expand the training set
    after the HC3 warm-start.

The output is a Hugging Face `Dataset` with columns:
  - prompt:  the chat-formatted prompt (what the policy sees)
  - chosen:  the human-style target (used by SFT)
  - source:  raw AI text (used by GRPO as the input to humanize)
  - human:   the matched human text (for reference at eval)

Examples that don't have an AI counterpart are still useful — for GRPO we only
need `source` (AI text); the reward comes from the detector, not a reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import Dataset, load_dataset

from ..humanizers.prompt import _SYSTEM_PROMPT, _USER_TEMPLATE


@dataclass
class DataConfig:
    n_examples: int = 5000
    min_words: int = 80
    max_words: int = 500
    seed: int = 42
    include_hc3: bool = True
    hc3_domain: str | None = None       # None = all; or "reddit_eli5", "open_qa", etc.
    include_wikitext: bool = False      # requires generation; off by default


def _wc(s: str) -> int:
    return len(s.split())


def _format_prompt(text: str) -> dict:
    return {
        "prompt": _USER_TEMPLATE.format(text=text),
        "system": _SYSTEM_PROMPT,
    }


def _hc3_rows(cfg: DataConfig) -> Iterable[dict]:
    name = cfg.hc3_domain or "all"
    ds = load_dataset("Hello-SimpleAI/HC3", name, split="train")
    for ex in ds:
        humans = ex.get("human_answers") or []
        ais = ex.get("chatgpt_answers") or []
        if not humans or not ais:
            continue
        h = humans[0].strip()
        a = ais[0].strip()
        if not (cfg.min_words <= _wc(a) <= cfg.max_words):
            continue
        if not (cfg.min_words <= _wc(h) <= cfg.max_words):
            continue
        yield {"source": a, "human": h, "domain": ex.get("source", "hc3")}


def build(cfg: DataConfig | None = None) -> Dataset:
    cfg = cfg or DataConfig()
    rows: list[dict] = []
    if cfg.include_hc3:
        for r in _hc3_rows(cfg):
            rows.append(r)
            if len(rows) >= cfg.n_examples:
                break
    # Attach prompt formatting up-front; SFT reads `prompt`+`chosen`, GRPO reads `prompt`.
    formatted = []
    for r in rows:
        f = _format_prompt(r["source"])
        formatted.append({**r, **f, "chosen": r["human"]})
    ds = Dataset.from_list(formatted)
    return ds.shuffle(seed=cfg.seed)


if __name__ == "__main__":
    ds = build(DataConfig(n_examples=200))
    print(ds)
    print(ds[0])
