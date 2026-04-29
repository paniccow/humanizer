"""Phase 1 of Run #8: build the reverse-translation training dataset.

Pulls human-written paragraphs from andythetechnerd03/AI-human-text,
asks gpt-4o-mini to rewrite each one K times in AI-style. Saves
(ai_input, human_target) pairs as JSONL.

Why this is the better paradigm:
  Standard humanizer training: AI text -> humanize -> reward by detector
    Problem: detector is binary, sparse signal, in Pangram's hard-negative
    mining set.

  Reverse-translation: AI(human_x) -> human_x as supervised target
    Every output token has a real human target. Dense, clean signal.
    Pangram can't have seen this paradigm because it's not adversarial
    against any detector.

Cost (gpt-4o-mini at $0.15 input / $0.60 output per 1M tokens):
  5000 human samples × 5 rewrites × ~250 tokens output = ~6.25M tokens
  ≈ $3.75 + a tiny bit of input cost
  ≈ ~$4 total. Override --n / --rewrites to tune.

Run:
  source ~/.humanizer-openrouter.env
  python cloud/build_dataset_v8.py --n 5000 --rewrites 5 --out cloud/dataset_v8.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"

# 5 distinct AI-styled rewriting instructions. Diversity matters — if all
# rewrites use the same prompt, the model just learns to undo that one prompt.
REWRITE_PROMPTS = [
    # Generic AI assistant tone
    "Rewrite the user's text in a polished, professional tone like an AI "
    "writing assistant would. Use slightly more formal vocabulary, smoother "
    "transitions, and balanced sentence lengths. Do not change facts. "
    "Output only the rewrite.",
    # Heavy AI tells deliberately
    "Rewrite the user's text in a typical AI-generated style: use words like "
    "'delve', 'leverage', 'intricate', 'multifaceted', 'paramount'. Start "
    "sentences with 'Furthermore', 'Moreover', 'Additionally'. Use uniform "
    "sentence lengths. Use 'It is important to note that...'. Output only "
    "the rewrite.",
    # Marketing / promotional AI tone
    "Rewrite the user's text in marketing-style prose with promotional "
    "language. Phrases like 'cutting-edge', 'transformative', 'seamless', "
    "'robust', 'comprehensive'. Use rule-of-three constructions. Output "
    "only the rewrite.",
    # Academic AI tone
    "Rewrite the user's text in academic AI-paper style. Use passive voice, "
    "hedging ('it could be argued', 'one might suggest'), em-dashes, and "
    "phrases like 'in this context', 'the implications are'. Output only "
    "the rewrite.",
    # Polished blog post tone
    "Rewrite the user's text as if a polished AI-assisted blog post. Each "
    "sentence flows neatly into the next. Add a conclusion that 'wraps "
    "things up nicely'. Output only the rewrite.",
]


def post_json(url, body, headers, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def rewrite(human_text: str, system_prompt: str, model: str, retries: int = 2) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Source:\n---\n{human_text}\n---"},
        ],
        "temperature": 0.9, "top_p": 0.95, "max_tokens": 600,
    }
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    for attempt in range(retries + 1):
        try:
            r = post_json(OPENROUTER, body, headers, timeout=90)
            return r["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                return f"__ERROR__: {e}"


import re

# Strip Wikipedia markup: section headers, citation refs, italics, etc.
_WIKI_HEADER_RE = re.compile(r"^\s*=+\s*.*?\s*=+\s*$", re.MULTILINE)
_WIKI_CITATION_RE = re.compile(r"\s*\[\s*\d+\s*\]")
_WIKI_BRACKETS_RE = re.compile(r"\s*<[^>]+>")  # any leftover tags
_WIKI_NEWLINES_RE = re.compile(r"\n{2,}")


def clean_wikipedia(text: str) -> str:
    """Strip wikitext-103 markup. Returns cleaned plaintext."""
    text = _WIKI_HEADER_RE.sub("", text)
    text = _WIKI_CITATION_RE.sub("", text)
    text = _WIKI_BRACKETS_RE.sub("", text)
    # Normalize "@-@" wikitext artifact (used for hyphens) and similar
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_paragraphs(article: str, min_words: int, max_words: int) -> list[str]:
    """Split a cleaned article into paragraph-shaped chunks. We use sentences
    since Wikipedia paragraphs tend to be huge — chunk into 60-200 word
    contiguous sentence blocks."""
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", article)
    out: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        wc = len(s.split())
        if cur_words + wc > max_words and cur_words >= min_words:
            out.append(" ".join(cur))
            cur = [s]; cur_words = wc
        else:
            cur.append(s); cur_words += wc
    if cur and cur_words >= min_words:
        out.append(" ".join(cur))
    return out


def load_human_samples(n: int, min_words: int, max_words: int,
                        source: str = "wikitext") -> list[str]:
    """Pull n human-written paragraphs.

    source='wikitext'  → WikiText-103 (100M tokens of verified Good/Featured
                          Wikipedia articles, human-reviewed encyclopedic text)
    source='ai-human'  → andythetechnerd03/AI-human-text (smaller, ~25K rows)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Need: pip install datasets", file=sys.stderr)
        sys.exit(1)

    if source == "wikitext":
        print(f"[data] streaming Salesforce/wikitext (wikitext-103-raw-v1)...", flush=True)
        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1",
            split="train", streaming=True,
        )
        out: list[str] = []
        # WikiText splits into individual lines; we need to re-aggregate articles.
        article_buf: list[str] = []
        for ex in ds:
            line = ex.get("text") or ""
            if line.startswith(" = ") and line.endswith(" = \n"):
                # New article header; flush prior article
                if article_buf:
                    article = clean_wikipedia("".join(article_buf))
                    paras = split_into_paragraphs(article, min_words, max_words)
                    for p in paras:
                        out.append(p)
                        if len(out) >= n:
                            print(f"[data] collected {len(out)} paragraphs", flush=True)
                            return out
                    article_buf = []
            else:
                article_buf.append(line)
            if len(out) and len(out) % 1000 == 0:
                print(f"  collected {len(out)}/{n}...", flush=True)
        # Flush final article
        if article_buf:
            article = clean_wikipedia("".join(article_buf))
            for p in split_into_paragraphs(article, min_words, max_words):
                out.append(p)
                if len(out) >= n:
                    break
        print(f"[data] {len(out)} paragraphs ready", flush=True)
        return out

    elif source == "ai-human":
        print(f"[data] streaming andythetechnerd03/AI-human-text...", flush=True)
        ds = load_dataset("andythetechnerd03/AI-human-text", split="train", streaming=True)
        out = []
        for ex in ds:
            if int(ex.get("generated", 0)) != 0:
                continue
            t = (ex.get("text") or "").strip()
            wc = len(t.split())
            if min_words <= wc <= max_words:
                out.append(t)
                if len(out) >= n:
                    break
        print(f"[data] {len(out)} human samples ready", flush=True)
        return out

    else:
        raise ValueError(f"unknown source {source!r}; use 'wikitext' or 'ai-human'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="Number of human samples")
    ap.add_argument("--rewrites", type=int, default=5, help="AI rewrites per human sample")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--out", default="cloud/dataset_v8.jsonl")
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8, help="Concurrent OpenRouter calls")
    ap.add_argument("--source", default="wikitext", choices=["wikitext", "ai-human"],
                    help="wikitext = Salesforce/wikitext-103 (100M tokens, recommended); "
                         "ai-human = andythetechnerd03/AI-human-text (smaller)")
    ap.add_argument("--limit-cost", type=float, default=10.0,
                    help="Estimated max $ to spend (rough check before starting)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. source ~/.humanizer-openrouter.env first.",
              file=sys.stderr)
        sys.exit(1)

    # Cost sanity check
    total_calls = args.n * args.rewrites
    est_tokens = total_calls * 350  # rough: ~100 input + ~250 output
    est_cost = est_tokens * 0.000_000_45  # weighted gpt-4o-mini average
    print(f"[plan] {args.n} humans × {args.rewrites} rewrites = {total_calls} LLM calls")
    print(f"[plan] est ~{est_tokens/1e6:.2f}M tokens, ~${est_cost:.2f}")
    if est_cost > args.limit_cost:
        print(f"[plan] over --limit-cost ${args.limit_cost} — pass --limit-cost to override")
        sys.exit(1)

    humans = load_human_samples(args.n, args.min_words, args.max_words, source=args.source)

    out_path = os.path.expanduser(args.out)
    written = 0
    t0 = time.time()
    print(f"[gen] starting AI rewrites with {args.workers} concurrent workers...")

    def one_pair(args_tuple):
        i, human, prompt_idx = args_tuple
        ai_text = rewrite(human, REWRITE_PROMPTS[prompt_idx], args.model)
        if ai_text.startswith("__ERROR__"):
            return None
        return {"ai": ai_text, "human": human, "prompt_idx": prompt_idx, "human_idx": i}

    # Build job list: every (human, prompt_idx) pair
    jobs = []
    for i, h in enumerate(humans):
        for k in range(args.rewrites):
            jobs.append((i, h, k % len(REWRITE_PROMPTS)))

    # Stream results to disk as they come in
    with open(out_path, "w") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for j, result in enumerate(pool.map(one_pair, jobs)):
            if result is None:
                continue
            f.write(json.dumps(result) + "\n")
            written += 1
            if (j + 1) % 50 == 0:
                rate = (j + 1) / (time.time() - t0)
                eta = (len(jobs) - j - 1) / max(rate, 0.1)
                print(f"  [{j+1}/{len(jobs)}] {rate:.1f} pairs/s, ETA {eta/60:.1f}m, written {written}", flush=True)

    print(f"\n[done] {written} pairs written to {out_path}")
    print(f"[done] elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
