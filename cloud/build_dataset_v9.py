"""Phase 1 of Run #9: multi-source human corpora for style diversity.

Run #8 trained on WikiText-only and won spectacularly on Pangram (5/5)
but the trained model emits encyclopedic prose for every input. For a
real product users paste in marketing copy, blog posts, casual essays
— the model needs to humanize into matching registers, not always into
Wikipedia.

This builds a diverse training corpus by mixing 3 distinct human
registers, then having gpt-4o-mini rewrite each in 5 AI styles:

  WikiText-103   (encyclopedic, formal)         2000 humans
  cnn_dailymail  (news prose, narrative)        2000 humans
  xsum           (BBC, concise British prose)   2000 humans
                                                ─────
                                                6000 humans × 5 rewrites
                                                = 30,000 (ai, human) pairs

The model learns to *match the style* of the source AI text rather
than always converging to one register.

Cost: ~$5 OpenRouter (gpt-4o-mini at ~$0.00016/call × 30k = ~$5).

Run:
  source ~/.humanizer-openrouter.env
  python cloud/build_dataset_v9.py --per-source 2000 --rewrites 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"

# Reuse the 5 AI-styled prompts from build_dataset_v8.py
REWRITE_PROMPTS = [
    "Rewrite the user's text in a polished, professional tone like an AI "
    "writing assistant would. Use slightly more formal vocabulary, smoother "
    "transitions, and balanced sentence lengths. Do not change facts. "
    "Output only the rewrite.",
    "Rewrite the user's text in a typical AI-generated style: use words like "
    "'delve', 'leverage', 'intricate', 'multifaceted', 'paramount'. Start "
    "sentences with 'Furthermore', 'Moreover', 'Additionally'. Use uniform "
    "sentence lengths. Use 'It is important to note that...'. Output only "
    "the rewrite.",
    "Rewrite the user's text in marketing-style prose with promotional "
    "language. Phrases like 'cutting-edge', 'transformative', 'seamless', "
    "'robust', 'comprehensive'. Use rule-of-three constructions. Output "
    "only the rewrite.",
    "Rewrite the user's text in academic AI-paper style. Use passive voice, "
    "hedging ('it could be argued', 'one might suggest'), em-dashes, and "
    "phrases like 'in this context', 'the implications are'. Output only "
    "the rewrite.",
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


# ---- Source loaders ----

_WIKI_HEADER_RE = re.compile(r"^\s*=+\s*.*?\s*=+\s*$", re.MULTILINE)
_WIKI_CITATION_RE = re.compile(r"\s*\[\s*\d+\s*\]")


def clean_wikipedia(text: str) -> str:
    text = _WIKI_HEADER_RE.sub("", text)
    text = _WIKI_CITATION_RE.sub("", text)
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_into_paragraphs(text: str, min_words: int, max_words: int) -> list[str]:
    """Sentence-aware chunking. Yields contiguous-sentence blocks of
    min..max words."""
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    out, cur, cur_w = [], [], 0
    for s in sents:
        s = s.strip()
        if not s: continue
        wc = len(s.split())
        if cur_w + wc > max_words and cur_w >= min_words:
            out.append(" ".join(cur))
            cur, cur_w = [s], wc
        else:
            cur.append(s); cur_w += wc
    if cur and cur_w >= min_words:
        out.append(" ".join(cur))
    return out


def load_wikitext(n: int, min_words: int, max_words: int) -> list[str]:
    from datasets import load_dataset
    print(f"[wikitext] streaming Salesforce/wikitext (wikitext-103-raw-v1)...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    out, article = [], []
    for ex in ds:
        line = ex.get("text") or ""
        if line.startswith(" = ") and line.endswith(" = \n"):
            if article:
                cleaned = clean_wikipedia("".join(article))
                for p in chunk_into_paragraphs(cleaned, min_words, max_words):
                    out.append(p)
                    if len(out) >= n:
                        print(f"[wikitext] {len(out)} paragraphs", flush=True)
                        return out
                article = []
        else:
            article.append(line)
    if article:
        cleaned = clean_wikipedia("".join(article))
        for p in chunk_into_paragraphs(cleaned, min_words, max_words):
            out.append(p)
            if len(out) >= n: break
    print(f"[wikitext] {len(out)} paragraphs", flush=True)
    return out


def load_cnn(n: int, min_words: int, max_words: int) -> list[str]:
    from datasets import load_dataset
    print(f"[cnn] streaming abisee/cnn_dailymail...", flush=True)
    # Use the article body (not the highlights/summary)
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0",
                      split="train", streaming=True)
    out = []
    for ex in ds:
        text = (ex.get("article") or "").strip()
        # Strip CNN-specific filler
        text = re.sub(r"\(CNN\)\s*--\s*", "", text)
        text = re.sub(r"^[A-Z][A-Z, ]+--\s*", "", text, count=1)  # dateline
        text = re.sub(r"\s+", " ", text).strip()
        for p in chunk_into_paragraphs(text, min_words, max_words):
            out.append(p)
            if len(out) >= n:
                print(f"[cnn] {len(out)} paragraphs", flush=True)
                return out
    print(f"[cnn] {len(out)} paragraphs", flush=True)
    return out


def load_xsum(n: int, min_words: int, max_words: int) -> list[str]:
    from datasets import load_dataset
    print(f"[xsum] streaming EdinburghNLP/xsum...", flush=True)
    ds = load_dataset("EdinburghNLP/xsum", split="train", streaming=True)
    out = []
    for ex in ds:
        text = (ex.get("document") or "").strip()
        text = re.sub(r"\s+", " ", text).strip()
        for p in chunk_into_paragraphs(text, min_words, max_words):
            out.append(p)
            if len(out) >= n:
                print(f"[xsum] {len(out)} paragraphs", flush=True)
                return out
    print(f"[xsum] {len(out)} paragraphs", flush=True)
    return out


SOURCE_LOADERS = {
    "wikitext": load_wikitext,
    "cnn":      load_cnn,
    "xsum":     load_xsum,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=2000,
                    help="Humans per source")
    ap.add_argument("--rewrites", type=int, default=5)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--out", default="cloud/dataset_v9.jsonl")
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", nargs="+", default=["wikitext", "cnn", "xsum"])
    ap.add_argument("--limit-cost", type=float, default=12.0)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

    total_calls = args.per_source * len(args.sources) * args.rewrites
    est_tokens = total_calls * 350
    est_cost = est_tokens * 0.000_000_45
    print(f"[plan] {len(args.sources)} sources × {args.per_source} humans × "
          f"{args.rewrites} rewrites = {total_calls} calls")
    print(f"[plan] est ~{est_tokens/1e6:.2f}M tokens, ~${est_cost:.2f}")
    if est_cost > args.limit_cost:
        print(f"[plan] over --limit-cost ${args.limit_cost} — pass --limit-cost to override")
        sys.exit(1)

    # Load humans from each source. Streaming through one at a time keeps
    # memory low.
    all_humans: list[tuple[str, str]] = []  # (source, text)
    for src in args.sources:
        if src not in SOURCE_LOADERS:
            print(f"[plan] unknown source {src!r}; skipping"); continue
        humans = SOURCE_LOADERS[src](args.per_source, args.min_words, args.max_words)
        for h in humans:
            all_humans.append((src, h))

    print(f"[plan] loaded {len(all_humans)} humans across {len(args.sources)} sources")

    # Build job list
    jobs = []
    for i, (src, h) in enumerate(all_humans):
        for k in range(args.rewrites):
            jobs.append((i, src, h, k % len(REWRITE_PROMPTS)))

    out_path = os.path.expanduser(args.out)

    # Resume: if file exists, skip jobs we've already done. Each pair is
    # uniquely identified by (source, human_idx, prompt_idx).
    done_keys: set[tuple[str, int, int]] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r["source"], r["human_idx"], r["prompt_idx"]))
                except Exception:
                    continue
    print(f"[resume] {len(done_keys)} pairs already in {out_path}")

    pending = [j for j in jobs if (j[1], j[0], j[3]) not in done_keys]
    print(f"[gen] {len(pending)} jobs remaining; using as_completed (no head-of-line blocking)")

    def one_pair(args_tuple):
        i, src, human, prompt_idx = args_tuple
        ai_text = rewrite(human, REWRITE_PROMPTS[prompt_idx], args.model)
        if ai_text.startswith("__ERROR__"):
            return None
        return {"ai": ai_text, "human": human, "source": src,
                "prompt_idx": prompt_idx, "human_idx": i}

    from concurrent.futures import as_completed
    written = len(done_keys)
    completed_now = 0
    failed = 0
    t0 = time.time()
    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_pair, j): j for j in pending}
        for fut in as_completed(futures):
            completed_now += 1
            try:
                result = fut.result(timeout=200)
            except Exception as e:
                failed += 1; continue
            if result is None:
                failed += 1; continue
            f.write(json.dumps(result) + "\n"); f.flush()
            written += 1
            if completed_now % 100 == 0:
                rate = completed_now / (time.time() - t0)
                eta = (len(pending) - completed_now) / max(rate, 0.1)
                print(f"  [{completed_now}/{len(pending)}] {rate:.1f}/s ETA {eta/60:.1f}m  "
                      f"written={written} failed={failed}", flush=True)

    print(f"\n[done] {written} total pairs in {out_path}, "
          f"new this run: {completed_now-failed}, failed: {failed}, "
          f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
