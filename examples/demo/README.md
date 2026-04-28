# Demo

Self-contained HTML demo for the `humanizer` service. No build step,
no framework. Just static `index.html` that hits a running service.

## Run

```bash
# 1. Start the service in one terminal
export OPENAI_API_KEY=sk-or-v1-...      # OpenRouter or OpenAI
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # if OpenRouter
# Optional: paid judges (any combination — auto-judge ensembles them)
export PANGRAM_API_KEY=...        # cheapest, $0.05/1K words
export ORIGINALITY_API_KEY=...    # $14.95/mo flat
export GPTZERO_API_KEY=...        # $135/mo, more expensive but the brand-name one
# Required for the demo: allow CORS from the static-server origin
export HUMANIZER_CORS_ORIGINS="http://localhost:7000"
humanizer serve --port 8000

# 2. Open the demo in another terminal
python -m http.server 7000 --directory examples/demo
# Then open http://localhost:7000 in browser
```

The demo's "endpoint" field defaults to `http://localhost:8000`. Change
it if you've deployed the service elsewhere. CORS may bite — for
production deploys, add a CORS middleware to the FastAPI app.

## What it shows

- **Humanize** — full rejection sampling against the configured judge.
  Surfaces `passed/exhausted` badge, per-detector breakdown, request
  stats (elapsed, attempts, judge calls, rounds).
- **Score original only** — calls `/detect` to show the AI-likelihood
  of the input *without* humanizing. Useful for benchmarking the input.
- **Reliability projection** — given the last p_ai, shows the math:
  best-of-8 / 16 / 32 reliability under the `1-(1-p)^N` formula.

## What it doesn't do (and won't)

No accounts, no payments, no rate limiting, no production polish.
This is a working *demo*, not a SaaS frontend. If you want a paid
product UI, fork the HTML and wire your own auth/billing on top.
