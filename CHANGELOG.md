# Changelog

## v0.6 — deployable service + per-detector transparency (2026-04)

The system is now deployable: a single uvicorn worker exposes the
rejection-sampling stack as an HTTP service.

### `humanizer/service/`

- `POST /humanize` — `{text, max_rounds?, candidates?, threshold?}` →
  `{text, passed, score, judge, rounds_used, judge_calls,
   per_detector, attempts, elapsed_ms}`. Lazy init of the rejection
  sampler + judge on first request.
- `POST /detect` — score arbitrary text with the configured judge,
  returns `p_ai` + per-detector breakdown. Useful for UI feedback and
  for testing the judge config without burning an LLM generation.
- `GET /health` — readiness probe + `paid_keys_set` list (which paid
  APIs are configured for the auto-judge ensemble).
- `GET /version`.

Optional bearer-token auth via `HUMANIZER_API_KEY` env. Auth on
`/humanize` and `/detect` only — health/version stay open for probes.

All env-var config reads use `field(default_factory=...)` so multi-
worker deployments and tests see fresh values, not values frozen at
class-definition time.

CLI: `humanizer serve --port 8000 [--workers N] [--reload]`. Optional
deps: `pip install -e '.[serve]'` (FastAPI + uvicorn).

### Per-detector breakdown surfaced through the rejection sampler

When the judge is an `EnsembleJudge` (≥2 paid detectors), the
rejection sampler now records per-detector scores for the chosen
candidate in `metadata["per_detector"]`. The `/humanize` and `/detect`
responses surface this. Lets the operator see which detector was the
weak link and decide whether to drop one from the ensemble.

### Run #5 — adversarial discriminator (in progress)

Build + first-attempt failure analysis + relaunch shipped in this
release. First attempt OOM'd at step ~150 on the second discriminator
update (fp32 disc + 8-batch × 30-step training added ~2GB peak on a
24GB-effective 4090). Fixes:
- Discriminator in bf16 instead of fp32 (load_discriminator).
- `disc_train_steps` 30 → 10, `disc_batch` 8 → 4.
- `gc.collect()` + `torch.cuda.empty_cache()` after each disc update.
- `set_to_none=True` on `zero_grad`.
- `done` sentinel written after each save_every checkpoint so the
  rescue stack can pull a partial adapter even if training dies later.

Second attempt at 7.7-8.5 GB VRAM (was OOM'ing at 24GB) — comfortable
headroom. Adversarial training stable at p_ai≈0.33 (the equilibrium
the discriminator pushes the policy toward).

### Tests

82 → 94:
- 8 service routing/auth/validation/contract tests with FastAPI
  TestClient + stubbed humanizer.
- 4 detect-endpoint + per-detector-breakdown tests (with-breakdown,
  without-breakdown, auth, response shape).

## v0.5 — rejection sampling against the live judge (2026-04)

The "money-back guarantee" inference layer. Single-shot ASR was the wrong
target — what users actually need is per-request reliability, and that's
what brute-force rejection sampling against the real-world detector gives
you. If the base humanizer's single-shot pass rate is `p`, best-of-N
rejection sampling lifts the per-request success rate to `1 - (1-p)^N`.
With `p = 0.7` and `N = 8`: 99.99%. With `p = 0.5` and `N = 8`: 99.6%.

### `humanizer reject`

  ```
  humanizer reject TEXT --judge auto -n 8 --rounds 4 --threshold 0.05
  ```

Sample N candidates → similarity-filter against the source → score each
through the judge → return on first `p_ai < threshold`. If none pass,
bump temperature and retry up to `--rounds`. Worst case: `n × rounds`
generations + judge calls; typical: passes round 1.

### Paid-detector clients (urllib only, no extra deps)

- **GPTZero** — `GPTZERO_API_KEY`. `p_ai = ai + 0.5*mixed`. ($135/mo for
  1M words.)
- **Originality.ai** — `ORIGINALITY_API_KEY`. `score.ai`. **9× cheaper
  than GPTZero** ($14.95/mo for 3M words).
- **Pangram** — `PANGRAM_API_KEY`. Tolerates 3 response shapes
  (`ai_likelihood`, `class_probabilities`, predicted-class+confidence).

Turnitin: no public API. Use `roberta-large-openai-detector` as a proxy
(correlates ~0.7 with Turnitin's signal in academic studies).

### `--judge auto` and `EnsembleJudge`

`humanizer.detectors.judge.judge_from_env()` inspects the operator's env
and returns whichever paid-detector ensemble is configured (or the local
RoBERTa-large fallback if none). The CLI's `--judge auto` uses this:
multi-detector ensemble when ≥2 keys are set, bare detector for 1 key,
local fallback when 0 keys. Never optimize against a single detector —
that's how you overfit.

### Pipeline integration

`humanizer pipeline --reject --judge auto` swaps the best-of-N + refine
stages for the rejection sampler, keeping scrub + burstiness + QA gate
around it.

### Tests

55 → 82 tests:
- 7 rejection-sampler tests (accept / escalate / exhaust / similarity-
  filter / temperature-ramp / telemetry).
- 15 paid-detector tests (response-parsing for all 3 APIs, including
  legacy schemas, mixed-class weighting, auth-failure surfaces).
- 9 judge-factory tests (ensemble aggregation, env-var inspection,
  prefer/fallback paths, integration with rejection sampler).

### Architecture notes

- All paid-detector clients use only `urllib` from the stdlib — no
  `requests` dependency. Tests stub `urllib.request.urlopen` at the
  module level (no network).
- `RejectionSamplingHumanizer` defers the `embedding_similarity` import
  via a wrapper so the module imports clean without torch.
- `EnsembleJudge` adapts a `DetectorEnsemble` to the single-detector
  `Detector` contract, returning the weighted aggregate `p_ai`.

## v0.4 — pipeline maturity (2026-04)

The system's deliverable shifted from "trained adapter" to "multi-stage
deterministic + LLM pipeline" after the cross-run benchmark in
`experiments/run-004/FINDINGS.md` showed that GRPO training did not move
detector evasion at our budget, while the deterministic scrub on top of
`Qwen2.5-3B-Instruct` produced cleaner output than any trained adapter.

### Core: scrub + patterns

- 13 pattern signals (was 10): added `abstract_subject`, `enumeration_shape`,
  `modality_overload`. The fingerprint's aggregate now reflects all 13.
- ~50 scrub rules covering:
  - Stiff transitions (Furthermore / Moreover / In conclusion / ...) at
    text-start AND mid-text
  - AI-favorite vocabulary (delve / leverage / intricate / multifaceted / ...)
  - Hedging boilerplate (It is important to note that / In today's...)
  - Em-dash thinning (parenthetical → comma, clause-separator → period)
  - Tricolon breaking (X, Y, and Z → X and Y. Z)
  - Sentence splitting at coordinators when > 28 words
  - Short-fragment merging via comma-join
  - Adaptive variance injection when sentence-length CV < 0.35
  - Archaic-formal markers dropped (whereby / thereof / hereby / henceforth /
    hitherto / wherein / whereupon)
  - a/an article repair after vowel-changing word swaps
  - Quoted-speech protection (double quotes never modified)
- Numbers: best `BASE+SCRUB` pattern aggregate **0.067** (Run #4 Qwen-3B).
  87% of 240 scrubbed eval outputs now trigger zero pattern flags.

### Pipeline

- `humanizer/pipeline/Pipeline` — composable orchestrator: scrub →
  paraphrase → best-of-N (16 default) → iterative refine (up to 3 passes)
  → burstiness post-process → QA gate.
- Targeted refine prompts: when iterating, the next-pass prompt names the
  specific signals still firing (e.g. "use concrete subjects instead of
  abstract nouns").
- `humanizer pipeline` CLI command, `--no-llm` flag for deterministic-only.

### Cloud training infrastructure

- TypeScript orchestration layer (`cloud/`) for RunPod 4090 rentals.
- Disconnect-safe `launch.ts` (training under nohup setsid; reconnect with
  `--resume <pod-id>`).
- Triple-redundant artifact rescue: `active_rescue.sh` (5-min poll for
  `done` sentinel) + `cost_cap.sh` (configurable max-hours, scps adapter
  before terminate) + `early_kill.sh` (kill pod if no step progress for
  25 min — direct fix for the run-#4-attempt-1 OOM that wasted ~$4).

### Tests

- 22 → 55 tests (15 patterns/scrub/pipeline + 19 advanced scrub stages
  + 14 safety/preservation).

### Reproducibility

- `scripts/headline_benchmark.py` — pure-python script that reads the 4
  eval JSONs and reproduces the README's headline numbers in ~5 seconds
  with no models or API key.

## v0.3 — multi-stage pipeline (2026-04)

- Added `humanizer/pipeline/scrub.py` (Stage 1) and `Pipeline` orchestrator.
- First demonstration that scrub + base outperforms trained adapter alone.

## v0.2 — patterns (2026-04)

- Added `humanizer/patterns/` — 10 signal AI fingerprint module with
  `analyze(text) -> Fingerprint` and the explain() bar-chart renderer.

## v0.1 — initial scaffold (2026-04)

- 4 GRPO training runs on RunPod (~$8 spend total, $10 budget).
- Run #1 baseline (Qwen-1.5B, simple reward) — marginal over base.
- Run #2 (multi-objective reward) — quality preservation big win, detector
  evasion still flat.
- Run #3 (4-detector reward) — rebalanced per-detector overfit.
- Run #4 (Qwen-3B + sim_floor 0.65 + w_det 1.5) — bigger base, similar
  evasion plateau. Confirmed: training-time GRPO at this budget does not
  move detector evasion meaningfully; the inference pipeline is the
  actual lever.
