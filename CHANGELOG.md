# Changelog

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
