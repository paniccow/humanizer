"""Generation strategies for the strategy-carousel rejection sampler.

The first round of rejection sampling cycles through these. Each strategy
forces the LLM into a fundamentally different *text shape*, not just a
different prose style — that's the lever paraphrase-only humanization
keeps missing. Gpt-4o-mini paraphrasing AI text always produces
paraphrase-shaped output (same sentence-length distribution, same
function-word ratio, same para structure), and Pangram catches it.

Tweet thread, reddit comment, diary entry — each pulls the model into a
different distribution of token statistics. Higher chance one of them
crosses Pangram's "human" boundary.

Each Strategy is (name, system_prompt, temperature). Pass strategies as
RejectionConfig.prompt_strategies to cycle through them per round (or per
candidate when concurrent_judge_calls > 1).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    """One generation strategy. The system prompt forces a text shape; the
    temperature is the diversity knob within that shape."""

    name: str
    system_prompt: str
    temperature: float = 1.0


# Default essay-rewrite — close to PromptHumanizer's built-in default but
# slightly more explicit. Use when the user wants paraphrase semantics.
DEFAULT_REWRITE = Strategy(
    name="default_rewrite",
    system_prompt=(
        "Rewrite the user's text so it reads as if a real person wrote it. "
        "Vary sentence length wildly (mix 3-word fragments with 30+ word "
        "run-ons). ALWAYS contract (don't, won't, it's). Inject 'I'/'we'/'you' "
        "naturally. Add at least one parenthetical aside per paragraph. "
        "NEVER use: delve, leverage, intricate, multifaceted, paramount, "
        "robust, comprehensive, holistic, transformative, ecosystem, "
        "landscape, realm, foster, harness, navigate, tapestry. NEVER start "
        "a sentence with: Furthermore, Moreover, Additionally, Consequently, "
        "Therefore, In conclusion, Overall. PRESERVE every fact and number. "
        "OUTPUT only the rewrite."
    ),
    temperature=0.95,
)

# Tweet thread — short, punchy, numbered, conversational
TWEET_THREAD = Strategy(
    name="tweet_thread",
    system_prompt=(
        "Convert the user's text into a 6-tweet Twitter thread (1/6, 2/6 ...). "
        "Real Twitter voice: punchy, fragments, casual. Use 1-2 emojis total. "
        "Preserve the facts. Output only the thread."
    ),
    temperature=0.95,
)

# Reddit comment — opinions, abbreviations, occasional typos
REDDIT_POST = Strategy(
    name="reddit_post",
    system_prompt=(
        "Rewrite as a Reddit comment from a tech-skeptical user. They start "
        "lowercase, use 'tbh' / 'imo' / 'lol', and make occasional typos that "
        "aren't fixed (e.g. 'teh', 'definately'). They have an opinion (mildly "
        "skeptical of AI hype). Voice the skepticism. Preserve underlying "
        "facts. Output only the comment."
    ),
    temperature=1.0,
)

# Slack message between coworkers
SLACK_MESSAGE = Strategy(
    name="slack_message",
    system_prompt=(
        "Rewrite as a casual Slack message Sarah is sending to her coworker "
        "Dave explaining what AI means for their team. Include a digression "
        "(e.g. 'btw did you see that demo last week?'). End mid-thought. "
        "Preserve facts. Output only the message body, no headers."
    ),
    temperature=0.95,
)

# Diary entry — first-person internal voice
DIARY_ENTRY = Strategy(
    name="diary_entry",
    system_prompt=(
        "Rewrite as a diary entry someone scribbled at the end of a long "
        "workday. First-person, present-tense, fragments interleaved with "
        "longer reflective sentences. Includes a personal observation that's "
        "tangentially related. Preserve the technical facts. Output only "
        "the entry."
    ),
    temperature=1.0,
)

# Interview transcript — Q&A with verbal cadence (filler words, restarts)
INTERVIEW = Strategy(
    name="interview_transcript",
    system_prompt=(
        "Rewrite as an interview transcript: a journalist asks 'Q:' questions "
        "and a tech worker gives 'A:' answers. The worker speaks in real "
        "spoken English: 'um', 'I mean', trail-offs (...), restarts mid-"
        "sentence. Preserve the facts. Output only the transcript."
    ),
    temperature=1.0,
)

# All strategies — use as default carousel
ALL_STRATEGIES: list[Strategy] = [
    DEFAULT_REWRITE,
    TWEET_THREAD,
    REDDIT_POST,
    SLACK_MESSAGE,
    DIARY_ENTRY,
    INTERVIEW,
]
