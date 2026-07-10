"""Internal LLM-judge that mirrors the leaderboard's two axes: accuracy + tone.

Judges each clip's caption set against frames it re-samples itself, so it is
an independent check of the pipeline (not the pipeline grading its own work
with its own fact sheet).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sevcap.config import clip_profile  # noqa: E402
from sevcap.fireworks import Gemma, extract_json  # noqa: E402
from sevcap.sampler import probe_duration, sample_keyframes  # noqa: E402

JUDGE_PROMPT = """You are an impartial judge grading video captions on two axes.

These {n} images are keyframes from the video that was captioned.

Captions to grade:
1. formal: {formal}
2. sarcastic: {sarcastic}
3. humorous_tech: {humorous_tech}
4. humorous_non_tech: {humorous_non_tech}

For EACH caption give:
- accuracy 1-5: are all concrete claims true of the video? (5 = fully accurate,
  1 = describes things not in the video)
- tone 1-5: does it genuinely read as its labeled style? (5 = unmistakable,
  1 = wrong style entirely)

Think briefly if needed, then END your response with one line containing ONLY
the JSON:
{{"scores": {{"formal": {{"accuracy": 5, "tone": 5}}, "sarcastic": {{...}},
"humorous_tech": {{...}}, "humorous_non_tech": {{...}}}}}}"""


async def judge_clip(llm: Gemma, video: str, captions: dict[str, str]) -> dict:
    frames = sample_keyframes(video, clip_profile(probe_duration(video)).n_frames)
    images = [f.b64() for f in frames]
    prompt = JUDGE_PROMPT.format(n=len(images), **captions)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = await llm.vision_chat(
                prompt, images, temperature=0.0 if attempt == 0 else 0.3,
                max_tokens=6000, tag="eval-judge",
                # "low" reasoning can produce a long "thought" preamble that
                # eats the token budget before reaching the JSON; "none"
                # skips straight to the answer. Bypass cache on retry so a
                # truncated response never replays.
                reasoning="none",
                seed=None if attempt == 0 else 900 + attempt,
                cache=attempt == 0,
            )
            return extract_json(raw).get("scores", {})
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"judge failed after retries: {last_err}")
