"""Internal LLM-judge that mirrors the leaderboard's two axes: accuracy + tone.

Judges each clip's caption set against frames it re-samples itself, so it is
an independent check of the pipeline (not the pipeline grading its own work
with its own fact sheet).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sevcap.fireworks import Gemma, extract_json  # noqa: E402
from sevcap.sampler import sample_keyframes  # noqa: E402

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
    frames = sample_keyframes(video, 8)
    images = [f.b64() for f in frames]
    raw = await llm.vision_chat(
        JUDGE_PROMPT.format(n=len(images), **captions),
        images, temperature=0.0, max_tokens=6000, tag="eval-judge",
    )
    return extract_json(raw).get("scores", {})
