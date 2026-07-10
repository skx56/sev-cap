"""Lightweight accuracy judge for draft vs SEV caption selection."""

from __future__ import annotations

import asyncio
import logging

from .config import clip_profile
from .fireworks import Gemma, extract_json
from .sampler import probe_duration, sample_keyframes

log = logging.getLogger("sevcap.prejudge")

SCORE_PROMPT = """You are grading ONE video caption for factual accuracy.

These {n} images are keyframes from the video.

Caption ({style}):
{caption}

Rate accuracy 1-5:
5 = every concrete claim matches the video
3 = mostly right with minor issues
1 = describes things not in the video or invents dialogue

Think briefly if needed, then END with one line containing ONLY:
{{"accuracy": <integer 1-5>}}"""


async def score_caption_accuracy(
    llm: Gemma, video: str, style: str, caption: str, frames=None
) -> int:
    if not caption or len(caption.strip()) < 8:
        return 1
    try:
        kf = frames or sample_keyframes(
            video, clip_profile(probe_duration(video)).n_frames,
        )
        images = [f.b64() for f in kf]
        raw = await llm.vision_chat(
            SCORE_PROMPT.format(n=len(images), style=style, caption=caption.strip()),
            images,
            temperature=0.0,
            max_tokens=800,
            tag="prejudge",
            reasoning="none",
        )
        score = int(extract_json(raw).get("accuracy", 3))
        return max(1, min(5, score))
    except Exception as e:  # noqa: BLE001
        log.warning("prejudge failed for %s: %s", style, str(e)[:100])
        return 3


async def pick_better_caption(
    llm: Gemma,
    video: str,
    style: str,
    draft_cap: str,
    sev_cap: str,
    frames=None,
) -> tuple[str, dict]:
    """Return the more accurate caption and score metadata."""
    if not sev_cap or sev_cap.strip() == draft_cap.strip():
        return draft_cap, {"picked": "draft", "draft": None, "sev": None}
    draft_score, sev_score = await asyncio.gather(
        score_caption_accuracy(llm, video, style, draft_cap, frames=frames),
        score_caption_accuracy(llm, video, style, sev_cap, frames=frames),
    )
    meta = {"draft": draft_score, "sev": sev_score}
    if sev_score > draft_score:
        return sev_cap, {**meta, "picked": "sev"}
    return draft_cap, {**meta, "picked": "draft"}
