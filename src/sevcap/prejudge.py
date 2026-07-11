"""Lightweight vision judge for caption candidate selection.

Scores both accuracy and style tone so we can pick captions that raise the
leaderboard-style average (acc+tone)/10, not accuracy alone.
"""

from __future__ import annotations

import asyncio
import logging

from .config import clip_profile
from .fireworks import Gemma, extract_json
from .sampler import probe_duration, sample_keyframes

log = logging.getLogger("sevcap.prejudge")

SCORE_PROMPT = """You are grading ONE video caption for factual accuracy AND style fit.

These {n} images are keyframes from the video.

Requested style: {style}
Caption:
{caption}

Rate:
- accuracy 1-5: every concrete claim matches what is visible
  (punish invented objects, brands, cities, outcomes, or plot)
- tone 1-5: does it clearly read as the requested style?
  formal=professional neutral; sarcastic=dry irony; humorous_tech=tech joke;
  humorous_non_tech=everyday humor, no tech jargon

Think briefly if needed, then END with one line containing ONLY:
{{"accuracy": <int>, "tone": <int>}}"""


async def score_caption_axes(
    llm: Gemma, video: str, style: str, caption: str, frames=None,
) -> tuple[int, int]:
    if not caption or len(caption.strip()) < 8:
        return 1, 1
    try:
        kf = frames or sample_keyframes(
            video, clip_profile(probe_duration(video)).n_frames,
        )
        # Keep prejudge cheap: at most 4 evenly spaced frames.
        if len(kf) > 4:
            step = len(kf) / 4
            kf = [kf[int(i * step)] for i in range(4)]
        images = [f.b64() for f in kf]
        raw = await llm.vision_chat(
            SCORE_PROMPT.format(n=len(images), style=style, caption=caption.strip()),
            images,
            temperature=0.0,
            max_tokens=200,
            tag="prejudge",
            reasoning="none",
            cache=False,
        )
        data = extract_json(raw)
        acc = max(1, min(5, int(data.get("accuracy", 3))))
        tone = max(1, min(5, int(data.get("tone", 3))))
        return acc, tone
    except Exception as e:  # noqa: BLE001
        log.warning("prejudge failed for %s: %s", style, str(e)[:100])
        return 3, 3


async def score_caption_accuracy(
    llm: Gemma, video: str, style: str, caption: str, frames=None,
) -> int:
    acc, _ = await score_caption_axes(llm, video, style, caption, frames=frames)
    return acc


def combined_rank(acc: int, tone: int) -> float:
    """Leaderboard-style rank; accuracy floor matters more for gate risk."""
    return (acc + tone) / 2.0 + (0.35 if acc >= 4 else 0.0) + (0.15 if tone >= 4 else 0.0)


async def pick_best_candidate(
    llm: Gemma,
    video: str,
    style: str,
    candidates: list[str],
    frames=None,
    is_valid=None,
) -> tuple[str, int, int]:
    """Pick the candidate with best (accuracy, tone), preferring valid ones."""
    uniq: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        key = c.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(c.strip())
    if not uniq:
        return "", 1, 1

    scores = await asyncio.gather(
        *(score_caption_axes(llm, video, style, cap, frames=frames) for cap in uniq)
    )
    scored: list[tuple[str, int, int, float, bool]] = []
    for cap, (acc, tone) in zip(uniq, scores):
        valid = True if is_valid is None else bool(is_valid(cap))
        rank = combined_rank(acc, tone) + (0.5 if valid else -1.0)
        scored.append((cap, acc, tone, rank, valid))

    scored.sort(key=lambda x: (x[4], x[3], x[1], x[2]), reverse=True)
    best = scored[0]
    return best[0], best[1], best[2]


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
    (d_acc, d_tone), (s_acc, s_tone) = await asyncio.gather(
        score_caption_axes(llm, video, style, draft_cap, frames=frames),
        score_caption_axes(llm, video, style, sev_cap, frames=frames),
    )
    meta = {"draft": d_acc, "sev": s_acc, "draft_tone": d_tone, "sev_tone": s_tone}
    if combined_rank(s_acc, s_tone) > combined_rank(d_acc, d_tone):
        return sev_cap, {**meta, "picked": "sev"}
    return draft_cap, {**meta, "picked": "draft"}
