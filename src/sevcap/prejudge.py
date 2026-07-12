"""Lightweight vision judge for caption candidate selection.

Scores accuracy and style tone so we pick captions that raise the Track 2
leaderboard average (acc + tone) / 10.
"""

from __future__ import annotations

import asyncio
import logging
import re

from .config import clip_profile
from .fireworks import Gemma, extract_json
from .sampler import probe_duration, sample_keyframes

log = logging.getLogger("sevcap.prejudge")

SCORE_PROMPT = """Grade ONE video caption. These {n} images are keyframes.

Requested style: {style}
Caption: {caption}

accuracy 1-5: every concrete claim matches what is visible
  (punish invented objects, brands, cities, outcomes, or plot)
tone 1-5: clearly reads as the requested style
  formal=professional; sarcastic=dry irony; humorous_tech=tech joke;
  humorous_non_tech=everyday humor, no tech jargon

Reply with ONLY this JSON object and nothing else:
{{"accuracy": <int>, "tone": <int>}}"""

_AXES_RE = re.compile(
    r'"?accuracy"?\s*[:=]\s*([1-5]).{0,40}"?tone"?\s*[:=]\s*([1-5])',
    re.I | re.S,
)
_AXES_RE_REV = re.compile(
    r'"?tone"?\s*[:=]\s*([1-5]).{0,40}"?accuracy"?\s*[:=]\s*([1-5])',
    re.I | re.S,
)


def _parse_axes(raw: str) -> tuple[int, int]:
    """Parse accuracy/tone from model output; raise if unusable."""
    try:
        data = extract_json(raw)
        if isinstance(data, dict) and "accuracy" in data and "tone" in data:
            return (
                max(1, min(5, int(data["accuracy"]))),
                max(1, min(5, int(data["tone"]))),
            )
    except Exception:  # noqa: BLE001
        pass
    m = _AXES_RE.search(raw)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _AXES_RE_REV.search(raw)
    if m:
        return int(m.group(2)), int(m.group(1))
    raise ValueError(f"No parseable axes in model response: {raw[:120]!r}")


async def score_caption_axes(
    llm: Gemma, video: str, style: str, caption: str, frames=None,
) -> tuple[int, int]:
    if not caption or len(caption.strip()) < 8:
        return 1, 1
    kf = frames or sample_keyframes(
        video, clip_profile(probe_duration(video)).n_frames,
    )
    if len(kf) > 4:
        step = len(kf) / 4
        kf = [kf[int(i * step)] for i in range(4)]
    images = [f.b64() for f in kf]
    prompt = SCORE_PROMPT.format(n=len(images), style=style, caption=caption.strip())

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = await llm.vision_chat(
                prompt,
                images,
                temperature=0.0 if attempt == 0 else 0.2,
                max_tokens=800 if attempt == 0 else 1200,
                tag="prejudge",
                reasoning="none",
                seed=None if attempt == 0 else 700 + attempt,
                cache=False,
            )
            return _parse_axes(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    log.warning("prejudge failed for %s: %s", style, str(last_err)[:100])
    return 0, 0


def combined_rank(acc: int, tone: int) -> float:
    """Leaderboard-style rank; accuracy floor matters more for gate risk."""
    if acc <= 0 and tone <= 0:
        return -10.0
    return (
        (acc + tone) / 2.0
        + (0.55 if acc >= 5 else 0.35 if acc >= 4 else -0.4)
        + (0.15 if tone >= 4 else 0.0)
    )


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
    if best[1] <= 0 and best[2] <= 0:
        for cap, acc, tone, _rank, valid in scored:
            if valid:
                return cap, 3, 3
        return uniq[0], 3, 3
    return best[0], best[1], best[2]
