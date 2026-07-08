"""Stage 1: K independent fact extractions from keyframes (Gemma vision).

Each extraction runs at moderate temperature with a distinct seed so the
samples are genuinely independent draws — the variance between them is the
signal semantic entropy needs (Farquhar et al., Nature 2024).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from .fireworks import Gemma, extract_json
from .sampler import Keyframe

log = logging.getLogger("sevcap.extractor")

CATEGORIES = ("objects", "actions", "setting", "events", "on_screen_text")

SYSTEM = (
    "You are a meticulous video analyst. You only state what is visibly "
    "present in the frames. You never guess brands, names, locations, or "
    "intentions that are not directly observable."
)

PROMPT = """These {n} images are keyframes sampled in chronological order from ONE short video clip (timestamps: {ts}).

List atomic facts about the video. An atomic fact is ONE short, self-contained, verifiable statement (max ~12 words), e.g. "a man in a red jacket rides a bicycle".

Rules:
- Only include what is clearly visible. If unsure, leave it out.
- No speculation about names, brands, emotions, or off-screen events.
- Describe changes over time as events (e.g. "the car stops at a crossing").
- If text is legible on screen, quote it exactly under on_screen_text.

Return ONLY JSON in exactly this shape:
{{
  "objects": ["..."],
  "actions": ["..."],
  "setting": ["..."],
  "events": ["..."],
  "on_screen_text": ["..."]
}}"""


@dataclass
class Extraction:
    """One sampled fact extraction (one of K)."""
    sample_id: int
    facts: dict[str, list[str]] = field(default_factory=dict)

    def flat(self) -> list[tuple[str, str]]:
        out = []
        for cat in CATEGORIES:
            for f in self.facts.get(cat, []):
                if isinstance(f, str) and f.strip():
                    out.append((cat, f.strip()))
        return out


def _parse(raw: str, sample_id: int) -> Extraction:
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("extraction response is not a JSON object")
    facts = {cat: [str(x) for x in data.get(cat, []) if str(x).strip()] for cat in CATEGORIES}
    return Extraction(sample_id=sample_id, facts=facts)


async def extract_facts(
    llm: Gemma, frames: list[Keyframe], k: int = 5, temperature: float = 0.7
) -> list[Extraction]:
    """Run K independent extractions concurrently; tolerate partial failures."""
    images = [f.b64() for f in frames]
    ts = ", ".join(f"{f.t:.0f}s" for f in frames)
    prompt = PROMPT.format(n=len(frames), ts=ts)

    async def one(i: int) -> Extraction | None:
        try:
            raw = await llm.vision_chat(
                prompt, images, temperature=temperature, seed=1000 + i,
                tag="extract", system=SYSTEM,
            )
            return _parse(raw, i)
        except Exception as e:  # noqa: BLE001
            log.warning("extraction sample %d failed: %s", i, e)
            return None

    results = await asyncio.gather(*(one(i) for i in range(k)))
    good = [r for r in results if r is not None and r.flat()]
    if not good:
        raise RuntimeError("all fact extractions failed")
    return good
