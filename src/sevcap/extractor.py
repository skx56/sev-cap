"""Stage 1: K independent fact extractions from keyframes (Gemma vision).

Each extraction runs at moderate temperature with a distinct seed so the
samples are genuinely independent draws — the variance between them is the
signal semantic entropy needs (Farquhar et al., Nature 2024).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from .config import settings
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
{audio_block}
List the SALIENT atomic facts about the video. An atomic fact is ONE short, self-contained, verifiable statement (max ~12 words), e.g. "a man in a red jacket rides a bicycle".

Rules:
- Only include what is clearly visible or clearly said. If unsure, leave it out.
- Prioritize the STORY: what happens, in order, goes under events (3-6 events).
- At most 6 facts per category — the most important ones only, no minor
  background details (individual leaves, rocks, lighting nuances).
- Name each subject consistently (e.g. always "the squirrel", not sometimes
  "the small creature").
- No speculation about names, brands, emotions, or off-screen events.
- If text is legible on screen, quote it exactly under on_screen_text.
- If the audio transcript contains a distinct spoken line, you may include ONE
  atomic fact under "events" like: the character says "<short quote>" — only
  if the transcript above is non-empty.

Think briefly if needed, then END your response with ONLY a JSON object in
exactly this shape (no other text after it):
{{
  "objects": ["..."],
  "actions": ["..."],
  "setting": ["..."],
  "events": ["..."],
  "on_screen_text": ["..."]
}}"""

_AUDIO_BLOCK = '\nAudio transcript of this clip (may be empty or imperfect ASR output): "{transcript}"\n'


@dataclass
class Extraction:
    """One sampled fact extraction (one of K)."""
    sample_id: int
    facts: dict[str, list[str]] = field(default_factory=dict)

    def flat(self) -> list[tuple[str, str]]:
        out = []
        for cat in CATEGORIES:
            for f in self.facts.get(cat, []):
                if not isinstance(f, str):
                    continue
                txt = f.strip()
                # drop junk "facts" like "...", "N/A", lone punctuation
                if len(txt) >= 8 and sum(c.isalpha() for c in txt) >= 5:
                    out.append((cat, txt))
        return out


def _junk(s: str) -> bool:
    """Template placeholders and non-statements must never become facts."""
    t = s.strip().strip(".").strip()
    return len(t) < 4 or t in {"..", "…", "etc", "n/a", "none", "N/A", "None"}


def _parse(raw: str, sample_id: int) -> Extraction:
    data = extract_json(raw)
    if isinstance(data, list):  # model emitted a bare list of facts
        data = {"events": [x for x in data if isinstance(x, str)]}
    if not isinstance(data, dict):
        raise ValueError("extraction response is not a JSON object")
    facts = {
        cat: [str(x).strip() for x in data.get(cat, [])
              if str(x).strip() and not _junk(str(x))]
        for cat in CATEGORIES
    }
    if not any(facts.values()):
        raise ValueError("extraction contained no usable facts")
    return Extraction(sample_id=sample_id, facts=facts)


async def extract_facts(
    llm: Gemma,
    frames: list[Keyframe],
    k: int = 5,
    temperature: float = settings.extract_temperature,
    transcript: str = "",
) -> list[Extraction]:
    """Run K independent extractions concurrently; tolerate partial failures.

    `transcript` (if any) is identical across all K samples, so any fact it
    grounds shows up consistently and earns its verification the normal way —
    audio is a second evidence channel into Stage 1, not a bypass around it.
    """
    images = [f.b64() for f in frames]
    ts = ", ".join(f"{f.t:.0f}s" for f in frames)
    audio_block = _AUDIO_BLOCK.format(transcript=transcript) if transcript else ""
    prompt = PROMPT.format(n=len(frames), ts=ts, audio_block=audio_block)

    async def one(i: int) -> Extraction | None:
        for attempt in range(2):
            try:
                # A fixed seed reproduces a bad sample deterministically, so
                # a retry must resample with a different seed to be useful.
                seed = 1000 + i if attempt == 0 else 9000 + i * 97 + attempt
                raw = await llm.vision_chat(
                    prompt, images, temperature=temperature, seed=seed,
                    tag="extract", system=SYSTEM, max_tokens=6000,
                    cache=attempt == 0,  # never replay a cached bad response
                )
                return _parse(raw, i)
            except Exception as e:  # noqa: BLE001
                log.warning("extraction sample %d attempt %d failed: %s",
                            i, attempt + 1, str(e)[:150])
        return None

    results = await asyncio.gather(*(one(i) for i in range(k)))
    good = [r for r in results if r is not None and r.flat()]
    if not good:
        raise RuntimeError("all fact extractions failed")
    return good
