"""Grounded captioning: describe frames → self-verify → write styled captions.

Matches the high-scoring Raccoon-Vision-Translator pattern (describe, verify,
then text-only style writes from one shared grounding description) while
keeping our style definitions and duration-scaled frame sampling.

Tuned against AMD Track 2 sample references: specific visual details first,
then short style-faithful captions with no hallucinations.
"""

from __future__ import annotations

import asyncio
import re

from .config import ClipProfile
from .fireworks import Gemma
from .sampler import Keyframe
from .styles import StyleConfig

_META_RE = re.compile(
    r"^(I cannot|I can't|As an AI|Based on the (frames|keyframes|images)|"
    r"The (fact sheet|verified facts))",
    re.I | re.M,
)


def _safe_format(template: str, **kwargs: str | int) -> str:
    """Format prompts without breaking on braces inside model/user text."""
    out = template
    for key, val in kwargs.items():
        out = out.replace("{" + key + "}", str(val))
    return out


def _clean_prose(raw: str) -> str:
    text = raw.strip().strip('"').strip()
    if not text:
        raise ValueError("empty description")
    if text.lower().startswith("description:"):
        text = text.split(":", 1)[1].strip()
    return text


def _clean_caption(raw: str) -> str:
    text = raw.strip().strip('"').strip()
    if text.upper().startswith("CAPTION:"):
        text = text.split(":", 1)[1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[-1] if len(lines) > 1 and lines[0].lower().startswith("here") else " ".join(lines)
    if not text or _META_RE.search(text):
        raise ValueError(f"meta or empty caption: {text[:80]!r}")
    return text


_DESCRIBE_PROMPT = """These {n} images are frames sampled across a {duration}-second video clip, in chronological order.
{audio_block}{long_hint}
Write a dense factual description of what is VISIBLY present. Include:
- setting / location type
- main subjects (people, animals, objects) with concrete attributes (color, size, clothing, breed if clear)
- what is happening / motion across the frames
- notable background details (buildings, weather, lighting) only if clearly visible
Do NOT invent names of cities, brands, companies, or dialogue you cannot clearly read.
If text is blurry or uncertain, omit it rather than guess.
Be specific; prefer "golden ginkgo trees" over "trees", "orange tabby kitten" over "cat".
Write 2-4 sentences. Output ONLY the description text, no preamble or labels."""

_VERIFY_PROMPT = """These {n} images are frames from the same video clip.

Draft description:
"{draft}"

Check the draft against the actual frames.
- Keep every detail that is clearly visible and specific.
- Remove or correct anything invented, wrong, or too generic.
- Drop uncertain brand names, city names, or signage you cannot clearly verify.
- Do not add new speculative details.
Output ONLY the final description — no preamble."""


async def describe_scene(
    llm: Gemma,
    frames: list[Keyframe],
    profile: ClipProfile,
    transcript: str = "",
    transcript_trusted: bool = False,
) -> str:
    audio_block = ""
    if transcript and transcript_trusted:
        audio_block = f'\nTrusted dialogue transcript: "{transcript}"\n'
    long_hint = ""
    if profile.long_form:
        long_hint = (
            "\nThis is a longer clip — summarize the MAIN arc (who, what happens, "
            "outcome) in 2-4 sentences. Omit minor background props.\n"
        )
    images = [f.b64() for f in frames]
    prompt = _safe_format(
        _DESCRIBE_PROMPT,
        n=len(images),
        duration=int(profile.duration_s),
        audio_block=audio_block,
        long_hint=long_hint,
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = await llm.vision_chat(
                prompt, images, temperature=0.2, max_tokens=450,
                tag="describe", reasoning="none", cache=attempt == 0,
            )
            return _clean_prose(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"describe_scene failed: {last_err}")


async def verify_description(
    llm: Gemma, frames: list[Keyframe], draft: str,
) -> str:
    images = [f.b64() for f in frames]
    prompt = _safe_format(_VERIFY_PROMPT, n=len(images), draft=draft.strip())
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = await llm.vision_chat(
                prompt, images, temperature=0.05, max_tokens=400,
                tag="verify", reasoning="none", cache=attempt == 0,
            )
            return _clean_prose(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"verify_description failed: {last_err}")


_LENGTH_HINT = {
    "formal": "1-2 dense professional sentences (about 20-45 words).",
    "sarcastic": "ONE dry sentence, punchy (about 12-28 words).",
    "humorous_tech": "ONE tech-joke sentence that still names something visible (about 15-35 words).",
    "humorous_non_tech": "ONE everyday-humor sentence grounded in the scene (about 12-30 words).",
}


async def write_styled_caption(
    llm: Gemma,
    description: str,
    style: StyleConfig,
    prior_captions: list[str],
    feedback: str | None = None,
) -> str:
    variety = ""
    if prior_captions:
        variety = (
            "\n\nCaptions already written in other styles (use a DIFFERENT sentence "
            "structure and angle): " + " | ".join(prior_captions)
        )
    fb = f"\n\nRewrite feedback: {feedback}" if feedback else ""
    tech_extra = ""
    if style.key == "humorous_tech":
        tech_extra = (
            "\nUse exactly ONE tech metaphor mapped to ONE visible action or object. "
            "Do not invent plot events.\n"
        )
    exemplars = "\n".join(f"- {cap}" for _, cap in style.exemplars[:2])
    length = _LENGTH_HINT.get(style.key, "1-2 sentences.")
    prompt = (
        f"Style: {style.label}\n{style.description}\n\n"
        f"Rules:\n" + "\n".join(f"- {r}" for r in style.rules) + "\n\n"
        f"Never:\n" + "\n".join(f"- {a}" for a in style.anti_patterns) + "\n\n"
        f"Example captions in this style (tone only — do not copy content):\n{exemplars}\n\n"
        f"Factual description of the video clip:\n{description}\n\n"
        f"{tech_extra}"
        f"Write ONE caption. Length target: {length} "
        f"Every concrete detail must come from the description — no invented city names, "
        f"brands, company names, signage, or events. If the description does not name a "
        f"brand or city, do not invent one. Write as if you personally watched the clip. "
        f"Never mention computer vision, models, detection, frames, or uncertainty. "
        f"Output ONLY the caption text."
        f"{variety}{fb}"
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = await llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=style.temperature,
                max_tokens=160,
                tag=f"write-{style.key}",
                reasoning="none",
                cache=attempt == 0,
            )
            return _clean_caption(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"write_styled_caption({style.key}) failed: {last_err}")


async def caption_all_styles(
    llm: Gemma,
    description: str,
    styles: list[str] | None = None,
) -> dict[str, str]:
    """Write requested styles sequentially from one verified description."""
    from .io_contract import normalize_captions
    from .styles import STYLES

    order = [k for k in (styles or list(STYLES)) if k in STYLES]
    if not order:
        order = list(STYLES.keys())

    captions: dict[str, str] = {}
    prior: list[str] = []
    fallback = description.split(".")[0].strip() + "." if description else ""
    for key in order:
        try:
            cap = await write_styled_caption(llm, description, STYLES[key], prior)
        except Exception:  # noqa: BLE001
            cap = fallback if key == "formal" else (prior[-1] if prior else fallback)
        captions[key] = cap
        prior.append(cap)
    return normalize_captions(captions, order)
