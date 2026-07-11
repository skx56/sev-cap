"""Grounded captioning: describe frames → self-verify → write styled captions.

Matches the high-scoring Raccoon-Vision-Translator pattern (describe, verify,
then text-only style writes from one shared grounding description) while
keeping our style definitions and duration-scaled frame sampling.
"""

from __future__ import annotations

import asyncio
import re

from .config import ClipProfile
from .fireworks import Gemma
from .sampler import Keyframe
from .styles import STYLE_ORDER, StyleConfig

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
Describe exactly what is visible: the setting/location, main subjects (people, animals, objects), their colors and actions, and any readable on-screen text.
Be concrete and specific — name what you see, not what you guess. No speculation beyond the frames{audio_clause}.
Write 2-4 factual sentences. Output ONLY the description text, no preamble or labels."""

_VERIFY_PROMPT = """These {n} images are frames from the same video clip.

Draft description:
"{draft}"

Check the draft against the actual frames. If accurate and specific, repeat it unchanged.
If anything is wrong, invented, or too generic, correct it.
Output ONLY the final description — no preamble."""


async def describe_scene(
    llm: Gemma,
    frames: list[Keyframe],
    profile: ClipProfile,
    transcript: str = "",
    transcript_trusted: bool = False,
) -> str:
    audio_block = ""
    audio_clause = ""
    if transcript and transcript_trusted:
        audio_block = f'\nTrusted dialogue transcript: "{transcript}"\n'
        audio_clause = " or clearly spoken in the transcript"
    long_hint = ""
    if profile.long_form:
        long_hint = (
            "\nThis is a longer clip — summarize the MAIN story arc (who, what happens, "
            "outcome) in 2-4 sentences. Omit minor background props.\n"
        )
    images = [f.b64() for f in frames]
    prompt = _safe_format(
        _DESCRIBE_PROMPT,
        n=len(images),
        duration=int(profile.duration_s),
        audio_block=audio_block,
        long_hint=long_hint,
        audio_clause=audio_clause,
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = await llm.vision_chat(
                prompt, images, temperature=0.3, max_tokens=400,
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
                prompt, images, temperature=0.1, max_tokens=350,
                tag="verify", reasoning="none", cache=attempt == 0,
            )
            return _clean_prose(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"verify_description failed: {last_err}")


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
    prompt = (
        f"Style: {style.label}\n{style.description}\n\n"
        f"Rules:\n" + "\n".join(f"- {r}" for r in style.rules) + "\n\n"
        f"Never:\n" + "\n".join(f"- {a}" for a in style.anti_patterns) + "\n\n"
        f"Example captions in this style (tone only — do not copy content):\n{exemplars}\n\n"
        f"Factual description of the video clip:\n{description}\n\n"
        f"{tech_extra}"
        f"Write ONE caption in 1-2 sentences (about 15-45 words). "
        f"Ground every claim in the description above. "
        f"Write as if you personally watched the clip — never mention frames, models, or uncertainty. "
        f"Output ONLY the caption text."
        f"{variety}{fb}"
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = await llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=style.temperature,
                max_tokens=180,
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
    from .io_contract import REQUIRED_STYLES, normalize_captions
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
