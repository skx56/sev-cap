"""Grounded captioning: describe frames → self-verify → write styled captions.

Accuracy-first for Track 2: styled captions may ONLY rephrase facts from the
verified description. Jokes change tone, never invent objects, motion, brands,
or outcomes that are not in the description.
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
- notable background details only if clearly visible
STRICT:
- Do NOT invent city names, brands, company logos, product models, or dialogue.
- Prefer generic labels when uncertain ("desktop computer" not "iMac", "city intersection" not "Shibuya").
- Prefer "golden ginkgo trees" / "orange tabby kitten" style specificity when clearly visible.
Write 2-4 sentences. Output ONLY the description text."""

_VERIFY_PROMPT = """These {n} images are frames from the same video clip.

Draft description:
"{draft}"

Check the draft against the actual frames.
- Keep clearly visible, specific details.
- Remove invented or uncertain brands, cities, product models, and plot.
- Do not add speculative details.
Output ONLY the final description."""


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
            "\nLonger clip — summarize the MAIN arc in 2-4 sentences; omit minor props.\n"
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
                prompt, images, temperature=0.15, max_tokens=450,
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
                prompt, images, temperature=0.0, max_tokens=400,
                tag="verify", reasoning="none", cache=attempt == 0,
            )
            return _clean_prose(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"verify_description failed: {last_err}")


_STYLE_WRITE_HINTS = {
    "formal": (
        "Write a clear professional caption that restates the description. "
        "1-2 dense sentences. No jokes, no opinions."
    ),
    "humorous_tech": (
        "Write ONE short tech-humor sentence (15-32 words). "
        "Joke about the PRIMARY visible action/subject (usually the first clause of the description). "
        "Map that one thing to ONE tech metaphor (deploy, retry, bug, agent, staging). "
        "Ignore background props unless they are the main subject. "
        "Must read as a joke. Do not invent extra objects or failure plots. "
        "Never say frame/frames."
    ),
    "humorous_non_tech": (
        "Write ONE short everyday-humor sentence (12-30 words). "
        "Personify or understate the PRIMARY subject/action from the description. "
        "No tech jargon. Must be funny. No invented interactions. Never say frame/frames."
    ),
    "sarcastic": (
        "Write ONE short dry sarcastic sentence (12-28 words). "
        "Needle the PRIMARY action/subject with ironic praise or understatement. "
        "MUST sound sarcastic when read alone — not a formal paraphrase. "
        "Do not invent new events. Never say frame/frames."
    ),
}


def _word_count(text: str) -> int:
    return len([w for w in text.replace("—", " ").split() if w.strip()])


def _too_similar(a: str, b: str) -> bool:
    """True if caption collapsed into a near-copy of formal."""
    if not a or not b:
        return False
    ta = {w.lower().strip(".,;:!?\"'") for w in a.split() if len(w) > 3}
    tb = {w.lower().strip(".,;:!?\"'") for w in b.split() if len(w) > 3}
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / max(1, len(ta))
    return overlap >= 0.72 or a.strip()[:80].lower() == b.strip()[:80].lower()


_FOREIGN_BLOCK = {
    "seoul", "tokyo", "shibuya", "nyc", "london", "paris",
    "chrome", "firefox", "safari", "windows", "macos", "android", "iphone",
    "ram", "cpu", "gpu", "5g", "wifi", "bluetooth", "usb",
    "aws", "azure", "kubernetes", "docker", "stackoverflow",
}
_META_WORDS = ("frame", "frames", "keyframe", "keyframes", "timestamp")


def _invents_foreign(caption: str, description: str) -> bool:
    """Reject jokes that drag in cities/products absent from the description."""
    desc = description.lower()
    low = caption.lower()
    for tok in _FOREIGN_BLOCK:
        if tok in low and tok not in desc:
            return True
    for tok in _META_WORDS:
        if re.search(rf"\b{tok}\b", low):
            return True
    return False


def _style_ok(style_key: str, caption: str, formal: str | None, description: str = "") -> bool:
    n = _word_count(caption)
    if style_key == "formal":
        return 12 <= n <= 70
    if n > 36 or n < 8:
        return False
    if formal and _too_similar(caption, formal):
        return False
    if description and _invents_foreign(caption, description):
        return False
    return True


async def write_styled_caption(
    llm: Gemma,
    description: str,
    style: StyleConfig,
    prior_captions: list[str],
    feedback: str | None = None,
    formal_anchor: str | None = None,
) -> str:
    variety = ""
    if prior_captions:
        variety = (
            "\nOther styles already written (different voice, same facts): "
            + " | ".join([c for c in prior_captions if _word_count(c) <= 40][:2])
        )
    fb = f"\nRewrite feedback: {feedback}" if feedback else ""
    anchor = ""
    if formal_anchor and style.key != "formal":
        anchor = (
            f"\nFact checklist (do NOT copy this wording; only reuse its facts):\n"
            f"{formal_anchor}\n"
            f"Your caption must be SHORTER and clearly {style.label}, not formal.\n"
        )
    exemplars = "\n".join(f"- {cap}" for _, cap in style.exemplars[:2])
    hint = _STYLE_WRITE_HINTS.get(style.key, "Write one caption.")
    prompt = (
        f"Style: {style.label}\n{style.description}\n\n"
        f"Rules:\n" + "\n".join(f"- {r}" for r in style.rules) + "\n\n"
        f"Never:\n" + "\n".join(f"- {a}" for a in style.anti_patterns) + "\n\n"
        f"Tone examples (do not copy content):\n{exemplars}\n\n"
        f"Verified description (ONLY allowed facts):\n{description}\n"
        f"{anchor}\n"
        f"{hint}\n"
        f"HARD CONSTRAINTS:\n"
        f"- Every concrete noun/action must come from the description.\n"
        f"- Do NOT name cities, brands, apps, or gadgets absent from the description "
        f"(no Seoul/Tokyo/Chrome/RAM/5G unless those words appear above).\n"
        f"- The joke's subject must be something visible in the description.\n"
        f"- Change voice only; do not invent plot.\n"
        f"- Never mention frames/models/uncertainty.\n"
        f"- Output ONLY the caption text."
        f"{variety}{fb}"
    )
    last_err: Exception | None = None
    last_cap = ""
    for attempt in range(4):
        try:
            raw = await llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=style.temperature if attempt == 0 else min(0.85, style.temperature + 0.12 * attempt),
                max_tokens=120 if style.key != "formal" else 180,
                tag=f"write-{style.key}",
                reasoning="none",
                cache=attempt == 0,
            )
            cap = _clean_caption(raw)
            last_cap = cap
            if _style_ok(style.key, cap, formal_anchor if style.key != "formal" else None, description):
                return cap
            reason = "bad length/style"
            if formal_anchor and _too_similar(cap, formal_anchor):
                reason = "too similar to formal"
            elif _invents_foreign(cap, description):
                reason = "invented city/product not in description"
            prompt = prompt + (
                f"\n\nPrevious attempt rejected ({reason}): {cap}\n"
                f"Write a NEW short {style.label} caption using ONLY description facts."
            )
            last_err = ValueError(f"style quality reject: {cap[:60]!r}")
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(1.2 * (attempt + 1))
    if last_cap:
        return last_cap
    raise RuntimeError(f"write_styled_caption({style.key}) failed: {last_err}")


async def safe_style_caption(
    llm: Gemma,
    description: str,
    style: StyleConfig,
    formal_anchor: str,
) -> str:
    """Conservative short rewrite when polish still fails accuracy — keeps style."""
    prompt = (
        f"Write a SHORT {style.label} caption about this scene.\n\n"
        f"Facts only:\n{description}\n\n"
        f"Formal reference (facts only — DO NOT copy its sentence):\n{formal_anchor}\n\n"
        f"Requirements:\n"
        f"- One sentence, under 30 words.\n"
        f"- Clearly {style.label} tone (not formal).\n"
        f"- Subject must be a visible thing from the facts "
        f"(trees/kitten/waves/knife/runner/etc — not Chrome/RAM/Seoul).\n"
        f"- No new objects/events/brands/cities.\n"
        f"- Output ONLY the caption."
    )
    last = ""
    for attempt in range(3):
        raw = await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.55 + 0.1 * attempt,
            max_tokens=100,
            tag=f"safe-{style.key}",
            reasoning="none",
            cache=False,
        )
        last = _clean_caption(raw)
        if _style_ok(style.key, last, formal_anchor, description):
            return last
        prompt += (
            "\nPrevious attempt failed (copied formal, too long, or invented "
            "outside cities/products). Make it shorter and grounded."
        )
    return last


async def caption_all_styles(
    llm: Gemma,
    description: str,
    styles: list[str] | None = None,
) -> dict[str, str]:
    """Write formal first, then other styles locked to that fact checklist."""
    from .io_contract import normalize_captions
    from .styles import STYLES

    order = [k for k in (styles or list(STYLES)) if k in STYLES]
    if not order:
        order = list(STYLES.keys())
    if "formal" in order:
        order = ["formal"] + [k for k in order if k != "formal"]

    captions: dict[str, str] = {}
    prior: list[str] = []
    fallback = description.split(".")[0].strip() + "." if description else ""
    formal_anchor = ""
    for key in order:
        try:
            cap = await write_styled_caption(
                llm, description, STYLES[key], prior,
                formal_anchor=formal_anchor or None,
            )
        except Exception:  # noqa: BLE001
            if key == "formal":
                cap = fallback
            else:
                try:
                    cap = await safe_style_caption(
                        llm, description, STYLES[key], formal_anchor or fallback,
                    )
                except Exception:  # noqa: BLE001
                    cap = formal_anchor or fallback
        if key != "formal" and formal_anchor and (
            _too_similar(cap, formal_anchor) or _invents_foreign(cap, description)
        ):
            try:
                cap = await safe_style_caption(llm, description, STYLES[key], formal_anchor)
            except Exception:  # noqa: BLE001
                pass
        captions[key] = cap
        if key == "formal":
            formal_anchor = cap
        prior.append(cap)
    return normalize_captions(captions, order)
