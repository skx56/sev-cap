"""Grounded captioning: describe → verify → multi-candidate styled writes.

Goal: robust average leaderboard score ≥ 0.95 on sample-like clips.
We generate several caption candidates per style and pick with a vision judge
on BOTH accuracy and tone (same axes as Track 2 scoring).
"""

from __future__ import annotations

import asyncio
import os
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

# How many candidates to draft per style before vision selection.
# 2 keeps quality high while staying inside harness time budgets.
N_CANDIDATES = int(os.environ.get("SEVCAP_CANDIDATES", "2"))


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
Write a SHORT factual description (2 sentences max) of what is clearly visible:
1) main subject(s) with concrete attributes (color, type, clothing)
2) setting + the main action/motion
STRICT:
- No city names, brands, product models, or dialogue unless unmistakably readable.
- Prefer generic labels when unsure ("desktop computer", "city intersection").
- Do NOT mention the camera, filming, aerial footage, zoom, pull-back, fade, or the clip ending.
- Prefer specificity when clear ("golden ginkgo trees", "orange tabby kitten").
Output ONLY the description text."""

_VERIFY_PROMPT = """These {n} images are frames from the same video clip.

Draft description:
"{draft}"

Correct against the frames:
- Keep only clearly visible facts.
- Remove brands/cities/product models/plot that are uncertain.
- Remove ALL camera-meta (aerial, zoom, pull-back, fade, blur, "the clip", exits the frame).
- Keep it to 2 tight sentences about subjects and setting only.
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
        long_hint = "\nLonger clip — cover the main arc only; omit minor props.\n"
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
                prompt, images, temperature=0.1, max_tokens=280,
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
                prompt, images, temperature=0.0, max_tokens=280,
                tag="verify", reasoning="none", cache=attempt == 0,
            )
            return _clean_prose(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"verify_description failed: {last_err}")


_STYLE_WRITE_HINTS = {
    "formal": (
        "Write a professional caption: 1-2 sentences, about 18-45 words. "
        "Restate the description precisely with concrete visible details. "
        "No jokes, opinions, or camera talk."
    ),
    "sarcastic": (
        "Write ONE dry sarcastic sentence (12-26 words). "
        "Ironic praise or understatement about the PRIMARY subject/action. "
        "Reuse concrete nouns from the description (colors, clothing, setting). "
        "Must sound sarcastic alone — not a formal paraphrase. No invented events."
    ),
    "humorous_tech": (
        "Write ONE tech-humor sentence (14-28 words). "
        "Reuse concrete visible nouns from the description, then map the PRIMARY "
        "visible state/action to ONE tech metaphor (deploy, retry, bug, agent, staging). "
        "The metaphor must NOT invent physical motion (rolling, crashing, collapsing) "
        "that is not in the description. "
        "No frame/camera talk."
    ),
    "humorous_non_tech": (
        "Write ONE everyday-humor sentence (12-26 words). "
        "Reuse concrete visible nouns from the description. "
        "Light personification or understatement of what is actually visible. "
        "Do NOT invent clothing, body jokes, arguments, or secret plans. "
        "No tech jargon."
    ),
}

_VISION_STYLE_PROMPT = """These {n} images are keyframes from the video.

Verified description (facts you may use):
{description}

Write ONE {label} caption for this video.
{hint}

HARD:
- Every concrete noun/action must be visible in the frames or description.
- No invented plot, brands, cities, or camera-meta.
- Output ONLY the caption text."""


async def vision_style_caption(
    llm: Gemma,
    frames: list[Keyframe],
    description: str,
    style: StyleConfig,
    formal_anchor: str | None = None,
) -> str:
    """Write a style caption while looking at frames — higher accuracy floor."""
    kf = frames
    if len(kf) > 4:
        step = len(kf) / 4
        kf = [kf[int(i * step)] for i in range(4)]
    images = [f.b64() for f in kf]
    hint = _STYLE_WRITE_HINTS.get(style.key, "Write one caption.")
    if formal_anchor and style.key != "formal":
        hint += f"\nFact checklist (reuse facts, not wording): {formal_anchor}"
    prompt = _safe_format(
        _VISION_STYLE_PROMPT,
        n=len(images),
        description=description.strip(),
        label=style.label,
        hint=hint,
    )
    last = ""
    for attempt in range(2):
        raw = await llm.vision_chat(
            prompt, images,
            temperature=0.35 + 0.15 * attempt,
            max_tokens=140 if style.key != "formal" else 200,
            tag=f"vision-write-{style.key}",
            reasoning="none",
            cache=False,
        )
        last = _clean_caption(raw)
        if _style_ok(style.key, last, formal_anchor if style.key != "formal" else None, description):
            return last
    return last


def _word_count(text: str) -> int:
    return len([w for w in text.replace("—", " ").split() if w.strip()])


def _too_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    ta = {w.lower().strip(".,;:!?\"'") for w in a.split() if len(w) > 3}
    tb = {w.lower().strip(".,;:!?\"'") for w in b.split() if len(w) > 3}
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / max(1, len(ta))
    return overlap >= 0.70 or a.strip()[:70].lower() == b.strip()[:70].lower()


_FOREIGN_BLOCK = {
    "seoul", "tokyo", "shibuya", "nyc", "london", "paris",
    "chrome", "firefox", "safari", "windows", "macos", "android", "iphone",
    "ram", "cpu", "gpu", "5g", "wifi", "bluetooth", "usb",
    "aws", "azure", "kubernetes", "docker", "stackoverflow",
    "/dev/null", "segfault", "zombie process",
}
_META_WORDS = (
    "frame", "frames", "keyframe", "keyframes", "timestamp",
    "motion blur", "long-exposure", "exits the frame", "camera",
    "aerial footage", "pulls back", "pull-back", "fades to black", "fade to black",
    "the clip", "zooms", "zoom out", "zoom in",
)
_INVENTED_MOTION = (
    "rolling back", "rolled back", "collapsing", "crashing", "exploding",
    "falling apart", "breaking apart", "arguing", "staring contest",
)


def _invents_foreign(caption: str, description: str) -> bool:
    desc = description.lower()
    low = caption.lower()
    for tok in _FOREIGN_BLOCK:
        if tok in low and tok not in desc:
            return True
    for tok in _META_WORDS:
        if tok in low and tok not in desc:
            return True
    for tok in _INVENTED_MOTION:
        if tok in low and tok not in desc:
            return True
    return False


def _content_overlap(caption: str, description: str) -> float:
    stop = {
        "the", "a", "an", "and", "or", "with", "from", "into", "onto", "that",
        "this", "its", "their", "then", "than", "while", "over", "under",
    }
    def toks(s: str) -> set[str]:
        return {
            w.lower().strip(".,;:!?\"'")
            for w in s.replace("—", " ").split()
            if len(w) > 3 and w.lower().strip(".,;:!?\"'") not in stop
        }
    ta, tb = toks(caption), toks(description)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def _style_ok(style_key: str, caption: str, formal: str | None, description: str = "") -> bool:
    n = _word_count(caption)
    if style_key == "formal":
        if not (12 <= n <= 55):
            return False
        if description and _invents_foreign(caption, description):
            return False
        return True
    if n > 34 or n < 8:
        return False
    if formal and _too_similar(caption, formal):
        return False
    if description and _invents_foreign(caption, description):
        return False
    # Humor must reuse enough visible nouns from the description (accuracy floor).
    if style_key.startswith("humorous") and description and _content_overlap(caption, description) < 0.28:
        return False
    return True


async def _draft_one(
    llm: Gemma,
    description: str,
    style: StyleConfig,
    prior_captions: list[str],
    formal_anchor: str | None,
    temperature: float,
    feedback: str | None = None,
) -> str:
    variety = ""
    if prior_captions:
        short = [c for c in prior_captions if _word_count(c) <= 40][:2]
        if short:
            variety = "\nOther styles already written (different voice, same facts): " + " | ".join(short)
    fb = f"\nRewrite feedback: {feedback}" if feedback else ""
    anchor = ""
    if formal_anchor and style.key != "formal":
        anchor = (
            f"\nFact checklist (do NOT copy wording; reuse facts only):\n{formal_anchor}\n"
            f"Must be clearly {style.label} and shorter than formal.\n"
        )
    exemplars = "\n".join(f"- {cap}" for _, cap in style.exemplars[:2])
    hint = _STYLE_WRITE_HINTS.get(style.key, "Write one caption.")
    prompt = (
        f"Style: {style.label}\n{style.description}\n\n"
        f"Rules:\n" + "\n".join(f"- {r}" for r in style.rules) + "\n\n"
        f"Never:\n" + "\n".join(f"- {a}" for a in style.anti_patterns) + "\n\n"
        f"Tone examples (do not copy content):\n{exemplars}\n\n"
        f"Verified description (ONLY allowed facts):\n{description}\n"
        f"{anchor}\n{hint}\n"
        f"HARD CONSTRAINTS:\n"
        f"- Concrete nouns/actions must come from the description.\n"
        f"- No cities/brands/apps/gadgets absent from the description.\n"
        f"- No camera-meta (frames, blur, exits the frame).\n"
        f"- Joke/subject must be the PRIMARY visible action/subject.\n"
        f"- Output ONLY the caption text."
        f"{variety}{fb}"
    )
    raw = await llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=110 if style.key != "formal" else 160,
        tag=f"write-{style.key}",
        reasoning="none",
        cache=False,
    )
    return _clean_caption(raw)


async def write_styled_caption(
    llm: Gemma,
    description: str,
    style: StyleConfig,
    prior_captions: list[str],
    feedback: str | None = None,
    formal_anchor: str | None = None,
) -> str:
    """Single draft (kept for polish/fallback callers)."""
    temps = [style.temperature, min(0.9, style.temperature + 0.15), max(0.2, style.temperature - 0.15)]
    last = ""
    for i, temp in enumerate(temps):
        try:
            cap = await _draft_one(
                llm, description, style, prior_captions, formal_anchor, temp, feedback,
            )
            last = cap
            if _style_ok(style.key, cap, formal_anchor if style.key != "formal" else None, description):
                return cap
        except Exception:  # noqa: BLE001
            continue
    if last:
        return last
    raise RuntimeError(f"write_styled_caption({style.key}) failed")


async def safe_style_caption(
    llm: Gemma,
    description: str,
    style: StyleConfig,
    formal_anchor: str,
) -> str:
    prompt = (
        f"Write a SHORT {style.label} caption.\n\n"
        f"Facts:\n{description}\n\n"
        f"Formal reference (facts only — do not copy):\n{formal_anchor}\n\n"
        f"- One sentence, under 28 words.\n"
        f"- Clearly {style.label}.\n"
        f"- Subject = primary visible thing from the facts.\n"
        f"- No new objects/brands/cities/camera talk.\n"
        f"Output ONLY the caption."
    )
    last = ""
    for attempt in range(3):
        raw = await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.5 + 0.1 * attempt,
            max_tokens=90,
            tag=f"safe-{style.key}",
            reasoning="none",
            cache=False,
        )
        last = _clean_caption(raw)
        if _style_ok(style.key, last, formal_anchor, description):
            return last
        prompt += "\nPrevious attempt failed validation. Shorter, more styled, same facts."
    return last


async def draft_style_candidates(
    llm: Gemma,
    description: str,
    style: StyleConfig,
    prior_captions: list[str],
    formal_anchor: str | None,
    n: int | None = None,
    include_safe: bool = False,
) -> list[str]:
    """Draft several diverse candidates for vision selection."""
    n = n or (2 if style.key == "formal" else N_CANDIDATES)
    base = style.temperature
    temps = [max(0.15, base - 0.12), base, min(0.9, base + 0.18)][:n]
    while len(temps) < n:
        temps.append(min(0.9, base + 0.1 * len(temps)))

    async def one(temp: float) -> str | None:
        try:
            return await _draft_one(
                llm, description, style, prior_captions, formal_anchor, temp,
            )
        except Exception:  # noqa: BLE001
            return None

    raw = await asyncio.gather(*(one(t) for t in temps))
    out: list[str] = []
    seen: set[str] = set()
    for cap in raw:
        if not cap:
            continue
        key = cap.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(cap)
    if include_safe and style.key != "formal" and formal_anchor:
        try:
            safe = await safe_style_caption(llm, description, style, formal_anchor)
            key = safe.lower().strip()
            if key not in seen:
                out.append(safe)
        except Exception:  # noqa: BLE001
            pass
    return out


async def caption_all_styles(
    llm: Gemma,
    description: str,
    styles: list[str] | None = None,
    video: str | None = None,
    frames=None,
) -> dict[str, str]:
    """Write formal first, then multi-candidate select other styles."""
    from .io_contract import normalize_captions
    from .prejudge import pick_best_candidate
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
        style = STYLES[key]
        candidates = await draft_style_candidates(
            llm, description, style, prior, formal_anchor or None,
            include_safe=(key != "formal"),
        )
        if video and frames is not None:
            try:
                vcap = await vision_style_caption(
                    llm, frames, description, style, formal_anchor or None,
                )
                if vcap:
                    candidates.append(vcap)
            except Exception:  # noqa: BLE001
                pass
        if not candidates:
            candidates = [fallback if key == "formal" else (formal_anchor or fallback)]

        if video and frames is not None:
            def _valid(cap: str, _key=key, _formal=formal_anchor) -> bool:
                return _style_ok(_key, cap, _formal if _key != "formal" else None, description)

            best, best_acc, best_tone = await pick_best_candidate(
                llm, video, key, candidates, frames=frames, is_valid=_valid,
            )
            cap = best or candidates[0]
            # If still weak, force another vision write and re-pick.
            if best_acc < 4 or best_tone < 4:
                try:
                    extra = await vision_style_caption(
                        llm, frames, description, style, formal_anchor or None,
                    )
                    if extra and extra.strip().lower() != cap.strip().lower():
                        pool = [cap, extra]
                        if formal_anchor and key != "formal":
                            try:
                                pool.append(
                                    await safe_style_caption(
                                        llm, description, style, formal_anchor,
                                    )
                                )
                            except Exception:  # noqa: BLE001
                                pass
                        better, a2, t2 = await pick_best_candidate(
                            llm, video, key, pool, frames=frames, is_valid=_valid,
                        )
                        if better and (a2 > best_acc or (a2 == best_acc and t2 >= best_tone)):
                            cap, best_acc, best_tone = better, a2, t2
                except Exception:  # noqa: BLE001
                    pass
        else:
            # No vision available: prefer first valid candidate.
            cap = next(
                (c for c in candidates if _style_ok(key, c, formal_anchor if key != "formal" else None, description)),
                candidates[0],
            )

        if key != "formal" and formal_anchor and (
            _too_similar(cap, formal_anchor) or _invents_foreign(cap, description)
        ):
            try:
                cap = await safe_style_caption(llm, description, style, formal_anchor)
            except Exception:  # noqa: BLE001
                pass

        captions[key] = cap
        if key == "formal":
            formal_anchor = cap
        prior.append(cap)

    return normalize_captions(captions, order)
