"""Stage 2: style-conditioned caption generation from verified facts ONLY.

The generator never sees the video or frames — only the semantic-entropy
verified fact sheet. That is the hallucination firewall: a detail that did
not survive verification cannot appear in a caption.
"""

from __future__ import annotations

from .entropy import FactSheet
from .fireworks import Gemma
from .styles import StyleConfig

SYSTEM = (
    "You are an award-winning caption writer. You write captions for short "
    "video clips based STRICTLY on a verified fact sheet. You never invent "
    "objects, actions, places, names, numbers or events that are not on the "
    "sheet. Style is yours; facts are not."
)

GEN_PROMPT = """Style: {label}
{description}

Style rules:
{rules}

Never do this (instant failure):
{anti}

Here are examples of PERFECT captions in this style:

{exemplars}

Now write ONE caption in the {label} style for the following video.

VERIFIED FACT SHEET:
{facts}
{feedback}
Think if you need to, but END your response with a single line of the form:
CAPTION: <the caption text>"""


def _parse_caption(raw: str) -> str:
    """Take the last CAPTION: line; fall back to the last *complete* line."""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.upper().startswith("CAPTION:"):
            return ln[len("CAPTION:"):].strip().strip('"').strip()
    # No CAPTION marker: prefer the last line that ends like a sentence
    # (a truncated generation leaves a dangling fragment we must not ship).
    for ln in reversed(lines):
        if ln.endswith((".", "!", "?", '"', "”")):
            return ln.strip('"').strip()
    return (lines[-1] if lines else raw).strip().strip('"').strip()


def _format_exemplars(style: StyleConfig) -> str:
    blocks = []
    for i, (facts, caption) in enumerate(style.exemplars, 1):
        blocks.append(f"Example {i}\nFACT SHEET:\n{facts}\nCAPTION: {caption}")
    return "\n\n".join(blocks)


async def generate_caption(
    llm: Gemma,
    fact_sheet: FactSheet,
    style: StyleConfig,
    feedback: str | None = None,
    seed: int | None = None,
) -> str:
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\nYour previous attempt failed review. Feedback to fix in this "
            f"rewrite:\n{feedback}\n"
        )
    prompt = GEN_PROMPT.format(
        label=style.label,
        description=style.description,
        rules="\n".join(f"- {r}" for r in style.rules),
        anti="\n".join(f"- {a}" for a in style.anti_patterns),
        exemplars=_format_exemplars(style),
        facts=fact_sheet.as_text(),
        feedback=feedback_block,
    )
    raw = await llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=style.temperature,
        max_tokens=2500,
        seed=seed,
        tag=f"gen-{style.key}",
        reasoning="none",
    )
    return _parse_caption(raw)


DRAFT_PROMPT = """These {n} images are keyframes (in order) from one short video clip.

Write exactly four captions for the clip, one per style:
1. formal — precise, neutral, professional; no humor, no contractions.
2. sarcastic — dry deadpan irony with a clear target; no exclamation marks.
3. humorous_tech — a joke built on software/tech culture that maps onto the scene.
4. humorous_non_tech — everyday observational humor, zero technical vocabulary.

Only describe what is visibly in the frames. Think briefly if needed, then END
your response with one line containing ONLY the JSON:
{{"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."}}"""


async def generate_draft(llm: Gemma, images_b64: list[str]) -> dict[str, str]:
    """Fast single-pass draft of all 4 styles straight from frames.

    This is the anytime-algorithm safety net: it is written to disk first so a
    harness timeout can never leave a clip without output.
    """
    from .fireworks import extract_json

    raw = await llm.vision_chat(
        DRAFT_PROMPT.format(n=len(images_b64)), images_b64,
        temperature=0.7, max_tokens=3000, tag="draft",
    )
    data = extract_json(raw)
    out = {}
    for key in ("formal", "sarcastic", "humorous_tech", "humorous_non_tech"):
        val = data.get(key, "") if isinstance(data, dict) else ""
        out[key] = str(val).strip() or "A short video clip."
    return out
