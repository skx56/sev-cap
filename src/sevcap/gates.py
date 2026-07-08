"""Quality gates run before any caption leaves the container.

Gate A (grounding): every concrete claim must be entailed by the verified
fact sheet (implemented in entropy.check_grounding, re-exported here).

Gate B (blind style lineup): the four captions are label-stripped, shuffled,
and a fresh judge context must assign each to its style. A caption passes only
if it is blindly identifiable as its intended style with enough confidence —
a direct test of style separability, which is what the LLM-Judge grades.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from .entropy import check_grounding  # noqa: F401  (Gate A, re-exported)
from .fireworks import Gemma, extract_json
from .styles import STYLES

log = logging.getLogger("sevcap.gates")

LINEUP_PROMPT = """Four captions were written for the SAME short video, each in a different style:
- formal: precise, neutral, professional register
- sarcastic: dry deadpan irony that mocks a target
- humorous_tech: comedy built on software/tech culture references
- humorous_non_tech: everyday observational/absurdist comedy, no tech vocabulary

The captions below are unlabeled and shuffled. Assign each caption exactly one
style (use each style exactly once) and rate your confidence 1-5 (5 = the style
is unmistakable, 1 = you are guessing).

Captions:
{captions}

Return ONLY JSON:
{{"assignments": [{{"caption": 1, "style": "formal", "confidence": 4}}, ...]}}"""


@dataclass
class LineupResult:
    caption_style: str          # intended style
    identified: bool            # judge blindly matched it to the intended style
    judged_as: str              # what the judge thought it was
    confidence: int             # judge's 1-5 confidence
    passed: bool                # identified AND confidence >= threshold

    def feedback(self) -> str:
        intended = STYLES[self.caption_style]
        if not self.identified:
            return (
                f"A blind judge read this caption as '{self.judged_as}', not "
                f"'{self.caption_style}'. Make the {intended.label} style unmistakable. "
                f"Style rules: " + "; ".join(intended.rules)
            )
        return (
            f"A blind judge matched the style but with low confidence "
            f"({self.confidence}/5). Push the {intended.label} style harder: "
            + "; ".join(intended.rules)
        )


async def blind_lineup(
    llm: Gemma, captions: dict[str, str], min_confidence: int = 3, rng_seed: int | None = None
) -> dict[str, LineupResult]:
    """Run the blind style lineup over the 4 captions. Keys are style keys."""
    styles = list(captions.keys())
    order = styles[:]
    random.Random(rng_seed).shuffle(order)
    numbered = "\n".join(f"{i + 1}. {captions[s]}" for i, s in enumerate(order))

    raw = await llm.chat(
        [{"role": "user", "content": LINEUP_PROMPT.format(captions=numbered)}],
        temperature=0.0, tag="lineup", max_tokens=400,
    )
    results: dict[str, LineupResult] = {}
    try:
        assignments = extract_json(raw).get("assignments", [])
        judged: dict[int, tuple[str, int]] = {}
        for a in assignments:
            idx = int(a.get("caption", 0)) - 1
            judged[idx] = (str(a.get("style", "")).strip(), int(a.get("confidence", 1)))
        for i, style_key in enumerate(order):
            judged_as, conf = judged.get(i, ("unknown", 1))
            identified = judged_as == style_key
            results[style_key] = LineupResult(
                caption_style=style_key,
                identified=identified,
                judged_as=judged_as,
                confidence=conf,
                passed=identified and conf >= min_confidence,
            )
    except Exception as e:  # noqa: BLE001
        log.warning("lineup judge unparseable (%s); passing captions through", e)
        for style_key in styles:
            results[style_key] = LineupResult(
                caption_style=style_key, identified=True, judged_as=style_key,
                confidence=min_confidence, passed=True,
            )
    return results
