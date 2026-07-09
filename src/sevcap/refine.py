"""Self-Refine loop (Madaan et al., NeurIPS 2023) over both gates.

Generate -> gate -> feed the judge's actionable feedback back into the
generator -> retry, hard-capped at settings.max_refine_rounds. We always keep
the best attempt seen so far, so a clip can degrade gracefully but never fail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import settings
from .entropy import FactSheet, check_grounding
from .fireworks import Gemma
from .gates import blind_lineup
from .generator import generate_caption
from .styles import STYLES

log = logging.getLogger("sevcap.refine")


@dataclass
class CaptionAttempt:
    text: str
    grounded: bool = False
    unsupported_claims: list[str] = field(default_factory=list)
    lineup_passed: bool = False
    lineup_judged_as: str = ""
    lineup_confidence: int = 0

    def score(self) -> tuple[int, int, int]:
        """Grounding dominates (accuracy axis), then lineup, then confidence."""
        return (int(self.grounded), int(self.lineup_passed), self.lineup_confidence)


@dataclass
class StyleOutcome:
    style: str
    final: CaptionAttempt
    attempts: int
    history: list[dict] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "style": self.style,
            "attempts": self.attempts,
            "grounded": self.final.grounded,
            "unsupported_claims": self.final.unsupported_claims,
            "lineup_identified_as": self.final.lineup_judged_as,
            "lineup_confidence": self.final.lineup_confidence,
            "lineup_passed": self.final.lineup_passed,
            "history": self.history,
        }


async def _assess(
    llm: Gemma, fact_sheet: FactSheet, captions: dict[str, str], round_id: int
) -> dict[str, CaptionAttempt]:
    """Run Gate A per caption and Gate B once over the full set."""
    lineup = await blind_lineup(
        llm, captions, min_confidence=settings.lineup_min_confidence, rng_seed=round_id
    )
    out: dict[str, CaptionAttempt] = {}
    for style_key, text in captions.items():
        grounded, unsupported = await check_grounding(llm, fact_sheet, text)
        lr = lineup[style_key]
        out[style_key] = CaptionAttempt(
            text=text,
            grounded=grounded,
            unsupported_claims=unsupported,
            lineup_passed=lr.passed,
            lineup_judged_as=lr.judged_as,
            lineup_confidence=lr.confidence,
        )
    return out


def _feedback(attempt: CaptionAttempt, style_key: str) -> str:
    parts = []
    if not attempt.grounded:
        claims = "; ".join(attempt.unsupported_claims) or "unverifiable details"
        parts.append(
            f"Remove or replace claims not on the fact sheet: {claims}. "
            "Only use facts from the sheet."
        )
    if not attempt.lineup_passed:
        style = STYLES[style_key]
        if attempt.lineup_judged_as != style_key:
            parts.append(
                f"A blind judge read it as '{attempt.lineup_judged_as}', not "
                f"'{style_key}'. Make the {style.label} style unmistakable: "
                + "; ".join(style.rules)
            )
        else:
            parts.append(
                f"Style was recognizable but weak (confidence "
                f"{attempt.lineup_confidence}/5). Push the {style.label} style harder."
            )
    return " ".join(parts)


async def refine_captions(
    llm: Gemma,
    fact_sheet: FactSheet,
    initial: dict[str, str],
    images_b64: list[str] | None = None,
) -> dict[str, StyleOutcome]:
    """Gate + Self-Refine the 4-caption set; return best attempt per style."""
    captions = dict(initial)
    best: dict[str, CaptionAttempt] = {}
    history: dict[str, list[dict]] = {k: [] for k in captions}
    attempts_used: dict[str, int] = {k: 1 for k in captions}

    for round_id in range(settings.max_refine_rounds + 1):
        assessed = await _assess(llm, fact_sheet, captions, round_id)
        retry_styles: list[str] = []
        for style_key, attempt in assessed.items():
            history[style_key].append(
                {"round": round_id, "text": attempt.text,
                 "grounded": attempt.grounded, "lineup_passed": attempt.lineup_passed,
                 "judged_as": attempt.lineup_judged_as,
                 "confidence": attempt.lineup_confidence}
            )
            if style_key not in best or attempt.score() > best[style_key].score():
                best[style_key] = attempt
            if not (attempt.grounded and attempt.lineup_passed):
                retry_styles.append(style_key)

        if not retry_styles or round_id == settings.max_refine_rounds:
            break
        log.info("refine round %d: retrying %s", round_id + 1, retry_styles)
        for style_key in retry_styles:
            fb = _feedback(assessed[style_key], style_key)
            try:
                captions[style_key] = await generate_caption(
                    llm, fact_sheet, STYLES[style_key], feedback=fb,
                    seed=2000 + round_id * 10,
                    images_b64=images_b64,
                )
                attempts_used[style_key] += 1
            except Exception as e:  # noqa: BLE001
                log.warning("refine generation failed for %s: %s", style_key, e)

    return {
        k: StyleOutcome(style=k, final=best[k], attempts=attempts_used[k], history=history[k])
        for k in captions
    }
