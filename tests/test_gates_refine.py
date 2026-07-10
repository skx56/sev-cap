import json

import pytest

from sevcap.entropy import FactSheet, VerifiedFact
from sevcap.gates import blind_lineup
from sevcap.refine import refine_captions

CAPTIONS = {
    "formal": "A dog retrieves a ball in a park.",
    "sarcastic": "Truly elite ball retrieval, pond and all. Nailed it.",
    "humorous_tech": "Fetch request succeeded after one unhandled pond exception.",
    "humorous_non_tech": "He wanted the ball. The pond wanted him more.",
}


def lineup_response(order, styles_conf):
    """Build a lineup JSON given shuffled order and {style: (judged, conf)}."""
    assignments = []
    for i, style in enumerate(order):
        judged, conf = styles_conf[style]
        assignments.append({"caption": i + 1, "style": judged, "confidence": conf})
    return json.dumps({"assignments": assignments})


@pytest.mark.asyncio
async def test_blind_lineup_pass_and_fail(fake_llm):
    import random

    order = list(CAPTIONS.keys())
    random.Random(7).shuffle(order)
    fake_llm.queue(lineup_response(order, {
        "formal": ("formal", 5),
        "sarcastic": ("humorous_non_tech", 4),   # misidentified -> fail
        "humorous_tech": ("humorous_tech", 5),
        "humorous_non_tech": ("humorous_non_tech", 2),  # low confidence -> fail
    }))
    results = await blind_lineup(fake_llm, CAPTIONS, min_confidence=3, rng_seed=7)
    assert results["formal"].passed
    assert results["humorous_tech"].passed
    assert not results["sarcastic"].passed and results["sarcastic"].judged_as == "humorous_non_tech"
    assert not results["humorous_non_tech"].passed  # identified but weak


@pytest.mark.asyncio
async def test_refine_keeps_best_attempt(fake_llm, monkeypatch):
    from sevcap import refine as refine_mod

    monkeypatch.setattr(refine_mod.settings, "max_refine_rounds", 1)
    sheet = FactSheet(
        verified=[VerifiedFact(text="a dog chases a ball", category="actions", support=5, k=5)],
        rejected=[], semantic_entropy=0.1, k=5,
    )

    all_pass = {k: (k, 5) for k in CAPTIONS}

    async def fake_assess_lineup(llm, captions, min_confidence=3, rng_seed=None):
        # round 0: sarcastic misidentified; round 1: everything passes
        from sevcap.gates import LineupResult
        first = not hasattr(fake_assess_lineup, "done")
        fake_assess_lineup.done = True
        out = {}
        for k in captions:
            judged, conf = ("formal", 4) if (first and k == "sarcastic") else all_pass[k]
            out[k] = LineupResult(k, judged == k, judged, conf, judged == k and conf >= 3)
        return out

    async def fake_grounding(llm, fs, caption, images_b64=None):
        return True, []

    async def fake_generate(llm, fs, style, feedback=None, seed=None, images_b64=None, duration_s=None):
        assert feedback and "sarcastic" in feedback.lower() or style.key == "sarcastic"
        return "Ah yes, the ball. Clearly the dog's finest hour."

    monkeypatch.setattr(refine_mod, "blind_lineup", fake_assess_lineup)
    monkeypatch.setattr(refine_mod, "check_grounding", fake_grounding)
    monkeypatch.setattr(refine_mod, "generate_caption", fake_generate)

    outcomes = await refine_captions(fake_llm, sheet, dict(CAPTIONS))
    assert outcomes["sarcastic"].attempts == 2
    assert outcomes["sarcastic"].final.lineup_passed
    assert outcomes["formal"].attempts == 1
    for k in CAPTIONS:
        assert outcomes[k].final.grounded
