"""Smoke tests for the scoring agent surface."""

from config import Config


def test_required_styles():
    assert Config.REQUIRED_STYLES == [
        "formal",
        "sarcastic",
        "humorous_tech",
        "humorous_non_tech",
    ]


def test_default_models():
    assert "minimax" in Config.FIREWORKS_VISION_MODEL
    assert "kimi" in Config.FIREWORKS_TEXT_MODEL
