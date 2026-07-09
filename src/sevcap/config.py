"""Central runtime configuration, sourced from environment variables.

Every knob the harness or a judge might need to tweak is an env var so the
public Docker image never needs a rebuild for tuning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
    )
    # One model family end-to-end: Gemma 3 27B is extractor, NLI judge,
    # generator, lineup judge and refiner ("best use of Gemma" story).
    model: str = field(
        default_factory=lambda: os.environ.get(
            "SEVCAP_MODEL", "accounts/fireworks/models/gemma-3-27b-it"
        )
    )
    vision_model: str = field(
        default_factory=lambda: os.environ.get(
            "SEVCAP_VISION_MODEL", "accounts/fireworks/models/gemma-3-27b-it"
        )
    )
    # If the primary vision model rejects image input on serverless, we fall
    # back to this VLM for Stage 1 only; all text stages stay on Gemma.
    fallback_vision_model: str = field(
        default_factory=lambda: os.environ.get(
            "SEVCAP_FALLBACK_VISION_MODEL",
            "accounts/fireworks/models/kimi-k2p6",
        )
    )
    # If the primary text model is not deployed on the account running the
    # container (e.g. Gemma not serverless), all text stages fall back here.
    fallback_model: str = field(
        default_factory=lambda: os.environ.get(
            "SEVCAP_FALLBACK_MODEL", "accounts/fireworks/models/kimi-k2p6"
        )
    )

    k_samples: int = field(default_factory=lambda: _int("SEVCAP_K", 5))
    min_support: int = field(default_factory=lambda: _int("SEVCAP_MIN_SUPPORT", 3))
    # 10->6: fewer images per vision-extraction call measurably cuts the
    # gemma-4-26b-a4b-it decoding-repetition-loop rate on this deployment
    # without losing story coverage for ~30-90s clips.
    n_frames: int = field(default_factory=lambda: _int("SEVCAP_FRAMES", 6))
    # K independent samples still need genuine diversity for the semantic-
    # entropy signal to mean anything; 0.55 cuts degeneracy vs 0.7 while
    # keeping samples non-identical (verified empirically, see eval notes).
    extract_temperature: float = field(default_factory=lambda: _float("SEVCAP_EXTRACT_TEMP", 0.55))
    time_budget_s: float = field(default_factory=lambda: _float("SEVCAP_TIME_BUDGET", 1800.0))
    # Drafts already score close to ceiling on the internal judge; one refine
    # round (not two) buys most of the grounding/style-lineup win per clip at
    # roughly half the Stage-2 LLM-call cost, so more clips fit the budget.
    max_refine_rounds: int = field(default_factory=lambda: _int("SEVCAP_MAX_REFINE_ROUNDS", 1))
    clip_concurrency: int = field(default_factory=lambda: _int("SEVCAP_CLIP_CONCURRENCY", 4))
    llm_concurrency: int = field(default_factory=lambda: _int("SEVCAP_LLM_CONCURRENCY", 4))
    lineup_min_confidence: int = field(default_factory=lambda: _int("SEVCAP_LINEUP_MIN_CONF", 3))
    # Hard cap on how long any single clip's Phase-2 SEV upgrade may occupy a
    # concurrency slot, so one clip stuck retrying a degenerating model can
    # never starve the others out of their turn (the bug that left 2/7 clips
    # at "placeholder" in a real run: all 3 concurrency slots were pinned on
    # slow clips for the whole budget).
    clip_upgrade_timeout_s: float = field(
        default_factory=lambda: _float("SEVCAP_CLIP_UPGRADE_TIMEOUT", 420.0)
    )

    cache_dir: str = field(default_factory=lambda: os.environ.get("SEVCAP_CACHE_DIR", ".sevcap_cache"))
    cache_enabled: bool = field(
        default_factory=lambda: os.environ.get("SEVCAP_CACHE", "1") not in ("0", "false", "no")
    )


settings = Settings()
