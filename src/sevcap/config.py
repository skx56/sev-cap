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
    n_frames: int = field(default_factory=lambda: _int("SEVCAP_FRAMES", 10))
    time_budget_s: float = field(default_factory=lambda: _float("SEVCAP_TIME_BUDGET", 1500.0))
    max_refine_rounds: int = field(default_factory=lambda: _int("SEVCAP_MAX_REFINE_ROUNDS", 2))
    clip_concurrency: int = field(default_factory=lambda: _int("SEVCAP_CLIP_CONCURRENCY", 3))
    llm_concurrency: int = field(default_factory=lambda: _int("SEVCAP_LLM_CONCURRENCY", 4))
    lineup_min_confidence: int = field(default_factory=lambda: _int("SEVCAP_LINEUP_MIN_CONF", 3))

    cache_dir: str = field(default_factory=lambda: os.environ.get("SEVCAP_CACHE_DIR", ".sevcap_cache"))
    cache_enabled: bool = field(
        default_factory=lambda: os.environ.get("SEVCAP_CACHE", "1") not in ("0", "false", "no")
    )


settings = Settings()
