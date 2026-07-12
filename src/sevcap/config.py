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


# Kimi K2.6 is the reliability-first default (serverless, strong VLM + text).
# Gemma remains available as an env-only override for the bonus chase.
_KIMI = "accounts/fireworks/models/kimi-k2p6"


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get("SEVCAP_MODEL", _KIMI)
    )
    vision_model: str = field(
        default_factory=lambda: os.environ.get("SEVCAP_VISION_MODEL", _KIMI)
    )
    fallback_vision_model: str = field(
        default_factory=lambda: os.environ.get(
            "SEVCAP_FALLBACK_VISION_MODEL", _KIMI
        )
    )
    fallback_model: str = field(
        default_factory=lambda: os.environ.get("SEVCAP_FALLBACK_MODEL", _KIMI)
    )

    n_frames: int = field(default_factory=lambda: _int("SEVCAP_FRAMES", 6))
    time_budget_s: float = field(default_factory=lambda: _float("SEVCAP_TIME_BUDGET", 900.0))
    clip_concurrency: int = field(default_factory=lambda: _int("SEVCAP_CLIP_CONCURRENCY", 2))
    llm_concurrency: int = field(default_factory=lambda: _int("SEVCAP_LLM_CONCURRENCY", 6))
    clip_upgrade_timeout_s: float = field(
        default_factory=lambda: _float("SEVCAP_CLIP_UPGRADE_TIMEOUT", 420.0)
    )

    cache_dir: str = field(default_factory=lambda: os.environ.get("SEVCAP_CACHE_DIR", ".sevcap_cache"))
    cache_enabled: bool = field(
        default_factory=lambda: os.environ.get("SEVCAP_CACHE", "1") not in ("0", "false", "no")
    )


# Harness clips are 30–120s; scale sampling and timeouts across that range.
CLIP_DURATION_MIN_S = 30.0
CLIP_DURATION_MAX_S = 120.0
LONG_CLIP_THRESHOLD_S = 75.0


@dataclass(frozen=True)
class ClipProfile:
    """Per-clip knobs derived from duration (30s floor → 120s ceiling)."""

    duration_s: float
    n_frames: int
    max_scenes: int
    upgrade_timeout_s: float
    long_form: bool


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clip_profile(duration_s: float) -> ClipProfile:
    """Scale keyframes, scenes, and upgrade budget for 30–120s clips."""
    d = max(CLIP_DURATION_MIN_S, min(CLIP_DURATION_MAX_S, float(duration_s)))
    t = (d - CLIP_DURATION_MIN_S) / (CLIP_DURATION_MAX_S - CLIP_DURATION_MIN_S)
    n_min = settings.n_frames
    n_max = _int("SEVCAP_FRAMES_MAX", 12)
    s_min = _int("SEVCAP_MAX_SCENES_MIN", 8)
    s_max = _int("SEVCAP_MAX_SCENES_MAX", max(s_min + 4, 16))
    to_min = settings.clip_upgrade_timeout_s
    to_max = _float("SEVCAP_CLIP_UPGRADE_TIMEOUT_MAX", max(to_min + 120.0, 540.0))
    return ClipProfile(
        duration_s=d,
        n_frames=int(round(_lerp(float(n_min), float(n_max), t))),
        max_scenes=int(round(_lerp(float(s_min), float(s_max), t))),
        upgrade_timeout_s=_lerp(to_min, to_max, t),
        long_form=d >= LONG_CLIP_THRESHOLD_S,
    )


settings = Settings()
