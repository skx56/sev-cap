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
# Gemma (dedicated deployment or serverless) remains available as an env-only
# override for the "best use of Gemma" bonus chase — never the default, because
# heavy multi-image Gemma calls have a real decoding-degeneracy / stall rate.
_KIMI = "accounts/fireworks/models/kimi-k2p6"
_GEMMA_BONUS = (
    "accounts/fireworks/models/gemma-4-26b-a4b-it"
    "#accounts/skx56/deployments/c4pafnfc"
)


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
    )
    # Primary: Kimi end-to-end (vision extraction + text stages). Override with
    # SEVCAP_MODEL / SEVCAP_VISION_MODEL to chase the Gemma bonus later.
    model: str = field(
        default_factory=lambda: os.environ.get("SEVCAP_MODEL", _KIMI)
    )
    vision_model: str = field(
        default_factory=lambda: os.environ.get("SEVCAP_VISION_MODEL", _KIMI)
    )
    # Fallbacks only matter when the primary is overridden to something that
    # rejects images / is undeployed (e.g. a Gemma bonus chase that cold-starts).
    fallback_vision_model: str = field(
        default_factory=lambda: os.environ.get(
            "SEVCAP_FALLBACK_VISION_MODEL", _KIMI
        )
    )
    fallback_model: str = field(
        default_factory=lambda: os.environ.get("SEVCAP_FALLBACK_MODEL", _KIMI)
    )

    # K=3 is the floor Farquhar et al.'s semantic-entropy method still gives a
    # meaningful agreement signal at (below that, "verification" degenerates
    # into majority-of-2). Fewer samples = fewer heavy multi-image VLM calls.
    k_samples: int = field(default_factory=lambda: _int("SEVCAP_K", 3))
    # With K=3, support>=2-of-3 is majority agreement (not a singleton) and
    # keeps the fact sheet from starving; support=3 would require unanimous
    # samples and is unrealistically strict under any per-call failure rate.
    min_support: int = field(default_factory=lambda: _int("SEVCAP_MIN_SUPPORT", 2))
    # Base keyframes at 30s; clip_profile() scales up to SEVCAP_FRAMES_MAX at 120s.
    n_frames: int = field(default_factory=lambda: _int("SEVCAP_FRAMES", 6))
    extract_temperature: float = field(default_factory=lambda: _float("SEVCAP_EXTRACT_TEMP", 0.55))
    time_budget_s: float = field(default_factory=lambda: _float("SEVCAP_TIME_BUDGET", 1800.0))
    # One refine round buys most of the grounding/style-lineup win per clip at
    # roughly half the Stage-2 LLM-call cost vs two rounds.
    max_refine_rounds: int = field(default_factory=lambda: _int("SEVCAP_MAX_REFINE_ROUNDS", 1))
    # Keep concurrency modest on serverless Kimi — flooding it with parallel
    # multi-image draft calls causes request timeouts that stall the batch.
    # 1 = fully serial drafts/upgrades (slow but reliable on Kimi serverless).
    clip_concurrency: int = field(default_factory=lambda: _int("SEVCAP_CLIP_CONCURRENCY", 1))
    llm_concurrency: int = field(default_factory=lambda: _int("SEVCAP_LLM_CONCURRENCY", 2))
    lineup_min_confidence: int = field(default_factory=lambda: _int("SEVCAP_LINEUP_MIN_CONF", 3))
    # Hard per-clip Stage-2 cap: abandon upgrade and keep the draft rather than
    # let one stuck clip consume the global budget. Composes with Deadline.
    # 420s fits K=3 Kimi extraction + clustering + gen + one refine when
    # reasoning is disabled (reasoning=none) so completion tokens stay small.
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
    long_form: bool  # arc-style captions, not scene laundry lists


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
