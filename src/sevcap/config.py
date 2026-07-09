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
    # Fewer images per vision call cuts cost/latency without losing story
    # coverage for ~30-90s clips.
    n_frames: int = field(default_factory=lambda: _int("SEVCAP_FRAMES", 6))
    extract_temperature: float = field(default_factory=lambda: _float("SEVCAP_EXTRACT_TEMP", 0.55))
    time_budget_s: float = field(default_factory=lambda: _float("SEVCAP_TIME_BUDGET", 1500.0))
    # One refine round buys most of the grounding/style-lineup win per clip at
    # roughly half the Stage-2 LLM-call cost vs two rounds.
    max_refine_rounds: int = field(default_factory=lambda: _int("SEVCAP_MAX_REFINE_ROUNDS", 1))
    # Keep concurrency modest on serverless Kimi — flooding it with 7 parallel
    # multi-image draft calls causes request timeouts that stall the batch.
    clip_concurrency: int = field(default_factory=lambda: _int("SEVCAP_CLIP_CONCURRENCY", 2))
    llm_concurrency: int = field(default_factory=lambda: _int("SEVCAP_LLM_CONCURRENCY", 3))
    lineup_min_confidence: int = field(default_factory=lambda: _int("SEVCAP_LINEUP_MIN_CONF", 3))
    # Hard per-clip Stage-2 cap: abandon upgrade and keep the draft rather than
    # let one stuck clip consume the global budget. Composes with Deadline.
    clip_upgrade_timeout_s: float = field(
        default_factory=lambda: _float("SEVCAP_CLIP_UPGRADE_TIMEOUT", 120.0)
    )

    cache_dir: str = field(default_factory=lambda: os.environ.get("SEVCAP_CACHE_DIR", ".sevcap_cache"))
    cache_enabled: bool = field(
        default_factory=lambda: os.environ.get("SEVCAP_CACHE", "1") not in ("0", "false", "no")
    )


settings = Settings()
