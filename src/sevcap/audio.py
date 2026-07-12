"""Optional audio transcript hook (disabled for Track 2 scoring by default)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TranscriptResult:
    text: str
    trusted: bool
    reason: str = ""


def transcribe_with_meta(video: str) -> TranscriptResult:
    """Return dialogue text when SEVCAP_AUDIO=1; otherwise empty (vision-only)."""
    del video  # unused when ASR is off
    if os.environ.get("SEVCAP_AUDIO", "0") not in ("1", "true", "yes"):
        return TranscriptResult(text="", trusted=False, reason="disabled")
    # Whisper is intentionally not bundled in the scoring image.
    return TranscriptResult(text="", trusted=False, reason="asr-unavailable")
