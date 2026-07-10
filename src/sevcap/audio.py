"""Audio-track transcription: a second, deterministic evidence source.

Fireworks deprecated its hosted Whisper endpoint (2026-06-10), so this runs
locally with faster-whisper — CPU-only, int8, small model, fast enough to
transcribe a 30s-2min clip well within the harness time budget.

The transcript is NOT injected straight into the verified fact sheet: it is
handed to every Stage-1 vision-extraction sample as extra context, the same
way a human annotator would use both eyes and ears. Because every sample sees
the identical transcript, any fact it grounds will naturally show high
cross-sample support and survive semantic-entropy verification on its own
merits — it does not bypass the hallucination firewall, it feeds it.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass

log = logging.getLogger("sevcap.audio")

_model = None
_MIN_WAV_BYTES = 2000  # smaller than this is silence/empty-track noise


@dataclass
class TranscriptResult:
    text: str
    trusted: bool
    reason: str = ""


def assess_transcript_quality(text: str) -> tuple[bool, str]:
    """Heuristic trust gate for Whisper output before it becomes facts."""
    if not text or len(text.strip()) < 3:
        return False, "empty"
    t = text.strip()
    words = t.split()
    if len(words) < 2:
        return False, "too-short"
    # Garbled ASR: heavy stutter / repetition ("you, you see, you see")
    if re.search(r"\b(\w+),\s*\1\b", t, re.I):
        return False, "stutter-repeat"
    uniq = len(set(w.lower() for w in words))
    if len(words) >= 8 and uniq / len(words) < 0.45:
        return False, "repetitive"
    # Run-on without sentence structure often means bad segmentation
    if len(t) > 120 and t.count(".") + t.count("?") + t.count("!") == 0:
        return False, "unpunctuated-runon"
    return True, "ok"


def transcribe_with_meta(video: str) -> TranscriptResult:
    text = transcribe(video)
    trusted, reason = assess_transcript_quality(text)
    if text and not trusted:
        log.info("transcript rejected for fact use (%s): %s", reason, text[:80])
    return TranscriptResult(text=text, trusted=trusted, reason=reason)


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # heavy import, load lazily

        size = os.environ.get("SEVCAP_WHISPER_MODEL", "base")
        download_root = os.environ.get("SEVCAP_WHISPER_CACHE_DIR") or None
        _model = WhisperModel(size, device="cpu", compute_type="int8", download_root=download_root)
    return _model


def _extract_audio(video: str, workdir: str) -> str | None:
    dest = os.path.join(workdir, "audio.wav")
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video,
         "-vn", "-ac", "1", "-ar", "16000", "-y", dest],
        capture_output=True,
    )
    if res.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) < _MIN_WAV_BYTES:
        return None
    return dest


def transcribe(video: str) -> str:
    """Best-effort dialogue transcript; '' on silence, no-speech, or failure.

    Never raises: a transcription failure must degrade to vision-only
    extraction, not sink the clip.
    """
    if os.environ.get("SEVCAP_AUDIO", "1") in ("0", "false", "no"):
        return ""
    try:
        with tempfile.TemporaryDirectory(prefix="sevcap_audio_") as workdir:
            wav = _extract_audio(video, workdir)
            if not wav:
                return ""
            model = _get_model()
            segments, _info = model.transcribe(wav, beam_size=1, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text if len(text) >= 3 else ""
    except Exception as e:  # noqa: BLE001
        log.warning("transcription failed for %s: %s", video, str(e)[:150])
        return ""
