"""Keyframe sampling with ffmpeg: uniform coverage + scene-change frames.

Uniform frames guarantee temporal coverage of slow/static clips; the
scene-change pass catches cuts and fast action that uniform sampling misses.
The union (deduped, time-ordered) is what Stage 1 sees.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

log = logging.getLogger("sevcap.sampler")

SCENE_THRESHOLD = 0.30
MAX_EDGE_PX = 768  # keep frames small: 5MB/image API cap, ~256 tokens each
JPEG_QUALITY = 4   # ffmpeg qscale (2=best); 4 is visually fine and compact


@dataclass
class Keyframe:
    t: float          # timestamp in seconds
    path: str         # jpeg on disk

    def b64(self) -> str:
        with open(self.path, "rb") as f:
            return base64.b64encode(f.read()).decode()


def probe_duration(video: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", video],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def _extract_at(video: str, t: float, dest: str) -> bool:
    scale = f"scale='min({MAX_EDGE_PX},iw)':-2"
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}",
         "-i", video, "-frames:v", "1", "-vf", scale, "-q:v", str(JPEG_QUALITY),
         "-y", dest],
        capture_output=True,
    )
    return res.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


def _scene_change_times(video: str, max_scenes: int) -> list[float]:
    """Timestamps of scene cuts via the select filter's showinfo output."""
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", video,
         "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    times: list[float] = []
    for line in res.stderr.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[1].split()[0]))
            except (ValueError, IndexError):
                continue
    if len(times) > max_scenes:
        # keep an even spread of the strongest coverage rather than the first N
        step = len(times) / max_scenes
        times = [times[int(i * step)] for i in range(max_scenes)]
    return times


def sample_keyframes(video: str, n_frames: int = 10, workdir: str | None = None) -> list[Keyframe]:
    """Return n_frames keyframes: ~60% uniform, ~40% scene-change (deduped)."""
    workdir = workdir or tempfile.mkdtemp(prefix="sevcap_frames_")
    os.makedirs(workdir, exist_ok=True)
    duration = probe_duration(video)

    n_uniform = max(3, int(round(n_frames * 0.6)))
    n_scene = max(0, n_frames - n_uniform)

    if duration <= 45:
        # Short clips: segment midpoints (best accuracy in evals).
        uniform_ts = [duration * (i + 0.5) / n_uniform for i in range(n_uniform)]
    else:
        # Longer clips: Raccoon-style 5% edge padding avoids black intro/outro.
        pad = duration * 0.05
        span = max(duration - 2 * pad, duration * 0.5)
        if n_uniform == 1:
            uniform_ts = [pad + span / 2]
        else:
            uniform_ts = [pad + i * span / (n_uniform - 1) for i in range(n_uniform)]
    scene_ts = _scene_change_times(video, n_scene) if n_scene else []

    # dedupe: drop scene frames within 1s of a uniform frame
    merged = list(uniform_ts)
    for t in scene_ts:
        if all(abs(t - u) > 1.0 for u in merged):
            merged.append(t)
    merged = sorted(t for t in merged if 0 <= t < duration)[: n_frames + 2]

    frames: list[Keyframe] = []
    for i, t in enumerate(merged):
        dest = os.path.join(workdir, f"frame_{i:03d}_{t:.1f}s.jpg")
        if _extract_at(video, t, dest):
            frames.append(Keyframe(t=t, path=dest))
        else:
            log.warning("frame extraction failed at t=%.2fs for %s", t, video)
    if not frames:
        raise RuntimeError(f"Could not extract any keyframes from {video}")
    return frames[:n_frames] if len(frames) > n_frames else frames
