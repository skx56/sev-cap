"""Harness I/O adapter: input discovery + atomic output writes.

The exact harness contract is not published, so this module is deliberately
forgiving: it discovers the input directory from env vars or well-known mount
points, and writes results both per-clip and as a single combined
captions.json, in a flat, obvious schema. All writes are atomic (temp file +
os.replace) so a mid-write kill can never corrupt or lose output — the
OUTPUT_MISSING / TIMEOUT defense.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("sevcap.io")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}

INPUT_ENV_VARS = ("INPUT_DIR", "SEVCAP_INPUT_DIR", "VIDEO_DIR", "CLIPS_DIR", "DATA_DIR")
OUTPUT_ENV_VARS = ("OUTPUT_DIR", "SEVCAP_OUTPUT_DIR", "RESULTS_DIR")
INPUT_CANDIDATES = ("/input", "/videos", "/clips", "/data", "./clips", "./input")
OUTPUT_CANDIDATES = ("/output", "/results", "./results")


def find_input_dir(cli_arg: str | None = None) -> Path:
    if cli_arg:
        return Path(cli_arg)
    for var in INPUT_ENV_VARS:
        val = os.environ.get(var)
        if val and Path(val).is_dir():
            return Path(val)
    for cand in INPUT_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and list_videos(p):
            return p
    # last resort: any dir under / or cwd that contains videos
    for base in (Path("/"), Path.cwd()):
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.name.startswith((".", "proc", "sys", "dev")):
                try:
                    if list_videos(child):
                        return child
                except PermissionError:
                    continue
    raise FileNotFoundError(
        "No input directory with video files found. Set INPUT_DIR or mount clips "
        "at /input."
    )


def find_output_dir(cli_arg: str | None = None) -> Path:
    for val in ([cli_arg] if cli_arg else []) + [os.environ.get(v) for v in OUTPUT_ENV_VARS]:
        if val:
            p = Path(val)
            p.mkdir(parents=True, exist_ok=True)
            return p
    for cand in OUTPUT_CANDIDATES:
        p = Path(cand)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            continue
    p = Path("./results")
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_videos(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


class ResultWriter:
    """Maintains per-clip JSONs plus a combined captions.json, atomically."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._combined: dict[str, dict] = {}

    def write(self, clip_id: str, captions: dict[str, str], meta: dict | None = None) -> None:
        record = {
            "clip": clip_id,
            "captions": {
                "formal": captions.get("formal", ""),
                "sarcastic": captions.get("sarcastic", ""),
                "humorous_tech": captions.get("humorous_tech", ""),
                "humorous_non_tech": captions.get("humorous_non_tech", ""),
            },
        }
        if meta:
            record["verification"] = meta
        atomic_write_json(self.output_dir / f"{clip_id}.json", record)
        self._combined[clip_id] = record
        atomic_write_json(self.output_dir / "captions.json", {"results": list(self._combined.values())})
        log.info("wrote output for %s (%d clips total)", clip_id, len(self._combined))
