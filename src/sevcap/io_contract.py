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

INPUT_ENV_VARS = (
    "INPUT_DIR",
    "SEVCAP_INPUT_DIR",
    "VIDEO_DIR",
    "CLIPS_DIR",
    "DATA_DIR",
    "VIDEO_PATH",
    "INPUT_FILE",
    "VIDEO_FILE",
)
OUTPUT_ENV_VARS = ("OUTPUT_DIR", "SEVCAP_OUTPUT_DIR", "RESULTS_DIR", "OUT_DIR")
INPUT_CANDIDATES = (
    "/input",
    "/inputs",
    "/videos",
    "/clips",
    "/data",
    "/mnt/input",
    "/mnt/inputs",
    "/mnt/data",
    "/workspace/input",
    "/app/input",
    "./clips",
    "./input",
    "./inputs",
    "./videos",
)
OUTPUT_CANDIDATES = (
    "/output",
    "/outputs",
    "/results",
    "/mnt/output",
    "/mnt/outputs",
    "/mnt/results",
    "/workspace/output",
    "/app/output",
    "./output",
    "./outputs",
    "./results",
)


def find_input_dir(cli_arg: str | None = None) -> Path:
    if cli_arg:
        p = Path(cli_arg)
        return p.parent if p.is_file() and p.suffix.lower() in VIDEO_EXTS else p
    for var in INPUT_ENV_VARS:
        val = os.environ.get(var)
        if val:
            p = Path(val)
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                return p.parent
            if p.is_dir() and list_videos(p):
                return p
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
    # Prefer an existing mount point. Creating /output inside the container when
    # the harness actually mounted /outputs is a classic OUTPUT_MISSING failure.
    for cand in OUTPUT_CANDIDATES:
        p = Path(cand)
        if p.is_dir():
            try:
                p.mkdir(parents=True, exist_ok=True)
                return p
            except OSError:
                continue
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
        self._mirror_dirs = self._discover_mirror_dirs(output_dir)

    @staticmethod
    def _discover_mirror_dirs(primary: Path) -> list[Path]:
        dirs = [primary]
        for var in OUTPUT_ENV_VARS:
            val = os.environ.get(var)
            if val:
                dirs.append(Path(val))
        for cand in OUTPUT_CANDIDATES:
            p = Path(cand)
            if p.is_dir():
                dirs.append(p)
        out: list[Path] = []
        seen: set[str] = set()
        for d in dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
                key = str(d.resolve())
            except OSError:
                continue
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

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
        self._combined[clip_id] = record
        combined = {"results": list(self._combined.values())}
        flat = {cid: rec["captions"] for cid, rec in self._combined.items()}
        for out_dir in self._mirror_dirs:
            try:
                atomic_write_json(out_dir / f"{clip_id}.json", record)
                atomic_write_json(out_dir / "captions.json", combined)
                atomic_write_json(out_dir / "results.json", combined)
                atomic_write_json(out_dir / "submission.json", combined)
                # Some simple scorers expect a direct clip_id -> captions map.
                atomic_write_json(out_dir / "predictions.json", flat)
            except OSError as e:
                log.warning("failed writing output mirror %s: %s", out_dir, str(e)[:120])
        log.info(
            "wrote output for %s (%d clips total, %d output dirs)",
            clip_id,
            len(self._combined),
            len(self._mirror_dirs),
        )
