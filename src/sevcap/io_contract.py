"""Harness I/O adapter: input discovery + atomic output writes.

AMD ACT II Track 2 contract:
  - read ``/input/tasks.json`` (array of ``task_id``, ``video_url``, ``styles``)
  - write ``/output/results.json`` (array of ``task_id``, ``captions``)

Local/dev mode still works when ``tasks.json`` is absent: discover ``.mp4`` files
under the input directory and write the same ``results.json`` schema using the
clip stem as ``task_id``. All writes are atomic (temp file + os.replace).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("sevcap.io")

REQUIRED_STYLES = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)

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
TASKS_FILENAMES = ("tasks.json",)
DOWNLOAD_TIMEOUT_S = 180
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


def _dir_has_input(p: Path) -> bool:
    return bool(list_videos(p) or find_tasks_file(p))


def find_tasks_file(input_dir: Path | None = None) -> Path | None:
    explicit = os.environ.get("INPUT_PATH")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        if p.is_dir():
            for name in TASKS_FILENAMES:
                candidate = p / name
                if candidate.is_file():
                    return candidate
    search_dirs = []
    if input_dir is not None:
        search_dirs.append(input_dir)
    search_dirs.extend(Path(cand) for cand in INPUT_CANDIDATES)
    seen: set[str] = set()
    for base in search_dirs:
        try:
            key = str(base.resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        if not base.is_dir():
            continue
        for name in TASKS_FILENAMES:
            candidate = base / name
            if candidate.is_file():
                return candidate
    return None


def find_input_dir(cli_arg: str | None = None) -> Path:
    if cli_arg:
        p = Path(cli_arg)
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTS:
                return p.parent
            if p.name in TASKS_FILENAMES:
                return p.parent
        if p.is_dir() and _dir_has_input(p):
            return p
    for var in INPUT_ENV_VARS:
        val = os.environ.get(var)
        if val:
            p = Path(val)
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                return p.parent
            if p.is_file() and p.name in TASKS_FILENAMES:
                return p.parent
            if p.is_dir() and _dir_has_input(p):
                return p
    for cand in INPUT_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and _dir_has_input(p):
            return p
    # last resort: any dir under / or cwd that contains videos or tasks.json
    for base in (Path("/"), Path.cwd()):
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.name.startswith((".", "proc", "sys", "dev")):
                try:
                    if _dir_has_input(child):
                        return child
                except PermissionError:
                    continue
    raise FileNotFoundError(
        "No input directory with tasks.json or video files found. Set INPUT_DIR "
        "or mount /input/tasks.json."
    )


def find_results_path(cli_arg: str | None = None) -> Path:
    explicit = os.environ.get("OUTPUT_PATH")
    if explicit:
        p = Path(explicit)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return find_output_dir(cli_arg) / "results.json"


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


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


@dataclass
class VideoTask:
    task_id: str
    video_url: str
    styles: list[str]


def load_tasks(tasks_path: Path) -> list[VideoTask]:
    raw = json.loads(tasks_path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{tasks_path} must contain a JSON array")
    tasks: list[VideoTask] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id", "")).strip()
        video_url = str(item.get("video_url", "")).strip()
        styles = item.get("styles") or [
            "formal",
            "sarcastic",
            "humorous_tech",
            "humorous_non_tech",
        ]
        if not task_id or not video_url:
            continue
        tasks.append(VideoTask(task_id=task_id, video_url=video_url, styles=list(styles)))
    if not tasks:
        raise ValueError(f"{tasks_path} contained no valid tasks")
    return tasks


def discover_tasks(input_dir: Path) -> list[VideoTask] | None:
    tasks_path = find_tasks_file(input_dir)
    if tasks_path is None:
        return None
    return load_tasks(tasks_path)


def empty_captions(styles: list[str] | None = None) -> dict[str, str]:
    keys = styles or list(REQUIRED_STYLES)
    return {k: "" for k in keys if k in REQUIRED_STYLES}


def normalize_captions(captions: dict[str, str], styles: list[str] | None = None) -> dict[str, str]:
    """Ensure every required style key exists with a string value."""
    keys = styles or list(REQUIRED_STYLES)
    out = empty_captions(keys)
    for key in out:
        val = captions.get(key, "")
        out[key] = val if isinstance(val, str) else str(val)
    return out


def download_video(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "sevcap/1.0"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    if dest.stat().st_size == 0:
        raise RuntimeError(f"downloaded video is empty: {url}")


class ResultWriter:
    """Maintains harness results.json plus optional debug mirrors, atomically."""

    def __init__(self, output_dir: Path, task_order: list[str] | None = None):
        self.output_dir = output_dir
        self.results_path = find_results_path(str(output_dir))
        self._task_order = list(task_order or [])
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

    def _harness_record(self, task_id: str, captions: dict[str, str], styles: list[str] | None = None) -> dict:
        normalized = normalize_captions(captions, styles)
        return {"task_id": task_id, "captions": normalized}

    def _ordered_results(self) -> list[dict]:
        if self._task_order:
            ordered_ids = list(self._task_order)
            for task_id in self._combined:
                if task_id not in ordered_ids:
                    ordered_ids.append(task_id)
        else:
            ordered_ids = sorted(self._combined)
        return [self._combined[task_id] for task_id in ordered_ids if task_id in self._combined]

    def write(
        self,
        task_id: str,
        captions: dict[str, str],
        meta: dict | None = None,
        styles: list[str] | None = None,
    ) -> None:
        record = self._harness_record(task_id, captions, styles)
        self._combined[task_id] = record
        if task_id not in self._task_order:
            self._task_order.append(task_id)
        self._flush(task_id, record, meta)

    def finalize(self) -> None:
        """Emit a valid (possibly empty) record for every expected task_id."""
        for task_id in self._task_order:
            if task_id not in self._combined:
                self.write(task_id, empty_captions(), meta={"stage": "missing"})

    def _flush(self, task_id: str, record: dict, meta: dict | None) -> None:
        harness_results = self._ordered_results()
        legacy = {
            "results": [
                {"clip": rec["task_id"], "captions": rec["captions"]}
                for rec in harness_results
            ]
        }
        flat = {rec["task_id"]: rec["captions"] for rec in harness_results}
        debug_record = dict(record)
        if meta:
            debug_record["verification"] = meta
        targets = {self.results_path.parent, *self._mirror_dirs}
        for out_dir in targets:
            try:
                atomic_write_json(out_dir / "results.json", harness_results)
                atomic_write_json(out_dir / f"{task_id}.json", debug_record)
                atomic_write_json(out_dir / "captions.json", legacy)
                atomic_write_json(out_dir / "submission.json", legacy)
                atomic_write_json(out_dir / "predictions.json", flat)
            except OSError as e:
                log.warning("failed writing output mirror %s: %s", out_dir, str(e)[:120])
        log.info(
            "wrote output for %s (%d tasks total, results=%s)",
            task_id,
            len(self._combined),
            self.results_path,
        )
