#!/usr/bin/env python3
"""Validate Track 2 harness output against the official schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


def validate_results(results_path: Path, expected_task_ids: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    if not results_path.is_file():
        return [f"missing file: {results_path}"]

    try:
        data = json.loads(results_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]

    if not isinstance(data, list):
        errors.append("results.json must be a JSON array")
        return errors

    seen: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"item {i} is not an object")
            continue
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            errors.append(f"item {i} missing task_id")
            continue
        seen.append(task_id)
        caps = item.get("captions")
        if not isinstance(caps, dict):
            errors.append(f"{task_id}: captions must be an object")
            continue
        for style in REQUIRED_STYLES:
            if style not in caps:
                errors.append(f"{task_id}: missing style {style}")
            elif not isinstance(caps[style], str):
                errors.append(f"{task_id}: captions.{style} must be a string")

    if expected_task_ids:
        missing = [tid for tid in expected_task_ids if tid not in seen]
        extra = [tid for tid in seen if tid not in expected_task_ids]
        if missing:
            errors.append(f"missing tasks: {missing}")
        if extra:
            errors.append(f"unexpected tasks: {extra}")
        if seen != expected_task_ids:
            errors.append(f"task order mismatch: got {seen}, want {expected_task_ids}")

    return errors


def main() -> int:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "/output/results.json")
    tasks = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    expected: list[str] | None = None
    if tasks and tasks.is_file():
        raw = json.loads(tasks.read_text())
        expected = [str(x["task_id"]) for x in raw if isinstance(x, dict) and x.get("task_id")]

    errors = validate_results(results, expected)
    if errors:
        print("INVALID")
        for err in errors:
            print(f"- {err}")
        return 1
    print(f"OK ({len(expected or []) or 'n/a'} tasks checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
