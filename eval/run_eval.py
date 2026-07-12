"""Grade harness results.json with the internal accuracy+tone judge.

Usage:
  FIREWORKS_API_KEY=... python eval/run_eval.py \\
    --results /path/to/results.json --videos /tmp/sevcap_tasks
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from judge import judge_clip  # noqa: E402
from sevcap.fireworks import Gemma  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.json")
    ap.add_argument(
        "--videos",
        required=True,
        help="Directory with {task_id}.mp4 files",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    if not isinstance(data, list):
        raise SystemExit("results.json must be a JSON array")

    llm = Gemma()
    await llm.resolve_text_model()
    videos = Path(args.videos)
    rows = []
    print(f"{'clip':6} {'style':20} {'acc':>4} {'tone':>4}")
    for row in data:
        tid = row["task_id"]
        video = videos / f"{tid}.mp4"
        if not video.exists():
            print(f"{tid} missing video {video}")
            continue
        scores = await judge_clip(llm, str(video), row["captions"])
        rows.append((tid, scores))
        for style, s in scores.items():
            a, t = s.get("accuracy", 0), s.get("tone", 0)
            flag = "  <-- weak" if min(a, t) <= 3 else ""
            print(f"{tid:6} {style:20} {a:>4} {t:>4}{flag}")

    all_a = [s.get("accuracy", 0) for _, sc in rows for s in sc.values()]
    all_t = [s.get("tone", 0) for _, sc in rows for s in sc.values()]
    if all_a:
        combined = (statistics.mean(all_a) + statistics.mean(all_t)) / 10
        print(f"\nmean accuracy {statistics.mean(all_a):.2f}")
        print(f"mean tone     {statistics.mean(all_t):.2f}")
        print(f"combined      {combined:.3f}  (n={len(all_a)} captions, {len(rows)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
