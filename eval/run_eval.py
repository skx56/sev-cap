"""Run the pipeline over clips/ and grade the results with the internal judge.

Usage:
    FIREWORKS_API_KEY=... .venv/bin/python eval/run_eval.py [--skip-pipeline]

Response caching (.sevcap_cache/) makes reruns free while tuning thresholds.
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
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--results", default="results")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="grade existing results/ without rerunning the pipeline")
    args = ap.parse_args()

    if not args.skip_pipeline:
        from sevcap.pipeline import run
        await run(args.clips, args.results)

    llm = Gemma()
    await llm.resolve_text_model()
    results_dir = Path(args.results)
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "captions.json":
            continue
        rec = json.loads(f.read_text())
        video = next(Path(args.clips).glob(f"{rec['clip']}.*"), None)
        if not video:
            continue
        scores = await judge_clip(llm, str(video), rec["captions"])
        rows.append((rec["clip"], scores))

    print(f"\n{'clip':24} {'style':18} {'acc':>4} {'tone':>4}")
    accs, tones = [], []
    for clip, scores in rows:
        for style, s in scores.items():
            a, t = s.get("accuracy", 0), s.get("tone", 0)
            accs.append(a)
            tones.append(t)
            flag = "  <-- weak" if min(a, t) <= 3 else ""
            print(f"{clip:24} {style:18} {a:>4} {t:>4}{flag}")
    if accs:
        print(f"\nmean accuracy {statistics.mean(accs):.2f}  "
              f"mean tone {statistics.mean(tones):.2f}  "
              f"(n={len(accs)} captions, {len(rows)} clips)")
        combined = (statistics.mean(accs) + statistics.mean(tones)) / 10
        print(f"combined (leaderboard-style 0-1): {combined:.2f}")
    out = Path("eval/out")
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_scores.json").write_text(json.dumps(
        {clip: scores for clip, scores in rows}, indent=2))
    print(f"llm usage: {llm.usage.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
