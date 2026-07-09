"""Anytime orchestrator.

Per clip, two phases:
  Phase 1 (fast): single-pass draft of all 4 styles straight from frames,
  written to disk IMMEDIATELY. From this moment the clip has valid output.
  Phase 2 (SEV upgrade): K-sample extraction -> semantic-entropy verification
  -> fact-conditioned generation -> grounding gate + blind lineup ->
  Self-Refine, atomically overwriting the draft when it survives the gates.

A global time budget guards the whole run: if the harness kills the container
or the budget expires mid-upgrade, every clip already has its best output so
far — degraded gracefully, never missing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from .config import settings
from .entropy import verify_facts
from .extractor import extract_facts
from .fireworks import Gemma
from .generator import generate_caption, generate_draft
from .io_contract import ResultWriter, find_input_dir, find_output_dir, list_videos
from .refine import refine_captions
from .sampler import sample_keyframes
from .styles import STYLE_ORDER, STYLES

log = logging.getLogger("sevcap.pipeline")


class Deadline:
    def __init__(self, budget_s: float):
        self.t0 = time.monotonic()
        self.budget = budget_s

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.t0)

    def expired(self, reserve: float = 0.0) -> bool:
        return self.remaining() <= reserve


async def process_clip(
    llm: Gemma, video: Path, writer: ResultWriter, deadline: Deadline
) -> dict:
    clip_id = video.stem
    log.info("[%s] sampling keyframes", clip_id)
    frames = await asyncio.to_thread(sample_keyframes, str(video), settings.n_frames)
    images = [f.b64() for f in frames]

    # ---- Phase 1: draft written immediately (TIMEOUT/OUTPUT_MISSING defense)
    draft = None
    for attempt in range(2):
        try:
            draft = await generate_draft(llm, images)
            break
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] draft attempt %d failed: %s", clip_id, attempt + 1, str(e)[:150])
    if draft is None:
        draft = {k: "A short video clip." for k in STYLE_ORDER}
    writer.write(clip_id, draft, meta={"stage": "draft", "keyframes": len(frames)})

    if deadline.expired(reserve=60.0):
        log.warning("[%s] budget exhausted after draft; keeping draft", clip_id)
        return {"clip": clip_id, "stage": "draft"}

    # ---- Phase 2: full SEV upgrade
    try:
        extractions = await extract_facts(llm, frames, k=settings.k_samples)
        if len(extractions) < 3:
            # Semantic entropy needs independent samples to agree; with fewer
            # than 3 there is no consensus signal and "verification" is noise.
            raise RuntimeError(f"only {len(extractions)} extraction samples succeeded")
        fact_sheet = await verify_facts(llm, extractions, settings.min_support)
        if not fact_sheet.verified:
            # Never caption from an empty sheet (the generator would write
            # meta-captions about missing facts). Keep the visual draft.
            raise RuntimeError("empty fact sheet after verification")

        captions: dict[str, str] = {}
        results = await asyncio.gather(
            *(generate_caption(llm, fact_sheet, STYLES[k], seed=100 + i)
              for i, k in enumerate(STYLE_ORDER)),
            return_exceptions=True,
        )
        for k, res in zip(STYLE_ORDER, results):
            ok = not isinstance(res, Exception) and str(res).strip()
            captions[k] = str(res).strip() if ok else draft[k]

        outcomes = await refine_captions(llm, fact_sheet, captions)
        final: dict[str, str] = {}
        for k in STYLE_ORDER:
            best = outcomes[k].final
            text = best.text.strip()
            # A caption that never passed grounding, or reads like a template
            # artifact, must not beat the vision-grounded draft.
            usable = text and best.grounded and not text.startswith(("[", "("))
            final[k] = text if usable else draft[k]
        meta = {
            "stage": "sev-verified",
            "keyframes": len(frames),
            "fact_verification": fact_sheet.report(),
            "style_gates": {k: outcomes[k].report() for k in STYLE_ORDER},
        }
        writer.write(clip_id, final, meta=meta)
        log.info("[%s] SEV upgrade complete", clip_id)
        return {"clip": clip_id, "stage": "sev-verified"}
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] SEV upgrade failed (%s); draft output stands", clip_id, e)
        return {"clip": clip_id, "stage": "draft", "error": str(e)}


async def run(input_dir: str | None = None, output_dir: str | None = None) -> dict:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    in_dir = find_input_dir(input_dir)
    out_dir = find_output_dir(output_dir)
    videos = list_videos(in_dir)
    log.info("input=%s output=%s clips=%d", in_dir, out_dir, len(videos))
    if not videos:
        raise FileNotFoundError(f"No video files in {in_dir}")

    llm = Gemma()
    writer = ResultWriter(out_dir)
    deadline = Deadline(settings.time_budget_s)
    clip_sem = asyncio.Semaphore(settings.clip_concurrency)

    # OUTPUT_MISSING defense, layer 0: every clip has (placeholder) output on
    # disk before any processing starts; everything after only upgrades it.
    for v in videos:
        writer.write(v.stem, {k: "A short video clip." for k in STYLE_ORDER},
                     meta={"stage": "placeholder"})

    try:
        model = await llm.resolve_text_model()
        log.info("text model resolved: %s", model)
    except Exception as e:  # noqa: BLE001
        log.error("no reachable text model: %s", e)
    try:
        vision = await llm.check_vision()
        log.info("vision model resolved: %s", vision)
    except Exception as e:  # noqa: BLE001
        log.error("no vision model available: %s", e)

    async def guarded(v: Path) -> dict:
        async with clip_sem:
            try:
                return await process_clip(llm, v, writer, deadline)
            except Exception as e:  # noqa: BLE001
                log.error("[%s] clip failed entirely: %s", v.stem, e)
                writer.write(v.stem, {k: "A short video clip." for k in STYLE_ORDER},
                             meta={"stage": "error", "error": str(e)})
                return {"clip": v.stem, "stage": "error", "error": str(e)}

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(guarded(v) for v in videos)),
            timeout=max(deadline.remaining(), 1.0),
        )
    except asyncio.TimeoutError:
        log.warning("global time budget hit; drafts already on disk")
        results = [{"stage": "timeout"}]

    summary = {
        "clips": len(videos),
        "stages": [r.get("stage") for r in results],
        "llm_usage": llm.usage.summary(),
    }
    log.info("run complete: %s", summary)
    return summary
