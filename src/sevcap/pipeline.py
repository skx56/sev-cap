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
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .audio import transcribe
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


@dataclass
class DraftResult:
    frames: list
    transcript: str
    draft: dict[str, str]


async def draft_phase(llm: Gemma, video: Path, writer: ResultWriter) -> DraftResult:
    """Phase 1 only: cheap, fast, run for every clip before any Phase 2 work.

    Decoupled from Phase 2 so a slow-to-upgrade clip can never prevent other
    clips from getting their immediate draft coverage (see module docstring).
    """
    clip_id = video.stem
    log.info("[%s] sampling keyframes", clip_id)
    frames_task = asyncio.to_thread(sample_keyframes, str(video), settings.n_frames)
    audio_task = asyncio.to_thread(transcribe, str(video))
    frames, transcript = await asyncio.gather(frames_task, audio_task)
    images = [f.b64() for f in frames]
    if transcript:
        log.info("[%s] audio transcript (%d chars): %s", clip_id, len(transcript), transcript[:100])

    draft = None
    for attempt in range(2):
        try:
            draft = await generate_draft(llm, images, transcript=transcript)
            break
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] draft attempt %d failed: %s", clip_id, attempt + 1, str(e)[:150])
    if draft is None:
        draft = {k: "A short video clip." for k in STYLE_ORDER}
    writer.write(clip_id, draft, meta={"stage": "draft", "keyframes": len(frames)})
    return DraftResult(frames=frames, transcript=transcript, draft=draft)


async def upgrade_phase(
    llm: Gemma, video: Path, writer: ResultWriter, deadline: Deadline, pre: DraftResult
) -> dict:
    clip_id = video.stem
    frames, transcript, draft = pre.frames, pre.transcript, pre.draft

    if deadline.expired(reserve=60.0):
        log.warning("[%s] budget exhausted before SEV upgrade; keeping draft", clip_id)
        return {"clip": clip_id, "stage": "draft"}

    # ---- Phase 2: full SEV upgrade
    try:
        extractions = await extract_facts(llm, frames, k=settings.k_samples, transcript=transcript)
        # Semantic entropy needs independent samples to agree; below 2 there
        # is no consensus signal and "verification" is noise. Scaled to
        # k_samples (not a fixed 3) since K itself is now as low as 3 and
        # requiring 3-of-3 successes here would defeat the point of lowering
        # K for reliability in the first place.
        min_needed = min(2, settings.k_samples)
        if len(extractions) < min_needed:
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


async def _upgrade_with_cap(
    llm: Gemma, video: Path, writer: ResultWriter, deadline: Deadline, pre: DraftResult
) -> dict:
    """Run upgrade_phase under a hard per-clip time cap.

    Bounds Phase 2 (extraction + verification + gates + refine) to
    settings.clip_upgrade_timeout_s (also capped by whatever remains of the
    global deadline) so one clip stuck retrying a degenerating model can
    never consume the whole run's time budget — it just falls back to its
    already-written draft. Composes with (does not replace) the global
    Deadline: the global deadline still shrinks per_clip_budget below the
    timeout as the run's overall time runs out.
    """
    clip_id = video.stem
    per_clip_budget = min(deadline.remaining(), settings.clip_upgrade_timeout_s)
    if per_clip_budget <= 0:
        return {"clip": clip_id, "stage": "draft"}
    try:
        return await asyncio.wait_for(
            upgrade_phase(llm, video, writer, deadline, pre), timeout=per_clip_budget
        )
    except asyncio.TimeoutError:
        log.warning("[%s] upgrade timed out after %.0fs; keeping draft",
                    clip_id, per_clip_budget)
        return {"clip": clip_id, "stage": "draft", "error": "upgrade-timeout"}


async def process_clip(
    llm: Gemma, video: Path, writer: ResultWriter, deadline: Deadline
) -> dict:
    """Single-clip convenience wrapper (draft phase, then capped SEV upgrade).

    Used by callers that only ever process one clip at a time and don't need
    the batch run()'s cross-clip draft/upgrade decoupling — e.g. the
    Streamlit demo, or ad-hoc scripts/tests.
    """
    pre = await draft_phase(llm, video, writer)
    if deadline.expired(reserve=60.0):
        log.warning("[%s] budget exhausted before SEV upgrade; keeping draft", video.stem)
        return {"clip": video.stem, "stage": "draft"}
    return await _upgrade_with_cap(llm, video, writer, deadline, pre)


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

    # ---- Phase 1 for EVERY clip first (before any Phase 2). Bounded by the
    # same clip_concurrency semaphore AND a hard per-draft timeout so a
    # single slow/timeouting draft call cannot pin the whole batch forever
    # (observed under Kimi serverless load: 2/7 drafts hung while 5 finished).
    # Clips that miss the draft timeout keep their placeholder and are still
    # eligible for a later repair-round draft+upgrade if budget remains.
    draft_timeout_s = float(os.environ.get("SEVCAP_DRAFT_TIMEOUT", "180"))

    async def guarded_draft(v: Path) -> tuple[Path, DraftResult | None]:
        async with clip_sem:
            try:
                return v, await asyncio.wait_for(
                    draft_phase(llm, v, writer), timeout=draft_timeout_s
                )
            except asyncio.TimeoutError:
                log.warning("[%s] draft timed out after %.0fs; keeping placeholder",
                            v.stem, draft_timeout_s)
                return v, None
            except Exception as e:  # noqa: BLE001
                log.error("[%s] draft phase failed entirely: %s", v.stem, e)
                writer.write(v.stem, {k: "A short video clip." for k in STYLE_ORDER},
                             meta={"stage": "error", "error": str(e)})
                return v, None

    draft_pairs = await asyncio.gather(*(guarded_draft(v) for v in videos))
    drafts: dict[Path, DraftResult] = {v: d for v, d in draft_pairs if d is not None}
    best: dict[str, dict] = {
        v.stem: {"clip": v.stem, "stage": "draft" if v in drafts else "error"}
        for v in videos
    }

    # ---- Phase 2: SEV upgrades, bounded by clip_concurrency AND a per-clip
    # timeout so one stuck clip can only ever hold up its own concurrency
    # slot for a bounded time, never the whole remaining budget.
    async def guarded_upgrade(v: Path) -> dict:
        pre = drafts.get(v)
        if pre is None:
            return best[v.stem]
        async with clip_sem:
            try:
                return await _upgrade_with_cap(llm, v, writer, deadline, pre)
            except Exception as e:  # noqa: BLE001
                log.error("[%s] clip upgrade failed entirely: %s", v.stem, e)
                return {"clip": v.stem, "stage": "draft", "error": str(e)}

    async def one_pass(targets: list[Path]) -> list[dict]:
        return await asyncio.gather(*(guarded_upgrade(v) for v in targets))

    # Repair loop: transient provider outages (rate-limit storms, spurious
    # 401 windows) can sink a whole pass. While budget remains, re-attempt
    # every clip that did not reach sev-verified — including clips whose
    # first draft timed out (re-run draft, then upgrade).
    targets = [v for v in videos if best[v.stem].get("stage") != "sev-verified"]
    for round_no in range(3):
        if not targets or deadline.expired(reserve=60.0):
            break
        # Re-draft any clip that never got a DraftResult (timeout/error).
        missing = [v for v in targets if v not in drafts]
        if missing:
            log.warning("repair round %d: re-drafting %s",
                        round_no + 1, [v.stem for v in missing])
            redraft = await asyncio.gather(*(guarded_draft(v) for v in missing))
            for v, d in redraft:
                if d is not None:
                    drafts[v] = d
                    best[v.stem] = {"clip": v.stem, "stage": "draft"}
        upgrade_targets = [v for v in targets if v in drafts]
        if not upgrade_targets:
            break
        round_results = await one_pass(upgrade_targets)
        for r in round_results:
            clip = r.get("clip")
            if clip and (r.get("stage") == "sev-verified"
                         or best[clip].get("stage") != "sev-verified"):
                best[clip] = r
        targets = [v for v in videos if best[v.stem].get("stage") != "sev-verified"]
        if not targets or deadline.expired(reserve=180.0):
            break
        log.warning("repair round %d: retrying %s",
                    round_no + 1, [v.stem for v in targets])
        await asyncio.sleep(min(60.0, max(deadline.remaining() * 0.05, 5.0)))
    results = list(best.values())

    summary = {
        "clips": len(videos),
        "stages": [r.get("stage") for r in results],
        "llm_usage": llm.usage.summary(),
    }
    log.info("run complete: %s", summary)
    return summary
