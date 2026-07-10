"""Anytime orchestrator — grounded describe → verify → style-write pipeline.

Phase 1 writes placeholder output immediately. Phase 2 runs the Raccoon-style
grounded path (describe frames, self-verify, write 4 styles from one shared
description) plus a light accuracy polish pass. Simple and reliable: ~10 LLM
calls/clip instead of 30+ through semantic-entropy gates that often fell back
to unverified drafts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .audio import transcribe_with_meta
from .config import ClipProfile, clip_profile, settings
from .fireworks import Gemma
from .grounded import caption_all_styles, describe_scene, verify_description, write_styled_caption
from .io_contract import (
    ResultWriter,
    discover_tasks,
    download_video,
    find_input_dir,
    find_output_dir,
    list_videos,
)
from .prejudge import score_caption_accuracy
from .sampler import probe_duration, sample_keyframes
from .styles import STYLE_ORDER, STYLES

log = logging.getLogger("sevcap.pipeline")

POLISH_MIN_SCORE = 4
POLISH_ENABLED = os.environ.get("SEVCAP_POLISH", "1") not in ("0", "false", "no")


class Deadline:
    def __init__(self, budget_s: float):
        self.t0 = time.monotonic()
        self.budget = budget_s

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.t0)

    def expired(self, reserve: float = 0.0) -> bool:
        return self.remaining() <= reserve


@dataclass
class ClipState:
    frames: list
    profile: ClipProfile
    transcript: str
    transcript_trusted: bool
    description: str = ""


@dataclass
class ClipJob:
    task_id: str
    video: Path


async def _resolve_jobs(in_dir: Path) -> tuple[list[ClipJob], list[str]]:
    tasks = discover_tasks(in_dir)
    if tasks:
        work = Path(tempfile.mkdtemp(prefix="sevcap_tasks_"))
        jobs: list[ClipJob] = []
        for task in tasks:
            dest = work / f"{task.task_id}.mp4"
            log.info("[%s] downloading %s", task.task_id, task.video_url)
            await asyncio.to_thread(download_video, task.video_url, dest)
            jobs.append(ClipJob(task_id=task.task_id, video=dest))
        return jobs, [t.task_id for t in tasks]
    videos = list_videos(in_dir)
    return [ClipJob(task_id=v.stem, video=v) for v in videos], [v.stem for v in videos]


async def _placeholder(writer: ResultWriter, task_id: str) -> None:
    writer.write(
        task_id,
        {k: "A short video clip." for k in STYLE_ORDER},
        meta={"stage": "placeholder"},
    )


async def _grounded_caption(
    llm: Gemma, job: ClipJob, writer: ResultWriter, state: ClipState | None = None,
) -> dict:
    task_id = job.task_id
    video = job.video
    try:
        profile = state.profile if state else clip_profile(probe_duration(str(video)))

        if state is None:
            log.info("[%s] sampling keyframes (duration=%.0fs, n=%d)",
                     task_id, profile.duration_s, profile.n_frames)
            frames_task = asyncio.to_thread(sample_keyframes, str(video), profile.n_frames)
            audio_task = asyncio.to_thread(transcribe_with_meta, str(video))
            frames, tr = await asyncio.gather(frames_task, audio_task)
            state = ClipState(
                frames=frames, profile=profile,
                transcript=tr.text, transcript_trusted=tr.trusted,
            )
        else:
            frames = state.frames

        log.info("[%s] describing scene", task_id)
        draft_desc = await describe_scene(
            llm, frames, profile,
            transcript=state.transcript, transcript_trusted=state.transcript_trusted,
        )
        log.info("[%s] verifying description", task_id)
        description = await verify_description(llm, frames, draft_desc)
        state.description = description

        log.info("[%s] writing styled captions", task_id)
        captions = await caption_all_styles(llm, description)

        if POLISH_ENABLED:
            for style_key in STYLE_ORDER:
                for polish_round in range(2):
                    try:
                        score = await score_caption_accuracy(
                            llm, str(video), style_key, captions[style_key], frames=frames,
                        )
                    except Exception as e:  # noqa: BLE001
                        log.warning("[%s] polish score failed for %s: %s", task_id, style_key, str(e)[:80])
                        break
                    if score >= POLISH_MIN_SCORE:
                        break
                    log.info("[%s] polishing %s round %d (score=%d)",
                             task_id, style_key, polish_round + 1, score)
                    prior = [captions[k] for k in STYLE_ORDER if k != style_key]
                    try:
                        candidate = await write_styled_caption(
                            llm, description, STYLES[style_key], prior,
                            feedback=(
                                f"The previous caption scored {score}/5 on factual accuracy. "
                                "Stay closer to the description; do not invent events or objects."
                            ),
                        )
                        new_score = await score_caption_accuracy(
                            llm, str(video), style_key, candidate, frames=frames,
                        )
                        if new_score >= score:
                            captions[style_key] = candidate
                            score = new_score
                    except Exception as e:  # noqa: BLE001
                        log.warning("[%s] polish failed for %s: %s", task_id, style_key, str(e)[:100])
                        break
                    if score >= POLISH_MIN_SCORE:
                        break

        meta = {
            "stage": "sev-verified",
            "keyframes": len(frames),
            "duration_s": profile.duration_s,
            "transcript_trusted": state.transcript_trusted,
            "grounding_description": description,
        }
        writer.write(task_id, captions, meta=meta)
        log.info("[%s] grounded captioning complete", task_id)
        return {"task_id": task_id, "stage": "sev-verified"}
    except Exception as e:  # noqa: BLE001
        log.error("[%s] grounded caption failed: %s", task_id, str(e)[:200])
        writer.write(
            task_id,
            {k: "A short video clip." for k in STYLE_ORDER},
            meta={"stage": "draft", "error": str(e)[:300]},
        )
        return {"task_id": task_id, "stage": "draft", "error": str(e)[:300]}


async def process_clip(
    llm: Gemma, job: ClipJob, writer: ResultWriter, deadline: Deadline,
) -> dict:
    task_id = job.task_id
    await _placeholder(writer, task_id)
    if deadline.expired(reserve=30.0):
        return {"task_id": task_id, "stage": "placeholder"}
    budget = min(deadline.remaining(), clip_profile(probe_duration(str(job.video))).upgrade_timeout_s)
    try:
        return await asyncio.wait_for(
            _grounded_caption(llm, job, writer), timeout=max(budget, 60.0),
        )
    except asyncio.TimeoutError:
        log.warning("[%s] grounded caption timed out after %.0fs", task_id, budget)
        writer.write(
            task_id,
            {k: "A short video clip." for k in STYLE_ORDER},
            meta={"stage": "draft", "error": "timeout"},
        )
        return {"task_id": task_id, "stage": "draft", "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        log.error("[%s] process_clip failed: %s", task_id, str(e)[:200])
        writer.write(
            task_id,
            {k: "A short video clip." for k in STYLE_ORDER},
            meta={"stage": "draft", "error": str(e)[:300]},
        )
        return {"task_id": task_id, "stage": "draft", "error": str(e)[:300]}


async def run(input_dir: str | None = None, output_dir: str | None = None) -> dict:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    try:
        in_dir = find_input_dir(input_dir)
        out_dir = find_output_dir(output_dir)
    except Exception as e:  # noqa: BLE001
        log.error("I/O setup failed: %s", e)
        out_dir = find_output_dir(output_dir)
        writer = ResultWriter(out_dir)
        writer.write(
            "_fatal",
            {k: "A short video clip." for k in STYLE_ORDER},
            meta={"stage": "error", "error": str(e)[:300]},
        )
        return {"clips": 0, "stages": ["error"], "error": str(e)}

    try:
        jobs, task_order = await _resolve_jobs(in_dir)
    except Exception as e:  # noqa: BLE001
        log.error("input resolution failed: %s", e)
        writer = ResultWriter(out_dir)
        writer.write(
            "_fatal",
            {k: "A short video clip." for k in STYLE_ORDER},
            meta={"stage": "error", "error": str(e)[:300]},
        )
        return {"clips": 0, "stages": ["error"], "error": str(e)}

    log.info("input=%s output=%s tasks=%d", in_dir, out_dir, len(jobs))
    writer = ResultWriter(out_dir, task_order=task_order)
    if not jobs:
        log.error("No tasks or video files in %s", in_dir)
        writer.write(
            "_empty",
            {k: "A short video clip." for k in STYLE_ORDER},
            meta={"stage": "error", "error": "no tasks or video files"},
        )
        return {"clips": 0, "stages": ["error"], "error": "no tasks or video files"}

    llm = Gemma()
    deadline = Deadline(settings.time_budget_s)
    sem = asyncio.Semaphore(settings.clip_concurrency)

    for job in jobs:
        await _placeholder(writer, job.task_id)

    try:
        model = await llm.resolve_text_model()
        log.info("text model resolved: %s", model)
    except Exception as e:  # noqa: BLE001
        log.error("no reachable text model: %s", e)

    async def one(job: ClipJob) -> dict:
        async with sem:
            try:
                return await process_clip(llm, job, writer, deadline)
            except Exception as e:  # noqa: BLE001
                log.error("[%s] clip failed: %s", job.task_id, e)
                return {"task_id": job.task_id, "stage": "error", "error": str(e)}

    results_list = await asyncio.gather(*(one(job) for job in jobs))
    by_id = {r["task_id"]: r for r in results_list}

    for round_no in range(2):
        failed = [j for j in jobs if by_id.get(j.task_id, {}).get("stage") != "sev-verified"]
        if not failed or deadline.expired(reserve=60.0):
            break
        log.warning("repair round %d: %s", round_no + 1, [j.task_id for j in failed])
        for r in await asyncio.gather(*(one(j) for j in failed)):
            task_id = r.get("task_id")
            if task_id:
                by_id[task_id] = r

    results = [by_id.get(j.task_id, {"task_id": j.task_id, "stage": "error"}) for j in jobs]

    summary = {
        "clips": len(jobs),
        "stages": [r.get("stage") for r in results],
        "llm_usage": llm.usage.summary(),
    }
    log.info("run complete: %s", summary)
    return summary
