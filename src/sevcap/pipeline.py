"""Anytime orchestrator — grounded describe → verify → style-write pipeline.

Describe frames, self-verify, write multi-candidate captions per style, then
vision-prejudge on accuracy + tone (Track 2 axes) with a polish/reselect pass.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .audio import transcribe_with_meta
from .config import ClipProfile, clip_profile, settings
from .fireworks import Gemma
from .grounded import (
    _style_ok,
    caption_all_styles,
    describe_scene,
    draft_style_candidates,
    safe_style_caption,
    verify_description,
)
from .io_contract import (
    ResultWriter,
    discover_tasks,
    download_video,
    empty_captions,
    find_input_dir,
    find_output_dir,
    find_tasks_file,
    list_videos,
    load_tasks,
)
from .sampler import probe_duration, sample_keyframes
from .styles import STYLE_ORDER, STYLES

log = logging.getLogger("sevcap.pipeline")

POLISH_MIN_SCORE = 4
POLISH_ENABLED = os.environ.get("SEVCAP_POLISH", "1") not in ("0", "false", "no")
POLISH_MIN_REMAINING_S = 45.0
DOWNLOAD_WORKDIR = Path(os.environ.get("SEVCAP_DOWNLOAD_DIR", "/tmp/sevcap_tasks"))


def _fallback_captions(description: str = "") -> dict[str, str]:
    """Description-grounded fallback — never the generic harness-killer phrase."""
    base = description.strip()
    if not base:
        return empty_captions()
    sentence = base.split(".")[0].strip() + "."
    return {k: sentence for k in STYLE_ORDER}


def _load_task_plan(in_dir: Path | None = None) -> tuple[list[str], dict[str, list[str]]]:
    tasks_path = find_tasks_file(in_dir) if in_dir else find_tasks_file()
    if tasks_path is None:
        return [], {}
    tasks = load_tasks(tasks_path)
    order = [t.task_id for t in tasks]
    styles = {t.task_id: t.styles for t in tasks}
    return order, styles


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
    styles: list[str]
    video_url: str | None = None
    video: Path | None = None


async def _resolve_jobs(in_dir: Path) -> tuple[list[ClipJob], list[str]]:
    tasks = discover_tasks(in_dir)
    if tasks:
        jobs = [
            ClipJob(task_id=t.task_id, styles=t.styles, video_url=t.video_url)
            for t in tasks
        ]
        return jobs, [t.task_id for t in tasks]
    videos = list_videos(in_dir)
    return [
        ClipJob(task_id=v.stem, styles=list(STYLE_ORDER), video=v)
        for v in videos
    ], [v.stem for v in videos]


async def _ensure_video(job: ClipJob) -> Path:
    if job.video is not None and job.video.exists():
        return job.video
    if not job.video_url:
        raise FileNotFoundError(f"[{job.task_id}] no video path or URL")
    DOWNLOAD_WORKDIR.mkdir(parents=True, exist_ok=True)
    dest = DOWNLOAD_WORKDIR / f"{job.task_id}.mp4"
    log.info("[%s] downloading %s", job.task_id, job.video_url)
    await asyncio.to_thread(download_video, job.video_url, dest)
    job.video = dest
    return dest


async def _placeholder(writer: ResultWriter, job: ClipJob) -> None:
    writer.write(
        job.task_id,
        empty_captions(job.styles),
        meta={"stage": "placeholder"},
        styles=job.styles,
    )


async def _grounded_caption(
    llm: Gemma,
    job: ClipJob,
    writer: ResultWriter,
    deadline: Deadline,
    state: ClipState | None = None,
) -> dict:
    task_id = job.task_id
    video = await _ensure_video(job)
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

        log.info("[%s] writing styled captions (multi-candidate)", task_id)
        captions = await caption_all_styles(
            llm, description, styles=job.styles, video=str(video), frames=frames,
        )

        # Light second pass only for styles still below the accuracy floor.
        if POLISH_ENABLED and deadline.remaining() > POLISH_MIN_REMAINING_S:
            from .prejudge import pick_best_candidate, score_caption_axes

            formal_anchor = captions.get("formal", "")
            polish_order = [k for k in ("sarcastic", "humorous_tech", "humorous_non_tech", "formal")
                            if k in job.styles]
            for style_key in polish_order:
                if deadline.expired(reserve=POLISH_MIN_REMAINING_S):
                    break
                try:
                    acc, tone = await score_caption_axes(
                        llm, str(video), style_key, captions[style_key], frames=frames,
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("[%s] polish score failed for %s: %s", task_id, style_key, str(e)[:80])
                    continue
                # Reselect unless both axes are solid (failed prejudge returns 0).
                if acc >= 4 and tone >= 4:
                    continue
                if acc == 0 and tone == 0:
                    log.info("[%s] polish: prejudge failed for %s — forcing reselect", task_id, style_key)
                elif acc >= POLISH_MIN_SCORE and tone >= 4:
                    continue
                log.info("[%s] reselecting %s (acc=%d tone=%d)", task_id, style_key, acc, tone)
                prior = [captions[k] for k in job.styles if k != style_key and captions.get(k)]
                try:
                    from .grounded import vision_style_caption

                    extra = await draft_style_candidates(
                        llm, description, STYLES[style_key], prior, formal_anchor or None,
                        include_safe=True,
                    )
                    try:
                        vcap = await vision_style_caption(
                            llm, frames, description, STYLES[style_key], formal_anchor or None,
                        )
                        if vcap:
                            extra.append(vcap)
                    except Exception:  # noqa: BLE001
                        pass
                    pool = [captions[style_key], *extra]

                    def _valid(cap: str, _key=style_key, _formal=formal_anchor) -> bool:
                        return _style_ok(_key, cap, _formal if _key != "formal" else None, description)

                    best, new_acc, new_tone = await pick_best_candidate(
                        llm, str(video), style_key, pool, frames=frames, is_valid=_valid,
                    )
                    if best and (
                        new_acc > acc
                        or (new_acc == acc and new_tone >= tone)
                        or (acc == 0 and new_acc > 0)
                    ):
                        captions[style_key] = best
                        if style_key == "formal":
                            formal_anchor = best
                        log.info("[%s] selected %s acc=%d tone=%d", task_id, style_key, new_acc, new_tone)
                except Exception as e:  # noqa: BLE001
                    log.warning("[%s] reselect failed for %s: %s", task_id, style_key, str(e)[:100])

        meta = {
            "stage": "sev-verified",
            "keyframes": len(frames),
            "duration_s": profile.duration_s,
            "transcript_trusted": state.transcript_trusted,
            "grounding_description": description,
        }
        writer.write(task_id, captions, meta=meta, styles=job.styles)
        log.info("[%s] grounded captioning complete", task_id)
        return {"task_id": task_id, "stage": "sev-verified"}
    except Exception as e:  # noqa: BLE001
        log.error("[%s] grounded caption failed: %s", task_id, str(e)[:200])
        desc = state.description if state and state.description else ""
        fallback = _fallback_captions(desc)
        writer.write(
            task_id,
            fallback if any(fallback.values()) else empty_captions(job.styles),
            meta={"stage": "draft", "error": str(e)[:300]},
            styles=job.styles,
        )
        return {"task_id": task_id, "stage": "draft", "error": str(e)[:300]}


async def process_clip(
    llm: Gemma, job: ClipJob, writer: ResultWriter, deadline: Deadline,
) -> dict:
    task_id = job.task_id
    await _placeholder(writer, job)
    if deadline.expired(reserve=30.0):
        return {"task_id": task_id, "stage": "placeholder"}
    try:
        video = await _ensure_video(job)
    except Exception as e:  # noqa: BLE001
        log.error("[%s] download failed: %s", task_id, str(e)[:200])
        writer.write(
            task_id,
            empty_captions(job.styles),
            meta={"stage": "error", "error": str(e)[:300]},
            styles=job.styles,
        )
        return {"task_id": task_id, "stage": "error", "error": str(e)[:300]}
    budget = min(deadline.remaining(), clip_profile(probe_duration(str(video))).upgrade_timeout_s)
    try:
        return await asyncio.wait_for(
            _grounded_caption(llm, job, writer, deadline), timeout=max(budget, 60.0),
        )
    except asyncio.TimeoutError:
        log.warning("[%s] grounded caption timed out after %.0fs", task_id, budget)
        writer.write(
            task_id,
            empty_captions(job.styles),
            meta={"stage": "draft", "error": "timeout"},
            styles=job.styles,
        )
        return {"task_id": task_id, "stage": "draft", "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        log.error("[%s] process_clip failed: %s", task_id, str(e)[:200])
        writer.write(
            task_id,
            empty_captions(job.styles),
            meta={"stage": "draft", "error": str(e)[:300]},
            styles=job.styles,
        )
        return {"task_id": task_id, "stage": "draft", "error": str(e)[:300]}


async def run(input_dir: str | None = None, output_dir: str | None = None) -> dict:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    task_order, _ = _load_task_plan(Path(input_dir) if input_dir else None)
    try:
        in_dir = find_input_dir(input_dir)
        out_dir = find_output_dir(output_dir)
    except Exception as e:  # noqa: BLE001
        log.error("I/O setup failed: %s", e)
        out_dir = find_output_dir(output_dir)
        writer = ResultWriter(out_dir, task_order=task_order)
        for task_id in task_order or ["_fatal"]:
            writer.write(
                task_id,
                empty_captions(),
                meta={"stage": "error", "error": str(e)[:300]},
            )
        writer.finalize()
        return {"clips": 0, "stages": ["error"], "error": str(e)}

    jobs, task_order = await _resolve_jobs(in_dir)
    log.info("input=%s output=%s tasks=%d", in_dir, out_dir, len(jobs))
    writer = ResultWriter(out_dir, task_order=task_order)
    if not jobs:
        log.error("No tasks or video files in %s", in_dir)
        for task_id in task_order or ["_empty"]:
            writer.write(
                task_id,
                empty_captions(),
                meta={"stage": "error", "error": "no tasks or video files"},
            )
        writer.finalize()
        return {"clips": 0, "stages": ["error"], "error": "no tasks or video files"}

    llm = Gemma()
    deadline = Deadline(settings.time_budget_s)
    sem = asyncio.Semaphore(settings.clip_concurrency)

    for job in jobs:
        await _placeholder(writer, job)

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
                writer.write(
                    job.task_id,
                    empty_captions(job.styles),
                    meta={"stage": "error", "error": str(e)[:300]},
                    styles=job.styles,
                )
                return {"task_id": job.task_id, "stage": "error", "error": str(e)}

    results_list = await asyncio.gather(*(one(job) for job in jobs))
    by_id = {r["task_id"]: r for r in results_list}

    for round_no in range(1):  # one repair pass — keep under TIMEOUT with UHD clips
        failed = [j for j in jobs if by_id.get(j.task_id, {}).get("stage") != "sev-verified"]
        if not failed or deadline.expired(reserve=90.0):
            break
        log.warning("repair round %d: %s", round_no + 1, [j.task_id for j in failed])
        for r in await asyncio.gather(*(one(j) for j in failed)):
            task_id = r.get("task_id")
            if task_id:
                by_id[task_id] = r

    writer.finalize()
    results = [by_id.get(j.task_id, {"task_id": j.task_id, "stage": "error"}) for j in jobs]

    summary = {
        "clips": len(jobs),
        "stages": [r.get("stage") for r in results],
        "llm_usage": llm.usage.summary(),
    }
    log.info("run complete: %s", summary)
    return summary
