"""Minimal Gradio demo: upload a video, get 4 SEV-Cap captions.

Uses the real pipeline (process_clip). Launch with share=True for a public URL.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr

# Ensure package import works when launched as `python demo/app.py`
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sevcap.fireworks import Gemma  # noqa: E402
from sevcap.io_contract import ResultWriter  # noqa: E402
from sevcap.pipeline import Deadline, process_clip  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("sevcap.demo")

# Demo budget: drafts land fast; SEV upgrade runs if time remains.
DEMO_BUDGET_S = float(os.environ.get("SEVCAP_DEMO_BUDGET", "240"))
OUT_DIR = Path(__file__).resolve().parent / "out" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_llm: Gemma | None = None
_llm_lock = asyncio.Lock()


async def _get_llm() -> Gemma:
    global _llm
    async with _llm_lock:
        if _llm is None:
            _llm = Gemma()
            try:
                await _llm.resolve_text_model()
                await _llm.check_vision()
            except Exception as e:  # noqa: BLE001
                log.warning("model warm-up failed (will retry on request): %s", e)
        return _llm


def _format_facts(meta: dict) -> str:
    fv = (meta or {}).get("fact_verification") or {}
    verified = fv.get("verified_facts") or []
    if not verified:
        return "(no verified facts yet — draft stage or empty sheet)"
    lines = [f"[{f.get('category','?')}] {f.get('text','')}  (support {f.get('support','?')})"
             for f in verified[:20]]
    return "\n".join(lines)


async def caption_video(video_path: str | None) -> tuple[str, str, str, str, str, str]:
    if not video_path:
        return ("", "", "", "", "Please upload an mp4 video.", "")

    src = Path(video_path)
    if not src.exists():
        return ("", "", "", "", f"Upload missing: {video_path}", "")

    work = Path(tempfile.mkdtemp(prefix="sevcap_demo_"))
    try:
        dest = work / f"upload{src.suffix or '.mp4'}"
        shutil.copy2(src, dest)

        llm = await _get_llm()
        writer = ResultWriter(OUT_DIR)
        deadline = Deadline(DEMO_BUDGET_S)
        t0 = time.monotonic()
        log.info("demo request: %s budget=%.0fs", dest.name, DEMO_BUDGET_S)
        result = await process_clip(llm, dest, writer, deadline)
        elapsed = time.monotonic() - t0

        clip_id = dest.stem
        rec_path = OUT_DIR / f"{clip_id}.json"
        if not rec_path.exists():
            return ("", "", "", "", f"No output written ({result})", "")

        rec = json.loads(rec_path.read_text())
        caps = rec.get("captions") or {}
        meta = rec.get("verification") or {}
        stage = meta.get("stage") or result.get("stage", "?")
        status = (
            f"Done in {elapsed:.0f}s · stage={stage} · "
            f"budget={DEMO_BUDGET_S:.0f}s (draft first, then SEV upgrade if time allows)"
        )
        facts = _format_facts(meta)
        return (
            caps.get("formal", ""),
            caps.get("sarcastic", ""),
            caps.get("humorous_tech", ""),
            caps.get("humorous_non_tech", ""),
            status,
            facts,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("demo caption failed")
        return ("", "", "", "", f"Error: {e}", "")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SEV-Cap Demo") as demo:
        gr.Markdown(
            """
# SEV-Cap — live caption demo
Upload a short mp4. The **real** pipeline runs (Gemma vision + Whisper audio +
semantic-entropy verification). Expect **~1–4 minutes** depending on clip length
and model warm-up (scale-to-zero may add a short wait on the first request).
"""
        )
        video = gr.Video(label="Video (mp4)", sources=["upload"])
        btn = gr.Button("Generate captions", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            formal = gr.Textbox(label="Formal", lines=3)
            sarcastic = gr.Textbox(label="Sarcastic", lines=3)
        with gr.Row():
            tech = gr.Textbox(label="Humorous (tech)", lines=3)
            nontech = gr.Textbox(label="Humorous (non-tech)", lines=3)
        facts = gr.Textbox(label="Verified facts (if SEV upgrade finished)", lines=8)

        btn.click(
            fn=caption_video,
            inputs=[video],
            outputs=[formal, sarcastic, tech, nontech, status, facts],
        )
    return demo


if __name__ == "__main__":
    if not os.environ.get("FIREWORKS_API_KEY"):
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("SEVCAP_DEMO_PORT", "7860")),
        share=True,
        show_error=True,
        prevent_thread_lock=True,
    )
    print("KEEPALIVE: Gradio share launched", flush=True)
    import time
    while True:
        time.sleep(3600)
