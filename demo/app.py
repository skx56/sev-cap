"""SEV-Cap demo — elegant dark Streamlit UI over the production pipeline.

Local:
  streamlit run demo/app.py --server.port 7860
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAX_UPLOAD_MB = int(os.environ.get("SEVCAP_DEMO_MAX_UPLOAD_MB", "1024"))
COMPRESS_ABOVE_MB = int(os.environ.get("SEVCAP_DEMO_COMPRESS_ABOVE_MB", "80"))
STYLE_ORDER = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


def _load_secrets() -> None:
    try:
        if "FIREWORKS_API_KEY" in st.secrets:
            os.environ.setdefault("FIREWORKS_API_KEY", str(st.secrets["FIREWORKS_API_KEY"]))
        for k, v in st.secrets.items():
            if isinstance(v, str) and (
                k.startswith("SEVCAP_")
                or k in {"FIREWORKS_VISION_MODEL", "FIREWORKS_TEXT_MODEL", "AUTO_TRANSCRIBE"}
            ):
                os.environ.setdefault(k, v)
    except Exception:  # noqa: BLE001
        pass
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_secrets()

from agent import process_single_task  # noqa: E402
from config import Config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("sevcap.demo")

OUT_DIR = Path(__file__).resolve().parent / "out" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


STYLES = [
    ("formal", "Formal", "Clear. Precise. Archival."),
    ("sarcastic", "Sarcastic", "Dry irony. Quiet bite."),
    ("humorous_tech", "Humorous · Tech", "One metaphor. On-screen facts."),
    ("humorous_non_tech", "Humorous · Non-Tech", "Warm. Everyday. No jargon."),
]


@st.cache_resource
def _warm() -> bool:
    try:
        Config.validate()
    except Exception as e:  # noqa: BLE001
        log.warning("config warm-up: %s", e)
        return False
    return True


def _save_upload(uploaded, dest: Path) -> int:
    """Stream upload to disk; return byte size."""
    size = 0
    with open(dest, "wb") as out:
        uploaded.seek(0)
        while True:
            chunk = uploaded.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            size += len(chunk)
    return size


def _compress_for_demo(src: Path, dest: Path) -> Path:
    """Downscale large clips so demo stays responsive."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(dest),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
        raise RuntimeError(f"ffmpeg compress failed: {(res.stderr or res.stdout)[:300]}")
    return dest


def _caption(video_path: Path) -> tuple[dict[str, str], dict]:
    _warm()
    t0 = time.monotonic()
    result = process_single_task({
        "task_id": video_path.stem,
        "video_url": str(video_path),
        "styles": list(STYLE_ORDER),
    })
    elapsed = time.monotonic() - t0
    caps = result.get("captions") or {}
    try:
        (OUT_DIR / f"{video_path.stem}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        pass
    return caps, {"elapsed": elapsed, "stage": "ok"}


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.stApp {
  background:
    radial-gradient(900px 520px at 12% -8%, rgba(201,162,93,0.10), transparent 55%),
    radial-gradient(700px 480px at 92% 8%, rgba(94,129,140,0.10), transparent 50%),
    linear-gradient(180deg, #09090b 0%, #0d0d10 48%, #080809 100%);
  color: #ece7df;
}
.block-container { max-width: 1080px; padding-top: 2.4rem; padding-bottom: 3.5rem; }
.brand {
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(3.4rem, 7vw, 5.4rem);
  font-weight: 600; letter-spacing: -0.02em; line-height: 0.95;
  color: #f4efe6; margin: 0 0 0.55rem 0;
}
.tag {
  display: inline-block; font-size: 0.72rem; letter-spacing: 0.18em;
  text-transform: uppercase; color: #c9a25d; margin-bottom: 1.1rem;
}
.lede {
  max-width: 40rem; font-weight: 300; font-size: 1.08rem; line-height: 1.55;
  color: #a7a29a; margin-bottom: 1.2rem;
}
.hint {
  font-size: 0.86rem; color: #8b8680; margin-bottom: 1.4rem;
}
.panel {
  border: 1px solid rgba(236,231,223,0.10);
  background: rgba(255,255,255,0.025);
  border-radius: 18px; padding: 1.15rem 1.2rem;
}
.panel-label {
  font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: #8b8680; margin-bottom: 0.75rem;
}
.style-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; margin: 1.2rem 0 1.6rem; }
.style-chip {
  border: 1px solid rgba(236,231,223,0.10); border-radius: 14px;
  padding: 0.85rem 0.9rem; background: rgba(255,255,255,0.02); min-height: 92px;
}
.style-chip .n { font-family: 'Cormorant Garamond', serif; font-size: 1.15rem; color: #f1ebe3; margin-bottom: 0.25rem; }
.style-chip .d { font-size: 0.78rem; color: #8f8a83; line-height: 1.35; }
.cap {
  border: 1px solid rgba(236,231,223,0.10); border-radius: 16px;
  padding: 1.1rem 1.15rem 1.2rem;
  background: linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
  min-height: 168px;
}
.cap .k { font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; color: #c9a25d; margin-bottom: 0.55rem; }
.cap .t { font-family: 'Cormorant Garamond', serif; font-size: 1.28rem; line-height: 1.35; color: #f3eee6; }
.stat-wrap { display:flex; gap:0.7rem; flex-wrap:wrap; margin: 0.4rem 0 1.1rem; }
.stat {
  border: 1px solid rgba(236,231,223,0.10); border-radius: 12px;
  padding: 0.65rem 0.9rem; background: rgba(255,255,255,0.02); min-width: 110px;
}
.stat .k { font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; color: #8b8680; }
.stat .v { font-family: 'Cormorant Garamond', serif; font-size: 1.35rem; color: #f1ebe3; margin-top: 0.15rem; }
.stButton > button {
  width: 100%;
  border: 1px solid rgba(201,162,93,0.45) !important;
  background: linear-gradient(180deg, #d4b06e, #b8893f) !important;
  color: #1a140c !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em;
  border-radius: 999px !important;
  padding: 0.75rem 1rem !important;
}
.stButton > button:disabled { opacity: 0.35 !important; }
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px dashed rgba(236,231,223,0.22) !important;
  border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #a7a29a !important; }
@media (max-width: 900px) { .style-row { grid-template-columns: 1fr 1fr; } }
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="SEV-Cap",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if not os.environ.get("FIREWORKS_API_KEY"):
        st.error("FIREWORKS_API_KEY is missing. Add it in Streamlit secrets or a local `.env`.")
        st.stop()

    st.markdown('<div class="tag">Video Captioning</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand">SEV-Cap</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lede">Four grounded captions from one clip — formal, sarcastic, '
        "humorous-tech, humorous-non-tech — scored for accuracy and tone before they ship.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hint">Tip: clips under ~{COMPRESS_ABOVE_MB}MB are fastest. '
        f"Larger files (up to {MAX_UPLOAD_MB}MB) are accepted and auto-compressed for demo.</div>",
        unsafe_allow_html=True,
    )

    chips = "".join(
        f'<div class="style-chip"><div class="n">{name}</div><div class="d">{desc}</div></div>'
        for _, name, desc in STYLES
    )
    st.markdown(f'<div class="style-row">{chips}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown('<div class="panel-label">Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "video",
            type=["mp4", "mov", "mkv", "webm"],
            label_visibility="collapsed",
            help=f"Max {MAX_UPLOAD_MB}MB. Large files are compressed automatically.",
        )
        ready = False
        size_mb = 0.0
        if uploaded is not None:
            try:
                # Prefer reported size; fall back to buffer length.
                raw_size = getattr(uploaded, "size", None)
                if raw_size is None:
                    pos = uploaded.tell()
                    uploaded.seek(0, os.SEEK_END)
                    raw_size = uploaded.tell()
                    uploaded.seek(pos)
                size_mb = float(raw_size) / (1024 * 1024)
                if size_mb <= 0:
                    st.error("Upload failed (empty file). Remove it and try again.")
                elif size_mb > MAX_UPLOAD_MB:
                    st.error(
                        f"File is {size_mb:.0f}MB — over the {MAX_UPLOAD_MB}MB limit. "
                        "Compress/export a shorter 720p clip and re-upload."
                    )
                else:
                    ready = True
                    st.caption(f"Ready · {uploaded.name} · {size_mb:.1f}MB")
            except Exception as e:  # noqa: BLE001
                st.error(f"Upload unreadable: {e}. Remove the red file chip and upload again.")
        run = st.button("Generate captions", type="primary", disabled=not ready, use_container_width=True)

    with right:
        st.markdown('<div class="panel-label">Preview</div>', unsafe_allow_html=True)
        if ready and uploaded is not None:
            try:
                st.video(uploaded)
            except Exception:  # noqa: BLE001
                st.info("Preview unavailable for this file — generation can still run.")
        else:
            st.markdown(
                '<div class="panel" style="min-height:180px;display:flex;align-items:center;'
                'color:#8b8680;font-weight:300;">Drop a short clip to begin.</div>',
                unsafe_allow_html=True,
            )

    if not run:
        # Show prior results if any
        if st.session_state.get("last_caps"):
            _render_results(st.session_state["last_caps"], st.session_state.get("last_info") or {})
        return
    if not ready or uploaded is None:
        st.warning("Upload a valid video first (red chip = failed upload).")
        return

    work = Path(tempfile.mkdtemp(prefix="sevcap_up_"))
    try:
        src = work / Path(uploaded.name).name
        with st.status("Preparing video…", expanded=True) as status:
            st.write("Saving upload to disk")
            try:
                nbytes = _save_upload(uploaded, src)
            except Exception as e:  # noqa: BLE001
                status.update(label="Save failed", state="error", expanded=True)
                st.error(f"Could not save upload: {e}")
                return
            st.write(f"Saved {nbytes / (1024 * 1024):.1f}MB")

            video_for_pipeline = src
            if nbytes >= COMPRESS_ABOVE_MB * 1024 * 1024:
                st.write(f"Compressing for demo (source ≥ {COMPRESS_ABOVE_MB}MB)…")
                compact = work / "demo_input.mp4"
                try:
                    _compress_for_demo(src, compact)
                    video_for_pipeline = compact
                    st.write(
                        f"Compressed to {compact.stat().st_size / (1024 * 1024):.1f}MB"
                    )
                except Exception as e:  # noqa: BLE001
                    status.update(label="Compress failed", state="error", expanded=True)
                    st.error(str(e))
                    return

            st.write("Sampling keyframes → describe/verify → style captions")
            try:
                caps, info = _caption(video_for_pipeline)
            except Exception as e:  # noqa: BLE001
                status.update(label="Pipeline error", state="error", expanded=True)
                st.error(str(e))
                return
            if info.get("error"):
                status.update(label="No output", state="error")
                st.error(info["error"])
                return
            status.update(label="Done", state="complete", expanded=False)

        st.session_state["last_caps"] = caps
        st.session_state["last_info"] = info
        _render_results(caps, info)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render_results(caps: dict[str, str], info: dict) -> None:
    st.markdown(
        f"""
        <div class="stat-wrap">
          <div class="stat"><div class="k">Elapsed</div><div class="v">{info.get('elapsed',0):.0f}s</div></div>
          <div class="stat"><div class="k">Stage</div><div class="v">{info.get('stage','?')}</div></div>
          <div class="stat"><div class="k">Keyframes</div><div class="v">{info.get('keyframes') or '—'}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    r1 = st.columns(2)
    r2 = st.columns(2)
    for cell, (key, name, _) in zip([r1[0], r1[1], r2[0], r2[1]], STYLES):
        text = (caps.get(key) or "—").replace("<", "&lt;").replace(">", "&gt;")
        with cell:
            st.markdown(
                f'<div class="cap"><div class="k">{name}</div><div class="t">{text}</div></div>',
                unsafe_allow_html=True,
            )
    desc = info.get("description") or ""
    if desc:
        with st.expander("Grounding description"):
            st.write(desc)


if __name__ == "__main__":
    main()
