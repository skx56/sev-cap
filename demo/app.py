"""SEV-Cap demo — elegant dark Streamlit UI over the production pipeline.

Local:
  streamlit run demo/app.py --server.port 7860

Streamlit Cloud:
  Main file path: demo/app.py
  Secrets: FIREWORKS_API_KEY
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_secrets() -> None:
    # Streamlit Cloud secrets
    try:
        if "FIREWORKS_API_KEY" in st.secrets:
            os.environ.setdefault("FIREWORKS_API_KEY", str(st.secrets["FIREWORKS_API_KEY"]))
        for k, v in st.secrets.items():
            if isinstance(v, str) and k.startswith("SEVCAP_"):
                os.environ.setdefault(k, v)
    except Exception:  # noqa: BLE001
        pass
    # Local .env
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_secrets()

from sevcap.fireworks import Gemma  # noqa: E402
from sevcap.io_contract import ResultWriter  # noqa: E402
from sevcap.pipeline import ClipJob, Deadline, process_clip  # noqa: E402
from sevcap.styles import STYLE_ORDER  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("sevcap.demo")

DEMO_BUDGET_S = float(os.environ.get("SEVCAP_DEMO_BUDGET", "600"))
OUT_DIR = Path(__file__).resolve().parent / "out" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLES = [
    ("formal", "Formal", "Clear. Precise. Archival."),
    ("sarcastic", "Sarcastic", "Dry irony. Quiet bite."),
    ("humorous_tech", "Humorous · Tech", "One metaphor. On-screen facts."),
    ("humorous_non_tech", "Humorous · Non-Tech", "Warm. Everyday. No jargon."),
]


@st.cache_resource
def _get_llm() -> Gemma:
    llm = Gemma()
    try:
        asyncio.run(llm.resolve_text_model())
        asyncio.run(llm.check_vision())
    except Exception as e:  # noqa: BLE001
        log.warning("model warm-up failed: %s", e)
    return llm


async def _caption(video_path: Path) -> tuple[dict[str, str], dict]:
    work = Path(tempfile.mkdtemp(prefix="sevcap_demo_"))
    try:
        dest = work / f"upload{video_path.suffix or '.mp4'}"
        shutil.copy2(video_path, dest)
        llm = _get_llm()
        writer = ResultWriter(OUT_DIR, task_order=[dest.stem])
        deadline = Deadline(DEMO_BUDGET_S)
        job = ClipJob(task_id=dest.stem, styles=list(STYLE_ORDER), video=dest)
        t0 = time.monotonic()
        result = await process_clip(llm, job, writer, deadline)
        elapsed = time.monotonic() - t0
        rec_path = OUT_DIR / f"{dest.stem}.json"
        if not rec_path.exists():
            return {}, {"error": f"No output written ({result})"}
        rec = json.loads(rec_path.read_text())
        caps = rec.get("captions") or {}
        meta = rec.get("verification") or {}
        return caps, {
            "elapsed": elapsed,
            "stage": meta.get("stage") or result.get("stage", "?"),
            "description": (meta.get("grounding_description") or "")[:700],
            "keyframes": meta.get("keyframes"),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


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
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 0.95;
  color: #f4efe6;
  margin: 0 0 0.55rem 0;
}
.tag {
  display: inline-block;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #c9a25d;
  margin-bottom: 1.1rem;
}
.lede {
  max-width: 38rem;
  font-weight: 300;
  font-size: 1.08rem;
  line-height: 1.55;
  color: #a7a29a;
  margin-bottom: 1.8rem;
}
.panel {
  border: 1px solid rgba(236,231,223,0.10);
  background: rgba(255,255,255,0.025);
  border-radius: 18px;
  padding: 1.15rem 1.2rem;
  backdrop-filter: blur(8px);
}
.panel-label {
  font-size: 0.7rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #8b8680;
  margin-bottom: 0.75rem;
}
.style-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; margin: 1.2rem 0 1.6rem; }
.style-chip {
  border: 1px solid rgba(236,231,223,0.10);
  border-radius: 14px;
  padding: 0.85rem 0.9rem;
  background: rgba(255,255,255,0.02);
  min-height: 92px;
}
.style-chip .n {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.15rem;
  color: #f1ebe3;
  margin-bottom: 0.25rem;
}
.style-chip .d { font-size: 0.78rem; color: #8f8a83; line-height: 1.35; }

.cap {
  border: 1px solid rgba(236,231,223,0.10);
  border-radius: 16px;
  padding: 1.1rem 1.15rem 1.2rem;
  background: linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
  min-height: 168px;
  transition: border-color .2s ease, transform .2s ease;
}
.cap:hover { border-color: rgba(201,162,93,0.35); transform: translateY(-1px); }
.cap .k {
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #c9a25d;
  margin-bottom: 0.55rem;
}
.cap .t {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.28rem;
  line-height: 1.35;
  color: #f3eee6;
}

.stat-wrap { display:flex; gap:0.7rem; flex-wrap:wrap; margin: 0.4rem 0 1.1rem; }
.stat {
  border: 1px solid rgba(236,231,223,0.10);
  border-radius: 12px;
  padding: 0.65rem 0.9rem;
  background: rgba(255,255,255,0.02);
  min-width: 110px;
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
  box-shadow: 0 10px 28px rgba(184,137,63,0.18);
  transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 14px 34px rgba(184,137,63,0.28);
}
.stButton > button:disabled { opacity: 0.35 !important; }

[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px dashed rgba(236,231,223,0.22) !important;
  border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #a7a29a !important; }
[data-testid="stStatusWidget"], .stAlert { border-radius: 14px; }
hr { border: none; border-top: 1px solid rgba(236,231,223,0.08); margin: 1.4rem 0; }

@media (max-width: 900px) {
  .style-row { grid-template-columns: 1fr 1fr; }
}
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

    st.markdown('<div class="tag">Track 2 · Video Captioning</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand">SEV-Cap</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lede">Four grounded captions from one clip — formal, sarcastic, '
        "humorous-tech, humorous-non-tech — scored for accuracy and tone before they ship.</div>",
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
        )
        run = st.button("Generate captions", type="primary", disabled=uploaded is None)
    with right:
        st.markdown('<div class="panel-label">Preview</div>', unsafe_allow_html=True)
        if uploaded is not None:
            st.video(uploaded)
        else:
            st.markdown(
                '<div class="panel" style="min-height:180px;display:flex;align-items:center;'
                'color:#8b8680;font-weight:300;">Drop a short clip to begin.</div>',
                unsafe_allow_html=True,
            )

    if not (run and uploaded is not None):
        return

    tmp = Path(tempfile.mkdtemp(prefix="sevcap_up_")) / uploaded.name
    tmp.write_bytes(uploaded.getvalue())
    try:
        with st.status("Composing captions…", expanded=True) as status:
            st.write("Sampling keyframes")
            st.write("Describe → verify")
            st.write("Multi-candidate styles + vision prejudge")
            try:
                caps, info = asyncio.run(_caption(tmp))
            except Exception as e:  # noqa: BLE001
                status.update(label="Pipeline error", state="error", expanded=True)
                st.error(str(e))
                return
            if info.get("error"):
                status.update(label="No output", state="error")
                st.error(info["error"])
                return
            status.update(label="Done", state="complete", expanded=False)

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
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
