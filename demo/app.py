"""Streamlit demo: upload an mp4, get 4 SEV-Cap captions from the real pipeline.

Launch:
  streamlit run demo/app.py --server.port 7860 --server.headless true
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

# Load .env if present (demo convenience; Docker/harness uses real env).
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from sevcap.fireworks import Gemma  # noqa: E402
from sevcap.io_contract import ResultWriter  # noqa: E402
from sevcap.pipeline import Deadline, process_clip  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("sevcap.demo")

DEMO_BUDGET_S = float(os.environ.get("SEVCAP_DEMO_BUDGET", "600"))
OUT_DIR = Path(__file__).resolve().parent / "out" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def _get_llm() -> Gemma:
    llm = Gemma()
    try:
        asyncio.run(llm.resolve_text_model())
        asyncio.run(llm.check_vision())
    except Exception as e:  # noqa: BLE001
        log.warning("model warm-up failed (will retry on request): %s", e)
    return llm


async def _caption(video_path: Path) -> tuple[dict[str, str], dict]:
    from sevcap.pipeline import ClipJob
    from sevcap.styles import STYLE_ORDER

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
        stage = meta.get("stage") or result.get("stage", "?")
        info = {
            "elapsed": elapsed,
            "stage": stage,
            "budget": DEMO_BUDGET_S,
            "description": (meta.get("grounding_description") or "")[:600],
            "keyframes": meta.get("keyframes"),
        }
        return caps, info
    finally:
        shutil.rmtree(work, ignore_errors=True)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

#MainMenu, header, footer {visibility: hidden;}

.stApp {
    background:
        radial-gradient(1100px 600px at 12% -10%, rgba(139,92,246,0.20), transparent 55%),
        radial-gradient(900px 600px at 88% 0%, rgba(236,72,153,0.16), transparent 55%),
        radial-gradient(800px 700px at 50% 120%, rgba(34,211,238,0.12), transparent 55%),
        #0b0b14;
    background-attachment: fixed;
}
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1150px; }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 3.4rem;
    line-height: 1.05;
    margin: 0;
    background: linear-gradient(100deg, #a78bfa 0%, #ec4899 45%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 1.15rem;
    color: #b7b7c9;
    margin-top: .35rem;
}
.hero-desc { color: #8b8ba3; font-size: .95rem; max-width: 720px; margin-top: .6rem; }

.pill-row { display: flex; gap: .5rem; flex-wrap: wrap; margin: 1rem 0 .4rem; }
.pill {
    font-family: 'Space Grotesk', sans-serif;
    font-size: .78rem; font-weight: 600;
    padding: .34rem .8rem; border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.04);
    color: #cfcfe2; backdrop-filter: blur(6px);
}
.pill.violet { border-color: rgba(139,92,246,0.5); color: #c4b5fd; }
.pill.pink   { border-color: rgba(236,72,153,0.5); color: #f9a8d4; }
.pill.cyan   { border-color: rgba(34,211,238,0.5); color: #67e8f9; }

/* the 4 output styles, explained up front */
.style-grid {
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: .7rem; margin: 1.15rem 0 .3rem;
}
.style-tile {
    display: flex; gap: .7rem; align-items: flex-start;
    padding: .8rem .95rem; border-radius: 14px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid var(--accent);
    transition: transform .15s ease, background .15s ease;
}
.style-tile:hover { transform: translateY(-2px); background: rgba(255,255,255,0.06); }
.st-name {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    font-size: .95rem; color: var(--accent);
}
.st-desc { color: #9a9ab0; font-size: .84rem; margin-top: .12rem; }
@media (max-width: 720px) { .style-grid { grid-template-columns: 1fr; } }

/* caption cards */
.cap-card {
    position: relative; border-radius: 18px; padding: 1.15rem 1.25rem 1.25rem;
    background: linear-gradient(160deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    height: 100%; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    overflow: hidden;
}
.cap-card::before {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
    background: var(--accent);
}
.cap-card:hover { transform: translateY(-4px); border-color: rgba(255,255,255,0.16);
    box-shadow: 0 18px 44px rgba(0,0,0,0.5); }
.cap-head { display:flex; align-items:center; gap:.55rem; margin-bottom:.55rem; }
.cap-name {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: .82rem;
    letter-spacing: .08em; text-transform: uppercase; color: var(--accent);
}
.cap-body { color: #ececf5; font-size: 1.02rem; line-height: 1.5; }

/* stat chips */
.stat-wrap { display:flex; gap:.7rem; flex-wrap: wrap; margin: .2rem 0 1.2rem; }
.stat {
    border-radius: 14px; padding: .7rem 1.1rem; min-width: 120px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
}
.stat .k { font-size: .72rem; color:#8b8ba3; text-transform: uppercase; letter-spacing:.06em; }
.stat .v { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3rem; color:#fff; }

/* fact chips */
.fact { display:inline-block; margin:.2rem .3rem .2rem 0; padding:.32rem .7rem;
    border-radius: 10px; font-size:.85rem; background: rgba(34,197,94,0.10);
    border:1px solid rgba(34,197,94,0.30); color:#bbf7d0; }
.fact .cat { color:#86efac; font-weight:600; font-size:.72rem; text-transform:uppercase; margin-right:.35rem; }
.fact .sup { color:#6ee7b7; opacity:.8; font-size:.72rem; margin-left:.3rem; }
.rej { display:inline-block; margin:.2rem .3rem .2rem 0; padding:.32rem .7rem;
    border-radius: 10px; font-size:.85rem; background: rgba(244,63,94,0.08);
    border:1px solid rgba(244,63,94,0.28); color:#fecdd3; text-decoration: line-through; opacity:.85; }

.stButton > button {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem;
    border-radius: 14px; border: none; padding: .7rem 1rem;
    background: linear-gradient(100deg, #8b5cf6, #ec4899);
    color: white; transition: transform .15s ease, box-shadow .15s ease;
    box-shadow: 0 8px 24px rgba(139,92,246,0.35);
}
.stButton > button:hover:not(:disabled) { transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(236,72,153,0.45); }
.stButton > button:disabled { opacity: .4; }

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03); border: 1.5px dashed rgba(139,92,246,0.4);
    border-radius: 16px;
}
hr { border-color: rgba(255,255,255,0.08); }
</style>
"""

STYLES = [
    ("formal", "Formal", "#a78bfa"),
    ("sarcastic", "Sarcastic", "#ec4899"),
    ("humorous_tech", "Humorous · Tech", "#22d3ee"),
    ("humorous_non_tech", "Humorous · Non-Tech", "#f59e0b"),
]


def _cap_card(key: str, name: str, accent: str, text: str) -> str:
    body = (text or "—").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <div class="cap-card" style="--accent:{accent}">
      <div class="cap-head">
        <span class="cap-name">{name}</span>
      </div>
      <div class="cap-body">{body}</div>
    </div>
    """


def main() -> None:
    st.set_page_config(page_title="SEV-Cap", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown('<div class="hero-title">SEV-Cap</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Semantic-Entropy Verified Video Captions</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-desc">Upload a short clip and SEV-Cap writes '
        "<b>four captions</b> for it &mdash; one in each style below. Every caption is "
        "built only from facts that pass our accuracy check, so it describes what is "
        "actually in the video, not what the model guesses.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="style-grid">'
        '<div class="style-tile" style="--accent:#a78bfa">'
        '<div><div class="st-name">Formal</div>'
        '<div class="st-desc">Clear, professional description of the scene.</div></div></div>'
        '<div class="style-tile" style="--accent:#ec4899">'
        '<div><div class="st-name">Sarcastic</div>'
        '<div class="st-desc">Dry, witty take with a bit of attitude.</div></div></div>'
        '<div class="style-tile" style="--accent:#22d3ee">'
        '<div><div class="st-name">Humorous · Tech</div>'
        '<div class="st-desc">Playful joke with a nerdy / tech twist.</div></div></div>'
        '<div class="style-tile" style="--accent:#f59e0b">'
        '<div><div class="st-name">Humorous · Non-Tech</div>'
        '<div class="st-desc">Light, everyday humor anyone gets.</div></div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    col_up, col_prev = st.columns([1.1, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drop a video", type=["mp4", "mov", "mkv", "webm"], label_visibility="collapsed"
        )
        run = st.button(
            "Generate Captions",
            type="primary",
            disabled=uploaded is None,
            use_container_width=True,
        )
    with col_prev:
        if uploaded is not None:
            st.video(uploaded)
        else:
            st.markdown(
                '<div style="color:#6b6b83;padding:1.2rem 0;font-size:.9rem;">'
                "Preview appears here once you pick a file.</div>",
                unsafe_allow_html=True,
            )

    if run and uploaded is not None:
        tmp = Path(tempfile.mkdtemp(prefix="sevcap_up_")) / uploaded.name
        tmp.write_bytes(uploaded.getvalue())

        with st.status("Running SEV-Cap pipeline…", expanded=True) as status:
            st.write("Sampling keyframes")
            st.write("Describe → verify scene")
            st.write("Multi-candidate styled captions + polish")
            st.caption("Typically 1–3 minutes.")
            try:
                caps, info = asyncio.run(_caption(tmp))
                if info.get("error"):
                    status.update(label="No output produced.", state="error")
                    st.error(info["error"])
                    return
                status.update(label="Done!", state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001
                status.update(label="Pipeline error.", state="error", expanded=True)
                st.error(f"Error: {e}")
                return
            finally:
                shutil.rmtree(tmp.parent, ignore_errors=True)

        stage = info.get("stage", "?")
        kf = info.get("keyframes") or "?"
        st.markdown(
            f"""
            <div class="stat-wrap">
              <div class="stat"><div class="k">Elapsed</div><div class="v">{info.get('elapsed',0):.0f}s</div></div>
              <div class="stat"><div class="k">Stage</div><div class="v">{stage}</div></div>
              <div class="stat"><div class="k">Keyframes</div><div class="v">{kf}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r1 = st.columns(2)
        r2 = st.columns(2)
        cells = [r1[0], r1[1], r2[0], r2[1]]
        for cell, (key, name, accent) in zip(cells, STYLES):
            with cell:
                st.markdown(
                    _cap_card(key, name, accent, caps.get(key)),
                    unsafe_allow_html=True,
                )

        desc = info.get("description") or ""
        if desc:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            with st.expander("Grounding description", expanded=False):
                st.write(desc)


if __name__ == "__main__":
    main()
