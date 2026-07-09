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

DEMO_BUDGET_S = float(os.environ.get("SEVCAP_DEMO_BUDGET", "240"))
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


def _format_facts(meta: dict) -> str:
    fv = (meta or {}).get("fact_verification") or {}
    verified = fv.get("verified_facts") or []
    if not verified:
        return ""
    
    lines = []
    for f in verified[:20]:
        cat = f.get('category', '?')
        text = f.get('text', '')
        support = f.get('support', '?')
        lines.append(f"- **{cat}**: {text} *(Support: {support})*")
    return "\n".join(lines)


async def _caption(video_path: Path) -> tuple[dict[str, str], str, str]:
    work = Path(tempfile.mkdtemp(prefix="sevcap_demo_"))
    try:
        dest = work / f"upload{video_path.suffix or '.mp4'}"
        shutil.copy2(video_path, dest)
        llm = _get_llm()
        writer = ResultWriter(OUT_DIR)
        deadline = Deadline(DEMO_BUDGET_S)
        t0 = time.monotonic()
        result = await process_clip(llm, dest, writer, deadline)
        elapsed = time.monotonic() - t0
        rec_path = OUT_DIR / f"{dest.stem}.json"
        if not rec_path.exists():
            return {}, f"No output written ({result})", ""
        rec = json.loads(rec_path.read_text())
        caps = rec.get("captions") or {}
        meta = rec.get("verification") or {}
        stage = meta.get("stage") or result.get("stage", "?")
        status = (
            f"Done in {elapsed:.0f}s · stage={stage} · "
            f"budget={DEMO_BUDGET_S:.0f}s"
        )
        return caps, status, _format_facts(meta)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    st.set_page_config(page_title="SEV-Cap Demo", page_icon="🎬", layout="wide")
    
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🎬 SEV-Cap")
    st.markdown("### Semantic-Entropy Verified Video Captions")
    st.caption(
        "Upload a short video and let our pipeline generate multi-style captions. "
        "We perform draft extraction followed by semantic-entropy verification to ensure accuracy."
    )
    st.markdown("---")

    uploaded = st.file_uploader("Upload Video (mp4, mov, mkv, webm)", type=["mp4", "mov", "mkv", "webm"])
    run = st.button("✨ Generate Captions", type="primary", disabled=uploaded is None, use_container_width=True)

    if run and uploaded is not None:
        tmp = Path(tempfile.mkdtemp(prefix="sevcap_up_")) / uploaded.name
        tmp.write_bytes(uploaded.getvalue())
        
        with st.status("Running SEV-Cap pipeline...", expanded=True) as status:
            st.write("⏳ Uploading and initializing...")
            st.write("🔍 Extracting frames and running semantic-entropy verification...")
            st.write("✨ Generating multi-style captions...")
            st.write("*(This process typically takes 1–3 minutes)*")
            try:
                caps, status_str, facts = asyncio.run(_caption(tmp))
                status.update(label="Pipeline finished successfully!", state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001
                status.update(label="Pipeline encountered an error.", state="error", expanded=True)
                st.error(f"Error: {e}")
                return
            finally:
                shutil.rmtree(tmp.parent, ignore_errors=True)

        st.success(f"✅ {status_str}")
        st.markdown("### 📝 Generated Captions")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**👔 Formal**")
            st.info(caps.get("formal") or "—")
            st.markdown("**😏 Sarcastic**")
            st.info(caps.get("sarcastic") or "—")
        with c2:
            st.markdown("**🤓 Humorous (Tech)**")
            st.info(caps.get("humorous_tech") or "—")
            st.markdown("**😂 Humorous (Non-Tech)**")
            st.info(caps.get("humorous_non_tech") or "—")
            
        with st.expander("🔍 View Verification Report"):
            if not facts:
                st.info("No verified facts available (draft stage or empty sheet).")
            else:
                st.markdown(facts)


if __name__ == "__main__":
    main()
