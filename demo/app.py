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
        return "(no verified facts yet — draft stage or empty sheet)"
    return "\n".join(
        f"[{f.get('category', '?')}] {f.get('text', '')}  (support {f.get('support', '?')})"
        for f in verified[:20]
    )


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
    st.set_page_config(page_title="SEV-Cap Demo", layout="wide")
    st.title("SEV-Cap")
    st.caption(
        "Semantic-entropy verified multi-style video captions. "
        "Upload a short mp4 — the real pipeline runs (draft first, then SEV upgrade)."
    )

    uploaded = st.file_uploader("Video (mp4)", type=["mp4", "mov", "mkv", "webm"])
    run = st.button("Generate captions", type="primary", disabled=uploaded is None)

    if run and uploaded is not None:
        tmp = Path(tempfile.mkdtemp(prefix="sevcap_up_")) / uploaded.name
        tmp.write_bytes(uploaded.getvalue())
        with st.spinner("Running SEV-Cap pipeline (often 1–3 minutes)…"):
            try:
                caps, status, facts = asyncio.run(_caption(tmp))
            except Exception as e:  # noqa: BLE001
                st.error(f"Error: {e}")
                return
            finally:
                shutil.rmtree(tmp.parent, ignore_errors=True)

        st.success(status)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Formal")
            st.write(caps.get("formal") or "—")
            st.subheader("Sarcastic")
            st.write(caps.get("sarcastic") or "—")
        with c2:
            st.subheader("Humorous (tech)")
            st.write(caps.get("humorous_tech") or "—")
            st.subheader("Humorous (non-tech)")
            st.write(caps.get("humorous_non_tech") or "—")
        with st.expander("Verified facts (if SEV upgrade finished)"):
            st.text(facts)


if __name__ == "__main__":
    main()
