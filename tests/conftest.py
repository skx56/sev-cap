import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("SEVCAP_CACHE", "0")


class FakeGemma:
    """Scripted Gemma stand-in: pops queued responses, records prompts."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def queue(self, *responses):
        self.responses.extend(responses)
        return self

    async def chat(self, messages, **kwargs):
        self.calls.append(("chat", messages, kwargs))
        if not self.responses:
            raise AssertionError("FakeGemma ran out of scripted responses")
        return self.responses.pop(0)

    async def vision_chat(self, prompt, images, **kwargs):
        self.calls.append(("vision", prompt, kwargs))
        if not self.responses:
            raise AssertionError("FakeGemma ran out of scripted responses")
        return self.responses.pop(0)


@pytest.fixture
def fake_llm():
    return FakeGemma()


@pytest.fixture(scope="session")
def synthetic_clip(tmp_path_factory):
    """A 35s synthetic test video generated with ffmpeg (no network needed)."""
    dest = tmp_path_factory.mktemp("clips") / "synthetic.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=35:size=320x240:rate=10",
         "-pix_fmt", "yuv420p", "-y", str(dest)],
        check=True,
    )
    return str(dest)
