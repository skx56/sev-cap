"""Quick single-clip smoke test of the refactored pipeline (not a permanent script)."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sevcap.fireworks import Gemma  # noqa: E402
from sevcap.io_contract import ResultWriter  # noqa: E402
from sevcap.pipeline import Deadline, process_clip  # noqa: E402


async def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "clips/ed_machine_90s.mp4"
    llm = Gemma()
    writer = ResultWriter(Path("results_smoke"))
    deadline = Deadline(600.0)
    t0 = time.monotonic()
    await llm.resolve_text_model()
    await llm.check_vision()
    result = await process_clip(llm, Path(clip), writer, deadline)
    print("result:", result)
    print("elapsed:", time.monotonic() - t0)
    print("usage:", llm.usage.summary())


if __name__ == "__main__":
    asyncio.run(main())
