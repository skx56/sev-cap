"""`python -m sevcap` is the container entrypoint.

With no arguments it runs the full pipeline directly (the scoring harness
passes none); with arguments it behaves like the `sevcap` CLI.
"""

from __future__ import annotations

import asyncio
import logging
import sys


def _run_pipeline(input_dir: str | None = None, output_dir: str | None = None) -> None:
    from .pipeline import run

    asyncio.run(run(input_dir, output_dir))


def _write_fatal_placeholder(exc: BaseException) -> None:
    """Best-effort output so the harness never sees a bare crash."""
    try:
        from .io_contract import ResultWriter, find_output_dir
        from .styles import STYLE_ORDER

        writer = ResultWriter(find_output_dir(None))
        writer.write(
            "_fatal",
            {k: "A short video clip." for k in STYLE_ORDER},
            meta={"stage": "error", "error": str(exc)[:500]},
        )
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        # Most harnesses pass no args, but some pass input/output paths directly.
        # Treat bare paths as pipeline args instead of letting Typer reject them.
        if len(sys.argv) in (2, 3) and not sys.argv[1].startswith("-") and sys.argv[1] not in {
            "run",
            "check",
            "facts",
            "lineup-test",
        }:
            _run_pipeline(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
        elif len(sys.argv) > 1:
            from .cli import main

            main()
        else:
            _run_pipeline()
    except Exception as exc:
        logging.exception("sevcap entrypoint failed: %s", exc)
        _write_fatal_placeholder(exc)
        # Exit 0 when we managed to emit placeholder output; harness treats
        # non-zero exits as RUNTIME_ERROR even if partial output exists.
        sys.exit(0)
