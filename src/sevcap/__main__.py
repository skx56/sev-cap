"""`python -m sevcap` is the container entrypoint.

With no arguments it runs the full pipeline directly (the scoring harness
passes none); with arguments it behaves like the `sevcap` CLI.
"""

from __future__ import annotations

import asyncio
import sys

if len(sys.argv) > 1:
    from .cli import main

    main()
else:
    from .pipeline import run

    asyncio.run(run(None, None))
