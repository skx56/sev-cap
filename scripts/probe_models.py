"""Probe which candidate models this Fireworks account can actually reach.

Usage: .venv/bin/python scripts/probe_models.py
Reads FIREWORKS_API_KEY from env or .env. Prints one line per candidate:
model id, reachable yes/no, and whether image input works.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if not os.environ.get("FIREWORKS_API_KEY"):
    env = Path(__file__).parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FIREWORKS_API_KEY="):
                os.environ["FIREWORKS_API_KEY"] = line.split("=", 1)[1].strip()

from openai import AsyncOpenAI  # noqa: E402

CANDIDATES = [
    # Gemma paths under various accounts/naming schemes
    "accounts/fireworks/models/gemma-3-27b-it",
    "accounts/fireworks/models/gemma-3-12b-it",
    "accounts/fireworks/models/gemma-3-4b-it",
    "accounts/google/models/gemma-3-27b-it",
    "accounts/fireworks/models/gemma-3n-e4b-it",
    "accounts/fireworks/models/gemma-2-9b-it",
    # VLMs listed as serverless on this account
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/kimi-k2p5",
    # other text models on the serverless list
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/glm-5p1",
]

TINY_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAAB"
    "AAAAAAAAAAAAAAAAAAAACv/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="
)


async def main() -> None:
    client = AsyncOpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai/inference/v1",
        timeout=30, max_retries=0,
    )
    for model in CANDIDATES:
        try:
            await client.chat.completions.create(
                model=model, max_tokens=5,
                messages=[{"role": "user", "content": "say OK"}],
            )
            text_ok = "text OK"
        except Exception as e:  # noqa: BLE001
            print(f"{model:55} UNREACHABLE ({type(e).__name__}: {str(e)[:80]})")
            continue
        try:
            await client.chat.completions.create(
                model=model, max_tokens=5,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "say OK"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{TINY_JPEG}"}},
                    ],
                }],
            )
            print(f"{model:55} {text_ok}, VISION OK")
        except Exception as e:  # noqa: BLE001
            print(f"{model:55} {text_ok}, vision NO ({str(e)[:80]})")


if __name__ == "__main__":
    asyncio.run(main())
