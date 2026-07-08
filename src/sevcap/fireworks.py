"""Async Fireworks AI client wrapper.

Features: exponential-backoff retries, on-disk response cache (so eval reruns
are free and protect the credit budget), global concurrency limit, call/token
accounting, lenient JSON extraction, and automatic vision-model fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from .config import settings

log = logging.getLogger("sevcap.fireworks")

RETRIABLE = (RateLimitError, APITimeoutError, APIError, ConnectionError, asyncio.TimeoutError)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    errors: int = 0
    per_tag: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "errors": self.errors,
            "per_tag": dict(self.per_tag),
        }


class VisionNotSupportedError(RuntimeError):
    """Raised when the model rejects image input, triggering VLM fallback."""


def _cache_key(model: str, messages: list[dict], kwargs: dict) -> str:
    payload = json.dumps({"m": model, "msgs": messages, "kw": kwargs}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_json(text: str) -> Any:
    """Leniently pull the first JSON object/array out of a model response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No parseable JSON in model response: {text[:300]!r}")


class Gemma:
    """Thin async wrapper for Gemma 3 (and fallback VLM) on Fireworks."""

    def __init__(self, api_key: str | None = None):
        self.client = AsyncOpenAI(
            api_key=api_key or settings.api_key or "MISSING",
            base_url=settings.base_url,
            timeout=120.0,
            max_retries=0,  # we do our own retries
        )
        self.usage = Usage()
        self._sem = asyncio.Semaphore(settings.llm_concurrency)
        self._vision_model_resolved: str | None = None

    # ------------------------------------------------------------------ core

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 1024,
        seed: int | None = None,
        tag: str = "general",
        cache: bool | None = None,
    ) -> str:
        model = model or settings.model
        kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            kwargs["seed"] = seed

        use_cache = settings.cache_enabled if cache is None else cache
        key = _cache_key(model, messages, kwargs)
        cache_path = os.path.join(settings.cache_dir, f"{key}.json")
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    self.usage.cache_hits += 1
                    return json.load(f)["content"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        delay = 2.0
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(
                        model=model, messages=messages, **kwargs
                    )
                content = resp.choices[0].message.content or ""
                self.usage.calls += 1
                self.usage.per_tag[tag] = self.usage.per_tag.get(tag, 0) + 1
                if resp.usage:
                    self.usage.prompt_tokens += resp.usage.prompt_tokens or 0
                    self.usage.completion_tokens += resp.usage.completion_tokens or 0
                if use_cache:
                    os.makedirs(settings.cache_dir, exist_ok=True)
                    tmp = cache_path + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump({"content": content}, f)
                    os.replace(tmp, cache_path)
                return content
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.usage.errors += 1
                msg = str(e).lower()
                # Non-retriable: model can't take images -> caller may fall back.
                if any(s in msg for s in ("image", "vision", "multimodal")) and any(
                    s in msg for s in ("not support", "unsupported", "invalid", "cannot")
                ):
                    raise VisionNotSupportedError(str(e)) from e
                # 4xx (except 429) will not fix itself: fail fast.
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                if not isinstance(e, RETRIABLE) and "429" not in msg and "500" not in msg:
                    if attempt >= 1:
                        raise
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    # ---------------------------------------------------------------- vision

    async def vision_chat(
        self,
        prompt: str,
        image_b64_list: list[str],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        seed: int | None = None,
        tag: str = "vision",
        system: str | None = None,
    ) -> str:
        """Vision call that resolves the working VLM once and sticks with it."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64 in image_b64_list:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        candidates = (
            [self._vision_model_resolved]
            if self._vision_model_resolved
            else [settings.vision_model, settings.fallback_vision_model]
        )
        last_err: Exception | None = None
        for m in candidates:
            try:
                out = await self.chat(
                    messages,
                    model=m,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    tag=tag,
                )
                self._vision_model_resolved = m
                return out
            except VisionNotSupportedError as e:
                log.warning("Model %s rejected image input, falling back: %s", m, e)
                last_err = e
        raise RuntimeError(f"No available vision model accepted image input: {last_err}")

    async def check_vision(self) -> str:
        """1-pixel smoke test; returns the model that accepted image input."""
        tiny = (
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
            "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAAB"
            "AAAAAAAAAAAAAAAAAAAACv/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="
        )
        await self.vision_chat(
            "Reply with the single word OK.", [tiny], temperature=0.0, tag="vision-check",
            max_tokens=10,
        )
        return self._vision_model_resolved or settings.vision_model
