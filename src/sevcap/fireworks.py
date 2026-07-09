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
from collections import Counter
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


class DegenerateOutputError(RuntimeError):
    """Raised when the model falls into a decoding repetition loop.

    Observed occasionally on gemma-4-26b-a4b-it at temperature>=0.7 ("too
    many too many too many..."). The API call itself succeeds (200 OK), so
    this must be caught after the fact and treated as a retriable failure —
    never cached, never trusted.
    """


def _is_degenerate(text: str) -> bool:
    """Cheap repetition-loop detector, scanned over the whole response.

    Catches shapes of decoding loop seen on this checkpoint: a single
    token/word repeated back-to-back many times ("de-identified de-identified
    ..."), a short phrase cycling ("too many too many too many..."), and a
    hyphen-chained repeat ("top-level-level-level-level-level...").
    """
    words = re.split(r"[\s-]+", text)
    if len(words) < 12:
        return False
    max_run = run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    if max_run >= 6:
        return True
    grams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if not grams:
        return False
    top = Counter(grams).most_common(1)[0][1]
    return top / len(grams) > 0.15


def _has_parseable_json_tail(text: str) -> bool:
    try:
        extract_json(text)
        return True
    except Exception:  # noqa: BLE001
        return False


def _cache_key(model: str, messages: list[dict], kwargs: dict) -> str:
    payload = json.dumps({"m": model, "msgs": messages, "kw": kwargs}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_json(text: str) -> Any:
    """Leniently pull JSON out of a model response.

    Reasoning models often think out loud before answering, so when several
    JSON-looking spans exist we prefer the LAST parseable one (the answer).
    """
    text = text.strip()
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for fenced in reversed(fences):
        try:
            return json.loads(fenced.strip())
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    spans: list[tuple[int, int, Any]] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        pos = 0
        while (start := text.find(opener, pos)) != -1:
            depth = 0
            end = -1
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                break
            try:
                spans.append((start, end, json.loads(text[start : end + 1])))
            except json.JSONDecodeError:
                pass
            pos = start + 1
    # drop spans nested inside another parseable span, keep the last top-level
    top = [s for s in spans
           if not any(o[0] <= s[0] and s[1] <= o[1] and o != s for o in spans)]
    if top:
        return max(top, key=lambda s: s[0])[2]
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
        self._text_model_resolved: str | None = None
        self._reasoning_supported: bool | None = None

    async def resolve_text_model(self) -> str:
        """Pick the first reachable text model (primary, then fallback).

        Keeps Gemma as the default wherever it is deployed, while letting the
        same container run on accounts where Gemma is not serverless.
        """
        if self._text_model_resolved:
            return self._text_model_resolved
        for m in (settings.model, settings.fallback_model):
            try:
                await self.chat(
                    [{"role": "user", "content": "Reply with OK."}],
                    model=m, temperature=0.0, max_tokens=5, tag="resolve", cache=False,
                )
                if m != settings.model:
                    log.warning("primary model %s unreachable; using %s", settings.model, m)
                self._text_model_resolved = m
                return m
            except Exception as e:  # noqa: BLE001
                log.warning("model %s unreachable: %s", m, str(e)[:120])
        raise RuntimeError("No reachable text model (primary or fallback)")

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
        reasoning: str | None = "none",
    ) -> str:
        model = model or self._text_model_resolved or settings.model
        kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            kwargs["seed"] = seed
        # Reasoning models burn most of the budget thinking; "none"/"low" keeps
        # judge calls fast and cheap. Silently dropped if the model rejects it.
        if reasoning and self._reasoning_supported is not False:
            kwargs["extra_body"] = {"reasoning_effort": reasoning}

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
        for attempt in range(8):  # 429 storms need patience, not failure
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(
                        model=model, messages=messages, **kwargs
                    )
                content = resp.choices[0].message.content or ""
                # Only treat repetition as fatal if there is no salvageable
                # JSON at all — a verbose "thinking" preamble can legitimately
                # repeat a phrase while still ending in a perfectly good
                # answer, and rejecting those hurts more than it helps.
                if _is_degenerate(content) and not _has_parseable_json_tail(content):
                    raise DegenerateOutputError(
                        f"repetition loop with no recoverable JSON (tail: {content[-80:]!r})"
                    )
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
                # Model rejects reasoning_effort -> drop it and retry once.
                if "reasoning" in msg and "extra_body" in kwargs:
                    self._reasoning_supported = False
                    kwargs.pop("extra_body", None)
                    continue
                # A fixed seed reproduces a decoding loop deterministically;
                # retrying identically would just hit the same loop again.
                if isinstance(e, DegenerateOutputError) and "seed" in kwargs:
                    kwargs["seed"] = kwargs["seed"] + 7919 * (attempt + 1)
                # 4xx will not fix itself, except 429 (rate limit) and 401:
                # Fireworks intermittently returns spurious 401s under load,
                # so give auth errors the full backoff before giving up.
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500 and status not in (401, 429):
                    raise
                if not isinstance(e, RETRIABLE) and "429" not in msg and "500" not in msg:
                    # One reseeded retry for a decoding loop; if the model is
                    # still looping after that, stop burning time here and
                    # let vision_chat's candidate fallback take over instead.
                    max_attempt = 1
                    if attempt >= max_attempt:
                        raise
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, str(e)[:150])
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
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
        reasoning: str | None = "low",
        cache: bool | None = None,
    ) -> str:
        """Vision call with per-call fallback: a model that degenerates on one
        call does not get permanently trusted for the rest of the run.

        Earlier this locked to whichever model answered check_vision() once
        and never reconsidered — so a model that is reachable but unstable
        (e.g. decoding loops on heavy multi-image prompts) kept getting
        retried in isolation for the whole run instead of ever falling
        through to the working alternative.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64 in image_b64_list:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        # Try whichever model most recently worked first (cheap optimization),
        # but always keep the other candidate available this call too.
        primary = self._vision_model_resolved or settings.vision_model
        candidates = [primary]
        if settings.fallback_vision_model and settings.fallback_vision_model != primary:
            candidates.append(settings.fallback_vision_model)
        if primary != settings.vision_model and settings.vision_model not in candidates:
            candidates.append(settings.vision_model)

        last_err: Exception | None = None
        for i, m in enumerate(candidates):
            try:
                out = await self.chat(
                    messages,
                    model=m,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    tag=tag,
                    reasoning=reasoning,
                    cache=cache,
                )
                self._vision_model_resolved = m
                return out
            except Exception as e:  # noqa: BLE001
                is_last = i == len(candidates) - 1
                # Only a hard "no image support" failure is worth giving up
                # on immediately without trying remaining candidates.
                if isinstance(e, VisionNotSupportedError) and not is_last:
                    log.warning("Vision model %s rejects images, trying next: %s", m, str(e)[:120])
                    last_err = e
                    continue
                if is_last:
                    raise
                log.warning("Vision model %s unusable this call, falling back: %s", m, str(e)[:120])
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
