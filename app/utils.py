"""Small, dependency-light helpers shared across the app.

Kept free of Twikit/pydantic imports so it can be unit tested in isolation.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from app.exceptions import NetworkTimeoutError, RateLimitError, ScraperError, TransientRequestError

T = TypeVar("T")

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")

#: Exceptions worth retrying: transient/network/rate-limit conditions.
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RateLimitError,
    NetworkTimeoutError,
    TransientRequestError,
)


def normalize_username(raw: str) -> str:
    """Strip whitespace, a leading '@', and surrounding slashes/URLs.

    Accepts plain handles ("elonmusk"), "@elonmusk", or profile URLs
    ("https://x.com/elonmusk") and returns the bare handle.
    """
    value = raw.strip()
    if not value:
        return value
    value = value.rstrip("/")
    if "://" in value or value.startswith("x.com/") or value.startswith("twitter.com/"):
        value = value.rstrip("/").split("/")[-1]
    value = value.lstrip("@")
    return value


def is_valid_username(username: str) -> bool:
    """Validate against X's handle rules: 1-15 chars, alphanumeric + underscore."""
    return bool(_USERNAME_RE.match(username))


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """De-duplicate case-insensitively while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def read_usernames_from_file(path: Path) -> list[str]:
    """Read one username per line from a text file, ignoring blanks and '#' comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    usernames = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        usernames.append(normalize_username(stripped))
    return usernames


def split_errors_by_stage(errors: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a `CategoryReport.errors` list into `(validation_errors,
    pipeline_errors)` by each error's `"stage"` field.

    "validation"-stage errors are candidate accounts rejected during
    discovery/validation, before ranking/selection ever runs (e.g. a
    discovered handle that doesn't exist or is protected) - a normal,
    expected part of dynamic discovery, not a failure of the final
    selected accounts. `app/category_agent.py`'s `accounts_failed`/
    `accounts_rate_limited`/`accounts_failed_other` stats already only
    count "tweets"-stage errors; this split lets callers report the two
    kinds distinctly instead of conflating them into one "N accounts
    failed" count. Any non-"validation" stage (currently just "tweets")
    is treated as a pipeline error, never silently dropped.
    """
    validation_errors = [e for e in errors if e.get("stage") == "validation"]
    pipeline_errors = [e for e in errors if e.get("stage") != "validation"]
    return validation_errors, pipeline_errors


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_seconds: float = 2.0,
    max_seconds: float = 30.0,
    rate_limit_base_seconds: float | None = None,
    rate_limit_max_seconds: float | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Run `func`, retrying on `retryable_exceptions` with exponential backoff + jitter.

    Attempt 0 is the initial call; up to `max_retries` further attempts follow.
    Delay grows as `base_seconds * 2**attempt`, capped at `max_seconds`, with
    up to 25% random jitter to avoid thundering-herd retries.

    `RateLimitError` uses a separate, much longer `rate_limit_base_seconds`/
    `rate_limit_max_seconds` track instead (falling back to `base_seconds`/
    `max_seconds` if not given) - ordinary transient errors clear in seconds,
    but X's rate-limit windows are commonly ~15 minutes, so retrying a
    rate-limited request on the fast track just burns retries for nothing.
    If `exc.retry_after` is populated (X's own reported reset time), it takes
    precedence over the exponential guess, capped at the rate-limit max as a
    sanity bound. Jitter is still applied on top even then: many accounts
    hitting the same shared rate-limit window would otherwise all compute the
    *same* reset time and retry in perfect lockstep the instant it lapses.
    """
    attempt = 0
    while True:
        try:
            return await func()
        except retryable_exceptions as exc:
            if attempt >= max_retries:
                raise
            if isinstance(exc, RateLimitError):
                rl_base = (
                    rate_limit_base_seconds if rate_limit_base_seconds is not None else base_seconds
                )
                rl_max = (
                    rate_limit_max_seconds if rate_limit_max_seconds is not None else max_seconds
                )
                delay = min(rl_base * (2**attempt), rl_max)
                if exc.retry_after:
                    delay = min(max(delay, exc.retry_after), rl_max)
            else:
                delay = min(base_seconds * (2**attempt), max_seconds)
            delay += random.uniform(0, delay * 0.25)
            if on_retry:
                on_retry(attempt + 1, exc, delay)
            await asyncio.sleep(delay)
            attempt += 1
        except ScraperError:
            raise
