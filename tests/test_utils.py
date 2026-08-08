"""Unit tests for app.utils: username normalization/validation and retry/backoff."""

from __future__ import annotations

import pytest

from app.exceptions import NetworkTimeoutError, RateLimitError, ScraperError, UserNotFoundError
from app.utils import (
    dedupe_preserve_order,
    is_valid_username,
    normalize_username,
    read_usernames_from_file,
    retry_with_backoff,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("elonmusk", "elonmusk"),
        ("@elonmusk", "elonmusk"),
        ("  @elonmusk  ", "elonmusk"),
        ("https://x.com/elonmusk", "elonmusk"),
        ("https://x.com/elonmusk/", "elonmusk"),
        ("https://twitter.com/elonmusk", "elonmusk"),
        ("x.com/elonmusk", "elonmusk"),
    ],
)
def test_normalize_username(raw: str, expected: str) -> None:
    assert normalize_username(raw) == expected


@pytest.mark.parametrize(
    ("username", "valid"),
    [
        ("elonmusk", True),
        ("open_ai", True),
        ("a", True),
        ("", False),
        ("this_handle_is_too_long_16", False),
        ("has space", False),
        ("has-dash", False),
    ],
)
def test_is_valid_username(username: str, valid: bool) -> None:
    assert is_valid_username(username) is valid


def test_dedupe_preserve_order_case_insensitive() -> None:
    result = dedupe_preserve_order(["elonmusk", "OpenAI", "elonmusk", "openai"])
    assert result == ["elonmusk", "OpenAI"]


def test_read_usernames_from_file(tmp_path) -> None:
    path = tmp_path / "usernames.txt"
    path.write_text("elonmusk\n# a comment\n\n@openai\nsatyanadella\n", encoding="utf-8")

    result = read_usernames_from_file(path)

    assert result == ["elonmusk", "openai", "satyanadella"]


async def test_retry_with_backoff_succeeds_first_try() -> None:
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        return "ok"

    result = await retry_with_backoff(func, max_retries=3, base_seconds=0.01, max_seconds=0.02)

    assert result == "ok"
    assert calls == 1


async def test_retry_with_backoff_retries_then_succeeds() -> None:
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise NetworkTimeoutError("temporary")
        return "ok"

    result = await retry_with_backoff(func, max_retries=5, base_seconds=0.01, max_seconds=0.02)

    assert result == "ok"
    assert calls == 3


async def test_retry_with_backoff_exhausts_retries() -> None:
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        raise NetworkTimeoutError("always fails")

    with pytest.raises(NetworkTimeoutError):
        await retry_with_backoff(func, max_retries=2, base_seconds=0.01, max_seconds=0.02)

    assert calls == 3  # initial attempt + 2 retries


async def test_retry_with_backoff_does_not_retry_non_retryable() -> None:
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        raise UserNotFoundError("ghost")

    with pytest.raises(ScraperError):
        await retry_with_backoff(func, max_retries=3, base_seconds=0.01, max_seconds=0.02)

    assert calls == 1


def _capture_sleep(monkeypatch) -> list[float]:
    """Patch asyncio.sleep so backoff tests don't actually wait, while
    still recording exactly what delay retry_with_backoff computed."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("app.utils.asyncio.sleep", fake_sleep)
    return delays


async def test_retry_with_backoff_rate_limit_uses_separate_longer_track(monkeypatch) -> None:
    delays = _capture_sleep(monkeypatch)
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("slow down")
        return "ok"

    result = await retry_with_backoff(
        func,
        max_retries=3,
        base_seconds=0.01,
        max_seconds=0.02,
        rate_limit_base_seconds=30.0,
        rate_limit_max_seconds=900.0,
    )

    assert result == "ok"
    assert len(delays) == 1
    assert delays[0] >= 30.0  # the rate-limit track, not the 0.01-0.02 generic one


async def test_retry_with_backoff_honors_retry_after_within_cap(monkeypatch) -> None:
    delays = _capture_sleep(monkeypatch)

    async def func():
        raise RateLimitError("slow down", retry_after=120)

    with pytest.raises(RateLimitError):
        await retry_with_backoff(
            func, max_retries=1, rate_limit_base_seconds=30.0, rate_limit_max_seconds=900.0
        )

    assert len(delays) == 1
    assert 120.0 <= delays[0] <= 120.0 * 1.25  # honored, plus up to 25% jitter


async def test_retry_with_backoff_caps_retry_after_at_rate_limit_max(monkeypatch) -> None:
    delays = _capture_sleep(monkeypatch)

    async def func():
        raise RateLimitError("slow down", retry_after=5000)  # far beyond any sane cap

    with pytest.raises(RateLimitError):
        await retry_with_backoff(
            func, max_retries=1, rate_limit_base_seconds=30.0, rate_limit_max_seconds=900.0
        )

    assert len(delays) == 1
    assert 900.0 <= delays[0] <= 900.0 * 1.25  # capped, then jittered


async def test_retry_with_backoff_rate_limit_falls_back_to_generic_track_if_not_given() -> None:
    """rate_limit_base_seconds/rate_limit_max_seconds are optional - existing
    callers that don't pass them keep working exactly as before."""
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("slow down")
        return "ok"

    result = await retry_with_backoff(func, max_retries=2, base_seconds=0.01, max_seconds=0.02)

    assert result == "ok"


async def test_retry_with_backoff_jitter_varies_across_calls(monkeypatch) -> None:
    """Many accounts hitting the same shared rate-limit window would all
    report the same retry_after - jitter must still spread their retries out
    rather than releasing them all in perfect lockstep."""
    delays = _capture_sleep(monkeypatch)

    async def func():
        raise RateLimitError("slow down", retry_after=100)

    for _ in range(8):
        with pytest.raises(RateLimitError):
            await retry_with_backoff(
                func, max_retries=1, rate_limit_base_seconds=30.0, rate_limit_max_seconds=900.0
            )

    assert all(d >= 100.0 for d in delays)
    assert len(set(delays)) > 1  # jitter actually varies the delay run to run
