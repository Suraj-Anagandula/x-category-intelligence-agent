"""Unit tests for app.scraper: cache-hit reporting and per-user failure isolation."""

from __future__ import annotations

from app.cache import ProfileCache
from app.config import Settings
from app.exceptions import UserNotFoundError
from app.models import UserProfile
from app.scraper import ProfileScraper


class _StubClient:
    """Fake Twikit client: returns a canned profile, or raises for 'ghost'."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_profile(self, username: str) -> UserProfile:
        self.calls.append(username)
        if username == "ghost":
            raise UserNotFoundError(username)
        return UserProfile(username=username, followers=100)


def _fast_settings(tmp_path) -> Settings:
    settings = Settings()
    settings.request_delay_seconds = 0
    settings.max_retries = 0
    settings.backoff_base_seconds = 0.01
    settings.backoff_max_seconds = 0.02
    settings.cache_dir = tmp_path
    settings.cache_ttl_seconds = 3600
    settings.cache_enabled = True
    return settings


async def test_scrape_one_marks_fresh_fetch_as_not_from_cache(tmp_path) -> None:
    settings = _fast_settings(tmp_path)
    client = _StubClient()
    cache = ProfileCache(cache_dir=settings.cache_dir, ttl_seconds=settings.cache_ttl_seconds)
    scraper = ProfileScraper(settings, client, cache)

    result = await scraper._scrape_one("elonmusk")

    assert result.success is True
    assert result.from_cache is False
    assert client.calls == ["elonmusk"]


async def test_scrape_one_marks_second_fetch_as_from_cache(tmp_path) -> None:
    settings = _fast_settings(tmp_path)
    client = _StubClient()
    cache = ProfileCache(cache_dir=settings.cache_dir, ttl_seconds=settings.cache_ttl_seconds)
    scraper = ProfileScraper(settings, client, cache)

    first = await scraper._scrape_one("elonmusk")
    second = await scraper._scrape_one("elonmusk")

    assert first.from_cache is False
    assert second.from_cache is True
    assert client.calls == ["elonmusk"]  # second call served from cache, no client hit


async def test_scrape_many_isolates_per_user_failure(tmp_path) -> None:
    settings = _fast_settings(tmp_path)
    client = _StubClient()
    cache = ProfileCache(cache_dir=settings.cache_dir, ttl_seconds=settings.cache_ttl_seconds)
    scraper = ProfileScraper(settings, client, cache)

    results = await scraper.scrape_many(["elonmusk", "ghost", "openai"])

    by_username = {r.username: r for r in results}
    assert by_username["elonmusk"].success is True
    assert by_username["openai"].success is True
    assert by_username["ghost"].success is False
    assert by_username["ghost"].error_type == "UserNotFoundError"


async def test_scrape_many_returns_empty_list_for_empty_input(tmp_path) -> None:
    """Regression guard: an empty username list must short-circuit before
    the Rich progress display renders - a zero-total task's spinner frame
    crashes with UnicodeEncodeError on Windows' legacy console, and this is
    a real path (a category with zero discovered candidates)."""
    settings = _fast_settings(tmp_path)
    client = _StubClient()
    cache = ProfileCache(cache_dir=settings.cache_dir, ttl_seconds=settings.cache_ttl_seconds)
    scraper = ProfileScraper(settings, client, cache)

    results = await scraper.scrape_many([])

    assert results == []
    assert client.calls == []


async def test_scrape_one_rejects_invalid_username(tmp_path) -> None:
    settings = _fast_settings(tmp_path)
    client = _StubClient()
    cache = ProfileCache(cache_dir=settings.cache_dir, ttl_seconds=settings.cache_ttl_seconds)
    scraper = ProfileScraper(settings, client, cache)

    result = await scraper._scrape_one("has space")

    assert result.success is False
    assert result.error_type == "InvalidUsernameError"
    assert client.calls == []
