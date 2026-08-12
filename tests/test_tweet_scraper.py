"""Unit tests for app.tweet_scraper: cache-hit reporting and per-account failure isolation."""

from __future__ import annotations

from app.cache import TweetCache
from app.config import Settings
from app.exceptions import NetworkTimeoutError, ProtectedAccountError, RateLimitError
from app.models import Tweet
from app.time_window import resolve_time_window
from app.tweet_scraper import TweetScraper


class _StubClient:
    """Fake Twikit client: returns canned tweets, or raises for special usernames."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.timeout_attempts = 0
        self.rate_limit_attempts = 0

    async def get_recent_tweets(self, username: str, count: int = 10) -> list[Tweet]:
        self.calls.append(username)
        if username == "protected_user":
            raise ProtectedAccountError(username)
        if username == "rate_limited":
            raise RateLimitError("slow down")
        if username == "rl_retry_ok":
            self.rate_limit_attempts += 1
            if self.rate_limit_attempts < 2:
                raise RateLimitError("slow down", retry_after=0.01)
        if username == "flaky":
            self.timeout_attempts += 1
            if self.timeout_attempts < 2:
                raise NetworkTimeoutError("timed out")
        return [Tweet(id=str(i), text=f"tweet {i}") for i in range(count)]


def _fast_settings(tmp_path) -> Settings:
    settings = Settings()
    settings.request_delay_seconds = 0
    settings.max_retries = 2
    settings.backoff_base_seconds = 0.01
    settings.backoff_max_seconds = 0.02
    # Rate-limit backoff defaults to 30s/900s in production - override to
    # keep these tests fast; the values themselves are covered separately
    # in tests/test_utils.py.
    settings.rate_limit_base_seconds = 0.01
    settings.rate_limit_max_seconds = 0.02
    settings.tweet_cache_dir = tmp_path
    settings.tweet_cache_ttl_seconds = 3600
    settings.cache_enabled = True
    settings.tweets_per_account = 5
    return settings


def _scraper(tmp_path) -> tuple[TweetScraper, _StubClient]:
    settings = _fast_settings(tmp_path)
    client = _StubClient()
    cache = TweetCache(
        cache_dir=settings.tweet_cache_dir, ttl_seconds=settings.tweet_cache_ttl_seconds
    )
    return TweetScraper(settings, client, cache), client


async def test_scrape_one_fetches_and_tags_username(tmp_path) -> None:
    scraper, client = _scraper(tmp_path)

    result = await scraper._scrape_one("elonmusk", count=5)

    assert result.success is True
    assert result.from_cache is False
    assert len(result.tweets) == 5
    assert all(tweet.username == "elonmusk" for tweet in result.tweets)
    assert client.calls == ["elonmusk"]


async def test_scrape_one_second_fetch_served_from_cache(tmp_path) -> None:
    scraper, client = _scraper(tmp_path)

    first = await scraper._scrape_one("elonmusk", count=5)
    second = await scraper._scrape_one("elonmusk", count=5)

    assert first.from_cache is False
    assert second.from_cache is True
    assert all(tweet.username == "elonmusk" for tweet in second.tweets)
    assert client.calls == ["elonmusk"]  # second call served from cache


async def test_scrape_one_rejects_invalid_username(tmp_path) -> None:
    scraper, client = _scraper(tmp_path)

    result = await scraper._scrape_one("has space", count=5)

    assert result.success is False
    assert result.error_type == "InvalidUsernameError"
    assert client.calls == []


async def test_scrape_one_reports_protected_account(tmp_path) -> None:
    scraper, _ = _scraper(tmp_path)

    result = await scraper._scrape_one("protected_user", count=5)

    assert result.success is False
    assert result.error_type == "ProtectedAccountError"


async def test_scrape_one_retries_transient_then_succeeds(tmp_path) -> None:
    scraper, client = _scraper(tmp_path)

    result = await scraper._scrape_one("flaky", count=5)

    assert result.success is True
    assert result.attempts == 2


async def test_scrape_many_isolates_per_account_failure(tmp_path) -> None:
    scraper, _ = _scraper(tmp_path)

    results = await scraper.scrape_many(["elonmusk", "protected_user", "openai"], count=3)

    by_username = {r.username: r for r in results}
    assert by_username["elonmusk"].success is True
    assert by_username["openai"].success is True
    assert by_username["protected_user"].success is False
    assert by_username["protected_user"].error_type == "ProtectedAccountError"


async def test_scrape_many_exhausts_retries_on_persistent_rate_limit(tmp_path) -> None:
    scraper, _ = _scraper(tmp_path)

    results = await scraper.scrape_many(["rate_limited"], count=3)

    assert results[0].success is False
    assert results[0].error_type == "RateLimitError"


async def test_scrape_many_returns_empty_list_for_empty_input(tmp_path) -> None:
    """A category with zero discovered candidates must not crash - see the
    matching regression test in test_scraper.py for the root cause."""
    scraper, client = _scraper(tmp_path)

    results = await scraper.scrape_many([])

    assert results == []
    assert client.calls == []


async def test_scrape_one_retries_rate_limit_then_succeeds(tmp_path) -> None:
    scraper, client = _scraper(tmp_path)

    result = await scraper._scrape_one("rl_retry_ok", count=5)

    assert result.success is True
    assert result.attempts == 2
    assert client.rate_limit_attempts == 2


async def test_scrape_many_preserves_successful_results_alongside_rate_limited_failures(
    tmp_path,
) -> None:
    """Spec: retain every successfully-collected account's tweets even when
    other accounts in the same batch are rate limited - one must never
    discard the other."""
    scraper, _ = _scraper(tmp_path)

    usernames = ["elonmusk", "rate_limited", "openai", "rl_retry_ok", "protected_user"]
    results = await scraper.scrape_many(usernames, count=3)

    by_username = {r.username: r for r in results}
    assert by_username["elonmusk"].success is True
    assert len(by_username["elonmusk"].tweets) == 3
    assert by_username["openai"].success is True
    assert by_username["rl_retry_ok"].success is True
    assert by_username["rate_limited"].success is False
    assert by_username["rate_limited"].error_type == "RateLimitError"
    assert by_username["protected_user"].success is False
    assert by_username["protected_user"].error_type == "ProtectedAccountError"

    succeeded = sum(1 for r in results if r.success)
    rate_limited = sum(1 for r in results if not r.success and r.error_type == "RateLimitError")
    other_failed = len(results) - succeeded - rate_limited
    assert succeeded == 3
    assert rate_limited == 1
    assert other_failed == 1


class _StubWindowAwareClient:
    """Unlike `_StubClient` above, accepts the `window`/`max_pages` kwargs
    `TweetScraper` passes through for a real (filtered) time window - used
    only by the window-specific tests below, so `_StubClient`'s exact
    original 2-arg signature (and every existing test using it) stays
    untouched.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def get_recent_tweets(self, username: str, count: int = 10, window=None, max_pages=10):
        self.calls.append((username, {"window": window, "max_pages": max_pages}))
        return [Tweet(id=str(i), text=f"tweet {i}") for i in range(count)]


def _window_scraper(tmp_path) -> tuple[TweetScraper, _StubWindowAwareClient]:
    settings = _fast_settings(tmp_path)
    client = _StubWindowAwareClient()
    cache = TweetCache(
        cache_dir=settings.tweet_cache_dir, ttl_seconds=settings.tweet_cache_ttl_seconds
    )
    return TweetScraper(settings, client, cache), client


async def test_scrape_one_latest_mode_calls_client_without_window_kwargs(tmp_path) -> None:
    """The default latest-mode call (no window given) must be indistinguishable
    from the pre-time-window-feature call shape - no window/max_pages
    kwargs reach the client at all."""
    scraper, client = _window_scraper(tmp_path)

    await scraper._scrape_one("elonmusk", count=5)

    assert client.calls == [("elonmusk", {"window": None, "max_pages": 10})]


async def test_scrape_one_real_window_passes_window_and_max_pages_through(tmp_path) -> None:
    scraper, client = _window_scraper(tmp_path)
    settings = scraper.settings
    settings.tweet_window_max_pages = 7
    window = resolve_time_window("7d")

    await scraper._scrape_one("elonmusk", count=5, window=window)

    assert len(client.calls) == 1
    called_username, kwargs = client.calls[0]
    assert called_username == "elonmusk"
    assert kwargs["window"] is window
    assert kwargs["max_pages"] == 7


async def test_scrape_one_real_window_bypasses_cache_read_and_write(tmp_path) -> None:
    """A windowed fetch must never read from or write to the plain
    "latest" cache - see app/tweet_scraper.py for why (cache is keyed by
    username only, with one TTL meant for "most recent tweets")."""
    scraper, client = _window_scraper(tmp_path)
    window = resolve_time_window("7d")

    first = await scraper._scrape_one("elonmusk", count=5, window=window)
    second = await scraper._scrape_one("elonmusk", count=5, window=window)

    assert first.from_cache is False
    assert second.from_cache is False  # never served from cache
    assert len(client.calls) == 2  # X was hit both times, no cache short-circuit


async def test_scrape_many_passes_window_through_to_every_account(tmp_path) -> None:
    scraper, client = _window_scraper(tmp_path)
    window = resolve_time_window("24h")

    await scraper.scrape_many(["a", "b"], count=3, window=window)

    assert len(client.calls) == 2
    assert all(kwargs["window"] is window for _, kwargs in client.calls)


async def test_scrape_many_defaults_to_latest_when_no_window_given(tmp_path) -> None:
    """Existing callers that don't pass `window` at all keep the exact
    original "latest" behavior."""
    scraper, client = _window_scraper(tmp_path)

    await scraper.scrape_many(["a"], count=3)

    assert client.calls == [("a", {"window": None, "max_pages": 10})]


def test_tweet_scraper_uses_dedicated_concurrency_setting_not_profile_concurrency(
    tmp_path,
) -> None:
    """Tweet scraping must use TWEET_SCRAPE_CONCURRENCY, independently of
    (and lower than) the profile scraper's CONCURRENCY_LIMIT."""
    settings = _fast_settings(tmp_path)
    settings.concurrency_limit = 20
    settings.tweet_scrape_concurrency = 3
    cache = TweetCache(
        cache_dir=settings.tweet_cache_dir, ttl_seconds=settings.tweet_cache_ttl_seconds
    )

    scraper = TweetScraper(settings, _StubClient(), cache)

    assert scraper._semaphore._value == 3
