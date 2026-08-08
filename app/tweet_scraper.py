"""Concurrent tweet-fetching orchestration.

`TweetScraper` mirrors `app/scraper.py`'s `ProfileScraper` shape - same
retry/backoff, TTL cache, per-account failure isolation, progress bar, and
logging - but calls `TwikitProfileClient.get_recent_tweets()` instead of
`get_profile()`. Concurrency uses its own, lower `TWEET_SCRAPE_CONCURRENCY`
setting rather than the profile scraper's `CONCURRENCY_LIMIT`: X's
timeline-read endpoint rate-limits noticeably more aggressively than the
profile-lookup endpoint, and rate-limit backoff uses its own, much longer
`RATE_LIMIT_BASE_SECONDS`/`RATE_LIMIT_MAX_SECONDS` track (see `app/utils.py`
`retry_with_backoff` and `app/client.py`'s `RateLimitError.retry_after`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from app.cache import TweetCache
from app.client import TwikitProfileClient
from app.config import Settings
from app.exceptions import InvalidUsernameError, ScraperError
from app.logger import get_logger
from app.models import TweetScrapeResult
from app.utils import is_valid_username, normalize_username, retry_with_backoff

logger = get_logger()


class TweetScraper:
    """Fetches recent tweets for many usernames, concurrently and resiliently."""

    def __init__(self, settings: Settings, client: TwikitProfileClient, cache: TweetCache) -> None:
        self.settings = settings
        self.client = client
        self.cache = cache
        self._semaphore = asyncio.Semaphore(settings.tweet_scrape_concurrency)

    async def _scrape_one(self, raw_username: str, count: int) -> TweetScrapeResult:
        username = normalize_username(raw_username)
        if not is_valid_username(username):
            error = InvalidUsernameError(raw_username)
            logger.error(str(error))
            return TweetScrapeResult(
                username=raw_username,
                success=False,
                error=str(error),
                error_type=error.__class__.__name__,
            )

        attempts = 0

        async def attempt() -> TweetScrapeResult:
            nonlocal attempts
            attempts += 1

            cached = await self.cache.get(username)
            if cached is not None:
                for tweet in cached:
                    tweet.username = username
                logger.info(f"@{username}: tweets served from cache")
                return TweetScrapeResult(
                    username=username,
                    success=True,
                    tweets=cached,
                    attempts=attempts,
                    from_cache=True,
                )

            async with self._semaphore:
                await asyncio.sleep(self.settings.request_delay_seconds)
                tweets = await self.client.get_recent_tweets(username, count=count)

            for tweet in tweets:
                tweet.username = username

            await self.cache.set(username, tweets)
            return TweetScrapeResult(
                username=username, success=True, tweets=tweets, attempts=attempts
            )

        def on_retry(attempt_no: int, exc: Exception, delay: float) -> None:
            logger.warning(
                f"@{username}: tweet retry {attempt_no}/{self.settings.max_retries} "
                f"after {exc.__class__.__name__} ({exc}); backing off {delay:.1f}s"
            )

        try:
            return await retry_with_backoff(
                attempt,
                max_retries=self.settings.max_retries,
                base_seconds=self.settings.backoff_base_seconds,
                max_seconds=self.settings.backoff_max_seconds,
                rate_limit_base_seconds=self.settings.rate_limit_base_seconds,
                rate_limit_max_seconds=self.settings.rate_limit_max_seconds,
                on_retry=on_retry,
            )
        except ScraperError as exc:
            logger.error(f"@{username}: tweet fetch failed permanently - {exc}")
            return TweetScrapeResult(
                username=username,
                success=False,
                error=str(exc),
                error_type=exc.__class__.__name__,
                attempts=attempts,
            )
        except Exception as exc:  # noqa: BLE001 - one account's failure can't kill the batch
            logger.exception(f"@{username}: unexpected tweet-fetch failure")
            return TweetScrapeResult(
                username=username,
                success=False,
                error=str(exc),
                error_type=exc.__class__.__name__,
                attempts=attempts,
            )

    async def scrape_many(
        self, usernames: Iterable[str], count: int | None = None
    ) -> list[TweetScrapeResult]:
        """Fetch recent tweets for all `usernames` concurrently."""
        usernames = list(usernames)
        count = count if count is not None else self.settings.tweets_per_account
        started_at = time.monotonic()
        logger.info(
            f"Starting tweet scrape for {len(usernames)} account(s) "
            f"(count={count}, concurrency={self.settings.tweet_scrape_concurrency})"
        )

        if not usernames:
            # See ProfileScraper.scrape_many: skip the Rich progress display
            # entirely for empty input to avoid a UnicodeEncodeError from a
            # zero-total task's spinner frame on Windows' legacy console.
            logger.info(
                "Tweet scrape summary:\n"
                "  Accounts requested: 0\n"
                "  Accounts succeeded: 0\n"
                "  Accounts rate limited: 0\n"
                "  Accounts failed for other reasons: 0\n"
                "  Tweets collected: 0"
            )
            return []

        progress_columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]Scraping tweets"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ]

        with Progress(*progress_columns) as progress:
            task_id = progress.add_task("scrape", total=len(usernames))

            async def run_one(username: str) -> TweetScrapeResult:
                result = await self._scrape_one(username, count)
                progress.advance(task_id)
                return result

            results = await asyncio.gather(*(run_one(u) for u in usernames))

        duration = time.monotonic() - started_at
        succeeded = sum(1 for r in results if r.success)
        rate_limited = sum(1 for r in results if not r.success and r.error_type == "RateLimitError")
        failed_other = len(results) - succeeded - rate_limited
        total_tweets = sum(len(r.tweets) for r in results if r.success)
        logger.info(
            "Tweet scrape summary:\n"
            f"  Accounts requested: {len(results)}\n"
            f"  Accounts succeeded: {succeeded}\n"
            f"  Accounts rate limited: {rate_limited}\n"
            f"  Accounts failed for other reasons: {failed_other}\n"
            f"  Tweets collected: {total_tweets}\n"
            f"  Duration: {duration:.2f}s"
        )
        return list(results)
