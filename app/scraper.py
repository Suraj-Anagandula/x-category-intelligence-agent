"""Concurrent scraping orchestration.

`ProfileScraper` ties together the Twikit client, the TTL cache, and the
retry/backoff helper to fetch many profiles concurrently (bounded by an
`asyncio.Semaphore`) while showing progress with Rich and logging
start/end/duration/errors for the whole run and per-user.
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

from app.cache import ProfileCache
from app.client import TwikitProfileClient
from app.config import Settings
from app.exceptions import InvalidUsernameError, ScraperError
from app.logger import get_logger
from app.models import ScrapeResult
from app.utils import is_valid_username, normalize_username, retry_with_backoff

logger = get_logger()


class ProfileScraper:
    """Fetches public profiles for many usernames, concurrently and resiliently."""

    def __init__(
        self, settings: Settings, client: TwikitProfileClient, cache: ProfileCache
    ) -> None:
        self.settings = settings
        self.client = client
        self.cache = cache
        self._semaphore = asyncio.Semaphore(settings.concurrency_limit)

    async def _scrape_one(self, raw_username: str) -> ScrapeResult:
        username = normalize_username(raw_username)
        if not is_valid_username(username):
            error = InvalidUsernameError(raw_username)
            logger.error(str(error))
            return ScrapeResult(
                username=raw_username,
                success=False,
                error=str(error),
                error_type=error.__class__.__name__,
            )

        attempts = 0

        async def attempt() -> ScrapeResult:
            nonlocal attempts
            attempts += 1

            cached = await self.cache.get(username)
            if cached is not None:
                logger.info(f"@{username}: served from cache")
                return ScrapeResult(
                    username=username,
                    success=True,
                    profile=cached,
                    attempts=attempts,
                    from_cache=True,
                )

            async with self._semaphore:
                await asyncio.sleep(self.settings.request_delay_seconds)
                profile = await self.client.get_profile(username)

            await self.cache.set(username, profile)
            return ScrapeResult(username=username, success=True, profile=profile, attempts=attempts)

        def on_retry(attempt_no: int, exc: Exception, delay: float) -> None:
            logger.warning(
                f"@{username}: retry {attempt_no}/{self.settings.max_retries} "
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
            logger.error(f"@{username}: failed permanently - {exc}")
            return ScrapeResult(
                username=username,
                success=False,
                error=str(exc),
                error_type=exc.__class__.__name__,
                attempts=attempts,
            )
        except Exception as exc:  # noqa: BLE001 - guarantee one user's failure can't kill the batch
            logger.exception(f"@{username}: unexpected failure")
            return ScrapeResult(
                username=username,
                success=False,
                error=str(exc),
                error_type=exc.__class__.__name__,
                attempts=attempts,
            )

    async def scrape_many(self, usernames: Iterable[str]) -> list[ScrapeResult]:
        """Scrape all `usernames` concurrently, returning one `ScrapeResult` each."""
        usernames = list(usernames)
        started_at = time.monotonic()
        logger.info(
            f"Starting scrape of {len(usernames)} usernames (concurrency={self.settings.concurrency_limit})"
        )

        if not usernames:
            # Skip the Rich progress display entirely for empty input - a
            # zero-total task still renders a spinner frame on exit, which
            # crashes with UnicodeEncodeError on Windows' legacy (cp1252)
            # console. Nothing to scrape means nothing to show progress for.
            logger.info("Finished scrape: 0 succeeded, 0 failed, duration=0.00s")
            return []

        results: list[ScrapeResult] = []
        progress_columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]Scraping profiles"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ]

        with Progress(*progress_columns) as progress:
            task_id = progress.add_task("scrape", total=len(usernames))

            async def run_one(username: str) -> ScrapeResult:
                result = await self._scrape_one(username)
                progress.advance(task_id)
                return result

            results = await asyncio.gather(*(run_one(u) for u in usernames))

        duration = time.monotonic() - started_at
        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        logger.info(
            f"Finished scrape: {succeeded} succeeded, {failed} failed, " f"duration={duration:.2f}s"
        )
        return list(results)
