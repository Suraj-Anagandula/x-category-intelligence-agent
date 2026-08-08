"""Runs the existing `CategoryIntelligenceAgent` pipeline for the Streamlit UI.

This module builds the same client/cache/scraper/LLM objects `main.py`'s
`_run_analyze()` does, and calls the same, unmodified
`CategoryIntelligenceAgent.run_pipeline()` - it does not reimplement
discovery, validation, ranking, scraping, retry/rate-limit handling, or
analysis.

Real-time stage progress is derived by tapping the pipeline's *existing*
loguru log messages via a temporary sink (added before the run, removed
after) - no callback/hook parameter is added to `app/category_agent.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.cache import ProfileCache, TweetCache
from app.category_agent import CategoryIntelligenceAgent
from app.client import TwikitProfileClient
from app.config import settings
from app.llm import build_llm_client
from app.logger import get_logger
from app.schemas import CategoryReport
from app.scraper import ProfileScraper
from app.tweet_scraper import TweetScraper

logger = get_logger()

#: (stage_key, log-message substring that marks it complete). Matched against
#: the exact lines app/category_agent.py's run_pipeline already logs, so
#: progress reflects real backend completion, never a simulated timer.
STAGE_MARKERS: list[tuple[str, str]] = [
    ("context", "Category context for"),
    ("discovery", "Candidates discovered:"),
    ("validation", "Profiles validated:"),
    ("ranking", "Accounts selected:"),
    ("tweets", "Tweet scrape summary:"),
    ("analysis", "Analysis completed"),
    ("export", "CSV:"),
]

#: Ordered stage keys + display labels, for rendering the 7-step checklist.
PIPELINE_STAGES: list[tuple[str, str]] = [
    ("context", "Category Context"),
    ("discovery", "Account Discovery"),
    ("validation", "Profile Validation"),
    ("ranking", "Account Ranking"),
    ("tweets", "Tweet Collection"),
    ("analysis", "AI Analysis"),
    ("export", "Export"),
]


def _build_client() -> TwikitProfileClient:
    """Identical to main.py's `_build_client()` - same auth/session wiring."""
    return TwikitProfileClient(
        auth_token=settings.x_auth_token,
        ct0=settings.x_ct0,
        username=settings.x_username,
        email=settings.x_email,
        password=settings.x_password,
        session_file=settings.x_session_file,
    )


async def _run_pipeline_async(
    category: str,
    candidate_limit: int,
    top_accounts: int,
    tweets_per_account: int,
) -> CategoryReport:
    """Identical wiring to main.py's `_run_analyze()`."""
    settings.ensure_directories()

    client = _build_client()
    await client.connect()

    profile_cache = ProfileCache(
        cache_dir=settings.cache_dir,
        ttl_seconds=settings.cache_ttl_seconds,
        enabled=settings.cache_enabled,
    )
    profile_scraper = ProfileScraper(settings, client, profile_cache)

    tweet_cache = TweetCache(
        cache_dir=settings.tweet_cache_dir,
        ttl_seconds=settings.tweet_cache_ttl_seconds,
        enabled=settings.cache_enabled,
    )
    tweet_scraper = TweetScraper(settings, client, tweet_cache)

    llm_client = build_llm_client(settings)

    agent = CategoryIntelligenceAgent(settings, profile_scraper, tweet_scraper, llm_client)
    return await agent.run_pipeline(
        category,
        candidate_limit=candidate_limit,
        top_n=top_accounts,
        tweets_per_account=tweets_per_account,
    )


def run_category_analysis(
    category: str,
    candidate_limit: int,
    top_accounts: int,
    tweets_per_account: int,
    on_stage: Callable[[str], None] | None = None,
) -> CategoryReport:
    """Synchronous entry point for Streamlit callbacks.

    Runs the existing pipeline end to end and returns its `CategoryReport`
    unchanged. If `on_stage` is given, it's called with each `STAGE_MARKERS`
    key the moment the pipeline's own logging reports that stage complete.

    Raises whatever `CategoryIntelligenceAgent.run_pipeline` raises
    (`AuthenticationError`, `LLMError`, or another `ScraperError`) - the
    caller handles these exactly as `main.py` already does.
    """
    sink_id = None
    if on_stage is not None:

        def _progress_sink(message) -> None:
            text = message.record["message"]
            for stage_key, marker in STAGE_MARKERS:
                if marker in text:
                    on_stage(stage_key)

        sink_id = logger.add(_progress_sink, level="INFO", format="{message}")

    try:
        return asyncio.run(
            _run_pipeline_async(category, candidate_limit, top_accounts, tweets_per_account)
        )
    finally:
        if sink_id is not None:
            logger.remove(sink_id)
