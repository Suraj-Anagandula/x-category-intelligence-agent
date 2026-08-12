"""Runs the existing `CategoryIntelligenceAgent` pipeline for the Streamlit UI.

This module builds the same client/cache/scraper/LLM objects `main.py`'s
`_run_analyze()` does, and calls the same, unmodified
`CategoryIntelligenceAgent.run_pipeline()` - it does not reimplement
discovery, validation, ranking, scraping, retry/rate-limit handling, or
analysis.

Real-time stage progress is now driven by `run_pipeline`'s own `on_stage`
callback parameter (a real hook, not string-matching against log output),
so progress reflects real backend completion - never a simulated timer -
and carries the real counts computed at each stage.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.cache import ProfileCache, TweetCache
from app.category_agent import CategoryIntelligenceAgent
from app.client import TwikitProfileClient
from app.config import settings
from app.llm import build_llm_client
from app.schemas import CategoryReport
from app.scraper import ProfileScraper
from app.time_window import TimeWindow
from app.tweet_scraper import TweetScraper

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
    on_stage: Callable[[str, dict], None] | None = None,
    time_window: TimeWindow | None = None,
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
        on_stage=on_stage,
        time_window=time_window,
    )


def run_category_analysis(
    category: str,
    candidate_limit: int,
    top_accounts: int,
    tweets_per_account: int,
    on_stage: Callable[[str, dict], None] | None = None,
    time_window: TimeWindow | None = None,
) -> CategoryReport:
    """Synchronous entry point for Streamlit callbacks.

    Runs the existing pipeline end to end and returns its `CategoryReport`
    unchanged. If `on_stage` is given, it's passed straight through to
    `CategoryIntelligenceAgent.run_pipeline`, which calls it with
    `(stage_key, payload)` the moment each stage actually completes -
    `payload` carries real counts computed at that point (e.g.
    `{"tweets_collected": 184}`), not just a completion marker.

    `time_window` defaults to `None` -> `CategoryIntelligenceAgent.run_pipeline`
    resolves that to "latest" (no filtering), so every existing caller that
    doesn't pass it keeps the original "most recent available tweets"
    behavior unchanged.

    Raises whatever `CategoryIntelligenceAgent.run_pipeline` raises
    (`AuthenticationError`, `LLMError`, or another `ScraperError`) - the
    caller handles these exactly as `main.py` already does.
    """
    return asyncio.run(
        _run_pipeline_async(
            category, candidate_limit, top_accounts, tweets_per_account, on_stage, time_window
        )
    )
