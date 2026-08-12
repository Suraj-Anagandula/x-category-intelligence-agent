"""Persistence for category-intelligence runs.

Reuses the existing `JSONExporter`/`CSVExporter`/`TweetCSVExporter` for the
top-N profiles and their tweets (no new export formats other than
`CategoryTweetCSVExporter`, added specifically for the consolidated
per-category tweet CSV), and additionally writes one consolidated per-run
snapshot to `data/tweets/<category>/<YYYY-MM-DD>.json` (spec section 19) so a
category's full history is browsable on disk. Filesystem-only for now; the
flat-dict shape here is deliberately simple so a future SQL/Mongo backend can
read the same fields without redesigning this module (spec section 19).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from app.config import Settings
from app.exporter import CategoryTweetCSVExporter, CSVExporter, JSONExporter, TweetCSVExporter
from app.logger import get_logger
from app.models import Tweet, UserProfile
from app.schemas import CategoryReport

logger = get_logger()


async def save_category_run(
    category: str,
    report: CategoryReport,
    profiles: list[UserProfile],
    tweets: list[Tweet],
    settings: Settings,
    discovery_reasons: dict[str, str] | None = None,
    index_fn: Callable[[list[Tweet], str, datetime], int] | None = None,
) -> Path:
    """Persist a category run's profiles, tweets, and report.

    Writes the top-N profiles via the existing `JSONExporter`/`CSVExporter`,
    each account's tweets via the existing `TweetCSVExporter`, a consolidated
    `data/csv/<category>_tweets.csv` (one row per tweet, joined with account
    info, rank/relevance from `report.accounts`, and `discovery_reasons`) via
    `CategoryTweetCSVExporter`, and a single consolidated
    `{category, scraped_at, accounts, tweets}` snapshot to
    `data/tweets/<category>/<date>.json`. Returns the snapshot path.

    `index_fn`, if given, is called as `index_fn(tweets, category,
    scraped_at)` right after the snapshot is written (see
    `app.rag.indexer.index_tweets` for the real implementation, wired in by
    `CategoryIntelligenceAgent`). Any failure is logged and swallowed -
    indexing is an optional enhancement, never allowed to fail a scrape run.
    """
    scraped_at = datetime.now(timezone.utc)
    discovery_reasons = discovery_reasons or {}

    if profiles:
        await JSONExporter(output_dir=settings.json_output_dir).export(profiles)
        await CSVExporter(output_dir=settings.csv_output_dir).export(profiles)

    tweets_by_username: dict[str, list[Tweet]] = {}
    for tweet in tweets:
        if tweet.username:
            tweets_by_username.setdefault(tweet.username, []).append(tweet)
    for username, user_tweets in tweets_by_username.items():
        await TweetCSVExporter(output_dir=settings.csv_output_dir, username=username).export(
            user_tweets
        )

    profiles_by_username = {profile.username: profile for profile in profiles}
    account_meta_by_username = {
        account.username: {
            "rank": account.rank,
            "relevance_score": account.category_relevance,
            "discovery_reason": discovery_reasons.get(account.username, ""),
        }
        for account in report.accounts
    }
    await CategoryTweetCSVExporter(output_dir=settings.csv_output_dir, category=category).export(
        tweets, profiles_by_username, scraped_at, account_meta_by_username
    )

    run_dir = settings.tweets_output_dir / category
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / f"{scraped_at.strftime('%Y-%m-%d')}.json"

    payload = {
        "category": category,
        "scraped_at": scraped_at.isoformat(),
        "accounts": [account.model_dump(mode="json") for account in report.accounts],
        "tweets": [tweet.to_flat_dict() for tweet in tweets],
        "tweet_statistics": report.tweet_statistics.model_dump(mode="json"),
        "analysis": report.analysis.model_dump(mode="json"),
        "errors": report.errors,
        # Additive key - a snapshot saved before time-window support existed
        # simply won't have this; every reader (app/report_compare.py,
        # app/rag/indexer.py, ui/*) uses `.get("time_window", {})` and must
        # never assume it's present. `scraped_at` above stays the
        # collection/run timestamp - "time_window" records the separate
        # concept of *which tweets this run analyzed*.
        "time_window": report.time_window.model_dump(mode="json"),
    }

    async with aiofiles.open(snapshot_path, "w", encoding="utf-8") as fh:
        await fh.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    logger.info(f"Saved category run for {category!r} to {snapshot_path}")

    if index_fn is not None:
        try:
            index_fn(tweets, category, scraped_at)
        except Exception as exc:  # noqa: BLE001 - indexing must never fail a scrape run
            logger.warning(f"RAG indexing failed for {category!r}, continuing without it: {exc}")

    return snapshot_path


async def load_latest_category_run(category: str, settings: Settings) -> dict | None:
    """Return the most recently saved run snapshot for `category`, or None."""
    run_dir = settings.tweets_output_dir / category
    if not run_dir.exists():
        return None

    files = sorted(run_dir.glob("*.json"))
    if not files:
        return None

    latest = files[-1]
    async with aiofiles.open(latest, encoding="utf-8") as fh:
        raw = await fh.read()
    return json.loads(raw)
