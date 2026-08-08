"""Export collected profiles/tweets to durable storage.

`BaseExporter` defines the interface; `JSONExporter`, `CSVExporter`,
`TweetCSVExporter`, and `CategoryTweetCSVExporter` are the concrete
implementations. Adding a new backend (SQLite, PostgreSQL, MongoDB, ...)
later means writing one more subclass that implements `export()` - no
changes needed elsewhere, since callers only depend on the abstract
interface via `ExporterRegistry`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import pandas as pd

from app.logger import get_logger
from app.models import Tweet, UserProfile

logger = get_logger()

# pandas.to_csv()/read_csv() default to the OS locale encoding when none is
# given - on Windows that's cp1252, which can't represent emoji at all and
# raises UnicodeEncodeError on write. utf-8-sig writes a BOM so Excel also
# detects UTF-8 correctly instead of mangling non-ASCII text; reading with
# utf-8-sig transparently strips that BOM if present and is otherwise
# equivalent to plain utf-8.
CSV_ENCODING = "utf-8-sig"


class BaseExporter(ABC):
    """Interface every export backend must implement."""

    @abstractmethod
    async def export(self, profiles: list[UserProfile]) -> Path:
        """Persist `profiles` and return the path/identifier written to."""


class JSONExporter(BaseExporter):
    """Writes one timestamped JSON array file per run."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def export(self, profiles: list[UserProfile]) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"profiles_{timestamp}.json"
        payload = [profile.to_flat_dict() for profile in profiles]

        async with aiofiles.open(path, "w", encoding="utf-8") as fh:
            await fh.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

        logger.info(f"Wrote {len(profiles)} profile(s) to {path}")
        return path


class CSVExporter(BaseExporter):
    """Upserts profiles into a single running `users.csv`, keyed by username."""

    def __init__(self, output_dir: Path, filename: str = "users.csv") -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / filename

    async def export(self, profiles: list[UserProfile]) -> Path:
        new_rows = pd.DataFrame([profile.to_flat_dict() for profile in profiles])
        if new_rows.empty:
            logger.info("No profiles to export to CSV.")
            return self.path

        if self.path.exists():
            # Keep all-digit id columns as strings on read - pandas would
            # otherwise infer int64 and silently lose type consistency across
            # append cycles (id/pinned_tweet_id are modeled as str for a reason:
            # they can exceed safe float/precision ranges).
            existing = pd.read_csv(
                self.path, dtype={"id": str, "pinned_tweet_id": str}, encoding=CSV_ENCODING
            )
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset="username", keep="last")
        else:
            combined = new_rows

        combined.to_csv(self.path, index=False, encoding=CSV_ENCODING)
        logger.info(f"Wrote {len(new_rows)} profile(s) to {self.path} ({len(combined)} total rows)")
        return self.path


class TweetCSVExporter(BaseExporter):
    """Upserts one user's tweets into `<username>_tweets.csv`, keyed by tweet id."""

    def __init__(self, output_dir: Path, username: str) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.username = username
        self.path = self.output_dir / f"{username}_tweets.csv"

    async def export(self, tweets: list[Tweet]) -> Path:
        new_rows = pd.DataFrame([tweet.to_flat_dict() for tweet in tweets])
        if new_rows.empty:
            logger.info(f"No tweets to export for @{self.username}.")
            return self.path

        if self.path.exists():
            # Force `id` to stay a string on read - it's all-digit, so pandas
            # would otherwise infer int64 and silently break the dedupe below
            # (new_rows keeps `id` as str, so "1" != 1 to drop_duplicates).
            existing = pd.read_csv(self.path, dtype={"id": str}, encoding=CSV_ENCODING)
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset="id", keep="last")
        else:
            combined = new_rows

        combined.to_csv(self.path, index=False, encoding=CSV_ENCODING)
        logger.info(f"Wrote {len(new_rows)} tweet(s) to {self.path} ({len(combined)} total rows)")
        return self.path


class CategoryTweetCSVExporter(BaseExporter):
    """Upserts a whole category run's tweets - with account context joined
    in - into `<category>_tweets.csv`, keyed by tweet_id. One row per tweet.

    This is additional to (not a replacement for) the per-account
    `TweetCSVExporter` output; both are written for a category run.
    """

    #: Note: no `conversation_id` column - twifork's `Tweet` object exposes
    #: no such field, and we don't invent data that isn't actually scraped.
    COLUMNS = [
        "category",
        "username",
        "user_id",
        "display_name",
        "followers_count",
        "tweet_id",
        "tweet_text",
        "created_at",
        "likes",
        "retweets",
        "replies",
        "views",
        "url",
        "media_urls",
        "hashtags",
        "mentions",
        "quoted_tweet_id",
        "account_rank",
        "relevance_score",
        "discovery_reason",
        "collected_at",
    ]

    def __init__(self, output_dir: Path, category: str) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.category = category
        self.path = self.output_dir / f"{category}_tweets.csv"

    async def export(
        self,
        tweets: list[Tweet],
        profiles_by_username: dict[str, UserProfile],
        collected_at: datetime,
        account_meta_by_username: dict[str, dict] | None = None,
    ) -> Path:
        account_meta_by_username = account_meta_by_username or {}
        rows = []
        for tweet in tweets:
            profile = profiles_by_username.get(tweet.username) if tweet.username else None
            meta = account_meta_by_username.get(tweet.username, {}) if tweet.username else {}
            rows.append(
                {
                    "category": self.category,
                    "username": tweet.username,
                    "user_id": profile.id if profile else None,
                    "display_name": profile.display_name if profile else None,
                    "followers_count": profile.followers if profile else None,
                    "tweet_id": tweet.id,
                    "tweet_text": tweet.text,
                    "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                    "likes": tweet.like_count,
                    "retweets": tweet.retweet_count,
                    "replies": tweet.reply_count,
                    "views": tweet.view_count,
                    "url": tweet.url,
                    "media_urls": ";".join(tweet.media_urls) if tweet.media_urls else None,
                    "hashtags": ";".join(tweet.hashtags) if tweet.hashtags else None,
                    "mentions": ";".join(tweet.mentions) if tweet.mentions else None,
                    "quoted_tweet_id": tweet.quoted_tweet_id,
                    "account_rank": meta.get("rank"),
                    "relevance_score": meta.get("relevance_score"),
                    "discovery_reason": meta.get("discovery_reason"),
                    "collected_at": collected_at.isoformat(),
                }
            )

        new_rows = pd.DataFrame(rows, columns=self.COLUMNS)
        if new_rows.empty:
            logger.info(f"No tweets to export for category {self.category!r}.")
            return self.path

        if self.path.exists():
            # Force id-like columns to stay strings on read - same
            # int64-inference pitfall as TweetCSVExporter/CSVExporter above.
            existing = pd.read_csv(
                self.path, dtype={"tweet_id": str, "user_id": str}, encoding=CSV_ENCODING
            )
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset="tweet_id", keep="last")
        else:
            combined = new_rows

        combined.to_csv(self.path, index=False, encoding=CSV_ENCODING)
        logger.info(f"Wrote {len(new_rows)} tweet(s) to {self.path} ({len(combined)} total rows)")
        return self.path


class ExporterRegistry:
    """Resolves export-format names to `BaseExporter` instances.

    Extend by registering new backends here (or dynamically via `register`)
    rather than branching on format strings throughout the codebase.
    """

    def __init__(self) -> None:
        self._factories: dict[str, type[BaseExporter]] = {}

    def register(self, name: str, exporter_cls: type[BaseExporter]) -> None:
        self._factories[name] = exporter_cls

    def create(self, name: str, **kwargs) -> BaseExporter:
        try:
            exporter_cls = self._factories[name]
        except KeyError as exc:
            raise ValueError(f"Unknown export format: {name!r}") from exc
        return exporter_cls(**kwargs)
