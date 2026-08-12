"""Tweet -> normalized text + metadata -> vector-store upsert.

Document text = the tweet's own text alone (URLs stripped, whitespace
collapsed; hashtags/@-mentions kept - they carry real semantic content for
a tweet). Everything else (username, timestamps, category, url, engagement)
lives purely in Chroma metadata, never concatenated into the embedded
string - embedding-space proximity stays about content, not identity.

No per-tweet `topic` field is invented here: none exists on `Tweet`/
`CategoryAnalysis` (trending topics are category-level, not per-tweet).
Topic-scoping for retrieval is handled by the `category` metadata filter,
not a fabricated per-tweet label.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.exceptions import RAGError
from app.logger import get_logger
from app.models import Tweet
from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorStore
from app.time_window import TimeWindow, time_window_from_dict

logger = get_logger()

_URL_RE = re.compile(r"https?://\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_tweet_text(tweet: Tweet) -> str:
    """Strip URLs and collapse whitespace - keeps hashtags/@-mentions."""
    text = _URL_RE.sub("", tweet.text or "")
    return _WHITESPACE_RE.sub(" ", text).strip()


def tweet_to_metadata(
    tweet: Tweet, category: str, scraped_at: datetime, window: TimeWindow | None = None
) -> dict:
    """Chroma metadata only supports flat scalar types (str/int/float/bool)
    - `created_at_epoch` (int) is stored alongside the human-readable ISO
    string specifically so date-range filtering works via Chroma's
    `$gte`/`$lte` numeric operators, which don't apply to string fields.

    `window`, if given, records which analysis window produced this tweet
    (`analysis_window_mode`/`_start`/`_end`) - purely informational
    enrichment; posts outside the window were already excluded upstream
    (`app/category_agent.py` filters before `save_category_run`/indexing
    ever see them), so this never widens what gets indexed.
    """
    metadata: dict = {
        "tweet_id": tweet.id,
        "username": tweet.username or "",
        "category": category,
        "url": tweet.url,
        "like_count": tweet.like_count or 0,
        "retweet_count": tweet.retweet_count or 0,
        "reply_count": tweet.reply_count or 0,
        "view_count": tweet.view_count or 0,
        "scraped_at": scraped_at.isoformat(),
        "created_at": "",
    }
    if tweet.created_at is not None:
        metadata["created_at"] = tweet.created_at.isoformat()
        metadata["created_at_epoch"] = int(tweet.created_at.timestamp())
    if window is not None:
        metadata["analysis_window_mode"] = window.mode
        if window.start is not None:
            metadata["analysis_window_start"] = window.start.isoformat()
        if window.end is not None:
            metadata["analysis_window_end"] = window.end.isoformat()
    return metadata


def index_tweets(
    tweets: list[Tweet],
    category: str,
    scraped_at: datetime,
    store: VectorStore,
    embedder: Embedder,
    window: TimeWindow | None = None,
) -> int:
    """Embed and upsert `tweets` into `store`.

    Idempotent: re-indexing the same `tweet.id` overwrites rather than
    duplicates (see `VectorStore.upsert`), so a same-day rerun that
    overwrites `data/tweets/<category>/<date>.json` is safe to re-index.
    Tweets with empty/whitespace-only text are skipped (nothing to embed);
    the skip count is logged, never silently dropped. Returns the number
    of tweets actually indexed.
    """
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    skipped = 0

    for tweet in tweets:
        text = normalize_tweet_text(tweet)
        if not text:
            skipped += 1
            continue
        ids.append(tweet.id)
        documents.append(text)
        metadatas.append(tweet_to_metadata(tweet, category, scraped_at, window))

    if skipped:
        logger.info(f"RAG indexing: skipped {skipped} tweet(s) with no text to embed")

    if not documents:
        return 0

    embeddings = embedder.embed_texts(documents)
    store.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info(f"RAG indexing: indexed {len(documents)} tweet(s) for category {category!r}")
    return len(documents)


def backfill_from_snapshots(
    settings: Settings, store: VectorStore, embedder: Embedder
) -> dict[str, int]:
    """Walk every existing `data/tweets/<category>/<date>.json` snapshot and
    index it, sharing the exact same `index_tweets` code path (and
    idempotency guarantee) as live indexing. Used by
    `scripts/backfill_rag_index.py` and the Ask Intelligence page's
    "Build/Refresh Index" button to make already-collected runs searchable
    without re-scraping X.

    Returns `{"<category>/<date>": count_indexed, ...}` for reporting; a
    snapshot that fails to parse is skipped (logged), not fatal to the rest
    of the backfill.
    """
    root = Path(settings.tweets_output_dir)
    results: dict[str, int] = {}
    if not root.exists():
        return results

    for path in sorted(root.glob("*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"RAG backfill: skipping unreadable snapshot {path}: {exc}")
            continue

        category = data.get("category", path.parent.name)
        scraped_at_raw = data.get("scraped_at")
        try:
            scraped_at = datetime.fromisoformat(scraped_at_raw) if scraped_at_raw else None
        except ValueError:
            scraped_at = None
        if scraped_at is None:
            scraped_at = datetime.fromtimestamp(path.stat().st_mtime)

        try:
            tweets = [Tweet.model_validate(t) for t in data.get("tweets", [])]
        except Exception as exc:  # noqa: BLE001 - a malformed snapshot must not abort the backfill
            logger.warning(f"RAG backfill: skipping snapshot with invalid tweets {path}: {exc}")
            continue

        # `time_window_from_dict` returns None for snapshots saved before
        # time-window support existed (no "time_window" key at all) - the
        # tweets themselves index exactly as before, just without the
        # analysis_window_* metadata enrichment.
        window = time_window_from_dict(data.get("time_window"))

        try:
            count = index_tweets(tweets, category, scraped_at, store, embedder, window)
        except RAGError as exc:
            logger.warning(f"RAG backfill: indexing failed for {path}: {exc}")
            continue

        results[f"{category}/{path.stem}"] = count

    return results
