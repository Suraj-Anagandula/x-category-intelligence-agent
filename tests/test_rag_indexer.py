"""Unit tests for app.rag.indexer.

Uses a deterministic bag-of-words stub embedder (no real sentence-transformers
model) and a real Chroma `VectorStore` backed by `tmp_path` - fast and fully
offline, matching this project's existing tmp_path-based test convention
(see tests/test_storage.py, tests/test_cache.py).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.config import Settings
from app.models import Tweet
from app.rag.indexer import (
    backfill_from_snapshots,
    index_tweets,
    normalize_tweet_text,
    tweet_to_metadata,
)
from app.rag.vector_store import VectorStore
from app.time_window import resolve_time_window


class _StubEmbedder:
    """Deterministic bag-of-words embedder over a tiny fixed vocabulary -
    no real model, so these tests stay fast and network-free."""

    _VOCAB = ["ai", "regulation", "mars", "rover", "compliance", "quantum"]

    def embed_texts(self, texts):
        return [[1.0 if word in text.lower() else 0.0 for word in self._VOCAB] for text in texts]


def _settings(tmp_path) -> Settings:
    settings = Settings()
    settings.tweets_output_dir = tmp_path / "tweets"
    return settings


def test_normalize_tweet_text_strips_urls_keeps_hashtags() -> None:
    tweet = Tweet(id="1", text="Big AI news https://t.co/abc123 #AIRegulation @openai")

    normalized = normalize_tweet_text(tweet)

    assert "https://t.co/abc123" not in normalized
    assert "#AIRegulation" in normalized
    assert "@openai" in normalized


def test_normalize_tweet_text_collapses_whitespace() -> None:
    tweet = Tweet(id="1", text="hello   \n\n  world")

    assert normalize_tweet_text(tweet) == "hello world"


def test_tweet_to_metadata_includes_expected_fields() -> None:
    created_at = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    tweet = Tweet(
        id="1",
        username="openai",
        text="hello",
        created_at=created_at,
        like_count=10,
        retweet_count=2,
        reply_count=1,
        view_count=100,
    )

    metadata = tweet_to_metadata(tweet, "technology", created_at)

    assert metadata["tweet_id"] == "1"
    assert metadata["username"] == "openai"
    assert metadata["category"] == "technology"
    assert metadata["url"] == tweet.url
    assert metadata["like_count"] == 10
    assert metadata["created_at"] == created_at.isoformat()
    assert metadata["created_at_epoch"] == int(created_at.timestamp())


def test_tweet_to_metadata_handles_missing_created_at() -> None:
    tweet = Tweet(id="1", username="openai", text="hello", created_at=None)

    metadata = tweet_to_metadata(tweet, "technology", datetime.now(timezone.utc))

    assert metadata["created_at"] == ""
    assert "created_at_epoch" not in metadata


def test_tweet_to_metadata_includes_window_fields_when_given() -> None:
    tweet = Tweet(id="1", username="openai", text="hello")
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )

    metadata = tweet_to_metadata(tweet, "technology", datetime.now(timezone.utc), window)

    assert metadata["analysis_window_mode"] == "custom"
    assert metadata["analysis_window_start"] == window.start.isoformat()
    assert metadata["analysis_window_end"] == window.end.isoformat()


def test_tweet_to_metadata_omits_window_fields_when_not_given() -> None:
    tweet = Tweet(id="1", username="openai", text="hello")

    metadata = tweet_to_metadata(tweet, "technology", datetime.now(timezone.utc))

    assert "analysis_window_mode" not in metadata
    assert "analysis_window_start" not in metadata


def test_index_tweets_skips_empty_text_tweets(tmp_path) -> None:
    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()
    tweets = [
        Tweet(id="1", username="a", text="AI regulation news"),
        Tweet(id="2", username="b", text="   "),
    ]

    count = index_tweets(tweets, "technology", datetime.now(timezone.utc), store, embedder)

    assert count == 1
    assert store.count() == 1


def test_index_tweets_is_idempotent_on_rerun(tmp_path) -> None:
    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()
    tweets = [Tweet(id="1", username="a", text="AI regulation news")]

    index_tweets(tweets, "technology", datetime.now(timezone.utc), store, embedder)
    index_tweets(tweets, "technology", datetime.now(timezone.utc), store, embedder)

    assert store.count() == 1


def test_index_tweets_returns_zero_for_no_documents(tmp_path) -> None:
    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()

    count = index_tweets([], "technology", datetime.now(timezone.utc), store, embedder)

    assert count == 0
    assert store.count() == 0


def test_backfill_from_snapshots_indexes_every_saved_run(tmp_path) -> None:
    settings = _settings(tmp_path)
    run_dir = settings.tweets_output_dir / "technology"
    run_dir.mkdir(parents=True)
    payload = {
        "category": "technology",
        "scraped_at": "2026-08-08T00:00:00+00:00",
        "tweets": [
            {"id": "1", "username": "a", "text": "AI regulation update"},
            {"id": "2", "username": "b", "text": "Quantum computing breakthrough"},
        ],
    }
    (run_dir / "2026-08-08.json").write_text(json.dumps(payload), encoding="utf-8")

    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()

    results = backfill_from_snapshots(settings, store, embedder)

    assert results == {"technology/2026-08-08": 2}
    assert store.count() == 2


def test_backfill_from_snapshots_enriches_metadata_with_the_snapshots_window(tmp_path) -> None:
    """Section 11: the analysis_window_* metadata enrichment is sourced
    from the snapshot's own stored "time_window" block."""
    settings = _settings(tmp_path)
    run_dir = settings.tweets_output_dir / "technology"
    run_dir.mkdir(parents=True)
    payload = {
        "category": "technology",
        "scraped_at": "2026-08-08T00:00:00+00:00",
        "time_window": {
            "mode": "7d",
            "start": "2026-08-01T00:00:00Z",  # Pydantic JSON-mode "Z" suffix
            "end": "2026-08-08T00:00:00Z",
            "posts_fetched": 10,
            "posts_in_window": 1,
        },
        "tweets": [{"id": "1", "username": "a", "text": "AI regulation update"}],
    }
    (run_dir / "2026-08-08.json").write_text(json.dumps(payload), encoding="utf-8")

    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()

    backfill_from_snapshots(settings, store, embedder)

    [embedding] = embedder.embed_texts(["AI regulation update"])
    hits = store.query(embedding, top_k=1)
    assert hits[0].metadata["analysis_window_mode"] == "7d"
    assert hits[0].metadata["analysis_window_start"] == "2026-08-01T00:00:00+00:00"


def test_backfill_from_snapshots_skips_unreadable_file(tmp_path) -> None:
    settings = _settings(tmp_path)
    run_dir = settings.tweets_output_dir / "technology"
    run_dir.mkdir(parents=True)
    (run_dir / "2026-08-08.json").write_text("not valid json", encoding="utf-8")

    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()

    results = backfill_from_snapshots(settings, store, embedder)

    assert results == {}


def test_backfill_from_snapshots_empty_when_no_runs_dir(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()

    assert backfill_from_snapshots(settings, store, embedder) == {}
