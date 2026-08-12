"""Unit tests for app.rag.retriever.

Uses the same deterministic bag-of-words stub embedder and real tmp_path-
backed `VectorStore` as tests/test_rag_indexer.py - fast, fully offline.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.models import Tweet
from app.rag.indexer import index_tweets
from app.rag.retriever import retrieve
from app.rag.vector_store import VectorStore


class _StubEmbedder:
    _VOCAB = ["ai", "regulation", "mars", "rover", "compliance", "quantum"]

    def embed_texts(self, texts):
        return [[1.0 if word in text.lower() else 0.0 for word in self._VOCAB] for text in texts]


def _seeded_store(tmp_path, category="technology"):
    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()
    tweets = [
        Tweet(id="1", username="openai", text="New AI regulation announced today"),
        Tweet(id="2", username="nasa", text="Mars rover sends back new photos"),
        Tweet(id="3", username="sec_gov", text="AI compliance rules tightened"),
    ]
    index_tweets(tweets, category, datetime.now(timezone.utc), store, embedder)
    return store, embedder


def test_retrieve_returns_empty_list_when_store_is_empty(tmp_path) -> None:
    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()

    chunks = retrieve("What is happening with AI regulation?", store, embedder)

    assert chunks == []


def test_retrieve_ranks_relevant_tweets_higher(tmp_path) -> None:
    store, embedder = _seeded_store(tmp_path)

    chunks = retrieve("AI regulation", store, embedder, category="technology", min_similarity=0.0)

    assert chunks[0].tweet_id in {"1", "3"}
    ids = {c.tweet_id for c in chunks}
    assert "2" in ids  # still returned (low similarity, not filtered out at threshold 0.0)


def test_retrieve_drops_hits_below_similarity_threshold(tmp_path) -> None:
    store, embedder = _seeded_store(tmp_path)

    chunks = retrieve("AI regulation", store, embedder, category="technology", min_similarity=0.99)

    # Only a near-perfect vector match survives a very high threshold.
    assert all(c.similarity >= 0.99 for c in chunks)


def test_retrieve_scopes_to_category_via_where_filter(tmp_path) -> None:
    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()
    index_tweets(
        [Tweet(id="1", username="a", text="AI regulation in tech")],
        "technology",
        datetime.now(timezone.utc),
        store,
        embedder,
    )
    index_tweets(
        [Tweet(id="2", username="b", text="AI regulation in politics")],
        "politics",
        datetime.now(timezone.utc),
        store,
        embedder,
    )

    chunks = retrieve("AI regulation", store, embedder, category="politics", min_similarity=0.0)

    assert all(c.category == "politics" for c in chunks)
    assert {c.tweet_id for c in chunks} == {"2"}


def test_retrieve_never_exceeds_available_documents(tmp_path) -> None:
    """Requesting more results than exist in the store must not raise."""
    store, embedder = _seeded_store(tmp_path)

    chunks = retrieve(
        "AI regulation", store, embedder, category="technology", top_k=1000, min_similarity=0.0
    )

    assert len(chunks) <= 3


def test_retrieve_includes_real_url_and_username_on_each_chunk(tmp_path) -> None:
    store, embedder = _seeded_store(tmp_path)

    chunks = retrieve("AI regulation", store, embedder, category="technology", min_similarity=0.0)

    for chunk in chunks:
        assert chunk.url.startswith("https://x.com/i/status/")
        assert chunk.username


def test_retrieve_scopes_to_tweet_ids_allow_list(tmp_path) -> None:
    """Regression for the real reported evidence leak: an older tweet
    indexed under the same category must never be returned once retrieval
    is restricted to a specific (e.g. current-report) set of tweet ids -
    enforced at the data/query layer via Chroma's `where` filter, not a
    post-hoc UI filter."""
    store = VectorStore(tmp_path / "chroma")
    embedder = _StubEmbedder()
    index_tweets(
        [Tweet(id="old1", username="a", text="AI regulation update from an old run")],
        "technology",
        datetime.now(timezone.utc),
        store,
        embedder,
    )
    index_tweets(
        [Tweet(id="new1", username="b", text="AI regulation update from the current run")],
        "technology",
        datetime.now(timezone.utc),
        store,
        embedder,
    )

    chunks = retrieve(
        "AI regulation",
        store,
        embedder,
        category="technology",
        tweet_ids={"new1"},
        min_similarity=0.0,
    )

    assert {c.tweet_id for c in chunks} == {"new1"}


def test_retrieve_empty_tweet_ids_returns_no_evidence(tmp_path) -> None:
    """An empty allow-list (e.g. the current report has zero tweets) must
    mean "no evidence", not "unrestricted" - never silently fall back to
    a broader search than the caller asked for."""
    store, embedder = _seeded_store(tmp_path)

    chunks = retrieve(
        "AI regulation", store, embedder, category="technology", tweet_ids=set(), min_similarity=0.0
    )

    assert chunks == []


def test_retrieve_tweet_ids_none_is_unchanged_from_before(tmp_path) -> None:
    """`tweet_ids=None` (the default) must be byte-for-byte the pre-existing
    category-only-scoped behavior - backward compatible for every existing
    caller that never passes it."""
    store, embedder = _seeded_store(tmp_path)

    chunks = retrieve(
        "AI regulation", store, embedder, category="technology", tweet_ids=None, min_similarity=0.0
    )

    assert {c.tweet_id for c in chunks} == {"1", "2", "3"}
