"""Swappable reranking function - v1 is a cheap deterministic heuristic
blending vector similarity with recency and engagement, not a second model.

A cross-encoder-based reranker would mean a second model download/load on
top of the embedder, directly against the "don't break the app to add a
reranker" guidance - this heuristic needs zero new imports, is trivially
unit-testable, and naturally favors recent evidence, which also serves
"what changed recently" questions well. Kept behind this same function
signature so a model-based reranker can replace it later with zero blast
radius elsewhere.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.rag.retriever import RetrievedChunk

#: Within the spec's suggested 5-8 evidence items sent to the LLM.
DEFAULT_TOP_N = 6


def _recency_score(created_at: str) -> float:
    if not created_at:
        return 0.0
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - parsed).total_seconds() / 86400
    return max(0.0, 1.0 - age_days / 30)


def _engagement_score(engagement: int) -> float:
    return min(1.0, engagement / 10_000)


def rerank(chunks: list[RetrievedChunk], top_n: int = DEFAULT_TOP_N) -> list[RetrievedChunk]:
    """Blend `0.7*similarity + 0.2*recency + 0.1*engagement`, sorted
    descending, truncated to `top_n`."""

    def _score(chunk: RetrievedChunk) -> float:
        return (
            0.7 * chunk.similarity
            + 0.2 * _recency_score(chunk.created_at)
            + 0.1 * _engagement_score(chunk.engagement)
        )

    return sorted(chunks, key=_score, reverse=True)[:top_n]
