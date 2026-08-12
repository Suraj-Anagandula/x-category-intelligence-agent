"""Question -> embed -> vector-store query -> similarity-threshold filter ->
`RetrievedChunk` list.

Never sends the whole indexed database anywhere - only the handful of
chunks this returns ever reach a prompt (see `app/rag/agent.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorStore

#: Candidates pulled from the vector store before threshold-filtering/
#: reranking - matches the spec's "top-20 candidates" guidance.
DEFAULT_TOP_K_CANDIDATES = 20


@dataclass
class RetrievedChunk:
    """One piece of retrieved evidence. Kept local to `app/rag/`, not added
    to `app/schemas.py`, which stays scoped to core pipeline output."""

    tweet_id: str
    text: str
    username: str
    url: str
    created_at: str
    category: str
    similarity: float
    engagement: int = 0


def _distance_to_similarity(distance: float) -> float:
    """`sentence-transformers` output is close to unit-norm, so `1 -
    distance` closely tracks cosine similarity for Chroma's default
    (squared L2) space here. Clamped to [0, 1] since this is used as a
    plain "how relevant" signal for thresholding/reranking, never
    presented as a statistical probability."""
    return max(0.0, min(1.0, 1.0 - distance))


def _build_where(category: str | None, tweet_ids: set[str] | list[str] | None) -> dict | None:
    """Combine an optional `category` filter with an optional `tweet_id`
    allow-list filter. Kept as a single `{"category": ...}` dict (not
    wrapped in `$and`) when only `category` is given, to match the exact
    shape this module has always produced - existing callers/tests that
    never pass `tweet_ids` see no behavior change at all.

    `tweet_ids`, when given, restricts retrieval to exactly that set of
    already-known-good tweet ids (e.g. the tweets that belong to a
    specific analysis run) - this is the data-layer enforcement that keeps
    evidence from a global/cross-run index from leaking into a single
    report's story evidence (see `app/story_brief.py`).
    """
    clauses = []
    if category:
        clauses.append({"category": category})
    if tweet_ids is not None:
        clauses.append({"tweet_id": {"$in": list(tweet_ids)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    category: str | None = None,
    tweet_ids: set[str] | list[str] | None = None,
    min_similarity: float = 0.30,
    top_k: int = DEFAULT_TOP_K_CANDIDATES,
) -> list[RetrievedChunk]:
    """Embed `question`, query `store` (optionally scoped to `category`
    and/or an explicit `tweet_ids` allow-list via Chroma's `where` filter),
    and drop any hit below `min_similarity`.

    `tweet_ids`, when given, is a hard allow-list: only chunks whose
    `tweet_id` metadata is in that set can ever be returned, regardless of
    what else is indexed. Passing an empty set/list returns `[]` (no
    evidence), not "unrestricted" - the caller controls that distinction.

    Returns `[]` - never raises - if the store is empty or nothing clears
    the threshold; the caller (`app.rag.agent.ask_intelligence`) treats an
    empty list as "insufficient evidence".
    """
    if tweet_ids is not None and not tweet_ids:
        return []

    embeddings = embedder.embed_texts([question])
    if not embeddings:
        return []
    embedding = embeddings[0]

    where = _build_where(category, tweet_ids)
    hits = store.query(embedding, top_k=top_k, where=where)

    chunks: list[RetrievedChunk] = []
    for hit in hits:
        similarity = _distance_to_similarity(hit.distance)
        if similarity < min_similarity:
            continue
        metadata = hit.metadata
        engagement = (
            int(metadata.get("like_count") or 0)
            + int(metadata.get("retweet_count") or 0)
            + int(metadata.get("reply_count") or 0)
        )
        chunks.append(
            RetrievedChunk(
                tweet_id=metadata.get("tweet_id", hit.id),
                text=hit.document,
                username=metadata.get("username", ""),
                url=metadata.get("url", ""),
                created_at=metadata.get("created_at", ""),
                category=metadata.get("category", ""),
                similarity=round(similarity, 4),
                engagement=engagement,
            )
        )
    return chunks
