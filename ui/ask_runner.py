"""Bridges Streamlit's synchronous world to the async RAG flows (Ask
Intelligence, Story Brief generation) and the vector store/embedder they
share - mirrors `ui/pipeline_runner.py`'s `asyncio.run` bridge pattern.

The store/embedder are module-level singletons (not per-session) so the
~80MB embedding model is loaded once per process, not once per question -
this is a shared, stateless resource, not per-user state.
"""

from __future__ import annotations

import asyncio

from app.config import settings
from app.exceptions import RAGError
from app.llm import build_llm_client
from app.rag.agent import AskIntelligenceAnswer, ask_intelligence
from app.rag.embeddings import Embedder
from app.rag.indexer import backfill_from_snapshots
from app.rag.vector_store import VectorStore
from app.story_brief import StoryBrief, generate_story_brief
from app.story_opportunities import StoryOpportunity

_embedder: Embedder | None = None
_store: VectorStore | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(settings.chroma_dir)
    return _store


def index_size() -> int:
    """Number of tweets currently indexed - 0 (never raises) if RAG
    dependencies aren't installed or the store can't be opened, so the UI
    can always render a count rather than crashing on this check."""
    try:
        return _get_store().count()
    except RAGError:
        return 0


def rebuild_index() -> dict[str, int]:
    """Backfill every existing run snapshot into the index.

    Raises `RAGError` if RAG dependencies aren't installed - the caller
    shows a clear message (see `ui.components.render_error_message`)
    rather than a raw traceback.
    """
    settings.ensure_directories()
    return backfill_from_snapshots(settings, _get_store(), _get_embedder())


def ask(
    question: str, category: str | None = None, tweet_ids: set[str] | None = None
) -> AskIntelligenceAnswer:
    """Synchronous entry point for the Ask Intelligence tab.

    `tweet_ids=None` searches all indexed history for `category`; pass the
    current report's own tweet ids to scope the answer to just this run
    (see `ui/pages/intelligence.py`'s "Include historical posts" toggle).

    Raises `RAGError` if RAG dependencies aren't installed.
    """
    llm_client = build_llm_client(settings)
    return asyncio.run(
        ask_intelligence(
            question,
            store=_get_store(),
            embedder=_get_embedder(),
            llm_client=llm_client,
            category=category,
            tweet_ids=tweet_ids,
            min_similarity=settings.rag_min_similarity,
        )
    )


def generate_brief(
    opportunity: StoryOpportunity,
    category: str | None = None,
    tweet_ids: set[str] | None = None,
) -> StoryBrief:
    """Synchronous entry point for the Story Opportunities tab's "Generate
    Brief" action - reuses the same store/embedder singletons as `ask`.

    `tweet_ids` should always be the current report's own tweet id set
    (see `ui/pages/reports.py::_render_brief_for`) so evidence can never
    leak in from another run's data - see `app.story_brief.generate_story_brief`.

    Raises `RAGError` if RAG dependencies aren't installed.
    """
    llm_client = build_llm_client(settings)
    return asyncio.run(
        generate_story_brief(
            opportunity,
            store=_get_store(),
            embedder=_get_embedder(),
            llm_client=llm_client,
            category=category,
            tweet_ids=tweet_ids,
        )
    )
