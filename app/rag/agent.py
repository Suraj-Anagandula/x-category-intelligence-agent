"""Ask Intelligence: retrieval-augmented Q&A over previously-collected X
posts. Uses ONLY the retrieved evidence for a given question - never the
full indexed database, never a fabricated citation or URL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.exceptions import LLMError
from app.llm import LLMProvider
from app.logger import get_logger
from app.rag.embeddings import Embedder
from app.rag.reranker import DEFAULT_TOP_N, rerank
from app.rag.retriever import DEFAULT_TOP_K_CANDIDATES, RetrievedChunk, retrieve
from app.rag.vector_store import VectorStore

logger = get_logger()


@dataclass
class Citation:
    username: str
    url: str
    text_excerpt: str
    created_at: str


@dataclass
class AskIntelligenceAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    insufficient_evidence: bool = False
    used_llm: bool = False


_PROMPT_TEMPLATE = (
    "You are answering a question using ONLY the X (Twitter) posts listed "
    "below as evidence. Every post includes its real URL - cite it by "
    "number when you reference it. Do not invent facts, statistics, or "
    "URLs not present in the evidence. If the evidence is insufficient or "
    "doesn't actually address the question, say so explicitly rather than "
    "guessing.\n\n"
    "Question: {question}\n\n"
    "Evidence:\n{evidence}\n\n"
    'Return ONLY JSON: {{"answer": "<answer text, citing posts as [n]>", '
    '"sufficient": <true/false>, "cited_indices": [<1-based ints>]}}'
)


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        when = chunk.created_at or "unknown time"
        lines.append(
            f'[{index}] @{chunk.username or "unknown"} ({when}): "{chunk.text}" — {chunk.url}'
        )
    return "\n".join(lines)


def resolve_citations(chunks: list[RetrievedChunk], indices: list[int]) -> list[Citation]:
    """Map 1-based `indices` (as returned by an LLM) back to real
    `RetrievedChunk`s - out-of-range/invalid indices are dropped, never
    fabricated, matching `app/account_ranker.py`'s existing defensive
    pattern for LLM-returned values. Shared by `ask_intelligence` and
    `app.story_brief.generate_story_brief`.
    """
    citations: list[Citation] = []
    for index in indices:
        if not isinstance(index, int) or not (1 <= index <= len(chunks)):
            continue
        chunk = chunks[index - 1]
        citations.append(
            Citation(
                username=chunk.username,
                url=chunk.url,
                text_excerpt=chunk.text,
                created_at=chunk.created_at,
            )
        )
    return citations


def _deterministic_answer(question: str, chunks: list[RetrievedChunk]) -> AskIntelligenceAnswer:
    """No LLM configured (or the LLM call failed) - degrade to "here are
    the raw matching posts" rather than fail outright, matching every
    other LLM-optional path in this codebase (e.g.
    `CategoryAgent._build_context_fallback`, `analyze_category`'s
    deterministic path)."""
    preview = "; ".join(f'@{c.username}: "{c.text}"' for c in chunks[:3])
    answer = f"No LLM answer is available right now. Closest matching posts: {preview}"
    citations = resolve_citations(chunks, list(range(1, len(chunks) + 1)))
    return AskIntelligenceAnswer(
        question=question, answer=answer, citations=citations, used_llm=False
    )


async def ask_intelligence(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    llm_client: LLMProvider | None,
    category: str | None = None,
    tweet_ids: set[str] | list[str] | None = None,
    min_similarity: float = 0.30,
    top_k_candidates: int = DEFAULT_TOP_K_CANDIDATES,
    top_n_context: int = DEFAULT_TOP_N,
) -> AskIntelligenceAnswer:
    """Retrieve evidence for `question` and answer using ONLY that
    evidence.

    `tweet_ids`, when given, restricts retrieval to exactly that set of
    tweets (e.g. the currently loaded report's own dataset) - `None` means
    an unscoped, cross-run historical search across everything indexed for
    `category`. Callers should default to scoping (pass the current
    report's tweet ids) and only search unscoped when the user has
    explicitly asked for historical/cross-run evidence (see
    `ui/pages/intelligence.py`'s "Include historical posts" toggle).

    Never calls the LLM when no evidence survives retrieval (honest and
    cheap - `insufficient_evidence=True` is returned immediately). Falls
    back to a deterministic answer if no LLM is configured or the LLM call
    fails.
    """
    candidates = retrieve(
        question,
        store,
        embedder,
        category=category,
        tweet_ids=tweet_ids,
        min_similarity=min_similarity,
        top_k=top_k_candidates,
    )
    chunks = rerank(candidates, top_n=top_n_context)

    if not chunks:
        return AskIntelligenceAnswer(
            question=question,
            answer="Insufficient evidence in the indexed data to answer this question.",
            insufficient_evidence=True,
        )

    if llm_client is None:
        return _deterministic_answer(question, chunks)

    prompt = _PROMPT_TEMPLATE.format(question=question, evidence=_format_evidence(chunks))
    try:
        result = await llm_client.generate_json(prompt)
    except LLMError as exc:
        logger.warning(f"Ask Intelligence LLM call failed, falling back to raw evidence: {exc}")
        return _deterministic_answer(question, chunks)

    if not isinstance(result, dict):
        return _deterministic_answer(question, chunks)

    answer_text = str(result.get("answer") or "").strip()
    if not answer_text:
        return _deterministic_answer(question, chunks)

    sufficient = bool(result.get("sufficient", True))
    raw_indices = result.get("cited_indices") or []
    indices = [i for i in raw_indices if isinstance(i, int)]
    citations = resolve_citations(chunks, indices)
    if not citations:
        citations = resolve_citations(chunks, list(range(1, len(chunks) + 1)))

    return AskIntelligenceAnswer(
        question=question,
        answer=answer_text,
        citations=citations,
        insufficient_evidence=not sufficient,
        used_llm=True,
    )
