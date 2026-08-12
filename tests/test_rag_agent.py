"""Unit tests for app.rag.agent.ask_intelligence and citation resolution.

Retrieval/reranking are stubbed directly (monkeypatched module-level
functions) so these tests focus purely on the ask-flow branching: zero-
evidence short-circuit, LLM-less deterministic fallback, malformed LLM
output, and citation-index validation - no real embedding model or vector
store needed here (those are covered by test_rag_retriever.py/test_rag_indexer.py).
"""

from __future__ import annotations

import app.rag.agent as agent_module
from app.exceptions import LLMError
from app.rag.agent import ask_intelligence, resolve_citations
from app.rag.retriever import RetrievedChunk


def _chunk(tweet_id: str, username: str = "openai", similarity: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        tweet_id=tweet_id,
        text=f"post {tweet_id}",
        username=username,
        url=f"https://x.com/i/status/{tweet_id}",
        created_at="2026-08-08T00:00:00+00:00",
        category="technology",
        similarity=similarity,
    )


class _StubLLMClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    async def generate_json(self, prompt: str):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.result

    async def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        return ""


def _patch_retrieval(monkeypatch, chunks: list[RetrievedChunk]) -> None:
    monkeypatch.setattr(agent_module, "retrieve", lambda *a, **k: chunks)
    monkeypatch.setattr(agent_module, "rerank", lambda chunks, top_n=6: chunks[:top_n])


async def test_ask_intelligence_passes_tweet_ids_through_to_retrieve(monkeypatch) -> None:
    """Wiring proof that scoping (e.g. to the current report's own tweets)
    actually reaches the retrieval call - see test_rag_retriever.py for the
    real data-layer enforcement this feeds."""
    captured = {}

    def _fake_retrieve(*args, **kwargs):
        captured.update(kwargs)
        return [_chunk("1")]

    monkeypatch.setattr(agent_module, "retrieve", _fake_retrieve)
    monkeypatch.setattr(agent_module, "rerank", lambda chunks, top_n=6: chunks[:top_n])

    await ask_intelligence("why?", store=None, embedder=None, llm_client=None, tweet_ids={"1", "2"})

    assert captured["tweet_ids"] == {"1", "2"}


async def test_ask_intelligence_zero_evidence_makes_no_llm_call(monkeypatch) -> None:
    _patch_retrieval(monkeypatch, [])
    llm = _StubLLMClient(result={"answer": "should never be called"})

    answer = await ask_intelligence("anything", store=None, embedder=None, llm_client=llm)

    assert answer.insufficient_evidence is True
    assert answer.citations == []
    assert llm.calls == []


async def test_ask_intelligence_deterministic_fallback_when_no_llm_configured(
    monkeypatch,
) -> None:
    chunks = [_chunk("1"), _chunk("2", username="nasa")]
    _patch_retrieval(monkeypatch, chunks)

    answer = await ask_intelligence("why?", store=None, embedder=None, llm_client=None)

    assert answer.used_llm is False
    assert answer.insufficient_evidence is False
    assert {c.url for c in answer.citations} == {
        "https://x.com/i/status/1",
        "https://x.com/i/status/2",
    }


async def test_ask_intelligence_uses_llm_json_response(monkeypatch) -> None:
    chunks = [_chunk("1"), _chunk("2")]
    _patch_retrieval(monkeypatch, chunks)
    llm = _StubLLMClient(
        result={
            "answer": "AI regulation is trending. [1]",
            "sufficient": True,
            "cited_indices": [1],
        }
    )

    answer = await ask_intelligence("why?", store=None, embedder=None, llm_client=llm)

    assert answer.used_llm is True
    assert answer.insufficient_evidence is False
    assert answer.answer == "AI regulation is trending. [1]"
    assert len(answer.citations) == 1
    assert answer.citations[0].url == "https://x.com/i/status/1"


async def test_ask_intelligence_respects_llm_reported_insufficiency(monkeypatch) -> None:
    chunks = [_chunk("1")]
    _patch_retrieval(monkeypatch, chunks)
    llm = _StubLLMClient(
        result={
            "answer": "The evidence doesn't address this.",
            "sufficient": False,
            "cited_indices": [],
        }
    )

    answer = await ask_intelligence("unrelated question", store=None, embedder=None, llm_client=llm)

    assert answer.insufficient_evidence is True


async def test_ask_intelligence_falls_back_when_llm_errors(monkeypatch) -> None:
    chunks = [_chunk("1")]
    _patch_retrieval(monkeypatch, chunks)
    llm = _StubLLMClient(error=LLMError("boom"))

    answer = await ask_intelligence("why?", store=None, embedder=None, llm_client=llm)

    assert answer.used_llm is False
    assert answer.citations  # still has real, non-fabricated citations


async def test_ask_intelligence_falls_back_when_llm_returns_non_dict(monkeypatch) -> None:
    chunks = [_chunk("1")]
    _patch_retrieval(monkeypatch, chunks)
    llm = _StubLLMClient(result=["not", "a", "dict"])

    answer = await ask_intelligence("why?", store=None, embedder=None, llm_client=llm)

    assert answer.used_llm is False


async def test_ask_intelligence_falls_back_when_answer_field_empty(monkeypatch) -> None:
    chunks = [_chunk("1")]
    _patch_retrieval(monkeypatch, chunks)
    llm = _StubLLMClient(result={"answer": "", "sufficient": True, "cited_indices": []})

    answer = await ask_intelligence("why?", store=None, embedder=None, llm_client=llm)

    assert answer.used_llm is False


def test_resolve_citations_drops_out_of_range_indices() -> None:
    chunks = [_chunk("1"), _chunk("2")]

    citations = resolve_citations(chunks, [0, 1, 2, 5, -1])

    assert [c.url for c in citations] == [
        "https://x.com/i/status/1",
        "https://x.com/i/status/2",
    ]


def test_resolve_citations_drops_non_integer_indices() -> None:
    chunks = [_chunk("1")]

    citations = resolve_citations(chunks, ["1", 1.5, None, 1])

    assert len(citations) == 1
    assert citations[0].url == "https://x.com/i/status/1"


def test_resolve_citations_empty_when_no_valid_indices() -> None:
    chunks = [_chunk("1")]

    assert resolve_citations(chunks, [99]) == []


async def test_ask_intelligence_falls_back_to_all_chunks_when_llm_cites_nothing_valid(
    monkeypatch,
) -> None:
    chunks = [_chunk("1"), _chunk("2")]
    _patch_retrieval(monkeypatch, chunks)
    llm = _StubLLMClient(
        result={
            "answer": "General answer with no valid citations.",
            "sufficient": True,
            "cited_indices": [99],
        }
    )

    answer = await ask_intelligence("why?", store=None, embedder=None, llm_client=llm)

    # No fabricated citation - falls back to citing all retrieved evidence
    # rather than leaving the answer with zero source attribution.
    assert len(answer.citations) == 2
