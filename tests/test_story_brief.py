"""Unit tests for app.story_brief.generate_story_brief.

Retrieval/reranking are stubbed directly (monkeypatched module-level
functions in app.story_brief), mirroring tests/test_rag_agent.py's style -
no real embedding model or vector store needed here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import app.story_brief as story_brief_module
from app.exceptions import LLMError
from app.models import Tweet
from app.rag.indexer import index_tweets
from app.rag.retriever import RetrievedChunk
from app.rag.vector_store import VectorStore
from app.story_brief import generate_story_brief
from app.story_opportunities import StoryOpportunity


def _chunk(tweet_id: str, username: str = "openai") -> RetrievedChunk:
    return RetrievedChunk(
        tweet_id=tweet_id,
        text=f"post {tweet_id}",
        username=username,
        url=f"https://x.com/i/status/{tweet_id}",
        created_at="2026-08-08T00:00:00+00:00",
        category="technology",
        similarity=0.9,
    )


def _opportunity(**overrides) -> StoryOpportunity:
    defaults = dict(
        title="AI regulation",
        why_it_matters="High signal, strong confidence.",
        signal_score=80.0,
        confidence_label="High",
        account_usernames=["openai"],
    )
    defaults.update(overrides)
    return StoryOpportunity(**defaults)


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
        return ""


def _patch_retrieval(monkeypatch, chunks: list[RetrievedChunk]) -> None:
    monkeypatch.setattr(story_brief_module, "retrieve", lambda *a, **k: chunks)
    monkeypatch.setattr(story_brief_module, "rerank", lambda chunks, top_n=6: chunks[:top_n])


async def test_generate_story_brief_degrades_on_zero_evidence(monkeypatch) -> None:
    _patch_retrieval(monkeypatch, [])
    llm = _StubLLMClient(result={"headline": "should not be called"})

    brief = await generate_story_brief(_opportunity(), store=None, embedder=None, llm_client=llm)

    assert brief.note
    assert brief.observed_facts == []
    assert brief.ai_interpretation == []
    assert llm.calls == []


async def test_generate_story_brief_degrades_when_no_llm_configured(monkeypatch) -> None:
    _patch_retrieval(monkeypatch, [_chunk("1")])

    brief = await generate_story_brief(_opportunity(), store=None, embedder=None, llm_client=None)

    assert "No LLM configured" in brief.note
    assert brief.headline == "AI regulation"


async def test_generate_story_brief_separates_fact_interpretation_and_questions(
    monkeypatch,
) -> None:
    _patch_retrieval(monkeypatch, [_chunk("1"), _chunk("2")])
    llm = _StubLLMClient(
        result={
            "headline": "AI regulation debate intensifies",
            "observed_facts": ["Multiple accounts posted about new AI compliance rules."],
            "ai_interpretation": ["This suggests growing regulatory pressure on AI companies."],
            "suggested_investigation_questions": ["Which companies are most affected?"],
            "supporting_posts": [1, 2],
        }
    )

    brief = await generate_story_brief(_opportunity(), store=None, embedder=None, llm_client=llm)

    assert brief.headline == "AI regulation debate intensifies"
    assert brief.observed_facts == ["Multiple accounts posted about new AI compliance rules."]
    assert brief.ai_interpretation == ["This suggests growing regulatory pressure on AI companies."]
    assert brief.suggested_investigation_questions == ["Which companies are most affected?"]
    assert brief.note == ""
    assert len(brief.supporting_posts) == 2


async def test_generate_story_brief_falls_back_on_llm_error(monkeypatch) -> None:
    _patch_retrieval(monkeypatch, [_chunk("1")])
    llm = _StubLLMClient(error=LLMError("boom"))

    brief = await generate_story_brief(_opportunity(), store=None, embedder=None, llm_client=llm)

    assert "AI brief generation failed" in brief.note


async def test_generate_story_brief_falls_back_on_non_dict_response(monkeypatch) -> None:
    _patch_retrieval(monkeypatch, [_chunk("1")])
    llm = _StubLLMClient(result=["not", "a", "dict"])

    brief = await generate_story_brief(_opportunity(), store=None, embedder=None, llm_client=llm)

    assert brief.note


async def test_generate_story_brief_never_fabricates_citations(monkeypatch) -> None:
    """Out-of-range supporting_posts indices must be dropped, and citations
    must still be real (non-fabricated) chunks - never invented URLs."""
    chunks = [_chunk("1"), _chunk("2")]
    _patch_retrieval(monkeypatch, chunks)
    llm = _StubLLMClient(
        result={
            "headline": "headline",
            "observed_facts": [],
            "ai_interpretation": [],
            "suggested_investigation_questions": [],
            "supporting_posts": [99, -1],
        }
    )

    brief = await generate_story_brief(_opportunity(), store=None, embedder=None, llm_client=llm)

    # Falls back to citing all retrieved evidence rather than zero citations.
    assert len(brief.supporting_posts) == 2
    assert all(c.url.startswith("https://x.com/i/status/") for c in brief.supporting_posts)


class _KeywordEmbedder:
    """Deterministic bag-of-words embedder over the vocabulary these
    end-to-end tests actually use - real Chroma query/where-filter
    behavior underneath, no monkeypatched `retrieve`/`rerank`."""

    _VOCAB = ["global", "health", "forum", "funding", "cuts", "outcomes"]

    def embed_texts(self, texts):
        return [[1.0 if word in text.lower() else 0.0 for word in self._VOCAB] for text in texts]


async def test_generate_story_brief_never_pulls_evidence_outside_tweet_ids(tmp_path) -> None:
    """Regression for the real reported leak: a June 2026 tweet from
    @healtheconomics must never appear as evidence for a story opportunity
    scoped to an August analysis run's own tweet_ids, even though both are
    indexed under the same "healthcare" category. Uses the real retrieve()/
    rerank() against a real (tmp_path) VectorStore - not a monkeypatched
    stub - so this actually exercises the data-layer enforcement, not just
    the call wiring."""
    store = VectorStore(tmp_path / "chroma")
    embedder = _KeywordEmbedder()
    old_tweet = Tweet(
        id="june1",
        username="healtheconomics",
        text="Global Health Forum funding cuts and health outcomes discussed",
    )
    new_tweet = Tweet(
        id="aug1",
        username="cdc",
        text="Global Health Forum funding cuts update from this week",
    )
    index_tweets(
        [old_tweet], "healthcare", datetime(2026, 6, 25, tzinfo=timezone.utc), store, embedder
    )
    index_tweets(
        [new_tweet], "healthcare", datetime(2026, 8, 10, tzinfo=timezone.utc), store, embedder
    )

    llm = _StubLLMClient(
        result={
            "headline": "Global Health Forum funding cuts",
            "observed_facts": ["Funding cuts were discussed."],
            "ai_interpretation": [],
            "suggested_investigation_questions": [],
            "supporting_posts": [1],
        }
    )
    opportunity = _opportunity(
        title="Global Health Forum funding cuts and health outcomes",
        account_usernames=["cdc"],
    )

    brief = await generate_story_brief(
        opportunity,
        store=store,
        embedder=embedder,
        llm_client=llm,
        category="healthcare",
        tweet_ids={"aug1"},
    )

    assert all(post.username != "healtheconomics" for post in brief.supporting_posts)
    assert "healtheconomics" not in brief.supporting_accounts
    assert set(brief.supporting_accounts) <= {"cdc"}


async def test_generate_story_brief_merges_opportunity_and_citation_accounts(monkeypatch) -> None:
    _patch_retrieval(monkeypatch, [_chunk("1", username="nasa")])
    llm = _StubLLMClient(
        result={
            "headline": "h",
            "observed_facts": [],
            "ai_interpretation": [],
            "suggested_investigation_questions": [],
            "supporting_posts": [1],
        }
    )

    brief = await generate_story_brief(
        _opportunity(account_usernames=["openai"]), store=None, embedder=None, llm_client=llm
    )

    assert set(brief.supporting_accounts) == {"openai", "nasa"}
