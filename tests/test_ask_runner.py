"""Unit tests for ui.ask_runner - the Streamlit sync bridge for Ask
Intelligence / Story Brief generation.

Uses a stub embedder (monkeypatched directly onto the module's singleton)
so no real sentence-transformers model loads in these tests, and a
tmp_path-backed chroma_dir so nothing touches the real `.chroma/` index.
"""

from __future__ import annotations

import json

from app.config import Settings
from app.story_brief import StoryBrief
from app.story_opportunities import StoryOpportunity
from ui import ask_runner


class _StubEmbedder:
    """Constant-direction embedder - fine for exercising the plumbing
    (empty-index handling, rebuild wiring, deterministic fallback), not for
    testing retrieval quality (that's covered by tests/test_rag_retriever.py
    with a more discriminating stub)."""

    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _isolated_settings(tmp_path) -> Settings:
    settings = Settings()
    settings.chroma_dir = tmp_path / "chroma"
    settings.tweets_output_dir = tmp_path / "tweets"
    return settings


def _reset_singletons(monkeypatch, settings) -> None:
    monkeypatch.setattr(ask_runner, "settings", settings)
    monkeypatch.setattr(ask_runner, "_store", None)
    monkeypatch.setattr(ask_runner, "_embedder", _StubEmbedder())
    # ask()/generate_brief() call build_llm_client(settings) - with no API
    # key configured, this returns None, exercising the deterministic path.
    monkeypatch.setattr(ask_runner, "build_llm_client", lambda _settings: None)


def test_index_size_zero_on_fresh_empty_store(tmp_path, monkeypatch) -> None:
    settings = _isolated_settings(tmp_path)
    _reset_singletons(monkeypatch, settings)

    assert ask_runner.index_size() == 0


def test_ask_returns_insufficient_evidence_on_empty_index(tmp_path, monkeypatch) -> None:
    """The core "graceful degradation" requirement: Ask Intelligence must
    not crash - and must say so explicitly - when nothing has been
    indexed yet (a fresh install, before any "Build/Refresh Index" click)."""
    settings = _isolated_settings(tmp_path)
    _reset_singletons(monkeypatch, settings)

    answer = ask_runner.ask("What is trending?", category="technology")

    assert answer.insufficient_evidence is True
    assert answer.citations == []


def test_rebuild_index_and_ask_finds_backfilled_evidence(tmp_path, monkeypatch) -> None:
    settings = _isolated_settings(tmp_path)
    _reset_singletons(monkeypatch, settings)

    run_dir = settings.tweets_output_dir / "technology"
    run_dir.mkdir(parents=True)
    payload = {
        "category": "technology",
        "scraped_at": "2026-08-08T00:00:00+00:00",
        "tweets": [{"id": "1", "username": "openai", "text": "New AI regulation announced"}],
    }
    (run_dir / "2026-08-08.json").write_text(json.dumps(payload), encoding="utf-8")

    results = ask_runner.rebuild_index()
    assert results == {"technology/2026-08-08": 1}
    assert ask_runner.index_size() == 1

    answer = ask_runner.ask("AI regulation", category="technology")

    assert answer.insufficient_evidence is False
    assert answer.used_llm is False  # no LLM configured -> deterministic fallback
    assert answer.citations
    assert answer.citations[0].url == "https://x.com/i/status/1"


def test_ask_with_tweet_ids_excludes_tweets_outside_the_allow_list(tmp_path, monkeypatch) -> None:
    """Regression for the reported evidence leak: scoping Ask Intelligence
    to a specific set of tweet ids (e.g. the currently loaded report's own
    tweets) must never surface a tweet outside that set, even though it's
    indexed under the same category."""
    settings = _isolated_settings(tmp_path)
    _reset_singletons(monkeypatch, settings)

    run_dir = settings.tweets_output_dir / "technology"
    run_dir.mkdir(parents=True)
    payload = {
        "category": "technology",
        "scraped_at": "2026-08-08T00:00:00+00:00",
        "tweets": [
            {"id": "old1", "username": "openai", "text": "AI regulation news"},
            {"id": "new1", "username": "openai", "text": "AI regulation update"},
        ],
    }
    (run_dir / "2026-08-08.json").write_text(json.dumps(payload), encoding="utf-8")
    ask_runner.rebuild_index()

    answer = ask_runner.ask("AI regulation", category="technology", tweet_ids={"new1"})

    assert {c.url for c in answer.citations} == {"https://x.com/i/status/new1"}


def test_generate_brief_passes_tweet_ids_through_to_story_brief_generation(
    tmp_path, monkeypatch
) -> None:
    settings = _isolated_settings(tmp_path)
    _reset_singletons(monkeypatch, settings)
    captured = {}

    async def _fake_generate_story_brief(
        opportunity, store, embedder, llm_client, category=None, tweet_ids=None
    ):
        captured["tweet_ids"] = tweet_ids
        return StoryBrief(headline="h", why_it_matters="w")

    monkeypatch.setattr(ask_runner, "generate_story_brief", _fake_generate_story_brief)

    opportunity = StoryOpportunity(title="t", why_it_matters="w", confidence_label="High")
    ask_runner.generate_brief(opportunity, category="technology", tweet_ids={"new1"})

    assert captured["tweet_ids"] == {"new1"}


def test_generate_brief_degrades_gracefully_with_no_evidence(tmp_path, monkeypatch) -> None:
    settings = _isolated_settings(tmp_path)
    _reset_singletons(monkeypatch, settings)

    opportunity = StoryOpportunity(
        title="Nonexistent topic", why_it_matters="test", confidence_label="High"
    )

    brief = ask_runner.generate_brief(opportunity, category="technology")

    assert brief.note  # explicit incompleteness notice, never a silently-thin brief
