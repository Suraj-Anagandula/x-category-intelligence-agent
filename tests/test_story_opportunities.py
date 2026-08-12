"""Unit tests for app.story_opportunities.derive_story_opportunities - pure
functions over already-computed fields, no I/O.
"""

from __future__ import annotations

from app.report_compare import AccountMover, ComparisonResult
from app.story_opportunities import SIGNAL_SCORE_THRESHOLD, derive_story_opportunities


def _report(topics=None, tweets=None):
    return {
        "tweets": tweets or [],
        "analysis": {"trending_topics": topics or []},
    }


def _tweets_for_topic(topic_word: str, count: int, distinct_authors: int) -> list[dict]:
    return [
        {
            "id": str(i),
            "username": f"author{i % distinct_authors}",
            "text": f"{topic_word} update {i}",
        }
        for i in range(count)
    ]


def test_derive_story_opportunities_includes_strong_signal() -> None:
    tweets = _tweets_for_topic("regulation", count=10, distinct_authors=6)
    report = _report(topics=["AI regulation"], tweets=tweets)

    opportunities = derive_story_opportunities(report)

    assert len(opportunities) == 1
    assert opportunities[0].title == "AI regulation"
    assert opportunities[0].kind == "signal"
    assert opportunities[0].confidence_label in ("Medium", "High")
    assert opportunities[0].evidence_tweet_ids


def test_derive_story_opportunities_excludes_weak_signal() -> None:
    """A topic with almost no supporting evidence must not become an
    opportunity - never invent a signal from thin evidence."""
    tweets = [{"id": "1", "username": "a", "text": "unrelated mars content"}]
    report = _report(topics=["Quantum computing breakthroughs"], tweets=tweets)

    opportunities = derive_story_opportunities(report)

    assert opportunities == []


def test_derive_story_opportunities_no_topics_returns_empty() -> None:
    report = _report(topics=[], tweets=[])

    assert derive_story_opportunities(report) == []


def test_derive_story_opportunities_includes_new_topic_from_comparison() -> None:
    tweets = _tweets_for_topic("compliance", count=8, distinct_authors=5)
    report = _report(topics=["AI compliance"], tweets=tweets)
    comparison = ComparisonResult(
        older_label="technology — 2026-08-08",
        newer_label="technology — 2026-08-10",
        topics_added=["AI compliance"],
    )

    opportunities = derive_story_opportunities(report, comparison=comparison)

    new_topic_opportunities = [o for o in opportunities if o.kind == "new_topic"]
    assert len(new_topic_opportunities) == 1
    assert "didn't appear" in new_topic_opportunities[0].why_it_matters


def test_derive_story_opportunities_includes_rising_account_mover() -> None:
    report = _report(topics=[], tweets=[])
    comparison = ComparisonResult(
        older_label="technology — 2026-08-08",
        newer_label="technology — 2026-08-10",
        movers=[AccountMover(username="tim_cook", rank_delta=12, score_delta=4.0)],
    )

    opportunities = derive_story_opportunities(report, comparison=comparison)

    rank_shift = [o for o in opportunities if o.kind == "rank_shift"]
    assert len(rank_shift) == 1
    assert rank_shift[0].account_usernames == ["tim_cook"]
    assert rank_shift[0].signal_score is None  # directly observed, not an inferred signal


def test_derive_story_opportunities_excludes_falling_movers() -> None:
    report = _report(topics=[], tweets=[])
    comparison = ComparisonResult(
        older_label="a",
        newer_label="b",
        movers=[AccountMover(username="declining_account", rank_delta=-5, score_delta=-10.0)],
    )

    opportunities = derive_story_opportunities(report, comparison=comparison)

    assert opportunities == []


def test_signal_score_threshold_is_a_named_constant() -> None:
    # Documents the actual threshold used - guards against silent drift.
    assert SIGNAL_SCORE_THRESHOLD == 60.0
