"""Unit tests for app.signal_score: pure functions, no I/O, no LLM/network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models import Tweet
from app.schemas import RankedAccount
from app.signal_score import compute_confidence, compute_momentum, compute_signal_score


def _account(**overrides) -> RankedAccount:
    defaults = dict(
        rank=1,
        username="espn",
        category_relevance=80.0,
        engagement_score=60.0,
        activity_score=50.0,
        audience_score=70.0,
        ranking_score=65.0,
    )
    defaults.update(overrides)
    return RankedAccount(**defaults)


def _tweet(hours_ago: float | None, tweet_id: str = "1") -> Tweet:
    created_at = None
    if hours_ago is not None:
        created_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return Tweet(id=tweet_id, text="hello", created_at=created_at)


def test_compute_momentum_favors_recent_tweets() -> None:
    all_recent = [_tweet(1, "1"), _tweet(2, "2"), _tweet(3, "3")]
    all_old = [_tweet(200, "1"), _tweet(300, "2"), _tweet(400, "3")]

    assert compute_momentum(all_recent) > compute_momentum(all_old)


def test_compute_momentum_zero_when_no_tweets() -> None:
    assert compute_momentum([]) == 0.0


def test_compute_momentum_ignores_missing_created_at_without_crashing() -> None:
    tweets = [_tweet(None, "1"), _tweet(None, "2")]

    assert compute_momentum(tweets) == 0.0


def test_compute_momentum_mixed_missing_and_dated_tweets() -> None:
    tweets = [_tweet(None, "1"), _tweet(1, "2")]

    # Only the dated tweet counts toward the fraction - must not crash.
    momentum = compute_momentum(tweets)
    assert momentum > 0.0


def test_compute_signal_score_uses_real_ranked_account_fields() -> None:
    account = _account(category_relevance=100.0, engagement_score=100.0, audience_score=100.0)

    score = compute_signal_score(account, account_tweets=[])

    # No momentum contribution (no tweets) but the other three terms are maxed.
    assert score == round(0.35 * 100 + 0.30 * 100 + 0.20 * 100, 2)


def test_compute_signal_score_never_exceeds_100() -> None:
    account = _account(category_relevance=100.0, engagement_score=100.0, audience_score=100.0)
    tweets = [_tweet(0.5, str(i)) for i in range(5)]

    score = compute_signal_score(account, account_tweets=tweets)

    assert 0.0 <= score <= 100.0


def test_compute_signal_score_handles_missing_created_at_without_crashing() -> None:
    account = _account()
    tweets = [_tweet(None, "1"), _tweet(None, "2")]

    score = compute_signal_score(account, account_tweets=tweets)

    assert score >= 0.0


def test_compute_confidence_zero_evidence_floors_low() -> None:
    label, score = compute_confidence(evidence_tweet_count=0, independent_account_count=0)

    assert label == "Low"
    assert score == 0.0


def test_compute_confidence_high_with_strong_evidence() -> None:
    label, score = compute_confidence(evidence_tweet_count=10, independent_account_count=9)

    assert label == "High"
    assert score == 100.0


def test_compute_confidence_medium_band() -> None:
    label, score = compute_confidence(evidence_tweet_count=3, independent_account_count=2)

    assert label == "Medium"
    assert 35.0 <= score < 70.0


def test_compute_confidence_never_exceeds_100() -> None:
    label, score = compute_confidence(evidence_tweet_count=1000, independent_account_count=1000)

    assert label == "High"
    assert score == 100.0
