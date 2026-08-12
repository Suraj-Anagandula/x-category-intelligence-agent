"""Signal Score and Confidence: pure functions over fields already computed
elsewhere in the pipeline. No I/O, fully unit-testable in isolation -
mirrors `app/account_ranker.py`'s convention.

Neither function invents a metric for the sake of having a number: every
term is either read directly off `RankedAccount` (already used in ranking)
or derived from real `Tweet.created_at` timestamps.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.account_ranker import normalize_log
from app.models import Tweet
from app.schemas import RankedAccount

#: (category_relevance, engagement_score, audience_score, momentum)
SIGNAL_SCORE_WEIGHTS = (0.35, 0.30, 0.20, 0.15)


def compute_momentum(tweets: list[Tweet]) -> float:
    """Recency-weighted frequency: the log-normalized share of `tweets`
    posted in the last 24 hours, based on real `Tweet.created_at`
    timestamps.

    `Tweet.created_at` is `datetime | None` - tweets with a missing
    timestamp are excluded from both the numerator and denominator rather
    than crashing. If none of the given tweets have a usable timestamp,
    momentum is 0.0 (never a fabricated value).
    """
    dated = [t for t in tweets if t.created_at is not None]
    if not dated:
        return 0.0

    now = datetime.now(timezone.utc)

    def _age_hours(tweet: Tweet) -> float:
        created = tweet.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (now - created).total_seconds() / 3600

    recent = sum(1 for tweet in dated if _age_hours(tweet) <= 24)
    fraction = recent / len(dated)
    return normalize_log(fraction * 100, cap=100.0)


def compute_signal_score(
    account: RankedAccount,
    account_tweets: list[Tweet] | None = None,
    weights: tuple[float, float, float, float] = SIGNAL_SCORE_WEIGHTS,
) -> float:
    """Composite 0-100 signal-strength score for one account:

        signal_score = 0.35 * category_relevance  (RankedAccount - LLM/heuristic relevance)
                      + 0.30 * engagement_score     (RankedAccount - real per-tweet engagement)
                      + 0.20 * audience_score       (RankedAccount - log-normalized followers)
                      + 0.15 * momentum             (derived here from real Tweet.created_at)

    `account_tweets` may be omitted (or empty) if per-tweet timestamps
    aren't available for this account in the current context - momentum
    then defaults to 0.0 rather than raising.
    """
    relevance_w, engagement_w, audience_w, momentum_w = weights
    momentum = compute_momentum(account_tweets or [])
    score = (
        relevance_w * account.category_relevance
        + engagement_w * account.engagement_score
        + audience_w * account.audience_score
        + momentum_w * momentum
    )
    return round(max(0.0, min(100.0, score)), 2)


def compute_confidence(
    evidence_tweet_count: int, independent_account_count: int
) -> tuple[str, float]:
    """Heuristic confidence bucket over countable evidence - NOT a
    statistical confidence interval. `independent_account_count` (distinct
    source accounts) and `evidence_tweet_count` (supporting posts) are both
    real, countable quantities; this only buckets them into a label a
    non-technical reader can act on.

    Zero evidence always floors at ("Low", 0.0) - confidence is never
    hidden or inferred without evidence to back it.
    """
    if evidence_tweet_count <= 0 or independent_account_count <= 0:
        return "Low", 0.0

    score = min(100.0, independent_account_count * 20 + min(evidence_tweet_count, 10) * 3)
    if score >= 70:
        label = "High"
    elif score >= 35:
        label = "Medium"
    else:
        label = "Low"
    return label, round(score, 2)
