"""Unit tests for app.account_ranker: score normalization, weighted ranking, top-N."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.account_ranker import (
    compute_activity_score,
    compute_audience_score,
    compute_category_relevance,
    compute_category_relevance_llm,
    compute_engagement_score,
    compute_tweet_engagement_score,
    normalize_log,
    rank_accounts,
    rerank_with_tweet_engagement,
    select_top_n,
)
from app.exceptions import LLMError
from app.models import Tweet, UserProfile
from app.schemas import CategoryContext


def _profile(**overrides) -> UserProfile:
    defaults = dict(
        username="examplesports",
        display_name="Example Sports",
        bio="Official sports news and analysis.",
        followers=1_000_000,
        tweets=10_000,
        likes=5_000,
        created_at=datetime.now(timezone.utc) - timedelta(days=1000),
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


def _ctx() -> CategoryContext:
    return CategoryContext(
        category="sports",
        subcategories=["cricket", "football"],
        keywords=["sports", "athletes", "championship"],
    )


def test_normalize_log_bounds() -> None:
    assert normalize_log(None, cap=100) == 0.0
    assert normalize_log(0, cap=100) == 0.0
    assert normalize_log(100, cap=100) == 100.0
    assert 0.0 < normalize_log(10, cap=100) < 100.0


def test_normalize_log_never_exceeds_100_above_cap() -> None:
    assert normalize_log(1_000_000, cap=100) == 100.0


def test_compute_audience_score_scales_with_followers() -> None:
    small = compute_audience_score(_profile(followers=100))
    large = compute_audience_score(_profile(followers=10_000_000))

    assert 0.0 <= small <= 100.0
    assert 0.0 <= large <= 100.0
    assert large > small


def test_compute_activity_score_zero_without_history() -> None:
    assert compute_activity_score(_profile(tweets=None, created_at=None)) == 0.0
    assert compute_activity_score(_profile(tweets=0)) == 0.0


def test_compute_activity_score_positive_with_history() -> None:
    score = compute_activity_score(
        _profile(tweets=5000, created_at=datetime.now(timezone.utc) - timedelta(days=500))
    )

    assert 0.0 < score <= 100.0


def test_compute_engagement_score_zero_without_tweets_count() -> None:
    assert compute_engagement_score(_profile(tweets=None)) == 0.0
    assert compute_engagement_score(_profile(tweets=0)) == 0.0


def test_compute_category_relevance_scores_matching_bio_higher() -> None:
    relevant = compute_category_relevance(_profile(), _ctx())
    irrelevant = compute_category_relevance(
        _profile(username="randomuser", display_name="Random", bio="I like cooking."), _ctx()
    )

    assert 0.0 <= relevant <= 100.0
    assert relevant > irrelevant


def test_compute_category_relevance_empty_vocabulary_is_zero() -> None:
    empty_ctx = CategoryContext(category="", subcategories=[], keywords=[])
    assert compute_category_relevance(_profile(), empty_ctx) == 0.0


class _StubLLMClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    async def generate_json(self, prompt: str):
        if self.error:
            raise self.error
        return self.result


async def test_compute_category_relevance_llm_uses_llm_score() -> None:
    client = _StubLLMClient(result={"relevance": 87})

    score = await compute_category_relevance_llm(_profile(), _ctx(), client)

    assert score == 87.0


async def test_compute_category_relevance_llm_falls_back_on_error() -> None:
    client = _StubLLMClient(error=LLMError("boom"))

    score = await compute_category_relevance_llm(_profile(), _ctx(), client)

    assert score == compute_category_relevance(_profile(), _ctx())


async def test_compute_category_relevance_llm_falls_back_on_bad_payload() -> None:
    client = _StubLLMClient(result={"unexpected": "shape"})

    score = await compute_category_relevance_llm(_profile(), _ctx(), client)

    assert score == compute_category_relevance(_profile(), _ctx())


def test_compute_tweet_engagement_score_zero_without_tweets() -> None:
    assert compute_tweet_engagement_score([]) == 0.0


def test_compute_tweet_engagement_score_positive_with_tweets() -> None:
    tweets = [Tweet(id="1", text="hi", like_count=1000, retweet_count=200, reply_count=50)]

    assert compute_tweet_engagement_score(tweets) > 0.0


def test_rank_accounts_produces_0_to_100_scores_sorted_descending() -> None:
    profiles = [
        _profile(username="high_relevance", bio="sports championship athletes cricket football"),
        _profile(username="low_relevance", bio="cooking recipes", followers=10, tweets=10),
    ]

    ranked = rank_accounts(profiles, _ctx())

    assert len(ranked) == 2
    assert all(0.0 <= account.ranking_score <= 100.0 for account in ranked)
    assert ranked[0].ranking_score >= ranked[1].ranking_score
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_rank_accounts_uses_precomputed_relevance_scores() -> None:
    profiles = [_profile(username="a"), _profile(username="b")]

    ranked = rank_accounts(profiles, _ctx(), relevance_scores={"a": 100.0, "b": 0.0})

    by_username = {account.username: account for account in ranked}
    assert by_username["a"].category_relevance == 100.0
    assert by_username["b"].category_relevance == 0.0
    assert ranked[0].username == "a"


def test_select_top_n_truncates() -> None:
    profiles = [_profile(username=f"user{i}") for i in range(5)]
    ranked = rank_accounts(profiles, _ctx())

    top = select_top_n(ranked, 2)

    assert len(top) == 2


def test_select_top_n_returns_all_when_pool_smaller_than_n() -> None:
    profiles = [_profile(username="only_one")]
    ranked = rank_accounts(profiles, _ctx())

    top = select_top_n(ranked, 20)

    assert len(top) == 1


def test_rerank_with_tweet_engagement_updates_score_and_order() -> None:
    profiles = [
        _profile(username="a", followers=100, tweets=10),
        _profile(username="b", followers=100, tweets=10),
    ]
    ranked = rank_accounts(profiles, _ctx(), relevance_scores={"a": 50.0, "b": 50.0})

    tweets_by_username = {
        "a": [Tweet(id="1", text="x", like_count=1, retweet_count=0, reply_count=0)],
        "b": [
            Tweet(id="2", text="y", like_count=100_000, retweet_count=50_000, reply_count=10_000)
        ],
    }

    updated = rerank_with_tweet_engagement(ranked, tweets_by_username)

    by_username = {account.username: account for account in updated}
    assert by_username["b"].engagement_score > by_username["a"].engagement_score
    assert updated[0].username == "b"
    assert updated[0].rank == 1
