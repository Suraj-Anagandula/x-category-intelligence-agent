"""Unit tests for app.analysis: deterministic aggregations and LLM fallback."""

from __future__ import annotations

from app.analysis import (
    analyze_category,
    most_discussed_accounts,
    sentiment_fallback,
    top_engagement_tweets,
    trending_topics_fallback,
)
from app.exceptions import LLMError
from app.models import Tweet


def _tweet(tweet_id: str, username: str, text: str = "hello", **overrides) -> Tweet:
    defaults = dict(like_count=0, retweet_count=0, reply_count=0, view_count=0)
    defaults.update(overrides)
    return Tweet(id=tweet_id, username=username, text=text, **defaults)


def test_most_discussed_accounts_orders_by_frequency() -> None:
    tweets = [_tweet("1", "a"), _tweet("2", "a"), _tweet("3", "b")]

    result = most_discussed_accounts(tweets)

    assert result[0] == "a"
    assert "b" in result


def test_most_discussed_accounts_ignores_tweets_without_username() -> None:
    tweets = [_tweet("1", None), _tweet("2", "a")]

    result = most_discussed_accounts(tweets)

    assert result == ["a"]


def test_top_engagement_tweets_sorts_descending() -> None:
    low = _tweet("1", "a", like_count=1)
    high = _tweet("2", "a", like_count=1000, retweet_count=500)

    result = top_engagement_tweets([low, high], n=2)

    assert result[0].id == "2"
    assert result[1].id == "1"


def test_top_engagement_tweets_respects_n() -> None:
    tweets = [_tweet(str(i), "a", like_count=i) for i in range(10)]

    result = top_engagement_tweets(tweets, n=3)

    assert len(result) == 3


def test_trending_topics_fallback_prefers_hashtags() -> None:
    tweets = [
        _tweet("1", "a", text="Great match today #WorldCup #WorldCup"),
        _tweet("2", "a", text="Another #WorldCup update"),
    ]

    topics = trending_topics_fallback(tweets, n=3)

    assert topics[0] == "#worldcup"


def test_trending_topics_fallback_falls_back_to_words_without_hashtags() -> None:
    tweets = [_tweet("1", "a", text="championship championship match results")]

    topics = trending_topics_fallback(tweets, n=3)

    assert "championship" in topics


def test_sentiment_fallback_empty_tweets() -> None:
    sentiment = sentiment_fallback([])

    assert sentiment.positive == 0.0
    assert sentiment.neutral == 0.0
    assert sentiment.negative == 0.0


def test_sentiment_fallback_classifies_positive_and_negative() -> None:
    tweets = [
        _tweet("1", "a", text="What an amazing victory, great win!"),
        _tweet("2", "a", text="Terrible loss, awful performance."),
        _tweet("3", "a", text="The match starts at noon."),
    ]

    sentiment = sentiment_fallback(tweets)

    assert sentiment.positive > 0
    assert sentiment.negative > 0
    # Each component is independently rounded to 1dp, so the sum can be off
    # by a hair (e.g. 33.3 * 3 == 99.9) - allow that rounding slack.
    assert abs(sentiment.positive + sentiment.neutral + sentiment.negative - 100.0) < 0.5


class _StubLLMClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    async def generate_json(self, prompt: str):
        if self.error:
            raise self.error
        return self.result


async def test_analyze_category_without_llm_uses_fallback() -> None:
    tweets = [_tweet("1", "a", text="Great win today #WorldCup")]

    analysis = await analyze_category("sports", tweets, llm_client=None)

    assert analysis.most_discussed_accounts == ["a"]
    assert analysis.summary != ""


async def test_analyze_category_uses_llm_when_available() -> None:
    tweets = [_tweet("1", "a", text="Great win today")]
    client = _StubLLMClient(
        result={
            "trending_topics": ["championship"],
            "sentiment": {"positive": 80, "neutral": 15, "negative": 5},
            "summary": "Positive sentiment around a big win.",
        }
    )

    analysis = await analyze_category("sports", tweets, llm_client=client)

    assert analysis.trending_topics == ["championship"]
    assert analysis.sentiment.positive == 80
    assert analysis.summary == "Positive sentiment around a big win."


async def test_analyze_category_falls_back_on_llm_error() -> None:
    tweets = [_tweet("1", "a", text="Great win today #WorldCup")]
    client = _StubLLMClient(error=LLMError("boom"))

    analysis = await analyze_category("sports", tweets, llm_client=client)

    assert analysis.most_discussed_accounts == ["a"]
    assert analysis.trending_topics  # deterministic fallback still produced something


async def test_analyze_category_empty_tweets() -> None:
    analysis = await analyze_category("sports", [], llm_client=None)

    assert analysis.most_discussed_accounts == []
    assert analysis.high_engagement_tweets == []
