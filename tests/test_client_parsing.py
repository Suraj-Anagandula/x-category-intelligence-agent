"""Unit tests for the Twikit User/Tweet -> app model parsing logic in app.client."""

from __future__ import annotations

import time
from types import SimpleNamespace

from app.client import TwikitProfileClient, _map_twikit_exception, _parse_created_at


def test_parse_created_at_legacy_twitter_format() -> None:
    result = _parse_created_at("Wed Oct 10 20:19:24 +0000 2018")

    assert result.year == 2018
    assert result.month == 10
    assert result.day == 10


def test_parse_created_at_passthrough_on_unrecognized_format() -> None:
    assert _parse_created_at("not-a-date") == "not-a-date"


def test_parse_created_at_passthrough_none() -> None:
    assert _parse_created_at(None) is None


def _fake_user(**overrides) -> SimpleNamespace:
    defaults = dict(
        id=12345,
        screen_name="elonmusk",
        name="Elon Musk",
        description="Bio text",
        location="Mars",
        url="https://t.co/short",
        entities={"url": {"urls": [{"expanded_url": "https://tesla.com"}]}},
        profile_image_url_https="https://example.com/avatar.jpg",
        profile_banner_url="https://example.com/banner.jpg",
        protected=False,
        verified=True,
        followers_count=200_000_000,
        friends_count=900,
        statuses_count=50_000,
        favourites_count=10_000,
        media_count=5_000,
        created_at="Wed Oct 10 20:19:24 +0000 2018",
        pinned_tweet_ids=[999],
        lang="en",
        is_blue_verified=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_to_profile_maps_all_fields() -> None:
    user = _fake_user()

    profile = TwikitProfileClient._to_profile("elonmusk", user)

    assert profile.id == "12345"
    assert profile.username == "elonmusk"
    assert profile.display_name == "Elon Musk"
    assert profile.bio == "Bio text"
    assert profile.location == "Mars"
    assert profile.website == "https://tesla.com"
    assert profile.profile_image == "https://example.com/avatar.jpg"
    assert profile.banner_image == "https://example.com/banner.jpg"
    assert profile.followers == 200_000_000
    assert profile.following == 900
    assert profile.tweets == 50_000
    assert profile.likes == 10_000
    assert profile.media_count == 5_000
    assert profile.pinned_tweet_id == "999"
    assert profile.language == "en"
    assert profile.is_blue_verified is True
    assert profile.profile_url == "https://x.com/elonmusk"
    assert profile.created_at.year == 2018


def test_to_profile_falls_back_to_plain_url_when_no_entities() -> None:
    user = _fake_user(entities=None, url="https://plain.example.com")

    profile = TwikitProfileClient._to_profile("elonmusk", user)

    assert profile.website == "https://plain.example.com"


def test_to_profile_handles_missing_optional_fields() -> None:
    user = SimpleNamespace(screen_name="minimal")

    profile = TwikitProfileClient._to_profile("minimal", user)

    assert profile.username == "minimal"
    assert profile.followers is None
    assert profile.verified is False
    assert profile.is_blue_verified is False


def _fake_tweet(**overrides) -> SimpleNamespace:
    defaults = dict(
        id=123456789,
        full_text="Just a tweet",
        text="Just a tweet",
        created_at="Wed Oct 10 20:19:24 +0000 2018",
        reply_count=5,
        retweet_count=10,
        favorite_count=100,
        quote_count=2,
        view_count="1234",
        retweeted_tweet=None,
        in_reply_to=None,
        lang="en",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_to_tweet_maps_all_fields() -> None:
    raw = _fake_tweet()

    tweet = TwikitProfileClient._to_tweet(raw)

    assert tweet.id == "123456789"
    assert tweet.text == "Just a tweet"
    assert tweet.created_at.year == 2018
    assert tweet.reply_count == 5
    assert tweet.retweet_count == 10
    assert tweet.like_count == 100
    assert tweet.quote_count == 2
    assert tweet.view_count == 1234
    assert tweet.is_retweet is False
    assert tweet.is_reply is False
    assert tweet.lang == "en"
    assert tweet.url == "https://x.com/i/status/123456789"


def test_to_tweet_detects_retweet_and_reply() -> None:
    raw = _fake_tweet(retweeted_tweet=SimpleNamespace(id=1), in_reply_to="987")

    tweet = TwikitProfileClient._to_tweet(raw)

    assert tweet.is_retweet is True
    assert tweet.is_reply is True


def test_to_tweet_handles_unparseable_view_count() -> None:
    raw = _fake_tweet(view_count="N/A")

    tweet = TwikitProfileClient._to_tweet(raw)

    assert tweet.view_count is None


def test_to_tweet_falls_back_to_text_without_full_text() -> None:
    raw = _fake_tweet(full_text=None, text="Short form")

    tweet = TwikitProfileClient._to_tweet(raw)

    assert tweet.text == "Short form"


class _StubRawClient:
    """Stands in for the underlying twikit.Client - X often returns more
    tweets than the requested page size, which get_recent_tweets must trim.
    """

    def __init__(self, tweets: list) -> None:
        self._tweets = tweets

    async def get_user_by_screen_name(self, username: str):
        return SimpleNamespace(id="1", screen_name=username)

    async def get_user_tweets(self, user_id: str, tweet_type: str, count: int):
        return self._tweets


async def test_get_recent_tweets_trims_to_requested_count() -> None:
    client = TwikitProfileClient(session_file=None)
    client._client = _StubRawClient([_fake_tweet(id=i) for i in range(20)])

    tweets = await client.get_recent_tweets("elonmusk", count=5)

    assert len(tweets) == 5


class TooManyRequests(Exception):
    """Stands in for twifork's `errors.TooManyRequests`, which exposes
    `rate_limit_reset` (an absolute Unix timestamp from the
    `x-rate-limit-reset` response header) when that header was present.
    `_map_twikit_exception` dispatches purely on class *name* (so it never
    imports twikit's real error types directly), so the name must match
    exactly."""

    def __init__(self, rate_limit_reset: float | None) -> None:
        super().__init__("Too Many Requests")
        self.rate_limit_reset = rate_limit_reset


def test_map_twikit_exception_extracts_retry_after_from_rate_limit_reset() -> None:
    reset_at = time.time() + 120
    exc = TooManyRequests(rate_limit_reset=reset_at)

    mapped = _map_twikit_exception(exc, "elonmusk")

    assert mapped.__class__.__name__ == "RateLimitError"
    assert mapped.retry_after is not None
    # Allow slack for test execution time between computing reset_at and the call.
    assert 115 <= mapped.retry_after <= 120


def test_map_twikit_exception_handles_missing_rate_limit_reset() -> None:
    exc = TooManyRequests(rate_limit_reset=None)

    mapped = _map_twikit_exception(exc, "elonmusk")

    assert mapped.__class__.__name__ == "RateLimitError"
    assert mapped.retry_after is None


def test_map_twikit_exception_clamps_past_reset_time_to_zero() -> None:
    """A reset timestamp already in the past (clock skew, slow request) must
    not produce a negative retry_after."""
    exc = TooManyRequests(rate_limit_reset=time.time() - 60)

    mapped = _map_twikit_exception(exc, "elonmusk")

    assert mapped.retry_after == 0.0
