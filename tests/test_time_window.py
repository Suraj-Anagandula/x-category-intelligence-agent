"""Unit tests for app.time_window - pure functions, no I/O, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.models import Tweet
from app.time_window import (
    TimeWindow,
    filter_tweets_to_window,
    oldest_created_at,
    resolve_time_window,
    time_window_from_dict,
    tweet_in_window,
)


def _tweet(created_at: datetime | None, tweet_id: str = "1") -> Tweet:
    return Tweet(id=tweet_id, text="hello", created_at=created_at)


# --- resolve_time_window -------------------------------------------------


def test_resolve_time_window_latest_has_no_bounds() -> None:
    window = resolve_time_window("latest")

    assert window.mode == "latest"
    assert window.start is None
    assert window.end is None
    assert window.is_filtered is False


def test_resolve_time_window_24h_calculates_correct_window() -> None:
    now = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)

    window = resolve_time_window("24h", now=now)

    assert window.mode == "24h"
    assert window.end == now
    assert window.start == now - timedelta(hours=24)
    assert window.is_filtered is True


def test_resolve_time_window_7d_calculates_correct_window() -> None:
    now = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)

    window = resolve_time_window("7d", now=now)

    assert window.end == now
    assert window.start == now - timedelta(days=7)


def test_resolve_time_window_30d_calculates_correct_window() -> None:
    now = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)

    window = resolve_time_window("30d", now=now)

    assert window.end == now
    assert window.start == now - timedelta(days=30)


def test_resolve_time_window_custom_uses_given_bounds() -> None:
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, 8, tzinfo=timezone.utc)

    window = resolve_time_window("custom", custom_start=start, custom_end=end)

    assert window.mode == "custom"
    assert window.start == start
    assert window.end == end
    assert window.is_filtered is True


def test_resolve_time_window_custom_requires_both_bounds() -> None:
    with pytest.raises(ValueError):
        resolve_time_window("custom", custom_start=datetime.now(timezone.utc))
    with pytest.raises(ValueError):
        resolve_time_window("custom", custom_end=datetime.now(timezone.utc))


def test_resolve_time_window_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        resolve_time_window("bogus")


def test_resolve_time_window_custom_naive_datetimes_treated_as_utc() -> None:
    start = datetime(2026, 8, 1)  # naive
    end = datetime(2026, 8, 8)  # naive

    window = resolve_time_window("custom", custom_start=start, custom_end=end)

    assert window.start.tzinfo is not None
    assert window.end.tzinfo is not None


# --- tweet_in_window: the exact boundary semantics the spec calls out ----


def test_tweet_exactly_at_start_is_included() -> None:
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    tweet = _tweet(datetime(2026, 8, 1, tzinfo=timezone.utc))  # == START

    assert tweet_in_window(tweet, window) is True


def test_tweet_exactly_at_end_is_excluded() -> None:
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    tweet = _tweet(datetime(2026, 8, 8, tzinfo=timezone.utc))  # == END

    assert tweet_in_window(tweet, window) is False


def test_tweet_before_start_is_excluded() -> None:
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    tweet = _tweet(datetime(2026, 7, 31, 23, 59, tzinfo=timezone.utc))

    assert tweet_in_window(tweet, window) is False


def test_tweet_after_end_is_excluded() -> None:
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    tweet = _tweet(datetime(2026, 8, 8, 0, 1, tzinfo=timezone.utc))

    assert tweet_in_window(tweet, window) is False


def test_tweet_in_window_latest_mode_always_true_even_without_timestamp() -> None:
    window = resolve_time_window("latest")

    assert tweet_in_window(_tweet(None), window) is True
    assert tweet_in_window(_tweet(datetime(1999, 1, 1, tzinfo=timezone.utc)), window) is True


def test_tweet_in_window_missing_created_at_excluded_when_filtered() -> None:
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )

    assert tweet_in_window(_tweet(None), window) is False


def test_tweet_in_window_naive_created_at_treated_as_utc() -> None:
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    tweet = _tweet(datetime(2026, 8, 4))  # naive - must be treated as UTC, not raise

    assert tweet_in_window(tweet, window) is True


def test_tweet_in_window_non_utc_timezone_converted_correctly() -> None:
    """A tweet timestamped in a non-UTC offset must be converted to UTC
    before comparison, not compared "as-is" against a UTC window."""
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc),
    )
    ist = timezone(timedelta(hours=5, minutes=30))
    # 2026-08-01 05:00 IST == 2026-07-31 23:30 UTC - just before the window.
    tweet = _tweet(datetime(2026, 8, 1, 5, 0, tzinfo=ist))

    assert tweet_in_window(tweet, window) is False

    # 2026-08-01 09:00 IST == 2026-08-01 03:30 UTC - inside the window.
    tweet_inside = _tweet(datetime(2026, 8, 1, 9, 0, tzinfo=ist))
    assert tweet_in_window(tweet_inside, window) is True


# --- filter_tweets_to_window ----------------------------------------------


def test_filter_tweets_to_window_keeps_only_in_range_tweets() -> None:
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    tweets = [
        _tweet(datetime(2026, 7, 30, tzinfo=timezone.utc), "old"),
        _tweet(datetime(2026, 8, 3, tzinfo=timezone.utc), "in_range"),
        _tweet(datetime(2026, 8, 9, tzinfo=timezone.utc), "future"),
        _tweet(None, "no_timestamp"),
    ]

    filtered = filter_tweets_to_window(tweets, window)

    assert [t.id for t in filtered] == ["in_range"]


def test_filter_tweets_to_window_latest_mode_returns_all_unchanged() -> None:
    window = resolve_time_window("latest")
    tweets = [_tweet(None, "1"), _tweet(datetime(2020, 1, 1, tzinfo=timezone.utc), "2")]

    filtered = filter_tweets_to_window(tweets, window)

    assert filtered == tweets
    assert filtered is not tweets  # never mutates/returns the same list object


def test_filter_tweets_to_window_does_not_mutate_input() -> None:
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    tweets = [_tweet(datetime(2020, 1, 1, tzinfo=timezone.utc), "old")]
    original_len = len(tweets)

    filter_tweets_to_window(tweets, window)

    assert len(tweets) == original_len


def test_filter_tweets_to_window_empty_window_returns_empty_list() -> None:
    window = resolve_time_window(
        "custom",
        custom_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        custom_end=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    tweets = [_tweet(datetime(2020, 1, 1, tzinfo=timezone.utc), "old")]

    filtered = filter_tweets_to_window(tweets, window)

    assert filtered == []


# --- oldest_created_at -----------------------------------------------------


def test_oldest_created_at_returns_the_minimum() -> None:
    tweets = [
        _tweet(datetime(2026, 8, 5, tzinfo=timezone.utc), "a"),
        _tweet(datetime(2026, 8, 1, tzinfo=timezone.utc), "b"),
        _tweet(datetime(2026, 8, 9, tzinfo=timezone.utc), "c"),
    ]

    assert oldest_created_at(tweets) == datetime(2026, 8, 1, tzinfo=timezone.utc)


def test_oldest_created_at_ignores_missing_timestamps() -> None:
    tweets = [_tweet(None, "a"), _tweet(datetime(2026, 8, 1, tzinfo=timezone.utc), "b")]

    assert oldest_created_at(tweets) == datetime(2026, 8, 1, tzinfo=timezone.utc)


def test_oldest_created_at_empty_or_all_missing_returns_none() -> None:
    assert oldest_created_at([]) is None
    assert oldest_created_at([_tweet(None, "a")]) is None


# --- time_window_from_dict (backward compatibility) ------------------------


def test_time_window_from_dict_reconstructs_a_real_window() -> None:
    data = {
        "mode": "custom",
        "start": "2026-08-01T00:00:00+00:00",
        "end": "2026-08-08T00:00:00+00:00",
    }

    window = time_window_from_dict(data)

    assert window is not None
    assert window.mode == "custom"
    assert window.start == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert window.end == datetime(2026, 8, 8, tzinfo=timezone.utc)


def test_time_window_from_dict_none_or_empty_returns_none() -> None:
    assert time_window_from_dict(None) is None
    assert time_window_from_dict({}) is None


def test_time_window_from_dict_handles_old_snapshot_with_no_time_window_key() -> None:
    """A snapshot saved before this feature existed has no "time_window"
    key at all - `data.get("time_window")` is None, must not raise."""
    old_snapshot = {"category": "sports", "scraped_at": "2026-08-08T00:00:00+00:00"}

    assert time_window_from_dict(old_snapshot.get("time_window")) is None


def test_time_window_from_dict_unknown_mode_returns_none() -> None:
    assert time_window_from_dict({"mode": "bogus", "start": None, "end": None}) is None


def test_time_window_from_dict_accepts_pydantic_z_suffixed_timestamps() -> None:
    """Regression test: `report.time_window.model_dump(mode="json")`
    (Pydantic's JSON mode) serializes UTC datetimes with a trailing "Z"
    (e.g. "2026-08-03T16:59:26.075703Z"), which Python 3.10's
    `datetime.fromisoformat()` cannot parse directly - this is exactly the
    stored format `app/storage.py` writes for every real snapshot's
    "time_window" block, so it must round-trip correctly, not raise."""
    data = {
        "mode": "custom",
        "start": "2026-08-01T00:00:00.000000Z",
        "end": "2026-08-08T00:00:00.000000Z",
    }

    window = time_window_from_dict(data)

    assert window is not None
    assert window.start == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert window.end == datetime(2026, 8, 8, tzinfo=timezone.utc)


def test_time_window_from_dict_unparseable_dates_returns_none() -> None:
    data = {"mode": "custom", "start": "not-a-date", "end": "also-not-a-date"}

    assert time_window_from_dict(data) is None


def test_time_window_label_property() -> None:
    assert TimeWindow(mode="latest").label == "Latest Available"
    assert TimeWindow(mode="7d").label == "Last 7 Days"
