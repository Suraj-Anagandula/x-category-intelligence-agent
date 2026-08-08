"""Unit tests for app.models: UserProfile / ScrapeResult."""

from __future__ import annotations

from datetime import datetime, timezone

from app.models import ScrapeResult, UserProfile


def test_user_profile_minimal_defaults() -> None:
    profile = UserProfile(username="elonmusk")

    assert profile.username == "elonmusk"
    assert profile.followers is None
    assert profile.verified is False
    assert profile.protected is False
    assert isinstance(profile.scraped_at, datetime)


def test_user_profile_resolved_profile_url_falls_back() -> None:
    profile = UserProfile(username="openai")

    assert profile.resolved_profile_url == "https://x.com/openai"


def test_user_profile_resolved_profile_url_respects_explicit_value() -> None:
    profile = UserProfile(username="openai", profile_url="https://x.com/OpenAI")

    assert profile.resolved_profile_url == "https://x.com/OpenAI"


def test_user_profile_to_flat_dict_serializes_datetimes() -> None:
    created = datetime(2018, 10, 10, 20, 19, 24, tzinfo=timezone.utc)
    profile = UserProfile(username="elonmusk", created_at=created, followers=100)

    flat = profile.to_flat_dict()

    assert flat["username"] == "elonmusk"
    assert flat["followers"] == 100
    assert isinstance(flat["created_at"], str)
    assert flat["created_at"].startswith("2018-10-10")


def test_scrape_result_success() -> None:
    profile = UserProfile(username="satyanadella")
    result = ScrapeResult(username="satyanadella", success=True, profile=profile)

    assert result.success is True
    assert result.profile is profile
    assert result.error is None
    assert result.from_cache is False


def test_scrape_result_defaults_from_cache_false() -> None:
    result = ScrapeResult(
        username="elonmusk", success=True, profile=UserProfile(username="elonmusk")
    )

    assert result.from_cache is False


def test_scrape_result_failure() -> None:
    result = ScrapeResult(
        username="ghost",
        success=False,
        error="User not found: @ghost",
        error_type="UserNotFoundError",
    )

    assert result.success is False
    assert result.profile is None
    assert result.error_type == "UserNotFoundError"
