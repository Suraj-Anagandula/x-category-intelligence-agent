"""Unit tests for app.storage: category run persistence."""

from __future__ import annotations

import json

import pandas as pd

from app.config import Settings
from app.models import Tweet, UserProfile
from app.schemas import CategoryAnalysis, CategoryReport, RankedAccount, TweetStatistics
from app.storage import load_latest_category_run, save_category_run


def _settings(tmp_path) -> Settings:
    settings = Settings()
    settings.json_output_dir = tmp_path / "json"
    settings.csv_output_dir = tmp_path / "csv"
    settings.tweets_output_dir = tmp_path / "tweets"
    return settings


def _report() -> CategoryReport:
    return CategoryReport(
        category="sports",
        accounts=[
            RankedAccount(
                rank=1,
                username="espn",
                display_name="ESPN",
                followers=1000,
                category_relevance=90.0,
                engagement_score=50.0,
                activity_score=60.0,
                audience_score=70.0,
                ranking_score=65.0,
            )
        ],
        tweet_statistics=TweetStatistics(
            accounts_processed=1, accounts_failed=0, tweets_collected=1
        ),
        analysis=CategoryAnalysis(summary="A summary."),
        errors=[],
    )


async def test_save_category_run_writes_snapshot(tmp_path) -> None:
    settings = _settings(tmp_path)
    report = _report()
    profiles = [UserProfile(username="espn", followers=1000)]
    tweets = [Tweet(id="1", username="espn", text="hello")]

    path = await save_category_run("sports", report, profiles, tweets, settings)

    assert path.exists()
    assert path.parent == settings.tweets_output_dir / "sports"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["category"] == "sports"
    assert len(payload["accounts"]) == 1
    assert payload["accounts"][0]["username"] == "espn"
    assert len(payload["tweets"]) == 1


async def test_save_category_run_writes_profile_and_tweet_exports(tmp_path) -> None:
    settings = _settings(tmp_path)
    report = _report()
    profiles = [UserProfile(username="espn", followers=1000)]
    tweets = [Tweet(id="1", username="espn", text="hello")]

    await save_category_run("sports", report, profiles, tweets, settings)

    assert (settings.csv_output_dir / "users.csv").exists()
    assert (settings.csv_output_dir / "espn_tweets.csv").exists()
    assert any(settings.json_output_dir.glob("profiles_*.json"))


async def test_save_category_run_writes_consolidated_category_tweet_csv(tmp_path) -> None:
    settings = _settings(tmp_path)
    report = _report()
    profiles = [UserProfile(id="111", username="espn", display_name="ESPN", followers=1000)]
    tweets = [
        Tweet(id="1", username="espn", text="hello", like_count=5, retweet_count=1),
        Tweet(id="2", username="espn", text="world", like_count=10, retweet_count=2),
    ]

    await save_category_run("sports", report, profiles, tweets, settings)

    category_csv = settings.csv_output_dir / "sports_tweets.csv"
    assert category_csv.exists()

    df = pd.read_csv(category_csv, dtype={"tweet_id": str, "user_id": str}, encoding="utf-8-sig")
    assert len(df) == 2
    assert set(df["tweet_id"]) == {"1", "2"}
    assert (df["username"] == "espn").all()
    assert (df["user_id"] == "111").all()
    assert (df["category"] == "sports").all()
    assert (df["display_name"] == "ESPN").all()
    assert (df["followers_count"] == 1000).all()

    # Per-account and profile exports must still be produced alongside it.
    assert (settings.csv_output_dir / "users.csv").exists()
    assert (settings.csv_output_dir / "espn_tweets.csv").exists()


async def test_save_category_run_consolidated_csv_upserts_across_runs(tmp_path) -> None:
    settings = _settings(tmp_path)
    report = _report()
    profiles = [UserProfile(id="111", username="espn", followers=1000)]

    await save_category_run(
        "sports", report, profiles, [Tweet(id="1", username="espn", text="v1")], settings
    )
    await save_category_run(
        "sports",
        report,
        profiles,
        [Tweet(id="1", username="espn", text="v2"), Tweet(id="2", username="espn", text="new")],
        settings,
    )

    df = pd.read_csv(settings.csv_output_dir / "sports_tweets.csv", encoding="utf-8-sig")
    assert len(df) == 2  # tweet 1 updated in place, not duplicated
    assert df[df["tweet_id"] == 1]["tweet_text"].iloc[0] == "v2"


async def test_save_category_run_handles_no_profiles_or_tweets(tmp_path) -> None:
    settings = _settings(tmp_path)
    report = CategoryReport(category="sports")

    path = await save_category_run("sports", report, [], [], settings)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["accounts"] == []
    assert payload["tweets"] == []


async def test_load_latest_category_run_returns_none_when_missing(tmp_path) -> None:
    settings = _settings(tmp_path)

    result = await load_latest_category_run("sports", settings)

    assert result is None


async def test_load_latest_category_run_round_trips(tmp_path) -> None:
    settings = _settings(tmp_path)
    report = _report()
    await save_category_run("sports", report, [], [], settings)

    loaded = await load_latest_category_run("sports", settings)

    assert loaded is not None
    assert loaded["category"] == "sports"
