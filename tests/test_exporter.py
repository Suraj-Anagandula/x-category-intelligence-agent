"""Unit tests for app.exporter: JSONExporter and CSVExporter."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from app.exporter import (
    CategoryTweetCSVExporter,
    CSVExporter,
    ExporterRegistry,
    JSONExporter,
    TweetCSVExporter,
)
from app.models import Tweet, UserProfile


def _profile(username: str, followers: int = 10) -> UserProfile:
    return UserProfile(username=username, display_name=username.title(), followers=followers)


def _tweet(
    tweet_id: str, text: str = "hello", like_count: int = 1, username: str | None = None
) -> Tweet:
    return Tweet(id=tweet_id, text=text, like_count=like_count, username=username)


async def test_json_exporter_writes_expected_payload(tmp_path) -> None:
    exporter = JSONExporter(output_dir=tmp_path)
    profiles = [_profile("elonmusk"), _profile("openai", followers=999)]

    path = await exporter.export(profiles)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert {row["username"] for row in payload} == {"elonmusk", "openai"}


async def test_json_exporter_handles_empty_list(tmp_path) -> None:
    exporter = JSONExporter(output_dir=tmp_path)

    path = await exporter.export([])

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == []


async def test_csv_exporter_creates_file(tmp_path) -> None:
    exporter = CSVExporter(output_dir=tmp_path)

    path = await exporter.export([_profile("elonmusk")])

    assert path.exists()
    df = pd.read_csv(path)
    assert list(df["username"]) == ["elonmusk"]


async def test_csv_exporter_upserts_by_username(tmp_path) -> None:
    exporter = CSVExporter(output_dir=tmp_path)

    await exporter.export([_profile("elonmusk", followers=100)])
    await exporter.export([_profile("elonmusk", followers=200), _profile("openai")])

    df = pd.read_csv(exporter.path)
    assert len(df) == 2
    row = df[df["username"] == "elonmusk"].iloc[0]
    assert int(row["followers"]) == 200


async def test_csv_exporter_no_op_on_empty_list(tmp_path) -> None:
    exporter = CSVExporter(output_dir=tmp_path)

    path = await exporter.export([])

    assert not path.exists()


async def test_tweet_csv_exporter_creates_file_named_after_username(tmp_path) -> None:
    exporter = TweetCSVExporter(output_dir=tmp_path, username="elonmusk")

    path = await exporter.export([_tweet("1"), _tweet("2")])

    assert path == tmp_path / "elonmusk_tweets.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df["id"]) == [1, 2]


async def test_tweet_csv_exporter_upserts_by_id(tmp_path) -> None:
    exporter = TweetCSVExporter(output_dir=tmp_path, username="elonmusk")

    await exporter.export([_tweet("1", like_count=10)])
    await exporter.export([_tweet("1", like_count=20), _tweet("2")])

    df = pd.read_csv(exporter.path)
    assert len(df) == 2
    row = df[df["id"] == 1].iloc[0]
    assert int(row["like_count"]) == 20


async def test_tweet_csv_exporter_handles_emoji_without_crashing(tmp_path) -> None:
    # pandas.to_csv() defaults to the OS locale encoding, which on Windows
    # (cp1252) can't represent emoji and raises UnicodeEncodeError - this
    # exercises that every emoji-bearing tweet doesn't just crash the export.
    emoji_text = "It will be awesome \U0001f60e and more \U0001f680"
    exporter = TweetCSVExporter(output_dir=tmp_path, username="elonmusk")

    path = await exporter.export([_tweet("1", text=emoji_text)])

    raw = path.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf"  # UTF-8 BOM, so Excel also detects it correctly
    assert emoji_text in raw.decode("utf-8-sig")


async def test_tweet_csv_exporter_preserves_emoji_across_upsert(tmp_path) -> None:
    emoji_text = "It will be awesome \U0001f60e"
    exporter = TweetCSVExporter(output_dir=tmp_path, username="elonmusk")

    await exporter.export([_tweet("1", text=emoji_text)])
    await exporter.export([_tweet("2", text="no emoji here")])

    decoded = exporter.path.read_bytes().decode("utf-8-sig")
    assert emoji_text in decoded


async def test_csv_exporter_handles_emoji_without_crashing(tmp_path) -> None:
    exporter = CSVExporter(output_dir=tmp_path)
    profile = _profile("elonmusk")
    profile.bio = "Building rockets \U0001f680"

    path = await exporter.export([profile])

    raw = path.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf"
    assert "\U0001f680" in raw.decode("utf-8-sig")


async def test_tweet_csv_exporter_no_op_on_empty_list(tmp_path) -> None:
    exporter = TweetCSVExporter(output_dir=tmp_path, username="elonmusk")

    path = await exporter.export([])

    assert not path.exists()


async def test_category_tweet_csv_exporter_creates_file_named_after_category(tmp_path) -> None:
    exporter = CategoryTweetCSVExporter(output_dir=tmp_path, category="sports")

    path = await exporter.export([], {}, datetime.now(timezone.utc))

    assert path == tmp_path / "sports_tweets.csv"


@pytest.mark.parametrize(
    "category", ["sports", "politics", "technology", "business", "finance", "vintage cars"]
)
async def test_category_tweet_csv_exporter_filename_is_dynamic_per_category(
    tmp_path, category: str
) -> None:
    """The filename must be derived from the category argument, never
    hardcoded - this must hold for arbitrary/custom categories too."""
    exporter = CategoryTweetCSVExporter(output_dir=tmp_path, category=category)
    tweet = _tweet("1", username="someaccount")

    path = await exporter.export([tweet], {}, datetime.now(timezone.utc))

    assert path == tmp_path / f"{category}_tweets.csv"
    assert path.exists()
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert df.iloc[0]["category"] == category


async def test_category_tweet_csv_exporter_row_count_equals_accounts_times_tweets(
    tmp_path,
) -> None:
    """20 accounts x 10 tweets/account -> up to 200 rows (spec section 11)."""
    exporter = CategoryTweetCSVExporter(output_dir=tmp_path, category="technology")
    accounts = [f"account{i}" for i in range(20)]
    tweets = [_tweet(f"{account}-{i}", username=account) for account in accounts for i in range(10)]

    path = await exporter.export(tweets, {}, datetime.now(timezone.utc))

    df = pd.read_csv(path, encoding="utf-8-sig")
    assert len(df) == 200
    assert df["username"].nunique() == 20


async def test_category_tweet_csv_exporter_one_row_per_tweet_with_account_context(tmp_path) -> None:
    exporter = CategoryTweetCSVExporter(output_dir=tmp_path, category="sports")
    tweets = [
        _tweet("1", text="Great match", like_count=100, username="espn"),
        _tweet("2", text="Final result", like_count=50, username="fifacom"),
    ]
    profiles = {
        "espn": UserProfile(id="111", username="espn", display_name="ESPN", followers=1000),
        "fifacom": UserProfile(id="222", username="fifacom", display_name="FIFA", followers=500),
    }
    collected_at = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)

    path = await exporter.export(tweets, profiles, collected_at)

    df = pd.read_csv(path, dtype={"tweet_id": str, "user_id": str}, encoding="utf-8-sig")
    assert list(df.columns) == CategoryTweetCSVExporter.COLUMNS
    assert len(df) == 2

    espn_row = df[df["username"] == "espn"].iloc[0]
    assert espn_row["user_id"] == "111"
    assert espn_row["tweet_id"] == "1"
    assert espn_row["tweet_text"] == "Great match"
    assert espn_row["category"] == "sports"
    assert espn_row["display_name"] == "ESPN"
    assert int(espn_row["followers_count"]) == 1000
    assert int(espn_row["likes"]) == 100
    assert espn_row["url"] == "https://x.com/i/status/1"
    assert espn_row["collected_at"] == collected_at.isoformat()


async def test_category_tweet_csv_exporter_handles_tweet_without_matching_profile(tmp_path) -> None:
    exporter = CategoryTweetCSVExporter(output_dir=tmp_path, category="sports")
    tweets = [_tweet("1", username="unknown_account")]

    path = await exporter.export(tweets, {}, datetime.now(timezone.utc))

    df = pd.read_csv(path, encoding="utf-8-sig")
    assert df.iloc[0]["username"] == "unknown_account"
    assert pd.isna(df.iloc[0]["user_id"])
    assert pd.isna(df.iloc[0]["display_name"])


async def test_category_tweet_csv_exporter_upserts_by_tweet_id(tmp_path) -> None:
    exporter = CategoryTweetCSVExporter(output_dir=tmp_path, category="sports")
    profiles = {"espn": UserProfile(id="111", username="espn", followers=1000)}

    await exporter.export(
        [_tweet("1", text="first version", like_count=1, username="espn")],
        profiles,
        datetime.now(timezone.utc),
    )
    await exporter.export(
        [
            _tweet("1", text="updated version", like_count=99, username="espn"),
            _tweet("2", text="new tweet", like_count=5, username="espn"),
        ],
        profiles,
        datetime.now(timezone.utc),
    )

    df = pd.read_csv(exporter.path, dtype={"tweet_id": str}, encoding="utf-8-sig")
    assert len(df) == 2
    row = df[df["tweet_id"] == "1"].iloc[0]
    assert row["tweet_text"] == "updated version"
    assert int(row["likes"]) == 99


async def test_category_tweet_csv_exporter_no_op_on_empty_tweets(tmp_path) -> None:
    exporter = CategoryTweetCSVExporter(output_dir=tmp_path, category="sports")

    path = await exporter.export([], {}, datetime.now(timezone.utc))

    assert not path.exists()


async def test_category_tweet_csv_exporter_preserves_existing_per_account_export(tmp_path) -> None:
    # The consolidated category CSV must not interfere with the existing
    # per-account TweetCSVExporter output living in the same directory.
    tweet = _tweet("1", text="hello", username="espn")

    per_account_path = await TweetCSVExporter(output_dir=tmp_path, username="espn").export([tweet])
    category_path = await CategoryTweetCSVExporter(output_dir=tmp_path, category="sports").export(
        [tweet], {}, datetime.now(timezone.utc)
    )

    assert per_account_path.exists()
    assert category_path.exists()
    assert per_account_path != category_path


async def test_exporter_registry_creates_correct_backend(tmp_path) -> None:
    registry = ExporterRegistry()
    registry.register("json", JSONExporter)
    registry.register("csv", CSVExporter)

    json_exporter = registry.create("json", output_dir=tmp_path)
    csv_exporter = registry.create("csv", output_dir=tmp_path)

    assert isinstance(json_exporter, JSONExporter)
    assert isinstance(csv_exporter, CSVExporter)


def test_exporter_registry_unknown_format_raises(tmp_path) -> None:
    registry = ExporterRegistry()

    try:
        registry.create("xml", output_dir=tmp_path)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
