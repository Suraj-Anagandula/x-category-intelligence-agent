"""Unit tests for main.py's CLI glue: interactive tweet-lookup mode.

These stub out TwikitProfileClient entirely, so no network/credentials are
needed - only main.py's argument parsing, prompting, and error handling are
under test.
"""

from __future__ import annotations

import io

from rich.console import Console
from typer.testing import CliRunner

import main as main_module
from app.exceptions import AuthenticationError, UserNotFoundError
from app.models import Tweet
from app.schemas import (
    CategoryAnalysis,
    CategoryReport,
    RankedAccount,
    SentimentBreakdown,
    TweetStatistics,
)

runner = CliRunner()

#: Representative Unicode that a legacy-Windows cp1252 console can't encode:
#: the reported U+2011 non-breaking hyphen, plus an em dash, curly quotes, an
#: emoji, and CJK text, so the fix is verified against more than one character.
_UNICODE_SAMPLE = "non‑breaking hyphen, em—dash, curly ‘quotes’, " "\U0001f680 emoji, 中文 text"


def _cp1252_stream() -> io.TextIOWrapper:
    """A stream that mimics Python's default stdout encoding on a legacy
    (non-UTF-8) Windows console - writing our Unicode sample to it directly
    reproduces the reported UnicodeEncodeError."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")


class _FakeClient:
    """Stands in for TwikitProfileClient without any network access."""

    tweets: list[Tweet] = []
    connect_error: Exception | None = None
    fetch_error: Exception | None = None

    def __init__(self, **_kwargs) -> None:
        pass

    async def connect(self) -> None:
        if self.connect_error:
            raise self.connect_error

    async def get_recent_tweets(self, username: str, count: int = 10) -> list[Tweet]:
        if self.fetch_error:
            raise self.fetch_error
        return self.tweets


def _isolate_settings(monkeypatch, tmp_path) -> None:
    for attr in ("output_dir", "json_output_dir", "csv_output_dir", "log_dir", "cache_dir"):
        monkeypatch.setattr(main_module.settings, attr, tmp_path / attr)
    monkeypatch.setattr(main_module.settings, "x_session_file", tmp_path / "session.json")


def test_no_args_prompts_for_username_and_shows_tweets(monkeypatch, tmp_path) -> None:
    _isolate_settings(monkeypatch, tmp_path)
    _FakeClient.tweets = [Tweet(id="1", text="Hello world", like_count=5, retweet_count=1)]
    _FakeClient.connect_error = None
    _FakeClient.fetch_error = None
    monkeypatch.setattr(main_module, "TwikitProfileClient", _FakeClient)

    result = runner.invoke(main_module.app, [], input="elonmusk\n")

    assert result.exit_code == 0
    assert "elonmusk" in result.stdout
    assert "Hello world" in result.stdout
    assert "Saved to" in result.stdout

    csv_path = main_module.settings.csv_output_dir / "elonmusk_tweets.csv"
    assert csv_path.exists()
    assert "Hello world" in csv_path.read_text(encoding="utf-8")


def test_no_args_rejects_invalid_username(monkeypatch, tmp_path) -> None:
    _isolate_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(main_module, "TwikitProfileClient", _FakeClient)

    result = runner.invoke(main_module.app, [], input="has space\n")

    assert result.exit_code == 1
    assert "Invalid username" in result.stdout


def test_no_args_reports_authentication_failure_cleanly(monkeypatch, tmp_path) -> None:
    _isolate_settings(monkeypatch, tmp_path)
    _FakeClient.connect_error = AuthenticationError("no cookies configured")
    _FakeClient.fetch_error = None
    monkeypatch.setattr(main_module, "TwikitProfileClient", _FakeClient)

    result = runner.invoke(main_module.app, [], input="elonmusk\n")

    assert result.exit_code == 1
    assert "Authentication error" in result.stdout


def test_no_args_reports_user_not_found_cleanly(monkeypatch, tmp_path) -> None:
    _isolate_settings(monkeypatch, tmp_path)
    _FakeClient.connect_error = None
    _FakeClient.fetch_error = UserNotFoundError("ghost")
    monkeypatch.setattr(main_module, "TwikitProfileClient", _FakeClient)

    result = runner.invoke(main_module.app, [], input="ghost\n")

    assert result.exit_code == 1
    assert "not found" in result.stdout.lower()


def test_no_args_with_no_tweets_shows_empty_message(monkeypatch, tmp_path) -> None:
    _isolate_settings(monkeypatch, tmp_path)
    _FakeClient.tweets = []
    _FakeClient.connect_error = None
    _FakeClient.fetch_error = None
    monkeypatch.setattr(main_module, "TwikitProfileClient", _FakeClient)

    result = runner.invoke(main_module.app, [], input="elonmusk\n")

    assert result.exit_code == 0
    assert "no recent tweets" in result.stdout
    assert "Saved to" not in result.stdout


def test_unreconfigured_cp1252_stream_reproduces_the_reported_crash() -> None:
    """Pins down the actual bug mechanism (independent of host OS/terminal):
    writing the Unicode sample to a plain cp1252 stream raises
    UnicodeEncodeError, exactly as main.py's console.print() did before the fix."""
    stream = _cp1252_stream()

    try:
        stream.write(_UNICODE_SAMPLE)
        raise AssertionError("expected UnicodeEncodeError on an unreconfigured cp1252 stream")
    except UnicodeEncodeError:
        pass


def test_make_utf8_safe_prevents_unicode_encode_error() -> None:
    stream = _cp1252_stream()

    main_module._make_utf8_safe(stream)
    stream.write(_UNICODE_SAMPLE)  # must not raise
    stream.flush()


def test_make_utf8_safe_leaves_streams_without_reconfigure_untouched() -> None:
    """Streams that don't support reconfiguration (some test harnesses,
    unusual redirections) must be left alone rather than crashing setup."""

    class _NoReconfigure:
        pass

    stream = _NoReconfigure()

    result = main_module._make_utf8_safe(stream)

    assert result is stream


def _category_report_with_unicode_summary() -> CategoryReport:
    return CategoryReport(
        category="healthcare",
        accounts=[
            RankedAccount(
                rank=1,
                username="who",
                display_name="WHO",
                followers=1000,
                category_relevance=90.0,
                engagement_score=80.0,
                activity_score=70.0,
                audience_score=60.0,
                ranking_score=85.0,
            )
        ],
        tweet_statistics=TweetStatistics(
            accounts_processed=1, accounts_failed=0, tweets_collected=5
        ),
        analysis=CategoryAnalysis(
            trending_topics=[f"Topic with {_UNICODE_SAMPLE}"],
            sentiment=SentimentBreakdown(positive=50.0, neutral=30.0, negative=20.0),
            summary=f"This analysis summary contains {_UNICODE_SAMPLE}.",
        ),
        errors=[],
    )


def test_print_category_report_does_not_raise_on_unicode_summary(monkeypatch) -> None:
    """Regression test for the reported crash: a Groq-generated analysis
    summary containing U+2011 (and other representative Unicode) must not
    raise UnicodeEncodeError when printed, even on a legacy cp1252 console."""
    stream = _cp1252_stream()
    main_module._make_utf8_safe(stream)
    fake_console = Console(file=stream, force_terminal=False)
    monkeypatch.setattr(main_module, "console", fake_console)

    main_module._print_category_report(_category_report_with_unicode_summary(), requested_top_n=1)

    stream.flush()


def test_print_category_report_raises_without_the_fix(monkeypatch) -> None:
    """Confirms the previous test is actually exercising the fix, not just a
    console that happened to tolerate the input: without _make_utf8_safe,
    the same report on the same kind of stream reproduces the crash."""
    stream = _cp1252_stream()  # deliberately NOT passed through _make_utf8_safe
    fake_console = Console(file=stream, force_terminal=False)
    monkeypatch.setattr(main_module, "console", fake_console)

    try:
        main_module._print_category_report(
            _category_report_with_unicode_summary(), requested_top_n=1
        )
        raise AssertionError("expected UnicodeEncodeError without the fix")
    except UnicodeEncodeError:
        pass
