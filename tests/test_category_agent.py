"""Unit tests for app.category_agent: normalization, context building, and pipeline wiring.

Every I/O dependency (discovery, profile scraper, tweet scraper, LLM client)
is stubbed or monkeypatched - no real X/LLM credentials or network needed.
"""

from __future__ import annotations

import pytest

from app.category_agent import (
    PREDEFINED_CATEGORIES,
    CategoryAgent,
    CategoryIntelligenceAgent,
    normalize_category,
)
from app.config import Settings
from app.exceptions import LLMError
from app.models import ScrapeResult, Tweet, TweetScrapeResult, UserProfile
from app.schemas import DiscoveredAccount


def _discovered(*usernames: str) -> list[DiscoveredAccount]:
    return [DiscoveredAccount(username=name, reason="test reason") for name in usernames]


def test_normalize_category_cases() -> None:
    assert normalize_category("Sports") == "sports"
    assert normalize_category("SPORTS") == "sports"
    assert normalize_category("  sports  ") == "sports"


class _StubLLMClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    async def generate_json(self, prompt: str):
        if self.error:
            raise self.error
        return self.result


async def test_build_context_predefined_category_without_llm() -> None:
    agent = CategoryAgent(llm_client=None)

    ctx = await agent.build_context("Sports")

    assert ctx.category == "sports"
    assert ctx == PREDEFINED_CATEGORIES["sports"]


async def test_build_context_custom_category_without_llm_uses_keywords() -> None:
    agent = CategoryAgent(llm_client=None)

    ctx = await agent.build_context("vintage cars")

    assert ctx.category == "vintage cars"
    assert "vintage" in ctx.keywords
    assert "cars" in ctx.keywords


async def test_build_context_uses_llm_when_available() -> None:
    client = _StubLLMClient(
        result={"subcategories": ["cricket"], "keywords": ["sports", "cricket"]}
    )
    agent = CategoryAgent(llm_client=client)

    ctx = await agent.build_context("sports")

    assert ctx.subcategories == ["cricket"]
    assert ctx.keywords == ["sports", "cricket"]


async def test_build_context_falls_back_when_llm_fails() -> None:
    client = _StubLLMClient(error=LLMError("boom"))
    agent = CategoryAgent(llm_client=client)

    ctx = await agent.build_context("sports")

    assert ctx == PREDEFINED_CATEGORIES["sports"]


class _StubProfileScraper:
    def __init__(self, results: list[ScrapeResult]) -> None:
        self.results = results
        self.calls: list[list[str]] = []

    async def scrape_many(self, usernames):
        self.calls.append(list(usernames))
        return self.results


class _StubTweetScraper:
    def __init__(self, results: list[TweetScrapeResult]) -> None:
        self.results = results
        self.calls: list[tuple[list[str], int | None]] = []

    async def scrape_many(self, usernames, count=None):
        self.calls.append((list(usernames), count))
        return self.results


def _profile(username: str, followers: int = 1000) -> UserProfile:
    return UserProfile(
        username=username,
        display_name=username.title(),
        bio="sports news",
        followers=followers,
        tweets=1000,
    )


def _settings(tmp_path) -> Settings:
    settings = Settings()
    settings.json_output_dir = tmp_path / "json"
    settings.csv_output_dir = tmp_path / "csv"
    settings.tweets_output_dir = tmp_path / "tweets"
    return settings


async def test_run_pipeline_wires_discovery_ranking_and_tweets(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)

    async def fake_discover(category, keywords, limit, llm_client=None):
        return _discovered("espn", "ghost", "fifacom")

    monkeypatch.setattr("app.category_agent.discover_candidates", fake_discover)

    profile_results = [
        ScrapeResult(username="espn", success=True, profile=_profile("espn")),
        ScrapeResult(
            username="ghost", success=False, error="not found", error_type="UserNotFoundError"
        ),
        ScrapeResult(username="fifacom", success=True, profile=_profile("fifacom")),
    ]
    tweet_results = [
        TweetScrapeResult(
            username="espn",
            success=True,
            tweets=[Tweet(id="1", username="espn", text="great match", like_count=10)],
        ),
        TweetScrapeResult(
            username="fifacom", success=False, error="protected", error_type="ProtectedAccountError"
        ),
    ]

    agent = CategoryIntelligenceAgent(
        settings=settings,
        profile_scraper=_StubProfileScraper(profile_results),
        tweet_scraper=_StubTweetScraper(tweet_results),
        llm_client=None,
    )

    report = await agent.run_pipeline("sports", candidate_limit=10, top_n=2, tweets_per_account=5)

    assert report.category == "sports"
    assert {account.username for account in report.accounts} == {"espn", "fifacom"}
    assert report.tweet_statistics.tweets_collected == 1
    assert report.tweet_statistics.accounts_failed == 1
    assert any(e["username"] == "ghost" and e["stage"] == "validation" for e in report.errors)


async def test_run_pipeline_reports_rate_limited_accounts_separately_from_other_failures(
    monkeypatch, tmp_path
) -> None:
    """Spec: the final report must distinguish rate-limited tweet-scrape
    failures from other failures, and must never overstate tweets collected."""
    settings = _settings(tmp_path)
    usernames = [f"account{i}" for i in range(4)]

    async def fake_discover(category, keywords, limit, llm_client=None):
        return _discovered(*usernames)

    monkeypatch.setattr("app.category_agent.discover_candidates", fake_discover)

    profile_results = [
        ScrapeResult(username=name, success=True, profile=_profile(name)) for name in usernames
    ]
    tweet_results = [
        TweetScrapeResult(
            username="account0",
            success=True,
            tweets=[Tweet(id="1", username="account0", text="hi")],
        ),
        TweetScrapeResult(
            username="account1", success=False, error="slow down", error_type="RateLimitError"
        ),
        TweetScrapeResult(
            username="account2", success=False, error="slow down", error_type="RateLimitError"
        ),
        TweetScrapeResult(
            username="account3", success=False, error="gone", error_type="UserNotFoundError"
        ),
    ]

    agent = CategoryIntelligenceAgent(
        settings=settings,
        profile_scraper=_StubProfileScraper(profile_results),
        tweet_scraper=_StubTweetScraper(tweet_results),
        llm_client=None,
    )

    report = await agent.run_pipeline("sports", top_n=4)

    stats = report.tweet_statistics
    assert stats.accounts_processed == 1
    assert stats.accounts_failed == 3
    assert stats.accounts_rate_limited == 2
    assert stats.accounts_failed_other == 1
    assert stats.tweets_collected == 1  # never overstated as if all 4 accounts succeeded
    assert any(e["username"] == "account1" and e["stage"] == "tweets" for e in report.errors)
    assert any(e["username"] == "account3" and e["stage"] == "tweets" for e in report.errors)


async def test_run_pipeline_uses_settings_defaults_when_not_overridden(
    monkeypatch, tmp_path
) -> None:
    settings = _settings(tmp_path)
    settings.category_candidate_limit = 7
    settings.top_accounts_limit = 1
    settings.tweets_per_account = 3

    captured = {}

    async def fake_discover(category, keywords, limit, llm_client=None):
        captured["limit"] = limit
        return _discovered("espn")

    monkeypatch.setattr("app.category_agent.discover_candidates", fake_discover)

    profile_results = [ScrapeResult(username="espn", success=True, profile=_profile("espn"))]
    tweet_results = [TweetScrapeResult(username="espn", success=True, tweets=[])]

    agent = CategoryIntelligenceAgent(
        settings=settings,
        profile_scraper=_StubProfileScraper(profile_results),
        tweet_scraper=_StubTweetScraper(tweet_results),
        llm_client=None,
    )

    report = await agent.run_pipeline("sports")

    assert captured["limit"] == 7
    assert len(report.accounts) == 1
    assert agent.tweet_scraper.calls[0][1] == 3  # settings.tweets_per_account propagated


async def test_run_pipeline_selects_fewer_than_top_n_when_insufficient_valid_accounts(
    monkeypatch, tmp_path
) -> None:
    """Spec section 10: if only 12/20 candidates validate, report 12 - never
    fabricate accounts to reach the requested top_n."""
    settings = _settings(tmp_path)
    usernames = [f"account{i}" for i in range(12)]

    async def fake_discover(category, keywords, limit, llm_client=None):
        return _discovered(*usernames)

    monkeypatch.setattr("app.category_agent.discover_candidates", fake_discover)

    profile_results = [
        ScrapeResult(username=name, success=True, profile=_profile(name)) for name in usernames
    ]
    tweet_results = [
        TweetScrapeResult(username=name, success=True, tweets=[]) for name in usernames
    ]

    agent = CategoryIntelligenceAgent(
        settings=settings,
        profile_scraper=_StubProfileScraper(profile_results),
        tweet_scraper=_StubTweetScraper(tweet_results),
        llm_client=None,
    )

    report = await agent.run_pipeline("sports", top_n=20)

    assert len(report.accounts) == 12


@pytest.mark.parametrize("category", ["sports", "politics", "technology"])
async def test_run_pipeline_is_generic_across_categories(
    monkeypatch, tmp_path, category: str
) -> None:
    """The same `run_pipeline` code path must work for any category - no
    category-specific branching in the pipeline itself."""
    settings = _settings(tmp_path)
    captured = {}

    async def fake_discover(cat, keywords, limit, llm_client=None):
        captured["category"] = cat
        captured["keywords"] = keywords
        return _discovered("account1", "account2")

    monkeypatch.setattr("app.category_agent.discover_candidates", fake_discover)

    profile_results = [
        ScrapeResult(username="account1", success=True, profile=_profile("account1")),
        ScrapeResult(username="account2", success=True, profile=_profile("account2")),
    ]
    tweet_results = [
        TweetScrapeResult(username="account1", success=True, tweets=[]),
        TweetScrapeResult(username="account2", success=True, tweets=[]),
    ]

    agent = CategoryIntelligenceAgent(
        settings=settings,
        profile_scraper=_StubProfileScraper(profile_results),
        tweet_scraper=_StubTweetScraper(tweet_results),
        llm_client=None,
    )

    report = await agent.run_pipeline(category, top_n=20)

    assert report.category == category
    assert captured["category"] == category
    assert len(report.accounts) == 2


async def test_run_pipeline_requests_configured_tweets_per_account(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)

    async def fake_discover(category, keywords, limit, llm_client=None):
        return _discovered("espn")

    monkeypatch.setattr("app.category_agent.discover_candidates", fake_discover)

    profile_results = [ScrapeResult(username="espn", success=True, profile=_profile("espn"))]
    tweet_scraper = _StubTweetScraper([TweetScrapeResult(username="espn", success=True, tweets=[])])

    agent = CategoryIntelligenceAgent(
        settings=settings,
        profile_scraper=_StubProfileScraper(profile_results),
        tweet_scraper=tweet_scraper,
        llm_client=None,
    )

    await agent.run_pipeline("sports", tweets_per_account=10)

    assert tweet_scraper.calls[0][1] == 10


async def test_run_pipeline_handles_zero_valid_candidates(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)

    async def fake_discover(category, keywords, limit, llm_client=None):
        return _discovered("ghost")

    monkeypatch.setattr("app.category_agent.discover_candidates", fake_discover)

    profile_results = [
        ScrapeResult(
            username="ghost", success=False, error="not found", error_type="UserNotFoundError"
        )
    ]

    agent = CategoryIntelligenceAgent(
        settings=settings,
        profile_scraper=_StubProfileScraper(profile_results),
        tweet_scraper=_StubTweetScraper([]),
        llm_client=None,
    )

    report = await agent.run_pipeline("sports", top_n=20)

    assert report.accounts == []
    assert report.tweet_statistics.tweets_collected == 0
