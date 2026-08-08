"""Pydantic models for the category-intelligence layer.

Kept separate from `app/models.py`, which stays scoped to raw scrape objects
(`UserProfile`, `Tweet`, `ScrapeResult`, `TweetScrapeResult`). Everything here
describes structured output *derived* from those raw objects: category
context, ranking results, and analysis/report output.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.models import Tweet


class CategoryContext(BaseModel):
    """Normalized category plus the keywords/subcategories used to drive
    discovery and relevance scoring."""

    category: str
    subcategories: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class DiscoveredAccount(BaseModel):
    """One candidate username surfaced by discovery, with the reason it was
    suggested. `reason` is discovery metadata (why the LLM proposed this
    account) - never a substitute for real profile/tweet data, which always
    comes from the existing scraper."""

    username: str
    reason: str = ""


class RankedAccount(BaseModel):
    """One account's position and component scores in a ranked result."""

    rank: int
    username: str
    display_name: str | None = None
    followers: int | None = None
    category_relevance: float
    engagement_score: float
    activity_score: float
    audience_score: float
    ranking_score: float


class TweetStatistics(BaseModel):
    accounts_processed: int = 0
    accounts_failed: int = 0
    #: Breakdown of `accounts_failed`: how many failed specifically due to
    #: X rate-limiting vs. any other reason (not found, protected, network,
    #: etc.) - `accounts_rate_limited + accounts_failed_other == accounts_failed`.
    accounts_rate_limited: int = 0
    accounts_failed_other: int = 0
    tweets_collected: int = 0


class SentimentBreakdown(BaseModel):
    positive: float = 0.0
    neutral: float = 0.0
    negative: float = 0.0


class CategoryAnalysis(BaseModel):
    trending_topics: list[str] = Field(default_factory=list)
    sentiment: SentimentBreakdown = Field(default_factory=SentimentBreakdown)
    most_discussed_accounts: list[str] = Field(default_factory=list)
    high_engagement_tweets: list[Tweet] = Field(default_factory=list)
    summary: str = ""


class CategoryReport(BaseModel):
    """Final output of `CategoryIntelligenceAgent.run_pipeline()`."""

    category: str
    accounts: list[RankedAccount] = Field(default_factory=list)
    tweet_statistics: TweetStatistics = Field(default_factory=TweetStatistics)
    analysis: CategoryAnalysis = Field(default_factory=CategoryAnalysis)
    errors: list[dict[str, str]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_flat_dict(self) -> dict:
        return self.model_dump(mode="json")
