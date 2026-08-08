"""Pydantic data models for public X profile information."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, computed_field

_MENTION_RE = re.compile(r"(?<!\w)@(\w+)")


class UserProfile(BaseModel):
    """Publicly available profile information for a single X account."""

    model_config = ConfigDict(populate_by_name=True)

    # --- Basic information ---
    id: str | None = Field(default=None, description="Numeric user ID")
    username: str = Field(description="Handle, without the leading '@'")
    display_name: str | None = Field(default=None, description="Human-readable display name")
    bio: str | None = Field(default=None, description="Profile description / bio text")
    location: str | None = Field(default=None, description="User-supplied location string")
    website: str | None = Field(default=None, description="URL from the profile's bio link")
    profile_image: str | None = Field(default=None, description="Profile (avatar) image URL")
    banner_image: str | None = Field(default=None, description="Profile banner/header image URL")
    protected: bool = Field(default=False, description="Whether the account's tweets are protected")
    verified: bool = Field(default=False, description="Legacy 'verified' flag")

    # --- Statistics ---
    followers: int | None = Field(default=None, ge=0, description="Followers count")
    following: int | None = Field(default=None, ge=0, description="Following (friends) count")
    tweets: int | None = Field(default=None, ge=0, description="Total tweet/status count")
    likes: int | None = Field(default=None, ge=0, description="Total likes ('favourites') count")
    media_count: int | None = Field(
        default=None, ge=0, description="Count of tweets containing media"
    )

    # --- Dates ---
    created_at: datetime | None = Field(default=None, description="Account creation timestamp")

    # --- Additional public metadata ---
    pinned_tweet_id: str | None = Field(default=None, description="ID of the pinned tweet, if any")
    language: str | None = Field(default=None, description="Account/profile language code")
    is_blue_verified: bool = Field(
        default=False, description="X Premium ('Blue') verification badge"
    )
    profile_url: str | None = Field(
        default=None, description="Canonical https://x.com/<username> URL"
    )

    # --- Bookkeeping ---
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this record was collected",
    )

    @computed_field  # type: ignore[misc]
    @property
    def resolved_profile_url(self) -> str:
        """Fall back to a derived profile URL if one wasn't explicitly set."""
        return self.profile_url or f"https://x.com/{self.username}"

    def to_flat_dict(self) -> dict:
        """Return a JSON/CSV-friendly flat dict (datetimes as ISO-8601 strings)."""
        data = self.model_dump(mode="json")
        return data


class Tweet(BaseModel):
    """A single public tweet from a user's timeline."""

    id: str
    text: str
    created_at: datetime | None = None
    reply_count: int | None = Field(default=None, ge=0)
    retweet_count: int | None = Field(default=None, ge=0)
    like_count: int | None = Field(default=None, ge=0)
    quote_count: int | None = Field(default=None, ge=0)
    view_count: int | None = Field(default=None, ge=0)
    is_retweet: bool = False
    is_reply: bool = False
    lang: str | None = None
    username: str | None = Field(
        default=None,
        description="Handle of the account this tweet belongs to; set when scraped as part of a multi-account batch",
    )
    hashtags: list[str] = Field(default_factory=list, description="Hashtags in the tweet text")
    media_urls: list[str] = Field(
        default_factory=list, description="URLs of attached photo/video/GIF media"
    )
    quoted_tweet_id: str | None = Field(
        default=None, description="ID of the quoted tweet, if this tweet quotes another"
    )

    @computed_field  # type: ignore[misc]
    @property
    def url(self) -> str:
        return f"https://x.com/i/status/{self.id}"

    @computed_field  # type: ignore[misc]
    @property
    def mentions(self) -> list[str]:
        """@-mentions found in the tweet's own text (derived, not a separate
        scraped field - Twikit/twifork expose no dedicated mentions entity)."""
        return _MENTION_RE.findall(self.text) if self.text else []

    def to_flat_dict(self) -> dict:
        return self.model_dump(mode="json")


class ScrapeResult(BaseModel):
    """Outcome of attempting to scrape a single username, success or failure."""

    username: str
    success: bool
    profile: UserProfile | None = None
    error: str | None = None
    error_type: str | None = None
    attempts: int = 1
    from_cache: bool = False


class TweetScrapeResult(BaseModel):
    """Outcome of attempting to fetch one username's recent tweets."""

    username: str
    success: bool
    tweets: list[Tweet] = Field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    attempts: int = 1
    from_cache: bool = False
