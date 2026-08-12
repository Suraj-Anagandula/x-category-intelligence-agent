"""Thin wrapper around Twikit's `Client`.

Responsibilities:
    * Establish (and cache) an authenticated session.
    * Fetch a single user's public profile fields by screen name.
    * Translate Twikit's exception types into this app's exception
      hierarchy so the rest of the codebase never imports Twikit directly.

Twikit is imported lazily inside methods (not at module import time) so
that modules which don't need live network access - `models`, `exporter`,
`utils`, and their tests - never pay the import cost or require the
dependency to be installed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.exceptions import (
    AccountSuspendedError,
    AuthenticationError,
    NetworkTimeoutError,
    ProtectedAccountError,
    RateLimitError,
    TransientRequestError,
    UserNotFoundError,
)
from app.logger import get_logger
from app.models import Tweet, UserProfile
from app.time_window import TimeWindow, oldest_created_at

logger = get_logger()


def _parse_created_at(value: Any) -> Any:
    """Twikit returns Twitter's legacy date format; let pydantic coerce strings,
    but normalize the common 'Wed Oct 10 20:19:24 +0000 2018' format explicitly
    since that one isn't ISO-8601.
    """
    if value is None or not isinstance(value, str):
        return value
    from datetime import datetime

    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return value


def _rate_limit_retry_after(exc: Exception) -> float | None:
    """Extract a relative retry-after delay (seconds) from a Twikit
    `TooManyRequests` exception, if it carried one.

    Twifork's `TooManyRequests.rate_limit_reset` (when the `x-rate-limit-reset`
    response header was present) is an *absolute* Unix timestamp for when
    X's rate-limit window resets - not a relative delay. Convert it to a
    relative number of seconds from now so `RateLimitError.retry_after` and
    `retry_with_backoff` can treat every rate-limit source the same way.
    """
    reset_at = getattr(exc, "rate_limit_reset", None)
    if not reset_at:
        return None
    try:
        delay = float(reset_at) - time.time()
    except (TypeError, ValueError):
        return None
    return max(delay, 0.0)


def _map_twikit_exception(exc: Exception, username: str) -> Exception:
    """Translate a Twikit/network exception into our exception hierarchy."""
    name = exc.__class__.__name__

    if name in {"UserNotFound", "UserNotFoundError"}:
        return UserNotFoundError(username)
    if name in {"AccountSuspended", "AccountSuspendedError"}:
        return AccountSuspendedError(username)
    if name in {"UserUnavailable"}:
        return ProtectedAccountError(username)
    if name in {"TooManyRequests", "RateLimitExceeded"}:
        return RateLimitError(
            f"Rate limited while fetching @{username}", retry_after=_rate_limit_retry_after(exc)
        )
    if name in {"Unauthorized", "Forbidden", "AuthenticationError"}:
        return AuthenticationError(f"Authentication failed while fetching @{username}: {exc}")
    if name in {"RequestTimeout", "TimeoutError"}:
        return NetworkTimeoutError(f"Timed out fetching @{username}")
    if name in {"ServerError", "CouldNotSendRequest", "ConnectionError"}:
        return TransientRequestError(f"Transient error fetching @{username}: {exc}")
    return TransientRequestError(f"Unexpected error fetching @{username}: {exc}")


class TwikitProfileClient:
    """Manages a single Twikit `Client` session and profile lookups."""

    def __init__(
        self,
        *,
        auth_token: str | None = None,
        ct0: str | None = None,
        username: str | None = None,
        email: str | None = None,
        password: str | None = None,
        session_file: Path,
        language: str = "en-US",
    ) -> None:
        self._auth_token = auth_token
        self._ct0 = ct0
        self._username = username
        self._email = email
        self._password = password
        self._session_file = session_file
        self._language = language
        self._client: Any = None

    async def connect(self) -> None:
        """Establish a session, preferring a cached session, then browser-exported
        cookies, then (legacy, likely non-functional) password login.

        X has retired third-party password login (see README "Authentication"),
        so `X_AUTH_TOKEN` / `X_CT0` cookies are the primary supported path.
        """
        try:
            from twikit import Client
        except ImportError as exc:
            raise AuthenticationError(
                "twikit is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self._client = Client(self._language)
        self._session_file.parent.mkdir(parents=True, exist_ok=True)

        if self._session_file.exists():
            try:
                self._client.load_cookies(str(self._session_file))
                await self._client.user_id()  # cheap authenticated call; raises if the session is stale
                logger.info(f"Loaded cached X session from {self._session_file}")
                return
            except Exception as exc:  # noqa: BLE001 - fall through to fresh auth
                logger.warning(f"Cached session invalid, will re-authenticate: {exc}")

        if self._auth_token and self._ct0:
            self._client.set_cookies({"auth_token": self._auth_token, "ct0": self._ct0})
            try:
                await self._client.user_id()
            except Exception as exc:  # noqa: BLE001
                raise AuthenticationError(
                    "X_AUTH_TOKEN/X_CT0 cookies were rejected; export fresh ones from your "
                    f"browser: {exc}"
                ) from exc

            self._client.save_cookies(str(self._session_file))
            logger.info(f"Authenticated via cookies; session cached to {self._session_file}")
            return

        if self._username and self._email and self._password:
            try:
                await self._client.login(
                    auth_info_1=self._username,
                    auth_info_2=self._email,
                    password=self._password,
                )
            except Exception as exc:  # noqa: BLE001
                raise AuthenticationError(
                    "Password login to X failed (X has retired third-party password "
                    f"login for most accounts; use X_AUTH_TOKEN/X_CT0 cookies instead): {exc}"
                ) from exc

            self._client.save_cookies(str(self._session_file))
            logger.info(
                f"Logged in to X as @{self._username}; session cached to {self._session_file}"
            )
            return

        raise AuthenticationError(
            "No cached session and no credentials configured. Set X_AUTH_TOKEN and X_CT0 "
            "(browser-exported cookies) in .env - see README 'Authentication'."
        )

    async def get_profile(self, username: str) -> UserProfile:
        """Fetch and normalize the public profile for `username`."""
        if self._client is None:
            raise AuthenticationError("Client not connected; call connect() first.")

        try:
            user = await self._client.get_user_by_screen_name(username)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error below
            raise _map_twikit_exception(exc, username) from exc

        if user is None:
            raise UserNotFoundError(username)

        return self._to_profile(username, user)

    async def get_recent_tweets(
        self,
        username: str,
        count: int = 10,
        tweet_type: str = "Tweets",
        *,
        window: TimeWindow | None = None,
        max_pages: int = 10,
    ) -> list[Tweet]:
        """Fetch a user's public tweets.

        With no `window` (or an unfiltered/"latest" one), this is exactly
        the original behavior: a single page of up to `count` most-recent
        tweets - no pagination, no behavior change for existing callers.

        With a real (filtered) `window`, pages backward through Twikit's
        cursor-based pagination (`Result.next()`) until one of:
          (a) a fetched page's oldest tweet is already older than
              `window.start` - every further page can only be older still,
              so the window is now fully covered from the "old" side;
          (b) X returns an empty page - no more tweets are available at
              all (there is no guarantee X exposes arbitrarily old
              history - see README for the real limitation);
          (c) `max_pages` pages have been fetched - a safety bound so a
              custom range far in an account's past can't trigger
              unbounded requests.
        Tweets are de-duplicated by id across pages. Returns the fetched
        superset - NOT yet cut to the exact `[start, end)` range; callers
        apply `app.time_window.filter_tweets_to_window` for that precise
        boundary (this keeps "posts fetched" vs. "posts inside window"
        distinctly countable for report metadata).

        `tweet_type` is one of Twikit's {'Tweets', 'Replies', 'Media', 'Likes'}.
        """
        if self._client is None:
            raise AuthenticationError("Client not connected; call connect() first.")

        try:
            user = await self._client.get_user_by_screen_name(username)
        except Exception as exc:  # noqa: BLE001
            raise _map_twikit_exception(exc, username) from exc

        if user is None:
            raise UserNotFoundError(username)

        try:
            page = await self._client.get_user_tweets(user.id, tweet_type, count=count)
        except Exception as exc:  # noqa: BLE001
            raise _map_twikit_exception(exc, username) from exc

        # Twikit's `count` is a page-size hint, not a hard cap - the API can return more.
        page_tweets = [self._to_tweet(tweet) for tweet in page][:count]

        if window is None or not window.is_filtered:
            return page_tweets

        collected: list[Tweet] = list(page_tweets)
        seen_ids = {tweet.id for tweet in page_tweets}
        pages_fetched = 1

        while True:
            oldest = oldest_created_at(page_tweets)
            if oldest is not None and oldest < window.start:
                break  # window covered - every further page is only older
            if pages_fetched >= max_pages:
                logger.warning(
                    f"@{username}: reached the {max_pages}-page pagination limit while "
                    "fetching tweets for the selected time window - results for this "
                    "account may not cover the full requested range."
                )
                break

            try:
                page = await page.next()
            except Exception as exc:  # noqa: BLE001
                raise _map_twikit_exception(exc, username) from exc

            page_tweets = [self._to_tweet(tweet) for tweet in page]
            if not page_tweets:
                break  # X has no more tweets to give for this account

            pages_fetched += 1
            for tweet in page_tweets:
                if tweet.id not in seen_ids:
                    seen_ids.add(tweet.id)
                    collected.append(tweet)

        return collected

    @staticmethod
    def _to_tweet(raw: Any) -> Tweet:
        """Map a Twikit `Tweet` object's fields onto our `Tweet` model."""

        def g(name: str, default: Any = None) -> Any:
            value = getattr(raw, name, None)
            return value if value not in (None, "") else default

        view_count = g("view_count")
        try:
            view_count = int(view_count) if view_count is not None else None
        except (TypeError, ValueError):
            view_count = None

        raw_media = getattr(raw, "media", None) or []
        media_urls = [item.media_url for item in raw_media if getattr(item, "media_url", None)]

        quoted_tweet_id = g("quoted_status_id")

        return Tweet(
            id=str(g("id")),
            text=g("full_text", default=g("text", default="")),
            created_at=_parse_created_at(g("created_at")),
            reply_count=g("reply_count"),
            retweet_count=g("retweet_count"),
            like_count=g("favorite_count"),
            quote_count=g("quote_count"),
            view_count=view_count,
            is_retweet=g("retweeted_tweet") is not None,
            is_reply=g("in_reply_to") is not None,
            lang=g("lang"),
            hashtags=g("hashtags", default=[]),
            media_urls=media_urls,
            quoted_tweet_id=str(quoted_tweet_id) if quoted_tweet_id else None,
        )

    @staticmethod
    def _to_profile(username: str, user: Any) -> UserProfile:
        """Map a Twikit `User` object's fields onto our `UserProfile` model."""

        def g(*names: str, default: Any = None) -> Any:
            """Return the first present, non-empty attribute value.

            X's API returns '' rather than omitting the field for unset
            profile text (location, bio, url, ...); treat that as absent too
            so callers get a clean None instead of an empty string.
            """
            for name in names:
                value = getattr(user, name, None)
                if value not in (None, ""):
                    return value
            return default

        website = None
        entities = g("entities", default=None)
        if isinstance(entities, dict):
            urls = entities.get("url", {}).get("urls") if entities.get("url") else None
            if urls:
                website = urls[0].get("expanded_url")
        website = website or g("url")

        pinned = g("pinned_tweet_ids", default=None)
        pinned_id = pinned[0] if isinstance(pinned, list) and pinned else g("pinned_tweet_id")

        return UserProfile(
            id=str(g("id")) if g("id") is not None else None,
            username=g("screen_name", default=username),
            display_name=g("name"),
            bio=g("description"),
            location=g("location"),
            website=website,
            profile_image=g("profile_image_url", "profile_image_url_https"),
            banner_image=g("profile_banner_url"),
            protected=bool(g("protected", default=False)),
            verified=bool(g("verified", default=False)),
            followers=g("followers_count"),
            following=g("following_count", "friends_count"),
            tweets=g("statuses_count"),
            likes=g("favourites_count"),
            media_count=g("media_count"),
            created_at=_parse_created_at(g("created_at")),
            pinned_tweet_id=str(pinned_id) if pinned_id else None,
            language=g("lang"),
            is_blue_verified=bool(g("is_blue_verified", default=False)),
            profile_url=f"https://x.com/{g('screen_name', default=username)}",
        )
