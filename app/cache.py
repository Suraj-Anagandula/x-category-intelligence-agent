"""TTL-based on-disk cache for scraped profiles.

Avoids re-requesting a profile that was already fetched recently. Storage
is one JSON file per username under `cache_dir`, each wrapping the cached
payload with a `cached_at` timestamp used to evaluate the TTL.

The interface is intentionally storage-agnostic-ish (get/set/invalidate)
so it could later be swapped for a Redis/SQLite backend without touching
callers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from app.logger import get_logger
from app.models import Tweet, UserProfile

logger = get_logger()


def _safe_key(username: str) -> str:
    return username.lower().strip().lstrip("@")


class ProfileCache:
    """Async, file-backed TTL cache mapping username -> UserProfile."""

    def __init__(self, cache_dir: Path, ttl_seconds: int, enabled: bool = True) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, username: str) -> Path:
        return self.cache_dir / f"{_safe_key(username)}.json"

    async def get(self, username: str) -> UserProfile | None:
        """Return a cached profile if present and not yet expired, else None."""
        if not self.enabled:
            return None

        path = self._path_for(username)
        if not path.exists():
            return None

        try:
            async with aiofiles.open(path, encoding="utf-8") as fh:
                raw = await fh.read()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Cache read failed for @{username}: {exc}")
            return None

        cached_at = datetime.fromisoformat(payload["cached_at"])
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()
        if age > self.ttl_seconds:
            logger.debug(f"Cache expired for @{username} (age={age:.0f}s)")
            return None

        logger.debug(f"Cache hit for @{username} (age={age:.0f}s)")
        return UserProfile.model_validate(payload["profile"])

    async def set(self, username: str, profile: UserProfile) -> None:
        """Persist a profile to the cache with the current timestamp."""
        if not self.enabled:
            return

        payload = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile.to_flat_dict(),
        }
        path = self._path_for(username)
        async with aiofiles.open(path, "w", encoding="utf-8") as fh:
            await fh.write(json.dumps(payload, default=str))

    async def invalidate(self, username: str) -> None:
        """Remove any cached entry for `username`."""
        path = self._path_for(username)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        """Remove all cached entries."""
        if not self.cache_dir.exists():
            return
        for file in self.cache_dir.glob("*.json"):
            file.unlink()


class TweetCache:
    """Async, file-backed TTL cache mapping username -> list[Tweet].

    Structurally identical to `ProfileCache` but stores a batch of tweets per
    username rather than a single profile, under its own directory/TTL since
    tweets go stale faster than profile metadata.
    """

    def __init__(self, cache_dir: Path, ttl_seconds: int, enabled: bool = True) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, username: str) -> Path:
        return self.cache_dir / f"{_safe_key(username)}.json"

    async def get(self, username: str) -> list[Tweet] | None:
        """Return cached tweets if present and not yet expired, else None."""
        if not self.enabled:
            return None

        path = self._path_for(username)
        if not path.exists():
            return None

        try:
            async with aiofiles.open(path, encoding="utf-8") as fh:
                raw = await fh.read()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Tweet cache read failed for @{username}: {exc}")
            return None

        cached_at = datetime.fromisoformat(payload["cached_at"])
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()
        if age > self.ttl_seconds:
            logger.debug(f"Tweet cache expired for @{username} (age={age:.0f}s)")
            return None

        logger.debug(f"Tweet cache hit for @{username} (age={age:.0f}s)")
        return [Tweet.model_validate(item) for item in payload["tweets"]]

    async def set(self, username: str, tweets: list[Tweet]) -> None:
        """Persist a batch of tweets to the cache with the current timestamp."""
        if not self.enabled:
            return

        payload = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "tweets": [tweet.to_flat_dict() for tweet in tweets],
        }
        path = self._path_for(username)
        async with aiofiles.open(path, "w", encoding="utf-8") as fh:
            await fh.write(json.dumps(payload, default=str))

    async def invalidate(self, username: str) -> None:
        """Remove any cached entry for `username`."""
        path = self._path_for(username)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        """Remove all cached entries."""
        if not self.cache_dir.exists():
            return
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
