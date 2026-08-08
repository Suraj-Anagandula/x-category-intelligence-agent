"""Unit tests for app.cache: TTL-based on-disk profile cache."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app.cache import ProfileCache
from app.models import UserProfile


async def test_cache_miss_when_empty(tmp_path) -> None:
    cache = ProfileCache(cache_dir=tmp_path, ttl_seconds=3600)

    assert await cache.get("elonmusk") is None


async def test_cache_set_then_get_hit(tmp_path) -> None:
    cache = ProfileCache(cache_dir=tmp_path, ttl_seconds=3600)
    profile = UserProfile(username="elonmusk", followers=100)

    await cache.set("elonmusk", profile)
    cached = await cache.get("elonmusk")

    assert cached is not None
    assert cached.username == "elonmusk"
    assert cached.followers == 100


async def test_cache_disabled_never_hits(tmp_path) -> None:
    cache = ProfileCache(cache_dir=tmp_path, ttl_seconds=3600, enabled=False)
    profile = UserProfile(username="elonmusk")

    await cache.set("elonmusk", profile)

    assert await cache.get("elonmusk") is None


async def test_cache_expired_entry_returns_none(tmp_path) -> None:
    cache = ProfileCache(cache_dir=tmp_path, ttl_seconds=1)
    profile = UserProfile(username="elonmusk")
    await cache.set("elonmusk", profile)

    # Manually backdate the cached_at timestamp past the TTL.
    path = cache._path_for("elonmusk")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stale = datetime.now(timezone.utc) - timedelta(seconds=10)
    payload["cached_at"] = stale.isoformat()
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")

    assert await cache.get("elonmusk") is None


async def test_cache_invalidate_removes_entry(tmp_path) -> None:
    cache = ProfileCache(cache_dir=tmp_path, ttl_seconds=3600)
    await cache.set("elonmusk", UserProfile(username="elonmusk"))

    await cache.invalidate("elonmusk")

    assert await cache.get("elonmusk") is None


def test_cache_key_is_case_insensitive(tmp_path) -> None:
    cache = ProfileCache(cache_dir=tmp_path, ttl_seconds=3600)

    assert cache._path_for("ElonMusk") == cache._path_for("elonmusk")
