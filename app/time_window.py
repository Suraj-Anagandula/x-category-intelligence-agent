"""Time-window resolution and tweet filtering for category analysis - pure
functions, no I/O, matching `app/account_ranker.py`'s convention.

All comparisons use timezone-aware UTC datetimes. X's own tweet timestamps
are already UTC (Twikit's legacy `created_at` format always carries a
`+0000` offset - see `app/client.py::_parse_created_at`); a naive datetime
is treated as already-UTC rather than raising, matching the same defensive
convention already used elsewhere in this codebase
(`ui/utils.py::freshness_state`, `app/signal_score.py::compute_momentum`).

"latest" mode means "no filtering at all" - it preserves the pipeline's
original behavior (most recent available tweets per account) exactly;
every other mode resolves to a concrete `[start, end)` UTC window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.models import Tweet

#: Fixed set of supported time-window modes, in the order they should be
#: offered in the UI.
TIME_WINDOW_MODES: tuple[str, ...] = ("latest", "24h", "7d", "30d", "custom")

TIME_WINDOW_MODE_LABELS: dict[str, str] = {
    "latest": "Latest Available",
    "24h": "Last 24 Hours",
    "7d": "Last 7 Days",
    "30d": "Last 30 Days",
    "custom": "Custom Range",
}


@dataclass
class TimeWindow:
    """A resolved analysis time window.

    `mode="latest"` means "no filtering" (`start`/`end` stay `None`) -
    this is the flag every downstream consumer (`app/client.py`,
    `app/tweet_scraper.py`, `app/category_agent.py`) checks via
    `is_filtered` before doing anything different from the original,
    unwindowed behavior.
    """

    mode: str
    start: datetime | None = None
    end: datetime | None = None

    @property
    def label(self) -> str:
        return TIME_WINDOW_MODE_LABELS.get(self.mode, self.mode)

    @property
    def is_filtered(self) -> bool:
        return self.mode != "latest"


def to_utc(value: datetime) -> datetime:
    """Treat a naive datetime as already-UTC; convert an aware one to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def resolve_time_window(
    mode: str,
    custom_start: datetime | None = None,
    custom_end: datetime | None = None,
    now: datetime | None = None,
) -> TimeWindow:
    """Resolve a UI-selected mode into a concrete `[start, end)` UTC window.

    - "latest": no window - preserves the original "most recent available
      tweets" behavior exactly.
    - "24h" / "7d" / "30d": `[now - duration, now)`.
    - "custom": `[custom_start, custom_end)`, both required.

    Raises `ValueError` for an unknown mode, or a "custom" mode missing
    either bound - both are genuine caller errors, not a data condition to
    silently paper over.
    """
    reference = to_utc(now) if now is not None else datetime.now(timezone.utc)

    if mode == "latest":
        return TimeWindow(mode="latest")
    if mode == "24h":
        return TimeWindow(mode="24h", start=reference - timedelta(hours=24), end=reference)
    if mode == "7d":
        return TimeWindow(mode="7d", start=reference - timedelta(days=7), end=reference)
    if mode == "30d":
        return TimeWindow(mode="30d", start=reference - timedelta(days=30), end=reference)
    if mode == "custom":
        if custom_start is None or custom_end is None:
            raise ValueError("A custom time window requires both a start and an end datetime.")
        return TimeWindow(mode="custom", start=to_utc(custom_start), end=to_utc(custom_end))

    raise ValueError(f"Unknown time window mode: {mode!r}")


def tweet_in_window(tweet: Tweet, window: TimeWindow) -> bool:
    """`START <= tweet.created_at < END`, using the tweet's real X creation
    timestamp - never `scraped_at`/run time as a substitute.

    A tweet with no `created_at` can never satisfy an *active* window (it's
    excluded, not fabricated as in-range). "latest" mode (no window) always
    returns `True` - no filtering.
    """
    if not window.is_filtered:
        return True
    if tweet.created_at is None:
        return False
    created = to_utc(tweet.created_at)
    return window.start <= created < window.end


def filter_tweets_to_window(tweets: list[Tweet], window: TimeWindow) -> list[Tweet]:
    """Filter `tweets` to those satisfying `tweet_in_window`.

    Returns a new list; never mutates the input. "latest" mode returns the
    input tweets unchanged (as a new list) - no filtering.
    """
    if not window.is_filtered:
        return list(tweets)
    return [tweet for tweet in tweets if tweet_in_window(tweet, window)]


def oldest_created_at(tweets: list[Tweet]) -> datetime | None:
    """The earliest `created_at` among `tweets` that actually have one, or
    `None` if none do. Used by pagination (`app/client.py`) to decide
    whether a just-fetched page has already gone back past a window's
    start - tweets with no timestamp are ignored here rather than treated
    as arbitrarily old or new.
    """
    dated = [to_utc(tweet.created_at) for tweet in tweets if tweet.created_at is not None]
    return min(dated) if dated else None


def _parse_iso(value: str) -> datetime:
    """`datetime.fromisoformat()` on Python 3.10 (this project's minimum
    supported version) doesn't accept a trailing "Z" - Pydantic's JSON
    mode (`model_dump(mode="json")`, used when writing `time_window` into
    the stored snapshot) serializes UTC datetimes with exactly that "Z"
    suffix. Normalize it to the "+00:00" Python 3.10 does accept."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def time_window_from_dict(data: dict | None) -> TimeWindow | None:
    """Reconstruct a `TimeWindow` from a stored snapshot's `"time_window"`
    dict (see `app/schemas.py::TimeWindowInfo`), or `None` if `data` is
    missing/empty/unparseable - snapshots saved before this feature existed
    simply have no `"time_window"` key, and this must not raise for them.
    """
    if not data:
        return None
    mode = data.get("mode", "latest")
    start_raw = data.get("start")
    end_raw = data.get("end")
    try:
        start = _parse_iso(start_raw) if start_raw else None
        end = _parse_iso(end_raw) if end_raw else None
    except ValueError:
        return None
    if mode not in TIME_WINDOW_MODES:
        return None
    return TimeWindow(mode=mode, start=start, end=end)
