"""Small, dependency-light helpers for the Streamlit UI.

Kept free of Streamlit imports so these are trivially unit-testable, mirroring
`app/utils.py`'s own design note.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Re-exported for existing call sites (ui/pages/overview.py,
# ui/pages/intelligence.py) - the actual implementation lives in
# app/topic_matching.py since app/story_opportunities.py (backend) needs it
# too, and app/ must never depend on ui/.
from app.topic_matching import (  # noqa: F401
    count_topic_mentions,
    distinct_authors_for_topic,
    group_tweets_by_topic,
)

#: Fixed, short list of categories offered as one-click starting points on
#: Overview/New Analysis - not fetched from anywhere, just a convenience
#: shortcut into the same free-text category field the backend already takes.
CATEGORY_QUICK_PICKS: list[str] = ["Technology", "Politics", "Healthcare", "Business", "Sports"]

#: depth_key -> (candidate_limit, top_accounts, tweets_per_account). Fixed
#: presets so normal users pick "Standard"/"Deep" instead of three raw
#: numbers; power users can still override via the Advanced expander.
DEPTH_PRESETS: dict[str, tuple[int, int, int]] = {
    "standard": (50, 20, 10),
    "deep": (100, 40, 20),
}


def resolve_depth_preset(depth_key: str) -> tuple[int, int, int]:
    """Map a depth-preset key to (candidate_limit, top_accounts, tweets_per_account).

    Falls back to the "standard" preset for an unrecognized key rather than
    raising, since this only ever drives UI defaults.
    """
    return DEPTH_PRESETS.get(depth_key.lower(), DEPTH_PRESETS["standard"])


def format_compact_number(value: int | float | None) -> str:
    """Render a large number compactly, e.g. 241200000 -> "241.2M".

    Returns "-" for None, matching the CLI's existing convention for
    missing values (see main.py's table-formatting f-strings).
    """
    if value is None:
        return "-"

    number = float(value)
    sign = "-" if number < 0 else ""
    number = abs(number)

    for suffix, threshold in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if number >= threshold:
            return f"{sign}{number / threshold:.1f}{suffix}"

    return f"{sign}{number:,.0f}"


def validate_time_window_params(
    mode: str, custom_start: datetime | None, custom_end: datetime | None
) -> str | None:
    """Validate the New Analysis page's time-window inputs before running
    the pipeline. Returns `None` if valid, otherwise a human-readable
    error message. Only "custom" mode has anything to validate - the
    preset modes ("latest"/"24h"/"7d"/"30d") are always valid since they
    carry no free-text input.
    """
    if mode != "custom":
        return None
    if custom_start is None or custom_end is None:
        return "Custom Range requires both a start and an end date/time."
    if custom_start >= custom_end:
        return "The custom range's start must be before its end."
    return None


def validate_pipeline_params(
    category: str,
    candidate_limit: int,
    top_accounts: int,
    tweets_per_account: int,
) -> str | None:
    """Validate Analyze Category inputs before running the pipeline.

    Returns None if valid, otherwise a human-readable error message
    describing the first rule violated.
    """
    if not category or not category.strip():
        return "Category is required."
    if candidate_limit < 1:
        return "Candidate limit must be at least 1."
    if top_accounts < 1:
        return "Top accounts must be at least 1."
    if tweets_per_account < 1:
        return "Tweets per account must be at least 1."
    if top_accounts > candidate_limit:
        return "Top accounts cannot exceed the candidate limit."
    return None


#: (state_key, human label) thresholds for freshness_state, in hours.
_FRESHNESS_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (6.0, "fresh"),
    (24.0, "aging"),
)


def freshness_state(scraped_at: str | None, now: datetime | None = None) -> tuple[str, str]:
    """Bucket a run's `scraped_at` ISO timestamp into a freshness state key
    ("fresh"/"aging"/"stale"/"unknown") plus a human-readable relative-time
    label. `now` is injectable so tests don't depend on real wall-clock
    time. Never raises on a missing/unparseable timestamp - returns
    ("unknown", "Unknown") instead.
    """
    if not scraped_at:
        return "unknown", "Unknown"
    try:
        parsed = datetime.fromisoformat(scraped_at)
    except ValueError:
        return "unknown", "Unknown"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    reference = now or datetime.now(timezone.utc)
    age_hours = (reference - parsed).total_seconds() / 3600
    if age_hours < 0:
        age_hours = 0.0

    state = "stale"
    for threshold, key in _FRESHNESS_THRESHOLDS:
        if age_hours <= threshold:
            state = key
            break

    if age_hours < 1:
        label = "Last analyzed less than an hour ago"
    elif age_hours < 24:
        hours = int(age_hours)
        label = f"Last analyzed {hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(age_hours // 24)
        label = f"Last analyzed {days} day{'s' if days != 1 else ''} ago"

    return state, label


def report_fallback_dict(report: Any) -> dict:
    """Defensive fallback if a run snapshot somehow wasn't found on disk
    right after a live run (`save_category_run` always writes it inside
    `run_pipeline`, so this should not normally trigger)."""
    return {
        "category": report.category,
        "accounts": [account.model_dump(mode="json") for account in report.accounts],
        "tweets": [],
        "tweet_statistics": report.tweet_statistics.model_dump(mode="json"),
        "analysis": report.analysis.model_dump(mode="json"),
        "errors": report.errors,
    }


def credential_status(settings: Any) -> dict[str, bool]:
    """Whether X auth and the configured LLM provider are set up - booleans
    only. Never touches or returns the underlying secret values themselves;
    wraps the existing `Settings.has_cookie_credentials`/`Settings.has_llm`
    properties rather than re-deriving credential presence."""
    return {
        "x_auth": bool(settings.has_cookie_credentials),
        "llm": bool(settings.has_llm),
        "cache": bool(settings.cache_enabled),
    }
