"""Read-only access to the existing on-disk storage formats.

Every function here only reads files `app.storage`/`app.exporter` already
write (`data/tweets/<category>/<date>.json`, `data/csv/*.csv`) - it never
triggers scraping and never modifies the storage format. Deliberately
un-cached (no `st.cache_data`): these are small local file reads, and
correctness (never showing stale data right after a fresh run) matters more
here than shaving milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.config import settings

CSV_ENCODING = "utf-8-sig"  # matches app/exporter.py's CSV_ENCODING


def list_run_files(category: str | None = None) -> list[Path]:
    """All saved category-run JSON snapshots, newest first.

    If `category` is given, only that category's runs.
    """
    root = settings.tweets_output_dir
    if not root.exists():
        return []

    pattern = f"{category}/*.json" if category else "*/*.json"
    files = list(root.glob(pattern))
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def load_run_json(path: Path | str) -> dict:
    """Parse a single saved run snapshot (the same format `save_category_run`
    in app/storage.py writes). Accepts a `str` as well as a `Path` - callers
    commonly pass a `history_df["path"]` value, which is stored as a plain
    string (see `load_run_history` below)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run_history(category: str | None = None) -> pd.DataFrame:
    """One row per saved run, newest first - for the Reports/Compare pages.

    If `category` is given, only that category's runs (used by Compare's
    date pickers and Ask Intelligence's category scoping); omitted/`None`
    preserves the original "all categories" behavior. Skips any file that
    fails to parse rather than aborting the whole listing.
    """
    rows = []
    for path in list_run_files(category):
        try:
            data = load_run_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "category": data.get("category", path.parent.name),
                "date": path.stem,
                "accounts": len(data.get("accounts", [])),
                "tweets": len(data.get("tweets", [])),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows, columns=["category", "date", "accounts", "tweets", "path"])


def load_latest_run(category: str) -> dict | None:
    """The most recently saved run snapshot for one category, or None."""
    files = list_run_files(category)
    if not files:
        return None
    try:
        return load_run_json(files[0])
    except (OSError, json.JSONDecodeError):
        return None


def load_category_csv(category: str) -> pd.DataFrame | None:
    """The consolidated `data/csv/<category>_tweets.csv`, or None if it
    hasn't been generated yet."""
    path = settings.csv_output_dir / f"{category}_tweets.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"tweet_id": str, "user_id": str}, encoding=CSV_ENCODING)


def load_users_csv() -> pd.DataFrame | None:
    """The cumulative `data/csv/users.csv`, or None if it doesn't exist yet."""
    path = settings.csv_output_dir / "users.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"id": str, "pinned_tweet_id": str}, encoding=CSV_ENCODING)


def get_discovery_reasons(category: str) -> dict[str, str]:
    """Per-account discovery reason, joined from the consolidated CSV.

    `RankedAccount`/the run JSON's "accounts" list don't carry
    `discovery_reason` (only `CategoryTweetCSVExporter`'s per-tweet rows do -
    see app/exporter.py), so this reads the existing CSV rather than
    changing what the backend persists.
    """
    df = load_category_csv(category)
    if df is None or "discovery_reason" not in df.columns or "username" not in df.columns:
        return {}

    reasons: dict[str, str] = {}
    for _, row in df.dropna(subset=["username"]).iterrows():
        username = str(row["username"])
        reason = row.get("discovery_reason")
        if username not in reasons and pd.notna(reason) and str(reason).strip():
            reasons[username] = str(reason)
    return reasons
