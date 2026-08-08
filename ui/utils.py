"""Small, dependency-light helpers for the Streamlit UI.

Kept free of Streamlit imports so these are trivially unit-testable, mirroring
`app/utils.py`'s own design note.
"""

from __future__ import annotations

from typing import Any


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
