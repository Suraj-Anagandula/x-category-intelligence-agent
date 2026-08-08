"""Custom exception hierarchy for the scraper.

Keeping these distinct (rather than catching bare Exception everywhere)
lets the orchestration layer decide, per failure kind, whether to retry,
skip, or abort.
"""

from __future__ import annotations


class ScraperError(Exception):
    """Base class for all application-raised errors."""


class ConfigurationError(ScraperError):
    """Raised when required configuration is missing or invalid."""


class AuthenticationError(ScraperError):
    """Raised when login/session establishment with X fails."""


class UserNotFoundError(ScraperError):
    """Raised when a requested username does not exist."""

    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(f"User not found: @{username}")


class AccountSuspendedError(ScraperError):
    """Raised when a requested account has been suspended."""

    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(f"Account suspended: @{username}")


class ProtectedAccountError(ScraperError):
    """Raised when an account's tweets/data are protected, if that blocks the read."""

    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(f"Account is protected: @{username}")


class InvalidUsernameError(ScraperError):
    """Raised when a username fails basic syntactic validation."""

    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(f"Invalid username: {username!r}")


class RateLimitError(ScraperError):
    """Raised when X responds with a rate-limit signal.

    Carries an optional `retry_after` hint (seconds) taken from the
    response headers, when the underlying client exposes one.
    """

    def __init__(
        self, message: str = "Rate limited by X", retry_after: float | None = None
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class NetworkTimeoutError(ScraperError):
    """Raised for network-level timeouts and transient connectivity failures."""


class TransientRequestError(ScraperError):
    """Raised for retryable, non-specific request failures (5xx, connection reset)."""


class LLMError(ScraperError):
    """Raised when an LLM call fails or returns unparseable output.

    Never fatal to the category-intelligence pipeline - callers catch this
    and fall back to deterministic logic (curated seeds, keyword-overlap
    scoring, frequency-based analysis).
    """
