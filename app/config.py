"""Centralized application configuration.

All tunables are sourced from environment variables / a `.env` file via
`pydantic-settings`, so the rest of the codebase never touches `os.environ`
directly. Import the module-level `settings` singleton wherever config is
needed.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_CONCURRENCY_LIMITS = (10, 20, 50, 100)


class Settings(BaseSettings):
    """Application settings, loaded from environment / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- X account / session (only used to establish an authenticated session) ---
    # X has retired third-party password login; browser-exported cookies are the
    # only currently working auth path (see README "Authentication" section).
    x_auth_token: str | None = Field(default=None, alias="X_AUTH_TOKEN")
    x_ct0: str | None = Field(default=None, alias="X_CT0")
    # Legacy fallback, kept in case X ever restores password login for Twikit-style clients.
    x_username: str | None = Field(default=None, alias="X_USERNAME")
    x_email: str | None = Field(default=None, alias="X_EMAIL")
    x_password: str | None = Field(default=None, alias="X_PASSWORD")
    x_session_file: Path = Field(default=Path(".data/session.json"), alias="X_SESSION_FILE")

    # --- Output ---
    output_dir: Path = Field(default=Path("data"), alias="OUTPUT_DIR")
    json_output_dir: Path = Field(default=Path("data/json"), alias="JSON_OUTPUT_DIR")
    csv_output_dir: Path = Field(default=Path("data/csv"), alias="CSV_OUTPUT_DIR")
    log_dir: Path = Field(default=Path("data/logs"), alias="LOG_DIR")

    # --- Concurrency & pacing ---
    concurrency_limit: int = Field(default=10, alias="CONCURRENCY_LIMIT")
    request_delay_seconds: float = Field(default=1.0, alias="REQUEST_DELAY_SECONDS")

    # --- Retry / backoff ---
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    backoff_base_seconds: float = Field(default=2.0, alias="BACKOFF_BASE_SECONDS")
    backoff_max_seconds: float = Field(default=30.0, alias="BACKOFF_MAX_SECONDS")
    # Rate limits get a much longer, separate backoff than ordinary transient
    # errors: X's read-endpoint rate-limit windows are commonly ~15 minutes,
    # so retrying every few seconds (the generic backoff above) just burns
    # through retries without the window ever resetting. When X's response
    # exposes a reset time (see app/client.py), that's used directly, capped
    # at this max as a sanity bound; otherwise this is the exponential base/cap.
    rate_limit_base_seconds: float = Field(default=30.0, alias="RATE_LIMIT_BASE_SECONDS")
    rate_limit_max_seconds: float = Field(default=900.0, alias="RATE_LIMIT_MAX_SECONDS")

    # --- Cache ---
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_dir: Path = Field(default=Path(".cache"), alias="CACHE_DIR")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- LLM (category agent / discovery / ranking / analysis) ---
    # Groq is the default/primary provider; Gemini remains available as an
    # explicit opt-in (LLM_PROVIDER=gemini). Whichever is selected, no API
    # key configured means "no LLM" - there is no curated-account fallback.
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-20b", alias="GROQ_MODEL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    # --- Category intelligence ---
    category_candidate_limit: int = Field(default=50, alias="CATEGORY_CANDIDATE_LIMIT")
    top_accounts_limit: int = Field(default=20, alias="TOP_ACCOUNTS_LIMIT")
    tweets_per_account: int = Field(default=10, alias="TWEETS_PER_ACCOUNT")
    tweets_output_dir: Path = Field(default=Path("data/tweets"), alias="TWEETS_OUTPUT_DIR")
    tweet_cache_dir: Path = Field(default=Path(".cache/tweets"), alias="TWEET_CACHE_DIR")
    tweet_cache_ttl_seconds: int = Field(default=900, alias="TWEET_CACHE_TTL_SECONDS")
    # Deliberately independent of (and lower than) CONCURRENCY_LIMIT: X's
    # timeline-read GraphQL endpoint rate-limits noticeably more
    # aggressively than the profile-lookup endpoint, so the profile
    # scraper's concurrency tier is too aggressive for tweet scraping.
    tweet_scrape_concurrency: int = Field(default=3, alias="TWEET_SCRAPE_CONCURRENCY")
    # Only consulted when a real (non-"latest") time window is selected -
    # see app/client.py::TwikitProfileClient.get_recent_tweets. Bounds how
    # many cursor-paginated pages of tweets are fetched per account while
    # trying to reach the requested window's start date, so a custom range
    # far in an account's past can't trigger unbounded requests.
    tweet_window_max_pages: int = Field(default=10, alias="TWEET_WINDOW_MAX_PAGES")

    # --- Ask Intelligence (RAG) ---
    # Only touched by app/rag/* (lazily imported - chromadb/sentence-transformers
    # are an optional extra, see pyproject.toml's `rag` group). A vector index is
    # derived/regenerable from data/tweets/*, so its directory lives alongside
    # .cache/ rather than data/ and is gitignored.
    chroma_dir: Path = Field(default=Path(".chroma"), alias="CHROMA_DIR")
    #: Minimum similarity (1 - cosine distance) for a retrieved chunk to count
    #: as evidence at all - needs empirical tuning against real indexed data,
    #: hence a config knob rather than a hardcoded constant.
    rag_min_similarity: float = Field(default=0.30, alias="RAG_MIN_SIMILARITY")

    @field_validator("concurrency_limit")
    @classmethod
    def _clamp_concurrency(cls, value: int) -> int:
        """Snap arbitrary values to the nearest supported tier (10/20/50/100)."""
        if value in ALLOWED_CONCURRENCY_LIMITS:
            return value
        closest = min(ALLOWED_CONCURRENCY_LIMITS, key=lambda tier: abs(tier - value))
        return closest

    def ensure_directories(self) -> None:
        """Create all output/cache/log directories if they don't already exist."""
        for path in (
            self.output_dir,
            self.json_output_dir,
            self.csv_output_dir,
            self.log_dir,
            self.cache_dir,
            self.x_session_file.parent,
            self.tweets_output_dir,
            self.tweet_cache_dir,
            self.chroma_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def has_llm(self) -> bool:
        provider = (self.llm_provider or "groq").strip().lower()
        if provider == "gemini":
            return bool(self.gemini_api_key)
        return bool(self.groq_api_key)

    @property
    def has_cookie_credentials(self) -> bool:
        return bool(self.x_auth_token and self.x_ct0)

    @property
    def has_password_credentials(self) -> bool:
        return bool(self.x_username and self.x_email and self.x_password)

    @property
    def has_credentials(self) -> bool:
        return self.has_cookie_credentials or self.has_password_credentials


settings = Settings()
