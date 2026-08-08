"""Unit tests for the Streamlit UI layer (ui/).

Function-level only - no real Streamlit process, no network, no real X/LLM
credentials. `ui.pipeline_runner.run_category_analysis` is tested by
stubbing `CategoryIntelligenceAgent`/`TwikitProfileClient`, the same pattern
`tests/test_category_agent.py` already uses for the pipeline itself.
"""

from __future__ import annotations

import json

import pandas as pd

from app.config import Settings
from app.logger import get_logger
from ui import data_loader, pipeline_runner
from ui.utils import credential_status, format_compact_number, validate_pipeline_params


def _settings(tmp_path) -> Settings:
    settings = Settings()
    settings.tweets_output_dir = tmp_path / "tweets"
    settings.csv_output_dir = tmp_path / "csv"
    return settings


def _pipeline_settings(tmp_path) -> Settings:
    """A Settings instance fully isolated to tmp_path, for tests that
    exercise run_category_analysis (which calls settings.ensure_directories()) -
    every directory it touches must be redirected, not just the two
    data_loader cares about."""
    settings = _settings(tmp_path)
    settings.output_dir = tmp_path / "data"
    settings.json_output_dir = tmp_path / "data" / "json"
    settings.log_dir = tmp_path / "data" / "logs"
    settings.cache_dir = tmp_path / "cache"
    settings.tweet_cache_dir = tmp_path / "tweet_cache"
    settings.x_session_file = tmp_path / "session.json"
    settings.cache_enabled = False
    return settings


def _write_run(tmp_path, category: str, date: str, accounts: list, tweets: list) -> None:
    run_dir = tmp_path / "tweets" / category
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "category": category,
        "scraped_at": f"{date}T00:00:00+00:00",
        "accounts": accounts,
        "tweets": tweets,
        "tweet_statistics": {
            "accounts_processed": len(accounts),
            "accounts_failed": 0,
            "accounts_rate_limited": 0,
            "accounts_failed_other": 0,
            "tweets_collected": len(tweets),
        },
        "analysis": {"trending_topics": [], "sentiment": {}, "summary": ""},
        "errors": [],
    }
    (run_dir / f"{date}.json").write_text(json.dumps(payload), encoding="utf-8")


# --- data_loader: run history / JSON -----------------------------------------


def test_list_run_files_empty_when_no_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_loader, "settings", _settings(tmp_path))

    assert data_loader.list_run_files() == []


def test_load_run_history_empty_dataframe_when_no_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_loader, "settings", _settings(tmp_path))

    history = data_loader.load_run_history()

    assert history.empty
    assert list(history.columns) == ["category", "date", "accounts", "tweets", "path"]


def test_list_run_files_and_load_run_history_reflect_saved_runs(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(data_loader, "settings", settings)

    _write_run(
        tmp_path, "sports", "2026-08-01", accounts=[{"username": "espn"}], tweets=[{"id": "1"}]
    )
    _write_run(
        tmp_path,
        "technology",
        "2026-08-02",
        accounts=[{"username": "openai"}, {"username": "nvidia"}],
        tweets=[{"id": "1"}, {"id": "2"}],
    )

    all_files = data_loader.list_run_files()
    assert len(all_files) == 2

    sports_files = data_loader.list_run_files("sports")
    assert len(sports_files) == 1

    history = data_loader.load_run_history()
    assert set(history["category"]) == {"sports", "technology"}
    tech_row = history[history["category"] == "technology"].iloc[0]
    assert tech_row["accounts"] == 2
    assert tech_row["tweets"] == 2


def test_load_latest_run_returns_none_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_loader, "settings", _settings(tmp_path))

    assert data_loader.load_latest_run("sports") is None


def test_load_latest_run_returns_parsed_snapshot(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(data_loader, "settings", settings)
    _write_run(tmp_path, "sports", "2026-08-01", accounts=[{"username": "espn"}], tweets=[])

    run = data_loader.load_latest_run("sports")

    assert run is not None
    assert run["category"] == "sports"


# --- data_loader: CSVs --------------------------------------------------------


def test_load_category_csv_returns_none_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_loader, "settings", _settings(tmp_path))

    assert data_loader.load_category_csv("sports") is None


def test_load_category_csv_reads_existing_file(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings.csv_output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(data_loader, "settings", settings)

    df = pd.DataFrame({"username": ["espn"], "tweet_id": ["123"], "discovery_reason": ["r"]})
    df.to_csv(settings.csv_output_dir / "sports_tweets.csv", index=False, encoding="utf-8-sig")

    loaded = data_loader.load_category_csv("sports")

    assert loaded is not None
    assert loaded.iloc[0]["username"] == "espn"
    assert loaded.iloc[0]["tweet_id"] == "123"  # kept as string, not inferred to int64


def test_load_users_csv_returns_none_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_loader, "settings", _settings(tmp_path))

    assert data_loader.load_users_csv() is None


def test_get_discovery_reasons_joins_from_csv(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings.csv_output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(data_loader, "settings", settings)

    df = pd.DataFrame(
        {
            "username": ["espn", "espn", "nasa"],
            "discovery_reason": ["Major sports outlet", "Major sports outlet", None],
        }
    )
    df.to_csv(settings.csv_output_dir / "sports_tweets.csv", index=False, encoding="utf-8-sig")

    reasons = data_loader.get_discovery_reasons("sports")

    assert reasons == {"espn": "Major sports outlet"}


def test_get_discovery_reasons_empty_when_no_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_loader, "settings", _settings(tmp_path))

    assert data_loader.get_discovery_reasons("sports") == {}


# --- ui.utils ------------------------------------------------------------------


def test_format_compact_number() -> None:
    assert format_compact_number(None) == "-"
    assert format_compact_number(500) == "500"
    assert format_compact_number(241_200_000) == "241.2M"
    assert format_compact_number(12_500) == "12.5K"


def test_validate_pipeline_params_accepts_valid_input() -> None:
    assert validate_pipeline_params("technology", 50, 20, 10) is None


def test_validate_pipeline_params_requires_category() -> None:
    assert validate_pipeline_params("  ", 50, 20, 10) is not None


def test_validate_pipeline_params_rejects_non_positive_values() -> None:
    assert validate_pipeline_params("technology", 0, 20, 10) is not None
    assert validate_pipeline_params("technology", 50, 0, 10) is not None
    assert validate_pipeline_params("technology", 50, 20, 0) is not None


def test_validate_pipeline_params_rejects_top_accounts_over_candidate_limit() -> None:
    assert validate_pipeline_params("technology", 10, 20, 10) is not None


def test_credential_status_never_leaks_secret_values() -> None:
    settings = Settings()
    settings.x_auth_token = "super-secret-token"
    settings.x_ct0 = "super-secret-ct0"
    settings.llm_provider = "groq"
    settings.groq_api_key = "super-secret-groq-key"

    status = credential_status(settings)

    assert status == {"x_auth": True, "llm": True, "cache": settings.cache_enabled}
    assert all(isinstance(v, bool) for v in status.values())
    rendered = repr(status)
    assert "super-secret-token" not in rendered
    assert "super-secret-ct0" not in rendered
    assert "super-secret-groq-key" not in rendered


def test_credential_status_false_when_not_configured() -> None:
    settings = Settings()
    settings.x_auth_token = None
    settings.x_ct0 = None
    settings.llm_provider = "groq"
    settings.groq_api_key = None

    status = credential_status(settings)

    assert status["x_auth"] is False
    assert status["llm"] is False


# --- ui.pipeline_runner ---------------------------------------------------------


class _StubTwikitClient:
    def __init__(self, **_kwargs) -> None:
        pass

    async def connect(self) -> None:
        pass


class _StubReport:
    def __init__(self, category: str) -> None:
        self.category = category


class _StubCategoryIntelligenceAgent:
    """Records how it was called and emits the same log lines the real
    pipeline does, so run_category_analysis's on_stage wiring is exercised
    end to end without any real scraping/LLM calls."""

    calls: list[dict] = []

    def __init__(self, settings, profile_scraper, tweet_scraper, llm_client) -> None:
        pass

    async def run_pipeline(
        self, category, candidate_limit=None, top_n=None, tweets_per_account=None
    ):
        _StubCategoryIntelligenceAgent.calls.append(
            {
                "category": category,
                "candidate_limit": candidate_limit,
                "top_n": top_n,
                "tweets_per_account": tweets_per_account,
            }
        )
        logger = get_logger()
        logger.info(f"Category context for {category!r}: 3 keyword(s), 2 subcategory(ies)")
        logger.info("Candidates discovered: 10 (requested up to 10)")
        logger.info("Profiles validated: 8/10")
        logger.info("Accounts selected: 5/5")
        logger.info("Tweet scrape summary:\n  Accounts requested: 5")
        logger.info("Analysis completed")
        logger.info("CSV: data/csv/example_tweets.csv")
        return _StubReport(category)


def test_run_category_analysis_invokes_pipeline_with_given_params(tmp_path, monkeypatch) -> None:
    settings = _pipeline_settings(tmp_path)

    _StubCategoryIntelligenceAgent.calls.clear()
    monkeypatch.setattr(pipeline_runner, "settings", settings)
    monkeypatch.setattr(pipeline_runner, "TwikitProfileClient", _StubTwikitClient)
    monkeypatch.setattr(
        pipeline_runner, "CategoryIntelligenceAgent", _StubCategoryIntelligenceAgent
    )

    report = pipeline_runner.run_category_analysis("sports", 50, 20, 10)

    assert report.category == "sports"
    assert _StubCategoryIntelligenceAgent.calls == [
        {"category": "sports", "candidate_limit": 50, "top_n": 20, "tweets_per_account": 10}
    ]


def test_run_category_analysis_reports_real_stage_progress(tmp_path, monkeypatch) -> None:
    """Proves progress is driven by the pipeline's own log output, not a
    simulated/fake timer: every STAGE_MARKERS key must fire, in order."""
    settings = _pipeline_settings(tmp_path)

    _StubCategoryIntelligenceAgent.calls.clear()
    monkeypatch.setattr(pipeline_runner, "settings", settings)
    monkeypatch.setattr(pipeline_runner, "TwikitProfileClient", _StubTwikitClient)
    monkeypatch.setattr(
        pipeline_runner, "CategoryIntelligenceAgent", _StubCategoryIntelligenceAgent
    )

    seen: list[str] = []
    pipeline_runner.run_category_analysis("sports", 50, 20, 10, on_stage=seen.append)

    expected_order = [key for key, _ in pipeline_runner.STAGE_MARKERS]
    assert seen == expected_order
