"""Streamlit entry point for the X Category Intelligence dashboard.

Run with:
    streamlit run streamlit_app.py

This is a presentation layer only - it calls the existing
`CategoryIntelligenceAgent` pipeline via `ui.pipeline_runner` and reads the
existing on-disk storage via `ui.data_loader`. It never reimplements
discovery, validation, ranking, scraping, retry/rate-limit handling, or
analysis, and it never mutates `main.py` or anything under `app/`.

The pipeline is only ever invoked from the explicit "Run Analysis" button on
the Analyze Category page - every other page reads exclusively from
`st.session_state`/disk, so simply browsing Dashboard/Accounts/Tweets/
Analytics/Run History never triggers a new X request.
"""

from __future__ import annotations

import streamlit as st

from app.config import settings
from app.exceptions import AuthenticationError, LLMError, RateLimitError, ScraperError
from ui.charts import (
    create_engagement_chart,
    create_followers_chart,
    create_ranking_chart,
    create_relevance_score_chart,
    create_sentiment_chart,
    create_tweet_distribution_chart,
)
from ui.components import (
    render_account_card,
    render_account_table,
    render_ai_summary,
    render_empty_state,
    render_error_message,
    render_pipeline_status,
    render_run_summary,
    render_sentiment_section,
    render_trending_topics,
    render_tweet_card,
)
from ui.data_loader import (
    get_discovery_reasons,
    list_run_files,
    load_run_history,
    load_run_json,
)
from ui.pipeline_runner import run_category_analysis
from ui.styles import inject_custom_css
from ui.utils import credential_status, validate_pipeline_params

NAV_OPTIONS = [
    "Dashboard",
    "Analyze Category",
    "Accounts",
    "Tweets",
    "Analytics",
    "Run History",
    "Settings",
]


def _init_session_state() -> None:
    st.session_state.setdefault("current_category", None)
    st.session_state.setdefault("current_report", None)
    st.session_state.setdefault("current_run_source", None)
    st.session_state.setdefault("pipeline_running", False)
    st.session_state.setdefault("pipeline_stages_done", set())


def _status_row(label: str, ok: bool) -> None:
    dot_class = "xi-status-ok" if ok else "xi-status-bad"
    text = "Configured" if ok else "Not configured"
    st.markdown(
        f'<div class="xi-status-row"><span class="xi-status-dot {dot_class}"></span>'
        f"{label}: {text}</div>",
        unsafe_allow_html=True,
    )


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<p class="xi-brand">X INTELLIGENCE</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="xi-brand-subtitle">Category Intelligence Platform</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio("Navigation", NAV_OPTIONS, key="nav_page", label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**System Status**")
        # Derived from actual configuration (Settings.has_cookie_credentials /
        # Settings.has_llm) - never hardcoded, never shows the secrets themselves.
        status = credential_status(settings)
        _status_row("X Session", status["x_auth"])
        _status_row(f"LLM Provider ({settings.llm_provider})", status["llm"])
        _status_row("Cache", status["cache"])
    return page


def _report_fallback_dict(report) -> dict:
    """Defensive fallback if the run snapshot somehow wasn't found on disk
    right after a live run (save_category_run always writes it inside
    run_pipeline, so this should not normally trigger)."""
    return {
        "category": report.category,
        "accounts": [account.model_dump(mode="json") for account in report.accounts],
        "tweets": [],
        "tweet_statistics": report.tweet_statistics.model_dump(mode="json"),
        "analysis": report.analysis.model_dump(mode="json"),
        "errors": report.errors,
    }


def _render_downloads(category: str) -> None:
    st.markdown("#### Downloads")
    files = list_run_files(category)
    cols = st.columns(3)

    if files:
        cols[0].download_button(
            "Download Run JSON",
            data=files[0].read_bytes(),
            file_name=files[0].name,
            mime="application/json",
            key=f"dl-json-{category}",
        )

    csv_path = settings.csv_output_dir / f"{category}_tweets.csv"
    if csv_path.exists():
        cols[1].download_button(
            "Download Consolidated CSV",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
            key=f"dl-csv-{category}",
        )

    users_path = settings.csv_output_dir / "users.csv"
    if users_path.exists():
        cols[2].download_button(
            "Download Users CSV",
            data=users_path.read_bytes(),
            file_name="users.csv",
            mime="text/csv",
            key=f"dl-users-{category}",
        )


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def _render_dashboard() -> None:
    st.title("X Category Intelligence")
    st.caption("Discover, validate, rank and analyze influential X accounts by category.")

    report = st.session_state.get("current_report")
    if not report:
        st.markdown(
            "Discover influential accounts. Validate them using real X data. "
            "Analyze current conversations."
        )
        render_empty_state(
            "No analysis has been run yet. Run a category analysis to see intelligence results."
        )
        if st.button("Analyze a Category", type="primary"):
            st.session_state["nav_page"] = "Analyze Category"
            st.rerun()
        return

    render_run_summary(report)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top Accounts")
        top = sorted(report.get("accounts", []), key=lambda a: a.get("rank", 999))[:5]
        if not top:
            render_empty_state("No ranked accounts available.")
        for account in top:
            st.write(f"**@{account['username']}** — {account['ranking_score']:.1f}")
    with col2:
        analysis = report.get("analysis", {}) or {}
        render_sentiment_section(
            analysis.get("sentiment", {}),
            report.get("tweet_statistics", {}).get("tweets_collected", 0),
        )

    render_trending_topics((report.get("analysis", {}) or {}).get("trending_topics", []))
    render_ai_summary((report.get("analysis", {}) or {}).get("summary", ""))


def _execute_pipeline(
    category: str, candidate_limit: int, top_accounts: int, tweets_per_account: int
) -> None:
    st.session_state["pipeline_running"] = True
    st.session_state["pipeline_stages_done"] = set()
    status_placeholder = st.empty()

    def on_stage(stage_key: str) -> None:
        st.session_state["pipeline_stages_done"].add(stage_key)
        with status_placeholder.container():
            render_pipeline_status(st.session_state["pipeline_stages_done"])

    with status_placeholder.container():
        render_pipeline_status(st.session_state["pipeline_stages_done"])

    try:
        with st.spinner(f"Running category intelligence pipeline for '{category}'..."):
            report = run_category_analysis(
                category, candidate_limit, top_accounts, tweets_per_account, on_stage=on_stage
            )

        files = list_run_files(report.category)
        run_data = load_run_json(files[0]) if files else _report_fallback_dict(report)

        st.session_state["current_report"] = run_data
        st.session_state["current_category"] = report.category
        st.session_state["current_run_source"] = "live"
    except AuthenticationError as exc:
        render_error_message("auth", str(exc))
    except LLMError as exc:
        render_error_message("llm", str(exc))
    except RateLimitError as exc:
        render_error_message("rate_limit", str(exc))
    except ScraperError as exc:
        render_error_message("generic", str(exc))
    finally:
        st.session_state["pipeline_running"] = False


def _render_analyze_page() -> None:
    st.title("Analyze Category")
    st.markdown("---")

    category = st.text_input(
        "Category", placeholder="e.g. Technology, Healthcare, Indian Agriculture"
    )
    col1, col2, col3 = st.columns(3)
    candidate_limit = col1.number_input(
        "Candidate limit", min_value=1, value=settings.category_candidate_limit, step=5
    )
    top_accounts = col2.number_input(
        "Top accounts", min_value=1, value=settings.top_accounts_limit, step=1
    )
    tweets_per_account = col3.number_input(
        "Tweets per account", min_value=1, value=settings.tweets_per_account, step=1
    )

    run_clicked = st.button(
        "🚀 Run Analysis", type="primary", disabled=st.session_state["pipeline_running"]
    )

    if run_clicked:
        error = validate_pipeline_params(
            category, candidate_limit, top_accounts, tweets_per_account
        )
        if error:
            st.error(error)
        else:
            _execute_pipeline(
                category.strip(), int(candidate_limit), int(top_accounts), int(tweets_per_account)
            )

    report = st.session_state.get("current_report")
    if report and st.session_state.get("current_run_source") == "live":
        st.markdown("---")
        render_run_summary(report)
        _render_downloads(report.get("category", ""))


def _render_accounts_page() -> None:
    st.title("Accounts")
    report = st.session_state.get("current_report")
    if not report:
        render_empty_state(
            "No analysis has been run yet. Run a category analysis to see ranked accounts."
        )
        return

    accounts = report.get("accounts", [])
    render_account_table(accounts)

    if not accounts:
        return

    st.markdown("---")
    st.markdown("#### Account Details")
    usernames = [account["username"] for account in accounts]
    selected = st.selectbox("Select an account", usernames)
    account = next((a for a in accounts if a["username"] == selected), None)
    if account is None:
        return

    reasons = get_discovery_reasons(report.get("category", ""))
    account = {**account, "discovery_reason": reasons.get(account["username"], "")}
    render_account_card(account)


def _render_tweets_page() -> None:
    st.title("Tweets")
    report = st.session_state.get("current_report")
    if not report:
        render_empty_state(
            "No analysis has been run yet. Run a category analysis to explore tweets."
        )
        return

    tweets = report.get("tweets", [])
    if not tweets:
        render_empty_state("No tweets available for this selection.")
        return

    usernames = ["All"] + sorted({t["username"] for t in tweets if t.get("username")})
    col1, col2 = st.columns([1, 2])
    account_filter = col1.selectbox("Account", usernames)
    search = col2.text_input("Search", placeholder="Search tweets...")

    filtered = tweets
    if account_filter != "All":
        filtered = [t for t in filtered if t.get("username") == account_filter]
    if search:
        needle = search.lower()
        filtered = [t for t in filtered if needle in (t.get("text") or "").lower()]

    st.caption(f"{len(filtered)} of {len(tweets)} tweet(s) shown.")
    if not filtered:
        render_empty_state("No tweets match this filter.")
        return

    for tweet in filtered:
        render_tweet_card(tweet)


def _render_analytics_page() -> None:
    st.title("Analytics")
    report = st.session_state.get("current_report")
    if not report:
        render_empty_state(
            "No analysis has been run yet. Run a category analysis to see analytics."
        )
        return

    accounts = report.get("accounts", [])
    tweets = report.get("tweets", [])
    analysis = report.get("analysis", {}) or {}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Ranking Score")
        st.plotly_chart(create_ranking_chart(accounts), width="stretch")
    with col2:
        st.markdown("#### Followers")
        st.plotly_chart(create_followers_chart(accounts), width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Relevance vs Score")
        st.plotly_chart(create_relevance_score_chart(accounts), width="stretch")
    with col4:
        st.markdown("#### Sentiment")
        st.plotly_chart(create_sentiment_chart(analysis.get("sentiment", {})), width="stretch")

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("#### Engagement by Account")
        st.plotly_chart(create_engagement_chart(tweets), width="stretch")
    with col6:
        st.markdown("#### Tweet Distribution")
        st.plotly_chart(create_tweet_distribution_chart(tweets), width="stretch")

    render_trending_topics(analysis.get("trending_topics", []))


def _render_history_page() -> None:
    st.title("Run History")

    history_df = load_run_history()
    if history_df.empty:
        render_empty_state("No previous runs found.")
        return

    st.dataframe(history_df.drop(columns=["path"]), width="stretch", hide_index=True)

    options = [f"{row['category']} — {row['date']}" for _, row in history_df.iterrows()]
    choice = st.selectbox("Select a run to view", options)
    idx = options.index(choice)
    row = history_df.iloc[idx]

    if st.button("Load selected run"):
        run_data = load_run_json(row["path"])
        st.session_state["current_report"] = run_data
        st.session_state["current_category"] = run_data.get("category")
        st.session_state["current_run_source"] = "history"
        st.success(f"Loaded '{choice}' from disk - no new X requests were made.")

    st.markdown("---")
    _render_downloads(row["category"])


def _render_settings_page() -> None:
    st.title("Settings")
    st.caption("Read-only view of the active configuration. Secrets are never displayed.")

    status = credential_status(settings)

    st.markdown("#### Credentials")
    col1, col2 = st.columns(2)
    col1.metric("X Authentication", "Configured" if status["x_auth"] else "Not configured")
    col2.metric(
        f"{settings.llm_provider.title()} API Key",
        "Configured" if status["llm"] else "Not configured",
    )

    st.markdown("#### LLM")
    model = settings.groq_model if settings.llm_provider == "groq" else settings.gemini_model
    st.write(f"**Provider:** {settings.llm_provider}")
    st.write(f"**Model:** {model}")

    st.markdown("#### Concurrency")
    st.write(f"**Profile concurrency:** {settings.concurrency_limit}")
    st.write(f"**Tweet concurrency:** {settings.tweet_scrape_concurrency}")

    st.markdown("#### Retry & Rate Limiting")
    st.write(f"**Generic retry base:** {settings.backoff_base_seconds} seconds")
    st.write(f"**Generic retry max:** {settings.backoff_max_seconds} seconds")
    st.write(f"**Rate limit retry base:** {settings.rate_limit_base_seconds} seconds")
    st.write(f"**Rate limit maximum:** {settings.rate_limit_max_seconds} seconds")

    st.markdown("#### Cache")
    st.write(f"**Enabled:** {'Yes' if settings.cache_enabled else 'No'}")
    st.write(f"**TTL:** {settings.cache_ttl_seconds} seconds")


def main() -> None:
    st.set_page_config(page_title="X Intelligence", page_icon="📊", layout="wide")
    inject_custom_css()
    _init_session_state()

    page = _render_sidebar()

    pages = {
        "Dashboard": _render_dashboard,
        "Analyze Category": _render_analyze_page,
        "Accounts": _render_accounts_page,
        "Tweets": _render_tweets_page,
        "Analytics": _render_analytics_page,
        "Run History": _render_history_page,
        "Settings": _render_settings_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()
