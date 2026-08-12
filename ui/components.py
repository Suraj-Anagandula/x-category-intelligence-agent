"""Reusable Streamlit render functions.

Keeps `streamlit_app.py` a thin router: every widget/section is built here
from plain dicts/lists shaped like the existing `data/tweets/<category>/<date>.json`
run snapshot (see `app/storage.py`) - no data is invented, only formatted.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.utils import split_errors_by_stage
from ui.pipeline_runner import PIPELINE_STAGES
from ui.utils import format_compact_number

_ERROR_MESSAGES: dict[str, tuple[str, str]] = {
    "auth": (
        "X authentication is unavailable.",
        "Please configure X_AUTH_TOKEN and X_CT0 in your .env file.",
    ),
    "llm": (
        "LLM analysis failed.",
        "Check your configured LLM provider and API key.",
    ),
    "rate_limit": (
        "X rate limit encountered.",
        "The pipeline is waiting according to the configured rate-limit reset/backoff policy.",
    ),
    "generic": (
        "The category pipeline failed.",
        "",
    ),
    "rag_empty": (
        "No indexed intelligence available yet.",
        "Run or load an analysis first, then build the index before asking questions.",
    ),
    "rag_insufficient": (
        "The retrieved evidence doesn't clearly answer this question.",
        "Try a more specific question, or run a new analysis to collect more posts.",
    ),
}


def render_empty_state(message: str) -> None:
    st.info(message)


def render_error_message(kind: str, detail: str = "") -> None:
    """Canned, user-friendly error messages - never displays credentials;
    `detail` is the real exception text, shown only inside a collapsed
    expander for troubleshooting (matches the CLI, which already logs and
    prints the same exception text - see main.py's ScraperError handling)."""
    headline, hint = _ERROR_MESSAGES.get(kind, _ERROR_MESSAGES["generic"])
    st.error(headline)
    if hint:
        st.caption(hint)
    if detail:
        with st.expander("Technical detail"):
            st.code(detail)


def render_metric_cards(stats: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Accounts Discovered", stats.get("discovered", "-"))
    cols[1].metric("Profiles Validated", stats.get("validated", "-"))
    cols[2].metric("Accounts Selected", stats.get("selected", "-"))
    cols[3].metric("Tweets Collected", stats.get("tweets_collected", "-"))


def render_pipeline_status(done_stages: set[str]) -> None:
    """The 7-stage checklist - a stage only ever shows complete once its key
    is actually in `done_stages`, which the caller only adds to when the
    pipeline's own logging reports that stage done (see ui/pipeline_runner.py)."""
    for key, label in PIPELINE_STAGES:
        icon = "✅" if key in done_stages else "⏳"
        st.write(f"{icon} {label}")


def render_run_summary(run_data: dict) -> None:
    """The post-run (or historical-run) headline: success/partial-failure
    banner plus the real counts - never rounds a partial run up to "complete".

    "validation"-stage errors (a discovered candidate handle that doesn't
    exist or is protected, rejected before ranking/selection) are a normal
    part of dynamic discovery, not a failure of the run's final selected
    accounts - see `app.utils.split_errors_by_stage`. Only "tweets"-stage
    errors (a selected account's own tweet collection failing) affect the
    success/partial-failure banner or count toward "failed during the
    pipeline run".
    """
    stats = run_data.get("tweet_statistics", {}) or {}
    errors = run_data.get("errors", []) or []
    validation_errors, pipeline_errors = split_errors_by_stage(errors)

    if pipeline_errors:
        st.warning("Analysis completed with partial failures.")
    else:
        st.success("Analysis completed successfully.")

    st.markdown(f"**Category:** {run_data.get('category', '-')}")

    window = run_data.get("time_window") or {}
    window_mode = window.get("mode", "latest")
    if window_mode != "latest" and window.get("start") and window.get("end"):
        start_label = window["start"].split("T")[0]
        end_label = window["end"].split("T")[0]
        st.caption(
            f"**Analysis Window:** {start_label} → {end_label}  ·  "
            f"{window.get('posts_in_window', 0)} of {window.get('posts_fetched', 0)} "
            "fetched post(s) were inside this window"
        )

    cols = st.columns(6)
    cols[0].metric("Accounts Selected", len(run_data.get("accounts", [])))
    cols[1].metric("Succeeded", stats.get("accounts_processed", 0))
    cols[2].metric("Rate Limited", stats.get("accounts_rate_limited", 0))
    cols[3].metric("Pipeline Failures", stats.get("accounts_failed", 0))
    cols[4].metric("Candidate Validation Issues", len(validation_errors))
    cols[5].metric("Tweets Collected", stats.get("tweets_collected", 0))

    if pipeline_errors:
        st.warning(
            f"⚠ {len(pipeline_errors)} of the selected account(s) failed during tweet "
            "collection (the pipeline run itself)."
        )
        with st.expander("Pipeline failure detail"):
            st.dataframe(pd.DataFrame(pipeline_errors), width="stretch", hide_index=True)

    if validation_errors:
        st.caption(
            f"{len(validation_errors)} candidate account(s) discovered for this category were "
            "rejected during validation (e.g. not found, protected) before ranking - they were "
            "never part of the final selected accounts above."
        )
        with st.expander("Candidate validation detail"):
            st.dataframe(pd.DataFrame(validation_errors), width="stretch", hide_index=True)


def render_account_table(accounts: list[dict]) -> None:
    if not accounts:
        render_empty_state("No ranked accounts available.")
        return

    df = pd.DataFrame(accounts)
    display = pd.DataFrame(
        {
            "Rank": df["rank"],
            "Username": "@" + df["username"].astype(str),
            "Display Name": df.get("display_name", pd.Series([""] * len(df))).fillna(""),
            "Followers": df["followers"].apply(format_compact_number),
            "Relevance": df["category_relevance"].round(1),
            "Engagement": df["engagement_score"].round(1),
            "Activity": df["activity_score"].round(1),
            "Audience": df["audience_score"].round(1),
            "Score": df["ranking_score"].round(1),
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)


def render_tweet_card(tweet: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**@{tweet.get('username') or 'unknown'}**")
        st.write(tweet.get("text", ""))

        cols = st.columns(4)
        cols[0].caption(f"❤️ {tweet.get('like_count') or 0}")
        cols[1].caption(f"🔁 {tweet.get('retweet_count') or 0}")
        cols[2].caption(f"💬 {tweet.get('reply_count') or 0}")
        views = tweet.get("view_count")
        cols[3].caption(f"👁 {views if views is not None else '-'}")

        hashtags = tweet.get("hashtags") or []
        if hashtags:
            st.caption(" ".join(f"#{tag}" for tag in hashtags))

        url = tweet.get("url")
        if url:
            st.caption(url)


def render_ai_summary(summary: str) -> None:
    if not summary:
        render_empty_state("No AI summary available for this run.")
        return
    st.markdown("#### AI Insight")
    st.info(summary)


def render_trending_topics(topics: list[str]) -> None:
    if not topics:
        render_empty_state("No trending topics identified for this run.")
        return
    st.markdown("#### Trending Topics")
    chips = " ".join(f'<span class="xi-chip">{topic}</span>' for topic in topics)
    st.markdown(chips, unsafe_allow_html=True)


def render_sentiment_section(sentiment: dict, total_tweets: int) -> None:
    if not sentiment or not any(sentiment.values()):
        render_empty_state("Sentiment data unavailable for this run.")
        return

    st.markdown("#### Sentiment Overview")
    cols = st.columns(3)
    cols[0].metric("Positive", f"{sentiment.get('positive', 0):.0f}%")
    cols[1].metric("Neutral", f"{sentiment.get('neutral', 0):.0f}%")
    cols[2].metric("Negative", f"{sentiment.get('negative', 0):.0f}%")
    st.caption(f"Based on {total_tweets} collected tweet(s).")
