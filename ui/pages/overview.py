"""Overview page: onboarding when no analysis exists yet, at-a-glance
Executive Summary / Key Signals / Top Sources / Sentiment when one does.
"""

from __future__ import annotations

import streamlit as st

from app.signal_score import compute_confidence
from ui.cards import render_freshness_badge, render_signal_card, render_source_card
from ui.components import (
    render_ai_summary,
    render_empty_state,
    render_run_summary,
    render_sentiment_section,
)
from ui.utils import CATEGORY_QUICK_PICKS, count_topic_mentions, distinct_authors_for_topic


def render_overview_page() -> None:
    st.title("X Intelligence")
    st.caption("Discover, validate, rank and analyze influential X accounts by category.")

    report = st.session_state.get("current_report")
    if not report:
        st.markdown(
            "Discover influential accounts. Validate them using real X data. "
            "Analyze current conversations."
        )
        render_empty_state(
            "No analysis has been run yet. Choose a category to begin, or run a custom analysis."
        )

        st.markdown("##### Choose a category to begin")
        cols = st.columns(len(CATEGORY_QUICK_PICKS))
        for col, category in zip(cols, CATEGORY_QUICK_PICKS, strict=True):
            if col.button(category, key=f"quick-pick-{category}"):
                st.session_state["prefill_category"] = category
                st.session_state["nav_target"] = "New Analysis"
                st.rerun()

        if st.button("Custom Category / Analyze Now", type="primary"):
            st.session_state["nav_target"] = "New Analysis"
            st.rerun()
        return

    render_freshness_badge(report.get("scraped_at"), report.get("category"))
    render_run_summary(report)
    st.markdown("---")

    analysis = report.get("analysis", {}) or {}
    summary = analysis.get("summary", "")
    if summary:
        st.markdown("#### Executive Summary")
        render_ai_summary(summary)
        st.markdown("---")

    tweets = report.get("tweets", [])
    accounts = sorted(report.get("accounts", []), key=lambda a: a.get("rank", 999))
    topics = (analysis.get("trending_topics", []) or [])[:3]

    col_signals, col_sources, col_sentiment = st.columns(3)
    with col_signals:
        st.markdown("#### Key Signals")
        if not topics:
            render_empty_state("No trending topics identified for this run.")
        for topic in topics:
            mentions = count_topic_mentions(tweets, topic)
            authors = distinct_authors_for_topic(tweets, topic)
            confidence_label, _ = compute_confidence(mentions, authors)
            render_signal_card(topic, mentions, confidence_label=confidence_label, compact=True)

    with col_sources:
        st.markdown("#### Top Sources")
        if not accounts:
            render_empty_state("No ranked accounts available.")
        for account in accounts[:3]:
            render_source_card(account, compact=True)
        if st.button("View all sources"):
            st.session_state["nav_target"] = "Sources"
            st.rerun()

    with col_sentiment:
        st.markdown("#### Sentiment")
        render_sentiment_section(
            analysis.get("sentiment", {}),
            report.get("tweet_statistics", {}).get("tweets_collected", 0),
        )
