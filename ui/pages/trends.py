"""Trends page: the same 6 Plotly charts as the old "Analytics" page,
reorganized under business-question subheaders instead of a flat grid.
"""

from __future__ import annotations

import streamlit as st

from ui.charts import (
    create_engagement_chart,
    create_followers_chart,
    create_ranking_chart,
    create_relevance_score_chart,
    create_sentiment_chart,
    create_tweet_distribution_chart,
)
from ui.components import render_empty_state, render_trending_topics


def render_trends_page() -> None:
    st.title("Trends")
    report = st.session_state.get("current_report")
    if not report:
        render_empty_state("No analysis has been run yet. Run a category analysis to see trends.")
        return

    accounts = report.get("accounts", [])
    tweets = report.get("tweets", [])
    analysis = report.get("analysis", {}) or {}

    st.markdown("#### Who has the most influence?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Ranking Score")
        st.plotly_chart(create_ranking_chart(accounts), width="stretch")
    with col2:
        st.markdown("##### Followers")
        st.plotly_chart(create_followers_chart(accounts), width="stretch")

    st.markdown("#### Is relevance matched by engagement?")
    st.plotly_chart(create_relevance_score_chart(accounts), width="stretch")

    st.markdown("---")

    st.markdown("#### What's the mood?")
    st.plotly_chart(create_sentiment_chart(analysis.get("sentiment", {})), width="stretch")

    st.markdown("#### Where's the attention concentrated?")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Engagement by Account")
        st.plotly_chart(create_engagement_chart(tweets), width="stretch")
    with col4:
        st.markdown("##### Tweet Distribution")
        st.plotly_chart(create_tweet_distribution_chart(tweets), width="stretch")

    render_trending_topics(analysis.get("trending_topics", []))
