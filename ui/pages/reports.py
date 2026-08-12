"""Reports page: historical runs (was "Run History"), plus Compare and
Story Opportunities tabs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.exceptions import RAGError
from app.report_compare import compare_reports
from app.story_opportunities import derive_story_opportunities
from ui import ask_runner
from ui.cards import render_brief_view, render_story_opportunity_card
from ui.components import render_empty_state, render_error_message
from ui.data_loader import load_run_history, load_run_json
from ui.pages._shared import render_downloads


def _render_all_runs_tab() -> None:
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
    render_downloads(row["category"])


def _render_compare_tab() -> None:
    st.caption(
        "Compare the SAME category across two different dates - e.g. Technology yesterday "
        "vs. today - to see what changed and why."
    )

    all_history = load_run_history()
    if all_history.empty:
        render_empty_state("No previous runs found. Run at least two analyses to compare them.")
        return

    categories = sorted(all_history["category"].unique())
    category = st.selectbox("Category", categories)

    category_history = load_run_history(category)
    if len(category_history) < 2:
        render_empty_state(
            f"Only one run exists for {category!r} so far - run this category's analysis "
            "again on a different date to enable comparison."
        )
        return

    dates = list(category_history["date"])
    col1, col2 = st.columns(2)
    older_date = col1.selectbox("Older date", dates, index=len(dates) - 1)
    newer_date = col2.selectbox("Newer date", dates, index=0)

    if older_date == newer_date:
        st.info("Pick two different dates to compare.")
        return

    older_row = category_history[category_history["date"] == older_date].iloc[0]
    newer_row = category_history[category_history["date"] == newer_date].iloc[0]
    older_run = load_run_json(older_row["path"])
    newer_run = load_run_json(newer_row["path"])

    result = compare_reports(older_run, newer_run)
    st.markdown(f"#### {result.older_label}  →  {result.newer_label}")
    st.caption(f"Analysis window: {result.older_window_label}   vs.   {result.newer_window_label}")

    metric_cols = st.columns(2)
    metric_cols[0].metric(
        "Tweets collected",
        newer_run.get("tweet_statistics", {}).get("tweets_collected", 0),
        delta=result.tweets_collected_delta,
    )
    metric_cols[1].metric(
        "Accounts succeeded",
        newer_run.get("tweet_statistics", {}).get("accounts_processed", 0),
        delta=result.accounts_processed_delta,
    )

    st.markdown("##### Sentiment shift")
    sentiment_cols = st.columns(3)
    for col, key in zip(sentiment_cols, ("positive", "neutral", "negative"), strict=True):
        col.metric(
            key.title(),
            f"{newer_run.get('analysis', {}).get('sentiment', {}).get(key, 0):.0f}%",
            delta=f"{result.sentiment_delta.get(key, 0):+.0f}pp",
        )

    st.markdown("##### Topic Changes")
    col_added, col_persisted, col_removed = st.columns(3)
    with col_added:
        st.caption("New Signals")
        if result.topics_added:
            for topic in result.topics_added:
                st.markdown(f"- {topic}")
        else:
            st.caption("None")
    with col_persisted:
        st.caption("Persistent Signals")
        if result.topics_persisted:
            for topic in result.topics_persisted:
                st.markdown(f"- {topic}")
        else:
            st.caption("None")
    with col_removed:
        st.caption("Declining Signals")
        if result.topics_removed:
            for topic in result.topics_removed:
                st.markdown(f"- {topic}")
        else:
            st.caption("None")

    st.markdown("##### Rising & falling accounts")
    if result.movers:
        movers_df = pd.DataFrame(
            [
                {
                    "Account": f"@{m.username}",
                    "Rank change": m.rank_delta,
                    "Score change": round(m.score_delta, 1),
                }
                for m in result.movers[:10]
            ]
        )
        st.dataframe(movers_df, width="stretch", hide_index=True)
    else:
        render_empty_state("No accounts appear in both runs to compare.")

    if result.accounts_new or result.accounts_dropped:
        st.markdown("##### Top Source Changes")
        st.caption(
            "These reflect changes in the ranked source set between runs; they do not "
            "necessarily mean an account stopped covering this topic."
        )
        col_new, col_dropped = st.columns(2)
        col_new.markdown("**New in Top Sources**")
        col_new.write(", ".join(f"@{u}" for u in result.accounts_new) or "None")
        col_dropped.markdown("**Not in Newer Top Sources**")
        col_dropped.write(", ".join(f"@{u}" for u in result.accounts_dropped) or "None")


def _render_brief_for(opportunity, report: dict) -> None:
    if st.button("← Back to Story Opportunities"):
        st.session_state["selected_opportunity_title"] = None
        st.rerun()

    category = report.get("category")
    # The current report's own tweet ids - the hard boundary that keeps
    # story evidence scoped to this exact analysis run (its category AND
    # its time window), never an older/global slice of the RAG index.
    tweet_ids = {t.get("id") for t in report.get("tweets", []) if t.get("id")}

    try:
        with st.spinner("Generating story brief..."):
            brief = ask_runner.generate_brief(opportunity, category=category, tweet_ids=tweet_ids)
    except RAGError as exc:
        render_error_message("rag_empty", str(exc))
        return

    render_brief_view(
        {
            "headline": brief.headline,
            "why_it_matters": brief.why_it_matters,
            "note": brief.note,
            "observed_facts": brief.observed_facts,
            "ai_interpretation": brief.ai_interpretation,
            "supporting_posts": [
                {
                    "username": c.username,
                    "text": c.text_excerpt,
                    "url": c.url,
                    "created_at": c.created_at,
                }
                for c in brief.supporting_posts
            ],
            "accounts": [{"username": u} for u in brief.supporting_accounts],
            "investigation_questions": brief.suggested_investigation_questions,
        }
    )


def _render_story_opportunities_tab() -> None:
    report = st.session_state.get("current_report")
    if not report:
        render_empty_state(
            "No analysis has been run yet. Run or load an analysis to see story opportunities."
        )
        return

    opportunities = derive_story_opportunities(report)
    st.session_state.setdefault("selected_opportunity_title", None)

    selected_title = st.session_state.get("selected_opportunity_title")
    if selected_title:
        opportunity = next((o for o in opportunities if o.title == selected_title), None)
        if opportunity is None:
            st.session_state["selected_opportunity_title"] = None
            st.rerun()
        _render_brief_for(opportunity, report)
        return

    if not opportunities:
        render_empty_state(
            "No signals in this run currently clear the confidence bar for a story "
            "opportunity - try a run with more collected tweets/accounts."
        )
        return

    for opportunity in opportunities:

        def _on_generate(title=opportunity.title) -> None:
            st.session_state["selected_opportunity_title"] = title
            st.rerun()

        render_story_opportunity_card(
            opportunity.title,
            opportunity.why_it_matters,
            opportunity.confidence_label,
            _on_generate,
        )


def render_reports_page() -> None:
    st.title("Reports")

    tab_all_runs, tab_compare, tab_story = st.tabs(["All Runs", "Compare", "Story Opportunities"])
    with tab_all_runs:
        _render_all_runs_tab()
    with tab_compare:
        _render_compare_tab()
    with tab_story:
        _render_story_opportunities_tab()
