"""Intelligence page: evidence browsing (topic-grouped) plus Ask
Intelligence - retrieval-augmented Q&A over the currently loaded report's
posts. Scoped by default to just this report's own tweets (never an
older/global slice of the category's index); an explicit "Include
historical posts" toggle opts into searching all indexed history for the
same category.
"""

from __future__ import annotations

import streamlit as st

from app.exceptions import RAGError
from ui import ask_runner
from ui.cards import render_evidence_card
from ui.components import render_empty_state, render_error_message, render_tweet_card
from ui.utils import group_tweets_by_topic


def _render_browse_evidence_tab(report: dict) -> None:
    tweets = report.get("tweets", [])
    if not tweets:
        render_empty_state("No tweets available for this selection.")
        return

    analysis = report.get("analysis", {}) or {}
    topics = analysis.get("trending_topics", []) or []

    usernames = ["All"] + sorted({t["username"] for t in tweets if t.get("username")})
    col1, col2, col3 = st.columns([1, 1, 2])
    account_filter = col1.selectbox("Account", usernames)
    topic_filter = col2.selectbox("Topic", ["All Topics", *topics])
    search = col3.text_input("Search", placeholder="Search tweets...")

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

    grouped = group_tweets_by_topic(filtered, topics)
    if topic_filter != "All Topics":
        grouped = {topic_filter: grouped.get(topic_filter, [])}

    for topic, topic_tweets in grouped.items():
        if not topic_tweets:
            continue
        st.markdown(f"#### {topic.upper()}")
        st.caption(f"{len(topic_tweets)} relevant post(s)")
        for tweet in topic_tweets:
            render_tweet_card(tweet)


def _render_ask_intelligence_tab(report: dict) -> None:
    category = report.get("category", "")
    st.session_state.setdefault("ask_history", [])

    try:
        indexed = ask_runner.index_size()
    except Exception:  # noqa: BLE001 - status check must never break the page
        indexed = 0

    col_status, col_button = st.columns([3, 1])
    col_status.caption(f"{indexed} post(s) indexed across all categories.")
    if col_button.button("Build / Refresh Index"):
        try:
            with st.spinner("Indexing collected posts..."):
                results = ask_runner.rebuild_index()
            total = sum(results.values())
            st.success(f"Indexed {total} post(s) across {len(results)} run(s).")
        except RAGError as exc:
            render_error_message("rag_empty", str(exc))

    if indexed == 0:
        render_empty_state(
            'No indexed intelligence available yet. Click "Build / Refresh Index" above '
            "to make this run's (and any past runs') posts searchable."
        )
        return

    include_historical = st.checkbox(
        "Include historical posts from earlier runs",
        value=False,
        help=(
            "Off (default): answers are scoped to only this loaded report's own posts. "
            "On: also searches previously indexed runs of this category - use this when "
            "you explicitly want historical context beyond the current analysis."
        ),
    )

    for turn in st.session_state["ask_history"]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn.get("insufficient"):
                st.caption("⚠ Insufficient evidence")
            for citation in turn.get("citations", []):
                render_evidence_card(citation, compact=True)

    question = st.chat_input(f"Ask a question about {category or 'this category'}...")
    if not question:
        return

    st.session_state["ask_history"].append({"role": "user", "content": question})
    tweet_ids = (
        None
        if include_historical
        else {t.get("id") for t in report.get("tweets", []) if t.get("id")}
    )
    try:
        answer = ask_runner.ask(question, category=category or None, tweet_ids=tweet_ids)
    except RAGError as exc:
        render_error_message("rag_empty", str(exc))
        return

    citation_dicts = [
        {
            "username": c.username,
            "text": c.text_excerpt,
            "url": c.url,
            "created_at": c.created_at,
        }
        for c in answer.citations
    ]
    st.session_state["ask_history"].append(
        {
            "role": "assistant",
            "content": answer.answer,
            "citations": citation_dicts,
            "insufficient": answer.insufficient_evidence,
        }
    )
    st.rerun()


def render_intelligence_page() -> None:
    st.title("Intelligence")
    report = st.session_state.get("current_report")
    if not report:
        render_empty_state(
            "No analysis has been run yet. Run a category analysis to explore its evidence."
        )
        return

    tab_evidence, tab_ask = st.tabs(["Browse Evidence", "Ask Intelligence"])
    with tab_evidence:
        _render_browse_evidence_tab(report)
    with tab_ask:
        _render_ask_intelligence_tab(report)
