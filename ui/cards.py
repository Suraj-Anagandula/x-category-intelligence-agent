"""Shared card-style render functions for the intelligence UX (signals,
sources, evidence, freshness). Kept separate from `ui/components.py`
(existing generic helpers) since this inventory is large enough on its own
- splits by concern, the same way the codebase already separates
`charts.py`/`data_loader.py`/`components.py`.

Every function here takes plain dicts/lists shaped exactly like the
existing `data/tweets/<category>/<date>.json` run snapshot - no new data
shape is invented, so a historical run loaded via Reports renders
identically to a fresh live run.
"""

from __future__ import annotations

import streamlit as st

from ui.utils import format_compact_number, freshness_state

_SIGNAL_BADGE_CLASS = {
    "High": "xi-signal-badge-high",
    "Medium": "xi-signal-badge-medium",
    "Low": "xi-signal-badge-low",
}

_FRESHNESS_DOT_CLASS = {
    "fresh": "xi-status-ok",
    "aging": "xi-status-warn",
    "stale": "xi-status-bad",
    "unknown": "xi-status-bad",
}


def _confidence_meter(score: float) -> None:
    pct = max(0.0, min(100.0, score))
    st.markdown(
        f'<div class="xi-confidence-meter">'
        f'<div class="xi-confidence-meter-fill" style="width:{pct:.0f}%"></div></div>',
        unsafe_allow_html=True,
    )


def render_source_card(account: dict, discovery_reason: str = "", compact: bool = False) -> None:
    """Relevance/engagement/activity sub-scores + follower count + "why this
    source matters" + a "View Profile" link, built entirely from real
    `RankedAccount` fields already on `account`."""
    username = account.get("username", "unknown")
    with st.container(border=True):
        st.markdown('<p class="xi-card-title">Source</p>', unsafe_allow_html=True)
        st.markdown(f"**@{username}**")

        if compact:
            st.caption(
                f"Score {account.get('ranking_score', 0):.1f} · "
                f"{format_compact_number(account.get('followers'))} followers"
            )
        else:
            cols = st.columns(3)
            cols[0].metric("Relevance", f"{account.get('category_relevance', 0):.0f}")
            cols[1].metric("Engagement", f"{account.get('engagement_score', 0):.0f}")
            cols[2].metric("Activity", f"{account.get('activity_score', 0):.0f}")
            st.caption(f"{format_compact_number(account.get('followers'))} followers")

        reason = discovery_reason or account.get("discovery_reason") or ""
        if reason and not compact:
            st.markdown("**Why this source matters**")
            st.info(reason)

        st.markdown(f"[View Profile](https://x.com/{username})")


def render_signal_card(
    topic: str,
    mention_count: int,
    signal_score: float | None = None,
    confidence_label: str | None = None,
    compact: bool = False,
) -> None:
    """Replaces the bare `.xi-chip` topic spans with a real card. `signal_score`
    and `confidence_label` are optional - pass them once Phase 2's scoring
    functions have computed them for this topic; a card with neither still
    renders cleanly with just the topic and mention count."""
    with st.container(border=True):
        st.markdown('<p class="xi-card-title">Signal</p>', unsafe_allow_html=True)
        st.markdown(f"**{topic}**")
        st.caption(f"{mention_count} supporting post(s)")

        if signal_score is not None:
            st.caption(f"Signal score: {signal_score:.0f}/100")
            _confidence_meter(signal_score)

        if confidence_label is not None:
            badge_class = _SIGNAL_BADGE_CLASS.get(confidence_label, "xi-signal-badge-low")
            st.markdown(
                f'<span class="xi-signal-badge {badge_class}">'
                f"Confidence: {confidence_label}</span>",
                unsafe_allow_html=True,
            )


def render_evidence_card(tweet: dict, compact: bool = False) -> None:
    """The shared citation/evidence unit - reused by Ask Intelligence
    citations and Story Brief "Supporting Posts". Always renders the real
    `.url` as "View on X" - never a fabricated link."""
    with st.container(border=True):
        st.markdown(f"**@{tweet.get('username') or 'unknown'}**")
        st.markdown(
            f'<div class="xi-evidence-quote">{tweet.get("text", "")}</div>',
            unsafe_allow_html=True,
        )

        if not compact:
            cols = st.columns(4)
            cols[0].caption(f"❤️ {tweet.get('like_count') or 0}")
            cols[1].caption(f"🔁 {tweet.get('retweet_count') or 0}")
            cols[2].caption(f"💬 {tweet.get('reply_count') or 0}")
            views = tweet.get("view_count")
            cols[3].caption(f"👁 {views if views is not None else '-'}")

        url = tweet.get("url")
        if url:
            st.markdown(f"[View on X]({url})")


def render_story_opportunity_card(
    title: str,
    why_it_matters: str,
    confidence_label: str | None,
    on_generate,
) -> None:
    """List-item card for the Story Opportunities tab. `confidence_label`
    renders as a text badge when available (never a numeric score here -
    the owning module decides if/how a score is shown); `on_generate` is a
    zero-arg callback invoked when the "Generate Brief" button is clicked."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(why_it_matters)
        if confidence_label is not None:
            badge_class = _SIGNAL_BADGE_CLASS.get(confidence_label, "xi-signal-badge-low")
            st.markdown(
                f'<span class="xi-signal-badge {badge_class}">'
                f"Confidence: {confidence_label}</span>",
                unsafe_allow_html=True,
            )
        if st.button("Generate Brief", key=f"generate-brief-{title}"):
            on_generate()


def render_brief_view(brief: dict) -> None:
    """Fixed six-section story-brief document layout: Headline -> Why It
    Matters -> Key Claims -> Supporting Posts -> Accounts Involved ->
    Investigation Questions. `brief` is a plain dict with keys `headline`,
    `why_it_matters`, `observed_facts`/`ai_interpretation` (the "Key Claims"
    section, kept visually distinct), `supporting_posts` (tweet-shaped
    dicts), `accounts` (account-shaped dicts), `investigation_questions`.
    A `note` key, if present, renders as an explicit incompleteness notice
    rather than letting a thin brief look complete.
    """
    st.markdown(f"## {brief.get('headline', 'Untitled')}")

    if brief.get("note"):
        st.warning(brief["note"])

    st.markdown("#### Why It Matters")
    st.write(brief.get("why_it_matters", ""))

    st.markdown("#### Key Claims")
    observed = brief.get("observed_facts") or []
    interpretation = brief.get("ai_interpretation") or []
    col_observed, col_interpretation = st.columns(2)
    with col_observed, st.container(border=True):
        st.markdown('<p class="xi-card-title">Observed in the evidence</p>', unsafe_allow_html=True)
        if observed:
            for fact in observed:
                st.markdown(f"- {fact}")
        else:
            st.caption("None reported.")
    with col_interpretation, st.container(border=True):
        st.markdown(
            '<p class="xi-card-title">AI interpretation (not directly observed)</p>',
            unsafe_allow_html=True,
        )
        if interpretation:
            for point in interpretation:
                st.markdown(f"- {point}")
        else:
            st.caption("None reported.")

    st.markdown("#### Supporting Posts")
    posts = brief.get("supporting_posts") or []
    if posts:
        for post in posts:
            render_evidence_card(post, compact=True)
    else:
        st.caption("No supporting posts available.")

    st.markdown("#### Accounts Involved")
    accounts = brief.get("accounts") or []
    if accounts:
        for account in accounts:
            # Only use the full source card (relevance/engagement/activity
            # sub-scores) when we actually have that RankedAccount data -
            # a username-only entry (from a citation, not the ranked-account
            # table) gets a plain link instead of implying a fabricated 0.0
            # score for fields we simply don't know here.
            if "ranking_score" in account:
                render_source_card(account, compact=True)
            else:
                username = account.get("username", "unknown")
                st.markdown(f"- [@{username}](https://x.com/{username})")
    else:
        st.caption("No accounts identified.")

    st.markdown("#### Investigation Questions")
    questions = brief.get("investigation_questions") or []
    if questions:
        for question in questions:
            st.markdown(f"- {question}")
    else:
        st.caption("None suggested.")


def render_freshness_badge(scraped_at: str | None, category: str | None = None) -> None:
    """Green-fresh / amber-aging / red-stale dot + "Last analyzed <relative
    time>" text, derived from the run snapshot's existing `scraped_at`
    field - never labels old cached data as live."""
    state, label = freshness_state(scraped_at)
    dot_class = _FRESHNESS_DOT_CLASS.get(state, "xi-status-bad")
    prefix = f"{category} — " if category else ""
    st.markdown(
        f'<div class="xi-status-row"><span class="xi-status-dot {dot_class}"></span>'
        f"{prefix}{label}</div>",
        unsafe_allow_html=True,
    )
