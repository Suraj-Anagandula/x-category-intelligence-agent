"""Sources page: ranked accounts driving the conversation. Renamed from the
old "Accounts" page; the detail card now shows relevance/engagement/activity
sub-scores, follower count, "why this source matters", and a View Profile
link via the shared `render_source_card`.
"""

from __future__ import annotations

import streamlit as st

from ui.cards import render_source_card
from ui.components import render_account_table, render_empty_state
from ui.data_loader import get_discovery_reasons


def render_sources_page() -> None:
    st.title("Sources")
    report = st.session_state.get("current_report")
    if not report:
        render_empty_state(
            "No analysis has been run yet. Run a category analysis to see ranked sources."
        )
        return

    accounts = report.get("accounts", [])
    render_account_table(accounts)

    if not accounts:
        return

    st.markdown("---")
    st.markdown("#### Source Detail")
    usernames = [account["username"] for account in accounts]
    selected = st.selectbox("Select a source", usernames)
    account = next((a for a in accounts if a["username"] == selected), None)
    if account is None:
        return

    reason = account.get("discovery_reason")
    if not reason:
        reasons = get_discovery_reasons(report.get("category", ""))
        reason = reasons.get(account["username"], "")
    render_source_card(account, discovery_reason=reason)
