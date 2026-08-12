"""Streamlit entry point for the X Intelligence dashboard.

Run with:
    streamlit run streamlit_app.py

This is a presentation layer only - it calls the existing
`CategoryIntelligenceAgent` pipeline via `ui.pipeline_runner` and reads the
existing on-disk storage via `ui.data_loader`. It never reimplements
discovery, validation, ranking, scraping, retry/rate-limit handling, or
analysis, and it never mutates `main.py` or anything under `app/`.

The pipeline is only ever invoked from the explicit "Analyze Now" button on
the New Analysis page - every other page reads exclusively from
`st.session_state`/disk, so simply browsing Overview/Sources/Intelligence/
Trends/Reports/Settings never triggers a new X request.

This file stays a thin router: page layout lives in `ui/pages/*.py`, one
module per nav destination.
"""

from __future__ import annotations

import streamlit as st

from app.config import settings
from ui.pages.intelligence import render_intelligence_page
from ui.pages.new_analysis import render_new_analysis_page
from ui.pages.overview import render_overview_page
from ui.pages.reports import render_reports_page
from ui.pages.settings import render_settings_page
from ui.pages.sources import render_sources_page
from ui.pages.trends import render_trends_page
from ui.styles import inject_custom_css
from ui.utils import credential_status

NAV_OPTIONS = [
    "Overview",
    "New Analysis",
    "Sources",
    "Intelligence",
    "Trends",
    "Reports",
    "Settings",
]


def _init_session_state() -> None:
    st.session_state.setdefault("current_category", None)
    st.session_state.setdefault("current_report", None)
    st.session_state.setdefault("current_run_source", None)
    st.session_state.setdefault("pipeline_running", False)
    st.session_state.setdefault("pipeline_stages_done", set())


def _apply_pending_navigation() -> None:
    """Apply any programmatic page-switch request before the `nav_page`
    radio widget below is instantiated this run.

    Streamlit forbids writing to a widget-bound `st.session_state` key
    after that widget has already rendered in the same script run (raises
    `StreamlitAPIException`). Pages that want to switch tabs (e.g. an
    Overview CTA button) can't write `st.session_state["nav_page"]`
    directly - the sidebar radio below already instantiated that key
    earlier in the very run where the button was clicked. Instead, they
    set `st.session_state["nav_target"]` and call `st.rerun()`; on the
    *next* run - here, before the radio widget exists yet - that request
    is copied into `nav_page` and cleared, which Streamlit accepts as the
    widget's initial value for this run.
    """
    pending = st.session_state.pop("nav_target", None)
    if pending is not None:
        st.session_state["nav_page"] = pending


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


def main() -> None:
    st.set_page_config(page_title="X Intelligence", page_icon="📊", layout="wide")
    inject_custom_css()
    _init_session_state()
    _apply_pending_navigation()

    page = _render_sidebar()

    pages = {
        "Overview": render_overview_page,
        "New Analysis": render_new_analysis_page,
        "Sources": render_sources_page,
        "Intelligence": render_intelligence_page,
        "Trends": render_trends_page,
        "Reports": render_reports_page,
        "Settings": render_settings_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()
