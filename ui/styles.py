"""Minimal, theme-neutral CSS for the Streamlit dashboard.

Uses semi-transparent (rgba) overlays rather than hardcoded light/dark
colors, so it doesn't fight Streamlit's own light/dark theme switching -
the same rule the project's published Artifacts follow. No gradients or
animations, to keep the look like a serious analytics product rather than a
demo.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(127, 127, 127, 0.2);
}

.xi-brand {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-bottom: 0;
}

.xi-brand-subtitle {
    font-size: 0.85rem;
    opacity: 0.7;
    margin-top: 0;
}

.xi-status-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
    padding: 0.15rem 0;
}

.xi-status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.xi-status-ok { background-color: #54A24B; }
.xi-status-bad { background-color: #E45756; }

div[data-testid="stMetric"] {
    background-color: rgba(127, 127, 127, 0.08);
    border: 1px solid rgba(127, 127, 127, 0.15);
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
}

.xi-chip {
    display: inline-block;
    background-color: rgba(76, 120, 168, 0.15);
    border: 1px solid rgba(76, 120, 168, 0.35);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    margin: 0.15rem 0.3rem 0.15rem 0;
    font-size: 0.85rem;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(127, 127, 127, 0.15);
    border-radius: 0.5rem;
}
</style>
"""


def inject_custom_css() -> None:
    """Call once, near the top of streamlit_app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)
