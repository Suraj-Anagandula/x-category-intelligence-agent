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
:root {
    --xi-space-xs: 0.25rem;
    --xi-space-sm: 0.5rem;
    --xi-space-md: 1rem;
    --xi-space-lg: 1.5rem;
}

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

.xi-card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.65;
    margin-bottom: var(--xi-space-xs);
}

/* Signal-strength badges - same rgba language as the status dots above,
   just a filled pill instead of a bare dot, plus a neutral/amber variant
   for "medium" states that the two-color status convention didn't need. */
.xi-signal-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.xi-signal-badge-high { background-color: rgba(84, 162, 75, 0.15); color: #54A24B; }
.xi-signal-badge-medium { background-color: rgba(224, 158, 24, 0.15); color: #B87F0F; }
.xi-signal-badge-low { background-color: rgba(127, 127, 127, 0.15); color: #7A7A7A; }

.xi-status-warn { background-color: #E09E18; }

.xi-evidence-quote {
    border-left: 3px solid rgba(76, 120, 168, 0.5);
    padding: var(--xi-space-xs) var(--xi-space-md);
    margin: var(--xi-space-xs) 0;
    font-style: italic;
    opacity: 0.9;
}

.xi-confidence-meter {
    position: relative;
    height: 6px;
    border-radius: 999px;
    background-color: rgba(127, 127, 127, 0.15);
    overflow: hidden;
    margin: var(--xi-space-xs) 0;
}
.xi-confidence-meter-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    border-radius: 999px;
    background-color: #4C78A8;
}
</style>
"""


def inject_custom_css() -> None:
    """Call once, near the top of streamlit_app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)
