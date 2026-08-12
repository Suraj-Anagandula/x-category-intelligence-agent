"""Settings page: 100% read-only, no editable widgets, secrets never
displayed. Split into an "Account & Status" section (credentials/LLM - what
a normal user might want to confirm) and a collapsed "Developer" expander
(concurrency/retry/cache - unchanged content, just demoted from the main flow).
"""

from __future__ import annotations

import streamlit as st

from app.config import settings
from ui.utils import credential_status


def render_settings_page() -> None:
    st.title("Settings")
    st.caption("Read-only view of the active configuration. Secrets are never displayed.")

    status = credential_status(settings)

    st.markdown("#### Account & Status")
    col1, col2 = st.columns(2)
    col1.metric("X Authentication", "Configured" if status["x_auth"] else "Not configured")
    col2.metric(
        f"{settings.llm_provider.title()} API Key",
        "Configured" if status["llm"] else "Not configured",
    )

    model = settings.groq_model if settings.llm_provider == "groq" else settings.gemini_model
    st.write(f"**LLM Provider:** {settings.llm_provider}")
    st.write(f"**LLM Model:** {model}")

    with st.expander("Developer settings"):
        st.markdown("#### Concurrency")
        st.write(f"**Profile concurrency:** {settings.concurrency_limit}")
        st.write(f"**Tweet concurrency:** {settings.tweet_scrape_concurrency}")

        st.markdown("#### Retry & Rate Limiting")
        st.write(f"**Generic retry base:** {settings.backoff_base_seconds} seconds")
        st.write(f"**Generic retry max:** {settings.backoff_max_seconds} seconds")
        st.write(f"**Rate limit retry base:** {settings.rate_limit_base_seconds} seconds")
        st.write(f"**Rate limit maximum:** {settings.rate_limit_max_seconds} seconds")

        st.markdown("#### Cache")
        st.write(f"**Enabled:** {'Yes' if settings.cache_enabled else 'No'}")
        st.write(f"**TTL:** {settings.cache_ttl_seconds} seconds")
