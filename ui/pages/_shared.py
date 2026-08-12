"""Small Streamlit-aware helpers shared by more than one page module.

Kept separate from `ui/utils.py` (which stays Streamlit-free/testable) since
`render_downloads` calls `st.*` directly and can't live there.
"""

from __future__ import annotations

import streamlit as st

from app.config import settings
from ui.data_loader import list_run_files


def render_downloads(category: str) -> None:
    st.markdown("#### Downloads")
    files = list_run_files(category)
    cols = st.columns(3)

    if files:
        cols[0].download_button(
            "Download Run JSON",
            data=files[0].read_bytes(),
            file_name=files[0].name,
            mime="application/json",
            key=f"dl-json-{category}",
        )

    csv_path = settings.csv_output_dir / f"{category}_tweets.csv"
    if csv_path.exists():
        cols[1].download_button(
            "Download Consolidated CSV",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
            key=f"dl-csv-{category}",
        )

    users_path = settings.csv_output_dir / "users.csv"
    if users_path.exists():
        cols[2].download_button(
            "Download Users CSV",
            data=users_path.read_bytes(),
            file_name="users.csv",
            mime="text/csv",
            key=f"dl-users-{category}",
        )
