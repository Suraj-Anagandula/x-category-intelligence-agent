"""New Analysis page: the only page that ever triggers a live pipeline run.

Productized inputs (category quick-picks + depth preset + time window)
resolve to the same parameters `CategoryIntelligenceAgent.run_pipeline`
already takes - power users can still override candidate/account/tweet
counts via the Advanced expander. Every other page reads exclusively from
`st.session_state`/disk, so browsing anywhere else never triggers a new X
request.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

import streamlit as st

from app.exceptions import AuthenticationError, LLMError, RateLimitError, ScraperError
from app.time_window import TIME_WINDOW_MODE_LABELS, TIME_WINDOW_MODES, resolve_time_window
from ui.components import (
    render_empty_state,
    render_error_message,
    render_pipeline_status,
    render_run_summary,
)
from ui.data_loader import list_run_files, load_run_json
from ui.pages._shared import render_downloads
from ui.pipeline_runner import run_category_analysis
from ui.utils import (
    CATEGORY_QUICK_PICKS,
    report_fallback_dict,
    resolve_depth_preset,
    validate_pipeline_params,
    validate_time_window_params,
)


def _execute_pipeline(
    category: str,
    candidate_limit: int,
    top_accounts: int,
    tweets_per_account: int,
    time_window,
) -> None:
    st.session_state["pipeline_running"] = True
    st.session_state["pipeline_stages_done"] = set()
    status_placeholder = st.empty()

    def on_stage(stage_key: str, _payload: dict) -> None:
        st.session_state["pipeline_stages_done"].add(stage_key)
        with status_placeholder.container(border=True):
            st.caption(
                "This runs the same steps every time - discovery, validation, ranking, "
                "tweet collection, analysis, export - real backend progress, updated live below."
            )
            render_pipeline_status(st.session_state["pipeline_stages_done"])

    with status_placeholder.container(border=True):
        st.caption(
            "This runs the same steps every time - discovery, validation, ranking, "
            "tweet collection, analysis, export - real backend progress, updated live below."
        )
        render_pipeline_status(st.session_state["pipeline_stages_done"])

    try:
        with st.spinner(f"Running category intelligence pipeline for '{category}'..."):
            report = run_category_analysis(
                category,
                candidate_limit,
                top_accounts,
                tweets_per_account,
                on_stage=on_stage,
                time_window=time_window,
            )

        files = list_run_files(report.category)
        run_data = load_run_json(files[0]) if files else report_fallback_dict(report)

        st.session_state["current_report"] = run_data
        st.session_state["current_category"] = report.category
        st.session_state["current_run_source"] = "live"

        if time_window.is_filtered and report.time_window.posts_in_window == 0:
            render_empty_state("No X posts were found in the selected time window.")
    except AuthenticationError as exc:
        render_error_message("auth", str(exc))
    except LLMError as exc:
        render_error_message("llm", str(exc))
    except RateLimitError as exc:
        render_error_message("rate_limit", str(exc))
    except ScraperError as exc:
        render_error_message("generic", str(exc))
    finally:
        st.session_state["pipeline_running"] = False


def render_new_analysis_page() -> None:
    st.title("New Analysis")
    st.markdown("---")

    st.markdown("##### What do you want to monitor?")
    default_category = st.session_state.pop("prefill_category", "")
    cols = st.columns(len(CATEGORY_QUICK_PICKS))
    for col, quick_pick in zip(cols, CATEGORY_QUICK_PICKS, strict=True):
        if col.button(quick_pick, key=f"na-quick-pick-{quick_pick}"):
            default_category = quick_pick

    category = st.text_input(
        "Category",
        value=default_category,
        placeholder="e.g. Technology, Healthcare, Indian Agriculture",
    )

    st.markdown("##### Analysis depth")
    depth_key = st.radio(
        "Analysis depth",
        options=["standard", "deep"],
        format_func=lambda key: "Standard" if key == "standard" else "Deep",
        captions=[
            "Top 20 of 50 candidates, 10 tweets each - a few minutes.",
            "Top 40 of 100 candidates, 20 tweets each - a longer run.",
        ],
        label_visibility="collapsed",
        horizontal=True,
    )
    preset_candidate_limit, preset_top_accounts, preset_tweets_per_account = resolve_depth_preset(
        depth_key
    )

    st.markdown("##### Time window")
    window_mode = st.selectbox(
        "Time window",
        options=TIME_WINDOW_MODES,
        format_func=lambda key: TIME_WINDOW_MODE_LABELS[key],
        label_visibility="collapsed",
    )

    custom_start = custom_end = None
    if window_mode == "custom":
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Start date")
            start_time = st.time_input("Start time (UTC)", value=time(0, 0))
        with col_end:
            end_date = st.date_input("End date")
            end_time = st.time_input("End time (UTC)", value=time(0, 0))
        custom_start = datetime.combine(start_date, start_time, tzinfo=timezone.utc)
        custom_end = datetime.combine(end_date, end_time, tzinfo=timezone.utc)
        st.caption("Custom range times are interpreted as UTC.")
    elif window_mode == "latest":
        st.caption("Analyzes the most recent available tweets per account - the original behavior.")
    else:
        st.caption(
            f"Only analyzes tweets actually posted in the {TIME_WINDOW_MODE_LABELS[window_mode].lower()}, "
            "based on their real X timestamp (UTC). Reaching further back in an account's "
            "history may require more requests and is not guaranteed for very old dates - "
            "see the README for X/Twikit's real historical-retrieval limits."
        )

    with st.expander("Advanced"):
        col1, col2, col3 = st.columns(3)
        candidate_limit = col1.number_input(
            "Candidate limit", min_value=1, value=preset_candidate_limit, step=5
        )
        top_accounts = col2.number_input(
            "Top accounts", min_value=1, value=preset_top_accounts, step=1
        )
        tweets_per_account = col3.number_input(
            "Tweets per account", min_value=1, value=preset_tweets_per_account, step=1
        )

    run_clicked = st.button(
        "Analyze Now", type="primary", disabled=st.session_state["pipeline_running"]
    )

    if run_clicked:
        error = validate_pipeline_params(
            category, candidate_limit, top_accounts, tweets_per_account
        ) or validate_time_window_params(window_mode, custom_start, custom_end)
        if error:
            st.error(error)
        else:
            time_window = resolve_time_window(window_mode, custom_start, custom_end)
            _execute_pipeline(
                category.strip(),
                int(candidate_limit),
                int(top_accounts),
                int(tweets_per_account),
                time_window,
            )

    report = st.session_state.get("current_report")
    if report and st.session_state.get("current_run_source") == "live":
        st.markdown("---")
        render_run_summary(report)
        render_downloads(report.get("category", ""))
