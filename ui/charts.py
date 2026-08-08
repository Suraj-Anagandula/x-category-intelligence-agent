"""Plotly chart builders for the Analytics page.

Pure functions: `list[dict]` (the same shape `app/storage.py` already writes
to `data/tweets/<category>/<date>.json` - `RankedAccount`/`Tweet` field
names) in, a `plotly.graph_objects.Figure` out. No Streamlit calls inside,
so these are usable for both a just-completed live report and a historical
one loaded from disk, and are independently testable.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_EMPTY_LAYOUT = {
    "xaxis": {"visible": False},
    "yaxis": {"visible": False},
    "margin": {"l": 20, "r": 20, "t": 30, "b": 20},
}


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14})
    fig.update_layout(**_EMPTY_LAYOUT)
    return fig


def create_ranking_chart(accounts: list[dict]) -> go.Figure:
    """Horizontal bar of ranking_score by account, highest at the top."""
    if not accounts:
        return _empty_figure("No ranked accounts available.")

    df = pd.DataFrame(accounts).sort_values("ranking_score", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df["ranking_score"],
            y=[f"@{u}" for u in df["username"]],
            orientation="h",
            marker_color="#4C78A8",
        )
    )
    fig.update_layout(
        xaxis_title="Ranking score",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=max(300, 28 * len(df)),
    )
    return fig


def create_followers_chart(accounts: list[dict]) -> go.Figure:
    """Horizontal bar of follower counts, log-scaled x-axis (follower counts
    vary by orders of magnitude across accounts in the same category)."""
    if not accounts:
        return _empty_figure("No account data available.")

    df = pd.DataFrame(accounts)
    df = df[df["followers"].notna()].sort_values("followers", ascending=True)
    if df.empty:
        return _empty_figure("No follower data available.")

    fig = go.Figure(
        go.Bar(
            x=df["followers"],
            y=[f"@{u}" for u in df["username"]],
            orientation="h",
            marker_color="#72B7B2",
        )
    )
    fig.update_layout(
        xaxis_title="Followers (log scale)",
        xaxis_type="log",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=max(300, 28 * len(df)),
    )
    return fig


def create_relevance_score_chart(accounts: list[dict]) -> go.Figure:
    """Scatter: category relevance (x) vs overall ranking score (y)."""
    if not accounts:
        return _empty_figure("No account data available.")

    df = pd.DataFrame(accounts)
    fig = go.Figure(
        go.Scatter(
            x=df["category_relevance"],
            y=df["ranking_score"],
            mode="markers+text",
            text=[f"@{u}" for u in df["username"]],
            textposition="top center",
            marker={"size": 10, "color": "#E45756"},
        )
    )
    fig.update_layout(
        xaxis_title="Category relevance",
        yaxis_title="Ranking score",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return fig


def create_sentiment_chart(sentiment: dict) -> go.Figure:
    """Donut chart of positive/neutral/negative sentiment shares."""
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        sentiment.get("positive", 0.0),
        sentiment.get("neutral", 0.0),
        sentiment.get("negative", 0.0),
    ]
    if not any(values):
        return _empty_figure("Sentiment data unavailable for this run.")

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker={"colors": ["#54A24B", "#B0B0B0", "#E45756"]},
        )
    )
    fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
    return fig


def create_engagement_chart(tweets: list[dict]) -> go.Figure:
    """Total engagement (likes + retweets + replies) per account."""
    if not tweets:
        return _empty_figure("No tweets available.")

    df = pd.DataFrame(tweets)
    if not {"username", "like_count", "retweet_count", "reply_count"}.issubset(df.columns):
        return _empty_figure("No engagement data available.")

    df = df.copy()
    df["engagement"] = (
        df["like_count"].fillna(0) + df["retweet_count"].fillna(0) + df["reply_count"].fillna(0)
    )
    grouped = df.groupby("username", as_index=False)["engagement"].sum()
    grouped = grouped.sort_values("engagement", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=grouped["engagement"],
            y=[f"@{u}" for u in grouped["username"]],
            orientation="h",
            marker_color="#F58518",
        )
    )
    fig.update_layout(
        xaxis_title="Total engagement (likes + retweets + replies)",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=max(300, 28 * len(grouped)),
    )
    return fig


def create_tweet_distribution_chart(tweets: list[dict]) -> go.Figure:
    """Number of collected tweets per account."""
    if not tweets:
        return _empty_figure("No tweets available.")

    df = pd.DataFrame(tweets)
    if "username" not in df.columns:
        return _empty_figure("No tweet data available.")

    counts = df["username"].value_counts().sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=[f"@{u}" for u in counts.index],
            orientation="h",
            marker_color="#B279A2",
        )
    )
    fig.update_layout(
        xaxis_title="Tweets collected",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=max(300, 28 * len(counts)),
    )
    return fig
