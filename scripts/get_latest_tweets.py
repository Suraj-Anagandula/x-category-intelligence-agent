"""Fetch and print a user's most recent public tweets.

Usage:
    python scripts/get_latest_tweets.py elonmusk [count]

Uses the same authenticated session (cached cookies) as `main.py`.
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.table import Table

from app.client import TwikitProfileClient
from app.config import settings


async def main(username: str, count: int) -> None:
    settings.ensure_directories()
    client = TwikitProfileClient(
        auth_token=settings.x_auth_token,
        ct0=settings.x_ct0,
        username=settings.x_username,
        email=settings.x_email,
        password=settings.x_password,
        session_file=settings.x_session_file,
    )
    await client.connect()

    tweets = await client.get_recent_tweets(username, count=count)

    console = Console()
    table = Table(title=f"Latest tweets from @{username}")
    table.add_column("Date")
    table.add_column("Text", overflow="fold", max_width=70)
    table.add_column("Likes", justify="right")
    table.add_column("Retweets", justify="right")
    table.add_column("Views", justify="right")

    for tweet in tweets:
        table.add_row(
            tweet.created_at.strftime("%Y-%m-%d %H:%M") if tweet.created_at else "-",
            tweet.text,
            f"{tweet.like_count:,}" if tweet.like_count is not None else "-",
            f"{tweet.retweet_count:,}" if tweet.retweet_count is not None else "-",
            f"{tweet.view_count:,}" if tweet.view_count is not None else "-",
        )

    console.print(table)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "elonmusk"
    tweet_count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    asyncio.run(main(target, tweet_count))
