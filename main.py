"""CLI entry point for the X public profile collector.

Usage
-----
    python main.py                          # interactive: prompts for a username, shows its latest tweets
    python main.py elonmusk
    python main.py elonmusk openai satyanadella
    python main.py usernames.txt
    python main.py --csv usernames.txt
    python main.py --json usernames.txt
    python main.py --both usernames.txt
    python main.py category sports
    python main.py analyze sports --candidate-limit 100 --top-accounts 20 --tweets-per-account 20
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from app.cache import ProfileCache, TweetCache
from app.category_agent import CategoryAgent, CategoryIntelligenceAgent
from app.client import TwikitProfileClient
from app.config import settings
from app.exceptions import AuthenticationError, ScraperError
from app.exporter import CSVExporter, ExporterRegistry, JSONExporter, TweetCSVExporter
from app.llm import build_llm_client
from app.logger import configure_logging, get_logger
from app.models import ScrapeResult, Tweet
from app.schemas import CategoryReport
from app.scraper import ProfileScraper
from app.tweet_scraper import TweetScraper
from app.utils import (
    dedupe_preserve_order,
    is_valid_username,
    normalize_username,
    read_usernames_from_file,
    split_errors_by_stage,
)


def _make_utf8_safe(stream):
    """Reconfigure a text stream to UTF-8 with errors='replace', if possible.

    On Windows, Python's default stdout/stderr encoding is the system's
    ANSI code page (commonly cp1252), which can't represent most Unicode
    punctuation (non-breaking hyphens, em/en dashes, curly quotes, emoji,
    ...) - exactly what LLM-generated text (category analysis summaries)
    commonly contains. Both Rich's legacy-Windows-console render path
    (`rich._win32_console.LegacyWindowsTerm.write_text`) and loguru's
    console sink write straight through to `stream.write()`, so this single
    reconfiguration at the shared console/logging boundary covers both.
    UTF-8 can represent any Unicode code point, so `errors="replace"` here
    is just a defensive backstop, not the primary fix - no character
    replacement list is involved, and arbitrary Unicode is unaffected.
    Never raises: encoding setup must not be able to crash the CLI itself,
    and streams that don't support reconfiguration (e.g. test harnesses)
    are left untouched.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(Exception):  # output setup must never crash the CLI
            reconfigure(encoding="utf-8", errors="replace")
    return stream


for _stream in (sys.stdout, sys.stderr):
    _make_utf8_safe(_stream)

app = typer.Typer(add_completion=False, help="Collect public X (Twitter) profile information.")
console = Console()


def _resolve_usernames(targets: list[str]) -> list[str]:
    """Interpret `targets` as either a single usernames-file path or a literal list."""
    if len(targets) == 1 and Path(targets[0]).is_file():
        return read_usernames_from_file(Path(targets[0]))
    return list(targets)


def _resolve_formats(*, as_json: bool, as_csv: bool, as_both: bool) -> list[str]:
    if as_both or (as_json and as_csv):
        return ["json", "csv"]
    if as_csv:
        return ["csv"]
    if as_json:
        return ["json"]
    return ["json"]  # default output format


def _print_summary(results: list[ScrapeResult]) -> None:
    table = Table(title="Scrape Results")
    table.add_column("Username")
    table.add_column("Status")
    table.add_column("Followers", justify="right")
    table.add_column("Verified")
    table.add_column("Detail")

    for result in results:
        if result.success and result.profile:
            table.add_row(
                f"@{result.profile.username}",
                "[green]OK[/green]",
                f"{result.profile.followers:,}" if result.profile.followers is not None else "-",
                "yes" if (result.profile.verified or result.profile.is_blue_verified) else "no",
                (
                    "cached"
                    if result.from_cache
                    else (
                        "fetched"
                        if result.attempts == 1
                        else f"fetched, {result.attempts} attempts"
                    )
                ),
            )
        else:
            table.add_row(
                f"@{result.username}", "[red]FAILED[/red]", "-", "-", result.error_type or ""
            )

    console.print(table)

    succeeded = sum(1 for r in results if r.success)
    console.print(f"\n[bold]{succeeded}/{len(results)}[/bold] profiles collected successfully.")


def _build_client() -> TwikitProfileClient:
    return TwikitProfileClient(
        auth_token=settings.x_auth_token,
        ct0=settings.x_ct0,
        username=settings.x_username,
        email=settings.x_email,
        password=settings.x_password,
        session_file=settings.x_session_file,
    )


def _print_tweets(username: str, tweets: list[Tweet]) -> None:
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
    if not tweets:
        console.print(f"[yellow]@{username} has no recent tweets to show.[/yellow]")


async def _run_tweets(username: str, count: int) -> tuple[list[Tweet], Path]:
    settings.ensure_directories()
    client = _build_client()
    await client.connect()
    tweets = await client.get_recent_tweets(username, count=count)

    exporter = TweetCSVExporter(output_dir=settings.csv_output_dir, username=username)
    csv_path = await exporter.export(tweets)

    return tweets, csv_path


def _print_category_report(report: CategoryReport, requested_top_n: int) -> None:
    table = Table(title=f"Top accounts - {report.category}")
    table.add_column("Rank", justify="right")
    table.add_column("Username")
    table.add_column("Followers", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Score", justify="right")

    for account in report.accounts:
        table.add_row(
            str(account.rank),
            f"@{account.username}",
            f"{account.followers:,}" if account.followers is not None else "-",
            f"{account.category_relevance:.1f}",
            f"{account.ranking_score:.1f}",
        )

    console.print(table)
    console.print(f"\n[bold]Accounts selected:[/bold] {len(report.accounts)}/{requested_top_n}")
    stats = report.tweet_statistics
    console.print(
        f"[bold]Accounts successfully scraped:[/bold] {stats.accounts_processed}  "
        f"[bold]Rate limited:[/bold] {stats.accounts_rate_limited}  "
        f"[bold]Failed (other):[/bold] {stats.accounts_failed_other}  "
        f"[bold]Tweets collected:[/bold] {stats.tweets_collected}"
    )
    if report.analysis.trending_topics:
        console.print(f"Trending topics: {', '.join(report.analysis.trending_topics)}")
    sentiment = report.analysis.sentiment
    console.print(
        f"Sentiment - positive {sentiment.positive}% / neutral {sentiment.neutral}% / "
        f"negative {sentiment.negative}%"
    )
    if report.analysis.summary:
        console.print(f"\n{report.analysis.summary}")
    validation_errors, pipeline_errors = split_errors_by_stage(report.errors)
    if pipeline_errors:
        console.print(
            f"\n[yellow]{len(pipeline_errors)} pipeline failure(s) among the selected accounts - "
            "see the log file for detail.[/yellow]"
        )
    if validation_errors:
        console.print(
            f"[dim]{len(validation_errors)} candidate account(s) were rejected during "
            "discovery/validation before ranking - not part of the final selected accounts.[/dim]"
        )

    category_csv_path = settings.csv_output_dir / f"{report.category}_tweets.csv"
    console.print(f"\n[bold]CSV:[/bold] {category_csv_path}")


async def _run_category(category: str) -> None:
    settings.ensure_directories()
    llm_client = build_llm_client(settings)
    get_logger().info(
        f"LLM provider: {settings.llm_provider}"
        + ("" if llm_client is not None else " (not configured - no API key)")
    )
    ctx = await CategoryAgent(llm_client).build_context(category)
    console.print_json(data=ctx.model_dump())


async def _run_analyze(
    category: str,
    candidate_limit: int | None,
    top_accounts_limit: int | None,
    tweets_per_account: int | None,
) -> CategoryReport:
    settings.ensure_directories()

    client = _build_client()
    await client.connect()

    profile_cache = ProfileCache(
        cache_dir=settings.cache_dir,
        ttl_seconds=settings.cache_ttl_seconds,
        enabled=settings.cache_enabled,
    )
    profile_scraper = ProfileScraper(settings, client, profile_cache)

    tweet_cache = TweetCache(
        cache_dir=settings.tweet_cache_dir,
        ttl_seconds=settings.tweet_cache_ttl_seconds,
        enabled=settings.cache_enabled,
    )
    tweet_scraper = TweetScraper(settings, client, tweet_cache)

    llm_client = build_llm_client(settings)
    get_logger().info(
        f"LLM provider: {settings.llm_provider}"
        + ("" if llm_client is not None else " (not configured - no API key)")
    )

    agent = CategoryIntelligenceAgent(settings, profile_scraper, tweet_scraper, llm_client)
    return await agent.run_pipeline(
        category,
        candidate_limit=candidate_limit,
        top_n=top_accounts_limit,
        tweets_per_account=tweets_per_account,
    )


async def _run(usernames: list[str], formats: list[str]) -> list[ScrapeResult]:
    settings.ensure_directories()

    client = _build_client()
    await client.connect()

    cache = ProfileCache(
        cache_dir=settings.cache_dir,
        ttl_seconds=settings.cache_ttl_seconds,
        enabled=settings.cache_enabled,
    )
    scraper = ProfileScraper(settings, client, cache)
    results = await scraper.scrape_many(usernames)

    profiles = [r.profile for r in results if r.success and r.profile]
    if profiles:
        registry = ExporterRegistry()
        registry.register("json", JSONExporter)
        registry.register("csv", CSVExporter)

        if "json" in formats:
            await registry.create("json", output_dir=settings.json_output_dir).export(profiles)
        if "csv" in formats:
            await registry.create("csv", output_dir=settings.csv_output_dir).export(profiles)

    return results


@app.command()
def main(
    targets: list[str] | None = typer.Argument(
        None,
        help=(
            "One or more usernames, or a single path to a newline-delimited usernames file. "
            "If omitted, you'll be prompted for a username and shown their latest tweets."
        ),
    ),
    tweet_count: int = typer.Option(
        10, "--tweet-count", help="Number of tweets to show in interactive mode."
    ),
    as_json: bool = typer.Option(False, "--json", help="Export results as JSON."),
    as_csv: bool = typer.Option(False, "--csv", help="Export results as CSV."),
    as_both: bool = typer.Option(False, "--both", help="Export results as both JSON and CSV."),
    concurrency: int | None = typer.Option(
        None, "--concurrency", help="Override CONCURRENCY_LIMIT (10/20/50/100)."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass the profile cache for this run."
    ),
    candidate_limit: int | None = typer.Option(
        None,
        "--candidate-limit",
        help="Category discovery candidate pool size (default: CATEGORY_CANDIDATE_LIMIT). "
        "Only used with 'category'/'analyze'.",
    ),
    top_accounts: int | None = typer.Option(
        None,
        "--top-accounts",
        help="Number of top-ranked accounts to select (default: TOP_ACCOUNTS_LIMIT). "
        "Only used with 'analyze'.",
    ),
    tweets_per_account_opt: int | None = typer.Option(
        None,
        "--tweets-per-account",
        help="Tweets to fetch per top account (default: TWEETS_PER_ACCOUNT). "
        "Only used with 'analyze'.",
    ),
) -> None:
    """Collect public profile information for one or more X usernames.

    Run with no arguments to be prompted for a single username and see
    their latest tweets instead of full profile export.

    Category intelligence:
        python main.py category <category>   # show discovery keywords/subcategories
        python main.py analyze <category> [--candidate-limit N --top-accounts N --tweets-per-account N]
    """
    configure_logging(settings.log_dir, settings.log_level)
    logger = get_logger()

    if concurrency is not None:
        settings.concurrency_limit = concurrency
    if no_cache:
        settings.cache_enabled = False

    if targets and targets[0] in {"category", "analyze"}:
        mode, *rest = targets
        if not rest:
            console.print(f"[red]Usage:[/red] python main.py {mode} <category> [options]")
            raise typer.Exit(code=1)
        category = " ".join(rest)

        logger.info(f"CLI invoked in {mode!r} mode for category {category!r}")
        try:
            if mode == "category":
                asyncio.run(_run_category(category))
            else:
                requested_top_n = top_accounts or settings.top_accounts_limit
                report = asyncio.run(
                    _run_analyze(category, candidate_limit, top_accounts, tweets_per_account_opt)
                )
                _print_category_report(report, requested_top_n)
        except AuthenticationError as exc:
            logger.error(f"Authentication failed: {exc}")
            console.print(f"[red]Authentication error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        except ScraperError as exc:
            logger.error(f"Category pipeline failed for {category!r}: {exc}")
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        return

    if not targets:
        raw_username = typer.prompt("Enter an X username")
        username = normalize_username(raw_username)
        if not is_valid_username(username):
            console.print(f"[red]Invalid username:[/red] {raw_username!r}")
            raise typer.Exit(code=1)

        logger.info(f"Interactive mode: fetching latest {tweet_count} tweet(s) for @{username}")
        try:
            tweets, csv_path = asyncio.run(_run_tweets(username, tweet_count))
        except AuthenticationError as exc:
            logger.error(f"Authentication failed: {exc}")
            console.print(f"[red]Authentication error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        except ScraperError as exc:
            logger.error(f"Failed to fetch tweets for @{username}: {exc}")
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        _print_tweets(username, tweets)
        if tweets:
            console.print(f"\n[bold]Saved to[/bold] {csv_path}")
        return

    usernames = dedupe_preserve_order(_resolve_usernames(targets))
    if not usernames:
        console.print("[red]No usernames provided.[/red]")
        raise typer.Exit(code=1)

    formats = _resolve_formats(as_json=as_json, as_csv=as_csv, as_both=as_both)
    logger.info(f"CLI invoked for {len(usernames)} username(s); formats={formats}")

    try:
        results = asyncio.run(_run(usernames, formats))
    except AuthenticationError as exc:
        logger.error(f"Authentication failed: {exc}")
        console.print(f"[red]Authentication error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_summary(results)

    if all(not r.success for r in results):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
