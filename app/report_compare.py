"""Compare two category-run snapshots - pure functions, no I/O, matching
`app/account_ranker.py`'s convention.

Primary use case: the SAME category across two different dates (e.g.
Technology 2026-08-08 vs. a later Technology run) - "what changed since
last time," not a same-date comparison across different categories. Both
inputs are the plain dict shape `ui/data_loader.py::load_run_json` already
returns (the `data/tweets/<category>/<date>.json` snapshot format) - no new
storage or representation is introduced.

All deltas are arithmetic over fields already stored by the pipeline
(`trending_topics`, `accounts[].rank`/`ranking_score`, `sentiment`,
`tweet_statistics`) - nothing here is a new computation model, just a diff.

Topic classification (added/removed/persisted) additionally reuses
`app.topic_matching.topic_keywords` (never a second, separate matching
system) to recognize the same underlying theme reworded between two runs
- see `topics_related`/`_reconcile_similar_topics` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.topic_matching import topic_keywords


@dataclass
class AccountMover:
    username: str
    rank_delta: int
    score_delta: float


@dataclass
class ComparisonResult:
    older_label: str
    newer_label: str
    #: Human-readable analysis-window periods (e.g. "2026-08-01 -> 2026-08-08"),
    #: distinct from `older_label`/`newer_label`'s run/collection dates.
    #: "Latest Available" when a snapshot has no time-window metadata
    #: (pre-existing snapshots) or was run in "latest" mode - never invented.
    older_window_label: str = "Latest Available"
    newer_window_label: str = "Latest Available"
    topics_added: list[str] = field(default_factory=list)
    topics_removed: list[str] = field(default_factory=list)
    topics_persisted: list[str] = field(default_factory=list)
    accounts_new: list[str] = field(default_factory=list)
    accounts_dropped: list[str] = field(default_factory=list)
    movers: list[AccountMover] = field(default_factory=list)
    sentiment_delta: dict[str, float] = field(default_factory=dict)
    tweets_collected_delta: int = 0
    accounts_processed_delta: int = 0


def _run_label(run: dict) -> str:
    category = run.get("category", "?")
    scraped_at = run.get("scraped_at", "")
    date = scraped_at.split("T")[0] if scraped_at else "?"
    return f"{category} — {date}"


def _window_label(run: dict) -> str:
    """Human-readable analysis-window period for one snapshot, e.g.
    "2026-08-01 -> 2026-08-08". Falls back to "Latest Available" for
    "latest"-mode runs and for snapshots saved before time-window support
    existed (no "time_window" key at all) - both read as the same thing to
    a viewer: no window was applied, the run just analyzed the most recent
    available tweets.
    """
    window = run.get("time_window") or {}
    mode = window.get("mode", "latest")
    start = window.get("start")
    end = window.get("end")
    if mode == "latest" or not start or not end:
        return "Latest Available"
    return f"{start.split('T')[0]} → {end.split('T')[0]}"


#: How much of the shorter topic's significant-keyword set must overlap for
#: two differently-worded topics to be considered the same underlying theme.
#: Deliberately conservative (not just "any shared keyword") - a single
#: shared generic-but-not-a-stopword word (e.g. "strategies", "updates")
#: between two otherwise unrelated short topic phrases must not be enough
#: to merge them; see tests/test_report_compare.py for the exact negative
#: case this threshold guards against.
TOPIC_RELATEDNESS_THRESHOLD = 0.5


def topics_related(
    topic_a: str, topic_b: str, min_overlap_ratio: float = TOPIC_RELATEDNESS_THRESHOLD
) -> bool:
    """Whether two trending-topic phrases represent the same/closely
    related underlying theme, despite different exact wording.

    Reuses `app.topic_matching.topic_keywords` (the existing keyword-
    extraction helper already used for tweet-to-topic matching elsewhere)
    rather than introducing a second, separate matching system. Exact
    match (case/whitespace-insensitive) is always related. Otherwise,
    related only if the two phrases' significant keywords overlap by at
    least `min_overlap_ratio` of the shorter phrase's keyword count - not
    merely "share one word," which would over-match unrelated topics that
    happen to share a single generic term.
    """
    a = topic_a.strip().lower()
    b = topic_b.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True

    keywords_a = set(topic_keywords(topic_a))
    keywords_b = set(topic_keywords(topic_b))
    if not keywords_a or not keywords_b:
        return False

    overlap = keywords_a & keywords_b
    if not overlap:
        return False

    smaller = min(len(keywords_a), len(keywords_b))
    return (len(overlap) / smaller) >= min_overlap_ratio


def _reconcile_similar_topics(
    added: list[str], removed: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Move topic pairs that `topics_related` considers the same underlying
    theme out of `added`/`removed` and into a third "matched" list, using
    the newer phrasing (from `added`) as the single displayed label - the
    most current articulation of an ongoing theme.

    Greedy, deterministic pairing over the given (already-sorted) lists:
    each newer topic is matched against at most one older topic, in list
    order, and each older topic can be consumed at most once. `added` and
    `removed` here have already had their exact-string overlap removed
    (that's `topics_persisted`'s existing exact-match computation, done by
    the caller before this runs) - this only reconciles the leftovers.
    """
    remaining_removed = list(removed)
    still_added: list[str] = []
    matched: list[str] = []

    for new_topic in added:
        match = next((old for old in remaining_removed if topics_related(new_topic, old)), None)
        if match is not None:
            remaining_removed.remove(match)
            matched.append(new_topic)
        else:
            still_added.append(new_topic)

    return sorted(still_added), sorted(remaining_removed), sorted(matched)


def compare_reports(older: dict, newer: dict) -> ComparisonResult:
    """Diff two run snapshots. Both are plain dicts loaded from disk -
    every lookup uses `.get(...)` with a default, since a snapshot saved
    before a given field existed (e.g. `discovery_reason`) must not crash
    a comparison against a newer one (see the 12 real 2026-08-08 snapshots,
    which predate several fields added since).
    """
    older_analysis = older.get("analysis", {}) or {}
    newer_analysis = newer.get("analysis", {}) or {}

    older_topics = set(older_analysis.get("trending_topics", []) or [])
    newer_topics = set(newer_analysis.get("trending_topics", []) or [])

    # Exact-string match first (unchanged from before) - only topics that
    # don't match verbatim go through the keyword-overlap reconciliation
    # pass, which additionally recognizes the same theme reworded between
    # runs (e.g. "Teacher preparation & education strategies" vs "Teacher
    # preparation and professional development") without over-matching
    # genuinely unrelated topics that happen to share one generic word.
    exact_added = sorted(newer_topics - older_topics)
    exact_removed = sorted(older_topics - newer_topics)
    exact_persisted = sorted(older_topics & newer_topics)
    topics_added, topics_removed, related_matches = _reconcile_similar_topics(
        exact_added, exact_removed
    )
    topics_persisted = sorted(set(exact_persisted) | set(related_matches))

    older_accounts = {a.get("username"): a for a in older.get("accounts", []) if a.get("username")}
    newer_accounts = {a.get("username"): a for a in newer.get("accounts", []) if a.get("username")}

    movers = []
    for username in set(older_accounts) & set(newer_accounts):
        old_account = older_accounts[username]
        new_account = newer_accounts[username]
        rank_delta = int(old_account.get("rank", 0)) - int(new_account.get("rank", 0))
        score_delta = float(new_account.get("ranking_score", 0)) - float(
            old_account.get("ranking_score", 0)
        )
        movers.append(
            AccountMover(username=username, rank_delta=rank_delta, score_delta=score_delta)
        )
    movers.sort(key=lambda m: abs(m.score_delta), reverse=True)

    older_sentiment = older_analysis.get("sentiment", {}) or {}
    newer_sentiment = newer_analysis.get("sentiment", {}) or {}
    sentiment_delta = {
        key: float(newer_sentiment.get(key, 0)) - float(older_sentiment.get(key, 0))
        for key in ("positive", "neutral", "negative")
    }

    older_stats = older.get("tweet_statistics", {}) or {}
    newer_stats = newer.get("tweet_statistics", {}) or {}

    return ComparisonResult(
        older_label=_run_label(older),
        newer_label=_run_label(newer),
        older_window_label=_window_label(older),
        newer_window_label=_window_label(newer),
        topics_added=topics_added,
        topics_removed=topics_removed,
        topics_persisted=topics_persisted,
        accounts_new=sorted(set(newer_accounts) - set(older_accounts)),
        accounts_dropped=sorted(set(older_accounts) - set(newer_accounts)),
        movers=movers,
        sentiment_delta=sentiment_delta,
        tweets_collected_delta=int(newer_stats.get("tweets_collected", 0))
        - int(older_stats.get("tweets_collected", 0)),
        accounts_processed_delta=int(newer_stats.get("accounts_processed", 0))
        - int(older_stats.get("accounts_processed", 0)),
    )
