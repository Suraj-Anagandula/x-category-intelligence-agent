"""Story Opportunities: signals worth a journalist's attention, derived
entirely from already-computed scores/topics - never invented. Pure
functions, no I/O, matching `app/account_ranker.py`'s convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.report_compare import ComparisonResult
from app.signal_score import compute_confidence
from app.topic_matching import topic_matches

#: A topic only becomes a story opportunity once it clears both bars - a
#: high enough signal score AND evidence strong enough that "Low"
#: confidence doesn't undermine it. Neither threshold is a new metric -
#: both reuse `app.signal_score.compute_confidence`'s existing 0-100 scale.
SIGNAL_SCORE_THRESHOLD = 60.0


@dataclass
class StoryOpportunity:
    title: str
    why_it_matters: str
    #: `None` for opportunities derived from a directly observed fact
    #: (e.g. a rank change) rather than an evidence-count-based signal -
    #: never a fabricated placeholder score.
    signal_score: float | None = None
    confidence_label: str | None = None
    evidence_tweet_ids: list[str] = field(default_factory=list)
    account_usernames: list[str] = field(default_factory=list)
    kind: Literal["signal", "new_topic", "rank_shift"] = "signal"


def _topic_opportunity(
    topic: str, tweets: list[dict], kind: str = "signal"
) -> StoryOpportunity | None:
    """Build one opportunity for `topic` from real matching tweets in
    `tweets`, or `None` if it doesn't clear the signal/confidence bar.

    A topic's "signal score" has no per-account relevance/engagement/
    audience fields to combine (those are account-level, computed by
    `app.signal_score.compute_signal_score`) - a topic's strength is
    instead grounded directly in how much real evidence backs it, so
    `compute_confidence`'s own 0-100 scale is reused rather than inventing
    a second scoring formula for the same underlying evidence.
    """
    matching = [t for t in tweets if topic_matches(t.get("text") or "", topic)]
    authors = sorted({t.get("username") for t in matching if t.get("username")})
    confidence_label, confidence_score = compute_confidence(len(matching), len(authors))

    if confidence_score < SIGNAL_SCORE_THRESHOLD or confidence_label == "Low":
        return None

    return StoryOpportunity(
        title=topic,
        why_it_matters=(
            f"{len(matching)} post(s) from {len(authors)} independent account(s) are "
            f"discussing this - confidence {confidence_label} ({confidence_score:.0f}/100)."
        ),
        signal_score=confidence_score,
        confidence_label=confidence_label,
        evidence_tweet_ids=[t.get("id") for t in matching if t.get("id")],
        account_usernames=authors,
        kind=kind,
    )


def derive_story_opportunities(
    report: dict, comparison: ComparisonResult | None = None
) -> list[StoryOpportunity]:
    """Build a list of `StoryOpportunity` from a run's real trending topics
    and, if given, a `ComparisonResult` from `app.report_compare` (new
    topics and the biggest rank movers between two dated runs of the same
    category). Never derived from an invented metric - see
    `_topic_opportunity` and the comparison-based branches below.
    """
    tweets = report.get("tweets", [])
    analysis = report.get("analysis", {}) or {}
    topics = analysis.get("trending_topics", []) or []

    opportunities: list[StoryOpportunity] = []
    for topic in topics:
        opportunity = _topic_opportunity(topic, tweets, kind="signal")
        if opportunity is not None:
            opportunities.append(opportunity)

    if comparison is not None:
        for topic in comparison.topics_added:
            opportunity = _topic_opportunity(topic, tweets, kind="new_topic")
            if opportunity is not None:
                opportunity.why_it_matters = (
                    f"New this run - didn't appear in {comparison.older_label}. "
                    + opportunity.why_it_matters
                )
                opportunities.append(opportunity)

        for mover in comparison.movers[:3]:
            if mover.score_delta <= 0:
                continue
            opportunities.append(
                StoryOpportunity(
                    title=f"@{mover.username} is rising fast",
                    why_it_matters=(
                        f"@{mover.username}'s ranking score changed by {mover.score_delta:+.1f} "
                        f"(rank moved by {mover.rank_delta:+d}) between {comparison.older_label} "
                        f"and {comparison.newer_label} - a directly observed change, not an "
                        f"inferred signal."
                    ),
                    account_usernames=[mover.username],
                    kind="rank_shift",
                )
            )

    return opportunities
