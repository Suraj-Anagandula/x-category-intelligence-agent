"""Deterministic account ranking engine.

Pure functions with no I/O, fully unit-testable in isolation. Every
component score is normalized to 0-100, combined via a fixed weighted
formula, and the result is 0-100 too:

    ranking_score = 0.40 * category_relevance
                  + 0.25 * engagement_score
                  + 0.20 * activity_score
                  + 0.15 * audience_score

`category_relevance` may be produced by an LLM classifier upstream (see
`app/llm.py`) or by the deterministic keyword-overlap heuristic here - either
way it arrives as a plain float before ranking, so the weighted-sum math
itself never depends on the LLM being available.

Below `RELEVANCE_FLOOR`, engagement/activity/audience are zeroed out of the
score entirely (see `_combined_score`) - real relevance is the one signal
that actually says whether an account belongs in this category at all, and
the other three combined (0.60) outweigh it (0.40) enough that a highly
engaged but irrelevant account could otherwise still out-rank a genuinely
relevant one. Known residual limitation: X exposes no verified-identity
signal, so an LLM-discovered handle that has since been abandoned/reused by
an unrelated account can still occasionally clear the floor if its bio text
happens to read as ambiguous rather than clearly off-topic - this reduces,
but cannot fully eliminate, that class of false positive.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from app.exceptions import LLMError
from app.llm import LLMProvider
from app.models import Tweet, UserProfile
from app.schemas import CategoryContext, RankedAccount
from app.utils import dedupe_preserve_order

#: (category_relevance, engagement, activity, audience)
DEFAULT_WEIGHTS = (0.40, 0.25, 0.20, 0.15)

#: category_relevance's weight (0.40) is the single largest term, but the
#: other three sum to 0.60 - without a floor, an account with near-zero
#: real relevance to the category can still out-rank a genuinely relevant
#: one purely on engagement/activity/audience (e.g. a reused/ambiguous
#: handle like "@hhs" being suggested for "healthcare" by discovery, but
#: actually belonging to an unrelated person whose unrelated posts still
#: get ordinary engagement). Below this floor, only the relevance term
#: contributes to the score - the weighted formula itself is unchanged
#: for every account that clears the bar, so this never affects a
#: genuinely relevant, high-engagement account.
RELEVANCE_FLOOR = 20.0


def _combined_score(
    relevance: float,
    engagement: float,
    activity: float,
    audience: float,
    weights: tuple[float, float, float, float],
) -> float:
    relevance_w, engagement_w, activity_w, audience_w = weights
    if relevance < RELEVANCE_FLOOR:
        return relevance_w * relevance
    return (
        relevance_w * relevance
        + engagement_w * engagement
        + activity_w * activity
        + audience_w * audience
    )


def normalize_log(value: float | None, cap: float) -> float:
    """Log-scale `value` into 0-100, where `cap` (or above) maps to ~100.

    Using log(1+x) prevents a handful of huge accounts/values from
    completely dominating the linear range while still rewarding growth.
    """
    if value is None or value <= 0 or cap <= 0:
        return 0.0
    score = math.log1p(value) / math.log1p(cap) * 100
    return max(0.0, min(100.0, score))


def compute_audience_score(profile: UserProfile) -> float:
    """Log-normalized follower count. A single mega-account should not
    automatically dominate the ranking over a smaller, more relevant one."""
    return normalize_log(profile.followers, cap=50_000_000)


def compute_activity_score(profile: UserProfile) -> float:
    """Posting-frequency proxy: total tweets over account age in days.

    Per-tweet recency isn't available at the 100-candidate stage without
    scraping tweets for all of them (only the top N get that treatment
    later) - total-tweets-over-account-age is the honest signal available
    from a profile alone.
    """
    if not profile.tweets or not profile.created_at:
        return 0.0
    created_at = profile.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = max((datetime.now(timezone.utc) - created_at).days, 1)
    frequency = profile.tweets / age_days
    return normalize_log(frequency, cap=5.0)


def compute_engagement_score(profile: UserProfile) -> float:
    """Profile-level engagement proxy (likes given / tweets posted).

    This is a rough placeholder available for the full candidate pool before
    any tweets have been scraped. Once the top N accounts' tweets are
    fetched, `rerank_with_tweet_engagement` replaces this with a real
    per-tweet engagement measure.
    """
    if not profile.tweets:
        return 0.0
    ratio = (profile.likes or 0) / profile.tweets
    return normalize_log(ratio, cap=10.0)


def compute_category_relevance(profile: UserProfile, ctx: CategoryContext) -> float:
    """Deterministic keyword-overlap relevance between the profile's public
    text fields and the category's keywords/subcategories."""
    text = " ".join(filter(None, [profile.username, profile.display_name, profile.bio])).lower()
    vocabulary = dedupe_preserve_order(
        [term for term in [ctx.category, *ctx.keywords, *ctx.subcategories] if term]
    )
    if not vocabulary or not text:
        return 0.0

    matches = sum(1 for term in vocabulary if term.lower() in text)
    # Partial keyword coverage is normal (a bio won't mention every keyword),
    # so scale up rather than requiring near-total overlap to score highly.
    return min(100.0, (matches / len(vocabulary)) * 200)


async def compute_category_relevance_llm(
    profile: UserProfile, ctx: CategoryContext, llm_client: LLMProvider
) -> float:
    """LLM-judged relevance (0-100), falling back to the deterministic
    heuristic on any LLM failure. The returned float is then used exactly
    like any other precomputed score - no further LLM calls happen once
    ranking runs."""
    prompt = (
        f"Category: {ctx.category}\n"
        f"Keywords/subcategories: {', '.join(ctx.keywords + ctx.subcategories)}\n"
        f"Account username: {profile.username}\n"
        f"Display name: {profile.display_name or ''}\n"
        f"Bio: {profile.bio or ''}\n\n"
        "On a scale of 0-100, how relevant is this account to the category? "
        "Base your judgment ONLY on the actual display name and bio text above - "
        "a username can be misleading, outdated, or reused/reassigned to an "
        "unrelated person or organization, so never infer relevance from the "
        "username alone. If the display name and bio give no real indication "
        "the account covers this category, score it low (under 20) even if the "
        "username itself looks related.\n"
        'Return ONLY JSON: {"relevance": <number>}'
    )
    try:
        result = await llm_client.generate_json(prompt)
        score = float(result["relevance"]) if isinstance(result, dict) else None
        if score is None:
            raise LLMError("Missing 'relevance' field in LLM response")
        return max(0.0, min(100.0, score))
    except (LLMError, KeyError, TypeError, ValueError):
        return compute_category_relevance(profile, ctx)


def compute_tweet_engagement_score(tweets: list[Tweet]) -> float:
    """Real per-tweet engagement, log-normalized average across `tweets`."""
    if not tweets:
        return 0.0
    totals = [
        (tweet.like_count or 0)
        + (tweet.retweet_count or 0)
        + (tweet.reply_count or 0)
        + (tweet.view_count or 0) * 0.01
        for tweet in tweets
    ]
    average = sum(totals) / len(totals)
    return normalize_log(average, cap=5000.0)


def rank_accounts(
    profiles: list[UserProfile],
    ctx: CategoryContext,
    relevance_scores: dict[str, float] | None = None,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> list[RankedAccount]:
    """Score every profile and return them sorted descending by ranking_score."""
    relevance_scores = relevance_scores or {}

    ranked: list[RankedAccount] = []
    for profile in profiles:
        relevance = relevance_scores.get(profile.username, compute_category_relevance(profile, ctx))
        engagement = compute_engagement_score(profile)
        activity = compute_activity_score(profile)
        audience = compute_audience_score(profile)
        score = _combined_score(relevance, engagement, activity, audience, weights)
        ranked.append(
            RankedAccount(
                rank=0,
                username=profile.username,
                display_name=profile.display_name,
                followers=profile.followers,
                category_relevance=round(relevance, 2),
                engagement_score=round(engagement, 2),
                activity_score=round(activity, 2),
                audience_score=round(audience, 2),
                ranking_score=round(score, 2),
            )
        )

    ranked.sort(key=lambda account: account.ranking_score, reverse=True)
    for index, account in enumerate(ranked, start=1):
        account.rank = index
    return ranked


def select_top_n(ranked: list[RankedAccount], n: int) -> list[RankedAccount]:
    """Return the top `n` (or fewer, if the ranked pool is smaller)."""
    return ranked[:n]


def rerank_with_tweet_engagement(
    ranked: list[RankedAccount],
    tweets_by_username: dict[str, list[Tweet]],
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> list[RankedAccount]:
    """Replace the profile-level engagement proxy with real per-tweet
    engagement for accounts whose tweets were collected, and re-sort."""
    updated: list[RankedAccount] = []
    for account in ranked:
        tweets = tweets_by_username.get(account.username)
        engagement = compute_tweet_engagement_score(tweets) if tweets else account.engagement_score
        score = _combined_score(
            account.category_relevance,
            engagement,
            account.activity_score,
            account.audience_score,
            weights,
        )
        updated.append(
            account.model_copy(
                update={"engagement_score": round(engagement, 2), "ranking_score": round(score, 2)}
            )
        )

    updated.sort(key=lambda account: account.ranking_score, reverse=True)
    for index, account in enumerate(updated, start=1):
        account.rank = index
    return updated
