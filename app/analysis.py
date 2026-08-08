"""Category-level tweet analysis.

`most_discussed_accounts` and `top_engagement_tweets` are pure aggregations
over collected tweets - always deterministic, never LLM-derived, since they
describe real data rather than requiring interpretation. `trending_topics`
and `sentiment` are qualitative judgments: an LLM (grounded in the actual
tweet texts, never fabricating facts) is used when configured, with a
deterministic fallback so the pipeline still produces a full report without
one (spec Rule 7 - tests must not require an LLM/network either).
"""

from __future__ import annotations

import re
from collections import Counter

from app.exceptions import LLMError
from app.llm import LLMProvider
from app.logger import get_logger
from app.models import Tweet
from app.schemas import CategoryAnalysis, SentimentBreakdown

logger = get_logger()

_HASHTAG_RE = re.compile(r"#(\w+)")
_WORD_RE = re.compile(r"[A-Za-z']{4,}")
_STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "just",
    "your",
    "about",
    "into",
    "will",
    "they",
    "them",
    "what",
    "when",
    "where",
    "which",
    "there",
    "their",
    "some",
    "more",
    "than",
    "then",
    "over",
    "here",
    "very",
    "would",
    "could",
    "should",
    "been",
    "were",
    "https",
}

_POSITIVE_WORDS = {
    "great",
    "win",
    "wins",
    "won",
    "amazing",
    "best",
    "excellent",
    "good",
    "love",
    "happy",
    "excited",
    "awesome",
    "fantastic",
    "congrats",
    "success",
    "positive",
    "beautiful",
    "incredible",
    "impressive",
    "victory",
}
_NEGATIVE_WORDS = {
    "bad",
    "worst",
    "lose",
    "loses",
    "lost",
    "terrible",
    "awful",
    "hate",
    "sad",
    "angry",
    "fail",
    "failure",
    "disappointing",
    "scandal",
    "crisis",
    "negative",
    "injury",
    "injured",
    "controversy",
    "criticism",
    "concern",
}


def most_discussed_accounts(tweets: list[Tweet], n: int = 10) -> list[str]:
    """Accounts with the most collected tweets, most-discussed first."""
    counts = Counter(tweet.username for tweet in tweets if tweet.username)
    return [username for username, _ in counts.most_common(n)]


def top_engagement_tweets(tweets: list[Tweet], n: int = 10) -> list[Tweet]:
    """Tweets sorted by total engagement (likes+retweets+replies+views), descending."""

    def engagement(tweet: Tweet) -> int:
        return (
            (tweet.like_count or 0)
            + (tweet.retweet_count or 0)
            + (tweet.reply_count or 0)
            + (tweet.view_count or 0)
        )

    return sorted(tweets, key=engagement, reverse=True)[:n]


def trending_topics_fallback(tweets: list[Tweet], n: int = 5) -> list[str]:
    """Deterministic hashtag/word-frequency extraction (no LLM)."""
    hashtags = Counter()
    words = Counter()
    for tweet in tweets:
        text = tweet.text or ""
        hashtags.update(tag.lower() for tag in _HASHTAG_RE.findall(text))
        words.update(
            word.lower() for word in _WORD_RE.findall(text) if word.lower() not in _STOPWORDS
        )

    if hashtags:
        return [f"#{tag}" for tag, _ in hashtags.most_common(n)]
    return [word for word, _ in words.most_common(n)]


def sentiment_fallback(tweets: list[Tweet]) -> SentimentBreakdown:
    """Deterministic lexicon-based sentiment split (no LLM/external dependency)."""
    if not tweets:
        return SentimentBreakdown(positive=0.0, neutral=0.0, negative=0.0)

    positive = neutral = negative = 0
    for tweet in tweets:
        text = (tweet.text or "").lower()
        pos_hits = sum(1 for word in _POSITIVE_WORDS if word in text)
        neg_hits = sum(1 for word in _NEGATIVE_WORDS if word in text)
        if pos_hits > neg_hits:
            positive += 1
        elif neg_hits > pos_hits:
            negative += 1
        else:
            neutral += 1

    total = len(tweets)
    return SentimentBreakdown(
        positive=round(positive / total * 100, 1),
        neutral=round(neutral / total * 100, 1),
        negative=round(negative / total * 100, 1),
    )


def _deterministic_summary(
    category: str, tweets: list[Tweet], topics: list[str], accounts: list[str]
) -> str:
    account_count = len({t.username for t in tweets if t.username})
    topic_text = ", ".join(topics[:3]) if topics else "no clear recurring topics"
    account_text = ", ".join(f"@{a}" for a in accounts[:3]) if accounts else "no single account"
    return (
        f"Collected {len(tweets)} tweet(s) from {account_count} account(s) in the "
        f"'{category}' category. Most active discussion centers on {topic_text}, "
        f"with {account_text} posting most frequently."
    )


async def analyze_category(
    category: str, tweets: list[Tweet], llm_client: LLMProvider | None = None
) -> CategoryAnalysis:
    """Produce a `CategoryAnalysis` for `tweets`, using an LLM when available."""
    discussed = most_discussed_accounts(tweets)
    top_tweets = top_engagement_tweets(tweets)

    if llm_client is not None and tweets:
        sample = [{"username": t.username, "text": t.text} for t in tweets[:60] if t.text]
        prompt = (
            f"You are analyzing real X (Twitter) posts collected for the category "
            f"'{category}'. Base your answer only on the posts below - do not invent "
            f"facts not present in them.\n\nPosts:\n{sample}\n\n"
            'Return ONLY JSON: {"trending_topics": [<up to 5 short topic strings>], '
            '"sentiment": {"positive": <pct 0-100>, "neutral": <pct 0-100>, "negative": <pct 0-100>}, '
            '"summary": "<2-3 sentence natural-language summary>"}'
        )
        try:
            result = await llm_client.generate_json(prompt)
            sentiment_raw = result.get("sentiment", {}) if isinstance(result, dict) else {}
            return CategoryAnalysis(
                trending_topics=list(result.get("trending_topics", []))[:5],
                sentiment=SentimentBreakdown(
                    positive=float(sentiment_raw.get("positive", 0.0)),
                    neutral=float(sentiment_raw.get("neutral", 0.0)),
                    negative=float(sentiment_raw.get("negative", 0.0)),
                ),
                most_discussed_accounts=discussed,
                high_engagement_tweets=top_tweets,
                summary=str(result.get("summary", "")),
            )
        except (LLMError, TypeError, ValueError, KeyError) as exc:
            logger.warning(
                f"LLM analysis failed for {category!r}, using deterministic fallback: {exc}"
            )

    topics = trending_topics_fallback(tweets)
    sentiment = sentiment_fallback(tweets)
    return CategoryAnalysis(
        trending_topics=topics,
        sentiment=sentiment,
        most_discussed_accounts=discussed,
        high_engagement_tweets=top_tweets,
        summary=_deterministic_summary(category, tweets, topics, discussed),
    )
