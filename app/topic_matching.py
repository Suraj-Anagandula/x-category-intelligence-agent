"""Keyword-overlap matching between a trending topic and tweet text - pure
functions, no I/O, matching `app/account_ranker.py`'s convention.

Lives under `app/` (not `ui/`) because both the UI's evidence-grouping/
signal-card display AND `app/story_opportunities.py` (backend) need it -
`app/` must never depend on `ui/`, so this stays the single source of truth
and `ui/utils.py` re-exports from here for existing call sites.
"""

from __future__ import annotations

#: Common stopwords excluded when extracting a topic's significant keywords
#: below - short/structural words like these would otherwise "match" almost
#: any tweet and make the topic-matching heuristic meaningless.
_TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def topic_keywords(topic: str) -> list[str]:
    """Significant words (length > 3, stopwords excluded) extracted from a
    trending-topic phrase, lowercased.

    `analysis.trending_topics` entries are often short LLM-abstracted
    phrases (e.g. "AI strategy and policy") that rarely appear verbatim in
    any single tweet's text - matching on the whole phrase as one substring
    would silently match nothing on real data. Matching on any of a topic's
    significant words is still a plain keyword heuristic (not a
    classifier), but actually finds the tweets that motivated the topic.
    """
    words = [w.strip(".,!?:;\"'()").lower() for w in topic.split()]
    return [w for w in words if len(w) > 3 and w not in _TOPIC_STOPWORDS]


#: For a topic phrase with this many significant keywords or more, a
#: single shared word is not enough evidence that a tweet actually
#: supports it - e.g. "Indigenous women's inclusive care" sharing only its
#: generic "care" keyword with an unrelated solar-eclipse eye-safety post
#: must not count as a match. Topics with only 1-2 keywords keep the
#: original single-keyword-match behavior (there's no larger set to
#: require more overlap from, and real trending topics are frequently
#: exactly this short - see tests/test_topic_matching.py's "AI strategy
#: and policy" case).
_MULTI_KEYWORD_MATCH_THRESHOLD = 3
_MIN_KEYWORD_HITS_ABOVE_THRESHOLD = 2


def topic_matches(text: str, topic: str) -> bool:
    """A tweet's `text` "matches" `topic` if it contains the full phrase
    (case insensitive - handles short/single-word topics, including the
    deterministic hashtag-based fallback in `app/analysis.py`), or its
    significant keywords overlap enough (see `_MULTI_KEYWORD_MATCH_THRESHOLD`
    above) - one keyword for a short (1-2 keyword) topic, at least two for
    a longer one, so a single broad/generic word shared with an otherwise
    unrelated topic can't attach a post as "evidence" for it."""
    if not topic:
        return False
    lowered = text.lower()
    if topic.lower() in lowered:
        return True
    keywords = topic_keywords(topic)
    if not keywords:
        return False
    required_hits = (
        _MIN_KEYWORD_HITS_ABOVE_THRESHOLD if len(keywords) >= _MULTI_KEYWORD_MATCH_THRESHOLD else 1
    )
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return hits >= required_hits


def count_topic_mentions(tweets: list[dict], topic: str) -> int:
    """Count of how many `tweets` mention `topic` - not a backend score,
    just a display/scoring helper. See `topic_matches` for the matching rule."""
    return sum(1 for t in tweets if topic_matches(t.get("text") or "", topic))


def distinct_authors_for_topic(tweets: list[dict], topic: str) -> int:
    """Count of distinct usernames among tweets mentioning `topic` - the
    "independent account" input `app.signal_score.compute_confidence`
    expects, using the same matching rule as `count_topic_mentions`."""
    authors = {
        t.get("username")
        for t in tweets
        if t.get("username") and topic_matches(t.get("text") or "", topic)
    }
    return len(authors)


def group_tweets_by_topic(tweets: list[dict], topics: list[str]) -> dict[str, list[dict]]:
    """Group `tweets` by which of `topics` they mention (see
    `topic_matches` for the matching rule) - an honest v1, not a
    classifier. A tweet matching more than one topic appears under each;
    tweets matching none land under "Uncategorized". Topic order is
    preserved; "Uncategorized" is always last and only included when
    non-empty.
    """
    grouped: dict[str, list[dict]] = {topic: [] for topic in topics}
    uncategorized: list[dict] = []

    for tweet in tweets:
        text = tweet.get("text") or ""
        matched = False
        for topic in topics:
            if topic_matches(text, topic):
                grouped[topic].append(tweet)
                matched = True
        if not matched:
            uncategorized.append(tweet)

    grouped = {topic: items for topic, items in grouped.items() if items}
    if uncategorized:
        grouped["Uncategorized"] = uncategorized
    return grouped
