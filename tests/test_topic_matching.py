"""Unit tests for app.topic_matching.

The key behavior under test: `analysis.trending_topics` entries are often
short LLM-abstracted phrases that never appear verbatim in any tweet's
text - matching must fall back to significant-word overlap, not just a
literal full-phrase substring, or every signal card/story opportunity would
show zero supporting posts against real data.
"""

from __future__ import annotations

from app.topic_matching import (
    count_topic_mentions,
    distinct_authors_for_topic,
    group_tweets_by_topic,
    topic_matches,
)


def _tweet(text: str, username: str = "someone") -> dict:
    return {"text": text, "username": username}


def test_count_topic_mentions_matches_full_phrase() -> None:
    tweets = [_tweet("Big news on AI regulation today")]

    assert count_topic_mentions(tweets, "AI regulation") == 1


def test_count_topic_mentions_matches_on_significant_word_when_phrase_absent() -> None:
    """The exact phrase "AI strategy and policy" never appears verbatim,
    but a tweet mentioning "strategy" or "policy" should still count."""
    tweets = [
        _tweet("Company X announces new AI strategy for 2026"),
        _tweet("Completely unrelated tweet about sports"),
    ]

    assert count_topic_mentions(tweets, "AI strategy and policy") == 1


def test_count_topic_mentions_ignores_stopwords_only_topic() -> None:
    tweets = [_tweet("this is just a normal tweet with no signal")]

    # A topic made entirely of short/stopwords must not match everything.
    assert count_topic_mentions(tweets, "in the and of") == 0


def test_count_topic_mentions_zero_when_no_match() -> None:
    tweets = [_tweet("nothing relevant here")]

    assert count_topic_mentions(tweets, "Quantum computing advances") == 0


def test_distinct_authors_for_topic_counts_unique_usernames() -> None:
    tweets = [
        _tweet("quantum computing breakthrough", username="alice"),
        _tweet("more on quantum computing", username="alice"),
        _tweet("quantum computing news", username="bob"),
    ]

    assert distinct_authors_for_topic(tweets, "quantum computing") == 2


def test_group_tweets_by_topic_places_unmatched_under_uncategorized() -> None:
    tweets = [_tweet("AI strategy news"), _tweet("totally unrelated content")]

    grouped = group_tweets_by_topic(tweets, ["AI strategy and policy"])

    assert len(grouped["AI strategy and policy"]) == 1
    assert len(grouped["Uncategorized"]) == 1


def test_group_tweets_by_topic_omits_empty_topics_and_uncategorized() -> None:
    tweets = [_tweet("AI strategy news")]

    grouped = group_tweets_by_topic(tweets, ["AI strategy and policy", "Unrelated Topic Here"])

    assert "Unrelated Topic Here" not in grouped
    assert "Uncategorized" not in grouped


def test_group_tweets_by_topic_tweet_can_match_multiple_topics() -> None:
    tweets = [_tweet("AI strategy and quantum computing both mentioned here")]

    grouped = group_tweets_by_topic(tweets, ["AI strategy", "quantum computing"])

    assert len(grouped["AI strategy"]) == 1
    assert len(grouped["quantum computing"]) == 1


def test_topic_matches_rejects_eclipse_post_for_indigenous_womens_care_topic() -> None:
    """Real reported false positive: a solar-eclipse eye-safety post must
    not be treated as evidence for an unrelated healthcare-equity topic
    just because both happen to use the single generic word "care"."""
    eclipse_text = (
        "Protect your eyes during tomorrow's solar eclipse! Wear certified "
        "eclipse glasses and take care to avoid direct viewing - eye safety "
        "matters. #SolarEclipse #EyeSafety"
    )

    assert topic_matches(eclipse_text, "Indigenous women's inclusive care") is False


def test_topic_matches_accepts_genuinely_relevant_post_for_its_topic() -> None:
    relevant_text = (
        "New report highlights gaps in inclusive care access for Indigenous "
        "women in rural clinics"
    )

    assert topic_matches(relevant_text, "Indigenous women's inclusive care") is True


def test_topic_matches_rejects_unrelated_post_sharing_only_the_broad_category() -> None:
    """A generic healthcare post must not be attached to a specific
    multi-keyword topic just because both are broadly "healthcare" - real
    overlap with the topic's own significant words is required, not a
    single incidental substring match ("health" inside "telehealth")."""
    unrelated_text = (
        "New study shows benefits of telehealth apps for rural patients this flu season"
    )

    assert topic_matches(unrelated_text, "Tribal health infrastructure funding") is False


def test_topic_matches_still_matches_short_topics_on_a_single_keyword() -> None:
    """Topics with only 1-2 significant keywords keep the original
    single-keyword-match behavior even when the full phrase never appears
    verbatim - real trending topics are frequently this short, and
    requiring 2+ hits here would silently zero out most signal cards/story
    opportunities against real data (see the "AI strategy and policy" case
    below - the topic phrase itself never appears in the tweet)."""
    assert (
        topic_matches("Company X announces new AI strategy for 2026", "AI strategy and policy")
        is True
    )
