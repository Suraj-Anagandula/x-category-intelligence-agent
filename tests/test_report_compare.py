"""Unit tests for app.report_compare.compare_reports - pure dict-diff logic,
no I/O, no model/network dependency.
"""

from __future__ import annotations

from app.report_compare import compare_reports, topics_related


def _run(
    category="technology",
    date="2026-08-08",
    topics=None,
    accounts=None,
    sentiment=None,
    tweets_collected=0,
    accounts_processed=0,
    time_window=None,
):
    payload = {
        "category": category,
        "scraped_at": f"{date}T00:00:00+00:00",
        "accounts": accounts or [],
        "tweet_statistics": {
            "tweets_collected": tweets_collected,
            "accounts_processed": accounts_processed,
        },
        "analysis": {
            "trending_topics": topics or [],
            "sentiment": sentiment or {},
        },
    }
    if time_window is not None:
        payload["time_window"] = time_window
    return payload


def test_compare_reports_topic_diff() -> None:
    older = _run(topics=["AI regulation", "Semiconductors"])
    newer = _run(date="2026-08-10", topics=["Semiconductors", "Open Source AI"])

    result = compare_reports(older, newer)

    assert result.topics_added == ["Open Source AI"]
    assert result.topics_removed == ["AI regulation"]
    assert result.topics_persisted == ["Semiconductors"]


def test_compare_reports_account_new_and_dropped() -> None:
    older = _run(accounts=[{"username": "a", "rank": 1, "ranking_score": 80}])
    newer = _run(accounts=[{"username": "b", "rank": 1, "ranking_score": 70}])

    result = compare_reports(older, newer)

    assert result.accounts_new == ["b"]
    assert result.accounts_dropped == ["a"]


def test_compare_reports_movers_sorted_by_score_delta_magnitude() -> None:
    older = _run(
        accounts=[
            {"username": "a", "rank": 5, "ranking_score": 50},
            {"username": "b", "rank": 3, "ranking_score": 60},
        ]
    )
    newer = _run(
        accounts=[
            {"username": "a", "rank": 1, "ranking_score": 90},  # big mover
            {"username": "b", "rank": 3, "ranking_score": 61},  # tiny mover
        ]
    )

    result = compare_reports(older, newer)

    assert result.movers[0].username == "a"
    assert result.movers[0].rank_delta == 4
    assert result.movers[0].score_delta == 40


def test_compare_reports_sentiment_delta() -> None:
    older = _run(sentiment={"positive": 30, "neutral": 50, "negative": 20})
    newer = _run(sentiment={"positive": 60, "neutral": 20, "negative": 20})

    result = compare_reports(older, newer)

    assert result.sentiment_delta == {"positive": 30, "neutral": -30, "negative": 0}


def test_compare_reports_volume_deltas() -> None:
    older = _run(tweets_collected=32, accounts_processed=18)
    newer = _run(tweets_collected=81, accounts_processed=20)

    result = compare_reports(older, newer)

    assert result.tweets_collected_delta == 49
    assert result.accounts_processed_delta == 2


def test_compare_reports_labels_include_category_and_date() -> None:
    older = _run(category="technology", date="2026-08-08")
    newer = _run(category="technology", date="2026-08-10")

    result = compare_reports(older, newer)

    assert result.older_label == "technology — 2026-08-08"
    assert result.newer_label == "technology — 2026-08-10"


def test_topics_related_recognizes_reworded_same_theme() -> None:
    """Real example from the education category (2026-08-08 vs 2026-08-10):
    the same underlying "teacher preparation" theme, reworded between runs -
    must be recognized as related, not treated as two unrelated topics."""
    older_topic = "Teacher preparation & education strategies"
    newer_topic = "Teacher preparation and professional development"

    assert topics_related(older_topic, newer_topic) is True


def test_topics_related_recognizes_second_real_reworded_pair() -> None:
    """Also from the same real education comparison: "delivery" vs
    "promotion" of the same "educational content" theme."""
    assert topics_related("Educational content delivery", "Educational content promotion") is True


def test_topics_related_rejects_genuinely_unrelated_topics() -> None:
    """Real example from the same education comparison - these two topics
    share no meaningful theme and must NOT be considered related, even
    though the matching logic is lenient enough to catch reworded pairs."""
    assert topics_related("Space exploration updates", "Student engagement and retention") is False


def test_topics_related_does_not_over_match_on_one_shared_generic_word() -> None:
    """Two short topics sharing exactly one generic-but-not-a-stopword word
    ("strategies") must not be merged just because of that one word - the
    overlap has to cover a meaningful share of the shorter phrase."""
    assert (
        topics_related(
            "Research-backed learning strategies", "Teacher preparation & education strategies"
        )
        is False
    )


def test_topics_related_exact_match_is_always_related() -> None:
    assert topics_related("AI regulation", "AI regulation") is True
    assert topics_related("AI regulation", "ai regulation") is True  # case-insensitive


def test_topics_related_empty_topic_is_never_related() -> None:
    assert topics_related("", "AI regulation") is False
    assert topics_related("AI regulation", "") is False


def test_compare_reports_recognizes_reworded_topic_as_persisted() -> None:
    """End-to-end: compare_reports itself (not just topics_related in
    isolation) must classify a reworded-but-same theme as Persistent, and
    remove it from both Added and Removed."""
    older = _run(
        topics=[
            "Teacher preparation & education strategies",
            "Space exploration updates",
        ]
    )
    newer = _run(
        date="2026-08-10",
        topics=[
            "Teacher preparation and professional development",
            "Student engagement and retention",
        ],
    )

    result = compare_reports(older, newer)

    assert result.topics_persisted == ["Teacher preparation and professional development"]
    assert result.topics_added == ["Student engagement and retention"]
    assert result.topics_removed == ["Space exploration updates"]


def test_compare_reports_real_education_example_full_reconstruction() -> None:
    """The exact 5-topic sets from the real education 2026-08-08 vs
    2026-08-10 comparison - locks in the improved classification against
    the actual data this feature was built to handle correctly."""
    older = _run(
        topics=[
            "Edutopia resources & research",
            "Teacher preparation & education strategies",
            "Space exploration updates",
            "Museum reopenings & visitor info",
            "Educational content delivery",
        ]
    )
    newer = _run(
        date="2026-08-10",
        topics=[
            "Research-backed learning strategies",
            "Online course completion and certificates",
            "Teacher preparation and professional development",
            "Student engagement and retention",
            "Educational content promotion",
        ],
    )

    result = compare_reports(older, newer)

    assert result.topics_persisted == [
        "Educational content promotion",
        "Teacher preparation and professional development",
    ]
    assert result.topics_added == [
        "Online course completion and certificates",
        "Research-backed learning strategies",
        "Student engagement and retention",
    ]
    assert result.topics_removed == [
        "Edutopia resources & research",
        "Museum reopenings & visitor info",
        "Space exploration updates",
    ]


def test_compare_reports_window_labels_show_the_analyzed_period() -> None:
    older = _run(
        date="2026-08-01",
        time_window={
            "mode": "custom",
            "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-08T00:00:00+00:00",
        },
    )
    newer = _run(
        date="2026-08-08",
        time_window={
            "mode": "custom",
            "start": "2026-08-08T00:00:00+00:00",
            "end": "2026-08-15T00:00:00+00:00",
        },
    )

    result = compare_reports(older, newer)

    assert result.older_window_label == "2026-08-01 → 2026-08-08"
    assert result.newer_window_label == "2026-08-08 → 2026-08-15"


def test_compare_reports_window_label_defaults_to_latest_available_for_latest_mode_runs() -> None:
    older = _run(time_window={"mode": "latest", "start": None, "end": None})
    newer = _run(time_window={"mode": "latest", "start": None, "end": None})

    result = compare_reports(older, newer)

    assert result.older_window_label == "Latest Available"
    assert result.newer_window_label == "Latest Available"


def test_compare_reports_window_label_backward_compatible_with_snapshots_missing_the_key() -> None:
    """Section 19/16: a snapshot saved before time-window support existed
    has no "time_window" key at all - must not crash, must read as
    "Latest Available" (the same meaning as an explicit "latest" mode)."""
    older = _run()  # no time_window key at all
    newer = _run()

    result = compare_reports(older, newer)

    assert result.older_window_label == "Latest Available"
    assert result.newer_window_label == "Latest Available"
    # And the existing, already-verified-correct arithmetic is untouched.
    assert result.tweets_collected_delta == 0
    assert result.accounts_processed_delta == 0


def test_compare_reports_handles_missing_fields_without_crashing() -> None:
    """A pre-existing snapshot (e.g. the real 2026-08-08 fixtures) may
    predate fields like discovery_reason - comparison must not assume
    every stored dict has every key."""
    older = {"category": "technology", "scraped_at": "2026-08-08T00:00:00+00:00"}
    newer = {"category": "technology", "scraped_at": "2026-08-10T00:00:00+00:00"}

    result = compare_reports(older, newer)

    assert result.topics_added == []
    assert result.accounts_new == []
    assert result.movers == []
    assert result.tweets_collected_delta == 0
