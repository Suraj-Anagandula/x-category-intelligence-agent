"""Category understanding and the end-to-end category-intelligence pipeline.

`CategoryAgent` owns normalization and keyword/subcategory generation
(spec section 6: "coordinate the account-discovery process" by producing the
context discovery/ranking need). `CategoryIntelligenceAgent` is the workflow
coordinator (spec section 20), wiring together discovery, the *existing*
`ProfileScraper` (validation), `account_ranker`, `TweetScraper`, `storage`,
and `analysis` - reusing every piece of existing scraping infrastructure
rather than re-implementing it.

No LangGraph/typed-graph-state here for the MVP (spec sections 21-22 are
explicitly conditional on LangGraph being used) - a plain async pipeline is
consistent with the rest of this codebase and simpler to test; it can be
swapped in later without touching the other modules.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from app.account_discovery import discover_candidates
from app.account_ranker import (
    compute_category_relevance_llm,
    rank_accounts,
    rerank_with_tweet_engagement,
    select_top_n,
)
from app.analysis import analyze_category
from app.config import Settings
from app.exceptions import LLMError
from app.llm import LLMProvider
from app.logger import get_logger
from app.models import Tweet
from app.schemas import CategoryContext, CategoryReport, TimeWindowInfo, TweetStatistics
from app.scraper import ProfileScraper
from app.storage import save_category_run
from app.time_window import TimeWindow, filter_tweets_to_window, resolve_time_window
from app.tweet_scraper import TweetScraper

logger = get_logger()

#: Static fallback context for the 10 predefined categories, used whenever no
#: LLM is configured (or the LLM call fails) - keeps custom categories and
#: tests fully functional without any external dependency (spec Rule 7).
PREDEFINED_CATEGORIES: dict[str, CategoryContext] = {
    "sports": CategoryContext(
        category="sports",
        subcategories=["cricket", "football", "basketball", "tennis", "formula 1"],
        keywords=["sports", "cricket", "football", "athletes", "teams", "championship"],
    ),
    "politics": CategoryContext(
        category="politics",
        subcategories=["elections", "government", "policy", "diplomacy", "legislation"],
        keywords=["politics", "election", "government", "policy", "senate", "president"],
    ),
    "technology": CategoryContext(
        category="technology",
        subcategories=["ai", "software", "startups", "gadgets", "cybersecurity"],
        keywords=["technology", "tech", "software", "ai", "startup", "innovation"],
    ),
    "business": CategoryContext(
        category="business",
        subcategories=["markets", "startups", "economy", "entrepreneurship", "trade"],
        keywords=["business", "market", "economy", "company", "industry", "entrepreneur"],
    ),
    "entertainment": CategoryContext(
        category="entertainment",
        subcategories=["movies", "music", "television", "celebrities", "streaming"],
        keywords=["entertainment", "movie", "music", "celebrity", "film", "show"],
    ),
    "news": CategoryContext(
        category="news",
        subcategories=["world news", "breaking news", "local news", "investigations"],
        keywords=["news", "breaking", "report", "coverage", "headline"],
    ),
    "science": CategoryContext(
        category="science",
        subcategories=["space", "biology", "physics", "climate", "research"],
        keywords=["science", "research", "study", "discovery", "space", "climate"],
    ),
    "education": CategoryContext(
        category="education",
        subcategories=["higher education", "edtech", "teaching", "online learning"],
        keywords=["education", "school", "university", "learning", "teaching", "student"],
    ),
    "gaming": CategoryContext(
        category="gaming",
        subcategories=["esports", "console gaming", "pc gaming", "mobile gaming"],
        keywords=["gaming", "game", "esports", "console", "playstation", "xbox"],
    ),
    "finance": CategoryContext(
        category="finance",
        subcategories=["stocks", "crypto", "investing", "personal finance", "banking"],
        keywords=["finance", "stock", "market", "investing", "crypto", "economy"],
    ),
}


def normalize_category(raw: str) -> str:
    """Normalize a raw category string: strip whitespace and lowercase.

    "Sports" / "SPORTS" / " sports " all normalize to "sports".
    """
    return raw.strip().lower()


class CategoryAgent:
    """Builds a `CategoryContext` (keywords/subcategories) for a category."""

    def __init__(self, llm_client: LLMProvider | None = None) -> None:
        self.llm_client = llm_client

    async def build_context(self, category: str) -> CategoryContext:
        normalized = normalize_category(category)

        if self.llm_client is not None:
            try:
                return await self._build_context_llm(normalized)
            except LLMError as exc:
                logger.warning(
                    f"LLM category-context generation failed for {normalized!r}, "
                    f"using fallback: {exc}"
                )

        return self._build_context_fallback(normalized)

    async def _build_context_llm(self, category: str) -> CategoryContext:
        prompt = (
            f"Generate discovery context for the X (Twitter) content category "
            f"'{category}'.\n"
            'Return ONLY JSON: {"subcategories": [<up to 6 short subcategory strings>], '
            '"keywords": [<up to 10 short keyword strings>]}'
        )
        result = await self.llm_client.generate_json(prompt)
        if not isinstance(result, dict):
            raise LLMError("Expected a JSON object from the category-context prompt")

        return CategoryContext(
            category=category,
            subcategories=[str(item) for item in result.get("subcategories", [])][:6],
            keywords=[str(item) for item in result.get("keywords", [])][:10],
        )

    def _build_context_fallback(self, category: str) -> CategoryContext:
        if category in PREDEFINED_CATEGORIES:
            return PREDEFINED_CATEGORIES[category]

        words = [word for word in re.split(r"[\s,]+", category) if word]
        return CategoryContext(category=category, subcategories=[], keywords=words or [category])


class CategoryIntelligenceAgent:
    """Coordinates the full category-intelligence pipeline.

    Every I/O dependency (profile scraper, tweet scraper, LLM client) is
    injected so the whole pipeline is testable with stubs and never needs
    real X/LLM credentials.
    """

    def __init__(
        self,
        settings: Settings,
        profile_scraper: ProfileScraper,
        tweet_scraper: TweetScraper,
        llm_client: LLMProvider | None = None,
        rag_indexer: Callable[[list[Tweet], str, object], int] | None = None,
    ) -> None:
        self.settings = settings
        self.profile_scraper = profile_scraper
        self.tweet_scraper = tweet_scraper
        self.llm_client = llm_client
        self.category_agent = CategoryAgent(llm_client)
        #: Optional `(tweets, category, scraped_at) -> count_indexed`
        #: callable - see `app.rag.indexer.index_tweets` for the real
        #: implementation. `None` (the default) means Ask Intelligence
        #: indexing is skipped entirely; every existing caller that
        #: constructs this class without it keeps working unmodified.
        self.rag_indexer = rag_indexer

    async def run_pipeline(
        self,
        category: str,
        candidate_limit: int | None = None,
        top_n: int | None = None,
        tweets_per_account: int | None = None,
        *,
        on_stage: Callable[[str, dict], None] | None = None,
        time_window: TimeWindow | None = None,
    ) -> CategoryReport:
        candidate_limit = candidate_limit or self.settings.category_candidate_limit
        top_n = top_n or self.settings.top_accounts_limit
        tweets_per_account = tweets_per_account or self.settings.tweets_per_account
        #: Defaults to "latest" (no filtering) - every existing caller that
        #: doesn't pass `time_window` keeps the original "most recent
        #: available tweets" behavior unchanged.
        time_window = time_window or resolve_time_window("latest")
        errors: list[dict[str, str]] = []

        def _emit(stage_key: str, payload: dict) -> None:
            if on_stage is not None:
                on_stage(stage_key, payload)

        normalized = normalize_category(category)
        logger.info(f"Category: {normalized}")
        logger.info(f"Category intelligence pipeline started for {normalized!r}")

        ctx = await self.category_agent.build_context(normalized)
        logger.info(
            f"Category context for {normalized!r}: {len(ctx.keywords)} keyword(s), "
            f"{len(ctx.subcategories)} subcategory(ies)"
        )
        _emit(
            "context",
            {"keywords": len(ctx.keywords), "subcategories": len(ctx.subcategories)},
        )

        logger.info(f"Dynamic discovery started for {normalized!r}")
        discovered = await discover_candidates(
            normalized, ctx.keywords + ctx.subcategories, candidate_limit, self.llm_client
        )
        candidates = [account.username for account in discovered]
        discovery_reasons = {account.username: account.reason for account in discovered}
        logger.info(f"Candidates discovered: {len(candidates)} (requested up to {candidate_limit})")
        _emit("discovery", {"candidates_discovered": len(candidates)})

        logger.info("Profile validation started")
        scrape_results = await self.profile_scraper.scrape_many(candidates)
        valid_profiles = [r.profile for r in scrape_results if r.success and r.profile]
        for result in scrape_results:
            if not result.success:
                errors.append(
                    {
                        "stage": "validation",
                        "username": result.username,
                        "error_type": result.error_type or "",
                        "error": result.error or "",
                    }
                )
        logger.info(f"Profiles validated: {len(valid_profiles)}/{len(scrape_results)}")
        _emit(
            "validation",
            {"profiles_validated": len(valid_profiles), "profiles_attempted": len(scrape_results)},
        )

        relevance_scores: dict[str, float] = {}
        if self.llm_client is not None:
            for profile in valid_profiles:
                relevance_scores[profile.username] = await compute_category_relevance_llm(
                    profile, ctx, self.llm_client
                )

        logger.info("Ranking started")
        ranked = rank_accounts(valid_profiles, ctx, relevance_scores)
        top_accounts = select_top_n(ranked, top_n)
        logger.info(f"Accounts selected: {len(top_accounts)}/{top_n}")
        _emit("ranking", {"accounts_selected": len(top_accounts), "top_n_requested": top_n})

        top_usernames = [account.username for account in top_accounts]
        logger.info(
            f"Starting tweet scraping: {len(top_usernames)} account(s), "
            f"{tweets_per_account} tweet(s)/account"
            + (f", time window={time_window.mode}" if time_window.is_filtered else "")
        )
        tweet_results = await self.tweet_scraper.scrape_many(
            top_usernames, count=tweets_per_account, window=time_window
        )

        all_tweets: list[Tweet] = []
        tweets_by_username: dict[str, list[Tweet]] = {}
        accounts_failed = 0
        accounts_rate_limited = 0
        for result in tweet_results:
            if result.success:
                all_tweets.extend(result.tweets)
                tweets_by_username[result.username] = result.tweets
            else:
                accounts_failed += 1
                if result.error_type == "RateLimitError":
                    accounts_rate_limited += 1
                errors.append(
                    {
                        "stage": "tweets",
                        "username": result.username,
                        "error_type": result.error_type or "",
                        "error": result.error or "",
                    }
                )

        # Apply the actual created_at time-window filter here, once, before
        # anything downstream (ranking, analysis, storage, RAG indexing)
        # ever sees the tweets - "latest" mode is a no-op (returns the same
        # tweets), so this line changes nothing when no window was selected.
        posts_fetched = len(all_tweets)
        if time_window.is_filtered:
            all_tweets = filter_tweets_to_window(all_tweets, time_window)
            tweets_by_username = {
                username: filter_tweets_to_window(tweets, time_window)
                for username, tweets in tweets_by_username.items()
            }
        posts_in_window = len(all_tweets)
        if time_window.is_filtered:
            logger.info(
                f"Time window {time_window.mode!r}: {posts_in_window}/{posts_fetched} "
                "fetched post(s) fell inside the requested range"
            )

        top_accounts = rerank_with_tweet_engagement(top_accounts, tweets_by_username)
        top_accounts = [
            account.model_copy(
                update={"discovery_reason": discovery_reasons.get(account.username, "")}
            )
            for account in top_accounts
        ]

        stats = TweetStatistics(
            accounts_processed=len(top_accounts) - accounts_failed,
            accounts_failed=accounts_failed,
            accounts_rate_limited=accounts_rate_limited,
            accounts_failed_other=accounts_failed - accounts_rate_limited,
            tweets_collected=len(all_tweets),
        )
        logger.info(
            "Tweet scrape summary:\n"
            f"  Accounts requested: {len(top_accounts)}\n"
            f"  Accounts succeeded: {stats.accounts_processed}\n"
            f"  Accounts rate limited: {stats.accounts_rate_limited}\n"
            f"  Accounts failed for other reasons: {stats.accounts_failed_other}\n"
            f"  Tweets collected: {stats.tweets_collected}"
        )
        _emit(
            "tweets",
            {
                "tweets_collected": stats.tweets_collected,
                "accounts_failed": accounts_failed,
                "accounts_rate_limited": accounts_rate_limited,
                "posts_fetched": posts_fetched,
                "posts_in_window": posts_in_window,
            },
        )

        logger.info(f"Analysis started for {normalized!r}")
        analysis = await analyze_category(normalized, all_tweets, self.llm_client)
        logger.info("Analysis completed")
        _emit("analysis", {"trending_topics": len(analysis.trending_topics)})

        report = CategoryReport(
            category=normalized,
            accounts=top_accounts,
            tweet_statistics=stats,
            analysis=analysis,
            errors=errors,
            time_window=TimeWindowInfo(
                mode=time_window.mode,
                start=time_window.start,
                end=time_window.end,
                posts_fetched=posts_fetched,
                posts_in_window=posts_in_window,
            ),
        )

        top_username_set = set(top_usernames)
        top_profiles = [
            profile for profile in valid_profiles if profile.username in top_username_set
        ]
        await save_category_run(
            normalized,
            report,
            top_profiles,
            all_tweets,
            self.settings,
            discovery_reasons,
            index_fn=self.rag_indexer,
        )

        category_csv_path = self.settings.csv_output_dir / f"{normalized}_tweets.csv"
        logger.info(f"CSV: {category_csv_path}")
        _emit("export", {"csv_path": str(category_csv_path)})

        return report
