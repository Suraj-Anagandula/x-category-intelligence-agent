"""Dynamic, LLM-only candidate X account discovery.

There is intentionally **no curated/seed account list and no fallback**:
discovery is entirely Gemini-driven. If no LLM client is configured, or the
LLM call fails or returns an unusable response, `discover_candidates` raises
`LLMError` with a clear message - the caller (`CategoryIntelligenceAgent`)
lets this propagate unhandled so the CLI reports it as a clean, typed error
(via the existing `ScraperError` handling in `main.py`) instead of silently
substituting predefined accounts or crashing with a raw traceback.

The LLM is responsible only for *discovery* (candidate usernames + a reason
each was suggested) - it never supplies profile/tweet facts. Every discovered
username still goes through the real `ProfileScraper`/`TweetScraper` before
anything about the account is treated as true.
"""

from __future__ import annotations

from typing import Protocol

from app.exceptions import LLMError
from app.llm import LLMProvider
from app.logger import get_logger
from app.schemas import DiscoveredAccount
from app.utils import is_valid_username, normalize_username

logger = get_logger()


class AccountDiscoveryProvider(Protocol):
    """Interface a discovery backend must implement."""

    async def discover(
        self, category: str, keywords: list[str], limit: int = 100
    ) -> list[DiscoveredAccount]: ...


class LLMDiscoveryProvider:
    """Asks Gemini for real, currently active X handles relevant to the
    category, each with a stated reason. This is the only discovery backend
    - there is no curated/offline fallback provider."""

    def __init__(self, llm_client: LLMProvider) -> None:
        self._llm_client = llm_client

    async def discover(
        self, category: str, keywords: list[str], limit: int = 100
    ) -> list[DiscoveredAccount]:
        keyword_list = ", ".join(keywords) if keywords else category
        prompt = (
            "You are helping build a candidate list of real, currently active X "
            f"(Twitter) accounts relevant to the category '{category}'.\n"
            f"Related keywords/subcategories: {keyword_list}.\n"
            f"Return up to {limit} accounts. Respond with ONLY a single valid JSON "
            "object matching exactly this structure - no explanation, no markdown "
            "code fences, no text before or after the JSON:\n"
            '{"accounts": [{"username": "<handle, no @>", "reason": '
            '"<short reason this account is relevant>"}]}\n'
            "Only include accounts you are confident actually exist - do not "
            "invent handles. Do not include follower counts, user IDs, tweet "
            "content, or any other profile/tweet data; only username and reason."
        )
        try:
            result = await self._llm_client.generate_json(prompt)
        except LLMError as exc:
            raise LLMError(
                f"Dynamic account discovery failed. Reason: {exc}. "
                "No curated account fallback is enabled."
            ) from exc

        accounts = result.get("accounts") if isinstance(result, dict) else None
        if not isinstance(accounts, list):
            raise LLMError(
                "Dynamic account discovery failed. Reason: Gemini did not return "
                "the expected {'accounts': [...]} structure. "
                "No curated account fallback is enabled."
            )

        discovered: list[DiscoveredAccount] = []
        for item in accounts:
            if isinstance(item, dict) and isinstance(item.get("username"), str):
                discovered.append(
                    DiscoveredAccount(username=item["username"], reason=str(item.get("reason", "")))
                )
        return discovered[:limit]


async def discover_candidates(
    category: str,
    keywords: list[str],
    limit: int = 100,
    llm_client: LLMProvider | None = None,
) -> list[DiscoveredAccount]:
    """Dynamically discover a candidate pool for `category` via the LLM.

    Raises `LLMError` - with no fallback - if `llm_client` is None or the
    LLM call fails/returns an unusable response.
    """
    if llm_client is None:
        raise LLMError(
            "Dynamic account discovery failed. Reason: no LLM provider is "
            "configured (set LLM_PROVIDER and the matching GROQ_API_KEY or "
            "GEMINI_API_KEY). No curated account fallback is enabled."
        )

    discovered = await LLMDiscoveryProvider(llm_client).discover(category, keywords, limit)

    deduped: list[DiscoveredAccount] = []
    seen: set[str] = set()
    for account in discovered:
        username = normalize_username(account.username)
        if not is_valid_username(username):
            continue
        key = username.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(DiscoveredAccount(username=username, reason=account.reason))

    deduped = deduped[:limit]
    logger.info(f"Discovery produced {len(deduped)} candidate(s) for category {category!r}")
    return deduped
