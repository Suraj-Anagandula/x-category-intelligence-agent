"""Unit tests for app.account_discovery: LLM-only discovery, dedupe, and
the no-curated-fallback error contract.

Note: discovery has no curated/seed fallback by design - these tests only
cover what still applies (structured LLM output parsing, dedupe/normalize,
limit handling, and the explicit LLMError contract when no LLM is
configured or the LLM call fails).
"""

from __future__ import annotations

from app.account_discovery import LLMDiscoveryProvider, discover_candidates
from app.exceptions import LLMError


class _StubLLMClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.prompts: list[str] = []

    async def generate_json(self, prompt: str):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.result


async def test_llm_discovery_provider_returns_username_and_reason() -> None:
    client = _StubLLMClient(
        result={
            "accounts": [
                {"username": "espn", "reason": "Major sports news outlet"},
                {"username": "@fifacom", "reason": "Official football body"},
            ]
        }
    )
    provider = LLMDiscoveryProvider(client)

    result = await provider.discover("sports", ["football"], limit=10)

    assert [a.username for a in result] == ["espn", "@fifacom"]
    assert result[0].reason == "Major sports news outlet"


async def test_llm_discovery_provider_rejects_non_dict_response() -> None:
    client = _StubLLMClient(result=["not", "the", "right", "shape"])
    provider = LLMDiscoveryProvider(client)

    try:
        await provider.discover("sports", [], limit=10)
        raise AssertionError("expected LLMError")
    except LLMError:
        pass


async def test_llm_discovery_provider_rejects_missing_accounts_key() -> None:
    client = _StubLLMClient(result={"not_accounts": []})
    provider = LLMDiscoveryProvider(client)

    try:
        await provider.discover("sports", [], limit=10)
        raise AssertionError("expected LLMError")
    except LLMError:
        pass


async def test_discover_candidates_dedupes_and_normalizes() -> None:
    client = _StubLLMClient(
        result={
            "accounts": [
                {"username": "ESPN", "reason": "a"},
                {"username": "@espn", "reason": "b"},
                {"username": "https://x.com/nba", "reason": "c"},
                {"username": "has space", "reason": "invalid"},
            ]
        }
    )

    result = await discover_candidates("sports", ["football"], limit=100, llm_client=client)
    usernames = [a.username for a in result]

    assert usernames.count("ESPN") + usernames.count("espn") == 1  # case-insensitive dedupe
    assert "nba" in usernames
    assert "has space" not in usernames  # invalid username syntax filtered out


async def test_discover_candidates_raises_without_llm_client() -> None:
    try:
        await discover_candidates("sports", ["football"], limit=100, llm_client=None)
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert "GEMINI_API_KEY" in str(exc)


async def test_discover_candidates_raises_on_llm_failure_no_fallback() -> None:
    client = _StubLLMClient(error=LLMError("boom"))

    try:
        await discover_candidates("sports", ["football"], limit=100, llm_client=client)
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert "No curated account fallback is enabled" in str(exc)


async def test_discover_candidates_respects_limit() -> None:
    client = _StubLLMClient(
        result={"accounts": [{"username": f"user{i}", "reason": "r"} for i in range(50)]}
    )

    result = await discover_candidates("sports", [], limit=5, llm_client=client)

    assert len(result) == 5
