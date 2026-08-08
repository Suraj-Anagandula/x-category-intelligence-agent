"""Unit tests for app.llm: shared JSON extraction, GroqProvider's JSON-mode
request/retry behavior, and the provider factory.

GroqProvider tests stub the underlying SDK client directly (no real
network/API calls, no real credentials needed) by setting the provider's
internal `_client` attribute, bypassing the lazy `_ensure_client` SDK-import
path entirely.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.exceptions import LLMError
from app.llm import GeminiProvider, GroqProvider, _extract_json, build_llm_client


def _groq_response(content: str | None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _StubGroqClient:
    """Stands in for `groq.AsyncGroq` - records every create() call's kwargs."""

    def __init__(self, responses: list) -> None:
        # Each item is either a response object to return or an Exception to raise.
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_extract_json_parses_plain_json() -> None:
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fences() -> None:
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_extracts_object_wrapped_in_stray_text() -> None:
    assert _extract_json('Sure, here you go: {"a": 1} - hope that helps!') == {"a": 1}


def test_extract_json_raises_on_unparseable_text() -> None:
    try:
        _extract_json("not json at all")
        raise AssertionError("expected LLMError")
    except LLMError:
        pass


async def test_groq_generate_json_succeeds_with_reasoning_params() -> None:
    client = _StubGroqClient([_groq_response('{"accounts": []}')])
    provider = GroqProvider(api_key="test-key", model="openai/gpt-oss-20b")
    provider._client = client

    result = await provider.generate_json("prompt")

    assert result == {"accounts": []}
    assert len(client.calls) == 1
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["reasoning_effort"] == "low"
    assert client.calls[0]["reasoning_format"] == "hidden"


async def test_groq_generate_json_falls_back_to_plain_request_on_json_validate_failed() -> None:
    """Reproduces the reported bug: the first (reasoning-params) attempt
    fails with Groq's json_validate_failed 400 - the retry must be a
    genuinely different request (no response_format/reasoning params), not
    the same one repeated."""
    json_validate_error = Exception(
        "Error code: 400 - {'error': {'message': \"Failed to validate JSON. "
        "Please adjust your prompt.\", 'type': 'invalid_request_error', "
        "'code': 'json_validate_failed', 'failed_generation': ''}}"
    )
    client = _StubGroqClient([json_validate_error, _groq_response('{"accounts": []}')])
    provider = GroqProvider(api_key="test-key", model="openai/gpt-oss-20b")
    provider._client = client

    result = await provider.generate_json("prompt")

    assert result == {"accounts": []}
    assert len(client.calls) == 2
    assert "response_format" not in client.calls[1]
    assert "reasoning_effort" not in client.calls[1]
    assert "reasoning_format" not in client.calls[1]


async def test_groq_generate_json_raises_when_both_attempts_fail() -> None:
    client = _StubGroqClient([Exception("boom1"), Exception("boom2")])
    provider = GroqProvider(api_key="test-key", model="openai/gpt-oss-20b")
    provider._client = client

    try:
        await provider.generate_json("prompt")
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert "boom2" in str(exc)


async def test_groq_generate_json_raises_on_empty_response() -> None:
    client = _StubGroqClient([_groq_response(None)])
    provider = GroqProvider(api_key="test-key", model="openai/gpt-oss-20b")
    provider._client = client

    try:
        await provider.generate_json("prompt")
        raise AssertionError("expected LLMError")
    except LLMError:
        pass


async def test_groq_generate_json_extracts_json_wrapped_in_text_on_fallback() -> None:
    """The plain-text fallback path has no server-enforced JSON mode, so
    _extract_json's stray-text handling must actually be exercised end to end."""
    client = _StubGroqClient(
        [
            Exception("json_validate_failed"),
            _groq_response('Here is the JSON: {"accounts": [{"username": "nasa", "reason": "x"}]}'),
        ]
    )
    provider = GroqProvider(api_key="test-key", model="openai/gpt-oss-20b")
    provider._client = client

    result = await provider.generate_json("prompt")

    assert result == {"accounts": [{"username": "nasa", "reason": "x"}]}


def test_build_llm_client_defaults_to_groq() -> None:
    settings = Settings()
    settings.llm_provider = "groq"
    settings.groq_api_key = "test-key"

    client = build_llm_client(settings)

    assert isinstance(client, GroqProvider)


def test_build_llm_client_returns_none_without_groq_api_key() -> None:
    settings = Settings()
    settings.llm_provider = "groq"
    settings.groq_api_key = None

    assert build_llm_client(settings) is None


def test_build_llm_client_selects_gemini_when_configured() -> None:
    settings = Settings()
    settings.llm_provider = "gemini"
    settings.gemini_api_key = "test-key"

    client = build_llm_client(settings)

    assert isinstance(client, GeminiProvider)


def test_build_llm_client_unknown_provider_returns_none() -> None:
    settings = Settings()
    settings.llm_provider = "not-a-real-provider"

    assert build_llm_client(settings) is None
