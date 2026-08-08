"""Provider-agnostic LLM interface for category-intelligence features.

`LLMProvider` is the interface the rest of the app depends on
(`generate_text`/`generate_json`) - callers (`category_agent`,
`account_discovery`, `account_ranker`, `analysis`) never need to know
whether the configured provider is Groq or Gemini. Each concrete provider
lazily imports its own SDK (same pattern as `app/client.py`'s lazy twikit
import), so modules that don't touch the LLM never pay the import cost or
require an unused SDK to be installed.

`build_llm_client(settings)` is the factory: it reads `settings.llm_provider`
("groq", the default, or "gemini") and returns the matching provider, or
None if that provider's API key isn't configured. Callers already treat
None/`LLMError` as "no LLM available" and act accordingly - there is no
curated-account fallback for discovery (see `app/account_discovery.py`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

from app.exceptions import LLMError
from app.logger import get_logger

if TYPE_CHECKING:
    from app.config import Settings

logger = get_logger()


class LLMProvider(Protocol):
    """Interface every LLM backend must implement."""

    async def generate_text(self, prompt: str) -> str: ...
    async def generate_json(self, prompt: str) -> dict | list: ...


def _extract_json(text: str) -> dict | list:
    """Strip Markdown code fences (if present) and parse JSON.

    Shared by every provider's `generate_json` - raises `LLMError` on
    failure rather than letting a raw `JSONDecodeError` escape. Falls back to
    extracting the outermost `{...}`/`[...]` substring before giving up, in
    case the model wrapped the JSON in a stray sentence despite instructions
    not to (relevant for providers/models used without server-enforced JSON
    mode, e.g. Groq's plain-text fallback path below).
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
    if starts and ends:
        start, end = min(starts), max(ends) + 1
        if end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError as exc:
                raise LLMError(f"LLM returned unparseable JSON: {exc}") from exc

    raise LLMError("LLM returned unparseable JSON: no JSON object/array found in response")


class GeminiProvider:
    """Google Gemini, via the `google-genai` SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise LLMError(
                    "google-genai is not installed. Run `pip install -r requirements.txt`."
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def generate_text(self, prompt: str) -> str:
        """Return the model's plain-text response to `prompt`."""
        client = self._ensure_client()
        try:
            response = await client.aio.models.generate_content(model=self._model, contents=prompt)
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure is non-fatal upstream
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMError("Gemini returned an empty response")
        return text

    async def generate_json(self, prompt: str) -> dict | list:
        """Ask the model for strict JSON and parse it.

        Raises `LLMError` if the call fails or the response isn't valid JSON.
        """
        client = self._ensure_client()
        try:
            from google.genai import types

            config = types.GenerateContentConfig(response_mime_type="application/json")
        except Exception:  # noqa: BLE001 - fall back to plain text-mode call
            config = None

        try:
            if config is not None:
                response = await client.aio.models.generate_content(
                    model=self._model, contents=prompt, config=config
                )
            else:
                response = await client.aio.models.generate_content(
                    model=self._model, contents=prompt
                )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMError("Gemini returned an empty response")
        return _extract_json(text)


class GroqProvider:
    """Groq, via the official `groq` Python SDK (OpenAI-compatible chat API)."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from groq import AsyncGroq
            except ImportError as exc:
                raise LLMError(
                    "groq is not installed. Run `pip install -r requirements.txt`."
                ) from exc
            self._client = AsyncGroq(api_key=self._api_key)
        return self._client

    async def generate_text(self, prompt: str) -> str:
        """Return the model's plain-text response to `prompt`."""
        client = self._ensure_client()
        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure is non-fatal upstream
            raise LLMError(f"Groq request failed: {exc}") from exc

        text = response.choices[0].message.content if response.choices else None
        if not text:
            raise LLMError("Groq returned an empty response")
        return text

    async def generate_json(self, prompt: str) -> dict | list:
        """Ask the model for strict JSON (Groq JSON mode) and parse it.

        Reasoning-capable Groq models (`openai/gpt-oss-20b`/`-120b`, the
        `qwen3` family) can fail Groq's server-side `json_object` validation
        with `json_validate_failed` / empty `failed_generation` - the
        installed SDK documents why: unless `reasoning_format="hidden"` is
        set, the model's reasoning tokens can leak into (or entirely
        replace) the completion content that Groq validates as JSON. We ask
        for hidden, low-effort reasoning so `content` is just the final
        JSON. `reasoning_effort`/`reasoning_format` are only meaningful for
        reasoning-capable models, so if the configured `GROQ_MODEL` rejects
        them (a different 400), we retry once with a genuinely different
        request - plain generation, no `response_format`/reasoning params,
        relying on the prompt's own JSON instruction plus `_extract_json`'s
        fallback parsing - rather than repeating the identical failing call.

        Raises `LLMError` if both attempts fail or neither response is valid JSON.
        """
        client = self._ensure_client()
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                reasoning_effort="low",
                reasoning_format="hidden",
            )
        except Exception as exc:  # noqa: BLE001 - retry below with a different request shape
            logger.warning(
                f"Groq JSON-mode request with reasoning params failed, retrying in "
                f"plain-text mode: {exc}"
            )
            try:
                response = await client.chat.completions.create(
                    model=self._model, messages=messages
                )
            except Exception as retry_exc:  # noqa: BLE001
                raise LLMError(f"Groq request failed: {retry_exc}") from retry_exc

        text = response.choices[0].message.content if response.choices else None
        if not text:
            raise LLMError("Groq returned an empty response")
        return _extract_json(text)


def build_llm_client(settings: Settings) -> LLMProvider | None:
    """Return the configured LLM provider, or None if unavailable.

    Reads `settings.llm_provider` ("groq" or "gemini") and dispatches to
    that provider's API key/model. Returns None (never raises) if the
    selected provider has no API key configured - "no LLM" is a normal,
    expected state for every caller (there is no curated-account fallback).
    """
    provider_name = (settings.llm_provider or "groq").strip().lower()

    if provider_name == "groq":
        if not settings.groq_api_key:
            return None
        return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model)

    if provider_name == "gemini":
        if not settings.gemini_api_key:
            return None
        return GeminiProvider(api_key=settings.gemini_api_key, model=settings.gemini_model)

    logger.warning(f"Unknown LLM_PROVIDER {provider_name!r} configured; no LLM will be used.")
    return None
