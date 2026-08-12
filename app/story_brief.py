"""Generate an evidence-backed story brief for a Story Opportunity.

Observed fact, AI interpretation, and suggested investigation are kept in
three structurally distinct fields, asked for separately in the prompt -
never a single summary split post-hoc by string heuristics, which is the
only way this separation (spec's fact/interpretation/suggestion
distinction) is actually reliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.exceptions import LLMError
from app.llm import LLMProvider
from app.logger import get_logger
from app.rag.agent import Citation, resolve_citations
from app.rag.embeddings import Embedder
from app.rag.reranker import DEFAULT_TOP_N, rerank
from app.rag.retriever import DEFAULT_TOP_K_CANDIDATES, RetrievedChunk, retrieve
from app.rag.vector_store import VectorStore
from app.story_opportunities import StoryOpportunity

logger = get_logger()


@dataclass
class StoryBrief:
    headline: str
    why_it_matters: str
    observed_facts: list[str] = field(default_factory=list)
    ai_interpretation: list[str] = field(default_factory=list)
    suggested_investigation_questions: list[str] = field(default_factory=list)
    supporting_posts: list[Citation] = field(default_factory=list)
    supporting_accounts: list[str] = field(default_factory=list)
    #: Set only on the degraded paths (no evidence / no LLM / LLM failure)
    #: so the UI can render an explicit "this brief is incomplete" notice
    #: rather than silently presenting a thin brief as if it were complete.
    note: str = ""


_PROMPT_TEMPLATE = (
    "You are helping a journalist evaluate a potential story using ONLY the "
    "X (Twitter) posts listed below as evidence. Separate what is directly "
    "observed in the posts from your own interpretation, and never invent "
    "facts, statistics, or URLs not present in the evidence.\n\n"
    "Story signal: {title}\n"
    "Why it was flagged: {why_it_matters}\n\n"
    "Evidence:\n{evidence}\n\n"
    'Return ONLY JSON: {{"headline": "<suggested headline>", '
    '"observed_facts": ["<fact literally reflected in the posts>"], '
    '"ai_interpretation": ["<your inference, explicitly labeled as interpretation>"], '
    '"suggested_investigation_questions": ["<open question a journalist should chase>"], '
    '"supporting_posts": [<1-based post indices actually used>]}}'
)


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        when = chunk.created_at or "unknown time"
        lines.append(
            f'[{index}] @{chunk.username or "unknown"} ({when}): "{chunk.text}" — {chunk.url}'
        )
    return "\n".join(lines)


def _degraded_brief(opportunity: StoryOpportunity, note: str) -> StoryBrief:
    """No evidence / no LLM / LLM failure - never present a thin brief as
    if it were complete."""
    return StoryBrief(
        headline=opportunity.title,
        why_it_matters=opportunity.why_it_matters,
        supporting_accounts=opportunity.account_usernames,
        note=note,
    )


async def generate_story_brief(
    opportunity: StoryOpportunity,
    store: VectorStore,
    embedder: Embedder,
    llm_client: LLMProvider | None,
    category: str | None = None,
    tweet_ids: set[str] | list[str] | None = None,
) -> StoryBrief:
    """Retrieve evidence scoped to `opportunity` (reuses Phase 3's
    retriever, not a new retrieval path) and ask the LLM for a brief.

    `tweet_ids`, when given, MUST be the current report's own tweet id set
    - every story brief is generated for a specific analysis run (a
    specific category + time window + account selection), and evidence
    must never be pulled from any other run's tweets or an older/global
    slice of the same category's index (see section 1 of the data-quality
    fix: a June tweet must never appear in an Aug 9-10 report). Callers
    that omit it get the pre-existing category-only-scoped behavior -
    kept only so tests that stub `retrieve` directly are unaffected; real
    UI call sites (`ui/ask_runner.py::generate_brief`) always pass it.
    """
    top_k = max(DEFAULT_TOP_K_CANDIDATES, len(tweet_ids)) if tweet_ids else DEFAULT_TOP_K_CANDIDATES
    candidates = retrieve(
        opportunity.title,
        store,
        embedder,
        category=category,
        tweet_ids=tweet_ids,
        top_k=top_k,
    )
    chunks = rerank(candidates, top_n=DEFAULT_TOP_N)

    if not chunks:
        return _degraded_brief(opportunity, "Insufficient evidence to generate a full brief.")
    if llm_client is None:
        return _degraded_brief(opportunity, "No LLM configured - showing the raw signal only.")

    prompt = _PROMPT_TEMPLATE.format(
        title=opportunity.title,
        why_it_matters=opportunity.why_it_matters,
        evidence=_format_evidence(chunks),
    )
    try:
        result = await llm_client.generate_json(prompt)
    except LLMError as exc:
        logger.warning(f"Story brief generation failed, showing raw signal only: {exc}")
        return _degraded_brief(opportunity, f"AI brief generation failed: {exc}")

    if not isinstance(result, dict):
        return _degraded_brief(
            opportunity, "AI returned an unexpected response - showing the raw signal only."
        )

    raw_indices = result.get("supporting_posts") or []
    indices = [i for i in raw_indices if isinstance(i, int)]
    citations = resolve_citations(chunks, indices)
    if not citations:
        citations = resolve_citations(chunks, list(range(1, len(chunks) + 1)))

    accounts = sorted(
        {c.username for c in citations if c.username} | set(opportunity.account_usernames)
    )

    return StoryBrief(
        headline=str(result.get("headline") or opportunity.title),
        why_it_matters=opportunity.why_it_matters,
        observed_facts=[str(f) for f in (result.get("observed_facts") or [])],
        ai_interpretation=[str(f) for f in (result.get("ai_interpretation") or [])],
        suggested_investigation_questions=[
            str(f) for f in (result.get("suggested_investigation_questions") or [])
        ],
        supporting_posts=citations,
        supporting_accounts=accounts,
    )
