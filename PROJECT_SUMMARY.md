# Project Summary: x-profile-scraper ("X Intelligence")

## Overview

A Python application that started as a public X/Twitter profile scraper and has grown into **"X Intelligence"** — a category-level social intelligence platform aimed at journalists and researchers. It:

1. Scrapes public X profile/tweet data for given usernames (no DMs, no protected content — public fields only).
2. Given a topic/category (e.g. "technology", "space"), uses an LLM to discover and rank relevant accounts, scrapes their tweets, and analyzes trending topics/sentiment.
3. Lets a user ask natural-language questions over previously collected tweets via a RAG (retrieval-augmented generation) pipeline, with cited evidence and an "insufficient evidence" fallback rather than fabrication.
4. Surfaces "Story Opportunities" for journalists by comparing runs over time and flagging significant shifts, backed by AI-generated briefs.

## Entry Points

Two entry points share the same core logic and on-disk data:

- **`main.py`** — Typer/Rich CLI.
  - Interactive mode (no args): prompts for a single username, prints latest tweets.
  - Batch profile scrape: `python main.py user1 user2` or a usernames file, with `--json/--csv/--both`, `--concurrency`, `--no-cache`.
  - Category-intelligence commands: `category <name>` (build discovery context only) and `analyze <name>` (full pipeline: discovery → ranking → tweet scraping → analysis → report).
- **`streamlit_app.py`** — a thin router into a Streamlit dashboard (`ui/pages/*`: Overview, New Analysis, Sources, Intelligence [Browse Evidence + Ask Intelligence], Trends, Reports [All Runs / Compare / Story Opportunities], Settings). It never reimplements scraping/analysis logic — it calls the same `CategoryIntelligenceAgent` pipeline as the CLI, reading/writing the same `.env` and `data/`.

## Authentication Model

X retired password login for third-party clients, so auth is cookie-based: `X_AUTH_TOKEN`/`X_CT0` exported from a logged-in browser session, cached in `.data/session.json`. The scraping client depends on `twifork==2.3.5` (a maintained Twikit fork, imported as `twikit`) rather than upstream `twikit`, because it patches X's `ondemand.s.js` changes.

## Architecture (core package `app/`)

| Area | Files | Responsibility |
|---|---|---|
| Config/models/infra | `config.py`, `models.py`, `schemas.py`, `exceptions.py`, `logger.py`, `utils.py` | Settings singleton, Pydantic data models, error hierarchy, logging, shared helpers (`retry_with_backoff`, username validation) |
| Scraping | `client.py`, `scraper.py`, `tweet_scraper.py`, `cache.py` | `TwikitProfileClient` (session/auth + GraphQL, no browser automation), concurrent profile/tweet scrapers, async TTL file cache |
| Storage/export | `exporter.py`, `storage.py` | JSON/CSV exporters (pluggable `ExporterRegistry`), category-run persistence to `data/tweets/<category>/<date>.json` + RAG indexing trigger |
| LLM/AI | `llm.py`, `account_discovery.py`, `account_ranker.py`, `analysis.py`, `category_agent.py` | Groq (default) / Gemini provider abstraction; LLM-only account discovery; deterministic + LLM-scored ranking; trending-topic/sentiment analysis; `CategoryIntelligenceAgent.run_pipeline()` top-level orchestrator |
| Pure logic | `time_window.py`, `topic_matching.py`, `report_compare.py`, `signal_score.py`, `story_opportunities.py`, `story_brief.py` | Time-window filtering, topic/tweet keyword matching, run-to-run diffing, 0–100 signal/confidence scoring, journalist story-opportunity derivation and AI brief generation |
| RAG (`app/rag/`) | `embeddings.py`, `vector_store.py`, `indexer.py`, `retriever.py`, `reranker.py`, `agent.py` | Local sentence-transformers embeddings, ChromaDB vector store, indexing, retrieval with similarity floor, `0.7*similarity + 0.2*recency + 0.1*engagement` reranking, `ask_intelligence()` Q&A orchestration with citations |

Dependency direction is intentionally one-way: `client`/`cache` → `scraper` → `category_agent` → `main.py`/`ui/`. `app/` never depends on `ui/`.

## Data Flow

**Simple profile mode:**
```
usernames → normalize/validate → cache check → TwikitProfileClient.get_profile()
  → UserProfile model → JSON/CSV export → data/json, data/csv
```

**Category-intelligence mode:**
```
category string
  → CategoryAgent.build_context() (LLM or 10 predefined fallback categories)
  → discover_candidates() (LLM-only account discovery)
  → ProfileScraper.scrape_many() (validate candidates are real accounts)
  → rank_accounts() / select_top_n() (relevance + engagement + activity + audience)
  → TweetScraper.scrape_many() (top-N accounts, time-window aware)
  → rerank_with_tweet_engagement()
  → analyze_category() (LLM trending topics/sentiment, deterministic fallback)
  → CategoryReport assembled → save_category_run()
      → JSON/CSV/tweet exports + optional RAG indexing (.chroma)
  → shown via CLI table (Rich) or Streamlit dashboard
```

**Journalist tooling (built on stored snapshots + vector index):**
```
saved runs → report_compare.compare_reports() → story_opportunities.derive_story_opportunities()
  → story_brief.generate_story_brief() → RAG retrieve/rerank → LLM → cited StoryBrief

OR: ui/ask_runner.py → app/rag/agent.ask_intelligence() → retrieve → rerank → LLM → cited answer
```

## Tech Stack

- **Scraping**: `twifork` (Twikit fork) — cookie-based session auth against X's internal GraphQL API, no Playwright/Selenium.
- **CLI**: Typer + Rich.
- **UI**: Streamlit + Plotly.
- **Data/config**: Pydantic + pydantic-settings, pandas (CSV), aiofiles, loguru.
- **LLM**: Groq SDK (default, model `openai/gpt-oss-20b`), `google-genai` (Gemini `gemini-2.0-flash`, alternate provider) — no LangChain.
- **RAG**: ChromaDB (vector store, cosine distance) + sentence-transformers (`all-MiniLM-L6-v2`, local embeddings).
- **Dev tooling**: pytest/pytest-asyncio (~24 test modules, 268 offline tests, no live credentials required), black/isort/ruff.

## Configuration (`.env.example`)

- **X auth**: `X_AUTH_TOKEN`, `X_CT0` (primary), legacy `X_USERNAME`/`X_EMAIL`/`X_PASSWORD`, `X_SESSION_FILE`.
- **Output paths**: `OUTPUT_DIR`, `JSON_OUTPUT_DIR`, `CSV_OUTPUT_DIR`, `LOG_DIR`, `TWEETS_OUTPUT_DIR`.
- **Concurrency/pacing**: `CONCURRENCY_LIMIT` (10/20/50/100 tiers), `REQUEST_DELAY_SECONDS`, `TWEET_SCRAPE_CONCURRENCY`.
- **Retry/backoff**: `MAX_RETRIES`, `BACKOFF_BASE_SECONDS`, `BACKOFF_MAX_SECONDS`, plus a separate longer rate-limit track (`RATE_LIMIT_BASE_SECONDS` default 30s, `RATE_LIMIT_MAX_SECONDS` default 900s).
- **Cache**: `CACHE_ENABLED`, `CACHE_DIR`, `CACHE_TTL_SECONDS`, `TWEET_CACHE_DIR`, `TWEET_CACHE_TTL_SECONDS`.
- **LLM**: `LLM_PROVIDER` (`groq` default / `gemini`), `GROQ_API_KEY`, `GROQ_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL`.
- **Category intelligence**: `CATEGORY_CANDIDATE_LIMIT` (50), `TOP_ACCOUNTS_LIMIT` (20), `TWEETS_PER_ACCOUNT` (10).
- **RAG**: `CHROMA_DIR` (default `.chroma`), `RAG_MIN_SIMILARITY` (default 0.30).

## Repository Layout

```
x-profile-scraper/
├── main.py                  # CLI entry point (Typer)
├── streamlit_app.py         # UI entry point (Streamlit router)
├── pyproject.toml / requirements.txt
├── .env.example
├── app/                     # core package
│   ├── config.py, models.py, schemas.py, exceptions.py, logger.py, utils.py
│   ├── client.py, scraper.py, tweet_scraper.py, cache.py
│   ├── exporter.py, storage.py
│   ├── llm.py, account_discovery.py, account_ranker.py, analysis.py, category_agent.py
│   ├── time_window.py, topic_matching.py, report_compare.py,
│   │   signal_score.py, story_opportunities.py, story_brief.py
│   └── rag/
│       ├── embeddings.py, vector_store.py, indexer.py, retriever.py, reranker.py, agent.py
├── ui/                      # Streamlit presentation layer
│   ├── ask_runner.py, pipeline_runner.py, data_loader.py,
│   │   charts.py, cards.py, components.py, styles.py, utils.py
│   └── pages/
│       ├── overview.py, new_analysis.py, sources.py, intelligence.py,
│       │   trends.py, reports.py, settings.py
├── scripts/
│   ├── backfill_rag_index.py, extract_chrome_cookies.py,
│   │   generate_sample_output.py, get_latest_tweets.py
├── tests/                   # ~24 modules mirroring app/ and ui/
└── data/
    ├── csv/, json/, logs/, tweets/<category>/<date>.json
```

## Current State

Not a fresh scaffold — `data/` already holds real accumulated output: ~230 per-account CSV tweet exports across many categories (news, sports, health, finance, space, tech, education, Indian startups, etc.), ~34 timestamped profile JSON batches (2026-08-08 to 2026-08-11), 5 days of rotating logs, and per-category tweet snapshot folders. A local `.chroma/` vector store and `.venv` are also present — this is an actively-used tool.

## Usage Reference

```
python main.py                                    # interactive
python main.py elonmusk openai satyanadella
python main.py usernames.txt --csv
python main.py --concurrency 50 usernames.txt
python main.py category technology                # discovery only
python main.py analyze technology --top-accounts 20
streamlit run streamlit_app.py                     # X Intelligence dashboard
python scripts/backfill_rag_index.py               # index existing runs for Ask Intelligence
pytest -q                                          # 268 offline tests
```
