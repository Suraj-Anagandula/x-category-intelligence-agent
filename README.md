# X Profile Scraper

A modular, async Python application that collects **publicly available profile
information** from X (formerly Twitter) for one or more usernames, using
[Twikit](https://github.com/d60/twikit) (via the
[`twifork`](https://github.com/PawiX25/twifork) maintained fork — see
"Authentication" below) as the underlying client.

It only reads public profile fields (bio, follower counts, verification
status, etc.) — never DMs, private tweets, or anything behind a protected
account's wall. Respect X's Terms of Service and applicable law in how you
use this tool.

> **Note on authentication.** X requires a logged-in session for nearly all
> reads today, even of public profiles, **and has retired password login for
> third-party clients** — `Client.login()` cannot be made to work anymore, no
> matter the library version. The supported path is exporting `auth_token`
> and `ct0` cookies from a browser session where you're already logged in to
> x.com. See "Authentication" below for exact steps. The app caches the
> resulting session locally so you don't have to re-export cookies on every
> run — only when they eventually expire.

---

## Features

- Fetches public profile fields (bio, stats, dates, verification, etc.) for
  any number of usernames.
- Concurrent scraping with a configurable limit (10 / 20 / 50 / 100).
- Exponential backoff + jitter retries for rate limits, timeouts, and
  transient network errors.
- TTL file cache so repeat lookups within the TTL window skip the network.
- JSON and CSV export, with an `Exporter` interface designed so
  SQLite/PostgreSQL/MongoDB backends can be added later without touching
  calling code.
- Rich progress bar + summary table; structured logging to `data/logs/`.
- One user's failure (suspended, not found, protected, rate-limited, ...)
  never stops the rest of the batch.
- A category-intelligence pipeline (LLM-driven account discovery → ranking
  → tweet collection → trending-topic/sentiment analysis) exposed through a
  polished "X Intelligence" Streamlit dashboard for journalists/researchers
  — see "Streamlit UI — X Intelligence" below.
- "Ask Intelligence": retrieval-augmented Q&A over previously collected
  posts, with real evidence citations and an explicit "insufficient
  evidence" fallback rather than fabricated answers.
- Same-category-across-two-dates comparison and journalist-facing "Story
  Opportunities" with evidence-backed, AI-assisted brief generation.

---

## Project Architecture

```text
x-profile-scraper/
│
├── app/
│   ├── client.py            # Twikit session management + Twikit User -> UserProfile mapping
│   ├── scraper.py            # Profile: concurrency, retry orchestration, progress bar, logging
│   ├── tweet_scraper.py      # Tweets: same pattern, its own (lower) concurrency tier
│   ├── models.py             # Pydantic UserProfile / Tweet / ScrapeResult
│   ├── schemas.py            # CategoryContext / RankedAccount / CategoryReport (derived output)
│   ├── exporter.py           # JSON/CSV exporters behind a BaseExporter interface
│   ├── cache.py              # TTL file caches (profiles, tweets)
│   ├── config.py             # pydantic-settings driven configuration
│   ├── logger.py             # loguru console + rotating file sinks
│   ├── exceptions.py         # Typed exception hierarchy
│   ├── utils.py              # Username parsing/validation, retry/backoff helper
│   ├── llm.py                # Provider-agnostic Groq/Gemini LLM client
│   ├── category_agent.py     # CategoryAgent + CategoryIntelligenceAgent (pipeline orchestrator)
│   ├── account_discovery.py  # LLM-only candidate account discovery
│   ├── account_ranker.py     # Deterministic weighted account ranking
│   ├── analysis.py           # Trending topics / sentiment / summary (LLM + fallback)
│   ├── storage.py            # Persists a category run's snapshot + exports
│   ├── signal_score.py       # Signal Score / Confidence - pure functions over real fields
│   ├── topic_matching.py     # Keyword-overlap topic<->tweet matching (shared by app/ and ui/)
│   ├── report_compare.py     # Diff two dated runs of the same category
│   ├── story_opportunities.py # Derives journalist-facing "worth investigating" signals
│   ├── story_brief.py        # Generates an evidence-backed brief for one opportunity
│   └── rag/                  # Ask Intelligence (RAG) - see "Ask Intelligence (RAG)" below
│       ├── embeddings.py      # sentence-transformers wrapper, lazily imported
│       ├── vector_store.py    # ChromaDB persistent-collection wrapper
│       ├── indexer.py         # Tweet -> normalized text + metadata -> upsert; backfill
│       ├── retriever.py       # question -> embed -> query -> similarity-filtered evidence
│       ├── reranker.py        # Swappable similarity+recency+engagement reranking heuristic
│       └── agent.py           # ask_intelligence() + shared citation-resolution helper
│
├── data/
│   ├── json/            # Timestamped JSON exports (sample_profiles.json checked in)
│   ├── csv/             # Running users.csv + per-category tweet CSVs
│   ├── logs/            # Daily rotating log files
│   └── tweets/          # data/tweets/<category>/<date>.json run snapshots
│
├── .chroma/               # Ask Intelligence vector store (gitignored, derived/regenerable)
│
├── ui/                    # Streamlit presentation layer (see "Streamlit UI" below)
│   ├── pages/              # One module per nav destination (overview/new_analysis/sources/...)
│   ├── ask_runner.py       # Sync bridge to the async Ask Intelligence / Story Brief flows
│   ├── pipeline_runner.py  # Calls the existing CategoryIntelligenceAgent - no reimplementation
│   ├── data_loader.py      # Read-only access to existing data/tweets, data/csv
│   ├── charts.py           # Plotly chart builders
│   ├── cards.py            # Signal/source/evidence/freshness/story-opportunity/brief cards
│   ├── components.py       # Other reusable st.* render functions
│   ├── styles.py           # Theme-neutral CSS + design tokens
│   └── utils.py            # Formatting/validation/depth-preset/freshness helpers
│
├── scripts/
│   ├── generate_sample_output.py  # Regenerates the checked-in sample fixtures
│   └── backfill_rag_index.py      # Indexes every existing run snapshot for Ask Intelligence
│
├── tests/                # pytest unit tests, one file per module above (no live credentials)
├── main.py               # Typer CLI entry point
├── streamlit_app.py       # Streamlit UI entry point (streamlit run streamlit_app.py) - thin router
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

Each module has one responsibility and depends only on the layers below it
(`client`/`cache` -> `scraper` -> `category_agent` -> `main`/`ui`), so any
piece can be swapped or tested independently — e.g. `exporter.py` doesn't
know Twikit exists, `models.py`/`utils.py`/`signal_score.py`/
`topic_matching.py`/`report_compare.py` have zero I/O, and `app/` never
depends on `ui/` (the reverse is fine, and is how `ui/utils.py` re-exports
`app/topic_matching.py`'s functions for existing call sites).

---

## Installation

Requires **Python 3.12+** (tested here against 3.10, which also works since
no 3.12-only syntax is used).

```bash
cd x-profile-scraper
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Setup

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

```dotenv
X_AUTH_TOKEN=
X_CT0=
X_SESSION_FILE=.data/session.json

OUTPUT_DIR=data
JSON_OUTPUT_DIR=data/json
CSV_OUTPUT_DIR=data/csv
LOG_DIR=data/logs

CONCURRENCY_LIMIT=10
REQUEST_DELAY_SECONDS=1.0
TWEET_SCRAPE_CONCURRENCY=3

MAX_RETRIES=3
BACKOFF_BASE_SECONDS=2.0
BACKOFF_MAX_SECONDS=30.0
RATE_LIMIT_BASE_SECONDS=30.0
RATE_LIMIT_MAX_SECONDS=900.0

CACHE_ENABLED=true
CACHE_DIR=.cache
CACHE_TTL_SECONDS=3600

LOG_LEVEL=INFO
```

## Authentication

**X has retired password login for third-party clients.** `Client.login()`
(username/email/password) fails outright with `Couldn't get KEY_BYTE indices`
or similar, on every current library — this isn't a bug in this app, it's a
platform-side change. The working alternative is exporting two cookies from a
browser session where you're already logged in to x.com:

1. Open [x.com](https://x.com) in a browser and make sure you're logged in.
2. Open DevTools (F12) → **Application** (Chrome) or **Storage** (Firefox) →
   **Cookies** → `https://x.com`.
3. Find the rows named `auth_token` and `ct0`, and copy their **Value**
   columns.
4. Paste them into `.env`:

   ```dotenv
   X_AUTH_TOKEN=<value of auth_token>
   X_CT0=<value of ct0>
   ```

5. Run the app. On success, the session is cached to `X_SESSION_FILE` so you
   don't need to repeat this until the cookies eventually expire (X sessions
   typically last weeks to months) — at which point `is_logged_in()` returns
   false and you'll be prompted to export fresh ones.

This project depends on [`twifork`](https://github.com/PawiX25/twifork)
rather than upstream `twikit` — it's a maintained fork that still fixes X's
March 2026 `ondemand.s.js` change (the actual root cause of the KEY_BYTE
error) for GraphQL requests, while still importing as `from twikit import
Client` so no application code differs. `X_USERNAME`/`X_EMAIL`/`X_PASSWORD`
remain as a legacy fallback in case password login is ever restored for some
account tier, but treat cookie auth as the supported path.

---

## Usage

```bash
# single username
python main.py elonmusk

# multiple usernames
python main.py elonmusk openai satyanadella

# usernames from a file (one per line, '#' comments and blanks ignored)
python main.py usernames.txt

# choose export format
python main.py --csv usernames.txt
python main.py --json usernames.txt
python main.py --both usernames.txt

# override concurrency tier for this run
python main.py --concurrency 50 usernames.txt

# bypass the cache
python main.py --no-cache elonmusk
```

Default output format (no flag) is JSON. `--both` writes both a timestamped
JSON file and appends/upserts into the running `data/csv/users.csv`.

### Sample usernames file

```text
# usernames.txt
elonmusk
openai
satyanadella
```

The CLI also supports category intelligence (dynamic account discovery,
ranking, tweet collection, and analysis for an arbitrary category):

```bash
python main.py category technology                                            # show discovery keywords/subcategories only
python main.py analyze technology --candidate-limit 50 --top-accounts 20 --tweets-per-account 10
```

---

## Streamlit UI — X Intelligence

A visual dashboard ("X Intelligence") is available as a second interface
over the same backend pipeline - it's purely presentational and reuses
`CategoryIntelligenceAgent` exactly as the CLI does; nothing about
discovery, validation, ranking, scraping, retry/rate-limit handling, or
storage is reimplemented. Its page-render code lives in `ui/pages/*.py`,
one module per nav destination, with `streamlit_app.py` staying a thin
router (sidebar → page dict → call).

Install the extra dependencies (already included in `requirements.txt`; if
installing via `pyproject.toml` extras, use `pip install -e ".[ui]"`), then:

```bash
streamlit run streamlit_app.py
```

Pages (navigate via the sidebar):

- **Overview** — landing page; an onboarding screen with category
  quick-picks when no analysis has run yet, otherwise an Executive Summary,
  Key Signals, Top Sources, and Sentiment at a glance.
- **New Analysis** — pick a category (quick-pick chips or custom text) and
  an analysis depth ("Standard"/"Deep" - resolves to the underlying
  candidate-limit/top-accounts/tweets-per-account numbers; power users can
  still override them in the "Advanced" expander), then click "Analyze
  Now". Progress is reported per stage (Category Context → Account
  Discovery → Profile Validation → Account Ranking → Tweet Collection → AI
  Analysis → Export) via `CategoryIntelligenceAgent.run_pipeline`'s
  `on_stage` callback as the backend actually completes each one - not
  simulated.
- **Sources** — the ranked account table, plus a per-account detail card
  with relevance/engagement/activity sub-scores, follower count, its LLM
  discovery reason ("why this source matters"), and a link to the profile.
- **Intelligence** — two tabs:
  - *Browse Evidence* — collected tweets grouped by trending topic
    (keyword-overlap matching against `analysis.trending_topics` - an
    honest heuristic, not a classifier), with account/topic/search filters.
  - *Ask Intelligence* — retrieval-augmented Q&A over this category's
    previously collected posts. See "Ask Intelligence (RAG)" below.
- **Trends** — the same Plotly charts (ranking score, followers, relevance
  vs. score, sentiment, engagement, tweet distribution), grouped under
  business-question subheaders.
- **Reports** — three tabs:
  - *All Runs* — every previously saved
    `data/tweets/<category>/<date>.json` run, browsable without making any
    new X requests.
  - *Compare* — compare the SAME category across two different dates
    (e.g. Technology yesterday vs. today) - tweet/account-count deltas,
    sentiment shift, topics added/removed/persisted, and the biggest
    account rank/score movers. See the storage-granularity note below.
  - *Story Opportunities* — signals from the loaded report that clear a
    confidence bar, each with a "Generate Brief" action producing an
    evidence-backed brief (headline, why it matters, observed facts vs. AI
    interpretation kept in visually distinct sections, supporting
    posts/accounts, investigation questions).
- **Settings** — the active configuration, split into "Account & Status"
  (credentials/LLM - shown only as "Configured"/"Not configured", never as
  values) and a collapsed "Developer" section (concurrency, retry/rate-limit
  backoff, cache).

Only the "Analyze Now" button on New Analysis ever triggers a live pipeline
run; every other page reads from the current session's report or from disk,
so simply browsing never makes an X/LLM request on its own (Ask
Intelligence/Story Brief generation do call the LLM, but only over already-
collected, already-indexed posts - never a new X scrape).

Both interfaces (`main.py` and `streamlit_app.py`) share the same `.env`
configuration and the same `data/` output.

### Ask Intelligence (RAG)

"Ask Intelligence" is retrieval-augmented Q&A over previously collected X
posts - memory/search over what's already been scraped, not a live web
search:

```
X Posts -> Clean/Normalize -> Embeddings -> Vector Store -> Retriever
-> Relevant X Posts -> Groq -> Evidence-backed Answer
```

- **Embeddings**: local `sentence-transformers` model (`all-MiniLM-L6-v2`,
  384-dim, ~80MB) - chosen over a hosted embeddings API specifically to
  avoid a new required API key/secret; chosen over a larger local model
  since it's fast on CPU and well-suited to short, informal tweet text.
  Lazily imported (same pattern as `app/llm.py`'s provider clients), so
  nothing outside `app/rag/*` pays the import/model-load cost.
- **Vector store**: ChromaDB, persisted to `.chroma/` (configurable via
  `CHROMA_DIR`) - one shared collection across all categories, with
  `category` as a filterable metadata field (simpler and more flexible
  than per-category collections). The collection is explicitly configured
  for cosine distance at creation time (`metadata={"hnsw:space": "cosine"}`)
  so retrieved-evidence similarity scores are meaningful regardless of the
  embedder's output scale.
- **Reranking**: a cheap deterministic heuristic (`0.7*similarity +
  0.2*recency + 0.1*engagement`), not a second model - avoids doubling
  cold-start latency/install surface for an explicitly optional feature.
- **Insufficient evidence**: if nothing clears `RAG_MIN_SIMILARITY` (default
  `0.30`, needs empirical tuning against your own data), Ask Intelligence
  says so explicitly and makes zero LLM calls, rather than guessing.

Install the extra dependencies (`pip install -e ".[rag]"`, or they're
already listed in `requirements.txt`) - the `sentence-transformers` model
weights and the CPU `torch` wheel download on first real use, not at
install time (expect ~500MB-1GB on first run).

Before asking questions, index the posts you've already collected:

```bash
python scripts/backfill_rag_index.py
```

This walks every existing `data/tweets/<category>/<date>.json` snapshot and
indexes it (idempotent - safe to re-run any time; re-indexing the same
tweet overwrites rather than duplicates it). The Ask Intelligence tab also
has a "Build / Refresh Index" button that does the same thing.

### Compare's storage granularity

Compare's two-run diff reuses the existing one-file-per-category-per-day
snapshot layout (`data/tweets/<category>/<date>.json`) unchanged - this is
a deliberate choice, not an oversight. Compare's real use case is "what
changed since a previous date," which the existing per-day granularity
already serves; adding intra-day run versioning would be scope creep that
also risks breaking `load_latest_category_run()`'s "sorted, take last"
assumption, for no benefit to this feature.

---

## Output Examples

**JSON** (`data/json/sample_profiles.json`, one array per run):

```json
{
  "id": "44196397",
  "username": "elonmusk",
  "display_name": "Elon Musk",
  "bio": "Mars, cars, chips, and internet tubes",
  "location": "Mars",
  "website": "https://tesla.com",
  "profile_image": "https://pbs.twimg.com/profile_images/sample/elonmusk.jpg",
  "banner_image": "https://pbs.twimg.com/profile_banners/sample/elonmusk",
  "protected": false,
  "verified": true,
  "followers": 200000000,
  "following": 900,
  "tweets": 45000,
  "likes": 30000,
  "media_count": 5000,
  "created_at": "2009-06-02T20:12:29Z",
  "pinned_tweet_id": "1234567890123456789",
  "language": "en",
  "is_blue_verified": true,
  "profile_url": "https://x.com/elonmusk",
  "scraped_at": "2026-08-07T12:44:06.062723Z"
}
```

**CSV** (`data/csv/sample_users.csv`, one running file, upserted by username):

```csv
id,username,display_name,...,followers,following,tweets,...,verified,profile_url
44196397,elonmusk,Elon Musk,...,200000000,900,45000,...,True,https://x.com/elonmusk
```

> These two sample files are fixtures generated by
> `python scripts/generate_sample_output.py` (illustrative data, not a live
> scrape) so you can see the exact shape of real output without needing
> credentials.

---

## Error Handling

Each username is scraped independently; a failure never aborts the batch.
Handled conditions, each mapped to a typed exception in `app/exceptions.py`:

| Condition | Exception | Retried? |
|---|---|---|
| Username doesn't exist | `UserNotFoundError` | no |
| Account suspended | `AccountSuspendedError` | no |
| Account protected | `ProtectedAccountError` | no |
| Malformed username | `InvalidUsernameError` | no |
| Login/session failure | `AuthenticationError` | no (aborts the run — nothing to scrape without a session) |
| Rate limited | `RateLimitError` | yes, separate long backoff (`RATE_LIMIT_BASE_SECONDS`/`RATE_LIMIT_MAX_SECONDS`) honoring X's own reported reset time when available |
| Network timeout | `NetworkTimeoutError` | yes, exponential backoff |
| Other transient 5xx/connection errors | `TransientRequestError` | yes, exponential backoff |

Tweet scraping (`app/tweet_scraper.py`) uses its own, lower concurrency
(`TWEET_SCRAPE_CONCURRENCY`, default 3) than profile validation
(`CONCURRENCY_LIMIT`) — X's timeline-read endpoint rate-limits more
aggressively than the profile-lookup endpoint. At the end of a tweet-scrape
run, a summary reports accounts requested/succeeded/rate-limited/failed and
tweets collected, so a partial run is never reported as if it were complete.

The end-of-run summary table lists per-username status, and full detail goes
to `data/logs/scraper_YYYY-MM-DD.log`.

---

## Testing

```bash
pytest -q
```

268 tests, all fully offline - no network, no X/LLM credentials, and no
real embedding model load (RAG tests use a small deterministic stub
embedder plus a real tmp_path-backed Chroma store). Covers the scraping
core (`app/exporter.py`, `app/models.py`, `app/utils.py`, `app/cache.py`,
the Twikit-response-to-`UserProfile` parsing logic in `app/client.py` via a
fake `SimpleNamespace` user object), the category-intelligence pipeline
(`app/category_agent.py`, `app/account_ranker.py`, `app/account_discovery.py`,
`app/analysis.py`, `app/storage.py`), and everything added by this
transformation - signal scoring/confidence, topic matching, report
comparison, story opportunities/briefs, and the full RAG stack
(`app/rag/*`, `ui/ask_runner.py`).

Lint/format:

```bash
ruff check .
black .
isort .
```

---

## Troubleshooting

- **`Couldn't get KEY_BYTE indices`** (or `AttributeError: 'ClientTransaction'
  object has no attribute 'key'`). X changed its `ondemand.s.js` bundle in
  March 2026, breaking upstream `twikit`'s login/transaction parsing — this
  is why the project depends on `twifork` instead (see "Authentication"). If
  it recurs even with `twifork`, check for a newer `twifork`/`twikit`
  release, since X's obfuscation changes periodically.
- **`Password login to X failed ... use X_AUTH_TOKEN/X_CT0 cookies instead`**
  Expected if `X_USERNAME`/`X_EMAIL`/`X_PASSWORD` are set but cookies aren't
  — X has retired password login outright. Follow "Authentication" above.
- **`No cached session and no credentials configured`**
  Neither a cached session nor `X_AUTH_TOKEN`/`X_CT0` are set in `.env`.
  Follow "Authentication" above.
- **`X_AUTH_TOKEN/X_CT0 cookies were rejected`**
  The cookies expired or were copied incorrectly (extra whitespace, wrong
  cookie, logged out in that browser since). Re-export fresh values.
- **Frequent `RateLimitError` / long backoff waits.** Lower
  `CONCURRENCY_LIMIT` and/or raise `REQUEST_DELAY_SECONDS` in `.env`. For
  tweet scraping specifically, lower `TWEET_SCRAPE_CONCURRENCY` (default 3)
  further, or raise `RATE_LIMIT_BASE_SECONDS`/`RATE_LIMIT_MAX_SECONDS` so
  retries wait long enough for X's rate-limit window to actually reset
  instead of repeatedly retrying into the same window.
- **Stale-looking data.** Results are served from cache for
  `CACHE_TTL_SECONDS`. Use `--no-cache` or delete the relevant file under
  `CACHE_DIR` to force a refresh.
- **A specific username always fails.** Check the log line's exception type
  — `UserNotFoundError`/`AccountSuspendedError`/`ProtectedAccountError` mean
  the account genuinely can't be read; that's expected, not a bug.
- **`ModuleNotFoundError: No module named 'app'`** when running a script
  under `scripts/` directly (e.g. `python scripts/backfill_rag_index.py`).
  Python puts the script's own directory on `sys.path`, not the project
  root, so `app`/`ui` aren't importable even when your shell's current
  directory is `x-profile-scraper/`. Either install the project in editable
  mode once (`pip install -e .`) or prefix the command with `PYTHONPATH=.`
  (`PYTHONPATH=. python scripts/backfill_rag_index.py`). `python main.py ...`
  is unaffected - it's the top-level entry point, already on the right path.

---

## Future Enhancements

The architecture leaves room for, without requiring changes to existing
modules:

- Followers/following analysis, subject to platform capabilities and ToS.
- Additional storage backends — implement `BaseExporter` in `exporter.py`
  for SQLite/PostgreSQL/MongoDB and register it with `ExporterRegistry`
  (the vector store's Chroma metadata could also move to whichever backend
  is chosen, though the flat-file snapshot format `app/rag/indexer.py`
  reads from would stay unchanged).
- A model-based reranker (e.g. a cross-encoder) as a drop-in replacement
  for `app/rag/reranker.py`'s current heuristic - the interface
  (`rerank(chunks, top_n) -> chunks`) is already swappable.
- A FastAPI service exposing `CategoryIntelligenceAgent`/Ask Intelligence
  programmatically, for integrations beyond the Streamlit UI.
- A scheduler (cron/APScheduler) invoking `main.py analyze <category>`
  periodically, so Compare/Story Opportunities always have a fresh
  same-category, different-date pair to work with.
- Docker packaging.
- Per-tweet topic labels (from a real classifier, not the current
  keyword-overlap heuristic in `app/topic_matching.py`) for more precise
  evidence grouping and story-opportunity scoring.
