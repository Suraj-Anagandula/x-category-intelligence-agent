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

---

## Project Architecture

```text
x-profile-scraper/
│
├── app/
│   ├── client.py       # Twikit session management + Twikit User -> UserProfile mapping
│   ├── scraper.py       # Concurrency, retry orchestration, progress bar, logging
│   ├── models.py        # Pydantic UserProfile / ScrapeResult
│   ├── exporter.py       # JSON/CSV exporters behind a BaseExporter interface
│   ├── cache.py         # TTL file cache keyed by username
│   ├── config.py         # pydantic-settings driven configuration
│   ├── logger.py         # loguru console + rotating file sinks
│   ├── exceptions.py     # Typed exception hierarchy
│   └── utils.py         # Username parsing/validation, retry/backoff helper
│
├── data/
│   ├── json/            # Timestamped JSON exports (sample_profiles.json checked in)
│   ├── csv/             # Running users.csv (sample_users.csv checked in)
│   └── logs/            # Daily rotating log files
│
├── ui/                    # Streamlit presentation layer (see "Streamlit UI" below)
│   ├── pipeline_runner.py # Calls the existing CategoryIntelligenceAgent - no reimplementation
│   ├── data_loader.py     # Read-only access to existing data/tweets, data/csv
│   ├── charts.py          # Plotly chart builders
│   ├── components.py      # Reusable st.* render functions
│   ├── styles.py          # Theme-neutral CSS
│   └── utils.py           # Formatting/validation helpers
│
├── scripts/
│   └── generate_sample_output.py  # Regenerates the checked-in sample fixtures
│
├── tests/                # pytest unit tests for exporter/models/utils/parsing/ui
├── main.py               # Typer CLI entry point
├── streamlit_app.py       # Streamlit UI entry point (streamlit run streamlit_app.py)
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

Each module has one responsibility and depends only on the layers below it
(`client`/`cache` -> `scraper` -> `main`), so any piece can be swapped or
tested independently — e.g. `exporter.py` doesn't know Twikit exists, and
`models.py`/`utils.py` have zero I/O.

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

## Streamlit UI

A visual dashboard is available as a second interface over the same backend
pipeline - it's purely presentational and reuses `CategoryIntelligenceAgent`
exactly as the CLI does; nothing about discovery, validation, ranking,
scraping, retry/rate-limit handling, or storage is reimplemented.

Install the extra dependencies (already included in `requirements.txt`; if
installing via `pyproject.toml` extras, use `pip install -e ".[ui]"`), then:

```bash
streamlit run streamlit_app.py
```

Pages (navigate via the sidebar):

- **Dashboard** — landing page; shows the current report's headline stats,
  top accounts, sentiment, trending topics, and AI summary once one exists.
- **Analyze Category** — enter a category and candidate limit / top accounts /
  tweets-per-account, then run the pipeline. Progress is reported per stage
  (Category Context → Account Discovery → Profile Validation → Account
  Ranking → Tweet Collection → AI Analysis → Export) as the backend actually
  completes each one - not simulated.
- **Accounts** — the ranked account table, plus a per-account detail view
  including its LLM discovery reason.
- **Tweets** — browse/search collected tweets by account and free-text.
- **Analytics** — Plotly charts: ranking score, followers (log scale),
  relevance vs. score, sentiment, engagement by account, tweet distribution.
- **Run History** — every previously saved `data/tweets/<category>/<date>.json`
  run, browsable without making any new X requests.
- **Settings** — the active configuration (LLM provider/model, concurrency,
  retry/rate-limit backoff, cache) - credentials are only ever shown as
  "Configured"/"Not configured", never as values.
- **Downloads** — the real on-disk run JSON and consolidated/users CSVs.

Only the "Run Analysis" button on the Analyze Category page ever triggers a
pipeline run; every other page reads from the current session's report or
from disk, so simply browsing the dashboard never makes X/LLM requests.

Both interfaces (`main.py` and `streamlit_app.py`) share the same `.env`
configuration and the same `data/` output.

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

Covers `app/exporter.py`, `app/models.py`, `app/utils.py`, `app/cache.py`,
and the Twikit-response-to-`UserProfile` parsing logic in `app/client.py`
(via a fake `SimpleNamespace` user object, so no network/credentials are
needed to run the suite).

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
- **`ModuleNotFoundError: No module named 'app'`** when running scripts
  directly. Run from the project root (`x-profile-scraper/`) so `app/` is
  importable, or run via `python main.py ...` which already does this.

---

## Future Enhancements

The architecture leaves room for, without requiring changes to existing
modules:

- Recent public posts collection, engagement analytics, sentiment analysis
  (new methods on `TwikitProfileClient` + new fields/models).
- Followers/following analysis, subject to platform capabilities and ToS.
- Additional storage backends — implement `BaseExporter` in `exporter.py`
  for SQLite/PostgreSQL/MongoDB and register it with `ExporterRegistry`.
- A FastAPI service or Streamlit dashboard wrapping `ProfileScraper`.
- A scheduler (cron/APScheduler) invoking `main.py` periodically.
- Docker packaging.
- AI-powered profile summarization as a post-processing step over
  `UserProfile` records.
