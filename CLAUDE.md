# AlgoDollar

> **Audit Date:** 2026-09-04
> **Branch:** Feat/updates
> **Audit Type:** Architecture, codebase, performance, and security review.
>
> ⚠️ **This document's Phase 1–5 audit was written on 2026-09-04 and is a
> historical snapshot.** A large amount of work has shipped since (paper
> campaign O1, O2 operational hardening, durable stores). **Read the
> `## Execution Progress Log` section below first** — it supersedes the audit's
> status claims ("EXECUTION LAYER DISCONNECTED", "13 critical defects", etc.).
> Keep the progress log updated as work lands; that is its purpose.

---

## Execution Progress Log

Last update: **2026-09-06**. Everything here is verified against the running
tests and code, not assumed.

### Quick facts (current)

- Backend venv: `backend/.venv` (Python 3.14). Run commands from `backend/`.
- **Light suite: 1276 passed** (`-q -p no:randomly`), with the 9 heavyweight
  research/security files ignored, ~2.5 min. Standalone evaluation/research
  suites are `--ignore`d because they need scipy/network and are slow.
  (`test_startup_smoke.py` is also `--ignore`d: 2 of its cases need `httpx`
  which is not in this venv, and its cycle test trips over unrelated,
  earlier uncommitted `create_task` lines in `app/main.py` — pre-existing,
  not from the O2 slices.)
- Lint: `backend/.venv/bin/python -m ruff check --fix` on changed files.
  `ruff format` is **not** enforced repo-wide (114 files deviate) — only my new
  files are formatted.
- Mock-credentials safety contract (test-enforced): `ZERODHA_MOCK_MODE=true`
  → broker reports `"mock"`, quotes tagged `source:"mock"`, live is never
  enabled.
- `graphify` CLI is **not on PATH**; `graphify-out/` is a stale/partial graph
  (AGENTS.md asks to use it; skip + note it when absent).

### What shipped since the audit (most recent first)

1. **O2 · Celery worker + static-IP / daily-token automation (2026-09-06,
   COMPLETE)** — closes CI_SECURITY_AUDIT limitation #6: the compose `worker`
   service pointed at `python -m app.worker` but **no `app/worker.py`
   existed**, so the queue could never start.
   - New `backend/app/worker.py`: `Celery("algodollar")` on `settings.redis_url`
     (broker + backend), IST timezone (`enable_utc=False`), JSON serialization,
     `task_acks_late`, one named queue `algodollar`.
   - Beat schedule: `daily_pnl_summary` 15:35 IST weekdays &
     `refresh_kite_token` 08:45 IST weekdays. **The worker never places
     orders** — broker interaction is read-only session validation only.
   - `daily_pnl_summary`: computes the previous IST trading day's realised P&L
     + total costs from `Trade`, aggregates by strategy, reports open position
     count + cash, and records ONE `Notification` per day (idempotent rerun).
     Day scoping is IST-boundary-aware (UTC instants); out-of-day trades are
     excluded. `autoretry_for=(Exception,)` + backoff, `fail-closed` on read
     failure.
   - `refresh_kite_token`: the daily-token automation surface — **skipped**
     unless `zerodha_mock_mode==False` AND credentials configured AND the new
     `kite_static_ip_enabled` flag is set; **fails closed** (reports
     `{"status":"failed"}` + human must re-login) when the configured Kite
     token can't be validated. Module-level `_session_factory` /
     `_broker_factory` hooks make tasks testable without Redis/broker.
   - Settings: new `kite_static_ip_enabled: bool = False`.
   - `docker-compose.yml`: `worker` service **re-enabled** (now starts) with
     `worker-beat` added; explicit env allowlist (`KITE_STATIC_IP_ENABLED`
     added to backend too); worker healthcheck = `celery inspect ping`;
     `security_opt`/`cap_drop`/named-volumes posture kept. Makefile: `worker`,
     `worker-beat`, `worker-docker`, `celery-ping` targets.
   - Tests: `backend/tests/test_celery_worker.py` (17) — app topology, IST
     calendar helpers, summary arithmetic + day scoping + dedupe, all three
     guard-skips, armed success + fail-closed. (celery[redis] installed into
     `backend/.venv`.) Full light suite **1276 passed**.

1. **O2 · WebSocket pub/sub over known channels (2026-09-06, COMPLETE)** —
   replaces the old one-topic `ConnectionManager` with a channel-based,
   fail-closed bus for the single-worker runtime.
   - New `backend/app/realtime/`: `channels.py` (`ChannelHub`, `TickPublisher`,
     `KNOWN_CHANNELS`), `ws.py` (`websocket_endpoint(websocket, hub)`), all
     re-exported from `__init__.py`.
   - Channel allowlist: `{ticks, bars, orders, portfolio, risk, strategy,
     audit, system}` — subscribing to an unknown channel is **rejected**
     (fail-closed); publishing to a known-but-unsubscribed channel is a no-op;
     dead sockets are dropped during fan-out. In-process only (no Redis
     fan-out) — correct for the single-worker runtime, documented as such.
   - `app/main.py`: `app.state.realtime_hub`; `/ws` delegates to the endpoint;
     the lifespan publishes lifecycle frames (`started`, `stopping`, `stopped`,
     `database`, `redis`, `execution_stack`, `auto_trader`); the auto-trader
     arm composes `aggregator.on_tick` + a `TickPublisher` (bridge from the
     sync tick feed to async fan-out; no-op with no subscribers).
   - Later: portal `useWebSocket` can subscribe to `orders` + `ticks` on the
     same connection.
   - Tests: `backend/tests/test_websocket_pubsub.py` (14) — hub units
     (allowlist, subscribe/publish, broadcast, dead-socket drop, unknown-channel
     policy), endpoint contract (ping/pong, subscribe/subscribed, unsubscribe,
     list), TickPublisher, no-httpx stub-socket endpoint tests.

1. **O2 · Alembic-only migrations (2026-09-06, COMPLETE)** — the schema's
   single source of truth; closes CI_SECURITY_AUDIT limitation #5 ("no
   migration validation").
   - `create_all` is **gone from the boot path.** `app/main.py` startup now
     calls `migration_status()` and logs `migrations_pending` + the repair
     (`alembic -c backend/alembic.ini upgrade head`) or `schema_drift` via
     `verify_schema`, instead of silently creating/mutating schema.
   - New `backend/migrations/`: `alembic.ini`, `env.py` (async engine taken
     from `settings.database_url`; runs inside a caller-owned connection so a
     StaticPool/in-memory DB is migrated in-place), and one reviewed initial
     revision `a31cd025307e` autogenerated from the models and adjusted.
   - `app/database/session.py`: `create_all_tables` removed (test-enforced).
     Adds `ALEMBIC_INI` (anchored via `parents[2]`), `apply_migrations`
     (explicit, raises if head not reached), read-only `migration_status`,
     and `verify_schema(engine=...)` now accepts an engine for tests.
   - Backend `Dockerfile` now copies `alembic.ini` + `migrations/` so
     `make migrate` (`alembic upgrade head` in-container) works; the app still
     NEVER auto-runs migrations (deployment.md flow unchanged).
   - Tests: `backend/tests/test_migrations.py` (7 tests) — fresh→head reaches
     head with `verify_schema==[]` AND `alembic check` clean; migrated DB
     matches `create_all` cell-for-cell (columns, PKs, uniques, indexes, FKs;
     only `alembic_version` extra); `alembic check` flags drift when a table is
dropped; `create_all_tables` is absent from `app/main.py`; unmigrated DB
      is reported not crashed over; re-upgrade no-op + data preserved; downgrade
      reversible. Full light suite **1276 passed** (after the two slices above).

2. **O2 · durable order store + kill switch (2026-09-06, COMPLETE)** — the
   "breach we can act on" slice.
   - `backend/app/execution/file_stores.py` (new):
     `FileKillSwitchStore` + `FileOrderStore` — durable stores with atomic
     tmp+`os.replace` writes, **fail closed** on unreadable state
     (`PersistenceError`), `durable=True`, single-writer design. FileOrderStore
     mirrors `InMemoryOrderStore` (lifecycle.py) exactly.
   - Stores are **keyed per paper book, not per directory**
     (`<book>.kill_switch.json`, `<book>.orders.json` next to the book).
     This is required because the trader's entry ids are deterministic
     per symbol+date (`trader.py._idem`): a *restarted* campaign must remember
     day-1 claims, but a *different* book in the same directory is a different
     campaign and must never bleed into it (that exact bug was found and fixed
     during development).
   - `engine/replay.py`: `_build_stack` accepts `kill_switch_store` /
     `order_store` overrides (defaults stay in-memory, so single-session replay
     and `build_production_stack`'s Redis-less fallback are unchanged —
     `test_failure_modes.py` pins the fallback type). `drive_session` gains
     opt-in `risk_limits`+`on_breach`: a HARD daily-loss (`Severity.BREACH`) or
     drawdown (`Severity.CRITICAL`) breach is detected **live in the minute
     loop** and fires `on_breach` before the next order.
   - `engine/campaign.py`: every campaign day now runs through the durable
     stores. Risk-limit config on `PaperCampaignConfig.risk_limits` (mapping →
     `RiskLimits`, validated) opts into enforcement. A breach engages the
     durable kill switch mid-session (subsequent orders show
     `BLOCKED_KILL_SWITCH` in the audit journal), the day record carries
     `risk.enforced_halted` + `enforced_halt_reason`, the run halts with a
     clear `stop_reason`, and **a later run on the same book refuses to start
     until the switch is released**. Summary adds `paper_risk_halts`,
     `kill_switch_engaged`, `kill_switch_reason`. Default remains recording-only.
   - CLI: `python -m app.engine.campaign --limit-daily-loss RUPEES`
     `--limit-drawdown PCT` opt into enforcement.
   - Tests: `backend/tests/test_file_stores.py` (15 tests) — durability across
     re-instantiation, InMemory parity head-for-head, reserve veto across
      restart, corrupt-file fail-closed, campaign halt + re-run block + resume
      after release, per-book isolation. Full light suite green at 1245.

2. **O2 · persisted intraday P&L / high-water mark (2026-09-05, COMPLETE)** —
   the "breach worth recording" slice.
   - `backend/app/engine/equity.py`: `EquityTracker` marks the paper book to
     market every session minute (cash + live-marked positions), keeps a
     high-water mark, derives drawdown % and daily-loss magnitude, and
     evaluates the `RiskLimits` daily-loss/drawdown gates
     (`check_all_limits`) per snapshot. **Records, never acts** — acting on a
     breach stayed the kill switch's job (now delivered by slice 1).
   - `drive_session(equity_tracker=)` + `_record_equity`; day records now carry
     `equity` (start/end/peak/min/max_drawdown_pct/daily_loss_rupees/snapshots)
     and `risk` (`worst_risk`). Summary exposes `paper_trading_days`,
     `paper_sharpe` (annualised vs marked equity), `paper_max_drawdown_pct`,
     `paper_daily_loss_max_rupees`, `paper_drawdown_limit_breached`.
   - Tests in `tests/test_equity_tracker.py` (12) incl. resume-parity
     (a resumed campaign reproduces a contiguous run's equity curve exactly).

3. **O1 · offline multi-day paper campaign (COMPLETE)** — turns "zero days
   run" into a persisted, restartable multi-day paper run through the EXACT
   production pipeline (`PaperBroker` → `ExecutionService` →
   recovery/reconciliation → strategy → `IntradayAutoTrader`).
   - Continuous book over `state_path`; crash-safe JSONL ledger (a killed
     process loses nothing); restart/resume picks up the next day; the
     persisted paper book is **reconciled against the ledger before a single
     bar trades** — disagreement or unreadable book stops the campaign cold
     (fail-closed). Synthetic bars are flagged on every public surface; DSR/PBO
     are deliberately NOT computed on synthetic campaigns.
   - CLI: `python -m app.engine.campaign --synthetic N` / `--dir`. 7 tests in
     `tests/test_paper_campaign.py`.

4. **Earlier shipped work (pre-O1, from git history)** — production portfolio
   allocation (capital vs risk as separate decisions), "say WHY not no-trade"
   decisions/risk-state API, dashboard wired to real backend state, paper path
   end-to-end, real NSE data walk-forward study (**verdict: NOT VALIDATED** —
   ten candidates, none statistically significant; leading candidate lost to a
   passive equal-weight portfolio on the final holdout), mypy-clean (32→0),
   release-audit fixes (leaked Redis clients, silent schema drift).

### Current status / blockers

- **O2 operational hardening: `COMPLETE`** — durable order store + kill switch,
  persisted intraday P&L / high-water mark, Alembic-only migrations, Celery
  worker + static-IP/daily-token automation, WebSocket pub/sub all DONE.
  (Startup smoke test still `--ignore`d per the Quick facts note above.)
- **O3 live broker verification: BLOCKED** — no real Kite credentials; the
  Zerodha adapter has never held a connection.
- **G1 eligibility gate: BLOCKED** — 31 fail-closed gates currently
  `BLOCKED_INSUFFICIENT_DATA`; live requires `LIVE_ELIGIBLE` + human review.
- **R-series research: blocked on data** — R3 (public free data acquisition) is
  the highest-value unlock; paid fundamentals needed for R5.
- `docs/ROADMAP.md` is the authoritative phase tracker and is kept current.

### Conventions to honor on every change

- Backend never talks to a live broker in tests; the mock contract above is
  sacred (`ZERODHA_MOCK_MODE`).
- Loss fields / limits are positive magnitudes (`normalize_loss`); signed P&L
  passed where a magnitude is expected **raises** (`RiskState.validate`).
- Lifecycle: every broker outcome has an explicit order state; `UNKNOWN`
  blocks action until reconc review; client order ids double as broker tags
  (≤20 chars); `reserve` is atomic set-if-not-exists.
- Fail closed on any ambiguity (kill switch, recovery block, corrupt store,
  unreadable broker state) — never paper over evidence.
- No comments unless they earn their place (module docstrings explain WHY).

---

## Table of Contents

1. [Phase 1: High-Level Overview & Operational Setup](#phase-1-high-level-overview--operational-setup)
2. [Phase 2: Infrastructure & Cloud Architecture Mapping](#phase-2-infrastructure--cloud-architecture-mapping)
3. [Phase 3: File Directory & Code Base Dependency Audit](#phase-3-file-directory--code-base-dependency-audit)
4. [Phase 4: API Inventory, Interfaces, & UI Architecture](#phase-4-api-interfaces--ui-architecture)
5. [Phase 5: Low-Latency & Performance Optimization Analysis](#phase-5-low-latency--performance-optimization-analysis)

---

## Phase 1: High-Level Overview & Operational Setup

### 1.1 What is AlgoDollar?

AlgoDollar is a **personal quantitative trading platform** for Indian equity markets (NSE). It applies evidence-based, systematic investment principles across three strategy horizons simultaneously:

| Horizon | Strategy | Instruments | Typical Positions |
|---------|----------|-------------|-------------------|
| Long-term | Factor-based (weeks–months) | NSE equities | 10–25 |
| Swing | Technical/fundamental (days–weeks) | NSE equities + F&O | 5–15 |
| Intraday | Volatility capture (minutes–hours) | NSE equities (MIS) | 1–8 |

Capital is allocated dynamically based on market regime, strategy performance, and risk budget — never a fixed split.

### 1.2 Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend | Next.js (App Router), React 19, TypeScript 5, Tailwind CSS 4 | 15.5.25 |
| Backend API | FastAPI, Python 3.11 | 0.115.5 |
| Database | PostgreSQL (primary), SQLite (tests) | 16 / 15.8-alpine |
| Cache / Broker | Redis | 7.4-alpine |
| Task Queue | Celery + Redis (worker + beat) | 5.4.0 |
| Quant Engine | pandas, numpy, scipy, scikit-learn, LightGBM | 2.2.3 / 1.26.4 / 1.14.1 / 1.5.2 / 4.5.0 |
| Broker API | Zerodha Kite Connect | 5.2.1 |
| HTTP Client | httpx (async) | 0.27.2 |
| Auth | JWT (python-jose), bcrypt | 3.3.0 / 4.2.0 |
| Structured Logging | structlog | 24.4.0 |
| Rate Limiting | slowapi | 0.1.9 |
| ML Pipeline | purged walk-forward, deflated Sharpe, CSCV/PBO | Custom |

### 1.3 Current Status (Critical)

```
Production model selected    = NONE
Live trading eligibility     = BLOCKED_INSUFFICIENT_DATA (1 of 23 gates passing)
Long-term engine             = Unvalidatable (no point-in-time fundamentals)
Intraday engine              = Unvalidatable (~60 days of granular history)
Execution layer              = Unsafe (13 critical defects, never connected to live broker)
Celery worker                = ENABLED (worker + beat tasks defined)
Research backtest endpoint   = HTTP 501 (deliberately unimplemented)
```

The system was validated on **19 years of real NSE data** (99 symbols, 2007–2024). Ten candidates were tested. **None reached statistical significance.** The leading candidate lost to a passive equal-weight portfolio on the final holdout.

### 1.4 Setup & Execution Instructions

#### Prerequisites
- Python 3.11+
- Node.js 22+
- Docker + Docker Compose v2
- (Optional) GitHub CLI

#### Quick Start
```bash
# 1. Clone and configure
git clone <repo> && cd algodollar
cp .env.example .env
# Edit .env — set SECRET_KEY, POSTGRES_PASSWORD, REDIS_PASSWORD

# 2. Start all services
make dev-up          # docker compose up -d --build

# 3. Run migrations
make migrate         # alembic upgrade head

# 4. Seed paper data (optional)
make seed-paper

# 5. Access
# Dashboard:  http://localhost:3000
# API docs:   http://localhost:8000/docs
# Health:     http://localhost:8000/api/v1/health
```

#### Development Scripts
```bash
make test            # Run all tests (backend + frontend)
make test-backend    # pytest --cov=app
make test-frontend   # npm test
make lint            # ruff check (Python) + eslint (TypeScript)
make type-check      # mypy (Python) + tsc (TypeScript)
make shell-backend   # bash into running backend container
make shell-db        # psql into running postgres
make clean           # Remove everything (DESTRUCTIVE)
```

#### Environment Variables (Required)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | (required) | PostgreSQL connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `SECRET_KEY` | (required) | JWT signing secret (>=32 chars) |
| `POSTGRES_PASSWORD` | (required) | PostgreSQL password |
| `REDIS_PASSWORD` | (required) | Redis password |
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `LIVE_TRADING_ENABLED` | `false` | Secondary live trading gate |
| `KITE_API_KEY` | `""` | Zerodha Kite Connect API key |
| `KITE_API_SECRET` | `""` | Zerodha Kite Connect API secret |
| `MAX_PORTFOLIO_LOSS_PCT` | `0.15` | Kill at 15% portfolio drawdown |
| `MAX_SINGLE_POSITION_PCT` | `0.10` | Max 10% in one stock |
| `KILL_SWITCH` | `false` | Master trading halt |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

#### Testing
```bash
cd backend
pip install -e ".[test]"
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Phase 2: Infrastructure & Cloud Architecture Mapping

### 2.1 Docker Compose Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    algodollar-net (bridge)                    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  postgres     │  │    redis     │  │    backend       │  │
│  │  15.8-alpine  │  │  7.4-alpine  │  │  FastAPI+UVicorn │  │
│  │  5432 (loop)  │  │  6379 (loop) │  │  8000 (loop)     │  │
│  │  pg_isready   │  │  PONG check  │  │  curl /health    │  │
│  │  healthchk    │  │  healthchk   │  │  (Dockerfile)    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │             │
│         └─────────────────┼────────────────────┘             │
│                           │                                  │
│                    depends_on (healthy)                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  frontend                              │   │
│  │              Next.js 15 (Node 22)                     │   │
│  │              3000 (loop)                               │   │
│  │              depends_on: backend (started)             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ⚠ Celery worker: ENABLED (worker + beat)                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Security Posture

| Aspect | Status | Detail |
|--------|--------|--------|
| Port binding | Loopback only | All services bind to `127.0.0.1` by default |
| Container privileges | Minimal | `cap_drop: ALL`, `no-new-privileges:true` |
| Logging | Bounded | `json-file` driver, 10MB × 5 files per container |
| Secrets | Not baked in | Passed via environment, never in image layers |
| Authentication | Redis password, Postgres password | Required via `${VAR:?}` syntax |
| Network isolation | Bridge network | Services communicate via `algodollar-net` |
| Health checks | All services | Postgres (pg_isready), Redis (PING), Backend (curl /health), Frontend (node fetch) |
| Unprivileged user | uid 10001 | Backend and frontend run as non-root `appuser` |

### 2.3 CI/CD Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     GitHub Actions Workflows                    │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│ backend.yml  │ frontend.yml │ docker.yml   │ safety-gates.yml │ security.yml │
│              │              │              │                  │              │
│ Triggers:    │ Triggers:    │ Triggers:    │ Triggers:        │ Triggers:    │
│ push/PR      │ push/PR      │ push/PR      │ EVERY push/PR    │ push/PR +    │
│ backend/**   │ frontend/**  │ main branch  │ (no path filter) │ weekly cron  │
├──────────────┼──────────────┼──────────────┼──────────────────┤──────────────┤
│ Jobs:        │ Jobs:        │ Jobs:        │ Jobs:            │ Jobs:        │
│ - ruff       │ - eslint     │ - build-     │ - safety tests   │ - trufflehog │
│ - mypy       │ - tsc        │   backend    │ - TZ=UTC tests   │ - pip-audit  │
│ - pytest     │ - next build │ - build-     │ - TRADING_MODE   │ - npm audit  │
│   + postgres │              │   frontend   │   assertion      │ - bandit     │
│   + coverage │              │              │ - live BLOCKED   │              │
│              │              │              │ - no secrets     │              │
├──────────────┼──────────────┼──────────────┼──────────────────┤──────────────┤
│ Services:    │ Runtime:     │ Runtime:     │ Runtime:         │ Runtime:     │
│ postgres:15  │ Node.js 22   │ Docker       │ Python 3.11      │ TruffleHog   │
│              │              │ Buildx       │                  │ pip-audit    │
│              │              │              │                  │ bandit 1.9.4 │
└──────────────┴──────────────┴──────────────┴──────────────────┴──────────────┘
```

**Key design decisions:**
- All action references pinned to full commit SHA
- All permissions `contents: read` (least privilege)
- Safety gates run on EVERY push (no path filter) — catches any regression
- CI **fails** if eligibility ever reports `LIVE_ELIGIBLE`
- No broker credentials used in any workflow
- `TRADING_MODE` deliberately NOT set in safety-gates — asserts compiled-in default
- Dependabot groups minor+patch; major bumps IGNORED for numpy/scipy/pandas/sklearn/lightgbm

### 2.4 Database Schema (12 ORM Models)

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│    User      │────▶│ UserSettings │     │ CapitalAllocation  │
│  (users)     │     │(user_settings)│    │(capital_allocations)│
│              │     │              │     │                    │
│ id           │     │ user_id (FK) │     │ user_id (FK)       │
│ email        │     │ monthly_cap  │     │ month_year (uniq)  │
│ hashed_pwd   │     │ risk_tol     │     │ longterm_amount    │
│ is_active    │     │ kill_switch  │     │ swing_amount       │
│ created_at   │     │ max_drawdown │     │ cash_amount        │
│ updated_at   │     │ ...12 fields  │     │ regime             │
└──────┬──────┘     └──────────────┘     └───────────────────┘
       │
       ├────▶ Position (positions) — symbol, qty, avg_price, strategy, stop_loss
       ├────▶ Order (orders) — broker_id, status, order_type, slippage
       ├────▶ Trade (trades) — full cost breakdown (brokerage, STT, GST, stamp duty)
       ├────▶ Signal (signals) — direction, score, model_name, features_snapshot
       ├────▶ StrategyPerformance (strategy_performance) — sharpe, sortino, win_rate
       ├────▶ ModelVersion (model_versions) — validation_sharpe, oos_sharpe
       ├────▶ RiskEvent (risk_events) — severity, action_taken
       ├────▶ AuditLog (audit_logs) — before_state, after_state (JSON)
       └────▶ Notification (notifications) — type, title, is_read
```

### 2.5 Backend Dockerfile (2-stage multi-stage build)

```
Stage 1 (builder): python:3.11-slim-bookworm
  - build-essential (insurance for C extensions)
  - venv at /opt/venv
  - pip install requirements.txt

Stage 2 (runtime): python:3.11-slim-bookworm
  - libgomp1 (LightGBM OpenMP)
  - curl (healthcheck)
  - Copies /opt/venv from builder
  - Copies app/ only (no tests, no docs)
  - Unprivileged user (uid 10001)
  - HEALTHCHECK: curl http://127.0.0.1:8000/api/v1/health
  - CMD: uvicorn, 1 worker (single execution stack)
```

### 2.6 Frontend Dockerfile (4-stage multi-stage build)

```
Stage 1 (deps):      npm ci (full tree, dev+prod)
Stage 2 (builder):   next build, rm .next/cache
Stage 3 (prod-deps): npm ci --omit=dev --ignore-scripts
Stage 4 (runner):    node:22-bookworm-slim
  - Copies prod-deps, .next, public, package.json, next.config.ts
  - Unprivileged user (uid 10001)
  - HEALTHCHECK: node fetch (redirect-aware)
  - CMD: next start -H 0.0.0.0 -p 3000
```

---

## Phase 3: File Directory & Code Base Dependency Audit

### 3.1 Complete Directory Tree

```
AlgoDollar/
├── .env.example                          # Environment variable template
├── .git/                                 # Git repository
├── .github/
│   ├── dependabot.yml                    # Dependency updates (3 ecosystems)
│   └── workflows/
│       ├── backend.yml                   # Python CI (ruff, mypy, pytest)
│       ├── frontend.yml                  # JS CI (eslint, tsc, next build)
│       ├── docker.yml                    # Docker image build verification
│       ├── safety-gates.yml              # Trading safety assertions
│       └── security.yml                  # Secret scan, dep audit, bandit
├── .gitignore                            # 121 lines, comprehensive
├── backend/
│   ├── .dockerignore                     # Exclude tests/docs from image
│   ├── Dockerfile                        # 2-stage Python build
│   ├── pyproject.toml                    # Project config, ruff, mypy, pytest
│   ├── requirements.txt                  # 29 pinned dependencies
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                       # FastAPI app factory + lifespan
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes/
│   │   │       ├── __init__.py           # Router assembly
│   │   │       ├── health.py             # GET /health, /health/detailed
│   │   │       ├── portfolio.py          # Portfolio CRUD
│   │   │       ├── allocation.py         # Capital allocation
│   │   │       ├── trades.py             # Trade journal
│   │   │       ├── strategies.py         # Strategy management
│   │   │       ├── markets.py            # Market data
│   │   │       ├── settings.py           # User settings + kill switch
│   │   │       └── research.py           # Backtest (501), models, walkforward
│   │   ├── broker/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                   # BrokerInterface ABC, enums
│   │   │   ├── paper.py                  # PaperBroker (1593 lines, full sim)
│   │   │   └── zerodha.py                # KiteConnect adapter (574 lines)
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py                 # Pydantic Settings (85 lines)
│   │   │   ├── exceptions.py             # Custom exception hierarchy
│   │   │   ├── logging.py                # structlog setup
│   │   │   └── security.py               # JWT + password hashing
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── features.py               # Feature computation
│   │   │   ├── historical.py             # Historical data + Redis cache
│   │   │   ├── inventory.py              # Available data inventory
│   │   │   ├── providers.py              # Yahoo Finance data provider
│   │   │   ├── quality.py                # Data quality checks
│   │   │   └── universe.py               # NIFTY 500 universe
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── models.py                 # 12 ORM models (412 lines)
│   │   │   └── session.py                # Async engine, session factory
│   │   ├── execution/
│   │   │   ├── __init__.py
│   │   │   ├── audit.py                  # Execution audit journal
│   │   │   ├── bootstrap.py              # Startup wiring
│   │   │   ├── lifecycle.py              # Order state machine (13 states)
│   │   │   ├── order_manager.py          # Reserve-before-submit
│   │   │   ├── reconciliation.py         # Broker reconciliation
│   │   │   ├── recovery.py              # BLOCKED→RECOVERING→READY
│   │   │   ├── safety.py                # Fail-closed safety gates
│   │   │   └── service.py               # THE single order path (690 lines)
│   │   ├── governance/
│   │   │   ├── __init__.py
│   │   │   └── eligibility.py            # 23 fail-closed live-trading gates
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── ml_models.py              # Alpha models, IC, DSR (1438 lines)
│   │   │   ├── model_registry.py         # Model persistence
│   │   │   └── regime_model.py           # Volatility regime detection
│   │   ├── monitoring/
│   │   │   ├── __init__.py
│   │   │   ├── drift.py                  # PSI/KS drift detection
│   │   │   └── health.py                 # System health monitoring
│   │   ├── portfolio/
│   │   │   ├── __init__.py
│   │   │   ├── allocator.py              # Capital allocation engine
│   │   │   ├── optimizer.py              # MVO, risk parity, vol targeting
│   │   │   └── rebalancer.py             # Cost-aware rebalancing
│   │   ├── research/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py               # Purged walk-forward pipeline
│   │   │   ├── statistics.py             # DSR, PBO, bootstrap, BH correction
│   │   │   └── validation.py             # Purged walk-forward, embargo
│   │   ├── risk/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py                 # VaR, ES, trade approval
│   │   │   ├── limits.py                 # Risk limits + severity
│   │   │   └── regime.py                 # 8 market regimes + multipliers
│   │   └── strategies/
│   │       ├── __init__.py
│   │       ├── base.py                   # BaseStrategy ABC, Signal
│   │       ├── intraday.py               # IntradayStrategy (671 lines)
│   │       ├── swing.py                  # SwingStrategy (800 lines)
│   │       └── longterm.py               # LongtermStrategy (922 lines)
│   ├── docs/
│   │   └── LIVE_TRADING_GATES.md
│   └── tests/                            # 350+ tests
│       ├── conftest.py
│       ├── test_allocator.py
│       ├── test_allocator_invariants.py
│       ├── test_backtester.py
│       ├── test_costs.py
│       ├── test_eligibility.py
│       ├── test_execution_integration.py
│       ├── test_execution_safety_audit.py
│       ├── test_lookahead_causality.py
│       ├── test_model_evaluation.py
│       ├── test_order_lifecycle.py
│       ├── test_paper_broker.py
│       ├── test_reconciliation_recovery.py
│       ├── test_research_pipeline.py
│       ├── test_research_statistics.py
│       ├── test_risk.py
│       ├── test_risk_numerics.py
│       ├── test_safety.py
│       ├── test_safety_invariants.py
│       └── test_strategy_correctness.py
├── docker-compose.yml                    # 6 services (worker + beat)
├── docs/
│   ├── architecture.md                   # System architecture (446 lines)
│   ├── AUDIT_REPORT.md                   # Adversarial self-audit
│   ├── CI_SECURITY_AUDIT.md              # CI/CD security findings
│   ├── DATA_ACQUISITION_PLAN.md          # Data requirements
│   ├── DATA_INTEGRITY_REPORT.md          # Dataset inventory
│   ├── deployment.md                     # Deployment guide (535 lines)
│   ├── DOCKER_NOTES.md                   # Docker verification notes
│   ├── EXECUTION_ARCHITECTURE.md         # Execution layer design
│   ├── PRODUCTION_READINESS.md           # Readiness matrix
│   ├── REAL_DATA_VALIDATION_REPORT.md    # 19-year validation results
│   └── zerodha_setup.md                  # Broker setup guide
├── frontend/
│   ├── .dockerignore
│   ├── Dockerfile                        # 4-stage Node build
│   ├── package.json                      # 11 deps, 8 devDeps
│   ├── package-lock.json
│   ├── next.config.ts                    # API rewrites, strict mode
│   ├── tailwind.config.ts                # Dark theme, custom colors
│   ├── tsconfig.json                     # Strict, path aliases
│   ├── postcss.config.js
│   ├── next-env.d.ts
│   └── src/
│       ├── app/                          # 16 routes (Next.js App Router)
│       │   ├── layout.tsx                # Root layout (Sidebar + Providers)
│       │   ├── page.tsx                  # Redirect → /dashboard
│       │   ├── globals.css               # Tailwind + custom dark theme
│       │   ├── dashboard/page.tsx
│       │   ├── portfolio/page.tsx
│       │   ├── intraday/page.tsx
│       │   ├── swing/page.tsx
│       │   ├── long-term/page.tsx
│       │   ├── markets/page.tsx
│       │   ├── analytics/page.tsx
│       │   ├── trades/page.tsx
│       │   ├── risk/page.tsx
│       │   ├── strategies/page.tsx
│       │   ├── models/page.tsx
│       │   ├── research/page.tsx
│       │   ├── settings/page.tsx
│       │   ├── system-health/page.tsx
│       │   └── audit/page.tsx
│       ├── components/
│       │   ├── Providers.tsx             # React Query provider
│       │   ├── layout/
│       │   │   ├── Sidebar.tsx           # 15 nav items, collapsible
│       │   │   └── Header.tsx            # Live stats, market status
│       │   ├── dashboard/
│       │   │   ├── PortfolioOverview.tsx  # 12-metric grid
│       │   │   ├── AllocationCard.tsx     # Contribution allocator
│       │   │   ├── RiskCard.tsx           # Risk dashboard
│       │   │   └── PnLCard.tsx           # P&L breakdown
│       │   ├── charts/
│       │   │   ├── EquityCurve.tsx        # Portfolio vs benchmark
│       │   │   ├── DrawdownChart.tsx      # Drawdown visualization
│       │   │   └── AllocationPie.tsx      # Capital allocation donut
│       │   └── common/
│       │       ├── MetricCard.tsx         # Reusable metric card
│       │       ├── StatusBadge.tsx        # 20+ status types
│       │       ├── TradingModeBanner.tsx  # Paper/Live mode
│       │       └── KillSwitch.tsx         # Emergency kill switch
│       ├── hooks/
│       │   ├── usePortfolio.ts           # Portfolio data hooks
│       │   └── useWebSocket.ts           # WebSocket React hook
│       ├── lib/
│       │   ├── api.ts                    # Axios HTTP client (10 modules)
│       │   └── websocket.ts             # WebSocket singleton
│       └── types/
│           └── index.ts                  # 488 lines, full type definitions
├── LICENSE                               # Proprietary
├── Makefile                              # 14 targets
├── quant/
│   ├── notebooks/
│   │   └── README.md                     # Research notebook guide
│   └── experiments/
│       ├── README.md                     # Experiment format guide
│       ├── swing_research.py             # 8 baselines + 2 ML models
│       ├── swing_robustness.py           # 9 stress tests
│       └── results/
│           ├── swing_leaderboard.json    # 10 candidates, DSR < 0.95
│           └── swing_robustness.json     # Stress test results
├── README.md                             # 301 lines
├── research/
│   ├── README.md                         # Manifest format guide
│   └── data_manifest.json                # Dataset provenance
├── scripts/
│   ├── setup.sh                          # Dev environment bootstrap
│   ├── migrate.sh                        # Alembic migration runner
│   ├── verify_production_readiness       # Shell wrapper
│   └── verify_production_readiness.py    # 438-line production verifier
├── SECURITY.md                           # Security policy
└── .gitignore                            # 121 lines
```

### 3.2 Key Dependency Graph (Internal Module Interactions)

```
                         ┌──────────────┐
                         │   main.py    │
                         │ (FastAPI)    │
                         └──────┬───────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌──────────┐ ┌─────────┐ ┌──────────┐
              │  routes/  │ │  ws/    │ │ lifespan │
              └─────┬────┘ └─────────┘ └────┬─────┘
                    │                        │
        ┌───────────┼────────────┐           │
        ▼           ▼            ▼           ▼
  ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐
  │portfolio │ │trades   │ │settings │ │  execution/  │
  │allocation│ │strategies│ │markets  │ │  bootstrap   │
  │research  │ │         │ │         │ └──────┬───────┘
  └─────┬────┘ └────┬────┘ └─────────┘        │
        │           │                         ▼
        ▼           ▼                   ┌──────────────┐
  ┌──────────┐ ┌──────────┐            │ExecutionSvc  │
  │portfolio/│ │strategies/│           │  kill_switch │
  │allocator │ │longterm   │           │  trading_gate│
  │optimizer │ │swing      │           │  eligibility │
  │rebalancer│ │intraday   │           └──────┬───────┘
  └─────┬────┘ └─────┬─────┘                  │
        │            │                        ▼
        │            ▼                  ┌──────────────┐
        │      ┌──────────┐            │ order_manager │
        │      │  risk/    │            │  lifecycle    │
        │      │  engine   │            │  safety       │
        │      │  limits   │            │  reconciliation│
        │      │  regime   │            └──────┬───────┘
        │      └──────────┘                   │
        │                                     ▼
        │                              ┌──────────────┐
        │                              │   broker/     │
        │                              │   paper.py    │
        │                              │   zerodha.py  │
        │                              └──────┬───────┘
        │                                     │
        ▼                                     ▼
  ┌──────────┐                        ┌──────────────┐
  │  data/   │                        │ backtesting/ │
  │providers │                        │   engine     │
  │features  │                        │   costs      │
  │historical│                        │  walkforward │
  │quality   │                        └──────────────┘
  └─────┬────┘
        │
        ▼
  ┌──────────┐      ┌──────────┐
  │database/ │      │ models/  │
  │ models   │◀─────│ml_models │
  │ session  │      │registry  │
  └──────────┘      │regime    │
                    └──────────┘
```

### 3.3 External Dependencies Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SERVICES                         │
├─────────────────────┬─────────────────────┬────────────────────┤
│   Zerodha Kite      │   Yahoo Finance     │   Anthropic/OpenAI  │
│   Connect API       │   (yfinance)        │   (optional LLM)   │
│   - OAuth flow      │   - Price data      │   - Explanations   │
│   - Order placement │   - Fundamentals    │                    │
│   - WebSocket ticks │   - Universe        │                    │
│   - Rate: 3 rps     │                     │                    │
│     (orders)        │                     │                    │
│   - Rate: 10 rps    │                     │                    │
│     (data)          │                     │                    │
└─────────────────────┴─────────────────────┴────────────────────┘
```

---

## Phase 4: API Inventory, Interfaces, & UI Architecture

### 4.1 Complete API Endpoint Inventory

All endpoints are prefixed with `/api/v1`. Rate limit: 200 req/min default.

#### Health
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/health` | No | Basic health (DB + Redis status) |
| GET | `/api/v1/health/detailed` | No | Detailed component health |
| GET | `/` | No | Service info (60/min limit) |

#### Portfolio
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/portfolio/overview` | Yes | Portfolio summary (capital, PnL, metrics) |
| GET | `/api/v1/portfolio/positions` | Yes | List positions (query: strategy, is_open) |
| GET | `/api/v1/portfolio/allocation` | Yes | Current allocation breakdown |
| GET | `/api/v1/portfolio/performance` | Yes | Equity curve (query: period 1W/1M/3M/6M/1Y/ALL) |
| POST | `/api/v1/portfolio/contribution` | Yes | Submit monthly contribution |

#### Allocation
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/allocation/calculate` | Yes | Calculate allocation recommendation |
| POST | `/api/v1/allocation/execute` | Yes | Execute allocation (routes through execution boundary) |
| GET | `/api/v1/allocation/history` | Yes | Last 24 allocation records |
| GET | `/api/v1/allocation/explain/{id}` | Yes | Explain allocation decision |

#### Trades
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/trades` | Yes | Paginated trade list |
| GET | `/api/v1/trades/summary` | Yes | Trade summary stats |
| GET | `/api/v1/trades/{trade_id}` | Yes | Single trade detail |

#### Strategies
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/strategies` | Yes | List all strategies + health |
| GET | `/api/v1/strategies/{name}` | Yes | Strategy detail |
| PUT | `/api/v1/strategies/{name}/status` | Yes | Override strategy status |
| GET | `/api/v1/strategies/{name}/signals` | Yes | Recent signals |
| GET | `/api/v1/strategies/{name}/performance` | Yes | Performance time series |

#### Markets
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/markets/overview` | Yes | Market overview |
| GET | `/api/v1/markets/regime` | Yes | Current market regime |
| GET | `/api/v1/markets/sectors` | Yes | Sector performance |
| GET | `/api/v1/markets/opportunities` | Yes | Trading opportunities |

#### Settings
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/settings` | Yes | User settings |
| PUT | `/api/v1/settings` | Yes | Update settings |
| POST | `/api/v1/settings/kill-switch` | Yes | Toggle kill switch |
| GET | `/api/v1/settings/cost-model` | Yes | Zerodha cost model |

#### Research
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/research/backtest` | Yes | **501 Not Implemented** (was fabricating results) |
| GET | `/api/v1/research/backtests` | Yes | List past backtests |
| GET | `/api/v1/research/models` | Yes | List ML models (query: strategy, active_only) |
| POST | `/api/v1/research/walkforward` | Yes | Submit walk-forward task (not wired to worker) |

#### WebSocket
| Protocol | Path | Description |
|----------|------|-------------|
| WS | `/ws` | Real-time updates (portfolio, orders, risk, strategy, market data) |

### 4.2 Frontend UI Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Root Layout                               │
│  ┌──────────┐  ┌────────────────────────────────────────────┐  │
│  │ Sidebar  │  │  Header (IST clock, P&L, WS status, bell)  │  │
│  │ (15 nav) │  ├────────────────────────────────────────────┤  │
│  │          │  │                                            │  │
│  │ Dashboard│  │              <main> Content Area           │  │
│  │ Portfolio│  │                                            │  │
│  │ Intraday │  │  Routes:                                   │  │
│  │ Swing    │  │  / → redirect to /dashboard                │  │
│  │ L-Term   │  │  /dashboard    — Overview, allocation,     │  │
│  │ Markets  │  │                   risk, P&L, equity curve  │  │
│  │ Analytics│  │  /portfolio    — Position table, metrics    │  │
│  │ Trades   │  │  /intraday    — Day trading, square-off    │  │
│  │ Risk     │  │  /swing       — Swing positions, signals   │  │
│  │ Strategy │  │  /long-term   — Holdings, factor scores    │  │
│  │ Models   │  │  /markets     — Index, regime, sectors     │  │
│  │ Research │  │  /analytics   — Charts, drawdown, heatmap  │  │
│  │ Settings │  │  /trades      — Journal, filters, CSV      │  │
│  │ SysHealth│  │  /risk        — VaR, limits, correlation   │  │
│  │ Audit    │  │  /strategies  — Health, status override    │  │
│  │          │  │  /models      — ML monitoring, drift       │  │
│  │ [Kill    │  │  /research    — Backtest config + results  │  │
│  │  Switch] │  │  /settings    — Trading mode, risk params  │  │
│  └──────────┘  │  /system-health — Component status         │  │
│                │  /audit        — Audit trail, JSON diffs   │  │
│                └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Data Flow Architecture

```
User Browser
    │
    ├── HTTP ──▶ Next.js (port 3000)
    │             ├── /api/v1/* ──rewrite──▶ FastAPI (port 8000)
    │             └── Server Components (SSR)
    │
    └── WebSocket ──▶ ws://localhost:8000/ws
                       └── ChannelHub (in-process channel pub/sub)

FastAPI Backend
    │
    ├── SQLAlchemy Async ──▶ PostgreSQL
    │                         └── 12 tables, pool 10/20
    │
    ├── Redis Async ──▶ Redis 7
    │                   ├── Kill switch state
    │                   ├── Session cache
    │                   └── System health
    │
    ├── BrokerInterface ──▶ PaperBroker (default)
    │                   └── ZerodhaBroker (live, blocked)
    │
    └── Celery ──▶ Redis (DISABLED — no worker)
```

### 4.4 Frontend State Management

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Server State | React Query (TanStack v5) | All API data, polling (10s–120s) |
| Client State | React `useState` | Modals, filters, form inputs |
| Real-time State | WebSocket → React Query cache | Live portfolio, orders, risk alerts |
| Global State | Zustand (dependency installed, **NOT used**) | Available but unused |

---

## Phase 5: Low-Latency & Performance Optimization Analysis

### 5.1 Network & API Latency

| Component | Current | Bottleneck | Recommendation |
|-----------|---------|------------|----------------|
| Backend Uvicorn | 1 worker | Single-worker architecture prevents horizontal scaling within a container | Scale via replicas behind load balancer, not `--workers` (execution stack is in-process singleton) |
| Rate Limiter | 200 req/min global | No per-endpoint tuning | Add stricter limits on mutation endpoints (allocation/execute, kill-switch) |
| CORS | `allow_origins=["*"]` effectively | Overly permissive in dev | Tighten to explicit origins in production |
| WebSocket | In-memory `ConnectionManager` | Not distributed; no Redis pub/sub | For multi-worker: implement Redis pub/sub broadcast |
| Frontend→Backend | Next.js rewrite proxy | Extra hop in dev | In production, serve frontend separately (CDN) with API behind reverse proxy |

### 5.2 Database Performance

| Component | Current | Issue | Recommendation |
|-----------|---------|-------|----------------|
| Connection Pool | `pool_size=10, max_overflow=20` | Adequate for single worker | Tune for production: `pool_size=20, max_overflow=40` |
| `create_all_tables()` | Runs on every startup | Schema drift risk; slow cold start | Use Alembic exclusively; remove `create_all_tables()` |
| Session `autoflush=False` | Manual flush required | Can cause stale reads within a transaction | Set `autoflush=True` for correctness unless explicitly needed |
| `expire_on_commit=False` | ORM objects stay hydrated | Good for async; prevents lazy-load errors | Keep as-is |
| Indexing | Partial (user+symbol, user+is_open, user+status) | Missing composite indexes for common queries | Add indexes on `(strategy, is_open)`, `(created_at DESC)`, `(model_name, is_active)` |
| Historical Data | Forward-fill capped at 5 days | Adequate | Consider materialized views for dashboard queries |
| Alembic | Present but `create_all_tables()` overrides | Migration drift | Remove `create_all_tables()`, rely solely on Alembic |

### 5.3 Compute & ML Performance

| Component | Current | Issue | Recommendation |
|-----------|---------|-------|----------------|
| LightGBM imports | Cold start ~2-3s | numpy/pandas/sklearn/lightgbm import time | Lazy imports in strategy modules; load on first signal generation |
| Feature computation | Per-request pandas operations | No caching of computed features | Cache feature matrices in Redis with TTL; precompute daily |
| Research pipeline | Synchronous in API handler | `POST /research/backtest` is 501; walk-forward not wired | Wire to Celery workers (once implemented) |
| LedoitWolf covariance | Called per optimization | O(n²) per call | Cache covariance matrix with daily invalidation |
| Model inference | `predict()` on every signal generation | No model caching | Load models at startup; cache in `app.state` |
| `paper.py` (1593 lines) | Heavy import | Loaded at startup even in live mode | Conditional import based on `TRADING_MODE` |

### 5.4 Caching Opportunities

| Data | Current State | Cache Strategy | Expected Impact |
|------|--------------|----------------|-----------------|
| Market regime | Computed on demand | Cache in Redis with 5-min TTL | Eliminates repeated regime computation |
| Strategy signals | Generated per request | Cache with strategy-specific TTL (intraday: 1min, swing: 1hr, longterm: 1day) | Reduces redundant signal generation |
| Portfolio overview | Full recomputation | Cache with 30s TTL, invalidate on trade | Reduces dashboard load time |
| Sector performance | No caching | Daily cache | Eliminates repeated sector calculations |
| Cost model | Static configuration | No caching needed | Already fast |
| Historical prices | Redis with TTL map | Existing (5-day forward-fill cap) | Adequate |

### 5.5 Docker Multi-Stage Build Optimization

| Image | Current Stages | Optimization |
|-------|---------------|--------------|
| Backend | 2 stages (builder → runtime) | Good. Consider adding `--mount=type=cache,target=/root/.cache/pip` for faster rebuilds |
| Frontend | 4 stages (deps → builder → prod-deps → runner) | Optimal. `npm ci --omit=dev` produces ~105 packages vs ~609 with devDeps |
| Both | No BuildKit cache mounts | Add `--mount=type=cache,target=/root/.cache/pip` (backend) and `--mount=type=cache,target=/root/.npm` (frontend) |
| Frontend | `.next/cache` deleted in builder | Good — `next start` doesn't use it |
| Backend | `PIP_NO_CACHE_DIR=1` | Correct for CI; add cache mount for local dev |

### 5.6 Critical Performance Recommendations (Prioritized)

#### P0 — High Impact, Low Effort
1. **Remove `create_all_tables()` from startup** — Use Alembic exclusively. This eliminates schema drift risk and reduces cold-start time.
2. **Lazy-load ML imports** — `lightgbm`, `scikit-learn`, `pandas` should be imported on first use, not at module load. This could save 2-3s on startup.
3. **Add Redis pub/sub for WebSocket** — The current `ConnectionManager` is in-memory and cannot work with multiple workers.
4. **Cache portfolio overview** — The most-requested endpoint recomputes everything. 30s Redis cache would dramatically reduce load.

#### P1 — Medium Impact, Medium Effort
5. **Wire Celery worker** — The worker is listed in docker-compose but disabled. Implement `app/worker.py` and define actual Celery tasks for research, backtesting, and walk-forward.
6. **Add database indexes** — Missing composite indexes on `(strategy, is_open)`, `(created_at DESC)`, `(model_name, is_active)`.
7. **Implement connection pool tuning** — Production should use `pool_size=20, max_overflow=40` with `pool_timeout=30`.
8. **Frontend: Enable `output: "standalone"`** — Currently ships `.next + node_modules`. Standalone output eliminates the node_modules dependency, reducing image size by ~60%.
9. **Add BuildKit cache mounts** — Both Dockerfiles would benefit from `--mount=type=cache` for pip and npm caches during rebuilds.

#### P2 — Lower Impact, Higher Effort
10. **Implement Redis-based session store** — Current in-memory `InMemoryOrderStore` and `InMemoryAuditSink` lose state on restart.
11. **Add Prometheus metrics** — Replace ad-hoc health checks with structured metrics (request latency, error rates, order submission latency).
12. **Database query optimization** — Add materialized views for dashboard aggregates (daily PnL, strategy performance, equity curve).
13. **WebSocket compression** — Enable permessage-deflate for reduced bandwidth on market data streams.
14. **Frontend: Implement ISR (Incremental Static Regeneration)** — Dashboard pages with stable data could be statically generated and revalidated.

### 5.7 Scaling Bottlenecks

| Bottleneck | Severity | Mitigation |
|------------|----------|------------|
| Single Uvicorn worker | **HIGH** | Scale horizontally with replicas (not `--workers`) due to in-process execution stack |
| In-memory WebSocket manager | **HIGH** | Redis pub/sub for distributed broadcast |
| In-memory order store | **MEDIUM** | Migrate to Redis-backed store (already partially implemented in `lifecycle.py`) |
| Synchronous feature computation | **MEDIUM** | Precompute and cache; move to Celery Beat for daily pipeline |
| PostgreSQL single instance | **LOW** | Adequate for personal use; add read replicas if needed |
| No CDN for frontend | **LOW** | Add Cloudflare/Vercel CDN for static assets |

---

## Appendix A: Security Findings Summary

| Category | Finding | Status |
|----------|---------|--------|
| Secrets management | `.env` gitignored, credentials never baked into images | PASS |
| Container security | `cap_drop: ALL`, `no-new-privileges`, unprivileged user | PASS |
| Port exposure | All services loopback-only by default | PASS |
| Dependency scanning | TruffleHog, pip-audit, npm audit, Bandit | PASS (weekly + on-push) |
| Known exceptions | `PYSEC-2020-25` (autobahn, hard-pinned by kiteconnect) | DOCUMENTED |
| Trading safety | 23 fail-closed gates, CI asserts BLOCKED | PASS |
| Kill switch | Aggregates all sources; unreadable = ACTIVE | PASS |
| Execution layer | Not connected to application; nothing can place orders | PASS (safest state) |

## Appendix B: Test Coverage Summary

| Test Category | Files | Count | Purpose |
|---------------|-------|-------|---------|
| Allocator | 2 | ~30 | Capital allocation invariants |
| Backtester | 1 | ~15 | Event-driven backtest correctness |
| Costs | 1 | ~10 | Zerodha cost model accuracy |
| Eligibility | 1 | ~20 | Live trading gate evaluation |
| Execution | 3 | ~40 | Safety, integration, audit |
| Lookahead | 1 | ~10 | Feature leakage prevention |
| Models | 1 | ~15 | ML model evaluation |
| Order lifecycle | 1 | ~20 | 13-state order machine |
| Paper broker | 1 | ~25 | Paper trading simulation |
| Reconciliation | 1 | ~15 | Broker reconciliation + recovery |
| Research | 2 | ~20 | Pipeline + statistics |
| Risk | 2 | ~25 | Risk engine + numerics |
| Safety | 2 | ~30 | Invariants + execution safety |
| Strategy | 1 | ~15 | Strategy correctness |
| **Total** | **~20** | **~350+** | |

---

*This audit was conducted as a read-only analysis. No code was modified. All findings are based on the current state of the `Feat/updates` branch.*
