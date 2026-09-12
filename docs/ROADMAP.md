# AlgoDollar — Roadmap & Remaining Phases

**Purpose:** the phases still left before live trading, what each requires, and
whether it is blocked on data, on software, or on credentials.

**Status values:** `DONE` · `BLOCKED` (cannot proceed without an external input) ·
`OPEN` (nothing external required).

---

## Completed

| Phase | Scope | Status |
|---|---|---|
| P1 Foundation | Paper end-to-end trading, persistence, restart/recovery, reconciliation, idempotency, kill switch, risk gates, API + frontend | **DONE** |
| P2 Execution safety | 13 critical execution defects fixed; order lifecycle, reserve-before-submit, fail-closed safety, eligibility gate wired into every order path | **DONE** |
| P3 Research integrity | Real-data validation, lookahead causality tests, DSR/PBO, untouched holdout, fabricated output removed | **DONE** |
| P4 Intraday paper auto-trader | Mock tick feed, minute-bar aggregation, auto-trader loop, square-off, broker-side SL-M stops | **DONE** |
| P5 Offline replay + intraday research rig | Replay harness through the production stack, config-grid surface, real-week experiment (NO_TRADE verdict), reproducible data fetch scripts | **DONE** |

---

## Remaining phases

### R1 — Intraday data acquisition  ·  `BLOCKED` (credentials/subscription)
- **Need:** 3–5 years of 1-minute (or 5-minute) NSE bars for the tradeable
  universe, plus bid/ask spread history. Minimum viable ≈ 60 sessions; a
  defensible claim needs >1 year.
- **Sources:** Zerodha Kite historical API (₹2,000/month subscription + the
  credentials the user does not yet have), GDFL, NSE data products.
- **Current fallback:** `scripts/fetch_intraday_bars.py` (yfinance, ~7 days,
  one regime). Used for pipeline validation only — documented as such.
- **Mock stand-in:** `ZERODHA_MOCK_MODE=true` serves deterministic mock quotes
  to non-trading routes; it is not research data.

### R2 — Intraday research program  ·  `BLOCKED` (on R1)
- Run the real-week experiment (`quant/experiments/intraday_replay_research.py`)
  across enough sessions for DSR/PBO. Current result: **NO_TRADE** on one quiet
  week (max net edge +11 bps against an 18 bps cost hurdle) — an honest
  measurement, but too few sessions to be a claim.

### R3 — Point-in-time universe + delisted prices  ·  `BLOCKED` (data)
- NSE index-change circulars → membership as of date *t*; bhavcopy archives →
  delisted names. Unblocks every historical performance claim for the swing and
  long-term horizons (survivorship bias is currently unquantified).
- Partly achievable from public sources with labour.

### R4 — Authoritative corporate actions  ·  `OPEN`
- Ingest NSE/BSE corporate-action feeds to convert the anomaly detector into a
  cross-check. Public feeds.

### R5 — Point-in-time fundamentals  ·  `BLOCKED` (paid data)
- CMIE Prowess / Capitaline with `publication_date`. Defer; the long-term
  engine remains guarded to paper mode until then.

### O1 — Paper trading campaign  ·  `DONE` (offline harness)
- `backend/app/engine/campaign.py` runs synthetic/csv paper campaigns through
  the production stack: `drive_session` (connect → recovery → snapshot →
  funds → live minute loop → EOD fills), a crash-safe JSONL ledger, and a
  persistent `PaperBroker` book resumed across restarts.
- Deliverable: daily P&L series, restart survival (resume N-days → +M-days is
  bit-identical to a contiguous run), audited refusals, and fail-closed checks
  (corrupt/forged state, cancelled ledger lines) — the operational evidence the
  eligibility gates demand. Evidence is operational-only: synthetic/csv bars are
  explicitly not a performance claim (no DSR/PBO per missing-data rules).
- CLI: `.venv/bin/python -m app.engine.campaign --synthetic N` (writes
  `data/campaign/paper/`). 7 tests in `tests/test_paper_campaign.py`; full light
  suite green.
- Next: O2 — persist intraday P&L + high-water mark so the daily-loss/drawdown
  gates become measurable on a real running campaign.

### O2 — Operational hardening  ·  `DONE`
- **DONE (slice): persisted intraday P&L + high-water mark.**
  `app/engine/equity.py` marks the paper book to market every minute during a
  campaign session (EquityTracker → high-water mark → drawdown / daily-loss
  magnitudes), evaluates the `RiskLimits` daily-loss & drawdown gates against
  each snapshot (`worst_risk`), and stores everything in the day records and
  summary (`written to ledger, never acts — recording is not enforcement`).
  Summary now exposes `paper_trading_days`, `paper_sharpe` (annualised, vs
  marked equity), `paper_max_drawdown_pct`, `paper_daily_loss_max_rupees` and
  `paper_drawdown_limit_breached` — the daily-loss/drawdown gates are now
  measurable on a running paper campaign (12 tests added).
- **DONE (slice): durable order store + kill switch (survive restart).**
  `app/execution/file_stores.py` ships `FileKillSwitchStore` +
  `FileOrderStore` (atomic tmp+`os.replace` writes, fail closed on unreadable
  state, `durable=True`). The paper campaign now runs every day through these
  stores — keyed **per book** (`<book>.kill_switch.json` / `<book>.orders.json`,
  so separate campaigns sharing a directory never bleed into each other) — so
  an engaged halt and the reserved order-id claims survive between days, across
  restarts, and across re-runs. `drive_session` gains an opt-in enforcement
  link (`risk_limits` + `on_breach`): a HARD daily-loss (`Severity.BREACH`) or
  drawdown (`Severity.CRITICAL`) breach engages the durable switch mid-session
  (next order refused), the campaign halts with a recorded
  `risk.enforced_halted` + reason, and a later run on the same book refuses to
  start until the switch is released. Recording-only is still the default
  (`risk_limits=None`); opt in via config mapping or CLI
  `--limit-daily-loss RUPEES` / `--limit-drawdown PCT`. Summary adds
  `paper_risk_halts`, `kill_switch_engaged`, `kill_switch_reason`. Non-campaign
  stacks (single-session replay, `build_production_stack` Redis-less fallback)
  still default to the in-memory stores (15 tests added, light suite 1238).
- **Alembic-only migrations DONE:** `create_all` is gone from the boot path.
  The schema now comes solely from `backend/migrations/` (alembic.ini,
  migrations/env.py via the async engine, one reviewed initial revision
  `a31cd025307e`). `app/main.py` startup checks `migration_status` and logs
  `migrations_pending` + the `alembic ... upgrade head` repair instead of
  silently creating schema. Operators apply
  `alembic -c backend/alembic.ini upgrade head` after review (unchanged from
  deployment.md); the backend image now ships alembic.ini + migrations so
  `make migrate` works. `app/database/session.py` exposes `apply_migrations`
  (explicit, raises on failure) and a read-only `migration_status`. Drift is
  guarded three ways: startup logs via `verify_schema`, CI-able `alembic check`,
  and a migration test suite that proves a fresh DB taken to head matches
  `create_all` cell-for-cell (7 tests added, light suite now **1245**).
- **Celery worker + static-IP / daily-token automation DONE:**
  `backend/app/worker.py` is the missing entrypoint the compose `worker` service
  was always pointing at (CI_SECURITY_AUDIT #6). `Celery("algodollar")` on
  `settings.redis_url`; IST beat schedule runs `daily_pnl_summary` (15:35 IST
  weekdays — previous IST trading day's realised P&L + costs, one idempotent
  `Notification` per day) and `refresh_kite_token` (08:45 IST weekdays —
  skips unless `zerodha_mock_mode==False` + credentials + `KITE_STATIC_IP_ENABLED`;
  fail-closed `{"status":"failed"}` when the configured token can't be
  validated). The worker never places orders. Compose now starts `worker` +
  `worker-beat` (pidbox-ping healthcheck, same env allowlist / security posture);
  Makefile `worker` / `worker-beat` / `worker-docker` / `celery-ping`. 17 tests
  (light suite now **1276**).
- **WebSocket pub/sub DONE:** `app/realtime/` replaces the old one-topic
  `ConnectionManager` with a `ChannelHub` over the known-channel allowlist
  `{ticks, bars, orders, portfolio, risk, strategy, audit, system}` — unknown
  subscriptions are rejected (fail-closed), publishing to a known-but-empty
  channel is a no-op, dead sockets dropped. `app/main.py` publishes lifecycle +
  execution-stack frames on `app.state.realtime_hub` and bridges the sync tick
  feed through a `TickPublisher`; `/ws` delegates to the new endpoint contract
  (ping/pong, subscribe/subscribed, unsubscribe, list). In-process only — correct
  for the single-worker runtime. 14 tests.
- Next (post-O2, non-blocking): drive the dashboard `useWebSocket` onto the new
  channel contract (`orders`, `ticks`) now that broker frames have a home.

### O3 — Live broker verification  ·  `BLOCKED` (credentials)
- The Zerodha adapter has never held a connection. Needs real credentials and a
  sandbox/small-capital pass before anything live is attempted.

### G1 — Eligibility gate → human go-live  ·  `BLOCKED` (on most of the above)
- 31 fail-closed gates currently `BLOCKED_INSUFFICIENT_DATA`. Live requires the
  gate to read `LIVE_ELIGIBLE` **and** a reviewed human decision. The pipeline
  already refuses to skip this regardless of configuration.

---

## Suggested next step

1. **O2 (operational hardening) is DONE** — all five slices shipped: intraday
   P&L / HWM (recorded breaches), durable order store + kill switch (engaged
   halts + reserved claims survive restarts), Alembic-only migrations (schema
   has a single reviewed source of truth; `create_all` dropped from boot),
   Celery worker + static-IP/daily-token automation (`app/worker.py`, worker +
   beat compose services), and WebSocket pub/sub (`ChannelHub` over known
   channels; `/ws` contract upgraded). Next non-blocking step: point the
   dashboard `useWebSocket` at `orders`/`ticks`.
2. **R3** is the highest-value data acquisition: public, free, and it unlocks
   the swing/long-term claims.
3. Mock credentials (`ZERODHA_MOCK_MODE`) are in place so any phase that needs
   the broker read surface can run locally today.