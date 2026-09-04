# AlgoDollar — Production Readiness Matrix

**Purpose:** a single table showing every requirement for live trading, its actual status, the evidence, and whether it blocks.

**Status values:** `PASS` (verified by a test) · `FAIL` (verified broken) · `BLOCKED` (cannot be satisfied with available data) · `N/A` (not wired in).

---

## Release audit — final state

Run against commit `HEAD` on the date of this document. Every row was verified by executing
it, not by inspection.

| Acceptance criterion | Status | Evidence |
|---|---|---|
| 1. PAPER end-to-end trading | **PASS** | Full chain driven through `app.main`'s real lifespan: startup → order SUBMITTED → 1 fill → position 10 → cash 1,000,000 → 974,952.25 → audit feed → clean shutdown |
| 2. Persistence | **PASS** | Orders, state transitions, fills, positions, cash, average cost, realised P&L, strategy attribution, reconciliation runs. `tests/test_e2e_paper_trade.py` |
| 3. Restart / recovery | **PASS** | Fixed this cycle — see the reconciliation defect below. `tests/test_failure_modes.py::TestRestart` |
| 4. Reconciliation | **PASS** | OK / MISMATCH / UNAVAILABLE distinguished; unreachable broker never reports OK |
| 5. Idempotency | **PASS** | UNIQUE `(user_id, client_order_id)` claimed before the broker call; retry-after-error and concurrent-duplicate both refused |
| 6. Risk gates | **PASS** | Daily loss, daily risk budget, position count, cash, holdings, sector, position size, liquidity, turnover |
| 7. Kill switch | **PASS** | Any source engaged = stop; an UNREADABLE switch counts as ENGAGED |
| 8. Frontend shows real state | **PASS** | API envelope/casing defect fixed; fabricated charts removed |
| 9. CI green | **PASS** | Backend CI, Frontend CI, Security Scanning, Safety Gates, Docker Build |
| 10. No critical security defects | **PASS** | pip-audit, Bandit, TruffleHog, npm audit all green in CI; no secret tracked |
| 11. Research status honest | **PASS** | `NOT VALIDATED`; stale report superseded |
| 12. Live trading gated | **PASS** | `blocked_insufficient_data`, 22 of 31 gates failing, `trading_mode=paper` |

### Defects this audit found and fixed

**Restart after any fill reported a false MISMATCH (CRITICAL).** Broker fills carry the
quantity under `qty`; reconciliation read `quantity`. Every broker fill therefore counted as
ZERO, so a restart compared `broker=0` against `local=10`, latched the kill switch, and left
the process permanently unable to trade. It failed *closed*, so nothing was at risk — but
restart-after-trading was impossible, which is one of the properties reconciliation exists to
provide.

**Redis clients leaked on every restart.** `build_production_stack()` opened two clients and
kept no handle, so shutdown could not release them. A container cycling repeatedly would
exhaust the server's client limit.

**Schema drift was silent.** `alembic` is a declared dependency but was never initialised;
tables come from `create_all`, which creates missing tables and never alters existing ones.
The columns added this cycle would never appear on an upgraded database, and the process
would fail at the first query touching them. `verify_schema()` now names every missing table
and column at startup.

### Known gaps, stated rather than closed

- **No migration tool.** `create_all` plus a drift check is the honest boundary of not having
  one. An upgraded database needs the reported changes applied by hand.
- **Daily loss and intraday capital limits are not measurable** — both need a persisted
  intraday P&L series. Reported as `measurable: false`, never as 0%.
- **Drawdown needs a persisted equity high-water mark**, which is not stored.
- **Exits of held names with no originating signal are skipped**, not submitted: creating the
  order would require fabricating a Signal into the audit trail.
- **Swing ranked-opportunities and long-term factor scores** are not yet exposed by the API.

---

## The structural fact that frames everything below

**The execution layer is not connected to the application.**

Verified by search: nothing outside `app/execution/` and `app/broker/` imports either package. No API route reaches the order manager. `reconcile()` is never called at startup. The eligibility gate is not consulted by any order path, because there is no order path.

This has two consequences that must be held simultaneously:

- **Nothing unsafe can happen today.** The system cannot place an order, live or paper, because nothing calls the code that would place one. This is the safest possible state.
- **None of the execution fixes are validated in integration.** They are verified by unit and scenario tests against the real classes, which is real evidence, but the assembled system has never run end to end.

Connecting this layer is therefore not a feature; it is the point at which every safety gate below starts to matter.

---

## Matrix

| Category | Requirement | Status | Evidence | Blocking? |
|---|---|---|---|---|
| **Data** | Real price history | **PASS** | 99 securities, 434,998 observations, 4,685 dates, 2006-01-02 → 2024-12-31 | No |
| **Data** | Point-in-time universe | **BLOCKED** | Absent. Today's membership applied to all history — a second look-ahead on top of survivorship | **YES** |
| **Data** | Delisted securities | **BLOCKED** | 0 present. SATYAMCOMP / DHFL / VIDEOIND return zero rows over periods when they were listed; RELIANCE control returns 720 | **YES** |
| **Data** | Corporate actions | **PARTIAL** | Vendor adjustment on 98/99; 12 unadjusted events detected via `adj_ratio == raw_ratio` and masked. Sub-40% events may remain | **YES** |
| **Data** | Point-in-time fundamentals | **BLOCKED** | None obtained. Long-term engine runs on `_MockFundamentalProvider`, guarded to raise outside paper mode | **YES** |
| **Data** | Intraday history | **BLOCKED** | ~60 days at 5-min. Cannot support a multiple-testing-corrected claim | **YES** |
| **Data** | Bid/ask spreads | **BLOCKED** | Absent. Slippage is modelled, never measured | **YES** |
| **Data** | Data-quality tests | **PASS** | `app/data/quality.py` runs pre-modelling, blocks on CRITICAL. Current: 0 CRITICAL / 6 WARNING | No |
| **Data** | Reproducibility manifest | **PASS** | `research/data_manifest.json`; content checksum detects a single value changed by 0.00001% | No |
| **Research** | No look-ahead in features | **PASS** | 21 features pass mechanical causality test; 2 negative controls confirm the checker is not vacuous | No |
| **Research** | Purged / embargoed validation | **PASS** | Every fold reports `n_purged == label_horizon` | No |
| **Research** | Untouched final holdout | **PASS** | 2022-2024 opened once; dev Sharpe +0.421 → holdout −0.067 | No |
| **Research** | Post-cost measurement | **PASS** | Turnover-proportional costs; verified MIS ₹53.64 / CNC ₹132.54 per ₹50k round trip | No |
| **Research** | DSR / PBO | **PASS** | Max DSR 0.461 across 10 candidates; PBO 0.143. Rejects a 1.89-Sharpe noise winner from 1000 trials | No |
| **Research** | Validated strategy | **FAIL** | None reached significance. **PRODUCTION MODEL = NONE** | **YES** |
| **Research** | Unbiased dataset for claims | **BLOCKED** | Survivorship-filtered; bias direction on excess return is indeterminate | **YES** |
| **Risk** | Capital limits | *see EXECUTION_SAFETY_AUDIT* | | **YES** |
| **Risk** | Position limits | *see EXECUTION_SAFETY_AUDIT* | | **YES** |
| **Risk** | Allocation caps cannot be normalized away | **PASS** | 2% cap resolves to exactly 2.0000%; sum invariant residual 0.000000 at ₹0 → ₹50L; PANIC → 100% cash | No |
| **Risk** | Risk numerics correct | **PASS** | CVaR matches Monte-Carlo ES to 0.048%; risk-parity dispersion 6.3e-11 | No |
| **Execution** | Non-idempotent calls never retried | **PASS** | `_call_kite(idempotent=False)` attempts once and raises `AmbiguousOrderStateError`; reads still retry 3× | No |
| **Execution** | Access token survives disconnect | **PASS** | `disconnect(invalidate_token=False)` default; previously killed the daily session on any restart | No |
| **Execution** | Order lifecycle state machine | *see EXECUTION_SAFETY_AUDIT* | | **YES** |
| **Execution** | Idempotent submission | *see EXECUTION_SAFETY_AUDIT* | | **YES** |
| **Execution** | Reconciliation fails closed | *see EXECUTION_SAFETY_AUDIT* | | **YES** |
| **Execution** | Restart recovery | *see EXECUTION_SAFETY_AUDIT* | | **YES** |
| **Execution** | Wired into the application | **N/A** | Nothing outside `app/execution` / `app/broker` imports them | **YES** |
| **Execution** | Live broker verified | **BLOCKED** | No credentials; never held a connection | **YES** |
| **Trading** | Paper broker accounting | *see EXECUTION_SAFETY_AUDIT* | | **YES** |
| **Trading** | Paper trading completed | **FAIL** | Zero days run | **YES** |
| **Security** | No tracked secrets | **PASS** | Verifier: no `.env`, no key material, no high-entropy literals | No |
| **Security** | Paper mode default | **PASS** | `TRADING_MODE=paper`; `is_live_trading_enabled` False | No |
| **Infrastructure** | Safety checks in CI | **PASS** | `.github/workflows/safety-gates.yml` asserts suites exist, runs them, runs session tests under `TZ=UTC`, and fails if eligibility ever reports LIVE_ELIGIBLE | No |
| **Infrastructure** | One-command verification | **PASS** | `scripts/verify_production_readiness` | No |
| **Governance** | Live eligibility gate | **PASS** (functioning) | 23 fail-closed gates; currently `BLOCKED_INSUFFICIENT_DATA`, 1/23 passing | No |
| **Governance** | Eligibility enforced at an order path | **N/A** | No order path exists to enforce it on | **YES** |

---

## Blockers, grouped

**Data (7)** — point-in-time universe, delisted securities, complete corporate actions, point-in-time fundamentals, intraday history, bid/ask spreads, and consequently any unbiased performance claim. All seven have a defined acquisition path in `DATA_ACQUISITION_PLAN.md`.

**Research (2)** — no validated strategy; no dataset capable of validating one.

**Execution (several)** — detailed in `EXECUTION_SAFETY_AUDIT.md`, plus the structural fact that the layer is not wired into the application and has never held a broker connection.

**Operational (1)** — zero days of paper trading, which cannot begin meaningfully until the paper broker's accounting is trustworthy.

---

## What this matrix is not

It is not a countdown. Clearing every execution row would leave `PRODUCTION MODEL = NONE` untouched, because software correctness and strategy profitability are different claims supported by different evidence. Fixing the execution layer makes future research trustworthy; it does not make an unvalidated strategy tradeable.

`LIVE_TRADING_ELIGIBILITY` remains **BLOCKED**.
