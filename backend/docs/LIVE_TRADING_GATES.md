# LIVE_TRADING_GATES

The gate between research and real money. Source: `app/governance/eligibility.py`.
Tests: `tests/test_eligibility.py` (414 tests, all passing).

---

## Verdict today

```
STATE          : BLOCKED_INSUFFICIENT_DATA
LIVE PERMITTED : NO
GATES          : 3/31 passed
BLOCKED BY     : market_data_current: last bar of price history is STALE:
                 2025-01-01 is 610 days ago, limit 7 days.
                 NO EVIDENCE RECORDED: market data feed health.
```

Reproduce with:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
.venv/bin/python -c "from app.governance import assess_repo_live_trading_eligibility as a; print(a().checklist())"
```

Three of thirty-one gates pass. All three are about the *shape* of the evidence
record, not about the strategy: the record is freshly stamped, its values are
well-formed, and there is real daily OHLCV on disk. Nothing about profitability,
validation, execution or risk has ever been demonstrated by this repository.

---

## CRITICAL: enforcement is not wired in

**`LIVE_TRADING_ELIGIBILITY != LIVE_ELIGIBLE` does not currently prevent
anything.** The audit that produced this document found that no module under
`app/` outside the governance package imported this one. `OrderManager.submit_order`
consults `ExecutionSafety` and nothing else; it would have placed a live order
happily while this file reported BLOCKED.

`require_live_eligible()` now exists and is the assertion that closes that hole,
but **the call site is owned by another module and has not been added.** Until
someone does this:

```python
# in execution/order_manager.py
from app.governance import require_live_eligible, OrderIntent

async def submit_order(self, signal, position_size, broker, ...):
    require_live_eligible(action=f"submit {signal.symbol} x{position_size}")
    ...

async def cancel_order(self, order_id, broker, db_session=None):
    require_live_eligible(action=f"cancel {order_id}", intent=OrderIntent.REDUCE_RISK)
    ...

async def emergency_flatten_all(self, broker, db_session=None):
    require_live_eligible(action="emergency flatten", intent=OrderIntent.REDUCE_RISK)
    ...
```

…the verdict in this document is advisory.

The system cannot quietly become *eligible but unenforced*, because gate
`enforcement_wired_into_order_path` reads `order_manager.py` and fails while the
call is absent. Wiring the call is therefore a precondition for LIVE_ELIGIBLE,
not a follow-up to it. When the call lands, update the assertion in
`test_the_enforcement_gate_detects_that_the_order_path_is_not_wired` deliberately.

> Note for whoever wires it: `tests/test_execution_safety_audit.py::test_CRITICAL_no_order_path_in_the_app_checks_anything`
> asserts that nothing outside `app/execution/` and `app/broker/` references
> those packages. Adding the import above will make it fail — correctly, because
> its premise will no longer hold. That test needs updating in the same change.

---

## State machine and severity ordering

`EligibilityState` is an ordered enum. Lower severity is more severe; the first
member is the default, so any code reaching for "the first state" lands somewhere
safe. The report's state is the **most severe** blocking state among the failed
gates — you are told the worst thing first, and the list is worked top to bottom.

| Severity | State | Meaning | Permits live? |
|---|---|---|---|
| 0 | `BLOCKED_INSUFFICIENT_DATA` | **Default.** The inputs are missing, stale, or impossible. | No |
| 1 | `BLOCKED_VALIDATION_INCOMPLETE` | Data is adequate; the validation protocol has not been run. | No |
| 2 | `BLOCKED_INSUFFICIENT_STATISTICAL_EVIDENCE` | Validation ran; the result is not distinguishable from luck. | No |
| 3 | `BLOCKED_POOR_OOS_PERFORMANCE` | The result is real but does not clear the return bar. | No |
| 4 | `BLOCKED_EXCESSIVE_DRAWDOWN` | Returns clear the bar; the path to them is too painful. | No |
| 5 | `BLOCKED_EXECUTION_NOT_VALIDATED` | The strategy is sound; the plumbing is unproven. | No |
| 6 | `BLOCKED_RISK_LIMIT_BREACH` | Everything is proven; a limit is breached *right now*. | No |
| 7 | `PAPER_ONLY` | Research bars cleared, operational bars not. **Still blocked.** | No |
| 8 | `LIVE_ELIGIBLE` | Every gate passes. | **Yes** |

`PAPER_ONLY` is the easiest state to misread as permission. It is not permission.
`state.is_blocked` is true for every state except `LIVE_ELIGIBLE`.

Severity 6 (`BLOCKED_RISK_LIMIT_BREACH`) sits below `PAPER_ONLY` deliberately: a
live limit breach is a *more* urgent thing to fix than a short paper-trading
record, even though both block.

---

## The 31 gates

"Current Status" is the actual result of running the code against this repository
today, not a target. "Test" names the test function that proves the gate blocks.
Every gate additionally appears in the parametrized sweeps
`test_every_gate_blocks_on_realistic_bad_evidence[<gate>]`,
`test_every_gate_can_block_on_its_own[<gate>]` and
`test_dropping_a_gate_cannot_buy_eligibility[<gate>]`; the Test column names the
most specific test where one exists.

### DATA — 3/8 passing

| Gate | Category | Requirement | Current Status | Evidence that would satisfy it | Fail-Closed | Test |
|---|---|---|---|---|---|---|
| `evidence_freshness` | data | Evidence record states when it was gathered, ≤24 h old, timezone-aware, not future-dated | **PASS** — record is 0.00 h old | The `Evidence.evidence_gathered_at` stamp set by whatever measured the facts, at measurement time | Absent stamp, naive stamp, stale stamp and future stamp all fail | `test_evidence_older_than_the_limit_blocks`, `test_future_dated_evidence_blocks` |
| `evidence_well_formed` | data | Every supplied value finite, correctly typed, inside its physically possible range; record not empty | **PASS** — all supplied values in-domain | Nothing extra; this is a schema check over whatever else is supplied | NaN, ±inf, out-of-domain, mistyped, empty record all fail | `test_physically_impossible_values_are_rejected` |
| `real_price_history` | data | Real daily OHLCV, ≥50 symbols, ≥10 years | **PASS** — 99 symbols, 19.0 years | Parquet cache under `data_cache/` with `SYMBOL__start__end.parquet` files from a real vendor | Synthetic data flag, short span, few symbols all fail | `test_every_gate_blocks_on_realistic_bad_evidence[real_price_history]` |
| `market_data_current` | data | Last bar ≤7 days old **and** feed currently healthy | **FAIL** — last bar 2025-01-01, 610 days ago; feed health unknown | A refreshed price cache plus a live health probe of the data feed | Stale, future-dated and unknown-health all fail | `test_stale_or_impossible_price_history_blocks` |
| `point_in_time_index_constituents` | data | Point-in-time index membership (AUDIT §9.4) | **FAIL** — not verified | A survivorship-free constituent history (e.g. NSE index-change archive) loaded and dated | `False` and `None` both fail; only literal `True` passes | `test_every_gate_blocks_on_realistic_bad_evidence[point_in_time_index_constituents]` |
| `point_in_time_fundamentals` | data | Point-in-time fundamentals with publication dates (AUDIT §9.3) | **FAIL** — provider is `_MockFundamentalProvider` | A vendor feed carrying report *publication* dates, not just period ends | Mock provider is detected and reported as `False` | `test_every_gate_blocks_on_realistic_bad_evidence[point_in_time_fundamentals]` |
| `intraday_history_span` | data | ≥504 days of intraday history | **FAIL** — 0 days | ~2 years of intraday bars from a source that retains them (Yahoo retains ~60 days) | `None` and short spans fail; NaN/inf fail | `test_every_gate_blocks_on_realistic_bad_evidence[intraday_history_span]` |
| `data_quality_audit` | data | Real OHLC (not synthesized from close) + corporate actions adjusted (AUDIT §9.5) | **FAIL** — no evidence recorded | A signed-off run of `app/data/quality.py` over the cache | All three sub-flags must be `True`; any `None` fails | `test_every_gate_blocks_on_realistic_bad_evidence[data_quality_audit]` |

### STATISTICAL — 0/5 passing

| Gate | Category | Requirement | Current Status | Evidence that would satisfy it | Fail-Closed | Test |
|---|---|---|---|---|---|---|
| `purged_walk_forward` | statistical | Purged, embargoed walk-forward completed (AUDIT §1.2 F5) | **FAIL** — no evidence recorded | A persisted `app/backtesting/walkforward.py` run record showing purge and embargo windows | Both sub-flags required; `None` fails | `test_every_gate_blocks_on_realistic_bad_evidence[purged_walk_forward]` |
| `deflated_sharpe_ratio` | statistical | DSR **> 0.95** (strict) | **FAIL** — no evidence recorded | A DSR computed by `app/research/statistics.py` over the full trial set | Boundary 0.95 fails; NaN/inf fail | `test_boundary_values_are_rejected` |
| `probability_of_backtest_overfitting` | statistical | PBO **< 0.50** (strict) | **FAIL** — no evidence recorded | A CSCV/PBO estimate over the trial set | Boundary 0.50 fails; p outside [0,1] fails as impossible | `test_boundary_values_are_rejected` |
| `beats_passive_benchmark` | statistical | Positive net annual excess return vs a **named** benchmark (AUDIT §12.6) | **FAIL** — no benchmark named | A named index (e.g. "NIFTY 50 total return") plus a net-of-cost excess return | Blank/whitespace/zero-width names rejected; 0.0 excess fails (strict) | `test_a_blank_benchmark_name_is_not_a_benchmark` |
| `honest_trial_count` | statistical | Trials recorded ≥1 **and** exactly the number the DSR was deflated for (AUDIT §12.7) | **FAIL** — no evidence recorded | A trial ledger counting every discarded variant, and a DSR deflated for that same count | Mismatched counts fail with "not honest"; `None` fails | `test_dishonest_trial_count_is_rejected` |

### PERFORMANCE — 0/4 passing

| Gate | Category | Requirement | Current Status | Evidence that would satisfy it | Fail-Closed | Test |
|---|---|---|---|---|---|---|
| `oos_sharpe_floor` | performance | OOS Sharpe ≥0.50, net of costs | **FAIL** — no evidence recorded | An out-of-sample equity curve from a walk-forward run, costed with `app/backtesting/costs.py` | NaN/inf/absurd values rejected; 0.49999 fails | `test_boundary_values_are_rejected` |
| `max_drawdown_limit` | performance | OOS max drawdown ≤15% | **FAIL** — no evidence recorded | Max drawdown from that same equity curve | **Negative drawdowns rejected as impossible** (see audit note below) | `test_physically_impossible_values_are_rejected` |
| `cost_and_slippage_stress` | performance | Survives ≥1.5× slippage **and** stressed-cost Sharpe ≥0.50 (AUDIT §12.9) | **FAIL** — no evidence recorded | A cost-stress sweep re-running the backtest at multiples of modelled slippage | Both sub-checks required; inf multiples rejected | `test_every_gate_blocks_on_realistic_bad_evidence[cost_and_slippage_stress]` |
| `return_concentration` | performance | Single-name ≤25%, sector ≤40%, single-year ≤50%, top-5 trades ≤50% of PnL | **FAIL** — no evidence recorded | A PnL attribution over the OOS period, by name, sector, year and trade | Negative shares rejected as sign errors; any `None` fails | `test_every_gate_blocks_on_realistic_bad_evidence[return_concentration]` |

### EXECUTION — 0/9 passing

| Gate | Category | Requirement | Current Status | Evidence that would satisfy it | Fail-Closed | Test |
|---|---|---|---|---|---|---|
| `broker_auth` | execution | Broker auth + token refresh verified against a live session (AUDIT §9.10) | **FAIL** — no evidence recorded | A signed-off live-session test including an expiry/refresh cycle | Only literal `True`; `1`/`"yes"` rejected | `test_every_gate_blocks_on_realistic_bad_evidence[broker_auth]` |
| `broker_connectivity` | execution | Broker answered ≤15 min ago, session valid, API **not** degraded | **FAIL** — no heartbeat, no session, degradation unknown | A live heartbeat timestamp from a real broker round-trip plus a status probe | **Broker unavailability blocks.** Unknown degradation blocks (must be positively `False`) | `test_broker_unavailability_blocks` |
| `order_lifecycle` | execution | Place / modify / cancel / partial fill / reject all verified | **FAIL** — no evidence recorded | A recorded live-session run exercising all five transitions | `None` and `False` fail | `test_every_gate_blocks_on_realistic_bad_evidence[order_lifecycle]` |
| `reconciliation` | execution | Startup reconciliation against real broker positions verified | **FAIL** — no evidence recorded | A recorded startup reconciliation against a funded account | `None` and `False` fail | `test_every_gate_blocks_on_realistic_bad_evidence[reconciliation]` |
| `reconciliation_current` | execution | Last reconciliation ≤24 h ago, **succeeded**, and left **zero** open breaks | **FAIL** — no timestamp, no outcome, no break count | A live `ReconciliationEngine` result with its timestamp and break count | **Failed reconciliation blocks. Stale reconciliation blocks. Any break count >0 blocks.** | `test_failed_reconciliation_blocks` |
| `duplicate_order_protection` | execution | Duplicate-order protection verified against a real broker session | **FAIL** — no evidence recorded | A live replay test proving the idempotency key suppresses a repeat | `None` and `False` fail | `test_every_gate_blocks_on_realistic_bad_evidence[duplicate_order_protection]` |
| `kill_switch` | execution | Kill switch verified to halt and flatten against a real broker session | **FAIL** — no evidence recorded | A live drill: trip the switch, confirm halt and flat book | `None` and `False` fail | `test_every_gate_blocks_on_realistic_bad_evidence[kill_switch]` |
| `timezone_and_squareoff` | execution | IST handling + intraday square-off verified on a UTC host (AUDIT §1.1 C4) | **FAIL** — no evidence recorded | A square-off drill run on a UTC-clock host across a session boundary | `None` and `False` fail | `test_every_gate_blocks_on_realistic_bad_evidence[timezone_and_squareoff]` |
| `enforcement_wired_into_order_path` | execution | The live order path calls `require_live_eligible()` | **FAIL** — `order_manager.py` does not reference it | The call, present in `order_manager.py`. The gate reads that file's source and checks | Unreadable file ⇒ `None` ⇒ fail. **A verdict nothing consults is decoration** | `test_the_enforcement_gate_detects_that_the_order_path_is_not_wired`, `test_unreadable_order_path_is_treated_as_unwired` |

### RISK — 0/1 passing

| Gate | Category | Requirement | Current Status | Evidence that would satisfy it | Fail-Closed | Test |
|---|---|---|---|---|---|---|
| `risk_limits_enforced` | operational → `BLOCKED_RISK_LIMIT_BREACH` | Limits loaded, **zero** active breaches, live drawdown ≤15%, kill switch **not** engaged | **FAIL** — nothing recorded | A live read from `app/risk/limits.py`: loaded-limits flag, current breach count, current drawdown, kill-switch state | **Any breach blocks. Unknown breach count blocks. An engaged kill switch is never countermanded** | `test_risk_limit_violations_block`, `test_an_engaged_kill_switch_is_never_countermanded` |

### OPERATIONAL — 0/4 passing

| Gate | Category | Requirement | Current Status | Evidence that would satisfy it | Fail-Closed | Test |
|---|---|---|---|---|---|---|
| `paper_trading_duration` | operational | ≥90 days paper trading on live data (AUDIT §12.15) | **FAIL** — no evidence recorded | A paper-trading log covering ≥90 sessions on live prices | 89 fails; `None` fails; absurd values rejected | `test_boundary_values_are_rejected` |
| `paper_matches_backtest` | operational | Paper Sharpe ≥0.5× backtest expected Sharpe (AUDIT §12.16) | **FAIL** — no paper Sharpe | Both Sharpes measured over the same period | **NaN in either input fails** (this was a real hole — see below); non-positive backtest expectation fails | `test_paper_vs_backtest_never_passes_on_a_non_number` |
| `human_enabled_live_mode` | operational | `TRADING_MODE == "live"` **and** a named human approver on a recorded date | **FAIL** — `TRADING_MODE` is `'paper'` | `settings.trading_mode = "live"` plus an approver name and date in the evidence record | Only the exact string `"live"`; `"LIVE"`, `"live "` rejected. Blank/whitespace/zero-width/1-char approvers rejected | `test_live_mode_without_a_human_approver_is_not_eligible`, `test_an_invisible_or_trivial_approver_is_not_a_human` |
| `approval_is_current` | operational | That approval is <90 days old | **FAIL** — no approval date | A dated sign-off refreshed at least quarterly | Stale and future-dated approvals both fail | `test_approval_that_has_gone_stale_blocks`, `test_future_dated_approval_blocks` |

---

## What would have to become true to reach LIVE_ELIGIBLE

All 31 gates, simultaneously, from a single evidence record gathered within the
last 24 hours. In dependency order:

1. **Data (5 remaining).** Refresh the price cache to within 7 days and add a
   feed-health probe. Acquire point-in-time index constituents and point-in-time
   fundamentals — the current Yahoo Finance source supplies neither, so this
   means a different vendor. Acquire ≥504 days of intraday history — Yahoo
   retains ~60, so again a different vendor. Run and sign off the data-quality
   audit.
2. **Statistical (5).** Build a strategy. Run purged, embargoed walk-forward
   validation. Compute DSR >0.95 and PBO <0.50 *over an honest trial ledger* that
   counts every discarded variant. Name a passive benchmark and beat it net of
   costs.
3. **Performance (4).** Demonstrate OOS Sharpe ≥0.50 and max drawdown ≤15% net of
   costs, surviving 1.5× slippage stress, with PnL not concentrated in one name,
   sector, year or handful of trades.
4. **Execution (9).** Verify auth, full order lifecycle, reconciliation,
   duplicate protection, kill switch and IST square-off against a *real* funded
   broker session. Maintain a live broker heartbeat and a clean, recent
   reconciliation. **Wire `require_live_eligible()` into the order path.**
5. **Risk (1).** Have limits loaded, zero active breaches, drawdown within
   bounds and the kill switch disengaged at the moment of asking.
6. **Operational (4).** ≥90 days of paper trading whose Sharpe is at least half
   the backtest's. Set `TRADING_MODE=live`. Record a named human approver and a
   date less than 90 days old.

None of steps 2–6 has any persisted artifact in this repository today. The honest
summary is that this is a data-and-plumbing project that has not yet produced a
strategy, and the gate says so.

---

## Enforcement points

| Point | Status | What it does |
|---|---|---|
| `require_live_eligible(action=..., intent=INCREASE_RISK)` | **Exists, not called** | Raises `LiveTradingBlocked` unless a freshly computed report clears every canonical gate |
| `OrderManager.submit_order` | **NOT WIRED** | Must call the above before touching the broker. This is the gap |
| `OrderManager.cancel_order`, `emergency_flatten_all` | **NOT WIRED** | Should call with `intent=REDUCE_RISK` — permitted while blocked, logged at WARNING |
| `EligibilityReport.permits_live_trading` | Advisory | A property. Safe to read, but a caller who forgets to branch trades anyway — prefer the raising function |

`require_live_eligible` is deliberately an **exception**, not a boolean. A caller
who ignores a returned `False` places the order; a caller who ignores an
exception places nothing.

It performs six independent checks, in order:

1. `REDUCE_RISK` intent short-circuits (permitted, logged at WARNING). An
   unrecognized intent blocks.
2. No report supplied ⇒ compute one now. An assessment that *raises* blocks.
3. The report must be **exactly** `EligibilityReport` — not a subclass, because a
   subclass can override `state` and `permits_live_trading`.
4. Provenance must be `COMPUTED`. Deserialized and hand-built reports are refused.
5. The report must be <15 minutes old.
6. The verdict is **re-derived from the results** against the canonical registry.
   `report.state` is not trusted.

### There is no override

No force flag, no environment variable, no config switch. This was a choice: the
system has never placed a live order, so no operational need exists that an
override would serve, and an override is the thing most likely to be reached for
at the exact moment judgement is worst. `test_no_bypass_mechanism_exists` asserts
the signature stays clean and that the module never reads `os.environ`.

The one carve-out is `OrderIntent.REDUCE_RISK`, scoped to flatten / square-off /
cancel, because a blocked system must still be able to get flat. It is logged at
WARNING on every use and it grants no eligibility — the report it returns still
reads `permits_live_trading = False`.

---

## How the module fails closed

1. **Silence is failure.** Every `Evidence` field defaults to `None`, and `None`
   is a FAILURE reading "NO EVIDENCE RECORDED", never a skip. A default
   `Evidence()` fails all 31 gates.
2. **Errors are failure.** A gate that raises is recorded FAILED with the
   exception text. A predicate returning the wrong shape is a broken gate, hence
   a failed gate.
3. **Nonsense is failure.** NaN, ±inf and out-of-domain values are rejected
   before any comparison happens, in *both* directions.
4. **The state is derived and provenance-gated.** `state` is a read-only property
   computed against a canonical registry captured at import. Only a report
   produced in-process by `assess_live_trading_eligibility` is `COMPUTED`;
   everything else is `UNTRUSTED` and authorizes nothing.
5. **The verdict is enforced, not merely reported** — see above, once wired.

Absent gates are injected as failures. Substituted gates may report failures (so
tests can inject them) but their *passes* are discarded. Rebinding the public
`ALL_GATES` does not change the verdict.

---

## Audit log — holes that were actually open

These were found by adversarial testing of the previous 23-gate version. Each is
now a named regression test.

| # | Hole | Severity | Fix |
|---|---|---|---|
| 1 | **Nothing enforced the verdict.** No module outside `app/governance/` imported it | CRITICAL | Added `require_live_eligible()` + gate `enforcement_wired_into_order_path` that detects its own absence |
| 2 | **Forged JSON.** `from_dict` trusted the payload's `passed` booleans; 23 `"passed": true` entries produced a checklist reading `LIVE PERMITTED : YES` | CRITICAL | `ReportProvenance`; deserialized reports are `UNTRUSTED` and `permits_live_trading` is False regardless of state |
| 3 | **Gate-name spoofing.** One same-named gate with a trivial predicate — the exact shape of the existing test helper — yielded LIVE_ELIGIBLE from a *completely empty* `Evidence` | CRITICAL | Gate identity checked against the canonical registry; substituted passes discarded |
| 4 | **NaN passed `paper_matches_backtest`.** `if paper < floor: return FAIL` falls through when `paper` is NaN, because NaN fails every comparison | HIGH | All numeric comparisons routed through `_as_finite()` |
| 5 | **No domain bounds.** `-inf` satisfied every `<=` cap and `+inf` every `>=` floor. A negative drawdown, a PBO of −3 and a Sharpe of 10⁶ all passed | HIGH | `_NUMERIC_DOMAINS` table + `evidence_well_formed` gate |
| 6 | **No freshness anywhere.** Evidence had no timestamp. An approval dated 1999, price history ending 1920, and a report evaluated in 2001 all passed | HIGH | `evidence_gathered_at`; gates `evidence_freshness`, `market_data_current`, `approval_is_current`; broker/reconciliation heartbeats; report-age check in the enforcer |
| 7 | **Severity downgrade.** A forged payload relabelling every failure `paper_only` misrepresented a broken data layer as nearly-ready | MEDIUM | `__post_init__` re-binds category/blocking-state/requirement from the canonical registry |
| 8 | **Monkeypatching `ALL_GATES = ()`** made an empty report LIVE_ELIGIBLE | MEDIUM | Private `_CANONICAL_GATES` captured at import; the public name is never consulted for a verdict |
| 9 | **Subclass override** of `state` / `permits_live_trading` | MEDIUM | Enforcer requires the exact type and re-derives from results |
| 10 | **Invisible approver.** `str.strip()` does not remove U+200B, so an approver of one zero-width space satisfied "a named human took responsibility" | LOW | `_is_blank()` covering the zero-width family; ≥2 visible characters required |
| 11 | **Epoch-fallback dates.** `price_history_start = 1900-01-01` (Excel epoch, a classic failed-date-parse artifact) satisfied the 10-year span check with data that cannot exist | LOW | `MIN_PLAUSIBLE_DATA_DATE`. *Found by the randomized property test, not by hand* |
| 12 | **`blocking_reason` could raise `StopIteration`** from inside the enforcer's block path, turning a clean refusal into an unhandled crash | LOW | Defaulted `next()`; the block path is now exception-safe |

Defences that were tested and **held** unchanged: empty reports, dropped gates,
result sets filtered to passes only, raising gates, garbage predicate returns,
frozen `Evidence`, truthy-but-not-`True` flags (`1`, `"yes"`, `[1]`), `"LIVE"` /
`"live "` variants, assignment to `state`, and flipping the serialized `state`
string.

---

## Uncalibrated thresholds — be honest about these

Most numeric bars in this module are **hand-chosen priors, not estimates**. They
are deliberately conservative, and they are not calibrated from data because
there is no validated strategy from which to calibrate them. This is the same
caveat AUDIT §9.8 applies to the regime multipliers.

**Pure priors — defensible, arbitrary:**

| Constant | Value | Note |
|---|---|---|
| `MIN_PRICE_HISTORY_YEARS` | 10 | "More than one regime" — but regimes are not 10 years |
| `MIN_PRICE_HISTORY_SYMBOLS` | 50 | Round number |
| `MIN_INTRADAY_HISTORY_DAYS` | 504 | ≈2 trading years; round number |
| `MIN_OOS_SHARPE` | 0.50 | Round number |
| `MAX_DRAWDOWN` | 0.15 | Mirrors `settings.max_portfolio_drawdown_pct`, itself a prior |
| `MIN_SLIPPAGE_STRESS_MULTIPLE` | 1.5 | Audit-cited, still hand-chosen |
| `MAX_SINGLE_NAME_PNL_SHARE` | 0.25 | Round number |
| `MAX_SECTOR_PNL_SHARE` | 0.40 | Round number |
| `MAX_SINGLE_YEAR_PNL_SHARE` | 0.50 | Round number |
| `MAX_TOP5_TRADES_PNL_SHARE` | 0.50 | Round number |
| `MIN_PAPER_TRADING_DAYS` | 90 | Audit says "3–6 months"; 90 is the low end |
| `MIN_PAPER_TO_BACKTEST_SHARPE_RATIO` | 0.5 | Pure judgement |
| `MAX_EVIDENCE_AGE_HOURS` | 24 | Judgement |
| `MAX_PRICE_DATA_STALENESS_DAYS` | 7 | Judgement; tolerates a long weekend plus holidays |
| `MAX_BROKER_HEARTBEAT_AGE_MINUTES` | 15 | Judgement |
| `MAX_RECONCILIATION_AGE_HOURS` | 24 | Judgement |
| `MAX_APPROVAL_AGE_DAYS` | 90 | Judgement |
| `MAX_REPORT_AGE_MINUTES` | 15 | Judgement |
| `MIN_PLAUSIBLE_DATA_DATE` | 1990-01-01 | Judgement, but with a specific job: catching 1900/1970 epoch fallbacks. NSE opened 1994 |
| `_NUMERIC_DOMAINS` bounds | various | Chosen as *impossible* bounds, not *acceptable* ones — they catch bugs, not bad strategies |

**Principled — has a source:**

| Constant | Value | Source |
|---|---|---|
| `MIN_DEFLATED_SHARPE` | >0.95 | Bailey & López de Prado: DSR is a probability; >0.95 is 95% confidence |
| `MAX_PBO` | <0.50 | Above 0.5, in-sample selection is worse than random — structurally meaningful |
| `MAX_ACTIVE_RISK_LIMIT_BREACHES` | 0 | Not a prior. A breach is a breach |
| `MAX_OPEN_RECONCILIATION_BREAKS` | 0 | Not a prior. An unexplained position is an unexplained position |

Two consequences worth stating plainly: passing these gates is **not** proof that
a strategy makes money, and the bars could be wrong in either direction. They are
a floor for "worth risking real money on", set by judgement.

---

## Test coverage

`tests/test_eligibility.py` — 414 tests, all passing.

- Every one of the 31 gates has a test proving it **blocks** on realistic bad
  evidence, a test proving it can block **alone**, and a test proving that
  **dropping** it does not buy eligibility.
- Every one of the 51 `Evidence` fields has a test proving it is load-bearing
  (`test_every_piece_of_evidence_is_load_bearing`) — erase any one and
  eligibility is lost.
- Every hole in the audit log above has a named regression test.
- `test_randomized_evidence_reaches_live_eligible_only_when_everything_passes` —
  5 seeds × 400 scenarios = **2000 scenarios, 0 violations**; 164 reached
  LIVE_ELIGIBLE and every one was genuinely fully compliant. It asserts five
  properties per scenario, including agreement with an independent oracle that
  does not consult the gate logic, and agreement between `permits_live_trading`
  and whether `require_live_eligible` raises.
- `test_randomized_single_field_corruption_always_blocks` — exhaustive sweep of
  every field × every bad value: **293 scenarios, 0 violations**.

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
.venv/bin/python -m pytest tests/test_eligibility.py -q
```
