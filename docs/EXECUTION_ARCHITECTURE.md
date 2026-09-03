# AlgoDollar — Execution Architecture

**Status of this document:** describes code that exists and is wired. Where something is not yet connected, it says so.

---

## The problem this architecture solves

Every safety component in this system worked, and none of them was reachable.

`ExecutionSafety` had twelve gates. `OrderManager` had idempotency. `ReconciliationEngine` compared broker state to local state. The eligibility gate had 23 checks. All had passing tests. And a repository-wide search for imports of `app.execution` or `app.broker` from anywhere else returned **nothing**.

The execution layer was unreachable dead code. `reconcile()` was never called at startup. Eligibility was enforced nowhere. Meanwhile `POST /allocation/execute` returned `executed: true` after checking a kill switch and writing a log line — it placed no orders, consulted no risk engine, and reached no broker, while telling the caller the allocation had been executed.

Safety properties that hold only in unit tests are not safety properties. This document describes the single path that now exists so they hold in the application.

---

## Execution flow

```
                    POST /api/v1/allocation/execute
                                 │
                                 ▼
                  request.app.state.execution_service
                                 │
    ┌────────────────────────────▼────────────────────────────┐
    │            ExecutionService.submit_signal()             │
    │              app/execution/service.py                   │
    │                                                         │
    │   1. KILL SWITCH        any source says stop → stop     │
    │   2. TRADING GATE       startup reconciliation OK?      │
    │   3. MODE + AUTH        paper default; live needs both  │
    │   4. ELIGIBILITY        evaluated always, enforced live │
    │                                                         │
    └────────────────────────────┬────────────────────────────┘
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │            OrderManager.submit_order()                  │
    │           app/execution/order_manager.py                │
    │                                                         │
    │   0. LIVE ELIGIBILITY   again, keyed on broker TYPE     │
    │   1. RESERVE            client order id, set-if-absent  │
    │   2. REFERENCE PRICE    rejects stale / missing / NaN   │
    │   3. SAFETY GATES       ExecutionSafety, 12 checks      │
    │   4. PERSIST SUBMITTED  before the broker call          │
    │   5. place_order()      exactly once, never retried     │
    │   6. AMBIGUOUS → UNKNOWN → reconcile by client tag      │
    └────────────────────────────┬────────────────────────────┘
                                 ▼
                  BrokerInterface (app/broker/base.py)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
          PaperBroker                     ZerodhaBroker
      (default; real prices,           (live; requires BOTH
       simulated fills)                 mode=live AND authorization)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                    Fill / Partial / Reject / Unknown
                                 │
                                 ▼
              Portfolio accounting + OrderStore persistence
                                 │
                                 ▼
                 AuditJournal (every attempt, including blocks)
```

Every arrow is a real call. There is no second path to a broker.

---

## Module responsibilities

| Module | Owns |
|---|---|
| `execution/service.py` | The application boundary. The only place an order originates. |
| `execution/bootstrap.py` | Assembly + startup reconciliation. Returns an `ExecutionStack`. |
| `execution/order_manager.py` | Idempotency, reference pricing, risk invocation, the broker call. |
| `execution/safety.py` | The twelve pre-trade gates. |
| `execution/lifecycle.py` | 13-state order state machine, client order ids, `OrderStore`. |
| `execution/reconciliation.py` | Broker vs local comparison; four explicit statuses. |
| `execution/recovery.py` | Startup sequence and the trading gate. |
| `execution/audit.py` | The audit journal. |
| `governance/eligibility.py` | 31 fail-closed gates and `require_live_eligible()`. |
| `broker/base.py` | The broker contract. |
| `broker/paper.py` | Paper broker: integer-paise accounting, enforced invariants. |
| `broker/zerodha.py` | Live adapter. |

---

## Order lifecycle

Thirteen states, with transitions declared in `LEGAL_TRANSITIONS` and enforced by `assert_transition`.

```
INTENT_CREATED → RISK_CHECK_PENDING → RISK_APPROVED → SUBMITTED
                        ↓                    ↓            ↓
                  RISK_REJECTED*      RISK_REJECTED*  ACKNOWLEDGED ⇄ PARTIALLY_FILLED
                                                          ↓   ↘           ↓
                                              CANCEL_PENDING  UNKNOWN   FILLED*
                                                          ↓      ⇣ (reconciled only)
                                         CANCELLED* / REJECTED* / EXPIRED*
```
`*` = terminal.

**`UNKNOWN` is the important state.** It is reached when a broker outcome is genuinely ambiguous — a timeout, a lost response, an unrecognised status. While an order is UNKNOWN, cancels, fills, and new orders in the same symbol all raise `OrderBlockedError`. It can only be left with `reconciled=True`, meaning the broker's order book was actually queried.

An unrecognised or missing broker status maps to UNKNOWN, never to success. There is no state that means "probably fine".

---

## Boundaries

### Risk boundary

`ExecutionSafety` runs twelve gates inside `OrderManager.submit_order`. Every order type passes through the same function; there is no order type with its own path.

The generic exception handler **fails closed**: any gate that raises marks the order rejected and records the failure. Previously `MarketClosedError` and `StaleDataError` subclassed `RuntimeError` rather than `SafetyCheckError`, so they fell through to a handler that recorded a *warning* and left `passed=True` — orders validated at 03:00 on a Sunday and on data that had never ticked.

Risk is measured as `qty × |entry − stop|`, or the full notional when there is no stop. It was previously the transaction cost, which meant a ₹2.9 crore position registered ₹2,029 of "risk" and twenty-four of them fitted inside a ₹50,000 daily cap.

### Eligibility boundary

Enforced in **two** places, deliberately:

1. `ExecutionService._check_eligibility` — the application boundary.
2. `order_manager._require_eligible_for_live` — the last function before `place_order`.

The second is not redundant. It is keyed on the **broker's actual type**, not a configuration flag: anything not positively identified as a `PaperBroker` is treated as live and must pass `require_live_eligible()`. A mode flag can be wrong, unset, or forgotten; the object about to receive the order cannot be.

`require_live_eligible()` re-derives the verdict from individual gate results rather than trusting a report's own `state` field. An adversarial audit found that a hand-built or deserialized report could otherwise assert its own eligibility — one of 71 successful bypasses found and closed.

Paper orders skip this check. Requiring LIVE_ELIGIBLE for paper trading would block the activity that produces the evidence the gates demand, and a paper order cannot lose money.

### Broker boundary

`BrokerInterface` is the only contract. `PaperBroker` and `ZerodhaBroker` implement it.

`ExecutionService` verifies at **construction** that the broker matches the mode — a paper-mode service holding a live broker refuses to be built. Checking at construction rather than per order means a misconfiguration fails at startup instead of at the first trade.

`place_order` carries an explicit `trigger_price` for SL/SL-M. Previously the live adapter had no trigger parameter at all: SL sent the limit price as its own trigger and SL-M got none, so strategies that believed they had broker-side stops did not have them. The paper broker used `price` as the trigger, so a stop tested in paper would have behaved differently live — a mismatch that only surfaces when the stop is actually needed.

---

## Reconciliation lifecycle

Four explicit statuses, with no "degraded but fine" value:

```
RECONCILIATION_OK           broker and local agree
RECONCILIATION_MISMATCH     they disagree
RECONCILIATION_UNAVAILABLE  an input could not be read
RECONCILIATION_ERROR        unexpected failure
```

Precedence is **ERROR > UNAVAILABLE > MISMATCH > OK**. Unavailability outranks a confirmed mismatch because when an input is missing the discrepancy list is necessarily incomplete.

Only `RECONCILIATION_OK` permits trading. `__bool__` is overridden on the status, the result, and the startup state so that a careless `if result:` cannot read unknown as success.

This exists because the previous implementation returned `[]` from every local-state fetch and swallowed broker errors into `[]` as well. An unreachable broker produced `[] vs []`, which compared equal, reported OK, set no kill switch, and let trading start blind.

---

## Startup and restart behaviour

`app/main.py` lifespan → `build_execution_stack()`:

```
establish dependencies
  → select broker for the configured mode
  → connect to broker
  → reconcile broker state against local state
  → only on RECONCILIATION_OK does the trading gate open
```

Connectivity is established **before** reconciliation, explicitly, because reconciling against a broker that was never connected is what produced the empty-vs-empty fail-open.

If any step fails, the service is still constructed but its trading gate stays closed. That distinction is deliberate: a service that exists and refuses to trade produces an audited rejection for every attempt, which is far more diagnosable than a missing object producing an `AttributeError` somewhere upstream.

`StartupState` is `BLOCKED → RECOVERING → READY | BLOCKED`. The initial state is BLOCKED: a process that has not reconciled does not know what it owns. `_enter_ready` is the only transition into READY and re-asserts `RECONCILIATION_OK`.

**Crash recovery** is tested at each dangerous point — before submission, during submission, after submission but before acknowledgement, after acknowledgement, after partial fill, after complete fill, during cancellation, during reconciliation. After a restart the system recovers state, reconciles, and never blindly resends an ambiguous order.

---

## Failure behaviour

Everything fails closed, including our own bugs.

| Condition | Result |
|---|---|
| Kill-switch store unreadable | Treated as ACTIVE |
| No kill-switch source configured | Treated as ACTIVE |
| Recovery manager absent | Trading gate CLOSED |
| Eligibility evaluation raises | Live orders BLOCKED |
| Any safety gate raises | Order REJECTED |
| Broker outcome ambiguous | `UNKNOWN`; never retried |
| Broker status unrecognised | `UNKNOWN` |
| Unexpected exception at the boundary | Order BLOCKED, audited as `ERROR` |
| Audit sink fails | Logged and attached to the record; never vetoes an order |

The last row is the one deliberate exception. An audit sink that can veto trading is a new outage mode; an audit sink that fails silently is a compliance hole. The failure is recorded on the record itself and logged at ERROR.

---

## Kill switch

The application previously had **two unconnected kill switches**: the API wrote `UserSettings.kill_switch_active` to the database while `ExecutionSafety` read a `kill_switch` key in Redis. Pressing the button in the UI did not stop the execution layer.

`KillSwitch` now aggregates every source. **Any** source saying stop means stop, and a source that raises counts as ACTIVE — a switch whose state cannot be read must be assumed engaged, because the alternative is trading through an outage in the one control designed to stop trading.

It is checked first, before every other gate, on every order.

---

## Paper / live separation

| | Paper | Live |
|---|---|---|
| Default | **Yes** | No |
| Requires `trading_mode=live` | — | Yes |
| Requires explicit authorization | — | Yes |
| Requires `LIVE_ELIGIBLE` | No | Yes |
| Broker type verified at construction | Yes | Yes |
| Fallback to the other mode | **Never** | **Never** |

There is deliberately no path from a failed live submission to a paper one, or from a failed paper submission to a live one. Such a fallback would silently change which account is at risk at the worst possible moment.

---

## What is still not wired

Stated plainly, because a diagram that implies more than exists is worse than none:

- **No strategy produces production signals.** `/allocation/execute` routes correctly through the boundary and reports honestly that there is nothing to submit. No strategy is validated, so this is the correct behaviour, not a gap to paper over.
- **No background worker or scheduler exists.** There is no automated trading loop. Orders originate only from an API call.
- **`InMemoryOrderStore` is the default.** It is a real store, not a stub, but it is not durable. Production requires `RedisOrderStore` or a database implementation.
- **The live broker has never held a connection.** Everything in the Zerodha adapter is verified by inspection and simulation only.
- **`LIVE_TRADING_ELIGIBILITY` is `BLOCKED_INSUFFICIENT_DATA`** at 4 of 31 gates. That is the correct state and nothing in this document changes it.

### Open item: unstamped quotes

`PaperBroker` treats a quote with **no timestamp** as fresh-but-unverified
(`strict_quote_staleness=False`). That is a deliberate, documented choice made
because the current data adapter does not stamp its quotes — but it means a
market order can be priced against data whose age cannot be checked.

The order path does **not** override this, so it remains configurable rather
than hardcoded. **Any real deployment must construct the broker with
`strict_quote_staleness=True`**, after which an unstamped quote is stale and
the freshness gate refuses the order.

This is recorded as an open item rather than quietly accepted. It is not the
same as the PB-STALE deadlock, which is fixed: genuine staleness is refused
(a 90-minute-old quote is rejected, verified by test), and what changed is
only that "never quoted before" no longer means "permanently untradeable".
