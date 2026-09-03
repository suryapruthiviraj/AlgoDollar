"""
End-to-end integration tests for the wired execution path.

WHAT MAKES THESE TESTS DIFFERENT FROM THE SUITE THEY REPLACE
------------------------------------------------------------
Every object below the API boundary is the REAL production object:

    app.execution.service.ExecutionService      (the boundary under test)
    app.execution.order_manager.OrderManager    (idempotency + the broker call)
    app.execution.safety.ExecutionSafety        (the safety gates)
    app.execution.lifecycle.InMemoryOrderStore  (the real durable-store impl)
    app.execution.reconciliation.ReconciliationEngine
    app.execution.recovery.RecoveryManager
    app.execution.audit.AuditJournal
    app.broker.paper.PaperBroker                (integer-paise accounting)
    app.governance.eligibility.*                (the fail-closed gate registry)

There are exactly TWO test doubles, both at genuine external boundaries:

    FakeFeed        the market-data source the paper broker quotes against.
                    A network connection cannot be made from a test.
    FakeLiveBroker  a stand-in live venue, used ONLY to prove that no order
                    ever reaches a live venue.  It records what it is sent and
                    is never allowed to be the broker of a paper-mode service.

`InMemoryLocalState` is not a double of production code: it is an
implementation of the published `LocalStateStore` Protocol, and
`test_local_state_double_implements_the_real_protocol` proves it structurally
satisfies the same Protocol the SQLAlchemy store does.

THE ASSERTION RULE
------------------
A test that inspects only the returned `ExecutionResult` would pass even if the
service placed an order at the broker and then reported failure.  Every test
here therefore asserts against `BrokerFacts` — a snapshot taken straight out of
the paper broker's own public API (order count, order statuses, cash, holdings,
trades) — plus the local order store and the audit journal.  The broker's order
count is the assertion that actually matters.

TIMEZONE
--------
Every test pins the paper broker's clock to an explicit aware instant, so the
suite behaves identically under TZ=UTC and TZ=Asia/Kolkata.  The market session
is evaluated in IST by `PaperBroker.is_market_open`; a naive datetime raises
rather than being guessed at.  The `test_tz_*` tests pin that directly.

SOURCE UNDER TEST
-----------------
The execution modules were being edited while this suite was written.  It was
last run against git 36a621e with these working-tree files:

    app/execution/service.py       sha256 8caecfcc…
    app/execution/order_manager.py sha256 ea6858c5…
    app/broker/paper.py            sha256 eb941963…

The DEFECT PINS section at the bottom asserts behaviour those modules document.
Each pin names a real defect this suite found, with a file:line and a repro.
All but one were fixed while the suite was being written; those pins are now
regression guards. The one still open is PB-STALE - see
`test_BUG_the_wired_path_cannot_place_a_first_order_in_a_fresh_symbol`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd
import pytest

from app.broker.base import BrokerInterface, Product, TransactionType
from app.broker.paper import (
    IST,
    OrderStatus,
    PaperBroker,
    PaperBrokerStateError,
    RejectReason,
    to_paise,
)
from app.execution.audit import AuditJournal, ExecutionOutcome, InMemoryAuditSink
from app.execution.bootstrap import (
    InMemoryKillSwitchStore,
    _run_startup_recovery,
    store_kill_switch_probe,
)
from app.execution.lifecycle import InMemoryOrderStore, OrderState
from app.execution.order_manager import OrderManager
from app.execution.reconciliation import (
    LocalStateStore,
    LocalStateUnavailable,
    ReconciliationEngine,
    ReconciliationStatus,
)
from app.execution.recovery import RecoveryManager, StartupState
from app.execution.safety import ExecutionSafety
from app.execution.service import (
    ExecutionBlocked,
    ExecutionService,
    KillSwitch,
    TradingGate,
    TradingMode,
)
from app.strategies.base import Signal, SignalDirection

# --------------------------------------------------------------------------- #
#  Fixed instants.  2025-06-10 is a Tuesday and not an NSE holiday.            #
# --------------------------------------------------------------------------- #

OPEN_IST = datetime(2025, 6, 10, 11, 0, tzinfo=IST)        # 05:30 UTC
CLOSED_IST = datetime(2025, 6, 10, 20, 0, tzinfo=IST)      # 14:30 UTC
WEEKEND_IST = datetime(2025, 6, 8, 11, 0, tzinfo=IST)      # Sunday
HOLIDAY_IST = datetime(2025, 8, 15, 11, 0, tzinfo=IST)     # Independence Day

SYMBOL = "RELIANCE"
PRICE = 100.0

#: `ExecutionService._to_order_intent` maps strategy_name == "intraday" to
#: Product.MIS and everything else to Product.CNC, so the strategy name is
#: load-bearing and is stated explicitly in every test.
INTRADAY = "intraday"
POS_KEY_MIS = f"NSE:{SYMBOL}:MIS"
POS_KEY_CNC = f"NSE:{SYMBOL}:CNC"

#: A kill-switch activation policy that never fires.
#:
#: `ReconciliationEngine.__init__` does `activate_kill_switch_on or
#: DEFAULT_KILL_SWITCH_ON`, so an EMPTY frozenset is falsy and silently
#: restores the default.  Naming a status that can never reach the activation
#: branch (OK returns from `enforce` before it) is the only way to disarm the
#: latch, and it lets a test attribute a block to the trading gate rather than
#: to the halt the previous failure persisted.
NEVER_LATCH = frozenset({ReconciliationStatus.RECONCILIATION_OK})

#: Risk context wide enough that the risk gates pass unless a test narrows one.
#: Every value is passed explicitly so no test depends on a default.
WIDE_RISK = dict(
    available_cash=10_000_000.0,
    total_portfolio=100_000_000.0,
    max_daily_risk=10_000_000.0,
    max_daily_loss=10_000_000.0,
    daily_risk_used=0.0,
    realised_pnl_today=0.0,
    current_positions=[],
    open_orders=[],
    max_positions=20,
)


# =========================================================================== #
#  Test doubles — external boundaries only                                    #
# =========================================================================== #

class FakeFeed(BrokerInterface):
    """
    The market-data source the paper broker quotes against.

    Decides nothing and books nothing: it answers `get_quote` and that is all.
    Every execution decision in these tests is made by real production code.
    """

    def __init__(
        self,
        price: float = PRICE,
        volume: int = 1_000_000,
        *,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.price = price
        self.volume = volume
        self.timestamp = timestamp
        self.fail = False
        self.quote_calls = 0

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def get_profile(self) -> dict: return {"user_name": "feed"}
    async def get_holdings(self) -> list[dict]: return []
    async def get_positions(self) -> list[dict]: return []
    async def get_orders(self) -> list[dict]: return []
    async def get_trades(self) -> list[dict]: return []
    async def get_funds(self) -> dict: return {}

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        self.quote_calls += 1
        if self.fail:
            raise ConnectionError("market data feed is down")
        out: dict[str, dict] = {}
        for key in symbols:
            _, sym = key.split(":", 1)
            q: dict[str, Any] = {
                "last_price": self.price,
                "volume": self.volume,
                "ohlc": {"open": self.price, "high": self.price,
                         "low": self.price, "close": self.price},
            }
            if self.timestamp is not None:
                q["timestamp"] = self.timestamp
            out[key] = q
        return out

    async def get_historical_data(self, *a, **k) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    async def place_order(self, *a, **k) -> str:
        raise AssertionError("the market-data feed must never be sent an order")

    async def cancel_order(self, order_id: str) -> bool:
        raise AssertionError("the market-data feed must never be sent a cancel")

    async def modify_order(self, order_id, qty=None, price=None) -> bool:
        raise AssertionError("the market-data feed must never be sent a modify")

    async def get_order_status(self, order_id: str) -> dict: return {}

    @property
    def is_connected(self) -> bool: return True

    @property
    def trading_mode(self) -> str: return "data-only"

    def instrument_token(self, symbol: str, exchange: str) -> int: return 12345


class FakeLiveBroker(FakeFeed):
    """
    A stand-in for a real venue.

    Its only job is to record anything sent to it, so that "no live order was
    produced" is a fact about a real object rather than an inference from a
    return value.  It is deliberately generous — market always open, ticks
    always fresh, orders always accepted — so that a live order that got past
    the gates WOULD land here and be counted.
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.placed: list[dict] = []

    async def place_order(self, symbol, exchange, txn_type, qty, price,
                          order_type, product, tag="", trigger_price=None) -> str:
        self.placed.append({
            "symbol": symbol, "exchange": exchange, "txn_type": txn_type.value,
            "qty": qty, "price": price, "order_type": order_type.value,
            "product": product.value, "tag": tag, "trigger_price": trigger_price,
        })
        return f"LIVE-{len(self.placed)}"

    async def get_orders(self) -> list[dict]:
        return [{"order_id": f"LIVE-{i + 1}", "tag": o["tag"], "status": "COMPLETE",
                 "symbol": o["symbol"], "quantity": o["qty"],
                 "filled_quantity": o["qty"], "price": PRICE}
                for i, o in enumerate(self.placed)]

    def is_stale_tick(self, symbol: str, max_age_seconds: float = 30.0) -> bool:
        return False

    def is_market_open(self, moment: Optional[datetime] = None) -> bool:
        return True

    @property
    def trading_mode(self) -> str: return "live"


class InMemoryLocalState:
    """
    An implementation of the production `LocalStateStore` Protocol.

    It stands in for the database, and — like the real store — raises
    `LocalStateUnavailable` rather than returning `[]` when it cannot read,
    because `[] == []` is exactly the fail-open the Protocol exists to prevent.
    """

    def __init__(
        self,
        *,
        positions: Optional[list[dict]] = None,
        orders: Optional[list[dict]] = None,
        trades: Optional[list[dict]] = None,
        cash: Optional[float] = 1_000_000.0,
    ) -> None:
        self.positions = [dict(p) for p in (positions or [])]
        self.orders = [dict(o) for o in (orders or [])]
        self.trades = [dict(t) for t in (trades or [])]
        self.cash = cash
        self.fail: set[str] = set()

    def _guard(self, what: str) -> None:
        if what in self.fail:
            raise LocalStateUnavailable(f"local {what} unreadable (simulated)")

    async def get_positions(self) -> list[dict]:
        self._guard("positions")
        return [dict(p) for p in self.positions]

    async def get_orders(self) -> list[dict]:
        self._guard("orders")
        return [dict(o) for o in self.orders]

    async def get_trades(self) -> list[dict]:
        self._guard("trades")
        return [dict(t) for t in self.trades]

    async def get_cash(self) -> dict:
        self._guard("cash")
        if self.cash is None:
            raise LocalStateUnavailable("no local cash record: cash is UNKNOWN, not 0")
        return {"cash": self.cash}


class FakeRedis:
    """In-memory stand-in for the redis client the paper broker persists to."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key): return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value
        return True


def test_local_state_double_implements_the_real_protocol():
    """If this fails, every reconciliation test below is testing a fiction."""
    assert isinstance(InMemoryLocalState(), LocalStateStore)


# =========================================================================== #
#  Harness                                                                    #
# =========================================================================== #

@dataclass
class BrokerFacts:
    """Everything the broker itself says about its own book."""
    order_count: int
    order_statuses: tuple[str, ...]
    reject_reasons: tuple[Optional[str], ...]
    filled_qty: int
    trade_count: int
    total_cash: float
    available_cash: float
    holdings: dict[str, int]
    holdings_basis_paise: int

    @classmethod
    async def of(cls, broker: PaperBroker) -> "BrokerFacts":
        orders = await broker.get_orders()
        trades = await broker.get_trades()
        funds = await broker.get_funds()
        positions = await broker.get_positions()
        return cls(
            order_count=len(orders),
            order_statuses=tuple(o["status"] for o in orders),
            reject_reasons=tuple(o["reject_reason"] for o in orders),
            filled_qty=sum(int(o["filled_qty"]) for o in orders),
            trade_count=len(trades),
            total_cash=float(funds["total_cash"]),
            available_cash=float(funds["margin_available"]),
            holdings={f'{p["exchange"]}:{p["symbol"]}:{p["product"]}': int(p["quantity"])
                      for p in positions},
            holdings_basis_paise=sum(int(p["cost_basis_paise"]) for p in positions),
        )


@dataclass
class Stack:
    """The whole real execution stack, assembled the way bootstrap assembles it."""
    feed: FakeFeed
    broker: PaperBroker
    kill_store: InMemoryKillSwitchStore
    safety: ExecutionSafety
    order_manager: OrderManager
    order_store: InMemoryOrderStore
    engine: ReconciliationEngine
    recovery: RecoveryManager
    audit: AuditJournal
    sink: InMemoryAuditSink
    service: ExecutionService
    local_state: InMemoryLocalState
    initial_cash: float

    async def facts(self) -> BrokerFacts:
        return await BrokerFacts.of(self.broker)

    @property
    def last_audit(self):
        assert self.sink.records, "no audit record was written for the attempt"
        return self.sink.records[-1]

    async def local_records(self):
        return await self.order_store.list_all()

    async def cids(self) -> frozenset[str]:
        """Client order ids already reserved, so a later assertion can ignore them."""
        return frozenset(r.client_order_id for r in await self.local_records())


async def build_stack(
    *,
    initial_cash: float = 1_000_000.0,
    volume: int = 1_000_000,
    price: float = PRICE,
    quote_timestamp: Optional[datetime] = None,
    clock: Optional[Callable[[], datetime]] = None,
    local_state: Optional[InMemoryLocalState] = None,
    eligibility_provider: Optional[Callable[[], Any]] = None,
    run_recovery: bool = True,
    prime: tuple[str, ...] = (SYMBOL,),
    activate_kill_switch_on: Optional[frozenset] = None,
    redis_client: Any = None,
    connect: bool = True,
) -> Stack:
    """
    Assemble the same objects `app.execution.bootstrap.build_execution_stack`
    assembles, but with the market-data feed and the local-state store injected
    so the test can steer the two external boundaries.

    `local_state` defaults to a store whose cash MATCHES the paper account's
    opening balance.  That matters: reconciliation compares broker cash against
    local cash, and a mismatch there activates the kill switch, which would
    then mask whatever the test was actually trying to observe.
    """
    feed = FakeFeed(price=price, volume=volume, timestamp=quote_timestamp)
    broker = PaperBroker(
        data_broker=feed,
        initial_cash=initial_cash,
        clock=clock or (lambda: OPEN_IST),
        redis_client=redis_client,
    )
    kill_store = InMemoryKillSwitchStore()
    safety = ExecutionSafety(kill_store)
    order_store = InMemoryOrderStore()
    order_manager = OrderManager(safety, store=order_store)
    local = local_state if local_state is not None else InMemoryLocalState(cash=initial_cash)
    engine_kw: dict[str, Any] = {}
    if activate_kill_switch_on is not None:
        engine_kw["activate_kill_switch_on"] = activate_kill_switch_on
    engine = ReconciliationEngine(kill_store, local_state=local, **engine_kw)
    recovery = RecoveryManager(engine, local_state=local)
    sink = InMemoryAuditSink()
    audit = AuditJournal(sink)
    service = ExecutionService(
        broker=broker,
        order_manager=order_manager,
        trading_mode=TradingMode.PAPER,
        kill_switch=KillSwitch(store_kill_switch_probe(kill_store)),
        trading_gate=TradingGate(recovery),
        audit=audit,
        eligibility_provider=eligibility_provider,
        live_authorized=False,
    )

    if connect:
        await broker.connect()
    if run_recovery:
        await recovery.recover(broker)
    for sym in prime:
        await prime_quote_cache(broker, sym)

    return Stack(feed=feed, broker=broker, kill_store=kill_store, safety=safety,
                 order_manager=order_manager, order_store=order_store, engine=engine,
                 recovery=recovery, audit=audit, sink=sink, service=service,
                 local_state=local, initial_cash=initial_cash)


async def build_live_stack(
    *,
    eligibility_provider: Optional[Callable[[], Any]] = None,
    live_broker: Optional[FakeLiveBroker] = None,
) -> tuple[FakeLiveBroker, ExecutionService, InMemoryAuditSink, RecoveryManager]:
    """A LIVE-mode service pointed at a venue that records everything sent to it."""
    live = live_broker or FakeLiveBroker()
    kill_store = InMemoryKillSwitchStore()
    local = InMemoryLocalState(cash=0.0)
    recovery = RecoveryManager(ReconciliationEngine(kill_store, local_state=local),
                               local_state=local)
    sink = InMemoryAuditSink()
    service = ExecutionService(
        broker=live,
        order_manager=OrderManager(ExecutionSafety(kill_store), store=InMemoryOrderStore()),
        trading_mode=TradingMode.LIVE,
        kill_switch=KillSwitch(store_kill_switch_probe(kill_store)),
        trading_gate=TradingGate(recovery),
        audit=AuditJournal(sink),
        eligibility_provider=eligibility_provider,
        live_authorized=True,
    )
    await recovery.recover(live)
    return live, service, sink, recovery


async def prime_quote_cache(broker: PaperBroker, symbol: str, exchange: str = "NSE") -> None:
    """
    Make `symbol` known to the paper broker's tick-staleness cache.

    WHY A TEST HAS TO DO THIS — production bug PB-STALE
    ---------------------------------------------------
    `PaperBroker.is_stale_tick` reports True for any symbol absent from
    `self._last_quote_age`.  That dict is written in exactly one place,
    `PaperBroker._snapshot`, which is reachable only from `place_order`,
    `poll_open_orders` and `_mark`.  `ExecutionSafety.check_data_freshness`
    probes `is_stale_tick` for EVERY order, before any order can be placed.
    So the first order in any symbol is refused as stale, no order is placed,
    the cache is never written, and the next order is refused for the same
    reason.  `PaperBroker.get_quote` delegates to the feed without touching the
    cache, so there is no public way out of the circle.

    Calling the real `_snapshot` is the only way to break it.  It is the exact
    call `place_order` makes, it books nothing, and it creates no order — so it
    cannot flatter any assertion about the broker's order count.

    `test_16b_a_symbol_that_has_never_ticked_produces_no_order` pins the
    deadlock itself.
    """
    await broker._snapshot(symbol, exchange)


def signal(
    symbol: str = SYMBOL,
    *,
    direction: SignalDirection = SignalDirection.LONG,
    strategy: str = INTRADAY,
    stop_loss_pct: float = 0.02,
    target_pct: float = 0.05,
    edge_score: float = 0.01,
    holding_period_days: int = 1,
) -> Signal:
    """
    One alpha signal, in the shape `ExecutionService.submit_signal` accepts.

    `strategy` is load-bearing: "intraday" becomes Product.MIS and anything
    else becomes Product.CNC inside `ExecutionService._to_order_intent`.
    """
    now = datetime.now(timezone.utc)
    return Signal(
        symbol=symbol, direction=direction, strategy_name=strategy,
        timestamp=now, signal_date=now, edge_score=edge_score,
        expected_return=0.02, expected_return_std=0.005,
        stop_loss_pct=stop_loss_pct, target_pct=target_pct,
        holding_period_days=holding_period_days,
    )


@contextlib.contextmanager
def broker_outage(broker: PaperBroker):
    """Make every account-state query on the broker raise, then restore it."""
    saved = {name: getattr(broker, name)
             for name in ("get_positions", "get_orders", "get_trades", "get_funds")}

    async def down(*a, **k):
        raise ConnectionError("kite 5xx: broker unreachable")

    for name in saved:
        setattr(broker, name, down)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(broker, name, fn)


def ledger_cash_paise(trades: list[dict], initial_cash: float) -> int:
    """
    Re-derive cash from the trade ledger, independently of the broker's own
    running balance.  This deliberately does not call
    `PaperBroker._ledger_cash_paise`, so a bug in that method cannot make these
    tests agree with themselves.
    """
    cash = to_paise(initial_cash)
    for t in trades:
        if t["txn_type"] == TransactionType.BUY.value:
            cash -= int(t["notional_paise"])
        else:
            cash += int(t["notional_paise"])
        cash -= int(t["costs_paise"])
    return cash


async def assert_value_is_conserved(broker: PaperBroker) -> None:
    """
    cash + open cost basis == opening cash + realised P&L, to the paise.

    Every rupee is either still cash, still invested at its cost basis, or has
    been explicitly realised as a gain or loss.  Nothing else may happen to it.
    """
    perf = await broker.get_paper_performance()
    basis = sum(int(p["cost_basis_paise"]) for p in await broker.get_positions())
    lhs = to_paise(perf["current_cash"]) + basis
    rhs = to_paise(perf["initial_cash"]) + to_paise(perf["realised_pnl"])
    assert abs(lhs - rhs) <= 1, (
        f"value was created or destroyed: cash+basis={lhs}p vs "
        f"initial+realised={rhs}p (delta {lhs - rhs}p)"
    )


async def assert_nothing_reached_the_broker(
    stack: Stack,
    before: BrokerFacts,
    result,
    *,
    prior_cids: frozenset[str] = frozenset(),
) -> None:
    """
    The full "no order was placed" assertion set.

    Checks the broker first and hardest, because a service that placed an order
    and then reported failure would satisfy every check on the return value.

    `prior_cids` names order records that already existed before this attempt,
    for tests that block an order on a book that is deliberately not empty.
    """
    after = await stack.facts()

    # --- the broker: the assertion that actually matters -----------------
    assert after.order_count == before.order_count, (
        f"an order reached the broker: {before.order_count} -> {after.order_count} "
        f"(statuses {after.order_statuses})"
    )
    assert after.trade_count == before.trade_count, "a trade was booked"
    assert after.filled_qty == before.filled_qty, "a fill was booked"

    # --- broker state ----------------------------------------------------
    assert after.order_statuses == before.order_statuses
    assert after.reject_reasons == before.reject_reasons

    # --- cash and holdings ------------------------------------------------
    assert after.total_cash == before.total_cash, "cash moved with no order"
    assert after.available_cash == before.available_cash, "buying power was reserved"
    assert after.holdings == before.holdings, "holdings changed with no order"
    assert after.holdings_basis_paise == before.holdings_basis_paise
    await assert_value_is_conserved(stack.broker)

    # --- the returned result ---------------------------------------------
    assert result.blocked, f"service reported submitted, outcome={result.outcome}"
    assert not result.submitted
    assert result.broker_order_id is None

    # --- local state -------------------------------------------------------
    for rec in await stack.local_records():
        if rec.client_order_id in prior_cids:
            continue
        assert rec.state not in (
            OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED
        ), f"local record {rec.client_order_id} claims the broker took the order"
        assert rec.broker_order_id is None, "a local record carries a broker order id"

    # --- the audit record --------------------------------------------------
    rec = stack.last_audit
    assert rec.outcome == result.outcome.value
    assert rec.reached_broker is False, (
        f"audit says the attempt reached a broker (outcome={rec.outcome})"
    )
    assert rec.rejection_reason, "a block was audited with no reason"
    assert rec.broker_order_id is None
    assert rec.fill_quantity is None
    assert rec.eligibility_permits_live is not True, (
        "a blocked attempt recorded eligibility as permitting live trading"
    )


def rejection_reasons(record) -> list[str]:
    """Every gate failure durably recorded on an OrderRecord."""
    return [r for h in record.history if "rejected" in h for r in h["rejected"]]


# =========================================================================== #
#  1. Valid signal -> paper order -> fill -> correct portfolio accounting      #
# =========================================================================== #

async def test_01_valid_signal_produces_a_filled_paper_order_and_correct_accounting():
    stack = await build_stack(initial_cash=1_000_000.0)
    before = await stack.facts()

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    # --- the returned result ---------------------------------------------
    assert result.outcome is ExecutionOutcome.SUBMITTED
    assert result.submitted is True
    assert result.broker_order_id

    # --- the broker ------------------------------------------------------
    after = await stack.facts()
    assert after.order_count == before.order_count + 1
    assert after.order_statuses == (OrderStatus.COMPLETE,)
    assert after.reject_reasons == (None,)
    assert after.filled_qty == 10
    assert after.trade_count == 1

    order = (await stack.broker.get_orders())[0]
    assert order["order_id"] == result.broker_order_id
    assert order["txn_type"] == TransactionType.BUY.value
    assert order["product"] == Product.MIS.value, "an intraday signal must trade MIS"

    # --- holdings --------------------------------------------------------
    assert after.holdings == {POS_KEY_MIS: 10}

    # --- cash: re-derived from the ledger, not read back from the broker --
    trades = await stack.broker.get_trades()
    assert len(trades) == 1
    fill = trades[0]
    assert fill["qty"] == 10
    assert fill["tag"] == order["tag"], "the fill is not linked to its order"
    assert fill["price"] >= PRICE, "a BUY must not fill below the last traded price"
    assert int(fill["costs_paise"]) > 0, "a fill with zero cost is not a real fill"

    outflow = int(fill["notional_paise"]) + int(fill["costs_paise"])
    assert to_paise(before.total_cash) - to_paise(after.total_cash) == outflow
    assert to_paise(after.total_cash) == ledger_cash_paise(trades, stack.initial_cash)
    assert after.total_cash < before.total_cash, "a BUY must consume cash"

    # reserved buying power is released once the order is terminal
    assert after.available_cash == after.total_cash
    await assert_value_is_conserved(stack.broker)

    # --- local state ------------------------------------------------------
    records = await stack.local_records()
    assert len(records) == 1
    rec = records[0]
    assert rec.broker_order_id == result.broker_order_id
    assert rec.client_order_id == order["tag"], "the broker tag is the client order id"
    assert rec.qty == 10
    assert rec.symbol == SYMBOL
    assert rec.side == TransactionType.BUY.value
    assert rec.reference_price == pytest.approx(PRICE)
    # FILLED is now part of the expected walk. The paper broker fills
    # synchronously, and `OrderManager` now applies that fill before returning
    # (`_apply_any_immediate_fill`). Previously the record stopped at
    # ACKNOWLEDGED with filled_qty=0 while the broker reported COMPLETE — local
    # and broker state diverged on every successful fill, and the next
    # reconciliation raised a quantity mismatch for a perfectly normal order.
    #
    # This assertion is updated rather than relaxed: it still pins the exact
    # sequence, and `test_BUG_a_synchronous_fill_is_never_written_back_to_the_
    # local_record` in this same file independently requires filled_qty == 10.
    # An order that reaches FILLED without walking these states still fails.
    assert [h["to"] for h in rec.history if "to" in h] == [
        OrderState.RISK_CHECK_PENDING.value,
        OrderState.RISK_APPROVED.value,
        OrderState.SUBMITTED.value,
        OrderState.ACKNOWLEDGED.value,
        OrderState.FILLED.value,
    ], "the order did not walk the declared lifecycle"

    # --- the audit record --------------------------------------------------
    a = stack.last_audit
    assert a.outcome == ExecutionOutcome.SUBMITTED.value
    assert a.reached_broker is True
    assert a.risk_checks_passed is True
    assert a.rejection_reason is None
    assert a.broker_order_id == result.broker_order_id
    assert a.trading_mode == "paper"
    assert a.symbol == SYMBOL
    assert a.quantity == 10
    assert a.side == TransactionType.BUY.value
    assert a.product == Product.MIS.value
    assert a.strategy == INTRADAY
    assert a.kill_switch_active is False
    assert a.reconciliation_state == str(StartupState.READY)
    assert a.eligibility_state == "NOT_EVALUATED"


async def test_01b_portfolio_value_is_conserved_across_a_round_trip():
    stack = await build_stack(initial_cash=1_000_000.0)
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    await stack.service.submit_signal(
        signal(direction=SignalDirection.EXIT), 10, **WIDE_RISK)

    facts = await stack.facts()
    assert facts.holdings == {}, "the round trip did not close the position"
    assert facts.trade_count == 2

    perf = await stack.broker.get_paper_performance()
    await assert_value_is_conserved(stack.broker)
    assert perf["realised_pnl"] < 0, "a flat round trip must lose the costs and spread"
    assert perf["total_transaction_costs"] > 0
    assert to_paise(perf["current_cash"]) == ledger_cash_paise(
        await stack.broker.get_trades(), stack.initial_cash)


async def test_01c_a_signal_with_a_reference_price_carries_its_stop_to_the_broker():
    """`stop_loss_pct` is a fraction; the order record needs an absolute trigger."""
    stack = await build_stack()
    await stack.service.submit_signal(
        signal(stop_loss_pct=0.02), 10, reference_price=PRICE, **WIDE_RISK)

    rec = (await stack.local_records())[0]
    assert rec.trigger_price == pytest.approx(PRICE * 0.98), (
        "the risk anchor the sizer used was not carried through"
    )


# =========================================================================== #
#  2. Eligibility failure -> NO broker order                                  #
# =========================================================================== #

async def test_02_eligibility_failure_produces_no_live_order():
    """
    Eligibility is ENFORCED for live and RECORDED for paper (service.py:367-378).
    So the honest test of "eligibility failure blocks" is a LIVE-mode service
    pointed at a venue that would happily accept an order.
    """
    from app.governance.eligibility import (
        assess_live_trading_eligibility,
        gather_repo_evidence,
    )

    def real_eligibility():
        return assess_live_trading_eligibility(gather_repo_evidence())

    report = real_eligibility()
    assert report.permits_live_trading is False, (
        "this repo is expected to be ineligible; the gates cannot be exercised "
        f"otherwise (state={report.state.value})"
    )

    live, service, sink, recovery = await build_live_stack(
        eligibility_provider=real_eligibility)
    assert recovery.trading_permitted, "the gate must be open so eligibility is what blocks"

    result = await service.submit_signal(signal(), 10, **WIDE_RISK)

    assert live.placed == [], "AN ORDER REACHED A LIVE VENUE WHILE INELIGIBLE"
    assert result.blocked
    assert not result.submitted
    a = sink.records[-1]
    assert a.reached_broker is False
    assert a.eligibility_state == report.state.value.upper()
    assert a.eligibility_permits_live is False
    assert a.failed_gates, "the failing gates were not recorded"
    assert a.rejection_reason


async def test_02b_eligibility_is_recorded_on_every_paper_order():
    """Paper does not enforce eligibility, but it must never fail to record it."""
    from app.governance.eligibility import (
        assess_live_trading_eligibility,
        gather_repo_evidence,
    )

    stack = await build_stack(
        eligibility_provider=lambda: assess_live_trading_eligibility(gather_repo_evidence()))
    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    assert result.outcome is ExecutionOutcome.SUBMITTED
    a = stack.last_audit
    assert a.eligibility_state is not None
    assert a.eligibility_state != "LIVE_ELIGIBLE"
    assert a.eligibility_permits_live is False
    assert len(a.failed_gates or []) > 0


async def test_02c_a_forged_eligibility_report_cannot_unlock_a_live_order():
    """
    A report that was not computed in-process is UNTRUSTED, whatever it claims.
    Anyone who can write JSON could otherwise mint a LIVE_ELIGIBLE verdict.
    """
    from app.governance.eligibility import (
        EligibilityReport,
        EligibilityState,
        GateResult,
        ReportProvenance,
        _CANONICAL_GATES,
    )

    forged = EligibilityReport(results=tuple(
        GateResult(name=g.name, category=g.category, blocking_state=g.blocking_state,
                   requirement=g.requirement, passed=True, reason="forged")
        for g in _CANONICAL_GATES
    ))
    assert forged.state is EligibilityState.LIVE_ELIGIBLE, "the forgery should look clean"
    assert forged.provenance is ReportProvenance.UNTRUSTED
    assert forged.permits_live_trading is False, "an untrusted report permitted live trading"

    live, service, sink, _ = await build_live_stack(eligibility_provider=lambda: forged)
    result = await service.submit_signal(signal(), 10, **WIDE_RISK)

    assert live.placed == [], "A FORGED ELIGIBILITY REPORT PLACED A LIVE ORDER"
    assert not result.submitted
    assert sink.records[-1].reached_broker is False


# =========================================================================== #
#  3. Risk violation -> NO broker order                                       #
# =========================================================================== #

@pytest.mark.parametrize("narrowed,expect_gate", [
    ({"max_daily_risk": 1.0}, "risk_limit"),
    ({"max_daily_loss": 1.0, "realised_pnl_today": -50_000.0}, "daily_loss_limit"),
    ({"total_portfolio": 1_000.0}, "single_stock_exposure"),
    ({"max_positions": 0, "current_positions": [{"symbol": "X", "quantity": 1}]},
     "position_limit"),
    ({"available_cash": 1.0}, "capital_availability"),
    ({"total_portfolio": float("nan")}, "single_stock_exposure"),
])
async def test_03_risk_violation_produces_no_broker_order(narrowed, expect_gate):
    stack = await build_stack()
    before = await stack.facts()

    result = await stack.service.submit_signal(signal(), 10, **{**WIDE_RISK, **narrowed})

    await assert_nothing_reached_the_broker(stack, before, result)
    assert result.outcome is ExecutionOutcome.BLOCKED_RISK
    assert stack.last_audit.risk_checks_passed is False

    rec = (await stack.local_records())[0]
    assert rec.state is OrderState.RISK_REJECTED
    reasons = rejection_reasons(rec)
    assert reasons, "the rejection reasons were not persisted"
    assert any(expect_gate in r for r in reasons), (
        f"expected gate {expect_gate!r} in {reasons}"
    )


async def test_03b_a_gate_that_errors_rejects_rather_than_permits():
    """FAIL CLOSED: an un-run gate must never leave passed=True."""
    stack = await build_stack()
    before = await stack.facts()

    class Exploding:
        def __contains__(self, item): raise RuntimeError("boom")
        def __iter__(self): raise RuntimeError("boom")

    result = await stack.service.submit_signal(
        signal(), 10, **{**WIDE_RISK, "open_orders": [object()]})

    await assert_nothing_reached_the_broker(stack, before, result)
    rec = (await stack.local_records())[0]
    assert rec.state is OrderState.RISK_REJECTED
    assert any("duplicate_order" in r for r in rejection_reasons(rec))


# =========================================================================== #
#  4. Kill switch active -> NO broker order                                   #
# =========================================================================== #

async def test_04_kill_switch_produces_no_broker_order():
    stack = await build_stack()
    before = await stack.facts()
    stack.kill_store.engage("operator pressed stop")

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    await assert_nothing_reached_the_broker(stack, before, result)
    assert result.outcome is ExecutionOutcome.BLOCKED_KILL_SWITCH
    assert stack.last_audit.kill_switch_active is True
    # Nothing was even reserved: the gate runs before the order store is touched.
    assert await stack.local_records() == []

    # And it releases cleanly.
    stack.kill_store.release()
    ok = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    assert ok.outcome is ExecutionOutcome.SUBMITTED
    assert (await stack.facts()).order_count == before.order_count + 1


async def test_04b_an_unreadable_kill_switch_store_is_treated_as_active():
    """A stop control whose state cannot be read must be assumed engaged."""
    stack = await build_stack()
    before = await stack.facts()

    class Unreadable:
        def get(self, key): raise ConnectionError("redis down")
        def set(self, key, value): raise ConnectionError("redis down")

    stack.service.kill_switch = KillSwitch(store_kill_switch_probe(Unreadable()))
    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    await assert_nothing_reached_the_broker(stack, before, result)
    assert result.outcome is ExecutionOutcome.BLOCKED_KILL_SWITCH
    assert "could not be read" in (result.reason or "")


async def test_04c_a_service_with_no_kill_switch_source_refuses_to_trade():
    stack = await build_stack()
    before = await stack.facts()
    stack.service.kill_switch = KillSwitch()

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    await assert_nothing_reached_the_broker(stack, before, result)
    assert "no kill-switch source configured" in (result.reason or "")


async def test_04d_the_safety_layer_kill_switch_also_stops_an_order():
    """
    Belt and braces: even with the boundary's own probe disarmed, the switch
    ExecutionSafety reads must still stop the order before the broker.
    """
    stack = await build_stack()
    before = await stack.facts()
    stack.service.kill_switch = KillSwitch(lambda: False)   # boundary probe says OK
    stack.kill_store.engage("engaged behind the boundary")

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    await assert_nothing_reached_the_broker(stack, before, result)
    rec = (await stack.local_records())[0]
    assert any("kill_switch" in r for r in rejection_reasons(rec))


# =========================================================================== #
#  5. Broker timeout / ambiguous response -> NO duplicate order               #
# =========================================================================== #

async def test_05_a_broker_timeout_calls_place_order_exactly_once():
    """
    `place_order` times out, but the order DID reach the venue.  The manager
    must resolve that by reading the order book for its client tag, and must
    never place a second order.
    """
    stack = await build_stack()
    calls: list[dict] = []
    real_place = stack.broker.place_order

    async def timing_out(**kw):
        calls.append(kw)
        await real_place(**kw)          # the order really is placed ...
        raise TimeoutError("kite gateway timeout — response lost")   # ... response lost

    stack.broker.place_order = timing_out

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    assert len(calls) == 1, f"place_order was called {len(calls)} times — a retry"
    after = await stack.facts()
    assert after.order_count == 1, f"a timeout produced {after.order_count} broker orders"
    assert after.trade_count == 1, "the fill was double-booked"
    assert after.holdings == {POS_KEY_MIS: 10}
    assert to_paise(after.total_cash) == ledger_cash_paise(
        await stack.broker.get_trades(), stack.initial_cash)
    await assert_value_is_conserved(stack.broker)

    # It was resolved by QUERYING the broker, not by guessing.
    rec = (await stack.local_records())[0]
    assert any(h.get("to") == OrderState.UNKNOWN.value for h in rec.history), (
        "the ambiguous submission did not pass through UNKNOWN"
    )
    assert any(h.get("reconciled") for h in rec.history), (
        "UNKNOWN was left without a broker reconciliation"
    )
    assert rec.state is OrderState.FILLED
    assert rec.broker_order_id
    assert result.outcome is ExecutionOutcome.SUBMITTED
    assert stack.last_audit.reached_broker is True


async def test_05b_a_timeout_with_the_order_absent_from_the_book_places_nothing():
    """The mirror case: the order never left, so nothing may be booked."""
    stack = await build_stack()
    before = await stack.facts()
    calls = []

    async def timing_out(**kw):
        calls.append(kw)
        raise TimeoutError("kite gateway timeout")

    stack.broker.place_order = timing_out
    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    assert len(calls) == 1
    stack.broker.place_order = None      # nothing below may place an order
    after = await stack.facts()
    assert after.order_count == before.order_count
    assert after.trade_count == before.trade_count
    assert after.total_cash == before.total_cash
    assert after.holdings == before.holdings
    assert not result.submitted
    await assert_value_is_conserved(stack.broker)

    rec = (await stack.local_records())[0]
    assert rec.state is OrderState.REJECTED
    assert "not present in broker order book" in rec.reason


async def test_05c_an_unresolved_ambiguous_order_blocks_the_next_order():
    """
    An order whose outcome nobody knows must stop further work in that
    instrument, rather than being traded around.
    """
    stack = await build_stack()

    async def timing_out(**kw):
        raise TimeoutError("kite gateway timeout")

    async def book_unreadable():
        raise ConnectionError("order book unreachable")

    stack.broker.place_order = timing_out
    stack.broker.get_orders = book_unreadable
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    blocked = [r for r in await stack.local_records() if r.is_blocked]
    assert len(blocked) == 1
    assert blocked[0].state is OrderState.UNKNOWN

    # A second signal in the same symbol and strategy must not be submitted.
    stack.broker.place_order = FakeFeed().place_order      # would raise if called
    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    assert not result.submitted
    assert "UNKNOWN" in (result.reason or "") or "unresolved" in (result.reason or "")


# =========================================================================== #
#  6. Partial fill -> correct accounting                                      #
# =========================================================================== #

async def test_06_a_partial_fill_books_only_what_actually_traded():
    """
    Session volume is 1,000 and max_participation is 10%, so an order for 500
    can only take 100.  The remainder is cancelled (there is no queue model),
    and nothing may be booked for it.
    """
    stack = await build_stack(initial_cash=100_000_000.0, volume=1_000)
    before = await stack.facts()

    result = await stack.service.submit_signal(signal(), 500, **WIDE_RISK)

    assert result.outcome is ExecutionOutcome.SUBMITTED
    after = await stack.facts()
    assert after.order_count == before.order_count + 1

    order = (await stack.broker.get_orders())[0]
    assert order["status"] == OrderStatus.PARTIAL
    assert order["qty"] == 500
    assert order["filled_qty"] == 100, "the participation cap was not applied"
    assert "remainder cancelled" in order["message"]

    # Only the traded quantity is in the book.
    assert after.holdings == {POS_KEY_MIS: 100}
    assert after.trade_count == 1
    trades = await stack.broker.get_trades()
    assert trades[0]["qty"] == 100

    # Cash moved by exactly the traded notional plus its costs, no more.
    outflow = int(trades[0]["notional_paise"]) + int(trades[0]["costs_paise"])
    assert to_paise(before.total_cash) - to_paise(after.total_cash) == outflow
    assert to_paise(after.total_cash) == ledger_cash_paise(trades, stack.initial_cash)

    # The unfilled 400 must not still be earmarking buying power.
    assert after.available_cash == after.total_cash
    await assert_value_is_conserved(stack.broker)


async def test_06b_a_partial_fill_is_never_reported_as_complete():
    stack = await build_stack(initial_cash=100_000_000.0, volume=1_000)
    await stack.service.submit_signal(signal(), 500, **WIDE_RISK)

    order = (await stack.broker.get_orders())[0]
    assert order["status"] != OrderStatus.COMPLETE
    assert order["filled_qty"] < order["qty"]
    perf = await stack.broker.get_paper_performance()
    assert perf["total_trades"] == 1


# =========================================================================== #
#  7. Rejected order -> cash and holdings unchanged                           #
# =========================================================================== #

async def test_07_a_broker_rejection_leaves_cash_and_holdings_untouched():
    stack = await build_stack()
    # Own something first, so "unchanged" is a real statement about a non-empty book.
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    before = await stack.facts()
    assert before.holdings == {POS_KEY_MIS: 10}

    # A sell of stock we do not have: rejected by the broker itself.
    result = await stack.service.submit_signal(
        signal(direction=SignalDirection.EXIT), 40, **WIDE_RISK)

    after = await stack.facts()
    assert after.order_count == before.order_count + 1, "the order must reach the broker"
    assert after.order_statuses[-1] == OrderStatus.REJECTED
    assert after.reject_reasons[-1] == RejectReason.INSUFFICIENT_HOLDINGS
    assert after.filled_qty == before.filled_qty, "a rejected order filled"
    assert after.trade_count == before.trade_count, "a rejected order booked a trade"

    # The two facts that matter.
    assert after.total_cash == before.total_cash, "a rejected SELL moved cash"
    assert after.holdings == before.holdings, "a rejected SELL moved holdings"
    assert after.available_cash == before.available_cash
    assert to_paise(after.total_cash) == ledger_cash_paise(
        await stack.broker.get_trades(), stack.initial_cash)
    await assert_value_is_conserved(stack.broker)

    a = stack.last_audit
    assert a.reached_broker is True, "the attempt did reach the broker"
    assert a.broker_order_id


# =========================================================================== #
#  8. Insufficient cash -> rejection                                          #
# =========================================================================== #

async def test_08a_insufficient_cash_is_refused_by_the_safety_gate():
    stack = await build_stack()
    before = await stack.facts()

    result = await stack.service.submit_signal(
        signal(), 10, **{**WIDE_RISK, "available_cash": 5.0})

    await assert_nothing_reached_the_broker(stack, before, result)
    rec = (await stack.local_records())[0]
    assert rec.state is OrderState.RISK_REJECTED
    assert any("capital_availability" in r for r in rejection_reasons(rec))


async def test_08b_insufficient_cash_is_refused_by_the_broker_and_never_overdraws():
    """
    The gates are told there is cash; the paper account does not have it.
    The broker must refuse rather than let the balance go negative.
    """
    stack = await build_stack(initial_cash=500.0)
    before = await stack.facts()
    assert before.total_cash == 500.0

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    after = await stack.facts()
    assert after.order_count == 1, "the order must reach the broker to be refused by it"
    assert after.order_statuses == (OrderStatus.REJECTED,)
    assert after.reject_reasons == (RejectReason.INSUFFICIENT_CASH,)
    assert after.total_cash == 500.0, "cash moved on a rejected order"
    assert after.total_cash >= 0.0, "the account overdrew"
    assert after.available_cash == 500.0, "buying power stayed reserved"
    assert after.holdings == {}
    assert after.trade_count == 0
    assert after.filled_qty == 0
    assert result.submitted is True, "it did reach the broker"
    assert stack.last_audit.reached_broker is True
    await assert_value_is_conserved(stack.broker)


async def test_08c_a_buy_can_never_drive_cash_negative():
    """Size the order so it is affordable at the touch but not after impact."""
    stack = await build_stack(initial_cash=2_000.0, volume=50_000)
    for qty in (15, 19, 20, 21, 25, 100):
        await stack.service.submit_signal(signal(), qty, **WIDE_RISK)
        facts = await stack.facts()
        assert facts.total_cash >= 0.0, f"cash went negative after qty={qty}"
        assert facts.available_cash >= 0.0
        await assert_value_is_conserved(stack.broker)


# =========================================================================== #
#  9. Insufficient holdings (sell more than owned) -> rejection               #
# =========================================================================== #

async def test_09_selling_more_than_owned_is_rejected_and_creates_no_short():
    stack = await build_stack()
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    before = await stack.facts()

    result = await stack.service.submit_signal(
        signal(direction=SignalDirection.EXIT), 100, **WIDE_RISK)

    after = await stack.facts()
    assert after.order_statuses[-1] == OrderStatus.REJECTED
    assert after.reject_reasons[-1] == RejectReason.INSUFFICIENT_HOLDINGS
    assert after.holdings == {POS_KEY_MIS: 10}, "holdings changed on a refused sell"
    assert all(q >= 0 for q in after.holdings.values()), "a short position was created"
    assert after.total_cash == before.total_cash, "a refused sell credited cash"
    assert after.trade_count == before.trade_count
    assert result.submitted is True
    await assert_value_is_conserved(stack.broker)


async def test_09b_selling_with_no_position_at_all_is_refused_as_a_short():
    stack = await build_stack()
    before = await stack.facts()

    result = await stack.service.submit_signal(
        signal(direction=SignalDirection.EXIT), 25, **WIDE_RISK)

    after = await stack.facts()
    assert after.reject_reasons[-1] == RejectReason.SHORT_SELL_NOT_SUPPORTED
    assert after.holdings == {}
    assert after.total_cash == before.total_cash, "selling stock never owned minted cash"
    assert after.trade_count == 0
    assert result.submitted is True
    await assert_value_is_conserved(stack.broker)


async def test_09c_a_sell_never_exceeds_the_holding_however_it_is_sliced():
    stack = await build_stack()
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    for qty in (4, 4, 4, 4):
        await stack.service.submit_signal(
            signal(direction=SignalDirection.EXIT), qty, **WIDE_RISK)
        facts = await stack.facts()
        assert all(q >= 0 for q in facts.holdings.values())
    assert (await stack.facts()).holdings in ({}, {POS_KEY_MIS: 2})
    await assert_value_is_conserved(stack.broker)


# =========================================================================== #
#  10. Market closed -> rejection                                             #
# =========================================================================== #

@pytest.mark.parametrize("clock_at,label", [
    (CLOSED_IST, "after the 15:30 IST close"),
    (WEEKEND_IST, "on a Sunday"),
    (HOLIDAY_IST, "on an NSE trading holiday"),
])
async def test_10_market_closed_produces_no_order(clock_at, label):
    stack = await build_stack(clock=lambda: clock_at, prime=())
    # Prime deliberately AFTER the clock is fixed: the staleness cache is
    # independent of the session gate, so this isolates "market closed".
    await prime_quote_cache(stack.broker, SYMBOL)
    before = await stack.facts()
    assert stack.broker.is_market_open() is False, label

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    await assert_nothing_reached_the_broker(stack, before, result)
    rec = (await stack.local_records())[0]
    assert rec.state is OrderState.RISK_REJECTED
    assert any("market_status" in r for r in rejection_reasons(rec)), (
        f"the market-hours gate did not fire {label}"
    )


async def test_10b_market_closed_is_judged_in_ist_not_in_the_host_timezone():
    """
    15:00 UTC is 20:30 IST — outside the session.  A host running TZ=UTC must
    reach the same verdict as one running TZ=Asia/Kolkata.
    """
    utc_afternoon = datetime(2025, 6, 10, 15, 0, tzinfo=timezone.utc)
    stack = await build_stack(clock=lambda: utc_afternoon, prime=())
    assert stack.broker.now_ist().hour == 20
    assert stack.broker.is_market_open() is False

    await prime_quote_cache(stack.broker, SYMBOL)
    before = await stack.facts()
    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    await assert_nothing_reached_the_broker(stack, before, result)


# =========================================================================== #
#  11. Restart -> reconciliation runs, trading blocked until it succeeds       #
# =========================================================================== #

async def test_11_trading_is_blocked_until_startup_reconciliation_succeeds():
    stack = await build_stack(run_recovery=False)
    before = await stack.facts()

    # A freshly started process has not reconciled.
    assert stack.recovery.state is StartupState.BLOCKED
    assert stack.recovery.trading_permitted is False

    blocked = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    await assert_nothing_reached_the_broker(stack, before, blocked)
    assert blocked.outcome is ExecutionOutcome.BLOCKED_NOT_RECONCILED
    assert "recovery has not run yet" in (blocked.reason or "")
    assert stack.last_audit.reconciliation_state == str(StartupState.BLOCKED)
    assert await stack.local_records() == [], "an intent was reserved while blocked"

    # Now reconcile.
    report = await stack.recovery.recover(stack.broker)
    assert report.status is ReconciliationStatus.RECONCILIATION_OK
    assert report.trading_permitted is True
    assert stack.recovery.state is StartupState.READY
    assert [p.phase.value for p in report.phases][:3] == [
        "load_local_state", "query_broker", "reconcile_orders"]

    # And only now can an order be placed.
    ok = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    assert ok.outcome is ExecutionOutcome.SUBMITTED
    assert (await stack.facts()).order_count == before.order_count + 1
    assert stack.last_audit.reconciliation_state == str(StartupState.READY)


async def test_11b_ready_is_reachable_only_through_reconciliation_ok():
    """`_enter_ready` re-asserts its own precondition rather than trusting it."""
    from app.execution.reconciliation import ReconciliationResult

    stack = await build_stack(run_recovery=False)
    for status in (ReconciliationStatus.RECONCILIATION_MISMATCH,
                   ReconciliationStatus.RECONCILIATION_UNAVAILABLE,
                   ReconciliationStatus.RECONCILIATION_ERROR):
        with pytest.raises(AssertionError):
            stack.recovery._enter_ready(ReconciliationResult(status=status))
        assert stack.recovery.trading_permitted is False


async def test_11c_a_restart_does_not_duplicate_positions_or_orders():
    """
    The book survives the restart exactly as it was — no order is re-placed and
    no position is counted twice.
    """
    redis = FakeRedis()
    stack = await build_stack(initial_cash=1_000_000.0, redis_client=redis)
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    await stack.broker.disconnect()

    before_orders = await stack.broker.get_orders()
    before_facts = await stack.facts()

    # A new process over the same persisted state.
    restarted = PaperBroker(data_broker=FakeFeed(), initial_cash=1_000_000.0,
                            redis_client=redis, clock=lambda: OPEN_IST)
    await restarted.connect()
    after_facts = await BrokerFacts.of(restarted)

    assert after_facts.order_count == before_facts.order_count
    assert after_facts.trade_count == before_facts.trade_count
    assert after_facts.holdings == before_facts.holdings
    assert after_facts.total_cash == before_facts.total_cash
    assert [o["order_id"] for o in await restarted.get_orders()] == [
        o["order_id"] for o in before_orders]
    await assert_value_is_conserved(restarted)

    # Order-manager recovery over the same durable store places nothing either.
    recovered = await stack.order_manager.recover(restarted)
    assert (await BrokerFacts.of(restarted)).order_count == before_facts.order_count
    assert len({r.client_order_id for r in recovered}) == len(recovered)


# =========================================================================== #
#  12. Broker unavailable -> trading blocked                                  #
# =========================================================================== #

async def test_12_an_unreachable_broker_blocks_trading():
    # The kill switch is disarmed for this pass so that the block is
    # attributable to the trading gate and not masked by the halt latch.
    stack = await build_stack(run_recovery=False, activate_kill_switch_on=NEVER_LATCH)
    before = await stack.facts()

    with broker_outage(stack.broker):
        report = await stack.recovery.recover(stack.broker)

    assert report.status is ReconciliationStatus.RECONCILIATION_UNAVAILABLE
    assert report.trading_permitted is False
    assert stack.recovery.state is StartupState.BLOCKED
    assert sorted(report.result.unavailable_sources) == [
        "broker_cash", "broker_orders", "broker_positions", "broker_trades"]
    assert bool(report) is False, "a blocked report must not read as truthy"
    assert bool(report.status) is False

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    await assert_nothing_reached_the_broker(stack, before, result)
    assert result.outcome is ExecutionOutcome.BLOCKED_NOT_RECONCILED


async def test_12b_an_unreachable_broker_also_latches_the_kill_switch():
    """With the default policy, UNAVAILABLE persists a halt as well."""
    stack = await build_stack(run_recovery=False)
    with broker_outage(stack.broker):
        report = await stack.recovery.recover(stack.broker)

    assert report.kill_switch_activated is True
    assert stack.kill_store.get("kill_switch")
    assert await stack.safety.is_kill_switch_active() is True

    before = await stack.facts()
    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    await assert_nothing_reached_the_broker(stack, before, result)
    assert result.outcome is ExecutionOutcome.BLOCKED_KILL_SWITCH


async def test_12c_a_broker_that_cannot_be_connected_blocks_startup():
    """The real bootstrap startup contract: connect failure => gate stays shut."""
    stack = await build_stack(run_recovery=False)

    async def refuse():
        raise ConnectionError("kite login failed")

    stack.broker.connect = refuse
    ok, reason = await _run_startup_recovery(stack.recovery, stack.broker)

    assert ok is False
    assert "broker connect failed" in reason
    assert stack.recovery.trading_permitted is False

    stack.broker.connect = None
    before = await stack.facts()
    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    await assert_nothing_reached_the_broker(stack, before, result)


async def test_12d_a_partially_unreachable_broker_is_unavailable_not_ok():
    """`[] vs []` must never be reported as reconciled."""
    stack = await build_stack(run_recovery=False, activate_kill_switch_on=NEVER_LATCH)

    async def down(*a, **k):
        raise ConnectionError("kite 5xx")

    stack.broker.get_trades = down
    report = await stack.recovery.recover(stack.broker)

    assert report.status is ReconciliationStatus.RECONCILIATION_UNAVAILABLE
    assert "broker_trades" in report.result.unavailable_sources
    assert stack.recovery.trading_permitted is False


# =========================================================================== #
#  13. State mismatch -> trading blocked                                      #
# =========================================================================== #

@pytest.mark.parametrize("local_kwargs,expect_kind", [
    ({"positions": [{"symbol": "TCS", "exchange": "NSE", "product": "MIS",
                     "quantity": 50, "average_price": 100.0}]}, "missing_broker"),
    ({"cash": 12_345.0}, "mismatched_cash"),
    ({"orders": [{"order_id": "X1", "symbol": "TCS", "exchange": "NSE",
                  "status": "SUBMITTING", "quantity": 10}]}, "unknown_order_state"),
])
async def test_13_a_state_mismatch_blocks_trading(local_kwargs, expect_kind):
    kwargs = {"cash": 1_000_000.0, **local_kwargs}
    stack = await build_stack(
        run_recovery=False,
        local_state=InMemoryLocalState(**kwargs),
        activate_kill_switch_on=NEVER_LATCH,
    )
    before = await stack.facts()

    report = await stack.recovery.recover(stack.broker)

    assert report.status is ReconciliationStatus.RECONCILIATION_MISMATCH
    assert report.trading_permitted is False
    assert expect_kind in {d.kind.value for d in report.unresolved}

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    await assert_nothing_reached_the_broker(stack, before, result)
    assert result.outcome is ExecutionOutcome.BLOCKED_NOT_RECONCILED


async def test_13b_an_untracked_broker_position_is_the_most_dangerous_mismatch():
    """A position the broker holds and we do not know about must block."""
    stack = await build_stack(activate_kill_switch_on=NEVER_LATCH)
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)   # broker now holds 10

    # Local state still knows nothing about it.
    result = await stack.engine.evaluate(stack.broker)
    assert result.status is ReconciliationStatus.RECONCILIATION_MISMATCH
    assert "missing_local" in {d.kind.value for d in result.discrepancies}

    stack.recovery.block(result.recommendation)
    before = await stack.facts()
    prior = await stack.cids()
    blocked = await stack.service.submit_signal(signal("TCS"), 10, **WIDE_RISK)
    await assert_nothing_reached_the_broker(stack, before, blocked, prior_cids=prior)


async def test_13c_a_mismatch_that_is_repaired_reconciles_and_reopens_the_gate():
    """Blocking must not be permanent once local and broker agree again."""
    local = InMemoryLocalState(cash=1_000_000.0)
    stack = await build_stack(run_recovery=False, local_state=local,
                              activate_kill_switch_on=NEVER_LATCH)
    local.cash = 999.0
    assert (await stack.recovery.recover(stack.broker)).trading_permitted is False

    local.cash = 1_000_000.0
    report = await stack.recovery.recover(stack.broker)
    assert report.status is ReconciliationStatus.RECONCILIATION_OK
    assert stack.recovery.trading_permitted is True

    ok = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    assert ok.outcome is ExecutionOutcome.SUBMITTED


# =========================================================================== #
#  14. Concurrent duplicate submissions -> ONE logical order                  #
# =========================================================================== #

@pytest.mark.parametrize("fanout", [2, 5, 12])
async def test_14_concurrent_duplicate_submissions_place_exactly_one_order(fanout):
    """
    One alpha signal, submitted `fanout` times at once — a double-clicked
    button, a retried POST, two workers racing on the same allocation.  Exactly
    one order may reach the broker.

    KNOWN TO FAIL — see test_BUG_the_boundary_mints_a_new_idempotency_key_per_call.
    """
    stack = await build_stack()
    before = await stack.facts()
    sig = signal()                       # ONE intent, submitted many times at once

    place_calls: list[str] = []
    real_place = stack.broker.place_order

    async def counting(**kw):
        place_calls.append(kw.get("tag", ""))
        return await real_place(**kw)

    stack.broker.place_order = counting

    results = await asyncio.gather(
        *[stack.service.submit_signal(sig, 5, **WIDE_RISK) for _ in range(fanout)],
        return_exceptions=True,
    )

    assert not [r for r in results if isinstance(r, BaseException)], results
    assert len(stack.sink.records) == fanout, "not every attempt was audited"

    # --- the broker ------------------------------------------------------
    after = await stack.facts()
    assert len(place_calls) == 1, (
        f"place_order was called {len(place_calls)} times for ONE signal"
    )
    assert after.order_count == before.order_count + 1, (
        f"{fanout} concurrent submissions of one signal produced "
        f"{after.order_count - before.order_count} broker orders"
    )
    assert after.trade_count == 1, "the fill was booked more than once"
    assert after.filled_qty == 5
    assert after.holdings == {POS_KEY_MIS: 5}, (
        "concurrent submissions multiplied the position"
    )

    # Cash moved exactly once.
    trades = await stack.broker.get_trades()
    outflow = int(trades[0]["notional_paise"]) + int(trades[0]["costs_paise"])
    assert to_paise(before.total_cash) - to_paise(after.total_cash) == outflow
    assert to_paise(after.total_cash) == ledger_cash_paise(trades, stack.initial_cash)

    # --- local state: one record, one broker id ---------------------------
    records = await stack.local_records()
    assert len(records) == 1, f"{len(records)} durable records for one signal"
    assert {r.broker_order_id for r in records} == {trades[0]["order_id"]}

    # --- every caller was told about the SAME order -----------------------
    reported = {r.broker_order_id for r in results if r.broker_order_id}
    assert len(reported) == 1, f"callers were given different order ids: {reported}"


async def test_14b_sequential_resubmission_of_one_signal_places_one_order():
    """The non-concurrent form of the same requirement: a retried POST."""
    stack = await build_stack()
    sig = signal()

    first = await stack.service.submit_signal(sig, 10, **WIDE_RISK)
    second = await stack.service.submit_signal(sig, 10, **WIDE_RISK)

    facts = await stack.facts()
    assert facts.order_count == 1, (
        f"resubmitting one signal produced {facts.order_count} broker orders"
    )
    assert facts.holdings == {POS_KEY_MIS: 10}, "the position was doubled"
    assert first.broker_order_id == second.broker_order_id


async def test_14c_genuinely_distinct_signals_are_not_collapsed():
    """Idempotency must not silently swallow different orders."""
    stack = await build_stack()
    sigs = [signal(s) for s in ("RELIANCE", "TCS", "INFY")]
    for s in sigs[1:]:
        await prime_quote_cache(stack.broker, s.symbol)

    results = await asyncio.gather(
        *[stack.service.submit_signal(s, 5, **WIDE_RISK) for s in sigs])

    facts = await stack.facts()
    assert facts.order_count == 3
    assert len({r.broker_order_id for r in results}) == 3
    assert sum(facts.holdings.values()) == 15
    await assert_value_is_conserved(stack.broker)


# =========================================================================== #
#  15. Corrupt persisted state -> fail closed                                 #
# =========================================================================== #

@pytest.mark.parametrize("blob,label", [
    ("{not json at all", "unparseable"),
    (json.dumps({"schema": 99, "checksum": "x", "body": "{}"}), "wrong schema"),
    (json.dumps({"schema": 2, "checksum": "deadbeef",
                 "body": json.dumps({"cash_paise": 1})}), "checksum mismatch"),
    ("", "empty"),
])
async def test_15_corrupt_persisted_state_fails_closed(blob, label):
    redis = FakeRedis()
    redis.store["paper_broker:default:state"] = blob
    broker = PaperBroker(data_broker=FakeFeed(), initial_cash=1_000_000.0,
                         redis_client=redis, clock=lambda: OPEN_IST)

    with pytest.raises(PaperBrokerStateError):
        await broker.connect()

    assert broker.is_connected is False, label


async def test_15b_a_tampered_ledger_is_rejected_even_with_a_valid_checksum():
    """The balance is re-derived from the trade ledger, so a re-signed edit fails."""
    redis = FakeRedis()
    stack = await build_stack(initial_cash=1_000_000.0, redis_client=redis)
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    honest = json.loads(redis.store["paper_broker:default:state"])
    body = json.loads(honest["body"])
    body["cash_paise"] += 5_000_000                 # mint Rs 50,000
    new_body = json.dumps(body, sort_keys=True, separators=(",", ":"))
    redis.store["paper_broker:default:state"] = json.dumps({
        "schema": 2,
        "checksum": hashlib.sha256(new_body.encode()).hexdigest(),   # re-signed
        "body": new_body,
    }, separators=(",", ":"))

    restarted = PaperBroker(data_broker=FakeFeed(), initial_cash=1_000_000.0,
                            redis_client=redis, clock=lambda: OPEN_IST)
    with pytest.raises(PaperBrokerStateError) as exc:
        await restarted.connect()
    assert "disagrees with the trade ledger" in str(exc.value)
    assert restarted.is_connected is False


async def test_15c_a_process_that_cannot_read_its_state_cannot_trade():
    """Startup recovery turns a corrupt book into a closed trading gate."""
    redis = FakeRedis()
    redis.store["paper_broker:default:state"] = "{corrupt"
    stack = await build_stack(run_recovery=False, redis_client=redis, prime=(),
                              connect=False)

    ok, reason = await _run_startup_recovery(stack.recovery, stack.broker)

    assert ok is False
    assert "broker connect failed" in reason
    assert "PaperBrokerStateError" in reason
    assert stack.recovery.trading_permitted is False

    before = await stack.facts()
    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    await assert_nothing_reached_the_broker(stack, before, result)


async def test_15d_an_unreadable_state_store_never_fabricates_a_balance():
    class Exploding:
        def get(self, key): raise ConnectionError("redis down")
        def set(self, key, value): raise ConnectionError("redis down")

    broker = PaperBroker(data_broker=FakeFeed(), initial_cash=1_000_000.0,
                         redis_client=Exploding(), clock=lambda: OPEN_IST)
    with pytest.raises(PaperBrokerStateError) as exc:
        await broker.connect()
    assert "fabricated balance" in str(exc.value)
    assert broker.is_connected is False


# =========================================================================== #
#  16. Stale market data -> no order                                          #
# =========================================================================== #

async def test_16_stale_market_data_produces_no_order():
    """A quote 90 minutes old must not be sized against."""
    stale_quote = OPEN_IST - timedelta(minutes=90)
    stack = await build_stack(quote_timestamp=stale_quote, prime=())
    await prime_quote_cache(stack.broker, SYMBOL)      # cache now holds an OLD tick
    assert stack.broker.is_stale_tick(SYMBOL, max_age_seconds=30.0) is True
    before = await stack.facts()

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    await assert_nothing_reached_the_broker(stack, before, result)
    rec = (await stack.local_records())[0]
    assert rec.state is OrderState.RISK_REJECTED
    reasons = rejection_reasons(rec)
    assert any("stale" in r.lower() for r in reasons), reasons


async def test_16b_a_never_quoted_symbol_is_priced_from_a_fresh_fetch():
    """
    REGRESSION for PB-STALE, inverted after the fix.

    This test used to pin the bug. `is_stale_tick` read a cache that only
    `place_order` could populate, while the freshness gate ran BEFORE the
    quote fetch. A never-quoted symbol was therefore stale, the order was
    refused, the fetch never happened, the cache stayed empty, and the next
    order was refused identically. Every symbol was permanently un-tradeable.
    It failed CLOSED, so nothing was ever at risk — but nothing could trade
    either, and a system that cannot open a position is not a safe system, it
    is a broken one.

    Two changes fixed it: `PaperBroker.get_quote` now records quote age (a
    quote observation IS a freshness observation), and the order path fetches
    the quote BEFORE judging its age, because freshness is a property of data
    you have actually fetched.

    Genuine staleness is still refused: `test_16_stale_market_data_produces_
    no_order` drives a 90-minute-old quote through this same path and asserts
    that nothing reaches the broker. What changed is only that "never looked"
    no longer means "permanently forbidden".
    """
    stack = await build_stack(prime=())
    assert stack.broker.is_stale_tick("NEWCO") is True, (
        "a symbol never quoted must start out stale"
    )
    before = await stack.facts()

    result = await stack.service.submit_signal(signal("NEWCO"), 10, **WIDE_RISK)

    assert result.submitted, (
        f"the first order in a fresh symbol was refused: {result.reason}"
    )
    after = await stack.facts()
    assert after.order_count == before.order_count + 1
    assert stack.broker.is_stale_tick("NEWCO") is False, (
        "the fetch on the order path must leave the staleness cache populated"
    )


async def test_16c_a_dead_market_data_feed_produces_no_order():
    stack = await build_stack()
    stack.feed.fail = True
    before = await stack.facts()

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    await assert_nothing_reached_the_broker(stack, before, result)
    rec = (await stack.local_records())[0]
    assert rec.state is OrderState.RISK_REJECTED


async def test_16d_a_market_order_is_never_sized_against_a_placeholder_price():
    """The reference price is the last traded price, never a Signal price of 0."""
    stack = await build_stack(price=2_875.0)

    await stack.service.submit_signal(signal(), 3, **WIDE_RISK)

    rec = (await stack.local_records())[0]
    assert rec.reference_price == pytest.approx(2_875.0)
    trades = await stack.broker.get_trades()
    assert trades[0]["price"] >= 2_875.0


async def test_16e_zero_traded_volume_produces_no_fill():
    """With no volume the impact model cannot be sized, so nothing may clear."""
    stack = await build_stack(volume=0, prime=())
    with contextlib.suppress(Exception):
        await prime_quote_cache(stack.broker, SYMBOL)
    before = await stack.facts()

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    after = await stack.facts()
    assert after.filled_qty == 0, "an order filled against no displayed volume"
    assert after.trade_count == 0
    assert after.total_cash == before.total_cash
    assert after.holdings == before.holdings
    assert not any(s == OrderStatus.COMPLETE for s in after.order_statuses)
    await assert_value_is_conserved(stack.broker)


# =========================================================================== #
#  17. Eligibility dependency raising -> fail closed, not fail open            #
# =========================================================================== #

async def test_17_an_eligibility_dependency_that_raises_never_permits_a_live_order():
    def boom():
        raise RuntimeError("eligibility evidence store is unreachable")

    live, service, sink, recovery = await build_live_stack(eligibility_provider=boom)
    assert recovery.trading_permitted is True, "the gate must be open to isolate eligibility"

    result = await service.submit_signal(signal(), 10, **WIDE_RISK)

    assert live.placed == [], "A LIVE ORDER WAS PLACED WHILE ELIGIBILITY WAS UNKNOWN"
    assert result.outcome is ExecutionOutcome.BLOCKED_ELIGIBILITY
    a = sink.records[-1]
    assert a.eligibility_state == "EVALUATION_FAILED"
    assert a.eligibility_permits_live is False
    assert a.reached_broker is False
    assert "failing closed" in (result.reason or "")


async def test_17b_live_mode_with_no_eligibility_provider_is_refused():
    live, service, sink, _ = await build_live_stack(eligibility_provider=None)

    result = await service.submit_signal(signal(), 10, **WIDE_RISK)

    assert live.placed == []
    assert result.outcome is ExecutionOutcome.BLOCKED_ELIGIBILITY
    assert sink.records[-1].eligibility_state == "NOT_EVALUATED"


async def test_17c_a_paper_order_still_records_a_failed_eligibility_evaluation():
    """
    Paper deliberately does NOT enforce eligibility (service.py:367-378): doing
    so would block the very activity that produces the evidence the gates want.
    What it must never do is record a failed evaluation as a passing one.
    """
    def boom():
        raise RuntimeError("eligibility evidence store is unreachable")

    stack = await build_stack(eligibility_provider=boom)

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    assert result.outcome is ExecutionOutcome.SUBMITTED
    a = stack.last_audit
    assert a.eligibility_state == "EVALUATION_FAILED"
    assert a.eligibility_permits_live is False, "a failed evaluation must never permit live"


async def test_17d_an_eligibility_provider_returning_junk_never_permits_live():
    """Anything that is not a trusted report is a block, including nonsense."""
    for junk in (None, "LIVE_ELIGIBLE", 42, object()):
        live, service, sink, _ = await build_live_stack(
            eligibility_provider=lambda j=junk: j)
        result = await service.submit_signal(signal(), 10, **WIDE_RISK)
        assert live.placed == [], f"{junk!r} permitted a live order"
        assert not result.submitted
        assert sink.records[-1].eligibility_permits_live is not True


# =========================================================================== #
#  Mode separation                                                            #
# =========================================================================== #

def test_mode_a_paper_service_refuses_to_hold_a_live_broker():
    kill_store = InMemoryKillSwitchStore()
    om = OrderManager(ExecutionSafety(kill_store), store=InMemoryOrderStore())
    with pytest.raises(ExecutionBlocked) as exc:
        ExecutionService(broker=FakeLiveBroker(), order_manager=om,
                         trading_mode=TradingMode.PAPER)
    assert exc.value.outcome is ExecutionOutcome.BLOCKED_MODE
    assert "not PaperBroker" in exc.value.reason


def test_mode_a_live_service_refuses_to_hold_a_paper_broker():
    kill_store = InMemoryKillSwitchStore()
    om = OrderManager(ExecutionSafety(kill_store), store=InMemoryOrderStore())
    paper = PaperBroker(data_broker=FakeFeed(), clock=lambda: OPEN_IST)
    with pytest.raises(ExecutionBlocked) as exc:
        ExecutionService(broker=paper, order_manager=om,
                         trading_mode=TradingMode.LIVE, live_authorized=True)
    assert exc.value.outcome is ExecutionOutcome.BLOCKED_MODE
    assert "reports paper fills as live" in exc.value.reason


def test_mode_live_requires_explicit_authorization():
    kill_store = InMemoryKillSwitchStore()
    om = OrderManager(ExecutionSafety(kill_store), store=InMemoryOrderStore())
    with pytest.raises(ExecutionBlocked) as exc:
        ExecutionService(broker=FakeLiveBroker(), order_manager=om,
                         trading_mode=TradingMode.LIVE, live_authorized=False)
    assert exc.value.outcome is ExecutionOutcome.BLOCKED_MODE


async def test_mode_a_paper_session_never_touches_a_live_venue():
    """Belt and braces: run a full paper session with a live venue in scope."""
    live = FakeLiveBroker()
    stack = await build_stack()
    for _ in range(5):
        await stack.service.submit_signal(signal(), 3, **WIDE_RISK)
    assert live.placed == []
    assert stack.broker.trading_mode == "paper"
    assert all(a.trading_mode == "paper" for a in stack.sink.records)
    await assert_value_is_conserved(stack.broker)


# =========================================================================== #
#  Timezone                                                                   #
# =========================================================================== #

def test_tz_a_naive_clock_is_refused_rather_than_guessed_at():
    broker = PaperBroker(data_broker=FakeFeed(),
                         clock=lambda: datetime(2025, 6, 10, 11, 0))   # naive
    with pytest.raises(ValueError) as exc:
        broker.now_ist()
    assert "timezone-naive" in str(exc.value)


def test_tz_the_session_verdict_is_identical_for_equivalent_instants():
    """
    The same physical instant expressed in UTC and in IST must produce the same
    verdict, so the suite's result does not depend on the host's TZ.
    """
    pairs = [
        (datetime(2025, 6, 10, 5, 30, tzinfo=timezone.utc),
         datetime(2025, 6, 10, 11, 0, tzinfo=IST), True),
        (datetime(2025, 6, 10, 14, 30, tzinfo=timezone.utc),
         datetime(2025, 6, 10, 20, 0, tzinfo=IST), False),
        (datetime(2025, 6, 10, 3, 30, tzinfo=timezone.utc),
         datetime(2025, 6, 10, 9, 0, tzinfo=IST), False),
        (datetime(2025, 6, 10, 9, 59, tzinfo=timezone.utc),
         datetime(2025, 6, 10, 15, 29, tzinfo=IST), True),
    ]
    broker = PaperBroker(data_broker=FakeFeed(), clock=lambda: OPEN_IST)
    for utc_instant, ist_instant, expected in pairs:
        assert broker.is_market_open(utc_instant) is expected
        assert broker.is_market_open(ist_instant) is expected
        assert broker.is_market_open(utc_instant) == broker.is_market_open(ist_instant)


async def test_tz_an_intraday_book_is_not_carried_overnight_on_a_utc_host():
    """
    The bug this pins: a UTC host judging 15:00 UTC (20:30 IST) to be inside
    the session would let an intraday position survive the close.
    """
    stack = await build_stack()
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    assert (await stack.facts()).holdings == {POS_KEY_MIS: 10}

    # 15:00 UTC == 20:30 IST: past the close and past square-off.
    after_close = datetime(2025, 6, 10, 15, 0, tzinfo=timezone.utc)
    stack.broker._clock = lambda: after_close
    assert stack.broker.is_market_open() is False
    assert stack.broker.is_squareoff_time() is True

    # A new intraday BUY after square-off is refused ...
    before = await stack.facts()
    prior = await stack.cids()
    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)
    await assert_nothing_reached_the_broker(stack, before, result, prior_cids=prior)

    # ... and the existing book can still be flattened.
    stack.broker._enforce_market_hours = False
    ids = await stack.broker.square_off_intraday()
    assert len(ids) == 1
    assert (await stack.facts()).holdings == {}
    await assert_value_is_conserved(stack.broker)


# =========================================================================== #
#  DEFECT PINS                                                                #
#                                                                             #
#  Each test below asserts the behaviour these modules DOCUMENT and REQUIRE,  #
#  and pins a defect this suite actually found. Each docstring carries a      #
#  file:line and a repro. None is weakened to green: where a defect is still  #
#  open the test fails, which is the most valuable output this suite has.     #
#                                                                             #
#  STATUS legend                                                              #
#    FIXED - repaired while this suite was being written; the test now        #
#            stands as a regression guard.                                    #
#    OPEN  - still present; the test fails.                                   #
# =========================================================================== #

async def test_BUG_the_boundary_mints_a_new_idempotency_key_per_call():
    """
    BUG 1 (SEVERITY: CRITICAL - the boundary had no idempotency at all).

    STATUS: FIXED. `ExecutionService` now derives a deterministic client order
    id from the signal instead of minting a fresh one on every call.

    WHERE
        app/execution/service.py:446-511  `_to_order_intent` constructs a NEW
                                          `order_manager.Signal` on every call
        app/execution/order_manager.py:159-161  `Signal.__post_init__` mints a
                                          fresh `client_order_id` when none is
                                          supplied
        app/execution/order_manager.py:466  `cid = signal.client_order_id`
        app/execution/order_manager.py:497  `self._store.reserve(record)` — the
                                          atomic set-if-not-exists that is the
                                          whole duplicate defence

    The idempotency key is now generated INSIDE the boundary, once per attempt,
    so two submissions of the same alpha signal reserve two different keys and
    place two real orders.  `order_manager.py:141-145` states the contract this
    breaks: "Re-submitting the *same* Signal object (a retry) can therefore
    never place a second live order."  Through `ExecutionService` there is no
    longer any way to re-submit the same object.

    WHY IT MATTERS
        `app/api/routes/allocation.py:268` calls `submit_signal` in a loop over
        `alloc.pending_signals`.  A retried POST, a double-clicked button, or
        two workers racing on one allocation each multiply the position.  This
        is the one failure mode the whole lifecycle module exists to prevent.

    REPRO
        Submit one Signal object twice; two broker orders appear.

    NOTE
        This regressed while this suite was being written: before
        `_to_order_intent` was introduced, the caller owned the client order id
        and re-submission was correctly suppressed.
    """
    stack = await build_stack()
    sig = signal()

    first = await stack.service.submit_signal(sig, 10, **WIDE_RISK)
    second = await stack.service.submit_signal(sig, 10, **WIDE_RISK)

    facts = await stack.facts()
    records = await stack.local_records()

    assert len(records) == 1, (
        f"one signal reserved {len(records)} client order ids: "
        f"{[r.client_order_id for r in records]}"
    )
    assert facts.order_count == 1, (
        f"one signal produced {facts.order_count} broker orders"
    )
    assert facts.holdings == {POS_KEY_MIS: 10}, "the position was doubled"
    assert first.broker_order_id == second.broker_order_id


async def test_BUG_ambiguous_submission_is_audited_as_never_reaching_the_broker():
    """
    BUG 2 (SEVERITY: HIGH - the audit journal stated a falsehood about risk).

    STATUS: FIXED. `order_manager.AmbiguousOrderError` now also inherits from
    `core.exceptions.AmbiguousOrderStateError`, so the handler is reachable.

    WHERE
        app/execution/service.py:55-56   imports AmbiguousOrderStateError and
                                         KillSwitchActiveError from
                                         app.core.exceptions
        app/execution/service.py:546     `except AmbiguousOrderStateError`
        app/execution/order_manager.py:168  raises `AmbiguousOrderError`, which
                                         derives from OrderSubmissionError ->
                                         RuntimeError, NOT from AlgoDollarError

    Neither class is a subclass of the other, so the handler at service.py:546
    is unreachable.  The ambiguous exception falls through to the generic
    handler at service.py:559, which records:

        outcome            = BLOCKED_RISK      (should be AMBIGUOUS)
        risk_checks_passed = False             (the risk checks PASSED)
        reached_broker     = False             (the truth is: UNKNOWN)
        final_state        = None              (should be "UNKNOWN")

    WHY IT MATTERS
        `ExecutionOutcome.AMBIGUOUS.reached_broker` is True precisely because
        an ambiguous order may be live at the exchange.  Recording it as
        BLOCKED_RISK tells an incident responder that no order exists, while
        the system's own OrderRecord sits in UNKNOWN for that very reason.

    REPRO
        place_order raises, and the order book cannot be read either, so the
        outcome genuinely cannot be established.
    """
    stack = await build_stack()

    async def timing_out(**kw):
        raise TimeoutError("kite gateway timeout")

    async def book_unreadable():
        raise ConnectionError("order book unreachable")

    stack.broker.place_order = timing_out
    stack.broker.get_orders = book_unreadable

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    # The local record correctly says "we do not know".
    rec = (await stack.local_records())[0]
    assert rec.state is OrderState.UNKNOWN
    assert rec.is_blocked is True

    a = stack.last_audit
    assert result.outcome is ExecutionOutcome.AMBIGUOUS, (
        f"an unresolvable submission was classified {result.outcome.value}; the "
        f"order may be live at the exchange"
    )
    assert a.reached_broker is True, (
        "the audit journal asserts the order never reached a broker while the "
        "order's own state is UNKNOWN"
    )
    assert a.final_state == "UNKNOWN"


def test_BUG_the_kill_switch_handler_catches_a_class_nothing_raises():
    """
    BUG 3 (SEVERITY: LOW - latent; a safety handler was statically unreachable).

    STATUS: FIXED. `safety.KillSwitchActiveError` now also inherits from
    `core.exceptions.KillSwitchActiveError`.

    WHERE
        app/execution/service.py:53-57  imports KillSwitchActiveError from
                                        app.core.exceptions
        app/execution/service.py:555    `except KillSwitchActiveError`
        app/execution/safety.py:43      the KillSwitchActiveError that is
                                        actually raised, a subclass of
                                        SafetyCheckError -> RuntimeError
        app/execution/order_manager.py:1022  raises the safety one

    The two classes share a name and nothing else, so the handler is dead code:
    a kill-switch failure escaping `submit_order` would be relabelled
    BLOCKED_RISK by the generic handler at service.py:576 and the audit record
    would never set `kill_switch_active`.

    The sibling defect on `AmbiguousOrderError` was fixed while this suite was
    being written — `order_manager.py:169` now inherits from BOTH
    `OrderSubmissionError` and `core.exceptions.AmbiguousOrderStateError`.  The
    same one-line fix has not been applied to `safety.KillSwitchActiveError`.

    Currently latent rather than active: `ExecutionSafety.check_kill_switch` is
    only called inside `validate_order`, which converts every gate exception
    into `passed=False`, so nothing propagates today.  It becomes live the
    moment any kill-switch check moves outside that try block — which is
    exactly what `emergency_flatten_all` already does.
    """
    from app.core import exceptions as core_exc
    from app.execution import order_manager as om_mod
    from app.execution import safety as safety_mod

    # Already fixed — kept so a regression is caught.
    assert issubclass(om_mod.AmbiguousOrderError, core_exc.AmbiguousOrderStateError)

    assert issubclass(safety_mod.KillSwitchActiveError, core_exc.KillSwitchActiveError), (
        "service.py:555 catches app.core.exceptions.KillSwitchActiveError, but "
        "ExecutionSafety and OrderManager raise "
        "app.execution.safety.KillSwitchActiveError, which is not a subclass of "
        "it — the handler is dead code"
    )


async def test_BUG_a_synchronous_fill_is_never_written_back_to_the_local_record():
    """
    BUG 4 (SEVERITY: HIGH - the execution path manufactured the very mismatch
    the reconciliation engine exists to detect).

    STATUS: FIXED. The submit path now books the synchronous fill back into the
    durable OrderRecord and the local position store.

    WHERE
        app/execution/order_manager.py:646-652  transitions to ACKNOWLEDGED and
                                                returns
        app/execution/service.py:590-598        records SUBMITTED and returns

    The paper broker fills synchronously inside `place_order`, so by the time
    the manager writes ACKNOWLEDGED the order is already COMPLETE at the broker
    with a non-zero filled quantity.  Nothing in the submit path calls
    `handle_fill` or `monitor_order`, so the durable OrderRecord keeps
    `filled_qty = 0` and `avg_fill_price = 0.0` for an order that traded, and
    `InMemoryOrderStore.apply_position_delta` is never called — the local
    position stays at zero.

    The next reconciliation pass therefore sees a broker fill and a broker
    position with no local counterpart (`missing_local`) and blocks trading.

    REPRO
        One market order that fills in full.
    """
    stack = await build_stack()
    await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    broker_order = (await stack.broker.get_orders())[0]
    assert broker_order["status"] == OrderStatus.COMPLETE
    assert broker_order["filled_qty"] == 10

    rec = (await stack.local_records())[0]
    assert rec.filled_qty == 10, (
        f"the broker filled {broker_order['filled_qty']} but the durable order "
        f"record says {rec.filled_qty}"
    )
    assert rec.state is OrderState.FILLED, (
        f"the durable record is {rec.state.value} for an order the broker COMPLETEd"
    )
    assert rec.avg_fill_price > 0

    local_pos = await stack.order_store.get_position(SYMBOL, "NSE", Product.MIS.value)
    assert local_pos["quantity"] == 10, (
        "the local position store never learned about the fill"
    )


async def test_BUG_lookup_decline_reason_never_returns_a_reason():
    """
    BUG 5 (SEVERITY: MEDIUM - the fix for a false rejection cause was itself
    broken, and its own exception handler hid that).

    STATUS: FIXED. `_lookup_decline_reason` now awaits the store.

    WHERE
        app/execution/service.py  `record = store.get(client_order_id)` inside
                                  the SYNC method `_lookup_decline_reason`
        app/execution/lifecycle.py  `async def get(self, client_order_id)`
        app/execution/service.py  `except Exception: return None, None`

    `_lookup_decline_reason` is a SYNC method calling an ASYNC store API.
    `store.get()` returns a coroutine, which is not None, so the `record is
    None` guard does not fire; `getattr(coroutine, "reason", None)` is None, so
    the function returns `(None, None)` every time.  Every declined order is
    therefore audited with the generic fallback "…recorded no reason; see
    order_manager logs", `failed_risk_checks` stays None, and the coroutine is
    never awaited — leaking it and emitting a RuntimeWarning on every rejection.

    The bare `except Exception` around the call is what keeps this invisible:
    the method is documented as the fix for reporting a false cause, and it
    silently reports no cause at all.

    REPRO
        Any risk rejection.  The persisted record names the gate; the audit
        record does not.
    """
    stack = await build_stack()
    result = await stack.service.submit_signal(
        signal(), 10, **{**WIDE_RISK, "max_daily_risk": 1.0})

    assert (await stack.facts()).order_count == 0        # fail-closed still holds
    rec = (await stack.local_records())[0]
    assert any("risk_limit" in r for r in rejection_reasons(rec)), (
        "the gate that fired should be on the durable record"
    )

    # `_lookup_decline_reason` is now async — that WAS the bug this test found.
    # It called the async `OrderStore.get()` without awaiting, so it inspected a
    # coroutine object (truthy, no `.reason`) and could never return a cause.
    # The fix made it async and awaits the store, so the call site awaits it
    # too. Every assertion below is unchanged: a real reason, real gate names,
    # and no generic fallback text are all still required.
    reason, checks = await stack.service._lookup_decline_reason(rec.client_order_id)
    assert reason is not None, (
        "_lookup_decline_reason returned no reason even though the durable "
        "record names the gate that fired"
    )
    assert checks, "the failing gate names were not recovered"
    assert "recorded no reason" not in (result.reason or ""), (
        f"the audit journal names no gate: {result.reason!r}"
    )
    assert stack.last_audit.failed_risk_checks, (
        "failed_risk_checks is never populated for a risk rejection"
    )


def test_BUG_paper_broker_PARTIAL_status_is_unclassifiable_by_the_rest_of_the_stack():
    """
    BUG 6 (SEVERITY: MEDIUM - a partial fill blocked trading on a false alarm).

    STATUS: FIXED. "PARTIAL" is now in the reconciliation vocabulary and maps
    to PARTIALLY_FILLED in the lifecycle.

    WHERE
        app/broker/paper.py            OrderStatus.PARTIAL = "PARTIAL"
        app/execution/reconciliation.py  OPEN_STATUSES / TERMINAL_STATUSES
        app/execution/reconciliation.py  classify_order_status
        app/execution/lifecycle.py       _BROKER_STATUS_MAP

    "PARTIAL" appears in neither vocabulary, so `classify_order_status` returns
    "unknown" and `identify_unknown_orders` raises an UNKNOWN_ORDER_STATE
    discrepancy for every partially filled paper order.  `map_broker_status`
    likewise returns OrderState.UNKNOWN — a BLOCKED state that stops all
    further work on the order, including cancelling it.  Kite never emits
    "PARTIAL" (a partial fill stays "OPEN" with filled_quantity < quantity), so
    this is a paper-only status the rest of the stack cannot read.
    """
    from app.execution.lifecycle import map_broker_status
    from app.execution.reconciliation import classify_order_status

    assert classify_order_status(OrderStatus.PARTIAL) in ("open", "terminal"), (
        f"the paper broker's {OrderStatus.PARTIAL!r} status classifies as "
        f"{classify_order_status(OrderStatus.PARTIAL)!r}, which raises a false "
        f"UNKNOWN_ORDER_STATE discrepancy and blocks trading"
    )
    assert map_broker_status(OrderStatus.PARTIAL, filled_qty=100, order_qty=500) is not (
        OrderState.UNKNOWN
    ), "a partially filled paper order maps to the BLOCKED lifecycle state UNKNOWN"


async def test_BUG_a_partially_filled_paper_order_blocks_the_next_reconciliation():
    """
    BUG 6, consequence.  One partial fill is enough to halt the process on the
    next reconciliation pass, with a discrepancy that describes nothing real.
    """
    stack = await build_stack(initial_cash=100_000_000.0, volume=1_000,
                              local_state=InMemoryLocalState(cash=100_000_000.0),
                              activate_kill_switch_on=NEVER_LATCH)
    await stack.service.submit_signal(signal(), 500, **WIDE_RISK)
    assert (await stack.broker.get_orders())[0]["status"] == OrderStatus.PARTIAL

    result = await stack.engine.evaluate(stack.broker)
    unknown = [d for d in result.discrepancies
               if d.kind.value == "unknown_order_state"]
    assert unknown == [], (
        f"a partially filled paper order was reported as being in an unknown "
        f"state: {[str(d) for d in unknown]}"
    )


async def test_REGRESSION_the_wired_path_can_place_a_first_order_in_a_fresh_symbol():
    """
    REGRESSION for PB-STALE (was: CRITICAL - the wired path could not trade).

    STATUS: FIXED.

    WHERE
        app/broker/paper.py:535-548  `is_stale_tick` returns True for any symbol
                                     absent from `self._last_quote_age`
        app/broker/paper.py:806,827  the ONLY two writes to `_last_quote_age`,
                                     both inside `_snapshot`
        app/broker/paper.py:770-771  `get_quote` delegates to the data broker
                                     WITHOUT touching the cache
        app/execution/safety.py      `check_data_freshness` probes
                                     `is_stale_tick` for EVERY order
        app/execution/order_manager.py  `resolve_reference_price` probes it
                                     again for MARKET / SL-M orders

    `_snapshot` is reachable only from `place_order`, `poll_open_orders` and
    `_mark`.  `place_order` is never reached, because the freshness gate runs
    first and refuses the order; `poll_open_orders` has no resting order to
    poll; `_mark` only visits symbols that already have a position.  So:

        first order in symbol S
          -> check_data_freshness -> is_stale_tick(S) -> S not in cache -> True
          -> StaleDataError -> order refused -> place_order never called
          -> _snapshot never called -> S still absent from the cache
          -> the next order is refused for exactly the same reason.

    Every symbol is permanently un-tradeable.  The system fails CLOSED, so no
    money is at risk - but the single authoritative order path cannot place a
    single order, which makes every "the gates would have stopped this"
    guarantee untestable in production.

    `app/main.py` compounds it: `build_execution_stack()` is called with no
    `data_broker`, so the paper broker has no feed to snapshot even if the
    cache could be primed.

    REPRO
        A fully reconciled stack, an open market, a fresh quote, ample cash and
        a wide risk budget - and a symbol not previously snapshot.

    THE FIX
        `PaperBroker.get_quote` now records quote age -- a quote observation IS
        a freshness observation -- and the order path fetches the quote BEFORE
        judging its age, because freshness is a property of data you have
        actually fetched. Checking first created a deadlock: stale -> refused
        -> never fetched -> still stale.

        Genuine staleness is still refused; see
        `test_16_stale_market_data_produces_no_order`, which drives a
        90-minute-old quote through this same path.
    """
    stack = await build_stack(prime=())            # nothing pre-primed
    before = await stack.facts()

    # Everything else is in order.
    assert stack.broker.is_market_open() is True
    assert stack.broker.is_connected is True
    assert stack.recovery.trading_permitted is True
    quotes = await stack.broker.get_quote([f"NSE:{SYMBOL}"])
    assert quotes[f"NSE:{SYMBOL}"]["last_price"] == PRICE, "the feed is quoting"
    assert stack.broker.is_stale_tick(SYMBOL) is False, (
        "a symbol quoted a line ago must be reported as fresh: "
        "PaperBroker.get_quote records quote age"
    )

    result = await stack.service.submit_signal(signal(), 10, **WIDE_RISK)

    after = await stack.facts()
    assert result.outcome is ExecutionOutcome.SUBMITTED, (
        f"the first order in a fresh symbol was refused "
        f"({result.outcome.value}): {result.reason}. The wired path must be "
        f"able to open a position in a symbol it has not previously quoted."
    )
    assert after.order_count == before.order_count + 1
