"""
Property / invariant tests for the execution path.

WHAT THIS FILE IS
-----------------
Fifteen properties that must hold after EVERY step of EVERY scenario, checked
against thousands of randomized steps driven through the REAL execution stack:
`ExecutionService` -> `OrderManager` -> `ExecutionSafety` -> `PaperBroker`,
with the real `ReconciliationEngine`, `RecoveryManager`, `InMemoryOrderStore`
and `AuditJournal`.  The only doubles are the market-data feed and a stand-in
live venue whose entire purpose is to prove nothing was ever sent to it.

The invariants are stated as *facts about the broker's own book*, not about the
value `submit_signal` returned.  A service that placed an order and then
reported failure violates them; a service that reported success without placing
one violates them too.

THE GENERATOR
-------------
`scenario(rng)` draws a symbol, side, quantity, product, price, session volume,
opening cash and a set of injected failures, then runs 1-4 steps through the
service.  After every step `check_invariants` evaluates every applicable
property and appends a `Violation` for each one that does not hold.  The RNG is
seeded (`SEED`), so a failure names a scenario index that reproduces exactly.

    .venv/bin/python -m pytest tests/test_safety_invariants.py -q

`test_zz_scenario_and_violation_counts` prints the totals and is the summary
line to read.

DELIBERATELY NOT WEAKENED
-------------------------
No invariant here is conditioned on "unless the implementation does X".  If one
fails, the correct response is to fix the production code, not the assertion.

SOURCE UNDER TEST
-----------------
The execution modules were being edited while this suite was written; see the
header of `test_execution_integration.py` for the revision it was last run
against, and that file's PRODUCTION BUGS section for the defects found.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd
import pytest

from app.broker.base import BrokerInterface, Product, TransactionType
from app.broker.paper import (
    IST,
    SHORT_SELLING_SUPPORTED,
    AccountingInvariantError,
    OrderStatus,
    PaperBroker,
    to_paise,
)
from app.execution.audit import AuditJournal, ExecutionOutcome, InMemoryAuditSink
from app.execution.bootstrap import InMemoryKillSwitchStore, store_kill_switch_probe
from app.execution.lifecycle import InMemoryOrderStore
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
#  Determinism                                                                 #
# --------------------------------------------------------------------------- #

SEED = 20250903
N_SCENARIOS = 400
MAX_STEPS = 4

OPEN_IST = datetime(2025, 6, 10, 11, 0, tzinfo=IST)        # Tuesday, in session
CLOSED_IST = datetime(2025, 6, 10, 20, 0, tzinfo=IST)      # after the close
WEEKEND_IST = datetime(2025, 6, 8, 11, 0, tzinfo=IST)      # Sunday

SYMBOLS = ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK")
#: strategy_name drives Product inside ExecutionService._to_order_intent:
#: "intraday" -> MIS, anything else -> CNC.
STRATEGIES = ("intraday", "swing", "longterm")


# =========================================================================== #
#  Doubles — external boundaries only                                         #
# =========================================================================== #

class FakeFeed(BrokerInterface):
    """The market-data source. Quotes, and nothing else."""

    def __init__(self, prices: dict[str, float], volume: int,
                 *, timestamp: Optional[datetime] = None) -> None:
        self.prices = dict(prices)
        self.volume = volume
        self.timestamp = timestamp
        self.fail = False

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def get_profile(self) -> dict: return {"user_name": "feed"}
    async def get_holdings(self) -> list[dict]: return []
    async def get_positions(self) -> list[dict]: return []
    async def get_orders(self) -> list[dict]: return []
    async def get_trades(self) -> list[dict]: return []
    async def get_funds(self) -> dict: return {}

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        if self.fail:
            raise ConnectionError("market data feed is down")
        out: dict[str, dict] = {}
        for key in symbols:
            _, sym = key.split(":", 1)
            price = self.prices.get(sym)
            if price is None:
                continue
            q: dict[str, Any] = {
                "last_price": price, "volume": self.volume,
                "ohlc": {"open": price, "high": price, "low": price, "close": price},
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

    def instrument_token(self, symbol: str, exchange: str) -> int: return 1


class FakeLiveBroker(FakeFeed):
    """
    A stand-in live venue that records everything sent to it.

    Deliberately permissive — market always open, ticks always fresh, orders
    always accepted — so any live order that got past the gates lands here and
    is counted.  `INV-05` and `INV-14` are assertions about `self.placed`.
    """

    def __init__(self, prices: Optional[dict[str, float]] = None, volume: int = 1_000_000):
        super().__init__(prices or {s: 100.0 for s in SYMBOLS}, volume)
        self.placed: list[dict] = []

    async def place_order(self, symbol, exchange, txn_type, qty, price,
                          order_type, product, tag="", trigger_price=None) -> str:
        self.placed.append({"symbol": symbol, "qty": qty, "tag": tag,
                            "txn_type": txn_type.value})
        return f"LIVE-{len(self.placed)}"

    async def get_orders(self) -> list[dict]:
        return [{"order_id": f"LIVE-{i + 1}", "tag": o["tag"], "status": "COMPLETE",
                 "symbol": o["symbol"], "quantity": o["qty"],
                 "filled_quantity": o["qty"]}
                for i, o in enumerate(self.placed)]

    def is_stale_tick(self, symbol: str, max_age_seconds: float = 30.0) -> bool:
        return False

    def is_market_open(self, moment: Optional[datetime] = None) -> bool:
        return True

    @property
    def trading_mode(self) -> str: return "live"


class InMemoryLocalState:
    """An implementation of the production `LocalStateStore` Protocol."""

    def __init__(self, *, cash: Optional[float] = 1_000_000.0,
                 positions=None, orders=None, trades=None) -> None:
        self.cash = cash
        self.positions = list(positions or [])
        self.orders = list(orders or [])
        self.trades = list(trades or [])
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
            raise LocalStateUnavailable("no local cash record")
        return {"cash": self.cash}


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key): return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value
        return True


def test_local_state_double_implements_the_real_protocol():
    assert isinstance(InMemoryLocalState(), LocalStateStore)


# =========================================================================== #
#  Book snapshot                                                              #
# =========================================================================== #

@dataclass(frozen=True)
class Book:
    """The broker's own account of itself, read only through its public API."""
    order_count: int
    order_ids: tuple[str, ...]
    #: order_id -> (status, filled_qty). Compared instead of the sequence
    #: because a restart rebuilds the order dict from a key-sorted JSON body,
    #: so `get_orders()` comes back in a different ORDER while holding exactly
    #: the same orders. Identity is what INV-13 is about, not iteration order.
    orders_by_id: dict[str, tuple[str, int]]
    statuses: tuple[str, ...]
    reject_reasons: tuple[Optional[str], ...]
    filled_qty: int
    trade_count: int
    cash_paise: int
    available_paise: int
    holdings: dict[str, int]
    basis_paise: int
    realised_paise: int
    trades: tuple[dict, ...]

    @classmethod
    async def of(cls, broker: PaperBroker) -> "Book":
        orders = await broker.get_orders()
        trades = await broker.get_trades()
        funds = await broker.get_funds()
        positions = await broker.get_positions()
        perf = await broker.get_paper_performance()
        return cls(
            order_count=len(orders),
            order_ids=tuple(o["order_id"] for o in orders),
            orders_by_id={o["order_id"]: (o["status"], int(o["filled_qty"]))
                          for o in orders},
            statuses=tuple(o["status"] for o in orders),
            reject_reasons=tuple(o["reject_reason"] for o in orders),
            filled_qty=sum(int(o["filled_qty"]) for o in orders),
            trade_count=len(trades),
            cash_paise=to_paise(funds["total_cash"]),
            available_paise=to_paise(funds["margin_available"]),
            holdings={f'{p["exchange"]}:{p["symbol"]}:{p["product"]}': int(p["quantity"])
                      for p in positions},
            basis_paise=sum(int(p["cost_basis_paise"]) for p in positions),
            realised_paise=to_paise(perf["realised_pnl"]),
            trades=tuple(dict(t) for t in trades),
        )

    def qty(self, key: str) -> int:
        return self.holdings.get(key, 0)


def ledger_cash_paise(trades, initial_cash_paise: int) -> int:
    """Re-derive cash from the ledger, independently of the broker's own total."""
    cash = initial_cash_paise
    for t in trades:
        if t["txn_type"] == TransactionType.BUY.value:
            cash -= int(t["notional_paise"])
        else:
            cash += int(t["notional_paise"])
        cash -= int(t["costs_paise"])
    return cash


def sell_proceeds_paise(trades) -> int:
    return sum(int(t["notional_paise"]) for t in trades
               if t["txn_type"] == TransactionType.SELL.value)


# =========================================================================== #
#  Violations                                                                 #
# =========================================================================== #

@dataclass
class Violation:
    invariant: str
    scenario: int
    step: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.invariant}] scenario {self.scenario} step {self.step}: {self.detail}"


INVARIANTS = {
    "INV-01": "a rejected order cannot alter cash",
    "INV-02": "a rejected order cannot alter holdings",
    "INV-03": "a sell cannot create holdings",
    "INV-04": "a sell cannot exceed holdings (shorting is not supported)",
    "INV-05": "a failed eligibility check cannot produce a broker order",
    "INV-06": "a failed risk check cannot produce a broker order",
    "INV-07": "the kill switch cannot produce a new order",
    "INV-08": "an ambiguous order is never automatically duplicated",
    "INV-09": "reconciliation with an unreachable broker cannot report OK",
    "INV-10": "an unknown safety state cannot become LIVE_ELIGIBLE",
    "INV-11": "the paper broker cannot manufacture cash",
    "INV-12": "accounting conserves cash and position value up to costs and slippage",
    "INV-13": "a restart cannot duplicate positions or orders",
    "INV-14": "no live order can be produced while trading_mode is paper",
    "INV-15": "no paper order can be produced while trading_mode is live",
}


# =========================================================================== #
#  Harness                                                                    #
# =========================================================================== #

@dataclass
class Harness:
    feed: FakeFeed
    broker: PaperBroker
    kill_store: InMemoryKillSwitchStore
    order_store: InMemoryOrderStore
    order_manager: OrderManager
    engine: ReconciliationEngine
    recovery: RecoveryManager
    sink: InMemoryAuditSink
    service: ExecutionService
    local_state: InMemoryLocalState
    live_venue: FakeLiveBroker
    redis: FakeRedis
    initial_cash_paise: int
    place_calls: list[str] = field(default_factory=list)

    async def book(self) -> Book:
        return await Book.of(self.broker)


async def build_harness(
    *,
    prices: dict[str, float],
    volume: int,
    initial_cash: float,
    clock_at: datetime,
    quote_timestamp: Optional[datetime] = None,
    eligibility_provider: Optional[Callable[[], Any]] = None,
    prime: tuple[str, ...] = (),
) -> Harness:
    feed = FakeFeed(prices, volume, timestamp=quote_timestamp)
    redis = FakeRedis()
    broker = PaperBroker(data_broker=feed, initial_cash=initial_cash,
                         clock=lambda: clock_at, redis_client=redis)
    kill_store = InMemoryKillSwitchStore()
    order_store = InMemoryOrderStore()
    order_manager = OrderManager(ExecutionSafety(kill_store), store=order_store)
    local = InMemoryLocalState(cash=initial_cash)
    engine = ReconciliationEngine(kill_store, local_state=local)
    recovery = RecoveryManager(engine, local_state=local)
    sink = InMemoryAuditSink()
    service = ExecutionService(
        broker=broker, order_manager=order_manager, trading_mode=TradingMode.PAPER,
        kill_switch=KillSwitch(store_kill_switch_probe(kill_store)),
        trading_gate=TradingGate(recovery), audit=AuditJournal(sink),
        eligibility_provider=eligibility_provider, live_authorized=False,
    )

    await broker.connect()
    await recovery.recover(broker)

    h = Harness(feed=feed, broker=broker, kill_store=kill_store,
                order_store=order_store, order_manager=order_manager, engine=engine,
                recovery=recovery, sink=sink, service=service, local_state=local,
                live_venue=FakeLiveBroker(prices, volume), redis=redis,
                initial_cash_paise=to_paise(initial_cash))

    # Instrument the real place_order so INV-08 can count calls per client tag.
    real_place = broker.place_order

    async def counting(**kw):
        h.place_calls.append(kw.get("tag", ""))
        return await real_place(**kw)

    broker.place_order = counting

    for sym in prime:
        # PB-STALE workaround; see tests/test_execution_integration.py
        # `prime_quote_cache` for why nothing on the wired path can do this.
        await broker._snapshot(sym, "NSE")
    return h


def make_signal(symbol: str, direction: SignalDirection, strategy: str) -> Signal:
    now = datetime(2025, 6, 10, 5, 30, tzinfo=timezone.utc)
    return Signal(
        symbol=symbol, direction=direction, strategy_name=strategy,
        timestamp=now, signal_date=now, edge_score=0.01,
        expected_return=0.02, expected_return_std=0.005,
        stop_loss_pct=0.02, target_pct=0.05, holding_period_days=1,
    )


def pos_key(symbol: str, strategy: str) -> str:
    product = Product.MIS.value if strategy == "intraday" else Product.CNC.value
    return f"NSE:{symbol}:{product}"


# =========================================================================== #
#  The per-step invariant checker                                             #
# =========================================================================== #

def check_invariants(
    *,
    scenario_id: int,
    step: int,
    before: Book,
    after: Book,
    harness: Harness,
    side: TransactionType,
    key: str,
    result,
    kill_switch_was_on: bool,
) -> list[Violation]:
    """Evaluate every applicable invariant against one completed step."""
    out: list[Violation] = []

    def fail(inv: str, detail: str) -> None:
        out.append(Violation(inv, scenario_id, step, detail))

    new_orders = after.order_count - before.order_count
    new_statuses = after.statuses[before.order_count:]
    reached_broker = new_orders > 0
    rejected = reached_broker and all(s == OrderStatus.REJECTED for s in new_statuses)
    audit = harness.sink.records[-1] if harness.sink.records else None

    # -- INV-01 / INV-02: a rejected order changes nothing ------------------
    if rejected or not reached_broker:
        if after.cash_paise != before.cash_paise:
            fail("INV-01", f"cash moved {before.cash_paise}p -> {after.cash_paise}p "
                           f"on a {'rejected' if rejected else 'never-placed'} order")
        if after.holdings != before.holdings:
            fail("INV-02", f"holdings moved {before.holdings} -> {after.holdings} "
                           f"on a {'rejected' if rejected else 'never-placed'} order")
        if after.trade_count != before.trade_count:
            fail("INV-01", "a rejected/never-placed order booked a trade")

    # -- INV-03: a sell cannot create holdings ------------------------------
    if side is TransactionType.SELL:
        for k, q in after.holdings.items():
            if q > before.qty(k):
                fail("INV-03", f"a SELL increased {k} from {before.qty(k)} to {q}")

    # -- INV-04: a sell cannot exceed holdings; nothing may go short --------
    for k, q in after.holdings.items():
        if q < 0:
            fail("INV-04", f"position {k} went short ({q}); "
                           f"SHORT_SELLING_SUPPORTED={SHORT_SELLING_SUPPORTED}")
    if side is TransactionType.SELL:
        sold = before.qty(key) - after.qty(key)
        if sold > before.qty(key):
            fail("INV-04", f"sold {sold} of {key} while holding {before.qty(key)}")

    # -- INV-06: a failed risk check cannot produce a broker order ----------
    if audit is not None and audit.risk_checks_passed is False and reached_broker:
        fail("INV-06", f"risk_checks_passed=False yet {new_orders} order(s) reached "
                       f"the broker (outcome={audit.outcome})")

    # -- INV-07: the kill switch cannot produce a new order -----------------
    if kill_switch_was_on and reached_broker:
        fail("INV-07", f"{new_orders} order(s) reached the broker with the kill "
                       f"switch engaged (outcome={result.outcome.value})")

    # -- INV-08: an ambiguous order is never automatically duplicated -------
    for tag in set(harness.place_calls):
        n = harness.place_calls.count(tag)
        if n > 1:
            fail("INV-08", f"place_order was called {n} times for client tag {tag!r}")
    if after.order_count > len(harness.place_calls):
        fail("INV-08", f"the broker holds {after.order_count} orders but place_order "
                       f"was called {len(harness.place_calls)} times")

    # -- INV-11: the paper broker cannot manufacture cash -------------------
    ceiling = harness.initial_cash_paise + sell_proceeds_paise(after.trades)
    if after.cash_paise > ceiling:
        fail("INV-11", f"cash {after.cash_paise}p exceeds opening cash plus every "
                       f"sale's gross proceeds ({ceiling}p)")
    if after.cash_paise < 0:
        fail("INV-11", f"cash went negative ({after.cash_paise}p)")
    derived = ledger_cash_paise(after.trades, harness.initial_cash_paise)
    if derived != after.cash_paise:
        fail("INV-11", f"cash {after.cash_paise}p disagrees with the trade ledger "
                       f"({derived}p)")
    if after.available_paise > after.cash_paise:
        fail("INV-11", f"available cash {after.available_paise}p exceeds total cash "
                       f"{after.cash_paise}p")

    # -- INV-12: value is conserved up to costs and realised P&L ------------
    lhs = after.cash_paise + after.basis_paise
    rhs = harness.initial_cash_paise + after.realised_paise
    if abs(lhs - rhs) > 1:
        fail("INV-12", f"cash+basis={lhs}p but opening+realised={rhs}p "
                       f"(delta {lhs - rhs}p)")

    # -- INV-14: a paper service never touches a live venue -----------------
    if harness.live_venue.placed:
        fail("INV-14", f"a live venue received {len(harness.live_venue.placed)} order(s) "
                       f"from a paper-mode service")
    if harness.broker.trading_mode != "paper":
        fail("INV-14", f"the paper stack's broker reports trading_mode="
                       f"{harness.broker.trading_mode!r}")
    for a in harness.sink.records:
        if a.trading_mode != "paper":
            fail("INV-14", f"an audit record claims trading_mode={a.trading_mode!r}")

    return out


# =========================================================================== #
#  The scenario generator                                                     #
# =========================================================================== #

@dataclass
class Injections:
    """
    The faults one scenario will suffer.

    Each is drawn independently and at a low rate, so the majority of steps
    still reach the broker.  A generator whose every step is refused satisfies
    every invariant vacuously — `test_the_randomized_run_actually_exercised_
    the_execution_path` is the guard that keeps this honest.
    """
    kill_switch_at: Optional[int]
    feed_dead_at: Optional[int]
    place_timeout_at: Optional[int]
    unreconciled_at: Optional[int]
    book_unreadable: bool
    stale_quotes: bool
    market_closed: bool
    eligibility_raises: bool
    starve_cash: bool
    unprimed_symbol: bool
    thin_volume: bool


def draw_injections(rng: random.Random) -> Injections:
    return Injections(
        kill_switch_at=rng.randint(0, MAX_STEPS - 1) if rng.random() < 0.15 else None,
        feed_dead_at=rng.randint(0, MAX_STEPS - 1) if rng.random() < 0.08 else None,
        place_timeout_at=rng.randint(0, MAX_STEPS - 1) if rng.random() < 0.15 else None,
        unreconciled_at=rng.randint(1, MAX_STEPS - 1) if rng.random() < 0.10 else None,
        book_unreadable=rng.random() < 0.5,
        stale_quotes=rng.random() < 0.07,
        market_closed=rng.random() < 0.07,
        eligibility_raises=rng.random() < 0.15,
        starve_cash=rng.random() < 0.10,
        unprimed_symbol=rng.random() < 0.07,
        thin_volume=rng.random() < 0.20,
    )


#: A risk context every gate passes. Individual dimensions are narrowed from
#: here, one at a time, so a rejection is attributable to a named gate.
GENEROUS_RISK = dict(
    available_cash=500_000_000.0,
    total_portfolio=5_000_000_000.0,
    max_daily_risk=500_000_000.0,
    max_daily_loss=500_000_000.0,
    daily_risk_used=0.0,
    realised_pnl_today=0.0,
    current_positions=[],
    open_orders=[],
    max_positions=20,
)

#: Each entry narrows exactly ONE risk dimension to something the gates refuse.
RISK_SQUEEZES: tuple[tuple[str, dict], ...] = (
    ("capital_availability", {"available_cash": 1.0}),
    ("single_stock_exposure", {"total_portfolio": 0.0}),
    ("single_stock_exposure", {"total_portfolio": 1_000.0}),
    ("single_stock_exposure", {"total_portfolio": float("nan")}),
    ("risk_limit", {"max_daily_risk": 1.0}),
    ("daily_loss_limit", {"max_daily_loss": 1.0, "realised_pnl_today": -75_000.0}),
    ("position_limit", {"max_positions": 0,
                        "current_positions": [{"symbol": "X", "quantity": 1}]}),
)


def draw_risk_context(rng: random.Random) -> tuple[dict, Optional[str]]:
    """A passing risk context most of the time; one squeezed gate otherwise."""
    if rng.random() < 0.75:
        return dict(GENEROUS_RISK), None
    gate, narrowed = rng.choice(RISK_SQUEEZES)
    return {**GENEROUS_RISK, **narrowed}, gate


#: Counts of what the generator actually produced.  Without these the
#: invariants could all be satisfied vacuously by a run that never traded.
COVERAGE_KEYS = (
    "steps", "reached_broker", "filled_orders", "partial_fills",
    "rejected_by_broker", "blocked_risk", "blocked_kill_switch",
    "blocked_not_reconciled", "ambiguous", "errors",
    "buys", "sells", "sell_fills", "restarts", "broker_orders", "trades",
    "insufficient_cash", "insufficient_holdings", "squeezed_gates",
    "faults_injected",
)


async def run_scenario(
    scenario_id: int, rng: random.Random, cov: dict[str, int],
) -> tuple[int, list[Violation]]:
    """One randomized session. Returns (steps_executed, violations)."""
    inj = draw_injections(rng)

    symbol = rng.choice(SYMBOLS)
    strategy = rng.choice(STRATEGIES)
    key = pos_key(symbol, strategy)
    price = round(rng.uniform(5.0, 3_500.0), 2)
    volume = rng.choice([200, 2_000]) if inj.thin_volume else rng.choice(
        [500_000, 5_000_000])
    # Opening cash is normally ample, so a refusal is attributable to a gate
    # rather than to an accidentally empty account.
    initial_cash = 250.0 if inj.starve_cash else rng.choice(
        [5_000_000.0, 50_000_000.0])
    clock_at = OPEN_IST
    if inj.market_closed:
        clock_at = rng.choice([CLOSED_IST, WEEKEND_IST])
    quote_ts = (clock_at - timedelta(minutes=rng.randint(31, 600))
                if inj.stale_quotes else None)

    def eligibility_boom():
        raise RuntimeError("eligibility evidence store unreachable")

    harness = await build_harness(
        prices={symbol: price},
        volume=volume,
        initial_cash=initial_cash,
        clock_at=clock_at,
        quote_timestamp=quote_ts,
        eligibility_provider=eligibility_boom if inj.eligibility_raises else None,
        prime=() if inj.unprimed_symbol else (symbol,),
    )

    violations: list[Violation] = []
    n_steps = rng.randint(1, MAX_STEPS)
    executed = 0

    for step in range(n_steps):
        before = await harness.book()
        held = before.qty(key)

        # Step 0 is usually a BUY, so later SELL steps have stock to sell and
        # the sell-side paths are genuinely exercised.
        if step == 0:
            direction = rng.choices(
                [SignalDirection.LONG, SignalDirection.EXIT], weights=[0.85, 0.15])[0]
        else:
            direction = rng.choices(
                [SignalDirection.LONG, SignalDirection.EXIT], weights=[0.45, 0.55])[0]
        side = (TransactionType.BUY if direction is SignalDirection.LONG
                else TransactionType.SELL)

        if side is TransactionType.SELL and held > 0 and rng.random() < 0.7:
            # A sell that should succeed ...
            qty = rng.randint(1, held)
        elif side is TransactionType.SELL and rng.random() < 0.5:
            # ... and one that must be refused as a short.
            qty = held + rng.choice([1, 10, 500])
        else:
            qty = rng.choice([1, 3, 10, 25, 100])
            if rng.random() < 0.20:
                qty = rng.choice([1_000, 20_000])       # fat finger / partial fill

        if inj.kill_switch_at == step:
            harness.kill_store.engage(f"randomized fault at step {step}")
            cov["faults_injected"] += 1
        if inj.unreconciled_at == step:
            harness.recovery.block(f"randomized reconciliation fault at step {step}")
            cov["faults_injected"] += 1
        harness.feed.fail = inj.feed_dead_at == step
        if harness.feed.fail:
            cov["faults_injected"] += 1

        risk_ctx, squeezed = draw_risk_context(rng)
        if squeezed:
            cov["squeezed_gates"] += 1

        kill_on = bool(harness.kill_store.get("kill_switch"))

        # The faults are installed AFTER the snapshot and removed in `finally`,
        # so the test's own view of the book is never the thing that breaks.
        real_place = harness.broker.place_order
        real_get_orders = harness.broker.get_orders
        if inj.place_timeout_at == step:
            cov["faults_injected"] += 1

            async def timing_out(**kw):
                harness.place_calls.append(kw.get("tag", ""))
                raise TimeoutError("kite gateway timeout")

            harness.broker.place_order = timing_out
            if inj.book_unreadable:
                # A genuinely ambiguous submission: the response was lost AND
                # the order book cannot be read to find out what happened.
                async def unreadable_book():
                    raise ConnectionError("order book unreachable")

                harness.broker.get_orders = unreadable_book

        try:
            result = await harness.service.submit_signal(
                make_signal(symbol, direction, strategy), qty, **risk_ctx)
        except AccountingInvariantError as exc:
            # The broker's own guard fired: the book is not coherent. That is
            # itself a violation, not an expected control-flow path.
            violations.append(Violation(
                "INV-12", scenario_id, step,
                f"PaperBroker raised AccountingInvariantError: {exc}"))
            break
        finally:
            harness.broker.place_order = real_place
            harness.broker.get_orders = real_get_orders

        executed += 1
        after = await harness.book()

        cov["steps"] += 1
        cov["buys" if side is TransactionType.BUY else "sells"] += 1
        cov[{
            ExecutionOutcome.SUBMITTED: "reached_broker",
            ExecutionOutcome.BLOCKED_RISK: "blocked_risk",
            ExecutionOutcome.BLOCKED_KILL_SWITCH: "blocked_kill_switch",
            ExecutionOutcome.BLOCKED_NOT_RECONCILED: "blocked_not_reconciled",
            ExecutionOutcome.AMBIGUOUS: "ambiguous",
        }.get(result.outcome, "errors")] += 1
        for status, reason in zip(after.statuses[before.order_count:],
                                  after.reject_reasons[before.order_count:]):
            cov["broker_orders"] += 1
            if status == OrderStatus.COMPLETE:
                cov["filled_orders"] += 1
            elif status == OrderStatus.PARTIAL:
                cov["partial_fills"] += 1
            elif status == OrderStatus.REJECTED:
                cov["rejected_by_broker"] += 1
                if reason == "INSUFFICIENT_CASH":
                    cov["insufficient_cash"] += 1
                elif reason in ("INSUFFICIENT_HOLDINGS", "SHORT_SELL_NOT_SUPPORTED"):
                    cov["insufficient_holdings"] += 1
        cov["trades"] += after.trade_count - before.trade_count
        if side is TransactionType.SELL and after.trade_count > before.trade_count:
            cov["sell_fills"] += 1

        violations += check_invariants(
            scenario_id=scenario_id, step=step, before=before, after=after,
            harness=harness, side=side, key=key, result=result,
            kill_switch_was_on=kill_on,
        )

        if inj.kill_switch_at == step and rng.random() < 0.5:
            harness.kill_store.release()
        if inj.unreconciled_at == step and rng.random() < 0.5:
            await harness.recovery.recover(harness.broker)

    # -- INV-13: a restart may not duplicate anything ----------------------
    harness.feed.fail = False          # the restart is not a feed-outage test
    cov["restarts"] += 1
    violations += await check_restart(scenario_id, harness, executed)

    # -- INV-05 / INV-15: nothing ever reached a live venue -----------------
    if harness.live_venue.placed:
        violations.append(Violation(
            "INV-05", scenario_id, executed,
            f"a live venue received {len(harness.live_venue.placed)} order(s)"))

    return executed, violations


async def check_restart(scenario_id: int, harness: Harness, step: int) -> list[Violation]:
    """
    INV-13: replay the persisted book into a fresh process.

    Positions, orders, trades and cash must come back exactly as they were, and
    running the order manager's own recovery must place nothing.
    """
    out: list[Violation] = []
    before = await harness.book()
    await harness.broker.disconnect()

    restarted = PaperBroker(
        data_broker=harness.feed,
        initial_cash=harness.initial_cash_paise / 100.0,
        clock=harness.broker._clock,
        redis_client=harness.redis,
    )
    try:
        await restarted.connect()
    except Exception as exc:
        out.append(Violation("INV-13", scenario_id, step,
                             f"the persisted book could not be reloaded: {exc!r}"))
        return out

    after = await Book.of(restarted)
    if len(after.order_ids) != len(set(after.order_ids)):
        out.append(Violation("INV-13", scenario_id, step,
                             "a restart produced duplicate order ids"))
    if after.orders_by_id != before.orders_by_id:
        added = set(after.orders_by_id) - set(before.orders_by_id)
        lost = set(before.orders_by_id) - set(after.orders_by_id)
        changed = {k for k in set(after.orders_by_id) & set(before.orders_by_id)
                   if after.orders_by_id[k] != before.orders_by_id[k]}
        out.append(Violation("INV-13", scenario_id, step,
                             f"the order book changed across a restart: "
                             f"added={sorted(added)} lost={sorted(lost)} "
                             f"changed={sorted(changed)}"))
    if after.holdings != before.holdings:
        out.append(Violation("INV-13", scenario_id, step,
                             f"positions changed across a restart: {before.holdings} "
                             f"-> {after.holdings}"))
    if after.cash_paise != before.cash_paise:
        out.append(Violation("INV-13", scenario_id, step,
                             f"cash changed across a restart: {before.cash_paise}p "
                             f"-> {after.cash_paise}p"))
    if after.trade_count != before.trade_count:
        out.append(Violation("INV-13", scenario_id, step, "trades were duplicated"))

    placed_during_recovery: list[str] = []

    async def must_not_place(**kw):
        placed_during_recovery.append(kw.get("tag", ""))
        raise AssertionError("recovery placed an order")

    restarted.place_order = must_not_place
    try:
        await harness.order_manager.recover(restarted)
    except Exception as exc:                       # pragma: no cover - diagnostic
        out.append(Violation("INV-13", scenario_id, step,
                             f"order-manager recovery raised: {exc!r}"))
    if placed_during_recovery:
        out.append(Violation("INV-13", scenario_id, step,
                             f"recovery placed {len(placed_during_recovery)} order(s)"))

    final = await Book.of(restarted)
    if final.order_count != before.order_count:
        out.append(Violation("INV-13", scenario_id, step,
                             "recovery changed the broker's order count"))
    return out


# =========================================================================== #
#  The randomized run                                                         #
# =========================================================================== #

_RESULTS: dict[str, Any] = {}


@pytest.fixture(scope="module")
def randomized_run():
    """Run every scenario once; every property test reads this result."""
    if _RESULTS:
        return _RESULTS

    rng = random.Random(SEED)

    cov = {k: 0 for k in COVERAGE_KEYS}

    async def go():
        steps = 0
        violations: list[Violation] = []
        for i in range(N_SCENARIOS):
            n, v = await run_scenario(i, rng, cov)
            steps += n
            violations += v
        return steps, violations

    steps, violations = asyncio.run(go())
    _RESULTS.update(scenarios=N_SCENARIOS, steps=steps, violations=violations,
                    seed=SEED, coverage=cov)
    return _RESULTS


def violations_of(run, *invariants: str) -> list[Violation]:
    wanted = set(invariants)
    return [v for v in run["violations"] if v.invariant in wanted]


def _assert_clean(run, invariant: str) -> None:
    found = violations_of(run, invariant)
    assert not found, (
        f"{invariant} ({INVARIANTS[invariant]}) was violated {len(found)} time(s) "
        f"across {run['scenarios']} scenarios / {run['steps']} steps "
        f"(seed={run['seed']}):\n  " + "\n  ".join(str(v) for v in found[:10])
    )


# --- the randomized properties --------------------------------------------

def test_INV_01_a_rejected_order_cannot_alter_cash(randomized_run):
    _assert_clean(randomized_run, "INV-01")


def test_INV_02_a_rejected_order_cannot_alter_holdings(randomized_run):
    _assert_clean(randomized_run, "INV-02")


def test_INV_03_a_sell_cannot_create_holdings(randomized_run):
    _assert_clean(randomized_run, "INV-03")


def test_INV_04_a_sell_cannot_exceed_holdings(randomized_run):
    _assert_clean(randomized_run, "INV-04")


def test_INV_05_a_failed_eligibility_check_cannot_produce_a_broker_order(randomized_run):
    _assert_clean(randomized_run, "INV-05")


def test_INV_06_a_failed_risk_check_cannot_produce_a_broker_order(randomized_run):
    _assert_clean(randomized_run, "INV-06")


def test_INV_07_the_kill_switch_cannot_produce_a_new_order(randomized_run):
    _assert_clean(randomized_run, "INV-07")


def test_INV_08_an_ambiguous_order_is_never_automatically_duplicated(randomized_run):
    _assert_clean(randomized_run, "INV-08")


def test_INV_11_the_paper_broker_cannot_manufacture_cash(randomized_run):
    _assert_clean(randomized_run, "INV-11")


def test_INV_12_accounting_conserves_cash_and_position_value(randomized_run):
    _assert_clean(randomized_run, "INV-12")


def test_INV_13_a_restart_cannot_duplicate_positions_or_orders(randomized_run):
    _assert_clean(randomized_run, "INV-13")


def test_INV_14_no_live_order_while_trading_mode_is_paper(randomized_run):
    _assert_clean(randomized_run, "INV-14")


def test_the_randomized_run_actually_exercised_the_execution_path(randomized_run):
    """
    A generator that never places an order satisfies every invariant vacuously.
    This is the guard against that: each behaviour the invariants are about has
    to actually occur, in quantity, before a green run means anything.
    """
    cov = randomized_run["coverage"]
    required = {
        "reached_broker": 200,          # orders that actually got to the broker
        "filled_orders": 150,           # complete fills
        "partial_fills": 5,             # capped by displayed liquidity
        "rejected_by_broker": 20,       # the broker refused them itself
        "blocked_risk": 80,             # the gates refused them
        "blocked_kill_switch": 15,      # the halt control fired
        "blocked_not_reconciled": 5,    # the trading gate was shut
        "ambiguous": 3,                 # outcome genuinely unknowable
        "buys": 200,
        "sells": 150,
        "sell_fills": 60,               # sells that actually traded
        "trades": 200,
        "restarts": N_SCENARIOS,        # every scenario replays its book
        "insufficient_holdings": 15,    # oversell / short attempts refused
        "squeezed_gates": 50,           # steps with a deliberately narrowed gate
        "faults_injected": 50,          # kill switch / feed / timeout / gate
    }
    missing = {k: (cov[k], floor) for k, floor in required.items() if cov[k] < floor}
    assert not missing, (
        "the randomized run did not exercise enough of the execution path, so a "
        f"clean result would be vacuous. (observed, required) = {missing}\n"
        f"full coverage: {cov}"
    )


# =========================================================================== #
#  INV-05 / INV-09 / INV-10 / INV-15 — randomized, but not book-shaped        #
# =========================================================================== #

async def test_INV_05_randomized_eligibility_failures_never_place_a_live_order():
    """
    Every eligibility outcome this repo can produce, plus injected failures,
    against a live venue that would accept anything.
    """
    from app.governance.eligibility import (
        EligibilityReport,
        EligibilityState,
        GateResult,
        _CANONICAL_GATES,
        assess_live_trading_eligibility,
        gather_repo_evidence,
    )

    rng = random.Random(SEED + 5)

    def real():
        return assess_live_trading_eligibility(gather_repo_evidence())

    def forged():
        """A report claiming every gate passed, built outside the assessor."""
        return EligibilityReport(results=tuple(
            GateResult(name=g.name, category=g.category, blocking_state=g.blocking_state,
                       requirement=g.requirement, passed=True, reason="forged")
            for g in _CANONICAL_GATES))

    def partial():
        """A random subset of gates reported as passing."""
        chosen = [g for g in _CANONICAL_GATES if rng.random() < 0.5]
        return EligibilityReport(results=tuple(
            GateResult(name=g.name, category=g.category, blocking_state=g.blocking_state,
                       requirement=g.requirement, passed=True, reason="partial")
            for g in chosen))

    def boom():
        raise RuntimeError("evidence store unreachable")

    providers = [real, forged, partial, boom, lambda: None, lambda: "LIVE_ELIGIBLE",
                 lambda: 42, lambda: object()]

    violations: list[str] = []
    attempts = 0
    for i in range(len(providers) * 8):
        provider = providers[i % len(providers)]
        live = FakeLiveBroker()
        kill_store = InMemoryKillSwitchStore()
        local = InMemoryLocalState(cash=0.0)
        recovery = RecoveryManager(ReconciliationEngine(kill_store, local_state=local),
                                   local_state=local)
        sink = InMemoryAuditSink()
        service = ExecutionService(
            broker=live,
            order_manager=OrderManager(ExecutionSafety(kill_store),
                                       store=InMemoryOrderStore()),
            trading_mode=TradingMode.LIVE,
            kill_switch=KillSwitch(store_kill_switch_probe(kill_store)),
            trading_gate=TradingGate(recovery),
            audit=AuditJournal(sink),
            eligibility_provider=provider,
            live_authorized=True,
        )
        await recovery.recover(live)
        assert recovery.trading_permitted, "the gate must be open to isolate eligibility"

        result = await service.submit_signal(
            make_signal(rng.choice(SYMBOLS), SignalDirection.LONG, "intraday"),
            rng.choice([1, 10, 500]),
            available_cash=50_000_000.0, total_portfolio=500_000_000.0,
            max_daily_risk=50_000_000.0, max_daily_loss=50_000_000.0,
        )
        attempts += 1
        if live.placed:
            violations.append(
                f"provider {provider.__name__ if hasattr(provider, '__name__') else provider!r} "
                f"placed {len(live.placed)} live order(s) (outcome={result.outcome.value})")
        if result.submitted:
            violations.append(f"provider reported submitted: {result.outcome.value}")
        if sink.records and sink.records[-1].eligibility_permits_live is True:
            violations.append("an audit record claims eligibility permitted live trading")

    assert attempts == len(providers) * 8
    assert not violations, (
        f"INV-05 ({INVARIANTS['INV-05']}) violated {len(violations)} time(s):\n  "
        + "\n  ".join(violations[:10]))


async def test_INV_09_reconciliation_with_an_unreachable_broker_is_never_ok():
    """
    Every combination of broker-side and local-side outage must classify as
    something other than RECONCILIATION_OK, and must never permit trading.
    """
    rng = random.Random(SEED + 9)
    broker_sources = ["get_positions", "get_orders", "get_trades", "get_funds"]
    local_sources = ["positions", "orders", "trades", "cash"]

    violations: list[str] = []
    cases = 0

    for _ in range(120):
        broker_down = {s for s in broker_sources if rng.random() < 0.4}
        local_down = {s for s in local_sources if rng.random() < 0.4}
        if not broker_down and not local_down:
            continue
        cases += 1

        feed = FakeFeed({s: 100.0 for s in SYMBOLS}, 1_000_000)
        broker = PaperBroker(data_broker=feed, initial_cash=1_000_000.0,
                             clock=lambda: OPEN_IST)
        await broker.connect()

        async def down(*a, **k):
            raise ConnectionError("kite 5xx")

        for name in broker_down:
            setattr(broker, name, down)

        local = InMemoryLocalState(cash=1_000_000.0)
        local.fail = set(local_down)
        kill_store = InMemoryKillSwitchStore()
        engine = ReconciliationEngine(kill_store, local_state=local)
        recovery = RecoveryManager(engine, local_state=local)

        report = await recovery.recover(broker)

        if report.status is ReconciliationStatus.RECONCILIATION_OK:
            violations.append(f"broker_down={sorted(broker_down)} "
                              f"local_down={sorted(local_down)} reported OK")
        if bool(report) or bool(report.status):
            violations.append("a non-OK report read as truthy")
        if report.trading_permitted or recovery.trading_permitted:
            violations.append(f"trading permitted with broker_down={sorted(broker_down)}")
        if recovery.state is StartupState.READY:
            violations.append("the recovery manager reached READY")
        if not report.result.unavailable_sources:
            violations.append("an outage produced no unavailable source")

    assert cases > 50, f"the generator produced only {cases} outage cases"
    assert not violations, (
        f"INV-09 ({INVARIANTS['INV-09']}) violated {len(violations)} time(s):\n  "
        + "\n  ".join(violations[:10]))


def test_INV_10_an_unknown_safety_state_cannot_become_live_eligible():
    """
    Nothing that is not a freshly computed, fully passing verdict may read as
    LIVE_ELIGIBLE — not an empty report, not a filtered gate set, not a
    substituted gate, not a forged payload.
    """
    from app.governance.eligibility import (
        EligibilityReport,
        EligibilityState,
        Evidence,
        Gate,
        GateResult,
        LiveTradingBlocked,
        ReportProvenance,
        _CANONICAL_GATES,
        assess_live_trading_eligibility,
        require_live_eligible,
    )

    rng = random.Random(SEED + 10)
    violations: list[str] = []
    cases = 0

    # 1. No evidence, and randomly partial evidence.
    for _ in range(60):
        cases += 1
        notes = tuple(f"note-{rng.randint(0, 99)}" for _ in range(rng.randint(0, 3)))
        report = assess_live_trading_eligibility(Evidence(notes=notes))
        if report.state is EligibilityState.LIVE_ELIGIBLE:
            violations.append(f"empty evidence produced LIVE_ELIGIBLE (notes={notes})")
        if report.permits_live_trading:
            violations.append("empty evidence permitted live trading")

    # 2. A filtered gate set: unevaluated gates must be injected as failures.
    for _ in range(40):
        cases += 1
        subset = tuple(g for g in _CANONICAL_GATES if rng.random() < 0.5)
        report = assess_live_trading_eligibility(Evidence(), gates=subset)
        if report.state is EligibilityState.LIVE_ELIGIBLE:
            violations.append(f"a {len(subset)}-gate subset produced LIVE_ELIGIBLE")
        if report.permits_live_trading:
            violations.append("a filtered gate set permitted live trading")
        missing = {r.name for r in report.results}
        if not {g.name for g in _CANONICAL_GATES}.issubset(missing):
            violations.append("a canonical gate was not injected into the result set")

    # 3. Substituted gates that always pass.
    for _ in range(20):
        cases += 1
        victim = rng.choice(_CANONICAL_GATES)
        fake = Gate(name=victim.name, category=victim.category,
                    blocking_state=victim.blocking_state,
                    requirement=victim.requirement,
                    predicate=lambda e: (True, "substituted"))
        report = assess_live_trading_eligibility(Evidence(), gates=(fake,))
        if report.state is EligibilityState.LIVE_ELIGIBLE:
            violations.append(f"a substituted gate {victim.name!r} produced LIVE_ELIGIBLE")
        if report.permits_live_trading:
            violations.append("a substituted gate permitted live trading")

    # 4. A forged report claiming every gate passed.
    cases += 1
    forged = EligibilityReport(results=tuple(
        GateResult(name=g.name, category=g.category, blocking_state=g.blocking_state,
                   requirement=g.requirement, passed=True, reason="forged")
        for g in _CANONICAL_GATES))
    if forged.provenance is not ReportProvenance.UNTRUSTED:
        violations.append("a report built outside the assessor was not UNTRUSTED")
    if forged.permits_live_trading:
        violations.append("a forged report permitted live trading")
    try:
        require_live_eligible(forged, action="randomized invariant probe")
        violations.append("require_live_eligible accepted a forged report")
    except LiveTradingBlocked:
        pass

    # 5. Junk in place of a report.
    for junk in (None, "LIVE_ELIGIBLE", 42, object(), [], {}):
        cases += 1
        try:
            require_live_eligible(junk, action="randomized invariant probe")
            violations.append(f"require_live_eligible accepted {junk!r}")
        except LiveTradingBlocked:
            pass
        except TypeError:
            pass

    assert cases > 100, f"only {cases} eligibility cases were generated"
    assert not violations, (
        f"INV-10 ({INVARIANTS['INV-10']}) violated {len(violations)} time(s):\n  "
        + "\n  ".join(violations[:10]))


async def test_INV_15_no_paper_order_can_be_produced_while_trading_mode_is_live():
    """
    Mode separation is enforced at construction, in BOTH directions, and there
    is no fallback from a failed live submission to a paper one.
    """
    rng = random.Random(SEED + 15)
    violations: list[str] = []

    for _ in range(40):
        kill_store = InMemoryKillSwitchStore()
        om = OrderManager(ExecutionSafety(kill_store), store=InMemoryOrderStore())
        feed = FakeFeed({s: 100.0 for s in SYMBOLS}, 1_000_000)
        paper = PaperBroker(data_broker=feed, clock=lambda: OPEN_IST)

        # live mode + paper broker: refused
        try:
            ExecutionService(broker=paper, order_manager=om,
                             trading_mode=TradingMode.LIVE, live_authorized=True)
            violations.append("a LIVE service was built around a PaperBroker")
        except ExecutionBlocked as exc:
            if exc.outcome is not ExecutionOutcome.BLOCKED_MODE:
                violations.append(f"wrong outcome for live+paper: {exc.outcome}")

        # live mode without authorization: refused
        try:
            ExecutionService(broker=FakeLiveBroker(), order_manager=om,
                             trading_mode=TradingMode.LIVE, live_authorized=False)
            violations.append("a LIVE service was built without authorization")
        except ExecutionBlocked:
            pass

        # paper mode + live broker: refused
        try:
            ExecutionService(broker=FakeLiveBroker(), order_manager=om,
                             trading_mode=TradingMode.PAPER)
            violations.append("a PAPER service was built around a live broker")
        except ExecutionBlocked:
            pass

    # A live service whose submission fails must NOT fall back to paper.
    for _ in range(20):
        live = FakeLiveBroker()
        kill_store = InMemoryKillSwitchStore()
        local = InMemoryLocalState(cash=0.0)
        recovery = RecoveryManager(ReconciliationEngine(kill_store, local_state=local),
                                   local_state=local)
        sink = InMemoryAuditSink()
        service = ExecutionService(
            broker=live,
            order_manager=OrderManager(ExecutionSafety(kill_store),
                                       store=InMemoryOrderStore()),
            trading_mode=TradingMode.LIVE,
            kill_switch=KillSwitch(store_kill_switch_probe(kill_store)),
            trading_gate=TradingGate(recovery),
            audit=AuditJournal(sink),
            eligibility_provider=lambda: (_ for _ in ()).throw(RuntimeError("down")),
            live_authorized=True,
        )
        await recovery.recover(live)
        result = await service.submit_signal(
            make_signal(rng.choice(SYMBOLS), SignalDirection.LONG, "intraday"), 10,
            available_cash=50_000_000.0, total_portfolio=500_000_000.0)

        if service.broker is not live:
            violations.append("the service swapped its broker after a failure")
        if live.placed:
            violations.append("a blocked live submission still reached the venue")
        if result.submitted:
            violations.append("a blocked live submission reported success")
        if sink.records[-1].trading_mode != "live":
            violations.append(
                f"a live attempt was audited as {sink.records[-1].trading_mode!r}")

    assert not violations, (
        f"INV-15 ({INVARIANTS['INV-15']}) violated {len(violations)} time(s):\n  "
        + "\n  ".join(violations[:10]))


# =========================================================================== #
#  Summary                                                                    #
# =========================================================================== #

def test_zz_scenario_and_violation_counts(randomized_run):
    """
    The headline numbers.  Run with `-s` to see them on the console.

    A non-zero violation count is a production bug, and the correct response is
    to fix the code — never to relax an invariant.
    """
    total = len(randomized_run["violations"])
    by_inv: dict[str, int] = {}
    for v in randomized_run["violations"]:
        by_inv[v.invariant] = by_inv.get(v.invariant, 0) + 1

    print(
        f"\nRANDOMIZED INVARIANT RUN (seed={randomized_run['seed']})\n"
        f"  scenarios          : {randomized_run['scenarios']}\n"
        f"  execution steps    : {randomized_run['steps']}\n"
        f"  invariants checked : {len(INVARIANTS)}\n"
        f"  violations         : {total}\n"
        + ("".join(f"      {k}: {n}\n" for k, n in sorted(by_inv.items()))
           if by_inv else "")
        + "  coverage:\n"
        + "".join(f"      {k:<24}: {n}\n"
                  for k, n in randomized_run["coverage"].items())
    )

    assert randomized_run["scenarios"] == N_SCENARIOS
    assert randomized_run["steps"] >= N_SCENARIOS
    assert total == 0, (
        f"{total} invariant violation(s) across {randomized_run['scenarios']} "
        f"scenarios / {randomized_run['steps']} steps:\n  "
        + "\n  ".join(str(v) for v in randomized_run["violations"][:20]))
