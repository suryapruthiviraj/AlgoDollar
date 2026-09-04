"""
Reconciliation + restart-recovery tests.

Every test here drives the REAL production classes —
``app.execution.reconciliation.ReconciliationEngine``,
``app.execution.reconciliation.SqlAlchemyLocalStateStore`` and
``app.execution.recovery.RecoveryManager``.  The only doubles are a broker
connection, a redis, and an in-memory implementation of the *published*
``LocalStateStore`` Protocol.  Nothing in this file re-implements the logic
under test.

The three defects being pinned as fixed:

D1  broker unreachable used to yield ``[] vs [] -> OK`` with no kill switch.
D2  ``_activate_kill_switch`` used to swallow the store error while
    ``reconcile()`` still claimed "Kill switch activated."
D3  trades were keyed by ``order_id``, collapsing partial fills.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.broker.base import BrokerInterface
from app.execution.reconciliation import (
    DiscrepancyKind,
    KillSwitchActivationError,
    LocalStateStore,
    LocalStateUnavailable,
    ReconciliationEngine,
    ReconciliationError,
    ReconciliationResult,
    ReconciliationStatus,
    Severity,
    SqlAlchemyLocalStateStore,
)
from app.execution.recovery import (
    RecoveryManager,
    RecoveryPhase,
    StartupState,
    TradingBlockedError,
)

OK = ReconciliationStatus.RECONCILIATION_OK
MISMATCH = ReconciliationStatus.RECONCILIATION_MISMATCH
UNAVAILABLE = ReconciliationStatus.RECONCILIATION_UNAVAILABLE
ERROR = ReconciliationStatus.RECONCILIATION_ERROR

START_CASH = 1_000_000.0


# =========================================================================== #
#  Test doubles — a broker connection, a redis, a persistent local store      #
# =========================================================================== #

class FakeRedis:
    """Minimal in-memory stand-in for the redis client the code expects."""

    def __init__(self, *, raise_on_set: bool = False, raise_on_get: bool = False,
                 swallow_writes: bool = False) -> None:
        self.store: dict[str, Any] = {}
        self.raise_on_set = raise_on_set
        self.raise_on_get = raise_on_get
        self.swallow_writes = swallow_writes   # set() "succeeds" but persists nothing

    def get(self, key):
        if self.raise_on_get:
            raise ConnectionError("redis down (read)")
        return self.store.get(key)

    def set(self, key, value):
        if self.raise_on_set:
            raise ConnectionError("redis down (write)")
        if self.swallow_writes:
            return True
        self.store[key] = value
        return True


class WriteOnlyStore:
    """A store with no .get() — activation cannot be verified by read-back."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def set(self, key, value):
        self.store[key] = value
        return True


class FakeBroker(BrokerInterface):
    """A stand-in broker connection. Fully configurable, records nothing it fakes."""

    def __init__(
        self,
        *,
        positions: Optional[list[dict]] = None,
        orders: Optional[list[dict]] = None,
        trades: Optional[list[dict]] = None,
        cash: Optional[float] = START_CASH,
        connected: bool = True,
        unreachable: bool = False,
        fail_fetch: Optional[set[str]] = None,
        order_status: Optional[dict[str, dict]] = None,
        fail_order_status: bool = False,
    ) -> None:
        self._positions = positions or []
        self._orders = orders or []
        self._trades = trades or []
        self._cash = cash
        self._connected = connected
        self._unreachable = unreachable
        self._fail_fetch = fail_fetch or set()
        self._order_status = order_status or {}
        self._fail_order_status = fail_order_status
        self.calls: list[str] = []

    def _maybe_fail(self, what: str) -> None:
        self.calls.append(what)
        if self._unreachable or what in self._fail_fetch:
            raise ConnectionError(f"kite 5xx while fetching {what}")

    # --- BrokerInterface ---------------------------------------------------
    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def get_profile(self) -> dict:
        self._maybe_fail("profile")
        return {"user_name": "fake"}

    async def get_holdings(self) -> list[dict]:
        self._maybe_fail("holdings")
        return []

    async def get_positions(self) -> list[dict]:
        self._maybe_fail("positions")
        return [dict(p) for p in self._positions]

    async def get_orders(self) -> list[dict]:
        self._maybe_fail("orders")
        return [dict(o) for o in self._orders]

    async def get_trades(self) -> list[dict]:
        self._maybe_fail("trades")
        return [dict(t) for t in self._trades]

    async def get_funds(self) -> dict:
        self._maybe_fail("funds")
        return {"cash": self._cash}

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        return {s: {"last_price": 100.0} for s in symbols}

    async def get_historical_data(self, symbol, exchange, interval, from_date, to_date):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    async def place_order(self, symbol, exchange, txn_type, qty, price,
                          order_type, product, tag="") -> str:
        raise AssertionError("recovery must never place an order")

    async def cancel_order(self, order_id: str) -> bool:
        raise AssertionError("recovery must never cancel an order by itself")

    async def modify_order(self, order_id, qty=None, price=None) -> bool:
        raise AssertionError("recovery must never modify an order")

    async def get_order_status(self, order_id: str) -> dict:
        self.calls.append(f"order_status:{order_id}")
        if self._fail_order_status:
            raise ConnectionError("kite 5xx while fetching order status")
        return dict(self._order_status.get(order_id, {"status": "UNKNOWN"}))

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def trading_mode(self) -> str:
        return "fake"

    def instrument_token(self, symbol: str, exchange: str) -> int:
        return 12345


class InMemoryLocalState:
    """
    An implementation of the production ``LocalStateStore`` Protocol.

    Stands in for the database.  A single instance is deliberately reused
    across simulated process restarts: it IS the persistent state.
    """

    def __init__(
        self,
        *,
        positions: Optional[list[dict]] = None,
        orders: Optional[list[dict]] = None,
        trades: Optional[list[dict]] = None,
        cash: Optional[float] = START_CASH,
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
            raise LocalStateUnavailable("no local cash record")
        return {"cash": self.cash}


def test_in_memory_double_satisfies_the_published_protocol():
    """The double implements the real Protocol, so these tests exercise the real path."""
    assert isinstance(InMemoryLocalState(), LocalStateStore)


# --------------------------------------------------------------------------- #
#  Builders                                                                    #
# --------------------------------------------------------------------------- #

def engine_with(redis: Optional[FakeRedis] = None, **kw) -> ReconciliationEngine:
    return ReconciliationEngine(redis if redis is not None else FakeRedis(), **kw)


def manager_for(engine: ReconciliationEngine, local: LocalStateStore, **kw) -> RecoveryManager:
    """A fresh RecoveryManager == a freshly started process."""
    return RecoveryManager(engine, local_state=local, **kw)


def order(oid, status="OPEN", qty=100, symbol="RELIANCE", filled=0, **kw) -> dict:
    d = {"order_id": oid, "symbol": symbol, "tradingsymbol": symbol, "exchange": "NSE",
         "status": status, "quantity": qty, "filled_quantity": filled, "price": 100.0}
    d.update(kw)
    return d


def position(symbol="RELIANCE", qty=100, avg=100.0, product="MIS") -> dict:
    return {"symbol": symbol, "tradingsymbol": symbol, "exchange": "NSE",
            "product": product, "quantity": qty, "average_price": avg}


def fill(oid, qty, price, symbol="RELIANCE", trade_id=None) -> dict:
    return {"order_id": oid, "trade_id": trade_id, "symbol": symbol,
            "tradingsymbol": symbol, "exchange": "NSE",
            "quantity": qty, "price": price, "average_price": price}


async def evaluate(broker, local, redis=None, **kw) -> ReconciliationResult:
    """Run a non-enforcing pass through the real engine."""
    return await engine_with(redis, **kw).evaluate(broker, local_state=local)


# =========================================================================== #
#  D1 — broker unreachable must be UNAVAILABLE, never OK                      #
# =========================================================================== #

class TestD1BrokerUnreachableFailsClosed:

    async def test_D1_unreachable_broker_is_UNAVAILABLE_not_OK(self):
        """
        THE D1 REGRESSION.

        Before: every broker fetch was wrapped ``except Exception: x = []``, the
        local side was a hardcoded ``[]``, ``[] == []`` compared equal, status
        was OK, no kill switch was set and trading started blind.
        """
        broker = FakeBroker(unreachable=True)
        local = InMemoryLocalState()          # genuinely empty — the exact [] vs [] shape

        result = await evaluate(broker, local)

        assert result.status is UNAVAILABLE
        assert result.status is not OK
        assert result.ok is False
        assert result.permits_trading is False
        assert bool(result) is False, "a non-OK result must never read as success"
        assert set(result.unavailable_sources) >= {
            "broker_positions", "broker_orders", "broker_trades", "broker_cash",
        }

    async def test_D1_empty_broker_data_is_None_not_empty_list(self):
        """The failed fetch must not be substituted with [] — that is the bug."""
        engine = engine_with()
        snap = await engine.gather(FakeBroker(unreachable=True), local_state=InMemoryLocalState())
        assert snap.broker_positions is None
        assert snap.broker_orders is None
        assert snap.broker_trades is None
        assert snap.broker_cash is None
        assert snap.is_complete() is False
        assert snap.positions_comparable() is False

    async def test_D1_unreachable_broker_trips_kill_switch_and_raises(self):
        redis = FakeRedis()
        engine = engine_with(redis)
        with pytest.raises(ReconciliationError) as exc:
            await engine.reconcile(FakeBroker(unreachable=True),
                                   local_state=InMemoryLocalState())
        assert exc.value.result.status is UNAVAILABLE
        assert redis.store["kill_switch"] == "1", "kill switch must actually be set"

    async def test_D1_unreachable_broker_blocks_trading(self):
        redis = FakeRedis()
        mgr = manager_for(engine_with(redis), InMemoryLocalState())
        report = await mgr.recover(FakeBroker(unreachable=True))

        assert report.status is UNAVAILABLE
        assert report.state is StartupState.BLOCKED
        assert report.trading_permitted is False
        assert bool(report) is False
        assert mgr.trading_permitted is False
        assert mgr.state is StartupState.BLOCKED
        assert redis.store["kill_switch"] == "1"
        with pytest.raises(TradingBlockedError):
            mgr.require_ready()
        assert report.phase(RecoveryPhase.QUERY_BROKER).ok is False

    async def test_D1_partial_broker_outage_is_still_UNAVAILABLE(self):
        """Only funds fails; positions/orders/trades match perfectly. Still not OK."""
        broker = FakeBroker(fail_fetch={"funds"})
        result = await evaluate(broker, InMemoryLocalState())
        assert result.status is UNAVAILABLE
        assert result.unavailable_sources == ["broker_cash"]

    async def test_D1_missing_local_state_source_is_UNAVAILABLE(self):
        """
        A healthy broker plus no local-state source must NOT reconcile clean.
        This is the other half of the old fail-open: db_session defaulted to
        None and the DB helpers returned [] regardless.
        """
        result = await ReconciliationEngine(FakeRedis()).evaluate(
            FakeBroker(), db_session=None
        )
        assert result.status is UNAVAILABLE
        assert set(result.unavailable_sources) == {
            "local_positions", "local_orders", "local_trades", "local_cash",
        }

    async def test_D1_unreadable_local_state_is_UNAVAILABLE(self):
        local = InMemoryLocalState()
        local.fail.add("positions")
        result = await evaluate(FakeBroker(), local)
        assert result.status is UNAVAILABLE
        assert "local_positions" in result.unavailable_sources

    async def test_D1_missing_local_cash_record_is_not_zero(self):
        """'No cash row' must be UNAVAILABLE, never reconciled as 0.0 == 0.0."""
        result = await evaluate(FakeBroker(cash=0.0), InMemoryLocalState(cash=None))
        assert result.status is UNAVAILABLE
        assert "local_cash" in result.unavailable_sources

    async def test_D1_broker_returning_garbage_is_UNAVAILABLE(self):
        """A broker that returns None instead of a list is unavailable, not empty."""
        broker = FakeBroker()
        broker.get_positions = lambda: _async_return(None)          # type: ignore[assignment]
        result = await evaluate(broker, InMemoryLocalState())
        assert result.status is UNAVAILABLE
        assert "broker_positions" in result.unavailable_sources

    def test_D1_unknown_states_never_read_as_success(self):
        assert bool(OK) is True
        for bad in (MISMATCH, UNAVAILABLE, ERROR):
            assert bool(bad) is False, f"{bad} must not read as success"
            assert bad.permits_trading is False
            assert bool(ReconciliationResult(status=bad)) is False
        assert bool(StartupState.READY) is True
        assert bool(StartupState.BLOCKED) is False
        assert bool(StartupState.RECOVERING) is False

    async def test_D1_reconciliation_crash_is_ERROR_not_OK(self):
        """An unexpected internal failure must classify as ERROR and block."""
        broker = FakeBroker()
        broker.get_positions = _raise_typeerror                      # type: ignore[assignment]
        result = await evaluate(broker, InMemoryLocalState())
        assert result.status in (UNAVAILABLE, ERROR)
        assert result.ok is False


async def _async_return(value):
    return value


def _raise_typeerror():
    raise TypeError("broker adapter is broken in an unexpected way")


# =========================================================================== #
#  Real comparison logic                                                      #
# =========================================================================== #

class TestComparisonLogic:

    async def test_broker_position_absent_locally_is_MISMATCH_and_blocks(self):
        """The most dangerous case: an untracked live position."""
        redis = FakeRedis()
        broker = FakeBroker(positions=[position("RELIANCE", 50)])
        local = InMemoryLocalState()
        mgr = manager_for(engine_with(redis), local)

        report = await mgr.recover(broker)

        assert report.status is MISMATCH
        assert report.state is StartupState.BLOCKED
        assert report.trading_permitted is False
        kinds = [d.kind for d in report.unresolved]
        assert DiscrepancyKind.MISSING_LOCAL in kinds
        missing = [d for d in report.unresolved if d.kind is DiscrepancyKind.MISSING_LOCAL]
        assert missing[0].broker_value == 50
        assert missing[0].local_value is None
        assert redis.store["kill_switch"] == "1"

    async def test_local_position_absent_at_broker_is_MISMATCH(self):
        result = await evaluate(
            FakeBroker(), InMemoryLocalState(positions=[position("TCS", 25)])
        )
        assert result.status is MISMATCH
        d = result.of_kind(DiscrepancyKind.MISSING_BROKER)
        assert len(d) == 1
        assert d[0].local_value == 25
        assert d[0].broker_value is None

    async def test_position_quantity_mismatch_detected(self):
        result = await evaluate(
            FakeBroker(positions=[position("RELIANCE", 100)]),
            InMemoryLocalState(positions=[position("RELIANCE", 70)]),
        )
        assert result.status is MISMATCH
        d = result.of_kind(DiscrepancyKind.MISMATCHED_QTY)
        assert (d[0].broker_value, d[0].local_value) == (100, 70)

    async def test_position_average_price_mismatch_beyond_tolerance(self):
        result = await evaluate(
            FakeBroker(positions=[position("RELIANCE", 100, avg=100.0)]),
            InMemoryLocalState(positions=[position("RELIANCE", 100, avg=105.0)]),
        )
        assert result.status is MISMATCH
        assert result.of_kind(DiscrepancyKind.MISMATCHED_PRICE)

    async def test_position_average_price_within_tolerance_is_OK(self):
        """100.00 vs 100.05 is 0.05 % — inside the 0.1 % tolerance."""
        result = await evaluate(
            FakeBroker(positions=[position("RELIANCE", 100, avg=100.0)]),
            InMemoryLocalState(positions=[position("RELIANCE", 100, avg=100.05)]),
        )
        assert result.status is OK

    async def test_cash_mismatch_beyond_tolerance_detected(self):
        result = await evaluate(
            FakeBroker(cash=900_000.0), InMemoryLocalState(cash=START_CASH)
        )
        assert result.status is MISMATCH
        d = result.of_kind(DiscrepancyKind.MISMATCHED_CASH)
        assert len(d) == 1
        assert d[0].symbol == "cash"
        assert (d[0].broker_value, d[0].local_value) == (900_000.0, 1_000_000.0)

    async def test_cash_within_tolerance_is_OK(self):
        result = await evaluate(
            FakeBroker(cash=START_CASH), InMemoryLocalState(cash=START_CASH - 0.50)
        )
        assert result.status is OK

    async def test_open_order_at_broker_absent_locally_is_MISMATCH(self):
        result = await evaluate(FakeBroker(orders=[order("O1")]), InMemoryLocalState())
        assert result.status is MISMATCH
        d = result.of_kind(DiscrepancyKind.MISSING_LOCAL)
        assert d[0].order_id == "O1"

    async def test_open_order_local_absent_at_broker_is_MISMATCH(self):
        result = await evaluate(
            FakeBroker(), InMemoryLocalState(orders=[order("O1")])
        )
        assert result.status is MISMATCH
        assert result.of_kind(DiscrepancyKind.MISSING_BROKER)[0].order_id == "O1"

    async def test_order_quantity_mismatch_detected(self):
        result = await evaluate(
            FakeBroker(orders=[order("O1", qty=100)]),
            InMemoryLocalState(orders=[order("O1", qty=50)]),
        )
        assert result.status is MISMATCH
        assert result.of_kind(DiscrepancyKind.MISMATCHED_QTY)

    async def test_terminal_orders_on_both_sides_do_not_trip_anything(self):
        result = await evaluate(
            FakeBroker(orders=[order("O1", status="CANCELLED")]),
            InMemoryLocalState(orders=[order("O1", status="CANCELLED")]),
        )
        assert result.status is OK

    async def test_clean_match_is_OK_and_only_then_permits_trading(self):
        redis = FakeRedis()
        broker = FakeBroker(
            positions=[position("RELIANCE", 100, avg=100.0)],
            orders=[order("O1", status="COMPLETE", filled=100)],
            trades=[fill("O1", 60, 100.0), fill("O1", 40, 100.0)],
            cash=START_CASH - 10_000.0,
        )
        local = InMemoryLocalState(
            positions=[position("RELIANCE", 100, avg=100.0)],
            orders=[order("O1", status="COMPLETE", filled=100)],
            trades=[fill("O1", 60, 100.0), fill("O1", 40, 100.0)],
            cash=START_CASH - 10_000.0,
        )
        mgr = manager_for(engine_with(redis), local)
        report = await mgr.recover(broker)

        assert report.status is OK
        assert report.state is StartupState.READY
        assert report.trading_permitted is True
        assert bool(report) is True
        assert mgr.trading_permitted is True
        mgr.require_ready()                       # must not raise
        assert "kill_switch" not in redis.store
        assert report.phase(RecoveryPhase.PERMIT_TRADING).ok is True

    async def test_every_phase_runs_in_the_required_order(self):
        mgr = manager_for(engine_with(), InMemoryLocalState())
        report = await mgr.recover(FakeBroker())
        assert [p.phase for p in report.phases] == [
            RecoveryPhase.LOAD_LOCAL_STATE,
            RecoveryPhase.QUERY_BROKER,
            RecoveryPhase.RECONCILE_ORDERS,
            RecoveryPhase.RECONCILE_POSITIONS,
            RecoveryPhase.RECONCILE_CASH,
            RecoveryPhase.IDENTIFY_UNKNOWN_ORDERS,
            RecoveryPhase.IDENTIFY_MISMATCHES,
            RecoveryPhase.RESOLVE,
            RecoveryPhase.PERMIT_TRADING,
        ]


# =========================================================================== #
#  D2 — kill-switch honesty                                                   #
# =========================================================================== #

class TestD2KillSwitchHonesty:

    MISMATCHING = dict(positions=[position("RELIANCE", 50)])

    async def _reconcile_mismatch(self, engine):
        return await engine.reconcile(
            FakeBroker(**self.MISMATCHING), local_state=InMemoryLocalState()
        )

    async def test_D2_store_write_failure_propagates_and_is_not_claimed(self):
        """
        THE D2 REGRESSION.

        Before: ``_activate_kill_switch`` caught the store error and logged it,
        then ``reconcile()`` raised "... Kill switch activated." while the store
        contained no ``kill_switch`` key at all.
        """
        redis = FakeRedis(raise_on_set=True)
        with pytest.raises(KillSwitchActivationError) as exc:
            await self._reconcile_mismatch(engine_with(redis))

        assert "kill_switch" not in redis.store
        message = str(exc.value)
        assert "NOT ACTIVATED" in message
        assert "Kill switch activated." not in message, "must not claim what did not happen"
        assert isinstance(exc.value, ReconciliationError)   # still catchable as one

    async def test_D2_absent_store_propagates(self):
        """Default kill_switch_store=None used to be a silent no-op."""
        with pytest.raises(KillSwitchActivationError) as exc:
            await self._reconcile_mismatch(ReconciliationEngine())
        assert "no kill_switch_store is configured" in str(exc.value)

    async def test_D2_write_that_does_not_persist_is_caught_by_readback(self):
        """set() returns success but nothing persists — read-back catches the lie."""
        redis = FakeRedis(swallow_writes=True)
        with pytest.raises(KillSwitchActivationError) as exc:
            await self._reconcile_mismatch(engine_with(redis))
        assert "read-back returned" in str(exc.value)
        assert "kill_switch" not in redis.store

    async def test_D2_unverifiable_write_propagates(self):
        redis = FakeRedis(raise_on_get=True)
        with pytest.raises(KillSwitchActivationError) as exc:
            await self._reconcile_mismatch(engine_with(redis))
        assert "UNVERIFIED" in str(exc.value)

    async def test_D2_store_without_get_is_accepted(self):
        """A write-only store cannot be verified, but the write did succeed."""
        store = WriteOnlyStore()
        with pytest.raises(ReconciliationError) as exc:
            await self._reconcile_mismatch(ReconciliationEngine(store))
        assert not isinstance(exc.value, KillSwitchActivationError)
        assert store.store["kill_switch"] == "1"

    async def test_D2_successful_activation_may_say_so(self):
        """Positive control: the claim is allowed exactly when it is true."""
        redis = FakeRedis()
        with pytest.raises(ReconciliationError) as exc:
            await self._reconcile_mismatch(engine_with(redis))
        assert not isinstance(exc.value, KillSwitchActivationError)
        assert "Kill switch activated." in str(exc.value)
        assert redis.store["kill_switch"] == "1"

    async def test_D2_recovery_propagates_kill_switch_failure(self):
        redis = FakeRedis(raise_on_set=True)
        mgr = manager_for(engine_with(redis), InMemoryLocalState())
        with pytest.raises(KillSwitchActivationError):
            await mgr.recover(FakeBroker(**self.MISMATCHING))
        assert mgr.state is StartupState.BLOCKED
        assert mgr.trading_permitted is False

    async def test_D2_recovery_reports_failure_when_not_raising(self):
        redis = FakeRedis(raise_on_set=True)
        mgr = manager_for(engine_with(redis), InMemoryLocalState())
        report = await mgr.recover(
            FakeBroker(**self.MISMATCHING), raise_on_kill_switch_failure=False
        )
        assert report.kill_switch_activated is False
        assert report.kill_switch_error is not None
        assert "NOT ACTIVATED" in report.kill_switch_error
        assert report.state is StartupState.BLOCKED
        assert report.trading_permitted is False
        assert "kill_switch" not in redis.store

    async def test_D2_no_kill_switch_on_a_clean_pass(self):
        redis = FakeRedis()
        result = await engine_with(redis).reconcile(
            FakeBroker(), local_state=InMemoryLocalState()
        )
        assert result.status is OK
        assert result.kill_switch_activated is False
        assert "kill_switch" not in redis.store


# =========================================================================== #
#  D3 — partial fills must not be collapsed                                   #
# =========================================================================== #

class TestD3PartialFills:

    THREE_FILLS = [fill("O1", 30, 100.0), fill("O1", 30, 101.0), fill("O1", 40, 150.0)]
    # quantity-weighted average = (3000 + 3030 + 6000) / 100 = 120.30

    async def _orders_findings(self, broker_trades, local_trades):
        engine = engine_with()
        snap = await engine.gather(
            FakeBroker(trades=broker_trades),
            local_state=InMemoryLocalState(trades=local_trades),
        )
        return engine.reconcile_orders(snap)

    async def test_D3_partial_fills_are_not_collapsed(self):
        """
        THE D3 REGRESSION.

        Before: ``{t["order_id"]: t for t in trades}`` kept only the LAST fill,
        so 100 shares executed against 30 recorded produced exactly one
        complaint — a price difference — and no quantity discrepancy at all.
        """
        found = await self._orders_findings(self.THREE_FILLS, [fill("O1", 30, 100.0)])
        kinds = [d.kind for d in found]

        assert DiscrepancyKind.MISMATCHED_QTY in kinds, "the 70 missing shares must surface"
        qty = [d for d in found if d.kind is DiscrepancyKind.MISMATCHED_QTY][0]
        assert (qty.broker_value, qty.local_value) == (100, 30)
        assert "3 fill(s)" in qty.details and "1 fill(s)" in qty.details
        assert kinds != [DiscrepancyKind.MISMATCHED_PRICE], "the old collapsed behaviour"

    async def test_D3_vwap_uses_every_fill_not_just_the_last(self):
        """Local recorded one 100-share fill at the true VWAP: quantity agrees."""
        found = await self._orders_findings(self.THREE_FILLS, [fill("O1", 100, 120.30)])
        kinds = [d.kind for d in found]
        assert DiscrepancyKind.MISMATCHED_PRICE not in kinds, (
            "VWAP across all three fills is 120.30; the old code compared 150.0"
        )
        assert DiscrepancyKind.MISMATCHED_QTY not in kinds
        assert DiscrepancyKind.PARTIAL_FILL_MISMATCH in kinds

    async def test_D3_fill_count_difference_is_reported(self):
        found = await self._orders_findings(
            [fill("O1", 50, 100.0), fill("O1", 50, 100.0)], [fill("O1", 100, 100.0)]
        )
        d = [x for x in found if x.kind is DiscrepancyKind.PARTIAL_FILL_MISMATCH]
        assert len(d) == 1
        assert (d[0].broker_value, d[0].local_value) == (2, 1)
        assert d[0].severity is Severity.WARNING

    async def test_D3_fill_count_difference_still_blocks_trading(self):
        """A 'warning' severity discrepancy is still not RECONCILIATION_OK."""
        result = await evaluate(
            FakeBroker(trades=[fill("O1", 50, 100.0), fill("O1", 50, 100.0)]),
            InMemoryLocalState(trades=[fill("O1", 100, 100.0)]),
        )
        assert result.status is MISMATCH
        assert result.permits_trading is False

    async def test_D3_identical_fill_sequences_reconcile_clean(self):
        found = await self._orders_findings(self.THREE_FILLS, list(self.THREE_FILLS))
        assert found == []

    async def test_D3_broker_fills_absent_locally_report_the_fill_count(self):
        found = await self._orders_findings(self.THREE_FILLS, [])
        d = [x for x in found if x.kind is DiscrepancyKind.MISSING_LOCAL][0]
        assert d.broker_value == 100
        assert "3 fill(s)" in d.details

    async def test_D3_local_fills_absent_at_broker(self):
        found = await self._orders_findings([], self.THREE_FILLS)
        d = [x for x in found if x.kind is DiscrepancyKind.MISSING_BROKER][0]
        assert d.local_value == 100


# =========================================================================== #
#  Unknown / ambiguous orders                                                 #
# =========================================================================== #

class TestUnknownOrders:

    async def test_unknown_order_state_blocks_trading(self):
        local = InMemoryLocalState(orders=[order("O1", status="UNKNOWN")])
        broker = FakeBroker(orders=[])
        mgr = manager_for(engine_with(), local, disambiguate_unknown_orders=False)
        report = await mgr.recover(broker)

        assert report.status is MISMATCH
        assert report.state is StartupState.BLOCKED
        assert any(d.kind is DiscrepancyKind.UNKNOWN_ORDER_STATE for d in report.unresolved)
        assert mgr.trading_permitted is False

    async def test_order_without_broker_id_is_unknown(self):
        """A crash between 'about to submit' and 'ack stored'."""
        local = InMemoryLocalState(orders=[order(None, status="PENDING")])
        result = await evaluate(FakeBroker(), local)
        d = result.of_kind(DiscrepancyKind.UNKNOWN_ORDER_STATE)
        assert len(d) == 1
        assert d[0].order_id is None
        assert result.status is MISMATCH

    async def test_order_without_broker_id_cannot_be_auto_resolved(self):
        local = InMemoryLocalState(orders=[order(None, status="PENDING")])
        mgr = manager_for(engine_with(), local)
        report = await mgr.recover(FakeBroker())
        assert report.state is StartupState.BLOCKED
        assert report.resolutions == []
        assert report.phase(RecoveryPhase.RESOLVE).ok is False

    async def test_unknown_order_unblocks_only_after_resolution_and_reverification(self):
        """
        Resolve, then RE-RECONCILE.  READY is reached only because the second,
        real reconciliation returned OK — not because the resolver said so.
        """
        local = InMemoryLocalState(orders=[order("O1", status="UNKNOWN")])
        broker = FakeBroker(
            orders=[order("O1", status="COMPLETE", filled=100)],
            trades=[fill("O1", 100, 100.0)],
            positions=[position("RELIANCE", 100, avg=100.0)],
            cash=START_CASH - 10_000.0,
            order_status={"O1": {"status": "COMPLETE", "filled_quantity": 100}},
        )

        async def repair(_broker, _result):
            """Adopt broker truth for the resolved order."""
            local.orders = [order("O1", status="COMPLETE", filled=100)]
            local.trades = [fill("O1", 100, 100.0)]
            local.positions = [position("RELIANCE", 100, avg=100.0)]
            local.cash = START_CASH - 10_000.0
            return ["local state repaired from broker truth"]

        mgr = manager_for(engine_with(), local, resolver=repair)
        report = await mgr.recover(broker)

        assert "order O1 disambiguated at broker as COMPLETE" in report.resolutions
        assert report.phase(RecoveryPhase.VERIFY) is not None
        assert report.phase(RecoveryPhase.VERIFY).ok is True
        assert report.status is OK
        assert report.state is StartupState.READY
        assert mgr.trading_permitted is True

    async def test_a_resolver_that_lies_cannot_unlock_trading(self):
        """READY is reachable ONLY via a fresh successful reconciliation."""
        local = InMemoryLocalState(orders=[order("O1", status="UNKNOWN")])
        broker = FakeBroker(
            orders=[order("O1", status="OPEN")],
            order_status={"O1": {"status": "OPEN"}},
        )

        async def liar(_broker, _result):
            return ["everything is fine, honest"]

        mgr = manager_for(engine_with(), local, resolver=liar)
        report = await mgr.recover(broker)

        assert report.resolutions                       # the resolver "did" something
        assert report.phase(RecoveryPhase.VERIFY).ok is False
        assert report.status is not OK
        assert report.state is StartupState.BLOCKED
        assert mgr.trading_permitted is False

    async def test_disambiguation_failure_leaves_the_order_unknown(self):
        local = InMemoryLocalState(orders=[order("O1", status="UNKNOWN")])
        broker = FakeBroker(orders=[], fail_order_status=True)
        mgr = manager_for(engine_with(), local)
        report = await mgr.recover(broker)
        assert report.resolutions == []
        assert report.state is StartupState.BLOCKED

    async def test_broker_reported_unrecognised_status_is_unknown(self):
        result = await evaluate(
            FakeBroker(orders=[order("O1", status="WEIRD NEW STATUS")]),
            InMemoryLocalState(),
        )
        assert result.of_kind(DiscrepancyKind.UNKNOWN_ORDER_STATE)
        assert result.status is MISMATCH


# =========================================================================== #
#  Crash-and-restart at every dangerous point                                 #
# =========================================================================== #

def _crash_scenarios() -> list[tuple[str, FakeBroker, InMemoryLocalState, bool]]:
    """
    (name, broker-side truth, surviving local state, may_trade_after_recovery)

    Each entry is the world as it exists when a NEW process starts after a
    crash at that point.  ``may_trade_after_recovery`` is True only where the
    two sides genuinely agree.
    """
    return [
        # 1. crash BEFORE submission: nothing was sent, nothing was written.
        ("before_submission", FakeBroker(), InMemoryLocalState(), True),

        # 2. crash DURING submission: the order reached the exchange, the
        #    response never came back, nothing was written locally.
        ("during_submission",
         FakeBroker(orders=[order("O1")]),
         InMemoryLocalState(),
         False),

        # 3. crash AFTER submission, BEFORE acknowledgement: an intent row
        #    exists locally with no broker id.
        ("after_submission_before_ack",
         FakeBroker(orders=[order("O1")]),
         InMemoryLocalState(orders=[order(None, status="PENDING")]),
         False),

        # 4. crash AFTER acknowledgement: both sides agree on an open order.
        ("after_acknowledgement",
         FakeBroker(orders=[order("O1")]),
         InMemoryLocalState(orders=[order("O1")]),
         True),

        # 5. crash AFTER a partial fill that was never recorded.
        ("after_partial_fill_unrecorded",
         FakeBroker(orders=[order("O1", filled=30)],
                    trades=[fill("O1", 30, 100.0)],
                    positions=[position("RELIANCE", 30, avg=100.0)],
                    cash=START_CASH - 3_000.0),
         InMemoryLocalState(orders=[order("O1")]),
         False),

        # 5b. crash after a partial fill that WAS recorded.
        ("after_partial_fill_recorded",
         FakeBroker(orders=[order("O1", filled=30)],
                    trades=[fill("O1", 30, 100.0)],
                    positions=[position("RELIANCE", 30, avg=100.0)],
                    cash=START_CASH - 3_000.0),
         InMemoryLocalState(orders=[order("O1", filled=30)],
                            trades=[fill("O1", 30, 100.0)],
                            positions=[position("RELIANCE", 30, avg=100.0)],
                            cash=START_CASH - 3_000.0),
         True),

        # 6. crash AFTER complete fill, before it was recorded.
        ("after_complete_fill_unrecorded",
         FakeBroker(orders=[order("O1", status="COMPLETE", filled=100)],
                    trades=[fill("O1", 30, 100.0), fill("O1", 70, 100.0)],
                    positions=[position("RELIANCE", 100, avg=100.0)],
                    cash=START_CASH - 10_000.0),
         InMemoryLocalState(orders=[order("O1")]),
         False),

        # 6b. crash after a complete fill that WAS recorded.
        ("after_complete_fill_recorded",
         FakeBroker(orders=[order("O1", status="COMPLETE", filled=100)],
                    trades=[fill("O1", 30, 100.0), fill("O1", 70, 100.0)],
                    positions=[position("RELIANCE", 100, avg=100.0)],
                    cash=START_CASH - 10_000.0),
         InMemoryLocalState(orders=[order("O1", status="COMPLETE", filled=100)],
                            trades=[fill("O1", 30, 100.0), fill("O1", 70, 100.0)],
                            positions=[position("RELIANCE", 100, avg=100.0)],
                            cash=START_CASH - 10_000.0),
         True),

        # 7. crash DURING cancellation: written off locally, still live at the
        #    broker — it can still fill.
        ("during_cancellation",
         FakeBroker(orders=[order("O1")]),
         InMemoryLocalState(orders=[order("O1", status="CANCELLED")]),
         False),

        # 8. crash DURING reconciliation: the broker call sequence dies part way.
        ("during_reconciliation",
         FakeBroker(orders=[order("O1")], fail_fetch={"trades", "funds"}),
         InMemoryLocalState(orders=[order("O1")]),
         False),
    ]


@pytest.mark.parametrize(
    "name,broker,local,may_trade",
    _crash_scenarios(),
    ids=[s[0] for s in _crash_scenarios()],
)
async def test_restart_after_crash_reaches_a_safe_state(name, broker, local, may_trade):
    """
    Simulated restart: a brand-new RecoveryManager over the surviving state.

    Recovery must always finish in a defined state and must never permit
    trading unless reconciliation actually succeeded.
    """
    redis = FakeRedis()
    mgr = manager_for(engine_with(redis), local, disambiguate_unknown_orders=False)

    assert mgr.state is StartupState.BLOCKED, "a process that has not recovered may not trade"
    report = await mgr.recover(broker)

    assert mgr.state in (StartupState.READY, StartupState.BLOCKED)
    assert mgr.state is not StartupState.RECOVERING, "recovery must terminate"
    assert report.trading_permitted is mgr.trading_permitted

    if may_trade:
        assert report.status is OK, f"{name}: {report.summary()}"
        assert mgr.state is StartupState.READY
        assert "kill_switch" not in redis.store
        mgr.require_ready()
    else:
        assert report.status is not OK, f"{name} must not reconcile clean"
        assert mgr.state is StartupState.BLOCKED
        assert mgr.trading_permitted is False
        assert report.unresolved, f"{name} must name what is wrong"
        assert redis.store["kill_switch"] == "1"
        with pytest.raises(TradingBlockedError):
            mgr.require_ready()


async def test_crash_during_reconciliation_leaves_process_blocked():
    """A process interrupted mid-reconciliation restarts BLOCKED, not READY."""
    mgr = manager_for(engine_with(), InMemoryLocalState())
    assert mgr.state is StartupState.BLOCKED
    assert mgr.trading_permitted is False
    assert mgr.blocked_reason
    with pytest.raises(TradingBlockedError):
        mgr.require_ready()


async def test_ready_process_can_be_reblocked_by_a_later_failure():
    """A periodic pass that fails must revoke permission."""
    local = InMemoryLocalState()
    mgr = manager_for(engine_with(), local)
    assert (await mgr.recover(FakeBroker())).state is StartupState.READY

    # Broker goes away mid-session.
    report = await mgr.recover(FakeBroker(unreachable=True))
    assert report.status is UNAVAILABLE
    assert mgr.state is StartupState.BLOCKED
    with pytest.raises(TradingBlockedError):
        mgr.require_ready()


async def test_manual_block_revokes_permission():
    mgr = manager_for(engine_with(), InMemoryLocalState())
    await mgr.recover(FakeBroker())
    assert mgr.trading_permitted is True
    mgr.block("operator halt")
    assert mgr.state is StartupState.BLOCKED
    assert mgr.blocked_reason == "operator halt"


async def test_untracked_broker_position_survives_restart_and_keeps_blocking():
    """Restarting again does not 'forget' the problem into an OK."""
    local = InMemoryLocalState()
    broker = FakeBroker(positions=[position("RELIANCE", 50)])
    for _ in range(3):
        mgr = manager_for(engine_with(), local)         # a fresh process each time
        report = await mgr.recover(broker)
        assert report.status is MISMATCH
        assert mgr.trading_permitted is False


# =========================================================================== #
#  The real ORM-backed local state store (replaces the `return []` stubs)     #
# =========================================================================== #

@pytest_asyncio.fixture
async def orm_session():
    from app.database.models import Base

    db = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with db.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(db, expire_on_commit=False)
    async with maker() as session:
        yield session
    await db.dispose()


class BrokenSession:
    """An AsyncSession whose every query fails — a database outage."""

    async def execute(self, *_a, **_kw):
        raise OperationalError("SELECT ...", {}, Exception("database is gone"))


async def _seed(session) -> int:
    from app.database.models import (
        AccountCash,
        Order,
        Position,
        Trade,
        User,
    )

    user = User(email="t@example.com", hashed_password="x")
    session.add(user)
    await session.flush()

    session.add(Position(
        user_id=user.id, symbol="RELIANCE", exchange="NSE", quantity=100,
        average_price=100.0, strategy="intraday",
        entry_date=datetime.now(timezone.utc), is_open=True,
    ))
    session.add(Position(
        user_id=user.id, symbol="TCS", exchange="NSE", quantity=10,
        average_price=3800.0, strategy="swing",
        entry_date=datetime.now(timezone.utc), is_open=False,      # closed: excluded
    ))
    o = Order(
        user_id=user.id, order_id_broker="O1", symbol="RELIANCE", exchange="NSE",
        transaction_type="BUY", quantity=100, price=100.0, order_type="LIMIT",
        status="COMPLETE", strategy="intraday",
    )
    session.add(o)
    await session.flush()

    # THREE separate fills of the same order — the D3 shape.
    for qty, px in ((30, 100.0), (30, 101.0), (40, 150.0)):
        session.add(Trade(
            user_id=user.id, order_id=o.id, symbol="RELIANCE", exchange="NSE",
            transaction_type="BUY", quantity=qty, price=px, value=qty * px,
            net_value=qty * px, strategy="intraday",
        ))
    # AccountCash, not CapitalAllocation. Local cash used to be read from the
    # newest CapitalAllocation row, which is a monthly capital BUDGET rather
    # than a balance — reconciling the broker's cash against a budget compares
    # two unrelated numbers and produces a permanent false MISMATCH.
    session.add(AccountCash(
        user_id=user.id, trading_mode="paper", cash=250_000.0,
        reserved=0.0, realized_pnl=0.0, total_costs=0.0,
    ))
    await session.commit()
    return user.id


class TestSqlAlchemyLocalStateStore:
    """The `return []` stubs are gone: these queries hit real rows."""

    async def test_positions_are_real_rows_not_empty(self, orm_session):
        user_id = await _seed(orm_session)
        rows = await SqlAlchemyLocalStateStore(orm_session, user_id).get_positions()
        assert rows != [], "the stub used to return [] unconditionally"
        assert [r["symbol"] for r in rows] == ["RELIANCE"]
        assert rows[0]["quantity"] == 100
        assert rows[0]["product"] == "MIS"

    async def test_orders_are_real_rows_not_empty(self, orm_session):
        user_id = await _seed(orm_session)
        rows = await SqlAlchemyLocalStateStore(orm_session, user_id).get_orders()
        assert [r["order_id"] for r in rows] == ["O1"]
        assert rows[0]["status"] == "COMPLETE"

    async def test_trades_keep_one_row_per_fill(self, orm_session):
        user_id = await _seed(orm_session)
        rows = await SqlAlchemyLocalStateStore(orm_session, user_id).get_trades()
        assert len(rows) == 3, "partial fills must not be collapsed by the store either"
        assert sum(r["quantity"] for r in rows) == 100
        assert all(r["order_id"] == "O1" for r in rows)

    async def test_cash_comes_from_the_account_balance(self, orm_session):
        user_id = await _seed(orm_session)
        cash = await SqlAlchemyLocalStateStore(
            orm_session, user_id, trading_mode="paper"
        ).get_cash()
        assert cash["cash"] == 250_000.0
        assert cash["margin_used"] == 0.0

    async def test_cash_is_scoped_to_the_trading_mode(self, orm_session):
        """A paper balance must never be reported for a live account."""
        user_id = await _seed(orm_session)
        store = SqlAlchemyLocalStateStore(orm_session, user_id, trading_mode="live")
        with pytest.raises(LocalStateUnavailable):
            await store.get_cash()

    async def test_absent_cash_row_raises_rather_than_returning_zero(self, orm_session):
        store = SqlAlchemyLocalStateStore(orm_session, 999)
        with pytest.raises(LocalStateUnavailable):
            await store.get_cash()

    async def test_orm_failure_raises_LocalStateUnavailable(self):
        """A dead database must raise, never return [] (that was the D1 root cause)."""
        store = SqlAlchemyLocalStateStore(BrokenSession(), 1)
        for getter in (store.get_positions, store.get_orders,
                       store.get_trades, store.get_cash):
            with pytest.raises(LocalStateUnavailable):
                await getter()

    async def test_engine_reports_UNAVAILABLE_when_the_orm_fails(self):
        result = await engine_with().evaluate(FakeBroker(), db_session=BrokenSession())
        assert result.status is UNAVAILABLE
        assert set(result.unavailable_sources) == {
            "local_positions", "local_orders", "local_trades", "local_cash",
        }

    async def test_end_to_end_orm_backed_reconciliation_detects_the_mismatch(
        self, orm_session
    ):
        """Real ORM rows vs a broker that disagrees — the full production path."""
        user_id = await _seed(orm_session)
        broker = FakeBroker(
            positions=[position("RELIANCE", 70, avg=100.0, product="MIS")],
            orders=[order("O1", status="COMPLETE", filled=100)],
            trades=[fill("O1", 30, 100.0), fill("O1", 30, 101.0), fill("O1", 40, 150.0)],
            cash=250_000.0,
        )
        engine = ReconciliationEngine(FakeRedis(), user_id=user_id)
        result = await engine.evaluate(broker, db_session=orm_session)

        assert result.status is MISMATCH
        qty = result.of_kind(DiscrepancyKind.MISMATCHED_QTY)
        assert any(d.broker_value == 70 and d.local_value == 100 for d in qty)

    async def test_end_to_end_orm_backed_reconciliation_is_OK_when_agreed(
        self, orm_session
    ):
        user_id = await _seed(orm_session)
        broker = FakeBroker(
            positions=[position("RELIANCE", 100, avg=100.0, product="MIS")],
            orders=[order("O1", status="COMPLETE", filled=100)],
            trades=[fill("O1", 30, 100.0), fill("O1", 30, 101.0), fill("O1", 40, 150.0)],
            cash=250_000.0,
        )
        engine = ReconciliationEngine(FakeRedis(), user_id=user_id)
        mgr = RecoveryManager(engine)
        report = await mgr.recover(broker, db_session=orm_session)

        assert report.status is OK, report.summary()
        assert report.state is StartupState.READY
        assert mgr.trading_permitted is True
