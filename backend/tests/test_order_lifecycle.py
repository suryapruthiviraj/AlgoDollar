"""
Order-lifecycle safety tests.

Every test here drives the REAL production classes — ``OrderState``,
``OrderRecord``, ``InMemoryOrderStore``, ``RedisOrderStore``, ``ExecutionSafety``
and ``OrderManager``.  Nothing under test is re-implemented here; the only test
doubles are a fake *broker connection* and a fake *redis connection*, i.e. the
two things that would otherwise require a network.

The fake broker keeps a real order book: ``place_order`` writes the order into
the book **before** it can fail, exactly like a broker that received the order
and then lost the response.  That is what makes the "lost response" and
"ambiguous timeout" scenarios meaningful rather than decorative.
"""
from __future__ import annotations

import asyncio
import math
from typing import Any, Optional

import pandas as pd
import pytest

from app.broker.base import BrokerInterface, OrderType, Product, TransactionType
from app.execution.lifecycle import (
    IllegalStateTransition,
    InMemoryOrderStore,
    OrderBlockedError,
    OrderRecord,
    OrderState,
    RedisOrderStore,
    assert_transition,
    make_client_order_id,
    map_broker_status,
)
from app.execution.order_manager import (
    AmbiguousOrderError,
    OrderManager,
    Signal,
    calculate_costs,
)
from app.execution.safety import (
    ExecutionSafety,
    KillSwitchActiveError,
    MarketClosedError,
    SafetyCheckError,
    StaleDataError,
)


# --------------------------------------------------------------------------- #
#  Test doubles: one broker connection, one redis connection                   #
# --------------------------------------------------------------------------- #


class FakeRedis:
    """In-memory stand-in supporting the SET ... NX EX that reserve() needs."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.raise_on_get = False
        self.raise_on_set = False

    def get(self, key):
        if self.raise_on_get:
            raise ConnectionError("redis down")
        return self.store.get(key)

    def set(self, key, value, nx: bool = False, ex: Optional[int] = None):
        if self.raise_on_set:
            raise ConnectionError("redis down")
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True

    def exists(self, key):
        return 1 if key in self.store else 0

    def keys(self, pattern="*"):
        prefix = pattern.rstrip("*")
        return [k for k in self.store if k.startswith(prefix)]


class FakeBroker(BrokerInterface):
    """
    A broker connection with a REAL order book.

    ``place_order`` appends to ``order_book`` *first*, so a subsequent failure
    reproduces "the exchange has your order, you just don't know it".
    """

    def __init__(
        self,
        *,
        connected: bool = True,
        market_open: bool = True,
        stale: bool = False,
        lose_response: bool = False,
        hang_after_accept: bool = False,
        reject_with: Optional[Exception] = None,
        positions: Optional[list[dict]] = None,
        orders: Optional[list[dict]] = None,
        get_orders_fails: bool = False,
        quote_price: float = 100.0,
        quote_volume: int = 1_000_000,
        quote_missing: bool = False,
    ) -> None:
        self._connected = connected
        self._market_open = market_open
        self._stale = stale
        self._lose_response = lose_response
        self._hang = hang_after_accept
        self._reject_with = reject_with
        self._positions = positions or []
        self.order_book: list[dict] = list(orders or [])
        self._get_orders_fails = get_orders_fails
        self._quote_price = quote_price
        self._quote_volume = quote_volume
        self._quote_missing = quote_missing
        self.placed: list[dict] = []        # every submission that reached us
        self.cancelled: list[str] = []
        self.get_orders_calls = 0

    # --- gate hooks ------------------------------------------------------ #
    def is_market_open(self) -> bool:
        return self._market_open

    def is_stale_tick(self, symbol: str, max_age_seconds: float = 30.0) -> bool:
        return self._stale

    # --- BrokerInterface -------------------------------------------------- #
    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def get_profile(self) -> dict:
        return {"user_name": "fake"}

    async def get_holdings(self) -> list[dict]:
        return []

    async def get_positions(self) -> list[dict]:
        return list(self._positions)

    async def get_orders(self) -> list[dict]:
        self.get_orders_calls += 1
        if self._get_orders_fails:
            raise ConnectionError("kite 5xx")
        return [dict(o) for o in self.order_book]

    async def get_trades(self) -> list[dict]:
        return []

    async def get_funds(self) -> dict:
        return {"cash": 1e9, "margin_available": 1e9, "margin_used": 0.0}

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        if self._quote_missing:
            return {}
        return {
            s: {"last_price": self._quote_price, "volume": self._quote_volume}
            for s in symbols
        }

    async def get_historical_data(self, symbol, exchange, interval, from_date, to_date):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    async def place_order(self, symbol, exchange, txn_type, qty, price,
                          order_type, product, tag="", trigger_price: float = 0.0) -> str:
        if self._reject_with is not None:
            # Deterministic rejection: never reaches the book.
            raise self._reject_with
        order_id = f"ORDER{len(self.placed) + 1:04d}"
        self.placed.append({
            "symbol": symbol, "qty": qty, "txn": txn_type.value, "tag": tag,
            "order_type": order_type.value, "price": price,
            "trigger_price": trigger_price,
        })
        self.order_book.append({
            "order_id": order_id, "tradingsymbol": symbol, "exchange": exchange,
            "tag": tag, "status": "OPEN", "quantity": qty, "filled_quantity": 0,
            "transaction_type": txn_type.value,
        })
        if self._hang:
            await asyncio.sleep(3600)          # the process dies here
        if self._lose_response:
            raise TimeoutError("read timeout after the exchange accepted the order")
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        for o in self.order_book:
            if o["order_id"] == order_id:
                o["status"] = "CANCELLED"
        return True

    async def modify_order(self, order_id, qty=None, price=None) -> bool:
        return True

    async def get_order_status(self, order_id: str) -> dict:
        for o in self.order_book:
            if o["order_id"] == order_id:
                return dict(o)
        return {"status": "OPEN", "filled_quantity": 0}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def trading_mode(self) -> str:
        # "paper", not "fake". This double cannot move real money, so it is a
        # paper venue as far as the safety layer is concerned.
        #
        # This matters: `order_manager._require_eligible_for_live` treats any
        # broker that does not positively declare itself paper as LIVE and
        # demands LIVE_ELIGIBLE. Declaring "fake" made every test order look
        # like a real one and blocked it — which is the rule working correctly,
        # not a defect in the rule. A broker that will not say what it is gets
        # treated as the expensive case.
        return "paper"

    def instrument_token(self, symbol: str, exchange: str) -> int:
        return 12345


class LegacyBrokerNoTrigger(FakeBroker):
    """A broker whose place_order cannot accept a trigger price (today's Kite)."""

    async def place_order(self, symbol, exchange, txn_type, qty, price,
                          order_type, product, tag="") -> str:
        return await FakeBroker.place_order(
            self, symbol, exchange, txn_type, qty, price, order_type, product, tag)


# --------------------------------------------------------------------------- #
#  Fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #


def _safety(redis: Optional[FakeRedis] = None) -> ExecutionSafety:
    return ExecutionSafety(redis or FakeRedis())


def _om(store=None, redis: Optional[FakeRedis] = None, **kw) -> OrderManager:
    return OrderManager(_safety(redis), store or InMemoryOrderStore(), **kw)


def _signal(**over) -> Signal:
    base = dict(
        symbol="RELIANCE", exchange="NSE", txn_type=TransactionType.BUY,
        order_type=OrderType.LIMIT, product=Product.MIS, price=100.0,
        strategy="momo",
    )
    base.update(over)
    return Signal(**base)


def _kwargs(**over) -> dict:
    base = dict(available_cash=1_000_000.0, total_portfolio=10_000_000.0,
                max_daily_risk=50_000.0)
    base.update(over)
    return base


# =========================================================================== #
#  A. THE STATE MACHINE                                                       #
# =========================================================================== #


def test_illegal_state_transitions_raise():
    """Undeclared transitions are rejected, not silently accepted."""
    with pytest.raises(IllegalStateTransition):
        assert_transition(OrderState.INTENT_CREATED, OrderState.FILLED)
    with pytest.raises(IllegalStateTransition):
        assert_transition(OrderState.INTENT_CREATED, OrderState.SUBMITTED)
    with pytest.raises(IllegalStateTransition):
        assert_transition(OrderState.ACKNOWLEDGED, OrderState.SUBMITTED)
    with pytest.raises(IllegalStateTransition):
        assert_transition(OrderState.RISK_REJECTED, OrderState.SUBMITTED)


def test_terminal_states_are_terminal():
    for terminal in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED,
                     OrderState.EXPIRED, OrderState.RISK_REJECTED):
        rec = OrderRecord(client_order_id="C1", symbol="X", exchange="NSE",
                          side="BUY", qty=1, order_type="LIMIT", product="MIS")
        rec.state = terminal
        assert rec.is_terminal
        for target in OrderState:
            with pytest.raises(IllegalStateTransition):
                rec.transition(target, reconciled=True)


def test_UNKNOWN_can_only_be_left_by_reconciliation():
    """UNKNOWN is never optimistically resolved to filled or not-placed."""
    rec = OrderRecord(client_order_id="C1", symbol="X", exchange="NSE", side="BUY",
                      qty=10, order_type="MARKET", product="MIS")
    rec.transition(OrderState.RISK_CHECK_PENDING)
    rec.transition(OrderState.RISK_APPROVED)
    rec.transition(OrderState.SUBMITTED)
    rec.transition(OrderState.UNKNOWN, reason="timeout")
    assert rec.is_blocked
    with pytest.raises(IllegalStateTransition, match="reconciliation"):
        rec.transition(OrderState.FILLED)
    rec.transition(OrderState.FILLED, reconciled=True)
    assert rec.state is OrderState.FILLED


def test_unrecognised_broker_status_maps_to_UNKNOWN_not_success():
    assert map_broker_status(None) is OrderState.UNKNOWN
    assert map_broker_status("") is OrderState.UNKNOWN
    assert map_broker_status("SOMETHING NEW") is OrderState.UNKNOWN
    assert map_broker_status("COMPLETE", 100, 100) is OrderState.FILLED
    assert map_broker_status("COMPLETE", 40, 100) is OrderState.PARTIALLY_FILLED
    assert map_broker_status("OPEN", 40, 100) is OrderState.PARTIALLY_FILLED
    assert map_broker_status("REJECTED") is OrderState.REJECTED


def test_every_order_carries_a_client_id_from_intent_creation():
    sig = _signal()
    assert sig.client_order_id
    assert len(sig.client_order_id) <= 20        # survives Kite tag truncation
    assert sig.client_order_id.startswith("momo")
    assert make_client_order_id("x") != make_client_order_id("x")


# =========================================================================== #
#  B. THE FOUR MANDATED SCENARIOS                                             #
# =========================================================================== #


async def test_scenario_A_lost_response_then_retry_places_exactly_one_order():
    """
    Request sent, broker receives it, RESPONSE IS LOST, system retries.
    Exactly ONE broker order must exist.
    """
    store = InMemoryOrderStore()
    om = _om(store)
    sig = _signal()

    broker = FakeBroker(lose_response=True)
    first = await om.submit_order(sig, 10, broker, **_kwargs())

    # The order really is live, and the system knows it — via tag reconciliation.
    assert len(broker.placed) == 1
    assert first == "ORDER0001"
    rec = await om.get_record(sig.client_order_id)
    assert rec.state is OrderState.ACKNOWLEDGED
    assert rec.broker_order_id == "ORDER0001"

    # The retry. The reservation already exists -> no second submission.
    second = await om.submit_order(sig, 10, broker, **_kwargs())
    assert second == "ORDER0001"
    assert len(broker.placed) == 1, "a retry placed a SECOND live order"
    assert len([o for o in broker.order_book
                if o["tag"] == sig.client_order_id]) == 1


async def test_scenario_B_ambiguous_timeout_queries_before_retrying():
    """
    Network timeout, broker status unknown -> the system QUERIES the broker
    order book; it never blind-retries place_order.
    """
    store = InMemoryOrderStore()
    om = _om(store)
    sig = _signal()
    broker = FakeBroker(lose_response=True)

    await om.submit_order(sig, 10, broker, **_kwargs())
    assert broker.get_orders_calls == 1, "no reconciliation query was made"
    assert len(broker.placed) == 1, "place_order was retried"

    # And when the order book itself is unreadable, the order STAYS blocked and
    # the caller is told so, rather than being handed a fake success.
    store2 = InMemoryOrderStore()
    om2 = _om(store2)
    sig2 = _signal()
    dark = FakeBroker(lose_response=True, get_orders_fails=True)
    with pytest.raises(AmbiguousOrderError):
        await om2.submit_order(sig2, 10, dark, **_kwargs())
    rec = await om2.get_record(sig2.client_order_id)
    assert rec.state is OrderState.UNKNOWN and rec.is_blocked
    assert len(dark.placed) == 1

    # UNKNOWN blocks everything downstream for that order.
    with pytest.raises(OrderBlockedError):
        await om2.cancel_order(sig2.client_order_id, dark)
    with pytest.raises(OrderBlockedError):
        await om2.handle_fill(sig2.client_order_id,
                              {"qty": 10, "price": 100.0, "fill_id": "F1"})
    # ...and blocks a fresh order in the same instrument/strategy.
    with pytest.raises(OrderBlockedError):
        await om2.submit_order(_signal(), 10, dark, **_kwargs())
    assert len(dark.placed) == 1


async def test_deterministic_broker_rejection_is_verified_not_assumed():
    """
    A broker exception is never *assumed* to mean "nothing was sent" — kite's
    NetworkException is a plain Exception, so a client-error-shaped exception is
    no evidence.  The order book is queried; absence is what makes it REJECTED.
    """
    store = InMemoryOrderStore()
    om = _om(store)
    sig = _signal()
    broker = FakeBroker(reject_with=ValueError("InputException: bad tradingsymbol"))

    assert await om.submit_order(sig, 10, broker, **_kwargs()) is None
    assert broker.get_orders_calls == 1, "rejection was assumed, not verified"
    assert broker.placed == []
    rec = await store.get(sig.client_order_id)
    assert rec.state is OrderState.REJECTED
    assert rec.history[-1]["reconciled"] is True


async def test_scenario_C_crash_after_submission_recovers_state_on_restart():
    """
    The process dies immediately after submission.  On restart the state is
    recovered from persistence plus broker reconciliation — the orphan live
    order is found, not lost.
    """
    store = InMemoryOrderStore()          # the durable store survives the crash
    om1 = _om(store)
    sig = _signal()
    broker = FakeBroker(hang_after_accept=True)

    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await asyncio.wait_for(om1.submit_order(sig, 10, broker, **_kwargs()), 0.05)

    # The record exists and says SUBMITTED — written BEFORE the broker call.
    crashed = await store.get(sig.client_order_id)
    assert crashed is not None, "no durable record of an in-flight live order"
    assert crashed.state is OrderState.SUBMITTED
    assert crashed.broker_order_id is None
    assert len(broker.placed) == 1        # it really is live at the broker

    # Restart: a brand-new OrderManager over the same store.
    broker2 = FakeBroker(orders=broker.order_book)
    om2 = OrderManager(_safety(), store)
    recovered = await om2.recover(broker2)

    assert [r.client_order_id for r in recovered] == [sig.client_order_id]
    rec = await store.get(sig.client_order_id)
    assert rec.state is OrderState.ACKNOWLEDGED
    assert rec.broker_order_id == "ORDER0001"
    assert len(broker2.placed) == 0, "recovery placed a new order"


async def test_scenario_C_recovery_leaves_unreadable_orders_blocked():
    """If the broker cannot be reached at recovery, the order stays UNKNOWN."""
    store = InMemoryOrderStore()
    om1 = _om(store)
    sig = _signal()
    broker = FakeBroker(hang_after_accept=True)
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await asyncio.wait_for(om1.submit_order(sig, 10, broker, **_kwargs()), 0.05)

    om2 = OrderManager(_safety(), store)
    await om2.recover(FakeBroker(get_orders_fails=True))
    rec = await store.get(sig.client_order_id)
    assert rec.state is OrderState.UNKNOWN
    assert (await om2.blocked_orders())[0].client_order_id == sig.client_order_id


async def test_scenario_D_duplicate_events_cause_no_duplicate_order_or_position():
    """
    A duplicate submit event and a duplicate fill message must change nothing:
    one broker order, one position delta.
    """
    store = InMemoryOrderStore()
    om = _om(store)
    sig = _signal()
    broker = FakeBroker()

    oid1 = await om.submit_order(sig, 10, broker, **_kwargs())
    oid2 = await om.submit_order(sig, 10, broker, **_kwargs())   # replayed event
    oid3 = await om.submit_order(sig, 10, broker, **_kwargs())   # replayed again
    assert oid1 == oid2 == oid3 == "ORDER0001"
    assert len(broker.placed) == 1, "duplicate events produced duplicate orders"

    fill = {"symbol": "RELIANCE", "exchange": "NSE", "qty": 10, "price": 100.0,
            "txn_type": "BUY", "product": "MIS", "fill_id": "TRADE-1"}
    r1 = await om.handle_fill(sig.client_order_id, dict(fill))
    r2 = await om.handle_fill(sig.client_order_id, dict(fill))   # duplicate message

    assert r1["duplicate"] is False and r1["state"] == "FILLED"
    assert r2["duplicate"] is True
    rec = await store.get(sig.client_order_id)
    assert rec.filled_qty == 10, "duplicate fill double-counted the quantity"
    pos = await store.get_position("RELIANCE", "NSE", "MIS")
    assert pos["quantity"] == 10, "duplicate fill double-counted the position"
    assert await store.list_trades(sig.client_order_id) != []


# =========================================================================== #
#  C. SIZING / CAPITAL / RISK                                                 #
# =========================================================================== #


async def test_ten_million_shares_against_one_rupee_is_REJECTED():
    """
    The headline reproduction: a MARKET signal whose ``price`` is the documented
    placeholder 0.0 previously produced trade_value=0, so 10,000,000 shares
    passed the capital and exposure gates against Rs 1 of cash.
    """
    store = InMemoryOrderStore()
    om = _om(store)
    broker = FakeBroker(quote_price=100.0)
    sig = _signal(order_type=OrderType.MARKET, price=0.0)

    oid = await om.submit_order(
        sig, 10_000_000, broker, available_cash=1.0, total_portfolio=1.0)

    assert oid is None, "10,000,000 shares placed against Rs 1"
    assert broker.placed == [], "an order reached the exchange"
    rec = await om.get_record(sig.client_order_id)
    assert rec.state is OrderState.RISK_REJECTED
    # The order was sized against the LAST TRADED PRICE, not against 0.0.
    assert rec.reference_price == 100.0
    assert "capital_availability" in rec.reason
    assert "10000.00" not in rec.reason        # sanity: real numbers, not zero
    assert "1000000000.00" in rec.reason       # Rs 1,00,00,00,000 required


async def test_market_order_with_stale_or_missing_tick_is_rejected():
    """A MARKET order cannot be sized without a fresh tick, so it is refused."""
    om = _om()
    stale = FakeBroker(stale=True)
    sig = _signal(order_type=OrderType.MARKET, price=0.0)
    assert await om.submit_order(sig, 10, stale, **_kwargs()) is None
    assert stale.placed == []

    om2 = _om()
    noquote = FakeBroker(quote_missing=True)
    sig2 = _signal(order_type=OrderType.MARKET, price=0.0)
    assert await om2.submit_order(sig2, 10, noquote, **_kwargs()) is None
    assert noquote.placed == []

    om3 = _om()
    nanquote = FakeBroker(quote_price=float("nan"))
    sig3 = _signal(order_type=OrderType.MARKET, price=0.0)
    assert await om3.submit_order(sig3, 10, nanquote, **_kwargs()) is None
    assert nanquote.placed == []


async def test_insufficient_cash_oversized_position_and_exposure_all_reject():
    broker = FakeBroker(quote_price=100.0)

    # 1. insufficient cash
    om = _om()
    assert await om.submit_order(
        _signal(), 100, broker,
        available_cash=100.0, total_portfolio=10_000_000.0) is None

    # 2. too many open positions
    om = _om()
    assert await om.submit_order(
        _signal(), 10, broker,
        current_positions=[{"quantity": 1}] * 20, max_positions=20,
        **_kwargs()) is None

    # 3. single-stock exposure > 10% of the portfolio
    om = _om()
    assert await om.submit_order(
        _signal(), 5_000, broker,
        available_cash=10_000_000.0, total_portfolio=1_000_000.0) is None

    # 4. sector exposure
    om = _om()
    assert await om.submit_order(
        _signal(), 10, broker, sector="IT", sector_value=9e8,
        **_kwargs()) is None

    assert broker.placed == [], "a rejected order still reached the exchange"


async def test_daily_risk_gate_measures_risk_not_commission():
    """
    Rs 2.9 crore of exposure used to register Rs 2,029 of "risk" (the fees), so
    24 such positions fitted inside a Rs 50,000 daily cap.
    """
    qty, entry = 10_000, 2_900.0
    costs = calculate_costs("RELIANCE", qty, entry, TransactionType.BUY,
                            Product.MIS, "NSE")
    assert costs["total"] < 3_000            # the old "risk" number

    om = _om()
    # With a stop 5 rupees away the real risk is 10,000 * 5 = Rs 50,000.
    risk = om._compute_trade_risk(qty, entry, entry - 5.0)
    assert risk == pytest.approx(50_000.0)
    assert risk > costs["total"] * 20

    # With no stop at all, the risk is the full notional — never the fees.
    assert om._compute_trade_risk(qty, entry, 0.0) == pytest.approx(qty * entry)

    # And the gate now actually bites: one such position exhausts the budget.
    broker = FakeBroker(quote_price=entry)
    om2 = _om()
    sig = _signal(price=entry, stop_price=entry - 5.0)
    assert await om2.submit_order(
        sig, qty, broker, available_cash=1e9, total_portfolio=1e12,
        max_daily_risk=50_000.0, daily_risk_used=1.0) is None
    assert broker.placed == []


async def test_realised_daily_loss_is_gated_separately_from_per_trade_risk():
    om = _om()
    broker = FakeBroker(quote_price=100.0)
    # Per-trade risk is fine, but the day's realised loss has blown the floor.
    assert await om.submit_order(
        _signal(stop_price=99.0), 10, broker,
        realised_pnl_today=-60_000.0, max_daily_loss=50_000.0,
        **_kwargs()) is None
    assert broker.placed == []
    # Same order passes when the day is flat.
    ok = await om.submit_order(
        _signal(stop_price=99.0), 10, broker,
        realised_pnl_today=0.0, max_daily_loss=50_000.0, **_kwargs())
    assert ok is not None


# =========================================================================== #
#  D. THE GATES THEMSELVES                                                    #
# =========================================================================== #


async def test_a_safety_gate_that_RAISES_results_in_rejection():
    """A gate that errors must never permit the order."""
    safety = _safety()

    async def boom():
        raise AttributeError("'NoneType' object has no attribute 'startswith'")

    safety.check_position_limit = lambda *a, **k: boom()
    res = await safety.validate_order(
        broker=FakeBroker(), symbol="RELIANCE", exchange="NSE", qty=10,
        price=100.0, trade_value=1000.0, trade_risk=1000.0, strategy="momo",
        current_positions=[], open_orders=[], daily_risk_used=0.0,
        max_daily_risk=50_000.0, available_cash=1e6, total_portfolio=1e7)
    assert res.passed is False
    assert any("position_limit" in f and "AttributeError" in f
               for f in res.failed_checks), res.failed_checks
    assert res.warnings == [], "an errored gate must not degrade to a warning"


async def test_nan_and_inf_prices_are_rejected():
    safety = _safety()
    for bad in (float("nan"), float("inf"), float("-inf")):
        res = await safety.validate_order(
            broker=FakeBroker(), symbol="RELIANCE", exchange="NSE", qty=10,
            price=bad, trade_value=bad, trade_risk=bad, strategy="momo",
            current_positions=[], open_orders=[], daily_risk_used=0.0,
            max_daily_risk=50_000.0, available_cash=1e6, total_portfolio=1e7)
        assert res.passed is False, f"{bad} passed every gate"
        assert any("order_validity" in f for f in res.failed_checks)
    assert math.isnan(float("nan"))


async def test_market_closed_and_stale_data_both_block():
    safety = _safety()
    common = dict(symbol="RELIANCE", exchange="NSE", qty=10, price=100.0,
                  trade_value=1000.0, trade_risk=1000.0, strategy="momo",
                  current_positions=[], open_orders=[], daily_risk_used=0.0,
                  max_daily_risk=50_000.0, available_cash=1e6,
                  total_portfolio=1e7)

    with pytest.raises(MarketClosedError):
        await safety.check_market_status(FakeBroker(market_open=False))
    closed = await safety.validate_order(
        broker=FakeBroker(market_open=False), **common)
    assert closed.passed is False
    assert any("market_status" in f for f in closed.failed_checks)

    with pytest.raises(StaleDataError):
        await safety.check_data_freshness(FakeBroker(stale=True), "RELIANCE")
    stale = await safety.validate_order(broker=FakeBroker(stale=True), **common)
    assert stale.passed is False
    assert any("data_freshness" in f for f in stale.failed_checks)

    # ...and an order at 03:00 on a Sunday does not reach the exchange.
    om = _om()
    assert await om.submit_order(
        _signal(), 10, FakeBroker(market_open=False), **_kwargs()) is None


async def test_gate_exceptions_all_derive_from_SafetyCheckError():
    """The hierarchy bug: MarketClosed/StaleData were bare RuntimeErrors."""
    assert issubclass(MarketClosedError, SafetyCheckError)
    assert issubclass(StaleDataError, SafetyCheckError)
    assert issubclass(KillSwitchActiveError, SafetyCheckError)


async def test_duplicate_gate_survives_null_tags_and_tag_truncation():
    safety = _safety()
    common = dict(symbol="RELIANCE", exchange="NSE", qty=10, price=100.0,
                  trade_value=1000.0, trade_risk=1000.0,
                  current_positions=[], daily_risk_used=0.0,
                  max_daily_risk=50_000.0, available_cash=1e6,
                  total_portfolio=1e7)

    # Kite returns tag=None for untagged orders; that must not kill the gate.
    res = await safety.validate_order(
        broker=FakeBroker(), strategy="momo",
        open_orders=[{"symbol": "RELIANCE", "tag": None, "status": "OPEN"},
                     {"symbol": "RELIANCE", "tag": "momo", "status": "OPEN"}],
        **common)
    assert res.passed is False
    assert any("duplicate_order" in f for f in res.failed_checks)

    # A >20-char strategy name whose tag was truncated by the broker.
    long_strategy = "mean_reversion_nifty50_v3"
    res2 = await safety.validate_order(
        broker=FakeBroker(), strategy=long_strategy,
        open_orders=[{"symbol": "RELIANCE", "tag": long_strategy[:20],
                      "status": "OPEN"}],
        **common)
    assert res2.passed is False
    assert any("duplicate_order" in f for f in res2.failed_checks)


# =========================================================================== #
#  E. KILL SWITCH                                                             #
# =========================================================================== #


async def test_kill_switch_blocks_every_order_path_including_exits():
    redis = FakeRedis()
    redis.store["kill_switch"] = "1"
    safety = ExecutionSafety(redis)
    om = OrderManager(safety, InMemoryOrderStore())
    broker = FakeBroker(positions=[{"tradingsymbol": "RELIANCE", "exchange": "NSE",
                                    "product": "MIS", "quantity": 50}])

    # entry
    assert await om.submit_order(_signal(), 10, broker, **_kwargs()) is None
    # exit
    assert await om.submit_order(
        _signal(txn_type=TransactionType.SELL, intent_kind="exit"),
        10, broker, **_kwargs()) is None
    # emergency flatten
    with pytest.raises(KillSwitchActiveError):
        await om.emergency_flatten_all(broker)
    assert broker.placed == [], "kill switch active but orders were placed"


async def test_kill_switch_fails_closed_when_store_is_unreadable():
    """A Redis blip must not silently disable the halt mechanism."""
    redis = FakeRedis()
    redis.raise_on_get = True
    safety = ExecutionSafety(redis)
    with pytest.raises(KillSwitchActiveError):
        await safety.check_kill_switch()
    assert await safety.is_kill_switch_active() is True

    om = OrderManager(safety, InMemoryOrderStore())
    broker = FakeBroker()
    assert await om.submit_order(_signal(), 10, broker, **_kwargs()) is None
    assert broker.placed == []


def test_execution_safety_refuses_to_build_without_a_kill_switch_backend():
    with pytest.raises(ValueError, match="kill_switch_store"):
        ExecutionSafety(None)
    with pytest.raises(ValueError):
        ExecutionSafety(object())          # no .get()


async def test_emergency_flatten_sets_the_kill_switch_and_is_idempotent():
    redis = FakeRedis()
    safety = ExecutionSafety(redis)
    om = OrderManager(safety, InMemoryOrderStore())
    broker = FakeBroker(positions=[
        {"tradingsymbol": "RELIANCE", "exchange": "NSE",
         "product": "MIS", "quantity": 50}])

    report = await om.emergency_flatten_all(broker, flatten_session="S1")
    assert redis.store["kill_switch"] == "1", "flatten did not set the kill switch"
    assert len(report["flattened"]) == 1
    assert len(broker.placed) == 1

    # A second call is refused because the switch it just set is now active.
    with pytest.raises(KillSwitchActiveError):
        await om.emergency_flatten_all(broker, flatten_session="S1")
    assert len(broker.placed) == 1, "flatten ran twice -> 100 shares sold, 50 held"

    # Even with an explicit human override, the deterministic client order id
    # de-duplicates the same position within the same session.
    report2 = await om.emergency_flatten_all(
        broker, override_kill_switch=True, flatten_session="S1")
    assert report2["skipped"] == ["RELIANCE"]
    assert len(broker.placed) == 1


async def test_kill_switch_write_failure_is_not_reported_as_success():
    redis = FakeRedis()
    redis.raise_on_set = True
    safety = ExecutionSafety(redis)
    with pytest.raises(SafetyCheckError):
        await safety.engage_kill_switch("test")


# =========================================================================== #
#  F. FILLS, MONITORING, PERSISTENCE                                          #
# =========================================================================== #


async def test_partial_fill_is_not_marked_COMPLETE():
    store = InMemoryOrderStore()
    om = _om(store)
    broker = FakeBroker()
    sig = _signal()
    await om.submit_order(sig, 100, broker, **_kwargs())

    out = await om.handle_fill(sig.client_order_id, {
        "symbol": "RELIANCE", "exchange": "NSE", "qty": 40, "price": 100.0,
        "txn_type": "BUY", "product": "MIS", "fill_id": "T1"})
    assert out["state"] == "PARTIALLY_FILLED"
    assert out["remaining_qty"] == 60
    rec = await store.get(sig.client_order_id)
    assert rec.state is OrderState.PARTIALLY_FILLED
    assert not rec.is_terminal

    out2 = await om.handle_fill(sig.client_order_id, {
        "symbol": "RELIANCE", "exchange": "NSE", "qty": 60, "price": 101.0,
        "txn_type": "BUY", "product": "MIS", "fill_id": "T2"})
    assert out2["state"] == "FILLED"
    rec = await store.get(sig.client_order_id)
    assert rec.filled_qty == 100
    assert rec.avg_fill_price == pytest.approx(100.6)


async def test_monitor_order_distinguishes_timeout_from_terminal_state():
    store = InMemoryOrderStore()
    om = _om(store)
    broker = FakeBroker()
    sig = _signal()
    await om.submit_order(sig, 10, broker, **_kwargs())

    out = await om.monitor_order(sig.client_order_id, broker,
                                 timeout=0.05, poll_interval=0.01)
    assert out["timed_out"] is True
    assert out["is_terminal"] is False
    assert out["state"] == OrderState.ACKNOWLEDGED.value

    broker.order_book[0]["status"] = "COMPLETE"
    broker.order_book[0]["filled_quantity"] = 10
    out2 = await om.monitor_order(sig.client_order_id, broker,
                                  timeout=0.5, poll_interval=0.01)
    assert out2["timed_out"] is False and out2["is_terminal"] is True
    assert out2["state"] == OrderState.FILLED.value


async def test_order_manager_refuses_to_build_without_persistence():
    with pytest.raises(ValueError, match="durable OrderStore"):
        OrderManager(_safety(), None)
    with pytest.raises(ValueError, match="durable OrderStore"):
        OrderManager(_safety(), None, redis_client=None)


async def test_persistence_is_real_and_survives_a_new_manager():
    redis = FakeRedis()
    store = RedisOrderStore(redis)
    om = OrderManager(ExecutionSafety(redis), store)
    sig = _signal()
    broker = FakeBroker()
    oid = await om.submit_order(sig, 10, broker, **_kwargs())

    assert any(k.startswith("order:rec:") for k in redis.store), \
        "nothing was written to the durable store"
    # A fresh manager over a fresh store object reading the same redis sees it.
    om2 = OrderManager(ExecutionSafety(redis), RedisOrderStore(redis))
    rec = await om2.get_record(sig.client_order_id)
    assert rec is not None and rec.broker_order_id == oid
    assert rec.state is OrderState.ACKNOWLEDGED


async def test_redis_store_reserve_is_atomic_set_if_not_exists():
    redis = FakeRedis()
    store = RedisOrderStore(redis)
    rec = OrderRecord(client_order_id="C1", symbol="X", exchange="NSE",
                      side="BUY", qty=1, order_type="LIMIT", product="MIS")
    assert await store.reserve(rec) is True
    assert await store.reserve(rec) is False, "reservation was not exclusive"


async def test_store_write_failure_prevents_submission():
    """An unwritable store must stop the order, not be logged and ignored."""
    redis = FakeRedis()
    redis.raise_on_set = True
    om = OrderManager(ExecutionSafety(FakeRedis()), RedisOrderStore(redis))
    broker = FakeBroker()
    from app.execution.lifecycle import PersistenceError
    with pytest.raises(PersistenceError):
        await om.submit_order(_signal(), 10, broker, **_kwargs())
    assert broker.placed == []


# =========================================================================== #
#  G. STOP-LOSS PLUMBING                                                      #
# =========================================================================== #


async def test_stop_loss_trigger_price_reaches_the_broker():
    om = _om()
    broker = FakeBroker(quote_price=100.0)
    sig = _signal(order_type=OrderType.SL, price=95.0, stop_price=96.0)
    oid = await om.submit_order(sig, 10, broker, **_kwargs())
    assert oid is not None
    assert broker.placed[0]["trigger_price"] == 96.0, \
        "stop-loss order sent with no trigger price"


async def test_sl_order_is_refused_when_the_broker_cannot_carry_a_trigger():
    """
    Fail closed: rather than sending a 'stop-loss' order with no stop, refuse.
    (ZerodhaBroker.place_order does not yet accept trigger_price — see the
    FOLLOW-UP note at the top of order_manager.py.)
    """
    om = _om()
    legacy = LegacyBrokerNoTrigger(quote_price=100.0)
    sig = _signal(order_type=OrderType.SL, price=95.0, stop_price=96.0)
    assert await om.submit_order(sig, 10, legacy, **_kwargs()) is None
    assert legacy.placed == []
    rec = await om.get_record(sig.client_order_id)
    assert "trigger_price" in rec.reason


async def test_sl_order_without_a_stop_price_is_refused():
    om = _om()
    broker = FakeBroker(quote_price=100.0)
    sig = _signal(order_type=OrderType.SL_M, price=0.0, stop_price=0.0)
    assert await om.submit_order(sig, 10, broker, **_kwargs()) is None
    assert broker.placed == []
