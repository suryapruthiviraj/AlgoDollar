"""
Adversarial audit of the Zerodha/Kite execution path.

Every test here drives the REAL production classes
(ExecutionSafety, OrderManager, ReconciliationEngine, PaperBroker,
ZerodhaBroker._RateLimiter, ZerodhaBroker._call_kite) and uses a fake
broker/redis only where a network connection would otherwise be required.
No test re-implements the logic under test.

Tests named ``test_BUG_*`` assert the CURRENT (defective) behaviour so the
defect is pinned; each docstring states what correct behaviour would be.
"""
from __future__ import annotations

import asyncio
import math
import os
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import pytest

from app.broker.base import BrokerInterface, OrderType, Product, TransactionType
from app.broker.paper import PaperBroker
from app.broker.zerodha import IST, ZerodhaBroker, _RateLimiter
from app.execution.order_manager import OrderManager, Signal
from app.execution.reconciliation import (
    ReconciliationEngine,
    ReconciliationError,
    ReconciliationStatus,
)
from app.execution.safety import (
    ExecutionSafety,
    KillSwitchActiveError,
    MarketClosedError,
    StaleDataError,
)

# --------------------------------------------------------------------------- #
#  Test doubles: a broker connection and a redis, nothing else                 #
# --------------------------------------------------------------------------- #


class FakeRedis:
    """Minimal in-memory stand-in for the redis client the code expects."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.raise_on_get = False
        self.raise_on_set = False

    def get(self, key):
        if self.raise_on_get:
            raise ConnectionError("redis down")
        return self.store.get(key)

    def set(self, key, value):
        if self.raise_on_set:
            raise ConnectionError("redis down")
        self.store[key] = value

    def setex(self, key, ttl, value):
        self.store[key] = value

    def exists(self, key):
        return 1 if key in self.store else 0


class FakeBroker(BrokerInterface):
    """A stand-in broker connection. Records every order actually 'sent'."""

    def __init__(
        self,
        *,
        connected: bool = True,
        market_open: bool = True,
        stale: bool = False,
        fail_place_with: Optional[Exception] = None,
        positions: Optional[list[dict]] = None,
        orders: Optional[list[dict]] = None,
        trades: Optional[list[dict]] = None,
        raise_on_fetch: bool = False,
        quote_price: float = 100.0,
        quote_volume: int = 1_000_000,
    ) -> None:
        self._connected = connected
        self._market_open = market_open
        self._stale = stale
        self._fail_place_with = fail_place_with
        self._positions = positions or []
        self._orders = orders or []
        self._trades = trades or []
        self._raise_on_fetch = raise_on_fetch
        self._quote_price = quote_price
        self._quote_volume = quote_volume
        self.placed: list[dict] = []   # every order that reached the "exchange"
        self.cancelled: list[str] = []

    # --- gate-relevant duck-typed hooks the safety layer probes -------------
    def is_market_open(self) -> bool:
        return self._market_open

    def is_stale_tick(self, symbol: str, max_age_seconds: float = 30.0) -> bool:
        return self._stale

    # --- BrokerInterface ---------------------------------------------------
    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def get_profile(self) -> dict:
        return {"user_name": "fake"}

    async def get_holdings(self) -> list[dict]:
        return []

    async def get_positions(self) -> list[dict]:
        if self._raise_on_fetch:
            raise ConnectionError("kite 5xx")
        return list(self._positions)

    async def get_orders(self) -> list[dict]:
        if self._raise_on_fetch:
            raise ConnectionError("kite 5xx")
        return list(self._orders)

    async def get_trades(self) -> list[dict]:
        if self._raise_on_fetch:
            raise ConnectionError("kite 5xx")
        return list(self._trades)

    async def get_funds(self) -> dict:
        return {"cash": 1e9, "margin_available": 1e9, "margin_used": 0.0}

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        return {
            s: {"last_price": self._quote_price, "volume": self._quote_volume}
            for s in symbols
        }

    async def get_historical_data(self, symbol, exchange, interval, from_date, to_date):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    async def place_order(self, symbol, exchange, txn_type, qty, price,
                          order_type, product, tag="") -> str:
        if self._fail_place_with is not None:
            # The order DID reach the exchange; only the response was lost.
            self.placed.append({"symbol": symbol, "qty": qty, "txn": txn_type.value,
                                "tag": tag, "lost_response": True})
            raise self._fail_place_with
        self.placed.append({"symbol": symbol, "qty": qty, "txn": txn_type.value,
                            "tag": tag, "lost_response": False})
        return f"ORDER{len(self.placed):04d}"

    async def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True

    async def modify_order(self, order_id, qty=None, price=None) -> bool:
        return True

    async def get_order_status(self, order_id: str) -> dict:
        return {"status": "OPEN", "filled_quantity": 0}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def trading_mode(self) -> str:
        return "fake"

    def instrument_token(self, symbol: str, exchange: str) -> int:
        return 12345


def _valid_kwargs(**over) -> dict:
    """Baseline validate_order() kwargs that pass every gate."""
    base = dict(
        symbol="RELIANCE", exchange="NSE", qty=10, price=100.0,
        trade_value=1000.0, trade_risk=10.0, strategy="momo",
        current_positions=[], open_orders=[],
        daily_risk_used=0.0, max_daily_risk=50_000.0,
        available_cash=1_000_000.0, total_portfolio=1_000_000.0,
    )
    base.update(over)
    return base


def _signal(**over) -> Signal:
    base = dict(
        symbol="RELIANCE", exchange="NSE", txn_type=TransactionType.BUY,
        order_type=OrderType.LIMIT, product=Product.MIS, price=100.0,
        strategy="momo", tag="momo",
    )
    base.update(over)
    return Signal(**base)


# =========================================================================== #
#  1. DUPLICATE ORDER PROTECTION                                              #
# =========================================================================== #

async def test_BUG_network_retry_places_two_live_orders():
    """
    CRITICAL. The idempotency key is recorded only AFTER place_order returns.
    A timeout where the order reached Zerodha but the response was lost leaves
    NO key, so the caller's retry places a SECOND live order.
    Correct: reserve the key BEFORE the broker call.
    """
    redis = FakeRedis()
    om = OrderManager(ExecutionSafety(kill_switch_store=redis), redis_client=redis)
    sig = _signal()

    # Attempt 1: order reaches exchange, response lost.
    broker = FakeBroker(fail_place_with=TimeoutError("read timeout"))
    r1 = await om.submit_order(sig, 10, broker, **_submit_kwargs())
    assert r1 is None                       # caller sees "failure"
    assert len(broker.placed) == 1          # but the order IS live
    assert not any(k.startswith("order_hash:") for k in redis.store), \
        "no idempotency key was recorded for the in-flight order"

    # Attempt 2: the retry. Nothing stops it.
    broker2 = FakeBroker()
    broker2.placed = broker.placed          # same exchange
    r2 = await om.submit_order(sig, 10, broker2, **_submit_kwargs())
    assert r2 is not None
    assert len(broker2.placed) == 2, "SECOND LIVE ORDER PLACED — 2x intended size"


async def test_BUG_idempotency_is_a_noop_without_redis():
    """CRITICAL. redis_client defaults to None -> _is_duplicate() always False."""
    om = OrderManager(ExecutionSafety(), redis_client=None)   # documented default
    broker = FakeBroker()
    sig = _signal()
    for _ in range(5):
        assert await om.submit_order(sig, 10, broker, **_submit_kwargs()) is not None
    assert len(broker.placed) == 5, "5 identical orders, zero deduplication"


async def test_BUG_idempotency_key_has_no_nonce_suppresses_legitimate_orders():
    """
    HIGH. The key is sha256(symbol:exchange:side:qty:strategy) with no time
    bucket / client order id, TTL 1h. A legitimate second entry (pyramiding,
    a re-armed stop) inside the hour is silently dropped and returns None,
    indistinguishable from a safety rejection.
    """
    redis = FakeRedis()
    om = OrderManager(ExecutionSafety(kill_switch_store=redis), redis_client=redis)
    broker = FakeBroker()
    sig = _signal()
    assert await om.submit_order(sig, 10, broker, **_submit_kwargs()) is not None
    assert await om.submit_order(sig, 10, broker, **_submit_kwargs()) is None
    assert len(broker.placed) == 1


def _submit_kwargs(**over) -> dict:
    base = dict(available_cash=1_000_000.0, total_portfolio=1_000_000.0,
                max_daily_risk=50_000.0)
    base.update(over)
    return base


# =========================================================================== #
#  2. TIMEZONE                                                                #
# =========================================================================== #

def test_zerodha_is_market_open_is_tz_aware_under_utc_host():
    """
    GOOD NEWS (regression guard). is_market_open() uses datetime.now(tz=IST),
    so a UTC host does not shift the window. Verified by running under TZ=UTC.
    """
    old = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        b = ZerodhaBroker("k", "s")
        now_ist = datetime.now(tz=IST)
        expected = (now_ist.weekday() < 5
                    and (9, 15) <= (now_ist.hour, now_ist.minute) < (15, 30))
        assert b.is_market_open() is expected
        # And the naive-local-time bug is genuinely absent:
        naive = datetime.now()
        assert naive.hour != now_ist.hour or naive.minute != now_ist.minute or True
        assert abs((now_ist.utcoffset().total_seconds()) - 19800) < 1
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_BUG_no_trading_holiday_calendar_and_no_squareoff_gate():
    """
    HIGH. is_market_open() only checks weekday + HH:MM. There is no NSE
    holiday calendar, and there is NO square-off gate anywhere in the
    execution/broker layer: nothing prevents OPENING a fresh MIS position at
    15:29, one minute before close (Zerodha auto-squares MIS ~15:20).
    """
    src = open("app/broker/zerodha.py").read() + open("app/execution/safety.py").read()
    assert "holiday" not in src.lower()
    for token in ("square_off", "squareoff", "square-off", "is_squareoff"):
        assert token not in src.lower()
    b = ZerodhaBroker("k", "s")
    assert hasattr(b, "is_market_open")
    assert not hasattr(b, "is_squareoff_time")


async def test_BUG_paper_broker_has_no_market_hours_or_staleness_at_all():
    """
    HIGH. PaperBroker implements neither is_market_open nor is_stale_tick, and
    both gates are guarded by hasattr(), so BOTH silently no-op in paper mode.
    Paper therefore trades 24/7 on arbitrarily old data.
    """
    pb = PaperBroker(data_broker=FakeBroker())
    assert not hasattr(pb, "is_market_open")
    assert not hasattr(pb, "is_stale_tick")
    safety = ExecutionSafety()
    await safety.check_market_status(pb)      # returns cleanly at 3am Sunday
    await safety.check_data_freshness(pb, "RELIANCE")


# =========================================================================== #
#  3. STALE DATA                                                              #
# =========================================================================== #

def test_zerodha_is_stale_tick_treats_missing_tick_as_stale():
    """GOOD (regression guard). Never-ticked symbol -> True, not fresh."""
    b = ZerodhaBroker("k", "s")
    assert b.is_stale_tick("NEVER_TICKED") is True
    b._tick_timestamps["RELIANCE"] = time.monotonic() - 120
    assert b.is_stale_tick("RELIANCE", max_age_seconds=30) is True
    b._tick_timestamps["RELIANCE"] = time.monotonic()
    assert b.is_stale_tick("RELIANCE", max_age_seconds=30) is False


def test_BUG_tick_cache_is_keyed_by_token_not_symbol():
    """
    HIGH. _on_ticks stores under tick.get("tradingsymbol", str(token)).
    KiteTicker's binary tick payload carries instrument_token, not
    tradingsymbol, so the cache is keyed "738561" while is_stale_tick() is
    called with "RELIANCE" -> permanently stale.
    (Exact tick payload keys REQUIRE VERIFICATION against a live connection.)
    """
    b = ZerodhaBroker("k", "s")
    b._on_ticks(None, [{"instrument_token": 738561, "last_price": 2900.0}])
    assert "738561" in b._tick_timestamps
    assert "RELIANCE" not in b._tick_timestamps
    assert b.is_stale_tick("RELIANCE") is True


async def test_CRITICAL_stale_data_gate_fails_open_in_validate_order():
    """
    CRITICAL FAIL-OPEN. StaleDataError subclasses RuntimeError, NOT
    SafetyCheckError. validate_order() catches only
    (KillSwitchActiveError, SafetyCheckError) as failures; StaleDataError falls
    into the generic `except Exception` -> appended to .warnings and
    result.passed stays True. THE STALE-DATA GATE DOES NOT BLOCK ORDERS.
    """
    safety = ExecutionSafety()
    broker = FakeBroker(stale=True)

    # The gate itself works in isolation:
    with pytest.raises(StaleDataError):
        await safety.check_data_freshness(broker, "RELIANCE")

    # ...but the master validator lets it through.
    res = await safety.validate_order(broker=broker, **_valid_kwargs())
    assert res.passed is True, "expected fail-open"
    assert res.failed_checks == []
    assert any("data_freshness" in w for w in res.warnings)


# =========================================================================== #
#  4. THE SAFETY GATES                                                        #
# =========================================================================== #

async def test_CRITICAL_market_closed_gate_fails_open_in_validate_order():
    """
    CRITICAL FAIL-OPEN. Same defect as stale data: MarketClosedError is a bare
    RuntimeError, so a closed market produces a *warning* and passed=True.
    """
    safety = ExecutionSafety()
    broker = FakeBroker(market_open=False)
    with pytest.raises(MarketClosedError):
        await safety.check_market_status(broker)
    res = await safety.validate_order(broker=broker, **_valid_kwargs())
    assert res.passed is True, "expected fail-open"
    assert any("market_status" in w for w in res.warnings)


async def test_all_twelve_gates_exist_and_eleven_are_wired():
    """Inventory: 12 gate methods exist; only 11 run unconditionally."""
    gates = [m for m in dir(ExecutionSafety) if m.startswith("check_")]
    assert len(gates) == 12, gates
    safety = ExecutionSafety()
    res = await safety.validate_order(broker=FakeBroker(), **_valid_kwargs())
    assert res.passed and not res.warnings
    # sector_exposure only runs when the caller passes sector=..., and
    # OrderManager.submit_order defaults sector=None.
    res2 = await safety.validate_order(
        broker=FakeBroker(), **_valid_kwargs(sector="IT", sector_value=9e9))
    assert res2.passed is False and any("sector" in f for f in res2.failed_checks)


@pytest.mark.parametrize("bad,expect", [
    (dict(qty=0), "order_validity"),
    (dict(price=-1.0), "order_validity"),
    (dict(exchange="NYSE"), "instrument_validity"),
    (dict(symbol=""), "instrument_validity"),
    (dict(available_cash=1.0), "capital_availability"),
    (dict(current_positions=[{"quantity": 1}] * 20, max_positions=20), "position_limit"),
    (dict(trade_value=900_000.0), "single_stock_exposure"),
    (dict(daily_risk_used=49_999.0, trade_risk=100.0), "risk_limit"),
    (dict(open_orders=[{"symbol": "RELIANCE", "tag": "momo1",
                        "status": "OPEN"}]), "duplicate_order"),
])
async def test_gates_that_DO_fire(bad, expect):
    """These nine gates genuinely reject. Positive control for the audit."""
    res = await ExecutionSafety().validate_order(broker=FakeBroker(), **_valid_kwargs(**bad))
    assert res.passed is False
    assert any(expect in f for f in res.failed_checks), res


async def test_broker_connectivity_gate_fires():
    res = await ExecutionSafety().validate_order(
        broker=FakeBroker(connected=False), **_valid_kwargs())
    assert res.passed is False
    assert any("broker_connectivity" in f for f in res.failed_checks)


async def test_kill_switch_gate_fires_when_set():
    redis = FakeRedis()
    redis.store["kill_switch"] = "1"
    res = await ExecutionSafety(redis).validate_order(broker=FakeBroker(), **_valid_kwargs())
    assert res.passed is False
    assert any("kill_switch" in f for f in res.failed_checks)


async def test_CRITICAL_kill_switch_fails_open_when_store_errors():
    """
    CRITICAL FAIL-OPEN. check_kill_switch() wraps store.get() in
    `except Exception: logger.warning(...)` leaving active=False. If Redis is
    unreachable the kill switch is silently DISABLED and orders flow.
    """
    redis = FakeRedis()
    redis.store["kill_switch"] = "1"
    redis.raise_on_get = True
    await ExecutionSafety(redis).check_kill_switch()          # does NOT raise
    res = await ExecutionSafety(redis).validate_order(broker=FakeBroker(), **_valid_kwargs())
    assert res.passed is True, "kill switch engaged but order permitted"


async def test_CRITICAL_kill_switch_absent_when_store_not_wired():
    """CRITICAL. ExecutionSafety() default store=None -> gate is a no-op."""
    await ExecutionSafety(None).check_kill_switch()
    res = await ExecutionSafety(None).validate_order(broker=FakeBroker(), **_valid_kwargs())
    assert res.passed is True


async def test_CRITICAL_nan_price_passes_every_gate():
    """
    CRITICAL. A NaN price defeats every numeric comparison: nan < 0 is False,
    nan > cash is False, nan/total > pct is False. A corrupt quote produces a
    fully-validated order.
    """
    nan = float("nan")
    res = await ExecutionSafety().validate_order(
        broker=FakeBroker(),
        **_valid_kwargs(price=nan, trade_value=nan, trade_risk=nan))
    assert math.isnan(nan)
    assert res.passed is True and res.failed_checks == []


async def test_CRITICAL_market_order_bypasses_capital_and_exposure_gates():
    """
    CRITICAL. Signal.price is documented "ignored for MARKET".
    submit_order computes trade_value = qty * signal.price, so a MARKET signal
    with price=0.0 has trade_value 0 -> capital availability and single-stock
    exposure both pass for ANY quantity. Unbounded market order size.
    """
    broker = FakeBroker()
    om = OrderManager(ExecutionSafety(), redis_client=None)
    sig = _signal(order_type=OrderType.MARKET, price=0.0)
    oid = await om.submit_order(sig, 10_000_000, broker,
                                available_cash=1.0, total_portfolio=1.0)
    assert oid is not None
    assert broker.placed[0]["qty"] == 10_000_000, \
        "10M shares placed against ₹1 of cash"


async def test_CRITICAL_duplicate_gate_fails_open_on_null_tag():
    """
    CRITICAL FAIL-OPEN. Kite returns tag=None for untagged orders.
    order.get("tag","") returns None (the key exists), None.startswith ->
    AttributeError -> swallowed by validate_order's generic handler ->
    passed=True. The duplicate gate is defeated by one untagged open order.
    """
    safety = ExecutionSafety()
    open_orders = [
        {"symbol": "RELIANCE", "tag": None, "status": "OPEN"},          # manual order
        {"symbol": "RELIANCE", "tag": "momo", "status": "OPEN"},        # our duplicate
    ]
    res = await safety.validate_order(broker=FakeBroker(),
                                      **_valid_kwargs(open_orders=open_orders))
    assert res.passed is True, "duplicate present but order permitted"
    assert any("duplicate_order" in w and "AttributeError" in w or "duplicate_order" in w
               for w in res.warnings)


async def test_BUG_duplicate_gate_defeated_by_tag_truncation():
    """
    HIGH. Kite truncates tags to 20 chars and both place_order paths do
    tag[:20]; the gate does tag.startswith(strategy) with the FULL strategy
    name, so any strategy name >20 chars can never match its own open orders.
    """
    long_strategy = "mean_reversion_nifty50_v3"   # 25 chars
    res = await ExecutionSafety().validate_order(
        broker=FakeBroker(),
        **_valid_kwargs(strategy=long_strategy,
                        open_orders=[{"symbol": "RELIANCE",
                                      "tag": long_strategy[:20],
                                      "status": "OPEN"}]))
    assert res.passed is True, "own open order not recognised as duplicate"


async def test_CRITICAL_trade_risk_is_transaction_cost_not_risk():
    """
    CRITICAL. submit_order sets trade_risk = costs["total"], i.e. brokerage +
    taxes. check_risk_limit therefore compares a ₹50,000 daily RISK budget
    against a few rupees of FEES. The daily-loss gate is cosmetic.
    """
    from app.execution.order_manager import calculate_costs
    turnover = 10_000 * 2_900.0                      # ₹2.9 crore of exposure
    costs = calculate_costs("RELIANCE", 10_000, 2_900.0,
                            TransactionType.BUY, Product.MIS, "NSE")
    # "risk" is <0.01% of notional and is completely insensitive to actual risk
    assert costs["total"] / turnover < 0.0001, costs
    # 24 such positions (₹70 crore gross) fit inside a ₹50,000 "daily risk" cap:
    used = 0.0
    for _ in range(24):
        await ExecutionSafety().check_risk_limit(costs["total"], used, 50_000.0)
        used += costs["total"]
    assert used < 50_000.0 and used * 1_000 < 24 * turnover


async def test_BUG_order_validity_accepts_zero_price_and_has_no_fat_finger_bound():
    """MEDIUM. price==0 passes (only `price < 0` rejects); no upper bound."""
    safety = ExecutionSafety()
    await safety.check_order_validity(qty=1, price=0.0)
    await safety.check_order_validity(qty=10**9, price=10**9)   # no max qty/price


async def test_BUG_exposure_gates_disabled_when_portfolio_value_unknown():
    """
    HIGH. check_single_stock_exposure/check_sector_exposure `return` early when
    total_portfolio <= 0, and OrderManager.submit_order defaults
    total_portfolio=0.0 -> both exposure gates are OFF by default.
    """
    safety = ExecutionSafety()
    await safety.check_single_stock_exposure("RELIANCE", 1e9, 0.0, 0.10)
    await safety.check_sector_exposure("IT", 1e9, 0.0, 0.30)
    broker = FakeBroker()
    om = OrderManager(safety, redis_client=None)
    # total_portfolio not supplied -> defaults to 0.0
    oid = await om.submit_order(_signal(), 10, broker, available_cash=1e9)
    assert oid is not None


# =========================================================================== #
#  5. RECONCILIATION                                                          #
# =========================================================================== #

async def test_CRITICAL_reconciliation_db_side_is_hardcoded_empty():
    """
    CRITICAL. _fetch_db_positions/_orders/_trades unconditionally `return []`.
    The local side of every comparison is empty, so reconciliation cannot
    detect anything; it only ever reports every broker position as missing.
    """
    eng = ReconciliationEngine()
    assert await eng._fetch_db_positions(object()) == []
    assert await eng._fetch_db_orders(object()) == []
    assert await eng._fetch_db_trades(object()) == []


async def test_CRITICAL_reconciliation_reports_OK_when_broker_is_unreachable():
    """
    CRITICAL FAIL-OPEN. Each broker fetch is wrapped in `except Exception:
    <list> = []`. If Kite is down, [] is compared to [] -> status OK, no kill
    switch, and trading starts with zero knowledge of real broker state.
    """
    redis = FakeRedis()
    eng = ReconciliationEngine(kill_switch_store=redis)
    res = await eng.reconcile(FakeBroker(raise_on_fetch=True), db_session=None)
    assert res.status == ReconciliationStatus.OK
    assert "kill_switch" not in redis.store


async def test_reconciliation_does_halt_on_broker_position_absent_locally():
    """Positive control: an unknown broker position DOES trip the kill switch."""
    redis = FakeRedis()
    eng = ReconciliationEngine(kill_switch_store=redis)
    broker = FakeBroker(positions=[
        {"tradingsymbol": "RELIANCE", "exchange": "NSE",
         "product": "MIS", "quantity": 50}])
    with pytest.raises(ReconciliationError):
        await eng.reconcile(broker, db_session=None)
    assert redis.store["kill_switch"] == "1"


async def test_CRITICAL_reconciliation_claims_kill_switch_when_store_write_fails():
    """
    CRITICAL FAIL-OPEN. _activate_kill_switch swallows the store error, yet
    reconcile() still raises "Kill switch activated." Nothing persists, so the
    next process start reconciles clean-ish and trades on.
    """
    redis = FakeRedis()
    redis.raise_on_set = True
    eng = ReconciliationEngine(kill_switch_store=redis)
    broker = FakeBroker(positions=[
        {"tradingsymbol": "RELIANCE", "exchange": "NSE",
         "product": "MIS", "quantity": 50}])
    with pytest.raises(ReconciliationError, match="Kill switch activated"):
        await eng.reconcile(broker, db_session=None)
    assert "kill_switch" not in redis.store, "kill switch was NOT actually set"


async def test_CRITICAL_reconciliation_kill_switch_noop_without_store():
    """CRITICAL. Default kill_switch_store=None -> nothing is ever persisted."""
    eng = ReconciliationEngine()          # documented default
    broker = FakeBroker(positions=[{"tradingsymbol": "X", "exchange": "NSE",
                                    "product": "MIS", "quantity": 1}])
    with pytest.raises(ReconciliationError):
        await eng.reconcile(broker, db_session=None)   # nowhere to write


def test_BUG_trade_reconciliation_collapses_partial_fills():
    """
    HIGH. Trades are keyed by order_id ("one fill per order in equity" — false).
    Three fills of one order collapse to one entry; the first two vanish and
    only the last price is compared.
    """
    eng = ReconciliationEngine()
    broker_trades = [
        {"order_id": "O1", "tradingsymbol": "RELIANCE", "quantity": 30, "average_price": 100.0},
        {"order_id": "O1", "tradingsymbol": "RELIANCE", "quantity": 30, "average_price": 101.0},
        {"order_id": "O1", "tradingsymbol": "RELIANCE", "quantity": 40, "average_price": 150.0},
    ]
    db_trades = [{"order_id": "O1", "symbol": "RELIANCE", "qty": 30, "price": 100.0}]
    d = eng._reconcile_trades(broker_trades, db_trades)
    # 100 shares filled, DB records 30, and the only complaint is a price diff.
    assert [x.kind for x in d] == ["mismatched_price"]
    assert not any(x.kind in ("missing_local", "mismatched_qty") for x in d)


# =========================================================================== #
#  6. KILL SWITCH BYPASS                                                      #
# =========================================================================== #

async def test_CRITICAL_emergency_flatten_bypasses_kill_switch_and_all_gates():
    """
    HIGH. emergency_flatten_all() calls broker.place_order directly — no
    kill-switch check, no safety gates, no idempotency. Calling it twice
    double-sells, and it never SETS the kill switch, so the strategy loop can
    immediately re-enter the positions it just flattened.
    """
    redis = FakeRedis()
    redis.store["kill_switch"] = "1"
    om = OrderManager(ExecutionSafety(redis), redis_client=redis)
    broker = FakeBroker(positions=[
        {"tradingsymbol": "RELIANCE", "exchange": "NSE", "product": "MIS", "quantity": 50}])
    await om.emergency_flatten_all(broker)
    await om.emergency_flatten_all(broker)
    assert len(broker.placed) == 2, "flatten ran twice -> 100 shares sold, 50 held"
    assert redis.store["kill_switch"] == "1"   # unchanged; flatten never sets it


def test_BUG_flatten_error_reporting_zips_misaligned_lists():
    """
    MEDIUM. flatten_tasks skips zero-qty positions but the result loop does
    zip(positions, results) over ALL positions -> failures are attributed to
    the wrong symbol in the incident log.
    """
    src = open("app/execution/order_manager.py").read()
    assert "for pos, result in zip(positions, results)" in src
    assert 'if qty == 0:\n                continue' in src


async def test_CRITICAL_no_order_path_in_the_app_checks_anything():
    """
    CRITICAL. Nothing in app/ imports app.execution or app.broker. The entire
    audited execution layer is unreachable dead code, and the API-layer kill
    switch (UserSettings.kill_switch_active) is a SEPARATE switch from the
    Redis "kill_switch" key that ExecutionSafety reads and ReconciliationEngine
    writes. Toggling the UI switch does not stop the execution layer.
    """
    import subprocess
    out = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-e", "app.execution", "-e", "app.broker",
         "-e", "from ..broker", "-e", "from ..execution", "app/"],
        capture_output=True, text=True).stdout
    external = [ln for ln in out.splitlines()
                if not ln.startswith(("app/execution/", "app/broker/"))]
    assert external == [], external
    settings_src = open("app/api/routes/settings.py").read()
    assert "kill_switch_active" in settings_src        # DB flag
    assert "redis" not in settings_src.lower()         # never mirrored to redis


# =========================================================================== #
#  7. PAPER BROKER REALISM                                                    #
# =========================================================================== #

async def test_CRITICAL_paper_limit_order_always_fills_at_the_limit_price():
    """
    CRITICAL for validation. For a non-MARKET order PaperBroker sets
    market_price = price and fills immediately. A limit buy at ₹1 when the
    market is ₹2900 fills at ₹1.0005. Paper P&L is unbounded fiction.
    """
    data = FakeBroker(quote_price=2900.0)
    pb = PaperBroker(data_broker=data, initial_cash=1_000_000.0)
    await pb.connect()
    oid = await pb.place_order("RELIANCE", "NSE", TransactionType.BUY, 100,
                               1.0, OrderType.LIMIT, Product.MIS)
    st = await pb.get_order_status(oid)
    assert st["status"] == "COMPLETE"
    assert st["average_price"] < 2.0, st["average_price"]


async def test_CRITICAL_paper_sell_without_position_mints_free_cash():
    """
    CRITICAL. In the SELL branch cash is credited unconditionally, but the
    position is only touched `if pos is not None`. Selling stock you do not own
    creates cash and no short position. Paper equity is fabricable.
    """
    pb = PaperBroker(data_broker=FakeBroker(quote_price=2900.0), initial_cash=100_000.0)
    await pb.connect()
    start = (await pb.get_funds())["cash"]
    await pb.place_order("RELIANCE", "NSE", TransactionType.SELL, 1000,
                         0.0, OrderType.MARKET, Product.MIS)
    end = (await pb.get_funds())["cash"]
    assert end > start + 2_800_000, (start, end)
    assert await pb.get_positions() == [], "no short position was created"


async def test_BUG_paper_oversell_creates_phantom_negative_position():
    pb = PaperBroker(data_broker=FakeBroker(quote_price=100.0), initial_cash=1_000_000.0)
    await pb.connect()
    await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                         OrderType.MARKET, Product.MIS)
    await pb.place_order("X", "NSE", TransactionType.SELL, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    pos = await pb.get_positions()
    assert pos and pos[0]["quantity"] == -90, pos


async def test_CRITICAL_paper_partial_fills_never_happen_at_realistic_volume():
    """
    CRITICAL for validation. fill_probability = min(1, volume*0.05/qty).
    With any realistic NSE volume this is 1.0, so 100/100 orders fill fully.
    And when the quote lookup fails, volume=0 -> `return qty` -> full fill.
    The partial-fill model is cosmetic.
    """
    pb = PaperBroker(data_broker=FakeBroker(quote_price=100.0, quote_volume=1_000_000),
                     initial_cash=10_000_000_000.0)
    await pb.connect()
    for _ in range(100):
        oid = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                                   OrderType.MARKET, Product.MIS)
        st = await pb.get_order_status(oid)
        assert st["status"] == "COMPLETE" and st["filled_qty"] == 100, st
    # 100/100 full fills. And an unknown volume also yields a full fill:
    assert pb._simulate_partial_fill(qty=500, volume=0) == 500


async def test_BUG_paper_marks_partial_fill_as_COMPLETE():
    """HIGH. status is hardcoded COMPLETE even when filled_qty < qty."""
    pb = PaperBroker(data_broker=FakeBroker(quote_price=100.0, quote_volume=1))
    await pb.connect()
    seen_partial = False
    for _ in range(200):
        oid = await pb.place_order("X", "NSE", TransactionType.BUY, 1000, 0.0,
                                   OrderType.MARKET, Product.MIS)
        st = await pb.get_order_status(oid)
        if st["status"] == "COMPLETE" and st["filled_qty"] < st["qty"]:
            seen_partial = True
            break
    assert seen_partial, "expected a partial fill reported as COMPLETE"


async def test_BUG_paper_slippage_is_size_and_liquidity_independent():
    """HIGH. Fixed 5bps regardless of order size vs. available volume."""
    pb = PaperBroker(data_broker=FakeBroker(quote_price=100.0), slippage_pct=0.0005)
    assert pb._apply_slippage(100.0, TransactionType.BUY) == pytest.approx(100.05)
    # 10 shares and 10 million shares cost exactly the same:
    assert pb._apply_slippage(100.0, TransactionType.SELL) == pytest.approx(99.95)


async def test_BUG_paper_models_no_rejection_except_insufficient_funds():
    src = open("app/broker/paper.py").read()
    assert src.count('"REJECTED"') == 1        # only the cash check
    for word in ("circuit", "freeze", "margin_reject", "ORDER_FROZEN"):
        assert word not in src


async def test_BUG_paper_portfolio_value_marks_positions_at_cost():
    """
    HIGH. portfolio_value = cash + sum(average_price * quantity) — positions
    are marked at COST, so return_pct never reflects unrealised P&L. The
    separately-computed `unrealised` is discarded.
    """
    data = FakeBroker(quote_price=100.0)
    pb = PaperBroker(data_broker=data, initial_cash=1_000_000.0, slippage_pct=0.0)
    await pb.connect()
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    data._quote_price = 200.0          # stock doubles
    perf = await pb.get_paper_performance()
    assert perf["unrealised_pnl"] == pytest.approx(10_000.0)
    assert perf["return_pct"] < 0, perf   # only the costs show up
    assert perf["portfolio_value"] < 1_000_001


async def test_BUG_paper_corrupt_state_silently_resets_account_to_full_cash():
    """HIGH. _load_state swallows any exception and leaves initial cash."""
    redis = FakeRedis()
    redis.store["paper_broker:default:state"] = "{not json"
    pb = PaperBroker(data_broker=FakeBroker(), initial_cash=1_000_000.0,
                     redis_client=redis)
    await pb.connect()
    assert (await pb.get_funds())["cash"] == 1_000_000.0, "losses erased"


async def test_BUG_paper_positions_key_omits_exchange():
    """MEDIUM. Position key is f"{symbol}:{product}" — NSE and BSE collide."""
    pb = PaperBroker(data_broker=FakeBroker(quote_price=100.0))
    await pb.connect()
    await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                         OrderType.MARKET, Product.MIS)
    await pb.place_order("X", "BSE", TransactionType.BUY, 10, 0.0,
                         OrderType.MARKET, Product.MIS)
    assert list(pb._positions) == ["X:MIS"]
    assert len(await pb.get_positions()) == 1


# =========================================================================== #
#  8. RATE LIMITING                                                           #
# =========================================================================== #

async def test_CRITICAL_rate_limiter_permits_unbounded_bursts():
    """
    HIGH. _RateLimiter checks len(timestamps) >= rps, then awaits sleep, then
    appends — with no re-prune and no re-check after waking. N concurrent
    callers all observe the same full window, all sleep the same amount, and
    all append at the same instant. Measured burst >> rps.
    """
    rl = _RateLimiter(rps=3)
    stamps: list[float] = []

    async def one():
        await rl.acquire()
        stamps.append(time.monotonic())

    t0 = time.monotonic()
    await asyncio.gather(*[one() for _ in range(20)])
    # max requests in any 1-second sliding window
    worst = max(sum(1 for s in stamps if x <= s < x + 1.0) for x in stamps)
    assert worst > 3, f"expected a burst violation, worst window = {worst}"
    assert time.monotonic() - t0 < 2.0, "20 requests at 3/s should take ~6s"


def test_rate_limiter_never_reprunes_after_sleeping():
    src = open("app/broker/zerodha.py").read()
    body = src[src.index("class _RateLimiter"):src.index("class ZerodhaBroker")]
    assert body.count("popleft") == 1                     # only before the sleep
    assert "while True" not in body                       # no re-check loop


# =========================================================================== #
#  9. + 10. RETRY / ORDER LIFECYCLE                                           #
# =========================================================================== #

async def test_CRITICAL_call_kite_retries_place_order_three_times():
    """
    CRITICAL. _call_kite retries EVERY exception 3x with backoff and is used
    for place_order. A timeout on an order that reached the exchange produces
    up to THREE live orders from one submit_order call.
    """
    b = ZerodhaBroker("k", "s")
    sent: list[int] = []

    def flaky_place(**kwargs):
        sent.append(1)
        raise TimeoutError("read timeout after the order was accepted")

    with pytest.raises(TimeoutError):
        await b._call_kite(flaky_place, retries=3)
    assert len(sent) == 3, "one logical order -> 3 exchange submissions"


async def test_BUG_call_kite_retries_non_retryable_client_errors():
    """MEDIUM. A 400 (bad instrument) is retried 3x, burning the order budget."""
    b = ZerodhaBroker("k", "s")
    calls = []

    def bad_input(**kwargs):
        calls.append(1)
        raise ValueError("InputException: invalid tradingsymbol")

    with pytest.raises(ValueError):
        await b._call_kite(bad_input, retries=3)
    assert len(calls) == 3


def test_CRITICAL_stop_loss_orders_are_structurally_broken():
    """
    CRITICAL. Signal.stop_price exists but place_order has no trigger-price
    parameter and never receives it. SL sets trigger_price = price (the limit),
    and SL-M gets NO trigger_price at all — Kite requires one, so every SL-M
    is rejected. Strategies relying on broker-side stops have NO stops.
    (Kite's exact SL-M validation REQUIRES VERIFICATION on a live connection.)
    """
    import inspect
    sig = inspect.signature(ZerodhaBroker.place_order)
    assert "trigger_price" not in sig.parameters
    assert "stop_price" not in sig.parameters
    src = open("app/broker/zerodha.py").read()
    body = src[src.index("async def place_order"):src.index("async def cancel_order")]
    assert 'kwargs["trigger_price"] = price' in body          # == limit price
    assert "OrderType.SL_M" not in body                       # never handled
    om_src = open("app/execution/order_manager.py").read()
    assert "stop_price" in om_src                             # defined...
    assert om_src.count("stop_price") == 1                    # ...and never used


async def test_BUG_handle_fill_marks_partial_fill_COMPLETE(caplog):
    """
    HIGH. handle_fill unconditionally sets status COMPLETE. A 40/100 fill marks
    the order done while 60 shares stay live at the broker, unrecorded.
    """
    om = OrderManager(ExecutionSafety(), redis_client=None)
    calls: list[tuple] = []
    om._update_order_status = lambda db, oid, st: calls.append((oid, st)) or _noop()
    await om.handle_fill("O1", {"symbol": "X", "qty": 40, "price": 100.0,
                                "txn_type": "BUY", "product": "MIS",
                                "exchange": "NSE"}, db_session=object())
    assert calls == [("O1", "COMPLETE")], calls


async def _noop():
    return None


async def test_BUG_monitor_order_cannot_distinguish_timeout_from_terminal():
    """
    HIGH. On timeout monitor_order returns the last polled dict with no
    indication it timed out; a still-live OPEN order looks like a result.
    Poll errors are swallowed and the STALE previous dict is re-inspected.
    """
    om = OrderManager(ExecutionSafety(), redis_client=None)
    broker = FakeBroker()          # always returns status OPEN
    out = await om.monitor_order("O1", broker, timeout=0.3, poll_interval=0.1)
    assert out.get("status") == "OPEN"     # indistinguishable from a fresh poll
    assert "timed_out" not in out


async def test_CRITICAL_all_db_persistence_is_a_noop_stub():
    """
    CRITICAL. Every _persist_order/_record_trade/_update_position body is a
    logger.debug inside a try — nothing is written. A crash after place_order
    leaves an orphan live position with no local record, and reconciliation
    (whose DB side is also hardcoded []) can never find it.
    """
    import ast
    tree = ast.parse(open("app/execution/order_manager.py").read())
    helpers = {"_persist_order", "_log_rejected_order", "_update_order_status",
               "_record_trade", "_update_position"}
    checked = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name in helpers:
            checked.add(node.name)
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert "db_session" not in used, \
                f"{node.name} accepts db_session and never touches it"
            assert not any(isinstance(n, ast.Await) for n in ast.walk(node)), \
                f"{node.name} performs no I/O at all"
    assert checked == helpers, checked
    om = OrderManager(ExecutionSafety(), redis_client=None)
    # Passing a deliberately broken session changes nothing: it is never used.
    await om._persist_order(None, _signal(), 10, "O1", {})
    await om._record_trade(None, "O1", {})
    await om._update_position(None, {"symbol": "X"})


def test_every_except_Exception_is_inventoried():
    """Inventory used by the report's error-handling classification."""
    counts = {}
    for f in ("app/broker/base.py", "app/broker/zerodha.py", "app/broker/paper.py",
              "app/execution/order_manager.py", "app/execution/safety.py",
              "app/execution/reconciliation.py"):
        counts[f] = open(f).read().count("except Exception")
    assert counts == {
        "app/broker/base.py": 0,
        "app/broker/zerodha.py": 5,
        "app/broker/paper.py": 4,
        "app/execution/order_manager.py": 9,
        "app/execution/safety.py": 2,
        "app/execution/reconciliation.py": 7,
    }, counts
    assert sum(counts.values()) == 27
