"""
test_paper_broker.py — regression + invariant suite for the REAL PaperBroker.

Every test here drives ``app.broker.paper.PaperBroker`` directly.  The only
test double is a price feed (``FakeFeed``) and an in-memory Redis
(``FakeRedis``), because a network connection would otherwise be required.
Nothing in this file re-implements the accounting or the execution model — the
previous suite did exactly that and therefore verified nothing.

Test naming
-----------
``test_D1_*`` .. ``test_D6_*``  pin the six audited defects as regressions.
``test_inv_*``                  assert the accounting invariants.
``test_exec_*``                 assert the execution model.
``test_hours_*``                assert IST market-session handling.
``test_state_*``                assert fail-closed persistence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import pytest

from app.backtesting.costs import ZerodhaCostModel
from app.broker.base import BrokerInterface, OrderType, Product, TransactionType
from app.broker.paper import (
    IST,
    SHORT_SELLING_SUPPORTED,
    AccountingInvariantError,
    OrderStatus,
    PaperBroker,
    PaperBrokerStateError,
    RejectReason,
    ensure_aware,
    to_paise,
)

# --------------------------------------------------------------------------- #
#  Fixed instants (a Tuesday that is not an NSE holiday in 2025)               #
# --------------------------------------------------------------------------- #

OPEN_IST = datetime(2025, 6, 10, 11, 0, tzinfo=IST)          # 05:30 UTC
LATE_IST = datetime(2025, 6, 10, 15, 25, tzinfo=IST)         # inside square-off
AFTER_IST = datetime(2025, 6, 10, 20, 0, tzinfo=IST)         # 14:30 UTC
SUNDAY_IST = datetime(2025, 6, 8, 11, 0, tzinfo=IST)
HOLIDAY_IST = datetime(2025, 8, 15, 11, 0, tzinfo=IST)       # Independence Day

ZERO_COST_MODEL = ZerodhaCostModel(config={
    "intraday_brokerage_pct": 0.0, "intraday_brokerage_max_rs": 0.0,
    "delivery_brokerage_pct": 0.0, "intraday_stt_sell_pct": 0.0,
    "delivery_stt_buy_pct": 0.0, "delivery_stt_sell_pct": 0.0,
    "nse_exchange_charge_pct": 0.0, "sebi_charge_pct": 0.0, "gst_pct": 0.0,
    "intraday_stamp_duty_pct": 0.0, "delivery_stamp_duty_pct": 0.0,
    "dp_charge_rs": 0.0,
})


# --------------------------------------------------------------------------- #
#  Test doubles: a price feed and a redis.  Nothing else.                      #
# --------------------------------------------------------------------------- #

class FakeFeed(BrokerInterface):
    """A price feed. Records nothing, decides nothing — it just quotes."""

    def __init__(
        self,
        price: float = 100.0,
        volume: int = 1_000_000,
        *,
        prices: Optional[dict[str, float]] = None,
        volumes: Optional[dict[str, int]] = None,
        timestamp: Optional[datetime] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        fail: bool = False,
    ) -> None:
        self.default_price = price
        self.default_volume = volume
        self.prices = dict(prices or {})
        self.volumes = dict(volumes or {})
        self.timestamp = timestamp
        self.bid = bid
        self.ask = ask
        self.fail = fail
        self.quote_calls = 0

    def set_price(self, symbol: str, price: float) -> None:
        self.prices[symbol] = price

    def set_volume(self, symbol: str, volume: int) -> None:
        self.volumes[symbol] = volume

    def price_of(self, symbol: str) -> float:
        return self.prices.get(symbol, self.default_price)

    # --- BrokerInterface --------------------------------------------------
    async def connect(self) -> None:  # pragma: no cover - trivial
        ...

    async def disconnect(self) -> None:  # pragma: no cover - trivial
        ...

    async def get_profile(self) -> dict:
        return {"user_name": "feed"}

    async def get_holdings(self) -> list[dict]:
        return []

    async def get_positions(self) -> list[dict]:
        return []

    async def get_orders(self) -> list[dict]:
        return []

    async def get_trades(self) -> list[dict]:
        return []

    async def get_funds(self) -> dict:
        return {}

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        self.quote_calls += 1
        if self.fail:
            raise ConnectionError("feed down")
        out: dict[str, dict] = {}
        for key in symbols:
            _, sym = key.split(":", 1)
            price = self.price_of(sym)
            if price is None:
                continue
            q: dict[str, Any] = {
                "last_price": price,
                "volume": self.volumes.get(sym, self.default_volume),
                "ohlc": {"open": price, "high": price * 1.01,
                         "low": price * 0.99, "close": price},
            }
            if self.timestamp is not None:
                q["timestamp"] = self.timestamp
            if self.bid is not None and self.ask is not None:
                q["bid"], q["ask"] = self.bid, self.ask
            out[key] = q
        return out

    async def get_historical_data(self, *a, **k) -> pd.DataFrame:
        return pd.DataFrame()

    async def place_order(self, *a, **k) -> str:  # pragma: no cover
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def modify_order(self, order_id, qty=None, price=None) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def get_order_status(self, order_id: str) -> dict:  # pragma: no cover
        return {}

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def trading_mode(self) -> str:
        return "live"

    def instrument_token(self, symbol: str, exchange: str) -> int:
        return abs(hash((symbol, exchange))) % 10_000_000


class FakeRedis:
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


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def make_broker(feed: FakeFeed | None = None, *, cash: float = 1_000_000.0, **kw) -> PaperBroker:
    """A broker with the clock pinned inside the IST session."""
    kw.setdefault("clock", lambda: OPEN_IST)
    return PaperBroker(data_broker=feed or FakeFeed(), initial_cash=cash, **kw)


async def connected(feed: FakeFeed | None = None, **kw) -> PaperBroker:
    pb = make_broker(feed, **kw)
    await pb.connect()
    return pb


def ledger_cash_paise(pb: PaperBroker, trades: list[dict]) -> int:
    """Re-derive cash from the trade tape, independently of the broker."""
    cash = to_paise(pb._initial_cash_paise / 100.0)
    for t in trades:
        sign = -1 if t["txn_type"] == "BUY" else 1
        cash += sign * t["qty"] * to_paise(t["price"])
        cash -= to_paise(t["costs"]["total"])
    return cash


async def status(pb: PaperBroker, oid: str) -> dict:
    return await pb.get_order_status(oid)


# =========================================================================== #
#  D1 — SELLING SHARES YOU DO NOT OWN MUST NOT MINT CASH                      #
# =========================================================================== #

async def test_D1_sell_without_holding_is_rejected_and_cash_unchanged():
    """
    Regression for D1.  Before: this credited +Rs 2,897,688.21 of cash and
    left ``get_positions() == []``.
    """
    pb = await connected(FakeFeed(price=2900.0), cash=100_000.0)
    before = (await pb.get_funds())["cash"]

    oid = await pb.place_order("RELIANCE", "NSE", TransactionType.SELL, 1000,
                               0.0, OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)

    assert st["status"] == OrderStatus.REJECTED, st
    assert st["reject_reason"] == RejectReason.SHORT_SELL_NOT_SUPPORTED, st
    assert st["filled_qty"] == 0
    assert (await pb.get_funds())["cash"] == before
    assert (await pb.get_funds())["total_cash"] == before
    assert await pb.get_positions() == []
    assert await pb.get_trades() == []


async def test_D1_overselling_is_rejected_no_phantom_negative_position():
    """Before: buy 10 then sell 100 left a phantom quantity of -90."""
    pb = await connected(FakeFeed(price=100.0))
    await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                         OrderType.MARKET, Product.MIS)
    cash_after_buy = (await pb.get_funds())["total_cash"]

    oid = await pb.place_order("X", "NSE", TransactionType.SELL, 100, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)

    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.INSUFFICIENT_HOLDINGS
    assert (await pb.get_funds())["total_cash"] == cash_after_buy
    pos = await pb.get_positions()
    assert len(pos) == 1 and pos[0]["quantity"] == 10, pos


async def test_D1_short_selling_is_explicitly_unsupported():
    assert SHORT_SELLING_SUPPORTED is False
    pb = await connected(FakeFeed(price=100.0))
    perf = await pb.get_paper_performance()
    assert perf["short_selling_supported"] is False


async def test_D1_partial_holding_sell_rejects_whole_order():
    """Selling 60 when only 50 are held is rejected outright, not clipped."""
    feed = FakeFeed(price=100.0)
    pb = await connected(feed)
    await pb.place_order("X", "NSE", TransactionType.BUY, 50, 0.0,
                         OrderType.MARKET, Product.MIS)
    oid = await pb.place_order("X", "NSE", TransactionType.SELL, 60, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.INSUFFICIENT_HOLDINGS
    assert (await pb.get_positions())[0]["quantity"] == 50


async def test_D1_resting_sell_reserves_stock_so_it_cannot_be_sold_twice():
    feed = FakeFeed(price=100.0)
    pb = await connected(feed)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    # A far-above-market limit sell rests and earmarks the shares.
    resting = await pb.place_order("X", "NSE", TransactionType.SELL, 100, 500.0,
                                   OrderType.LIMIT, Product.MIS)
    assert (await status(pb, resting))["status"] == OrderStatus.OPEN

    second = await pb.place_order("X", "NSE", TransactionType.SELL, 100, 0.0,
                                  OrderType.MARKET, Product.MIS)
    assert (await status(pb, second))["status"] == OrderStatus.REJECTED

    assert await pb.cancel_order(resting) is True
    third = await pb.place_order("X", "NSE", TransactionType.SELL, 100, 0.0,
                                 OrderType.MARKET, Product.MIS)
    assert (await status(pb, third))["status"] == OrderStatus.COMPLETE


# =========================================================================== #
#  D2 — LIMIT ORDERS MUST NOT FILL AT AN UNMARKETABLE LIMIT                   #
# =========================================================================== #

async def test_D2_limit_buy_far_below_market_does_not_fill():
    """Before: a limit BUY at Rs 1 filled 100 shares at Rs 1.0005 vs Rs 2,900."""
    pb = await connected(FakeFeed(price=2900.0))
    cash_before = (await pb.get_funds())["total_cash"]

    oid = await pb.place_order("RELIANCE", "NSE", TransactionType.BUY, 100,
                               1.0, OrderType.LIMIT, Product.MIS)
    st = await status(pb, oid)

    assert st["status"] == OrderStatus.OPEN, st
    assert st["filled_qty"] == 0, st
    assert st["average_price"] == 0.0, st
    assert await pb.get_positions() == []
    assert await pb.get_trades() == []
    # Buying power is earmarked but no cash has moved.
    assert (await pb.get_funds())["total_cash"] == cash_before


async def test_D2_limit_sell_far_above_market_does_not_fill():
    feed = FakeFeed(price=100.0)
    pb = await connected(feed)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    cash = (await pb.get_funds())["total_cash"]

    oid = await pb.place_order("X", "NSE", TransactionType.SELL, 100, 10_000.0,
                               OrderType.LIMIT, Product.MIS)
    st = await status(pb, oid)

    assert st["status"] == OrderStatus.OPEN, st
    assert st["filled_qty"] == 0
    assert (await pb.get_funds())["total_cash"] == cash
    assert (await pb.get_positions())[0]["quantity"] == 100


async def test_D2_marketable_limit_buy_fills_at_or_below_the_limit():
    pb = await connected(FakeFeed(price=100.0))
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 105.0,
                               OrderType.LIMIT, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.COMPLETE
    assert st["average_price"] <= 105.0
    # ...and never better than the ask it had to cross.
    assert st["average_price"] >= 100.0


async def test_D2_marketable_limit_sell_fills_at_or_above_the_limit():
    feed = FakeFeed(price=100.0)
    pb = await connected(feed)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    oid = await pb.place_order("X", "NSE", TransactionType.SELL, 100, 95.0,
                               OrderType.LIMIT, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.COMPLETE
    assert 95.0 <= st["average_price"] <= 100.0


async def test_D2_resting_limit_fills_only_when_the_market_comes_to_it():
    feed = FakeFeed(price=100.0)
    pb = await connected(feed)
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 90.0,
                               OrderType.LIMIT, Product.MIS)
    assert (await status(pb, oid))["status"] == OrderStatus.OPEN
    assert await pb.poll_open_orders() == 0        # still not marketable

    feed.set_price("X", 89.0)                      # market comes down
    assert await pb.poll_open_orders() == 1
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.COMPLETE
    assert st["average_price"] <= 90.0


# =========================================================================== #
#  D3/D4 — EXECUTION REALISM                                                  #
# =========================================================================== #

async def test_exec_partial_fill_when_order_is_large_versus_volume():
    """
    Before: ``fill_probability = min(1, volume*0.05/qty)`` made 100/100 orders
    fill in full at any realistic volume.
    """
    feed = FakeFeed(price=100.0, volume=1_000)
    pb = await connected(feed, cash=10_000_000.0, max_participation=0.10)

    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 500, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.PARTIAL, st
    assert st["filled_qty"] == 100, st            # 10% of the 1,000 traded
    assert (await pb.get_positions())[0]["quantity"] == 100


async def test_exec_full_fill_when_order_is_small_versus_volume():
    pb = await connected(FakeFeed(price=100.0, volume=1_000_000))
    for _ in range(25):
        oid = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                                   OrderType.MARKET, Product.MIS)
        st = await status(pb, oid)
        assert st["status"] == OrderStatus.COMPLETE and st["filled_qty"] == 100


async def test_exec_fills_are_deterministic_not_random():
    """No RNG in the fill model: the same inputs give the same fill twice."""
    results = []
    for _ in range(2):
        pb = await connected(FakeFeed(price=100.0, volume=50_000), cash=5_000_000.0)
        oid = await pb.place_order("X", "NSE", TransactionType.BUY, 9_000, 0.0,
                                   OrderType.MARKET, Product.MIS)
        st = await status(pb, oid)
        results.append((st["filled_qty"], st["average_price"]))
    assert results[0] == results[1], results


async def test_exec_slippage_increases_with_order_size():
    """Before: a flat 5 bps regardless of size (10 shares cost the same as 10 m)."""
    sizes = [100, 1_000, 10_000, 100_000]
    fills = []
    for qty in sizes:
        pb = await connected(
            FakeFeed(price=1_000.0, volume=1_000_000),
            cash=10_000_000_000.0,
            slippage_pct=0.0, impact_coef=0.02, max_participation=1.0, tick_size=0.01,
        )
        oid = await pb.place_order("X", "NSE", TransactionType.BUY, qty, 0.0,
                                   OrderType.MARKET, Product.MIS)
        st = await status(pb, oid)
        assert st["filled_qty"] == qty
        fills.append(st["average_price"])

    assert fills == sorted(fills), fills
    assert fills[-1] > fills[0], fills
    # Square-root law: 1000x the size costs ~sqrt(1000) = 31.6x the impact.
    assert (fills[-1] - 1_000.0) > 10 * (fills[0] - 1_000.0), fills


async def test_exec_sell_slippage_is_also_adverse_and_size_dependent():
    small, large = [], []
    for qty, bucket in ((100, small), (100_000, large)):
        feed = FakeFeed(price=1_000.0, volume=1_000_000)
        pb = await connected(feed, cash=10_000_000_000.0, slippage_pct=0.0,
                             impact_coef=0.02, max_participation=1.0, tick_size=0.01)
        await pb.place_order("X", "NSE", TransactionType.BUY, qty, 0.0,
                             OrderType.MARKET, Product.MIS)
        oid = await pb.place_order("X", "NSE", TransactionType.SELL, qty, 0.0,
                                   OrderType.MARKET, Product.MIS)
        bucket.append((await status(pb, oid))["average_price"])
    assert large[0] < small[0] < 1_000.0, (small, large)


async def test_exec_spread_is_wider_for_less_liquid_names():
    """A thin name must cost more to cross than an index heavyweight."""
    prices = []
    for volume in (10_000_000, 1_000):        # Rs 100 cr vs Rs 1 lakh turnover
        pb = await connected(FakeFeed(price=100.0, volume=volume),
                             cash=100_000_000.0, slippage_pct=0.0, impact_coef=0.0,
                             max_participation=1.0, tick_size=0.01)
        oid = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                                   OrderType.MARKET, Product.MIS)
        prices.append((await status(pb, oid))["average_price"])
    liquid, illiquid = prices
    assert illiquid > liquid, prices


async def test_exec_unknown_volume_is_rejected_not_filled_in_full():
    """Before: volume==0 fell through to ``return qty`` — a free full fill."""
    pb = await connected(FakeFeed(price=100.0, volume=0))
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.NO_LIQUIDITY_DATA


async def test_exec_stop_through_a_gap_fills_at_the_gapped_price():
    """A stop-loss must not be honoured at its trigger when the market gapped."""
    feed = FakeFeed(price=100.0, volume=1_000_000)
    pb = await connected(feed)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)

    stop = await pb.place_order("X", "NSE", TransactionType.SELL, 100, 95.0,
                                OrderType.SL_M, Product.MIS)
    assert (await status(pb, stop))["status"] == OrderStatus.OPEN

    feed.set_price("X", 80.0)                    # gap down through the stop
    assert await pb.poll_open_orders() == 1
    st = await status(pb, stop)
    assert st["status"] == OrderStatus.COMPLETE
    assert st["average_price"] < 81.0, st        # gapped price, not the Rs 95 trigger
    assert st["average_price"] < 95.0


async def test_exec_stop_buy_through_a_gap_fills_above_the_trigger():
    feed = FakeFeed(price=100.0, volume=1_000_000)
    pb = await connected(feed, cash=10_000_000.0)
    stop = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 105.0,
                                OrderType.SL_M, Product.MIS)
    assert (await status(pb, stop))["status"] == OrderStatus.OPEN
    feed.set_price("X", 130.0)                   # gap up through the trigger
    assert await pb.poll_open_orders() == 1
    st = await status(pb, stop)
    assert st["average_price"] > 129.0, st       # not the Rs 105 trigger


async def test_exec_stale_price_is_rejected():
    stale_ts = OPEN_IST - timedelta(seconds=300)
    pb = await connected(FakeFeed(price=100.0, timestamp=stale_ts))
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.STALE_PRICE
    assert pb.is_stale_tick("X", max_age_seconds=30.0) is True


async def test_exec_fresh_quote_is_accepted():
    pb = await connected(FakeFeed(price=100.0, timestamp=OPEN_IST - timedelta(seconds=2)))
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                               OrderType.MARKET, Product.MIS)
    assert (await status(pb, oid))["status"] == OrderStatus.COMPLETE
    assert pb.is_stale_tick("X", max_age_seconds=30.0) is False


async def test_exec_missing_price_is_rejected():
    feed = FakeFeed(price=100.0)
    feed.prices["X"] = None
    pb = await connected(feed)
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.NO_PRICE


async def test_exec_feed_failure_is_rejected_not_guessed():
    pb = await connected(FakeFeed(fail=True))
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                               OrderType.MARKET, Product.MIS)
    assert (await status(pb, oid))["reject_reason"] == RejectReason.NO_PRICE


@pytest.mark.parametrize("qty", [0, -5, 1.5, True])
async def test_exec_invalid_quantity_is_rejected(qty):
    pb = await connected(FakeFeed(price=100.0))
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, qty, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.INVALID_QUANTITY


@pytest.mark.parametrize("price", [0.0, -10.0, float("nan"), float("inf")])
async def test_exec_invalid_limit_price_is_rejected(price):
    pb = await connected(FakeFeed(price=100.0))
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 10, price,
                               OrderType.LIMIT, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.INVALID_PRICE


async def test_exec_positions_are_keyed_by_exchange_too():
    """Before: the key was ``symbol:product``, so NSE and BSE lines collided."""
    pb = await connected(FakeFeed(price=100.0))
    await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                         OrderType.MARKET, Product.MIS)
    await pb.place_order("X", "BSE", TransactionType.BUY, 10, 0.0,
                         OrderType.MARKET, Product.MIS)
    assert len(await pb.get_positions()) == 2


async def test_exec_square_off_closes_intraday_book():
    """The old broker had no square-off at all."""
    feed = FakeFeed(price=100.0)
    pb = await connected(feed)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    assert len(await pb.get_positions()) == 1
    ids = await pb.square_off_intraday()
    assert len(ids) == 1
    assert (await status(pb, ids[0]))["status"] == OrderStatus.COMPLETE
    assert await pb.get_positions() == []


async def test_exec_new_intraday_position_rejected_inside_squareoff_window():
    pb = await connected(FakeFeed(price=100.0), clock=lambda: LATE_IST)
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.SQUARE_OFF_WINDOW
    # ...but CNC (delivery) is unaffected, and an exit is always allowed.
    cnc = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                               OrderType.MARKET, Product.CNC)
    assert (await status(pb, cnc))["status"] == OrderStatus.COMPLETE


# =========================================================================== #
#  ACCOUNTING INVARIANTS                                                      #
# =========================================================================== #

async def test_inv_buy_reduces_cash_by_exactly_qty_times_price_plus_costs():
    pb = await connected(FakeFeed(price=100.0))
    before = to_paise((await pb.get_funds())["total_cash"])

    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 250, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.COMPLETE

    after = to_paise((await pb.get_funds())["total_cash"])
    expected = st["filled_qty"] * to_paise(st["average_price"]) + to_paise(st["costs"]["total"])
    assert before - after == expected, (before, after, expected, st)


async def test_inv_sell_increases_cash_by_exactly_qty_times_price_minus_costs():
    feed = FakeFeed(price=100.0)
    pb = await connected(feed)
    await pb.place_order("X", "NSE", TransactionType.BUY, 250, 0.0,
                         OrderType.MARKET, Product.MIS)
    before = to_paise((await pb.get_funds())["total_cash"])

    oid = await pb.place_order("X", "NSE", TransactionType.SELL, 250, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    after = to_paise((await pb.get_funds())["total_cash"])
    expected = st["filled_qty"] * to_paise(st["average_price"]) - to_paise(st["costs"]["total"])
    assert after - before == expected, (before, after, expected, st)


async def test_inv_round_trip_conserves_cash_to_the_paise():
    """
    Total account value is conserved across a round trip except for the
    modelled price difference and the explicit charges.  Asserted in integer
    paise, so there is no float slack anywhere.
    """
    feed = FakeFeed(price=100.0, volume=1_000_000)
    pb = await connected(feed, cash=1_000_000.0)
    start = to_paise((await pb.get_funds())["total_cash"])

    buy = await status(pb, await pb.place_order(
        "X", "NSE", TransactionType.BUY, 500, 0.0, OrderType.MARKET, Product.MIS))
    sell = await status(pb, await pb.place_order(
        "X", "NSE", TransactionType.SELL, 500, 0.0, OrderType.MARKET, Product.MIS))
    end = to_paise((await pb.get_funds())["total_cash"])

    assert buy["filled_qty"] == sell["filled_qty"] == 500
    buy_notional = 500 * to_paise(buy["average_price"])
    sell_notional = 500 * to_paise(sell["average_price"])
    costs = to_paise(buy["costs"]["total"]) + to_paise(sell["costs"]["total"])

    # 1. exact ledger identity
    assert end == start - buy_notional + sell_notional - costs

    # 2. every paise of leakage is explained by the spread plus the charges
    leakage = start - end
    spread_cost = buy_notional - sell_notional
    assert leakage == spread_cost + costs, (leakage, spread_cost, costs)

    # 3. and the book is flat again
    assert await pb.get_positions() == []


async def test_inv_round_trip_with_zero_charges_leaks_only_the_spread():
    """
    With every charge zeroed, the round-trip delta must equal the price
    difference exactly — no unexplained paise anywhere in the accounting.
    """
    feed = FakeFeed(price=100.0, volume=10_000_000)
    pb = await connected(feed, cash=1_000_000.0, cost_model=ZERO_COST_MODEL)
    start = to_paise((await pb.get_funds())["total_cash"])

    buy = await status(pb, await pb.place_order(
        "X", "NSE", TransactionType.BUY, 500, 0.0, OrderType.MARKET, Product.MIS))
    sell = await status(pb, await pb.place_order(
        "X", "NSE", TransactionType.SELL, 500, 0.0, OrderType.MARKET, Product.MIS))
    end = to_paise((await pb.get_funds())["total_cash"])

    assert buy["costs"]["total"] == 0.0 and sell["costs"]["total"] == 0.0
    delta = end - start
    assert delta == 500 * (to_paise(sell["average_price"]) - to_paise(buy["average_price"]))


async def test_inv_costs_come_from_the_shared_zerodha_cost_model():
    """A single cost model: paper must agree with the backtester to the paise."""
    pb = await connected(FakeFeed(price=100.0))
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 250, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    reference = ZerodhaCostModel().calculate_costs(
        "BUY", st["filled_qty"], st["average_price"], "NSE", "MIS")
    assert st["costs"]["total"] == pytest.approx(reference.total, abs=1e-4)
    assert st["costs"]["brokerage"] == pytest.approx(reference.brokerage, abs=1e-4)
    assert st["costs"]["stt"] == pytest.approx(reference.stt, abs=1e-4)


async def test_inv_insufficient_cash_rejects():
    pb = await connected(FakeFeed(price=2900.0), cash=10_000.0)
    oid = await pb.place_order("RELIANCE", "NSE", TransactionType.BUY, 1000, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, oid)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.INSUFFICIENT_CASH
    assert (await pb.get_funds())["total_cash"] == 10_000.0


async def test_inv_cash_never_goes_negative_under_repeated_buying():
    pb = await connected(FakeFeed(price=100.0, volume=100_000_000), cash=50_000.0)
    rejected = 0
    for _ in range(60):
        oid = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                                   OrderType.MARKET, Product.MIS)
        if (await status(pb, oid))["status"] == OrderStatus.REJECTED:
            rejected += 1
        funds = await pb.get_funds()
        assert funds["total_cash"] >= 0.0, funds
        assert funds["cash"] >= 0.0, funds
    assert rejected > 0, "expected the account to run out of money"


async def test_inv_resting_buy_orders_cannot_double_spend_the_same_cash():
    feed = FakeFeed(price=100.0)
    pb = await connected(feed, cash=20_000.0)
    a = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 99.0,
                             OrderType.LIMIT, Product.MIS)
    b = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 99.0,
                             OrderType.LIMIT, Product.MIS)
    assert (await status(pb, a))["status"] == OrderStatus.OPEN
    assert (await status(pb, b))["status"] == OrderStatus.OPEN
    funds = await pb.get_funds()
    assert funds["margin_used"] > 0
    assert funds["cash"] < funds["total_cash"]

    c = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 99.0,
                             OrderType.LIMIT, Product.MIS)
    st = await status(pb, c)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.INSUFFICIENT_CASH

    # Both resting orders become marketable at once; cash must still not go under.
    feed.set_price("X", 95.0)
    await pb.poll_open_orders()
    assert (await pb.get_funds())["total_cash"] >= 0.0


async def test_inv_position_quantity_never_goes_negative():
    feed = FakeFeed(price=100.0)
    pb = await connected(feed)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    for qty in (101, 500, 10_000):
        await pb.place_order("X", "NSE", TransactionType.SELL, qty, 0.0,
                             OrderType.MARKET, Product.MIS)
        for pos in await pb.get_positions():
            assert pos["quantity"] >= 0, pos
    assert (await pb.get_positions())[0]["quantity"] == 100


async def test_inv_internal_guard_fires_if_the_book_is_corrupted_in_memory():
    """The invariants are enforced in code, not merely asserted in tests."""
    pb = await connected(FakeFeed(price=100.0))
    pb._cash_paise -= 1                      # simulate a stray mutation
    with pytest.raises(AccountingInvariantError):
        pb._assert_invariants()

    pb2 = await connected(FakeFeed(price=100.0))
    await pb2.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                          OrderType.MARKET, Product.MIS)
    list(pb2._positions.values())[0].quantity = -1
    with pytest.raises(AccountingInvariantError):
        pb2._assert_invariants()


async def test_inv_ledger_and_running_balance_agree_after_every_order():
    feed = FakeFeed(price=100.0, volume=1_000_000)
    pb = await connected(feed, cash=500_000.0)
    for i in range(30):
        side = TransactionType.BUY if i % 3 else TransactionType.SELL
        await pb.place_order("X", "NSE", side, 50, 0.0, OrderType.MARKET, Product.MIS)
        feed.set_price("X", 100.0 + (i % 7))
        trades = await pb.get_trades()
        assert ledger_cash_paise(pb, trades) == pb._cash_paise


# =========================================================================== #
#  D5 — MARK TO MARKET                                                        #
# =========================================================================== #

async def test_D5_positions_are_marked_to_market_not_to_cost():
    """Before: portfolio_value used average_price, so a doubling showed -0.0004%."""
    feed = FakeFeed(price=100.0)
    pb = await connected(feed, cash=1_000_000.0)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)

    feed.set_price("X", 200.0)                       # the stock doubles
    perf = await pb.get_paper_performance()

    assert perf["unrealised_pnl"] > 9_000.0, perf
    assert perf["return_pct"] > 0.9, perf            # ~+1% of a Rs 10 lakh book
    assert perf["portfolio_value"] > 1_009_000.0, perf
    assert perf["market_value"] == pytest.approx(20_000.0, abs=1.0)

    pos = (await pb.get_positions())[0]
    assert pos["last_price"] == 200.0
    assert pos["unrealised"] > 9_000.0


async def test_D5_a_loss_is_also_reflected_immediately():
    feed = FakeFeed(price=100.0)
    pb = await connected(feed, cash=1_000_000.0)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    feed.set_price("X", 50.0)
    perf = await pb.get_paper_performance()
    assert perf["unrealised_pnl"] < -4_900.0, perf
    assert perf["return_pct"] < 0, perf


async def test_D5_portfolio_value_equals_cash_plus_marked_positions():
    feed = FakeFeed(price=100.0)
    pb = await connected(feed, cash=1_000_000.0)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    feed.set_price("X", 137.5)
    perf = await pb.get_paper_performance()
    assert perf["portfolio_value"] == pytest.approx(
        perf["current_cash"] + 100 * 137.5, abs=0.01)


# =========================================================================== #
#  MARKET HOURS — IST, PROVEN UNDER TZ=UTC                                    #
# =========================================================================== #

@pytest.fixture(params=["UTC", "Asia/Kolkata"])
def system_tz(request):
    """
    Run the session tests under both a UTC server clock and an IST one.

    The platform previously had a naive-datetime bug that, on a UTC server,
    would have judged an IST evening to be inside the session.  These tests
    must give identical answers under both.
    """
    old = os.environ.get("TZ")
    os.environ["TZ"] = request.param
    time.tzset()
    yield request.param
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time.tzset()


def test_hours_ist_session_boundaries_are_tz_independent(system_tz):
    pb = PaperBroker(data_broker=FakeFeed())
    # Expressed in UTC on purpose: 05:30 UTC == 11:00 IST (open),
    # 14:30 UTC == 20:00 IST (closed).  A naive implementation on a UTC box
    # gets both of these backwards.
    assert pb.is_market_open(datetime(2025, 6, 10, 5, 30, tzinfo=timezone.utc)) is True
    assert pb.is_market_open(datetime(2025, 6, 10, 14, 30, tzinfo=timezone.utc)) is False
    assert pb.is_market_open(datetime(2025, 6, 10, 3, 0, tzinfo=timezone.utc)) is False
    # 03:44 UTC == 09:14 IST, one minute before the open.
    assert pb.is_market_open(datetime(2025, 6, 10, 3, 44, tzinfo=timezone.utc)) is False
    assert pb.is_market_open(datetime(2025, 6, 10, 3, 45, tzinfo=timezone.utc)) is True
    # 09:59:59 UTC == 15:29:59 IST, the last second of the session.
    assert pb.is_market_open(datetime(2025, 6, 10, 9, 59, 59, tzinfo=timezone.utc)) is True
    assert pb.is_market_open(datetime(2025, 6, 10, 10, 0, tzinfo=timezone.utc)) is False
    # Weekend and NSE holiday.
    assert pb.is_market_open(SUNDAY_IST) is False
    assert pb.is_market_open(HOLIDAY_IST) is False


def test_hours_naive_datetime_raises(system_tz):
    pb = PaperBroker(data_broker=FakeFeed())
    with pytest.raises(ValueError, match="naive"):
        pb.is_market_open(datetime(2025, 6, 10, 11, 0))
    with pytest.raises(ValueError, match="naive"):
        ensure_aware(datetime(2025, 6, 10, 11, 0))
    naive_clock = PaperBroker(data_broker=FakeFeed(),
                             clock=lambda: datetime(2025, 6, 10, 11, 0))
    with pytest.raises(ValueError, match="naive"):
        naive_clock.now_ist()


def test_hours_default_clock_is_timezone_aware(system_tz):
    pb = PaperBroker(data_broker=FakeFeed())
    assert pb.now_ist().tzinfo is not None
    assert pb.now_ist().utcoffset() == timedelta(hours=5, minutes=30)


async def test_hours_market_closed_rejects_orders(system_tz):
    """The core requirement: no trading outside the IST session."""
    for moment in (AFTER_IST, SUNDAY_IST, HOLIDAY_IST,
                   datetime(2025, 6, 10, 14, 30, tzinfo=timezone.utc)):
        pb = await connected(FakeFeed(price=100.0), clock=lambda m=moment: m)
        oid = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                                   OrderType.MARKET, Product.MIS)
        st = await status(pb, oid)
        assert st["status"] == OrderStatus.REJECTED, (moment, st)
        assert st["reject_reason"] == RejectReason.MARKET_CLOSED, (moment, st)
        assert (await pb.get_funds())["total_cash"] == 100.0 * 10_000  # untouched
        assert await pb.get_trades() == []


async def test_hours_open_session_accepts_orders(system_tz):
    """The mirror image: 05:30 UTC is 11:00 IST and must trade."""
    pb = await connected(FakeFeed(price=100.0),
                         clock=lambda: datetime(2025, 6, 10, 5, 30, tzinfo=timezone.utc))
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                               OrderType.MARKET, Product.MIS)
    assert (await status(pb, oid))["status"] == OrderStatus.COMPLETE


def test_hours_holiday_calendar_is_consulted(system_tz):
    pb = PaperBroker(data_broker=FakeFeed())
    assert pb.is_trading_holiday(HOLIDAY_IST) is True
    assert pb.is_trading_holiday(OPEN_IST) is False
    custom = PaperBroker(data_broker=FakeFeed(),
                         holidays=[datetime(2025, 6, 10).date()])
    assert custom.is_market_open(OPEN_IST) is False


# =========================================================================== #
#  D6 — PERSISTENCE MUST FAIL CLOSED                                          #
# =========================================================================== #

async def test_D6_corrupt_state_refuses_to_start():
    """Before: corrupt JSON was swallowed and the account reset to full cash."""
    redis = FakeRedis()
    redis.store["paper_broker:default:state"] = "{not json"
    pb = make_broker(FakeFeed(), cash=1_000_000.0, redis_client=redis)
    with pytest.raises(PaperBrokerStateError, match="corrupt"):
        await pb.connect()
    assert pb.is_connected is False


async def test_D6_tampered_state_refuses_to_start():
    """A hand-edited balance breaks the checksum and is refused."""
    redis = FakeRedis()
    pb = make_broker(FakeFeed(price=100.0), cash=1_000_000.0, redis_client=redis)
    await pb.connect()
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    await pb.disconnect()

    envelope = json.loads(redis.store["paper_broker:default:state"])
    body = json.loads(envelope["body"])
    body["cash_paise"] = 999_999_999            # "restore" the losses
    envelope["body"] = json.dumps(body, sort_keys=True, separators=(",", ":"))
    redis.store["paper_broker:default:state"] = json.dumps(envelope)

    reopened = make_broker(FakeFeed(price=100.0), cash=1_000_000.0, redis_client=redis)
    with pytest.raises(PaperBrokerStateError, match="checksum"):
        await reopened.connect()


async def test_D6_tampered_state_with_recomputed_checksum_still_refuses():
    """Even a correctly re-signed blob fails: cash must match the trade tape."""
    redis = FakeRedis()
    pb = make_broker(FakeFeed(price=100.0), cash=1_000_000.0, redis_client=redis)
    await pb.connect()
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    await pb.disconnect()

    envelope = json.loads(redis.store["paper_broker:default:state"])
    body = json.loads(envelope["body"])
    body["cash_paise"] = 100_000_000
    new_body = json.dumps(body, sort_keys=True, separators=(",", ":"))
    redis.store["paper_broker:default:state"] = json.dumps({
        "schema": PaperBroker.STATE_SCHEMA_VERSION,
        "checksum": hashlib.sha256(new_body.encode()).hexdigest(),
        "body": new_body,
    })

    reopened = make_broker(FakeFeed(price=100.0), cash=1_000_000.0, redis_client=redis)
    with pytest.raises(PaperBrokerStateError, match="ledger"):
        await reopened.connect()


async def test_D6_unreadable_store_refuses_to_start():
    redis = FakeRedis()
    redis.raise_on_get = True
    pb = make_broker(FakeFeed(), cash=1_000_000.0, redis_client=redis)
    with pytest.raises(PaperBrokerStateError, match="unreadable"):
        await pb.connect()


async def test_D6_reset_requires_an_explicit_human_decision():
    redis = FakeRedis()
    redis.store["paper_broker:default:state"] = "{not json"
    pb = make_broker(FakeFeed(), cash=1_000_000.0, redis_client=redis,
                     allow_state_reset=True)
    await pb.connect()
    assert (await pb.get_funds())["total_cash"] == 1_000_000.0


async def test_state_survives_a_restart():
    redis = FakeRedis()
    feed = FakeFeed(price=100.0)
    pb = make_broker(feed, cash=1_000_000.0, redis_client=redis)
    await pb.connect()
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    cash = (await pb.get_funds())["total_cash"]
    await pb.disconnect()

    revived = make_broker(FakeFeed(price=100.0), cash=1_000_000.0, redis_client=redis)
    await revived.connect()
    assert (await revived.get_funds())["total_cash"] == cash
    assert cash < 1_000_000.0                        # the loss was NOT erased
    pos = await revived.get_positions()
    assert len(pos) == 1 and pos[0]["quantity"] == 100
    assert len(await revived.get_trades()) == 1


async def test_state_survives_a_restart_via_a_file(tmp_path):
    path = tmp_path / "paper.json"
    feed = FakeFeed(price=100.0)
    pb = make_broker(feed, cash=1_000_000.0, state_path=path)
    await pb.connect()
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    cash = (await pb.get_funds())["total_cash"]
    await pb.disconnect()

    revived = make_broker(FakeFeed(price=100.0), cash=1_000_000.0, state_path=path)
    await revived.connect()
    assert (await revived.get_funds())["total_cash"] == cash


async def test_state_write_failure_degrades_to_rejecting_new_orders():
    redis = FakeRedis()
    pb = make_broker(FakeFeed(price=100.0), cash=1_000_000.0, redis_client=redis)
    await pb.connect()
    redis.raise_on_set = True
    await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                         OrderType.MARKET, Product.MIS)
    nxt = await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                               OrderType.MARKET, Product.MIS)
    st = await status(pb, nxt)
    assert st["status"] == OrderStatus.REJECTED
    assert st["reject_reason"] == RejectReason.PERSISTENCE_DEGRADED


# =========================================================================== #
#  RANDOMIZED PROPERTY TEST                                                   #
# =========================================================================== #

SCENARIOS = 400
MAX_ORDERS_PER_SCENARIO = 12


async def _run_scenario(seed: int) -> list[str]:
    """Drive one random-but-valid order sequence; return invariant violations."""
    rng = random.Random(seed)
    symbols = ["AAA", "BBB", "CCC"]
    feed = FakeFeed(
        prices={s: rng.uniform(25.0, 3_000.0) for s in symbols},
        volumes={s: rng.choice([2_000, 50_000, 1_000_000, 25_000_000]) for s in symbols},
    )
    initial = rng.choice([50_000.0, 250_000.0, 1_000_000.0, 5_000_000.0])
    pb = PaperBroker(
        data_broker=feed,
        initial_cash=initial,
        clock=lambda: OPEN_IST,
        slippage_pct=rng.choice([0.0, 0.0005, 0.002]),
        impact_coef=rng.choice([0.0, 0.01, 0.05]),
        max_participation=rng.choice([0.05, 0.10, 0.5]),
        tick_size=rng.choice([0.01, 0.05]),
    )
    await pb.connect()

    violations: list[str] = []

    def check(label: str) -> None:
        funds_cash = pb._cash_paise
        reserved = pb._reserved_cash_paise
        if funds_cash < 0:
            violations.append(f"{label}: cash negative ({funds_cash}p)")
        if reserved < 0 or reserved > funds_cash:
            violations.append(f"{label}: reserved {reserved}p vs cash {funds_cash}p")
        for key, pos in pb._positions.items():
            if pos.quantity < 0:
                violations.append(f"{label}: {key} quantity {pos.quantity}")
            if pos.cost_basis_paise < 0:
                violations.append(f"{label}: {key} basis {pos.cost_basis_paise}p")
        derived = ledger_cash_paise(pb, pb._trades)
        if derived != funds_cash:
            violations.append(f"{label}: ledger {derived}p != cash {funds_cash}p")
        for o in pb._orders.values():
            if o.filled_qty > o.qty:
                violations.append(f"{label}: order overfilled {o.filled_qty}/{o.qty}")
            if o.status == OrderStatus.REJECTED and o.filled_qty:
                violations.append(f"{label}: rejected order has a fill")

    check("start")
    for step in range(rng.randint(3, MAX_ORDERS_PER_SCENARIO)):
        sym = rng.choice(symbols)
        # Random but *valid* inputs: positive int quantity, positive prices.
        qty = rng.choice([1, 7, 50, 250, 1_000, 5_000])
        side = rng.choice([TransactionType.BUY, TransactionType.SELL])
        otype = rng.choice([OrderType.MARKET, OrderType.LIMIT,
                            OrderType.SL_M, OrderType.SL])
        last = feed.price_of(sym)
        price = 0.0 if otype == OrderType.MARKET else round(
            last * rng.uniform(0.8, 1.2), 2)

        cash_before = pb._cash_paise
        positions_before = {k: v.quantity for k, v in pb._positions.items()}

        oid = await pb.place_order(sym, "NSE", side, qty, price, otype,
                                   rng.choice([Product.MIS, Product.CNC]))
        st = await pb.get_order_status(oid)
        check(f"seed={seed} step={step} after-place")

        if st["status"] == OrderStatus.REJECTED:
            if pb._cash_paise != cash_before:
                violations.append(f"seed={seed} step={step}: rejection moved cash")
            now = {k: v.quantity for k, v in pb._positions.items()}
            if now != positions_before:
                violations.append(f"seed={seed} step={step}: rejection moved stock")

        # Move the market, then let resting orders trade.
        feed.set_price(sym, max(1.0, last * rng.uniform(0.85, 1.15)))
        await pb.poll_open_orders()
        check(f"seed={seed} step={step} after-poll")

    # Unwind whatever is left; a full unwind must never fail an invariant.
    for pos in list(pb._positions.values()):
        for o in list(pb._orders.values()):
            if o.status == OrderStatus.OPEN:
                await pb.cancel_order(o.order_id)
        if pos.quantity > 0:
            await pb.place_order(pos.symbol, pos.exchange, TransactionType.SELL,
                                 pos.quantity, 0.0, OrderType.MARKET,
                                 Product(pos.product))
    check(f"seed={seed} unwind")
    return violations


async def test_property_accounting_invariants_hold_across_random_sequences(capsys):
    """
    Several hundred random valid order sequences.  After every order and every
    poll the following must hold:

      * cash >= 0 and 0 <= reserved <= cash
      * every position quantity >= 0 (short selling is unsupported)
      * the running cash balance equals cash re-derived from the trade tape
      * no order is filled beyond its quantity
      * a REJECTED order moves neither cash nor stock
    """
    all_violations: list[str] = []
    for seed in range(SCENARIOS):
        all_violations.extend(await _run_scenario(seed))

    with capsys.disabled():
        print(f"\n[property] scenarios={SCENARIOS} "
              f"violations={len(all_violations)}")
    assert all_violations == [], all_violations[:20]


async def test_property_reports_scenario_and_violation_counts():
    """A machine-readable record of the property run for the audit trail."""
    violations = []
    for seed in range(SCENARIOS, SCENARIOS + 50):
        violations.extend(await _run_scenario(seed))
    assert (len(violations), 50) == (0, 50), violations[:10]


# =========================================================================== #
#  MISCELLANEOUS CONTRACT CHECKS                                              #
# =========================================================================== #

async def test_broker_conforms_to_the_interface():
    pb = await connected(FakeFeed(price=100.0))
    assert isinstance(pb, BrokerInterface)
    assert pb.trading_mode == "paper"
    assert pb.is_connected is True
    assert (await pb.get_profile())["broker"] == "PAPER"
    assert isinstance(await pb.get_orders(), list)
    assert isinstance(await pb.get_holdings(), list)
    assert isinstance(await pb.get_funds(), dict)
    assert isinstance(await pb.get_historical_data("X", "NSE", "day", "a", "b"),
                      pd.DataFrame)
    assert pb.instrument_token("X", "NSE") > 0


async def test_cancel_and_modify_release_reservations():
    pb = await connected(FakeFeed(price=100.0), cash=100_000.0)
    oid = await pb.place_order("X", "NSE", TransactionType.BUY, 100, 90.0,
                               OrderType.LIMIT, Product.MIS)
    assert (await pb.get_funds())["margin_used"] > 0
    assert await pb.cancel_order(oid) is True
    assert (await pb.get_funds())["margin_used"] == 0.0
    assert (await pb.get_funds())["cash"] == 100_000.0
    assert await pb.cancel_order(oid) is False       # already terminal
    assert await pb.modify_order(oid, qty=10) is False
    assert await pb.get_order_status("nope") == {}


async def test_holdings_only_lists_delivery_positions():
    pb = await connected(FakeFeed(price=100.0))
    await pb.place_order("X", "NSE", TransactionType.BUY, 10, 0.0,
                         OrderType.MARKET, Product.CNC)
    await pb.place_order("Y", "NSE", TransactionType.BUY, 10, 0.0,
                         OrderType.MARKET, Product.MIS)
    holdings = await pb.get_holdings()
    assert [h["symbol"] for h in holdings] == ["X"]
    assert len(await pb.get_positions()) == 2


async def test_realised_pnl_is_recorded_on_a_profitable_exit():
    feed = FakeFeed(price=100.0, volume=10_000_000)
    pb = await connected(feed, cash=1_000_000.0)
    await pb.place_order("X", "NSE", TransactionType.BUY, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    feed.set_price("X", 150.0)
    await pb.place_order("X", "NSE", TransactionType.SELL, 100, 0.0,
                         OrderType.MARKET, Product.MIS)
    perf = await pb.get_paper_performance()
    assert perf["realised_pnl"] > 4_800.0, perf
    assert perf["portfolio_value"] > 1_004_000.0, perf
    assert perf["total_transaction_costs"] > 0
    assert math.isclose(perf["portfolio_value"], perf["current_cash"], abs_tol=0.01)
