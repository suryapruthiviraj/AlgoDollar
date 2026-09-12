"""
Tests for the paper intraday data layer: TickBarAggregator, MockTickFeed and
TickQuoteSource (app/broker/tickdata.py).

A note on statements like "30 minutes": the IST month is 2025-06-10 (a
Tuesday, not an NSE holiday), and every timestamp in these tests is aware IST,
so the aggregator's session logic sees exactly what it sees in production.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.broker.marketdata import MarketDataUnavailable
from app.broker.tickdata import (
    IST,
    MockTickFeed,
    TickBarAggregator,
    TickQuoteSource,
)

P = "RELIANCE"
PRICE = 100.0


def _at(h: int, m: int = 0) -> datetime:
    return datetime(2025, 6, 10, h, m, tzinfo=IST)


def _push_minute(agg: TickBarAggregator, symbol: str, ts: datetime, price: float,
                 volume: float = 50_000) -> None:
    agg.on_tick(symbol, ts, price, volume)


class _FixedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


# --------------------------------------------------------------------------- #
#  TickBarAggregator                                                           #
# --------------------------------------------------------------------------- #

class TestAggregator:
    def test_one_minute_folds_multiple_ticks_into_ohlc(self) -> None:
        agg = TickBarAggregator()
        t = _at(10, 0)
        agg.on_tick(P, t, 100.0, 1000)
        agg.on_tick(P, t + timedelta(seconds=20), 102.0, 2000)
        agg.on_tick(P, t + timedelta(seconds=40), 99.0, 3000)

        df = agg.bars(P)
        assert len(df) == 1                      # one minute bucket
        assert int(df["open"].iloc[0]) == 100
        assert int(df["high"].iloc[0]) == 102
        assert int(df["low"].iloc[0]) == 99
        assert int(df["close"].iloc[0]) == 99
        assert int(df["volume"].iloc[0]) == 6000
        assert df["time"].iloc[0] == t.replace(second=0, microsecond=0)

    def test_separate_minutes_are_sorted_ascending(self) -> None:
        agg = TickBarAggregator()
        _push_minute(agg, P, _at(10, 0), 100.0)
        _push_minute(agg, P, _at(10, 1), 101.0)
        _push_minute(agg, P, _at(9, 59), 99.0)
        df = agg.bars(P)
        assert list(df["close"]) == [99.0, 100.0, 101.0]
        assert list(df["time"]) == [_at(9, 59), _at(10, 0), _at(10, 1)]

    def test_last_price_vwap_symbols_and_has_ticked(self) -> None:
        agg = TickBarAggregator()
        df = pd.DataFrame(
            {
                "time": [_at(10, 0), _at(10, 1), _at(10, 2)],
                "open": [100.0, 102.0, 103.0],
                "high": [101.0, 103.0, 104.0],
                "low": [99.0, 101.0, 103.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [100, 300, 600],
            }
        )
        for _, row in df.iterrows():
            agg.on_tick(P, row["time"], row["open"], 1)
            agg.on_tick(P, row["time"], row["high"], 1)
            agg.on_tick(P, row["time"], row["low"], 1)
            agg.on_tick(P, row["time"], row["close"], int(row["volume"]))

        assert agg.last_price(P) == 104.0
        agg_df = agg.bars(P)
        expected = (agg_df["high"] + agg_df["low"] + agg_df["close"]) / 3.0
        assert agg.vwap(P) == pytest.approx(
            float((expected * agg_df["volume"]).sum() / agg_df["volume"].sum())
        )
        assert agg.symbols() == [P]
        assert agg.has_ticked(P)
        assert agg.last_tick_time(P) == _at(10, 2)

    def test_unknown_symbol_is_an_empty_frame_not_an_error(self) -> None:
        agg = TickBarAggregator()
        df = agg.bars("UNKNOWN")
        assert df.empty
        assert list(df.columns) == ["time", "open", "high", "low", "close", "volume"]
        assert agg.last_price("UNKNOWN") is None
        assert agg.vwap("UNKNOWN") is None
        assert not agg.has_ticked("UNKNOWN")

    def test_session_resets_bars_when_the_ist_date_changes(self) -> None:
        agg = TickBarAggregator()
        _push_minute(agg, P, _at(10, 0), 100.0)
        assert len(agg.bars(P)) == 1
        # New IST session date -> intraday state must never leak across days.
        agg.on_tick(P, datetime(2025, 6, 11, 10, 0, tzinfo=IST), 110.0, 1000)
        assert len(agg.bars(P)) == 1
        assert agg.last_price(P) == 110.0
        assert agg.has_ticked(P)
        assert agg.symbols() == [P]

    def test_invalid_ticks_are_dropped_not_crashed(self) -> None:
        agg = TickBarAggregator()
        agg.on_tick(P, _at(10, 0), 100.0, 1000)
        agg.on_tick(P, _at(10, 0), 0.0, 1000)      # non-positive price
        agg.on_tick(P, _at(10, 0), -5.0, 1000)
        agg.on_tick(P, _at(10, 0), float("nan"), 1000)
        agg.on_tick(P, _at(10, 0), 101.0, 1000)   # still a valid tick
        assert agg.last_price(P) == 101.0
        assert len(agg.bars(P)) == 1

    def test_naive_timestamps_are_rejected(self) -> None:
        agg = TickBarAggregator()
        with pytest.raises(ValueError):
            agg.on_tick(P, datetime(2025, 6, 10, 10, 0), 100.0, 1000)

    def test_reset_session_clears_state(self) -> None:
        agg = TickBarAggregator()
        _push_minute(agg, P, _at(10, 0), 100.0)
        agg.reset_session()
        assert agg.bars(P).empty
        assert not agg.has_ticked(P)
        assert agg.symbols() == []
        # And a clean session can start fresh.
        _push_minute(agg, P, _at(10, 0), 101.0)
        assert agg.last_price(P) == 101.0


# --------------------------------------------------------------------------- #
#  MockTickFeed                                                                #
# --------------------------------------------------------------------------- #

def _feed(now=None, **kw) -> MockTickFeed:
    return MockTickFeed(
        ["RELIANCE", "TCS"],
        {"RELIANCE": 100.0, "TCS": 500.0},
        seed=7,
        clock=_FixedClock(now or _at(10, 0)),
        **kw,
    )


class TestFeed:
    def test_is_deterministic_for_the_same_seed(self) -> None:
        async def scenario() -> None:
            agg_a, agg_b = TickBarAggregator(), TickBarAggregator()
            feed_a, feed_b = _feed(), _feed()
            for _ in range(3):
                await feed_a.run_once(agg_a.on_tick)
                await feed_b.run_once(agg_b.on_tick)
            assert agg_a.bars("RELIANCE").equals(agg_b.bars("RELIANCE"))
            assert agg_a.bars("TCS").equals(agg_b.bars("TCS"))

        asyncio.run(scenario())

    def test_different_seed_produces_different_paths(self) -> None:
        async def scenario() -> None:
            agg_a, agg_b = TickBarAggregator(), TickBarAggregator()
            feed_a = MockTickFeed(
                ["RELIANCE"], {"RELIANCE": 100.0}, seed=1, tick_size=0.01,
            )
            feed_b = MockTickFeed(
                ["RELIANCE"], {"RELIANCE": 100.0}, seed=2, tick_size=0.01,
            )
            for _ in range(200):
                await feed_a.run_once(agg_a.on_tick)
                await feed_b.run_once(agg_b.on_tick)
            # Two different seeds must produce different walks; the rounded
            # first few ticks can coincide, so compare whole paths.
            assert not agg_a.bars("RELIANCE").equals(agg_b.bars("RELIANCE"))

        asyncio.run(scenario())

    def test_session_gate_reflects_the_clock(self) -> None:
        inside = _feed(now=_at(10, 0))
        outside = _feed(now=_at(16, 0))
        assert inside.is_session_time()
        assert not outside.is_session_time()

    def test_run_emits_nothing_outside_session_and_one_pass_inside(self) -> None:
        async def scenario() -> list[int]:
            counts = []
            for ts in (_at(16, 0), _at(10, 0)):
                agg = TickBarAggregator()
                feed = _feed(now=ts)
                stop = asyncio.Event()
                task = asyncio.create_task(feed.run(stop, agg.on_tick))
                await asyncio.sleep(0.05)
                stop.set()
                counts.append(await task)
            return counts

        outside, inside = asyncio.run(scenario())
        assert outside == 0
        assert inside == 2        # one tick per symbol

    def test_run_once_emits_one_tick_per_symbol(self) -> None:
        async def scenario() -> tuple[int, list[str]]:
            agg = TickBarAggregator()
            feed = _feed()
            n = await feed.run_once(agg.on_tick)
            return n, agg.symbols()

        n, symbols = asyncio.run(scenario())
        assert n == 2
        assert symbols == ["RELIANCE", "TCS"]


# --------------------------------------------------------------------------- #
#  TickQuoteSource                                                             #
# --------------------------------------------------------------------------- #

class _Delegate:
    """A data broker that can answer coarser history and quoted fallback."""

    def __init__(self, price: float) -> None:
        self.price = price

    async def get_quote(self, symbols):
        return {s: {"last_price": self.price, "volume": 1_000_000} for s in symbols}

    async def get_historical_data(self, symbol, exchange, interval, from_date=None, to_date=None):
        return pd.DataFrame({"interval": [interval]})


@pytest.mark.asyncio
class TestQuoteSource:
    async def test_quotes_are_keyed_by_both_raw_and_bare_symbol(self) -> None:
        agg = TickBarAggregator()
        _push_minute(agg, P, _at(10, 0), 112.5)
        src = TickQuoteSource(agg)
        quotes = await src.get_quote([f"NSE:{P}"])
        assert f"NSE:{P}" in quotes
        assert P in quotes
        assert quotes[P]["last_price"] == 112.5
        assert quotes[P]["synthetic"] is True
        assert quotes[P]["source"] == "MockTickFeed"
        assert quotes[f"NSE:{P}"]["last_price"] == quotes[P]["last_price"]

    async def test_unknown_symbol_is_omitted_without_error(self) -> None:
        agg = TickBarAggregator()
        src = TickQuoteSource(agg)
        assert await src.get_quote(["NSE:UNKNOWN"]) == {}

    async def test_falls_back_to_delegate_when_aggregator_has_no_bar(self) -> None:
        agg = TickBarAggregator()
        src = TickQuoteSource(agg, delegate=_Delegate(555.0))
        quotes = await src.get_quote(["NSE:UNKNOWN"])
        assert quotes.get("UNKNOWN", {}).get("last_price") == 555.0

    async def test_one_minute_history_comes_from_the_aggregator(self) -> None:
        agg = TickBarAggregator()
        _push_minute(agg, P, _at(10, 0), 100.0)
        src = TickQuoteSource(agg, delegate=_Delegate(0.0))
        df = await src.get_historical_data(P, "NSE", "1m")
        assert list(df["close"]) == [100.0]
        assert len(df) == 1

    async def test_coarser_history_delegates_or_raises(self) -> None:
        agg = TickBarAggregator()
        src = TickQuoteSource(agg, delegate=_Delegate(0.0))
        df = await src.get_historical_data(P, "NSE", "5m")
        assert list(df["interval"]) == ["5m"]

        bare = TickQuoteSource(agg)   # no delegate
        with pytest.raises(MarketDataUnavailable):
            await bare.get_historical_data(P, "NSE", "5m")
