"""
The read-only market data feed.

The headline test here is `test_BUG_the_symbol_is_not_double_suffixed`. The
feed shipped with `suffix=".NS"` while `YahooDataProvider.fetch_symbol` already
builds `f"{symbol}.NS"` itself, so every request went out as `RELIANCE.NS.NS`,
Yahoo answered 404, and `get_quote` returned an EMPTY DICT for every symbol.

That failed closed — a missing quote refuses the order — so no money was ever
at risk. But it meant the production paper path could not price anything at
all, and it was invisible because "no quote" is also what a genuinely
unpriceable symbol looks like.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import pytest

from app.broker.marketdata import MarketDataBroker, MarketDataUnavailable

pytestmark = pytest.mark.asyncio

IST = timezone(timedelta(hours=5, minutes=30))


class RecordingProvider:
    """Records exactly what symbol string it was asked for."""

    def __init__(self, *, rows: int = 5, bad: bool = False) -> None:
        self.requested: list[str] = []
        self.rows = rows
        self.bad = bad

    def fetch_symbol(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        self.requested.append(symbol)
        if self.bad or self.rows == 0:
            return None
        idx = pd.date_range(end="2025-06-03", periods=self.rows, freq="B")
        return pd.DataFrame(
            {
                "open": [100.0] * self.rows,
                "high": [101.0] * self.rows,
                "low": [99.0] * self.rows,
                "close": [100.5] * self.rows,
                "adj_close": [100.5] * self.rows,
                "volume": [1_000_000] * self.rows,
            },
            index=idx,
        )


class TestSymbolNamespacing:

    async def test_BUG_the_symbol_is_not_double_suffixed(self):
        """
        The provider namespaces symbols itself; the feed must not do it again.

        `RELIANCE.NS.NS` is a 404 for every name, which made get_quote return
        an empty dict and the paper broker unable to price anything.
        """
        provider = RecordingProvider()
        feed = MarketDataBroker(provider)

        await feed.get_quote(["RELIANCE"])

        assert provider.requested == ["RELIANCE"], (
            f"the provider was asked for {provider.requested!r}; anything with a "
            "duplicated .NS is a 404 and prices nothing"
        )
        assert not any(s.count(".NS") > 1 for s in provider.requested)

    async def test_an_exchange_prefix_is_stripped_before_the_provider_sees_it(self):
        provider = RecordingProvider()
        feed = MarketDataBroker(provider)
        await feed.get_quote(["NSE:RELIANCE"])
        assert provider.requested == ["RELIANCE"]

    async def test_an_explicit_suffix_is_still_honoured(self):
        """The parameter still works for a provider that does NOT namespace."""
        provider = RecordingProvider()
        feed = MarketDataBroker(provider, suffix=".BO")
        await feed.get_quote(["RELIANCE"])
        assert provider.requested == ["RELIANCE.BO"]


class TestQuoteHonesty:

    async def test_a_quote_carries_the_bar_timestamp_not_now(self):
        """
        Stamping a daily close with `now()` would defeat every freshness check.

        The staleness gate is what stops an order being priced off data that no
        longer reflects the market; a quote that always looks fresh disables it.
        """
        feed = MarketDataBroker(RecordingProvider())
        quotes = await feed.get_quote(["RELIANCE"])
        ts = datetime.fromisoformat(quotes["RELIANCE"]["timestamp"])

        assert ts.date() == datetime(2025, 6, 3).date()
        assert (ts.hour, ts.minute) == (15, 30), "not stamped at the NSE close"
        assert abs((datetime.now(IST) - ts).total_seconds()) > 3600, (
            "the quote is stamped at roughly now(), so it would always look fresh"
        )

    async def test_an_unpriceable_symbol_is_omitted_not_placeholdered(self):
        feed = MarketDataBroker(RecordingProvider(bad=True))
        assert await feed.get_quote(["NOSUCH"]) == {}

    async def test_a_quote_reports_that_its_depth_is_synthetic(self):
        """A daily bar has no book; the payload must not pretend otherwise."""
        feed = MarketDataBroker(RecordingProvider())
        q = (await feed.get_quote(["RELIANCE"]))["RELIANCE"]
        assert q["synthetic_depth"] is True
        assert q["resolution"] == "1d"


class TestItIsNotAnExecutionVenue:
    """A price source that could also trade would be a second path to a venue."""

    async def test_every_mutating_call_is_refused(self):
        feed = MarketDataBroker(RecordingProvider())
        for call in (
            feed.place_order(symbol="X"),
            feed.cancel_order("1"),
            feed.modify_order("1"),
            feed.get_order_status("1"),
        ):
            with pytest.raises(NotImplementedError):
                await call

    async def test_it_does_not_claim_to_be_paper_or_live(self):
        feed = MarketDataBroker(RecordingProvider())
        assert feed.trading_mode == "data", (
            "reporting 'paper' or 'live' would let the mode check mistake this "
            "feed for a trading venue"
        )

    async def test_an_intraday_interval_is_refused_not_approximated(self):
        """Returning daily bars under an intraday label corrupts silently."""
        feed = MarketDataBroker(RecordingProvider())
        with pytest.raises(MarketDataUnavailable, match="daily bars only"):
            await feed.get_historical_data("RELIANCE", "NSE", "5minute", "2025-01-01", "2025-06-01")

    async def test_a_provider_is_required(self):
        with pytest.raises(ValueError, match="provider"):
            MarketDataBroker(None)


class TestCaching:

    async def test_a_cache_hit_keeps_the_original_bar_timestamp(self):
        """Caching must not be able to make stale data look fresh."""
        provider = RecordingProvider()
        feed = MarketDataBroker(provider, quote_ttl_sec=3600)

        first = (await feed.get_quote(["RELIANCE"]))["RELIANCE"]["timestamp"]
        second = (await feed.get_quote(["RELIANCE"]))["RELIANCE"]["timestamp"]

        assert first == second
        assert len(provider.requested) == 1, "the cache did not hold"
