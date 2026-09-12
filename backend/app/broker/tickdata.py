"""
tickdata.py — the paper intraday price feed.

Three pieces, one seam, one invariant:

  TickBarAggregator   ticks -> 1-minute OHLCV bars
  MockTickFeed        deterministic synthetic (seeded) NSE-style tick stream
  TickQuoteSource     a BrokerInterface-shaped price source the PaperBroker
                      quotes against and fills on

WHY THEY SHARE ONE MODULE
-------------------------
The intraday sleeve needs three views of the same stream: the strategy
consumes bars, the paper broker consumes (and fills against) point quotes,
and the auto-trader drives both.  If the aggregator is authoritative and the
other two are views OF IT, then a strategy signal, a paper fill and an exit
decision all resolve the SAME price at the SAME moment.  That was the property
the old design had to fetch from two agreeing sources and silently trust.

SAFETY
------
This is a PAPER-ONLY feed designed for TICK_MODE=mock.  It never connects to
a venue, and its prices are synthetic: a seeded, mean-reverting walk around
the previous daily close of each symbol.  Nothing here can be mistaken for
market data, and every public result says so in its payload
(``"source": "mock"``, ``"synthetic": True``).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# Mock session is the NSE regular session plus a 10-minute pre-open overlap so
# the loop has a warm bar before the opening-range window ends.
_MOCK_OPEN = time(9, 15)
_MOCK_CLOSE = time(15, 30)

# Typical SparkLine / Zerodha tick cadence per symbol.  The feed emits one
# logical tick per symbol per `interval_ms`; volumes below are per that tick.
_DEFAULT_TICK_INTERVAL_MS = 1_000

# Per-tick volume so that a symbol passes the intraday liquidity filter
# (min_intraday_volume / 375 per-minute shares) even for a short session.
# 4,000 shares/s tick * 60 ticks/min = 240,000 shares/min >> 2,666 threshold.
_DEFAULT_VOLUME_MEAN = 4_000.0


def to_ist(ts: datetime) -> datetime:
    """Normalise an aware timestamp to IST; raise on a naive one."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("tickdata timestamps must be timezone-aware")
    return ts.astimezone(IST)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _MinuteBar:
    start: datetime  # aware IST minute start
    open: float
    high: float
    low: float
    close: float
    volume: int


class TickBarAggregator:
    """
    In-memory 1-minute OHLCV bars, built from ticks.

    ``on_tick`` is the only write door; ``bars`` / ``last_price`` / ``vwap``
    are the read doors.  Bars reset automatically when the IST session date
    changes (intraday state must never leak across sessions — one of the two
    defects the 2023 UTC-clock bug produced was a stale, un-squared book).
    """

    def __init__(self) -> None:
        self._bars: Dict[str, Dict[int, _MinuteBar]] = {}
        self._session_date: Optional[date] = None
        self._source_name = "MockTickFeed"

    # ------------------------------------------------------------------ #
    #  Write door                                                         #
    # ------------------------------------------------------------------ #

    def on_tick(self, symbol: str, ts: datetime, price: float, volume: float) -> None:
        """
        Fold one tick into the current session's bars.

        A tick with a non-positive price is dropped (it cannot be a bar), but
        the invalid price is still an intraday event, so it is logged.
        """
        if price is None or not math.isfinite(float(price)) or price <= 0:
            logger.debug("Dropping invalid tick %s @ %s (price=%r)", symbol, ts, price)
            return
        ist = to_ist(ts)
        vol = int(max(0.0, float(volume or 0.0)))
        self._ensure_session(ist)
        minute_idx = int(ist.replace(second=0, microsecond=0).timestamp())
        bucket = self._bars.setdefault(symbol, {})
        bar = bucket.get(minute_idx)
        if bar is None:
            bucket[minute_idx] = _MinuteBar(
                start=ist.replace(second=0, microsecond=0),
                open=float(price),
                high=float(price),
                low=float(price),
                close=float(price),
                volume=vol,
            )
            return
        bar.high = max(bar.high, float(price))
        bar.low = min(bar.low, float(price))
        bar.close = float(price)
        bar.volume += vol

    # ------------------------------------------------------------------ #
    #  Read doors                                                         #
    # ------------------------------------------------------------------ #

    def bars(self, symbol: str) -> pd.DataFrame:
        """
        Current-session 1-minute bars for ``symbol`` as a DataFrame with
        columns ``time, open, high, low, close, volume`` in ascending order.

        The ``time`` column is minute-start, timezone-aware (IST).  An empty
        DataFrame (0 rows) is returned when the symbol has no bars yet —
        callers treat that as "not tradeable yet", never as a zero price.
        """
        bucket = self._bars.get(symbol)
        if not bucket:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        rows = [bucket[k] for k in sorted(bucket)]
        return pd.DataFrame(
            {
                "time": [r.start for r in rows],
                "open": [r.open for r in rows],
                "high": [r.high for r in rows],
                "low": [r.low for r in rows],
                "close": [r.close for r in rows],
                "volume": [r.volume for r in rows],
            }
        )

    def session_bars(self, symbol: str) -> pd.DataFrame:
        """Alias of ``bars`` — kept explicit so callers name the invariant."""
        return self.bars(symbol)

    def last_price(self, symbol: str) -> Optional[float]:
        """Most recent tick price for ``symbol``, or None before the first tick."""
        bucket = self._bars.get(symbol)
        if not bucket:
            return None
        idx = max(bucket)
        return bucket[idx].close

    def vwap(self, symbol: str) -> Optional[float]:
        """
        Session volume-weighted average price for ``symbol``.

        A minute bar's typical price (T+2C+L)/4 ... (H+L+C)/3 is volume-
        weighted; this mirrors the strategy's own VWAP so the exit monitor
        and the strategy see the same number.
        """
        df = self.bars(symbol)
        if df.empty:
            return None
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_v = df["volume"].astype(float).cumsum().iloc[-1]
        if cum_v <= 0:
            return None
        return float((typical * df["volume"]).sum() / cum_v)

    def symbols(self) -> List[str]:
        return sorted(self._bars.keys())

    def has_ticked(self, symbol: str) -> bool:
        return symbol in self._bars and bool(self._bars[symbol])

    def last_tick_time(self, symbol: str) -> Optional[datetime]:
        bucket = self._bars.get(symbol)
        if not bucket:
            return None
        return bucket[max(bucket)].start

    # ------------------------------------------------------------------ #
    #  Session lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def reset_session(self) -> None:
        """Clear all bars.  Called on start-up and on a new IST session date."""
        self._bars.clear()
        self._session_date = None

    def _ensure_session(self, ist: datetime) -> None:
        ist_date = ist.date()
        if self._session_date is None:
            self._session_date = ist_date
            return
        if self._session_date != ist_date:
            logger.info(
                "New IST session date %s; resetting intraday bars.",
                ist_date,
            )
            self._bars.clear()
        self._session_date = ist_date


class MockTickFeed:
    """
    Deterministic synthetic tick generator for the paper intraday sleeve.

    Prices walk a mean-reverting Ornstein-Uhlenbeck path around each symbol's
    base price (the previous daily close), maybe themselves seeded fallbacks.
    Every symbol's path is derived from `seed` + a symbol hash, so a session
    is exactly reproducible: same flags, same bars.

    The feed is a *stream*, not a market.  `run(stop_event, on_tick)` emits
    ticks inside the mock session (9:15–15:30 IST) and does nothing outside it.
    `run_once(on_tick)` emits a single pass over the universe and is the hook
    the tests and the auto-trader use for synchronous stepping.
    """

    def __init__(
        self,
        universe: Sequence[str],
        base_prices: Mapping[str, float],
        *,
        seed: int = 42,
        interval_ms: int = _DEFAULT_TICK_INTERVAL_MS,
        volume_mean: float = _DEFAULT_VOLUME_MEAN,
        intraday_sigma: float = 2.5e-4,
        mean_revert_strength: float = 0.12,
        tick_size: float = 0.05,
        clock: Callable[[], datetime] = utc_now,
        session_start: time = _MOCK_OPEN,
        session_end: time = _MOCK_CLOSE,
    ) -> None:
        self.universe = list(universe)
        self.base_prices = {s: float(base_prices.get(s, 1000.0)) for s in universe}
        self.seed = int(seed)
        self.interval_ms = int(interval_ms)
        self.volume_mean = float(volume_mean)
        self.intraday_sigma = float(intraday_sigma)
        self.mean_revert_strength = float(mean_revert_strength)
        self.tick_size = float(tick_size)
        self._clock = clock
        self.session_start = session_start
        self.session_end = session_end
        self._rngs: Dict[str, np.random.Generator] = {}
        self._deviation: Dict[str, float] = {}
        self._current_price: Dict[str, float] = {}
        self._last_seen_date: Optional[date] = None
        self._reset_state()

    def _reset_state(self) -> None:
        for sym in self.universe:
            h = int(hashlib.sha256(sym.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng((self.seed ^ h) & 0xFFFFFFFF)
            self._rngs[sym] = rng
            self._deviation[sym] = 0.0
            self._current_price[sym] = self.base_prices[sym]

    def source_name(self) -> str:
        return "MockTickFeed"

    def is_session_time(self) -> bool:
        return self.session_start <= to_ist(self._clock()).time() < self.session_end

    def _now(self) -> datetime:
        now = to_ist(self._clock())
        if self._last_seen_date is not None and self._last_seen_date != now.date():
            # A new IST session started: restart the walk at the same base so
            # multi-day runs do not drift with yesterday's cumulative noise.
            self._reset_state()
        self._last_seen_date = now.date()
        return now

    @staticmethod
    def _symbol_seed(symbol: str, seed: int) -> int:
        h = hashlib.sha256(symbol.encode()).hexdigest()[:8]
        return (int(seed) ^ int(h, 16)) & 0xFFFFFFFF

    def _next_tick(self, symbol: str) -> Tuple[float, int]:
        rng = self._rngs[symbol]
        dev = self._deviation[symbol]
        dev = (1.0 - self.mean_revert_strength) * dev + self.intraday_sigma * float(
            rng.standard_normal()
        )
        self._deviation[symbol] = dev
        log_p = math.log(max(self.base_prices[symbol], 0.5)) + dev
        raw = math.exp(log_p)
        price = round(raw / self.tick_size) * self.tick_size
        self._current_price[symbol] = max(price, self.tick_size)
        vol = max(1, int(math.exp(math.log(self.volume_mean) + 1.4 * float(rng.standard_normal()))))
        return self._current_price[symbol], vol

    async def run(self, stop_event: asyncio.Event, on_tick: Callable) -> int:
        """
        Emit ticks until ``stop_event`` is set.

        Outside the mock session the feed sleeps untouched and emits nothing —
        the loop, not the feed, owns session gating for its own decisions, but
        the feed must not manufacture prices after hours.
        """
        emitted = 0
        while not stop_event.is_set():
            if not self.is_session_time():
                await asyncio.sleep(1.0)
                continue
            emitted += await self.run_once(on_tick)
            await asyncio.sleep(self.interval_ms / 1000.0)
        return emitted

    async def run_once(self, on_tick: Callable) -> int:
        """
        One synchronous pass over the universe: emit one tick per symbol.

        ``on_tick(symbol, ts, price, volume)`` is called for every symbol,
        in universe order, timestamped with the feed clock.
        """
        if not self.universe:
            return 0
        now = self._now()
        for sym in self.universe:
            price, vol = self._next_tick(sym)
            on_tick(sym, now, price, vol)
        return len(self.universe)


class TickQuoteSource:
    """
    A BrokerInterface-shaped price source backed by the live aggregator.

    It is NOT a broker: it has no order interface.  The PaperBroker only needs
    a quote feed, and this is that feed.  Delegating the daily/interday reads
    to a `delegate` data broker keeps 1-minute precision where the aggregator
    owns it and lets the underlying provider own everything coarser.

    ``get_quote`` returns quotes keyed by BOTH the caller's raw key
    (e.g. ``"NSE:RELIANCE"``) and the bare symbol (``"RELIANCE"``).  PaperBroker
    looks up fills via the raw key and the order path via either form, so
    answering both is what keeps a fill and a quote cache in the same truth.
    """

    trading_mode = "data"
    instrument_token_prefix = "MOCK"

    def __init__(
        self,
        aggregator: TickBarAggregator,
        delegate: Optional[Any] = None,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._aggregator = aggregator
        self._delegate = delegate
        self._clock = clock
        self._connected = False

    # ------------------------------------------------------------------ #
    #  BrokerInterface baseline surface                                   #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "MockTickQuoteSource"

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        h = hashlib.sha256(f"{exchange}:{symbol}".encode()).hexdigest()[:8]
        return int(h, 16) % (2**31)

    # ------------------------------------------------------------------ #
    #  Quotes                                                             #
    # ------------------------------------------------------------------ #

    def _now(self) -> datetime:
        return to_ist(self._clock())

    def _quote_for_price(self, symbol: str, price: float, volume: int, ts: datetime) -> dict:
        return {
            "last_price": float(price),
            "volume": int(volume),
            "ohlc": {
                "open": float(price),
                "high": float(price),
                "low": float(price),
                "close": float(price),
            },
            "timestamp": ts.isoformat(),
            "depth": {
                "buy": [{"price": float(price), "quantity": int(volume)}],
                "sell": [{"price": float(price), "quantity": int(volume)}],
            },
            "source": self._aggregator._source_name,
            "synthetic": True,
        }

    async def get_quote(self, symbols: Sequence[str]) -> Dict[str, dict]:
        """
        Current price for each requested symbol, keyed by both the raw key and
        the bare symbol.  Symbols with no fresh price are omitted — callers
        treat that as "not tradeable", which is what PaperBroker already does
        with a missing quote.
        """
        out: Dict[str, dict] = {}
        now = self._now()
        for raw in symbols:
            bare = str(raw).split(":")[-1]
            price = self._aggregator.last_price(bare)
            if price is None and self._delegate is not None:
                try:
                    quotes = await self._delegate.get_quote([raw])
                    price = float((quotes.get(raw) or quotes.get(bare) or {}).get("last_price", 0.0)) or None
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Mock quote fallback failed for %s: %s", raw, exc)
            if price is None:
                continue
            price = float(price)
            bar_df = self._aggregator.bars(bare)
            vol = int(bar_df["volume"].iloc[-1]) if not bar_df.empty else 0
            quote = self._quote_for_price(bare, price, vol, now)
            out[bare] = quote
            if raw != bare:
                out[raw] = quote
        return out

    async def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: Optional[date] = None,  # noqa: ARG002
        to_date: Optional[date] = None,    # noqa: ARG002
    ) -> pd.DataFrame:
        """
        1-minute intervals come from the aggregator; everything else delegates
        to the underlying data broker.  Intraday precision only exists while a
        session is flowing, which is exactly what the auto-trader needs.
        """
        if interval in ("1m", "minute", "1minute"):
            return self._aggregator.bars(symbol)
        if self._delegate is not None:
            return await self._delegate.get_historical_data(
                symbol, exchange, interval, from_date, to_date
            )
        from app.broker.marketdata import MarketDataUnavailable

        raise MarketDataUnavailable(
            f"TickQuoteSource has no {interval} history for {symbol}: "
            "no delegate data broker configured and 1-minute bars are the only "
            "intraday resolution the aggregator keeps."
        )
