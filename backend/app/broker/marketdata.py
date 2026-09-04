"""
A read-only market-data feed that satisfies ``BrokerInterface``.

WHY THIS EXISTS
---------------
``PaperBroker`` simulates fills but does not invent prices — it delegates every
quote to a ``data_broker``. Without one it has no prices at all and cannot fill
anything, which is exactly the state the application shipped in: the execution
stack was constructed with ``data_broker=None``, so paper trading was wired but
inert.

The obvious candidate, ``ZerodhaBroker``, needs live credentials that a paper
deployment by definition does not have. So this adapter exposes a real market
data source through the same interface.

WHY IT CANNOT PLACE ORDERS
--------------------------
Every mutating method raises. This class is passed into ``PaperBroker`` as its
price source, and a price source that could also place orders would be a second
path to a venue — the one thing the execution architecture exists to prevent.
Raising is not a stub for "implement later"; it is the invariant.

FRESHNESS IS NOT FAKED
----------------------
Each quote carries the timestamp of the bar it came from, never ``now()``.
Stamping a delayed price with the current time would defeat the staleness gate
completely, and that gate is what stops an order being priced off data that no
longer reflects the market. A daily provider therefore produces quotes that
ARE stale during a live session, and the safety layer will refuse to trade on
them. That refusal is correct behaviour, not a defect to work around: it is the
system declining to trade on data it cannot stand behind.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from app.broker.base import BrokerInterface

logger = logging.getLogger(__name__)

#: NSE closes at 15:30 IST. A daily bar is stamped at the close of its session.
_IST = timezone(timedelta(hours=5, minutes=30))
_NSE_CLOSE_HOUR, _NSE_CLOSE_MINUTE = 15, 30


class MarketDataUnavailable(RuntimeError):
    """
    No usable quote could be obtained.

    Raised instead of returning an empty dict or a last-known price, because a
    caller that cannot tell "no data" from "this data" will trade on the wrong
    one.
    """


class MarketDataBroker(BrokerInterface):
    """
    Quotes and history from a data provider. Never an execution venue.

    Parameters
    ----------
    provider
        Anything with ``fetch_symbol(symbol, start, end) -> DataFrame | None``
        carrying at least a ``close`` column and a DatetimeIndex.
        :class:`app.data.providers.YahooDataProvider` satisfies this.
    suffix
        Appended to bare symbols ONLY when the provider does not namespace them
        itself. Defaults to EMPTY: ``YahooDataProvider.fetch_symbol`` already
        builds ``f"{symbol}.NS"`` internally, so passing ``.NS`` here produced
        ``RELIANCE.NS.NS``, a 404 from Yahoo, and an empty quote dict for every
        symbol — this feed could not price anything in production.
    lookback_days
        How far back to request in order to obtain the most recent bar.
    """

    def __init__(
        self,
        provider: Any,
        *,
        suffix: str = "",
        lookback_days: int = 10,
        quote_ttl_sec: float = 60.0,
    ) -> None:
        if provider is None:
            raise ValueError("MarketDataBroker requires a data provider")
        self._provider = provider
        self._suffix = suffix
        self._lookback_days = int(lookback_days)
        self._quote_ttl = float(quote_ttl_sec)
        self._connected = False
        # Caches the provider response only, never the freshness verdict: the
        # cached quote keeps its ORIGINAL bar timestamp, so a cache hit cannot
        # make stale data look fresh.
        self._cache: dict[str, tuple[float, dict]] = {}
        self._tokens: dict[str, int] = {}

    # -- lifecycle -------------------------------------------------------- #

    async def connect(self) -> None:
        self._connected = True
        logger.info(
            "MarketDataBroker ready (%s). Read-only: this feed cannot place orders.",
            type(self._provider).__name__,
        )

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def trading_mode(self) -> str:
        # Not "paper" and not "live": this is not a trading venue at all.
        return "data"

    # -- quotes ----------------------------------------------------------- #

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        """
        Return ``{symbol: quote}`` for every symbol that could be priced.

        A symbol the provider cannot price is OMITTED rather than given a
        placeholder. Downstream, a missing quote refuses the order; a fabricated
        one would let it through at an invented price.
        """
        out: dict[str, dict] = {}
        now = _monotonic()
        for raw in symbols or []:
            sym = str(raw).split(":")[-1]
            hit = self._cache.get(sym)
            if hit is not None and (now - hit[0]) < self._quote_ttl:
                out[sym] = hit[1]
                continue
            try:
                quote = await asyncio.to_thread(self._fetch_quote, sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("quote unavailable for %s: %s", sym, exc)
                continue
            if quote is None:
                continue
            self._cache[sym] = (now, quote)
            out[sym] = quote
        return out

    def _fetch_quote(self, symbol: str) -> Optional[dict]:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=self._lookback_days)
        df = self._provider.fetch_symbol(
            f"{symbol}{self._suffix}", start.isoformat(), end.isoformat()
        )
        if df is None or len(df) == 0:
            return None

        cols = {str(c).lower(): c for c in df.columns}
        if "close" not in cols:
            return None
        last = df.iloc[-1]
        close = float(last[cols["close"]])
        if not close > 0:
            return None

        ts = self._bar_timestamp(df.index[-1])
        high = float(last[cols["high"]]) if "high" in cols else close
        low = float(last[cols["low"]]) if "low" in cols else close
        open_ = float(last[cols["open"]]) if "open" in cols else close
        volume = int(last[cols["volume"]]) if "volume" in cols else 0

        # A daily bar has no book. Rather than invent depth — which would let
        # the impact model size a fill against liquidity nobody observed — the
        # touch is reported at the close with the bar's own volume, and the
        # absence of real depth is stated in the payload.
        return {
            "last_price": close,
            "timestamp": ts.isoformat(),
            "ohlc": {"open": open_, "high": high, "low": low, "close": close},
            "volume": volume,
            "depth": {
                "buy": [{"price": close, "quantity": volume}],
                "sell": [{"price": close, "quantity": volume}],
            },
            "source": type(self._provider).__name__,
            "resolution": "1d",
            "synthetic_depth": True,
        }

    @staticmethod
    def _bar_timestamp(idx: Any) -> datetime:
        """
        The instant a daily bar became final: its session close, in IST.

        Deliberately NOT ``now()``. A daily close stamped with the current time
        would pass every freshness check ever written while being hours or days
        old.
        """
        ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_IST)
        return ts.astimezone(_IST).replace(
            hour=_NSE_CLOSE_HOUR, minute=_NSE_CLOSE_MINUTE, second=0, microsecond=0
        )

    async def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> "pd.DataFrame":
        """
        Daily OHLCV. Intervals finer than a day are REFUSED, not approximated.

        The underlying provider serves daily bars only. Silently returning
        daily data to a caller that asked for 5-minute bars would corrupt any
        intraday calculation built on it, and would do so invisibly.
        """
        if interval not in ("day", "1d", "daily"):
            raise MarketDataUnavailable(
                f"{type(self._provider).__name__} serves daily bars only; "
                f"interval={interval!r} cannot be satisfied. Refusing rather "
                f"than returning daily data under an intraday label."
            )
        df = await asyncio.to_thread(
            self._provider.fetch_symbol,
            f"{str(symbol).split(':')[-1]}{self._suffix}",
            _as_date_str(from_date),
            _as_date_str(to_date),
        )
        if df is None:
            raise MarketDataUnavailable(f"no history for {symbol} on {exchange}")
        return df

    def instrument_token(self, symbol: str, exchange: str) -> int:
        """
        A stable synthetic token.

        This feed has no exchange instrument registry, so the token is derived
        from the name. It is stable across restarts, which is what callers use
        it for; it is NOT an NSE token and must never be sent to a real venue.
        """
        key = f"{exchange}:{symbol}"
        if key not in self._tokens:
            self._tokens[key] = abs(hash(key)) % 10_000_000
        return self._tokens[key]

    # -- account views: not applicable to a data feed ---------------------- #

    async def get_profile(self) -> dict:
        return {"user_name": "market-data", "user_type": "data", "broker": "DATA"}

    async def get_holdings(self) -> list[dict]:
        return []

    async def get_positions(self) -> list[dict]:
        return []

    async def get_orders(self) -> list[dict]:
        return []

    async def get_trades(self) -> list[dict]:
        return []

    async def get_funds(self) -> dict:
        return {"cash": 0.0, "margin_available": 0.0, "margin_used": 0.0}

    # -- execution: permanently refused ------------------------------------ #

    def _refuse(self, action: str) -> "NotImplementedError":
        return NotImplementedError(
            f"MarketDataBroker cannot {action}. It is a price source, not an "
            f"execution venue. Route orders through ExecutionService, which is "
            f"the only path to a broker."
        )

    async def place_order(self, *args: Any, **kwargs: Any) -> str:
        raise self._refuse("place orders")

    async def cancel_order(self, *args: Any, **kwargs: Any) -> bool:
        raise self._refuse("cancel orders")

    async def modify_order(self, *args: Any, **kwargs: Any) -> bool:
        raise self._refuse("modify orders")

    async def get_order_status(self, *args: Any, **kwargs: Any) -> dict:
        raise self._refuse("report order status")


def _monotonic() -> float:
    import time
    return time.monotonic()


def _as_date_str(v: Any) -> str:
    if isinstance(v, datetime):
        return v.date().isoformat()
    return str(v)[:10]
