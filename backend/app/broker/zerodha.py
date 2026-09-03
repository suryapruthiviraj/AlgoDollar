"""Zerodha / KiteConnect broker adapter."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import pytz

from .base import (
    BrokerInterface,
    Exchange,
    OrderType,
    Product,
    TransactionType,
)

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = (9, 15)   # HH, MM
MARKET_CLOSE = (15, 30)

# Kite rate-limit buckets (requests per second)
_RATE_LIMIT_ORDERS = 3
_RATE_LIMIT_DATA = 10

_KITE_ORDER_TYPE_MAP = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "SL",
    OrderType.SL_M: "SL-M",
}
_KITE_TXN_MAP = {
    TransactionType.BUY: "BUY",
    TransactionType.SELL: "SELL",
}
_KITE_PRODUCT_MAP = {
    Product.CNC: "CNC",
    Product.MIS: "MIS",
    Product.NRML: "NRML",
    Product.CO: "CO",
    Product.BO: "BO",
}


class _RateLimiter:
    """Token-bucket rate limiter (synchronous, used inside async context)."""

    def __init__(self, rps: int) -> None:
        self._rps = rps
        self._timestamps: deque[float] = deque()

    async def acquire(self) -> None:
        now = time.monotonic()
        # drop timestamps older than 1 second
        while self._timestamps and now - self._timestamps[0] > 1.0:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._rps:
            wait = 1.0 - (now - self._timestamps[0])
            if wait > 0:
                await asyncio.sleep(wait)
        self._timestamps.append(time.monotonic())


class ZerodhaBroker(BrokerInterface):
    """Live broker adapter wrapping the kiteconnect SDK."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = access_token
        self._kite = None          # KiteConnect instance
        self._kws = None           # KiteTicker instance
        self._connected = False
        self._last_ticks: dict[str, dict] = {}
        self._tick_timestamps: dict[str, float] = {}
        self._instruments: dict[tuple[str, str], int] = {}   # (symbol, exchange) -> token
        self._order_rl = _RateLimiter(_RATE_LIMIT_ORDERS)
        self._data_rl = _RateLimiter(_RATE_LIMIT_DATA)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """Initialise KiteConnect and verify the session."""
        try:
            from kiteconnect import KiteConnect  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "kiteconnect package is not installed. "
                "Run: pip install kiteconnect"
            ) from exc

        self._kite = KiteConnect(api_key=self._api_key)
        if self._access_token:
            self._kite.set_access_token(self._access_token)
            # validate by fetching profile
            loop = asyncio.get_event_loop()
            profile = await loop.run_in_executor(None, self._kite.profile)
            logger.info("ZerodhaBroker connected as %s", profile.get("user_name"))
            self._connected = True
            await self._load_instruments()
        else:
            logger.warning(
                "ZerodhaBroker created without access_token; "
                "call exchange_token() before trading."
            )

    async def disconnect(self) -> None:
        """Close WebSocket and invalidate session."""
        if self._kws is not None:
            self._kws.stop()
            self._kws = None
        if self._kite is not None and self._connected:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._kite.invalidate_access_token)
            except Exception:
                pass
        self._connected = False
        logger.info("ZerodhaBroker disconnected.")

    # ------------------------------------------------------------------ #
    #  Auth helpers                                                        #
    # ------------------------------------------------------------------ #

    def generate_login_url(self) -> str:
        """Return the Kite login URL for OAuth."""
        from kiteconnect import KiteConnect  # type: ignore[import]
        kite = KiteConnect(api_key=self._api_key)
        return kite.login_url()

    async def exchange_token(self, request_token: str) -> str:
        """Exchange a one-time request_token for a persistent access_token."""
        from kiteconnect import KiteConnect  # type: ignore[import]
        kite = KiteConnect(api_key=self._api_key)
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: kite.generate_session(request_token, api_secret=self._api_secret),
        )
        self._access_token = data["access_token"]
        self._kite = kite
        self._kite.set_access_token(self._access_token)
        self._connected = True
        await self._load_instruments()
        logger.info("Access token obtained and session active.")
        return self._access_token

    # ------------------------------------------------------------------ #
    #  Instrument cache                                                    #
    # ------------------------------------------------------------------ #

    async def _load_instruments(self) -> None:
        """Download and cache instrument list from Kite."""
        loop = asyncio.get_event_loop()
        try:
            instruments = await loop.run_in_executor(
                None, self._kite.instruments
            )
            self._instruments = {
                (i["tradingsymbol"], i["exchange"]): i["instrument_token"]
                for i in instruments
            }
            logger.info("Loaded %d instruments from Kite.", len(self._instruments))
        except Exception as exc:
            logger.warning("Could not load instruments: %s", exc)

    def instrument_token(self, symbol: str, exchange: str) -> int:
        key = (symbol, exchange)
        if key not in self._instruments:
            raise KeyError(f"Instrument not found: {symbol} / {exchange}")
        return self._instruments[key]

    # ------------------------------------------------------------------ #
    #  WebSocket (streaming quotes)                                       #
    # ------------------------------------------------------------------ #

    def _setup_ws(self, tokens: list[int]) -> None:
        """Initialise and connect KiteTicker for the given instrument tokens."""
        from kiteconnect import KiteTicker  # type: ignore[import]
        self._kws = KiteTicker(self._api_key, self._access_token)
        self._kws.on_ticks = self._on_ticks
        self._kws.on_connect = self._on_ws_connect(tokens)
        self._kws.on_close = self._on_ws_close
        self._kws.on_error = self._on_ws_error
        self._kws.connect(threaded=True)

    def _on_ws_connect(self, tokens: list[int]):
        def handler(ws, response):
            logger.info("WebSocket connected; subscribing %d tokens.", len(tokens))
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
        return handler

    def _on_ticks(self, ws, ticks: list[dict]) -> None:
        now = time.monotonic()
        for tick in ticks:
            token = tick.get("instrument_token")
            # look up symbol
            symbol = tick.get("tradingsymbol", str(token))
            self._last_ticks[symbol] = tick
            self._tick_timestamps[symbol] = now

    def _on_ws_close(self, ws, code, reason) -> None:
        logger.warning("WebSocket closed: %s %s", code, reason)

    def _on_ws_error(self, ws, code, reason) -> None:
        logger.error("WebSocket error: %s %s", code, reason)

    def is_stale_tick(self, symbol: str, max_age_seconds: float = 30.0) -> bool:
        ts = self._tick_timestamps.get(symbol)
        if ts is None:
            return True
        return (time.monotonic() - ts) > max_age_seconds

    # ------------------------------------------------------------------ #
    #  Market hours                                                        #
    # ------------------------------------------------------------------ #

    def is_market_open(self) -> bool:
        now_ist = datetime.now(tz=IST)
        if now_ist.weekday() >= 5:   # Saturday / Sunday
            return False
        t = (now_ist.hour, now_ist.minute)
        return MARKET_OPEN <= t < MARKET_CLOSE

    # ------------------------------------------------------------------ #
    #  Retry helper                                                        #
    # ------------------------------------------------------------------ #

    async def _call_kite(self, fn, *args, retries: int = 3, **kwargs):
        """Run a blocking kite call in executor with exponential-backoff retry."""
        loop = asyncio.get_event_loop()
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
            except Exception as exc:
                last_exc = exc
                backoff = 0.5 * (2 ** attempt)
                logger.warning(
                    "Kite call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    fn.__name__, attempt + 1, retries, exc, backoff,
                )
                await asyncio.sleep(backoff)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    #  Account methods                                                     #
    # ------------------------------------------------------------------ #

    async def get_profile(self) -> dict:
        await self._data_rl.acquire()
        return await self._call_kite(self._kite.profile)

    async def get_holdings(self) -> list[dict]:
        await self._data_rl.acquire()
        return await self._call_kite(self._kite.holdings)

    async def get_positions(self) -> list[dict]:
        await self._data_rl.acquire()
        pos = await self._call_kite(self._kite.positions)
        # kite returns {"net": [...], "day": [...]}
        return pos.get("net", [])

    async def get_orders(self) -> list[dict]:
        await self._data_rl.acquire()
        return await self._call_kite(self._kite.orders)

    async def get_trades(self) -> list[dict]:
        await self._data_rl.acquire()
        return await self._call_kite(self._kite.trades)

    async def get_funds(self) -> dict:
        await self._data_rl.acquire()
        raw = await self._call_kite(self._kite.margins)
        equity = raw.get("equity", {})
        return {
            "cash": equity.get("available", {}).get("cash", 0.0),
            "margin_available": equity.get("available", {}).get("live_balance", 0.0),
            "margin_used": equity.get("utilised", {}).get("debits", 0.0),
        }

    # ------------------------------------------------------------------ #
    #  Market data                                                         #
    # ------------------------------------------------------------------ #

    async def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        await self._data_rl.acquire()
        raw = await self._call_kite(self._kite.quote, symbols)
        result: dict[str, dict] = {}
        for key, data in raw.items():
            result[key] = {
                "last_price": data.get("last_price"),
                "bid": data.get("depth", {}).get("buy", [{}])[0].get("price"),
                "ask": data.get("depth", {}).get("sell", [{}])[0].get("price"),
                "ohlc": data.get("ohlc", {}),
                "volume": data.get("volume"),
                "oi": data.get("oi"),
                "timestamp": data.get("timestamp"),
            }
        return result

    async def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        await self._data_rl.acquire()
        token = self.instrument_token(symbol, exchange)
        raw = await self._call_kite(
            self._kite.historical_data,
            token,
            from_date,
            to_date,
            interval,
            continuous=False,
            oi=False,
        )
        if not raw:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(raw)
        df.rename(columns={"date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume"]].copy()

    # ------------------------------------------------------------------ #
    #  Order management                                                    #
    # ------------------------------------------------------------------ #

    async def place_order(
        self,
        symbol: str,
        exchange: str,
        txn_type: TransactionType,
        qty: int,
        price: float,
        order_type: OrderType,
        product: Product,
        tag: str = "",
    ) -> str:
        await self._order_rl.acquire()
        kwargs: dict = dict(
            variety=self._kite.VARIETY_REGULAR,
            exchange=exchange,
            tradingsymbol=symbol,
            transaction_type=_KITE_TXN_MAP[txn_type],
            quantity=qty,
            product=_KITE_PRODUCT_MAP[product],
            order_type=_KITE_ORDER_TYPE_MAP[order_type],
            tag=tag[:20] if tag else None,
        )
        if order_type in (OrderType.LIMIT, OrderType.SL):
            kwargs["price"] = price
        if order_type == OrderType.SL:
            kwargs["trigger_price"] = price  # caller sets stop; adjust if needed
        order_id = await self._call_kite(self._kite.place_order, **kwargs)
        logger.info(
            "Order placed: %s %s %s x%d @ %.2f → order_id=%s",
            txn_type.value, symbol, exchange, qty, price, order_id,
        )
        return str(order_id)

    async def cancel_order(self, order_id: str) -> bool:
        await self._order_rl.acquire()
        try:
            await self._call_kite(
                self._kite.cancel_order,
                variety=self._kite.VARIETY_REGULAR,
                order_id=order_id,
            )
            logger.info("Order cancelled: %s", order_id)
            return True
        except Exception as exc:
            logger.error("Cancel order %s failed: %s", order_id, exc)
            return False

    async def modify_order(
        self,
        order_id: str,
        qty: Optional[int] = None,
        price: Optional[float] = None,
    ) -> bool:
        await self._order_rl.acquire()
        kwargs: dict = dict(
            variety=self._kite.VARIETY_REGULAR,
            order_id=order_id,
        )
        if qty is not None:
            kwargs["quantity"] = qty
        if price is not None:
            kwargs["price"] = price
        try:
            await self._call_kite(self._kite.modify_order, **kwargs)
            logger.info("Order modified: %s qty=%s price=%s", order_id, qty, price)
            return True
        except Exception as exc:
            logger.error("Modify order %s failed: %s", order_id, exc)
            return False

    async def get_order_status(self, order_id: str) -> dict:
        await self._data_rl.acquire()
        history = await self._call_kite(self._kite.order_history, order_id=order_id)
        # most recent entry
        return history[-1] if history else {}

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def trading_mode(self) -> str:
        return "live"
