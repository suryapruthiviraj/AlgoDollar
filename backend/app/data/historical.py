"""
historical.py — Async historical OHLCV data provider for AlgoDollar.

Wraps the broker's get_historical_data() call with:
  - Redis caching (TTL varies by interval).
  - Missing-data handling (forward-fill ≤ 5 days, else NaN).
  - Split/bonus adjustment via broker-supplied adjustment_factor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
from datetime import date, datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL per interval (seconds)
# ---------------------------------------------------------------------------
_TTL: Dict[str, int] = {
    "day": 86_400,       # 24 h
    "week": 86_400,
    "month": 86_400,
    "60minute": 3_600,
    "30minute": 1_800,
    "15minute": 900,
    "5minute": 300,
    "minute": 300,       # 5 min — covers most intraday needs
}

# Maximum consecutive missing days to forward-fill; beyond this mark as NaN.
_MAX_FFILL_DAYS = 5


def _cache_key(symbol: str, exchange: str, from_date: str, to_date: str, interval: str) -> str:
    raw = f"{symbol}:{exchange}:{from_date}:{to_date}:{interval}"
    return "algodollar:ohlcv:" + hashlib.sha256(raw.encode()).hexdigest()


class HistoricalDataProvider:
    """
    Async provider for OHLCV historical data.

    Parameters
    ----------
    broker_client
        Broker client instance with ``get_historical_data()`` and
        ``get_adjustment_factor()`` async methods.
    redis_client
        Async Redis client (aioredis or redis.asyncio).  Pass ``None`` to
        disable caching (useful in unit tests).
    """

    def __init__(self, broker_client, redis_client=None):
        self._broker = broker_client
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_ohlcv(
        self,
        symbol: str,
        exchange: str = "NSE",
        from_date: str | date | datetime = "",
        to_date: str | date | datetime = "",
        interval: str = "day",
        adjust: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a single symbol.

        Parameters
        ----------
        symbol : str
            E.g. ``"RELIANCE"``.
        exchange : str
            ``"NSE"`` or ``"BSE"``.
        from_date, to_date : str | date | datetime
            Inclusive date range.  Strings accepted as ``"YYYY-MM-DD"``.
        interval : str
            ``"day"``, ``"minute"``, ``"5minute"``, ``"15minute"``,
            ``"30minute"``, ``"60minute"``, ``"week"``, ``"month"``.
        adjust : bool
            Whether to apply split/bonus adjustment.

        Returns
        -------
        pd.DataFrame
            Columns: ``date, open, high, low, close, volume``.
            Index: RangeIndex.  ``date`` column is ``datetime64``.
            Rows with > ``_MAX_FFILL_DAYS`` consecutive missing closes are
            NaN (not forward-filled) to avoid stale-price look-ahead.
        """
        from_str = self._to_str(from_date)
        to_str = self._to_str(to_date)
        cache_key = _cache_key(symbol, exchange, from_str, to_str, interval)
        ttl = _TTL.get(interval, 3_600)

        # Try cache first
        cached = await self._cache_get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: %s", cache_key)
            return cached

        # Fetch from broker
        raw = await self._broker.get_historical_data(
            symbol=symbol,
            exchange=exchange,
            from_date=from_str,
            to_date=to_str,
            interval=interval,
        )
        df = self._normalize(raw)

        # Adjust for splits / bonuses
        if adjust:
            df = await self._apply_adjustment(df, symbol, exchange)

        # Handle missing data
        df = self._handle_missing(df)

        # Cache result
        await self._cache_set(cache_key, df, ttl)
        return df

    async def get_multiple(
        self,
        symbols: list[str],
        from_date: str | date | datetime,
        to_date: str | date | datetime,
        interval: str = "day",
        adjust: bool = True,
        max_concurrent: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple symbols concurrently.

        Parameters
        ----------
        symbols : list[str]
        from_date, to_date : str | date | datetime
        interval : str
        adjust : bool
        max_concurrent : int
            Max simultaneous broker calls to avoid rate-limiting.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys are symbols; missing/errored symbols are excluded with a
            warning logged.
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _fetch(sym: str) -> tuple[str, Optional[pd.DataFrame]]:
            async with sem:
                try:
                    df = await self.get_ohlcv(
                        sym, from_date=from_date, to_date=to_date,
                        interval=interval, adjust=adjust,
                    )
                    return sym, df
                except Exception as exc:
                    logger.warning("Failed to fetch %s: %s", sym, exc)
                    return sym, None

        tasks = [_fetch(s) for s in symbols]
        results = await asyncio.gather(*tasks)
        return {sym: df for sym, df in results if df is not None}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_str(d: str | date | datetime) -> str:
        if isinstance(d, str):
            return d
        if isinstance(d, (date, datetime)):
            return d.strftime("%Y-%m-%d")
        return str(d)

    @staticmethod
    def _normalize(raw) -> pd.DataFrame:
        """
        Normalize broker response into a standard DataFrame.

        The broker may return a list of dicts or a DataFrame.  We produce:
          date (datetime64), open, high, low, close, volume (all float).
        """
        if isinstance(raw, pd.DataFrame):
            df = raw.copy()
        elif isinstance(raw, list):
            df = pd.DataFrame(raw)
        else:
            raise ValueError(f"Unexpected broker response type: {type(raw)}")

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Ensure 'date' column exists
        date_col_candidates = ["date", "datetime", "timestamp", "time"]
        for c in date_col_candidates:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
        if "date" not in df.columns:
            raise ValueError(f"No date column found. Columns: {df.columns.tolist()}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["date", "open", "high", "low", "close", "volume"]]

    async def _apply_adjustment(
        self, df: pd.DataFrame, symbol: str, exchange: str
    ) -> pd.DataFrame:
        """
        Apply split/bonus adjustment factors from the broker.

        Broker should provide adjustment_factor as a time series.  We multiply
        all OHLC columns by the factor up to each event date.  Volume is
        divided by the factor (shares outstanding increase on splits).

        If the broker does not support this, returns df unchanged.
        """
        try:
            factors = await self._broker.get_adjustment_factor(symbol, exchange)
        except (AttributeError, NotImplementedError):
            logger.debug("Broker does not support adjustment factors for %s", symbol)
            return df

        if not factors:
            return df

        # factors: list of dict {date: "YYYY-MM-DD", factor: float}
        df = df.copy()
        for event in sorted(factors, key=lambda x: x["date"], reverse=True):
            event_date = pd.to_datetime(event["date"])
            factor = float(event["factor"])
            if factor <= 0:
                continue
            mask = df["date"] < event_date
            price_cols = ["open", "high", "low", "close"]
            df.loc[mask, price_cols] = df.loc[mask, price_cols] * factor
            df.loc[mask, "volume"] = df.loc[mask, "volume"] / factor

        return df

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle gaps in OHLCV data.

        Strategy
        --------
        1. Forward-fill consecutive NaN runs of close ≤ _MAX_FFILL_DAYS.
        2. Longer runs are left as NaN (do not invent stale prices).
        3. For open/high/low, use close value when unavailable (conservative).
        4. Volume NaN → 0 (no activity assumed).
        """
        if df.empty:
            return df

        df = df.copy()
        close = df["close"]

        # Identify consecutive NaN groups
        is_nan = close.isna()
        group = (is_nan != is_nan.shift()).cumsum()
        nan_run_lengths = is_nan.groupby(group).transform("sum")

        # Forward-fill only short runs
        ffill_mask = is_nan & (nan_run_lengths <= _MAX_FFILL_DAYS)
        df["close"] = close.ffill().where(
            ~is_nan | ffill_mask, other=np.nan
        )

        # OHLC: if open/high/low missing but close present, use close
        for col in ["open", "high", "low"]:
            missing = df[col].isna() & df["close"].notna()
            df.loc[missing, col] = df.loc[missing, "close"]

        # Volume: fill NaN with 0
        df["volume"] = df["volume"].fillna(0.0)

        # Re-apply ffill logic properly for close after fixing OHLC
        # (re-do to capture any OHLC-induced close fills)
        close2 = df["close"]
        is_nan2 = close2.isna()
        group2 = (is_nan2 != is_nan2.shift()).cumsum()
        nan_run2 = is_nan2.groupby(group2).transform("sum")
        ffill_mask2 = is_nan2 & (nan_run2 <= _MAX_FFILL_DAYS)
        df["close"] = close2.ffill().where(
            ~is_nan2 | ffill_mask2, other=np.nan
        )
        return df

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    async def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        if self._redis is None:
            return None
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as exc:
            logger.warning("Redis get failed for %s: %s", key, exc)
            return None

    async def _cache_set(self, key: str, df: pd.DataFrame, ttl: int) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.setex(key, ttl, pickle.dumps(df))
        except Exception as exc:
            logger.warning("Redis set failed for %s: %s", key, exc)
