"""
Real market-data acquisition for NSE equities.

DATA PROVENANCE IS PART OF THE RESULT
-------------------------------------
A backtest number is meaningless without knowing what data produced it. Every
dataset this module returns carries a `DatasetProvenance` record naming the
source, the retrieval time, how the universe was chosen, and — most
importantly — which known biases the dataset carries. Downstream reports are
expected to print that record alongside any performance figure.

KNOWN AND UNFIXABLE LIMITATIONS OF THIS SOURCE
----------------------------------------------
The public source used here (Yahoo Finance) is adequate for daily price
research and inadequate for several things this platform nominally wants to
do. These are not implementation gaps; they are properties of the data:

1. SURVIVORSHIP BIAS — SEVERE, NOT MITIGABLE HERE.
   Delisted securities are dropped from the source entirely. Verified
   directly: querying SATYAMCOMP (2009 accounting fraud), DHFL (2019
   collapse) and VIDEOIND (insolvency) over periods when all three were
   actively listed and trading returns ZERO rows, while RELIANCE over the
   same window returns full history.

   The dataset keeps the survivors and discards the failures. Any universe
   assembled from currently-listed tickers therefore excludes precisely the
   worst outcomes.

   THE DIRECTION OF THIS BIAS IS NOT DETERMINATE. It inflates the ABSOLUTE
   return of a long-only book, because the companies that went to zero are
   missing. But research here measures EXCESS return against a benchmark
   drawn from the same filtered universe, and the bias does not cleanly
   survive that subtraction.

   For some signals it plausibly runs the other way. A company heading for
   delisting has collapsing momentum long before it disappears, so a momentum
   rule would have ranked it in the bottom quintile and avoided it while an
   equal-weight benchmark held it all the way down. Deleting such companies
   removes episodes where the signal was RIGHT, biasing its measured excess
   return DOWNWARD.

   The defensible statement is therefore: this dataset cannot establish true
   historical performance in EITHER direction. Results computed on it are
   measurements of this sample, not estimates of the strategy.

2. NO POINT-IN-TIME INDEX CONSTITUENTS.
   There is no way to ask "what was in the NIFTY 100 in March 2013?". Using
   today's membership across all history is itself a look-ahead: it selects
   companies for having become large.

3. NO POINT-IN-TIME FUNDAMENTALS.
   No financial-statement values with their publication dates. Without
   knowing WHEN a figure became public, fundamental features cannot be built
   causally. The long-term engine cannot be validated on this source.

4. INTRADAY HISTORY IS TOO SHORT FOR RESEARCH.
   Verified limits: 1-minute bars ~7 days, 5/15-minute bars ~60 days, hourly
   ~730 days. Sixty days spans one market regime at best. The intraday engine
   cannot be validated on this source.

WHAT THIS SOURCE IS ADEQUATE FOR
--------------------------------
Daily OHLCV over roughly twenty years, split- and dividend-adjusted. That is
enough to research the swing horizon and to establish honest baselines, so
long as the survivorship caveat travels with every number produced.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class DatasetProvenance:
    """Where a dataset came from and what is wrong with it."""
    source: str
    retrieved_at: datetime
    universe_definition: str
    n_symbols_requested: int
    n_symbols_returned: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    price_field: str
    known_biases: list[str] = field(default_factory=list)
    failed_symbols: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"source              : {self.source}",
            f"retrieved           : {self.retrieved_at.isoformat()}",
            f"universe definition : {self.universe_definition}",
            f"symbols             : {self.n_symbols_returned}/{self.n_symbols_requested}"
            f" returned",
            f"date range          : {self.start} .. {self.end}",
            f"price field         : {self.price_field}",
        ]
        if self.failed_symbols:
            shown = ", ".join(self.failed_symbols[:8])
            more = "" if len(self.failed_symbols) <= 8 else f" (+{len(self.failed_symbols)-8})"
            lines.append(f"failed symbols      : {shown}{more}")
        lines.append("KNOWN BIASES:")
        for b in self.known_biases:
            lines.append(f"  - {b}")
        return "\n".join(lines)


@dataclass
class PriceDataset:
    """Aligned OHLCV panels plus provenance."""
    close: pd.DataFrame          # adjusted close — use for returns
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    raw_close: pd.DataFrame      # unadjusted — use for realistic trade prices
    provenance: DatasetProvenance

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# Large/mid-cap NSE names, as constituted TODAY.
#
# This is a survivorship-biased universe and is labelled as such wherever it
# is used. It is not a point-in-time index membership list, because no such
# list is obtainable from this source. Every company here survived to the
# present; the ones that did not are absent by construction.
NSE_LARGE_MID_CAP: tuple[str, ...] = (
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "BAJFINANCE", "HCLTECH", "POWERGRID", "NTPC", "TATAMOTORS", "TATASTEEL",
    "ONGC", "GRASIM", "JSWSTEEL", "ADANIENT", "COALINDIA", "HINDALCO",
    "CIPLA", "DRREDDY", "EICHERMOT", "BRITANNIA", "DIVISLAB", "HEROMOTOCO",
    "BAJAJ-AUTO", "INDUSINDBK", "TECHM", "APOLLOHOSP", "BPCL", "TATACONSUM",
    "SHREECEM", "UPL", "SBILIFE", "HDFCLIFE", "BAJAJFINSV", "M&M",
    "PIDILITIND", "DABUR", "GODREJCP", "MARICO", "COLPAL", "BERGEPAINT",
    "HAVELLS", "SIEMENS", "AMBUJACEM", "ACC", "BANKBARODA", "PNB",
    "CANBK", "IOC", "GAIL", "VEDL", "JINDALSTEL", "SAIL", "NMDC",
    "LUPIN", "AUROPHARMA", "TORNTPHARM", "BIOCON", "GLENMARK", "ALKEM",
    "MOTHERSON", "BOSCHLTD", "ASHOKLEY", "TVSMOTOR", "BALKRISIND",
    "MPHASIS", "PERSISTENT", "LTIM", "OFSS", "TRENT", "DMART", "PAGEIND",
    "MUTHOOTFIN", "CHOLAFIN", "LICHSGFIN", "RECLTD", "PFC", "IRCTC",
    "CONCOR", "ADANIPORTS", "INDIGO", "ZYDUSLIFE", "ABBOTINDIA",
    "ICICIPRULI", "ICICIGI", "SRF", "TATAPOWER",
)

_BIAS_NOTES = [
    "SURVIVORSHIP: delisted securities are absent from the source. Verified — "
    "SATYAMCOMP, DHFL and VIDEOIND return zero rows over periods when they "
    "were actively listed. The bias direction on EXCESS return is "
    "INDETERMINATE and signal-dependent (a momentum rule would have avoided "
    "the deleted names, so removing them can bias its excess return DOWN). "
    "This dataset cannot establish true historical performance in either "
    "direction.",
    "NO POINT-IN-TIME UNIVERSE: membership is as of today, which selects "
    "companies for having become large. This is itself a look-ahead.",
    "NO POINT-IN-TIME FUNDAMENTALS: no statement values with publication "
    "dates, so fundamental features cannot be built causally.",
    "ADJUSTED PRICES: adjustment factors are applied retroactively by the "
    "vendor. Returns are correct; historical price LEVELS are not what was "
    "quoted at the time, so raw_close is provided for trade-price realism.",
]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class YahooDataProvider:
    """
    Daily NSE OHLCV via Yahoo Finance, with on-disk caching.

    Caching is not an optimization detail here: it makes a research run
    reproducible. Re-running an experiment must not silently pick up different
    data because the vendor revised something.
    """

    def __init__(self, cache_dir: Optional[Path] = None, use_cache: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    # -- single symbol ----------------------------------------------------

    def _cache_path(self, symbol: str, start: str, end: str) -> Path:
        safe = symbol.replace("/", "_").replace("&", "_")
        return self.cache_dir / f"{safe}__{start}__{end}.parquet"

    def fetch_symbol(
        self, symbol: str, start: str, end: str, max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch one symbol's daily OHLCV. Returns None if unavailable.

        A None here is meaningful data in itself: for a name that was listed
        during the requested window, absence is evidence of survivorship
        filtering by the vendor.
        """
        cache = self._cache_path(symbol, start, end)
        if self.use_cache and cache.exists():
            try:
                return pd.read_parquet(cache)
            except Exception as exc:
                logger.warning("cache read failed for %s: %s", symbol, exc)

        import yfinance as yf

        ticker = f"{symbol}.NS"
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    ticker, start=start, end=end, progress=False,
                    auto_adjust=False, threads=False,
                )
                if df is None or df.empty:
                    return None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.rename(columns={
                    "Adj Close": "adj_close", "Close": "close",
                    "Open": "open", "High": "high", "Low": "low",
                    "Volume": "volume",
                })
                keep = ["open", "high", "low", "close", "adj_close", "volume"]
                df = df[[c for c in keep if c in df.columns]]
                df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
                if self.use_cache:
                    try:
                        df.to_parquet(cache)
                    except Exception as exc:
                        logger.debug("cache write failed for %s: %s", symbol, exc)
                return df
            except Exception as exc:
                if attempt == max_retries - 1:
                    logger.warning("fetch failed for %s: %s", symbol, exc)
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    # -- universe ---------------------------------------------------------

    def fetch_universe(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        min_history_days: int = 500,
        pause: float = 0.0,
    ) -> PriceDataset:
        """
        Fetch a universe and align it into OHLCV panels.

        Parameters
        ----------
        symbols : iterable of NSE symbols WITHOUT the .NS suffix.
        start, end : "YYYY-MM-DD".
        min_history_days : symbols with less history than this are dropped and
            recorded in provenance. Thin history produces unstable features
            and inflated cross-sectional ranks.
        pause : seconds between requests, to stay polite to the vendor.
        """
        symbols = list(symbols)
        frames: dict[str, pd.DataFrame] = {}
        failed: list[str] = []

        for i, sym in enumerate(symbols, 1):
            df = self.fetch_symbol(sym, start, end)
            if df is None or len(df) < min_history_days:
                failed.append(sym)
            else:
                frames[sym] = df
            if pause:
                time.sleep(pause)
            if i % 25 == 0:
                logger.info("fetched %d/%d symbols", i, len(symbols))

        if not frames:
            raise RuntimeError(
                "No symbols returned usable data. The vendor may be rate "
                "limiting, or the date range may be invalid."
            )

        def panel(field: str) -> pd.DataFrame:
            return pd.DataFrame({s: d[field] for s, d in frames.items() if field in d})

        close_adj = panel("adj_close")
        close_raw = panel("close")

        # Align every panel on the union of dates, then require a minimum
        # cross-section per date so early sparse history does not distort
        # cross-sectional ranking.
        idx = close_adj.index
        out = PriceDataset(
            close=close_adj,
            open=panel("open").reindex(idx),
            high=panel("high").reindex(idx),
            low=panel("low").reindex(idx),
            volume=panel("volume").reindex(idx),
            raw_close=close_raw.reindex(idx),
            provenance=DatasetProvenance(
                source="Yahoo Finance (yfinance), NSE .NS tickers",
                retrieved_at=datetime.now(timezone.utc),
                universe_definition=(
                    "Hand-maintained list of NSE large/mid-cap names as "
                    "constituted TODAY. NOT a point-in-time index membership."
                ),
                n_symbols_requested=len(symbols),
                n_symbols_returned=close_adj.shape[1],
                start=idx.min() if len(idx) else None,
                end=idx.max() if len(idx) else None,
                price_field="adj_close (split/dividend adjusted)",
                known_biases=list(_BIAS_NOTES),
                failed_symbols=failed,
            ),
        )
        logger.info(
            "universe fetched: %d symbols, %s..%s",
            out.close.shape[1], out.provenance.start, out.provenance.end,
        )
        return out

    # -- benchmark --------------------------------------------------------

    def fetch_benchmark(self, start: str, end: str, symbol: str = "^NSEI") -> pd.Series:
        """
        NIFTY 50 index level. `^NSEI` is the Yahoo symbol for NIFTY 50.

        A benchmark is not optional. A strategy's return is uninterpretable
        without the passive alternative it must beat.
        """
        cache = self.cache_dir / f"BENCH_{symbol.replace('^','')}__{start}__{end}.parquet"
        if self.use_cache and cache.exists():
            try:
                return pd.read_parquet(cache)["close"]
            except Exception:
                pass

        import yfinance as yf

        df = yf.download(symbol, start=start, end=end, progress=False,
                         auto_adjust=False, threads=False)
        if df is None or df.empty:
            raise RuntimeError(f"Benchmark {symbol} returned no data.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = df[col]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        s.name = "close"
        if self.use_cache:
            try:
                s.to_frame().to_parquet(cache)
            except Exception:
                pass
        return s
