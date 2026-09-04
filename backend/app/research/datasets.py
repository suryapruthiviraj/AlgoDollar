"""
Parquet-backed implementations of the research data interfaces.

The research engine never imports this module by name — it takes providers as
arguments. Swapping the store for Postgres or a vendor API means writing
another implementation of :mod:`app.research.interfaces`, not editing the
engine.

WHAT IS AND IS NOT PROVIDED
---------------------------
Daily bars, the benchmark and sector classification are real. Intraday bars,
point-in-time universe membership, corporate actions and point-in-time
fundamentals are NOT, and :func:`build_dataset_bundle` wires explicit
"unavailable" stand-ins for them so a study that needs one fails loudly with a
stated reason rather than running on a substitute.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from app.research.interfaces import (
    Adjustment,
    Coverage,
    DataUnavailable,
    UnavailableProvider,
    unavailable,
)

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "research_data"


class ParquetDailyBars:
    """
    Daily OHLCV from one Parquet file per symbol.

    ``point_in_time`` is True for the BARS themselves: a bar recorded for
    2015-03-04 is what traded that day, and no later information is folded into
    it. That is a narrower claim than it sounds, and it says nothing about the
    UNIVERSE the bars were selected for — which is not point-in-time at all.
    """

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self._root = Path(root)
        self._prices = self._root / "prices"
        if not self._prices.exists():
            raise DataUnavailable(
                "daily_bars",
                f"no price store at {self._prices}. Run scripts/acquire_data.py "
                f"first; this module never fabricates bars.",
            )
        self._cache: dict[str, pd.DataFrame] = {}
        self._stale_counts: dict[str, int] = {}
        self._symbols = sorted(p.stem for p in self._prices.glob("*.parquet"))

    @property
    def name(self) -> str:
        return f"parquet-daily[{self._prices}]"

    @property
    def point_in_time(self) -> bool:
        return True

    def symbols(self) -> list[str]:
        return list(self._symbols)

    @staticmethod
    def stale_bar_mask(df: pd.DataFrame) -> pd.Series:
        """
        Rows that are NOT a real trading session.

        The vendor emits a bar for some non-trading days with volume 0 and
        open == high == low == close, carrying the previous close forward.
        Every symbol in this dataset has them (7 for RELIANCE, up to 15 for
        others).

        They matter because they are not neutral: a carried-forward price
        produces a return of EXACTLY ZERO, which is not a real observation. A
        handful of fake zeros per symbol compresses measured volatility,
        inflates any Sharpe computed from the series, and adds spurious
        zero-return days to a hit-rate calculation.

        They are identified, never repaired. The caller decides whether to drop
        them, and the count is reported so the decision is visible.
        """
        if "volume" not in df.columns:
            return pd.Series(False, index=df.index)
        flat = pd.Series(True, index=df.index)
        for a, b in (("open", "high"), ("high", "low"), ("low", "close")):
            if a in df.columns and b in df.columns:
                flat &= df[a] == df[b]
        return (df["volume"] == 0) & flat

    def bars(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        if symbol not in self._cache:
            path = self._prices / f"{symbol}.parquet"
            if not path.exists():
                raise DataUnavailable("daily_bars", f"no bars for {symbol}")
            df = pd.read_parquet(path).sort_index()
            df.index = pd.DatetimeIndex(df.index)
            self._cache[symbol] = df
        df = self._cache[symbol]
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def panel(
        self,
        symbols: Sequence[str],
        field: str = "close",
        start: Optional[date] = None,
        end: Optional[date] = None,
        adjustment: Adjustment = Adjustment.ADJUSTED,
        drop_stale: bool = True,
    ) -> pd.DataFrame:
        """
        Wide dates x symbols frame.

        For ``field="close"`` under ADJUSTED, ``adj_close`` is used. This is not
        a convenience: computing a return from the unadjusted close books every
        split as a catastrophic single-day loss, and every bonus issue as the
        same. Raw close is available and is what an execution simulation must
        fill at, but it must never be differenced.
        """
        col = field
        if adjustment is Adjustment.ADJUSTED and field == "close":
            col = "adj_close"

        series: dict[str, pd.Series] = {}
        for sym in symbols:
            try:
                df = self.bars(sym, start, end)
            except DataUnavailable:
                continue
            if drop_stale:
                # NaN, not dropped: the DATE still exists for other symbols, and
                # NaN is what "this name did not trade" honestly looks like.
                # Removing the row would silently shorten the panel for everyone.
                stale = self.stale_bar_mask(df)
                if stale.any():
                    df = df.copy()
                    df.loc[stale, [c for c in df.columns if c != "volume"]] = float("nan")
                    self._stale_counts[sym] = int(stale.sum())
            if col in df.columns:
                series[sym] = df[col]
            elif field in df.columns:
                # Fall back to the requested field, but say so: silently using
                # an unadjusted series where an adjusted one was asked for is
                # exactly the substitution this codebase refuses elsewhere.
                logger.warning(
                    "%s has no %r column; using %r. Returns computed from this "
                    "series are NOT split/dividend adjusted.", sym, col, field,
                )
                series[sym] = df[field]
        if not series:
            return pd.DataFrame()
        # Union of dates, NaN where a symbol did not trade. NaN is the honest
        # value — not zero, and not the previous price carried forward.
        return pd.DataFrame(series).sort_index()

    @property
    def stale_bars_masked(self) -> dict[str, int]:
        """Stale bars NaN'd by the last panel() call, per symbol."""
        return dict(self._stale_counts)

    def coverage(self) -> Coverage:
        rows = 0
        first: Optional[pd.Timestamp] = None
        last: Optional[pd.Timestamp] = None
        for sym in self._symbols:
            df = self.bars(sym)
            rows += len(df)
            if len(df):
                first = df.index[0] if first is None else min(first, df.index[0])
                last = df.index[-1] if last is None else max(last, df.index[-1])
        return Coverage(
            symbols=len(self._symbols),
            rows=rows,
            first_date=first.date() if first is not None else None,
            last_date=last.date() if last is not None else None,
            point_in_time=True,
            notes=(
                "Bars are point-in-time; the SYMBOL SET they were selected for "
                "is not. See the universe provider.",
            ),
        )


class ParquetBenchmark:
    """The index series, from the same vendor and calendar as the bars."""

    def __init__(self, root: Path | str = DEFAULT_ROOT, symbol: str = "^NSEI") -> None:
        self._path = Path(root) / "benchmark.parquet"
        self._symbol = symbol
        if not self._path.exists():
            raise DataUnavailable(
                "benchmark", f"no benchmark at {self._path}. Without it no "
                f"relative performance can be computed at all."
            )
        self._df = pd.read_parquet(self._path).sort_index()
        self._df.index = pd.DatetimeIndex(self._df.index)

    @property
    def name(self) -> str:
        return f"parquet-benchmark[{self._symbol}]"

    @property
    def point_in_time(self) -> bool:
        return True

    def series(
        self, start: Optional[date] = None, end: Optional[date] = None
    ) -> pd.Series:
        col = "adj_close" if "adj_close" in self._df.columns else "close"
        s = self._df[col]
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
        return s

    def returns(
        self, start: Optional[date] = None, end: Optional[date] = None
    ) -> pd.Series:
        return self.series(start, end).pct_change().dropna()

    def coverage(self) -> Coverage:
        return Coverage(
            symbols=1, rows=len(self._df),
            first_date=self._df.index[0].date() if len(self._df) else None,
            last_date=self._df.index[-1].date() if len(self._df) else None,
            point_in_time=True,
        )


class SnapshotUniverse:
    """
    Present-day index membership, applied to every date.

    ``point_in_time`` is FALSE and :meth:`members` returns the same list
    whatever ``as_of`` it is given. That is stated rather than hidden because
    it is the single largest known bias in every result computed here: the
    membership list contains only companies that survived to today, so any
    backtest over it never holds a name that went to zero.

    The direction of that bias is NOT reliably "conservative". For a momentum
    strategy the removed names are disproportionately the ones that fell
    hardest, which the strategy would have been short of or out of anyway; for
    a value or reversal strategy they are disproportionately the cheap names
    that never recovered, which the strategy WOULD have bought. Assuming the
    bias works in your favour is itself an error.
    """

    def __init__(self, symbols: Sequence[str]) -> None:
        self._symbols = sorted(symbols)

    @property
    def name(self) -> str:
        return "snapshot-universe(NOT point-in-time)"

    @property
    def point_in_time(self) -> bool:
        return False

    def members(self, as_of: date) -> list[str]:
        return list(self._symbols)

    def coverage(self) -> Coverage:
        return Coverage(
            symbols=len(self._symbols), rows=len(self._symbols),
            first_date=None, last_date=None, point_in_time=False,
            notes=(
                "SURVIVORSHIP BIAS: present-day membership applied to all dates.",
            ),
        )


class StaticSectorMap:
    """
    Sector labels from the repository's static map.

    Not point-in-time: a company that changed sector classification carries its
    CURRENT label backwards. That matters for sector-neutral studies and for
    exposure limits, and is small compared with the universe bias but is still
    a reconstruction rather than a record.
    """

    def __init__(self, mapping: Optional[dict[str, str]] = None) -> None:
        if mapping is None:
            try:
                from app.data.universe import StockUniverse

                mapping = getattr(StockUniverse, "SECTOR_MAP", None) or {}
            except Exception:  # noqa: BLE001
                mapping = {}
        self._map = dict(mapping)

    @property
    def name(self) -> str:
        return "static-sector-map(NOT point-in-time)"

    @property
    def point_in_time(self) -> bool:
        return False

    def sector(self, symbol: str, as_of: Optional[date] = None) -> str:
        return self._map.get(symbol, "UNKNOWN")

    def mapping(self, as_of: Optional[date] = None) -> dict[str, str]:
        return dict(self._map)

    def coverage(self) -> Coverage:
        known = sum(1 for v in self._map.values() if v and v != "UNKNOWN")
        return Coverage(
            symbols=len(self._map), rows=known, first_date=None, last_date=None,
            point_in_time=False,
            notes=("Current classification applied to all dates.",),
        )


@dataclass
class DatasetBundle:
    """Everything a research run is given, including what it does NOT have."""

    daily: ParquetDailyBars
    benchmark: ParquetBenchmark
    universe: SnapshotUniverse
    sectors: StaticSectorMap
    intraday: UnavailableProvider
    corporate_actions: UnavailableProvider
    fundamentals: UnavailableProvider
    manifest: dict[str, Any]

    def limitations(self) -> list[str]:
        """
        Every known bias and gap, in one list.

        Produced from the providers themselves rather than written by hand, so
        it cannot drift out of date relative to what was actually loaded.
        """
        out: list[str] = []
        if not self.universe.point_in_time:
            out.append(
                "SURVIVORSHIP BIAS: the universe is a present-day snapshot "
                "applied to all dates. Delisted, merged and removed companies "
                "are absent. Magnitude unknown; direction NOT reliably "
                "conservative."
            )
        if not self.sectors.point_in_time:
            out.append(
                "Sector labels are current, not point-in-time; reclassified "
                "companies carry today's label backwards."
            )
        for prov in (self.intraday, self.corporate_actions, self.fundamentals):
            out.append(f"NOT AVAILABLE — {prov.reason}")
        failed = (self.manifest or {}).get("symbols_failed") or {}
        if failed:
            out.append(
                f"{len(failed)} requested symbols returned no data and are "
                f"absent from the study ({', '.join(sorted(failed)[:8])}"
                f"{'...' if len(failed) > 8 else ''}). Several are renamed or "
                f"merged tickers, which is direct evidence of the symbol-change "
                f"gap rather than a transient fetch error."
            )
        return out


def build_dataset_bundle(root: Path | str = DEFAULT_ROOT) -> DatasetBundle:
    """
    Load everything available and wire explicit stand-ins for everything absent.

    The absent datasets are wired in as raising providers rather than left as
    None: ``None`` invites a quiet ``if provider:`` skip, whereas these force
    the caller to record a limitation.
    """
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    daily = ParquetDailyBars(root)
    return DatasetBundle(
        daily=daily,
        benchmark=ParquetBenchmark(root),
        universe=SnapshotUniverse(daily.symbols()),
        sectors=StaticSectorMap(),
        intraday=unavailable("intraday_bars"),
        corporate_actions=unavailable("corporate_actions"),
        fundamentals=unavailable("fundamentals_point_in_time"),
        manifest=manifest,
    )
