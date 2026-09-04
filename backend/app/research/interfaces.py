"""
The data contracts the research engine is written against.

The engine must not know whether prices live in Parquet, CSV, Postgres or a
vendor API. It asks these interfaces; something else decides where the bytes
come from.

THE POINT OF THE `DataUnavailable` EXCEPTION
--------------------------------------------
Several of these datasets DO NOT EXIST for this project. Point-in-time index
membership, corporate action ex-dates and publication-dated fundamentals are
all unobtainable from the free vendor this system uses.

The temptation is to return something plausible — today's index members as if
they were 2015's, or a fundamentals row stamped with the fiscal period end
instead of its publication date. Both silently inject look-ahead or
survivorship bias into every result computed downstream, and neither leaves a
trace in the output.

So the contract is: a provider that cannot answer RAISES
:class:`DataUnavailable`. The research runner catches it and records a stated
LIMITATION on the result. A study that needed the dataset is marked as such
rather than being quietly run on a substitute.

POINT-IN-TIME vs RECONSTRUCTED
------------------------------
Every provider declares :attr:`DataProvider.point_in_time`. False means the
data reflects TODAY'S state of the world, not the state as of the date being
studied. That flag propagates into the manifest and the validation report, so
no result can be read without also seeing how the data was built.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd


class DataUnavailable(RuntimeError):
    """
    The requested dataset does not exist, or does not exist point-in-time.

    Raised — never substituted with an approximation — because a study run on
    a substitute reads exactly like a study run on the real thing.
    """

    def __init__(self, dataset: str, reason: str) -> None:
        super().__init__(f"{dataset}: {reason}")
        self.dataset = dataset
        self.reason = reason


class Adjustment(str, Enum):
    """
    Which price series a caller wants.

    RAW is the traded price and is what an execution simulation must fill at.
    ADJUSTED is back-adjusted for splits and dividends and is the ONLY series
    from which returns may be computed — a 1:5 split in a raw series looks like
    an 80% single-day loss.
    """

    RAW = "raw"
    ADJUSTED = "adjusted"


@dataclass(frozen=True)
class Coverage:
    """What a provider actually holds, so a caller can check before relying."""

    symbols: int
    rows: int
    first_date: Optional[date]
    last_date: Optional[date]
    point_in_time: bool
    notes: tuple[str, ...] = ()


@runtime_checkable
class DataProvider(Protocol):
    """Common surface: every provider says what it is and what it holds."""

    @property
    def name(self) -> str: ...

    @property
    def point_in_time(self) -> bool:
        """
        False means this reflects TODAY'S world, not the studied date's.

        A False here is not a defect to hide; it is a property that must travel
        with every number computed from the provider.
        """
        ...

    def coverage(self) -> Coverage: ...


@runtime_checkable
class DailyBarProvider(DataProvider, Protocol):
    """Daily OHLCV."""

    def symbols(self) -> list[str]: ...

    def bars(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        OHLCV for one symbol, DatetimeIndex ascending.

        Columns: open, high, low, close, volume, and adj_close when the vendor
        supplies it. Missing sessions are ABSENT rows, never filled — a gap is
        evidence of a suspension or a listing date and must stay visible.
        """
        ...

    def panel(
        self,
        symbols: Sequence[str],
        field: str = "close",
        start: Optional[date] = None,
        end: Optional[date] = None,
        adjustment: Adjustment = Adjustment.ADJUSTED,
    ) -> pd.DataFrame:
        """
        Wide frame: dates x symbols for one field.

        Aligned on the union of trading dates. A symbol that did not trade on a
        date is NaN, which is the honest representation — it is not zero and it
        is not the previous price.
        """
        ...


@runtime_checkable
class IntradayBarProvider(DataProvider, Protocol):
    """Intraday OHLCV. No implementation exists for this project."""

    def bars(
        self, symbol: str, interval: str, start: date, end: date
    ) -> pd.DataFrame: ...


@runtime_checkable
class BenchmarkProvider(DataProvider, Protocol):
    """The index a strategy is measured against."""

    def series(
        self, start: Optional[date] = None, end: Optional[date] = None
    ) -> pd.Series:
        """Benchmark close, DatetimeIndex ascending."""
        ...

    def returns(
        self, start: Optional[date] = None, end: Optional[date] = None
    ) -> pd.Series: ...


@runtime_checkable
class UniverseProvider(DataProvider, Protocol):
    """
    Index membership.

    ``point_in_time`` is the whole question here. A provider returning today's
    membership for a 2015 date has silently removed every company that failed,
    which inflates any backtest run on it.
    """

    def members(self, as_of: date) -> list[str]: ...


@runtime_checkable
class CorporateActionProvider(DataProvider, Protocol):
    """Splits, bonuses and dividends with their ex-dates."""

    def splits(self, symbol: str) -> pd.DataFrame: ...

    def dividends(self, symbol: str) -> pd.DataFrame: ...


@runtime_checkable
class FundamentalsProvider(DataProvider, Protocol):
    """
    Fundamentals AS PUBLISHED.

    The publication date is the entire point. A statement for the quarter
    ending 31 March is not knowable on 31 March — it is knowable when it is
    filed, weeks later. Indexing fundamentals by period end rather than
    publication date is one of the most common and most damaging sources of
    look-ahead bias in equity research.
    """

    def as_of(self, symbol: str, when: date) -> dict[str, Any]: ...


@runtime_checkable
class SectorProvider(DataProvider, Protocol):
    """Sector classification, used for neutralisation and exposure limits."""

    def sector(self, symbol: str, as_of: Optional[date] = None) -> str: ...

    def mapping(self, as_of: Optional[date] = None) -> dict[str, str]: ...


# --------------------------------------------------------------------------- #
#  Explicitly-absent providers                                                  #
# --------------------------------------------------------------------------- #

class UnavailableProvider:
    """
    Stands in for a dataset this project does not have.

    It is wired in deliberately rather than left as ``None``. ``None`` invites
    ``if provider is not None:`` and a quiet skip; this raises with a specific
    reason that lands in the research report, so an absent dataset is reported
    as a LIMITATION instead of vanishing.
    """

    def __init__(self, dataset: str, reason: str) -> None:
        self._dataset = dataset
        self._reason = reason

    @property
    def name(self) -> str:
        return f"unavailable:{self._dataset}"

    @property
    def point_in_time(self) -> bool:
        return False

    @property
    def reason(self) -> str:
        return self._reason

    def coverage(self) -> Coverage:
        return Coverage(
            symbols=0, rows=0, first_date=None, last_date=None,
            point_in_time=False, notes=(self._reason,),
        )

    def _raise(self) -> Any:
        raise DataUnavailable(self._dataset, self._reason)

    # Every access path fails the same way.
    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)

        def _fail(*_a: Any, **_k: Any) -> Any:
            raise DataUnavailable(self._dataset, self._reason)

        return _fail


#: The datasets this project does NOT have, and why. Imported by the research
#: runner so every report states them rather than leaving them to be inferred.
KNOWN_UNAVAILABLE: dict[str, str] = {
    "intraday_bars": (
        "No intraday feed is available. The vendor serves daily bars only, so "
        "no intraday strategy can be researched or validated here."
    ),
    "point_in_time_universe": (
        "No historical index membership source is available. Only a PRESENT-DAY "
        "snapshot exists, so any universe reconstructed for a past date excludes "
        "every company that was delisted, merged or removed — survivorship bias "
        "of unknown magnitude and direction."
    ),
    "corporate_actions": (
        "No corporate-action dataset with ex-dates, split ratios or dividend "
        "amounts is available. Splits and dividends are only implicit in the "
        "vendor's adj_close, which is sufficient for computing returns and "
        "insufficient for anything that needs the event itself."
    ),
    "fundamentals_point_in_time": (
        "No fundamentals source with PUBLICATION dates is available. Fundamentals "
        "indexed by fiscal period end embed look-ahead bias, so value and quality "
        "factors cannot be validly researched here."
    ),
    "delistings": (
        "No delisting record is available. Delisted names are simply absent, "
        "which is the survivorship problem above rather than a separate gap."
    ),
    "symbol_changes": (
        "No ticker-change mapping is available. A renamed symbol appears as a "
        "new name with truncated history and its prior history is unreachable."
    ),
}


def unavailable(dataset: str) -> UnavailableProvider:
    """Build the stand-in for a known-absent dataset."""
    reason = KNOWN_UNAVAILABLE.get(
        dataset, "no provider is configured for this dataset"
    )
    return UnavailableProvider(dataset, reason)
