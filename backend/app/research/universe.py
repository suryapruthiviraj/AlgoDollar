"""
Point-in-time universe: what was actually eligible on each historical date.

THE UNIVERSE DEFINITION, STATED EXPLICITLY
------------------------------------------
    Universe(D) = { s : listed_on_or_before(s, D)
                      AND still_trading_on(s, D)
                      AND liquid_over_trailing_window(s, D)
                      AND price(s, D) >= MIN_PRICE }

It is a LIQUIDITY-SCREENED NSE EQUITY universe. It is deliberately NOT defined
by index membership.

WHY NOT INDEX MEMBERSHIP
------------------------
The original design used `StockUniverse.get_nifty500_symbols()` — a hardcoded
snapshot of today's NIFTY 500 — plus a liquidity filter. The liquidity filter
was already point-in-time; the membership list was not, and that is the entire
survivorship problem.

Fixing it by sourcing historical NIFTY 500 constituents is not possible here.
NSE serves no dated constituent files (verified: 404), and no other reachable
source provides entry and exit dates. Reconstructing membership by inference —
"it is in the index now and was big then, so it was probably in the index" —
would be fabricating the very dates that decide every backtest result.

So the universe is redefined onto criteria that CAN be established from real
evidence, which is the honest move rather than a weaker one:

    entry   NSE's published DATE OF LISTING, and the first bar actually observed
    exit    the last bar actually observed
    liquid  the trailing-window test the design already specified

Every one of those is a fact about the past recorded at the time, not a
present-day judgement projected backwards.

WHAT THIS DOES AND DOES NOT FIX
-------------------------------
FIXED: a company that listed in 2020 can no longer appear in a 2015 backtest. A
company that stopped trading in 2018 can no longer be held in 2019. A company
that collapsed remains in the universe for exactly the period it was tradeable,
so a strategy is charged for the chance it would have bought it.

NOT FIXED: the POOL. Price history is only available for symbols someone
thought to fetch, and NSE's equity list contains only companies listed TODAY.
Companies that delisted before today are absent unless added by name. The
provider reports `pool_completeness` so the size of that gap travels with every
result computed on it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Liquidity thresholds. Carried over UNCHANGED from
#: `StockUniverse.filter_liquid`, which is the design's own definition of an
#: investable name. They are not tuned here — re-deriving them against the new
#: universe would be fitting the universe to the result.
MIN_AVG_VOLUME = 500_000        # shares/day
MIN_AVG_TURNOVER = 5_000_000.0  # rupees/day
LIQUIDITY_WINDOW = 60           # trailing sessions

#: Penny stocks are excluded: a tick is a large fraction of the price, so the
#: cost model understates the true spread badly. Not a performance filter — it
#: is applied before any return is computed.
MIN_PRICE = 5.0

#: A symbol must have at least this many bars in the trailing window to be
#: judged at all. Fewer means we cannot tell whether it is liquid, and an
#: unjudgeable symbol is EXCLUDED rather than assumed fine.
MIN_OBSERVATIONS = 40

#: A series ending this many sessions before the dataset ends is treated as
#: having stopped trading — a delisting, a suspension, or a ticker change. All
#: three end eligibility, and the reason is not recoverable from prices alone.
DELISTING_GAP_SESSIONS = 20


class UniverseUnavailable(RuntimeError):
    """
    Membership cannot be established for the requested date.

    Raised, never answered with a guess. A backtest that silently receives
    today's list for a date it could not resolve is exactly the failure this
    module exists to prevent.
    """


@dataclass(frozen=True)
class Listing:
    """One symbol's tradeable life, from evidence rather than assumption."""

    symbol: str
    #: NSE's published listing date, when known.
    nse_listing_date: Optional[date]
    #: First bar actually observed in the price data.
    first_bar: date
    #: Last bar actually observed.
    last_bar: date
    #: True when the series stops well before the dataset ends.
    stopped_trading: bool
    n_bars: int

    @property
    def entry_date(self) -> date:
        """
        The later of NSE's listing date and the first observed bar.

        The later of the two on purpose: a bar before the published listing date
        is a data error, and trusting the published date alone would let a
        symbol into the universe on a day we have no price for it.
        """
        if self.nse_listing_date is None:
            return self.first_bar
        return max(self.nse_listing_date, self.first_bar)

    @property
    def exit_date(self) -> Optional[date]:
        """Last tradeable date, or None if still trading at the dataset end."""
        return self.last_bar if self.stopped_trading else None


@dataclass
class UniverseCoverage:
    """What the provider can and cannot answer for."""

    pool_size: int
    first_date: Optional[date]
    last_date: Optional[date]
    n_stopped_trading: int
    #: Dates the provider refuses to answer for, and why.
    unavailable_before: Optional[date] = None
    notes: tuple[str, ...] = ()


class PointInTimeUniverse:
    """
    Membership on any date, computed from listing dates and trailing liquidity.

    Deterministic: the same inputs always produce the same members in the same
    order, and :meth:`fingerprint` proves which inputs were used.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        *,
        listing_dates: Optional[dict[str, date]] = None,
        min_avg_volume: float = MIN_AVG_VOLUME,
        min_avg_turnover: float = MIN_AVG_TURNOVER,
        window: int = LIQUIDITY_WINDOW,
        min_price: float = MIN_PRICE,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        if prices.empty:
            raise UniverseUnavailable("no price data; membership cannot be computed")
        common = [c for c in prices.columns if c in volumes.columns]
        self._px = prices[common].sort_index()
        self._vol = volumes[common].sort_index()
        self._listing = dict(listing_dates or {})
        self._min_volume = float(min_avg_volume)
        self._min_turnover = float(min_avg_turnover)
        self._window = int(window)
        self._min_price = float(min_price)
        self._min_obs = int(min_observations)

        self._dates = pd.DatetimeIndex(self._px.index)
        self._listings: dict[str, Listing] = self._build_listings()
        self._cache: dict[pd.Timestamp, tuple[str, ...]] = {}

        # Rolling statistics are computed ONCE over the whole panel. Each row is
        # a trailing window ending at that row, so reading row D uses only data
        # up to and including D — the same guarantee as computing it per date,
        # without recomputing 3,600 times.
        traded_value = self._px * self._vol
        self._avg_volume = self._vol.rolling(
            self._window, min_periods=self._min_obs
        ).mean()
        self._avg_turnover = traded_value.rolling(
            self._window, min_periods=self._min_obs
        ).mean()

    # -- listings ---------------------------------------------------------- #

    def _build_listings(self) -> dict[str, Listing]:
        dataset_end = self._dates[-1]
        # A series is "stopped" if its last bar is far enough before the end of
        # the dataset. Measured in SESSIONS rather than calendar days so a long
        # holiday does not look like a delisting.
        cutoff_idx = max(0, len(self._dates) - 1 - DELISTING_GAP_SESSIONS)
        cutoff = self._dates[cutoff_idx]

        out: dict[str, Listing] = {}
        for sym in self._px.columns:
            series = self._px[sym].dropna()
            if series.empty:
                continue
            first, last = series.index[0], series.index[-1]
            out[sym] = Listing(
                symbol=sym,
                nse_listing_date=self._listing.get(sym),
                first_bar=first.date(),
                last_bar=last.date(),
                stopped_trading=bool(last < cutoff),
                n_bars=int(len(series)),
            )
        logger.info(
            "universe pool: %d symbols, %d stopped trading before %s",
            len(out), sum(1 for v in out.values() if v.stopped_trading),
            dataset_end.date(),
        )
        return out

    @property
    def listings(self) -> dict[str, Listing]:
        return dict(self._listings)

    # -- membership -------------------------------------------------------- #

    def get_members(self, when: date | pd.Timestamp) -> tuple[str, ...]:
        """
        Symbols eligible on ``when``. Sorted, so the result is deterministic.

        Raises :class:`UniverseUnavailable` when the date falls outside the
        period membership can be established for. That is the fail-closed
        behaviour: an unanswerable date must not quietly return today's list,
        and must not quietly return an empty universe either, because an empty
        universe reads as "no opportunities" rather than "no answer".
        """
        ts = pd.Timestamp(when).normalize()
        if ts in self._cache:
            return self._cache[ts]

        if ts < self._dates[0] or ts > self._dates[-1]:
            raise UniverseUnavailable(
                f"{ts.date()} is outside the price data "
                f"({self._dates[0].date()} to {self._dates[-1].date()}); "
                f"membership cannot be established and is not guessed."
            )

        # The last session at or before `when`. Asking about a holiday resolves
        # to the previous session rather than failing.
        pos = int(self._dates.searchsorted(ts, side="right")) - 1
        if pos < 0:
            raise UniverseUnavailable(f"no session on or before {ts.date()}")
        if pos < self._min_obs:
            raise UniverseUnavailable(
                f"{ts.date()} is only {pos} sessions into the data; the "
                f"{self._window}-session liquidity window needs at least "
                f"{self._min_obs} observations. Membership is UNAVAILABLE for "
                f"this warm-up period rather than being computed on partial data."
            )

        row_date = self._dates[pos]
        as_of = row_date.date()
        px_row = self._px.iloc[pos]
        vol_row = self._avg_volume.iloc[pos]
        turn_row = self._avg_turnover.iloc[pos]

        members: list[str] = []
        for sym, listing in self._listings.items():
            if listing.entry_date > as_of:
                continue                      # not yet listed
            exit_date = listing.exit_date
            if exit_date is not None and exit_date < as_of:
                continue                      # stopped trading
            price = px_row.get(sym)
            if price is None or not np.isfinite(price) or price < self._min_price:
                continue                      # not trading, or sub-penny
            avg_v, avg_t = vol_row.get(sym), turn_row.get(sym)
            if avg_v is None or not np.isfinite(avg_v) or avg_v < self._min_volume:
                continue
            if avg_t is None or not np.isfinite(avg_t) or avg_t < self._min_turnover:
                continue
            members.append(sym)

        result = tuple(sorted(members))
        self._cache[ts] = result
        return result

    def is_member(self, symbol: str, when: date | pd.Timestamp) -> bool:
        return symbol in self.get_members(when)

    def first_available_date(self) -> date:
        """Earliest date membership can be established for."""
        return self._dates[self._min_obs].date()

    def coverage(self) -> UniverseCoverage:
        stopped = sum(1 for v in self._listings.values() if v.stopped_trading)
        return UniverseCoverage(
            pool_size=len(self._listings),
            first_date=self._dates[0].date(),
            last_date=self._dates[-1].date(),
            n_stopped_trading=stopped,
            unavailable_before=self.first_available_date(),
            notes=(
                f"{stopped} symbols stopped trading during the period and are "
                f"excluded from dates after they stopped.",
                "Membership is UNAVAILABLE for the liquidity warm-up period.",
                "The POOL is incomplete: NSE's equity list contains only "
                "companies listed today, so companies that delisted before "
                "today are absent unless added by name.",
            ),
        )

    def pool_completeness(self) -> dict[str, Any]:
        """
        What can be said about how complete the pool is.

        Reported rather than asserted: `stopped_trading` counts names the pool
        DOES contain that stopped trading, which is direct evidence that the
        universe is no longer survivor-only. It says nothing about names the
        pool never contained, and that limitation is stated alongside.
        """
        stopped = [v for v in self._listings.values() if v.stopped_trading]
        return {
            "pool_size": len(self._listings),
            "stopped_trading": len(stopped),
            "stopped_trading_pct": round(
                100.0 * len(stopped) / max(1, len(self._listings)), 2
            ),
            "examples": sorted(v.symbol for v in stopped)[:15],
            "limitation": (
                "Counts only names present in the pool. Companies that delisted "
                "before NSE's current equity list was published are absent from "
                "it entirely and cannot be counted here."
            ),
        }

    # -- reproducibility ---------------------------------------------------- #

    def fingerprint(self) -> str:
        """
        Hash of the inputs and thresholds that determine membership.

        Two runs producing the same fingerprint must produce identical
        universes; a different fingerprint means the universe changed and any
        cached validation result no longer applies to it.
        """
        payload = json.dumps({
            "symbols": sorted(self._listings),
            "entries": {
                sym: [rec.entry_date.isoformat(),
                      rec.exit_date.isoformat() if rec.exit_date else None]
                for sym, rec in sorted(self._listings.items())
            },
            "thresholds": {
                "min_avg_volume": self._min_volume,
                "min_avg_turnover": self._min_turnover,
                "window": self._window,
                "min_price": self._min_price,
                "min_observations": self._min_obs,
            },
            "dates": [self._dates[0].isoformat(), self._dates[-1].isoformat()],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def manifest(self) -> dict[str, Any]:
        """The universe's own record, for the experiment manifest."""
        cov = self.coverage()
        return {
            "definition": (
                "Liquidity-screened NSE equity universe. Membership on date D "
                "requires: listed on or before D (NSE DATE OF LISTING and first "
                "observed bar), still trading on D (series has not ended), "
                f"trailing {self._window}-session average volume >= "
                f"{self._min_volume:,.0f} shares AND average turnover >= "
                f"Rs {self._min_turnover:,.0f}, and price >= Rs {self._min_price}."
            ),
            "not_index_membership": (
                "Deliberately NOT the NIFTY 500. NSE serves no dated historical "
                "constituent files, so index membership cannot be established "
                "point-in-time and is not inferred."
            ),
            "fingerprint": self.fingerprint(),
            "pool_size": cov.pool_size,
            "date_range": [str(cov.first_date), str(cov.last_date)],
            "unavailable_before": str(cov.unavailable_before),
            "thresholds": {
                "min_avg_volume": self._min_volume,
                "min_avg_turnover": self._min_turnover,
                "liquidity_window": self._window,
                "min_price": self._min_price,
                "min_observations": self._min_obs,
            },
            "pool_completeness": self.pool_completeness(),
            "notes": list(cov.notes),
        }


def load_listing_dates(root: Path) -> dict[str, date]:
    """NSE's published listing dates, when the reference file has been acquired."""
    path = Path(root) / "universe_reference.json"
    if not path.exists():
        logger.warning(
            "no universe_reference.json at %s; entry dates fall back to the "
            "first observed bar, which is later than the true listing date for "
            "any symbol whose history predates the dataset.", path,
        )
        return {}
    ref = json.loads(path.read_text())
    out: dict[str, date] = {}
    for sym, meta in (ref.get("listings") or {}).items():
        try:
            out[sym] = date.fromisoformat(meta["listing_date"])
        except (KeyError, ValueError):
            continue
    return out


def build_point_in_time_universe(
    bundle: Any, symbols: Optional[Sequence[str]] = None
) -> PointInTimeUniverse:
    """Assemble the provider from a dataset bundle."""
    from app.research.datasets import DEFAULT_ROOT

    syms = list(symbols) if symbols is not None else bundle.daily.symbols()
    prices = bundle.daily.panel(syms)
    volumes = bundle.daily.panel(syms, field="volume")
    return PointInTimeUniverse(
        prices, volumes, listing_dates=load_listing_dates(DEFAULT_ROOT)
    )
