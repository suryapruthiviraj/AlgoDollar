"""
Automated data-quality auditing.

WHY THIS RUNS BEFORE ANY MODEL
------------------------------
Most "alpha" discovered in a hurry is a data defect. An unadjusted split looks
like a -50% one-day return and any mean-reversion model will happily learn to
buy it. A stale price repeated across a holiday looks like zero volatility. A
symbol that only has history from 2019 onward will dominate cross-sectional
ranks in 2018 by being absent. None of these are modelling problems and none
of them are visible in a Sharpe ratio.

So the panel is audited first, the findings are attached to the research
result, and severe findings block the run rather than being noted in passing.

SEVERITY
--------
CRITICAL  Invalidates research. The run must not proceed.
WARNING   Materially affects interpretation. Must appear in the report.
INFO      Worth recording for provenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Finding:
    severity: Severity
    check: str
    message: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value:8s}] {self.check}: {self.message}"


@dataclass
class QualityReport:
    findings: list[Finding]
    n_symbols: int
    n_dates: int
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.CRITICAL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        """A panel with any CRITICAL finding must not be used for research."""
        return not self.critical

    def summary(self) -> str:
        head = (
            f"Data quality: {self.n_symbols} symbols x {self.n_dates} dates "
            f"({self.start.date()} .. {self.end.date()})\n"
            f"  CRITICAL={len(self.critical)}  "
            f"WARNING={len(self.warnings)}  "
            f"INFO={len(self.findings) - len(self.critical) - len(self.warnings)}"
        )
        body = "\n".join(f"  {f}" for f in self.findings)
        verdict = "USABLE" if self.passed else "NOT USABLE — critical findings present"
        return f"{head}\n{body}\n  VERDICT: {verdict}"

    def raise_if_critical(self) -> None:
        if not self.passed:
            msgs = "\n".join(f"  - {f}" for f in self.critical)
            raise ValueError(
                f"Data quality audit failed with {len(self.critical)} critical "
                f"finding(s):\n{msgs}"
            )


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

def audit_price_panel(
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    raw_close: Optional[pd.DataFrame] = None,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    *,
    min_cross_section: int = 10,
    extreme_return_threshold: float = 0.40,
    max_stale_run: int = 10,
) -> QualityReport:
    """
    Audit a daily price panel.

    Parameters
    ----------
    close : DataFrame (T x N), adjusted close, DatetimeIndex.
    volume, raw_close, high, low : optional aligned panels.
    min_cross_section : dates with fewer live names than this are flagged;
        cross-sectional ranking over a handful of names is noise.
    extreme_return_threshold : absolute daily return above which a move is
        treated as a possible unadjusted corporate action.
    max_stale_run : consecutive identical closes above which a series is
        treated as stale rather than merely quiet.
    """
    findings: list[Finding] = []
    idx = pd.DatetimeIndex(close.index)

    # -- index integrity ---------------------------------------------------

    if not idx.is_monotonic_increasing:
        findings.append(Finding(
            Severity.CRITICAL, "index_monotonic",
            "Dates are not sorted ascending. Every rolling feature and every "
            "purged split would be computed over the wrong window.",
        ))

    dupes = idx[idx.duplicated()]
    if len(dupes) > 0:
        findings.append(Finding(
            Severity.CRITICAL, "duplicate_dates",
            f"{len(dupes)} duplicated dates. Duplicated bars double-count "
            f"returns and corrupt compounding.",
            {"examples": [str(d.date()) for d in dupes[:5]]},
        ))

    if getattr(idx, "tz", None) is not None:
        findings.append(Finding(
            Severity.WARNING, "timezone",
            f"Index is timezone-aware ({idx.tz}). Daily NSE bars should be "
            f"naive dates; a tz-aware index risks off-by-one-day alignment "
            f"against other sources.",
        ))

    # -- calendar ----------------------------------------------------------

    if len(idx) > 2:
        bdays = pd.bdate_range(idx.min(), idx.max())
        missing = len(bdays) - len(idx)
        pct = missing / max(len(bdays), 1)
        # Indian markets close roughly 12-18 days a year for holidays, so a
        # gap of a few percent is expected, not a defect.
        sev = Severity.INFO if pct < 0.10 else Severity.WARNING
        findings.append(Finding(
            sev, "trading_calendar",
            f"{missing} weekdays absent from the panel ({pct:.1%} of business "
            f"days). Expected range for NSE holidays is roughly 4-7%.",
            {"missing_days": int(missing), "pct": round(float(pct), 4)},
        ))

        # A gap far longer than a holiday break suggests missing sessions.
        gaps = pd.Series(idx).diff().dt.days.dropna()
        long_gaps = gaps[gaps > 10]
        if len(long_gaps) > 0:
            findings.append(Finding(
                Severity.WARNING, "session_gaps",
                f"{len(long_gaps)} gaps longer than 10 calendar days. These "
                f"are likely missing sessions rather than holidays.",
                {"max_gap_days": int(long_gaps.max())},
            ))

    # -- price sanity ------------------------------------------------------

    n_nonpositive = int((close <= 0).sum().sum())
    if n_nonpositive > 0:
        findings.append(Finding(
            Severity.CRITICAL, "nonpositive_prices",
            f"{n_nonpositive} non-positive prices. Log returns are undefined "
            f"and will produce -inf that survives dropna().",
        ))

    if not np.isfinite(close.to_numpy(dtype=float, na_value=np.nan)[
        ~np.isnan(close.to_numpy(dtype=float, na_value=np.nan))
    ]).all():
        findings.append(Finding(
            Severity.CRITICAL, "non_finite_prices",
            "Panel contains inf/-inf values.",
        ))

    # -- OHLC coherence ----------------------------------------------------

    if high is not None and low is not None:
        bad = (high < low).sum().sum()
        if bad > 0:
            findings.append(Finding(
                Severity.CRITICAL, "ohlc_coherence",
                f"{int(bad)} bars where high < low.",
            ))
        outside = ((close > high * 1.001) | (close < low * 0.999)).sum().sum()
        if outside > 0:
            findings.append(Finding(
                Severity.WARNING, "close_outside_range",
                f"{int(outside)} bars where close lies outside [low, high]. "
                f"Expected when comparing ADJUSTED close against UNADJUSTED "
                f"high/low, but confirm the adjustment convention.",
            ))

    # -- corporate actions -------------------------------------------------

    rets = np.log(close / close.shift(1))
    extreme = (rets.abs() > extreme_return_threshold)
    n_extreme = int(extreme.sum().sum())
    if n_extreme > 0:
        worst = rets.abs().max().max()
        by_sym = extreme.sum()
        top = by_sym[by_sym > 0].sort_values(ascending=False).head(5)
        findings.append(Finding(
            Severity.WARNING, "extreme_returns",
            f"{n_extreme} daily moves above {extreme_return_threshold:.0%} "
            f"(largest {worst:.1%}). Some are genuine; others indicate an "
            f"unadjusted split or bonus. A mean-reversion model will treat an "
            f"unadjusted split as a huge opportunity.",
            {"worst_abs_log_return": round(float(worst), 4),
             "top_symbols": {k: int(v) for k, v in top.items()}},
        ))

    if raw_close is not None:
        aligned = raw_close.reindex_like(close)
        differs = (aligned - close).abs().gt(1e-6).any().sum()
        findings.append(Finding(
            Severity.INFO, "adjustment_applied",
            f"{int(differs)}/{close.shape[1]} symbols show adjusted != raw "
            f"close, i.e. corporate actions were applied.",
        ))

    # -- staleness ---------------------------------------------------------

    stale_syms = {}
    for sym in close.columns:
        s = close[sym].dropna()
        if len(s) < 2:
            continue
        same = (s.diff() == 0)
        if not same.any():
            continue
        # longest consecutive run of identical closes
        grp = (~same).cumsum()
        run = same.groupby(grp).sum().max()
        if run >= max_stale_run:
            stale_syms[sym] = int(run)
    if stale_syms:
        findings.append(Finding(
            Severity.WARNING, "stale_prices",
            f"{len(stale_syms)} symbols have a run of >= {max_stale_run} "
            f"identical closes. Realized volatility is understated for these "
            f"names and any vol-scaled position will be oversized.",
            {"symbols": dict(list(stale_syms.items())[:8])},
        ))

    # -- cross-sectional coverage -----------------------------------------

    live_per_date = close.notna().sum(axis=1)
    thin = live_per_date[live_per_date < min_cross_section]
    if len(thin) > 0:
        findings.append(Finding(
            Severity.WARNING, "thin_cross_section",
            f"{len(thin)} dates have fewer than {min_cross_section} live "
            f"symbols. Cross-sectional ranks on those dates are unreliable "
            f"and should be excluded from IC computation.",
            {"first": str(thin.index.min().date()),
             "last": str(thin.index.max().date())},
        ))

    # Symbols whose history starts late enter the panel as NaN. That is
    # correct, but worth surfacing: a backtest starting in 2006 with a name
    # listed in 2019 is not testing what it appears to.
    first_valid = close.apply(lambda c: c.first_valid_index())
    late = first_valid[first_valid > idx.min() + pd.Timedelta(days=365)]
    if len(late) > 0:
        findings.append(Finding(
            Severity.INFO, "staggered_listings",
            f"{len(late)}/{close.shape[1]} symbols have history starting more "
            f"than a year after the panel start. The effective universe grows "
            f"over time.",
            {"example": {str(k): str(v.date()) for k, v in list(late.items())[:5]}},
        ))

    # -- liquidity ---------------------------------------------------------

    if volume is not None:
        zero_vol = (volume.fillna(0) <= 0).sum().sum()
        total = volume.shape[0] * volume.shape[1]
        if zero_vol > 0:
            findings.append(Finding(
                Severity.WARNING, "zero_volume_bars",
                f"{int(zero_vol)} zero-volume bars ({zero_vol/total:.2%}). "
                f"A fill cannot be assumed on a bar with no trading.",
            ))
        turnover = (volume * close).median()
        illiquid = turnover[turnover < 1e7]  # < ~Rs 1 crore median daily
        if len(illiquid) > 0:
            findings.append(Finding(
                Severity.WARNING, "low_liquidity_symbols",
                f"{len(illiquid)} symbols have median daily turnover below "
                f"Rs 1 crore. Modelled slippage will understate real cost.",
                {"symbols": list(illiquid.index[:8])},
            ))

    return QualityReport(
        findings=findings,
        n_symbols=close.shape[1],
        n_dates=close.shape[0],
        start=idx.min(),
        end=idx.max(),
    )


# ---------------------------------------------------------------------------
# Corporate-action artifact masking
# ---------------------------------------------------------------------------

def mask_corporate_action_artifacts(
    close: pd.DataFrame,
    raw_close: Optional[pd.DataFrame] = None,
    threshold: float = 0.40,
    ratio_tolerance: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Neutralize unadjusted splits, bonuses and demergers.

    HOW THESE ARE IDENTIFIED
    ------------------------
    When a vendor fails to adjust for a corporate action, the adjusted and
    unadjusted series move by the SAME ratio on that date — the adjustment
    factor was simply never applied to either. A genuine 40% price move, by
    contrast, appears in both series but the adjusted and raw ratios differ
    wherever any dividend adjustment is active.

    So a bar is treated as an artifact when the absolute log return exceeds
    `threshold` AND the adjusted and raw ratios agree to within
    `ratio_tolerance`. When `raw_close` is unavailable the ratio test cannot
    run and the threshold alone is used, which is more aggressive.

    WHY MASK RATHER THAN DROP THE SYMBOL
    ------------------------------------
    Dropping a symbol for one bad bar discards nineteen years of valid history
    and biases the universe toward companies that never restructured. Masking
    removes only the contaminated observation.

    WHY THIS MATTERS
    ----------------
    Left in place, an unadjusted 1:5 split reads as a -80% single-day return.
    Any mean-reversion model will learn to buy it and will book an enormous
    imaginary profit on the "recovery" that, in reality, never happened —
    the shareholder simply held five times as many shares all along.

    Returns
    -------
    (masked_close, mask, events)
        masked_close : close with artifact bars set to NaN
        mask         : boolean DataFrame, True where an artifact was masked
        events       : list of dicts describing each masked event
    """
    rets = np.log(close / close.shift(1))
    suspicious = rets.abs() > threshold

    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    events: list[dict] = []

    for sym in close.columns:
        hits = suspicious[sym]
        for date in hits[hits].index:
            adj_ratio = close[sym].loc[date] / close[sym].shift(1).loc[date]
            is_artifact = True
            raw_ratio = None

            if raw_close is not None and sym in raw_close.columns:
                prev = raw_close[sym].shift(1).loc[date]
                curr = raw_close[sym].loc[date]
                if pd.notna(prev) and pd.notna(curr) and prev > 0:
                    raw_ratio = curr / prev
                    # Ratios agreeing means no adjustment was applied at all.
                    is_artifact = abs(raw_ratio - adj_ratio) < ratio_tolerance

            if is_artifact:
                mask.loc[date, sym] = True
                events.append({
                    "symbol": sym,
                    "date": str(pd.Timestamp(date).date()),
                    "log_return": round(float(rets[sym].loc[date]), 4),
                    "adj_ratio": round(float(adj_ratio), 4),
                    "raw_ratio": round(float(raw_ratio), 4) if raw_ratio else None,
                })

    masked = close.mask(mask)
    if events:
        logger.warning(
            "Masked %d corporate-action artifacts across %d symbols.",
            len(events), len({e["symbol"] for e in events}),
        )
    return masked, mask, events


# ---------------------------------------------------------------------------
# Survivorship probe
# ---------------------------------------------------------------------------

# NSE names that were actively listed and then delisted or suspended. If a
# data source returns nothing for these over windows when they traded, the
# source is survivorship-filtered.
KNOWN_DELISTED_PROBES: tuple[tuple[str, str, str], ...] = (
    ("SATYAMCOMP", "2006-01-01", "2008-12-01"),   # 2009 accounting fraud
    ("DHFL",       "2016-01-01", "2019-01-01"),   # 2019 collapse
    ("VIDEOIND",   "2010-01-01", "2016-01-01"),   # insolvency
)


def probe_survivorship(provider, control: str = "RELIANCE") -> Finding:
    """
    Empirically test whether a data source drops delisted securities.

    This does not estimate the SIZE of the bias — that would require the
    missing data. It establishes only whether the bias is present, which is
    enough to determine how a result may be interpreted.
    """
    missing, present = [], []
    for sym, start, end in KNOWN_DELISTED_PROBES:
        df = provider.fetch_symbol(sym, start, end)
        (present if (df is not None and len(df) > 0) else missing).append(sym)

    ctrl = provider.fetch_symbol(control, KNOWN_DELISTED_PROBES[0][1],
                                 KNOWN_DELISTED_PROBES[0][2])
    ctrl_ok = ctrl is not None and len(ctrl) > 0

    if not ctrl_ok:
        return Finding(
            Severity.CRITICAL, "survivorship_probe",
            f"Control symbol {control} also returned no data — the probe is "
            f"inconclusive because the source appears unreachable.",
        )

    if missing:
        return Finding(
            Severity.CRITICAL, "survivorship_probe",
            f"Source is SURVIVORSHIP-FILTERED: {len(missing)}/"
            f"{len(KNOWN_DELISTED_PROBES)} known-delisted names returned no data "
            f"over periods when they were listed ({', '.join(missing)}), "
            f"while the control {control} returned full history. Backtests on "
            f"this source measure a filtered sample. The bias direction on "
            f"EXCESS return is INDETERMINATE and signal-dependent, so results "
            f"establish true historical performance in NEITHER direction.",
            {"missing": missing, "present": present},
        )

    return Finding(
        Severity.INFO, "survivorship_probe",
        f"All {len(KNOWN_DELISTED_PROBES)} delisted probes returned data. "
        f"The source appears to retain delisted securities.",
    )
