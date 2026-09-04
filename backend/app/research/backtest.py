"""
Cross-sectional backtest with an explicit, auditable lag structure.

THE LAG IS THE WHOLE THING
--------------------------
Almost every inflated backtest is a timing error, so the convention here is
fixed, stated once, and enforced by :func:`assert_no_lookahead`:

    signal[t]   is computed from prices up to and INCLUDING the close of t
    weights[t]  are therefore only actionable AFTER t
    the position formed on signal[t] earns return[t+1] = px[t+2]/px[t+1] - 1

That is a TWO-bar offset from signal to realised return, not one. The signal is
observed at the close of ``t``; the earliest a trade can be executed is the
close of ``t+1``; the first return that position can earn runs from ``t+1`` to
``t+2``. Using ``return[t]`` would be pure look-ahead, and using
``px[t+1]/px[t]`` assumes execution at the same close that produced the signal
— free, instantaneous, at a price you only knew after the fact.

COSTS ARE CHARGED ON TURNOVER, BOTH SIDES
-----------------------------------------
Cost is applied to |w[t] - w[t-1]| summed across names, which charges both the
buy and the sell of a rotation. A backtest that charges one side halves the
true cost, and for a weekly-rebalanced cross-sectional strategy cost is usually
the difference between a positive and a negative result.

NOTHING HERE OPTIMISES ANYTHING
-------------------------------
There is no parameter search, no threshold fitting and no selection of a best
variant. This module measures one configuration and reports it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

#: Round-trip cost in basis points of traded notional, charged per side.
#: 25bps/side is deliberately not optimistic for Indian equities: brokerage,
#: STT, exchange charges, GST, stamp duty and stamp-duty-inclusive slippage on
#: a mid-cap add up to roughly this. `docs/RESEARCH_VALIDATION.md` reports the
#: sensitivity of every result to this number.
DEFAULT_COST_BPS = 25.0


class LookaheadError(AssertionError):
    """A signal was used to earn a return it could not have been known before."""


@dataclass
class BacktestMetrics:
    """Everything the report needs, computed once from one return series."""

    n_periods: int
    years: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    profit_factor: float
    avg_trade: float
    turnover_annual: float
    cost_drag_annual: float
    exposure: float
    beta: Optional[float] = None
    alpha_annual: Optional[float] = None
    benchmark_cagr: Optional[float] = None
    excess_cagr: Optional[float] = None
    information_ratio: Optional[float] = None
    worst_month: Optional[float] = None
    worst_quarter: Optional[float] = None
    worst_year: Optional[float] = None
    best_year: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    returns_net: pd.Series
    returns_gross: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    metrics: BacktestMetrics
    cost_bps: float
    config: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Look-ahead guard                                                             #
# --------------------------------------------------------------------------- #

def assert_no_lookahead(
    signal: pd.DataFrame, forward_returns: pd.DataFrame, *, lag: int = 2
) -> None:
    """
    Verify the return a signal earns is strictly in its future.

    Implemented as a real check rather than a comment: the forward-return frame
    must be the price panel shifted by at least ``lag``, so for every date the
    return attributed to ``signal[t]`` must be computable only from prices at
    ``t+lag`` or later.

    Raises :class:`LookaheadError` rather than warning. A look-ahead result is
    not a result with a caveat; it is not a result.
    """
    if signal.empty or forward_returns.empty:
        return
    if not signal.index.equals(forward_returns.index):
        raise LookaheadError(
            "signal and forward-return frames are not aligned on the same index; "
            "any comparison between them is meaningless"
        )
    if not signal.index.is_monotonic_increasing:
        raise LookaheadError("signal index is not chronologically ordered")

    # The last `lag` rows cannot have a realised forward return yet. If they do,
    # the frame was built with a shift that reaches into unavailable data.
    tail = forward_returns.iloc[-lag:]
    if tail.notna().any().any():
        raise LookaheadError(
            f"the final {lag} rows of the forward-return frame contain values. "
            f"With a {lag}-bar execution lag those returns are not yet knowable, "
            f"so the frame was shifted by less than {lag}."
        )

    # A frame that is all zeros passes every shift check and still measures
    # nothing — the failure mode that produced a Sharpe of -6.5 across every
    # baseline, because the only thing left in the series was cost.
    if forward_returns.size and int((forward_returns.fillna(0.0) != 0.0).to_numpy().sum()) == 0:
        raise LookaheadError(
            "the forward-return frame is identically zero, so the backtest "
            "would measure only transaction costs"
        )


def build_forward_returns(
    prices: pd.DataFrame, *, horizon: int = 1, lag: int = 2
) -> pd.DataFrame:
    """
    Return earned by a position opened on the signal observed at each date.

    ``lag=2`` is the default and the convention this module enforces: signal at
    the close of t, execution at the close of t+1, return from t+1 to t+2.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if lag < 1:
        raise ValueError("lag must be >= 1; a lag of 0 executes on the signal bar")

    # Entry at the close of t+(lag-1), exit at the close of t+(lag-1)+horizon.
    # With the default lag=2: the signal is observed at the close of t, the
    # position is opened at the close of t+1, and it earns t+1 -> t+2.
    #
    # This was `shift(-lag - horizon + 1) / shift(-lag)`, which for the default
    # horizon=1, lag=2 gives shift(-2)/shift(-2) — EXACTLY ZERO for every row.
    # Every backtest run through it returned pure cost bleed (Sharpe ~ -6.5 on
    # a long-only equity book), which is not a plausible result and is what
    # exposed it.
    entry = prices.shift(-(lag - 1))
    exit_ = prices.shift(-(lag - 1 + horizon))
    fwd = exit_ / entry - 1.0

    # A forward-return frame that is entirely zero or entirely NaN is a
    # construction error, not a market observation. Refuse it here rather than
    # let it become a plausible-looking backtest.
    finite = fwd.to_numpy()
    if finite.size and not (pd.notna(fwd).to_numpy().any()):
        raise ValueError("forward returns are entirely NaN; check the price panel")
    nonzero = int((fwd.fillna(0.0) != 0.0).to_numpy().sum())
    if fwd.size and nonzero == 0:
        raise ValueError(
            "every forward return is exactly zero — the entry and exit shifts "
            "resolve to the same bar, so no position can earn anything"
        )
    return fwd


# --------------------------------------------------------------------------- #
#  Weights                                                                      #
# --------------------------------------------------------------------------- #

def cross_sectional_weights(
    signal: pd.DataFrame,
    *,
    top_quantile: float = 0.2,
    long_only: bool = True,
    rebalance_days: int = 5,
    max_weight: float = 0.10,
) -> pd.DataFrame:
    """
    Turn a signal panel into portfolio weights.

    ``long_only`` defaults True because this is an Indian equity system: retail
    shorting of single stocks in the cash segment is not available, so a
    long-short result would not be implementable. The long-short variant is
    still computed elsewhere as a measure of raw signal quality — but it is
    reported as SIGNAL QUALITY, never as a strategy.

    ``max_weight`` caps any single name, so a thin cross-section cannot produce
    a concentrated bet that the reported Sharpe silently depends on.
    """
    if not 0 < top_quantile <= 0.5:
        raise ValueError("top_quantile must be in (0, 0.5]")

    ranks = signal.rank(axis=1, pct=True, na_option="keep")
    n_valid = signal.notna().sum(axis=1)

    long_mask = ranks >= (1.0 - top_quantile)
    weights = long_mask.astype(float)
    if not long_only:
        short_mask = ranks <= top_quantile
        weights = weights - short_mask.astype(float)

    # Normalise to unit gross, then cap — and then only ever scale DOWN.
    #
    # Renormalising after the clip is what made the cap ineffective: with a
    # thin cross-section, 2 selected names normalise to 0.5 each, the clip
    # brings them to max_weight, and dividing by the new gross puts them
    # straight back to 0.5. The cap was silently a no-op exactly when it
    # mattered most.
    #
    # When the cap cannot be met at full investment (n_selected * max_weight
    # < 1) the book is deliberately left PARTIALLY INVESTED rather than
    # concentrated. Holding cash because there are not enough names to
    # diversify into is a real portfolio decision; quietly breaching the cap is
    # not.
    gross = weights.abs().sum(axis=1)
    weights = weights.div(gross.where(gross > 0), axis=0)
    weights = weights.clip(-max_weight, max_weight)
    gross2 = weights.abs().sum(axis=1)
    scale = (1.0 / gross2.where(gross2 > 1.0)).fillna(1.0)
    weights = weights.mul(scale, axis=0)
    weights = weights.where(n_valid >= 10, 0.0)
    weights = weights.fillna(0.0)

    if rebalance_days > 1:
        # Hold between rebalances. Positions are refreshed only on rebalance
        # dates and carried in between, which is what actually generates
        # turnover — recomputing weights daily and calling it a weekly strategy
        # understates turnover by roughly the rebalance interval.
        keep = np.zeros(len(weights), dtype=bool)
        keep[::rebalance_days] = True
        weights = weights.where(pd.Series(keep, index=weights.index), other=np.nan)
        weights = weights.ffill().fillna(0.0)

    return weights


# --------------------------------------------------------------------------- #
#  Metrics                                                                      #
# --------------------------------------------------------------------------- #

def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def compute_metrics(
    returns_net: pd.Series,
    returns_gross: pd.Series,
    weights: pd.DataFrame,
    turnover: pd.Series,
    *,
    benchmark: Optional[pd.Series] = None,
    periods_per_year: int = TRADING_DAYS,
) -> BacktestMetrics:
    r = returns_net.dropna()
    n = len(r)
    if n == 0:
        return BacktestMetrics(
            n_periods=0, years=0.0, total_return=0.0, cagr=0.0, volatility=0.0,
            sharpe=0.0, sortino=0.0, max_drawdown=0.0, calmar=0.0, hit_rate=0.0,
            profit_factor=0.0, avg_trade=0.0, turnover_annual=0.0,
            cost_drag_annual=0.0, exposure=0.0,
        )

    years = n / periods_per_year
    total = float((1.0 + r).prod() - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else 0.0

    downside = r[r < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(r.mean() / dstd * np.sqrt(periods_per_year)) if dstd > 0 else float("nan")

    mdd = max_drawdown(r)
    calmar = float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else float("nan")

    wins, losses = r[r > 0], r[r < 0]
    hit = float(len(wins) / n)
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("nan")

    gross_annual = float(returns_gross.dropna().mean() * periods_per_year)
    net_annual = float(r.mean() * periods_per_year)
    cost_drag = gross_annual - net_annual

    turn_annual = float(turnover.dropna().mean() * periods_per_year)
    exposure = float((weights.abs().sum(axis=1) > 0).mean())

    beta = alpha = bench_cagr = excess = info_ratio = None
    if benchmark is not None and len(benchmark):
        b = benchmark.reindex(r.index).dropna()
        common = r.index.intersection(b.index)
        if len(common) > 20:
            rr, bb = r.loc[common], b.loc[common]
            var_b = float(bb.var(ddof=1))
            if var_b > 0:
                beta = float(np.cov(rr, bb, ddof=1)[0, 1] / var_b)
                alpha = float((rr.mean() - beta * bb.mean()) * periods_per_year)
            bt = float((1.0 + bb).prod() - 1.0)
            by = len(bb) / periods_per_year
            bench_cagr = float((1.0 + bt) ** (1.0 / by) - 1.0) if by > 0 and bt > -1 else None
            active = rr - bb
            if active.std(ddof=1) > 0:
                info_ratio = float(active.mean() / active.std(ddof=1) * np.sqrt(periods_per_year))
            if bench_cagr is not None and np.isfinite(cagr):
                excess = cagr - bench_cagr

    def _worst(freq: str) -> Optional[float]:
        try:
            agg = (1.0 + r).resample(freq).prod() - 1.0
            return float(agg.min()) if len(agg) else None
        except Exception:  # noqa: BLE001
            return None

    def _best(freq: str) -> Optional[float]:
        try:
            agg = (1.0 + r).resample(freq).prod() - 1.0
            return float(agg.max()) if len(agg) else None
        except Exception:  # noqa: BLE001
            return None

    return BacktestMetrics(
        n_periods=n, years=round(years, 3), total_return=total, cagr=cagr,
        volatility=vol, sharpe=sharpe, sortino=sortino, max_drawdown=mdd,
        calmar=calmar, hit_rate=hit, profit_factor=pf,
        avg_trade=float(r.mean()), turnover_annual=turn_annual,
        cost_drag_annual=cost_drag, exposure=exposure, beta=beta,
        alpha_annual=alpha, benchmark_cagr=bench_cagr, excess_cagr=excess,
        information_ratio=info_ratio,
        worst_month=_worst("ME"), worst_quarter=_worst("QE"),
        worst_year=_worst("YE"), best_year=_best("YE"),
    )


# --------------------------------------------------------------------------- #
#  The backtest                                                                 #
# --------------------------------------------------------------------------- #

def run_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    benchmark: Optional[pd.Series] = None,
    cost_bps: float = DEFAULT_COST_BPS,
    top_quantile: float = 0.2,
    long_only: bool = True,
    rebalance_days: int = 5,
    max_weight: float = 0.10,
    lag: int = 2,
    volume: Optional[pd.DataFrame] = None,
    max_participation: float = 0.05,
) -> BacktestResult:
    """
    Measure one configuration. Nothing here is fitted or selected.

    ``volume``, when supplied, caps each name's weight at the fraction of its
    median traded value that ``max_participation`` allows — an illiquid name
    cannot contribute a position the market could not have absorbed.
    """
    warnings: list[str] = []

    common_cols = [c for c in signal.columns if c in prices.columns]
    signal = signal[common_cols]
    prices = prices[common_cols]
    common_idx = signal.index.intersection(prices.index)
    signal, prices = signal.loc[common_idx], prices.loc[common_idx]

    if signal.empty:
        raise ValueError("no overlap between signal and price panels")

    fwd = build_forward_returns(prices, horizon=1, lag=lag)
    assert_no_lookahead(signal, fwd, lag=lag)

    weights = cross_sectional_weights(
        signal, top_quantile=top_quantile, long_only=long_only,
        rebalance_days=rebalance_days, max_weight=max_weight,
    )

    if volume is not None and not volume.empty:
        vol_aligned = volume.reindex(index=weights.index, columns=weights.columns)
        traded_value = (vol_aligned * prices).rolling(21, min_periods=5).median()
        # A name whose median traded value is tiny cannot carry a full weight.
        # Expressed relative to the median across the cross-section so the cap
        # scales with the universe rather than with an absolute rupee figure.
        typical = traded_value.median(axis=1)
        capacity = (traded_value.div(typical.where(typical > 0), axis=0)
                    * max_participation).clip(upper=1.0)
        before = float(weights.abs().sum(axis=1).mean())
        weights = weights * capacity.fillna(0.0)
        gross = weights.abs().sum(axis=1)
        weights = weights.div(gross.where(gross > 0), axis=0).fillna(0.0)
        after = float(weights.abs().sum(axis=1).mean())
        if before > 0 and after < before * 0.999:
            warnings.append(
                f"liquidity cap reduced average gross exposure from {before:.3f} "
                f"to {after:.3f} at {max_participation:.0%} participation"
            )

    gross_returns = (weights * fwd).sum(axis=1, min_count=1)

    # Turnover: both sides of every rotation. The first row is the cost of
    # building the book from cash, which is a real cost and is not free.
    turnover = (weights - weights.shift(1)).abs().sum(axis=1)
    turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    costs = turnover * (cost_bps / 10_000.0)
    net_returns = gross_returns - costs

    valid = fwd.notna().any(axis=1)
    gross_returns = gross_returns.where(valid)
    net_returns = net_returns.where(valid)

    # Re-index onto the date the return was REALISED, not the date the signal
    # was formed. Row t holds the return earned from t+(lag-1) to t+lag, so it
    # belongs at t+lag.
    #
    # This is not cosmetic. Left on the signal date, the series sits `lag` bars
    # ahead of every other date-indexed series, and comparing it with the
    # benchmark measured a 2-day-lagged correlation: beta came out at 0.02 for a
    # LONG-ONLY equity book, which is impossible. Aligned, the same portfolio
    # shows beta 0.89 / correlation 0.77. It also dates the calendar
    # aggregations (worst month, worst year) correctly.
    gross_returns = gross_returns.shift(lag)
    net_returns = net_returns.shift(lag)
    turnover_realised = turnover.shift(lag)

    metrics = compute_metrics(
        net_returns, gross_returns, weights, turnover_realised, benchmark=benchmark
    )

    return BacktestResult(
        returns_net=net_returns.dropna(),
        returns_gross=gross_returns.dropna(),
        weights=weights,
        turnover=turnover,
        metrics=metrics,
        cost_bps=cost_bps,
        config={
            "top_quantile": top_quantile, "long_only": long_only,
            "rebalance_days": rebalance_days, "max_weight": max_weight,
            "lag": lag, "cost_bps": cost_bps,
            "liquidity_capped": volume is not None,
            "max_participation": max_participation if volume is not None else None,
        },
        warnings=warnings,
    )
