"""
The study: walk-forward evaluation, robustness, and a pass/fail verdict.

THE ACCEPTANCE CRITERIA ARE FIXED BEFORE THE STUDY RUNS
-------------------------------------------------------
They are module constants, declared here and never touched in response to a
result. That is the entire mechanism preventing "the strategy is validated
because I moved the bar to where it landed". If a criterion is ever changed,
the change is a diff in this file, reviewable next to the result it produced.

HOW SELECTION WORKS, AND WHY IT MATTERS
---------------------------------------
Six baseline signals are evaluated. Reporting the best of six as if it were the
only one tested overstates its significance by roughly the number of trials —
this is the multiple-comparisons problem, and it is the single commonest way a
backtest lies.

Three separate defences are applied:

1. **Selection happens inside each walk-forward fold, on TRAIN data only.**
   The winner is chosen on the training window and then earns its return on the
   following test window, which it has never seen. Concatenating those test
   windows gives an out-of-sample track record for the SELECTION PROCEDURE, not
   for a signal picked with hindsight.
2. **The Deflated Sharpe Ratio** is computed with ``n_trials`` set to the real
   number of configurations examined, which discounts the observed Sharpe by
   what the best of N random strategies would have produced anyway.
3. **PBO (CSCV)** asks the different question of whether picking the in-sample
   best helps out-of-sample at all, or whether the ranking is noise.

WHAT IS NOT DONE
----------------
No parameter is searched. No window is tuned. No symbol or period is excluded
after the fact. The baselines use textbook parameters fixed in
``app/research/signals.py`` before any result was seen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from app.research.backtest import (
    DEFAULT_COST_BPS,
    TRADING_DAYS,
    BacktestResult,
    run_backtest,
)
from app.research.signals import BASELINE_SIGNALS, UNTESTABLE_SIGNALS
from app.research.statistics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    stationary_bootstrap,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  ACCEPTANCE CRITERIA — fixed in advance                                       #
# --------------------------------------------------------------------------- #

#: Out-of-sample Sharpe must clear this. 0.5 net of costs is a low bar for a
#: daily equity strategy and is deliberately low: the goal is to detect signal,
#: not to certify a good product.
MIN_OOS_SHARPE = 0.50

#: The 5th percentile of the bootstrapped Sharpe distribution must be positive.
#: A point estimate above zero with a confidence interval straddling it is not
#: evidence of anything.
MIN_BOOTSTRAP_LOWER = 0.0

#: Deflated Sharpe Ratio: probability the true Sharpe exceeds zero after
#: accounting for the number of trials and for non-normal returns.
MIN_DSR = 0.95

#: Probability of Backtest Overfitting. Above 0.5 means the in-sample winner
#: lands below the out-of-sample median more often than not — i.e. selection is
#: actively harmful.
MAX_PBO = 0.50

#: Must beat buy-and-hold on the same universe after costs. A strategy that
#: underperforms the index is not a strategy; it is an expensive index fund.
MIN_EXCESS_CAGR = 0.0

#: Must survive a doubling of the assumed transaction cost. A result that only
#: works at optimistic costs is a result about the cost assumption.
STRESS_COST_BPS = 50.0

#: The fraction of calendar years that must be positive on an excess-return
#: basis. Guards against a result driven by one extraordinary period.
MIN_POSITIVE_YEAR_FRACTION = 0.55


@dataclass
class FoldOutcome:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_signal: str
    train_sharpe: float
    test_sharpe: float
    test_return: float


@dataclass
class Criterion:
    name: str
    passed: bool
    observed: Any
    threshold: Any
    detail: str = ""


@dataclass
class StudyResult:
    verdict: str
    criteria: list[Criterion] = field(default_factory=list)
    in_sample: dict[str, dict] = field(default_factory=dict)
    folds: list[FoldOutcome] = field(default_factory=list)
    oos_metrics: dict[str, Any] = field(default_factory=dict)
    robustness: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    n_trials: int = 0

    @property
    def validated(self) -> bool:
        return self.verdict == "VALIDATED"


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    total = float((1.0 + r).prod() - 1.0)
    years = len(r) / TRADING_DAYS
    if years <= 0 or total <= -1:
        return float("nan")
    return float((1.0 + total) ** (1.0 / years) - 1.0)


# --------------------------------------------------------------------------- #
#  Walk-forward                                                                 #
# --------------------------------------------------------------------------- #

def walk_forward_selection(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    *,
    benchmark: Optional[pd.Series] = None,
    n_folds: int = 6,
    embargo_days: int = 10,
    cost_bps: float = DEFAULT_COST_BPS,
    min_train_days: int = 504,
) -> tuple[list[FoldOutcome], pd.Series]:
    """
    Select on train, measure on test, walk forward. Return the stitched OOS series.

    An EMBARGO of ``embargo_days`` sessions separates each train window from the
    test window that follows it. Without it the last training observations and
    the first test observations overlap through the signal's own trailing
    windows — a 12-month momentum signal computed on the first test day is built
    almost entirely from training-period prices. The embargo is what makes
    "out-of-sample" mean something.
    """
    dates = prices.index
    n = len(dates)
    if n < min_train_days + n_folds * 2:
        raise ValueError(f"not enough history: {n} sessions")

    # Equal test windows over the post-warmup portion of the timeline.
    test_size = (n - min_train_days) // n_folds
    folds: list[FoldOutcome] = []
    oos_chunks: list[pd.Series] = []

    for k in range(n_folds):
        train_end_i = min_train_days + k * test_size
        test_start_i = train_end_i + embargo_days
        test_end_i = min(test_start_i + test_size, n)
        if test_start_i >= n or test_end_i - test_start_i < 20:
            break

        train_slice = slice(0, train_end_i)
        test_slice = slice(test_start_i, test_end_i)

        # -- select on TRAIN only ---------------------------------------- #
        best_name, best_sharpe = None, -np.inf
        for name, sig in signals.items():
            tr = run_backtest(
                sig.iloc[train_slice], prices.iloc[train_slice],
                benchmark=benchmark, cost_bps=cost_bps,
            )
            s = _sharpe(tr.returns_net)
            if s > best_sharpe:
                best_name, best_sharpe = name, s

        if best_name is None:
            # No signal produced a finite Sharpe on this training window. Skip
            # the fold rather than fall back to an arbitrary choice — a fold
            # with no basis for selection contributes no honest evidence.
            logger.warning("fold %d: no signal was selectable on train; skipped", k)
            continue

        # -- measure on TEST, which selection never saw ------------------- #
        te = run_backtest(
            signals[best_name].iloc[test_slice], prices.iloc[test_slice],
            benchmark=benchmark, cost_bps=cost_bps,
        )
        te_r = te.returns_net
        oos_chunks.append(te_r)

        folds.append(FoldOutcome(
            fold=k,
            train_start=str(dates[0].date()),
            train_end=str(dates[train_end_i - 1].date()),
            test_start=str(dates[test_start_i].date()),
            test_end=str(dates[test_end_i - 1].date()),
            selected_signal=best_name,
            train_sharpe=round(float(best_sharpe), 4),
            test_sharpe=round(_sharpe(te_r), 4),
            test_return=round(float((1.0 + te_r.dropna()).prod() - 1.0), 4),
        ))

    oos = pd.concat(oos_chunks).sort_index() if oos_chunks else pd.Series(dtype=float)
    # Overlapping fold boundaries can duplicate a date; keep the first.
    oos = oos[~oos.index.duplicated(keep="first")]
    return folds, oos


# --------------------------------------------------------------------------- #
#  Robustness                                                                   #
# --------------------------------------------------------------------------- #

def cost_sensitivity(
    signal: pd.DataFrame, prices: pd.DataFrame, benchmark: Optional[pd.Series],
    levels: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0, 100.0),
) -> dict[str, dict]:
    """Sharpe and CAGR at several cost assumptions, including zero."""
    out = {}
    for bps in levels:
        m = run_backtest(signal, prices, benchmark=benchmark, cost_bps=bps).metrics
        out[f"{bps:g}bps"] = {
            "sharpe": round(m.sharpe, 4), "cagr": round(m.cagr, 4),
            "excess_cagr": round(m.excess_cagr, 4) if m.excess_cagr is not None else None,
        }
    return out


def subperiod_analysis(returns: pd.Series, benchmark: Optional[pd.Series]) -> dict:
    """Per-calendar-year returns, and the same versus the benchmark."""
    if returns.empty:
        return {}
    yr = (1.0 + returns).resample("YE").prod() - 1.0
    out: dict[str, Any] = {
        "by_year": {str(i.year): round(float(v), 4) for i, v in yr.items()},
        "positive_years": int((yr > 0).sum()),
        "total_years": int(len(yr)),
    }
    if benchmark is not None and len(benchmark):
        b = benchmark.reindex(returns.index).dropna()
        byr = (1.0 + b).resample("YE").prod() - 1.0
        common = yr.index.intersection(byr.index)
        excess = yr.loc[common] - byr.loc[common]
        out["excess_by_year"] = {str(i.year): round(float(v), 4) for i, v in excess.items()}
        out["positive_excess_years"] = int((excess > 0).sum())
        out["total_excess_years"] = int(len(excess))
    return out


def regime_analysis(returns: pd.Series, benchmark: pd.Series) -> dict:
    """
    Performance split by BENCHMARK regime, classified on trailing data only.

    The regime label for a date uses the benchmark's trailing 126-day return, so
    it is knowable on that date. Labelling regimes with hindsight — "this was
    the 2020 crash" — would let the analysis condition on the future.
    """
    b = benchmark.reindex(returns.index).dropna()
    common = returns.index.intersection(b.index)
    if len(common) < 200:
        return {"error": "insufficient overlap for regime analysis"}
    r, bb = returns.loc[common], b.loc[common]
    trailing = (1.0 + bb).rolling(126, min_periods=60).apply(np.prod, raw=True) - 1.0

    labels = pd.Series("sideways", index=common)
    labels[trailing > 0.10] = "bull"
    labels[trailing < -0.10] = "bear"

    out = {}
    for name in ("bull", "bear", "sideways"):
        mask = labels == name
        if mask.sum() < 20:
            out[name] = {"days": int(mask.sum()), "note": "too few days to measure"}
            continue
        out[name] = {
            "days": int(mask.sum()),
            "sharpe": round(_sharpe(r[mask]), 4),
            "mean_daily": round(float(r[mask].mean()), 6),
            "benchmark_mean_daily": round(float(bb[mask].mean()), 6),
        }
    return out


def bootstrap_sharpe_ci(returns: pd.Series, n_sims: int = 2000) -> dict:
    """
    Stationary-bootstrap confidence interval for the Sharpe ratio.

    Block resampling, not IID: daily equity returns have volatility clustering
    and autocorrelation, and IID resampling destroys both — producing intervals
    that are far too narrow for exactly the strategies whose risk matters most.
    """
    r = returns.dropna().to_numpy()
    if len(r) < 60:
        return {"error": "too few observations for a bootstrap"}
    sims = stationary_bootstrap(r, n_simulations=n_sims, random_seed=42)
    sharpes = []
    for path in sims:
        sd = float(np.std(path, ddof=1))
        sharpes.append(float(np.mean(path) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0)
    arr = np.asarray(sharpes)
    return {
        "point_estimate": round(_sharpe(returns), 4),
        "mean": round(float(arr.mean()), 4),
        "p05": round(float(np.percentile(arr, 5)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "fraction_above_zero": round(float((arr > 0).mean()), 4),
        "n_simulations": int(len(arr)),
    }


def concentration_analysis(result: BacktestResult) -> dict:
    """How much of the result rests on a few names or a few days."""
    r = result.returns_net.dropna()
    if r.empty:
        return {}
    total = float((1.0 + r).prod() - 1.0)
    top10 = r.nlargest(10)
    without_top10 = r.drop(top10.index)
    w = result.weights
    names_held = int((w.abs() > 0).any(axis=0).sum())
    return {
        "total_return": round(total, 4),
        "top_10_days_contribution": round(float((1.0 + top10).prod() - 1.0), 4),
        "return_excluding_top_10_days": round(
            float((1.0 + without_top10).prod() - 1.0), 4
        ),
        "sharpe_excluding_top_10_days": round(_sharpe(without_top10), 4),
        "distinct_names_held": names_held,
        "mean_positions_per_day": round(float((w.abs() > 0).sum(axis=1).mean()), 2),
    }


def parameter_perturbation(
    signal_fn: Callable[..., pd.DataFrame], prices: pd.DataFrame,
    benchmark: Optional[pd.Series], *, variants: dict[str, dict],
) -> dict:
    """
    Re-measure under neighbouring configurations.

    This is a STABILITY check, not a search. Every variant is reported; none is
    selected. A result that collapses when the rebalance interval moves from 5
    days to 10 was a property of the parameter, not of the market.
    """
    # dict[str, Any] because a variant that RAISES records its error string
    # here rather than being dropped. A silently missing variant reads as one
    # that was never tried, which is the opposite of what a stability check is
    # supposed to show.
    out: dict[str, dict[str, Any]] = {}
    for label, kwargs in variants.items():
        try:
            m = run_backtest(
                signal_fn(prices), prices, benchmark=benchmark, **kwargs
            ).metrics
            out[label] = {"sharpe": round(m.sharpe, 4), "cagr": round(m.cagr, 4)}
        except Exception as exc:  # noqa: BLE001
            out[label] = {"error": str(exc)}
    return out


# --------------------------------------------------------------------------- #
#  The study                                                                    #
# --------------------------------------------------------------------------- #

def run_study(
    prices: pd.DataFrame,
    *,
    benchmark: Optional[pd.Series] = None,
    stress_prices: Optional[pd.DataFrame] = None,
    cost_bps: float = DEFAULT_COST_BPS,
    n_folds: int = 6,
    limitations: Optional[list[str]] = None,
    volume: Optional[pd.DataFrame] = None,
) -> StudyResult:
    """Run the full evaluation and return a verdict with its evidence."""
    signals = {name: fn(prices) for name, fn in BASELINE_SIGNALS.items()}
    n_trials = len(signals)

    # ---- in-sample, for reference only ------------------------------------ #
    in_sample: dict[str, dict] = {}
    is_results: dict[str, BacktestResult] = {}
    for name, sig in signals.items():
        res = run_backtest(
            sig, prices, benchmark=benchmark, cost_bps=cost_bps, volume=volume
        )
        is_results[name] = res
        m = res.metrics
        in_sample[name] = {
            "sharpe": round(m.sharpe, 4), "cagr": round(m.cagr, 4),
            "max_drawdown": round(m.max_drawdown, 4),
            "excess_cagr": round(m.excess_cagr, 4) if m.excess_cagr is not None else None,
            "beta": round(m.beta, 4) if m.beta is not None else None,
            "turnover_annual": round(m.turnover_annual, 2),
            "cost_drag_annual": round(m.cost_drag_annual, 4),
            "NOTE": "IN-SAMPLE, full period. NOT evidence of anything.",
        }

    # ---- walk-forward ----------------------------------------------------- #
    folds, oos = walk_forward_selection(
        signals, prices, benchmark=benchmark, n_folds=n_folds, cost_bps=cost_bps
    )

    oos_sharpe = _sharpe(oos)
    oos_cagr = _cagr(oos)
    bench_oos_cagr = None
    excess_cagr = None
    if benchmark is not None and len(oos):
        b = benchmark.reindex(oos.index).dropna()
        if len(b) > 20:
            bench_oos_cagr = _cagr(b)
            if np.isfinite(oos_cagr) and np.isfinite(bench_oos_cagr):
                excess_cagr = oos_cagr - bench_oos_cagr

    boot = bootstrap_sharpe_ci(oos)

    dsr = deflated_sharpe_ratio(oos.dropna().tolist(), n_trials=n_trials)
    dsr_value = float(getattr(dsr, "deflated_sharpe_ratio", 0.0))

    # PBO across the six configurations over the same timeline.
    perf = np.column_stack([
        is_results[n].returns_net.reindex(
            is_results[list(signals)[0]].returns_net.index
        ).fillna(0.0).to_numpy()
        for n in signals
    ])
    try:
        pbo = probability_of_backtest_overfitting(perf, n_partitions=8)
        pbo_value = float(getattr(pbo, "pbo", 1.0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("PBO failed: %s", exc)
        pbo_value = float("nan")

    sub = subperiod_analysis(oos, benchmark)
    regimes = regime_analysis(oos, benchmark) if benchmark is not None else {}

    # ---- robustness on the most-selected signal --------------------------- #
    picked = (
        max({f.selected_signal for f in folds},
            key=lambda s: sum(1 for f in folds if f.selected_signal == s))
        if folds else list(signals)[0]
    )
    picked_fn = BASELINE_SIGNALS[picked]

    robustness: dict[str, Any] = {
        "most_selected_signal": picked,
        "cost_sensitivity": cost_sensitivity(signals[picked], prices, benchmark),
        "parameter_perturbation": parameter_perturbation(
            picked_fn, prices, benchmark,
            variants={
                "rebalance_1d": {"rebalance_days": 1},
                "rebalance_5d": {"rebalance_days": 5},
                "rebalance_10d": {"rebalance_days": 10},
                "rebalance_21d": {"rebalance_days": 21},
                "top_decile": {"top_quantile": 0.1},
                "top_quintile": {"top_quantile": 0.2},
                "top_third": {"top_quantile": 0.33},
            },
        ),
        "concentration": concentration_analysis(is_results[picked]),
        "subperiod": sub,
        "regimes": regimes,
        "bootstrap": boot,
    }

    # Universe sensitivity: re-run with known corporate failures added back.
    if stress_prices is not None and not stress_prices.empty:
        combined = pd.concat([prices, stress_prices], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        stressed = run_backtest(
            picked_fn(combined), combined, benchmark=benchmark, cost_bps=cost_bps
        ).metrics
        base = is_results[picked].metrics
        robustness["universe_sensitivity"] = {  # type: ignore[assignment]
            "method": (
                "Re-run with known Indian corporate failures added to the "
                "survivor-only universe. A hand-picked failure set is itself "
                "biased, so this bounds the effect rather than removing it."
            ),
            "added_symbols": int(stress_prices.shape[1]),
            "survivor_only": {
                "sharpe": round(base.sharpe, 4), "cagr": round(base.cagr, 4),
            },
            "with_failures": {
                "sharpe": round(stressed.sharpe, 4), "cagr": round(stressed.cagr, 4),
            },
            "sharpe_delta": round(stressed.sharpe - base.sharpe, 4),
            "cagr_delta": round(stressed.cagr - base.cagr, 4),
        }

    # ---- criteria --------------------------------------------------------- #
    stress = run_backtest(
        signals[picked], prices, benchmark=benchmark, cost_bps=STRESS_COST_BPS
    ).metrics
    pos_excess_frac = (
        sub.get("positive_excess_years", 0) / sub["total_excess_years"]
        if sub.get("total_excess_years") else 0.0
    )

    criteria = [
        Criterion(
            "oos_sharpe", oos_sharpe >= MIN_OOS_SHARPE, round(oos_sharpe, 4),
            MIN_OOS_SHARPE,
            "Walk-forward out-of-sample Sharpe, net of costs.",
        ),
        Criterion(
            "bootstrap_lower_bound",
            bool(boot.get("p05", -1) > MIN_BOOTSTRAP_LOWER),
            boot.get("p05"), MIN_BOOTSTRAP_LOWER,
            "5th percentile of the stationary-bootstrap Sharpe distribution.",
        ),
        Criterion(
            "deflated_sharpe", dsr_value >= MIN_DSR, round(dsr_value, 4), MIN_DSR,
            f"P(true Sharpe > 0) after deflating for {n_trials} trials.",
        ),
        Criterion(
            "pbo", bool(np.isfinite(pbo_value) and pbo_value <= MAX_PBO),
            round(pbo_value, 4) if np.isfinite(pbo_value) else None, MAX_PBO,
            "Probability the in-sample winner underperforms out-of-sample.",
        ),
        Criterion(
            "excess_cagr_vs_benchmark",
            bool(excess_cagr is not None and excess_cagr > MIN_EXCESS_CAGR),
            round(excess_cagr, 4) if excess_cagr is not None else None,
            MIN_EXCESS_CAGR,
            "Out-of-sample CAGR minus benchmark CAGR over the same dates.",
        ),
        Criterion(
            "survives_double_costs", stress.sharpe >= MIN_OOS_SHARPE,
            round(stress.sharpe, 4), MIN_OOS_SHARPE,
            f"In-sample Sharpe at {STRESS_COST_BPS:g}bps/side.",
        ),
        Criterion(
            "consistent_across_years", pos_excess_frac >= MIN_POSITIVE_YEAR_FRACTION,
            round(pos_excess_frac, 4), MIN_POSITIVE_YEAR_FRACTION,
            "Fraction of out-of-sample years with positive excess return.",
        ),
    ]

    lims = list(limitations or [])
    # A data-integrity gate. Even a clean statistical pass cannot certify a
    # strategy measured on a universe that excludes every company that failed.
    survivorship_blocking = any("SURVIVORSHIP" in x.upper() for x in lims)
    criteria.append(Criterion(
        "point_in_time_universe", not survivorship_blocking,
        "present-day snapshot" if survivorship_blocking else "point-in-time",
        "point-in-time",
        "A universe that excludes delisted companies cannot support a "
        "validation claim, however strong the statistics computed on it.",
    ))

    verdict = "VALIDATED" if all(c.passed for c in criteria) else "NOT VALIDATED"

    return StudyResult(
        verdict=verdict,
        criteria=criteria,
        in_sample=in_sample,
        folds=folds,
        oos_metrics={
            "n_observations": int(len(oos.dropna())),
            "sharpe": round(oos_sharpe, 4),
            "cagr": round(oos_cagr, 4) if np.isfinite(oos_cagr) else None,
            "benchmark_cagr": round(bench_oos_cagr, 4) if bench_oos_cagr else None,
            "excess_cagr": round(excess_cagr, 4) if excess_cagr is not None else None,
            "deflated_sharpe_ratio": round(dsr_value, 4),
            "pbo": round(pbo_value, 4) if np.isfinite(pbo_value) else None,
            "n_trials": n_trials,
        },
        robustness=robustness,
        limitations=lims + [
            f"UNTESTABLE: {k} — {v}" for k, v in UNTESTABLE_SIGNALS.items()
        ],
        n_trials=n_trials,
    )
