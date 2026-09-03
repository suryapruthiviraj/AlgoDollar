"""
walkforward.py — Walk-forward validation and Monte Carlo simulation.

Walk-Forward Protocol
---------------------
  1. Train model on TRAIN period (e.g. 3 years).
  2. Tune hyperparameters on VAL period (e.g. 1 year).
  3. Evaluate on TEST period (e.g. 1 year) — NEVER touch after selection.
  4. Roll window forward by step_years.
  5. Collect all OOS test-period results into a combined equity curve.

The combined OOS equity curve is the only honest assessment of strategy
performance.  In-sample and val-period results are reported for diagnostics
but must NOT be used as the primary performance claim.

Overfitting detection
---------------------
If train_sharpe >> oos_sharpe consistently across windows, the strategy is
likely overfit.  flag_overfit() returns True if:
  mean(train_sharpe) > 2 × mean(oos_sharpe) AND mean(oos_sharpe) < 0.5

Monte Carlo simulation
----------------------
Bootstrap resampling of daily returns:
  - Resample WITH replacement to preserve marginal distribution.
  - Does NOT preserve autocorrelation structure — treat as lower bound on
    true distribution of outcomes.

Survivorship bias note
----------------------
All results inherit the bias of the input universe.  A static universe
overstates returns.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from app.backtesting.engine import (
    BacktestMetrics, BacktestResult, EventDrivenBacktester
)
from app.research.statistics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    stationary_bootstrap,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RF_DAILY = 0.065 / 252


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Results for one walk-forward window."""
    train_start: str
    train_end:   str
    val_start:   str
    val_end:     str
    test_start:  str
    test_end:    str
    train_sharpe: float
    val_sharpe:   float
    oos_sharpe:   float
    oos_equity_curve: pd.Series
    oos_metrics:  BacktestMetrics
    best_params:  dict


@dataclass
class WalkForwardResult:
    windows:                   List[WindowResult]
    combined_oos_equity_curve: pd.Series
    combined_metrics:          BacktestMetrics
    parameter_stability:       Dict[str, Any]
    oos_sharpe_distribution:   List[float]
    mean_train_sharpe:         float
    mean_oos_sharpe:           float

    def overfitting_flags(self) -> dict[str, bool]:
        """
        Independent overfitting warning signs, each evaluated separately.

        The previous single criterion (train > 2x OOS *and* OOS < 0.5) missed
        the most common real case: a strategy with train Sharpe 3.0 and OOS
        Sharpe 0.6 has clearly overfit, but passed because 0.6 > 0.5. These
        checks are combined with OR, not AND — any one of them firing is
        reason to distrust the result.
        """
        oos = np.asarray(self.oos_sharpe_distribution, dtype=float)
        n = len(oos)

        flags = {
            # Large in-sample/out-of-sample gap, regardless of the OOS level.
            "large_is_oos_gap": (
                self.mean_train_sharpe - self.mean_oos_sharpe > 1.0
            ),
            # OOS performance that does not clear a plausible cost hurdle.
            "weak_oos": self.mean_oos_sharpe < 0.5,
            # Performance concentrated in a minority of windows: a strategy
            # that works in 2 of 7 periods is a regime bet, not an edge.
            "inconsistent_across_windows": (
                n > 0 and float((oos > 0).mean()) < 0.6
            ),
            # High dispersion relative to the mean: the point estimate is not
            # distinguishable from noise across windows.
            "unstable_oos": (
                n > 1
                and self.mean_oos_sharpe > 0
                and float(oos.std(ddof=1)) > abs(self.mean_oos_sharpe)
            ),
            # A suspiciously high OOS Sharpe usually means a leak, not alpha.
            "implausibly_high_oos": self.mean_oos_sharpe > 3.0,
            # Parameters that jump between windows describe noise, not a
            # stable phenomenon.
            "unstable_parameters": self._parameters_unstable(),
        }
        return flags

    def _parameters_unstable(self) -> bool:
        """True if any selected numeric parameter varies wildly across windows."""
        for spec in self.parameter_stability.values():
            mean, std = spec.get("mean"), spec.get("std")
            if mean and std and abs(mean) > 1e-12 and std / abs(mean) > 0.5:
                return True
        return False

    def flag_overfit(self) -> bool:
        """True if ANY overfitting warning sign is present."""
        if not self.oos_sharpe_distribution:
            return False
        return any(self.overfitting_flags().values())

    def overfitting_report(self) -> str:
        """Human-readable list of which warning signs fired."""
        flags = self.overfitting_flags()
        fired = [k for k, v in flags.items() if v]
        if not fired:
            return "No overfitting warning signs detected."
        return "OVERFITTING WARNINGS: " + ", ".join(fired)


@dataclass
class MonteCarloResult:
    n_simulations:        int
    percentiles:          Dict[str, float]   # "p5", "p25", "p50", "p75", "p95"
    risk_of_ruin_pct:     float              # P(equity < 0.5 × initial) at 3 yr
    prob_loss_at_1yr:     float              # P(return < 0) at 1 year
    prob_loss_at_3yr:     float              # P(return < 0) at 3 years
    var_95_annual:        float              # 95% Value at Risk, annualized fraction
    confidence_intervals: Dict[str, Tuple[float, float]]
    simulated_paths:      Optional[np.ndarray]  # shape (n_sim, T), only if saved


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

class WalkForwardEngine:
    """
    Anchored or rolling walk-forward validation.

    Parameters
    ----------
    backtester : EventDrivenBacktester instance.
    param_grid : dict of param_name → list_of_values.
        Hyperparameter search space.  All combinations are tried on val period.
    """

    def __init__(
        self,
        backtester: Optional[EventDrivenBacktester] = None,
        param_grid: Optional[Dict[str, List]] = None,
    ):
        self._backtester = backtester or EventDrivenBacktester()
        self._param_grid = param_grid or {}

    def run(
        self,
        strategy_class: Type,
        base_params: dict,
        prices_df: pd.DataFrame,
        features_fn: Callable,
        cost_model=None,
        universe: Optional[List[str]] = None,
        train_years: int = 3,
        val_years: int = 1,
        test_years: int = 1,
        step_years: int = 1,
        initial_capital: float = 100_000.0,
        volume_df: Optional[pd.DataFrame] = None,
        nifty_df: Optional[pd.DataFrame] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Parameters
        ----------
        strategy_class : class (not instance), the strategy to instantiate.
        base_params : dict, default parameters for the strategy.
        prices_df : DataFrame (T × N), close prices, DatetimeIndex.
        features_fn : callable(prices_df, volume_df, nifty_df) → features_df.
        cost_model : ZerodhaCostModel or compatible.
        universe : list[str] or None (uses prices_df.columns).
        train_years, val_years, test_years, step_years : int.
        initial_capital : float.
        volume_df, nifty_df : DataFrames (optional).

        Returns
        -------
        WalkForwardResult
        """
        if universe is None:
            universe = list(prices_df.columns)

        dates = prices_df.index
        start_ts = dates[0]
        end_ts   = dates[-1]

        windows: List[WindowResult] = []
        oos_equity_pieces: List[pd.Series] = []
        combined_capital = initial_capital

        # Generate window boundaries
        window_start = start_ts
        while True:
            train_end = window_start + pd.DateOffset(years=train_years)
            val_end   = train_end   + pd.DateOffset(years=val_years)
            test_end  = val_end     + pd.DateOffset(years=test_years)

            if test_end > end_ts:
                break

            # Slice dates to trading-day boundaries
            train_dates = dates[(dates >= window_start) & (dates <= train_end)]
            val_dates   = dates[(dates > train_end) & (dates <= val_end)]
            test_dates  = dates[(dates > val_end) & (dates <= test_end)]

            if len(train_dates) < 63 or len(val_dates) < 21 or len(test_dates) < 21:
                break

            logger.info(
                "Walk-forward window: train=%s→%s val=%s→%s test=%s→%s",
                train_dates[0].date(), train_dates[-1].date(),
                val_dates[0].date(),   val_dates[-1].date(),
                test_dates[0].date(),  test_dates[-1].date(),
            )

            # ---- Model fitting on train + val ----
            best_params = self._select_params_on_val(
                strategy_class=strategy_class,
                base_params=base_params,
                prices_df=prices_df,
                features_fn=features_fn,
                cost_model=cost_model,
                universe=universe,
                train_dates=train_dates,
                val_dates=val_dates,
                initial_capital=initial_capital,
                volume_df=volume_df,
                nifty_df=nifty_df,
            )

            # ---- Train Sharpe (in-sample) ----
            train_result = self._run_window(
                strategy_class, {**base_params, **best_params},
                prices_df, features_fn, cost_model, universe,
                train_dates, initial_capital, volume_df, nifty_df,
            )
            train_sharpe = train_result.metrics.sharpe_ratio

            # ---- Val Sharpe ----
            val_result = self._run_window(
                strategy_class, {**base_params, **best_params},
                prices_df, features_fn, cost_model, universe,
                val_dates, initial_capital, volume_df, nifty_df,
            )
            val_sharpe = val_result.metrics.sharpe_ratio

            # ---- OOS test (NEVER used for selection) ----
            test_result = self._run_window(
                strategy_class, {**base_params, **best_params},
                prices_df, features_fn, cost_model, universe,
                test_dates, combined_capital, volume_df, nifty_df,
            )
            oos_sharpe = test_result.metrics.sharpe_ratio
            oos_equity = test_result.equity_curve

            # Chain OOS equity: normalize to continue from previous window
            if oos_equity_pieces:
                scale = combined_capital / (oos_equity.iloc[0] or 1.0)
                oos_equity = oos_equity * scale
            combined_capital = float(oos_equity.iloc[-1])
            oos_equity_pieces.append(oos_equity)

            windows.append(WindowResult(
                train_start=str(train_dates[0].date()),
                train_end=  str(train_dates[-1].date()),
                val_start=  str(val_dates[0].date()),
                val_end=    str(val_dates[-1].date()),
                test_start= str(test_dates[0].date()),
                test_end=   str(test_dates[-1].date()),
                train_sharpe=train_sharpe,
                val_sharpe=  val_sharpe,
                oos_sharpe=  oos_sharpe,
                oos_equity_curve=oos_equity,
                oos_metrics=test_result.metrics,
                best_params=best_params,
            ))

            window_start += pd.DateOffset(years=step_years)

        if not windows:
            raise ValueError("No complete walk-forward windows could be formed.")

        # ---- Combined OOS equity curve ----
        combined_oos = pd.concat(oos_equity_pieces).sort_index()
        combined_oos = combined_oos[~combined_oos.index.duplicated(keep="last")]

        daily_rets = combined_oos.pct_change().dropna()
        combined_metrics = EventDrivenBacktester._compute_metrics(
            equity_curve=combined_oos,
            daily_returns=daily_rets,
            trade_records=[],
            initial_capital=initial_capital,
            start_date=combined_oos.index[0],
            end_date=combined_oos.index[-1],
        )

        oos_sharpe_dist = [w.oos_sharpe for w in windows]
        train_sharpe_dist = [w.train_sharpe for w in windows]

        # Parameter stability: how much do best_params vary across windows
        all_keys = set()
        for w in windows:
            all_keys.update(w.best_params.keys())
        stability = {}
        for k in all_keys:
            vals = [w.best_params.get(k) for w in windows if w.best_params.get(k) is not None]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                stability[k] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "values": vals,
                }

        result = WalkForwardResult(
            windows=windows,
            combined_oos_equity_curve=combined_oos,
            combined_metrics=combined_metrics,
            parameter_stability=stability,
            oos_sharpe_distribution=oos_sharpe_dist,
            mean_train_sharpe=float(np.mean(train_sharpe_dist)),
            mean_oos_sharpe=float(np.mean(oos_sharpe_dist)),
        )
        logger.info(
            "Walk-forward complete: %d windows | mean_train_sharpe=%.2f | "
            "mean_oos_sharpe=%.2f | overfit=%s",
            len(windows),
            result.mean_train_sharpe,
            result.mean_oos_sharpe,
            result.flag_overfit(),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_window(
        self,
        strategy_class,
        params: dict,
        prices_df: pd.DataFrame,
        features_fn: Callable,
        cost_model,
        universe: List[str],
        dates: pd.DatetimeIndex,
        initial_capital: float,
        volume_df,
        nifty_df,
    ) -> BacktestResult:
        strategy = strategy_class(**params)
        return self._backtester.run(
            strategy=strategy,
            universe=universe,
            prices_df=prices_df.loc[dates],
            features_fn=features_fn,
            cost_model=cost_model,
            initial_capital=initial_capital,
            volume_df=volume_df.loc[dates] if volume_df is not None else None,
            nifty_df=nifty_df.loc[dates] if nifty_df is not None else None,
        )

    def _select_params_on_val(
        self,
        strategy_class,
        base_params: dict,
        prices_df: pd.DataFrame,
        features_fn: Callable,
        cost_model,
        universe: List[str],
        train_dates: pd.DatetimeIndex,
        val_dates: pd.DatetimeIndex,
        initial_capital: float,
        volume_df,
        nifty_df,
    ) -> dict:
        """
        Grid search over param_grid; select best on val Sharpe.

        If param_grid is empty, return {} (use base_params as-is).
        """
        if not self._param_grid:
            return {}

        best_sharpe = -np.inf
        best_params: dict = {}

        param_combinations = _cartesian(self._param_grid)
        for combo in param_combinations:
            try:
                result = self._run_window(
                    strategy_class, {**base_params, **combo},
                    prices_df, features_fn, cost_model, universe,
                    val_dates, initial_capital, volume_df, nifty_df,
                )
                sharpe = result.metrics.sharpe_ratio
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = combo
            except Exception as exc:
                logger.debug("Param combo failed: %s — %s", combo, exc)

        logger.debug("Best val params: %s (Sharpe=%.3f)", best_params, best_sharpe)
        return best_params


# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """
    Bootstrap Monte Carlo simulation from an empirical return series.

    Default method: STATIONARY BLOCK BOOTSTRAP (Politis-Romano 1994).

    The obvious approach — resampling daily returns independently — is wrong
    for trading strategies. It destroys autocorrelation and volatility
    clustering, and drawdown is driven almost entirely by those two
    properties. A momentum strategy's real losing streaks come from
    consecutive correlated losses; IID resampling scatters those losses across
    the path and produces drawdown distributions that are far too shallow.

    Block resampling preserves local dependence, so simulated tails resemble
    the tails the strategy would actually experience. `method="iid"` is kept
    for comparison, and the gap between the two is itself diagnostic: a large
    gap means the strategy's risk is dominated by serial dependence.
    """

    def run(
        self,
        equity_curve: pd.Series,
        n_simulations: int = 1000,
        horizon_1yr: int = 252,
        horizon_3yr: int = 756,
        risk_of_ruin_threshold: float = 0.50,
        save_paths: bool = False,
        random_seed: int = 42,
        method: str = "block",
        mean_block_length: Optional[float] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        equity_curve : pd.Series, portfolio equity curve (levels, not returns).
        n_simulations : int
        horizon_1yr : int, trading days in 1 year (252).
        horizon_3yr : int, trading days in 3 years (756).
        risk_of_ruin_threshold : float
            Define "ruin" as equity falling below this fraction of initial.
        save_paths : bool, if True include all simulated paths in result.
        random_seed : int
        method : {"block", "iid"}
            "block" preserves autocorrelation and is the correct default.
        mean_block_length : float, optional
            Expected block length. Defaults to T**(1/3).

        Returns
        -------
        MonteCarloResult
        """
        rng = np.random.default_rng(random_seed)
        daily_returns = equity_curve.pct_change().dropna().values
        n_obs = len(daily_returns)

        if n_obs < 30:
            raise ValueError(
                f"Equity curve too short for Monte Carlo ({n_obs} returns). "
                "Need at least 30 observations."
            )

        initial = float(equity_curve.iloc[0])
        horizon = max(horizon_1yr, horizon_3yr)

        if method == "block":
            sampled_returns = stationary_bootstrap(
                daily_returns,
                n_simulations=n_simulations,
                horizon=horizon,
                mean_block_length=mean_block_length,
                random_seed=random_seed,
            )
        elif method == "iid":
            indices = rng.integers(0, n_obs, size=(n_simulations, horizon))
            sampled_returns = daily_returns[indices]
        else:
            raise ValueError(f"method must be 'block' or 'iid', got {method!r}")

        # Compound into equity paths
        paths = initial * np.cumprod(1 + sampled_returns, axis=1)  # (n_sim, horizon)

        final_1yr = paths[:, horizon_1yr - 1] if horizon_1yr <= paths.shape[1] else paths[:, -1]
        final_3yr = paths[:, horizon_3yr - 1] if horizon_3yr <= paths.shape[1] else paths[:, -1]

        # P(loss) at 1yr and 3yr
        prob_loss_1yr = float(np.mean(final_1yr < initial))
        prob_loss_3yr = float(np.mean(final_3yr < initial))

        # Risk of ruin: P(equity ever drops below threshold × initial)
        ruin_level = initial * risk_of_ruin_threshold
        ruin_count = int(np.sum(np.any(paths < ruin_level, axis=1)))
        ror_pct = ruin_count / n_simulations * 100.0

        # Annual VaR at 95%: worst 5% of 1-year outcomes
        annual_returns = (final_1yr - initial) / initial
        var_95 = float(np.percentile(annual_returns, 5))  # 5th percentile (loss)

        # Final equity percentiles at 3yr horizon
        final_values = paths[:, -1]
        pctile_labels = ["p5", "p25", "p50", "p75", "p95"]
        pctiles_vals  = np.percentile(final_values, [5, 25, 50, 75, 95])
        percentiles = dict(zip(pctile_labels, [round(float(v), 2) for v in pctiles_vals]))

        # Confidence intervals for annualized return (3yr)
        ann_rets_3yr = (final_3yr / initial) ** (1 / 3) - 1
        ci_95 = (float(np.percentile(ann_rets_3yr, 2.5)), float(np.percentile(ann_rets_3yr, 97.5)))
        ci_80 = (float(np.percentile(ann_rets_3yr, 10)), float(np.percentile(ann_rets_3yr, 90)))
        ci_50 = (float(np.percentile(ann_rets_3yr, 25)), float(np.percentile(ann_rets_3yr, 75)))

        result = MonteCarloResult(
            n_simulations=n_simulations,
            percentiles=percentiles,
            risk_of_ruin_pct=round(ror_pct, 3),
            prob_loss_at_1yr=round(prob_loss_1yr, 4),
            prob_loss_at_3yr=round(prob_loss_3yr, 4),
            var_95_annual=round(var_95, 6),
            confidence_intervals={
                "95": ci_95,
                "80": ci_80,
                "50": ci_50,
            },
            simulated_paths=paths if save_paths else None,
        )
        logger.info(
            "Monte Carlo (%d sims): P(loss@1yr)=%.1f%% P(loss@3yr)=%.1f%% "
            "RoR=%.2f%% VaR95=%.1f%%",
            n_simulations,
            prob_loss_1yr * 100,
            prob_loss_3yr * 100,
            ror_pct,
            var_95 * 100,
        )
        return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _cartesian(param_grid: Dict[str, List]) -> List[Dict]:
    """Return Cartesian product of parameter grid as list of dicts."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    from itertools import product
    combos = []
    for vals in product(*[param_grid[k] for k in keys]):
        combos.append(dict(zip(keys, vals)))
    return combos
