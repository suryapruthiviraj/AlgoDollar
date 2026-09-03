"""Core risk engine: portfolio risk calculation and position sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRisk:
    portfolio_variance: float
    portfolio_vol: float                    # annualised standard deviation
    portfolio_beta: float
    var_95: float                           # 1-day VaR at 95% confidence (₹)
    cvar_95: float                          # Expected Shortfall at 95% (₹)
    marginal_risk: dict[str, float]         # symbol -> marginal risk contribution
    sector_concentrations: dict[str, float]  # sector -> weight fraction


@dataclass
class StrategyHealth:
    name: str
    status: str    # ACTIVE | REDUCED | PAUSED | STOPPED


class RiskEngine:
    """Stateless risk calculation engine."""

    # ------------------------------------------------------------------ #
    #  Portfolio risk                                                      #
    # ------------------------------------------------------------------ #

    def calculate_portfolio_risk(
        self,
        positions: list[dict],
        prices: dict[str, float],
        covariance_matrix: Optional[np.ndarray] = None,
        market_beta: Optional[dict[str, float]] = None,
        sector_map: Optional[dict[str, str]] = None,
        portfolio_value: float = 0.0,
        trading_days_per_year: int = 252,
    ) -> PortfolioRisk:
        """
        Compute full portfolio risk metrics.

        Parameters
        ----------
        positions       : list of dicts with 'symbol' and 'quantity'
        prices          : symbol -> last price
        covariance_matrix : annualised covariance matrix (N x N), optional
        market_beta     : symbol -> beta vs Nifty, optional
        sector_map      : symbol -> sector name, optional
        portfolio_value : total portfolio value for VaR scaling
        """
        if not positions:
            return PortfolioRisk(
                portfolio_variance=0.0,
                portfolio_vol=0.0,
                portfolio_beta=1.0,
                var_95=0.0,
                cvar_95=0.0,
                marginal_risk={},
                sector_concentrations={},
            )

        symbols = [p["symbol"] for p in positions]
        quantities = np.array([p.get("quantity", 0) for p in positions], dtype=float)
        price_arr = np.array([prices.get(s, 0.0) for s in symbols], dtype=float)

        values = quantities * price_arr
        total_value = values.sum()
        if total_value == 0:
            total_value = max(portfolio_value, 1.0)

        weights = values / total_value  # fractional weights

        # --- covariance and portfolio variance -------------------------
        n = len(symbols)
        if covariance_matrix is not None and covariance_matrix.shape == (n, n):
            cov = covariance_matrix
        else:
            # Fallback: identity (uncorrelated) with flat 20% vol each
            annual_vol = 0.20
            daily_var = (annual_vol / np.sqrt(trading_days_per_year)) ** 2
            cov = np.diag(np.full(n, daily_var * trading_days_per_year))

        portfolio_variance = float(weights @ cov @ weights)
        portfolio_vol = float(np.sqrt(portfolio_variance))  # annualised

        # --- beta ------------------------------------------------------
        if market_beta:
            betas = np.array([market_beta.get(s, 1.0) for s in symbols])
        else:
            betas = np.ones(n)
        portfolio_beta = float(weights @ betas)

        # --- VaR / CVaR (parametric, normal) ---------------------------
        daily_vol = portfolio_vol / np.sqrt(trading_days_per_year)
        # 1-day 95% VaR
        z_95 = 1.6449
        var_95 = float(daily_vol * z_95 * total_value)
        # CVaR ≈ var * phi(z) / (1 - conf) for normal distribution
        # phi(1.6449) ≈ 0.1031, (1-0.95) = 0.05
        cvar_95 = float(var_95 * 0.1031 / 0.05)

        # --- marginal risk per position --------------------------------
        # marginal_risk_i = (Sigma @ w)_i / portfolio_vol
        if portfolio_vol > 0:
            sigma_w = cov @ weights
            marginal = sigma_w / portfolio_vol
            component_risk = weights * marginal  # % contribution
        else:
            component_risk = np.zeros(n)
        marginal_risk = {s: float(component_risk[i]) for i, s in enumerate(symbols)}

        # --- sector concentrations -------------------------------------
        sector_values: dict[str, float] = {}
        if sector_map:
            for sym, val in zip(symbols, values):
                sector = sector_map.get(sym, "Unknown")
                sector_values[sector] = sector_values.get(sector, 0.0) + val
        sector_concentrations = {s: v / total_value for s, v in sector_values.items()}

        return PortfolioRisk(
            portfolio_variance=portfolio_variance,
            portfolio_vol=portfolio_vol,
            portfolio_beta=portfolio_beta,
            var_95=var_95,
            cvar_95=cvar_95,
            marginal_risk=marginal_risk,
            sector_concentrations=sector_concentrations,
        )

    # ------------------------------------------------------------------ #
    #  Position sizing                                                     #
    # ------------------------------------------------------------------ #

    def scale_position_for_risk(
        self,
        signal_price: float,
        stop_price: float,
        capital: float,
        max_risk_per_trade_pct: float = 0.01,
        max_single_stock_pct: float = 0.10,
        adv_shares: Optional[int] = None,     # average daily volume in shares
        adv_cap_pct: float = 0.05,            # max 5% of ADV
    ) -> float:
        """
        Calculate position size in ₹ using fixed-fractional risk.

        position_size = risk_per_trade / stop_distance_pct

        Returns size in ₹ (may be zero if stop distance is negligible).
        """
        if signal_price <= 0 or stop_price <= 0:
            return 0.0

        stop_distance_pct = abs(signal_price - stop_price) / signal_price
        if stop_distance_pct < 1e-6:
            logger.warning("Stop distance is negligible; skipping position sizing.")
            return 0.0

        risk_amount = capital * max_risk_per_trade_pct
        position_size = risk_amount / stop_distance_pct  # in ₹

        # Cap at single-stock limit
        max_by_concentration = capital * max_single_stock_pct
        position_size = min(position_size, max_by_concentration)

        # Cap at liquidity constraint (5% of ADV)
        if adv_shares and adv_shares > 0:
            max_by_liquidity = adv_shares * adv_cap_pct * signal_price
            position_size = min(position_size, max_by_liquidity)

        logger.debug(
            "Position sizing: stop_dist=%.2f%% risk=₹%.0f size=₹%.0f",
            stop_distance_pct * 100, risk_amount, position_size,
        )
        return round(position_size, 2)

    # ------------------------------------------------------------------ #
    #  Dynamic risk scaling                                                #
    # ------------------------------------------------------------------ #

    def dynamic_risk_scaling(
        self,
        strategy_health: StrategyHealth,
        regime: str,                # MarketRegime.value
        portfolio_drawdown: float,  # fraction, e.g. 0.12 = 12%
    ) -> float:
        """
        Return a scale factor in [0.0, 1.0] to multiply base position sizes.

        Rules applied in order (smallest wins):
        - Strategy PAUSED  → 0.0
        - Strategy REDUCED → 0.5
        - Drawdown > 15%   → 0.25
        - Drawdown > 10%   → 0.5
        - Regime PANIC     → 0.5 (if not already lower)
        - Regime STRONG_BEAR → 0.5
        - Default          → 1.0
        """
        factor = 1.0

        # Strategy-level overrides (highest priority)
        if strategy_health.status == "PAUSED" or strategy_health.status == "STOPPED":
            return 0.0
        if strategy_health.status == "REDUCED":
            factor = min(factor, 0.5)

        # Portfolio drawdown
        if portfolio_drawdown >= 0.15:
            factor = min(factor, 0.25)
        elif portfolio_drawdown >= 0.10:
            factor = min(factor, 0.5)

        # Market regime
        bear_regimes = {"STRONG_BEAR", "PANIC"}
        if regime in bear_regimes:
            factor = min(factor, 0.5)

        logger.debug(
            "Dynamic risk scale: strategy=%s regime=%s drawdown=%.1f%% → factor=%.2f",
            strategy_health.status, regime, portfolio_drawdown * 100, factor,
        )
        return factor

    # ------------------------------------------------------------------ #
    #  Covariance estimation                                               #
    # ------------------------------------------------------------------ #

    def estimate_covariance(
        self,
        returns_df: pd.DataFrame,
        method: str = "ledoit_wolf",
        trading_days_per_year: int = 252,
    ) -> np.ndarray:
        """
        Estimate an annualised covariance matrix from daily returns.

        Parameters
        ----------
        returns_df : DataFrame where each column is a symbol and each row a daily return
        method     : "ledoit_wolf" (shrinkage) or "sample"

        Returns
        -------
        np.ndarray of shape (n_symbols, n_symbols) — annualised covariance
        """
        returns = returns_df.dropna()
        if returns.empty:
            n = returns_df.shape[1]
            return np.eye(n) * (0.20 / np.sqrt(trading_days_per_year)) ** 2 * trading_days_per_year

        if method == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf  # type: ignore[import]
                lw = LedoitWolf(assume_centered=False)
                lw.fit(returns.values)
                daily_cov = lw.covariance_
            except ImportError:
                logger.warning("sklearn not installed; falling back to sample covariance.")
                daily_cov = returns.cov().values
        else:
            daily_cov = returns.cov().values

        # Annualise
        annualised_cov = daily_cov * trading_days_per_year
        return annualised_cov

    # ------------------------------------------------------------------ #
    #  Portfolio VaR                                                       #
    # ------------------------------------------------------------------ #

    def calculate_portfolio_var(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_value: float = 1.0,
        confidence: float = 0.95,
        trading_days_per_year: int = 252,
    ) -> float:
        """
        Parametric normal VaR for a given portfolio.

        Returns the 1-day VaR in ₹ (positive = potential loss).
        """
        import scipy.stats as stats  # type: ignore[import]

        annual_portfolio_var = float(weights @ cov_matrix @ weights)
        daily_portfolio_vol = np.sqrt(annual_portfolio_var / trading_days_per_year)
        z = float(stats.norm.ppf(confidence))
        var = daily_portfolio_vol * z * portfolio_value
        return round(var, 2)
