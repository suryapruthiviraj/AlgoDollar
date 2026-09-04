"""Core risk engine: portfolio risk calculation and position sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.exceptions import RiskEngineError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Normal-distribution constants (exact to double precision).                   #
#      Z_95   = scipy.stats.norm.ppf(0.95)                                      #
#      PHI_Z95 = scipy.stats.norm.pdf(Z_95)                                     #
#  Expected Shortfall for a normal distribution is                              #
#      ES_alpha = sigma * phi(z_alpha) / (1 - alpha)                            #
#  i.e. it scales SIGMA, *not* VaR. Multiplying VaR by phi(z)/(1-alpha) would   #
#  double-count the z-score and overstate ES by a factor of z (~64%).           #
# --------------------------------------------------------------------------- #
Z_95: float = 1.6448536269514722
PHI_Z95: float = 0.10313564037537139
ES_SIGMA_MULTIPLIER_95: float = PHI_Z95 / 0.05          # ≈ 2.0627128
ES_OVER_VAR_95: float = ES_SIGMA_MULTIPLIER_95 / Z_95   # ≈ 1.2540

# Exposure below this (in ₹) is treated as "flat" — guards against exact float
# equality tests on a hedged book that nets to ~1e-9 rather than exactly 0.
_EXPOSURE_EPS: float = 1e-6


class MissingPriceError(ValueError):
    """Raised when a position's price is absent, non-finite or non-positive.

    A missing price previously defaulted to 0.0, which made the position
    invisible to risk (weight 0 despite real exposure) and, if *every* price
    was missing, reported zero risk on a live book. Risk must never be
    silently understated, so this is a hard error.
    """


@dataclass
class PortfolioRisk:
    portfolio_variance: float
    portfolio_vol: float                    # annualised standard deviation
    portfolio_beta: float
    var_95: float                           # 1-day VaR at 95% confidence (₹)
    cvar_95: float                          # Expected Shortfall at 95% (₹)
    # symbol -> component risk contribution in ANNUALISED VOL UNITS.
    # These sum to `portfolio_vol` (Euler decomposition), NOT to 1.0.
    risk_contributions: dict[str, float]
    # symbol -> component risk as a FRACTION of total risk. Sums to 1.0.
    risk_contribution_pct: dict[str, float]
    # sector -> signed exposure as a fraction of GROSS exposure.
    sector_concentrations: dict[str, float]
    gross_exposure: float = 0.0             # sum(|position value|) in ₹
    net_exposure: float = 0.0               # sum(position value) in ₹

    @property
    def marginal_risk(self) -> dict[str, float]:
        """Deprecated alias for :attr:`risk_contributions`.

        Kept for backwards compatibility. Note that these are annualised-vol
        units summing to ``portfolio_vol`` — use ``risk_contribution_pct`` if
        you want percentages that sum to 1.0.
        """
        return self.risk_contributions


@dataclass
class StrategyHealth:
    name: str
    status: str    # ACTIVE | REDUCED | PAUSED | STOPPED


class RiskEngine:
    """
    Stateless risk calculation engine.

    "Stateless" means it holds no portfolio of its own — every calculation is
    given the positions and prices it needs. :meth:`approve_trade` is the one
    exception in shape only: strategies call it with just
    ``(symbol, size, sleeve)``, so the portfolio context it judges against must
    be supplied first via :meth:`set_portfolio_context`.
    """

    def __init__(self) -> None:
        self._ctx: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    #  Per-trade approval (the gate strategies call)                       #
    # ------------------------------------------------------------------ #

    def set_portfolio_context(
        self,
        *,
        portfolio_value: float,
        available_cash: float,
        positions: Optional[list[dict]] = None,
        max_positions: Optional[int] = None,
    ) -> None:
        """
        Supply the book that :meth:`approve_trade` judges against.

        Must be called before each sizing pass. Stale context is worse than
        none: approving a trade against last hour's portfolio value is how a
        concentration limit silently stops binding.
        """
        if portfolio_value is None or portfolio_value <= 0:
            raise ValueError(
                "portfolio_value must be > 0 to evaluate a concentration limit"
            )
        self._ctx = {
            "portfolio_value": float(portfolio_value),
            "available_cash": float(available_cash),
            "positions": list(positions or []),
            "max_positions": (
                int(max_positions) if max_positions is not None
                else int(settings.max_positions)
            ),
        }

    def clear_portfolio_context(self) -> None:
        self._ctx = None

    def approve_trade(self, symbol: str, size: float, sleeve: str) -> bool:
        """
        Approve one proposed trade of ``size`` rupees in ``symbol``.

        FAILS CLOSED. Called from ``BaseStrategy._risk_engine_approves``, which
        treats a raise as a block — so every path out of here that is not a
        clear "yes" stops the trade.

        This method did not exist. Strategies have always called it, and
        ``_risk_engine_approves`` raises RiskEngineError when it is absent, so
        every strategy sizing call against a real RiskEngine blocked. That
        failed safe, but it also meant the risk engine and the strategies were
        never actually connected.

        The checks are the configured limits, not new policy:
          * single-stock concentration  (settings.max_single_stock_pct)
          * intraday capital cap        (settings.max_intraday_capital_pct)
          * position count             (settings.max_positions)
          * available cash
        """
        if self._ctx is None:
            raise RiskEngineError(
                f"approve_trade({symbol!r}) called with no portfolio context. "
                "Call set_portfolio_context() first — approving a trade without "
                "knowing the book means the concentration limit is not applied."
            )
        if not size or size <= 0:
            return False

        ctx = self._ctx
        pv = ctx["portfolio_value"]
        sleeve_l = (sleeve or "").lower()

        if size > ctx["available_cash"]:
            logger.info(
                "RISK BLOCK %s (%s): size Rs %.2f exceeds available cash Rs %.2f",
                symbol, sleeve, size, ctx["available_cash"],
            )
            return False

        cap = pv * float(settings.max_single_stock_pct)
        # Existing exposure counts: a limit applied per ORDER rather than per
        # POSITION is trivially defeated by splitting one order into several.
        held = sum(
            float(p.get("quantity", 0) or 0) * float(p.get("average_price", 0) or 0)
            for p in ctx["positions"]
            if str(p.get("symbol", "")).upper() == str(symbol).upper()
        )
        if held + size > cap:
            logger.info(
                "RISK BLOCK %s (%s): Rs %.2f existing + Rs %.2f proposed exceeds "
                "the %.1f%% single-stock cap (Rs %.2f)",
                symbol, sleeve, held, size, settings.max_single_stock_pct * 100, cap,
            )
            return False

        if sleeve_l == "intraday":
            intraday_cap = pv * float(settings.max_intraday_capital_pct)
            if size > intraday_cap:
                logger.info(
                    "RISK BLOCK %s (intraday): Rs %.2f exceeds the %.1f%% "
                    "intraday capital cap (Rs %.2f)",
                    symbol, size, settings.max_intraday_capital_pct * 100, intraday_cap,
                )
                return False

        open_symbols = {
            str(p.get("symbol", "")).upper()
            for p in ctx["positions"]
            if float(p.get("quantity", 0) or 0) != 0
        }
        if str(symbol).upper() not in open_symbols:
            if len(open_symbols) >= ctx["max_positions"]:
                logger.info(
                    "RISK BLOCK %s (%s): %d open positions is at the limit of %d",
                    symbol, sleeve, len(open_symbols), ctx["max_positions"],
                )
                return False

        return True

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
        prices          : symbol -> last price (must be present, finite and > 0
                          for every position, otherwise MissingPriceError)
        covariance_matrix : annualised covariance matrix (N x N), optional
        market_beta     : symbol -> beta vs Nifty, optional
        sector_map      : symbol -> sector name, optional
        portfolio_value : total portfolio value; used to scale VaR/CVaR only
                          when gross exposure is ~0 (i.e. a flat book)

        Notes
        -----
        Weights and ₹ risk figures are computed against **gross** exposure
        ``sum(|quantity * price|)``, not net. A market-neutral long/short book
        nets to ~0, and dividing by that produced absurd volatilities
        (e.g. 1.26e6 annualised for a fully hedged pair).

        Raises
        ------
        MissingPriceError
            If any position's price is absent, non-finite or non-positive.
        """
        if not positions:
            return PortfolioRisk(
                portfolio_variance=0.0,
                portfolio_vol=0.0,
                portfolio_beta=1.0,
                var_95=0.0,
                cvar_95=0.0,
                risk_contributions={},
                risk_contribution_pct={},
                sector_concentrations={},
                gross_exposure=0.0,
                net_exposure=0.0,
            )

        symbols = [p["symbol"] for p in positions]
        quantities = np.array([p.get("quantity", 0) for p in positions], dtype=float)

        # --- price validation: never default a missing price to zero ---------
        missing = [s for s in symbols if s not in prices]
        if missing:
            raise MissingPriceError(
                f"No price supplied for {sorted(set(missing))}; refusing to "
                f"compute risk with zero-weighted live exposure."
            )
        price_arr = np.array([prices[s] for s in symbols], dtype=float)
        bad = [
            s for s, px in zip(symbols, price_arr)
            if not np.isfinite(px) or px <= 0.0
        ]
        if bad:
            raise MissingPriceError(
                f"Invalid (non-finite or non-positive) price for "
                f"{sorted(set(bad))}; refusing to compute risk."
            )
        if not np.all(np.isfinite(quantities)):
            raise ValueError("Position quantities must be finite.")

        values = quantities * price_arr
        net_exposure = float(values.sum())
        gross_exposure = float(np.abs(values).sum())

        # Gross exposure is the correct risk denominator: it is invariant to
        # long/short netting and is only ~0 when the book is genuinely flat.
        if gross_exposure <= _EXPOSURE_EPS:
            logger.warning(
                "Gross exposure is ~0 (₹%.2e); falling back to portfolio_value "
                "for risk scaling.", gross_exposure,
            )
            risk_base = max(portfolio_value, 1.0)
        else:
            risk_base = gross_exposure

        weights = values / risk_base  # signed weights; sum(|w|) == 1 when gross>0

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
        daily_sigma_rupees = daily_vol * risk_base
        # 1-day 95% VaR = sigma * z
        var_95 = float(daily_sigma_rupees * Z_95)
        # 1-day 95% Expected Shortfall = SIGMA * phi(z) / (1 - conf).
        # The multiplier applies to sigma, not to VaR (see module constants).
        cvar_95 = float(daily_sigma_rupees * ES_SIGMA_MULTIPLIER_95)

        # --- risk contributions per position ---------------------------
        # Euler decomposition: RC_i = w_i * (Sigma @ w)_i / portfolio_vol,
        # which sums to portfolio_vol (annualised vol units, NOT percentages).
        if portfolio_vol > 0:
            sigma_w = cov @ weights
            marginal = sigma_w / portfolio_vol   # dSigma_p / dw_i
            component_risk = weights * marginal  # annualised vol units
            component_pct = component_risk / portfolio_vol  # sums to 1.0
        else:
            component_risk = np.zeros(n)
            component_pct = np.zeros(n)
        risk_contributions = {s: float(component_risk[i]) for i, s in enumerate(symbols)}
        risk_contribution_pct = {s: float(component_pct[i]) for i, s in enumerate(symbols)}

        # --- sector concentrations -------------------------------------
        # Signed sector exposure as a fraction of GROSS exposure.
        sector_values: dict[str, float] = {}
        if sector_map:
            for sym, val in zip(symbols, values):
                sector = sector_map.get(sym, "Unknown")
                sector_values[sector] = sector_values.get(sector, 0.0) + val
        sector_concentrations = {s: v / risk_base for s, v in sector_values.items()}

        return PortfolioRisk(
            portfolio_variance=portfolio_variance,
            portfolio_vol=portfolio_vol,
            portfolio_beta=portfolio_beta,
            var_95=var_95,
            cvar_95=cvar_95,
            risk_contributions=risk_contributions,
            risk_contribution_pct=risk_contribution_pct,
            sector_concentrations=sector_concentrations,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
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
        # NaN fails every comparison, so guard it explicitly before the
        # `<= 0` test (otherwise NaN slips through and yields a NaN size).
        if not np.isfinite(signal_price) or not np.isfinite(stop_price) or not np.isfinite(capital):
            logger.warning("Non-finite input to position sizing; returning 0.")
            return 0.0
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
