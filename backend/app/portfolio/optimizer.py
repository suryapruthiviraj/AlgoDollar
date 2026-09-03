"""
optimizer.py — Portfolio optimization routines for AlgoDollar.

Methods
-------
mean_variance_optimize   — maximize Sharpe (scipy + LedoitWolf covariance)
minimum_volatility_portfolio — minimize portfolio variance
risk_parity_portfolio    — equal risk contribution
volatility_target_size   — single-asset vol-targeted weight
efficient_frontier       — generate frontier points for display

All methods are pure functions (no side effects) and use only information
available at the time of the call (no future data required — the caller
must supply already-computed expected_returns and cov_matrix).

Numerical stability notes
-------------------------
- LedoitWolf shrinkage is applied to raw sample covariance to stabilize
  the matrix for small N or short history.
- If optimization fails (rare with LedoitWolf), we fall back to equal weight
  with vol scaling.
- For N < 5, we skip MV optimization and default directly to equal weight.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

# Risk-free rate used for Sharpe calculation (India 10-year G-Sec, ~6.5%)
_RF_DAILY = 0.065 / 252


# ---------------------------------------------------------------------------
# Covariance helpers
# ---------------------------------------------------------------------------

def _ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """
    Estimate covariance matrix using LedoitWolf shrinkage.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N), daily log returns.

    Returns
    -------
    np.ndarray, shape (N, N), annualized covariance.
    """
    lw = LedoitWolf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lw.fit(returns)
    # Returns is daily; annualize
    return lw.covariance_ * 252


def _is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def _make_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Nudge a near-PSD matrix to strict PD by adding a small diagonal term.
    """
    eigvals = np.linalg.eigvalsh(matrix)
    if eigvals.min() < epsilon:
        shift = max(epsilon - eigvals.min(), epsilon)
        matrix = matrix + shift * np.eye(matrix.shape[0])
        logger.debug("Made covariance PSD (shift=%.2e).", shift)
    return matrix


# ---------------------------------------------------------------------------
# Mean-variance optimization
# ---------------------------------------------------------------------------

def mean_variance_optimize(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    max_weight: float = 0.30,
    min_weight: float = 0.0,
    long_only: bool = True,
    rf_daily: float = _RF_DAILY,
) -> np.ndarray:
    """
    Maximize Sharpe ratio (mean-variance optimization).

    Parameters
    ----------
    expected_returns : np.ndarray, shape (N,), annualized expected returns.
    cov_matrix : np.ndarray, shape (N, N), annualized covariance.
    max_weight : float, maximum weight per asset.
    min_weight : float, minimum weight per asset (0 for long-only).
    long_only : bool, if True enforce w >= 0.
    rf_daily : float, daily risk-free rate.

    Returns
    -------
    np.ndarray, shape (N,), portfolio weights summing to 1.

    Fallback: equal weight with vol scaling if optimization fails.
    """
    n = len(expected_returns)
    if n < 2:
        return np.array([1.0]) if n == 1 else np.array([])
    if n < 5:
        logger.info("N=%d < 5; using equal weight with vol scaling.", n)
        return _equal_weight_vol_scaled(cov_matrix)

    cov_matrix = _make_psd(cov_matrix)
    rf_annual = rf_daily * 252

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(expected_returns @ w)
        port_vol = float(np.sqrt(w @ cov_matrix @ w))
        if port_vol < 1e-10:
            return 1e6
        return -(port_ret - rf_annual) / port_vol

    # Gradient for faster convergence
    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        port_ret = float(expected_returns @ w)
        port_var = float(w @ cov_matrix @ w)
        port_vol = np.sqrt(port_var)
        if port_vol < 1e-10:
            return np.zeros(n)
        sharpe = (port_ret - rf_annual) / port_vol
        d_ret = expected_returns
        d_vol = (cov_matrix @ w) / port_vol
        return -(d_ret / port_vol - sharpe * d_vol / port_vol)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    lb = 0.0 if long_only else -max_weight
    bounds = [(max(lb, min_weight), max_weight)] * n
    w0 = np.ones(n) / n  # start at equal weight

    try:
        result = minimize(
            neg_sharpe,
            w0,
            jac=neg_sharpe_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if result.success and np.isfinite(result.fun):
            w = np.array(result.x)
            w = np.clip(w, 0, max_weight)
            w /= w.sum()
            logger.debug("MV optimization succeeded (neg_sharpe=%.4f).", result.fun)
            return w
        else:
            logger.warning("MV optimization did not converge: %s. Using equal weight.", result.message)
    except Exception as exc:
        logger.warning("MV optimization exception: %s. Using equal weight.", exc)

    return _equal_weight_vol_scaled(cov_matrix)


# ---------------------------------------------------------------------------
# Minimum volatility portfolio
# ---------------------------------------------------------------------------

def minimum_volatility_portfolio(
    cov_matrix: np.ndarray,
    max_weight: float = 0.30,
    min_weight: float = 0.0,
) -> np.ndarray:
    """
    Find the portfolio with minimum variance.

    Parameters
    ----------
    cov_matrix : np.ndarray, shape (N, N), annualized covariance.
    max_weight : float
    min_weight : float

    Returns
    -------
    np.ndarray, shape (N,), weights.
    """
    n = cov_matrix.shape[0]
    if n < 2:
        return np.array([1.0]) if n == 1 else np.array([])

    cov_matrix = _make_psd(cov_matrix)

    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ cov_matrix @ w)

    def var_grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * cov_matrix @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.ones(n) / n

    try:
        result = minimize(
            portfolio_variance,
            w0,
            jac=var_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if result.success:
            w = np.clip(result.x, 0, max_weight)
            w /= w.sum()
            return w
    except Exception as exc:
        logger.warning("Min-vol optimization failed: %s.", exc)

    return _equal_weight_vol_scaled(cov_matrix)


# ---------------------------------------------------------------------------
# Risk parity portfolio
# ---------------------------------------------------------------------------

def risk_parity_portfolio(
    cov_matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Equal risk contribution (ERC / risk parity) portfolio.

    Solved via iterative algorithm (Maillard, Roncalli & Teïletche 2010).

    The objective is: w_i * (Sigma @ w)_i = 1/N * portfolio_variance
    for all i, subject to sum(w) = 1.

    Parameters
    ----------
    cov_matrix : np.ndarray, shape (N, N), annualized covariance.
    max_iter : int
    tol : float, convergence tolerance.

    Returns
    -------
    np.ndarray, shape (N,), weights.
    """
    n = cov_matrix.shape[0]
    if n < 2:
        return np.array([1.0]) if n == 1 else np.array([])

    cov_matrix = _make_psd(cov_matrix)

    # Cyclical coordinate descent (CCD)
    w = np.ones(n) / n
    for iteration in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            # Analytical update: minimize risk contribution of asset i
            # w_i = sqrt(w_{-i}' Sigma_{-i,-i} w_{-i} / Sigma_{ii})
            # Simplified closed-form via partial derivative
            cov_i = cov_matrix[i, :]
            sigma_ii = cov_matrix[i, i]
            if sigma_ii < 1e-14:
                continue
            a = sigma_ii
            b = float(cov_i @ w) - w[i] * sigma_ii  # Sigma_{i, -i} @ w_{-i}
            c = -1.0 / n  # target risk contribution * portfolio_variance ≈ normalized
            # Quadratic: a * w_i^2 + b * w_i + c = 0
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                continue
            w_i_new = (-b + np.sqrt(discriminant)) / (2 * a)
            w[i] = max(w_i_new, 1e-8)

        # Normalize
        w = np.maximum(w, 0)
        w /= w.sum()

        # Convergence check
        if np.max(np.abs(w - w_prev)) < tol:
            logger.debug("Risk parity converged at iteration %d.", iteration)
            break
    else:
        logger.warning("Risk parity did not converge in %d iterations.", max_iter)

    return w


# ---------------------------------------------------------------------------
# Single-asset vol-targeting
# ---------------------------------------------------------------------------

def volatility_target_size(
    signal_vol: float,
    target_vol: float = 0.15,
    max_weight: float = 0.50,
    min_weight: float = 0.01,
) -> float:
    """
    Compute the portfolio weight for a single asset given its volatility.

    weight = target_vol / signal_vol

    Parameters
    ----------
    signal_vol : float, annualized realized volatility of the asset.
    target_vol : float, target volatility contribution (annualized).
    max_weight : float, cap.
    min_weight : float, floor.

    Returns
    -------
    float, weight in [min_weight, max_weight].
    """
    if signal_vol <= 0:
        logger.warning("signal_vol <= 0; returning min_weight.")
        return min_weight
    weight = target_vol / signal_vol
    return float(np.clip(weight, min_weight, max_weight))


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

def efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 50,
    max_weight: float = 0.30,
) -> List[Dict]:
    """
    Generate efficient frontier points.

    Parameters
    ----------
    expected_returns : np.ndarray, shape (N,), annualized.
    cov_matrix : np.ndarray, shape (N, N), annualized.
    n_points : int, number of frontier points.
    max_weight : float

    Returns
    -------
    list[dict] with keys: target_return, portfolio_vol, weights, sharpe.
    """
    n = len(expected_returns)
    if n < 2:
        return []

    cov_matrix = _make_psd(cov_matrix)
    min_ret = float(expected_returns.min())
    max_ret = float(expected_returns.max())
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    for target in target_returns:
        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ cov_matrix @ w)

        def var_grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * cov_matrix @ w

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: float(expected_returns @ w) - target},
        ]
        bounds = [(0.0, max_weight)] * n
        w0 = np.ones(n) / n
        try:
            result = minimize(
                portfolio_variance,
                w0,
                jac=var_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            if result.success:
                w = np.clip(result.x, 0, max_weight)
                w /= w.sum()
                port_vol = float(np.sqrt(w @ cov_matrix @ w))
                realized_ret = float(expected_returns @ w)
                sharpe = (realized_ret - 0.065) / port_vol if port_vol > 0 else 0.0
                frontier.append({
                    "target_return": float(target),
                    "portfolio_vol": port_vol,
                    "weights": w.tolist(),
                    "sharpe": sharpe,
                })
        except Exception:
            continue

    return frontier


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _equal_weight_vol_scaled(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Fallback: equal weight with inverse-volatility scaling.

    Each asset gets weight proportional to 1/sigma_i, normalized to sum=1.
    """
    n = cov_matrix.shape[0]
    vols = np.sqrt(np.diag(cov_matrix))
    vols = np.where(vols < 1e-10, 1.0, vols)
    inv_vol = 1.0 / vols
    w = inv_vol / inv_vol.sum()
    return w


def compute_ledoit_wolf_covariance(
    prices_df: pd.DataFrame,
    window: Optional[int] = None,
) -> np.ndarray:
    """
    Compute LedoitWolf-shrunk annualized covariance matrix from price DataFrame.

    Parameters
    ----------
    prices_df : pd.DataFrame, shape (T, N), close prices.
    window : int or None, use only the last `window` rows.

    Returns
    -------
    np.ndarray, shape (N, N), annualized covariance.
    """
    if window is not None:
        prices_df = prices_df.tail(window + 1)

    log_rets = np.log(prices_df / prices_df.shift(1)).dropna()
    if log_rets.shape[0] < log_rets.shape[1]:
        logger.warning(
            "T=%d < N=%d; covariance estimate will be unreliable. "
            "Using LedoitWolf shrinkage.",
            log_rets.shape[0],
            log_rets.shape[1],
        )
    return _ledoit_wolf_cov(log_rets.values)
