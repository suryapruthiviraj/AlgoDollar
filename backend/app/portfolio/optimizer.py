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
- If optimization fails (rare with LedoitWolf), we fall back to inverse-vol
  weighting *projected onto the weight caps* and log a warning. The fallback
  is never silent and never violates max_weight.
- Every N >= 2 is genuinely mean-variance optimized. (Previously N < 5 silently
  discarded `expected_returns` and returned inverse-vol weights, so a 4-name
  sleeve was never optimized and identical weights came back for opposite
  return forecasts.)
- Weight caps are checked for feasibility up front: sum(w) = 1 is impossible
  when n * max_weight < 1, so that raises ValueError rather than quietly
  returning weights that breach the cap.
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


def _check_weight_bounds_feasible(
    n: int,
    max_weight: float,
    min_weight: float,
    caller: str,
) -> None:
    """
    Verify that bounds admit a fully-invested portfolio (sum(w) == 1).

    Raises ValueError when ``n * max_weight < 1`` (cap too tight) or
    ``n * min_weight > 1`` (floor too high). Without this precheck SLSQP
    reports failure and the caller silently returns fallback weights that
    breach the cap (e.g. max_weight=0.10 on n=5 returned 0.1667).
    """
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError(
            f"{caller}: infeasible bounds — {n} assets capped at "
            f"max_weight={max_weight:.4f} can hold at most "
            f"{n * max_weight:.4f} of the portfolio, but weights must sum to "
            f"1.0. Raise max_weight to at least {1.0 / n:.4f}."
        )
    if n * min_weight > 1.0 + 1e-12:
        raise ValueError(
            f"{caller}: infeasible bounds — {n} assets floored at "
            f"min_weight={min_weight:.4f} require at least "
            f"{n * min_weight:.4f} of the portfolio, but weights must sum to "
            f"1.0. Lower min_weight to at most {1.0 / n:.4f}."
        )


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
    min_weight : float, minimum weight per asset. For long_only=False this may
        be negative to permit shorts; leave at 0.0 to let the short bound
        default to -max_weight.
    long_only : bool, if True enforce w >= 0. If False the lower bound becomes
        ``min_weight`` when it is negative, else ``-max_weight`` — and the
        result is *not* clipped at zero, so shorts survive.
    rf_daily : float, daily risk-free rate.

    Returns
    -------
    np.ndarray, shape (N,), portfolio weights summing to 1.

    Raises
    ------
    ValueError
        If the weight bounds cannot produce a fully-invested portfolio
        (see :func:`_check_weight_bounds_feasible`).

    Notes
    -----
    All N >= 2 are optimized. There is no N < 5 shortcut: silently ignoring
    `expected_returns` made a 2- or 4-name sleeve return identical weights for
    opposite return forecasts.

    Fallback: cap-respecting inverse-vol weights (with a warning) if the
    optimizer fails to converge.
    """
    n = len(expected_returns)
    if n < 2:
        return np.array([1.0]) if n == 1 else np.array([])

    expected_returns = np.asarray(expected_returns, dtype=float)
    if not np.all(np.isfinite(expected_returns)):
        raise ValueError("expected_returns contains non-finite values.")
    if cov_matrix.shape != (n, n):
        raise ValueError(
            f"cov_matrix shape {cov_matrix.shape} does not match "
            f"expected_returns length {n}."
        )

    _check_weight_bounds_feasible(n, max_weight, min_weight, "mean_variance_optimize")

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
    # Lower bound. Previously this was `max(lb, min_weight)`, which with the
    # default min_weight=0.0 collapsed to 0.0 and made long_only=False a no-op.
    if long_only:
        lower = max(min_weight, 0.0)
    else:
        lower = min_weight if min_weight < 0.0 else -abs(max_weight)
    bounds = [(lower, max_weight)] * n
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
            # Clip to the ACTUAL bounds (not [0, max]) so shorts survive.
            w = np.clip(np.asarray(result.x, dtype=float), lower, max_weight)
            total = w.sum()
            if abs(total) < 1e-8:
                logger.warning(
                    "MV optimization produced weights summing to %.2e; "
                    "falling back to capped inverse-vol weights.", total,
                )
            else:
                w = w / total
                logger.debug("MV optimization succeeded (neg_sharpe=%.4f).", result.fun)
                return w
        else:
            logger.warning(
                "MV optimization did not converge: %s. Falling back to capped "
                "inverse-vol weights.", result.message,
            )
    except Exception as exc:
        logger.warning(
            "MV optimization exception: %s. Falling back to capped inverse-vol "
            "weights.", exc,
        )

    return _equal_weight_vol_scaled(cov_matrix, max_weight=max_weight)


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
    np.ndarray, shape (N,), weights — always within [min_weight, max_weight].

    Raises
    ------
    ValueError
        If the weight bounds cannot produce a fully-invested portfolio.
    """
    n = cov_matrix.shape[0]
    if n < 2:
        return np.array([1.0]) if n == 1 else np.array([])

    # Feasibility precheck: n * max_weight < 1 makes sum(w)=1 impossible, so
    # SLSQP fails and the old code silently returned uncapped fallback weights.
    _check_weight_bounds_feasible(n, max_weight, min_weight, "minimum_volatility_portfolio")

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
            w = np.clip(result.x, min_weight, max_weight)
            w /= w.sum()
            return w
        logger.warning(
            "Min-vol optimization did not converge: %s. Falling back to "
            "cap-respecting inverse-vol weights.", result.message,
        )
    except Exception as exc:
        logger.warning(
            "Min-vol optimization failed: %s. Falling back to cap-respecting "
            "inverse-vol weights.", exc,
        )

    return _equal_weight_vol_scaled(
        cov_matrix, max_weight=max_weight, min_weight=min_weight
    )


# ---------------------------------------------------------------------------
# Risk parity portfolio
# ---------------------------------------------------------------------------

def risk_contribution_dispersion(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Max relative deviation of risk contributions from the 1/N target.

    RC_i = w_i * (Sigma @ w)_i. For a true ERC portfolio every RC_i is equal,
    so the fractions RC_i / sum(RC) all equal 1/N and this returns ~0.

    Returns
    -------
    float, e.g. 0.0063 == 0.63% worst-case relative error.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    if n == 0:
        return 0.0
    rc = w * (cov_matrix @ w)
    total = rc.sum()
    if abs(total) < 1e-18:
        return 0.0
    frac = rc / total
    target = 1.0 / n
    return float(np.max(np.abs(frac - target)) / target)


def risk_parity_portfolio(
    cov_matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
    dispersion_tol: float = 0.01,
) -> np.ndarray:
    """
    Equal risk contribution (ERC / risk parity) portfolio.

    Solved by cyclical coordinate descent (Griveau-Billion, Richard & Roncalli
    2013; Maillard, Roncalli & Teïletche 2010).

    The objective is: w_i * (Sigma @ w)_i equal for all i, with sum(w) = 1.

    Per-coordinate update. Fixing all w_j (j != i), equal risk contribution
    requires RC_i = w_i * (Sigma @ w)_i = (w' Sigma w) / N, which rearranges to
    the quadratic

        Sigma_ii * w_i^2 + (Sigma_{i,-i} @ w_{-i}) * w_i - (w' Sigma w)/N = 0

    so the constant term is ``-portfolio_variance/N``, recomputed from the
    CURRENT weights on every coordinate step — NOT the constant ``-1/N``.

    Why the constant -1/N was wrong here. It is a valid update only for
    iterates held at the particular scale where w'Sigma w = 1, i.e. it is
    correct only if you never renormalize mid-solve (or if Sigma is diagonal,
    where the coupling vanishes). This routine renormalizes to sum(w) = 1 after
    every sweep, which rescales w away from that fixed scale, so the iteration
    settled on a different — and wrong — fixed point. It converged, silently,
    to the wrong portfolio: on an 8-name equity-like covariance it put 20.4% of
    risk in one name against a 12.5% target (63% relative error) and missed the
    true ERC weights by 8.8pp on a single line (22.1pp L1).

    The variance form above is scale-INVARIANT (w -> k*w scales b by k and the
    constant by k^2, so the root scales by k), which is what makes it safe to
    combine with per-sweep normalization. The equivalent Roncalli form uses
    ``-sigma(w)/N`` but is only valid without renormalization; empirically it
    also needs ~7x more iterations to converge because the iterate has to drift
    to the sigma(w) = 1 scale on its own. Do not swap the forms without also
    changing the normalization.

    Parameters
    ----------
    cov_matrix : np.ndarray, shape (N, N), annualized covariance.
    max_iter : int
    tol : float, convergence tolerance on the weight vector.
    dispersion_tol : float, post-solve check. If the achieved risk-contribution
        dispersion exceeds this, a warning is logged and an SLSQP polish is
        attempted (never a silent wrong answer).

    Returns
    -------
    np.ndarray, shape (N,), long-only weights summing to 1.
    """
    n = cov_matrix.shape[0]
    if n < 2:
        return np.array([1.0]) if n == 1 else np.array([])

    cov_matrix = _make_psd(cov_matrix)

    # Cyclical coordinate descent (CCD)
    w = np.ones(n) / n
    converged = False
    for iteration in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            sigma_ii = float(cov_matrix[i, i])
            if sigma_ii < 1e-14:
                continue
            # Portfolio variance at the CURRENT weights. Recomputing this each
            # coordinate step is the fix: the old code used the constant -1/n.
            port_var = float(max(w @ cov_matrix @ w, 0.0))
            a = sigma_ii
            b = float(cov_matrix[i, :] @ w) - w[i] * sigma_ii  # Sigma_{i,-i} @ w_{-i}
            c = -port_var / n
            # Quadratic: a * w_i^2 + b * w_i + c = 0
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                continue
            w_i_new = (-b + np.sqrt(discriminant)) / (2 * a)
            w[i] = max(w_i_new, 1e-12)

        # Normalize (ERC is scale-invariant, so this is safe mid-solve)
        w = np.maximum(w, 0.0)
        w /= w.sum()

        # Convergence check
        if np.max(np.abs(w - w_prev)) < tol:
            logger.debug("Risk parity converged at iteration %d.", iteration)
            converged = True
            break
    if not converged:
        logger.warning("Risk parity did not converge in %d iterations.", max_iter)

    # --- verify the answer instead of trusting convergence ----------------
    dispersion = risk_contribution_dispersion(w, cov_matrix)
    if dispersion > dispersion_tol:
        logger.warning(
            "Risk parity risk-contribution dispersion %.4f%% exceeds tolerance "
            "%.4f%%; polishing with SLSQP.",
            dispersion * 100, dispersion_tol * 100,
        )
        w_polished = _erc_slsqp(cov_matrix, w0=w)
        if w_polished is not None:
            polished_dispersion = risk_contribution_dispersion(w_polished, cov_matrix)
            if polished_dispersion < dispersion:
                logger.info(
                    "SLSQP polish improved ERC dispersion %.4f%% -> %.4f%%.",
                    dispersion * 100, polished_dispersion * 100,
                )
                return w_polished
        logger.error(
            "Risk parity could not reach equal risk contributions "
            "(dispersion %.4f%%); returned weights are approximate.",
            dispersion * 100,
        )

    return w


def _erc_slsqp(cov_matrix: np.ndarray, w0: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Direct SLSQP solve of the ERC problem — used to polish/verify CCD output.

    Minimizes the variance of the risk contributions subject to sum(w)=1, w>=0.
    Returns None if the solve fails.
    """
    n = cov_matrix.shape[0]
    if w0 is None:
        w0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        rc = w * (cov_matrix @ w)
        return float(np.sum((rc - rc.mean()) ** 2)) * 1e6

    try:
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=[(1e-10, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"ftol": 1e-16, "maxiter": 2000},
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ERC SLSQP polish raised: %s.", exc)
        return None
    w = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    return w / total


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
    float, weight in [min_weight, max_weight]. Always finite.
    """
    # NaN compares False against every operator, so `signal_vol <= 0` lets NaN
    # through and np.clip(nan) returns nan -> a NaN position size downstream.
    # Guard non-finite inputs explicitly.
    if not np.isfinite(signal_vol):
        logger.warning(
            "signal_vol is not finite (%r); returning min_weight.", signal_vol
        )
        return float(min_weight)
    if not np.isfinite(target_vol):
        logger.warning(
            "target_vol is not finite (%r); returning min_weight.", target_vol
        )
        return float(min_weight)
    if signal_vol <= 0:
        logger.warning("signal_vol <= 0; returning min_weight.")
        return float(min_weight)
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

    _check_weight_bounds_feasible(n, max_weight, 0.0, "efficient_frontier")

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

def _equal_weight_vol_scaled(
    cov_matrix: np.ndarray,
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
) -> np.ndarray:
    """
    Fallback: equal weight with inverse-volatility scaling.

    Each asset gets weight proportional to 1/sigma_i, normalized to sum=1.
    When ``max_weight`` is given the result is projected onto the cap by
    water-filling, so the fallback can never return weights that breach a cap
    the caller asked for (the previous fallback returned 0.1667 under a 0.10
    cap).
    """
    n = cov_matrix.shape[0]
    vols = np.sqrt(np.abs(np.diag(cov_matrix)))
    vols = np.where((vols < 1e-10) | ~np.isfinite(vols), 1.0, vols)
    inv_vol = 1.0 / vols
    w = inv_vol / inv_vol.sum()

    if max_weight is None:
        return w

    cap = float(max_weight)
    if n * cap < 1.0 - 1e-12:
        # Caller-side infeasibility; the closest respectful answer is the cap.
        logger.warning(
            "Inverse-vol fallback cannot satisfy sum(w)=1 under cap %.4f for "
            "n=%d; returning equal weight at the cap.", cap, n,
        )
        return np.full(n, cap)

    # Water-filling: cap the overweights, redistribute to the rest.
    floor = max(float(min_weight), 0.0)
    for _ in range(n):
        over = w > cap + 1e-15
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over & (w < cap - 1e-15)
        if not free.any():
            break
        w[free] += excess * (w[free] / w[free].sum())
    w = np.clip(w, floor, cap)
    total = w.sum()
    if total > 0:
        w = w / total
    return np.clip(w, 0.0, cap)


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

    Raises
    ------
    ValueError
        If fewer than two usable return rows remain after filtering.

    Notes
    -----
    A price of 0.0 makes ``log(p_t / p_{t-1})`` produce ``-inf`` (and the next
    row ``+inf``). ``dropna()`` does NOT remove infinities, so LedoitWolf then
    raised an unhandled ``ValueError: Input X contains infinity``. Rows are
    filtered with ``np.isfinite`` instead.
    """
    if window is not None:
        prices_df = prices_df.tail(window + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_rets = np.log(prices_df / prices_df.shift(1))

    n_raw = len(log_rets)
    # dropna() alone leaves +/-inf behind; keep only all-finite rows.
    finite_rows = np.isfinite(log_rets.to_numpy(dtype=float)).all(axis=1)
    log_rets = log_rets[finite_rows]
    n_dropped = n_raw - len(log_rets) - 1  # -1 for the unavoidable first NaN row
    if n_dropped > 0:
        logger.warning(
            "Dropped %d non-finite return row(s) (zero/negative/missing prices) "
            "before covariance estimation.", n_dropped,
        )

    if log_rets.shape[0] < 2:
        raise ValueError(
            f"Only {log_rets.shape[0]} usable return row(s) after filtering "
            f"non-finite values; cannot estimate a covariance matrix. Check "
            f"for zero or negative prices in the input."
        )

    if log_rets.shape[0] < log_rets.shape[1]:
        logger.warning(
            "T=%d < N=%d; covariance estimate will be unreliable. "
            "Using LedoitWolf shrinkage.",
            log_rets.shape[0],
            log_rets.shape[1],
        )
    return _ledoit_wolf_cov(log_rets.values)
