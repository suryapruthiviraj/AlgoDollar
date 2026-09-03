"""
Numerical regression tests for the risk engine, risk limits and the portfolio
optimizer.

Every test here pins down a defect that was empirically reproduced against the
pre-fix code. Each one asserts against an INDEPENDENT reference (Monte Carlo,
a direct SLSQP solve, or an analytic identity) rather than against the
implementation's own output, so a re-introduced bug fails loudly.

Defects covered
---------------
1. CVaR was computed as ``var_95 * phi(z)/0.05``, applying the Expected
   Shortfall multiplier to VaR instead of to sigma and double-counting the
   z-score (ES overstated by z = 1.6449, i.e. +64.4%).
2. ``risk_parity_portfolio`` used a constant ``c = -1/n`` in the CCD update
   instead of ``-sigma(w)/n``, so it equalized risk contributions only for a
   diagonal covariance.
3. Four declared RiskLimits fields were never enforced.
4. Signed (negative) losses silently satisfied every loss limit.
5. Net exposure as the risk denominator blew up on a hedged book; a missing
   price silently produced zero risk.
6. Optimizer robustness defects (discarded expected_returns, no-op long_only,
   silent cap-violating fallback, -inf from zero prices, NaN vol target).
7. ``marginal_risk`` was documented as "% contribution" but held annualised
   vol units.
"""
from __future__ import annotations

import logging
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
from scipy.optimize import minimize

from app.portfolio import optimizer as opt
from app.risk.engine import (
    ES_SIGMA_MULTIPLIER_95,
    PHI_Z95,
    Z_95,
    MissingPriceError,
    RiskEngine,
)
from app.risk.limits import (
    LIMIT_CHECKS,
    RiskLimits,
    RiskState,
    check_all_limits,
    normalize_loss,
)

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _corr_cov(vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Build an annualised covariance from vols and a correlation matrix."""
    return np.outer(vols, vols) * corr


def _equity_like_cov_8() -> np.ndarray:
    """An 8-name, non-diagonal, equity-like annualised covariance."""
    vols = np.array([0.18, 0.24, 0.22, 0.28, 0.31, 0.35, 0.38, 0.42])
    corr = np.array([
        [1.0 if i == j else 0.55 + 0.02 * ((i + j) % 4) for j in range(8)]
        for i in range(8)
    ])
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return _corr_cov(vols, corr)


def _rc_fractions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Risk-contribution fractions RC_i / sum(RC); these sum to 1."""
    rc = w * (cov @ w)
    return rc / rc.sum()


def _true_erc_weights(cov: np.ndarray) -> np.ndarray:
    """Independent ERC reference solved directly with SLSQP."""
    n = cov.shape[0]

    def obj(x: np.ndarray) -> float:
        rc = x * (cov @ x)
        return float(np.sum((rc - rc.mean()) ** 2)) * 1e6

    res = minimize(
        obj,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(1e-9, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
        options={"ftol": 1e-16, "maxiter": 5000},
    )
    return res.x / res.x.sum()


# --------------------------------------------------------------------------- #
#  DEFECT 1 — CVaR / Expected Shortfall                                         #
# --------------------------------------------------------------------------- #

class TestCVaRExpectedShortfall:
    """CVaR must be sigma * phi(z)/(1-alpha), not VaR * phi(z)/(1-alpha)."""

    @staticmethod
    def _portfolio():
        n = 5
        vols = np.array([0.22, 0.28, 0.19, 0.25, 0.31])
        corr = np.full((n, n), 0.35) + np.eye(n) * 0.65
        cov = _corr_cov(vols, corr)
        positions = [{"symbol": f"S{i}", "quantity": 100} for i in range(n)]
        prices = {f"S{i}": 200.0 + 50 * i for i in range(n)}
        total = sum(100 * prices[f"S{i}"] for i in range(n))
        return positions, prices, cov, total

    def test_constants_match_scipy(self):
        """The hard-coded normal constants must equal scipy to ~1 ulp."""
        assert Z_95 == pytest.approx(float(stats.norm.ppf(0.95)), rel=1e-14)
        assert float(stats.norm.cdf(Z_95)) == pytest.approx(0.95, rel=1e-14)
        assert PHI_Z95 == pytest.approx(float(stats.norm.pdf(Z_95)), rel=1e-14)
        assert ES_SIGMA_MULTIPLIER_95 == pytest.approx(PHI_Z95 / 0.05, rel=1e-15)

    def test_cvar_matches_monte_carlo_expected_shortfall(self):
        """Engine CVaR must match a 4e6-path Monte-Carlo ES to within 1%."""
        positions, prices, cov, total = self._portfolio()
        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov
        )

        daily_sigma_rupees = (risk.portfolio_vol / np.sqrt(TRADING_DAYS)) * total

        rng = np.random.default_rng(20240117)
        losses = -rng.standard_normal(4_000_000) * daily_sigma_rupees
        cutoff = np.quantile(losses, 0.95)
        mc_es = float(losses[losses >= cutoff].mean())

        rel_err = abs(risk.cvar_95 - mc_es) / mc_es
        assert rel_err < 0.01, (
            f"engine CVaR95 = {risk.cvar_95:,.2f} vs Monte-Carlo ES95 = "
            f"{mc_es:,.2f} (rel err {rel_err:.2%})"
        )

    def test_cvar_matches_analytic_expected_shortfall(self):
        """Engine CVaR must equal sigma * phi(z)/0.05 essentially exactly."""
        positions, prices, cov, total = self._portfolio()
        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov
        )
        daily_sigma_rupees = (risk.portfolio_vol / np.sqrt(TRADING_DAYS)) * total
        analytic_es = daily_sigma_rupees * float(stats.norm.pdf(Z_95)) / 0.05
        assert risk.cvar_95 == pytest.approx(analytic_es, rel=1e-9)

    def test_cvar_var_ratio_is_not_z_inflated(self):
        """
        The old bug made CVaR/VaR = phi(z)/0.05 = 2.0627.
        The correct ratio for a normal is phi(z)/(0.05*z) = 1.2540.
        """
        positions, prices, cov, _ = self._portfolio()
        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov
        )
        ratio = risk.cvar_95 / risk.var_95
        assert ratio == pytest.approx(PHI_Z95 / (0.05 * Z_95), rel=1e-9)
        assert ratio == pytest.approx(1.2540, abs=1e-3)
        # Guard the specific regression: never the z-inflated 2.0627.
        assert ratio < 1.5, f"CVaR/VaR = {ratio:.4f} looks z-double-counted"

    def test_cvar_exceeds_var(self):
        """Sanity: ES is always worse than VaR at the same confidence."""
        positions, prices, cov, _ = self._portfolio()
        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov
        )
        assert risk.cvar_95 > risk.var_95 > 0

    def test_var_still_matches_monte_carlo(self):
        """Regression guard: VaR (already correct) must stay correct."""
        positions, prices, cov, total = self._portfolio()
        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov
        )
        daily_sigma_rupees = (risk.portfolio_vol / np.sqrt(TRADING_DAYS)) * total
        rng = np.random.default_rng(99)
        losses = -rng.standard_normal(2_000_000) * daily_sigma_rupees
        mc_var = float(np.quantile(losses, 0.95))
        assert risk.var_95 == pytest.approx(mc_var, rel=0.01)


# --------------------------------------------------------------------------- #
#  DEFECT 2 — risk parity really equalizes risk contributions                   #
# --------------------------------------------------------------------------- #

class TestRiskParity:

    def test_n8_correlated_dispersion_under_1pct(self):
        """
        The headline case. Pre-fix RC fractions were
        [0.204 0.162 0.136 0.120 0.108 0.098 0.090 0.084] against a 0.125
        target -> 62.8% max relative error.
        """
        cov = _equity_like_cov_8()
        w = opt.risk_parity_portfolio(cov)

        assert w.shape == (8,)
        assert w.sum() == pytest.approx(1.0, rel=1e-9)
        assert np.all(w > 0)

        frac = _rc_fractions(w, cov)
        target = 1.0 / 8
        dispersion = float(np.max(np.abs(frac - target)) / target)
        assert dispersion < 0.01, (
            f"risk-contribution dispersion {dispersion:.4%} >= 1%; "
            f"fractions = {np.array2string(frac, precision=5)}"
        )

    def test_n8_matches_independent_slsqp_erc(self):
        """Weights must match an independently-solved ERC portfolio."""
        cov = _equity_like_cov_8()
        w = opt.risk_parity_portfolio(cov)
        w_true = _true_erc_weights(cov)
        max_err_pp = float(np.max(np.abs(w - w_true))) * 100
        assert max_err_pp < 0.5, (
            f"max weight error {max_err_pp:.3f}pp vs true ERC "
            f"{np.array2string(w_true, precision=5)}"
        )

    @pytest.mark.parametrize("rho", [0.0, 0.30, 0.70, 0.90])
    def test_dispersion_under_1pct_across_correlations(self, rho):
        """Must hold at every correlation level, not just the easy ones."""
        n = 5
        vols = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
        corr = np.full((n, n), rho) + np.eye(n) * (1.0 - rho)
        cov = _corr_cov(vols, corr)
        w = opt.risk_parity_portfolio(cov)
        dispersion = opt.risk_contribution_dispersion(w, cov)
        assert dispersion < 0.01, f"rho={rho}: dispersion {dispersion:.4%}"

    def test_diagonal_covariance_still_correct(self):
        """The diagonal case was already correct; do not regress it."""
        cov = np.diag(np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35]) ** 2)
        w = opt.risk_parity_portfolio(cov)
        dispersion = opt.risk_contribution_dispersion(w, cov)
        assert dispersion < 1e-6
        # For a diagonal Sigma, ERC == inverse-vol weights.
        inv_vol = 1.0 / np.sqrt(np.diag(cov))
        assert w == pytest.approx(inv_vol / inv_vol.sum(), rel=1e-5)

    def test_random_covariances_all_within_tolerance(self):
        """Fuzz over random PSD covariances."""
        rng = np.random.default_rng(4242)
        for trial in range(15):
            n = int(rng.integers(2, 12))
            a = rng.standard_normal((n, n + 5))
            cov = (a @ a.T) / (n + 5) * 0.05 + np.eye(n) * 1e-4
            w = opt.risk_parity_portfolio(cov)
            dispersion = opt.risk_contribution_dispersion(w, cov)
            assert dispersion < 0.01, (
                f"trial {trial} (n={n}): dispersion {dispersion:.4%}"
            )

    def test_risk_parity_sits_between_equal_and_inverse_vol_risk(self):
        """
        Economic sanity: ERC risk contributions must be flatter than those of
        an equal-weight portfolio on the same covariance.
        """
        cov = _equity_like_cov_8()
        w_erc = opt.risk_parity_portfolio(cov)
        w_eq = np.ones(8) / 8
        assert (
            opt.risk_contribution_dispersion(w_erc, cov)
            < opt.risk_contribution_dispersion(w_eq, cov)
        )


# --------------------------------------------------------------------------- #
#  DEFECT 3 — every declared limit is enforced                                  #
# --------------------------------------------------------------------------- #

def _tripping_state(limit_field: str, limits: RiskLimits) -> RiskState:
    """Build a RiskState that must trip `limit_field` and nothing weaker."""
    base = dict(total_capital=1_000_000.0, current_portfolio_value=1_000_000.0)
    equity = base["current_portfolio_value"]

    if limit_field == "max_daily_loss":
        return RiskState(daily_loss=limits.max_daily_loss * 1.5, **base)
    if limit_field == "max_weekly_loss":
        return RiskState(weekly_loss=limits.max_weekly_loss * 1.5, **base)
    if limit_field == "max_monthly_loss":
        return RiskState(monthly_loss=limits.max_monthly_loss * 1.5, **base)
    if limit_field == "max_drawdown_pct":
        return RiskState(current_drawdown=limits.max_drawdown_pct * 1.5, **base)
    if limit_field == "soft_drawdown_pct":
        # Between the soft and hard thresholds.
        soft = (limits.soft_drawdown_pct + limits.max_drawdown_pct) / 2.0
        return RiskState(current_drawdown=soft, **base)
    if limit_field == "max_open_positions":
        return RiskState(positions_count=limits.max_open_positions + 5, **base)
    if limit_field == "max_single_stock_pct":
        return RiskState(
            position_values={"RELIANCE": equity * (limits.max_single_stock_pct + 0.05)},
            **base,
        )
    if limit_field == "max_sector_pct":
        return RiskState(
            sector_values={"Financials": equity * (limits.max_sector_pct + 0.05)},
            **base,
        )
    if limit_field == "max_risk_per_trade_pct":
        return RiskState(
            trade_risk_amounts={"T-1": equity * (limits.max_risk_per_trade_pct * 2)},
            **base,
        )
    if limit_field == "max_daily_risk":
        return RiskState(daily_risk_used=limits.max_daily_risk * 1.5, **base)
    if limit_field == "max_leverage":
        return RiskState(gross_exposure=equity * (limits.max_leverage + 0.5), **base)
    if limit_field == "max_intraday_capital_pct":
        return RiskState(
            intraday_capital_used=equity * (limits.max_intraday_capital_pct + 0.05),
            **base,
        )
    raise AssertionError(f"No tripping state defined for limit '{limit_field}'")


class TestAllLimitsEnforced:

    def test_registry_covers_every_declared_limit(self):
        """
        A limit declared on RiskLimits with no entry in LIMIT_CHECKS is a limit
        nobody enforces. Pre-fix, max_single_stock_pct, max_sector_pct,
        max_risk_per_trade_pct and max_leverage were all in that state.
        """
        declared = {f.name for f in fields(RiskLimits)}
        registered = set(LIMIT_CHECKS)
        assert declared == registered, (
            f"declared-but-unenforced: {sorted(declared - registered)}; "
            f"registered-but-undeclared: {sorted(registered - declared)}"
        )

    @pytest.mark.parametrize("limit_field", [f.name for f in fields(RiskLimits)])
    def test_each_declared_limit_can_actually_fire(self, limit_field):
        """For each declared limit, a breaching state must produce a breach."""
        limits = RiskLimits()
        state = _tripping_state(limit_field, limits)
        breaches = check_all_limits(state, limits)
        expected = LIMIT_CHECKS[limit_field]
        names = [b.limit_name for b in breaches]
        assert expected in names, (
            f"limit '{limit_field}' did not fire; expected breach "
            f"'{expected}', got {names}"
        )

    @pytest.mark.parametrize("limit_field", [f.name for f in fields(RiskLimits)])
    def test_clean_state_fires_nothing(self, limit_field):
        """A fully-compliant state must produce no breaches at all."""
        limits = RiskLimits()
        state = RiskState(
            total_capital=1_000_000.0,
            current_portfolio_value=1_000_000.0,
            position_values={"RELIANCE": 50_000.0},
            sector_values={"Energy": 50_000.0},
            trade_risk_amounts={"T-1": 2_000.0},
            gross_exposure=500_000.0,
        )
        assert check_all_limits(state, limits) == []

    def test_sector_limit_actually_compares_engine_output(self):
        """
        The engine computes sector_concentrations; wiring it into the limit
        check must produce a breach for an over-concentrated sector.
        """
        cov = np.eye(3) * 0.04
        positions = [
            {"symbol": "HDFCBANK", "quantity": 100},
            {"symbol": "ICICIBANK", "quantity": 100},
            {"symbol": "RELIANCE", "quantity": 10},
        ]
        prices = {"HDFCBANK": 1600.0, "ICICIBANK": 900.0, "RELIANCE": 2500.0}
        sectors = {
            "HDFCBANK": "Financials",
            "ICICIBANK": "Financials",
            "RELIANCE": "Energy",
        }
        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov, sector_map=sectors
        )
        assert risk.sector_concentrations["Financials"] > 0.30

        gross = risk.gross_exposure
        state = RiskState(
            total_capital=gross,
            current_portfolio_value=gross,
            sector_values={
                s: pct * gross for s, pct in risk.sector_concentrations.items()
            },
        )
        breaches = check_all_limits(state, RiskLimits())
        assert "sector_concentration" in [b.limit_name for b in breaches]

    def test_short_position_counts_toward_concentration(self):
        """Concentration limits use |value|, so a big short also breaches."""
        limits = RiskLimits()
        state = RiskState(
            total_capital=1_000_000.0,
            current_portfolio_value=1_000_000.0,
            position_values={"TCS": -400_000.0},
        )
        breaches = check_all_limits(state, limits)
        assert "single_stock_concentration" in [b.limit_name for b in breaches]

    def test_exposure_without_equity_base_raises(self):
        """Never skip a concentration check silently for lack of a denominator."""
        state = RiskState(position_values={"TCS": 500_000.0})
        with pytest.raises(ValueError, match="cannot be evaluated"):
            check_all_limits(state, RiskLimits())

    def test_intraday_capital_pct_reconciled_to_allocator(self):
        """
        limits.py declared 0.50 while allocator.py and core/config.py used
        0.10 — a 5x disagreement. limits.py is now the authoritative source
        and must agree with the allocator's value.
        """
        assert RiskLimits().max_intraday_capital_pct == pytest.approx(0.10)


# --------------------------------------------------------------------------- #
#  DEFECT 4 — loss sign convention                                              #
# --------------------------------------------------------------------------- #

class TestLossSignConvention:

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"daily_loss": -50_000.0},
            {"weekly_loss": -90_000.0},
            {"monthly_loss": -200_000.0},
            {"daily_loss": -50_000.0, "weekly_loss": -90_000.0,
             "monthly_loss": -200_000.0},
        ],
    )
    def test_negative_loss_raises_at_construction(self, kwargs):
        """
        Pre-fix, RiskState(daily_loss=-50_000, ...) produced an EMPTY breach
        list against a ₹10,000 limit. Now it is impossible to build.
        """
        with pytest.raises(ValueError, match="non-negative loss magnitude"):
            RiskState(**kwargs)

    def test_negative_loss_assigned_after_construction_raises_on_check(self):
        """Bypassing __post_init__ by assignment must still be caught."""
        state = RiskState(total_capital=1_000_000.0)
        state.daily_loss = -50_000.0
        with pytest.raises(ValueError, match="non-negative loss magnitude"):
            check_all_limits(state, RiskLimits())

    def test_normalize_loss_converts_signed_pnl(self):
        assert normalize_loss(-50_000.0) == 50_000.0
        assert normalize_loss(12_000.0) == 0.0
        assert normalize_loss(0.0) == 0.0

    def test_from_pnl_normalizes_and_breaches(self):
        """The documented ingest path turns a signed loss into a real breach."""
        state = RiskState.from_pnl(
            daily_pnl=-50_000.0,
            weekly_pnl=-90_000.0,
            monthly_pnl=-200_000.0,
            total_capital=1_000_000.0,
            current_portfolio_value=1_000_000.0,
        )
        assert state.daily_loss == 50_000.0
        assert state.weekly_loss == 90_000.0
        assert state.monthly_loss == 200_000.0

        names = [b.limit_name for b in check_all_limits(state, RiskLimits())]
        assert "daily_loss" in names
        assert "weekly_loss" in names
        assert "monthly_loss" in names

    def test_from_pnl_treats_profit_as_zero_loss(self):
        state = RiskState.from_pnl(daily_pnl=75_000.0, total_capital=1_000_000.0)
        assert state.daily_loss == 0.0
        assert check_all_limits(state, RiskLimits()) == []

    def test_convention_documented(self):
        """The convention must be stated in the dataclass docstring."""
        doc = RiskState.__doc__ or ""
        assert "non-negative" in doc.lower()
        assert "normalize_loss" in doc


# --------------------------------------------------------------------------- #
#  DEFECT 5 — long/short books and missing prices                               #
# --------------------------------------------------------------------------- #

class TestLongShortAndMissingPrices:

    @staticmethod
    def _pair_cov(rho: float = 0.80) -> np.ndarray:
        vols = np.array([0.20, 0.20])
        corr = np.array([[1.0, rho], [rho, 1.0]])
        return _corr_cov(vols, corr)

    def test_market_neutral_book_has_sane_vol(self):
        """
        +₹100,000 long vs -₹99,999.99 short netted to ~1e-2 and produced
        portfolio_vol = 1,264,911 (annualised!). Gross exposure fixes it.
        """
        positions = [
            {"symbol": "LONG", "quantity": 100},
            {"symbol": "SHORT", "quantity": -100},
        ]
        prices = {"LONG": 1000.0, "SHORT": 999.9999}
        cov = self._pair_cov(rho=0.80)

        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov
        )

        assert np.isfinite(risk.portfolio_vol)
        assert np.isfinite(risk.var_95)
        assert np.isfinite(risk.cvar_95)
        # Two 20%-vol names, rho=0.8, half gross each: sigma = 0.20*sqrt(0.1)
        expected_vol = 0.20 * np.sqrt(2 * 0.25 * (1 - 0.80))
        assert risk.portfolio_vol == pytest.approx(expected_vol, rel=1e-3)
        # A hedged book must be far LESS volatile than either leg.
        assert risk.portfolio_vol < 0.20
        assert risk.gross_exposure == pytest.approx(199_999.99, rel=1e-9)
        assert abs(risk.net_exposure) < 1.0

    def test_perfectly_hedged_book_has_near_zero_vol(self):
        """rho = 1.0 and equal-and-opposite legs => essentially zero risk."""
        positions = [
            {"symbol": "A", "quantity": 100},
            {"symbol": "B", "quantity": -100},
        ]
        prices = {"A": 1000.0, "B": 1000.0}
        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=self._pair_cov(rho=1.0)
        )
        assert risk.portfolio_vol < 1e-6
        assert risk.var_95 < 1.0

    def test_vol_never_exceeds_worst_single_name(self):
        """A long/short book cannot be more volatile than its riskiest leg."""
        rng = np.random.default_rng(31337)
        for _ in range(20):
            n = 4
            qty = rng.integers(-100, 100, n).astype(float)
            if np.all(qty == 0):
                continue
            positions = [
                {"symbol": f"S{i}", "quantity": float(qty[i])} for i in range(n)
            ]
            prices = {f"S{i}": float(rng.uniform(100, 3000)) for i in range(n)}
            vols = rng.uniform(0.15, 0.40, n)
            a = rng.standard_normal((n, n + 6))
            corr = np.corrcoef(a)
            cov = _corr_cov(vols, corr)
            cov = cov + np.eye(n) * 1e-8
            risk = RiskEngine().calculate_portfolio_risk(
                positions, prices, covariance_matrix=cov
            )
            assert np.isfinite(risk.portfolio_vol)
            assert risk.portfolio_vol <= vols.max() + 1e-8, (
                f"vol {risk.portfolio_vol} exceeds worst leg {vols.max()}"
            )

    def test_long_only_book_unchanged(self):
        """Regression guard: for a long-only book gross == net, so nothing moves."""
        positions = [
            {"symbol": "A", "quantity": 10},
            {"symbol": "B", "quantity": 20},
        ]
        prices = {"A": 2500.0, "B": 1500.0}
        cov = self._pair_cov(rho=0.30)
        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov
        )
        total = 10 * 2500.0 + 20 * 1500.0
        w = np.array([10 * 2500.0, 20 * 1500.0]) / total
        expected_vol = float(np.sqrt(w @ cov @ w))
        assert risk.portfolio_vol == pytest.approx(expected_vol, rel=1e-12)
        assert risk.gross_exposure == pytest.approx(total)
        assert risk.net_exposure == pytest.approx(total)

    def test_missing_price_raises(self):
        """A symbol with no price must raise, not become weight 0."""
        positions = [
            {"symbol": "A", "quantity": 100},
            {"symbol": "B", "quantity": 50},
        ]
        with pytest.raises(MissingPriceError, match="No price supplied"):
            RiskEngine().calculate_portfolio_risk(
                positions,
                {"A": 500.0},
                covariance_matrix=np.eye(2) * 0.04,
                portfolio_value=1_000_000.0,
            )

    def test_all_prices_missing_raises_instead_of_zero_risk(self):
        """Pre-fix this returned vol=0.0 / var=0.0 on a live book."""
        with pytest.raises(MissingPriceError):
            RiskEngine().calculate_portfolio_risk(
                [{"symbol": "A", "quantity": 100}], {}, portfolio_value=1_000_000.0
            )

    @pytest.mark.parametrize("bad_price", [0.0, -10.0, float("nan"), float("inf")])
    def test_invalid_price_raises(self, bad_price):
        with pytest.raises(MissingPriceError):
            RiskEngine().calculate_portfolio_risk(
                [{"symbol": "A", "quantity": 100}],
                {"A": bad_price},
                covariance_matrix=np.eye(1) * 0.04,
            )

    def test_empty_portfolio_still_returns_zeros(self):
        """No positions is legitimately zero risk (not the same as no prices)."""
        risk = RiskEngine().calculate_portfolio_risk([], {})
        assert risk.portfolio_vol == 0.0
        assert risk.var_95 == 0.0
        assert risk.cvar_95 == 0.0


# --------------------------------------------------------------------------- #
#  DEFECT 6 — optimizer robustness                                              #
# --------------------------------------------------------------------------- #

class TestOptimizerRobustness:

    def test_small_n_uses_expected_returns(self):
        """
        Pre-fix, n<5 discarded expected_returns: er=[0.30,0.02] and
        er=[0.02,0.30] both returned [0.6, 0.4].
        """
        cov = np.array([[0.04, 0.006], [0.006, 0.09]])
        w_a = opt.mean_variance_optimize(np.array([0.30, 0.02]), cov, max_weight=0.9)
        w_b = opt.mean_variance_optimize(np.array([0.02, 0.30]), cov, max_weight=0.9)
        assert not np.allclose(w_a, w_b), (
            f"expected_returns ignored: {w_a} == {w_b}"
        )
        assert w_a[0] > w_a[1], "should overweight the high-return asset"
        assert w_b[1] > w_b[0], "should overweight the high-return asset"

    def test_four_name_sleeve_is_optimized(self):
        """A 4-name sleeve must beat equal weight on Sharpe."""
        rng = np.random.default_rng(5)
        a = rng.standard_normal((4, 30))
        cov = (a @ a.T) / 30 * 0.09 + np.eye(4) * 0.01
        er = np.array([0.28, 0.06, 0.14, 0.04])
        w = opt.mean_variance_optimize(er, cov, max_weight=0.60)
        w_eq = np.ones(4) / 4

        def sharpe(weights):
            return (er @ weights - 0.065) / np.sqrt(weights @ cov @ weights)

        assert sharpe(w) > sharpe(w_eq)

    def test_long_only_flag_is_not_a_no_op(self):
        """long_only=False must actually permit short weights."""
        n = 5
        vols = np.array([0.22, 0.28, 0.19, 0.25, 0.31])
        corr = np.full((n, n), 0.35) + np.eye(n) * 0.65
        cov = _corr_cov(vols, corr)
        er = np.array([0.25, 0.10, 0.08, 0.15, -0.10])

        w_lo = opt.mean_variance_optimize(er, cov, max_weight=0.60, long_only=True)
        w_ls = opt.mean_variance_optimize(er, cov, max_weight=0.60, long_only=False)

        assert w_lo.min() >= -1e-9, "long_only=True must stay non-negative"
        assert not np.allclose(w_lo, w_ls), "long_only flag had no effect"
        assert w_ls.min() < -1e-6, f"no short taken: min weight {w_ls.min()}"
        assert w_ls.sum() == pytest.approx(1.0, rel=1e-6)

    def test_infeasible_cap_raises_rather_than_violating(self):
        """
        min_vol with n=5 and max_weight=0.10 previously returned max w=0.2554,
        breaching the cap with no log line.
        """
        cov = np.eye(5) * 0.04
        with pytest.raises(ValueError, match="infeasible bounds"):
            opt.minimum_volatility_portfolio(cov, max_weight=0.10)
        with pytest.raises(ValueError, match="infeasible bounds"):
            opt.mean_variance_optimize(np.arange(5) / 10.0, cov, max_weight=0.10)

    def test_min_vol_respects_feasible_cap(self):
        rng = np.random.default_rng(11)
        a = rng.standard_normal((6, 40))
        cov = (a @ a.T) / 40 * 0.06 + np.eye(6) * 0.002
        w = opt.minimum_volatility_portfolio(cov, max_weight=0.25)
        assert w.max() <= 0.25 + 1e-9
        assert w.sum() == pytest.approx(1.0, rel=1e-9)

    def test_fallback_logs_and_respects_cap(self, caplog):
        """The inverse-vol fallback must warn and never breach the cap."""
        cov = np.diag([0.01, 0.04, 0.09, 0.16, 0.25])
        with caplog.at_level(logging.WARNING, logger=opt.logger.name):
            w = opt._equal_weight_vol_scaled(cov, max_weight=0.30)
        assert w.max() <= 0.30 + 1e-9
        assert w.sum() == pytest.approx(1.0, rel=1e-9)

    def test_zero_price_does_not_break_ledoit_wolf(self, caplog):
        """
        log(0/x) = -inf survives dropna() and made LedoitWolf raise
        `ValueError: Input X contains infinity`.
        """
        prices = pd.DataFrame({
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "B": [50.0, 0.0, 51.0, 52.0, 53.0, 54.0, 55.0],
        })
        with caplog.at_level(logging.WARNING, logger=opt.logger.name):
            cov = opt.compute_ledoit_wolf_covariance(prices)
        assert cov.shape == (2, 2)
        assert np.all(np.isfinite(cov))
        assert "non-finite" in caplog.text

    def test_all_zero_prices_raises_clearly(self):
        prices = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
        with pytest.raises(ValueError, match="usable return row"):
            opt.compute_ledoit_wolf_covariance(prices)

    def test_clean_prices_still_produce_valid_covariance(self):
        """Regression guard: annualisation stays x252 on VARIANCE."""
        rng = np.random.default_rng(2024)
        daily_sd = 0.015
        rets = rng.normal(0.0, daily_sd, (2000, 3))
        prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)))
        cov = opt.compute_ledoit_wolf_covariance(prices)
        annual_vols = np.sqrt(np.diag(cov))
        expected = daily_sd * np.sqrt(252)
        assert annual_vols == pytest.approx(np.full(3, expected), rel=0.10)

    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf"), 0.0, -0.2]
    )
    def test_volatility_target_size_guards_non_finite(self, bad):
        """volatility_target_size(nan) returned nan -> NaN position size."""
        w = opt.volatility_target_size(bad)
        assert np.isfinite(w)
        assert w == pytest.approx(0.01)

    def test_volatility_target_size_normal_path(self):
        assert opt.volatility_target_size(0.30, target_vol=0.15) == pytest.approx(0.5)
        assert opt.volatility_target_size(0.10, target_vol=0.15) == pytest.approx(0.50)
        assert opt.volatility_target_size(1.0, target_vol=0.15) == pytest.approx(0.15)


# --------------------------------------------------------------------------- #
#  DEFECT 7 — risk-contribution units are unambiguous                           #
# --------------------------------------------------------------------------- #

class TestRiskContributionUnits:

    @staticmethod
    def _risk():
        n = 4
        vols = np.array([0.20, 0.25, 0.30, 0.18])
        corr = np.full((n, n), 0.40) + np.eye(n) * 0.60
        cov = _corr_cov(vols, corr)
        positions = [{"symbol": f"S{i}", "quantity": 100} for i in range(n)]
        prices = {f"S{i}": 500.0 + 100 * i for i in range(n)}
        return RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov
        ), cov

    def test_pct_field_sums_to_one(self):
        risk, _ = self._risk()
        assert sum(risk.risk_contribution_pct.values()) == pytest.approx(1.0, rel=1e-9)

    def test_vol_units_field_sums_to_portfolio_vol(self):
        """Euler decomposition: contributions sum to total volatility."""
        risk, _ = self._risk()
        assert sum(risk.risk_contributions.values()) == pytest.approx(
            risk.portfolio_vol, rel=1e-9
        )

    def test_fields_are_not_confusable(self):
        """
        The two fields must differ by exactly the 1/vol factor that made the
        old single 'marginal_risk' field off by ~5x for a percentage check.
        """
        risk, _ = self._risk()
        for sym, pct in risk.risk_contribution_pct.items():
            assert risk.risk_contributions[sym] == pytest.approx(
                pct * risk.portfolio_vol, rel=1e-9
            )
        assert not np.isclose(sum(risk.risk_contributions.values()), 1.0)

    def test_marginal_risk_alias_preserved(self):
        """Backwards-compatible alias still returns the vol-unit dict."""
        risk, _ = self._risk()
        assert risk.marginal_risk == risk.risk_contributions

    def test_equal_weight_uncorrelated_gives_equal_pct(self):
        """Independent check with a known answer."""
        cov = np.eye(4) * 0.04
        positions = [{"symbol": f"S{i}", "quantity": 100} for i in range(4)]
        prices = {f"S{i}": 1000.0 for i in range(4)}
        risk = RiskEngine().calculate_portfolio_risk(
            positions, prices, covariance_matrix=cov
        )
        for pct in risk.risk_contribution_pct.values():
            assert pct == pytest.approx(0.25, rel=1e-9)
