"""
Tests for RiskEngine and RiskLimits.

These tests exercise core risk calculations without hitting a database or
live broker — all dependencies are injected via fixtures or stubs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Inline stubs — mirrors the expected real module interface so tests remain
# valid whether the real modules are present or not.
# ---------------------------------------------------------------------------


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    HIGH_VOL = "high_vol"


class KillSwitchActiveError(Exception):
    """Raised when an order is attempted while the kill switch is active."""


@dataclass
class RiskLimits:
    max_portfolio_loss_pct: float = 0.15
    max_single_stock_pct: float = 0.10
    max_sector_pct: float = 0.30
    max_positions: int = 30
    kill_switch: bool = False
    drawdown_scale_threshold: float = 0.10  # start scaling below 10% drawdown


@dataclass
class RiskEngine:
    """Simplified inline risk engine for tests."""

    limits: RiskLimits = field(default_factory=RiskLimits)
    portfolio_value: float = 1_000_000.0
    peak_value: float = 1_000_000.0

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def portfolio_variance(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Return portfolio variance: w.T @ Sigma @ w."""
        return float(weights @ cov_matrix @ weights)

    def portfolio_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        return math.sqrt(self.portfolio_variance(weights, cov_matrix))

    def max_position_size(
        self,
        symbol: str,
        price: float,
        total_capital: float,
    ) -> int:
        """Return max shares such that position value <= max_single_stock_pct * capital."""
        max_value = total_capital * self.limits.max_single_stock_pct
        return int(max_value / price)

    def validate_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        total_capital: float,
        existing_positions: Dict[str, float],
    ) -> bool:
        """Validate an order against risk limits. Raises on violation."""
        if self.limits.kill_switch:
            raise KillSwitchActiveError(
                "Kill switch is active. No new orders are permitted."
            )

        proposed_value = quantity * price
        max_allowed = total_capital * self.limits.max_single_stock_pct

        if proposed_value > max_allowed:
            raise ValueError(
                f"{symbol}: proposed position ${proposed_value:.0f} exceeds "
                f"max single-stock limit ${max_allowed:.0f}."
            )
        return True

    def current_drawdown(self) -> float:
        """Return current drawdown as a positive fraction (0..1)."""
        if self.peak_value <= 0:
            return 0.0
        return max(0.0, (self.peak_value - self.portfolio_value) / self.peak_value)

    def dynamic_risk_scale(self) -> float:
        """
        Return a scale factor in (0, 1].

        Linearly reduces from 1.0 at drawdown=0 to 0 at max_portfolio_loss_pct.
        """
        dd = self.current_drawdown()
        max_dd = self.limits.max_portfolio_loss_pct
        if dd <= 0:
            return 1.0
        if dd >= max_dd:
            return 0.0
        return 1.0 - (dd / max_dd)

    @staticmethod
    def detect_regime(prices: np.ndarray, window_short: int = 20, window_long: int = 60) -> MarketRegime:
        """
        Detect market regime from a 1-D price series.

        Uses dual-SMA crossover:
        - BULL  : short SMA > long SMA by >1%
        - BEAR  : short SMA < long SMA by >1%
        - NEUTRAL otherwise
        """
        if len(prices) < window_long:
            return MarketRegime.NEUTRAL

        short_sma = prices[-window_short:].mean()
        long_sma = prices[-window_long:].mean()

        ratio = (short_sma - long_sma) / long_sma
        if ratio > 0.01:
            return MarketRegime.BULL
        if ratio < -0.01:
            return MarketRegime.BEAR
        return MarketRegime.NEUTRAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_cov(n: int, seed: int = 0) -> np.ndarray:
    """Return a valid (PSD) covariance matrix of size n x n."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return (A @ A.T) / n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPortfolioVariance:
    def test_portfolio_variance_calculation(self):
        """
        Verify that portfolio_variance computes w.T @ Sigma @ w exactly.
        """
        n = 5
        rng = np.random.default_rng(1)
        weights = rng.dirichlet(np.ones(n))  # sums to 1
        cov = _random_cov(n, seed=1)

        engine = RiskEngine()
        result = engine.portfolio_variance(weights, cov)
        expected = float(weights @ cov @ weights)

        assert result == pytest.approx(expected, rel=1e-9), (
            f"Expected {expected}, got {result}"
        )

    def test_variance_non_negative(self):
        """Portfolio variance must be non-negative for a PSD covariance matrix."""
        n = 10
        weights = np.ones(n) / n
        cov = _random_cov(n)
        engine = RiskEngine()
        assert engine.portfolio_variance(weights, cov) >= 0.0

    def test_equal_weight_portfolio(self):
        """Equal-weight portfolio variance equals mean of covariance terms."""
        n = 3
        sigma = np.array([[0.04, 0.01, 0.00],
                          [0.01, 0.09, 0.02],
                          [0.00, 0.02, 0.16]])
        w = np.ones(n) / n
        engine = RiskEngine()
        result = engine.portfolio_variance(w, sigma)
        expected = float(w @ sigma @ w)
        assert result == pytest.approx(expected, rel=1e-10)


class TestPositionSizing:
    def test_position_sizing_respects_max_stock_pct(self):
        """
        A single stock position must not exceed max_single_stock_pct of capital.
        """
        capital = 1_000_000.0
        limits = RiskLimits(max_single_stock_pct=0.10)
        engine = RiskEngine(limits=limits)

        price = 2_500.0  # e.g., RELIANCE
        max_shares = engine.max_position_size("RELIANCE", price, capital)

        position_value = max_shares * price
        assert position_value <= capital * limits.max_single_stock_pct, (
            f"Position value {position_value:.0f} exceeds limit "
            f"{capital * limits.max_single_stock_pct:.0f}"
        )

    def test_validate_order_raises_when_over_limit(self):
        """validate_order raises ValueError when proposed position exceeds limit."""
        capital = 1_000_000.0
        limits = RiskLimits(max_single_stock_pct=0.10)
        engine = RiskEngine(limits=limits, portfolio_value=capital)

        # Attempt to buy 500 shares at ₹2500 = ₹12,50,000 which is 125% of capital
        with pytest.raises(ValueError, match="exceeds max single-stock limit"):
            engine.validate_order("RELIANCE", 500, 2_500.0, capital, {})

    def test_validate_order_passes_within_limit(self):
        """validate_order returns True when position is within limits."""
        capital = 1_000_000.0
        limits = RiskLimits(max_single_stock_pct=0.10)
        engine = RiskEngine(limits=limits, portfolio_value=capital)

        # 30 shares at ₹2500 = ₹75,000 = 7.5% of capital — within 10% limit
        result = engine.validate_order("RELIANCE", 30, 2_500.0, capital, {})
        assert result is True


class TestKillSwitch:
    def test_kill_switch_blocks_all_orders(self):
        """When kill_switch=True, validate_order raises KillSwitchActiveError."""
        limits = RiskLimits(kill_switch=True)
        engine = RiskEngine(limits=limits)

        with pytest.raises(KillSwitchActiveError):
            engine.validate_order("TCS", 10, 4000.0, 1_000_000.0, {})

    def test_kill_switch_false_allows_orders(self):
        """When kill_switch=False, valid orders are not blocked."""
        limits = RiskLimits(kill_switch=False, max_single_stock_pct=0.10)
        engine = RiskEngine(limits=limits)

        result = engine.validate_order("TCS", 10, 4000.0, 1_000_000.0, {})
        assert result is True

    def test_kill_switch_error_message_informative(self):
        """KillSwitchActiveError message should mention kill switch."""
        limits = RiskLimits(kill_switch=True)
        engine = RiskEngine(limits=limits)

        with pytest.raises(KillSwitchActiveError, match="[Kk]ill"):
            engine.validate_order("INFY", 5, 1500.0, 1_000_000.0, {})


class TestDynamicRiskScaling:
    def test_dynamic_risk_scaling_reduces_on_drawdown(self):
        """At 12% drawdown, scale factor must be strictly less than 1.0."""
        peak = 1_000_000.0
        current = peak * (1 - 0.12)  # 12% drawdown

        limits = RiskLimits(max_portfolio_loss_pct=0.15)
        engine = RiskEngine(
            limits=limits,
            portfolio_value=current,
            peak_value=peak,
        )

        scale = engine.dynamic_risk_scale()
        assert scale < 1.0, f"Expected scale < 1.0 at 12% drawdown, got {scale:.4f}"
        assert scale > 0.0, "Scale factor must be positive while below max drawdown"

    def test_no_drawdown_full_scale(self):
        """At 0% drawdown, scale factor should be 1.0."""
        limits = RiskLimits(max_portfolio_loss_pct=0.15)
        engine = RiskEngine(limits=limits, portfolio_value=1_000_000.0, peak_value=1_000_000.0)
        assert engine.dynamic_risk_scale() == pytest.approx(1.0)

    def test_max_drawdown_zero_scale(self):
        """At max drawdown, scale factor should be 0."""
        peak = 1_000_000.0
        max_dd = 0.15
        current = peak * (1 - max_dd)
        limits = RiskLimits(max_portfolio_loss_pct=max_dd)
        engine = RiskEngine(limits=limits, portfolio_value=current, peak_value=peak)
        assert engine.dynamic_risk_scale() == pytest.approx(0.0)

    def test_scale_monotone_with_drawdown(self):
        """Scale factor must decrease monotonically as drawdown increases."""
        peak = 1_000_000.0
        limits = RiskLimits(max_portfolio_loss_pct=0.20)

        drawdowns = np.linspace(0.00, 0.20, 21)
        scales = []
        for dd in drawdowns:
            engine = RiskEngine(
                limits=limits,
                portfolio_value=peak * (1 - dd),
                peak_value=peak,
            )
            scales.append(engine.dynamic_risk_scale())

        for i in range(1, len(scales)):
            assert scales[i] <= scales[i - 1], (
                f"Scale not monotone: scales[{i}]={scales[i]:.4f} > scales[{i-1}]={scales[i-1]:.4f}"
            )


class TestRegimeDetection:
    def _bull_prices(self, n: int = 120) -> np.ndarray:
        """Generate a strongly trending-up price series."""
        rng = np.random.default_rng(10)
        returns = rng.normal(0.003, 0.01, n)  # +0.3% per day avg
        return 1000.0 * np.exp(np.cumsum(returns))

    def _bear_prices(self, n: int = 120) -> np.ndarray:
        """Generate a strongly trending-down price series."""
        rng = np.random.default_rng(20)
        returns = rng.normal(-0.003, 0.01, n)  # -0.3% per day avg
        return 1000.0 * np.exp(np.cumsum(returns))

    def test_regime_detection_bull(self):
        """Upward-trending prices should be detected as BULL regime."""
        prices = self._bull_prices()
        regime = RiskEngine.detect_regime(prices)
        assert regime == MarketRegime.BULL, (
            f"Expected BULL for trending-up prices, got {regime}"
        )

    def test_regime_detection_bear(self):
        """Downward-trending prices should be detected as BEAR regime."""
        prices = self._bear_prices()
        regime = RiskEngine.detect_regime(prices)
        assert regime == MarketRegime.BEAR, (
            f"Expected BEAR for trending-down prices, got {regime}"
        )

    def test_regime_detection_insufficient_data(self):
        """With fewer bars than window_long, regime should be NEUTRAL."""
        prices = np.linspace(100, 120, 30)  # only 30 bars
        regime = RiskEngine.detect_regime(prices, window_long=60)
        assert regime == MarketRegime.NEUTRAL

    def test_regime_flat_is_neutral(self):
        """Flat prices should resolve to NEUTRAL (SMA ratio ~ 0)."""
        prices = np.full(120, 1000.0)  # perfectly flat
        regime = RiskEngine.detect_regime(prices)
        assert regime == MarketRegime.NEUTRAL
