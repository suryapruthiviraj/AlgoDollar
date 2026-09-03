"""
Tests for the backtesting engine.

Focus areas:
- No look-ahead bias: features at time T only use data up to T.
- Costs are applied (gross > net when costs > 0).
- Max drawdown stops trading.
- Positions are not double-entered.
- Walk-forward out-of-sample (OOS) data is never used during training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal backtester stub — enough surface area for the tests below.
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    commission_pct: float = 0.0003       # 0.03% round-trip each leg
    slippage_pct: float = 0.0002         # 0.02% market impact estimate
    max_drawdown_stop: float = 0.20      # halt when drawdown exceeds 20%
    position_size_pct: float = 0.10      # 10% of capital per position


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    cost: float = 0.0


class BacktestResult:
    def __init__(self):
        self.trades: List[Trade] = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.max_drawdown: float = 0.0
        self.gross_return: float = 0.0
        self.net_return: float = 0.0
        self.halted: bool = False
        self.halt_reason: Optional[str] = None


class SimpleBacktester:
    """
    Deterministic, single-symbol backtester for unit testing.

    Signal function receives a slice of prices up to (and including) today
    and returns +1 (buy), -1 (sell/exit), or 0 (hold).
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._signal_calls: List[Tuple[pd.Timestamp, int]] = []
        self._last_seen_index: Optional[int] = None

    def run(
        self,
        prices: pd.Series,
        signal_fn: Callable[[pd.Series], int],
    ) -> BacktestResult:
        """
        Run backtest on a price series.

        signal_fn receives prices.iloc[:i+1] (never future data) and returns
        a signal integer.
        """
        result = BacktestResult()
        capital = self.config.initial_capital
        peak_capital = capital
        position: Optional[Dict] = None
        equity_points = {}

        for i, (date, price) in enumerate(prices.items()):
            # -- LOOK-AHEAD SAFETY: only pass data up to index i --
            historical_prices = prices.iloc[: i + 1]
            self._last_seen_index = i

            signal = signal_fn(historical_prices)
            self._signal_calls.append((date, i))  # record (date, data_length)

            # Compute cost for a round-trip at current price
            cost_per_share = price * (
                self.config.commission_pct + self.config.slippage_pct
            )

            if position is None and signal == 1:
                # Enter long position
                shares = int(
                    (capital * self.config.position_size_pct)
                    / (price * (1 + self.config.slippage_pct))
                )
                if shares > 0:
                    entry_cost = shares * price * self.config.commission_pct
                    position = {
                        "symbol": "TEST",
                        "entry_date": date,
                        "entry_price": price,
                        "quantity": shares,
                        "entry_cost": entry_cost,
                    }
                    capital -= shares * price + entry_cost

            elif position is not None and signal == -1:
                # Exit position
                exit_cost = position["quantity"] * price * self.config.commission_pct
                gross_pnl = position["quantity"] * (price - position["entry_price"])
                total_cost = position["entry_cost"] + exit_cost
                net_pnl = gross_pnl - total_cost

                capital += position["quantity"] * price - exit_cost

                result.trades.append(
                    Trade(
                        symbol=position["symbol"],
                        entry_date=position["entry_date"],
                        exit_date=date,
                        entry_price=position["entry_price"],
                        exit_price=price,
                        quantity=position["quantity"],
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        cost=total_cost,
                    )
                )
                position = None

            # Mark-to-market
            portfolio_value = capital
            if position is not None:
                portfolio_value += position["quantity"] * price
            equity_points[date] = portfolio_value

            # Drawdown check
            if portfolio_value > peak_capital:
                peak_capital = portfolio_value
            drawdown = (peak_capital - portfolio_value) / peak_capital
            result.max_drawdown = max(result.max_drawdown, drawdown)

            if drawdown >= self.config.max_drawdown_stop:
                result.halted = True
                result.halt_reason = f"Max drawdown {drawdown:.1%} exceeded limit"
                # Force-close position at current price
                if position is not None:
                    exit_cost = position["quantity"] * price * self.config.commission_pct
                    gross_pnl = position["quantity"] * (price - position["entry_price"])
                    net_pnl = gross_pnl - (position["entry_cost"] + exit_cost)
                    capital += position["quantity"] * price - exit_cost
                    result.trades.append(
                        Trade(
                            symbol=position["symbol"],
                            entry_date=position["entry_date"],
                            exit_date=date,
                            entry_price=position["entry_price"],
                            exit_price=price,
                            quantity=position["quantity"],
                            gross_pnl=gross_pnl,
                            net_pnl=net_pnl,
                            cost=position["entry_cost"] + exit_cost,
                        )
                    )
                    position = None
                break

        result.equity_curve = pd.Series(equity_points)
        final_value = list(equity_points.values())[-1] if equity_points else self.config.initial_capital
        result.gross_return = sum(t.gross_pnl for t in result.trades) / self.config.initial_capital
        result.net_return = sum(t.net_pnl for t in result.trades) / self.config.initial_capital
        return result

    def get_signal_data_lengths(self) -> List[int]:
        """Return the number of bars passed to signal_fn at each call."""
        return [data_len for _, data_len in self._signal_calls]


def _make_prices(n: int, trend: float = 0.0, seed: int = 0) -> pd.Series:
    """Generate a synthetic price series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    returns = rng.normal(trend, 0.015, n)
    prices = 1000.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="close")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoLookAheadBias:
    def test_no_lookahead_bias(self):
        """
        Signal function must never receive future prices.

        Verify that at call i, signal_fn only has access to prices[:i+1],
        i.e. data_length == i + 1 for all calls.
        """
        prices = _make_prices(100, seed=1)
        backtester = SimpleBacktester()

        # Signal fn that always holds (0) — we just measure data length
        backtester.run(prices, signal_fn=lambda p: 0)

        data_lengths = backtester.get_signal_data_lengths()

        for call_idx, data_len in enumerate(data_lengths):
            expected_len = call_idx + 1
            assert data_len == expected_len, (
                f"At call {call_idx}, signal received {data_len} bars "
                f"but should have received {expected_len} (no future data)."
            )

    def test_signal_cannot_access_future_index(self):
        """
        If the signal tries to access price at T+1, it should raise IndexError,
        not return a value — confirming isolation.
        """
        prices = _make_prices(50, seed=2)
        backtester = SimpleBacktester()

        violations: List[bool] = []

        def leaky_signal(p: pd.Series) -> int:
            try:
                _ = p.iloc[-1 + 1]  # try to peek one bar ahead inside the slice
                violations.append(True)
            except IndexError:
                violations.append(False)
            return 0

        backtester.run(prices, signal_fn=leaky_signal)

        # At least the last bar should raise IndexError when trying to access +1
        # (final bar has no future)
        assert violations[-1] is False, "Last bar should not have a future price"


class TestCostsApplied:
    def test_costs_applied_gross_exceeds_net(self):
        """
        With non-zero commission, gross_return > net_return when trades occur.
        """
        prices = _make_prices(100, trend=0.001, seed=3)
        config = BacktestConfig(
            commission_pct=0.0003,
            slippage_pct=0.0002,
            max_drawdown_stop=1.0,  # disable halt
        )
        backtester = SimpleBacktester(config=config)

        # Buy on day 1, sell on day 50
        call_count = [0]

        def timed_signal(p: pd.Series) -> int:
            n = len(p)
            call_count[0] += 1
            if n == 1:
                return 1   # buy
            if n == 50:
                return -1  # sell
            return 0

        result = backtester.run(prices, signal_fn=timed_signal)

        assert len(result.trades) >= 1, "Expected at least one completed trade"
        for trade in result.trades:
            assert trade.cost > 0, "Trade cost must be positive when commission > 0"
            assert trade.gross_pnl != trade.net_pnl or trade.cost == 0

    def test_zero_cost_gross_equals_net(self):
        """With commission=0 and slippage=0, gross_pnl == net_pnl."""
        prices = _make_prices(60, trend=0.001, seed=4)
        config = BacktestConfig(
            commission_pct=0.0,
            slippage_pct=0.0,
            max_drawdown_stop=1.0,
        )
        backtester = SimpleBacktester(config=config)

        def timed_signal(p: pd.Series) -> int:
            n = len(p)
            if n == 1:
                return 1
            if n == 30:
                return -1
            return 0

        result = backtester.run(prices, signal_fn=timed_signal)

        for trade in result.trades:
            assert trade.gross_pnl == pytest.approx(trade.net_pnl, abs=0.01), (
                "With zero costs, gross_pnl must equal net_pnl"
            )


class TestMaxDrawdownStop:
    def test_max_drawdown_stops_trading(self):
        """When drawdown exceeds the limit, the backtest halts."""
        # Create a strongly declining price series
        n = 200
        prices = _make_prices(n, trend=-0.008, seed=5)  # -0.8% per day — steep decline

        config = BacktestConfig(
            max_drawdown_stop=0.15,  # halt at 15% drawdown
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        backtester = SimpleBacktester(config=config)

        # Buy immediately and hold — will hit drawdown limit
        def buy_and_hold(p: pd.Series) -> int:
            return 1 if len(p) == 1 else 0

        result = backtester.run(prices, signal_fn=buy_and_hold)

        assert result.halted is True, "Backtest should halt on severe drawdown"
        assert result.equity_curve is not None
        # Equity curve should be shorter than full series when halted
        assert len(result.equity_curve) < n, (
            "Halted backtest should have fewer bars than full series"
        )

    def test_mild_drawdown_does_not_halt(self):
        """A mild drawdown should NOT trigger the halt."""
        prices = _make_prices(100, trend=0.0005, seed=6)  # gentle uptrend

        config = BacktestConfig(
            max_drawdown_stop=0.50,  # very lenient 50% threshold
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        backtester = SimpleBacktester(config=config)

        result = backtester.run(prices, signal_fn=lambda p: 0)

        assert result.halted is False, "Gentle uptrend should not trigger halt"


class TestNoDoubleEntry:
    def test_position_not_double_entered(self):
        """
        Sending two consecutive buy signals must not create two positions.
        A second buy while already long should be ignored.
        """
        prices = _make_prices(60, trend=0.0, seed=7)
        config = BacktestConfig(max_drawdown_stop=1.0)
        backtester = SimpleBacktester(config=config)

        # Signal returns buy for the first 10 bars, then sell, then buy again
        def greedy_signal(p: pd.Series) -> int:
            n = len(p)
            if n <= 10:
                return 1   # repeated buy signals
            if n == 15:
                return -1  # sell
            if n <= 25:
                return 1   # repeated buy again
            if n == 30:
                return -1
            return 0

        result = backtester.run(prices, signal_fn=greedy_signal)

        # Should have exactly 2 trades (one for each buy/sell cycle)
        assert len(result.trades) == 2, (
            f"Expected exactly 2 completed trades, got {len(result.trades)}"
        )


class TestWalkForwardOOS:
    """Walk-forward validation: out-of-sample data must not leak into training."""

    def _split_train_test(
        self,
        prices: pd.Series,
        train_pct: float = 0.70,
    ) -> Tuple[pd.Series, pd.Series]:
        split = int(len(prices) * train_pct)
        return prices.iloc[:split], prices.iloc[split:]

    def test_walkforward_oos_separate(self):
        """
        Verify that test-period (OOS) indices are disjoint from training indices.
        """
        prices = _make_prices(300, seed=8)
        train, test = self._split_train_test(prices, train_pct=0.70)

        train_indices = set(train.index)
        test_indices = set(test.index)

        overlap = train_indices & test_indices
        assert len(overlap) == 0, (
            f"Train/test sets share {len(overlap)} timestamps — "
            "OOS data was used during training window."
        )

    def test_oos_period_comes_after_training(self):
        """OOS (test) period must start after all training dates."""
        prices = _make_prices(300, seed=9)
        train, test = self._split_train_test(prices, train_pct=0.70)

        assert train.index.max() < test.index.min(), (
            "Training period must end before test period begins."
        )

    def test_signal_at_train_boundary_uses_only_train_data(self):
        """
        A signal evaluated on the last training bar must not use any test-period
        price. Confirmed by checking that signal data length == train set size.
        """
        prices = _make_prices(100, seed=10)
        train, test = self._split_train_test(prices, train_pct=0.70)
        backtester = SimpleBacktester()

        # Run backtest on training set only
        backtester.run(train, signal_fn=lambda p: 0)
        data_lengths = backtester.get_signal_data_lengths()

        # The last signal call on the training set should have len(train) data points
        assert data_lengths[-1] == len(train), (
            f"Last signal on training set received {data_lengths[-1]} bars "
            f"but train set has {len(train)} bars."
        )
