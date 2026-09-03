"""
Tests for the REAL event-driven backtester.

This file previously defined its own `SimpleBacktester` stub and tested that,
never importing `EventDrivenBacktester`. It reported passing tests while the
production engine was entirely uncovered — and the stub itself contained an
off-by-one bug, so even its own assertions had begun failing. Nothing here
tested anything that ships.

Every test below imports and exercises the production engine.

THE CENTRAL TEST
----------------
`TestNullHypothesis` feeds the engine data with provably zero predictable
structure (independent geometric random walks) and asserts it reports no edge.
This is the most informative test in the suite: a backtester that manufactures
profit from noise will manufacture it from anything, and every result it
produces is void. It runs across many independent random worlds rather than
one, because a single seed proves nothing.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.backtesting.costs import ZerodhaCostModel
from app.backtesting.engine import EventDrivenBacktester
from app.strategies.base import (
    BaseStrategy,
    Signal,
    SignalDirection,
    StrategyHealth,
)


# ---------------------------------------------------------------------------
# Synthetic data with known properties
# ---------------------------------------------------------------------------

def gbm_panel(
    n_days: int = 500,
    n_symbols: int = 20,
    seed: int = 0,
    annual_vol: float = 0.30,
    annual_drift: float = 0.0,
) -> pd.DataFrame:
    """
    Independent geometric Brownian motions.

    Log returns are IID by construction: no autocorrelation, no
    cross-sectional predictability. No strategy can have a real edge here.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    lr = rng.normal(
        (annual_drift - 0.5 * annual_vol**2) * dt,
        annual_vol * np.sqrt(dt),
        size=(n_days, n_symbols),
    )
    px = 100.0 * np.exp(np.cumsum(lr, axis=0))
    return pd.DataFrame(
        px,
        index=pd.bdate_range("2015-01-01", periods=n_days),
        columns=[f"S{i:02d}" for i in range(n_symbols)],
    )


class MomentumStrategy(BaseStrategy):
    """
    Minimal 21-day momentum strategy used as a probe.

    Deliberately plausible-looking: on real data it might do something; on
    random data it must not.
    """

    def __init__(self, top_k: int = 3, stop_pct: float = 0.05,
                 target_pct: float = 0.10, weight: float = 0.15):
        self.top_k = top_k
        self.stop_pct = stop_pct
        self.target_pct = target_pct
        self.weight = weight
        self._name = "probe_momentum"
        self._health = StrategyHealth.HEALTHY

    @property
    def name(self) -> str:
        return self._name

    @property
    def holding_period(self) -> str:
        return "2_to_20_days"

    @property
    def health(self) -> StrategyHealth:
        return self._health

    def generate_signals(self, universe, features_df, market_data,
                         existing_positions=None):
        existing_positions = existing_positions or {}
        scores = {}
        for sym in universe:
            md = market_data.get(sym)
            if md is None or len(md) < 25:
                continue
            c = md["close"]
            scores[sym] = float(np.log(c.iloc[-1] / c.iloc[-22]))
        if not scores:
            return []

        now = datetime.now()
        out = []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for sym, sc in ranked[: self.top_k]:
            if sym in existing_positions:
                continue
            out.append(Signal(
                symbol=sym, direction=SignalDirection.LONG,
                strategy_name=self._name, timestamp=now, signal_date=now,
                edge_score=max(sc, 1e-6), expected_return=sc,
                expected_return_std=0.02, stop_loss_pct=self.stop_pct,
                target_pct=self.target_pct, holding_period_days=10,
            ))
        return out

    def calculate_position_size(self, signal, available_capital, risk_engine=None):
        return available_capital * self.weight

    def should_exit(self, position, current_data):
        px = current_data["price"]
        if px <= position["stop_loss"] or px >= position["target"]:
            return True
        return (current_data["date"] - position["entry_date"]).days >= 15


def _no_features(prices, volume, nifty):
    return pd.DataFrame(index=prices.index)


def run_bt(prices: pd.DataFrame, strategy=None, **kwargs):
    bt = EventDrivenBacktester(cost_model=ZerodhaCostModel(), **kwargs)
    return bt.run(
        strategy=strategy if strategy is not None else MomentumStrategy(),
        universe=list(prices.columns),
        prices_df=prices,
        features_fn=_no_features,
        cost_model=ZerodhaCostModel(),
        initial_capital=1_000_000.0,
    )


# ===========================================================================
# THE NULL HYPOTHESIS TEST
# ===========================================================================

class TestNullHypothesis:
    """The engine must not create edge that does not exist in the data."""

    def test_no_gross_edge_across_many_random_worlds(self):
        """
        Across independent random worlds, gross returns must be centred near
        zero. A consistent positive gross return on IID data means the engine
        is peeking at future prices.
        """
        gross = [
            run_bt(gbm_panel(n_days=500, n_symbols=20, seed=s)).metrics.gross_return
            for s in range(15)
        ]
        g = np.asarray(gross)
        frac_positive = float((g > 0).mean())

        assert frac_positive <= 0.75, (
            f"{frac_positive:.0%} of random worlds were profitable GROSS. "
            f"On IID data this should be near chance. Mean gross return "
            f"{g.mean():+.4f}. Suspect look-ahead in the execution path."
        )
        assert g.mean() < 0.05, (
            f"mean gross return on pure noise was {g.mean():+.4f}; the "
            f"engine appears to manufacture edge."
        )

    def test_net_returns_are_negative_after_costs_on_noise(self):
        """
        With no edge and real transaction costs, trading must lose money.
        A net-profitable result on noise would prove a costing bug.
        """
        nets = [
            run_bt(gbm_panel(n_days=500, n_symbols=20, seed=s)).metrics.net_return
            for s in range(10)
        ]
        n = np.asarray(nets)
        assert n.mean() < 0, (
            f"mean NET return on pure noise was {n.mean():+.4f}. "
            f"Trading noise must lose money after costs."
        )

    def test_costs_always_reduce_returns(self):
        for seed in range(5):
            m = run_bt(gbm_panel(seed=seed)).metrics
            if m.num_trades > 0:
                assert m.total_costs > 0
                assert m.net_return < m.gross_return


# ===========================================================================
# Execution mechanics
# ===========================================================================

class TestExecutionMechanics:

    def test_quantities_are_whole_shares(self):
        """Indian equities cannot be traded fractionally."""
        res = run_bt(gbm_panel(seed=1))
        assert res.metrics.num_trades > 0
        for q in res.trades["qty"]:
            assert float(q).is_integer(), f"fractional quantity {q}"

    def test_equity_never_goes_non_positive(self):
        res = run_bt(gbm_panel(seed=3))
        assert (res.equity_curve > 0).all()

    def test_slippage_multiplier_increases_costs(self):
        """Degradation stress test: worse execution must reduce returns."""
        prices = gbm_panel(seed=4)
        base = run_bt(prices, slippage_multiplier=1.0)
        worse = run_bt(prices, slippage_multiplier=3.0)
        assert worse.metrics.net_return < base.metrics.net_return, (
            f"3x slippage did not reduce returns: "
            f"{base.metrics.net_return:+.4f} -> {worse.metrics.net_return:+.4f}"
        )

    def test_product_setting_actually_changes_costs(self):
        """
        MIS and CNC have materially different cost schedules. Hardcoding CNC
        (the previous behaviour) overstates intraday round-trip cost by
        roughly 2.5x and would reject viable intraday strategies.
        """
        prices = gbm_panel(seed=5)
        cnc = run_bt(prices, product="CNC")
        mis = run_bt(prices, product="MIS")
        assert cnc.metrics.total_costs != mis.metrics.total_costs, (
            "product setting had no effect on costs — it is being ignored"
        )

    def test_large_cap_tier_is_reachable(self):
        """
        The large-cap slippage tier was previously dead: no call site passed
        the symbol list, and the max-slippage cap truncated the small-cap tier
        so every symbol received identical slippage.
        """
        prices = gbm_panel(seed=14)
        wide = run_bt(prices, large_cap_symbols=None)
        tight = run_bt(prices, large_cap_symbols=list(prices.columns))
        assert tight.metrics.net_return != wide.metrics.net_return, (
            "large_cap_symbols had no effect — the tier is still dead"
        )


# ===========================================================================
# Failure modes must be loud
# ===========================================================================

class TestFailuresAreLoud:

    def test_broken_feature_function_raises(self):
        """
        A feature pipeline that throws on every bar previously produced a flat
        equity curve indistinguishable from 'the strategy declined to trade'.
        It must fail instead.
        """
        def exploding_features(prices, volume, nifty):
            raise ValueError("simulated feature pipeline failure")

        prices = gbm_panel(seed=6)
        bt = EventDrivenBacktester(cost_model=ZerodhaCostModel())
        with pytest.raises(RuntimeError, match="features_fn failed"):
            bt.run(
                strategy=MomentumStrategy(),
                universe=list(prices.columns),
                prices_df=prices,
                features_fn=exploding_features,
                cost_model=ZerodhaCostModel(),
                initial_capital=1_000_000.0,
            )

    def test_broken_strategy_raises(self):
        class BrokenStrategy(MomentumStrategy):
            def generate_signals(self, *a, **k):
                raise ValueError("simulated strategy failure")

        prices = gbm_panel(seed=7)
        bt = EventDrivenBacktester(cost_model=ZerodhaCostModel())
        with pytest.raises(RuntimeError, match="generate_signals failed"):
            bt.run(
                strategy=BrokenStrategy(),
                universe=list(prices.columns),
                prices_df=prices,
                features_fn=_no_features,
                cost_model=ZerodhaCostModel(),
                initial_capital=1_000_000.0,
            )

    def test_insufficient_data_raises(self):
        tiny = gbm_panel(n_days=1, n_symbols=3, seed=8)
        with pytest.raises(ValueError, match="Insufficient trading days"):
            run_bt(tiny)


# ===========================================================================
# Risk controls
# ===========================================================================

class TestRiskControls:

    def test_drawdown_halt_stops_the_simulation(self):
        """A catastrophic decline must halt the run, not trade through it."""
        prices = gbm_panel(n_days=500, n_symbols=10, seed=9)
        prices = prices.mul(np.linspace(1.0, 0.2, len(prices)), axis=0)

        bt = EventDrivenBacktester(
            cost_model=ZerodhaCostModel(), max_drawdown_halt=0.25
        )
        res = bt.run(
            strategy=MomentumStrategy(weight=0.9),
            universe=list(prices.columns),
            prices_df=prices,
            features_fn=_no_features,
            cost_model=ZerodhaCostModel(),
            initial_capital=1_000_000.0,
        )
        assert res.equity_curve.index[-1] < prices.index[-1], (
            "backtest ran to the end despite breaching the drawdown halt"
        )


# ===========================================================================
# Metrics correctness
# ===========================================================================

class TestMetrics:

    def test_metrics_are_internally_consistent(self):
        res = run_bt(gbm_panel(seed=11))
        m = res.metrics
        final = res.equity_curve.iloc[-1]
        assert np.isclose(m.total_return, final / 1_000_000.0 - 1.0, atol=1e-6)
        assert m.annualized_volatility >= 0
        assert -1.0 <= m.max_drawdown <= 1.0
        assert m.num_trades == len(res.trades)

    def test_zero_trade_run_does_not_crash(self):
        class NeverTrades(MomentumStrategy):
            def generate_signals(self, *a, **k):
                return []

        res = run_bt(gbm_panel(seed=12), strategy=NeverTrades())
        assert res.metrics.num_trades == 0
        assert np.isclose(res.equity_curve.iloc[-1], 1_000_000.0)

    def test_deterministic_given_same_inputs(self):
        prices = gbm_panel(seed=13)
        a = run_bt(prices).metrics
        b = run_bt(prices).metrics
        assert a.net_return == b.net_return
        assert a.num_trades == b.num_trades
