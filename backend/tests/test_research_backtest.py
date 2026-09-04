"""
The backtester's timing, costs and metrics.

Two of these tests exist because the bugs they describe were REAL and shipped
in the first version of this module:

* `test_BUG_forward_returns_are_not_identically_zero` — the entry and exit
  shifts resolved to the same bar, so every forward return was exactly 0.0.
  Every baseline came back with a Sharpe near -6.5, which is what a portfolio
  paying costs and earning nothing looks like.
* `test_BUG_returns_are_indexed_by_realisation_date` — returns were indexed by
  SIGNAL date while the benchmark was indexed by realisation date. The 2-bar
  offset gave a long-only equity portfolio a beta of 0.02.

Both were caught by a number being implausible rather than by a test, which is
why they now have tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.backtest import (
    LookaheadError,
    assert_no_lookahead,
    build_forward_returns,
    compute_metrics,
    cross_sectional_weights,
    max_drawdown,
    run_backtest,
)


def make_prices(n: int = 800, k: int = 20, seed: int = 7) -> pd.DataFrame:
    """Deterministic geometric random walks with cross-sectional drift spread."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    out = {}
    for i in range(k):
        drift = (i - k / 2) * 0.0004
        steps = drift + rng.normal(0, 0.012, n)
        out[f"S{i:02d}"] = 100.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(out, index=idx)


class TestForwardReturns:

    def test_BUG_forward_returns_are_not_identically_zero(self):
        """
        The entry and exit shifts must not resolve to the same bar.

        They did: `shift(-lag - horizon + 1) / shift(-lag)` with the default
        horizon=1, lag=2 is `shift(-2)/shift(-2)`. Every backtest through it
        measured nothing but transaction costs.
        """
        px = make_prices()
        fwd = build_forward_returns(px, horizon=1, lag=2)
        nonzero = int((fwd.fillna(0.0) != 0.0).to_numpy().sum())
        assert nonzero > 0, "every forward return is exactly zero"
        assert nonzero > 0.9 * fwd.notna().to_numpy().sum()

    def test_the_return_is_exactly_t1_to_t2(self):
        """Signal at t, execute at t+1, earn t+1 -> t+2. Checked by hand."""
        px = make_prices()
        fwd = build_forward_returns(px, horizon=1, lag=2)
        s = px.iloc[:, 0]
        manual = s.shift(-2) / s.shift(-1) - 1.0
        assert np.allclose(fwd.iloc[:, 0].dropna(), manual.dropna())

    def test_a_longer_horizon_spans_more_bars(self):
        px = make_prices()
        s = px.iloc[:, 0]
        f5 = build_forward_returns(px, horizon=5, lag=2).iloc[:, 0]
        manual = s.shift(-6) / s.shift(-1) - 1.0
        assert np.allclose(f5.dropna(), manual.dropna())

    def test_a_zero_lag_is_refused(self):
        """lag=0 executes at the same close that produced the signal."""
        with pytest.raises(ValueError, match="lag must be"):
            build_forward_returns(make_prices(), lag=0)

    def test_the_tail_has_no_realised_return(self):
        fwd = build_forward_returns(make_prices(), horizon=1, lag=2)
        assert fwd.iloc[-2:].notna().to_numpy().sum() == 0


class TestLookaheadGuard:

    def test_a_correctly_lagged_frame_is_accepted(self):
        px = make_prices()
        sig = px.pct_change(21)
        assert_no_lookahead(sig, build_forward_returns(px, lag=2), lag=2)

    def test_an_insufficiently_lagged_frame_is_rejected(self):
        px = make_prices()
        sig = px.pct_change(21)
        with pytest.raises(LookaheadError):
            assert_no_lookahead(sig, build_forward_returns(px, lag=1), lag=2)

    def test_an_all_zero_frame_is_rejected(self):
        """It passes every shift check and still measures only costs."""
        px = make_prices()
        zeros = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        zeros.iloc[-2:] = np.nan
        with pytest.raises(LookaheadError, match="identically zero"):
            assert_no_lookahead(px.pct_change(21), zeros, lag=2)

    def test_misaligned_indexes_are_rejected(self):
        px = make_prices()
        fwd = build_forward_returns(px, lag=2)
        with pytest.raises(LookaheadError, match="not aligned"):
            assert_no_lookahead(px.pct_change(21).iloc[5:], fwd, lag=2)


class TestReturnAlignment:

    def test_BUG_returns_are_indexed_by_realisation_date(self):
        """
        A long-only equity book must have a beta near 1, not near 0.

        Indexed by signal date, the series sat 2 bars ahead of every other
        date-indexed series and the measured beta was 0.02.
        """
        px = make_prices(n=1200, k=30)
        equal_weight = px.pct_change().mean(axis=1).dropna()
        sig = px.pct_change(21)

        res = run_backtest(sig, px, benchmark=equal_weight, cost_bps=0.0)
        beta = res.metrics.beta

        assert beta is not None
        assert 0.4 < beta < 1.6, (
            f"beta {beta:.3f} for a long-only book drawn from the same universe "
            "as the benchmark — the return series is misaligned in time"
        )

    def test_the_series_starts_after_the_signal_it_came_from(self):
        px = make_prices()
        res = run_backtest(px.pct_change(21), px, cost_bps=0.0)
        assert res.returns_net.index[0] > px.index[0]


class TestWeights:

    def test_gross_exposure_never_exceeds_one(self):
        """Never levered. Under-invested is allowed; over-invested is not."""
        px = make_prices()
        w = cross_sectional_weights(px.pct_change(21), rebalance_days=1)
        assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9

    def test_a_wide_cross_section_is_fully_invested(self):
        """With enough names to satisfy the cap, gross reaches exactly 1."""
        px = make_prices(k=60)
        w = cross_sectional_weights(
            px.pct_change(21), top_quantile=0.2, max_weight=0.10, rebalance_days=1
        )
        gross = w.abs().sum(axis=1)
        live = gross[gross > 0]
        assert np.allclose(live, 1.0, atol=1e-9)

    def test_a_thin_cross_section_holds_cash_rather_than_breaching_the_cap(self):
        """
        4 names at a 10% cap is 40% invested, not 4 positions of 25%.

        Holding cash because there are not enough names to diversify into is a
        real portfolio decision; quietly breaching the cap is not.
        """
        px = make_prices(k=20)
        w = cross_sectional_weights(
            px.pct_change(21), top_quantile=0.2, max_weight=0.10, rebalance_days=1
        )
        gross = w.abs().sum(axis=1)
        live = gross[gross > 0]
        assert live.max() < 1.0
        assert w.max().max() <= 0.10 + 1e-9

    def test_long_only_never_shorts(self):
        px = make_prices()
        w = cross_sectional_weights(px.pct_change(21), long_only=True)
        assert (w >= -1e-12).all().all()

    def test_the_position_cap_binds(self):
        px = make_prices(k=12)
        w = cross_sectional_weights(
            px.pct_change(21), top_quantile=0.1, max_weight=0.15, rebalance_days=1
        )
        assert w.max().max() <= 0.15 + 1e-9

    def test_holding_between_rebalances_reduces_turnover(self):
        """A weekly strategy must not be measured at daily turnover."""
        px = make_prices()
        sig = px.pct_change(21)
        daily = cross_sectional_weights(sig, rebalance_days=1)
        weekly = cross_sectional_weights(sig, rebalance_days=5)
        t_daily = (daily - daily.shift(1)).abs().sum(axis=1).mean()
        t_weekly = (weekly - weekly.shift(1)).abs().sum(axis=1).mean()
        assert t_weekly < t_daily


class TestCosts:

    def test_higher_costs_never_improve_the_result(self):
        px = make_prices()
        sig = px.pct_change(21)
        prev = np.inf
        for bps in (0.0, 10.0, 25.0, 50.0, 100.0):
            cagr = run_backtest(sig, px, cost_bps=bps).metrics.cagr
            assert cagr <= prev + 1e-9, f"cost {bps}bps improved CAGR"
            prev = cagr

    def test_cost_drag_matches_the_gross_net_gap(self):
        px = make_prices()
        res = run_backtest(px.pct_change(21), px, cost_bps=25.0)
        gap = (
            res.returns_gross.mean() - res.returns_net.mean()
        ) * 252
        assert abs(res.metrics.cost_drag_annual - gap) < 1e-6

    def test_zero_cost_leaves_gross_and_net_identical(self):
        px = make_prices()
        res = run_backtest(px.pct_change(21), px, cost_bps=0.0)
        assert np.allclose(res.returns_net, res.returns_gross)

    def test_building_the_first_book_is_not_free(self):
        """
        Going from flat to invested is a real trade and must be charged.

        Not at row 0 — the signal has a warm-up during which weights are zero.
        The charge lands on the first row that actually takes a position.
        """
        px = make_prices()
        res = run_backtest(px.pct_change(21), px, cost_bps=25.0)
        first_live = (res.weights.abs().sum(axis=1) > 0).idxmax()
        assert res.turnover.loc[first_live] > 0, (
            "the first move from cash into a full book was charged nothing"
        )


class TestMetrics:

    def test_max_drawdown_on_a_known_path(self):
        r = pd.Series([0.10, -0.50, 0.10])
        assert max_drawdown(r) == pytest.approx(-0.5, abs=1e-9)

    def test_a_monotonic_riser_has_no_drawdown(self):
        assert max_drawdown(pd.Series([0.01] * 50)) == pytest.approx(0.0, abs=1e-12)

    def test_sharpe_of_a_constant_series_is_not_infinite(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        r = pd.Series(0.001, index=idx)
        m = compute_metrics(r, r, pd.DataFrame(index=idx), pd.Series(0.0, index=idx))
        assert np.isfinite(m.sharpe) or m.sharpe == 0.0

    def test_empty_returns_do_not_raise(self):
        m = compute_metrics(
            pd.Series(dtype=float), pd.Series(dtype=float),
            pd.DataFrame(), pd.Series(dtype=float),
        )
        assert m.n_periods == 0
