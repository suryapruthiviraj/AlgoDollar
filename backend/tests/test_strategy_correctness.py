"""
test_strategy_correctness.py — regression tests for verified strategy defects.

Each test names the defect it pins.  They are deliberately behavioural (drive
the public API, assert on emitted Signals) rather than unit tests of private
helpers, because every one of these defects was invisible at the unit level
and only showed up in the numbers that reached downstream consumers.
"""
from __future__ import annotations

import os
import time as time_mod
import warnings
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.strategies.base import (
    DELIVERY_ROUND_TRIP_COST,
    INTRADAY_ROUND_TRIP_COST,
    MAX_GROSS_EXPOSURE,
    PerformanceMetrics,
    RiskEngineError,
    Signal,
    SignalDirection,
    StrategyHealth,
    _annualized_sharpe_se,
)
from app.strategies.intraday import (
    _CONTINUATION_COEF,
    _SQUARE_OFF_TIME,
    IST,
    IntradayStrategy,
    NaiveDatetimeError,
    now_ist,
)
from app.strategies.longterm import (
    LongtermStrategy,
    MockDataInLiveModeError,
    _MockFundamentalProvider,
)
from app.strategies.swing import SwingStrategy

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_universe(n_syms=50, n_days=300, seed=7, drift=0.0, vol=0.018):
    """Synthetic daily OHLCV panel. drift=0.0 => a pure random walk, no edge."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    md = {}
    for i in range(n_syms):
        rets = rng.normal(drift, vol, n_days)
        close = 1000 * np.exp(np.cumsum(rets))
        md[f"S{i:02d}"] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": rng.integers(1e5, 1e6, n_days),
            },
            index=dates,
        )
    return list(md.keys()), md


def make_intraday(n_syms=6, n_bars=120, seed=3, price_drift=0.0, vol_spike=1.0):
    """Synthetic 1-minute bars for one session (09:15 IST onwards)."""
    rng = np.random.default_rng(seed)
    start = datetime(2024, 5, 2, 9, 15, tzinfo=IST)
    stamps = [start + timedelta(minutes=i) for i in range(n_bars)]
    data = {}
    for i in range(n_syms):
        rets = rng.normal(price_drift, 0.0008, n_bars)
        close = 500 * np.exp(np.cumsum(rets))
        vols = np.full(n_bars, 5000.0)
        vols[-1] *= vol_spike
        data[f"I{i:02d}"] = pd.DataFrame({
            "time": stamps,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": vols,
        })
    return list(data.keys()), data


class ApprovingRiskEngine:
    def approve_trade(self, symbol, size, sleeve):
        return True


class ExplodingRiskEngine:
    """A broken risk engine — the thing that used to be silently ignored."""

    def approve_trade(self, symbol, size, sleeve):
        raise ConnectionError("risk service unreachable")


def entry_signal(symbol="X", **kw):
    """A minimal valid ENTRY signal."""
    defaults = dict(
        symbol=symbol,
        direction=SignalDirection.LONG,
        strategy_name="test",
        timestamp=datetime(2024, 5, 2),
        signal_date=datetime(2024, 5, 2),
        edge_score=0.01,
        expected_return=0.012,
        expected_return_std=0.02,
        stop_loss_pct=0.02,
        target_pct=0.04,
        holding_period_days=5,
        feature_snapshot={"realized_vol_21d": 0.25, "realized_vol_63d": 0.25},
    )
    defaults.update(kw)
    return Signal(**defaults)


# ===========================================================================
# DEFECT 1 — swing reported a 12-month return as a 5-day expected return
# ===========================================================================

def test_defect1_swing_edge_near_zero_on_zero_drift_random_walk():
    """
    A zero-drift random walk has no edge BY CONSTRUCTION.  The old code scored
    it with raw 12-1 momentum (a 231-trading-day number) and emitted signals
    claiming a 41-71% FIVE-DAY return.
    """
    uni, md = make_universe(drift=0.0)
    strat = SwingStrategy(paper_mode=True)

    scores = strat._score_universe(uni, pd.DataFrame(), md, None)
    assert not scores.empty
    assert scores["score_source"].eq("momentum_prior").all()

    # The score itself must be a plausible 5-day return, not an annual one.
    assert scores["raw_score"].abs().max() < 0.02, (
        f"raw 5-day score {scores['raw_score'].abs().max():.4f} is implausibly "
        "large for a zero-drift random walk"
    )

    signals = strat.generate_signals(uni, pd.DataFrame(), md)
    for sig in signals:
        assert abs(sig.edge_score) < 0.02, (
            f"{sig.symbol}: edge_score {sig.edge_score:.4f} on a no-edge universe"
        )
        assert abs(sig.expected_return) < 0.02


def test_defect1_expected_return_is_on_the_signals_own_horizon():
    """expected_return must scale with holding_period_days, not be horizon-free."""
    uni, md = make_universe(drift=0.0003, seed=21)

    s5 = SwingStrategy(paper_mode=True, horizon_days=5)
    s20 = SwingStrategy(paper_mode=True, horizon_days=20)
    sc5 = s5._score_universe(uni, pd.DataFrame(), md, None)
    sc20 = s20._score_universe(uni, pd.DataFrame(), md, None)

    # sigma scales with sqrt(T), so a 20-day forecast is ~2x a 5-day one.
    ratio = sc20["raw_score"].abs().max() / sc5["raw_score"].abs().max()
    assert 1.9 < ratio < 2.1, f"horizon scaling ratio {ratio:.3f} != ~2"

    for sig in s20.generate_signals(uni, pd.DataFrame(), md):
        assert sig.holding_period_days == 20
        assert sig.metadata["return_units"] == "simple_return_over_20_trading_days"
        # edge = gross - cost, both round-trip fractions of notional
        assert sig.edge_score == pytest.approx(
            sig.expected_return - DELIVERY_ROUND_TRIP_COST, abs=1e-12
        )


def test_defect1_alpha_model_emitting_annual_returns_is_rejected():
    """A model whose predictions cannot be 5-day returns is a units bug."""
    from app.strategies.swing import SignalUnitsError

    uni, md = make_universe(n_syms=5, seed=2)
    features = pd.DataFrame(
        {f"f{j}__{s}": [0.5] for s in uni for j in range(3)}
    )

    class AnnualReturnModel:
        def predict(self, X):
            return np.full(X.shape[0], 0.42)  # 42% — plainly not a 5-day number

    strat = SwingStrategy(alpha_model=AnnualReturnModel(), paper_mode=True)
    with pytest.raises(SignalUnitsError):
        strat._score_universe(uni, features, md, [f"f{j}" for j in range(3)])


# ===========================================================================
# DEFECT 2 — fictional, non-reproducible mock fundamentals
# ===========================================================================

def test_defect2_mock_fundamentals_are_deterministic_per_symbol():
    """
    The class-level RNG advanced on every call, so RELIANCE returned
    roe 28.2 then 21.6, pe 17.2 then 64.0 — buy/sell decisions were coin flips.
    """
    a = _MockFundamentalProvider.get_fundamentals("RELIANCE")
    _ = _MockFundamentalProvider.get_fundamentals("TCS")  # interleave another symbol
    b = _MockFundamentalProvider.get_fundamentals("RELIANCE")
    assert a == b, "mock fundamentals must be reproducible per symbol"

    m1 = _MockFundamentalProvider.get_sector_medians("IT")
    m2 = _MockFundamentalProvider.get_sector_medians("IT")
    assert m1 == m2

    # ...and different symbols must still differ (not a constant stub).
    assert a != _MockFundamentalProvider.get_fundamentals("TCS")


def test_defect2_composite_score_is_stable_across_repeated_scoring():
    """The same stock straddled the sell threshold across consecutive calls."""
    uni, md = make_universe(n_syms=3, seed=5)
    lt = LongtermStrategy(paper_mode=True)
    scores = [
        lt._compute_composite(uni[0], pd.DataFrame(), md)["composite"]
        for _ in range(3)
    ]
    assert scores[0] == scores[1] == scores[2], f"unstable composite: {scores}"


def test_defect2_mock_data_in_live_mode_raises():
    """Synthetic fundamentals must never be able to drive a real order."""
    # Cannot even be constructed live.
    with pytest.raises(MockDataInLiveModeError):
        LongtermStrategy(paper_mode=False)

    # ...and flipping the flag after construction does not defeat the guard.
    lt = LongtermStrategy(paper_mode=True)
    assert lt.uses_mock_data and lt.data_source == "MOCK"
    uni, md = make_universe(n_syms=25, seed=9)

    lt.paper_mode = False
    with pytest.raises(MockDataInLiveModeError):
        lt.generate_signals(uni, pd.DataFrame(), md)
    with pytest.raises(MockDataInLiveModeError):
        lt.calculate_position_size(entry_signal(), 1_000_000, None)


def test_defect2_real_provider_runs_live_without_raising():
    """The guard keys on the provider, not on paper_mode alone."""

    class RealProvider:
        IS_MOCK = False
        DATA_SOURCE = "screener.in"

        def get_fundamentals(self, symbol):
            return {"roe": 20.0, "roce": 22.0, "pe_ratio": 20.0, "sector": "IT"}

        def get_sector_medians(self, sector):
            return {"pe_median": 22.0, "roe_median": 15.0}

    lt = LongtermStrategy(fundamental_provider=RealProvider(), paper_mode=False)
    assert not lt.uses_mock_data
    assert lt.data_source == "screener.in"


# ===========================================================================
# DEFECT 3 — intraday used naive local time (no square-off on a UTC host)
# ===========================================================================

@pytest.fixture
def utc_system_clock():
    """Force the PROCESS's local time zone to UTC, like a cloud host."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time_mod.tzset()
    yield
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time_mod.tzset()


def test_defect3_square_off_fires_at_1515_ist_on_a_utc_host(utc_system_clock):
    """
    15:15 IST == 09:45 UTC.  With naive local time the old code compared 09:45
    against 15:15 and returned False: the intraday book was carried overnight.
    """
    assert datetime.now().tzinfo is None          # the host clock really is naive
    assert now_ist().utcoffset() == timedelta(hours=5, minutes=30)

    strat = IntradayStrategy(paper_mode=True)
    position = {
        "symbol": "I00", "entry_price": 500.0, "direction": "LONG",
        "stop_loss": 480.0, "target": 520.0,
    }
    square_off_utc = datetime(2024, 5, 2, 9, 45, tzinfo=timezone.utc)
    assert square_off_utc.astimezone(IST).time() == _SQUARE_OFF_TIME

    assert strat.should_exit(position, {"price": 500.0, "time": square_off_utc}) is True

    # ...and a UTC stamp before the cut-off must NOT square off.
    before = datetime(2024, 5, 2, 9, 30, tzinfo=timezone.utc)  # 15:00 IST
    assert strat.should_exit(position, {"price": 500.0, "time": before}) is False


def test_defect3_session_gates_use_ist_not_host_time(utc_system_clock):
    """11:00 IST (05:30 UTC) is mid-session and must NOT be 'opening window'."""
    uni, idata = make_intraday()
    strat = IntradayStrategy(paper_mode=True)

    mid_session_utc = datetime(2024, 5, 2, 5, 30, tzinfo=timezone.utc)  # 11:00 IST
    assert mid_session_utc.astimezone(IST).time() == time(11, 0)
    # Not blocked by the opening-window gate (may still return [] on no edge).
    assert isinstance(
        strat.generate_signals(uni, pd.DataFrame(), {}, current_time=mid_session_utc,
                               intraday_data=idata),
        list,
    )

    # 15:20 IST (09:50 UTC) is past the 14:45 cutoff — no new positions.
    past_cutoff_utc = datetime(2024, 5, 2, 9, 50, tzinfo=timezone.utc)
    assert strat.generate_signals(uni, pd.DataFrame(), {},
                                  current_time=past_cutoff_utc,
                                  intraday_data=idata) == []


def test_defect3_naive_datetime_raises_rather_than_being_assumed():
    strat = IntradayStrategy(paper_mode=True)
    position = {"symbol": "I00", "entry_price": 500.0, "direction": "LONG",
                "stop_loss": 480.0, "target": 520.0}

    with pytest.raises(NaiveDatetimeError):
        strat.should_exit(position, {"price": 500.0,
                                     "time": datetime(2024, 5, 2, 15, 20)})
    with pytest.raises(NaiveDatetimeError):
        strat.should_exit(position, {"price": 500.0, "time": time(15, 20)})
    with pytest.raises(NaiveDatetimeError):
        strat.generate_signals([], pd.DataFrame(), {},
                               current_time=datetime(2024, 5, 2, 11, 0))


# ===========================================================================
# DEFECT 4 — long-term emitted BUYs while over the position limit
# ===========================================================================

def test_defect4_over_position_limit_yields_zero_buy_signals():
    """
    max_positions=20 holding 30 => available_slots=-10, and `list[:-10]`
    returns items rather than [] — 18 BUY signals were emitted.
    """
    uni, md = make_universe(n_syms=60, seed=13)
    held = {s: {"value": 100.0} for s in uni[:30]}
    lt = LongtermStrategy(paper_mode=True, max_positions=20)

    signals = lt.generate_signals(uni, pd.DataFrame(), md, existing_positions=held)
    buys = [s for s in signals if s.direction == SignalDirection.LONG]
    assert buys == [], f"emitted {len(buys)} BUYs while 30/20 positions held"


def test_defect4_exactly_at_limit_also_yields_zero_buys():
    uni, md = make_universe(n_syms=60, seed=14)
    held = {s: {"value": 100.0} for s in uni[:20]}
    lt = LongtermStrategy(paper_mode=True, max_positions=20)
    buys = [
        s for s in lt.generate_signals(uni, pd.DataFrame(), md,
                                       existing_positions=held)
        if s.direction == SignalDirection.LONG
    ]
    assert buys == []


def test_defect4_below_limit_still_buys():
    """The no-trade guard must not become an always-no-trade strategy."""
    uni, md = make_universe(n_syms=60, seed=15)
    lt = LongtermStrategy(paper_mode=True, max_positions=20)
    buys = [
        s for s in lt.generate_signals(uni, pd.DataFrame(), md)
        if s.direction == SignalDirection.LONG
    ]
    assert len(buys) > 0
    assert len(buys) <= 20


# ===========================================================================
# DEFECT 5 — no portfolio budget (2-3x silent leverage)
# ===========================================================================

CAPITAL = 10_000_000.0


def test_defect5_swing_position_sizes_never_exceed_sleeve_capital():
    """4 swing signals allocated 112% of capital; a full book was 300%."""
    uni, md = make_universe(n_syms=60, seed=17, drift=0.0006)
    strat = SwingStrategy(paper_mode=True, horizon_days=20)
    signals = strat.generate_signals(uni, pd.DataFrame(), md)
    assert signals, "need signals to make this assertion meaningful"

    total = sum(strat.calculate_position_size(s, CAPITAL, None) for s in signals)
    assert total <= CAPITAL * MAX_GROSS_EXPOSURE + 1e-6, (
        f"swing allocated {total / CAPITAL:.1%} of sleeve capital"
    )

    batch = strat.allocate_capital(signals, CAPITAL, None)
    assert sum(batch.values()) <= CAPITAL + 1e-6


def test_defect5_longterm_position_sizes_never_exceed_sleeve_capital():
    """20 longterm names x the 10% per-position cap was 200% gross."""
    uni, md = make_universe(n_syms=80, seed=19)
    lt = LongtermStrategy(paper_mode=True, max_positions=20)
    signals = [
        s for s in lt.generate_signals(uni, pd.DataFrame(), md)
        if s.direction == SignalDirection.LONG
    ]
    assert signals

    total = sum(lt.calculate_position_size(s, CAPITAL, None) for s in signals)
    assert total <= CAPITAL * MAX_GROSS_EXPOSURE + 1e-6, (
        f"longterm allocated {total / CAPITAL:.1%} of sleeve capital"
    )
    assert sum(lt.allocate_capital(signals, CAPITAL, None).values()) <= CAPITAL + 1e-6


def test_defect5_unstamped_signals_are_capped_so_a_full_book_fits():
    """Signals from another producer must still not blow the budget."""
    for strat, n in (
        (SwingStrategy(paper_mode=True), 10),
        (LongtermStrategy(paper_mode=True, max_positions=20), 20),
        (IntradayStrategy(paper_mode=True), 5),
    ):
        sigs = [entry_signal(symbol=f"U{i}") for i in range(n)]
        total = sum(strat.calculate_position_size(s, CAPITAL, None) for s in sigs)
        assert total <= CAPITAL + 1e-6, f"{strat.name} sized a full book at {total/CAPITAL:.1%}"
        assert sum(strat.allocate_capital(sigs, CAPITAL, None).values()) <= CAPITAL + 1e-6


def test_defect5_existing_positions_consume_the_budget():
    """A half-full book leaves at most half the sleeve for new signals."""
    uni, md = make_universe(n_syms=60, seed=23, drift=0.0006)
    strat = SwingStrategy(paper_mode=True, horizon_days=20, max_positions=10)
    held = {s: {"value": 1.0} for s in uni[:5]}
    signals = strat.generate_signals(uni, pd.DataFrame(), md, existing_positions=held)
    total = sum(strat.calculate_position_size(s, CAPITAL, None) for s in signals)
    assert total <= CAPITAL * 0.5 + 1e-6


@pytest.mark.parametrize(
    "strategy",
    [
        SwingStrategy(paper_mode=True),
        IntradayStrategy(paper_mode=True),
        LongtermStrategy(paper_mode=True),
    ],
    ids=["swing", "intraday", "longterm"],
)
def test_defect5_risk_engine_exception_blocks_the_trade(strategy):
    """
    `except Exception: pass` made a broken risk engine FAIL OPEN — every trade
    was permitted.  It must now fail closed, loudly.
    """
    sig = entry_signal()
    with pytest.raises(RiskEngineError):
        strategy.calculate_position_size(sig, CAPITAL, ExplodingRiskEngine())

    # A risk engine that cannot answer at all is also a block.
    class Useless:
        pass

    with pytest.raises(RiskEngineError):
        strategy.calculate_position_size(sig, CAPITAL, Useless())

    # A working engine that says no returns 0; one that says yes sizes.
    class Rejecting:
        def approve_trade(self, symbol, size, sleeve):
            return False

    assert strategy.calculate_position_size(sig, CAPITAL, Rejecting()) == 0.0
    assert strategy.calculate_position_size(sig, CAPITAL, ApprovingRiskEngine()) > 0.0


def test_defect5_risk_engine_exception_propagates_through_batch_sizing():
    strat = SwingStrategy(paper_mode=True)
    with pytest.raises(RiskEngineError):
        strat.allocate_capital([entry_signal()], CAPITAL, ExplodingRiskEngine())


# ===========================================================================
# DEFECT 6 — health auto-DISABLE fired inside the noise band
# ===========================================================================

def test_defect6_no_degradation_on_a_small_sample():
    """
    ~21 observations cannot support a Sharpe-driven degradation decision:
    SE ~ 3.5, so a -1.26 point estimate is noise around a true Sharpe of 1.0.
    """
    strat = SwingStrategy(paper_mode=True)
    for sharpe in (-0.2, -0.8, -1.5, -4.0):
        strat.health = StrategyHealth.HEALTHY
        strat.update_health(PerformanceMetrics(
            rolling_sharpe_30d=sharpe,
            rolling_sharpe_90d=1.0,
            current_drawdown_pct=0.0,
            win_rate_30d=0.55,
            num_trades_30d=12,
            num_observations_30d=21,
        ))
        assert strat.health == StrategyHealth.HEALTHY, (
            f"degraded to {strat.health} on 21 observations at sharpe={sharpe}"
        )


def test_defect6_unknown_sample_size_does_not_degrade():
    """An unspecified n must be treated as 'too small', not as 'infinite'."""
    strat = SwingStrategy(paper_mode=True)
    strat.update_health(PerformanceMetrics(
        rolling_sharpe_30d=-5.0, rolling_sharpe_90d=1.0,
        current_drawdown_pct=0.0, win_rate_30d=0.5, num_trades_30d=0,
    ))
    assert strat.health == StrategyHealth.HEALTHY


def test_defect6_large_sample_with_significant_underperformance_does_degrade():
    """The mechanism must still work when the evidence is real."""
    strat = SwingStrategy(paper_mode=True)
    n, observed = 250, -4.0
    se = _annualized_sharpe_se(observed, n)
    assert observed < -1.0 - 2 * se  # the observation IS >2 SE below the level
    strat.update_health(PerformanceMetrics(
        rolling_sharpe_30d=observed, rolling_sharpe_90d=-2.0,
        current_drawdown_pct=0.0, win_rate_30d=0.2, num_trades_30d=80,
        num_observations_30d=n,
    ))
    assert strat.health == StrategyHealth.DISABLED

    # ...but the same point estimate on the same sample, only 1 SE below the
    # level, is not evidence and must not degrade.
    strat2 = SwingStrategy(paper_mode=True)
    strat2.update_health(PerformanceMetrics(
        rolling_sharpe_30d=-1.0 - se, rolling_sharpe_90d=-1.0,
        current_drawdown_pct=0.0, win_rate_30d=0.2, num_trades_30d=80,
        num_observations_30d=n,
    ))
    assert strat2.health != StrategyHealth.DISABLED


def test_defect6_drawdown_still_degrades_unconditionally():
    """A drawdown is observed, not estimated: it must not need a sample size."""
    strat = SwingStrategy(paper_mode=True)
    strat.update_health(PerformanceMetrics(
        rolling_sharpe_30d=2.0, rolling_sharpe_90d=2.0,
        current_drawdown_pct=0.25, win_rate_30d=0.5, num_trades_30d=3,
        num_observations_30d=5,
    ))
    assert strat.health == StrategyHealth.DISABLED


def test_defect6_disabled_remains_sticky():
    """DISABLED is terminal — including against an accidental 'upgrade'."""
    strat = SwingStrategy(paper_mode=True)
    strat.health = StrategyHealth.DISABLED
    # Mildly bad numbers used to fall into the PAUSED branch and un-disable it.
    strat.update_health(PerformanceMetrics(
        rolling_sharpe_30d=-0.6, rolling_sharpe_90d=-0.6,
        current_drawdown_pct=0.12, win_rate_30d=0.4, num_trades_30d=40,
        num_observations_30d=120,
    ))
    assert strat.health == StrategyHealth.DISABLED
    # Nor does a good month bring it back automatically.
    strat.update_health(PerformanceMetrics(
        rolling_sharpe_30d=2.0, rolling_sharpe_90d=2.0,
        current_drawdown_pct=0.0, win_rate_30d=0.7, num_trades_30d=40,
        num_observations_30d=120,
    ))
    assert strat.health == StrategyHealth.DISABLED


def test_defect6_false_disable_rate_is_small_for_a_profitable_strategy():
    """
    Monte Carlo: a strategy with a TRUE annualized Sharpe of 1.0 was
    auto-DISABLED (manual-only recovery) 28.5% of months.  Cap the false
    monthly rate at 1%, i.e. well under ~12% a year.
    """
    rng = np.random.default_rng(101)
    n_trials, n_obs = 3000, 63
    disabled = 0
    strat = SwingStrategy(paper_mode=True)
    for _ in range(n_trials):
        r = rng.normal(1.0 / np.sqrt(252), 1.0, n_obs)
        sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))
        strat.health = StrategyHealth.HEALTHY
        strat.update_health(PerformanceMetrics(
            rolling_sharpe_30d=sharpe, rolling_sharpe_90d=1.0,
            current_drawdown_pct=0.0, win_rate_30d=0.55, num_trades_30d=20,
            num_observations_30d=n_obs,
        ))
        disabled += strat.health == StrategyHealth.DISABLED
    rate = disabled / n_trials
    assert rate < 0.01, f"false auto-disable rate {rate:.2%} per evaluation"


# ===========================================================================
# DEFECT 7 — intraday mixed units; a volume spike manufactured a return
# ===========================================================================

def _flat_frame(n=60, last_vol=6900.0, last_move=0.0):
    """Flat 1-minute bars with an optional volume spike / tiny final move."""
    stamps = [datetime(2024, 5, 2, 9, 15, tzinfo=IST) + timedelta(minutes=i)
              for i in range(n)]
    close = np.full(n, 500.0)
    close[-1] = 500.0 * (1 + last_move)
    vols = np.full(n, 1000.0)
    vols[-1] = last_vol
    return pd.DataFrame({
        "time": stamps, "open": close, "high": close, "low": close,
        "close": close, "volume": vols,
    })


def test_defect7_volume_spike_alone_does_not_manufacture_expected_return():
    """
    Flat price + rel_vol 6.9 produced expected_return=1.18%, edge_score=0.0103
    and TRADES=True.  With no price movement there is no return to expect.
    """
    now = datetime(2024, 5, 2, 11, 0, tzinfo=IST)
    flat = _flat_frame(last_vol=6900.0)
    rel_vol = flat["volume"].iloc[-1] / flat["volume"].tail(20).mean()
    assert rel_vol > 5

    # No signal at all — and not merely because of the long-only filter.
    for long_only in (True, False):
        strat = IntradayStrategy(paper_mode=True, long_only=long_only)
        assert strat._build_signal("FLAT", flat, pd.DataFrame(), now, True) is None

    # With a genuine but tiny move, the volume spike may scale it — bounded.
    strat = IntradayStrategy(paper_mode=True)
    sig = strat._build_signal("TINY", _flat_frame(last_vol=6900.0, last_move=0.0002),
                              pd.DataFrame(), now, True)
    assert sig is not None
    assert sig.feature_snapshot["rel_vol"] > 5
    assert sig.expected_return < 2e-4, (
        f"a 2 bp move plus a volume spike produced {sig.expected_return:.4%}"
    )
    assert sig.edge_score < 0     # nowhere near the cost hurdle
    assert not sig.is_valid()     # ...and therefore not tradeable


def test_defect7_volume_scales_but_never_creates_a_move():
    """rel_vol is a bounded, multiplicative confidence term."""
    strat = IntradayStrategy(paper_mode=True)
    n = 60
    stamps = [datetime(2024, 5, 2, 9, 15, tzinfo=IST) + timedelta(minutes=i)
              for i in range(n)]
    close = 500 * np.exp(np.cumsum(np.full(n, 0.0004)))  # a genuine uptrend

    def build(last_vol):
        vols = np.full(n, 1000.0)
        vols[-1] = last_vol
        df = pd.DataFrame({"time": stamps, "open": close, "high": close * 1.001,
                           "low": close * 0.999, "close": close, "volume": vols})
        return strat._build_signal("UP", df, pd.DataFrame(),
                                   datetime(2024, 5, 2, 11, 0, tzinfo=IST), True)

    quiet, loud = build(1000.0), build(20_000.0)
    assert loud.expected_return > quiet.expected_return > 0

    # Volume enters MULTIPLICATIVELY and boundedly: expected_return is exactly
    # the (return-valued) directional blend, shrunk and scaled by confidence.
    for sig in (quiet, loud):
        fs = sig.feature_snapshot
        assert 0.5 <= fs["volume_confidence"] <= 1.25
        assert fs["expected_return_signed"] == pytest.approx(
            _CONTINUATION_COEF * fs["directional_return"] * fs["volume_confidence"]
        )
    assert (loud.feature_snapshot["volume_confidence"]
            / quiet.feature_snapshot["volume_confidence"]) <= 1.25


def test_defect7_cost_estimate_covers_the_real_cost_model():
    """The constant must not understate the modelled round-trip cost."""
    from app.backtesting.costs import ZerodhaCostModel

    model = ZerodhaCostModel()
    for ticket in (50_000, 100_000, 500_000):
        fees = model.breakeven_return(ticket / 1000.0, 1000.0, product="MIS")
        assert INTRADAY_ROUND_TRIP_COST > fees, (
            f"intraday hurdle {INTRADAY_ROUND_TRIP_COST} <= modelled fees {fees}"
        )
        cnc = model.breakeven_return(ticket / 1000.0, 1000.0, product="CNC")
        assert DELIVERY_ROUND_TRIP_COST > cnc

    assert IntradayStrategy(paper_mode=True).cost_estimate == INTRADAY_ROUND_TRIP_COST


def test_defect7_pure_noise_bars_are_still_rejected():
    """Preserve the working no-trade mechanism."""
    uni, idata = make_intraday(n_syms=8, seed=77)
    strat = IntradayStrategy(paper_mode=True, min_intraday_volume=1)
    signals = strat.generate_signals(
        uni, pd.DataFrame(), {},
        current_time=datetime(2024, 5, 2, 11, 0, tzinfo=IST),
        intraday_data=idata,
    )
    assert signals == []


# ===========================================================================
# DEFECT 8 — long-term BUY threshold was statistically unreachable
# ===========================================================================

def test_defect8_buy_rule_is_relative_and_reachable():
    """
    P(composite >= 65) was 0.133% — one stock in 750.  The top decile of the
    scored universe must always be reachable.
    """
    uni, md = make_universe(n_syms=200, seed=31)
    lt = LongtermStrategy(paper_mode=True, max_positions=20)
    scores = [
        (s, lt._compute_composite(s, pd.DataFrame(), md)["composite"], {})
        for s in uni
    ]
    buy_cut, sell_cut = lt._threshold_cuts(scores)
    comps = np.array([c for _, c, _ in scores])

    assert buy_cut is not None and sell_cut is not None
    assert buy_cut > sell_cut
    p_buy = float((comps >= buy_cut).mean())
    p_sell = float((comps < sell_cut).mean())
    assert 0.05 <= p_buy <= 0.15, f"P(buy)={p_buy:.3%} is not ~the top decile"
    assert 0.15 <= p_sell <= 0.25, f"P(sell)={p_sell:.3%} is not ~the bottom quintile"

    buys = [s for s in lt.generate_signals(uni, pd.DataFrame(), md)
            if s.direction == SignalDirection.LONG]
    assert len(buys) == 20  # fills the book from a 200-name universe


def test_defect8_thresholds_track_the_distribution_not_a_constant():
    """Shifting the whole cross-section must shift the cuts with it."""
    uni, md = make_universe(n_syms=60, seed=37)
    lt = LongtermStrategy(paper_mode=True)
    base = [(s, 40.0 + i * 0.1, {}) for i, s in enumerate(uni)]
    shifted = [(s, c + 20.0, d) for s, c, d in base]
    b0, s0 = lt._threshold_cuts(base)
    b1, s1 = lt._threshold_cuts(shifted)
    assert b1 == pytest.approx(b0 + 20.0)
    assert s1 == pytest.approx(s0 + 20.0)


def test_defect8_tiny_universe_emits_no_buys():
    """Percentiles of 5 names are noise; the no-trade path is correct here."""
    uni, md = make_universe(n_syms=5, seed=41)
    lt = LongtermStrategy(paper_mode=True)
    buys = [s for s in lt.generate_signals(uni, pd.DataFrame(), md)
            if s.direction == SignalDirection.LONG]
    assert buys == []


def test_defect8_expected_return_matches_the_declared_horizon():
    uni, md = make_universe(n_syms=60, seed=43)
    lt = LongtermStrategy(paper_mode=True)
    for sig in lt.generate_signals(uni, pd.DataFrame(), md):
        if sig.direction != SignalDirection.LONG:
            continue
        assert sig.holding_period_days == 252
        assert 0.0 < sig.expected_return <= 0.09      # ~annual, not a 0-100 score
        assert sig.edge_score == pytest.approx(
            sig.expected_return - DELIVERY_ROUND_TRIP_COST, abs=1e-12)


# ===========================================================================
# DEFECT 9 — assorted correctness issues
# ===========================================================================

def test_defect9a_nan_imputation_is_per_feature_across_the_cross_section():
    """
    `np.nanmedian(vec)` took the median of ONE stock's heterogeneous features:
    [rsi=70, ret5=0.02, volume=NaN] -> volume became 35.01.
    """
    strat = SwingStrategy(paper_mode=True)
    uni = ["A", "B", "C"]
    features = pd.DataFrame({
        "rsi__A": [70.0], "ret5__A": [0.02], "volume__A": [np.nan],
        "rsi__B": [60.0], "ret5__B": [0.01], "volume__B": [1_000_000.0],
        "rsi__C": [50.0], "ret5__C": [0.03], "volume__C": [1_200_000.0],
    })
    X, syms = strat._build_feature_matrix(uni, features, ["rsi", "ret5", "volume"])
    assert syms == uni
    imputed = X[0, 2]
    assert imputed == pytest.approx(1_100_000.0), (
        f"missing volume imputed as {imputed} (cross-sectional median expected)"
    )
    assert X[0, 0] == 70.0 and X[0, 1] == 0.02


def test_defect9b_missing_feature_drops_the_symbol_not_the_column():
    """
    `available = [c for c in feat_cols if c in last_row.index]` silently
    SHORTENED the vector, producing a ragged vstack whose ValueError was
    swallowed and downgraded to the momentum heuristic.
    """
    strat = SwingStrategy(paper_mode=True)
    uni = ["A", "B"]
    features = pd.DataFrame({
        "f1__A": [1.0], "f2__A": [2.0], "f3__A": [3.0],
        "f1__B": [1.0], "f2__B": [2.0],  # f3 missing for B
    })
    X, syms = strat._build_feature_matrix(uni, features, ["f1", "f2", "f3"])
    assert syms == ["A"]
    assert X.shape == (1, 3)


def test_defect9b_feature_order_is_canonical_not_dataframe_order():
    """Column order in the panel must not permute the model's inputs."""
    strat = SwingStrategy(paper_mode=True)
    uni = ["A"]
    order = ["alpha", "beta", "gamma"]
    forward = pd.DataFrame({"alpha__A": [1.0], "beta__A": [2.0], "gamma__A": [3.0]})
    shuffled = pd.DataFrame({"gamma__A": [3.0], "alpha__A": [1.0], "beta__A": [2.0]})

    x1, _ = strat._build_feature_matrix(uni, forward, order)
    x2, _ = strat._build_feature_matrix(uni, shuffled, order)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(x1[0], [1.0, 2.0, 3.0])

    # Without explicit names the derived order is at least deterministic.
    x3, _ = strat._build_feature_matrix(uni, shuffled, None)
    x4, _ = strat._build_feature_matrix(uni, forward, None)
    np.testing.assert_array_equal(x3, x4)


def test_defect9b_model_errors_are_not_silently_downgraded_to_momentum():
    uni, md = make_universe(n_syms=4, seed=47)
    features = pd.DataFrame({f"f{j}__{s}": [0.1] for s in uni for j in range(2)})

    class BrokenModel:
        def predict(self, X):
            raise RuntimeError("model artifact corrupt")

    strat = SwingStrategy(alpha_model=BrokenModel(), paper_mode=True)
    with pytest.raises(RuntimeError, match="model artifact corrupt"):
        strat._score_universe(uni, features, md, ["f0", "f1"])


def test_defect9c_degradation_exit_uses_the_entry_rank():
    """The documented 'degrade vs ENTRY rank' logic was computed then discarded."""
    strat = SwingStrategy(paper_mode=True)
    entered_high = entry_signal(feature_snapshot={"rank": 0.95})
    position = {"symbol": "A", "entry_price": 100.0, "direction": "LONG",
                "stop_loss": 90.0, "target": 130.0, "signal": entered_high}

    # 0.60 is above the absolute 0.30 floor but below 0.70 * 0.95 = 0.665.
    assert strat.should_exit(position, {"price": 100.0, "current_rank": 0.60}) is True
    assert strat.should_exit(position, {"price": 100.0, "current_rank": 0.80}) is False

    # A name entered mid-pack is held to the absolute floor instead.
    entered_mid = entry_signal(feature_snapshot={"rank": 0.40})
    position["signal"] = entered_mid
    assert strat.should_exit(position, {"price": 100.0, "current_rank": 0.35}) is False
    assert strat.should_exit(position, {"price": 100.0, "current_rank": 0.25}) is True


def test_defect9d_risk_reward_rule_is_enforced_not_merely_documented():
    """risk_reward_ratio() had zero call sites; nothing enforced RR > 1."""
    bad = entry_signal(stop_loss_pct=0.04, target_pct=0.02)
    assert bad.risk_reward_ratio() == 0.5
    assert not bad.is_valid()
    assert SwingStrategy(paper_mode=True).calculate_position_size(bad, CAPITAL, None) == 0.0

    good = entry_signal(stop_loss_pct=0.02, target_pct=0.04)
    assert good.is_valid()
    assert good.breakeven_win_rate() == pytest.approx(1 / 3)

    # EXIT signals are never gated — closing risk is always actionable.
    exit_sig = entry_signal(direction=SignalDirection.EXIT, stop_loss_pct=0.0,
                            target_pct=0.0, edge_score=1.0)
    assert exit_sig.is_valid()


def test_defect9e_breadth_uses_the_session_open_not_the_first_bar():
    """`idf["open"].iloc[0]` was the first bar of whatever window was passed."""
    prev = [datetime(2024, 5, 1, 9, 15, tzinfo=IST) + timedelta(minutes=i)
            for i in range(10)]
    today = [datetime(2024, 5, 2, 9, 15, tzinfo=IST) + timedelta(minutes=i)
             for i in range(10)]
    # Yesterday traded at 100; today gapped up to 200 and drifted DOWN to 190.
    frame = pd.DataFrame({
        "time": prev + today,
        "open": [100.0] * 10 + [200.0] + [195.0] * 9,
        "high": [101.0] * 10 + [201.0] * 10,
        "low": [99.0] * 10 + [189.0] * 10,
        "close": [100.0] * 10 + [195.0] * 9 + [190.0],
        "volume": [1000.0] * 20,
    })
    assert IntradayStrategy._session_open_price(frame) == 200.0

    # Down 5% on the session, even though it is far above yesterday's open.
    breadth = IntradayStrategy._compute_breadth(["X"], {"X": frame})
    assert breadth == 0.0


# ===========================================================================
# Preserved behaviour — the working no-trade mechanism
# ===========================================================================

def test_paused_and_disabled_strategies_emit_nothing():
    uni, md = make_universe(n_syms=30, seed=53)
    for health in (StrategyHealth.PAUSED, StrategyHealth.DISABLED):
        sw = SwingStrategy(paper_mode=True)
        sw.health = health
        assert sw.generate_signals(uni, pd.DataFrame(), md) == []

        lt = LongtermStrategy(paper_mode=True)
        lt.health = health
        assert lt.generate_signals(uni, pd.DataFrame(), md) == []


def test_blocked_regimes_emit_nothing():
    uni, md = make_universe(n_syms=30, seed=59, drift=0.001)
    sw = SwingStrategy(paper_mode=True, horizon_days=20)
    for regime in ("BEAR", "PANIC", "BEAR_HIGH_VOL"):
        assert sw.generate_signals(uni, pd.DataFrame(), md, regime=regime) == []


def test_crashing_universe_emits_no_swing_signals():
    """
    A cross-sectional z-score is purely RELATIVE — in a market where every
    name is down, the least-bad names still rank highly.  The absolute-trend
    filter is what keeps the sleeve flat: no name whose own 12-1 momentum is
    negative may ever be signalled long.
    """
    sw = SwingStrategy(paper_mode=True, horizon_days=20)

    # Deep crash: nothing has a positive absolute trend, so nothing trades.
    uni, md = make_universe(n_syms=40, seed=61, drift=-0.005)
    assert sw.generate_signals(uni, pd.DataFrame(), md) == []

    # Milder crash: only names with a genuinely positive own trend survive.
    uni, md = make_universe(n_syms=40, seed=61, drift=-0.003)
    signals = sw.generate_signals(uni, pd.DataFrame(), md)
    assert len(signals) <= 0.10 * len(uni)
    for sig in signals:
        px = md[sig.symbol]["close"]
        own_momentum = float(np.log(px.iloc[-21] / px.iloc[-252]))
        assert own_momentum > 0, (
            f"{sig.symbol} signalled long with 12-1 momentum {own_momentum:.3f}"
        )


def test_every_signal_carries_a_feature_snapshot():
    uni, md = make_universe(n_syms=60, seed=67, drift=0.0006)
    sw = SwingStrategy(paper_mode=True, horizon_days=20)
    for sig in sw.generate_signals(uni, pd.DataFrame(), md):
        assert sig.feature_snapshot
        assert "raw_score" in sig.feature_snapshot

    lt = LongtermStrategy(paper_mode=True)
    for sig in lt.generate_signals(uni, pd.DataFrame(), md):
        assert sig.feature_snapshot
