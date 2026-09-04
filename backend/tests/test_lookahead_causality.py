"""
Automated look-ahead (causality) detection.

THE TEST
--------
A feature function f is *causal* if the value it assigns to bar T depends only
on data at or before T. That gives a mechanically checkable property:

    f(prices[:T])[-1]  ==  f(prices[:T+k])[T-1]      for every T, k > 0

Computing the feature on a truncated history must reproduce exactly what the
same feature produced at that timestamp when it had the full history available.
If the two disagree, the feature is reading data that did not exist yet.

This is stronger than reading the code and asserting causality in a docstring,
which is how look-ahead bias normally survives review: it is an empirical test
that fails loudly, and it runs against every feature in the engine.

WHAT A FAILURE MEANS
--------------------
A failure here invalidates every backtest that used the feature. The backtest
would have been trading on information the strategy could not have possessed,
so its returns are fictional. Treat a failure as a release blocker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.features import FeatureEngine

# ---------------------------------------------------------------------------
# Synthetic data with known properties
# ---------------------------------------------------------------------------

def make_gbm_series(
    n_days: int = 600,
    seed: int = 0,
    annual_vol: float = 0.30,
    annual_drift: float = 0.0,
    s0: float = 100.0,
) -> pd.Series:
    """
    Geometric Brownian motion: IID log returns, zero predictable structure.

    By construction there is no autocorrelation and no predictable component,
    so no feature computed from this series can have genuine forecasting power.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu, sigma = annual_drift, annual_vol
    log_rets = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days)
    prices = s0 * np.exp(np.cumsum(log_rets))
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.Series(prices, index=idx, name="SYNTH")


def make_volume_series(n_days: int = 600, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    vol = rng.lognormal(mean=13.0, sigma=0.4, size=n_days)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.Series(vol, index=idx, name="SYNTH")


# ---------------------------------------------------------------------------
# The causality checker
# ---------------------------------------------------------------------------

def assert_causal(
    fn,
    *series_args: pd.Series,
    cut_points: tuple[int, ...] = (300, 400, 500),
    tol: float = 1e-9,
    name: str = "",
) -> None:
    """
    Assert that `fn` is causal.

    For each cut point T, recompute the feature on data truncated at T and
    compare the final value against the value the full-history computation
    produced at that same timestamp.

    Parameters
    ----------
    fn : callable(*series) -> pd.Series
    series_args : the input series (all must share an index)
    cut_points : indices at which to truncate
    tol : absolute tolerance for the comparison
    name : label used in the failure message
    """
    full = fn(*series_args)
    label = name or getattr(fn, "__name__", "feature")

    for cut in cut_points:
        truncated_inputs = [s.iloc[:cut] for s in series_args]
        partial = fn(*truncated_inputs)

        v_partial = partial.iloc[-1]
        v_full = full.iloc[cut - 1]

        # Both NaN (insufficient lookback) is a consistent, acceptable outcome.
        if pd.isna(v_partial) and pd.isna(v_full):
            continue

        if pd.isna(v_partial) != pd.isna(v_full):
            raise AssertionError(
                f"LOOK-AHEAD in {label}: at cut={cut} NaN-ness differs "
                f"(truncated={v_partial!r}, full={v_full!r}). The feature "
                f"behaves differently when future data is present."
            )

        if abs(float(v_partial) - float(v_full)) > tol:
            raise AssertionError(
                f"LOOK-AHEAD in {label}: at cut={cut} truncated value "
                f"{float(v_partial):.10g} != full-history value "
                f"{float(v_full):.10g} (diff="
                f"{abs(float(v_partial) - float(v_full)):.3g}). "
                f"The feature is using data from after bar {cut - 1}."
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prices() -> pd.Series:
    return make_gbm_series()


@pytest.fixture(scope="module")
def volume() -> pd.Series:
    return make_volume_series()


# ---------------------------------------------------------------------------
# Causality tests — one per feature family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("periods", [1, 5, 21, 63])
def test_log_return_is_causal(prices, periods):
    assert_causal(
        lambda p: FeatureEngine.log_return(p, periods),
        prices,
        name=f"log_return_{periods}d",
    )


def test_momentum_12_1_is_causal(prices):
    assert_causal(FeatureEngine.momentum_12_1, prices, name="momentum_12_1")


def test_rsi_is_causal(prices):
    assert_causal(lambda p: FeatureEngine.rsi(p, 14), prices, name="rsi_14")


def test_distance_from_52w_high_is_causal(prices):
    assert_causal(
        FeatureEngine.distance_from_52w_high, prices, name="distance_from_52w_high"
    )


def test_distance_from_52w_low_is_causal(prices):
    assert_causal(
        FeatureEngine.distance_from_52w_low, prices, name="distance_from_52w_low"
    )


@pytest.mark.parametrize("window", [20, 50, 200])
def test_price_to_sma_is_causal(prices, window):
    assert_causal(
        lambda p: FeatureEngine.price_to_sma(p, window),
        prices,
        name=f"price_to_sma{window}",
    )


@pytest.mark.parametrize("window", [10, 21, 63])
def test_realized_vol_is_causal(prices, window):
    assert_causal(
        lambda p: FeatureEngine.realized_vol(p, window),
        prices,
        name=f"realized_vol_{window}d",
    )


def test_ewma_vol_is_causal(prices):
    assert_causal(FeatureEngine.ewma_vol, prices, name="ewma_vol")


def test_vol_ratio_is_causal(prices):
    assert_causal(FeatureEngine.vol_ratio, prices, name="vol_ratio")


def test_volume_ratio_is_causal(volume):
    assert_causal(
        lambda v: FeatureEngine.volume_ratio(v, 10), volume, name="volume_ratio_10d"
    )


def test_relative_volume_zscore_is_causal(volume):
    assert_causal(
        lambda v: FeatureEngine.relative_volume_zscore(v, 20),
        volume,
        name="relative_volume_zscore",
    )


def test_pvt_is_causal(prices, volume):
    assert_causal(FeatureEngine.pvt, prices, volume, name="pvt")


def test_obv_slope_is_causal(prices, volume):
    assert_causal(
        lambda p, v: FeatureEngine.obv_slope(p, v, 10),
        prices,
        volume,
        name="obv_slope_10d",
    )


def test_excess_return_vs_nifty_is_causal(prices):
    nifty = make_gbm_series(seed=99)
    assert_causal(
        lambda p, n: FeatureEngine.excess_return_vs_nifty(p, n, 5),
        prices,
        nifty,
        name="excess_return_vs_nifty_5d",
    )


# ---------------------------------------------------------------------------
# Negative control: the checker must actually catch a known leak
# ---------------------------------------------------------------------------

def test_checker_detects_deliberate_leak(prices):
    """
    Negative control.

    A test that never fails proves nothing. Feed the checker a feature that
    deliberately peeks one bar into the future and confirm it is caught. If
    this test does not raise, `assert_causal` is broken and every passing
    causality test above is meaningless.
    """

    def leaky_feature(p: pd.Series) -> pd.Series:
        # shift(-1) pulls tomorrow's price into today's row: textbook leakage.
        return p.shift(-1) / p - 1.0

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_causal(leaky_feature, prices, name="deliberately_leaky")


def test_checker_detects_full_sample_normalization(prices):
    """
    Negative control for the most common real-world leak: normalizing a feature
    using statistics computed over the whole sample (including the future).
    """

    def full_sample_zscore(p: pd.Series) -> pd.Series:
        rets = np.log(p / p.shift(1))
        # .mean()/.std() over the ENTIRE series uses future observations.
        return (rets - rets.mean()) / rets.std()

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_causal(full_sample_zscore, prices, name="full_sample_zscore")
