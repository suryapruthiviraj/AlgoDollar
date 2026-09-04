"""
Tests for the multiple-testing and validation machinery.

These tests are built around known ground truth. Pure-noise inputs must be
reported as having no edge; genuinely skilful inputs must be reported as
skilful. A statistic that cannot tell those two cases apart is worse than no
statistic, because it lends false confidence to a coin flip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.statistics import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    holm_bonferroni,
    probability_of_backtest_overfitting,
    stationary_bootstrap,
)
from app.research.validation import (
    PurgedWalkForward,
    assert_no_train_test_overlap,
    deflated_tstat,
    effective_sample_size,
)

# ===========================================================================
# Purged walk-forward
# ===========================================================================

def _times(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def test_purging_removes_overlapping_labels():
    """
    With a 5-period label horizon, the last training observation must end at
    least 5 positions before the first test observation. Otherwise its label
    was computed from test-period data.
    """
    n, horizon = 1000, 5
    splitter = PurgedWalkForward(n_splits=5, label_horizon=horizon, embargo_frac=0.0)

    splits = list(splitter.split(_times(n)))
    assert len(splits) > 0

    for sp in splits:
        gap = int(sp.test_idx.min()) - int(sp.train_idx.max())
        assert gap > horizon, (
            f"Only {gap} positions between last train and first test, but the "
            f"label horizon is {horizon}. Labels overlap the test window."
        )
        assert_no_train_test_overlap(sp, _times(n), horizon)


def test_zero_horizon_disables_purging():
    """A horizon of 0 means labels are contemporaneous; nothing to purge."""
    splitter = PurgedWalkForward(n_splits=4, label_horizon=0, embargo_frac=0.0)
    splits = list(splitter.split(_times(500)))
    assert all(sp.n_purged == 0 for sp in splits)


def test_purge_count_matches_horizon():
    """Exactly `label_horizon` observations should be purged per split."""
    horizon = 10
    splitter = PurgedWalkForward(n_splits=4, label_horizon=horizon, embargo_frac=0.0)
    for sp in splitter.split(_times(1000)):
        assert sp.n_purged == horizon, (
            f"expected {horizon} purged, got {sp.n_purged}"
        )


def test_embargo_is_applied():
    """Embargo must drop observations; with expanding walk-forward the right
    side is unused, so the count is recorded but training stays causal."""
    splitter = PurgedWalkForward(n_splits=4, label_horizon=1, embargo_frac=0.05)
    splits = list(splitter.split(_times(1000)))
    assert any(sp.n_embargoed > 0 for sp in splits)


def test_training_data_is_always_in_the_past():
    """Walk-forward must never train on data after the test window."""
    splitter = PurgedWalkForward(n_splits=5, label_horizon=5)
    for sp in splitter.split(_times(1000)):
        assert sp.train_idx.max() < sp.test_idx.min()


def test_unsorted_times_raises():
    """Unsorted input would make purging silently wrong, so it must raise."""
    t = _times(100).tolist()
    t[50], t[10] = t[10], t[50]
    with pytest.raises(ValueError, match="sorted"):
        list(PurgedWalkForward(n_splits=3, label_horizon=5).split(t))


def test_leakage_assertion_catches_a_bad_split():
    """Negative control: the leakage assertion must reject a bad split."""
    from app.research.validation import Split

    bad = Split(
        train_idx=np.arange(0, 100),
        test_idx=np.arange(100, 200),   # only 1 position of gap
        n_purged=0, n_embargoed=0,
        train_start=pd.Timestamp("2015-01-01"),
        train_end=pd.Timestamp("2015-05-01"),
        test_start=pd.Timestamp("2015-05-02"),
        test_end=pd.Timestamp("2015-09-01"),
    )
    with pytest.raises(AssertionError, match="LABEL LEAKAGE"):
        assert_no_train_test_overlap(bad, _times(200), label_horizon=5)


def test_effective_sample_size_deflates_overlapping_labels():
    assert effective_sample_size(1000, 1) == 1000.0
    assert effective_sample_size(1000, 5) == 200.0

    # A t-stat on overlapping labels must shrink by ~sqrt(horizon).
    t_naive = deflated_tstat(mean=0.01, std=0.10, n_obs=1000, label_horizon=1)
    t_defl = deflated_tstat(mean=0.01, std=0.10, n_obs=1000, label_horizon=5)
    assert t_defl < t_naive
    assert np.isclose(t_naive / t_defl, np.sqrt(5), rtol=1e-6)


# ===========================================================================
# Deflated Sharpe Ratio
# ===========================================================================

def test_expected_max_sharpe_grows_with_trials():
    """More trials -> higher bar. This is the whole point of the correction."""
    v = 1.0 / 251
    bars = [expected_max_sharpe_under_null(n, v) for n in (1, 10, 100, 1000)]
    assert bars[0] == 0.0
    assert bars[1] < bars[2] < bars[3]


def test_winner_selected_from_pure_noise_is_rejected():
    """
    THE CENTRAL TEST.

    Generate 200 strategies with zero true skill, keep the best-looking one,
    and confirm the Deflated Sharpe Ratio refuses to call it significant even
    though its raw annualized Sharpe looks respectable.
    """
    rng = np.random.default_rng(7)
    T, N = 756, 200  # 3 years daily, 200 configurations

    panel = rng.normal(0.0, 0.01, size=(T, N))  # zero mean == zero true edge
    sharpes = panel.mean(axis=0) / panel.std(axis=0, ddof=1)
    best = int(np.argmax(sharpes))
    best_returns = panel[:, best]

    raw_annual_sharpe = sharpes[best] * np.sqrt(252)

    res = deflated_sharpe_ratio(
        returns=best_returns,
        n_trials=N,
        sharpe_variance=float(np.var(sharpes, ddof=1)),
    )

    # The luckiest of 200 noise strategies looks tradeable on a raw Sharpe.
    assert raw_annual_sharpe > 0.4, (
        f"test is not exercising the intended regime: raw annualized Sharpe "
        f"was only {raw_annual_sharpe:.2f}"
    )
    # But it must not survive the multiple-testing adjustment.
    assert not res.is_significant, (
        f"DSR wrongly certified a pure-noise winner: {res.summary()}"
    )
    assert res.deflated_sharpe_ratio < 0.95


def test_genuine_skill_survives_deflation():
    """
    Positive control. A strategy with real, large edge must still be called
    significant after correction, otherwise the test above proves nothing
    beyond 'this function always says no'.
    """
    rng = np.random.default_rng(11)
    T = 1260  # 5 years daily
    # ~1.5 annualized Sharpe: mean/sd = 1.5/sqrt(252) per period.
    per_period_sr = 1.5 / np.sqrt(252)
    returns = rng.normal(per_period_sr * 0.01, 0.01, size=T)

    res = deflated_sharpe_ratio(returns=returns, n_trials=10)
    assert res.is_significant, f"real edge rejected: {res.summary()}"


def test_more_trials_makes_significance_harder():
    """The same return series must become less significant as N rises."""
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0008, 0.01, size=1000)

    few = deflated_sharpe_ratio(returns, n_trials=2)
    many = deflated_sharpe_ratio(returns, n_trials=10_000)
    assert many.deflated_sharpe_ratio < few.deflated_sharpe_ratio


def test_zero_variance_returns_raise():
    with pytest.raises(ValueError, match="zero variance"):
        deflated_sharpe_ratio(np.zeros(100), n_trials=1)


# ===========================================================================
# Probability of Backtest Overfitting
# ===========================================================================

def test_pbo_high_when_configs_are_pure_noise():
    """
    With no real differences between configurations, picking the in-sample
    best should be about as good as picking at random, so PBO should sit near
    0.5 rather than near 0.
    """
    rng = np.random.default_rng(5)
    M = rng.normal(0.0, 0.01, size=(1000, 40))

    res = probability_of_backtest_overfitting(M, n_partitions=10)
    assert 0.25 < res.pbo < 0.75, (
        f"PBO on pure noise should be near 0.5, got {res.pbo:.3f}"
    )


def test_pbo_low_when_one_config_is_genuinely_better():
    """
    Positive control. If one configuration has persistently higher expected
    return, selecting on in-sample performance should generalize, so PBO
    should be low.
    """
    rng = np.random.default_rng(13)
    T, N = 1200, 20
    M = rng.normal(0.0, 0.01, size=(T, N))
    M[:, 0] += 0.0015  # one genuinely superior configuration

    res = probability_of_backtest_overfitting(M, n_partitions=10)
    assert res.pbo < 0.2, f"PBO should be low with a real winner: {res.summary()}"
    assert not res.is_overfit


def test_pbo_rejects_odd_partitions():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(
            rng.normal(size=(500, 10)), n_partitions=7
        )


def test_pbo_requires_multiple_configs():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="at least 2"):
        probability_of_backtest_overfitting(rng.normal(size=(500, 1)))


# ===========================================================================
# Stationary bootstrap
# ===========================================================================

def test_stationary_bootstrap_preserves_autocorrelation():
    """
    The reason to prefer block resampling over IID resampling.

    Build a strongly autocorrelated series, resample it both ways, and confirm
    the block bootstrap retains dependence that IID resampling destroys.
    """
    rng = np.random.default_rng(17)
    T = 2000
    ar = np.zeros(T)
    for t in range(1, T):
        ar[t] = 0.6 * ar[t - 1] + rng.normal(0, 0.01)

    def lag1(x: np.ndarray) -> float:
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    original = lag1(ar)
    assert original > 0.4

    block_paths = stationary_bootstrap(
        ar, n_simulations=200, mean_block_length=50.0, random_seed=2
    )
    block_ac = float(np.mean([lag1(p) for p in block_paths]))

    iid = rng.choice(ar, size=(200, T), replace=True)
    iid_ac = float(np.mean([lag1(p) for p in iid]))

    assert abs(iid_ac) < 0.10, f"IID resampling should destroy AR, got {iid_ac:.3f}"
    assert block_ac > 0.30, f"block bootstrap lost the AR structure: {block_ac:.3f}"


def test_stationary_bootstrap_shape_and_support():
    rng = np.random.default_rng(19)
    r = rng.normal(0, 0.01, size=500)
    out = stationary_bootstrap(r, n_simulations=50, horizon=250)
    assert out.shape == (50, 250)
    # Every resampled value must come from the original sample.
    assert np.isin(out, r).all()


def test_stationary_bootstrap_rejects_tiny_samples():
    with pytest.raises(ValueError, match="at least 10"):
        stationary_bootstrap(np.zeros(5))


# ===========================================================================
# Multiple-testing corrections
# ===========================================================================

def test_benjamini_hochberg_controls_fdr_on_pure_noise():
    """With all nulls true, BH should make very few discoveries."""
    rng = np.random.default_rng(23)
    p = rng.uniform(0, 1, size=1000)  # p-values under the null are uniform
    rejected, q = benjamini_hochberg(p, alpha=0.05)
    assert rejected.sum() <= 5, f"too many false discoveries: {rejected.sum()}"
    assert np.all(q >= p - 1e-12)


def test_benjamini_hochberg_finds_real_effects():
    rng = np.random.default_rng(29)
    p = np.concatenate([rng.uniform(0, 1, 500), np.full(50, 1e-6)])
    rejected, _ = benjamini_hochberg(p, alpha=0.05)
    assert rejected[500:].sum() >= 45, "BH missed clearly real effects"


def test_holm_is_stricter_than_bh():
    rng = np.random.default_rng(31)
    p = np.concatenate([rng.uniform(0, 1, 200), np.full(20, 1e-4)])
    bh_rej, _ = benjamini_hochberg(p, alpha=0.05)
    holm_rej, _ = holm_bonferroni(p, alpha=0.05)
    assert holm_rej.sum() <= bh_rej.sum()


def test_corrections_reject_invalid_p_values():
    for fn in (benjamini_hochberg, holm_bonferroni):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            fn([0.1, 1.5, 0.3])
