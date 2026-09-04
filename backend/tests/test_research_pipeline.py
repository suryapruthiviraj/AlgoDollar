"""
End-to-end validation of the cross-sectional research pipeline.

The pipeline is tested against synthetic data whose ground truth is known by
construction, which is the only setting where "is this number right?" has a
definite answer.

Two cases matter, and both must hold:

  NULL CASE    Prices are independent random walks. No feature can predict
               anything. The pipeline must report an information coefficient
               indistinguishable from zero. A pipeline that finds edge here
               has a leak, and every result it ever produces is worthless.

  SIGNAL CASE  A known predictive relationship is injected. The pipeline must
               recover it. A pipeline that finds nothing here is broken in the
               opposite direction — it would discard real alpha.

Passing only the null case is easy: `return 0.0` passes it. Passing only the
signal case is also easy: a leaky pipeline passes it. Both together are what
gives the machinery credibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.research.pipeline import (
    build_forward_return_labels,
    cross_sectional_ic,
    long_short_returns,
    run_cross_sectional_research,
    stack_to_panel,
)

N_SYMBOLS = 30
N_DAYS = 1500
HORIZON = 5


def _dates(n: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def _model_factory():
    # Scaler is fitted inside the pipeline, hence per-fold on training data
    # only. Fitting a scaler once on the whole panel would leak test-period
    # distribution information into training.
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


# ---------------------------------------------------------------------------
# Synthetic worlds
# ---------------------------------------------------------------------------

def make_null_world(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Independent random walks. No cross-sectional predictability exists.

    Returns (prices, features_wide).
    """
    rng = np.random.default_rng(seed)
    idx = _dates()
    syms = [f"S{i:02d}" for i in range(N_SYMBOLS)]

    rets = rng.normal(0, 0.015, size=(N_DAYS, N_SYMBOLS))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=syms
    )

    # Features are pure noise, deliberately unrelated to future returns.
    feats = {}
    for f in ("featA", "featB", "featC"):
        for s in syms:
            feats[f"{f}__{s}"] = pd.Series(
                rng.normal(0, 1, N_DAYS), index=idx
            )
    return prices, pd.DataFrame(feats, index=idx)


def make_signal_world(
    seed: int = 0, signal_strength: float = 0.6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A world with a KNOWN cross-sectional signal.

    Construction order matters. Generate the price path FIRST, then compute
    the forward return exactly as the pipeline computes its label, then build
    `featA` as a noisy observation of that quantity. Deriving the feature from
    the realized label guarantees the intended relationship actually exists in
    the data.

    (An earlier version of this generator built prices from a separately drawn
    forward-return array. That produced a label equal to a rolling average of
    those draws rather than the draw itself, so the "signal" feature had zero
    real predictive power — the generator was broken, not the pipeline.)

    `signal_strength` weights the true component against noise, bounding the
    achievable IC well below 1 so the task stays realistic.
    """
    rng = np.random.default_rng(seed)
    idx = _dates()
    syms = [f"S{i:02d}" for i in range(N_SYMBOLS)]

    daily = rng.normal(0, 0.015, size=(N_DAYS, N_SYMBOLS))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(daily, axis=0)), index=idx, columns=syms
    )

    # The exact quantity the pipeline will use as its label.
    fwd = build_forward_return_labels(prices, HORIZON)

    # Cross-sectionally standardize per date, so the feature carries ranking
    # information rather than the day's market-wide move.
    fwd_z = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1) + 1e-12, axis=0)

    noise = pd.DataFrame(
        rng.normal(0, 1.0, size=(N_DAYS, N_SYMBOLS)), index=idx, columns=syms
    )
    feat_a = signal_strength * fwd_z + np.sqrt(1 - signal_strength**2) * noise

    feats = {}
    for s in syms:
        feats[f"featA__{s}"] = feat_a[s]
        feats[f"featB__{s}"] = pd.Series(rng.normal(0, 1, N_DAYS), index=idx)
        feats[f"featC__{s}"] = pd.Series(rng.normal(0, 1, N_DAYS), index=idx)

    return prices, pd.DataFrame(feats, index=idx)


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def test_forward_labels_are_forward_looking():
    prices, _ = make_null_world()
    labels = build_forward_return_labels(prices, HORIZON)

    # Last HORIZON rows have no realized outcome yet and must stay NaN.
    assert labels.iloc[-HORIZON:].isna().all().all()

    # Spot-check the arithmetic on one cell.
    s = prices.columns[0]
    expected = np.log(prices[s].iloc[HORIZON] / prices[s].iloc[0])
    assert np.isclose(labels[s].iloc[0], expected)


def test_forward_labels_reject_bad_horizon():
    prices, _ = make_null_world()
    with pytest.raises(ValueError, match="horizon must be"):
        build_forward_return_labels(prices, 0)


def test_stack_preserves_canonical_feature_order():
    """
    Columns must land in the same model position for every symbol. Gathering
    them by name-suffix in DataFrame order is how a feature silently ends up
    in the wrong slot when one symbol is missing a column.
    """
    prices, feats = make_null_world()
    labels = build_forward_return_labels(prices, HORIZON)
    X, y, dates = stack_to_panel(feats, labels, list(prices.columns))

    assert list(X.columns) == sorted(X.columns)
    assert X.index.names == ["date", "symbol"]
    assert len(y) == len(X)


# ---------------------------------------------------------------------------
# Cross-sectional IC behaves correctly
# ---------------------------------------------------------------------------

def test_pooled_correlation_would_be_misleading():
    """
    Demonstrates WHY IC must be computed per-date.

    This model has zero stock-picking skill — within any single day its
    ranking is random — but its overall level tracks the market. Pooling all
    observations together reports a large correlation; the correct per-date
    computation reports approximately zero.
    """
    rng = np.random.default_rng(4)
    dates = _dates(200)
    syms = [f"S{i:02d}" for i in range(20)]

    rows, preds, labels = [], [], []
    for d in dates:
        market = rng.normal(0, 0.02)          # common factor for the day
        for s in syms:
            rows.append((d, s))
            # Prediction knows the market move but nothing about the name.
            preds.append(market + rng.normal(0, 0.001))
            labels.append(market + rng.normal(0, 0.02))

    midx = pd.MultiIndex.from_tuples(rows, names=["date", "symbol"])
    p = pd.Series(preds, index=midx)
    y = pd.Series(labels, index=midx)

    from scipy import stats as _st
    pooled, _ = _st.spearmanr(p.values, y.values)
    per_date = cross_sectional_ic(p, y).mean()

    assert pooled > 0.4, f"expected large pooled correlation, got {pooled:.3f}"
    assert abs(per_date) < 0.10, (
        f"per-date IC should be ~0 for a zero-skill model, got {per_date:.3f}"
    )


def test_cross_sectional_ic_detects_real_ranking_skill():
    """Positive control: a perfect ranker must score IC = 1.0."""
    dates = _dates(50)
    syms = [f"S{i:02d}" for i in range(20)]
    rng = np.random.default_rng(6)

    rows, preds, labels = [], [], []
    for d in dates:
        vals = rng.normal(0, 0.02, len(syms))
        for s, v in zip(syms, vals):
            rows.append((d, s))
            preds.append(v)     # prediction == outcome: perfect ranking
            labels.append(v)

    midx = pd.MultiIndex.from_tuples(rows, names=["date", "symbol"])
    ic = cross_sectional_ic(pd.Series(preds, index=midx), pd.Series(labels, index=midx))
    assert np.isclose(ic.mean(), 1.0, atol=1e-9)


def test_long_short_returns_track_ranking():
    dates = _dates(60)
    syms = [f"S{i:02d}" for i in range(20)]
    rng = np.random.default_rng(8)

    rows, preds, labels = [], [], []
    for d in dates:
        vals = rng.normal(0, 0.02, len(syms))
        for s, v in zip(syms, vals):
            rows.append((d, s))
            preds.append(v)
            labels.append(v)

    midx = pd.MultiIndex.from_tuples(rows, names=["date", "symbol"])
    ls = long_short_returns(
        pd.Series(preds, index=midx), pd.Series(labels, index=midx)
    )
    # A perfect ranker's top quintile must beat its bottom quintile every day.
    assert (ls > 0).all()


# ---------------------------------------------------------------------------
# THE TWO DECISIVE END-TO-END TESTS
# ---------------------------------------------------------------------------

def test_null_world_reports_no_edge():
    """
    NULL CASE. Independent random walks, noise features.

    The pipeline must report an IC statistically indistinguishable from zero.
    If this fails, something is leaking and every other result is void.
    """
    prices, feats = make_null_world(seed=1)
    labels = build_forward_return_labels(prices, HORIZON)
    X, y, _ = stack_to_panel(feats, labels, list(prices.columns))

    res = run_cross_sectional_research(
        X=X, y=y, model_factory=_model_factory,
        label_horizon=HORIZON, n_splits=5, n_trials=1,
    )

    assert abs(res.mean_ic) < 0.05, (
        f"pipeline found edge in pure noise (IC={res.mean_ic:+.4f}). "
        f"This indicates leakage.\n{res.summary()}"
    )
    assert abs(res.ic_tstat_deflated) < 3.0, (
        f"spurious significance on noise: t={res.ic_tstat_deflated:.2f}"
    )


def test_signal_world_recovers_known_edge():
    """
    SIGNAL CASE. A known predictive feature is present.

    The pipeline must find it. Without this, the null test above would be
    satisfied by a pipeline that simply never detects anything.
    """
    prices, feats = make_signal_world(seed=2, signal_strength=0.6)
    labels = build_forward_return_labels(prices, HORIZON)
    X, y, _ = stack_to_panel(feats, labels, list(prices.columns))

    res = run_cross_sectional_research(
        X=X, y=y, model_factory=_model_factory,
        label_horizon=HORIZON, n_splits=5, n_trials=1,
    )

    assert res.mean_ic > 0.15, (
        f"pipeline failed to recover an injected signal "
        f"(IC={res.mean_ic:+.4f}).\n{res.summary()}"
    )
    assert res.ls_sharpe_annual > 0, "long-short book should be profitable"


def test_purging_is_actually_applied_in_the_pipeline():
    """Every fold must report purged observations equal to the label horizon."""
    prices, feats = make_null_world(seed=3)
    labels = build_forward_return_labels(prices, HORIZON)
    X, y, _ = stack_to_panel(feats, labels, list(prices.columns))

    res = run_cross_sectional_research(
        X=X, y=y, model_factory=_model_factory,
        label_horizon=HORIZON, n_splits=5, n_trials=1,
    )
    assert all(f.n_purged == HORIZON for f in res.folds), (
        f"purge counts: {[f.n_purged for f in res.folds]}"
    )


def test_deflated_tstat_is_smaller_than_naive():
    """Overlapping labels must deflate significance, never inflate it."""
    prices, feats = make_signal_world(seed=4)
    labels = build_forward_return_labels(prices, HORIZON)
    X, y, _ = stack_to_panel(feats, labels, list(prices.columns))

    res = run_cross_sectional_research(
        X=X, y=y, model_factory=_model_factory,
        label_horizon=HORIZON, n_splits=5, n_trials=1,
    )
    assert abs(res.ic_tstat_deflated) < abs(res.ic_tstat_naive)
    assert any("Effective sample size" in n for n in res.notes)


def test_many_trials_reduce_reported_significance():
    """
    The same result, honestly reported as the product of a large search, must
    be treated as weaker evidence.

    A deliberately marginal signal is used (long-short Sharpe ~1.0). Stronger
    signals saturate the Deflated Sharpe Ratio at 1.0 for any realistic trial
    count — correctly, since a Sharpe of 9 is overwhelming evidence however
    many configurations were tried — which would make this test vacuous.

    The Sharpe ~1.0 regime is where the adjustment decides real cases: found
    on the first attempt it is convincing; cherry-picked from 5000 attempts it
    is not.
    """
    prices, feats = make_signal_world(seed=5, signal_strength=0.02)
    labels = build_forward_return_labels(prices, HORIZON)
    X, y, _ = stack_to_panel(feats, labels, list(prices.columns))

    one = run_cross_sectional_research(
        X=X, y=y, model_factory=_model_factory,
        label_horizon=HORIZON, n_splits=5, n_trials=1,
    )
    many = run_cross_sectional_research(
        X=X, y=y, model_factory=_model_factory,
        label_horizon=HORIZON, n_splits=5, n_trials=5000,
    )
    assert one.dsr is not None and many.dsr is not None

    # Identical returns, identical Sharpe — only the honesty about how many
    # configurations were tried differs.
    assert np.isclose(one.ls_sharpe_annual, many.ls_sharpe_annual)
    assert many.dsr < one.dsr
    assert one.dsr_significant and not many.dsr_significant, (
        f"expected the search-adjusted verdict to flip: "
        f"DSR(1 trial)={one.dsr:.4f}, DSR(5000 trials)={many.dsr:.4f}"
    )
