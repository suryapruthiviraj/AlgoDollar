"""
Statistical-correctness tests for app/models/ml_models.py.

Each test names the defect it guards against.  Several are written so that they
FAIL against the previous implementation:

  Defect 1  IC was pooled over a stacked panel (measures market co-movement,
            not stock-picking skill).
  Defect 2  `test_icir = test_ic` — a dispersion-free number named ICIR.
  Defect 3  Model selection gated on `test_directional_accuracy` and ranked on
            `(val_ic + test_ic) / 2`, i.e. it selected on the held-out set, with
            no multiple-comparison correction.
  Defect 4  `RidgeCV(cv=5)` -> KFold(shuffle=False): future rows chose the penalty.
  Defect 5  `_long_short_sharpe` returned a single-period return named "sharpe".
  Defect 6  The no-look-ahead contract was prose only: no embargo, no handling
            of overlapping labels in the reported significance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from app.models.ml_models import (
    DEFAULT_PERIODS_PER_YEAR,
    AlphaModelBase,
    LinearAlphaModel,
    ModelCompetition,
    _embargoed_time_series_splits,
    _information_coefficient,
    _pooled_rank_correlation,
    cross_sectional_ic_series,
    deflated_sharpe_ratio,
    information_coefficient,
    long_short_sharpe,
)

N_DATES = 250
N_NAMES = 50


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _zero_skill_panel(n_dates=N_DATES, n_names=N_NAMES, seed=7):
    """
    A model with ZERO stock-picking skill whose prediction LEVEL tracks the market.

    On each date every name gets (approximately) the same prediction — the market
    move — so the model ranks no stock above another, yet a pooled correlation
    over the stacked panel is large because both series share the market factor.
    """
    rng = np.random.default_rng(seed)
    dates, preds, ys = [], [], []
    for t in range(n_dates):
        mkt = rng.normal(0.0, 0.02)
        idio = rng.normal(0.0, 0.02, n_names)
        ys.append(mkt + idio)
        # noise is independent of idio => no cross-sectional information at all
        preds.append(mkt + rng.normal(0.0, 0.002, n_names))
        dates.append(np.full(n_names, t))
    return np.concatenate(dates), np.concatenate(preds), np.concatenate(ys)


def _skilled_panel(n_dates=N_DATES, n_names=N_NAMES, seed=11, skill=0.35):
    """A model with genuine, injected CROSS-SECTIONAL skill."""
    rng = np.random.default_rng(seed)
    dates, preds, ys = [], [], []
    for t in range(n_dates):
        mkt = rng.normal(0.0, 0.02)
        signal = rng.normal(0.0, 1.0, n_names)
        idio = rng.normal(0.0, 1.0, n_names)
        ys.append(mkt + 0.02 * (skill * signal + math.sqrt(1 - skill ** 2) * idio))
        preds.append(signal)
        dates.append(np.full(n_names, t))
    return np.concatenate(dates), np.concatenate(preds), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Defect 1 — IC must be cross-sectional, not pooled
# ---------------------------------------------------------------------------

def test_zero_skill_market_tracking_model_scores_near_zero_ic():
    """
    REGRESSION TEST FOR DEFECT 1.

    Fails against the old pooled `spearmanr(y_pred, y_true)`, which scored this
    zero-skill model at ~0.7 and cleared MIN_TEST_IC=0.02 by more than 30x.
    """
    dates, preds, y = _zero_skill_panel()

    res = information_coefficient(preds, y, dates)
    assert abs(res.mean_ic) < 0.05, f"zero-skill model scored IC={res.mean_ic:.4f}"

    # And the statistic the old code computed is enormous on the SAME data —
    # this is what made the defect invisible.
    pooled = _pooled_rank_correlation(preds, y)
    assert pooled > 0.5
    assert pooled > 10 * abs(res.mean_ic)

    # Zero skill must also be statistically indistinguishable from zero.
    assert abs(res.t_stat) < 2.0
    assert res.p_value > 0.05


def test_information_coefficient_requires_a_date_axis():
    """The old two-argument (pooled) call form must be impossible, not silent."""
    _, preds, y = _zero_skill_panel(n_dates=20)
    with pytest.raises(TypeError):
        _information_coefficient(preds, y)  # type: ignore[call-arg]


def test_genuine_cross_sectional_skill_scores_clearly_positive_ic():
    """The metric is not merely always ~0: real skill must show up."""
    dates, preds, y = _skilled_panel()
    res = information_coefficient(preds, y, dates)
    assert res.mean_ic > 0.15, f"skilled model only scored IC={res.mean_ic:.4f}"
    assert res.t_stat > 3.0
    assert res.p_value < 0.01


def test_ic_is_computed_within_each_date_then_averaged():
    """Definitional check against a hand-rolled per-date loop."""
    dates, preds, y = _skilled_panel(n_dates=40, n_names=30, seed=3)
    manual = [
        spearmanr(preds[dates == t], y[dates == t]).statistic for t in np.unique(dates)
    ]
    res = information_coefficient(preds, y, dates)
    assert res.n_dates == len(manual)
    np.testing.assert_allclose(np.sort(res.ic_series), np.sort(manual), atol=1e-12)
    assert res.mean_ic == pytest.approx(float(np.mean(manual)), abs=1e-12)


def test_ic_is_invariant_to_a_per_date_level_shift():
    """
    Adding a market-level offset to every prediction on a date changes nothing
    cross-sectionally.  A pooled IC would move a lot; a cross-sectional one must not.
    """
    dates, preds, y = _skilled_panel(n_dates=60, seed=5)
    rng = np.random.default_rng(0)
    shifts = {t: rng.normal(0, 5.0) for t in np.unique(dates)}
    shifted = preds + np.array([shifts[t] for t in dates])

    base = information_coefficient(preds, y, dates)
    bumped = information_coefficient(shifted, y, dates)
    assert bumped.mean_ic == pytest.approx(base.mean_ic, abs=1e-12)
    # The pooled statistic is NOT invariant — that is exactly the defect.
    assert abs(_pooled_rank_correlation(shifted, y)
               - _pooled_rank_correlation(preds, y)) > 0.05


# ---------------------------------------------------------------------------
# Defect 2 — ICIR must be mean/std across dates, not a copy of IC
# ---------------------------------------------------------------------------

def test_icir_equals_mean_over_std_across_dates_and_differs_from_ic():
    """REGRESSION TEST FOR DEFECT 2 (`test_icir = test_ic`)."""
    dates, preds, y = _skilled_panel()
    res = information_coefficient(preds, y, dates)

    ic_dates, ics = cross_sectional_ic_series(preds, y, dates)
    expected = float(np.mean(ics)) / float(np.std(ics, ddof=1))

    assert res.icir == pytest.approx(expected, rel=1e-12)
    assert res.std_ic > 0
    # ICIR carries a dispersion term, so it cannot equal the IC itself.
    assert res.icir != pytest.approx(res.mean_ic, abs=1e-6)
    assert abs(res.icir - res.mean_ic) > 0.1


def test_icir_annualization_and_tstat_use_the_overlap_deflated_sample():
    """REGRESSION TEST FOR DEFECT 6 (overlapping labels)."""
    dates, preds, y = _skilled_panel()
    h = 5
    daily = information_coefficient(preds, y, dates, label_horizon=1)
    overlap = information_coefficient(preds, y, dates, label_horizon=h)

    # Raw ICIR is horizon-independent; annualisation and significance are not.
    assert overlap.icir == pytest.approx(daily.icir, rel=1e-12)
    assert overlap.icir_annualized == pytest.approx(
        daily.icir * math.sqrt(DEFAULT_PERIODS_PER_YEAR / h), rel=1e-12
    )
    assert overlap.n_effective == pytest.approx(daily.n_dates / h)
    # Overlapping labels must SHRINK the t-stat by ~sqrt(h).
    assert overlap.t_stat == pytest.approx(daily.t_stat / math.sqrt(h), rel=1e-9)
    assert overlap.t_stat < daily.t_stat


def test_single_cross_section_reports_nan_icir_rather_than_faking_one():
    """A one-date sample has no dispersion, so ICIR must be NaN, not the IC."""
    _, preds, y = _skilled_panel(n_dates=1, n_names=60, seed=2)
    res = information_coefficient(preds, y, np.zeros(len(y)))
    assert np.isfinite(res.mean_ic)
    assert np.isnan(res.icir)
    assert np.isnan(res.t_stat)


# ---------------------------------------------------------------------------
# Defect 5 — a field named "sharpe" must be a Sharpe ratio
# ---------------------------------------------------------------------------

def test_long_short_sharpe_is_a_real_sharpe_not_a_single_period_spread():
    """REGRESSION TEST FOR DEFECT 5."""
    dates, preds, y = _skilled_panel()
    res = long_short_sharpe(preds, y, dates, label_horizon=1)

    assert res.n_periods > 100
    assert res.volatility > 0
    expected = res.mean_return / res.volatility * math.sqrt(DEFAULT_PERIODS_PER_YEAR)
    assert res.sharpe == pytest.approx(expected, rel=1e-12)

    # The old return value (a bare spread) is still available, honestly named,
    # and is nowhere near the Sharpe in magnitude.
    assert res.ls_spread == pytest.approx(res.mean_return, rel=1e-12)
    assert abs(res.sharpe - res.ls_spread) > 1.0
    assert res.sharpe > 1.0  # genuine skill => a real, positive annualised Sharpe


def test_long_short_sharpe_is_scale_invariant_but_spread_is_not():
    """A Sharpe has a volatility denominator; a raw spread does not."""
    dates, preds, y = _skilled_panel(n_dates=120, seed=13)
    base = long_short_sharpe(preds, y, dates)
    scaled = long_short_sharpe(preds, 3.0 * y, dates)
    assert scaled.sharpe == pytest.approx(base.sharpe, rel=1e-9)
    assert scaled.ls_spread == pytest.approx(3.0 * base.ls_spread, rel=1e-9)


def test_zero_skill_long_short_sharpe_is_not_significant():
    dates, preds, y = _zero_skill_panel()
    res = long_short_sharpe(preds, y, dates)
    assert abs(res.sharpe) < 2.0


# ---------------------------------------------------------------------------
# Defects 4 & 6 — causal, embargoed internal splits
# ---------------------------------------------------------------------------

def test_internal_cv_splits_are_causal_and_embargoed():
    """REGRESSION TEST FOR DEFECTS 4 & 6."""
    embargo = 5
    splits = _embargoed_time_series_splits(n_samples=200, n_splits=5, embargo=embargo)
    assert len(splits) >= 2
    for train_idx, val_idx in splits:
        assert train_idx.max() < val_idx.min(), "training rows must precede validation"
        assert val_idx.min() - train_idx.max() - 1 >= embargo, "embargo not applied"

    # Contrast: the KFold that `cv=5` resolved to trains on FUTURE rows.
    kf_train, kf_val = next(iter(KFold(n_splits=5, shuffle=False).split(np.zeros((200, 1)))))
    assert kf_train.max() > kf_val.min(), "KFold sanity check: it is acausal"


def test_splits_never_cut_a_date_cross_section_in_half():
    dates = np.repeat(np.arange(100), 10)
    splits = _embargoed_time_series_splits(
        n_samples=len(dates), n_splits=4, embargo=2, dates=dates
    )
    assert splits
    for train_idx, val_idx in splits:
        assert not set(dates[train_idx]) & set(dates[val_idx])
        assert dates[train_idx].max() < dates[val_idx].min()
        assert dates[val_idx].min() - dates[train_idx].max() - 1 >= 2


def test_linear_alpha_model_uses_causal_cv_for_the_ridge_penalty():
    rng = np.random.default_rng(4)
    n = 400
    X = rng.normal(size=(n, 3))
    y = 0.5 * X[:, 0] + rng.normal(scale=1.0, size=n)
    model = LinearAlphaModel(label_horizon=3, n_splits=4)
    model.fit(X, y, X[-50:], y[-50:], feature_names=["a", "b", "c"])

    assert model._cv_splits, "no CV splits were recorded"
    for train_idx, val_idx in model._cv_splits:
        assert train_idx.max() < val_idx.min()
        assert val_idx.min() - train_idx.max() - 1 >= 3  # embargo == label_horizon


# ---------------------------------------------------------------------------
# Defect 3 — selection must be validation-only, with a multiple-testing check
# ---------------------------------------------------------------------------

class _FeatureModel(AlphaModelBase):
    """Deterministic model: prediction = a fixed linear combination of features."""

    def __init__(self, name: str, weights):
        self.model_name = name
        self._w = np.asarray(weights, dtype=float)

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None,
            dates_train=None, dates_val=None):
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self._w

    def predict_proba(self, X):
        p = self.predict(X)
        return 1.0 / (1.0 + np.exp(-p / (p.std() or 1.0)))

    def feature_importance(self):
        import pandas as pd
        return pd.Series(np.abs(self._w))


def _competition_panel(seed=21, n_dates=180, n_names=40):
    """
    Two informative features.  Feature 0 dominates on the validation set, so a
    validation-only selector must always pick the model that trades feature 0.
    """
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.arange(n_dates), n_names)
    X = rng.normal(size=(len(dates), 2))
    mkt = np.repeat(rng.normal(0, 0.02, n_dates), n_names)
    y = mkt + 0.02 * (1.0 * X[:, 0] + 0.45 * X[:, 1]) + 0.02 * rng.normal(size=len(dates))
    cut1, cut2 = n_dates // 2 * n_names, (n_dates * 3 // 4) * n_names
    sl = (slice(0, cut1), slice(cut1, cut2), slice(cut2, None))
    return [(X[s], y[s], dates[s]) for s in sl]


def test_model_selection_is_invariant_to_the_test_set():
    """
    REGRESSION TEST FOR DEFECT 3.

    Only the TEST data is perturbed.  The old code gated on
    `test_directional_accuracy` and ranked on `(val_ic + test_ic) / 2`, so the
    perturbation flipped the winner.  Selection must now be unchanged.
    """
    (Xtr, ytr, dtr), (Xv, yv, dv), (Xte, yte, dte) = _competition_panel()
    comp = ModelCompetition()

    def run(y_test):
        models = [_FeatureModel("uses_f0", [1.0, 0.0]),
                  _FeatureModel("uses_f1", [0.0, 1.0])]
        return comp.compare_models(
            models, Xtr, ytr, Xv, yv, Xte, y_test,
            dates_train=dtr, dates_val=dv, dates_test=dte, label_horizon=1,
        )

    baseline = run(yte)
    # Perturb ONLY the test labels, in the way most favourable to the OTHER model.
    perturbed = run(Xte[:, 1].copy())

    assert baseline.best_model_name == "uses_f0"
    assert perturbed.best_model_name == baseline.best_model_name, (
        "selection changed when only the TEST set changed"
    )

    # Validation metrics are untouched; test metrics moved a great deal.
    base_by = {m.model_name: m for m in baseline.metrics}
    pert_by = {m.model_name: m for m in perturbed.metrics}
    for name in base_by:
        assert pert_by[name].val_ic == pytest.approx(base_by[name].val_ic, abs=1e-12)
        assert pert_by[name].val_icir == pytest.approx(base_by[name].val_icir, abs=1e-12)
    assert pert_by["uses_f1"].test_ic > 0.99  # f1 now perfectly predicts the test set
    assert pert_by["uses_f1"].test_ic - base_by["uses_f1"].test_ic > 0.5

    # Proof the perturbation WOULD have flipped the old (val_ic + test_ic)/2 rule.
    def old_rule(by):
        return max(by.values(), key=lambda m: (m.val_ic + m.test_ic) / 2).model_name
    assert old_rule(base_by) == "uses_f0"
    assert old_rule(pert_by) == "uses_f1", "perturbation was not material enough"
    assert perturbed.selection_basis == "validation-only"


def test_selection_gate_reads_validation_directional_accuracy_only():
    """A test set that is pure noise must not disqualify a model with val skill."""
    (Xtr, ytr, dtr), (Xv, yv, dv), (Xte, yte, dte) = _competition_panel(seed=31)
    rng = np.random.default_rng(99)
    noise_test = rng.normal(size=len(yte))
    res = ModelCompetition().compare_models(
        [_FeatureModel("uses_f0", [1.0, 0.0]), _FeatureModel("uses_f1", [0.0, 1.0])],
        Xtr, ytr, Xv, yv, Xte, noise_test,
        dates_train=dtr, dates_val=dv, dates_test=dte,
    )
    winner = next(m for m in res.metrics if m.model_name == res.best_model_name)
    assert res.passed_selection_gate is True
    assert winner.val_directional_accuracy > ModelCompetition.MIN_DIR_ACC
    assert winner.test_directional_accuracy < ModelCompetition.MIN_DIR_ACC  # noise test set


def test_noise_winner_is_flagged_as_not_significant_by_multiple_testing():
    """
    REGRESSION TEST FOR DEFECT 3(b)/(c).

    30 pure-noise candidates, a plain `max` over them, and a winner that looks
    respectable.  The adjustment must refuse to call it skill, and N must be
    recorded in the result.
    """
    n_dates, n_names, n_models = 180, 40, 30
    rng = np.random.default_rng(1234)
    dates = np.repeat(np.arange(n_dates), n_names)
    X = rng.normal(size=(len(dates), 6))
    y = rng.normal(size=len(dates))  # labels independent of every feature
    cut1, cut2 = n_dates // 2 * n_names, (n_dates * 3 // 4) * n_names

    models = [
        _FeatureModel(f"noise_{i}", rng.normal(size=6)) for i in range(n_models)
    ]
    res = ModelCompetition().compare_models(
        models,
        X[:cut1], y[:cut1], X[cut1:cut2], y[cut1:cut2], X[cut2:], y[cut2:],
        dates_train=dates[:cut1], dates_val=dates[cut1:cut2], dates_test=dates[cut2:],
        label_horizon=5,
    )

    assert res.n_candidates == n_models, "N must be recorded for auditability"
    mt = res.multiple_testing
    assert mt is not None and mt.n_trials == n_models

    winner = next(m for m in res.metrics if m.model_name == res.best_model_name)
    assert winner.val_ic > 0  # the winner LOOKS positive — that is the trap
    assert res.winner_survives_multiple_testing is False
    assert mt.deflated_sharpe_ratio < mt.threshold
    assert "DOES NOT SURVIVE" in mt.verdict and "DOES NOT SURVIVE" in res.selection_reason
    assert mt.expected_max_sr_sqrt2logn > 0


def test_deflated_sharpe_ratio_still_passes_genuinely_strong_evidence():
    """The adjustment must not reject everything — a strong result survives."""
    strong = deflated_sharpe_ratio(0.60, n_obs=250, n_trials=30)
    assert strong.significant is True
    assert strong.deflated_sharpe_ratio > 0.95
    assert "SURVIVES" in strong.verdict

    # Same statistic, far more trials and far less data => no longer credible.
    weak = deflated_sharpe_ratio(0.60, n_obs=20, n_trials=1000)
    assert weak.significant is False


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    sds = 1.0 / math.sqrt(100)
    e10 = deflated_sharpe_ratio(0.0, n_obs=100, n_trials=10).expected_max_sr_null
    e100 = deflated_sharpe_ratio(0.0, n_obs=100, n_trials=100).expected_max_sr_null
    assert 0 < e10 < e100
    # Bracketed by the coarse sqrt(2 ln N) benchmark's order of magnitude.
    assert e100 == pytest.approx(math.sqrt(2 * math.log(100)) * sds, rel=0.35)


def test_compare_models_requires_a_date_axis():
    (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _) = _competition_panel(n_dates=40)
    with pytest.raises(ValueError, match="cross-sectional|date axis|CROSS-SECTIONAL"):
        ModelCompetition().compare_models(
            [_FeatureModel("m", [1.0, 0.0])], Xtr, ytr, Xv, yv, Xte, yte
        )
