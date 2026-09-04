"""
A LightGBM cross-sectional ranker, measured against the simple baselines.

THE TARGET IS DEFINED ONCE AND EXPLICITLY
-----------------------------------------
Predict the CROSS-SECTIONAL forward return: for each date, which stocks
outperform the others over the next ``horizon`` sessions. The label is
demeaned within each date, so the model cannot score by predicting the market
— it has to rank names against each other on the same day, which is the only
thing a long-only top-quantile portfolio actually uses.

HOW LEAKAGE IS PREVENTED
------------------------
1. Every feature is a TRAILING window of prices up to and including ``t``.
2. The label is the forward return from ``t+1`` to ``t+1+horizon``, matching the
   backtester's execution lag exactly.
3. Training and prediction are separated by a PURGE of ``horizon`` sessions plus
   an embargo. Without the purge, the last training labels overlap the first
   prediction dates — the model is then fitted on the very returns it is asked
   to predict, which produces a spectacular and entirely fake IC.
4. Features are standardised CROSS-SECTIONALLY (within each date), never across
   time, so no sample-wide mean or standard deviation leaks backwards.

WHAT WOULD MAKE IT REJECTED
---------------------------
It must beat the simple baselines out-of-sample AFTER costs. A model that
matches a 12-1 momentum rule is not adding anything a rule cannot do, and
carries model risk, retraining risk and opacity that the rule does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.research.backtest import run_backtest
from app.research.signals import MONTH, zscore_cross_section

logger = logging.getLogger(__name__)

#: Forward horizon of the prediction target, in sessions. One week. Chosen to
#: match the weekly rebalance the backtester uses, not searched.
DEFAULT_HORIZON = 5


@dataclass
class MLComparison:
    trained: bool
    reason: str = ""
    oos_sharpe: Optional[float] = None
    oos_cagr: Optional[float] = None
    oos_ic_mean: Optional[float] = None
    baseline_best_sharpe: Optional[float] = None
    baseline_best_name: Optional[str] = None
    improvement: Optional[float] = None
    verdict: str = "NOT EVALUATED"
    folds: list[dict] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)


def build_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Trailing price features, one frame per feature, all knowable at ``t``.

    Deliberately few and conventional. A wide feature set searched for
    predictive power on this dataset would be the overfitting the validation
    layer exists to catch, applied at the one point it cannot see.
    """
    ret = prices.pct_change()
    return {
        "mom_12_1": np.log(prices.shift(MONTH) / prices.shift(12 * MONTH)),
        "mom_6_1": np.log(prices.shift(MONTH) / prices.shift(6 * MONTH)),
        "reversal_21": -np.log(prices / prices.shift(MONTH)),
        "vol_63": ret.rolling(3 * MONTH, min_periods=MONTH).std(),
        "trend_50_200": (
            prices.rolling(50, min_periods=50).mean()
            / prices.rolling(200, min_periods=200).mean() - 1.0
        ),
        "range_pos_126": (
            (prices - prices.shift(1).rolling(126, min_periods=63).min())
            / (
                prices.shift(1).rolling(126, min_periods=63).max()
                - prices.shift(1).rolling(126, min_periods=63).min()
            ).replace(0.0, np.nan)
        ),
    }


def build_label(prices: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """
    Cross-sectionally demeaned forward return, matching the execution lag.

    Entry at the close of t+1, exit at t+1+horizon — identical to the
    backtester's ``lag=2`` convention. Demeaning per date removes the market
    component so the model must rank, not forecast the index.
    """
    fwd = prices.shift(-(1 + horizon)) / prices.shift(-1) - 1.0
    return fwd.sub(fwd.mean(axis=1), axis=0)


def _stack(
    features: dict[str, pd.DataFrame], label: pd.DataFrame, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Long-format (X, y, date_index) over the given dates, NaN rows dropped."""
    names = sorted(features)
    frames = []
    for feat in names:
        f = features[feat].reindex(dates)
        frames.append(zscore_cross_section(f).stack(future_stack=True).rename(feat))
    y = label.reindex(dates).stack(future_stack=True).rename("y")
    joined = pd.concat(frames + [y], axis=1).dropna()
    if joined.empty:
        return np.empty((0, len(names))), np.empty(0), pd.DatetimeIndex([]), names
    X = joined[names].to_numpy(dtype=float)
    yy = joined["y"].to_numpy(dtype=float)
    idx = joined.index.get_level_values(0)
    return X, yy, pd.DatetimeIndex(idx), names


def run_ml_comparison(
    prices: pd.DataFrame,
    *,
    benchmark: Optional[pd.Series] = None,
    baseline_best_sharpe: Optional[float] = None,
    baseline_best_name: Optional[str] = None,
    horizon: int = DEFAULT_HORIZON,
    n_folds: int = 4,
    min_train_days: int = 756,
    embargo_days: int = 10,
    cost_bps: float = 25.0,
) -> MLComparison:
    """
    Walk-forward LightGBM, compared with the best simple baseline.

    Returns a comparison rather than a model: the question is whether ML earns
    its complexity, and the answer is allowed to be no.
    """
    try:
        import lightgbm as lgb
    except Exception as exc:  # noqa: BLE001
        return MLComparison(
            trained=False,
            reason=f"LightGBM unavailable: {exc}",
            verdict="NOT EVALUATED",
        )

    features = build_features(prices)
    label = build_label(prices, horizon)
    dates = prices.index
    n = len(dates)
    if n < min_train_days + n_folds * 60:
        return MLComparison(
            trained=False, reason=f"insufficient history ({n} sessions)",
            verdict="NOT EVALUATED",
        )

    test_size = (n - min_train_days) // n_folds
    oos_signal = pd.DataFrame(index=dates, columns=prices.columns, dtype=float)
    fold_records: list[dict] = []
    ics: list[float] = []
    feat_names: list[str] = []

    for k in range(n_folds):
        train_end_i = min_train_days + k * test_size
        # PURGE the horizon, then EMBARGO. The label at the last training date
        # depends on prices `horizon` sessions later, which fall inside the test
        # window; without removing them the model trains on its own test data.
        train_end_i -= horizon
        test_start_i = train_end_i + horizon + embargo_days
        test_end_i = min(test_start_i + test_size, n)
        if test_start_i >= n or test_end_i - test_start_i < 20:
            break

        tr_dates = dates[:train_end_i]
        te_dates = dates[test_start_i:test_end_i]

        Xtr, ytr, _, feat_names = _stack(features, label, tr_dates)
        Xte, yte, te_idx, _ = _stack(features, label, te_dates)
        if len(Xtr) < 5000 or len(Xte) < 200:
            continue

        model = lgb.LGBMRegressor(
            # Deliberately small and heavily regularised. This is a BASELINE
            # model, not a tuned one — no hyperparameter search was run, because
            # searching here is exactly the overfitting the study measures.
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        # Rebuild the wide prediction panel for the backtester.
        _, _, _, _ = None, None, None, None
        frames = []
        for feat in sorted(features):
            f = features[feat].reindex(te_dates)
            frames.append(zscore_cross_section(f).stack(future_stack=True).rename(feat))
        yv = label.reindex(te_dates).stack(future_stack=True).rename("y")
        joined = pd.concat(frames + [yv], axis=1).dropna()
        pred_s = pd.Series(pred, index=joined.index)
        wide = pred_s.unstack()
        oos_signal.loc[wide.index, wide.columns] = wide.to_numpy()

        # Per-date cross-sectional IC (Spearman), the honest measure of ranking
        # skill. A POOLED correlation over all dates would mostly measure market
        # co-movement and reads far higher than any real skill.
        tmp = pd.DataFrame({"p": pred, "y": yte}, index=joined.index)
        per_date = tmp.groupby(level=0).apply(
            lambda g: g["p"].corr(g["y"], method="spearman") if len(g) > 5 else np.nan
        )
        fold_ic = float(per_date.mean()) if len(per_date) else float("nan")
        if np.isfinite(fold_ic):
            ics.append(fold_ic)

        fold_records.append({
            "fold": k,
            "train_end": str(dates[train_end_i - 1].date()),
            "test_start": str(dates[test_start_i].date()),
            "test_end": str(dates[test_end_i - 1].date()),
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "mean_ic": round(fold_ic, 5) if np.isfinite(fold_ic) else None,
        })

    valid = oos_signal.dropna(how="all")
    if valid.empty:
        return MLComparison(
            trained=False, reason="no out-of-sample predictions were produced",
            verdict="NOT EVALUATED", folds=fold_records,
        )

    sub_prices = prices.loc[valid.index]
    res = run_backtest(
        valid, sub_prices, benchmark=benchmark, cost_bps=cost_bps
    )
    m = res.metrics
    improvement = (
        m.sharpe - baseline_best_sharpe if baseline_best_sharpe is not None else None
    )

    if baseline_best_sharpe is None:
        verdict = "NO BASELINE TO COMPARE"
    elif improvement is not None and improvement > 0.20:
        verdict = "ML ADDS VALUE OVER BASELINE"
    else:
        verdict = "ML REJECTED — no meaningful improvement over the simple baseline"

    return MLComparison(
        trained=True,
        reason="",
        oos_sharpe=round(m.sharpe, 4),
        oos_cagr=round(m.cagr, 4),
        oos_ic_mean=round(float(np.mean(ics)), 5) if ics else None,
        baseline_best_sharpe=baseline_best_sharpe,
        baseline_best_name=baseline_best_name,
        improvement=round(improvement, 4) if improvement is not None else None,
        verdict=verdict,
        folds=fold_records,
        feature_names=feat_names,
    )
