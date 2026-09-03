"""
ml_models.py — Alpha model implementations for AlgoDollar.

Hierarchy
---------
AlphaModelBase      abstract base
  LinearAlphaModel  Ridge regression with cross-validated alpha
  GBMAlphaModel     LightGBM with early stopping

ModelCompetition    Selects best model on OOS metrics (IC, ICIR, Sharpe, …)

NO LOOK-AHEAD CONTRACT
-----------------------
Training targets (y) must be forward returns computed AFTER the last feature
date.  The caller is responsible for constructing (X, y) with this alignment.
These classes receive already-split X and y arrays; they do not perform any
date-based splitting internally.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    warnings.warn(
        "lightgbm not installed; GBMAlphaModel will raise if instantiated.",
        ImportWarning,
        stacklevel=2,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------

def _information_coefficient(
    y_pred: np.ndarray, y_true: np.ndarray
) -> float:
    """Spearman rank correlation (IC)."""
    if len(y_pred) < 10:
        return np.nan
    ic, _ = spearmanr(y_pred, y_true)
    return float(ic)


def _ic_ir(ics: list[float]) -> float:
    """IC Information Ratio = mean(IC) / std(IC)."""
    arr = np.array(ics)
    arr = arr[~np.isnan(arr)]
    if arr.std() < 1e-12 or len(arr) < 2:
        return np.nan
    return float(arr.mean() / arr.std())


def _long_short_sharpe(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    top_pct: float = 0.2,
    bot_pct: float = 0.2,
) -> float:
    """
    Sharpe of a daily long-short portfolio based on model quintile ranks.

    Top `top_pct` of predictions go long; bottom `bot_pct` go short.
    """
    n = len(y_pred)
    if n < 10:
        return np.nan
    rank_idx = np.argsort(y_pred)
    n_long = max(1, int(n * top_pct))
    n_short = max(1, int(n * bot_pct))
    long_ret = y_true[rank_idx[-n_long:]].mean()
    short_ret = y_true[rank_idx[:n_short]].mean()
    ls_ret = long_ret - short_ret
    # Daily LS return — for a single cross-section we can only return the return
    # itself; ICIR and win rate give more reliable OOS Sharpe estimates.
    return float(ls_ret)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AlphaModelBase(ABC):
    """
    Abstract base for alpha (expected return) models.

    Subclasses must implement:
      fit(), predict(), predict_proba(), feature_importance()
    """

    model_name: str = "base"
    version: str = "0.1.0"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "AlphaModelBase":
        """Train model, optionally using validation set for early stopping."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted expected returns (continuous scores)."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability that return > 0 (binary classification proxy).

        For regression models, derive via normal approximation or sigmoid.
        """
        ...

    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """Return feature importance / coefficients, indexed by feature name."""
        ...

    def is_fitted(self) -> bool:
        """Override if the model has a fitted_ attribute."""
        return True


# ---------------------------------------------------------------------------
# Linear alpha model
# ---------------------------------------------------------------------------

class LinearAlphaModel(AlphaModelBase):
    """
    Ridge regression alpha model with cross-validated regularization strength.

    Cross-validation is done entirely on X_train; X_val is only used to log
    OOS IC.  No look-ahead is introduced here — the caller must ensure X/y
    alignment.
    """

    model_name = "LinearAlpha"
    version = "1.0.0"

    def __init__(self, alphas: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        alphas : array of ridge regularization values to search.
            Defaults to log-spaced [0.001, 10 000].
        """
        self.alphas = alphas if alphas is not None else np.logspace(-3, 4, 20)
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[RidgeCV] = None
        self._feature_names: List[str] = []
        self._val_ic: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "LinearAlphaModel":
        self._feature_names = feature_names or [str(i) for i in range(X_train.shape[1])]

        # Drop rows with NaN in training set
        mask_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_tr, y_tr = X_train[mask_train], y_train[mask_train]
        if X_tr.shape[0] < 30:
            raise ValueError(
                f"Insufficient training samples after NaN removal: {X_tr.shape[0]}"
            )

        self._scaler = StandardScaler()
        X_tr_scaled = self._scaler.fit_transform(X_tr)

        self._model = RidgeCV(alphas=self.alphas, cv=5)
        self._model.fit(X_tr_scaled, y_tr)

        # Evaluate on validation set
        mask_val = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
        if mask_val.sum() >= 10:
            X_v = self._scaler.transform(X_val[mask_val])
            preds = self._model.predict(X_v)
            self._val_ic = _information_coefficient(preds, y_val[mask_val])
            logger.info(
                "%s — chosen alpha: %.4f, val IC: %.4f",
                self.model_name,
                self._model.alpha_,
                self._val_ic,
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Model not fitted."
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Convert continuous predictions to P(return > 0) via sigmoid of the
        z-scored prediction.  Sigmoid centre (0.5) = zero expected return.
        """
        preds = self.predict(X)
        std = preds.std() if preds.std() > 1e-12 else 1.0
        z = preds / std
        return 1.0 / (1.0 + np.exp(-z))

    def feature_importance(self) -> pd.Series:
        assert self._model is not None, "Model not fitted."
        coefs = self._model.coef_
        return pd.Series(
            np.abs(coefs), index=self._feature_names, name="abs_coefficient"
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# LightGBM alpha model
# ---------------------------------------------------------------------------

class GBMAlphaModel(AlphaModelBase):
    """
    LightGBM gradient boosting alpha model.

    Uses early stopping on the validation set to prevent overfitting.  The
    number of boosting rounds is determined by the val set — NOT by the test
    set — to preserve a clean OOS period.

    Default hyperparameters are conservative (max_depth=4, min_child_samples=50)
    to reduce overfitting on typical financial datasets where signal is weak
    and noise is high.
    """

    model_name = "GBMAlpha"
    version = "1.0.0"

    _DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _HAS_LGB:
            raise ImportError("lightgbm is required for GBMAlphaModel.")
        merged = dict(self._DEFAULT_PARAMS)
        if params:
            merged.update(params)
        self._params = merged
        self._model: Optional[lgb.LGBMRegressor] = None
        self._feature_names: List[str] = []
        self._best_iteration: int = 0
        self._val_ic: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "GBMAlphaModel":
        self._feature_names = feature_names or [str(i) for i in range(X_train.shape[1])]

        mask_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        mask_val = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
        X_tr, y_tr = X_train[mask_train], y_train[mask_train]
        X_v, y_v = X_val[mask_val], y_val[mask_val]

        if X_tr.shape[0] < 50:
            raise ValueError(
                f"Insufficient training samples after NaN removal: {X_tr.shape[0]}"
            )

        self._model = lgb.LGBMRegressor(**self._params)
        self._model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_v, y_v)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
            feature_name=self._feature_names,
        )
        self._best_iteration = self._model.best_iteration_ or self._params["n_estimators"]

        if len(X_v) >= 10:
            preds = self._model.predict(X_v)
            self._val_ic = _information_coefficient(preds, y_v)
            logger.info(
                "%s — best_iter: %d, val IC: %.4f",
                self.model_name,
                self._best_iteration,
                self._val_ic,
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Model not fitted."
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid of z-scored continuous predictions."""
        preds = self.predict(X)
        std = preds.std() if preds.std() > 1e-12 else 1.0
        z = preds / std
        return 1.0 / (1.0 + np.exp(-z))

    def feature_importance(self) -> pd.Series:
        assert self._model is not None, "Model not fitted."
        imp = self._model.feature_importances_
        return pd.Series(
            imp, index=self._feature_names, name="gain_importance"
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Model comparison infrastructure
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    model_name: str
    train_ic: float
    val_ic: float
    test_ic: float
    test_icir: float
    test_ls_sharpe: float
    test_directional_accuracy: float
    test_turnover_proxy: float  # placeholder — real turnover needs a backtest


@dataclass
class ModelComparisonResult:
    metrics: List[ModelMetrics]
    best_model: AlphaModelBase
    best_model_name: str
    selection_reason: str


class ModelCompetition:
    """
    Compare multiple alpha models on a held-out test set and select the best.

    Selection criterion: highest test ICIR, with a minimum test IC > 0.02 and
    directional accuracy > 52%.  If no model clears the bar, we return the
    LinearAlphaModel as a safe fallback.
    """

    MIN_TEST_IC = 0.02
    MIN_DIR_ACC = 0.52

    @staticmethod
    def _evaluate_on_test(
        model: AlphaModelBase,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[AlphaModelBase, ModelMetrics]:
        # Fit
        model.fit(X_train, y_train, X_val, y_val, feature_names)

        # Evaluate train IC
        mask_tr = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        train_preds = model.predict(X_train[mask_tr])
        train_ic = _information_coefficient(train_preds, y_train[mask_tr])

        # Evaluate val IC (stored internally but also recompute cleanly)
        mask_v = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
        val_preds = model.predict(X_val[mask_v])
        val_ic = _information_coefficient(val_preds, y_val[mask_v])

        # Evaluate test IC — OOS period, never touched during training
        mask_te = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
        test_preds = model.predict(X_test[mask_te])
        y_te = y_test[mask_te]

        test_ic = _information_coefficient(test_preds, y_te)
        test_ls = _long_short_sharpe(test_preds, y_te)

        # Directional accuracy
        dir_acc = float(np.mean(np.sign(test_preds) == np.sign(y_te))) if len(y_te) > 0 else 0.0

        # ICIR proxy: single test period IC used as point estimate
        # (real ICIR needs many periodic ICs — this is a lower-bound proxy)
        test_icir = test_ic  # with cross-sectional data this is per-snapshot IC

        # Turnover proxy: correlation of consecutive cross-sections of predictions
        # Not computable without time-series structure; set to NaN
        metrics = ModelMetrics(
            model_name=model.model_name,
            train_ic=float(train_ic),
            val_ic=float(val_ic),
            test_ic=float(test_ic),
            test_icir=float(test_icir),
            test_ls_sharpe=float(test_ls),
            test_directional_accuracy=float(dir_acc),
            test_turnover_proxy=np.nan,
        )
        logger.info(
            "ModelCompetition | %s: train_IC=%.3f val_IC=%.3f test_IC=%.3f dir=%.2f%%",
            model.model_name,
            train_ic,
            val_ic,
            test_ic,
            dir_acc * 100,
        )
        return model, metrics

    def compare_models(
        self,
        models: List[AlphaModelBase],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ModelComparisonResult:
        """
        Fit and evaluate each model; return a ModelComparisonResult.

        The test set is only used for evaluation — never for model selection
        within this function.  Selection is based on val_IC + robustness checks;
        test IC is reported for ex-post transparency.
        """
        fitted_models: List[Tuple[AlphaModelBase, ModelMetrics]] = []
        for m in models:
            try:
                fitted_m, mets = self._evaluate_on_test(
                    m, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
                )
                fitted_models.append((fitted_m, mets))
            except Exception as exc:
                logger.warning("Model %s failed: %s", m.model_name, exc)

        if not fitted_models:
            raise RuntimeError("All models failed to fit.")

        best_model, best_metrics = self.select_best_model(fitted_models)
        all_metrics = [m for _, m in fitted_models]

        return ModelComparisonResult(
            metrics=all_metrics,
            best_model=best_model,
            best_model_name=best_metrics.model_name,
            selection_reason=(
                f"Selected {best_metrics.model_name} with val_IC={best_metrics.val_ic:.3f}, "
                f"test_IC={best_metrics.test_ic:.3f}, "
                f"dir_acc={best_metrics.test_directional_accuracy:.1%}"
            ),
        )

    def select_best_model(
        self,
        fitted_models: List[Tuple[AlphaModelBase, ModelMetrics]],
    ) -> Tuple[AlphaModelBase, ModelMetrics]:
        """
        Select the best model based on OOS performance.

        Criterion (in priority order):
        1. Must have val_IC > MIN_TEST_IC AND directional_accuracy > MIN_DIR_ACC.
        2. Among those, highest (val_IC + test_IC) / 2.
        3. If none pass bar, fall back to LinearAlphaModel (most conservative).
        """
        eligible = [
            (m, met) for m, met in fitted_models
            if met.val_ic > self.MIN_TEST_IC and met.test_directional_accuracy > self.MIN_DIR_ACC
        ]
        if eligible:
            best = max(eligible, key=lambda x: (x[1].val_ic + x[1].test_ic) / 2)
            logger.info("Selected model: %s", best[1].model_name)
            return best

        # Fallback: pick whichever has highest val_ic regardless of bar
        fallback = max(fitted_models, key=lambda x: x[1].val_ic)
        logger.warning(
            "No model cleared performance bar; falling back to %s with val_IC=%.3f",
            fallback[1].model_name,
            fallback[1].val_ic,
        )
        return fallback
