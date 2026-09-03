"""Model drift detection: feature drift, prediction drift, performance drift."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelHealth(str, Enum):
    HEALTHY = "HEALTHY"
    MONITORING = "MONITORING"
    DRIFTED = "DRIFTED"
    RETRAIN_REQUIRED = "RETRAIN_REQUIRED"


@dataclass
class FeatureDriftResult:
    feature: str
    ks_pvalue: float
    psi: float
    drifted: bool
    severity: str        # "none" | "minor" | "significant"


@dataclass
class DriftReport:
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    overall_drifted: bool = False
    drifted_features: list[str] = field(default_factory=list)
    psi_max: float = 0.0
    recommendation: str = ""


@dataclass
class _ModelRecord:
    """Internal state stored per model."""
    predictions: list[dict] = field(default_factory=list)    # {timestamp, prediction}
    outcomes: list[dict] = field(default_factory=list)        # {timestamp, actual_return}
    feature_history: list[dict] = field(default_factory=list) # {timestamp, features}


class DriftDetector:
    """
    Tracks model predictions and outcomes to detect concept/feature drift.

    Thread-safety: not guaranteed; call from a single async task.
    """

    PSI_SIGNIFICANT = 0.2     # PSI > 0.2 → significant drift
    PSI_MINOR = 0.1           # PSI in (0.1, 0.2) → minor shift
    KS_ALPHA = 0.05           # reject H0 at 5%
    PERF_Z_THRESHOLD = -2.0   # z-score below which performance is drifted

    def __init__(self) -> None:
        self._models: dict[str, _ModelRecord] = {}

    def _get_model(self, model_name: str) -> _ModelRecord:
        if model_name not in self._models:
            self._models[model_name] = _ModelRecord()
        return self._models[model_name]

    # ------------------------------------------------------------------ #
    #  Event tracking                                                      #
    # ------------------------------------------------------------------ #

    def track_prediction(
        self,
        model_name: str,
        features: dict[str, float],
        prediction: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a model prediction and its input features."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        rec = self._get_model(model_name)
        rec.predictions.append({"timestamp": ts, "prediction": prediction})
        rec.feature_history.append({"timestamp": ts, "features": dict(features)})
        logger.debug("DriftDetector: tracked prediction for %s at %s", model_name, ts)

    def track_outcome(
        self,
        model_name: str,
        symbol: str,
        actual_return: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record the realised outcome for a previous prediction."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        rec = self._get_model(model_name)
        rec.outcomes.append({
            "timestamp": ts,
            "symbol": symbol,
            "actual_return": actual_return,
        })

    # ------------------------------------------------------------------ #
    #  Feature drift                                                       #
    # ------------------------------------------------------------------ #

    def detect_feature_drift(
        self,
        recent_features_df: pd.DataFrame,
        reference_features_df: pd.DataFrame,
    ) -> DriftReport:
        """
        Compare feature distributions using KS test and PSI.

        Parameters
        ----------
        recent_features_df    : recent window (e.g. last 20 days)
        reference_features_df : reference / training distribution

        Returns
        -------
        DriftReport with per-feature results
        """
        from scipy import stats  # type: ignore[import]

        feature_results: list[FeatureDriftResult] = []
        drifted: list[str] = []
        psi_max = 0.0

        common_cols = [
            c for c in recent_features_df.columns
            if c in reference_features_df.columns
        ]

        for col in common_cols:
            recent = recent_features_df[col].dropna().values
            reference = reference_features_df[col].dropna().values

            if len(recent) < 5 or len(reference) < 5:
                continue

            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(recent, reference)

            # PSI
            psi = self._compute_psi(reference, recent)
            psi_max = max(psi_max, psi)

            is_drifted = psi > self.PSI_SIGNIFICANT or ks_pvalue < self.KS_ALPHA
            if psi > self.PSI_SIGNIFICANT:
                severity = "significant"
            elif psi > self.PSI_MINOR or ks_pvalue < self.KS_ALPHA:
                severity = "minor"
            else:
                severity = "none"

            result = FeatureDriftResult(
                feature=col,
                ks_pvalue=round(float(ks_pvalue), 6),
                psi=round(float(psi), 6),
                drifted=is_drifted,
                severity=severity,
            )
            feature_results.append(result)
            if is_drifted:
                drifted.append(col)
                logger.warning(
                    "Feature drift detected: %s PSI=%.4f KS_p=%.4f",
                    col, psi, ks_pvalue,
                )

        overall = len(drifted) > 0
        if psi_max > self.PSI_SIGNIFICANT:
            recommendation = "Significant feature drift detected. Consider retraining."
        elif drifted:
            recommendation = f"Minor drift in: {drifted}. Monitor closely."
        else:
            recommendation = "No significant feature drift detected."

        return DriftReport(
            feature_results=feature_results,
            overall_drifted=overall,
            drifted_features=drifted,
            psi_max=round(psi_max, 6),
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------ #
    #  Prediction drift                                                    #
    # ------------------------------------------------------------------ #

    def detect_prediction_drift(
        self,
        recent_preds: list[float],
        reference_preds: list[float],
    ) -> DriftReport:
        """
        Compare distributions of model predictions (KS + PSI).
        Useful for detecting model output shift even if features look stable.
        """
        from scipy import stats  # type: ignore[import]

        recent = np.array(recent_preds, dtype=float)
        reference = np.array(reference_preds, dtype=float)

        if len(recent) < 5 or len(reference) < 5:
            return DriftReport(
                overall_drifted=False,
                recommendation="Insufficient data for prediction drift analysis.",
            )

        ks_stat, ks_pvalue = stats.ks_2samp(recent, reference)
        psi = self._compute_psi(reference, recent)
        drifted = psi > self.PSI_SIGNIFICANT or ks_pvalue < self.KS_ALPHA

        result = FeatureDriftResult(
            feature="prediction_distribution",
            ks_pvalue=round(float(ks_pvalue), 6),
            psi=round(float(psi), 6),
            drifted=drifted,
            severity="significant" if psi > self.PSI_SIGNIFICANT else (
                "minor" if drifted else "none"
            ),
        )

        rec = "Prediction distribution has drifted." if drifted else "Prediction drift: OK."
        return DriftReport(
            feature_results=[result],
            overall_drifted=drifted,
            drifted_features=["prediction_distribution"] if drifted else [],
            psi_max=round(float(psi), 6),
            recommendation=rec,
        )

    # ------------------------------------------------------------------ #
    #  Performance drift                                                   #
    # ------------------------------------------------------------------ #

    def detect_performance_drift(
        self,
        model_name: str,
        lookback: int = 20,
    ) -> bool:
        """
        Compare recent hit_rate (last `lookback` outcomes) vs historical.

        A prediction is a "hit" if sign(prediction) == sign(actual_return).

        Returns True if recent performance is statistically significantly worse.
        """
        rec = self._get_model(model_name)
        preds = rec.predictions
        outcomes = rec.outcomes

        if len(preds) < lookback * 2 or len(outcomes) < lookback * 2:
            logger.debug(
                "Not enough data for performance drift on %s (%d preds, %d outcomes).",
                model_name, len(preds), len(outcomes),
            )
            return False

        # Match predictions to outcomes by index (assume same order)
        n = min(len(preds), len(outcomes))
        preds_arr = np.array([p["prediction"] for p in preds[:n]])
        outcomes_arr = np.array([o["actual_return"] for o in outcomes[:n]])

        hits = (np.sign(preds_arr) == np.sign(outcomes_arr)).astype(float)

        recent_hits = hits[-lookback:]
        historical_hits = hits[:-lookback]

        recent_rate = float(recent_hits.mean())
        historical_rate = float(historical_hits.mean())
        historical_std = float(historical_hits.std()) or 1e-9

        z_score = (recent_rate - historical_rate) / (historical_std / math.sqrt(lookback))

        logger.debug(
            "Performance drift %s: recent_hr=%.3f historical_hr=%.3f z=%.2f",
            model_name, recent_rate, historical_rate, z_score,
        )

        if z_score < self.PERF_Z_THRESHOLD:
            logger.warning(
                "Performance drift detected for %s: z=%.2f (threshold=%.2f)",
                model_name, z_score, self.PERF_Z_THRESHOLD,
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Model health                                                        #
    # ------------------------------------------------------------------ #

    def get_model_health(self, model_name: str) -> ModelHealth:
        """
        Aggregate health classification.

        HEALTHY          : no drift detected
        MONITORING       : minor drift, worth watching
        DRIFTED          : significant feature or prediction drift
        RETRAIN_REQUIRED : performance has degraded significantly
        """
        rec = self._get_model(model_name)

        if not rec.predictions:
            return ModelHealth.HEALTHY

        # Build quick DFs from stored history
        if rec.feature_history:
            recent_n = min(len(rec.feature_history), 20)
            ref_n = len(rec.feature_history) - recent_n
            if ref_n >= 5:
                recent_rows = [r["features"] for r in rec.feature_history[-recent_n:]]
                ref_rows = [r["features"] for r in rec.feature_history[:ref_n]]
                recent_df = pd.DataFrame(recent_rows)
                ref_df = pd.DataFrame(ref_rows)
                try:
                    feature_report = self.detect_feature_drift(recent_df, ref_df)
                    if feature_report.psi_max > self.PSI_SIGNIFICANT:
                        # Also check performance
                        if self.detect_performance_drift(model_name):
                            return ModelHealth.RETRAIN_REQUIRED
                        return ModelHealth.DRIFTED
                    if feature_report.overall_drifted:
                        return ModelHealth.MONITORING
                except Exception as exc:
                    logger.warning("Feature drift check failed: %s", exc)

        if self.detect_performance_drift(model_name):
            return ModelHealth.RETRAIN_REQUIRED

        return ModelHealth.HEALTHY

    # ------------------------------------------------------------------ #
    #  PSI helper                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_psi(
        reference: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Population Stability Index between two 1D arrays.

        PSI = Σ (actual_% - reference_%) * ln(actual_% / reference_%)
        """
        # Use reference quantiles as bin edges
        eps = 1e-6
        breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)

        def bucket_pcts(arr: np.ndarray) -> np.ndarray:
            counts, _ = np.histogram(arr, bins=breakpoints)
            pcts = counts / max(len(arr), 1)
            return np.clip(pcts, eps, None)

        ref_pct = bucket_pcts(reference)
        act_pct = bucket_pcts(actual)

        # Normalise so they sum to 1 (handle edge cases)
        ref_pct = ref_pct / ref_pct.sum()
        act_pct = act_pct / act_pct.sum()

        psi = float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))
        return max(0.0, psi)
