"""
model_registry.py — Versioned alpha model storage and retrieval.

ModelRegistry manages the lifecycle of trained alpha models:
  - Registration with rich metadata (git commit, training dates, OOS metrics).
  - Persistence to disk via joblib.
  - Active-model selection per strategy.
  - Model listing and auditing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from app.models.ml_models import AlphaModelBase  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    """
    Metadata associated with a registered model version.

    Fields
    ------
    model_name : str
        Matches AlphaModelBase.model_name.
    version : str
        Semantic version string, e.g. "1.2.0".
    strategy : str
        Which strategy this model serves, e.g. "swing", "intraday", "longterm".
    git_commit : str
        SHA of the code commit that produced the model.
    training_start : datetime
        First date of training data used.
    training_end : datetime
        Last date of training data used.  This is the LAST date whose label
        (forward return) is fully observed — i.e., training ends at least
        holding_period days before the current date.
    oos_sharpe : float
        Out-of-sample walk-forward Sharpe (annualized).
    is_active : bool
        Whether this is the currently active model for its strategy.
    registered_at : datetime
        Wall-clock timestamp of registration.
    notes : str
        Free-form audit notes.
    """
    model_name: str
    version: str
    strategy: str
    git_commit: str
    training_start: datetime
    training_end: datetime
    oos_sharpe: float
    is_active: bool = False
    registered_at: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "strategy": self.strategy,
            "git_commit": self.git_commit,
            "training_start": self.training_start.isoformat(),
            "training_end": self.training_end.isoformat(),
            "oos_sharpe": self.oos_sharpe,
            "is_active": self.is_active,
            "registered_at": self.registered_at.isoformat(),
            "notes": self.notes,
        }


@dataclass
class ModelVersion:
    """Lightweight summary used by list_models()."""
    model_name: str
    version: str
    strategy: str
    oos_sharpe: float
    is_active: bool
    training_end: datetime
    path: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Thread-safe (single-process) alpha model registry.

    Models are saved as compressed joblib files.  Metadata is stored in memory
    and optionally persisted to a JSON manifest file for auditability.

    Parameters
    ----------
    base_dir : str | Path
        Root directory for model artefacts.  Defaults to ``./models``.
    """

    _MANIFEST_FILE = "registry_manifest.json"

    def __init__(self, base_dir: str | Path = "./models"):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, List[tuple[AlphaModelBase, ModelMetadata]]] = {}
        self._load_manifest()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model: AlphaModelBase,
        metadata: ModelMetadata,
        auto_save: bool = True,
    ) -> str:
        """
        Register a model version.

        Parameters
        ----------
        model : AlphaModelBase
        metadata : ModelMetadata
        auto_save : bool
            Immediately persist to disk.

        Returns
        -------
        str : path where the model was saved.
        """
        strategy = metadata.strategy
        if strategy not in self._registry:
            self._registry[strategy] = []

        # If this is marked active, deactivate all previous active models
        if metadata.is_active:
            for _, prev_meta in self._registry[strategy]:
                prev_meta.is_active = False

        path = ""
        if auto_save:
            path = self.save_model(model, metadata)
            metadata.notes += f" | saved_to={path}"

        self._registry[strategy].append((model, metadata))
        self._save_manifest()
        logger.info(
            "Registered model %s v%s for strategy=%s (OOS Sharpe=%.2f, active=%s)",
            metadata.model_name,
            metadata.version,
            strategy,
            metadata.oos_sharpe,
            metadata.is_active,
        )
        return path

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_active_model(self, strategy: str) -> AlphaModelBase:
        """
        Return the currently active model for a strategy.

        Raises
        ------
        KeyError if no active model is registered for the strategy.
        """
        versions = self._registry.get(strategy, [])
        for model, meta in reversed(versions):
            if meta.is_active:
                return model
        raise KeyError(
            f"No active model found for strategy '{strategy}'. "
            f"Registered versions: {[m.version for _, m in versions]}"
        )

    def list_models(self) -> List[ModelVersion]:
        """Return summary of all registered model versions across all strategies."""
        result = []
        for strategy, versions in self._registry.items():
            for _, meta in versions:
                result.append(ModelVersion(
                    model_name=meta.model_name,
                    version=meta.version,
                    strategy=strategy,
                    oos_sharpe=meta.oos_sharpe,
                    is_active=meta.is_active,
                    training_end=meta.training_end,
                    path=self._model_path(meta),
                ))
        return sorted(result, key=lambda x: (x.strategy, x.training_end), reverse=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, model: AlphaModelBase, metadata: ModelMetadata) -> str:
        """
        Save model to disk using joblib compression.

        Returns the absolute path of the saved file.
        """
        fname = self._model_path(metadata)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        payload = {"model": model, "metadata": metadata}
        joblib.dump(payload, fname, compress=3)
        logger.info("Saved model to %s", fname)
        return fname

    def load_model(self, path: str) -> AlphaModelBase:
        """
        Load a model from disk.

        Parameters
        ----------
        path : str
            Absolute or relative path to the joblib file.

        Returns
        -------
        AlphaModelBase
        """
        payload = joblib.load(path)
        model: AlphaModelBase = payload["model"]
        meta: ModelMetadata = payload["metadata"]
        logger.info(
            "Loaded model %s v%s (strategy=%s, OOS Sharpe=%.2f)",
            meta.model_name,
            meta.version,
            meta.strategy,
            meta.oos_sharpe,
        )
        return model

    def load_model_with_metadata(
        self, path: str
    ) -> tuple[AlphaModelBase, ModelMetadata]:
        payload = joblib.load(path)
        return payload["model"], payload["metadata"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _model_path(self, metadata: ModelMetadata) -> str:
        """Deterministic file path from metadata fields."""
        safe_version = metadata.version.replace(".", "_")
        fname = f"{metadata.model_name}_{safe_version}_{metadata.training_end.strftime('%Y%m%d')}.joblib"
        return str(self._base_dir / metadata.strategy / fname)

    def _save_manifest(self) -> None:
        """Persist metadata manifest as JSON for auditability."""
        import json
        manifest = []
        for strategy, versions in self._registry.items():
            for _, meta in versions:
                row = meta.to_dict()
                row["file_path"] = self._model_path(meta)
                manifest.append(row)
        manifest_path = self._base_dir / self._MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    def _load_manifest(self) -> None:
        """Load manifest on startup (metadata only; models loaded lazily)."""
        import json
        manifest_path = self._base_dir / self._MANIFEST_FILE
        if not manifest_path.exists():
            return
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            for row in manifest:
                # Only load metadata into the registry; skip actual model objects
                # (load_model() is called lazily when needed)
                meta = ModelMetadata(
                    model_name=row["model_name"],
                    version=row["version"],
                    strategy=row["strategy"],
                    git_commit=row.get("git_commit", "unknown"),
                    training_start=datetime.fromisoformat(row["training_start"]),
                    training_end=datetime.fromisoformat(row["training_end"]),
                    oos_sharpe=float(row["oos_sharpe"]),
                    is_active=bool(row["is_active"]),
                    registered_at=datetime.fromisoformat(row["registered_at"]),
                    notes=row.get("notes", ""),
                )
                strategy = meta.strategy
                if strategy not in self._registry:
                    self._registry[strategy] = []
                # Placeholder — actual model object not in registry until loaded
                self._registry[strategy].append((_LazyModelPlaceholder(row["file_path"]), meta))
        except Exception as exc:
            logger.warning("Could not load model registry manifest: %s", exc)


class _LazyModelPlaceholder(AlphaModelBase):
    """
    Placeholder that loads the real model from disk on first use.

    This avoids loading all model files into memory on registry startup.
    """

    model_name = "_lazy_placeholder"
    version = "0"

    def __init__(self, path: str):
        self._path = path
        self._real: Optional[AlphaModelBase] = None

    def _ensure_loaded(self):
        if self._real is None:
            payload = joblib.load(self._path)
            self._real = payload["model"]

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None):
        raise NotImplementedError("Lazy placeholder: call load_model() first.")

    def predict(self, X):
        self._ensure_loaded()
        return self._real.predict(X)

    def predict_proba(self, X):
        self._ensure_loaded()
        return self._real.predict_proba(X)

    def feature_importance(self):
        self._ensure_loaded()
        return self._real.feature_importance()
