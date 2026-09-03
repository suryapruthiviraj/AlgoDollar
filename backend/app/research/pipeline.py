"""
End-to-end cross-sectional research pipeline.

WHY THIS MODULE EXISTS
----------------------
The alpha models, the strategies and the feature engine all existed in this
codebase but nothing connected them. There was no code path that took price
data, built labels, trained a model on a causal split, produced out-of-sample
predictions and scored them. The "machine learning" was therefore untested and
unreachable: the strategies fell through to a hardcoded heuristic, and the
model classes had no call sites at all.

This module is that missing path. It is deliberately model-agnostic — it takes
any object exposing `fit(X, y)` and `predict(X)` — so the model layer can
change without touching the validation logic.

THE MEASUREMENT THAT MATTERS
----------------------------
For a cross-sectional stock-ranking strategy the relevant question is not
"does the model predict returns accurately?" but "on a given day, does the
model rank stocks in the right order?" Those are different questions and they
have different answers.

A model that predicts every stock will rise 2% on days the market rises, and
fall 2% on days it falls, has excellent pooled correlation with realized
returns and zero stock-picking skill. Pooling across dates measures market
timing and disguises it as selection skill. So IC here is always computed
WITHIN each date's cross-section and then averaged.

CAUSALITY
---------
A label at date t is the forward return from t to t+h, which is by
construction unknown at t. That is fine for training — it is the thing being
predicted — but it means the last h observations before any test window carry
labels contaminated by test-period prices. `PurgedWalkForward` removes them.
Turning purging off will make results look better; that improvement is
leakage, not alpha.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np
import pandas as pd
from scipy import stats

from app.research.statistics import deflated_sharpe_ratio
from app.research.validation import (
    PurgedWalkForward,
    assert_no_train_test_overlap,
    deflated_tstat,
    effective_sample_size,
)

logger = logging.getLogger(__name__)


class SupportsFitPredict(Protocol):
    """Minimal model interface this pipeline needs."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_forward_return_labels(
    prices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Forward log return over `horizon` periods.

    label[t, s] = log(price[t + horizon, s] / price[t, s])

    This is FUTURE information relative to t — that is the point, it is the
    prediction target. The final `horizon` rows are NaN because their outcome
    has not happened yet, and must never be filled.

    Parameters
    ----------
    prices : DataFrame (T x N), close prices, DatetimeIndex.
    horizon : int, forward periods.

    Returns
    -------
    DataFrame (T x N) of forward returns, NaN in the last `horizon` rows.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    fwd = np.log(prices.shift(-horizon) / prices)
    return fwd


def stack_to_panel(
    features_wide: pd.DataFrame,
    labels: pd.DataFrame,
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Convert wide per-symbol columns into a long (date, symbol) panel.

    The feature engine emits columns named "{feature}__{symbol}", which is a
    poor layout for cross-sectional modelling: every date/symbol pair needs to
    be one training row. This reshapes to a MultiIndex panel and aligns labels
    to it.

    Parameters
    ----------
    features_wide : DataFrame (T x (F*N)) with "{feature}__{symbol}" columns.
    labels : DataFrame (T x N) of forward returns.
    symbols : list of symbols to include.

    Returns
    -------
    (X, y, dates)
        X : DataFrame indexed by (date, symbol), one column per feature.
        y : Series indexed by (date, symbol).
        dates : the unique dates present, ascending.
    """
    feat_names = sorted({c.split("__")[0] for c in features_wide.columns if "__" in c})
    if not feat_names:
        raise ValueError(
            "No columns of the form '{feature}__{symbol}' found. "
            f"Got: {list(features_wide.columns)[:5]}"
        )

    frames = []
    for sym in symbols:
        cols = {}
        for f in feat_names:
            col = f"{f}__{sym}"
            if col in features_wide.columns:
                cols[f] = features_wide[col]
        if not cols:
            continue
        sub = pd.DataFrame(cols)
        # Canonical, stable column order. Gathering columns by suffix in
        # DataFrame order lets features silently land in the wrong model
        # position when a symbol is missing one of them.
        sub = sub.reindex(columns=feat_names)
        sub["__symbol__"] = sym
        if sym in labels.columns:
            sub["__label__"] = labels[sym]
        else:
            sub["__label__"] = np.nan
        frames.append(sub)

    if not frames:
        raise ValueError("No symbols produced any features.")

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.set_index("__symbol__", append=True)
    panel.index.names = ["date", "symbol"]
    panel = panel.sort_index()

    y = panel.pop("__label__")
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    return panel, y, dates


# ---------------------------------------------------------------------------
# Cross-sectional scoring
# ---------------------------------------------------------------------------

def cross_sectional_ic(
    predictions: pd.Series,
    labels: pd.Series,
    min_names_per_date: int = 5,
) -> pd.Series:
    """
    Spearman rank correlation between prediction and outcome, WITHIN each date.

    This is the correct information coefficient for a cross-sectional strategy.
    Pooling all dates together instead measures whether the model's overall
    level tracks the market, which a model with no stock-selection skill can
    do perfectly.

    Parameters
    ----------
    predictions, labels : Series indexed by (date, symbol).
    min_names_per_date : int
        Dates with fewer valid names are skipped; a rank correlation over 2
        or 3 names is noise.

    Returns
    -------
    Series indexed by date, one IC per date.
    """
    df = pd.DataFrame({"pred": predictions, "label": labels}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    out = {}
    for date, grp in df.groupby(level="date"):
        if len(grp) < min_names_per_date:
            continue
        if grp["pred"].nunique() < 2 or grp["label"].nunique() < 2:
            continue
        ic, _ = stats.spearmanr(grp["pred"].values, grp["label"].values)
        if np.isfinite(ic):
            out[date] = float(ic)

    return pd.Series(out).sort_index()


def long_short_returns(
    predictions: pd.Series,
    labels: pd.Series,
    quantile: float = 0.2,
    min_names_per_date: int = 5,
) -> pd.Series:
    """
    Per-date return of a long-top-quantile / short-bottom-quantile portfolio.

    This is the tradeable expression of the model's ranking. It is what the
    IC should translate into, and comparing the two is a useful consistency
    check: a high IC that produces no long-short spread usually means the
    signal is concentrated in names you cannot actually trade.

    Returns
    -------
    Series indexed by date of equal-weighted long-short returns (gross of
    costs — costs are applied by the backtester, not here).
    """
    df = pd.DataFrame({"pred": predictions, "label": labels}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    out = {}
    for date, grp in df.groupby(level="date"):
        if len(grp) < min_names_per_date:
            continue
        k = max(1, int(len(grp) * quantile))
        ranked = grp.sort_values("pred", ascending=False)
        long_leg = ranked.head(k)["label"].mean()
        short_leg = ranked.tail(k)["label"].mean()
        out[date] = float(long_leg - short_leg)

    return pd.Series(out).sort_index()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    n_purged: int
    n_embargoed: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    mean_ic: float
    ic_std: float
    ls_return_mean: float


@dataclass
class ResearchResult:
    """Out-of-sample result of a full purged walk-forward research run."""
    folds: list[FoldResult]
    oos_ic_series: pd.Series          # per-date IC, concatenated across folds
    oos_ls_returns: pd.Series         # per-date long-short return
    mean_ic: float
    ic_std: float
    icir: float                       # mean IC / std IC, annualized
    ic_tstat_naive: float             # t-stat using nominal n (overstated)
    ic_tstat_deflated: float          # t-stat using effective n
    ls_sharpe_annual: float
    label_horizon: int
    n_trials: int
    dsr: Optional[float] = None       # deflated Sharpe of the long-short book
    dsr_significant: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"OOS mean cross-sectional IC : {self.mean_ic:+.4f} "
            f"(sd {self.ic_std:.4f}, n={len(self.oos_ic_series)} dates)",
            f"ICIR (annualized)           : {self.icir:+.3f}",
            f"IC t-stat (nominal n)       : {self.ic_tstat_naive:+.2f}  "
            f"<- overstated with overlapping labels",
            f"IC t-stat (effective n)     : {self.ic_tstat_deflated:+.2f}  "
            f"<- use this one",
            f"Long-short Sharpe (annual)  : {self.ls_sharpe_annual:+.3f}",
        ]
        if self.dsr is not None:
            verdict = "SIGNIFICANT" if self.dsr_significant else "NOT SIGNIFICANT"
            lines.append(
                f"Deflated Sharpe ({self.n_trials} trials): "
                f"{self.dsr:.4f} [{verdict}]"
            )
        for n in self.notes:
            lines.append(f"NOTE: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def run_cross_sectional_research(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], SupportsFitPredict],
    label_horizon: int,
    n_splits: int = 5,
    embargo_frac: float = 0.01,
    n_trials: int = 1,
    min_names_per_date: int = 5,
    verify_splits: bool = True,
) -> ResearchResult:
    """
    Train and evaluate a cross-sectional alpha model under purged walk-forward.

    Parameters
    ----------
    X : DataFrame indexed by (date, symbol), features in columns.
    y : Series indexed by (date, symbol), forward returns.
    model_factory : callable returning a FRESH unfitted model each call.
        A fresh instance per fold is required: reusing a fitted model would
        carry information from later folds backwards.
    label_horizon : int
        Forward periods in the label. Drives purging and the effective sample
        size. Passing 1 when labels are 5-day returns silently reintroduces
        leakage.
    n_splits : int
    embargo_frac : float
    n_trials : int
        Number of configurations tried before choosing this one. Used for the
        multiple-testing adjustment. Report it honestly.
    min_names_per_date : int
    verify_splits : bool
        Assert every split is leakage-free. Cheap; leave it on.

    Returns
    -------
    ResearchResult
    """
    if not isinstance(X.index, pd.MultiIndex):
        raise TypeError("X must have a (date, symbol) MultiIndex")

    dates = pd.DatetimeIndex(
        X.index.get_level_values("date").unique()
    ).sort_values()

    splitter = PurgedWalkForward(
        n_splits=n_splits,
        label_horizon=label_horizon,
        embargo_frac=embargo_frac,
        expanding=True,
    )

    folds: list[FoldResult] = []
    ic_pieces: list[pd.Series] = []
    ls_pieces: list[pd.Series] = []
    notes: list[str] = []

    for i, split in enumerate(splitter.split(dates)):
        if verify_splits:
            assert_no_train_test_overlap(split, dates, label_horizon)

        train_dates = dates[split.train_idx]
        test_dates = dates[split.test_idx]

        tr_mask = X.index.get_level_values("date").isin(train_dates)
        te_mask = X.index.get_level_values("date").isin(test_dates)

        X_tr_raw, y_tr_raw = X[tr_mask], y[tr_mask]
        X_te_raw, y_te_raw = X[te_mask], y[te_mask]

        # Drop rows with missing features or labels. Imputation here would be
        # a modelling decision; make it explicit upstream instead of hiding it.
        tr_ok = X_tr_raw.notna().all(axis=1) & y_tr_raw.notna()
        te_ok = X_te_raw.notna().all(axis=1)

        X_tr, y_tr = X_tr_raw[tr_ok], y_tr_raw[tr_ok]
        X_te = X_te_raw[te_ok]
        y_te = y_te_raw[te_ok]

        if len(X_tr) < 50 or len(X_te) < min_names_per_date:
            notes.append(
                f"fold {i} skipped: {len(X_tr)} train rows, {len(X_te)} test rows"
            )
            continue

        model = model_factory()
        model.fit(X_tr.values, y_tr.values)
        preds = pd.Series(model.predict(X_te.values), index=X_te.index)

        ic = cross_sectional_ic(preds, y_te, min_names_per_date)
        ls = long_short_returns(preds, y_te, min_names_per_date=min_names_per_date)

        ic_pieces.append(ic)
        ls_pieces.append(ls)

        folds.append(FoldResult(
            fold=i,
            n_train=len(X_tr), n_test=len(X_te),
            n_purged=split.n_purged, n_embargoed=split.n_embargoed,
            train_start=split.train_start, train_end=split.train_end,
            test_start=split.test_start, test_end=split.test_end,
            mean_ic=float(ic.mean()) if len(ic) else float("nan"),
            ic_std=float(ic.std(ddof=1)) if len(ic) > 1 else float("nan"),
            ls_return_mean=float(ls.mean()) if len(ls) else float("nan"),
        ))
        logger.info(
            "fold %d | %s | mean IC=%+.4f", i, split.describe(), folds[-1].mean_ic
        )

    if not folds:
        raise RuntimeError(
            "No usable folds. Check that features and labels overlap in time "
            "and that enough non-NaN rows survive."
        )

    all_ic = pd.concat(ic_pieces).sort_index() if ic_pieces else pd.Series(dtype=float)
    all_ls = pd.concat(ls_pieces).sort_index() if ls_pieces else pd.Series(dtype=float)

    mean_ic = float(all_ic.mean()) if len(all_ic) else float("nan")
    ic_std = float(all_ic.std(ddof=1)) if len(all_ic) > 1 else float("nan")

    # ICIR annualized by the number of independent (non-overlapping)
    # observation periods per year, not by the raw date count.
    periods_per_year = 252.0 / max(label_horizon, 1)
    icir = (
        float(mean_ic / ic_std * np.sqrt(periods_per_year))
        if ic_std and np.isfinite(ic_std) and ic_std > 0
        else float("nan")
    )

    t_naive = (
        float(mean_ic / (ic_std / np.sqrt(len(all_ic))))
        if ic_std and ic_std > 0 and len(all_ic) > 1
        else float("nan")
    )
    t_defl = deflated_tstat(mean_ic, ic_std, len(all_ic), label_horizon)

    ls_sharpe = float("nan")
    dsr_val: Optional[float] = None
    dsr_sig = False
    if len(all_ls) > 2 and all_ls.std(ddof=1) > 0:
        ls_sharpe = float(
            all_ls.mean() / all_ls.std(ddof=1) * np.sqrt(periods_per_year)
        )
        try:
            d = deflated_sharpe_ratio(all_ls.values, n_trials=n_trials)
            dsr_val = d.deflated_sharpe_ratio
            dsr_sig = d.is_significant
        except ValueError as exc:
            notes.append(f"DSR unavailable: {exc}")

    if label_horizon > 1:
        notes.append(
            f"Labels overlap ({label_horizon}-period horizon sampled every "
            f"period). Effective sample size is "
            f"{effective_sample_size(len(all_ic), label_horizon):.0f}, not "
            f"{len(all_ic)}. The deflated t-stat accounts for this."
        )

    return ResearchResult(
        folds=folds,
        oos_ic_series=all_ic,
        oos_ls_returns=all_ls,
        mean_ic=mean_ic,
        ic_std=ic_std,
        icir=icir,
        ic_tstat_naive=t_naive,
        ic_tstat_deflated=t_defl,
        ls_sharpe_annual=ls_sharpe,
        label_horizon=label_horizon,
        n_trials=n_trials,
        dsr=dsr_val,
        dsr_significant=dsr_sig,
        notes=notes,
    )
