"""
ml_models.py — Alpha model implementations for AlgoDollar.

Hierarchy
---------
AlphaModelBase      abstract base
  LinearAlphaModel  Ridge regression with cross-validated alpha
  GBMAlphaModel     LightGBM with early stopping

ModelCompetition    Selects best model on VALIDATION metrics (IC, ICIR, …) and
                    reports a multiple-testing-adjusted assessment of the winner.

NO LOOK-AHEAD CONTRACT
-----------------------
1. Training targets (y) must be forward returns computed AFTER the last feature
   date.  The caller is responsible for constructing (X, y) with this alignment.
2. These classes never split randomly.  Any internal split (e.g. the ridge
   penalty search) is chronological AND embargoed: `label_horizon` observation
   dates are dropped between the end of a train fold and the start of the
   following validation fold, so that a training label whose forward window
   overlaps the validation period cannot leak into penalty selection.
3. OVERLAPPING LABELS.  With an `label_horizon`-day forward label sampled every
   period, adjacent labels share (label_horizon - 1) / label_horizon of their
   forward window.  Consecutive per-date ICs are therefore NOT independent and
   the effective sample size is
       n_effective = n_dates / label_horizon
   Every t-statistic produced here is computed on `n_effective`, not `n_dates`,
   and every annualisation uses sqrt(periods_per_year / label_horizon).  Callers
   MUST pass the true `label_horizon`; leaving it at the default of 1 while
   using multi-day labels overstates significance by ~sqrt(label_horizon).
4. All skill metrics are CROSS-SECTIONAL: computed within each date, then
   averaged across dates.  A pooled correlation over a stacked panel measures
   time-series co-movement with the market, not stock-picking skill, and is
   never used for any gate or ranking decision.

Purged/embargoed splitting: if `app.research.validation` is available it is used;
otherwise a minimal embargoed chronological splitter is implemented inline (see
`_embargoed_time_series_splits`).
"""

from __future__ import annotations

import inspect
import logging
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from scipy.stats import t as student_t
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

# Optional: a full purged/embargoed splitter owned by app/research/validation.py.
# If that module is not present we fall back to the inline embargoed splitter
# below (which is intentionally minimal: chronological + embargo, no purge of
# train labels that straddle the fold boundary from the *left*).
try:  # pragma: no cover - depends on a module owned by another workstream
    from app.research import validation as _research_validation
    _HAS_RESEARCH_VALIDATION = True
except Exception:  # ImportError or anything raised at import time
    _research_validation = None
    _HAS_RESEARCH_VALIDATION = False

logger = logging.getLogger(__name__)

DEFAULT_PERIODS_PER_YEAR = 252
_EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------

@dataclass
class ICResult:
    """
    Cross-sectional information coefficient statistics.

    Attributes
    ----------
    mean_ic : average of the per-date cross-sectional Spearman ICs.
    std_ic  : sample standard deviation (ddof=1) of the per-date IC series.
    icir    : RAW information ratio, mean_ic / std_ic, in per-period units.
    icir_annualized : icir * sqrt(periods_per_year / label_horizon).  The
              overlap divisor is applied because overlapping labels produce
              only periods_per_year / label_horizon independent bets a year.
    t_stat  : mean_ic / (std_ic / sqrt(n_effective)) with
              n_effective = n_dates / label_horizon (overlap deflation).
    p_value : two-sided p-value of t_stat under Student-t(n_effective - 1).
    n_dates : number of dates with a computable IC.
    n_effective : deflated (independent-observation) sample size.
    ic_series / ic_dates : the per-date IC series and its date labels.
    """

    mean_ic: float
    std_ic: float
    icir: float
    icir_annualized: float
    t_stat: float
    p_value: float
    n_dates: int
    n_effective: float
    label_horizon: int
    periods_per_year: int
    ic_series: np.ndarray = field(default_factory=lambda: np.array([]))
    ic_dates: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def empty(cls, label_horizon: int = 1,
              periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> "ICResult":
        return cls(
            mean_ic=np.nan, std_ic=np.nan, icir=np.nan, icir_annualized=np.nan,
            t_stat=np.nan, p_value=np.nan, n_dates=0, n_effective=0.0,
            label_horizon=label_horizon, periods_per_year=periods_per_year,
        )


def _group_slices(dates: Sequence[Any]) -> List[Tuple[Any, np.ndarray]]:
    """Return [(date, row_indices), …] grouped by date, in sorted date order."""
    arr = np.asarray(dates)
    if arr.ndim != 1:
        raise ValueError(f"`dates` must be 1-D, got shape {arr.shape}")
    codes, uniques = pd.factorize(arr, sort=True)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    chunks = np.split(order, boundaries)
    out: List[Tuple[Any, np.ndarray]] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        out.append((uniques[sorted_codes[chunk[0]]], chunk))
    return out


def cross_sectional_ic_series(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates: Sequence[Any],
    min_names_per_date: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spearman IC computed WITHIN each date's cross-section.

    Returns (dates_with_ic, ic_values).  Dates with fewer than
    `min_names_per_date` usable names, or with a constant prediction / return
    vector (rank correlation undefined), are dropped.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if len(y_pred) != len(y_true) or len(y_pred) != len(np.asarray(dates)):
        raise ValueError(
            "y_pred, y_true and dates must have equal length; got "
            f"{len(y_pred)}, {len(y_true)}, {len(np.asarray(dates))}"
        )

    out_dates: List[Any] = []
    out_ics: List[float] = []
    for date, idx in _group_slices(dates):
        p, a = y_pred[idx], y_true[idx]
        ok = np.isfinite(p) & np.isfinite(a)
        if ok.sum() < min_names_per_date:
            continue
        p, a = p[ok], a[ok]
        # Rank correlation is undefined when either side has no dispersion.
        if np.ptp(p) <= 0 or np.ptp(a) <= 0:
            continue
        ic = spearmanr(p, a).statistic
        if not np.isfinite(ic):
            continue
        out_dates.append(date)
        out_ics.append(float(ic))
    return np.asarray(out_dates), np.asarray(out_ics, dtype=float)


def information_coefficient(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates: Sequence[Any],
    *,
    label_horizon: int = 1,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    min_names_per_date: int = 5,
    min_dates: int = 2,
) -> ICResult:
    """
    Mean CROSS-SECTIONAL information coefficient and its dispersion.

    `dates` is REQUIRED and is the date (group) axis of the panel: the Spearman
    correlation is computed inside each date, and only then averaged.  Pooling a
    stacked panel into one correlation measures market co-movement, not
    stock-picking skill — a model whose prediction level merely tracks the
    market scores ~0.7 pooled and ~0.0 cross-sectionally.

    Significance is reported on the OVERLAP-DEFLATED sample size
    n_effective = n_dates / label_horizon (see module docstring, item 3).
    """
    label_horizon = max(1, int(label_horizon))
    ic_dates, ics = cross_sectional_ic_series(
        y_pred, y_true, dates, min_names_per_date=min_names_per_date
    )
    n = len(ics)
    if n == 0:
        return ICResult.empty(label_horizon, periods_per_year)

    mean_ic = float(np.mean(ics))
    if n < max(2, min_dates):
        # A single cross-section gives a point IC but no dispersion, hence no
        # ICIR and no t-statistic.  Say so with NaN rather than faking one.
        return ICResult(
            mean_ic=mean_ic, std_ic=np.nan, icir=np.nan, icir_annualized=np.nan,
            t_stat=np.nan, p_value=np.nan, n_dates=n,
            n_effective=float(n) / label_horizon,
            label_horizon=label_horizon, periods_per_year=periods_per_year,
            ic_series=ics, ic_dates=ic_dates,
        )

    std_ic = float(np.std(ics, ddof=1))
    n_eff = max(1.0, float(n) / label_horizon)
    if std_ic < 1e-12:
        icir = icir_ann = t_stat = p_value = np.nan
    else:
        icir = mean_ic / std_ic
        icir_ann = icir * math.sqrt(periods_per_year / label_horizon)
        t_stat = mean_ic / (std_ic / math.sqrt(n_eff))
        dof = max(1.0, n_eff - 1.0)
        p_value = float(2.0 * student_t.sf(abs(t_stat), df=dof))

    return ICResult(
        mean_ic=mean_ic, std_ic=std_ic, icir=float(icir),
        icir_annualized=float(icir_ann), t_stat=float(t_stat),
        p_value=float(p_value), n_dates=n, n_effective=n_eff,
        label_horizon=label_horizon, periods_per_year=periods_per_year,
        ic_series=ics, ic_dates=ic_dates,
    )


def _information_coefficient(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates: Sequence[Any],
    **kwargs: Any,
) -> float:
    """
    Mean cross-sectional IC as a bare float.

    `dates` is deliberately REQUIRED (positional): the previous two-argument
    pooled version silently reported market co-movement as skill, so any
    surviving old-style call site must fail loudly instead of returning a
    plausible-looking number.
    """
    return information_coefficient(y_pred, y_true, dates, **kwargs).mean_ic


def _pooled_rank_correlation(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Spearman correlation over a stacked panel, ignoring the date axis.

    NOT a measure of stock-picking skill — it is dominated by market-level
    co-movement.  Retained only for diagnostic logging when no date axis is
    available; never feeds a gate, a ranking or a selection decision.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    ok = np.isfinite(y_pred) & np.isfinite(y_true)
    if ok.sum() < 10 or np.ptp(y_pred[ok]) <= 0 or np.ptp(y_true[ok]) <= 0:
        return float("nan")
    return float(spearmanr(y_pred[ok], y_true[ok]).statistic)


def _ic_ir(
    ics: Iterable[float],
    *,
    label_horizon: int = 1,
    periods_per_year: Optional[int] = None,
) -> float:
    """
    IC information ratio = mean(IC) / std(IC) over a per-date IC series.

    With `periods_per_year` supplied the result is annualised by
    sqrt(periods_per_year / label_horizon); otherwise it is returned in raw
    per-period units.  Called from `information_coefficient`.
    """
    arr = np.asarray(list(ics), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return float("nan")
    ir = float(np.mean(arr)) / std
    if periods_per_year is not None:
        ir *= math.sqrt(periods_per_year / max(1, int(label_horizon)))
    return float(ir)


@dataclass
class LongShortResult:
    """Long-short quantile portfolio statistics built from a per-date return series."""

    sharpe: float             # annualised, overlap-adjusted
    sharpe_raw: float         # per-period mean/std, not annualised
    mean_return: float        # mean per-period long-short return
    volatility: float         # std (ddof=1) of the per-period long-short returns
    ls_spread: float          # == mean_return; the single-period spread, named honestly
    n_periods: int
    periods_per_year: int
    label_horizon: int
    returns: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def empty(cls, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
              label_horizon: int = 1) -> "LongShortResult":
        return cls(
            sharpe=np.nan, sharpe_raw=np.nan, mean_return=np.nan,
            volatility=np.nan, ls_spread=np.nan, n_periods=0,
            periods_per_year=periods_per_year, label_horizon=label_horizon,
        )


def long_short_returns(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates: Sequence[Any],
    top_pct: float = 0.2,
    bot_pct: float = 0.2,
    min_names_per_date: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-date long-short return series: mean(top `top_pct`) - mean(bottom `bot_pct`),
    where the quantiles are taken WITHIN each date's cross-section.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    out_dates: List[Any] = []
    out_rets: List[float] = []
    for date, idx in _group_slices(dates):
        p, a = y_pred[idx], y_true[idx]
        ok = np.isfinite(p) & np.isfinite(a)
        if ok.sum() < min_names_per_date:
            continue
        p, a = p[ok], a[ok]
        n = len(p)
        rank_idx = np.argsort(p)
        n_long = max(1, int(n * top_pct))
        n_short = max(1, int(n * bot_pct))
        out_dates.append(date)
        out_rets.append(float(a[rank_idx[-n_long:]].mean() - a[rank_idx[:n_short]].mean()))
    return np.asarray(out_dates), np.asarray(out_rets, dtype=float)


def long_short_sharpe(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates: Sequence[Any],
    *,
    top_pct: float = 0.2,
    bot_pct: float = 0.2,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    label_horizon: int = 1,
    min_names_per_date: int = 5,
) -> LongShortResult:
    """
    A real Sharpe ratio for the long-short quantile portfolio:

        sharpe = mean(ls_returns) / std(ls_returns) * sqrt(periods_per_year / label_horizon)

    The per-period long-short return series is built across dates, so there IS a
    volatility denominator.  The previous implementation returned a single-period
    spread with no denominator and no annualisation under the name "sharpe"; that
    quantity is still available, correctly named, as `ls_spread`.
    """
    label_horizon = max(1, int(label_horizon))
    _, rets = long_short_returns(
        y_pred, y_true, dates, top_pct, bot_pct, min_names_per_date
    )
    if len(rets) == 0:
        return LongShortResult.empty(periods_per_year, label_horizon)

    mean_r = float(np.mean(rets))
    if len(rets) < 2:
        return LongShortResult(
            sharpe=np.nan, sharpe_raw=np.nan, mean_return=mean_r, volatility=np.nan,
            ls_spread=mean_r, n_periods=len(rets), periods_per_year=periods_per_year,
            label_horizon=label_horizon, returns=rets,
        )
    vol = float(np.std(rets, ddof=1))
    if vol < 1e-15:
        sharpe_raw = sharpe = np.nan
    else:
        sharpe_raw = mean_r / vol
        sharpe = sharpe_raw * math.sqrt(periods_per_year / label_horizon)
    return LongShortResult(
        sharpe=float(sharpe), sharpe_raw=float(sharpe_raw), mean_return=mean_r,
        volatility=vol, ls_spread=mean_r, n_periods=len(rets),
        periods_per_year=periods_per_year, label_horizon=label_horizon, returns=rets,
    )


# ---------------------------------------------------------------------------
# Multiple-testing adjustment (Bailey & López de Prado, 2014)
# ---------------------------------------------------------------------------

@dataclass
class MultipleTestingAssessment:
    """
    Multiple-comparison adjustment for the winner of a model competition.

    Picking the best of N candidates inflates the winner's apparent skill even
    when every candidate is pure noise, so the raw winning statistic cannot be
    read as evidence on its own.

    Attributes
    ----------
    n_trials : N, the number of candidate models evaluated (the multiple-testing
        burden — recorded so it is auditable after the fact).
    observed_sr : the winner's per-period Sharpe-equivalent statistic.  For a
        model competition this is the raw validation ICIR (the Sharpe ratio of
        the per-date IC series).
    n_obs_effective : overlap-deflated number of independent observations.
    expected_max_sr_null : E[max SR] over N trials under the null of zero skill,
        via Bailey & López de Prado's Gumbel approximation.
    expected_max_sr_sqrt2logn : the simpler sqrt(2 ln N) benchmark, expressed in
        the same Sharpe units (sqrt(2 ln N) * sd(SR under null)).
    deflated_sharpe_ratio : P(true SR > 0 | selection over N trials).
    significant : deflated_sharpe_ratio >= threshold AND observed_sr exceeds
        expected_max_sr_null.  If False, the winner is indistinguishable from
        the best of N coin flips.
    """

    n_trials: int
    observed_sr: float
    n_obs_effective: float
    sr_std_under_null: float
    expected_max_sr_null: float
    expected_max_sr_sqrt2logn: float
    deflated_sharpe_ratio: float
    threshold: float
    significant: bool
    verdict: str


def expected_max_sharpe_under_null(
    n_trials: int,
    sr_std_under_null: float,
) -> float:
    """
    E[max SR] across `n_trials` independent trials of a zero-skill strategy
    (Bailey & López de Prado 2014, eq. for the expected maximum of N Gumbel-
    distributed Sharpe estimates):

        E[max] = sd(SR) * [ (1 - γ) Z⁻¹(1 - 1/N) + γ Z⁻¹(1 - 1/(N e)) ]

    with γ the Euler-Mascheroni constant.
    """
    n = max(1, int(n_trials))
    if n == 1 or not np.isfinite(sr_std_under_null) or sr_std_under_null <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(sr_std_under_null * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def expected_max_sharpe_sqrt2logn(n_trials: int, sr_std_under_null: float) -> float:
    """The coarser E[max] ≈ sqrt(2 ln N) benchmark, in Sharpe units."""
    n = max(1, int(n_trials))
    if n == 1:
        return 0.0
    return float(math.sqrt(2.0 * math.log(n)) * sr_std_under_null)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: float,
    n_trials: int,
    *,
    sr_variance: Optional[float] = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    threshold: float = 0.95,
) -> MultipleTestingAssessment:
    """
    Deflated Sharpe Ratio — Bailey & López de Prado (2014).

        DSR = Φ( (SR̂ - SR*) √(n-1) / √(1 - γ₃ SR̂ + (γ₄-1)/4 · SR̂²) )

    where SR* = E[max SR] under the null across `n_trials` and SR̂ is the
    winner's per-period Sharpe.  DSR is the probability that the winner's true
    Sharpe is positive AFTER accounting for having selected the max of N.

    `sr_variance` is the variance of the candidates' Sharpe estimates; when not
    supplied it defaults to the null variance 1/n_obs.  All Sharpes must be in
    per-observation (non-annualised) units.
    """
    n_trials = max(1, int(n_trials))
    n_obs = float(n_obs)
    if not np.isfinite(observed_sr) or n_obs < 2:
        return MultipleTestingAssessment(
            n_trials=n_trials, observed_sr=float(observed_sr), n_obs_effective=n_obs,
            sr_std_under_null=np.nan, expected_max_sr_null=np.nan,
            expected_max_sr_sqrt2logn=np.nan, deflated_sharpe_ratio=np.nan,
            threshold=threshold, significant=False,
            verdict="NOT ASSESSABLE: too few effective observations to deflate.",
        )

    if sr_variance is None or not np.isfinite(sr_variance) or sr_variance <= 0:
        sr_variance = 1.0 / n_obs  # variance of SR̂ under the null of zero skill
    sr_std = math.sqrt(sr_variance)

    sr_star = expected_max_sharpe_under_null(n_trials, sr_std)
    sr_star_simple = expected_max_sharpe_sqrt2logn(n_trials, sr_std)

    denom_sq = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2
    if denom_sq <= 0:
        dsr = float("nan")
    else:
        z = (observed_sr - sr_star) * math.sqrt(n_obs - 1.0) / math.sqrt(denom_sq)
        dsr = float(norm.cdf(z))

    significant = bool(
        np.isfinite(dsr) and dsr >= threshold and observed_sr > sr_star
    )
    if significant:
        verdict = (
            f"SURVIVES multiple-testing adjustment: winner SR={observed_sr:.3f} > "
            f"E[max|null, N={n_trials}]={sr_star:.3f}, DSR={dsr:.3f} >= {threshold:.2f}."
        )
    else:
        verdict = (
            f"DOES NOT SURVIVE multiple-testing adjustment: winner SR={observed_sr:.3f} vs "
            f"E[max|null, N={n_trials}]={sr_star:.3f} "
            f"(sqrt(2·lnN) benchmark {sr_star_simple:.3f}), DSR={dsr:.3f} < {threshold:.2f}. "
            f"Treat as indistinguishable from the best of {n_trials} coin flips."
        )

    return MultipleTestingAssessment(
        n_trials=n_trials, observed_sr=float(observed_sr), n_obs_effective=n_obs,
        sr_std_under_null=float(sr_std), expected_max_sr_null=float(sr_star),
        expected_max_sr_sqrt2logn=float(sr_star_simple),
        deflated_sharpe_ratio=dsr, threshold=threshold,
        significant=significant, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Causal (embargoed) internal splitting
# ---------------------------------------------------------------------------

def _embargoed_time_series_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo: int = 1,
    dates: Optional[Sequence[Any]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Chronological expanding-window splits with an embargo — never KFold.

    A plain `KFold(shuffle=False)` (which is what `cv=5` resolves to) scores
    fold 0 with a model trained on folds 1..4, i.e. FUTURE rows select the
    hyper-parameter.  Here fold k's training set is strictly earlier than its
    validation set, minus an `embargo` of the last `embargo` observation dates
    before the validation block, so a training label whose forward window
    overlaps the validation block cannot leak in.

    When `dates` is supplied the split is on unique dates, so a date's
    cross-section is never cut in half; otherwise it is on row order (rows are
    assumed chronologically ordered) and the embargo counts rows.

    If `app.research.validation` exposes a purged/embargoed splitter it is used
    instead of this minimal implementation.
    """
    embargo = max(0, int(embargo))
    n_splits = max(1, int(n_splits))

    if _HAS_RESEARCH_VALIDATION:
        external = _try_research_validation_splits(n_samples, n_splits, embargo, dates)
        if external is not None:
            return external

    if dates is not None:
        groups = _group_slices(dates)
        units: List[np.ndarray] = [idx for _, idx in groups]
    else:
        units = [np.array([i]) for i in range(n_samples)]

    n_units = len(units)
    # Need at least one train unit, `embargo` gap units, and one val unit per fold.
    max_splits = max(1, min(n_splits, n_units - embargo - 1))
    fold_size = max(1, (n_units - embargo) // (max_splits + 1))

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(max_splits):
        val_start = (k + 1) * fold_size + embargo
        val_end = val_start + fold_size if k < max_splits - 1 else n_units
        if val_start >= n_units or val_end <= val_start:
            continue
        train_end = val_start - embargo
        if train_end <= 0:
            continue
        tr = np.concatenate(units[:train_end]) if train_end else np.array([], dtype=int)
        va = np.concatenate(units[val_start:val_end])
        if len(tr) == 0 or len(va) == 0:
            continue
        splits.append((np.sort(tr), np.sort(va)))

    if not splits:
        # Degenerate sample: fall back to one chronological holdout, still embargoed.
        cut = max(1, int(n_units * 0.8))
        train_end = max(1, cut - embargo)
        if train_end < n_units and cut < n_units:
            tr = np.concatenate(units[:train_end])
            va = np.concatenate(units[cut:])
            if len(tr) and len(va):
                splits.append((np.sort(tr), np.sort(va)))
    return splits


def _try_research_validation_splits(
    n_samples: int,
    n_splits: int,
    embargo: int,
    dates: Optional[Sequence[Any]],
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Best-effort adapter to `app.research.validation` (owned by another engineer).

    Any signature mismatch degrades to the inline splitter rather than raising.
    """
    candidates = (
        "PurgedTimeSeriesSplit", "PurgedEmbargoedSplit", "PurgedKFold",
        "EmbargoedTimeSeriesSplit",
    )
    for name in candidates:
        cls = getattr(_research_validation, name, None)
        if cls is None:
            continue
        for kwargs in (
            {"n_splits": n_splits, "embargo": embargo},
            {"n_splits": n_splits, "embargo_periods": embargo},
            {"n_splits": n_splits},
        ):
            try:
                splitter = cls(**kwargs)
                X_dummy = np.zeros((n_samples, 1))
                splits = [
                    (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
                    for tr, va in splitter.split(X_dummy, groups=dates)
                ]
                splits = [(tr, va) for tr, va in splits if len(tr) and len(va)]
                if splits:
                    logger.info(
                        "Using app.research.validation.%s for internal CV splits.", name
                    )
                    return splits
            except Exception:  # pragma: no cover - signature probing
                continue
    logger.debug(
        "app.research.validation present but no compatible splitter found; "
        "using inline embargoed splitter."
    )
    return None


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
        dates_train: Optional[Sequence[Any]] = None,
        dates_val: Optional[Sequence[Any]] = None,
    ) -> "AlphaModelBase":
        """
        Train model, optionally using validation set for early stopping.

        `dates_train` / `dates_val` are the panel's date axis.  They are optional
        for backward compatibility, but without them no CROSS-SECTIONAL IC can be
        computed and any logged IC is a pooled diagnostic only (see module
        docstring, item 4).  Subclasses accepting them is detected via
        `inspect.signature`, so older subclasses keep working unchanged.
        """
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

    def _record_val_ic(
        self,
        preds: np.ndarray,
        y_val: np.ndarray,
        dates_val: Optional[Sequence[Any]],
        label_horizon: int = 1,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> Optional[ICResult]:
        """
        Store the validation IC.  With a date axis this is the mean CROSS-SECTIONAL
        IC; without one, `_val_ic` is left NaN (a pooled correlation is logged as a
        diagnostic but must never be mistaken for skill).
        """
        if dates_val is None:
            pooled = _pooled_rank_correlation(preds, y_val)
            self._val_ic = float("nan")
            self._val_ic_result = None
            self._val_ic_pooled = pooled
            logger.warning(
                "%s — no date axis passed to fit(); cross-sectional val IC is not "
                "computable. Pooled rank corr=%.4f is a DIAGNOSTIC ONLY (it measures "
                "market co-movement, not stock-picking skill) and is not used for "
                "model selection.",
                self.model_name, pooled,
            )
            return None

        res = information_coefficient(
            preds, y_val, dates_val,
            label_horizon=label_horizon, periods_per_year=periods_per_year,
        )
        self._val_ic = res.mean_ic
        self._val_ic_result = res
        self._val_ic_pooled = None
        logger.info(
            "%s — val IC (cross-sectional): mean=%.4f sd=%.4f ICIR=%.3f "
            "t=%.2f over %d dates (n_eff=%.1f, label_horizon=%d)",
            self.model_name, res.mean_ic, res.std_ic, res.icir, res.t_stat,
            res.n_dates, res.n_effective, res.label_horizon,
        )
        return res


# ---------------------------------------------------------------------------
# Linear alpha model
# ---------------------------------------------------------------------------

class LinearAlphaModel(AlphaModelBase):
    """
    Ridge regression alpha model with cross-validated regularization strength.

    The ridge penalty is chosen entirely on X_train, with a CAUSAL, EMBARGOED
    chronological split (never KFold — see `_embargoed_time_series_splits`).
    X_val is only used to log OOS IC.  No look-ahead is introduced here — the
    caller must ensure X/y alignment.
    """

    model_name = "LinearAlpha"
    version = "1.1.0"

    def __init__(
        self,
        alphas: Optional[np.ndarray] = None,
        n_splits: int = 5,
        label_horizon: int = 1,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ):
        """
        Parameters
        ----------
        alphas : array of ridge regularization values to search.
            Defaults to log-spaced [0.001, 10 000].
        n_splits : number of chronological CV folds for the penalty search.
        label_horizon : forward-return horizon of y, in observation periods.
            Used both as the embargo size between train and validation folds and
            as the overlap deflation factor for reported significance.
        """
        self.alphas = alphas if alphas is not None else np.logspace(-3, 4, 20)
        self.n_splits = int(n_splits)
        self.label_horizon = max(1, int(label_horizon))
        self.periods_per_year = int(periods_per_year)
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[RidgeCV] = None
        self._feature_names: List[str] = []
        self._val_ic: Optional[float] = None
        self._val_ic_result: Optional[ICResult] = None
        self._val_ic_pooled: Optional[float] = None
        self._cv_splits: List[Tuple[np.ndarray, np.ndarray]] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        dates_train: Optional[Sequence[Any]] = None,
        dates_val: Optional[Sequence[Any]] = None,
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

        # CAUSAL penalty search.  `cv=5` resolved to KFold(shuffle=False), which
        # scores fold 0 with a model trained on rows 2..9 — i.e. future rows
        # selected the ridge penalty.  Splits are now strictly forward-looking
        # and embargoed by `label_horizon` to handle overlapping labels.
        dates_tr = np.asarray(dates_train)[mask_train] if dates_train is not None else None
        self._cv_splits = _embargoed_time_series_splits(
            n_samples=X_tr_scaled.shape[0],
            n_splits=self.n_splits,
            embargo=self.label_horizon,
            dates=dates_tr,
        )
        if len(self._cv_splits) >= 1:
            cv: Any = self._cv_splits
        else:  # pragma: no cover - only for pathologically small samples
            logger.warning(
                "%s — could not build any causal CV split (n=%d); falling back to "
                "the mid-range ridge penalty rather than an acausal KFold.",
                self.model_name, X_tr_scaled.shape[0],
            )
            cv = [(np.arange(X_tr_scaled.shape[0]), np.arange(X_tr_scaled.shape[0]))]

        self._model = RidgeCV(alphas=self.alphas, cv=cv)
        self._model.fit(X_tr_scaled, y_tr)

        # Evaluate on validation set
        mask_val = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
        if mask_val.sum() >= 10:
            X_v = self._scaler.transform(X_val[mask_val])
            preds = self._model.predict(X_v)
            d_val = np.asarray(dates_val)[mask_val] if dates_val is not None else None
            logger.info(
                "%s — chosen alpha: %.4f (%d causal CV folds, embargo=%d)",
                self.model_name, self._model.alpha_, len(self._cv_splits),
                self.label_horizon,
            )
            self._record_val_ic(
                preds, y_val[mask_val], d_val,
                label_horizon=self.label_horizon,
                periods_per_year=self.periods_per_year,
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

    This class performs NO internal split — the early-stopping validation set is
    supplied by the caller, so the caller owns the train/val embargo (see module
    docstring, item 2).  `label_horizon` is used here only to deflate the
    effective sample size of the reported validation IC statistics.
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

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        label_horizon: int = 1,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ):
        if not _HAS_LGB:
            raise ImportError("lightgbm is required for GBMAlphaModel.")
        merged = dict(self._DEFAULT_PARAMS)
        if params:
            merged.update(params)
        self._params = merged
        self.label_horizon = max(1, int(label_horizon))
        self.periods_per_year = int(periods_per_year)
        self._model: Optional[lgb.LGBMRegressor] = None
        self._feature_names: List[str] = []
        self._best_iteration: int = 0
        self._val_ic: Optional[float] = None
        self._val_ic_result: Optional[ICResult] = None
        self._val_ic_pooled: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        dates_train: Optional[Sequence[Any]] = None,
        dates_val: Optional[Sequence[Any]] = None,
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
            d_val = np.asarray(dates_val)[mask_val] if dates_val is not None else None
            logger.info(
                "%s — best_iter: %d", self.model_name, self._best_iteration
            )
            self._record_val_ic(
                preds, y_v, d_val,
                label_horizon=self.label_horizon,
                periods_per_year=self.periods_per_year,
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
    """
    Per-model metrics.  Every `*_ic` field is a mean CROSS-SECTIONAL IC (computed
    within each date, then averaged), never a pooled panel correlation.

    Fields prefixed `val_` are the ONLY ones permitted to influence selection.
    Fields prefixed `test_` are reported for ex-post transparency and are
    deliberately unused by `select_best_model`.
    """

    model_name: str
    # --- selection-eligible (validation) ---
    val_ic: float
    val_ic_std: float
    val_icir: float                 # raw mean(IC)/std(IC) across dates
    val_icir_annualized: float      # * sqrt(periods_per_year / label_horizon)
    val_ic_tstat: float             # on the OVERLAP-DEFLATED sample size
    val_ic_pvalue: float
    val_n_dates: int
    val_n_effective: float
    val_directional_accuracy: float
    val_ls_sharpe: float            # annualised, from a per-date LS return series
    val_ls_spread: float            # mean single-period LS spread (NOT a Sharpe)
    # --- in-sample diagnostic ---
    train_ic: float
    # --- report-only (must never enter a gate or a ranking) ---
    test_ic: float
    test_ic_std: float
    test_icir: float
    test_icir_annualized: float
    test_ic_tstat: float
    test_ic_pvalue: float
    test_n_dates: int
    test_n_effective: float
    test_directional_accuracy: float
    test_ls_sharpe: float           # a REAL Sharpe: mean/std * sqrt(ppy/horizon)
    test_ls_spread: float           # the single-period spread, honestly named
    test_turnover_proxy: float      # placeholder — real turnover needs a backtest
    label_horizon: int = 1
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR


@dataclass
class ModelComparisonResult:
    metrics: List[ModelMetrics]
    best_model: AlphaModelBase
    best_model_name: str
    selection_reason: str
    n_candidates: int                              # N — the multiple-testing burden
    selection_basis: str = "validation-only"
    multiple_testing: Optional[MultipleTestingAssessment] = None
    winner_survives_multiple_testing: bool = False
    passed_selection_gate: bool = False


class ModelCompetition:
    """
    Compare multiple alpha models and select the best on VALIDATION metrics.

    Selection criterion: highest validation ICIR, subject to
    val_IC > MIN_VAL_IC and validation directional accuracy > MIN_DIR_ACC.
    If no model clears the bar we fall back to the LinearAlphaModel (the most
    conservative candidate), or, failing that, to the highest val_IC.

    THE TEST SET IS NEVER READ BY THE SELECTION LOGIC.  Test metrics are computed
    and reported, but `select_best_model` only ever touches `val_*` fields — the
    previous version gated on `test_directional_accuracy` and ranked on
    `(val_ic + test_ic) / 2`, which turned the held-out set into a selection set.

    Because the winner is the max over N candidates, its apparent skill is
    upward-biased even under the null.  Every result therefore carries N and a
    Deflated-Sharpe-Ratio assessment of the winner (`multiple_testing`,
    `winner_survives_multiple_testing`).
    """

    MIN_VAL_IC = 0.02
    MIN_TEST_IC = MIN_VAL_IC  # backward-compatible alias; the bar is a VAL bar
    MIN_DIR_ACC = 0.52
    DSR_THRESHOLD = 0.95

    @staticmethod
    def _directional_accuracy(
        preds: np.ndarray, y: np.ndarray, dates: Optional[Sequence[Any]]
    ) -> float:
        """
        Fraction of names whose predicted direction matched the realised one.

        Computed on values demeaned WITHIN each date, so that a model which only
        forecasts the market's direction (and picks no stocks) scores ~50%.
        """
        if len(y) == 0:
            return 0.0
        if dates is None:
            return float(np.mean(np.sign(preds) == np.sign(y)))
        hits: List[float] = []
        for _, idx in _group_slices(dates):
            p, a = preds[idx], y[idx]
            ok = np.isfinite(p) & np.isfinite(a)
            if ok.sum() < 2:
                continue
            p, a = p[ok] - p[ok].mean(), a[ok] - a[ok].mean()
            hits.extend((np.sign(p) == np.sign(a)).astype(float).tolist())
        return float(np.mean(hits)) if hits else 0.0

    @classmethod
    def _fit_and_evaluate(
        cls,
        model: AlphaModelBase,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dates_train: Sequence[Any],
        dates_val: Sequence[Any],
        dates_test: Sequence[Any],
        feature_names: Optional[List[str]] = None,
        label_horizon: int = 1,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> Tuple[AlphaModelBase, ModelMetrics]:
        # Fit.  Pass the date axis only to models whose fit() accepts it, so
        # older AlphaModelBase subclasses keep working unchanged.
        fit_kwargs: Dict[str, Any] = {}
        try:
            params = inspect.signature(model.fit).parameters
            if "dates_train" in params:
                fit_kwargs["dates_train"] = dates_train
            if "dates_val" in params:
                fit_kwargs["dates_val"] = dates_val
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            pass
        model.fit(X_train, y_train, X_val, y_val, feature_names, **fit_kwargs)

        ic_kw = dict(label_horizon=label_horizon, periods_per_year=periods_per_year)

        # Train IC (in-sample diagnostic only)
        mask_tr = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        train_preds = model.predict(X_train[mask_tr])
        train_ic = information_coefficient(
            train_preds, y_train[mask_tr], np.asarray(dates_train)[mask_tr], **ic_kw
        )

        # Validation metrics — the ONLY input to selection
        mask_v = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
        val_preds = model.predict(X_val[mask_v])
        y_v, d_v = y_val[mask_v], np.asarray(dates_val)[mask_v]
        val = information_coefficient(val_preds, y_v, d_v, **ic_kw)
        val_ls = long_short_sharpe(val_preds, y_v, d_v, **ic_kw)
        val_dir = cls._directional_accuracy(val_preds, y_v, d_v)

        # Test metrics — OOS period, reported but NEVER used for selection
        mask_te = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
        test_preds = model.predict(X_test[mask_te])
        y_te, d_te = y_test[mask_te], np.asarray(dates_test)[mask_te]
        test = information_coefficient(test_preds, y_te, d_te, **ic_kw)
        test_ls = long_short_sharpe(test_preds, y_te, d_te, **ic_kw)
        test_dir = cls._directional_accuracy(test_preds, y_te, d_te)

        metrics = ModelMetrics(
            model_name=model.model_name,
            val_ic=float(val.mean_ic),
            val_ic_std=float(val.std_ic),
            val_icir=float(val.icir),
            val_icir_annualized=float(val.icir_annualized),
            val_ic_tstat=float(val.t_stat),
            val_ic_pvalue=float(val.p_value),
            val_n_dates=int(val.n_dates),
            val_n_effective=float(val.n_effective),
            val_directional_accuracy=float(val_dir),
            val_ls_sharpe=float(val_ls.sharpe),
            val_ls_spread=float(val_ls.ls_spread),
            train_ic=float(train_ic.mean_ic),
            test_ic=float(test.mean_ic),
            test_ic_std=float(test.std_ic),
            test_icir=float(test.icir),
            test_icir_annualized=float(test.icir_annualized),
            test_ic_tstat=float(test.t_stat),
            test_ic_pvalue=float(test.p_value),
            test_n_dates=int(test.n_dates),
            test_n_effective=float(test.n_effective),
            test_directional_accuracy=float(test_dir),
            test_ls_sharpe=float(test_ls.sharpe),
            test_ls_spread=float(test_ls.ls_spread),
            test_turnover_proxy=np.nan,
            label_horizon=int(label_horizon),
            periods_per_year=int(periods_per_year),
        )
        logger.info(
            "ModelCompetition | %s: train_IC=%.3f | SELECTION val_IC=%.3f "
            "val_ICIR=%.3f val_t=%.2f (n_eff=%.1f) val_dir=%.1f%% | REPORT-ONLY "
            "test_IC=%.3f test_ICIR=%.3f test_LS_Sharpe=%.2f",
            model.model_name, metrics.train_ic, metrics.val_ic, metrics.val_icir,
            metrics.val_ic_tstat, metrics.val_n_effective,
            metrics.val_directional_accuracy * 100, metrics.test_ic,
            metrics.test_icir, metrics.test_ls_sharpe,
        )
        return model, metrics

    # Backward-compatible alias for the old (misleadingly named) entry point.
    _evaluate_on_test = _fit_and_evaluate

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
        dates_train: Optional[Sequence[Any]] = None,
        dates_val: Optional[Sequence[Any]] = None,
        dates_test: Optional[Sequence[Any]] = None,
        label_horizon: int = 1,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> ModelComparisonResult:
        """
        Fit and evaluate each model; return a ModelComparisonResult.

        The test set is only used for evaluation — never for model selection
        within this function.  Selection reads `val_*` metrics exclusively; test
        metrics are reported for ex-post transparency.  (This is now enforced by
        `select_best_model`, which is passed validation fields only.)

        `dates_train/val/test` are REQUIRED: without a date axis no
        cross-sectional IC exists, and a pooled correlation would score a
        zero-skill market-tracking model at ~0.7.

        `label_horizon` is the forward-return horizon of y in periods.  It sets
        the embargo used inside each model's internal CV and deflates the
        effective sample size for every reported t-statistic.
        """
        missing = [
            n for n, d in (("dates_train", dates_train), ("dates_val", dates_val),
                           ("dates_test", dates_test)) if d is None
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must be supplied: model skill is CROSS-SECTIONAL "
                "and cannot be measured without a date axis. Pooling a panel into one "
                "correlation measures market co-movement, not stock-picking skill."
            )

        fitted_models: List[Tuple[AlphaModelBase, ModelMetrics]] = []
        for m in models:
            try:
                fitted_m, mets = self._fit_and_evaluate(
                    m, X_train, y_train, X_val, y_val, X_test, y_test,
                    dates_train, dates_val, dates_test, feature_names,
                    label_horizon=label_horizon, periods_per_year=periods_per_year,
                )
                fitted_models.append((fitted_m, mets))
            except Exception as exc:
                logger.warning("Model %s failed: %s", m.model_name, exc)

        if not fitted_models:
            raise RuntimeError("All models failed to fit.")

        best_model, best_metrics, passed_gate = self._select(fitted_models)
        all_metrics = [m for _, m in fitted_models]
        n_candidates = len(models)  # every model TRIED counts toward the burden

        assessment = self.assess_multiple_testing(
            best_metrics, all_metrics, n_trials=n_candidates
        )

        reason = (
            f"Selected {best_metrics.model_name} on VALIDATION metrics only "
            f"(val_IC={best_metrics.val_ic:.3f}, val_ICIR={best_metrics.val_icir:.3f}, "
            f"val_t={best_metrics.val_ic_tstat:.2f} on n_eff={best_metrics.val_n_effective:.1f}, "
            f"val_dir_acc={best_metrics.val_directional_accuracy:.1%}); "
            f"gate {'PASSED' if passed_gate else 'NOT passed (fallback)'}; "
            f"test_IC={best_metrics.test_ic:.3f} reported but NOT used for selection. "
            f"N candidates tried = {n_candidates}. {assessment.verdict}"
        )
        if not assessment.significant:
            logger.warning("Multiple-testing: %s", assessment.verdict)

        return ModelComparisonResult(
            metrics=all_metrics,
            best_model=best_model,
            best_model_name=best_metrics.model_name,
            selection_reason=reason,
            n_candidates=n_candidates,
            selection_basis="validation-only",
            multiple_testing=assessment,
            winner_survives_multiple_testing=bool(assessment.significant),
            passed_selection_gate=passed_gate,
        )

    def assess_multiple_testing(
        self,
        best_metrics: ModelMetrics,
        all_metrics: Sequence[ModelMetrics],
        n_trials: int,
    ) -> MultipleTestingAssessment:
        """
        Deflate the winner's validation ICIR for the fact that it is the max of
        `n_trials` candidates.

        The ICIR is the Sharpe ratio of the per-date IC series, so it plugs
        directly into the Deflated Sharpe Ratio.  The variance of the candidates'
        ICIRs is used as V[SR] when there are enough candidates to estimate it;
        otherwise the null variance 1/n_obs is used.
        """
        sr_variance: Optional[float] = None
        irs = np.array(
            [m.val_icir for m in all_metrics if np.isfinite(m.val_icir)], dtype=float
        )
        if len(irs) >= 3:
            sr_variance = float(np.var(irs, ddof=1))
        return deflated_sharpe_ratio(
            best_metrics.val_icir,
            n_obs=best_metrics.val_n_effective,
            n_trials=n_trials,
            sr_variance=sr_variance,
            threshold=self.DSR_THRESHOLD,
        )

    def _select(
        self,
        fitted_models: List[Tuple[AlphaModelBase, ModelMetrics]],
    ) -> Tuple[AlphaModelBase, ModelMetrics, bool]:
        """Internal: returns (model, metrics, passed_gate)."""
        eligible = [
            (m, met) for m, met in fitted_models
            if (np.isfinite(met.val_ic) and met.val_ic > self.MIN_VAL_IC
                and met.val_directional_accuracy > self.MIN_DIR_ACC)
        ]
        if eligible:
            # Rank on validation ICIR (dispersion-aware); val_IC breaks ties and
            # covers the degenerate case where ICIR is undefined (a single date).
            best = max(
                eligible,
                key=lambda x: (
                    x[1].val_icir if np.isfinite(x[1].val_icir) else -np.inf,
                    x[1].val_ic,
                ),
            )
            logger.info(
                "Selected model on validation metrics: %s (val_ICIR=%.3f)",
                best[1].model_name, best[1].val_icir,
            )
            return best[0], best[1], True

        # Fallback: the most conservative candidate (LinearAlphaModel), else the
        # highest val_IC.  Still validation-only.
        linear = [
            (m, met) for m, met in fitted_models if isinstance(m, LinearAlphaModel)
        ]
        pool = linear or fitted_models
        fallback = max(
            pool,
            key=lambda x: x[1].val_ic if np.isfinite(x[1].val_ic) else -np.inf,
        )
        logger.warning(
            "No model cleared the validation bar; falling back to %s with val_IC=%.3f",
            fallback[1].model_name, fallback[1].val_ic,
        )
        return fallback[0], fallback[1], False

    def select_best_model(
        self,
        fitted_models: List[Tuple[AlphaModelBase, ModelMetrics]],
    ) -> Tuple[AlphaModelBase, ModelMetrics]:
        """
        Select the best model on VALIDATION metrics only.

        Criterion (in priority order):
        1. Must have val_IC > MIN_VAL_IC AND val_directional_accuracy > MIN_DIR_ACC.
           (Both gates read VALIDATION; the previous version gated on
           `test_directional_accuracy`.)
        2. Among those, the highest validation ICIR, val_IC breaking ties.
           (The previous version ranked on `(val_ic + test_ic) / 2`, i.e. it
           selected on the held-out set.)
        3. If none pass the bar, fall back to LinearAlphaModel (most conservative).

        No `test_*` field is read anywhere in this method.  A winner chosen here
        is still the max of N and must be read alongside
        `ModelComparisonResult.multiple_testing`.
        """
        model, metrics, _ = self._select(fitted_models)
        return model, metrics
