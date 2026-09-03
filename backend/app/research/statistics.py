"""
Multiple-testing and overfitting statistics.

THE PROBLEM THIS SOLVES
-----------------------
Run enough backtests and one of them will look excellent purely by chance.
Try 100 parameter settings on pure noise and the best will show an annualized
Sharpe near 0.5-1.0 with no edge whatsoever. A raw Sharpe ratio therefore
answers the wrong question. It tells you how the winner performed; it does
not tell you whether the winner is distinguishable from the luckiest of many
coin flips.

Three tools here address that:

  deflated_sharpe_ratio  Adjusts an observed Sharpe for (a) the number of
                         trials that produced it and (b) the non-normality of
                         the return distribution, returning the probability
                         that true skill is positive.

  probability_of_backtest_overfitting
                         Asks directly: when a configuration is selected for
                         being best in-sample, how often does it land below
                         median out-of-sample? A PBO above ~0.5 means the
                         selection procedure is worse than useless.

  stationary_bootstrap   Resamples a return series in blocks so autocorrelation
                         and volatility clustering survive resampling. IID
                         resampling destroys both and understates drawdown risk
                         for any trend-following or momentum strategy.

REFERENCES
----------
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
    Overfitting", Journal of Computational Finance.
Politis & Romano (1994), "The Stationary Bootstrap", JASA.
Benjamini & Hochberg (1995), "Controlling the False Discovery Rate", JRSS-B.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

@dataclass
class DeflatedSharpeResult:
    observed_sharpe: float          # per-period, as supplied
    expected_max_sharpe_null: float # per-period, E[max SR] under zero skill
    deflated_sharpe_ratio: float    # P(true SR > 0), in [0, 1]
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float
    is_significant: bool            # DSR > threshold

    def summary(self) -> str:
        verdict = "SIGNIFICANT" if self.is_significant else "NOT SIGNIFICANT"
        return (
            f"DSR={self.deflated_sharpe_ratio:.4f} [{verdict}] | "
            f"observed SR={self.observed_sharpe:.4f} vs "
            f"E[max|null]={self.expected_max_sharpe_null:.4f} over "
            f"N={self.n_trials} trials, T={self.n_observations} obs"
        )


def expected_max_sharpe_under_null(
    n_trials: int,
    sharpe_variance: float,
) -> float:
    """
    Expected maximum Sharpe ratio across `n_trials` strategies that all have
    ZERO true skill.

    This is the bar a candidate must clear to be interesting. It grows roughly
    with sqrt(2*ln(N)): searching harder raises the bar, which is exactly why
    an unadjusted Sharpe from a large parameter sweep means little.

    Parameters
    ----------
    n_trials : int
        Number of independent configurations tried. Be honest here — it
        includes every variant you discarded along the way, not just the ones
        you wrote down.
    sharpe_variance : float
        Variance of the Sharpe estimates across those trials.

    Returns
    -------
    float : expected maximum Sharpe under the null, in per-period units.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if sharpe_variance < 0:
        raise ValueError(f"sharpe_variance must be >= 0, got {sharpe_variance}")
    if n_trials == 1:
        return 0.0

    sigma = np.sqrt(sharpe_variance)
    g = _EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sigma * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int,
    sharpe_variance: Optional[float] = None,
    benchmark_sharpe: Optional[float] = None,
    significance_threshold: float = 0.95,
) -> DeflatedSharpeResult:
    """
    Probability that a strategy's true Sharpe exceeds zero, after accounting
    for multiple testing and non-normal returns.

    Parameters
    ----------
    returns : sequence of per-period returns (NOT annualized).
    n_trials : int
        Number of configurations tried before selecting this one.
    sharpe_variance : float, optional
        Variance of Sharpe across trials. If omitted, a conservative default
        of 1/(T-1) is used — the sampling variance of a single Sharpe estimate
        under the null.
    benchmark_sharpe : float, optional
        Explicit hurdle. Defaults to E[max SR] under the null.
    significance_threshold : float
        DSR above this is reported as significant. 0.95 is the usual choice.

    Returns
    -------
    DeflatedSharpeResult

    Notes
    -----
    Returns must be per-period. Passing an annualized Sharpe here inflates the
    result badly, because the deflation term is scaled by sqrt(T-1).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3:
        raise ValueError(f"Need at least 3 finite returns, got {T}")

    sd = r.std(ddof=1)
    if sd <= 0:
        raise ValueError("Return series has zero variance; Sharpe is undefined.")

    sr = float(r.mean() / sd)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # non-excess: normal == 3

    if sharpe_variance is None:
        sharpe_variance = 1.0 / (T - 1)

    sr0 = (
        benchmark_sharpe
        if benchmark_sharpe is not None
        else expected_max_sharpe_under_null(n_trials, sharpe_variance)
    )

    # Denominator: the standard error of the Sharpe estimator, corrected for
    # skewness and kurtosis. Negative skew and fat tails (both typical of
    # trading returns) inflate it, correctly making significance harder.
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denom_sq <= 0:
        logger.warning(
            "Non-normality correction is non-positive (%.4f); the Sharpe "
            "estimator variance is not well defined for this sample. "
            "Reporting DSR=0.", denom_sq,
        )
        dsr = 0.0
    else:
        z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom_sq)
        dsr = float(stats.norm.cdf(z))

    return DeflatedSharpeResult(
        observed_sharpe=sr,
        expected_max_sharpe_null=float(sr0),
        deflated_sharpe_ratio=dsr,
        n_trials=n_trials,
        n_observations=T,
        skewness=skew,
        kurtosis=kurt,
        is_significant=dsr > significance_threshold,
    )


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------

@dataclass
class PBOResult:
    pbo: float                       # P(IS-best config lands below OOS median)
    n_combinations: int
    n_configs: int
    logits: np.ndarray
    oos_ranks: np.ndarray            # relative OOS rank of the IS-best, in (0,1)
    median_oos_rank: float
    is_overfit: bool                 # pbo > 0.5

    def summary(self) -> str:
        verdict = "OVERFIT" if self.is_overfit else "acceptable"
        return (
            f"PBO={self.pbo:.3f} [{verdict}] over {self.n_combinations} "
            f"splits of {self.n_configs} configs | median OOS rank of the "
            f"IS-winner = {self.median_oos_rank:.3f}"
        )


def probability_of_backtest_overfitting(
    performance_matrix: np.ndarray,
    n_partitions: int = 10,
    max_combinations: int = 5000,
    random_seed: int = 42,
) -> PBOResult:
    """
    Combinatorially Symmetric Cross-Validation estimate of overfitting risk.

    The question it answers is not "did my best configuration do well?" but
    "does picking the in-sample best actually help out-of-sample?"

    Procedure: partition the timeline into S blocks; for every way of choosing
    S/2 blocks as in-sample (the rest out-of-sample), find the configuration
    with the best in-sample Sharpe and record where it ranks out-of-sample.
    PBO is the fraction of splits where that in-sample winner lands below the
    out-of-sample median.

    Interpretation:
        PBO ~ 0.0   selection generalizes well
        PBO ~ 0.5   selection is no better than picking at random
        PBO > 0.5   selection is actively harmful — you are choosing the
                    configuration most fitted to noise

    Parameters
    ----------
    performance_matrix : ndarray, shape (T, N)
        Per-period returns. Column n is configuration n's return series.
        Use returns, not equity curves.
    n_partitions : int
        S. Must be even. C(S, S/2) combinations are evaluated.
    max_combinations : int
        Cap on combinations; a random subset is used beyond this.
    random_seed : int

    Returns
    -------
    PBOResult
    """
    M = np.asarray(performance_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"performance_matrix must be 2-D (T,N), got {M.shape}")
    T, N = M.shape
    if N < 2:
        raise ValueError(f"Need at least 2 configurations to compare, got {N}")
    if n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be even, got {n_partitions}")
    if T < n_partitions * 2:
        raise ValueError(
            f"Need at least {n_partitions * 2} observations for "
            f"{n_partitions} partitions, got {T}"
        )

    rng = np.random.default_rng(random_seed)

    # Contiguous, time-ordered partitions preserve local serial structure.
    bounds = np.linspace(0, T, n_partitions + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_partitions)]

    half = n_partitions // 2
    all_combos = list(combinations(range(n_partitions), half))
    if len(all_combos) > max_combinations:
        pick = rng.choice(len(all_combos), size=max_combinations, replace=False)
        all_combos = [all_combos[i] for i in pick]

    logits, ranks = [], []

    for is_blocks in all_combos:
        oos_blocks = [b for b in range(n_partitions) if b not in is_blocks]
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])

        is_perf = _sharpe_by_column(M[is_idx, :])
        oos_perf = _sharpe_by_column(M[oos_idx, :])

        if not np.isfinite(is_perf).any() or not np.isfinite(oos_perf).any():
            continue

        best = int(np.nanargmax(is_perf))

        # Relative rank of the IS winner within the OOS distribution.
        finite = np.isfinite(oos_perf)
        if finite.sum() < 2 or not finite[best]:
            continue
        rank = float((oos_perf[finite] <= oos_perf[best]).sum())
        omega = rank / (finite.sum() + 1.0)
        omega = min(max(omega, 1e-6), 1 - 1e-6)  # keep the logit finite

        ranks.append(omega)
        logits.append(np.log(omega / (1.0 - omega)))

    if not logits:
        raise RuntimeError(
            "No valid CSCV combinations produced a finite result. The "
            "performance matrix is likely degenerate (zero-variance columns)."
        )

    logits_arr = np.asarray(logits)
    ranks_arr = np.asarray(ranks)
    pbo = float((logits_arr < 0).mean())

    return PBOResult(
        pbo=pbo,
        n_combinations=len(logits),
        n_configs=N,
        logits=logits_arr,
        oos_ranks=ranks_arr,
        median_oos_rank=float(np.median(ranks_arr)),
        is_overfit=pbo > 0.5,
    )


def _sharpe_by_column(block: np.ndarray) -> np.ndarray:
    """Per-period Sharpe for each column; NaN where variance is zero."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mu / sd, np.nan)
    return out


# ---------------------------------------------------------------------------
# Stationary block bootstrap
# ---------------------------------------------------------------------------

def stationary_bootstrap(
    returns: Sequence[float],
    n_simulations: int = 1000,
    horizon: Optional[int] = None,
    mean_block_length: Optional[float] = None,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Politis-Romano stationary bootstrap.

    Blocks of geometrically distributed length are drawn with circular
    wrap-around, so autocorrelation and volatility clustering survive
    resampling. IID resampling destroys both, which makes simulated drawdowns
    far shallower than reality for any strategy with persistence — precisely
    the strategies whose tail risk you most need to measure.

    Parameters
    ----------
    returns : sequence of per-period returns.
    n_simulations : int
    horizon : int, optional
        Length of each simulated path. Defaults to len(returns).
    mean_block_length : float, optional
        Expected block length. Defaults to T**(1/3), the standard rule of
        thumb; larger values preserve more dependence.
    random_seed : int

    Returns
    -------
    ndarray, shape (n_simulations, horizon) of resampled returns.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 10:
        raise ValueError(f"Need at least 10 finite returns, got {T}")

    H = horizon if horizon is not None else T
    L = mean_block_length if mean_block_length is not None else max(2.0, T ** (1 / 3))
    p = 1.0 / L  # probability of starting a new block at each step

    rng = np.random.default_rng(random_seed)
    out = np.empty((n_simulations, H), dtype=float)

    # Vectorized across simulations: at each step either continue the current
    # block (advance the pointer, wrapping) or jump to a fresh random start.
    pos = rng.integers(0, T, size=n_simulations)
    for t in range(H):
        out[:, t] = r[pos]
        new_block = rng.random(n_simulations) < p
        pos = np.where(new_block, rng.integers(0, T, size=n_simulations), (pos + 1) % T)

    return out


# ---------------------------------------------------------------------------
# Multiple-testing corrections
# ---------------------------------------------------------------------------

def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg false discovery rate control.

    Controls the expected *proportion* of false positives among rejections.
    Appropriate for screening many candidate signals, where tolerating some
    false discoveries is preferable to missing every real one.

    Returns
    -------
    (rejected, q_values)
        rejected : bool array, True where the null is rejected at FDR=alpha
        q_values : BH-adjusted p-values
    """
    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    out_q = np.empty(n)
    out_q[order] = q
    return out_q <= alpha, out_q


def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni family-wise error rate control.

    Stricter than Benjamini-Hochberg: controls the probability of *any* false
    positive. Use when a single false discovery would be costly — for example
    before committing capital to a newly "discovered" strategy.

    Returns
    -------
    (rejected, adjusted_p_values)
    """
    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * (n - np.arange(n))
    adj = np.maximum.accumulate(adj)
    adj = np.minimum(adj, 1.0)

    out = np.empty(n)
    out[order] = adj
    return out <= alpha, out
