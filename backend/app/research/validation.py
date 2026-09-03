"""
Purged and embargoed time-series cross-validation.

WHY THIS MODULE EXISTS
----------------------
Standard k-fold cross-validation is invalid for financial data, and the usual
fix — "just split by time instead of shuffling" — is *still* invalid whenever
labels are built from forward returns.

Consider a 5-day forward-return label sampled daily. The label attached to
Monday is determined by prices through Friday. If the training set ends on
Monday and the test set begins on Tuesday, then Monday's training label was
computed from Tuesday-through-Friday prices — data that lives inside the test
period. The model is fitted on the answer sheet. Nothing in a naive
train/test boundary prevents this.

Two corrections are required (Lopez de Prado, *Advances in Financial Machine
Learning*, ch. 7):

  PURGING  Drop any training observation whose label horizon reaches into the
           test period. This removes the direct overlap described above.

  EMBARGO  Additionally drop training observations that fall shortly *after*
           the test period. Serial correlation in features means a bar
           immediately following the test window still carries information
           about it, even though its label does not overlap.

The cost is real: purging and embargoing discard data and will *lower* your
reported performance. That is the point. A backtest that improves when you
remove these safeguards was measuring leakage, not skill.

WHAT THIS DOES NOT FIX
----------------------
Overlapping labels remain non-IID even after purging. With a horizon of h
sampled every period, roughly (h-1)/h of adjacent labels overlap, so the
effective sample size is closer to n/h than to n. Any t-statistic computed on
such a sample is inflated by approximately sqrt(h). Use
`effective_sample_size()` below when reporting significance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    """
    One train/test split with full provenance.

    Attributes
    ----------
    train_idx, test_idx : np.ndarray
        Positional indices into the original sample.
    n_purged : int
        Training observations dropped because their label horizon overlapped
        the test window.
    n_embargoed : int
        Training observations dropped by the post-test embargo.
    train_start, train_end, test_start, test_end : pd.Timestamp
        Wall-clock boundaries, for logging and audit.
    """
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_purged: int
    n_embargoed: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"train {self.train_start.date()}→{self.train_end.date()} "
            f"({len(self.train_idx)} obs, {self.n_purged} purged, "
            f"{self.n_embargoed} embargoed) | "
            f"test {self.test_start.date()}→{self.test_end.date()} "
            f"({len(self.test_idx)} obs)"
        )


# ---------------------------------------------------------------------------
# Purged walk-forward
# ---------------------------------------------------------------------------

class PurgedWalkForward:
    """
    Walk-forward splitter with purging and an embargo.

    Parameters
    ----------
    n_splits : int
        Number of sequential test windows.
    label_horizon : int
        Number of periods into the future that a label at time t depends on.
        A 5-day forward return has label_horizon=5. Set to 1 for a label that
        depends only on the next observation. This is the single most
        important parameter: setting it to 0 disables purging entirely.
    embargo_frac : float
        Embargo length as a fraction of total sample length, applied after
        each test window. Lopez de Prado suggests ~0.01.
    expanding : bool
        True  → anchored/expanding training window (train always starts at 0).
        False → rolling window of fixed length `max_train_size`.
    max_train_size : int, optional
        Cap on training observations. Required when expanding=False.

    Notes
    -----
    Splits are yielded in chronological order. The final test window is the
    most recent data, so a caller reserving a true holdout should slice it off
    *before* calling this class.
    """

    def __init__(
        self,
        n_splits: int = 5,
        label_horizon: int = 1,
        embargo_frac: float = 0.01,
        expanding: bool = True,
        max_train_size: Optional[int] = None,
    ):
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if label_horizon < 0:
            raise ValueError(f"label_horizon must be >= 0, got {label_horizon}")
        if not 0.0 <= embargo_frac < 1.0:
            raise ValueError(f"embargo_frac must be in [0,1), got {embargo_frac}")
        if not expanding and max_train_size is None:
            raise ValueError("max_train_size is required when expanding=False")

        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo_frac = embargo_frac
        self.expanding = expanding
        self.max_train_size = max_train_size

    # -- core -------------------------------------------------------------

    def split(self, times: Sequence) -> Iterator[Split]:
        """
        Generate purged, embargoed train/test splits.

        Parameters
        ----------
        times : sequence of timestamps, length n_samples
            The observation time of each row, in ascending order. Must be
            sorted; unsorted input silently corrupts every split, so it is
            checked.

        Yields
        ------
        Split
        """
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(times))))
        n = len(idx)
        if n < self.n_splits + 1:
            raise ValueError(
                f"Need at least {self.n_splits + 1} observations for "
                f"{self.n_splits} splits, got {n}"
            )
        if not idx.is_monotonic_increasing:
            raise ValueError(
                "`times` must be sorted ascending. Unsorted input would make "
                "purging silently incorrect."
            )

        embargo_n = int(n * self.embargo_frac)
        fold_bounds = np.linspace(0, n, self.n_splits + 1).astype(int)

        for k in range(self.n_splits):
            test_start = fold_bounds[k]
            test_end = fold_bounds[k + 1]  # exclusive

            # The first fold has no history to train on.
            if test_start == 0:
                continue

            test_idx = np.arange(test_start, test_end)

            # ---- candidate training set: everything outside the test window
            left = np.arange(0, test_start)
            right = np.arange(test_end, n)

            # ---- PURGE: drop left-side rows whose label reaches into test.
            # A row at position i carries a label determined by data through
            # position i + label_horizon. It must be dropped if that reaches
            # the first test observation.
            if self.label_horizon > 0 and len(left) > 0:
                purge_from = max(0, test_start - self.label_horizon)
                kept_left = left[left < purge_from]
                n_purged = len(left) - len(kept_left)
                left = kept_left
            else:
                n_purged = 0

            # ---- EMBARGO: drop right-side rows immediately after the test
            # window. Their features overlap the test period through serial
            # correlation even though their labels do not.
            if embargo_n > 0 and len(right) > 0:
                embargo_until = test_end + embargo_n
                kept_right = right[right >= embargo_until]
                n_embargoed = len(right) - len(kept_right)
                right = kept_right
            else:
                n_embargoed = 0

            # Right-side data is future relative to the test window. Including
            # it is legitimate only for a non-causal research diagnostic, never
            # for simulating live trading, so walk-forward uses the past only.
            train_idx = left

            if self.max_train_size is not None and len(train_idx) > self.max_train_size:
                train_idx = train_idx[-self.max_train_size:]

            if len(train_idx) == 0:
                logger.warning(
                    "Split %d has no training observations after purging "
                    "(test_start=%d, label_horizon=%d). Skipping.",
                    k, test_start, self.label_horizon,
                )
                continue

            yield Split(
                train_idx=train_idx,
                test_idx=test_idx,
                n_purged=n_purged,
                n_embargoed=n_embargoed,
                train_start=idx[train_idx[0]],
                train_end=idx[train_idx[-1]],
                test_start=idx[test_idx[0]],
                test_end=idx[test_idx[-1]],
            )

    def get_n_splits(self) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# Effective sample size under overlapping labels
# ---------------------------------------------------------------------------

def effective_sample_size(n_obs: int, label_horizon: int) -> float:
    """
    Effective (non-overlapping) sample size for overlapping forward labels.

    With a horizon of h sampled every period, consecutive labels share h-1
    periods of their return path. They are not independent draws. Treating
    them as independent inflates t-statistics by roughly sqrt(h).

    This returns the deflated count n/h, which should be used in place of n
    whenever computing a standard error or t-statistic on such a sample.

    Parameters
    ----------
    n_obs : int
    label_horizon : int

    Returns
    -------
    float
    """
    if label_horizon <= 1:
        return float(n_obs)
    return float(n_obs) / float(label_horizon)


def deflated_tstat(mean: float, std: float, n_obs: int, label_horizon: int) -> float:
    """
    t-statistic computed against the *effective* sample size.

    Using the nominal count on overlapping labels is the most common way an
    unremarkable signal is reported as significant.
    """
    n_eff = effective_sample_size(n_obs, label_horizon)
    if std <= 0 or n_eff <= 1:
        return float("nan")
    return float(mean / (std / np.sqrt(n_eff)))


# ---------------------------------------------------------------------------
# Leakage assertion helper
# ---------------------------------------------------------------------------

def assert_no_train_test_overlap(
    split: Split,
    times: Sequence,
    label_horizon: int,
) -> None:
    """
    Verify a Split really is leakage-free. Raises AssertionError if not.

    Checks:
      1. No positional index appears in both train and test.
      2. No training label horizon reaches the first test timestamp.

    Cheap enough to call on every split in a research run, and it converts a
    silent statistical error into a loud failure.
    """
    overlap = np.intersect1d(split.train_idx, split.test_idx)
    if len(overlap) > 0:
        raise AssertionError(
            f"train/test index overlap at positions {overlap[:10].tolist()}"
        )

    if label_horizon > 0 and len(split.train_idx) > 0:
        last_train = int(split.train_idx.max())
        first_test = int(split.test_idx.min())
        if last_train + label_horizon >= first_test:
            raise AssertionError(
                f"LABEL LEAKAGE: last training row {last_train} has a "
                f"{label_horizon}-period label reaching position "
                f"{last_train + label_horizon}, which is at or beyond the "
                f"first test row {first_test}. Purging did not work."
            )
