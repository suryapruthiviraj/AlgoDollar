"""
Baseline cross-sectional signals: simple, interpretable, and unassumed.

Each function takes an adjusted-price panel (dates x symbols) and returns a
signal panel on the same index, where a HIGHER value means MORE ATTRACTIVE.
Sign conventions are made explicit in each docstring because the commonest
silent error in factor code is a flipped sign that turns a losing signal into a
winning one and nobody notices.

WHAT THESE ARE FOR
------------------
They are the null hypothesis, not the product. Every one of them is a textbook
construction with textbook parameters, chosen BEFORE looking at any result on
this dataset and never adjusted afterwards. If a machine-learning model cannot
beat these after costs, the model is not adding anything and is rejected.

WHY THE PARAMETERS ARE NOT TUNED
--------------------------------
12-1 momentum is 12 months skipping the most recent one because that is the
standard definition, not because 11 or 13 tested better here. Every window
below is similarly conventional. Searching them would manufacture exactly the
overfitting the validation layer exists to detect — and would do so in the one
place the validation layer cannot see.

EVERY SIGNAL USES ONLY PAST DATA
--------------------------------
All windows are trailing and every rolling call is left-closed at the current
bar. The backtester adds a further two-bar execution lag on top of this, so a
signal computed from the close of ``t`` cannot earn a return before ``t+1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: One trading month / year, in sessions. Conventional, not fitted.
MONTH = 21
YEAR = 252


def _require_panel(prices: pd.DataFrame) -> None:
    if prices.empty:
        raise ValueError("price panel is empty")
    if not prices.index.is_monotonic_increasing:
        raise ValueError(
            "price panel is not chronologically ordered; every trailing window "
            "below would silently mix future data into the past"
        )


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Classic 12-1 momentum: the 12-month return, skipping the last month.

    HIGHER = more attractive (buy past winners).

    The one-month skip is not decoration. Short-horizon returns reverse, so
    including the most recent month mixes a reversal effect into a momentum
    signal and blunts both.
    """
    _require_panel(prices)
    return np.log(prices.shift(MONTH) / prices.shift(YEAR))


def short_term_reversal(prices: pd.DataFrame, window: int = MONTH) -> pd.DataFrame:
    """
    One-month reversal.

    HIGHER = more attractive, so the sign is NEGATED: a stock that fell over
    the past month scores high. This is the opposite convention to momentum and
    is the single easiest sign to get wrong.
    """
    _require_panel(prices)
    return -np.log(prices / prices.shift(window))


def trend_following(
    prices: pd.DataFrame, fast: int = 50, slow: int = 200
) -> pd.DataFrame:
    """
    Distance between a fast and slow moving average, scaled by price.

    HIGHER = more attractive (price above its long average).
    50/200 is the conventional pair.
    """
    _require_panel(prices)
    f = prices.rolling(fast, min_periods=fast).mean()
    s = prices.rolling(slow, min_periods=slow).mean()
    return (f - s) / s


def low_volatility(prices: pd.DataFrame, window: int = 3 * MONTH) -> pd.DataFrame:
    """
    Trailing realised volatility, NEGATED.

    HIGHER = more attractive, i.e. LOW volatility scores high. This is the
    low-volatility anomaly; the negation is what expresses it.
    """
    _require_panel(prices)
    return -prices.pct_change().rolling(window, min_periods=window // 2).std()


def breakout(prices: pd.DataFrame, window: int = 6 * MONTH) -> pd.DataFrame:
    """
    Position within the trailing high-low range (0 = at the low, 1 = at the high).

    HIGHER = more attractive. The window is shifted by one bar so the current
    close is compared against a range that EXCLUDES it — otherwise every new
    high scores exactly 1.0 by construction, which is a tautology rather than a
    signal.
    """
    _require_panel(prices)
    hi = prices.shift(1).rolling(window, min_periods=window // 2).max()
    lo = prices.shift(1).rolling(window, min_periods=window // 2).min()
    rng = (hi - lo).replace(0.0, np.nan)
    return (prices - lo) / rng


def volatility_scaled_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    12-1 momentum divided by trailing volatility.

    HIGHER = more attractive. A risk-adjusted variant: it prefers a steady
    riser to a volatile one that arrived at the same place.
    """
    _require_panel(prices)
    mom = momentum_12_1(prices)
    vol = prices.pct_change().rolling(3 * MONTH, min_periods=MONTH).std()
    return mom / vol.replace(0.0, np.nan)


#: The baseline battery. Registered once, run identically, all reported —
#: including the ones that lose. Selecting the winner from this table and
#: reporting only that is the multiple-testing error the study explicitly
#: corrects for.
BASELINE_SIGNALS = {
    "momentum_12_1": momentum_12_1,
    "short_term_reversal": short_term_reversal,
    "trend_50_200": trend_following,
    "low_volatility": low_volatility,
    "breakout_126d": breakout,
    "vol_scaled_momentum": volatility_scaled_momentum,
}

#: Signals requiring datasets this project does NOT have. Named so the report
#: states them as untested rather than leaving their absence to be inferred.
UNTESTABLE_SIGNALS = {
    "value_earnings_yield": (
        "requires point-in-time fundamentals with publication dates; not available"
    ),
    "quality_roe_accruals": (
        "requires point-in-time fundamentals with publication dates; not available"
    ),
    "intraday_reversal": "requires intraday bars; not available",
}


def zscore_cross_section(signal: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """
    Standardise each DATE across symbols, not each symbol across time.

    Cross-sectional is the correct axis: the question is which stock is more
    attractive than the others TODAY. Standardising along the time axis would
    leak the whole sample's mean and standard deviation into every historical
    date — a look-ahead so common it has its own name.
    """
    mu = signal.mean(axis=1)
    sd = signal.std(axis=1, ddof=1)
    z = signal.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)
    return z.clip(-clip, clip)
