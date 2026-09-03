"""Market regime detection using rule-based technical indicators."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """
    THE canonical market-regime vocabulary for the whole system.

    Every module that reasons about regime (allocator, regime_model,
    risk engine, reporting) must key its tables on these members — never on
    ad-hoc strings.  A table keyed on strings silently degrades when the
    vocabulary drifts; a table keyed on this enum fails loudly.
    """

    STRONG_BULL = "STRONG_BULL"
    WEAK_BULL = "WEAK_BULL"
    SIDEWAYS = "SIDEWAYS"
    WEAK_BEAR = "WEAK_BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    HIGH_VOL = "HIGH_VOL"
    PANIC = "PANIC"
    RECOVERY = "RECOVERY"


# Sleeve names used by every allocation table in the system.
SLEEVES: tuple[str, ...] = ("longterm", "swing", "intraday")


# ---------------------------------------------------------------------------
# Canonical regime → sleeve multiplier table
# ---------------------------------------------------------------------------
#
# THIS IS THE SINGLE SOURCE OF TRUTH for how a regime scales each sleeve's
# opportunity score.  app.portfolio.allocator imports it from here; there is
# no second copy anywhere.
#
# Semantics: a multiplier in [0, 1] applied to a sleeve's *opportunity score*.
# There is deliberately NO "cash" key: cash is the residual plug in the
# allocator (available_capital - deployed), so a "cash multiplier" would be a
# second, contradictory source of truth.  The previous version of this table
# had one, its rows summed to 3.0/2.4/1.9, and nothing read it — which made it
# very easy to mistake for a set of portfolio weights.
#
# CALIBRATION STATUS: UNCALIBRATED PRIORS.  Hand-chosen monotone priors
# (more risk in calmer/more bullish states, near-zero in PANIC).  They have NOT
# been estimated from data and must be validated before they drive live money.
# They are ordered consistently with regime_model._EQUITY_CAP.
REGIME_SLEEVE_MULTIPLIERS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.STRONG_BULL: {"longterm": 1.00, "swing": 1.00, "intraday": 1.00},
    MarketRegime.WEAK_BULL:   {"longterm": 0.85, "swing": 0.75, "intraday": 0.65},
    MarketRegime.RECOVERY:    {"longterm": 0.75, "swing": 0.55, "intraday": 0.45},
    MarketRegime.SIDEWAYS:    {"longterm": 0.60, "swing": 0.55, "intraday": 0.50},
    MarketRegime.HIGH_VOL:    {"longterm": 0.45, "swing": 0.30, "intraday": 0.20},
    MarketRegime.WEAK_BEAR:   {"longterm": 0.35, "swing": 0.20, "intraday": 0.15},
    MarketRegime.STRONG_BEAR: {"longterm": 0.20, "swing": 0.10, "intraday": 0.05},
    MarketRegime.PANIC:       {"longterm": 0.15, "swing": 0.00, "intraday": 0.00},
}

# Fail at import time if the table ever drifts from the enum.
_missing = set(MarketRegime) - set(REGIME_SLEEVE_MULTIPLIERS)
if _missing:
    raise RuntimeError(
        f"REGIME_SLEEVE_MULTIPLIERS is missing regimes: "
        f"{sorted(r.value for r in _missing)}"
    )
del _missing


def regime_sleeve_multipliers(regime: MarketRegime) -> dict[str, float]:
    """
    Return the sleeve multipliers for `regime`.

    Raises
    ------
    KeyError
        If `regime` is not a member of MarketRegime, or is a member with no
        row in the table.  This is deliberate: a silent default here is how
        regime detection stops affecting allocation without anyone noticing.
    """
    if not isinstance(regime, MarketRegime):
        raise KeyError(
            f"Unknown market regime {regime!r}. Expected a MarketRegime member; "
            f"valid regimes: {[r.value for r in MarketRegime]}"
        )
    return dict(REGIME_SLEEVE_MULTIPLIERS[regime])


class RegimeDetector:
    """
    Rule-based market regime classifier for Indian equities (Nifty).

    No hidden Markov models — purely technical indicators for robustness
    and interpretability.
    """

    # Thresholds
    VIX_PANIC = 30.0
    VIX_HIGH_VOL = 20.0
    RECENT_VOL_MULTIPLIER = 1.5    # recent vol > 1.5x historical → HIGH_VOL
    QUARTERLY_RETURN_BULL = 0.05   # >5% over 63 days → bullish return signal
    QUARTERLY_RETURN_BEAR = -0.05  # <-5% → bearish
    MONTHLY_RETURN_BULL = 0.03     # >3% over 21 days → bullish (works on short history)
    MONTHLY_RETURN_BEAR = -0.03    # <-3% → bearish
    PANIC_DRAWDOWN = -0.20         # ≤ -20% off the trailing peak, still falling → PANIC
    PANIC_CRASH_RETURN = -0.10     # ≤ -10% in 10 sessions → PANIC (price-only trigger)
    PANIC_STILL_FALLING_RETURN = -0.03  # confirmation that the drawdown is live
    MIN_TREND_BARS = 20            # shortest trailing window we will trust for a trend

    def detect_regime(
        self,
        nifty_prices: pd.Series,
        vix: Optional[pd.Series] = None,
        market_breadth: Optional[float] = None,  # fraction of stocks above 50-day MA
    ) -> MarketRegime:
        """
        Detect the current market regime.

        Parameters
        ----------
        nifty_prices    : pd.Series of daily close prices, indexed by date, recent last
        vix             : optional pd.Series of VIX values (same index)
        market_breadth  : optional float in [0, 1]; fraction of stocks above their 50-day MA

        Returns
        -------
        MarketRegime enum value
        """
        if len(nifty_prices) < 10:
            logger.warning(
                "Not enough price history for regime detection; defaulting to SIDEWAYS."
            )
            return MarketRegime.SIDEWAYS

        scores: dict[str, int] = {
            "bull": 0,
            "bear": 0,
            "vol": 0,
            "recovery": 0,
        }

        last_price = float(nifty_prices.iloc[-1])

        # 1. Long trend: price vs its long trailing mean.
        #    Degrades gracefully on short history (uses the longest trailing
        #    window available, min MIN_TREND_BARS) so that a 50-bar crash is not
        #    classified as SIDEWAYS just because 200 bars are unavailable.
        sma200, win_long = self._sma_adaptive(nifty_prices, 200)
        if sma200 is not None:
            if last_price > sma200:
                scores["bull"] += 2
            else:
                scores["bear"] += 2

        # 2. Golden / death cross: short trailing mean vs long trailing mean.
        #    Only scored when the two windows genuinely differ — otherwise the
        #    two means are identical and the comparison would always read bear.
        sma50, win_short = self._sma_adaptive(nifty_prices, 50)
        if sma50 is not None and sma200 is not None and win_short < win_long:
            if sma50 > sma200:
                scores["bull"] += 2
            else:
                scores["bear"] += 2

        # 3. 20-day vs historical realised vol
        recent_vol = self._realised_vol(nifty_prices, 20)
        historical_vol = self._realised_vol(nifty_prices, 252)
        if recent_vol is not None and historical_vol is not None and historical_vol > 0:
            vol_ratio = recent_vol / historical_vol
            if vol_ratio > self.RECENT_VOL_MULTIPLIER:
                scores["vol"] += 3

        # 4. Momentum return: 63-day (quarterly) when available, else 21-day.
        #    Without the short fallback, a series of <63 bars contributes no
        #    directional signal at all.
        if len(nifty_prices) >= 63:
            quarterly_return = self._trailing_return(nifty_prices, 63)
            if quarterly_return > self.QUARTERLY_RETURN_BULL:
                scores["bull"] += 1
            elif quarterly_return < self.QUARTERLY_RETURN_BEAR:
                scores["bear"] += 1
        elif len(nifty_prices) >= 21:
            monthly_return = self._trailing_return(nifty_prices, 21)
            if monthly_return > self.MONTHLY_RETURN_BULL:
                scores["bull"] += 1
            elif monthly_return < self.MONTHLY_RETURN_BEAR:
                scores["bear"] += 1

        # 5. VIX
        if vix is not None and not vix.empty:
            current_vix = float(vix.iloc[-1])
            if current_vix >= self.VIX_PANIC:
                scores["vol"] += 5   # will override to PANIC below
            elif current_vix >= self.VIX_HIGH_VOL:
                scores["vol"] += 2

        # 6. Market breadth
        if market_breadth is not None:
            if market_breadth >= 0.70:
                scores["bull"] += 1
            elif market_breadth <= 0.30:
                scores["bear"] += 1

        # 7. Recovery: recent bounce from low (price crossed back above 50-day after being below)
        if sma50 is not None:
            if len(nifty_prices) >= 10:
                prev_price = float(nifty_prices.iloc[-10])
                if prev_price < sma50 < last_price:
                    scores["recovery"] += 2

        # --- Decision logic -------------------------------------------
        regime = self._scores_to_regime(
            scores, vix, panic_from_price=self._is_price_panic(nifty_prices)
        )
        logger.debug(
            "Regime detection scores: %s → %s", scores, regime.value
        )
        return regime

    def detect(self, nifty_prices: pd.Series, **kwargs) -> MarketRegime:
        """Alias for :meth:`detect_regime` (kept for callers using `detect`)."""
        return self.detect_regime(nifty_prices, **kwargs)

    # ------------------------------------------------------------------ #
    #  Panic detection from price alone                                    #
    # ------------------------------------------------------------------ #

    def _is_price_panic(self, prices: pd.Series) -> bool:
        """
        PANIC detection that does NOT require a VIX series.

        Before this existed, PANIC was reachable only via `vix >= 30`, so any
        deployment that had no VIX feed could never see a panic regime — the
        most important state in the table was unreachable in practice.

        Two price-only triggers (both trailing-window only, no look-ahead):
          1. A crash: ≤ PANIC_CRASH_RETURN over the last 10 sessions.
          2. A deep drawdown from the trailing peak that is still unfolding —
             price below its short trailing mean AND still falling over the
             last 10 sessions.  The confirmation conditions stop a market that
             already fell 20% and has been recovering for months from being
             labelled PANIC forever.
        """
        if len(prices) >= 10:
            if self._trailing_return(prices, 10) <= self.PANIC_CRASH_RETURN:
                return True

        drawdown = self._drawdown_from_peak(prices, 252)
        if drawdown is not None and drawdown <= self.PANIC_DRAWDOWN:
            sma_short, _ = self._sma_adaptive(prices, 20, min_bars=10)
            still_falling = (
                self._trailing_return(prices, 10) <= self.PANIC_STILL_FALLING_RETURN
            )
            if sma_short is not None and still_falling:
                if float(prices.iloc[-1]) < sma_short:
                    return True
        return False

    # ------------------------------------------------------------------ #
    #  Score → Regime mapping                                              #
    # ------------------------------------------------------------------ #

    def _scores_to_regime(
        self,
        scores: dict[str, int],
        vix: Optional[pd.Series],
        panic_from_price: bool = False,
    ) -> MarketRegime:
        # PANIC overrides everything: VIX if we have it, price action if we don't.
        if vix is not None and not vix.empty:
            if float(vix.iloc[-1]) >= self.VIX_PANIC:
                return MarketRegime.PANIC
        if panic_from_price:
            return MarketRegime.PANIC

        # High vol
        if scores["vol"] >= 3:
            return MarketRegime.HIGH_VOL

        bull = scores["bull"]
        bear = scores["bear"]

        # Recovery
        if scores["recovery"] >= 2 and bull >= bear:
            return MarketRegime.RECOVERY

        # Symmetric bands.  The previous version mapped net=+1 to SIDEWAYS but
        # net=-1 to WEAK_BEAR, i.e. it was structurally more bearish than
        # bullish for identical evidence strength.
        net = bull - bear
        if net >= 4:
            return MarketRegime.STRONG_BULL
        elif net >= 2:
            return MarketRegime.WEAK_BULL
        elif net >= -1:            # -1, 0, +1 → no meaningful edge either way
            return MarketRegime.SIDEWAYS
        elif net >= -3:
            return MarketRegime.WEAK_BEAR
        else:
            return MarketRegime.STRONG_BEAR

    # ------------------------------------------------------------------ #
    #  Allocation multipliers                                              #
    # ------------------------------------------------------------------ #

    def get_regime_allocation_multipliers(
        self, regime: MarketRegime
    ) -> dict[str, float]:
        """
        Return sleeve multipliers for `regime` from the canonical table.

        Keys: "longterm", "swing", "intraday" (see SLEEVES).  There is no
        "cash" key — cash is the residual plug in the allocator.

        Raises KeyError on an unknown regime rather than defaulting to
        SIDEWAYS: a silent default here would hide a vocabulary drift.
        """
        return regime_sleeve_multipliers(regime)

    # ------------------------------------------------------------------ #
    #  Technical helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sma(prices: pd.Series, period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        return float(prices.iloc[-period:].mean())

    @classmethod
    def _sma_adaptive(
        cls, prices: pd.Series, period: int, min_bars: Optional[int] = None
    ) -> tuple[Optional[float], int]:
        """
        Trailing mean over min(period, len(prices)) bars.

        Returns (sma, window_used).  Returns (None, 0) when there is less
        history than `min_bars` (default MIN_TREND_BARS).  Trailing window
        only — no look-ahead.
        """
        floor_bars = cls.MIN_TREND_BARS if min_bars is None else min_bars
        n = len(prices)
        if n < floor_bars:
            return None, 0
        window = min(period, n)
        return float(prices.iloc[-window:].mean()), window

    @staticmethod
    def _trailing_return(prices: pd.Series, period: int) -> float:
        """
        Simple return from `period` bars ago to the last bar (0.0 if there is
        not enough history).  Same convention as the original quarterly-return
        signal: prices.iloc[-1] / prices.iloc[-period] - 1.
        """
        if len(prices) < period:
            return 0.0
        start = float(prices.iloc[-period])
        if start == 0:
            return 0.0
        return float(prices.iloc[-1]) / start - 1.0

    @staticmethod
    def _drawdown_from_peak(prices: pd.Series, window: int) -> Optional[float]:
        """Drawdown of the last price from the peak of the trailing window."""
        if len(prices) < 2:
            return None
        sub = prices.iloc[-min(window, len(prices)):]
        peak = float(sub.max())
        if peak <= 0:
            return None
        return float(prices.iloc[-1]) / peak - 1.0

    @staticmethod
    def _realised_vol(prices: pd.Series, period: int) -> Optional[float]:
        """Annualised realised volatility from log returns over `period` days."""
        if len(prices) < period + 1:
            return None
        sub = prices.iloc[-(period + 1):]
        log_returns = np.log(sub / sub.shift(1)).dropna()
        if log_returns.empty:
            return None
        daily_vol = float(log_returns.std())
        return daily_vol * np.sqrt(252)   # annualised
