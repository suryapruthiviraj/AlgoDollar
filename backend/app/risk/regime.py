"""Market regime detection using rule-based technical indicators."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    WEAK_BULL = "WEAK_BULL"
    SIDEWAYS = "SIDEWAYS"
    WEAK_BEAR = "WEAK_BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    HIGH_VOL = "HIGH_VOL"
    PANIC = "PANIC"
    RECOVERY = "RECOVERY"


# Default allocation multipliers per regime and strategy type
_REGIME_ALLOCATIONS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.STRONG_BULL: {
        "longterm": 1.0, "swing": 1.0, "intraday": 0.8, "cash": 0.2,
    },
    MarketRegime.WEAK_BULL: {
        "longterm": 0.8, "swing": 0.7, "intraday": 0.6, "cash": 0.3,
    },
    MarketRegime.SIDEWAYS: {
        "longterm": 0.5, "swing": 0.5, "intraday": 0.5, "cash": 0.4,
    },
    MarketRegime.WEAK_BEAR: {
        "longterm": 0.3, "swing": 0.2, "intraday": 0.3, "cash": 0.6,
    },
    MarketRegime.STRONG_BEAR: {
        "longterm": 0.1, "swing": 0.0, "intraday": 0.2, "cash": 0.8,
    },
    MarketRegime.HIGH_VOL: {
        "longterm": 0.5, "swing": 0.3, "intraday": 0.2, "cash": 0.5,
    },
    MarketRegime.PANIC: {
        "longterm": 0.5, "swing": 0.0, "intraday": 0.0, "cash": 0.5,
    },
    MarketRegime.RECOVERY: {
        "longterm": 0.6, "swing": 0.5, "intraday": 0.4, "cash": 0.4,
    },
}


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
            logger.warning("Not enough price history for regime detection; defaulting to SIDEWAYS.")
            return MarketRegime.SIDEWAYS

        scores: dict[str, int] = {
            "bull": 0,
            "bear": 0,
            "vol": 0,
            "recovery": 0,
        }

        last_price = float(nifty_prices.iloc[-1])

        # 1. 200-day SMA
        sma200 = self._sma(nifty_prices, 200)
        if sma200 is not None:
            if last_price > sma200:
                scores["bull"] += 2
            else:
                scores["bear"] += 2

        # 2. Golden / death cross: 50-day vs 200-day SMA
        sma50 = self._sma(nifty_prices, 50)
        if sma50 is not None and sma200 is not None:
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

        # 4. 63-day (quarterly) return
        if len(nifty_prices) >= 63:
            quarterly_return = (
                float(nifty_prices.iloc[-1]) / float(nifty_prices.iloc[-63]) - 1.0
            )
            if quarterly_return > self.QUARTERLY_RETURN_BULL:
                scores["bull"] += 1
            elif quarterly_return < self.QUARTERLY_RETURN_BEAR:
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
        regime = self._scores_to_regime(scores, vix)
        logger.debug(
            "Regime detection scores: %s → %s", scores, regime.value
        )
        return regime

    # ------------------------------------------------------------------ #
    #  Score → Regime mapping                                              #
    # ------------------------------------------------------------------ #

    def _scores_to_regime(
        self,
        scores: dict[str, int],
        vix: Optional[pd.Series],
    ) -> MarketRegime:
        # VIX overrides
        if vix is not None and not vix.empty:
            if float(vix.iloc[-1]) >= self.VIX_PANIC:
                return MarketRegime.PANIC

        # High vol
        if scores["vol"] >= 3:
            return MarketRegime.HIGH_VOL

        bull = scores["bull"]
        bear = scores["bear"]

        # Recovery
        if scores["recovery"] >= 2 and bull >= bear:
            return MarketRegime.RECOVERY

        net = bull - bear
        if net >= 4:
            return MarketRegime.STRONG_BULL
        elif net >= 2:
            return MarketRegime.WEAK_BULL
        elif net == 1 or net == 0:
            return MarketRegime.SIDEWAYS
        elif net == -1 or net == -2:
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
        Return allocation multipliers for each strategy type under the given regime.

        Keys: "longterm", "swing", "intraday", "cash"
        Values: float in [0.0, 1.0] — multiply base allocation by this factor.
        """
        return _REGIME_ALLOCATIONS.get(regime, _REGIME_ALLOCATIONS[MarketRegime.SIDEWAYS])

    # ------------------------------------------------------------------ #
    #  Technical helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sma(prices: pd.Series, period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        return float(prices.iloc[-period:].mean())

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
