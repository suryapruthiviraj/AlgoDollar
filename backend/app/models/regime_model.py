"""
regime_model.py — Market and volatility regime classification.

Re-exports MarketRegime and RegimeDetector from risk.regime.
Adds:
  VolatilityRegimeModel  — classify vol environment (LOW / MEDIUM / HIGH)
  CombinedRegimeModel    — join price-trend regime and vol regime
  RegimeHistory          — persistent log of regime transitions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

# Re-export from risk.regime if it exists; otherwise define local stubs so
# this module is usable stand-alone during development.
try:
    from backend.app.risk.regime import MarketRegime, RegimeDetector  # type: ignore
except ImportError:
    try:
        from app.risk.regime import MarketRegime, RegimeDetector  # type: ignore
    except ImportError:
        # ------- local fallback stubs -------
        class MarketRegime(str, Enum):  # type: ignore
            BULL = "BULL"
            BEAR = "BEAR"
            SIDEWAYS = "SIDEWAYS"
            PANIC = "PANIC"

        class RegimeDetector:  # type: ignore
            """Stub RegimeDetector — replace with real implementation from risk.regime."""

            def detect(self, nifty_close: pd.Series) -> MarketRegime:
                """Simple 200-day SMA rule as baseline."""
                if len(nifty_close) < 10:
                    return MarketRegime.SIDEWAYS
                last = nifty_close.iloc[-1]
                sma200 = nifty_close.tail(200).mean()
                if last > sma200 * 1.05:
                    return MarketRegime.BULL
                elif last < sma200 * 0.90:
                    return MarketRegime.BEAR
                return MarketRegime.SIDEWAYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Volatility regime
# ---------------------------------------------------------------------------

class VolatilityRegime(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class VolatilityRegimeModel:
    """
    Classify the current volatility environment using realized vol relative to
    its own rolling history.

    Thresholds
    ----------
    LOW    : current_vol < percentile_25 of historical vol
    MEDIUM : percentile_25 <= current_vol <= percentile_75
    HIGH   : current_vol > percentile_75

    Uses ONLY past data (rolling percentile computed on data up to bar T-1)
    to avoid look-ahead bias.
    """

    def __init__(
        self,
        vol_window: int = 21,
        percentile_window: int = 252,
        low_pct: float = 25.0,
        high_pct: float = 75.0,
    ):
        """
        Parameters
        ----------
        vol_window : int
            Window for realized volatility computation (trading days).
        percentile_window : int
            Rolling window for computing historical percentile boundaries.
        low_pct : float
            Percentile below which vol is "LOW".
        high_pct : float
            Percentile above which vol is "HIGH".
        """
        self.vol_window = vol_window
        self.percentile_window = percentile_window
        self.low_pct = low_pct
        self.high_pct = high_pct

    def classify_series(
        self, prices: pd.Series
    ) -> pd.Series:
        """
        Return a Series of VolatilityRegime values for each date in `prices`.

        Parameters
        ----------
        prices : pd.Series, close prices with DatetimeIndex.

        Returns
        -------
        pd.Series of VolatilityRegime, same index as prices.

        NO LOOK-AHEAD: percentile thresholds at T use vol history up to T-1.
        """
        log_rets = np.log(prices / prices.shift(1))
        realized_vol = (
            log_rets.rolling(self.vol_window, min_periods=self.vol_window // 2).std()
            * np.sqrt(252)
        )

        # Shift by 1 so the rolling percentile at T uses vol[T-1] and earlier
        rv_shifted = realized_vol.shift(1)

        regimes = pd.Series(index=prices.index, dtype=object)
        for i in range(len(prices)):
            t = prices.index[i]
            history = rv_shifted.iloc[max(0, i - self.percentile_window) : i + 1]
            history = history.dropna()
            current_rv = realized_vol.iloc[i]
            if pd.isna(current_rv) or len(history) < 10:
                regimes.iloc[i] = np.nan
                continue
            p_low = float(np.percentile(history, self.low_pct))
            p_high = float(np.percentile(history, self.high_pct))
            if current_rv <= p_low:
                regimes.iloc[i] = VolatilityRegime.LOW
            elif current_rv >= p_high:
                regimes.iloc[i] = VolatilityRegime.HIGH
            else:
                regimes.iloc[i] = VolatilityRegime.MEDIUM

        return regimes

    def classify_current(
        self, prices: pd.Series
    ) -> Optional[VolatilityRegime]:
        """Classify the most recent bar only."""
        series = self.classify_series(prices)
        val = series.dropna().iloc[-1] if not series.dropna().empty else None
        return val


# ---------------------------------------------------------------------------
# Combined regime
# ---------------------------------------------------------------------------

@dataclass
class CombinedRegime:
    """
    Combined market state from price-trend regime and volatility regime.

    Attributes
    ----------
    price_regime : MarketRegime
    vol_regime   : VolatilityRegime
    label        : str, e.g. "BULL_LOW_VOL"
    equity_fraction_cap : float
        Suggested maximum equity allocation for this regime.
    """
    price_regime: MarketRegime
    vol_regime: VolatilityRegime
    label: str = ""
    equity_fraction_cap: float = 1.0

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.price_regime.value}_{self.vol_regime.value}_VOL"


# Regime → recommended max equity fraction
_EQUITY_CAP: dict[tuple, float] = {
    (MarketRegime.BULL, VolatilityRegime.LOW):    1.00,
    (MarketRegime.BULL, VolatilityRegime.MEDIUM): 0.90,
    (MarketRegime.BULL, VolatilityRegime.HIGH):   0.75,
    (MarketRegime.SIDEWAYS, VolatilityRegime.LOW):    0.75,
    (MarketRegime.SIDEWAYS, VolatilityRegime.MEDIUM): 0.60,
    (MarketRegime.SIDEWAYS, VolatilityRegime.HIGH):   0.45,
    (MarketRegime.BEAR, VolatilityRegime.LOW):    0.50,
    (MarketRegime.BEAR, VolatilityRegime.MEDIUM): 0.35,
    (MarketRegime.BEAR, VolatilityRegime.HIGH):   0.20,
    (MarketRegime.PANIC, VolatilityRegime.LOW):    0.30,
    (MarketRegime.PANIC, VolatilityRegime.MEDIUM): 0.15,
    (MarketRegime.PANIC, VolatilityRegime.HIGH):   0.05,
}


class CombinedRegimeModel:
    """
    Combine price-trend regime (from RegimeDetector) and volatility regime
    (from VolatilityRegimeModel) into a single CombinedRegime.
    """

    def __init__(
        self,
        price_detector: Optional[RegimeDetector] = None,
        vol_model: Optional[VolatilityRegimeModel] = None,
    ):
        self._price_detector = price_detector or RegimeDetector()
        self._vol_model = vol_model or VolatilityRegimeModel()

    def detect(self, nifty_close: pd.Series) -> CombinedRegime:
        """
        Detect current combined regime.

        Parameters
        ----------
        nifty_close : pd.Series, NIFTY 50 close prices with DatetimeIndex.

        Returns
        -------
        CombinedRegime
        """
        price_regime = self._price_detector.detect(nifty_close)
        vol_regime = self._vol_model.classify_current(nifty_close)
        if vol_regime is None:
            vol_regime = VolatilityRegime.MEDIUM

        cap = _EQUITY_CAP.get((price_regime, vol_regime), 0.50)
        return CombinedRegime(
            price_regime=price_regime,
            vol_regime=vol_regime,
            equity_fraction_cap=cap,
        )

    def detect_series(
        self,
        nifty_close: pd.Series,
    ) -> pd.DataFrame:
        """
        Detect regime at every date in nifty_close.

        Returns DataFrame with columns: price_regime, vol_regime, label,
        equity_fraction_cap.
        """
        vol_series = self._vol_model.classify_series(nifty_close)
        records = []
        for i, t in enumerate(nifty_close.index):
            price_regime = self._price_detector.detect(nifty_close.iloc[: i + 1])
            vol_regime = vol_series.iloc[i]
            if pd.isna(vol_regime):
                vol_regime = VolatilityRegime.MEDIUM
            cr = CombinedRegime(
                price_regime=price_regime,
                vol_regime=vol_regime,
                equity_fraction_cap=_EQUITY_CAP.get((price_regime, vol_regime), 0.50),
            )
            records.append({
                "date": t,
                "price_regime": cr.price_regime.value,
                "vol_regime": cr.vol_regime.value,
                "label": cr.label,
                "equity_fraction_cap": cr.equity_fraction_cap,
            })
        df = pd.DataFrame(records).set_index("date")
        return df


# ---------------------------------------------------------------------------
# Regime history tracking
# ---------------------------------------------------------------------------

@dataclass
class RegimeTransition:
    timestamp: datetime
    from_label: Optional[str]
    to_label: str
    price_regime: str
    vol_regime: str


class RegimeHistory:
    """
    Maintains a running log of regime transitions.

    Usage
    -----
    history = RegimeHistory()
    history.update(current_combined_regime, timestamp=datetime.utcnow())
    transitions = history.get_transitions()
    """

    def __init__(self):
        self._transitions: List[RegimeTransition] = []
        self._current: Optional[CombinedRegime] = None

    def update(
        self,
        regime: CombinedRegime,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Record a transition if the regime has changed.

        Returns True if a transition occurred.
        """
        ts = timestamp or datetime.utcnow()
        if self._current is None or self._current.label != regime.label:
            self._transitions.append(
                RegimeTransition(
                    timestamp=ts,
                    from_label=self._current.label if self._current else None,
                    to_label=regime.label,
                    price_regime=regime.price_regime.value,
                    vol_regime=regime.vol_regime.value,
                )
            )
            logger.info(
                "Regime transition: %s → %s at %s",
                self._current.label if self._current else "None",
                regime.label,
                ts,
            )
            self._current = regime
            return True
        return False

    def get_transitions(self) -> List[RegimeTransition]:
        return list(self._transitions)

    def current_regime(self) -> Optional[CombinedRegime]:
        return self._current

    def to_dataframe(self) -> pd.DataFrame:
        if not self._transitions:
            return pd.DataFrame(
                columns=["timestamp", "from_label", "to_label", "price_regime", "vol_regime"]
            )
        return pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "from_label": t.from_label,
                "to_label": t.to_label,
                "price_regime": t.price_regime,
                "vol_regime": t.vol_regime,
            }
            for t in self._transitions
        ])
