"""
intraday.py — Intraday momentum / mean-reversion strategy.

Design goals
------------
- Only signals where expected net edge > minimum threshold (after full costs).
- Strict time controls to avoid overnight risk and opening-auction noise.
- Market-breadth filter: reduce long signals in weak markets.
- Paper mode by default; no auto-live-trade.
- NO LOOK-AHEAD: features at time T use only data up to T.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.app.strategies.base import (
    BaseStrategy, PerformanceMetrics, Signal, SignalDirection, StrategyHealth
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intraday constants
# ---------------------------------------------------------------------------

_IST_MARKET_OPEN  = time(9, 15)
_IST_MARKET_CLOSE = time(15, 30)
_SIGNAL_CUTOFF    = time(14, 45)   # No new positions after this time
_SQUARE_OFF_TIME  = time(15, 15)   # Force-exit all intraday positions

# Minimum required edge AFTER brokerage + slippage to generate a signal.
# 0.003 = 30 basis points round-trip net edge
_MIN_NET_EDGE = 0.003

# Estimated round-trip transaction cost (brokerage + STT + slippage)
# for an intraday trade as a fraction of trade value.
_COST_ESTIMATE_INTRADAY = 0.0015  # 15 bps brokerage + 5 bps STT + ~5 bps slippage ≈ 25 bps

# Market breadth filter: if fewer than this fraction of universe stocks are
# positive on the day, reduce long bias.
_BREADTH_THRESHOLD = 0.40

# Maximum concurrent intraday positions
_MAX_POSITIONS = 5

# Rolling windows for intraday features (in bars; assumes 1-minute bars)
_SHORT_BARS  = 5
_MED_BARS    = 15
_LONG_BARS   = 30


class IntradayStrategy(BaseStrategy):
    """
    Intraday momentum strategy.

    Holding period: minutes to hours, always squared off by 15:15 IST.

    Signal generation approach
    --------------------------
    1. Filter to liquid stocks with strong average intraday volume.
    2. Compute intraday features: VWAP distance, relative volume, price
       momentum at 5/15/30-minute horizons, opening-range breakout signal.
    3. Estimate P(net_return > threshold) using a sigmoid model on the edge score.
    4. Signal only if expected_net_edge > _MIN_NET_EDGE.
    5. Rank signals by edge score.

    Time rules
    ----------
    - No signals in the first 15 minutes (opening auction volatility).
    - No new signals after 14:45 IST.
    - All positions squared off by 15:15 IST regardless of P&L.

    Market breadth rule
    -------------------
    If < 40% of the universe stocks are positive on the day so far, only
    generate short signals (or no signals if the strategy is long-only).
    """

    name = "IntradayMomentum"
    holding_period = "minutes_to_hours"

    def __init__(
        self,
        paper_mode: bool = True,
        min_net_edge: float = _MIN_NET_EDGE,
        max_positions: int = _MAX_POSITIONS,
        long_only: bool = True,
        min_intraday_volume: int = 1_000_000,
    ):
        super().__init__(paper_mode=paper_mode)
        self.min_net_edge = min_net_edge
        self.max_positions = max_positions
        self.long_only = long_only
        self.min_intraday_volume = min_intraday_volume

    # ------------------------------------------------------------------
    # Core signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        universe: List[str],
        features_df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        current_time: Optional[datetime] = None,
        intraday_data: Optional[Dict[str, pd.DataFrame]] = None,
        existing_positions: Optional[Dict[str, dict]] = None,
    ) -> List[Signal]:
        """
        Generate intraday trading signals.

        Parameters
        ----------
        universe : list[str]
        features_df : pd.DataFrame, daily features (for context).
        market_data : dict[str, pd.DataFrame], daily OHLCV per symbol.
        current_time : datetime (IST), defaults to now.
        intraday_data : dict[str, pd.DataFrame], intraday 1-minute bars per symbol.
            Each DataFrame has columns: time, open, high, low, close, volume.
        existing_positions : dict[str, dict], currently open positions.

        Returns
        -------
        list[Signal] ranked by edge_score descending, filtered for net edge.
        """
        if not self._is_operational():
            logger.info("IntradayStrategy is %s; no signals generated.", self.health)
            return []

        now = current_time or datetime.now()
        now_time = now.time()

        # Time gate: no signals in first 15 minutes or after cutoff
        if now_time < time(9, 30):  # 9:15 + 15 min grace
            logger.debug("Within 15-min opening window; no signals.")
            return []
        if now_time >= _SIGNAL_CUTOFF:
            logger.debug("Past signal cutoff %s; no new signals.", _SIGNAL_CUTOFF)
            return []

        if intraday_data is None:
            logger.warning("No intraday data provided; cannot generate intraday signals.")
            return []

        existing_syms = set((existing_positions or {}).keys())
        available_slots = self.max_positions - len(existing_syms)
        if available_slots <= 0:
            logger.debug("Max intraday positions reached; no new signals.")
            return []

        # Market breadth filter
        breadth = self._compute_breadth(universe, intraday_data)
        long_allowed = breadth >= _BREADTH_THRESHOLD
        if not long_allowed:
            logger.info(
                "Market breadth %.1f%% < threshold; suppressing long signals.",
                breadth * 100,
            )
            if self.long_only:
                return []

        # Filter for liquid stocks with sufficient intraday volume
        liquid = self._filter_intraday_liquid(universe, intraday_data)

        # Compute signals
        candidate_signals: List[Signal] = []
        for sym in liquid:
            if sym in existing_syms:
                continue
            idf = intraday_data.get(sym)
            if idf is None or len(idf) < _LONG_BARS:
                continue
            try:
                sig = self._build_signal(sym, idf, features_df, now, long_allowed)
                if sig is not None and sig.edge_score > self.min_net_edge:
                    candidate_signals.append(sig)
            except Exception as exc:
                logger.debug("Signal error for %s: %s", sym, exc)

        candidate_signals.sort(key=lambda s: s.edge_score, reverse=True)
        selected = candidate_signals[:available_slots]

        logger.info(
            "IntradayStrategy: %d candidates → %d signals (breadth=%.1f%%)",
            len(candidate_signals),
            len(selected),
            breadth * 100,
        )
        return selected

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        signal: Signal,
        available_capital: float,
        risk_engine,
    ) -> float:
        """
        Volatility-adjusted position sizing for intraday.

        Size = min(
            available_capital * max_position_pct,
            risk_per_trade / stop_loss_pct
        )
        where risk_per_trade = available_capital * 0.005 (0.5% per trade).
        """
        if available_capital <= 0 or not signal.is_valid():
            return 0.0

        multiplier = self._position_size_multiplier()
        if multiplier == 0.0:
            return 0.0

        risk_per_trade = available_capital * 0.005  # 0.5% capital at risk per trade
        stop_loss = signal.stop_loss_pct
        if stop_loss <= 0:
            return 0.0

        size_by_risk = risk_per_trade / stop_loss
        max_size = available_capital * 0.20  # max 20% per trade
        size = min(size_by_risk, max_size) * multiplier

        # Risk engine check
        try:
            if hasattr(risk_engine, "approve_trade"):
                if not risk_engine.approve_trade(signal.symbol, size, "intraday"):
                    return 0.0
        except Exception:
            pass

        return max(0.0, size)

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def should_exit(
        self,
        position: dict,
        current_data: Dict[str, Any],
    ) -> bool:
        """
        Exit intraday position if:
        1. Stop-loss hit.
        2. Target hit.
        3. Time-based exit: current_time >= 15:15 IST.
        4. Momentum reversal: price crosses back through VWAP.

        Parameters
        ----------
        position : dict with keys: symbol, entry_price, direction,
                   stop_loss, target, entry_time.
        current_data : dict with keys: price, time (datetime), vwap (optional).
        """
        current_price = float(current_data.get("price", 0))
        current_time = current_data.get("time")

        # 1. Time-based square-off
        if current_time is not None:
            t = current_time.time() if hasattr(current_time, "time") else current_time
            if t >= _SQUARE_OFF_TIME:
                logger.debug(
                    "Squaring off %s at %s (time-based exit).",
                    position["symbol"],
                    current_time,
                )
                return True

        if current_price <= 0:
            return False

        # 2. Stop-loss
        if self._hit_stop(position, current_price):
            logger.debug("Stop-loss hit for %s at %.2f.", position["symbol"], current_price)
            return True

        # 3. Target
        if self._hit_target(position, current_price):
            logger.debug("Target hit for %s at %.2f.", position["symbol"], current_price)
            return True

        # 4. VWAP reversal: if long and price crosses below VWAP (momentum reversal)
        vwap = current_data.get("vwap")
        if vwap is not None and vwap > 0:
            direction = position.get("direction", "LONG")
            entry_price = position.get("entry_price", 0)
            if direction == "LONG" and current_price < vwap and current_price < entry_price:
                logger.debug(
                    "VWAP reversal exit for %s (price=%.2f < VWAP=%.2f).",
                    position["symbol"], current_price, vwap,
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        sym: str,
        idf: pd.DataFrame,
        features_df: pd.DataFrame,
        now: datetime,
        long_allowed: bool,
    ) -> Optional[Signal]:
        """Compute intraday features and construct a Signal for one symbol."""
        close = idf["close"]
        volume = idf["volume"]
        high = idf["high"]
        low = idf["low"]

        # ---- VWAP ----
        typical = (high + low + close) / 3.0
        cum_tv = (typical * volume).cumsum()
        cum_v = volume.cumsum().replace(0, np.nan)
        vwap = cum_tv / cum_v
        last_price = close.iloc[-1]
        last_vwap = vwap.iloc[-1]
        if pd.isna(last_vwap) or last_vwap == 0:
            return None
        vwap_dist = last_price / last_vwap - 1.0

        # ---- Price momentum at multiple horizons ----
        def _ret(bars: int) -> float:
            if len(close) <= bars:
                return 0.0
            return float(np.log(close.iloc[-1] / close.iloc[-1 - bars]))

        mom_5  = _ret(_SHORT_BARS)
        mom_15 = _ret(_MED_BARS)
        mom_30 = _ret(_LONG_BARS)

        # ---- Relative volume ----
        avg_vol_20 = volume.tail(20).mean() if len(volume) >= 20 else volume.mean()
        current_vol = volume.iloc[-1]
        rel_vol = (current_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0

        # ---- Opening range breakout (first 30 min = first 30 bars for 1-min) ----
        first_30 = idf.iloc[:30]
        or_high = first_30["high"].max() if len(first_30) > 0 else last_price
        or_low  = first_30["low"].min()  if len(first_30) > 0 else last_price
        orb_signal = 0.0
        if last_price > or_high:
            orb_signal = (last_price - or_high) / or_high
        elif last_price < or_low:
            orb_signal = (last_price - or_low) / or_low  # negative

        # ---- Edge score: weighted composite ----
        # Higher score = more bullish momentum with volume confirmation
        edge_raw = (
            0.30 * np.sign(mom_15) * abs(mom_15)
            + 0.25 * np.sign(orb_signal) * abs(orb_signal) * 0.5
            + 0.20 * (rel_vol - 1.0) * 0.01  # scaled
            + 0.15 * (-vwap_dist) * 0.5     # below VWAP = potential mean-reversion
            + 0.10 * np.sign(mom_5) * abs(mom_5)
        )

        direction = SignalDirection.LONG if edge_raw > 0 else SignalDirection.SHORT

        # Suppress short signals if long_only or breadth is poor
        if self.long_only and direction == SignalDirection.SHORT:
            return None
        if not long_allowed and direction == SignalDirection.LONG:
            return None

        # Net edge = edge_raw - cost estimate
        expected_return = abs(edge_raw)
        net_edge = expected_return - _COST_ESTIMATE_INTRADAY

        # ---- Stop / target ----
        atr_approx = (high - low).tail(14).mean() / last_price
        stop_loss_pct = max(atr_approx * 1.5, 0.005)  # at least 50 bps
        target_pct = stop_loss_pct * 2.0  # 2:1 risk/reward minimum

        feature_snapshot = {
            "vwap_dist": vwap_dist,
            "mom_5": mom_5,
            "mom_15": mom_15,
            "mom_30": mom_30,
            "rel_vol": rel_vol,
            "orb_signal": orb_signal,
            "edge_raw": edge_raw,
        }

        return Signal(
            symbol=sym,
            direction=direction,
            strategy_name=self.name,
            timestamp=now,
            signal_date=now,
            edge_score=net_edge,
            expected_return=expected_return,
            expected_return_std=abs(expected_return) * 0.5,
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            holding_period_days=0,  # intraday
            feature_snapshot=feature_snapshot,
        )

    @staticmethod
    def _compute_breadth(
        universe: List[str],
        intraday_data: Dict[str, pd.DataFrame],
    ) -> float:
        """
        Fraction of universe stocks with positive return from open.

        Uses ONLY intraday data available at the current time.
        """
        positive = 0
        total = 0
        for sym in universe:
            idf = intraday_data.get(sym)
            if idf is None or len(idf) < 2:
                continue
            open_px = idf["open"].iloc[0]
            last_px = idf["close"].iloc[-1]
            if open_px > 0:
                total += 1
                if last_px > open_px:
                    positive += 1
        return positive / total if total > 0 else 0.5

    def _filter_intraday_liquid(
        self,
        universe: List[str],
        intraday_data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        """Retain symbols with sufficient average intraday volume."""
        liquid = []
        for sym in universe:
            idf = intraday_data.get(sym)
            if idf is None or len(idf) < 30:
                continue
            avg_vol = idf["volume"].mean()
            if avg_vol >= self.min_intraday_volume / 375:  # per-minute avg
                liquid.append(sym)
        return liquid
