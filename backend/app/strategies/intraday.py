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
from datetime import datetime, time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.strategies.base import (
    INTRADAY_ROUND_TRIP_COST,
    MAX_GROSS_EXPOSURE,
    BaseStrategy,
    Signal,
    SignalDirection,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Time zone
# ---------------------------------------------------------------------------
# Every session decision in this module is made in EXCHANGE time, never in the
# host's local time.  On a UTC server (the norm for cloud deployment) naive
# datetime.now() made IST 11:00 look like 05:30 ("opening window", blocked)
# and IST 15:20 look like 09:50 (past square-off, yet treated as tradeable):
# the strategy was muted through the real session, allowed to open positions
# after the 14:45 cutoff, and NEVER squared off — carrying an intraday book
# overnight at intraday leverage.
IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Intraday constants (all IST wall-clock)
# ---------------------------------------------------------------------------

_IST_MARKET_OPEN  = time(9, 15)
_IST_MARKET_CLOSE = time(15, 30)
_OPENING_GRACE_END = time(9, 30)   # No new signals in the first 15 minutes
_SIGNAL_CUTOFF    = time(14, 45)   # No new positions after this time
_SQUARE_OFF_TIME  = time(15, 15)   # Force-exit all intraday positions

# Estimated ROUND-TRIP transaction cost for an intraday (MIS) trade, as a
# fraction of trade value.  Reconciled with app.backtesting.costs:
#   ZerodhaCostModel().breakeven_return(qty, px, product="MIS") returns
#   4.5-10.7 bps round trip (brokerage is ₹20-capped, so the percentage falls
#   as the ticket grows) — call it ~8 bps — plus ~5 bps of slippage per leg.
# The old 0.0015 constant contradicted its own "≈ 25 bps" comment and, more
# importantly, understated the hurdle it was compared against.
_COST_ESTIMATE_INTRADAY = INTRADAY_ROUND_TRIP_COST  # ~18 bps

# Minimum required edge AFTER costs to generate a signal.
# 0.003 = 30 basis points of NET edge on top of the cost hurdle.
_MIN_NET_EDGE = 0.003

# Fraction of a recent intraday move that is expected to persist over the
# holding window.  Intraday momentum is weak and mean-reverting at the margin;
# assuming a move repeats in full is not a forecast, it is an extrapolation.
_CONTINUATION_COEF = 0.25

# Weights on the (dimensionally homogeneous) RETURN components of the edge.
# These sum to 1.0 so the blend stays in return units.
_W_MOM15   = 0.45
_W_ORB     = 0.30
_W_MOM5    = 0.15
_W_VWAP_MR = 0.10

# Relative volume enters as a dimensionless CONFIDENCE MULTIPLIER, never as
# an additive return term.  Bounded so a volume spike can modulate — but never
# manufacture — an expected return.
_VOL_CONF_MIN = 0.50
_VOL_CONF_MAX = 1.25
_VOL_CONF_SLOPE = 0.25

# Market breadth filter: if fewer than this fraction of universe stocks are
# positive on the day, reduce long bias.
_BREADTH_THRESHOLD = 0.40

# Maximum concurrent intraday positions
_MAX_POSITIONS = 5

# Per-position notional cap as a fraction of sleeve capital.
_MAX_POSITION_PCT = 0.20

# Rolling windows for intraday features (in bars; assumes 1-minute bars)
_SHORT_BARS  = 5
_MED_BARS    = 15
_LONG_BARS   = 30


class NaiveDatetimeError(ValueError):
    """
    Raised when a timestamp used for a SESSION decision is timezone-naive.

    A naive timestamp is unusable here: it is only correct if the host happens
    to be in IST, which is exactly the assumption that broke square-off on UTC
    servers.  Callers must pass an aware datetime; this module never guesses.
    """


def to_ist(ts: datetime, field: str = "timestamp") -> datetime:
    """
    Convert an AWARE datetime to IST.  Raises on a naive datetime.

    This is the single door through which every session-time decision passes.
    """
    if not isinstance(ts, datetime):
        raise NaiveDatetimeError(
            f"{field} must be a timezone-aware datetime, got "
            f"{type(ts).__name__}: a bare time carries no zone and cannot be "
            "resolved to an exchange session time."
        )
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise NaiveDatetimeError(
            f"{field} is timezone-naive ({ts!r}).  Session decisions must use "
            "an aware datetime — pass datetime.now(IST) or attach a tzinfo. "
            "Assuming the host's local zone silently breaks square-off on any "
            "non-IST (e.g. UTC) server."
        )
    return ts.astimezone(IST)


def now_ist() -> datetime:
    """Current exchange time.  Correct on a host in any time zone."""
    return datetime.now(IST)


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
        cost_estimate: float = _COST_ESTIMATE_INTRADAY,
    ):
        """
        Parameters
        ----------
        min_net_edge : float
            Net edge (expected return minus cost, as a fraction of notional)
            required to emit a signal.
        cost_estimate : float
            Round-trip cost hurdle as a fraction of notional.  Defaults to
            ~18 bps, reconciled with ZerodhaCostModel MIS costs plus slippage.
            Inject the real per-ticket number when it is known.
        """
        super().__init__(paper_mode=paper_mode)
        self.min_net_edge = min_net_edge
        self.max_positions = max_positions
        self.long_only = long_only
        self.min_intraday_volume = min_intraday_volume
        self.cost_estimate = float(cost_estimate)

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
        current_time : timezone-AWARE datetime, defaults to now in IST.
            Converted to IST internally; a naive datetime raises
            NaiveDatetimeError rather than being assumed to be exchange time.
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

        # All session gating happens in IST, whatever the host's clock is set to.
        now = to_ist(current_time, "current_time") if current_time is not None else now_ist()
        now_time = now.time()

        # Time gate: no signals in first 15 minutes or after cutoff
        if now_time < _OPENING_GRACE_END:  # 9:15 + 15 min grace
            logger.debug("Within 15-min opening window (%s IST); no signals.", now_time)
            return []
        if now_time >= _SIGNAL_CUTOFF:
            logger.debug(
                "Past signal cutoff %s IST (now %s IST); no new signals.",
                _SIGNAL_CUTOFF, now_time,
            )
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

        # Portfolio budget across the emitted set (per-position caps alone do
        # not bound gross exposure).
        budget = self._entry_budget(self.max_positions, len(existing_syms))
        self._stamp_target_weights(
            selected,
            [self._raw_target_weight(s) for s in selected],
            budget=budget,
        )

        logger.info(
            "IntradayStrategy: %d candidates → %d signals (breadth=%.1f%%, "
            "gross intent %.1f%% of sleeve)",
            len(candidate_signals),
            len(selected),
            breadth * 100,
            sum(s.metadata.get("target_weight", 0.0) for s in selected) * 100,
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
        Volatility-adjusted position sizing for intraday, bounded by the
        portfolio budget.

        Raw weight = min(0.5% risk / stop_loss_pct, 20% of capital); the batch
        normalisation stamped by generate_signals() then guarantees the
        emitted set never intends more than the sleeve's remaining budget.

        Raises
        ------
        RiskEngineError : if a supplied risk engine cannot approve the trade.
        """
        if available_capital <= 0 or not signal.is_valid():
            return 0.0

        multiplier = self._position_size_multiplier()
        if multiplier == 0.0:
            return 0.0

        stamped = signal.metadata.get("target_weight")
        if stamped is not None:
            target_weight = float(stamped)
        else:
            target_weight = min(
                self._raw_target_weight(signal),
                MAX_GROSS_EXPOSURE / max(1, self.max_positions),
            )
        if target_weight <= 0:
            return 0.0

        size = available_capital * target_weight * multiplier

        if not self._risk_engine_approves(risk_engine, signal.symbol, size, "intraday"):
            return 0.0

        return max(0.0, size)

    def _raw_target_weight(self, signal: Signal) -> float:
        """
        Pre-normalisation weight: 0.5% of capital at risk per trade, capped at
        _MAX_POSITION_PCT of the sleeve.
        """
        stop_loss = signal.stop_loss_pct
        if stop_loss is None or not np.isfinite(stop_loss) or stop_loss <= 0:
            return 0.0
        weight_by_risk = 0.005 / stop_loss  # 0.5% capital at risk per trade
        return float(min(weight_by_risk, _MAX_POSITION_PCT))

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
        3. Time-based exit: current_time >= 15:15 IST (evaluated in IST, so it
           fires correctly on a UTC host: 09:45 UTC == 15:15 IST).
        4. Momentum reversal: price crosses back through VWAP.

        Parameters
        ----------
        position : dict with keys: symbol, entry_price, direction,
                   stop_loss, target, entry_time.
        current_data : dict with keys: price, time (timezone-AWARE datetime),
                       vwap (optional).

        Raises
        ------
        NaiveDatetimeError : if current_data["time"] is naive.  Square-off is
            the last line of defence against carrying an intraday book
            overnight; guessing a zone here is not acceptable.
        """
        current_price = float(current_data.get("price", 0))
        current_time = current_data.get("time")

        # 1. Time-based square-off (in IST, whatever the host clock says)
        if current_time is not None:
            ist_now = to_ist(current_time, 'current_data["time"]')
            if ist_now.time() >= _SQUARE_OFF_TIME:
                logger.info(
                    "Squaring off %s at %s IST (time-based exit; source ts %s).",
                    position["symbol"],
                    ist_now.time(),
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

        # ---- Opening range breakout (first 30 min of THIS session) ----
        first_30 = self._session_frame(idf).iloc[:30]
        or_high = first_30["high"].max() if len(first_30) > 0 else last_price
        or_low  = first_30["low"].min()  if len(first_30) > 0 else last_price
        orb_signal = 0.0
        if last_price > or_high:
            orb_signal = (last_price - or_high) / or_high
        elif last_price < or_low:
            orb_signal = (last_price - or_low) / or_low  # negative

        # ---- Expected return: a RETURN, built only from return terms ----
        #
        # Every term below is a price return (dimensionless fraction of
        # price), so the weighted blend is a return and can legitimately be
        # compared against a cost, which is also a fraction of notional.
        # The blend is then shrunk by _CONTINUATION_COEF, because only part of
        # a recent move persists.
        #
        # Relative volume is DELIBERATELY absent from this sum.  It used to
        # enter as `0.20 * (rel_vol - 1.0) * 0.01`, converting a dimensionless
        # ratio into return units via an arbitrary constant: with zero price
        # movement and rel_vol=6.9 that alone manufactured a 1.18% "expected
        # return" and a tradeable signal out of nothing but a volume spike.
        directional_return = (
            _W_MOM15 * mom_15
            + _W_ORB * orb_signal
            + _W_MOM5 * mom_5
            + _W_VWAP_MR * (-vwap_dist)  # below VWAP = mean-reversion pull up
        )
        expected_move = _CONTINUATION_COEF * directional_return

        # Volume informs CONFIDENCE only: a dimensionless multiplier bounded
        # away from zero and from inflation.  It can scale a real expected
        # move up or down; it can never create one.
        volume_confidence = float(np.clip(
            1.0 + _VOL_CONF_SLOPE * (rel_vol - 1.0),
            _VOL_CONF_MIN,
            _VOL_CONF_MAX,
        ))
        expected_return_signed = expected_move * volume_confidence

        if expected_return_signed == 0.0:
            # No directional information at all.  Under the old formula this
            # case still produced a signal, because the volume term supplied a
            # non-zero "return" on its own.
            logger.debug("%s: zero expected move; no signal.", sym)
            return None

        direction = (
            SignalDirection.LONG if expected_return_signed > 0
            else SignalDirection.SHORT
        )

        # Suppress short signals if long_only or breadth is poor
        if self.long_only and direction == SignalDirection.SHORT:
            return None
        if not long_allowed and direction == SignalDirection.LONG:
            return None

        # Net edge = |expected return| - round-trip cost.  Both are fractions
        # of notional, so this subtraction is dimensionally valid.
        expected_return = abs(expected_return_signed)
        net_edge = expected_return - self.cost_estimate

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
            "directional_return": float(directional_return),
            "volume_confidence": volume_confidence,
            "expected_return_signed": float(expected_return_signed),
        }

        return Signal(
            symbol=sym,
            direction=direction,
            strategy_name=self.name,
            timestamp=now,
            signal_date=now,
            edge_score=net_edge,
            expected_return=expected_return,
            # Dispersion of the intraday move itself (ATR-scaled), not a
            # fraction of the point estimate.
            expected_return_std=float(max(atr_approx, 1e-6)),
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            holding_period_days=0,  # intraday
            feature_snapshot=feature_snapshot,
            metadata={
                "return_units": "simple_return_over_intraday_holding_window",
                "cost_estimate": self.cost_estimate,
                "session_date": now.date().isoformat(),
            },
        )

    @staticmethod
    def _session_frame(idf: pd.DataFrame) -> pd.DataFrame:
        """
        Bars belonging to the CURRENT SESSION only.

        The caller may pass a rolling window or several days of bars, so
        `idf.iloc[0]` is not the session open and `idf.iloc[:30]` is not the
        opening range.  Bars are grouped by their own date and the last date's
        bars are returned.  With no usable timestamps the frame is returned
        unchanged (and the caller's fallback is documented as approximate).
        """
        if idf is None or idf.empty:
            return idf

        stamps = None
        if "time" in idf.columns:
            stamps = pd.to_datetime(idf["time"], errors="coerce")
        elif isinstance(idf.index, pd.DatetimeIndex):
            stamps = pd.Series(idf.index)

        if stamps is not None and stamps.notna().any():
            try:
                dates = stamps.dt.date.to_numpy()
                mask = dates == dates[-1]
                if mask.any():
                    return idf.loc[mask]
            except (AttributeError, TypeError, ValueError):
                pass

        logger.debug(
            "Intraday frame has no usable timestamps; treating the whole "
            "frame as one session."
        )
        return idf

    @classmethod
    def _session_open_price(cls, idf: pd.DataFrame) -> Optional[float]:
        """Open of the current session (see _session_frame)."""
        if idf is None or idf.empty or "open" not in idf.columns:
            return None
        session = cls._session_frame(idf)
        if session is None or session.empty:
            return None
        return float(session["open"].iloc[0])

    @classmethod
    def _compute_breadth(
        cls,
        universe: List[str],
        intraday_data: Dict[str, pd.DataFrame],
    ) -> float:
        """
        Fraction of universe stocks with positive return from the SESSION
        OPEN.

        Uses ONLY intraday data available at the current time.
        """
        positive = 0
        total = 0
        for sym in universe:
            idf = intraday_data.get(sym)
            if idf is None or len(idf) < 2:
                continue
            open_px = cls._session_open_price(idf)
            last_px = idf["close"].iloc[-1]
            if open_px is not None and open_px > 0:
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
