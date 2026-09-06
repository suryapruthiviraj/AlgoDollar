"""
monitor.py — the intraday exit brain of the auto-trader.

One job: given the currently open positions, the session's 1-minute bars and
the clock, decide whether each position must be exited now and WHY it must.

It is deliberately the READ-ONLY half of the loop.  It never places, cancels
or even suggests an order — it returns `ExitDecision` objects.  That keeps it
unit-testable without a broker and keeps every exit reason auditable, because
the decision is the thing the trader logs before it acts on it.

The reasons are the same ones the strategy itself documents, evaluated from
the SAME helpers the strategy uses (`BaseStrategy._hit_stop` / `_hit_target`)
so the monitor and the strategy can never disagree about what "hit" means:

  STOP           the price crossed the stop trigger
  TARGET         the price crossed the profit target
  VWAP_REVERSAL  a long that was losing its momentum crossed back under VWAP
  SQUARE_OFF     the clock passed the mandatory intraday square-off time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# IST square-off is the strategy's last line of defence, kept in-sync here by
# construction (default), overridable in tests.
_SQUARE_OFF_TIME = time(15, 15)


class ExitReason(str, Enum):
    STOP = "stop"
    TARGET = "target"
    VWAP_REVERSAL = "vwap_reversal"
    SQUARE_OFF = "square_off"
    # Produced only by the auto-trader, never by the monitor: an entry whose
    # protective SL-M stop could not be placed must be exited again, and this
    # is the honest label for that.
    STOP_PLACEMENT_FAILED = "stop_placement_failed"


@dataclass
class ExitDecision:
    symbol: str
    reason: ExitReason
    price: float
    vwap: Optional[float]
    exit_time: datetime


class PositionExitMonitor:
    """
    Decide whether open intraday positions must be exited right now.

    ``evaluate`` returns one `ExitDecision` per position that must go.  It is
    a pure function of its inputs — no broker, no state, no orders.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        *,
        square_off_time: time = _SQUARE_OFF_TIME,
    ) -> None:
        self._strategy = strategy
        self.square_off_time = square_off_time

    # ------------------------------------------------------------------ #
    #  Main entry                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        positions: Sequence[dict],
        bars: Mapping[str, pd.DataFrame],
        now: datetime,
    ) -> List[ExitDecision]:
        """
        Evaluate every open position against the current bar state and clock.

        Positions that cannot be priced (no bar yet, no last_price) are left
        alone: the broker-side SL-M stop is their protection, not this monitor.
        """
        decisions: List[ExitDecision] = []
        for pos in positions:
            symbol = pos.get("symbol")
            if not symbol:
                continue
            df = bars.get(symbol)
            price = self._current_price(pos, df)
            if price is None or price <= 0:
                logger.debug("No usable price for %s; monitor leaves it to the stop.", symbol)
                continue
            vwap = self._session_vwap(df)
            data: Dict = {"price": price, "time": now, "vwap": vwap}
            try:
                should = self._strategy.should_exit(pos, data)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Exit evaluation failed for %s (%s); keeping position and stop.",
                    symbol, exc,
                )
                continue
            if not should:
                continue
            reason = self._reason(pos, price, vwap, now)
            logger.info(
                "EXIT %s reason=%s price=%.2f vwap=%s",
                symbol, reason.value, price,
                f"{vwap:.2f}" if vwap else "n/a",
            )
            decisions.append(ExitDecision(
                symbol=symbol, reason=reason, price=price,
                vwap=vwap, exit_time=now,
            ))
        return decisions

    # ------------------------------------------------------------------ #
    #  Reason classification                                              #
    # ------------------------------------------------------------------ #

    def _reason(
        self,
        position: dict,
        price: float,
        vwap: Optional[float],
        now: datetime,
    ) -> ExitReason:
        """Attach a reason: square-off wins, then stop, then target, then vwap."""
        if now.time() >= self.square_off_time:
            return ExitReason.SQUARE_OFF
        if self._strategy._hit_stop(position, price):
            return ExitReason.STOP
        if self._strategy._hit_target(position, price):
            return ExitReason.TARGET
        if self._vwap_reversal(position, price, vwap):
            return ExitReason.VWAP_REVERSAL
        return ExitReason.SQUARE_OFF  # defensive: should_exit said yes, so label it

    def _vwap_reversal(self, position: dict, price: float, vwap: Optional[float]) -> bool:
        if vwap is None or vwap <= 0:
            return False
        direction = str(position.get("direction", "LONG")).upper()
        entry = float(position.get("entry_price", 0) or 0)
        if direction != "LONG":
            return False
        return price < vwap and price < entry

    # ------------------------------------------------------------------ #
    #  Pricing helpers (must agree with the strategy's own math)          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _current_price(position: dict, bars: Optional[pd.DataFrame]) -> Optional[float]:
        """Newest close from the bars, else the position's last mark."""
        if bars is not None and not bars.empty:
            return float(bars["close"].iloc[-1])
        last = position.get("last_price")
        return float(last) if last else None

    @staticmethod
    def _session_vwap(bars: Optional[pd.DataFrame]) -> Optional[float]:
        """
        Session VWAP from typical price (H+L+C)/3 — exactly the strategy's
        ``_build_signal`` formula, so the monitor and the strategy read the
        same VWAP and the exit decision cannot diverge from the entry logic.
        """
        if bars is None or bars.empty:
            return None
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
        cum_v = float(bars["volume"].astype(float).sum())
        if cum_v <= 0:
            return None
        return float((typical * bars["volume"]).sum() / cum_v)
