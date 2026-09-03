"""
base.py — Abstract base for all AlgoDollar trading strategies.

Design principles
-----------------
- Paper mode is DEFAULT.  Live trading requires explicit opt-in.
- No forced trades: generate_signals() can return an empty list.
- Position sizing is risk-driven, not capital-driven.
- Strategies self-monitor health and downgrade gracefully.
- All signals are timestamped and carry the feature snapshot used to
  generate them (for reproducibility and debugging).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StrategyHealth(str, Enum):
    """
    Strategy operational health.

    HEALTHY   : Operating normally.
    REDUCED   : Operating at reduced capacity (e.g. smaller position sizes,
                fewer signals) due to recent underperformance.
    PAUSED    : Not generating new signals; existing positions managed to exit.
    DISABLED  : Completely inactive.  Requires manual intervention to re-enable.
    """
    HEALTHY  = "HEALTHY"
    REDUCED  = "REDUCED"
    PAUSED   = "PAUSED"
    DISABLED = "DISABLED"


class SignalDirection(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    EXIT  = "EXIT"


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """
    A trading signal produced by a strategy.

    Fields
    ------
    symbol : str
    direction : SignalDirection
    strategy_name : str
    timestamp : datetime
        Wall-clock time the signal was generated.  Used to detect stale signals.
    signal_date : datetime
        The market date whose data produced this signal.
    edge_score : float
        Expected net edge after estimated costs.  Signals with edge_score ≤ 0
        should not be executed.
    expected_return : float
        Model's point estimate of forward return (e.g. 5-day expected return).
    expected_return_std : float
        Uncertainty in the expected return estimate.
    stop_loss_pct : float
        Stop-loss level as fraction of entry price.  E.g. 0.02 = 2%.
    target_pct : float
        Target as fraction of entry price.
    holding_period_days : int
        Expected holding horizon.  Not a hard constraint.
    feature_snapshot : dict
        Copy of the feature values at signal generation.  Used for debugging.
    metadata : dict
        Any strategy-specific extra info.
    """
    symbol: str
    direction: SignalDirection
    strategy_name: str
    timestamp: datetime
    signal_date: datetime
    edge_score: float
    expected_return: float
    expected_return_std: float
    stop_loss_pct: float
    target_pct: float
    holding_period_days: int
    feature_snapshot: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """A signal is valid only if it has positive expected net edge."""
        return self.edge_score > 0.0

    def risk_reward_ratio(self) -> float:
        """target_pct / stop_loss_pct — must be > 1.0 to be acceptable."""
        if self.stop_loss_pct <= 0:
            return np.nan
        return self.target_pct / self.stop_loss_pct


# ---------------------------------------------------------------------------
# Performance thresholds for health auto-update
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Rolling performance metrics fed to update_health()."""
    rolling_sharpe_30d: float    # 30-day rolling annualized Sharpe
    rolling_sharpe_90d: float    # 90-day rolling annualized Sharpe
    current_drawdown_pct: float  # Current drawdown from peak (positive = drawdown)
    win_rate_30d: float          # Win rate over last 30 trades
    num_trades_30d: int          # Trade count for statistical significance


# Health thresholds (conservative defaults; tune per strategy).
_HEALTH_THRESHOLDS = {
    # (from_state, to_state): condition
    "reduce":  {"sharpe_30d": 0.0,  "drawdown": 0.05},  # Sharpe < 0 OR DD > 5%
    "pause":   {"sharpe_30d": -0.5, "drawdown": 0.10},  # Sharpe < -0.5 OR DD > 10%
    "disable": {"sharpe_30d": -1.0, "drawdown": 0.20},  # Sharpe < -1.0 OR DD > 20%
    "recover": {"sharpe_90d": 0.3,  "drawdown": 0.03},  # Sustained recovery
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Abstract base class for AlgoDollar trading strategies.

    Subclasses implement the specific signal logic for their time horizon.
    All common infrastructure (health management, signal validation,
    position-size guardrails) lives here.
    """

    # Subclasses must set these
    name: str = "base"
    holding_period: str = "unspecified"

    # Paper mode is default.  Set to False only in production with explicit
    # confirmation from the system's operator.
    paper_mode: bool = True

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.health: StrategyHealth = StrategyHealth.HEALTHY
        self._health_log: List[dict] = []
        if paper_mode:
            logger.info("Strategy '%s' initialized in PAPER mode.", self.name)
        else:
            logger.warning(
                "Strategy '%s' initialized in LIVE mode. "
                "Ensure risk controls are active.",
                self.name,
            )

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(
        self,
        universe: List[str],
        features_df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
    ) -> List[Signal]:
        """
        Produce a list of trading signals.

        Implementations MUST:
        1. Use only data available at the signal generation time (no look-ahead).
        2. Return an empty list if no edge is found (no forced trades).
        3. Return Signal objects with edge_score > 0 only.
        4. Respect self.health — PAUSED and DISABLED must return [].

        Parameters
        ----------
        universe : list[str]
            Eligible symbols (pre-filtered for liquidity, etc.).
        features_df : pd.DataFrame
            Feature panel indexed by (date, symbol) or wide format.
        market_data : dict[str, pd.DataFrame]
            Raw OHLCV data per symbol.

        Returns
        -------
        list[Signal], ranked by edge_score descending.
        """
        ...

    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        available_capital: float,
        risk_engine,
    ) -> float:
        """
        Calculate position size (number of shares or rupee amount) for a signal.

        Must return 0 if:
          - available_capital <= 0
          - signal.edge_score <= 0
          - risk_engine indicates breach of risk limits

        Parameters
        ----------
        signal : Signal
        available_capital : float (INR)
        risk_engine : RiskEngine instance

        Returns
        -------
        float : rupee value of position (0 = no trade).
        """
        ...

    @abstractmethod
    def should_exit(
        self,
        position: dict,
        current_data: Dict[str, Any],
    ) -> bool:
        """
        Decide whether to exit an open position.

        Parameters
        ----------
        position : dict with at least:
            symbol, entry_price, entry_date, direction, stop_loss, target,
            signal (the original Signal object).
        current_data : dict with current OHLCV, time, features.

        Returns
        -------
        bool : True = exit now.
        """
        ...

    # ------------------------------------------------------------------
    # Health management
    # ------------------------------------------------------------------

    def update_health(
        self,
        metrics: PerformanceMetrics,
        timestamp: Optional[datetime] = None,
    ) -> StrategyHealth:
        """
        Auto-update strategy health based on rolling performance metrics.

        Logic (transitions apply in order; first match wins):
        1. HEALTHY → REDUCED  if Sharpe_30d < 0 or drawdown > 5%
        2. HEALTHY/REDUCED → PAUSED if Sharpe_30d < -0.5 or drawdown > 10%
        3. Any → DISABLED if Sharpe_30d < -1.0 or drawdown > 20%
        4. REDUCED/PAUSED → HEALTHY if Sharpe_90d > 0.3 and drawdown < 3%
           (recovery requires at least 10 trades for statistical significance)

        Parameters
        ----------
        metrics : PerformanceMetrics
        timestamp : datetime, defaults to now.

        Returns
        -------
        StrategyHealth (new state)
        """
        ts = timestamp or datetime.utcnow()
        old_health = self.health
        t = _HEALTH_THRESHOLDS

        sharpe_30 = metrics.rolling_sharpe_30d
        sharpe_90 = metrics.rolling_sharpe_90d
        dd = metrics.current_drawdown_pct
        n_trades = metrics.num_trades_30d

        new_health = old_health

        # Disable (most severe)
        if sharpe_30 < t["disable"]["sharpe_30d"] or dd > t["disable"]["drawdown"]:
            new_health = StrategyHealth.DISABLED
        # Pause
        elif sharpe_30 < t["pause"]["sharpe_30d"] or dd > t["pause"]["drawdown"]:
            new_health = StrategyHealth.PAUSED
        # Reduce
        elif (
            (sharpe_30 < t["reduce"]["sharpe_30d"] or dd > t["reduce"]["drawdown"])
            and old_health == StrategyHealth.HEALTHY
        ):
            new_health = StrategyHealth.REDUCED
        # Recovery (from REDUCED or PAUSED only)
        elif (
            old_health in (StrategyHealth.REDUCED, StrategyHealth.PAUSED)
            and sharpe_90 >= t["recover"]["sharpe_90d"]
            and dd <= t["recover"]["drawdown"]
            and n_trades >= 10
        ):
            new_health = StrategyHealth.HEALTHY

        self.health = new_health
        if new_health != old_health:
            logger.warning(
                "Strategy '%s' health: %s → %s at %s (sharpe30=%.2f, dd=%.1f%%)",
                self.name,
                old_health.value,
                new_health.value,
                ts,
                sharpe_30,
                dd * 100,
            )
            self._health_log.append({
                "timestamp": ts,
                "from": old_health.value,
                "to": new_health.value,
                "sharpe_30d": sharpe_30,
                "drawdown": dd,
            })
        return self.health

    # ------------------------------------------------------------------
    # Guard helpers
    # ------------------------------------------------------------------

    def _is_operational(self) -> bool:
        """Return True if strategy can generate new signals."""
        return self.health in (StrategyHealth.HEALTHY, StrategyHealth.REDUCED)

    def _position_size_multiplier(self) -> float:
        """
        Scale factor for position sizes based on health.
        HEALTHY → 1.0, REDUCED → 0.5, PAUSED/DISABLED → 0.0.
        """
        return {
            StrategyHealth.HEALTHY:  1.0,
            StrategyHealth.REDUCED:  0.5,
            StrategyHealth.PAUSED:   0.0,
            StrategyHealth.DISABLED: 0.0,
        }[self.health]

    # ------------------------------------------------------------------
    # Common stop-loss / target checks
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_stop(
        position: dict, current_price: float, atr_buffer: float = 0.0
    ) -> bool:
        """Return True if current_price has hit or passed the stop-loss level."""
        direction = position.get("direction", "LONG")
        stop = position["stop_loss"]
        if direction == "LONG":
            return current_price <= stop * (1 - atr_buffer)
        else:
            return current_price >= stop * (1 + atr_buffer)

    @staticmethod
    def _hit_target(position: dict, current_price: float) -> bool:
        """Return True if current_price has hit or passed the profit target."""
        direction = position.get("direction", "LONG")
        target = position["target"]
        if direction == "LONG":
            return current_price >= target
        else:
            return current_price <= target

    def get_health_log(self) -> List[dict]:
        """Return audit log of all health transitions."""
        return list(self._health_log)
